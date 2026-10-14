"""One file from a public HTTPS address, downloaded once, resumably, and checked before it is used.

- **A file on disk is whole.** Bytes arrive in `<destination>.part` and are renamed into place only
  once their length matches what the server announced and their digest matches the one the caller
  expects, so an interrupted or wrong transfer never sits at the path a reader opens.
- **An interrupted transfer resumes.** A later call asks for the rest of `<destination>.part` with a
  `Range` header, and starts over when the server answers with the whole file instead.
- **A present file is not downloaded again.** Its digest is still checked when one is expected.
- **Only HTTPS.** Plain HTTP is accepted for a loopback address alone, which is what tests serve.

Response headers are kept only from `RECORDED_HEADERS`, the names a provenance record needs, so a
cookie or a signed-URL parameter a server echoes never reaches a manifest.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from foundry.digests import sha256_file


USER_AGENT = "synthyra-foundry/1 (scientific-reproducibility download)"
CHUNK_BYTES = 8 * 1024 * 1024
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
RECORDED_HEADERS = frozenset(
    {
        "accept-ranges",
        "cache-control",
        "content-disposition",
        "content-length",
        "content-range",
        "content-type",
        "date",
        "etag",
        "last-modified",
        "server",
        "x-amz-version-id",
    }
)


class DigestMismatch(ValueError):
    """The downloaded or cached bytes are not the bytes the caller expects."""


class IncompleteDownload(OSError):
    """The transfer ended before the length the server announced."""


@dataclass(frozen=True, slots=True)
class Download:
    """Where one address's file now is, and what is known of its bytes.

    `status` is "downloaded" when bytes arrived during this call and "cached" when the file was
    already there, in which case `response_headers` is empty: no request was made.
    """

    url: str
    path: Path
    size_bytes: int
    sha256: str
    response_headers: dict[str, str]
    status: Literal["downloaded", "cached"]


def require_https(url: str) -> None:
    """Raise `ValueError` unless `url` is HTTPS, or plain HTTP to a loopback address."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and parsed.hostname in LOOPBACK_HOSTS:
        return
    raise ValueError(f"Source URL must use HTTPS: {url!r}")


def recorded_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """The headers named in `RECORDED_HEADERS`, their names lowercased."""
    return {name.lower(): str(value) for name, value in headers.items() if name.lower() in RECORDED_HEADERS}


def probe_headers(url: str, *, timeout_seconds: float = 30.0, user_agent: str = USER_AGENT) -> dict[str, str]:
    """The recorded headers of a HEAD request to `url`, or an empty dict when the request fails.

    For filling in the provenance of a file already on disk, where a missing header costs a
    manifest field and nothing else. Never use it to decide whether a file exists.
    """
    require_https(url)
    request = urllib.request.Request(url, headers={"User-Agent": user_agent}, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return recorded_headers(dict(response.headers.items()))
    except (OSError, urllib.error.URLError):
        return {}


def download_file(
    url: str,
    destination: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
    timeout_seconds: float = 120.0,
    user_agent: str = USER_AGENT,
) -> Download:
    """The file at `url`, at `destination`, downloaded unless it is already there.

    Raises `DigestMismatch` when `expected_sha256` is given and the bytes differ, removing a
    download that does not match, and `IncompleteDownload` when fewer bytes arrive than the server
    announced, keeping them in `<destination>.part` for the next call to resume.
    """
    require_https(url)
    destination = Path(destination)
    if destination.is_file():
        found_sha256 = sha256_file(destination)
        _require_digest(found_sha256, expected_sha256, destination)
        return Download(url, destination, destination.stat().st_size, found_sha256, {}, "cached")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request_headers = {"User-Agent": user_agent}
    if offset:
        request_headers["Range"] = f"bytes={offset}-"

    request = urllib.request.Request(url, headers=request_headers, method="GET")
    try:
        response = urllib.request.urlopen(request, timeout=timeout_seconds)
    except urllib.error.HTTPError as error:
        # 416 to a resume means the range starts at or past the end: the partial file may be whole.
        if error.code != 416 or not offset:
            raise
        headers = recorded_headers(dict(error.headers.items()))
        total = _announced_total(headers.get("content-range", ""))
        if total is None or offset != total:
            raise IncompleteDownload(
                f"{url}: the server refused to resume without showing that {partial} is complete"
            ) from error
        return _finish(url, partial, destination, expected_sha256, headers)

    with response:
        response_status = getattr(response, "status", None)
        status = int(response.getcode() if response_status is None else response_status)
        append = offset > 0 and status == 206
        headers = recorded_headers(dict(response.headers.items()))
        with partial.open("ab" if append else "wb") as handle:
            shutil.copyfileobj(response, handle, length=CHUNK_BYTES)
            handle.flush()
            os.fsync(handle.fileno())

    expected_size = _expected_size(headers, offset if append else 0)
    size = partial.stat().st_size
    if expected_size is not None and size != expected_size:
        raise IncompleteDownload(f"{url}: expected {expected_size} bytes, received {size}")
    return _finish(url, partial, destination, expected_sha256, headers)


def download_file_segmented(
    url: str,
    destination: str | os.PathLike[str],
    *,
    connections: int = 8,
    segment_bytes: int = 256 * 1024 * 1024,
    expected_sha256: str | None = None,
    timeout_seconds: float = 120.0,
    user_agent: str = USER_AGENT,
) -> Download:
    """The file at `url`, fetched as byte ranges over several connections, with the guarantees of `download_file`.

    Some servers cap each connection (EBI's FTP mirror gives about 2 MB/s), so one transfer of a 10 GB file takes
    over an hour while eight in parallel take minutes. The file is split into `segment_bytes` ranges, each written in
    place into `<destination>.part`; the finished ranges are listed in `<destination>.part.segments`, so an interrupted
    call redoes only the unfinished ones. A server that does not accept ranges is read as one stream by `download_file`.
    """
    require_https(url)
    destination = Path(destination)
    if destination.is_file():
        return download_file(url, destination, expected_sha256=expected_sha256, timeout_seconds=timeout_seconds, user_agent=user_agent)
    headers = probe_headers(url, timeout_seconds=timeout_seconds, user_agent=user_agent)
    total = int(headers["content-length"]) if headers.get("content-length", "").isdigit() else None
    if total is None or headers.get("accept-ranges", "").lower() != "bytes" or connections < 2:
        return download_file(url, destination, expected_sha256=expected_sha256, timeout_seconds=timeout_seconds, user_agent=user_agent)

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    ledger = destination.with_name(destination.name + ".part.segments")
    segments = [(start, min(start + segment_bytes, total) - 1) for start in range(0, total, segment_bytes)]
    finished: set[int] = set()
    if partial.exists() and ledger.exists() and partial.stat().st_size == total:
        recorded = json.loads(ledger.read_text(encoding="utf-8"))
        if recorded["total"] == total and recorded["segment_bytes"] == segment_bytes:
            finished = set(recorded["finished"])
    else:
        with partial.open("wb") as handle:
            handle.truncate(total)

    def fetch(index: int) -> int:
        start, end = segments[index]
        request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Range": f"bytes={start}-{end}"}, method="GET")
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response, partial.open("r+b") as handle:
            if int(getattr(response, "status", response.getcode())) != 206:
                raise IncompleteDownload(f"{url}: the server answered a range request with the whole file")
            handle.seek(start)
            received = 0
            while block := response.read(CHUNK_BYTES):
                handle.write(block)
                received += len(block)
            handle.flush()
            os.fsync(handle.fileno())
        if received != end - start + 1:
            raise IncompleteDownload(f"{url}: bytes {start}-{end} arrived as {received} bytes")
        return index

    def record() -> None:
        ledger.write_text(json.dumps({"total": total, "segment_bytes": segment_bytes, "finished": sorted(finished)}), encoding="utf-8")

    failure: BaseException | None = None
    with ThreadPoolExecutor(max_workers=connections) as pool:
        futures = [pool.submit(fetch, index) for index in range(len(segments)) if index not in finished]
        for future in futures:
            try:
                finished.add(future.result())
            except (OSError, urllib.error.URLError, IncompleteDownload) as error:
                failure = failure or error
            record()
    if failure is not None:
        raise IncompleteDownload(f"{url}: {len(segments) - len(finished)} of {len(segments)} ranges are unfinished: {failure}") from failure
    ledger.unlink()
    return _finish(url, partial, destination, expected_sha256, headers)


def _finish(
    url: str, partial: Path, destination: Path, expected_sha256: str | None, headers: dict[str, str]
) -> Download:
    found_sha256 = sha256_file(partial)
    try:
        _require_digest(found_sha256, expected_sha256, destination)
    except DigestMismatch:
        # Resuming wrong bytes would only extend them.
        partial.unlink()
        raise
    size = partial.stat().st_size
    os.replace(partial, destination)
    return Download(url, destination, size, found_sha256, headers, "downloaded")


def _require_digest(found: str, expected: str | None, path: Path) -> None:
    if expected is not None and found != expected:
        raise DigestMismatch(f"{path}: expected sha256 {expected}, found {found}")


def _announced_total(content_range: str) -> int | None:
    """The total length in a `Content-Range` value such as `bytes 0-99/1234`, or None."""
    total = content_range.rsplit("/", 1)[-1] if "/" in content_range else ""
    return int(total) if total.isdigit() else None


def _expected_size(headers: Mapping[str, str], offset: int) -> int | None:
    total = _announced_total(headers.get("content-range", ""))
    if total is not None:
        return total
    length = headers.get("content-length")
    return None if length is None else offset + int(length)


__all__ = [
    "CHUNK_BYTES",
    "RECORDED_HEADERS",
    "USER_AGENT",
    "DigestMismatch",
    "Download",
    "IncompleteDownload",
    "download_file",
    "download_file_segmented",
    "probe_headers",
    "recorded_headers",
    "require_https",
]
