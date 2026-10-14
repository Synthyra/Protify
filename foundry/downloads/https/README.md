---
id: foundry.downloads.https
kind: library_module
status: active
aliases: [foundry.downloads.https]
---
# foundry.downloads.https

One file from a public HTTPS address, downloaded once, resumably, and checked before it is used.

| Name | Does |
|---|---|
| `download_file(url, destination, *, expected_sha256=None, timeout_seconds=120.0, user_agent=USER_AGENT)` | The file at `destination`, downloaded unless present, and a `Download` record: `url`, `path`, `size_bytes`, `sha256`, `response_headers`, and `status` ("downloaded" or "cached") |
| `download_file_segmented(url, destination, *, connections=8, segment_bytes=256 MiB, expected_sha256=None, ...)` | The same file fetched as byte ranges over several connections, for hosts that cap each connection (EBI's FTP mirror gives about 2 MB/s). Finished ranges are listed in `<destination>.part.segments`, so an interrupted call redoes only the rest; a server that does not accept ranges is read as one stream |
| `probe_headers(url)` | The recorded headers of a HEAD request, or `{}` when it fails; for a manifest, never for deciding whether a file exists |
| `recorded_headers(headers)` | The headers in `RECORDED_HEADERS`, lowercased; never a cookie or an authorization header |
| `require_https(url)` | `ValueError` unless HTTPS, or plain HTTP to a loopback address |
| `DigestMismatch` (a `ValueError`), `IncompleteDownload` (an `OSError`) | Wrong bytes, and fewer bytes than the server announced |

The rules:

- Bytes arrive in `<destination>.part` and are renamed into place only once their length matches
  what the server announced and their digest matches `expected_sha256`, so the destination never
  holds an interrupted or wrong transfer. Wrong bytes are removed; short ones stay for the next
  call to resume.
- A resume asks for the rest with a `Range` header. A server that answers 200 instead restarts
  the file, and one that answers 416 finishes it only when its `Content-Range` total equals what
  is already held.
- A present file is not downloaded again, and its digest is checked when one is expected.

Taken on 2026-09-26 from phenomics' `data/download.py`, which keeps its source table and its
per-source license manifest and now calls this. `foundry.downloads.string_db` calls it too.

Tests: `tests/tier1_unit/test_downloads_https.py`, against a loopback server that serves byte
ranges.
