"""One file from the Hugging Face Hub at a pinned revision, checked against its digest.

Some files the code reads are neither source nor build output: runtime data a service loads, an
evidence table a report draws on, a generated lookup module. They are too large for the tree, so
they live in a private Hub repository and reach a machine through the Hugging Face cache instead
of an unmanaged local copy.

- **The pin is a commit hash, never a branch.** A branch moves, and an offline cache cannot
  resolve one. Forty hex characters or the pin is refused.
- **The digest is checked on every call,** a cache hit included, so a truncated, replaced, or
  half-written file fails here and not inside the code that reads it.
- **A cached file is read without the network.** `cached_path` answers from the cache alone, so an
  offline suite can run against a file it already holds and skip when it does not.

Authentication belongs to `huggingface_hub`, through `HF_TOKEN` or the token `hf auth login`
stored. Nothing here accepts, reads, or returns a token.

The package also holds what pushes share: `hub_copy_differences` (`copies`), which checks a pushed
folder against the repository, and `upload_resuming`, `hub_call` and `rate_limited` (`rate_limit`),
which outlast the account's rate limit.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from huggingface_hub import hf_hub_download, try_to_load_from_cache

from foundry.digests import sha256_file as digest
from .copies import file_digests, hub_copy_differences
from .rate_limit import hub_call, rate_limited, upload_resuming


COMMIT_HASH = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
REPO_TYPES = frozenset({"dataset", "model"})


class HubFileError(RuntimeError):
    """A pinned Hub file could not be provided as declared."""


class DigestMismatch(HubFileError):
    """The bytes on disk are not the bytes the pin names."""


class HubFileUnavailable(HubFileError):
    """The file is not in the cache and could not be downloaded."""


@dataclass(frozen=True, slots=True)
class HubFile:
    """One file of one Hub repository at one immutable revision.

    Declared beside the code that reads it, so the pin and the digest sit in source and a reader
    can see which bytes the code expects.
    """

    repo_id: str
    filename: str
    revision: str
    sha256: str
    repo_type: Literal["dataset", "model"] = "dataset"

    def __post_init__(self) -> None:
        if len(self.repo_id.split("/")) != 2:
            raise ValueError(f"repo_id must be 'namespace/name', got {self.repo_id!r}")
        if not COMMIT_HASH.fullmatch(self.revision):
            raise ValueError(
                f"{self.repo_id} {self.filename}: revision must be a full commit hash, got {self.revision!r}"
            )
        if not SHA256.fullmatch(self.sha256):
            raise ValueError(f"{self.repo_id} {self.filename}: sha256 must be 64 hex characters")
        if self.repo_type not in REPO_TYPES:
            raise ValueError(f"repo_type must be one of {sorted(REPO_TYPES)}, got {self.repo_type!r}")

    def cached_path(self) -> Path | None:
        """The cache's copy of this revision's file, verified, or None when it is not cached."""
        cached = try_to_load_from_cache(
            repo_id=self.repo_id, filename=self.filename, repo_type=self.repo_type, revision=self.revision
        )
        if not isinstance(cached, str):
            # None means "not cached"; the sentinel means "known absent from the revision".
            return None
        path = Path(cached)
        self.verify(path)
        return path

    def path(self) -> Path:
        """This revision's file in the cache, downloading it once if the cache lacks it."""
        cached = self.cached_path()
        if cached is not None:
            return cached
        try:
            downloaded = hf_hub_download(
                repo_id=self.repo_id, filename=self.filename, repo_type=self.repo_type, revision=self.revision
            )
        except Exception as error:
            raise HubFileUnavailable(
                f"{self.filename} is not in the Hugging Face cache and could not be downloaded from "
                f"{self.repo_id} at {self.revision[:12]}. The repository is private: run `hf auth login` "
                f"once, or set HF_TOKEN, and make sure this machine is online."
            ) from error
        path = Path(downloaded)
        self.verify(path)
        return path

    def verify(self, path: Path) -> None:
        """Raise unless `path` holds exactly the bytes this pin names."""
        found_sha256 = digest(path)
        if found_sha256 != self.sha256:
            raise DigestMismatch(
                f"{path} is not {self.repo_id} {self.filename} at {self.revision[:12]}: "
                f"expected sha256 {self.sha256}, found {found_sha256}"
            )


__all__ = [
    "DigestMismatch", "HubFile", "HubFileError", "HubFileUnavailable", "digest", "file_digests", "hub_call",
    "hub_copy_differences", "rate_limited", "upload_resuming",
]
