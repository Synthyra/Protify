"""Publish a dataset folder to the Hugging Face Hub as one verified commit.

The folder is the release. The published revision holds exactly the folder's files, plus the
Hub's own `.gitattributes` when the folder brings none. Remote files the folder no longer holds
are deleted in the same commit, so stale shards cannot join a split. A revision is reported only
after every file has been downloaded back from it and compared byte for byte.

- A new repository is created private and is deleted again if anything fails before it verifies.
- An existing repository is committed to a throwaway branch and verified there first, so `main`
  only ever moves to a revision that already passed. Both commits upload from one snapshot of
  the folder, so they carry the same bytes even if the folder changes in between.
- Visibility is applied last, after verification.

Authentication belongs to `huggingface_hub`, through `HF_TOKEN` or a cached login. Nothing here
accepts, reads, or returns a token, and a file that contains the value of a credential-named
environment variable stops the publish before anything is uploaded.

Promoted from annotation_vocabulary's release publisher; gem_integration is the second consumer.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import uuid

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download

from foundry.logging import is_credential


REPO_TYPE = "dataset"
GITATTRIBUTES = ".gitattributes"
STAGING_PREFIX = "foundry-staging-"
# The line the Hub appends to `.gitattributes` for a file it stored through LFS: one literal
# path, never a glob, a comment, or a negation.
LFS_LINE = re.compile(rb"[^\s#!*?\[\]][^\s*?\[\]]* filter=lfs diff=lfs merge=lfs -text(?:\r?\n)?")
# Shorter values are too common to mean anything when found in a file.
MINIMUM_SECRET_LENGTH = 8
CHUNK_BYTES = 1 << 20

# Called with each revision that passed byte verification, while the downloaded copies still
# exist, keyed by path in the repository. Raising rejects the revision: `main` does not move to
# it, and a new repository is deleted.
RevisionCheck = Callable[[str, Mapping[str, Path]], None]


class PublicationError(RuntimeError):
    """A publish stopped. Unless the message says otherwise, `main` did not move."""


class IntegrityError(PublicationError):
    """The revision on the Hub does not hold the folder's files."""


@dataclass(frozen=True, slots=True)
class VerifiedFile:
    """One published file: the folder's bytes, and the bytes downloaded back from the revision."""

    path: str
    sha256: str
    size: int
    published_sha256: str
    published_size: int

    @property
    def byte_identical(self) -> bool:
        """False only for a `.gitattributes` the Hub appended LFS lines to, which is allowed."""
        return self.sha256 == self.published_sha256 and self.size == self.published_size


@dataclass(frozen=True, slots=True)
class PublishedRevision:
    """The repository and the immutable revision a dataset manifest's `hf` entry pins."""

    repo_id: str
    revision: str
    commit_url: str
    private: bool
    files: tuple[VerifiedFile, ...]


@dataclass(frozen=True, slots=True)
class _ReleaseFile:
    path: str  # path in the repository, posix
    source: Path  # the bytes that are uploaded: the folder's file, or its snapshot
    sha256: str
    size: int


def publish_dataset(
    folder: Path,
    repo_id: str,
    *,
    commit_message: str,
    private: bool,
    check: RevisionCheck | None = None,
) -> PublishedRevision:
    """Make `repo_id` hold exactly `folder` at a new revision, verified, and return that revision.

    `private` has no default, so no caller publishes publicly by leaving it out. `check` runs
    project-specific verification, such as loading each dataset config, against a revision
    before `main` moves to it.
    """
    _require_repo_id(repo_id)
    if not isinstance(private, bool):
        raise TypeError("private must be True or False")
    message = commit_message.strip()
    if not message:
        raise ValueError("commit_message must not be empty")
    files = _release_files(Path(folder))

    api = HfApi()
    exists = api.repo_exists(repo_id=repo_id, repo_type=REPO_TYPE)
    if not isinstance(exists, bool):
        raise TypeError("Hugging Face repo_exists must return a boolean")

    with tempfile.TemporaryDirectory(prefix="foundry-publish-") as scratch:
        if exists:
            snapshot = _snapshot(files, Path(scratch), {file.path for file in files})
            commit, verified = _publish_existing(api, repo_id, snapshot, message, check)
        else:
            # One commit uploads straight from the folder, and byte verification catches a
            # file that changes mid-upload. `.gitattributes` is the exception: the Hub may
            # append to it, so it is not compared exactly and its bytes are pinned instead.
            pinned = _snapshot(files, Path(scratch), {GITATTRIBUTES})
            commit, verified = _publish_new(api, repo_id, pinned, message, check)
        revision = _response_text(commit, "oid")
        commit_url = _response_text(commit, "commit_url")

    api.update_repo_settings(repo_id=repo_id, repo_type=REPO_TYPE, private=private)
    return PublishedRevision(repo_id, revision, commit_url, private, verified)


def _publish_new(
    api: HfApi,
    repo_id: str,
    files: tuple[_ReleaseFile, ...],
    message: str,
    check: RevisionCheck | None,
) -> tuple[object, tuple[VerifiedFile, ...]]:
    """Create the repository private, and delete it again unless the commit verifies."""
    api.create_repo(repo_id=repo_id, repo_type=REPO_TYPE, private=True, exist_ok=False)
    try:
        parent = _head(api, repo_id)
        remote = set() if parent is None else _list_files(api, repo_id, parent)
        commit = _commit(api, repo_id, files, remote, message, parent_commit=parent, revision=None)
        verified = _verify(api, repo_id, _response_text(commit, "oid"), files, check)
    except BaseException:
        try:
            api.delete_repo(repo_id=repo_id, repo_type=REPO_TYPE)
        except Exception as cleanup_error:
            raise PublicationError(
                f"publishing to the new repository {repo_id} failed and deleting it failed too; delete it by hand"
            ) from cleanup_error
        raise
    return commit, verified


def _publish_existing(
    api: HfApi,
    repo_id: str,
    files: tuple[_ReleaseFile, ...],
    message: str,
    check: RevisionCheck | None,
) -> tuple[object, tuple[VerifiedFile, ...]]:
    """Verify the change on a throwaway branch, then apply the same change to `main`.

    Both commits name the same parent, so if `main` moved in between, the second is refused
    rather than applied on top of work this publish never saw.
    """
    parent = _head(api, repo_id)
    if parent is None:
        raise PublicationError(f"{repo_id} exists but has no commit on main to stage from")
    remote = _list_files(api, repo_id, parent)
    branch = f"{STAGING_PREFIX}{uuid.uuid4().hex}"
    api.create_branch(repo_id=repo_id, branch=branch, repo_type=REPO_TYPE, revision=parent, exist_ok=False)
    try:
        staged = _commit(api, repo_id, files, remote, message, parent_commit=parent, revision=branch)
        _verify(api, repo_id, _response_text(staged, "oid"), files, check)
    finally:
        api.delete_branch(repo_id=repo_id, branch=branch, repo_type=REPO_TYPE)

    commit = _commit(api, repo_id, files, remote, message, parent_commit=parent, revision="main")
    return commit, _verify(api, repo_id, _response_text(commit, "oid"), files, check)


def _commit(
    api: HfApi,
    repo_id: str,
    files: tuple[_ReleaseFile, ...],
    remote: set[str],
    message: str,
    *,
    parent_commit: str | None,
    revision: str | None,
) -> object:
    """One commit that adds every file and deletes every remote file the folder does not hold."""
    local = {file.path for file in files}
    operations: list[CommitOperationAdd | CommitOperationDelete] = [
        CommitOperationAdd(path_in_repo=file.path, path_or_fileobj=file.source) for file in files
    ]
    # A folder without `.gitattributes` keeps the Hub's, which decides what is stored through LFS.
    operations += [CommitOperationDelete(path_in_repo=path) for path in sorted(remote - local - {GITATTRIBUTES})]
    arguments: dict[str, str] = {}
    if parent_commit is not None:
        arguments["parent_commit"] = parent_commit
    if revision is not None:
        arguments["revision"] = revision
    return api.create_commit(
        repo_id=repo_id, repo_type=REPO_TYPE, operations=operations, commit_message=message, **arguments
    )


def _verify(
    api: HfApi,
    repo_id: str,
    revision: str,
    files: tuple[_ReleaseFile, ...],
    check: RevisionCheck | None,
) -> tuple[VerifiedFile, ...]:
    """Download every file from `revision` and compare it with the folder, then run `check`."""
    local = {file.path for file in files}
    published = _list_files(api, repo_id, revision)
    missing = sorted(local - published)
    unexpected = sorted(published - local - ({GITATTRIBUTES} - local))
    if missing or unexpected:
        raise IntegrityError(
            f"{repo_id} at {revision} holds a different file set than the folder: "
            f"missing={missing}, unexpected={unexpected}"
        )

    verified: list[VerifiedFile] = []
    downloaded: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="foundry-verify-") as cache:
        for file in files:
            path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file.path,
                    repo_type=REPO_TYPE,
                    revision=revision,
                    cache_dir=cache,
                    force_download=True,
                )
            )
            if not path.is_file():
                raise IntegrityError(f"Downloaded path for {file.path!r} is not a file")
            sha256, size, _ = _scan(path, {})
            if file.path == GITATTRIBUTES:
                _require_hub_managed_gitattributes(file.source.read_bytes(), path.read_bytes())
            elif size != file.size:
                raise IntegrityError(f"Downloaded size mismatch for {file.path!r}: expected {file.size}, got {size}")
            elif sha256 != file.sha256:
                raise IntegrityError(f"Downloaded SHA-256 mismatch for {file.path!r}")
            verified.append(VerifiedFile(file.path, file.sha256, file.size, sha256, size))
            downloaded[file.path] = path
        if check is not None:
            check(revision, downloaded)
    return tuple(verified)


def _require_hub_managed_gitattributes(uploaded: bytes, published: bytes) -> None:
    """The Hub may append an LFS line for each file it stored through LFS, and nothing else."""
    if published == uploaded:
        return
    lines = published.splitlines(keepends=True)
    if len(set(lines)) != len(lines):
        raise IntegrityError(".gitattributes at the revision repeats a line")
    bare = [line.rstrip(b"\r\n") for line in lines]
    kept = set(uploaded.splitlines())
    remaining = iter(bare)
    if not all(any(candidate == wanted for candidate in remaining) for wanted in uploaded.splitlines()):
        raise IntegrityError(".gitattributes at the revision does not keep the uploaded lines in order")
    added = [line for line, text in zip(lines, bare, strict=True) if text not in kept]
    unexpected = [line for line in added if not LFS_LINE.fullmatch(line)]
    if unexpected:
        raise IntegrityError(f".gitattributes at the revision adds lines that are not LFS tracking: {unexpected!r}")


def _release_files(folder: Path) -> tuple[_ReleaseFile, ...]:
    """Every file in the folder, hashed, after checking none holds a live credential.

    `.git` and `.cache/huggingface` are skipped, as `huggingface_hub` skips them. A symbolic
    link stops the publish: it can point anywhere, including at a credential file.
    """
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a directory")
    secrets = _live_credentials()
    files: list[_ReleaseFile] = []
    for path in sorted(folder.rglob("*"), key=lambda item: item.relative_to(folder).as_posix()):
        relative = path.relative_to(folder)
        if ".git" in relative.parts or relative.parts[:2] == (".cache", "huggingface"):
            continue
        if path.is_symlink():
            raise PublicationError(f"{relative.as_posix()} is a symbolic link; a release holds regular files only")
        if not path.is_file():
            continue
        sha256, size, leaked = _scan(path, secrets)
        if leaked:
            raise PublicationError(
                f"{relative.as_posix()} contains the value of {', '.join(leaked)}; nothing was published"
            )
        files.append(_ReleaseFile(relative.as_posix(), path, sha256, size))
    if not files:
        raise ValueError(f"{folder} holds no files to publish")
    return tuple(files)


def _live_credentials() -> dict[str, bytes]:
    """Values of credential-named environment variables, keyed by name. Never logged."""
    return {
        name: value.encode()
        for name, value in os.environ.items()
        if is_credential(name) and len(value) >= MINIMUM_SECRET_LENGTH
    }


def _scan(path: Path, secrets: Mapping[str, bytes]) -> tuple[str, int, list[str]]:
    """sha256, size, and the names of any `secrets` whose value occurs in the file."""
    digest = hashlib.sha256()
    size = 0
    found: set[str] = set()
    # Keep enough of the previous chunk that a value split across two chunks is still seen.
    overlap = max((len(value) for value in secrets.values()), default=1) - 1
    tail = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
            size += len(chunk)
            window = tail + chunk
            found.update(name for name, value in secrets.items() if value in window)
            tail = window[-overlap:] if overlap else b""
    return digest.hexdigest(), size, sorted(found)


def _snapshot(files: tuple[_ReleaseFile, ...], scratch: Path, paths: set[str]) -> tuple[_ReleaseFile, ...]:
    """Copy the named files into `scratch`, and check each copy is the bytes that were hashed."""
    snapshot: list[_ReleaseFile] = []
    for file in files:
        if file.path not in paths:
            snapshot.append(file)
            continue
        copy = scratch / file.path
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file.source, copy)
        sha256, size, _ = _scan(copy, {})
        if (sha256, size) != (file.sha256, file.size):
            raise IntegrityError(f"{file.path} changed while the publish was reading it")
        snapshot.append(_ReleaseFile(file.path, copy, sha256, size))
    return tuple(snapshot)


def _head(api: HfApi, repo_id: str) -> str | None:
    sha = getattr(api.repo_info(repo_id=repo_id, repo_type=REPO_TYPE), "sha", None)
    if sha is not None and (not isinstance(sha, str) or not sha.strip()):
        raise PublicationError(f"the Hub returned an invalid revision for {repo_id}")
    return sha


def _list_files(api: HfApi, repo_id: str, revision: str) -> set[str]:
    paths = api.list_repo_files(repo_id=repo_id, repo_type=REPO_TYPE, revision=revision)
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise PublicationError(f"the Hub returned an invalid file list for {repo_id} at {revision}")
    return set(paths)


def _response_text(response: object, attribute: str) -> str:
    value = getattr(response, attribute, None)
    if not isinstance(value, str) or not value.strip():
        raise PublicationError(f"the Hub's commit response has no {attribute}")
    return value


def _require_repo_id(repo_id: str) -> None:
    if not isinstance(repo_id, str):
        raise TypeError("repo_id must be a string")
    parts = repo_id.split("/")
    if repo_id.strip() != repo_id or len(parts) != 2 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"repo_id must be 'namespace/name', got {repo_id!r}")
