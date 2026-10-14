"""Compare the files of a folder with those of the Hub dataset repository it was pushed to, by name, size and digest.

Nothing is downloaded. The Hub records the SHA-256 of each file it stored through LFS and checks the bytes against it when they arrive. For
any other file it records the git blob id: the SHA-1 of `blob <size>`, a NUL byte and the content. Reading those records back says whether
the repository holds exactly the files of the folder. Moved here from `production_suite.hub_copy` on its second consumer, the embedding
stores' push (`foundry.embedding.hub`).
"""

from __future__ import annotations

import hashlib
import sys

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from huggingface_hub import RepoFile
from tqdm.auto import tqdm


READ_BYTES = 16 << 20
PROGRESS_SECONDS = 30.0
HUB_MANAGED = ".gitattributes"  # the Hub appends the LFS lines to it, so its bytes are never the folder's


def file_digests(path: Path) -> tuple[str, str]:
    """The SHA-256 of the file's bytes and its git blob id, from one read of the file."""
    sha256 = hashlib.sha256()
    blob_id = hashlib.sha1(f"blob {path.stat().st_size}\0".encode(), usedforsecurity=False)  # git names a blob by its SHA-1
    with path.open("rb") as handle:
        while chunk := handle.read(READ_BYTES):
            sha256.update(chunk)
            blob_id.update(chunk)
    return sha256.hexdigest(), blob_id.hexdigest()


def hub_copy_differences(api: Any, repo: str, root: Path, files: Sequence[Path], *, digests: bool = True, workers: int = 8) -> list[str]:
    """What the dataset repository lacks or holds differently from `files`, the published files under `root`; empty when it is a copy.

    The repository may hold more than the folder does (its card, its `.gitattributes`). With ``digests`` false only names and sizes are
    compared, which reads no local file.
    """
    remote = {entry.path: entry for entry in api.list_repo_tree(repo, repo_type="dataset", recursive=True) if isinstance(entry, RepoFile)}
    local = {name: path for path in files if (name := path.relative_to(root).as_posix()) != HUB_MANAGED}
    differences = [f"{name} is not in {repo}" for name in local if name not in remote]
    sizes = {name: path.stat().st_size for name, path in local.items() if name in remote}
    differences += [f"{name} has {size} bytes here and {remote[name].size} in {repo}" for name, size in sizes.items() if size != remote[name].size]
    if not digests:
        return differences
    same_size = [(name, local[name]) for name, size in sizes.items() if size == remote[name].size]
    with ThreadPoolExecutor(workers) as pool:
        found = tqdm(pool.map(lambda item: file_digests(item[1]), same_size), total=len(same_size), desc=f"digests against {repo}", unit="file",
                     mininterval=PROGRESS_SECONDS, file=sys.stderr)
        for (name, _), (sha256, blob_id) in zip(same_size, found, strict=True):
            lfs = remote[name].lfs
            here, there = (sha256, lfs.sha256) if lfs is not None else (blob_id, remote[name].blob_id)
            if here != there:
                differences.append(f"{name} is {here} here and {there} in {repo}")
    return differences


__all__ = ["file_digests", "hub_copy_differences"]
