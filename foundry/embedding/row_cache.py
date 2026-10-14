"""Copy selected Hub rows with original complete-contract identities and commit validation."""

from __future__ import annotations

import gzip
import hashlib
import json

from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from .canonical import manifest_features, sequence_inventory
from .contracts import FeatureSpec
from .hub import RowAddress
from .readers import checked_row_identity


@lru_cache(maxsize=64)
def source_part_identities(download: Callable[[str], Path], directory: str, number: int) -> tuple[list[str], list[dict[str, Any]]]:
    """Read and hash the pinned source sidecar once per recently used part."""
    marker = json.loads(download(f"{directory}/run.json").read_text())
    part = next(part for part in marker["parts"] if part["part"] == number)
    sidecar = part["row_metadata"]
    if sidecar["file"] != f"part-{number:05d}.rows.json.gz":
        raise ValueError("source identity sidecar filename differs from its part")
    payload = download(f"{directory}/{sidecar['file']}").read_bytes()
    if hashlib.sha256(payload).hexdigest() != sidecar["sha256"]:
        raise ValueError("source identity sidecar differs from its commit hash")
    identities = json.loads(gzip.decompress(payload))
    if len(identities) != len(part["digests"]):
        raise ValueError("source identity count differs from its committed rows")
    return part["digests"], identities


def commit_cached_rows(root: Path, store: Any, name: str, rows: Mapping[str, Any],
                       addresses: Mapping[str, Mapping[str, RowAddress]], download: Callable[[str], Path],
                       manifest: dict[str, Any], *, folder: str, revision: str) -> None:
    """Copy identities and validate their unchanged contract at the commit boundary."""
    if not rows:
        return
    inventory = sequence_inventory(list(rows))
    feature = FeatureSpec(store.spec.descriptor)
    grouped: dict[tuple[str, int], list[tuple[int, str, RowAddress]]] = {}
    for index, (sequence, key) in enumerate(inventory.items()):
        address = addresses[key.sha256][name]
        directory = f"{folder}/{store.spec.key}/segments/{address.segment}"
        grouped.setdefault((directory, address.part), []).append((index, sequence, address))
    # The two streams can have different shard orders. Resolve each part once for this
    # batch instead of thrashing the bounded LRU when hundreds of parts are interleaved.
    by_index: dict[int, dict[str, Any]] = {}
    for (directory, number), requests in grouped.items():
        digests, records = source_part_identities(download, directory, number)
        for index, sequence, address in requests:
            key = inventory[sequence]
            if address.row < 0 or address.row >= len(digests) or digests[address.row] != key.sha256:
                raise ValueError("source row address differs from its committed sequence")
            identity = records[address.row]
            bound = checked_row_identity(identity, feature, key)
            if store.spec.layout in {"ragged", "ragged_topk"} and address.rows != len(bound.retained_positions):
                raise ValueError("source row count differs from its identity")
            by_index[index] = identity
    identities = [by_index[index] for index in range(len(inventory))]

    def unchanged_contract() -> None:
        if json.loads((root / "canonical.json").read_text()) != manifest:
            raise ValueError("cache manifest changed before commit")
        if manifest_features(root, manifest, [name])[name] != store.spec:
            raise ValueError("cache descriptor changed before commit")
        for sequence, identity in zip(inventory, identities, strict=True):
            checked_row_identity(identity, feature, inventory[sequence])

    fingerprint = hashlib.sha256("\n".join(key.sha256 for key in inventory.values()).encode()).hexdigest()[:16]
    with store.segment("rows-" + fingerprint, {"origin_revision": revision, "stream": name}, before_commit=unchanged_contract) as writer:
        writer.append_bounded(list(inventory), list(rows.values()), row_metadata=identities, max_tensor_bytes=256 * 1024**2)
