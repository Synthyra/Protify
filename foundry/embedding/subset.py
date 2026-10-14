"""Materialize unchanged rows from a pinned Hub store into the native feature store.

Range reads preserve every source descriptor and row identity. The receipt verifies
received bytes and local parts, while source whole-part hashes remain references.
No embedding inference runs here; canonical jobs can append missing rows afterward.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import pyarrow.parquet as parquet
import torch

from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from filelock import FileLock
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from foundry.digests import sha256_file
from foundry.serialization.atomic import write_json_atomic
from .canonical import MANIFEST_FILE, MANIFEST_SCHEMA_V3, manifest_features, open_canonical_view, sequence_inventory, write_manifest
from .hub import ROWS_INDEX, RowAddress, addresses_from_rows, hub_range_reader, read_rows
from .readers import ReadBudget
from .row_cache import commit_cached_rows


SCHEMA = "canonical_embedding_subset_v1"
ORIGIN_FILE = "subset_origin.json"
AUDIT_DIRECTORY = ".cache/subset_reads"


if TYPE_CHECKING:
    from fastplms.features import StoredFeature


def row_sha256(row: Any) -> str:
    """Hash tensor shapes, dtypes and values, including sparse positions when present."""
    tensors = {"values": row} if isinstance(row, torch.Tensor) else {
        name: getattr(row, name) for name in ("indices", "values", "positions") if getattr(row, name, None) is not None
    }  # dense/ragged (w,)/(l+2,w); sparse indices/values (nnz,)/(l+2,k), positions (nnz,) if present.
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        description = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        digest.update(description + b"\0")
        values = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)  # (stored_bytes,)
        digest.update(memoryview(values.numpy()))
    return digest.hexdigest()


def row_tensor_bytes(feature: StoredFeature, address: RowAddress) -> int:
    """Exact dense/top-k payload bytes, or a conservative CSR codebook-width bound."""
    size = torch.empty((), dtype=feature.dtype).element_size()  # scalar () determines bytes per stored value.
    if feature.layout == "ragged":
        return address.rows * feature.width * size
    if feature.layout == "ragged_topk":
        return address.rows * feature.sparse_count * (size + 4)
    if feature.layout == "csr":
        return feature.width * (size + 4 + (8 if feature.positions else 0))
    return feature.width * size


def row_batches(sequences: Sequence[str], addresses: Mapping[str, Mapping[str, RowAddress]], stream: str,
                feature: StoredFeature, *, max_rows: int, max_bytes: int) -> Iterator[list[str]]:
    """Keep source-part locality and cap materialized tensor bytes before each range read."""
    keys = sequence_inventory(sequences)
    ordered = sorted(keys, key=lambda text: (addresses[keys[text].sha256][stream].segment,
                                           addresses[keys[text].sha256][stream].part,
                                           addresses[keys[text].sha256][stream].row))
    batch: list[str] = []
    used = 0
    for sequence in ordered:
        address = addresses[keys[sequence].sha256][stream]
        estimate = row_tensor_bytes(feature, address)
        if estimate > max_bytes:
            raise ValueError(f"One {stream} row exceeds the subset read byte limit.")
        if batch and (len(batch) >= max_rows or used + estimate > max_bytes):
            yield batch
            batch, used = [], 0
        batch.append(sequence)
        used += estimate
    if batch:
        yield batch


def materialize_subset(repository: str, folder: str, sequences: Sequence[str], *, revision: str,
                       manifest_sha256: str, root: Path, streams: Sequence[str] | None = None,
                       max_rows: int = 64, max_bytes: int = 256 * 1024**2, workers: int = 8,
                       download: Callable[[str], Path] | None = None,
                       read_range: Callable[[str, int, int], bytes] | None = None) -> dict[str, Any]:
    """Copy requested rows, reuse committed hits, and verify source-value parity on cold open.

    The full original manifest is retained, including streams not requested yet.
    A durable batch audit precedes each commit. Resume copies only absent rows but
    rechecks every requested local value against its audited range-read digest.
    The root is writable and is neither a published mirror nor a coverage claim.
    """
    from fastplms.features import FeatureStore

    if re.fullmatch(r"[0-9a-f]{40}", revision) is None or re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None:
        raise ValueError("A subset requires an immutable revision and pinned manifest SHA-256.")
    if min(max_rows, max_bytes, workers) < 1:
        raise ValueError("Subset read limits and workers must be positive.")
    inventory = sequence_inventory(sequences)
    if not inventory:
        raise ValueError("A subset requires at least one sequence.")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source_files: dict[str, dict[str, Any]] = {}
    fetched: dict[str, Path] = {}

    def fetch(path: str) -> Path:
        if path in fetched:
            return fetched[path]
        local = download(path) if download is not None else Path(hf_hub_download(
            repository, path, repo_type="dataset", revision=revision))
        source_files[path] = {"sha256": sha256_file(local), "bytes": local.stat().st_size}
        fetched[path] = local
        return local

    with FileLock(str(root / ".subset.lock"), timeout=600):
        source_manifest = fetch(f"{folder}/{MANIFEST_FILE}")
        if sha256_file(source_manifest) != manifest_sha256:
            raise ValueError("Pinned source manifest checksum mismatch.")
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        if manifest.get("schema") != MANIFEST_SCHEMA_V3:
            raise ValueError("Only complete profile feature stores support a canonical subset.")
        names = tuple(manifest["streams"]) if streams is None else tuple(streams)
        if not names or len(names) != len(set(names)) or set(names) - set(manifest["streams"]):
            raise ValueError("Subset streams are empty, repeated or absent from the source manifest.")
        origin = {"schema": SCHEMA, "repository": repository, "revision": revision, "folder": folder,
                  "manifest_sha256": manifest_sha256, "published_mirror": False,
                  "canonical_append_compatible": True}
        origin_path = root / ORIGIN_FILE
        if origin_path.exists() and json.loads(origin_path.read_text(encoding="utf-8")) != origin:
            raise ValueError("Subset root already names another pinned source.")
        write_manifest(root, manifest)
        if not origin_path.exists():
            write_json_atomic(origin_path, origin)
        for stream in names:
            relative = f"{manifest['streams'][stream]['key']}/feature.json"
            payload = fetch(f"{folder}/{relative}").read_bytes()
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != payload:
                raise ValueError("Subset descriptor differs from the pinned source.")
            if not target.exists():
                target.write_bytes(payload)
        features = manifest_features(root, manifest, names)
        print(f"subset: locate {len(inventory):,} requested sequences in {len(names)} streams", flush=True)
        index = fetch(f"{folder}/{ROWS_INDEX}")
        columns = ["sha256", *(f"{name}__{field}" for name in names for field in ("segment", "part", "row", "rows"))]
        addresses = addresses_from_rows(parquet.read_table(index, columns=columns, use_threads=False,
            pre_buffer=False, memory_map=True, filters=[("sha256", "in", [key.sha256 for key in inventory.values()])]).to_pylist())
        print(f"subset: resolved {len(addresses):,} source sequence locators", flush=True)
        if any(set(names) - set(addresses.get(key.sha256, {})) for key in inventory.values()):
            raise KeyError("The pinned store lacks requested rows; compute them with the canonical embedding job.")
        audit_root = root / AUDIT_DIRECTORY
        audit_root.mkdir(parents=True, exist_ok=True)
        expected: dict[str, dict[str, str]] = {name: {} for name in names}
        for path in audit_root.glob("*.json"):
            audit = json.loads(path.read_text(encoding="utf-8"))
            if audit["stream"] in expected:
                expected[audit["stream"]].update(audit["row_sha256"])
        read = read_range or hub_range_reader(repository, revision)
        received_bytes, received_requests = 0, 0
        resumed_rows: dict[str, int] = {}
        source_parts: dict[str, dict[str, Any]] = {}
        markers: dict[str, dict[str, Any]] = {}
        range_lock = threading.Lock()
        budget = ReadBudget(max_rows=max_rows, max_padded_residues=max_rows * 2048, max_bytes=max_bytes * 2)
        for stream, feature in features.items():
            for key in inventory.values():
                address = addresses[key.sha256][stream]
                directory = f"{folder}/{feature.key}/segments/{address.segment}"
                marker_path = f"{directory}/run.json"
                if marker_path not in markers:
                    markers[marker_path] = json.loads(fetch(marker_path).read_text(encoding="utf-8"))
                part = next(item for item in markers[marker_path]["parts"] if int(item["part"]) == address.part)
                part_path = f"{directory}/part-{address.part:05d}.safetensors"
                source_parts[part_path] = {"sha256": part["sha256"], "verification": "referenced_not_downloaded",
                                           "marker_path": marker_path, "marker_sha256": source_files[marker_path]["sha256"]}
            store = FeatureStore.open(root, feature, deep_verify=False)
            absent = set(store.missing(list(inventory)))
            resumed_rows[stream] = len(inventory) - len(absent)
            batches = list(row_batches(list(inventory), addresses, stream, feature,
                                       max_rows=max_rows, max_bytes=max_bytes))
            for batch in tqdm(batches, desc=f"subset {stream}", mininterval=15):
                need_read = [text for text in batch if text in absent or inventory[text].sha256 not in expected[stream]]
                if not need_read:
                    continue
                ranges: list[dict[str, Any]] = []

                def measured_range(path: str, start: int, end: int) -> bytes:
                    nonlocal received_bytes, received_requests
                    payload = read(path, start, end)
                    with range_lock:
                        received_bytes += len(payload)
                        received_requests += 1
                        ranges.append({"path": path, "start": start, "end": start + len(payload), "requested_end": end,
                                       "sha256": hashlib.sha256(payload).hexdigest()})
                    return payload

                rows = read_rows(repository, folder, need_read, revision=revision, streams=[stream],
                                 workers=workers, download=fetch, read_range=measured_range, addresses=addresses)
                hashes = {inventory[text].sha256: row_sha256(rows[text][stream]) for text in need_read}
                locators: dict[str, dict[str, Any]] = {}
                for text in need_read:
                    key = inventory[text].sha256
                    address = addresses[key][stream]
                    directory = f"{folder}/{feature.key}/segments/{address.segment}"
                    part_path = f"{directory}/part-{address.part:05d}.safetensors"
                    locators[key] = {"segment": address.segment, "part": address.part,
                                     "row": address.row, "rows": address.rows, "source_part": part_path}
                fingerprint = hashlib.sha256((stream + "\n" + "\n".join(sorted(hashes))).encode()).hexdigest()
                audit = {"stream": stream, "row_sha256": hashes, "locators": locators,
                         "ranges": sorted(ranges, key=lambda item: (item["path"], item["start"], item["end"]))}
                # Persist transport evidence first, including a row whose later commit is interrupted.
                write_json_atomic(audit_root / f"{fingerprint}.json", audit)
                expected[stream].update(hashes)
                selected = {text: rows[text][stream] for text in need_read if text in absent}
                commit_cached_rows(root, store, stream, selected, addresses, fetch, manifest, folder=folder, revision=revision)
        copy_seconds = time.monotonic() - started
        verification_started = time.monotonic()
        # All previous reader handles are closed. Ignore receipts and verify committed parts and row identities in full.
        with open_canonical_view(root, inventory, streams=names, budget=budget, deep=True) as view:
            pins = view.content_pins()
            for stream, feature in features.items():
                for batch in tqdm(row_batches(list(inventory), addresses, stream, feature,
                                              max_rows=max_rows, max_bytes=max_bytes), desc=f"verify {stream}", mininterval=15):
                    rows = view.rows(stream, batch)  # native rows: (w,), (l+2,w), TopKRow or SparseRow.
                    if any(row_sha256(row) != expected[stream][inventory[text].sha256]
                           for text, row in zip(batch, rows, strict=True)):
                        raise ValueError("A local subset row differs from its pinned source range values.")
        receipt = {**origin, "streams": list(names), "proteins": len(inventory),
                   "selection_sha256": hashlib.sha256("\n".join(sorted(key.sha256 for key in inventory.values())).encode()).hexdigest(),
                   "feature_keys": {name: spec.key for name, spec in features.items()}, "source_files": source_files,
                   "tensor_payload_bytes_upper_bound": {name: sum(row_tensor_bytes(spec, addresses[key.sha256][name])
                                                                   for key in inventory.values()) for name, spec in features.items()},
                   "local_committed_bytes": {name: sum((root / features[name].key / path).stat().st_size for path in paths)
                                               for name, paths in pins.items()},
                   "source_parts": source_parts, "content_pins": pins, "resumed_rows": resumed_rows,
                   "received_range_bytes_this_invocation": received_bytes, "range_requests_this_invocation": received_requests,
                   "copy_seconds": copy_seconds, "cold_verification_seconds": time.monotonic() - verification_started,
                   "elapsed_seconds": time.monotonic() - started,
                   "verified": {"local_part_sha256": True, "full_sequence_and_crop_identity": True,
                                "every_row_source_value_sha256": True, "source_whole_part_sha256": False}}
        write_json_atomic(root / f"subset-{receipt['selection_sha256'][:16]}.json", receipt)
        return receipt
