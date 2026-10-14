"""Measure sparse Parquet release loading and training-label assembly on a Linux fixture.

The artificial sequences and labels establish storage behavior only. They are not an
experimental source, a homology-qualified split, or evidence of complete training capacity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path
from typing import Any
from tqdm.auto import tqdm

from foundry.digests import sha256_file
from protify.data import declared_dataset, label_batches
from protify.data.declared_dataset import SPARSE_ENCODING, SPARSE_SCHEMA, SPLIT_NAMES, declared_target_fields, load_declared_dataset
from protify.data.label_batches import SparseObservedTargets, collate_sparse_labels
from protify.testing_suite.run_label_storage_pilot import ProceduralRows, require, resident_bytes


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def fixture_row(index: int, width: int) -> dict[str, Any]:
    """An ordinal encodes one unique artificial protein and its independently checkable truth."""
    ordinal = index
    residues = []
    for _ in range(12):
        ordinal, remainder = divmod(ordinal, len(AMINO_ACIDS))
        residues.append(AMINO_ACIDS[remainder])
    require(ordinal == 0, "fixture sequence space exhausted")
    sequence = "M" + "".join(residues)
    positive, negative = index % width, (index + 7) % width
    observed = index % 11 != 0
    complete = observed and index % 13 == 0
    return {
        "observation_id": f"assay_{index}", "cluster_id": f"fixture_cluster_{index}",
        "protein_id": hashlib.sha256(sequence.encode("ascii")).hexdigest(), "seqs": sequence,
        "labels": {"width": width, "indices": [positive] if observed else [], "values": [1.0] if observed else []},
        "label_mask": {"default_observed": complete, "exceptions": sorted((positive, negative)) if observed and not complete else []},
    }


def write_fixture(root: Path, rows: int, width: int) -> tuple[Path, dict[str, tuple[int, int]]]:
    """Stream fresh artifacts; no source row allocates a dense target or mask."""
    schema = pa.schema([
        *(pa.field(name, pa.string(), nullable=False) for name in ("observation_id", "protein_id", "cluster_id", "seqs")),
        *declared_target_fields("multilabel", compact=True),
    ])
    manifest = {
        "schema": SPARSE_SCHEMA, "label_encoding": SPARSE_ENCODING,
        "dataset_id": "sparse_release_software_fixture", "endpoint": "artificial_observed_truth",
        "task_type": "multilabel", "units": "binary",
        "target_transform": {"name": "identity", "source_units": "binary", "inverse": "identity"},
        "label_vocabulary": [f"fixture_{index}" for index in range(width)],
        "evidence_policy_sha256": "e" * 64, "split_audit_sha256": "a" * 64, "splits": {},
    }
    boundaries, start = {}, 0
    for split, count in zip(SPLIT_NAMES, (rows * 17 // 20, rows // 20, rows // 20, rows // 20), strict=True):
        path = root / f"{split}.parquet"
        stop = start + count
        with pq.ParquetWriter(path, schema, compression="zstd") as writer:
            for first in tqdm(range(start, stop, 4096), desc=f"Write {split}", unit="batches"):
                batch = [fixture_row(index, width) for index in range(first, min(first + 4096, stop))]
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        manifest["splits"][split] = {"path": path.name, "rows": count, "sha256": sha256_file(path)}
        boundaries[split] = start, stop
        start = stop
    path = root / "dataset.json"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path, boundaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1_200_000)
    parser.add_argument("--width", type=int, default=9951)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rss-growth-gib", type=int, default=4)
    arguments = parser.parse_args()
    require(arguments.root.is_absolute() and not arguments.root.exists(), "root must be a fresh absolute directory")
    require(arguments.rows >= 20 and arguments.rows % 20 == 0 and arguments.width > 7, "invalid fixture dimensions")
    require(0 < arguments.batch_size <= arguments.rows // 20 and arguments.max_rss_growth_gib > 0, "invalid batch size or budget")
    baseline_rss = resident_bytes()
    baseline_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    started = time.monotonic()
    arguments.root.mkdir(parents=True)
    manifest_path, boundaries = write_fixture(arguments.root, arguments.rows, arguments.width)
    written = time.monotonic()
    release = load_declared_dataset(manifest_path, sha256_file(manifest_path))
    loaded = time.monotonic()
    loaded_rss = resident_bytes()
    originals = ProceduralRows(arguments.rows, arguments.width, masks=False)
    original_masks = ProceduralRows(arguments.rows, arguments.width, masks=True)
    summaries = {}
    for split in tqdm(SPLIT_NAMES, desc="Assemble and verify labels", unit="splits"):
        dataset = release.splits[split]
        start, stop = boundaries[split]
        storage = SparseObservedTargets.from_rows(dataset["labels"], dataset["label_mask"])
        observed = sum(index % 11 != 0 for index in range(start, stop))
        complete = sum(index % 11 != 0 and index % 13 == 0 for index in range(start, stop))
        require(storage.values.shape == (stop - start, arguments.width), "stored corpus shape changed")
        require(storage.values.nnz == observed and storage.mask_entries.nnz == 2 * (observed - complete), "source observation counts changed")
        require(int(storage.complete_rows.sum()) == complete, "complete observation status changed")
        require(storage.storage_nbytes < (stop - start) * 100, "storage grew with vocabulary width")
        indices = np.linspace(0, stop - start - 1, num=arguments.batch_size, dtype=np.int64).tolist()
        batch = collate_sparse_labels([storage.row(index) for index in indices])  # (b, c) targets and masks
        expected_masks = np.stack([original_masks[start + index] for index in indices])  # (b, c)
        expected_labels = np.where(expected_masks, np.stack([originals[start + index] for index in indices]), 0)  # (b, c)
        require(np.array_equal(batch["labels"].numpy(), expected_labels), "selected targets differ from independent dense fixture")
        require(np.array_equal(batch["label_mask"].numpy(), expected_masks), "selected masks differ from independent dense fixture")
        summaries[split] = {
            "rows": len(dataset), "positive_entries": int(storage.values.nnz), "stored_mask_entries": int(storage.mask_entries.nnz),
            "complete_rows": complete, "compact_array_bytes": storage.storage_nbytes,
            "selected_rows": indices, "independent_batch_parity": True,
        }
        del storage, batch, expected_labels, expected_masks
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(peak - baseline_peak < arguments.max_rss_growth_gib * 1024**3, "source-loading pilot exceeded its measured RSS growth budget")
    receipt = {
        "schema": "sparse_release_storage_pilot_v1", "kind": "procedural_software_fixture_only",
        "rows": arguments.rows, "width": arguments.width, "splits": summaries,
        "manifest_sha256": sha256_file(manifest_path), "artifacts": release.manifest["splits"],
        "parquet_bytes": sum((arguments.root / record["path"]).stat().st_size for record in release.manifest["splits"].values()),
        "equivalent_dense_float32_targets_and_boolean_masks_bytes": arguments.rows * arguments.width * 5,
        "baseline_rss_bytes": baseline_rss, "rss_after_loading_bytes": loaded_rss,
        "baseline_peak_rss_bytes": baseline_peak, "peak_rss_bytes": peak, "peak_rss_growth_bytes": peak - baseline_peak,
        "rss_growth_limit_bytes": arguments.max_rss_growth_gib * 1024**3,
        "write_seconds": written - started, "load_seconds": loaded - written, "total_seconds": time.monotonic() - started,
        "implementation_sha256": {"declared_dataset": sha256_file(declared_dataset.__file__), "label_batches": sha256_file(label_batches.__file__)},
        "pilot_sha256": sha256_file(__file__), "dense_reference_sha256": sha256_file(Path(__file__).with_name("run_label_storage_pilot.py")),
        "scope": "sparse artifact writing, verified loading, compact assembly, and selected batches; source science, homology, full training and evaluation memory remain unqualified",
    }
    with (arguments.root / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
