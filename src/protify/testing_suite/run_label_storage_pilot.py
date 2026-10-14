"""Measure compact training-label storage on a large procedural software fixture on Linux."""

from __future__ import annotations

import argparse
import json
import resource
import numpy as np
import torch

from collections.abc import Iterator, Sequence
from pathlib import Path
from tqdm.auto import tqdm

from foundry.digests import sha256_file
from protify.data import label_batches
from protify.data.label_batches import SparseObservedTargets, collate_sparse_labels


class ProceduralRows(Sequence):
    """Every row is generated independently; no dense corpus is held by the fixture."""

    def __init__(self, rows: int, width: int, *, masks: bool) -> None:
        self.rows, self.width, self.masks = rows, width, masks

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> np.ndarray:
        if not isinstance(index, int):
            raise TypeError("the procedural source supports individual rows only")
        if not 0 <= index < self.rows:
            raise IndexError(index)
        positive, negative = index % self.width, (index + 7) % self.width
        if self.masks:
            mask = np.zeros(self.width, dtype=np.bool_)  # (c,)
            if index % 11:
                if index % 13:
                    mask[[positive, negative]] = True
                else:
                    mask[:] = True
            return mask  # (c,)
        target = np.full(self.width, np.nan, dtype=np.float32)  # (c,)
        if index % 11:
            if index % 13 == 0:
                target[:] = 0
            target[positive], target[negative] = 1, 0
        return target  # (c,)

    def __iter__(self) -> Iterator[np.ndarray]:
        indices = range(self.rows) if self.masks else tqdm(range(self.rows), desc="Compress observed targets", unit="rows")
        for index in indices:
            yield self[index]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resident_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux process RSS is unavailable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1_200_000)
    parser.add_argument("--width", type=int, default=9951)
    parser.add_argument("--batch-size", type=int, default=64)
    arguments = parser.parse_args()
    require(arguments.rows > 13 and arguments.width > 7 and 0 < arguments.batch_size <= arguments.rows, "invalid fixture dimensions")
    require(arguments.output.is_absolute() and not arguments.output.exists(), "output must be a fresh absolute path")
    source = ProceduralRows(arguments.rows, arguments.width, masks=False)
    masks = ProceduralRows(arguments.rows, arguments.width, masks=True)
    baseline_rss = resident_bytes()
    baseline_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    storage = SparseObservedTargets.from_rows(source, masks)
    compressed_rss = resident_bytes()
    observed_rows = sum(index % 11 != 0 for index in range(arguments.rows))
    complete_rows = sum(index % 11 != 0 and index % 13 == 0 for index in range(arguments.rows))
    require(storage.values.shape == (arguments.rows, arguments.width), "stored shape differs from the full requested corpus")
    require(storage.values.nnz == observed_rows, "positive targets were dropped or duplicated")
    require(storage.mask_entries.nnz == 2 * (observed_rows - complete_rows), "measured negatives or unknown targets changed")
    require(int(storage.complete_rows.sum()) == complete_rows, "complete observation flags differ")
    require(storage.storage_nbytes < arguments.rows * 100, "storage grew with vocabulary width instead of observed entries")
    indices = [int(index) for index in np.linspace(0, arguments.rows - 1, num=arguments.batch_size)]
    batch = collate_sparse_labels([storage.row(index) for index in indices])
    expected_masks = np.stack([masks[index] for index in indices])  # (b, c)
    expected_labels = np.where(expected_masks, np.stack([source[index] for index in indices]), 0)  # (b, c)
    require(np.array_equal(batch["labels"].numpy(), expected_labels), "selected target batch differs from procedural originals")
    require(np.array_equal(batch["label_mask"].numpy(), expected_masks), "selected mask batch differs from procedural originals")
    require(batch["labels"].dtype == torch.float32 and batch["label_mask"].dtype == torch.bool, "batch dtypes changed")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(peak - baseline_peak < 1024**3, "label storage pilot exceeded one GiB of peak RSS growth")
    receipt = {
        "schema": "compact_label_storage_pilot_v1", "kind": "procedural_software_fixture_only",
        "rows": arguments.rows, "width": arguments.width, "batch_size": arguments.batch_size,
        "positive_entries": int(storage.values.nnz), "stored_mask_entries": int(storage.mask_entries.nnz),
        "complete_rows": complete_rows, "compact_array_bytes": storage.storage_nbytes,
        "equivalent_dense_float32_targets_and_boolean_masks_bytes": arguments.rows * arguments.width * 5,
        "baseline_rss_bytes": baseline_rss, "rss_after_storage_bytes": compressed_rss,
        "baseline_peak_rss_bytes": baseline_peak, "peak_rss_bytes": peak, "peak_rss_growth_bytes": peak - baseline_peak,
        "selected_batch_bytes": sum(tensor.numel() * tensor.element_size() for tensor in batch.values()),
        "selected_rows": indices, "independent_selected_row_readback": True,
        "implementation_sha256": sha256_file(label_batches.__file__), "pilot_sha256": sha256_file(__file__),
        "scope": "compact label assembly only; dense source export, workflow-wide prediction arrays and full training memory remain unqualified",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
