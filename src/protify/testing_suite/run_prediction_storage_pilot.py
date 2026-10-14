"""Measure prediction collection, NPZ writing, and verified reload on a Linux software fixture.

The procedural head and truth establish storage behavior only. Calibration, reporting,
training, scientific evidence, and any pretrained model's capacity remain outside this pilot.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import shutil
import tempfile
import time
import numpy as np
import torch

from pathlib import Path
from types import SimpleNamespace
from torch import nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from foundry.digests import sha256_file
from protify.testing_suite.run_label_storage_pilot import require, resident_bytes
from protify.uncertainty import prediction_arrays, predictions
from protify.uncertainty.prediction_arrays import DiskPredictionArray, release_prediction_pages
from protify.uncertainty.predictions import collect_predictions, load_seed_predictions, save_seed_predictions


class OrdinalRows(Dataset):
    def __init__(self, rows: int) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> int:
        return index


class ProceduralHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width

    def forward(self, embeddings: torch.Tensor) -> SimpleNamespace:
        # embeddings: (b, 1)
        columns = torch.arange(self.width, device=embeddings.device).reshape(1, -1)  # (1, c)
        logits = ((embeddings + columns) % 7 - 3) / 4  # (b, c), exactly representable quarters
        return SimpleNamespace(logits=logits)


def collate_ordinals(indices: list[int], width: int) -> dict[str, torch.Tensor]:
    rows = torch.tensor(indices, dtype=torch.int64)  # (b,)
    labels = torch.zeros((len(rows), width), dtype=torch.float32)  # (b, c)
    mask = torch.zeros_like(labels, dtype=torch.bool)  # (b, c)
    for row, index in enumerate(indices):
        if index % 11:
            labels[row, index % width] = 1
            mask[row, [index % width, (index + 7) % width]] = True
            if index % 13 == 0:
                mask[row] = True
    return {"embeddings": rows.reshape(-1, 1), "labels": labels, "label_mask": mask}  # (b, 1), (b, c), (b, c)


def verify_all_rows(restored: predictions.SplitPredictions, width: int, batch_size: int) -> None:
    """Independent NumPy arithmetic checks every stored entry, including unobserved truth."""
    for start in tqdm(range(0, len(restored.labels), batch_size), desc="Verify every prediction", unit="batches"):
        stop = min(start + batch_size, len(restored.labels))
        rows = np.arange(start, stop, dtype=np.int64)  # (b,)
        outputs = ((rows[:, None] + np.arange(width)) % 7 - 3).astype(np.float32) / 4  # (b, c)
        labels = np.zeros_like(outputs)  # (b, c)
        mask = np.zeros(outputs.shape, dtype=bool)  # (b, c)
        observed = rows % 11 != 0  # (b,)
        positions = np.flatnonzero(observed)  # (k,)
        labels[positions, rows[observed] % width] = 1
        mask[positions, rows[observed] % width] = True
        mask[positions, (rows[observed] + 7) % width] = True
        mask[observed & (rows % 13 == 0)] = True
        for name, expected in (("outputs", outputs), ("labels", labels), ("label_mask", mask)):
            actual = getattr(restored, name)
            require(np.array_equal(actual[start:stop], expected), f"stored {name} differs from independent truth")
            release_prediction_pages(actual)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=60_000)
    parser.add_argument("--width", type=int, default=9951)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rss-growth-mib", type=int, default=1024)
    arguments = parser.parse_args()
    require(arguments.root.is_absolute() and not arguments.root.exists(), "root must be a fresh absolute directory")
    require(arguments.rows > 0 and arguments.width > 7 and arguments.batch_size > 0 and arguments.max_rss_growth_mib > 0, "invalid dimensions or budget")
    expected_bytes = arguments.rows * arguments.width * 9  # float32 outputs/labels and boolean masks
    require(shutil.disk_usage(tempfile.gettempdir()).free > expected_bytes * 2 + 1024**3, "insufficient temporary storage")
    arguments.root.mkdir(parents=True)
    require(shutil.disk_usage(arguments.root).free > expected_bytes * 2 + 1024**3, "insufficient archive storage")
    started = time.monotonic()
    baseline_rss = resident_bytes()
    baseline_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    head = ProceduralHead(arguments.width)
    def collator(indices: list[int]) -> dict[str, torch.Tensor]:
        return collate_ordinals(indices, arguments.width)  # (b, 1), (b, c), (b, c)
    split_predictions, inference_seconds = collect_predictions(head, OrdinalRows(arguments.rows), collator,
                                                    task_kind="multilabel", batch_size=arguments.batch_size,
                                                    device=torch.device("cpu"), disk_backed=True)
    collected = time.monotonic()
    require(all(isinstance(getattr(split_predictions, name), DiskPredictionArray) for name in ("outputs", "labels", "label_mask")), "collector returned complete heap arrays")
    archive = arguments.root / "seed_42.npz"
    save_seed_predictions(str(archive), {"test": split_predictions})
    written = time.monotonic()
    del split_predictions
    gc.collect()
    restored = load_seed_predictions(str(archive), mmap_mode="r")["test"]
    loaded = time.monotonic()
    require(all(isinstance(getattr(restored, name), DiskPredictionArray) for name in ("outputs", "labels", "label_mask")), "reload returned complete heap arrays")
    verify_all_rows(restored, arguments.width, arguments.batch_size)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(peak - baseline_peak < arguments.max_rss_growth_mib * 1024**2, "prediction storage exceeded its RSS growth budget")
    receipt = {
        "schema": "prediction_storage_pilot_v1", "kind": "procedural_software_fixture_only",
        "rows": arguments.rows, "width": arguments.width, "batch_size": arguments.batch_size,
        "array_bytes": expected_bytes, "archive_bytes": archive.stat().st_size, "archive_sha256": sha256_file(archive),
        "temporary_directory": tempfile.gettempdir(), "all_entries_independently_verified": True,
        "baseline_rss_bytes": baseline_rss, "baseline_peak_rss_bytes": baseline_peak,
        "peak_rss_bytes": peak, "peak_rss_growth_bytes": peak - baseline_peak,
        "rss_growth_limit_bytes": arguments.max_rss_growth_mib * 1024**2,
        "collection_seconds": collected - started, "writing_seconds": written - collected,
        "loading_seconds": loaded - written, "total_seconds": time.monotonic() - started,
        "plain_forward_seconds": inference_seconds,
        "implementation_sha256": {"prediction_arrays": sha256_file(prediction_arrays.__file__), "predictions": sha256_file(predictions.__file__)},
        "pilot_sha256": sha256_file(__file__),
        "scope": "bounded prediction collection, NPZ serialization and checked reload only; full fitting, calibration, reporting, native features, and scientific acceptance remain unqualified",
    }
    with (arguments.root / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
