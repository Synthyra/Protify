"""Measure a complete canonical multilabel report, including exact global ranking metrics.

Procedural probabilities and observation masks exercise the software path without training
or biological inputs. Independent block counts/losses check support, F1, Brier, NLL, and
selective scores. Exact AP/AUROC/PR area and equal-mass ECE retain their existing algorithms;
their full-width workspace is included in the RSS measurement, not claimed to be bounded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import tempfile
import time
import numpy as np
import torch

from pathlib import Path
from tqdm.auto import tqdm

from foundry.digests import sha256_file
from protify.testing_suite.run_label_storage_pilot import require, resident_bytes
from protify.uncertainty import metric_contracts, metrics, prediction_arrays, report
from protify.uncertainty.methods import Predictive
from protify.uncertainty.prediction_arrays import PredictionAccumulator, release_prediction_pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=60_000)
    parser.add_argument("--width", type=int, default=9951)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rss-growth-mib", type=int, default=65536)
    arguments = parser.parse_args()
    require(arguments.root.is_absolute() and not arguments.root.exists(), "root must be a fresh absolute directory")
    require(arguments.rows > 10 and arguments.width > 1 and arguments.batch_size > 0 and arguments.max_rss_growth_mib > 0, "invalid dimensions or budget")
    require(shutil.disk_usage(tempfile.gettempdir()).free > arguments.rows * arguments.width * 24 + 1024**3, "insufficient temporary prediction storage")
    arguments.root.mkdir(parents=True)
    started = time.monotonic()
    baseline_rss = resident_bytes()
    baseline_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    generator = np.random.default_rng(817)
    positive = np.zeros(arguments.width, dtype=np.int64)  # (c,)
    negative = np.zeros_like(positive)  # (c,)
    tp, fp, fn = (np.zeros_like(positive) for _ in range(3))  # each (c,)
    losses, uncertainty = np.full(arguments.rows, np.nan), generator.random(arguments.rows)  # each (n,)
    observed = 0
    brier_sum, nll_sum = 0.0, 0.0
    digests = {name: hashlib.sha256() for name in ("probs", "labels", "mask", "uncertainty")}
    with PredictionAccumulator(arguments.rows, disk_backed=True) as accumulator:
        for start in tqdm(range(0, arguments.rows, arguments.batch_size), desc="Prepare report inputs", unit="batches"):
            stop = min(start + arguments.batch_size, arguments.rows)
            shape = (stop - start, arguments.width)
            probs = generator.random(shape).astype(np.float32)  # (b, c)
            labels = (generator.random(shape) < probs).astype(np.float32)  # (b, c)
            mask = generator.random(shape) > 0.2  # (b, c)
            mask[np.arange(start, stop) % 11 == 0] = False
            positive += ((labels == 1) & mask).sum(axis=0)
            negative += ((labels == 0) & mask).sum(axis=0)
            decisions = probs > 0.5  # (b, c)
            tp += (decisions & (labels == 1) & mask).sum(axis=0)
            fp += (decisions & (labels == 0) & mask).sum(axis=0)
            fn += (~decisions & (labels == 1) & mask).sum(axis=0)
            counts = mask.sum(axis=1)  # (b,)
            errors = ((decisions != labels) & mask).sum(axis=1)  # (b,)
            losses[start:stop] = np.divide(errors, counts, out=np.full(len(counts), np.nan), where=counts > 0)
            observed += int(mask.sum())
            truth, selected = labels[mask], probs[mask]  # (m,), observed entries in this block
            brier_sum += float(np.sum((selected - truth) ** 2, dtype=np.float64))
            clipped = np.clip(selected.astype(np.float64), 1e-12, 1 - 1e-12)  # (m,)
            nll_sum -= float(np.sum(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped)))
            labels[~mask] = np.nan
            batch = {"probs": probs, "labels": labels, "mask": mask, "uncertainty": uncertainty[start:stop]}
            for name, values in batch.items():
                digests[name].update(values.tobytes())
            accumulator.append(batch)
        source = accumulator.finish()
    prepared = time.monotonic()
    print("Score complete canonical report including exact micro and macro rankings", flush=True)
    scores = report.score_rows("multilabel", Predictive(probs=source["probs"], uncertainty=source["uncertainty"]),
                               source["labels"], (), label_mask=source["mask"], canonical=True)
    support = report.label_support("multilabel", source["labels"], source["mask"])
    scored = time.monotonic()
    require(scores["count"] == arguments.rows and scores["observed_labels"] == observed, "report row or observation counts changed")
    require(scores["unobserved_labels"] == arguments.rows * arguments.width - observed, "unknown entries were lost")
    require(np.isclose(scores["brier"], brier_sum / observed, rtol=2e-6, atol=1e-12), "Brier score differs from independent block arithmetic")
    require(np.isclose(scores["nll"], nll_sum / observed, rtol=1e-12, atol=1e-12), "NLL differs from independent block arithmetic")
    micro_f1 = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())
    denominator = 2 * tp + fp + fn  # (c,)
    macro_f1 = np.mean(2 * tp[denominator > 0] / denominator[denominator > 0])
    require(np.isclose(scores["micro_f1"], micro_f1, rtol=1e-12) and np.isclose(scores["macro_f1"], macro_f1, rtol=1e-12), "F1 differs from independent counts")
    require(scores["f1"] == scores["micro_f1"], "primary F1 is not fixed-threshold micro F1")
    valid = np.isfinite(losses)  # (n,)
    expected_selective = metrics.selective_metrics("multilabel", losses[valid], uncertainty[valid])
    for name, value in expected_selective.items():
        require(scores[name] is None if value is None else np.isclose(scores[name], value, rtol=1e-12, atol=1e-12), f"selective score differs: {name}")
    require(scores["observed_rows"] == int(valid.sum()), "wholly unknown rows reached scoring")
    require(support["observed"] == observed and support["unknown"] == arguments.rows * arguments.width - observed, "support totals differ")
    for index, entry in enumerate(support["labels"]):
        require(entry["positive"] == positive[index] and entry["negative"] == negative[index], "per-label support differs")
        require(entry["unknown"] == arguments.rows - positive[index] - negative[index], "per-label unknown count differs")
    for name, value in scores.items():
        require(value is None or np.isfinite(value), f"nonfinite report field: {name}")
        if any(name.endswith(metric) for metric in ("average_precision", "pr_auc_trapezoidal", "auroc")) or name == "ece":
            require(value is None or 0 <= value <= 1, f"ranking or ECE outside its range: {name}")
    restored = {name: hashlib.sha256() for name in source}
    for start in tqdm(range(0, arguments.rows, arguments.batch_size), desc="Verify unchanged inputs", unit="batches"):
        for name, values in source.items():
            restored[name].update(values[start:start + arguments.batch_size].tobytes())
            release_prediction_pages(values)
    require(all(restored[name].digest() == digest.digest() for name, digest in digests.items()), "reporting changed input data")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(peak - baseline_peak < arguments.max_rss_growth_mib * 1024**2, "complete report exceeded its declared RSS budget")
    receipt = {
        "schema": "mapped_report_pilot_v1", "kind": "procedural_software_fixture_only",
        "rows_in_one_scored_split": arguments.rows, "width": arguments.width, "batch_size": arguments.batch_size,
        "input_array_bytes": sum(values.nbytes for values in source.values()), "input_sha256": {name: digest.hexdigest() for name, digest in digests.items()},
        "all_inputs_unchanged": True, "scores": scores, "observed_targets": observed,
        "independent_checks": ["support_every_label", "fixed_threshold_micro_macro_f1", "brier", "nll", "row_losses_and_selective_metrics"],
        "ranking_verification": "existing exact algorithms; bounded-fixture parity tests; full fixture finite and range checks only",
        "baseline_rss_bytes": baseline_rss, "baseline_peak_rss_bytes": baseline_peak, "peak_rss_bytes": peak,
        "peak_rss_growth_bytes": peak - baseline_peak, "rss_growth_limit_bytes": arguments.max_rss_growth_mib * 1024**2,
        "preparation_seconds": prepared - started, "report_seconds": scored - prepared, "total_seconds": time.monotonic() - started,
        "torch_threads": torch.get_num_threads(),
        "implementation_sha256": {module.__name__: sha256_file(module.__file__) for module in (metric_contracts, metrics, prediction_arrays, report)},
        "pilot_sha256": sha256_file(__file__),
        "scope": "one canonical multilabel report; global rankings retain input-dependent RAM; no training, package, conformal classification/regression capacity, or accepted-source qualification",
    }
    with (arguments.root / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
