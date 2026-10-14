"""Measure bounded temperature fitting on mapped, procedurally generated multilabel truth.

All observed logits have magnitude a. A fraction q agree with the binary label, so the exact
optimum is T = a / log(q / (1 - q)). This checks the complete fitting pass without a dense
reference computation. Unknown entries contain NaNs and must never affect the objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import shutil
import tempfile
import time
import numpy as np
import torch

from pathlib import Path
from tqdm.auto import tqdm

from foundry.digests import sha256_file
from foundry.eval import temperature
from protify.testing_suite.run_label_storage_pilot import require, resident_bytes
from protify.uncertainty import methods, prediction_arrays
from protify.uncertainty.prediction_arrays import PredictionAccumulator, release_prediction_pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=60_000)
    parser.add_argument("--width", type=int, default=9951)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rss-growth-mib", type=int, default=512)
    arguments = parser.parse_args()
    require(arguments.root.is_absolute() and not arguments.root.exists(), "root must be a fresh absolute directory")
    require(arguments.rows > 0 and arguments.width > 7 and arguments.batch_size > 0 and arguments.max_rss_growth_mib > 0,
            "invalid dimensions or budget")
    array_bytes = arguments.rows * arguments.width * 9
    require(shutil.disk_usage(tempfile.gettempdir()).free > array_bytes + 1024**3, "insufficient temporary storage")
    arguments.root.mkdir(parents=True)
    started = time.monotonic()
    baseline_rss = resident_bytes()
    baseline_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    observed_count = correct_count = 0
    magnitude = np.float32(2 * math.log(3))
    columns = np.arange(arguments.width, dtype=np.int64).reshape(1, -1)  # (1, c)
    source_hashes = {name: hashlib.sha256() for name in ("outputs", "labels", "mask")}
    with PredictionAccumulator(arguments.rows, disk_backed=True) as accumulator:
        for start in tqdm(range(0, arguments.rows, arguments.batch_size), desc="Prepare calibration", unit="batches"):
            rows = np.arange(start, min(start + arguments.batch_size, arguments.rows)).reshape(-1, 1)  # (b, 1)
            category = (rows * arguments.width + columns) % 8  # (b, c)
            positive = category < 4  # (b, c)
            labels = np.where(positive, category != 3, category == 7).astype(np.float32)  # (b, c)
            outputs = np.where(positive, magnitude, -magnitude)  # (b, c)
            mask = (rows % 11 != 0) & ((columns % 17 != 0) | (rows % 13 == 0))  # (b, c)
            observed_count += int(np.count_nonzero(mask))
            correct_count += int(np.count_nonzero(mask & (positive == labels)))
            outputs[~mask] = labels[~mask] = np.nan
            batch = {"outputs": outputs, "labels": labels, "mask": mask}
            for name, values in batch.items():
                source_hashes[name].update(values.tobytes())
            accumulator.append(batch)
        mapped = accumulator.finish()
    prepared = time.monotonic()
    require(observed_count > 0, "fixture has no observed targets")
    agreement = correct_count / observed_count
    require(0.5 < agreement < 1, "fixture requires a finite, positive temperature optimum")
    expected = float(magnitude) / math.log(agreement / (1 - agreement))
    print("Fitting one temperature on every observed target", flush=True)
    fitted = methods._fit_observed_temperature(mapped["outputs"], mapped["labels"], "multilabel", mapped["mask"])
    fitted_at = time.monotonic()
    require(math.isclose(fitted, expected, rel_tol=2e-4), "fitted temperature differs from the analytic optimum")
    fitted_scale = float(magnitude) / fitted
    optimum_scale = float(magnitude) / expected
    fitted_nll = float(np.logaddexp(0, fitted_scale) - agreement * fitted_scale)
    optimum_nll = float(np.logaddexp(0, optimum_scale) - agreement * optimum_scale)
    uncalibrated_nll = float(np.logaddexp(0, float(magnitude)) - agreement * float(magnitude))
    require(abs(fitted_nll - optimum_nll) < 1e-8 and fitted_nll < uncalibrated_nll,
            "calibration did not minimize the independent count-based objective")
    restored_hashes = {name: hashlib.sha256() for name in mapped}
    for start in tqdm(range(0, arguments.rows, arguments.batch_size), desc="Verify unchanged inputs", unit="batches"):
        for name, values in mapped.items():
            restored_hashes[name].update(values[start:start + arguments.batch_size].tobytes())
            release_prediction_pages(values)
    for name in mapped:
        require(source_hashes[name].digest() == restored_hashes[name].digest(), f"fitting changed {name}")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(peak - baseline_peak < arguments.max_rss_growth_mib * 1024**2, "temperature fitting exceeded its RSS growth budget")
    receipt = {
        "schema": "temperature_storage_pilot_v1", "kind": "procedural_software_fixture_only",
        "rows": arguments.rows, "width": arguments.width, "batch_size": arguments.batch_size,
        "batch_elements": temperature.TEMPERATURE_BATCH_ELEMENTS, "array_bytes": array_bytes,
        "observed_count": observed_count, "correct_count": correct_count, "logit_magnitude": float(magnitude),
        "fitted_temperature": fitted, "analytic_temperature": expected,
        "fitted_nll": fitted_nll, "analytic_nll": optimum_nll, "uncalibrated_nll": uncalibrated_nll,
        "all_inputs_unchanged": True, "input_sha256": {name: value.hexdigest() for name, value in source_hashes.items()},
        "baseline_rss_bytes": baseline_rss, "baseline_peak_rss_bytes": baseline_peak,
        "peak_rss_bytes": peak, "peak_rss_growth_bytes": peak - baseline_peak,
        "rss_growth_limit_bytes": arguments.max_rss_growth_mib * 1024**2,
        "preparation_seconds": prepared - started, "fitting_seconds": fitted_at - prepared,
        "total_seconds": time.monotonic() - started, "torch_threads": torch.get_num_threads(),
        "implementation_sha256": {"temperature": sha256_file(temperature.__file__),
                                  "methods": sha256_file(methods.__file__), "prediction_arrays": sha256_file(prediction_arrays.__file__)},
        "pilot_sha256": sha256_file(__file__),
        "scope": "mapped scalar temperature fitting only; probability transforms, reporting, full training, native features and scientific acceptance remain unqualified",
    }
    with (arguments.root / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
