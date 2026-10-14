"""Measure full fast classification method construction on mapped software-fixture inputs.

Build two heads' plain and calibrated predictions plus their ensemble over three splits.
Independent NumPy/SciPy block arithmetic checks every resulting probability and uncertainty.
This exercises method construction, not reporting, learned models, or source qualification.
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
from scipy.special import expit
from tqdm.auto import tqdm

from foundry.digests import sha256_file
from protify.testing_suite.run_label_storage_pilot import require, resident_bytes
from protify.uncertainty import methods, prediction_arrays
from protify.uncertainty.prediction_arrays import DiskPredictionArray, PredictionAccumulator, release_prediction_pages
from protify.uncertainty.predictions import SplitPredictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=60_000)
    parser.add_argument("--width", type=int, default=9951)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rss-growth-mib", type=int, default=768)
    arguments = parser.parse_args()
    require(arguments.root.is_absolute() and not arguments.root.exists(), "root must be a fresh absolute directory")
    require(arguments.rows >= 9 and arguments.width > 1 and arguments.batch_size > 0 and arguments.max_rss_growth_mib > 0,
            "invalid dimensions or budget")
    require(shutil.disk_usage(tempfile.gettempdir()).free > arguments.rows * arguments.width * 64 + 1024**3,
            "insufficient space for simultaneous input and derived prediction arrays")
    arguments.root.mkdir(parents=True)
    started = time.monotonic()
    baseline_rss = resident_bytes()
    baseline_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    generator = np.random.default_rng(2026)
    digests = {name: hashlib.sha256() for name in ("first", "second", "labels", "mask")}
    with PredictionAccumulator(arguments.rows, disk_backed=True) as accumulator:
        for start in tqdm(range(0, arguments.rows, arguments.batch_size), desc="Prepare method inputs", unit="batches"):
            stop = min(start + arguments.batch_size, arguments.rows)
            shape = (stop - start, arguments.width)
            first = generator.normal(0, 2, size=shape).astype(np.float32)  # (b, c)
            second = first * 1.5  # (b, c)
            labels = (generator.random(shape) < expit(first / 1.8)).astype(np.float32)  # (b, c)
            mask = generator.random(shape) > 0.2  # (b, c)
            rows = np.arange(start, stop)  # (b,)
            mask[rows % 13 == 0] = True
            mask[rows % 11 == 0] = False
            labels[~mask] = np.nan
            batch = {"first": first, "second": second, "labels": labels, "mask": mask}
            for name, values in batch.items():
                digests[name].update(values.tobytes())
            accumulator.append(batch)
        source = accumulator.finish()
    endpoints = (0, arguments.rows // 3, 2 * arguments.rows // 3, arguments.rows)
    slices = {name: slice(left, right) for name, left, right in zip(("v_fit", "v_cal", "test"), endpoints[:-1], endpoints[1:], strict=True)}
    seeds = {seed: {name: SplitPredictions(source["labels"][selected], source[field][selected], label_mask=source["mask"][selected])
                    for name, selected in slices.items()} for seed, field in ((42, "first"), (43, "second"))}
    prepared = time.monotonic()
    print("Build both heads, their calibrations, and the calibrated ensemble", flush=True)
    built = methods.build_methods(seeds, "projected", "multilabel")
    completed = time.monotonic()
    require([method.name for method in built] == ["plain", "temperature", "ensemble", "ensemble_temperature"],
            "unexpected fast method inventory")
    derived_bytes = 0
    fitted = {}
    for method in tqdm(built, desc="Verify methods", unit="methods"):
        for run in method.runs:
            fitted[f"{method.name}:{run.seed}"] = dict(run.fitted)
            for name, prediction in run.splits.items():
                require(isinstance(prediction.probs, DiskPredictionArray) and isinstance(prediction.uncertainty, DiskPredictionArray),
                        "method construction materialized a complete heap output")
                require(not prediction.probs.flags.writeable and not prediction.uncertainty.flags.writeable, "mapped result is writable")
                derived_bytes += prediction.probs.nbytes + prediction.uncertainty.nbytes
                selected = slices[name]
                for offset in range(0, selected.stop - selected.start, arguments.batch_size):
                    start = selected.start + offset
                    stop = min(start + arguments.batch_size, selected.stop)
                    first = np.asarray(source["first"][start:stop])  # (b, c)
                    second = np.asarray(source["second"][start:stop])  # (b, c)
                    if run.seed is not None:
                        logits = first if run.seed == 42 else second  # (b, c)
                        expected = expit(logits if method.name == "plain" else logits.astype(np.float64) / run.fitted["temperature"])  # (b, c)
                    else:
                        expected = np.mean([expit(first), expit(second)], axis=0)  # (b, c)
                        if method.name == "ensemble_temperature":
                            clipped = np.clip(expected.astype(np.float64), 1e-12, 1 - 1e-12)  # (b, c)
                            expected = expit((np.log(clipped) - np.log1p(-clipped)) / run.fitted["temperature"])  # (b, c)
                    require(prediction.probs.dtype == expected.dtype, "probability dtype changed")
                    require(np.allclose(prediction.probs[offset:offset + len(first)], expected, rtol=2e-7, atol=1e-14),
                            "stored probabilities differ from independent block arithmetic")
                    clipped = np.clip(expected.astype(np.float64), 1e-12, 1 - 1e-12)  # (b, c)
                    uncertainty = -np.mean(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped), axis=1)  # (b,)
                    require(np.allclose(prediction.uncertainty[offset:offset + len(first)], uncertainty, rtol=2e-7, atol=1e-14),
                            "stored uncertainty differs from independent block arithmetic")
                    for values in (*source.values(), prediction.probs, prediction.uncertainty):
                        release_prediction_pages(values)
    restored_digests = {name: hashlib.sha256() for name in source}
    for start in tqdm(range(0, arguments.rows, arguments.batch_size), desc="Verify unchanged sources", unit="batches"):
        for name, values in source.items():
            restored_digests[name].update(values[start:start + arguments.batch_size].tobytes())
            release_prediction_pages(values)
    require(all(digests[name].digest() == restored_digests[name].digest() for name in source), "method building changed source data")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(peak - baseline_peak < arguments.max_rss_growth_mib * 1024**2, "mapped method building exceeded its RSS budget")
    receipt = {
        "schema": "mapped_methods_pilot_v1", "kind": "procedural_software_fixture_only",
        "rows_across_splits": arguments.rows, "width": arguments.width, "seeds": [42, 43], "batch_size": arguments.batch_size,
        "split_rows": {name: selected.stop - selected.start for name, selected in slices.items()},
        "input_array_bytes": sum(values.nbytes for values in source.values()), "derived_array_bytes": derived_bytes,
        "input_sha256": {name: digest.hexdigest() for name, digest in digests.items()},
        "all_outputs_independently_checked": True, "all_inputs_unchanged": True, "fitted_parameters": fitted,
        "baseline_rss_bytes": baseline_rss, "baseline_peak_rss_bytes": baseline_peak,
        "peak_rss_bytes": peak, "peak_rss_growth_bytes": peak - baseline_peak,
        "rss_growth_limit_bytes": arguments.max_rss_growth_mib * 1024**2,
        "preparation_seconds": prepared - started, "method_build_seconds": completed - prepared,
        "total_seconds": time.monotonic() - started, "torch_threads": torch.get_num_threads(),
        "implementation_sha256": {"methods": sha256_file(methods.__file__), "prediction_arrays": sha256_file(prediction_arrays.__file__)},
        "pilot_sha256": sha256_file(__file__),
        "scope": "plain, temperature, ensemble and ensemble_temperature construction only; reports, MC dropout capacity, optional methods, full training and source acceptance remain unqualified",
    }
    with (arguments.root / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
