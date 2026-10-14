"""Run the supported source-to-package path on a procedural release, and record what it costs.

The release and its ordinary feature stores come from `procedural_release_fixture`, whose proteins and truth are
artificial: this establishes capacity and numerical agreement, never a homology-qualified split or a scientific result.
The path is the fit cell every declared release trains through (`protify.uncertainty.workflow.run_uncertainty_cell`),
either whole (`full`: fit, calibrate on V_cal, score test, report) or as the production fit stage (`validation_only`:
train and validation only). A rerun must then train nothing, and a `full` run continues into `build_package`,
`load_package`, and a prediction through the package.

Every function of the path that owns a stage of it runs inside a named stage that records its seconds, the peak resident
memory of the process, the peak device memory, and the peak disk the run added. The receipt holds those stages, the
release and store identities, the exact cell settings, the digest of every workspace source file the run loaded, the
host, and the throughput they imply. It is `suite_capacity_receipt_v1`; `production_suite.capacity` reads it. A run that
fails partway still writes its receipt, with the stage it failed in and `complete` false.

With `--parity` (small dimensions only) the same truth is also written as a version 1 dense release and trained through
the same cell, and the two must agree on the report and on every saved prediction, label, and mask.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shutil
import sys
import threading
import time
import traceback
import numpy as np
import pyarrow.parquet as pq
import torch

from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

from foundry.digests import sha256_file
from protify.data.branch_features import branch_embedding_arguments, parse_branch_spec
from protify.testing_suite.procedural_release_fixture import Dimensions, write_feature_stores, write_release
from protify.uncertainty import workflow
from protify.uncertainty.cell import cell_from_mapping
from protify.uncertainty.package import build_package
from protify.uncertainty.packaged import load_package
from protify.uncertainty.predictions import load_seed_predictions


RECEIPT_SCHEMA = "suite_capacity_receipt_v1"
MODES = ("full", "validation_only")
# The functions of `workflow` that each own one stage of the path, and the name their stage goes by.
WORKFLOW_STAGES = {
    "load_split_cell_data": "cell.load_release",
    "_record_observations": "cell.record_observations",
    "resolve_validation_roles": "cell.resolve_roles",
    "load_cell_branches": "cell.load_features",
    "fit_joined_standardizer": "cell.fit_standardizer",
    "JoinedBranchDataset": "cell.build_dataset",
    "_train_seed": "seed",
    "_train_head": "seed.train",
    "collect_predictions": "seed.predict",
    "trunk_features": "seed.trunk_features",
    "nearest_neighbor_distance": "seed.nearest_neighbors",
    "save_seed_predictions": "seed.save_predictions",
    "_primary": "seed.primary_metric",
    "_head_nll": "seed.nll",
    "_report": "cell.report",
    "load_seeds": "report.load_seeds",
    "build_methods": "report.build_methods",
    "score_method": "report.score_method",
}
PACKAGE_PREDICTION_ROWS = 2048
PACKAGE_BATCH_SIZE = 64
TRAINING_BATCH_ARGUMENT = "probe_batch_size"


def resident_bytes() -> int:
    """This process's resident set from Linux's `/proc`; 0 elsewhere, so only a Linux run is evidence."""
    with contextlib.suppress(OSError), open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def tree_bytes(root: Path) -> int:
    """The bytes of every file under `root`."""
    return sum(Path(directory, name).stat().st_size for directory, _, names in os.walk(root) for name in names)


class StageRecorder:
    """Times nested stages, and samples this process's resident memory and disk use while they run.

    A stage's peaks cover its children. Device memory is credited at stage boundaries from the allocator's running
    peaks, which are then restarted, so a parent keeps the highest peak of any child. With `progress_path` the
    recorder also writes what it has so far each time a stage closes and every `report_seconds`, so a process the
    kernel kills for memory still leaves the stage it died in and the peaks it had reached.
    """

    def __init__(self, root: Path, *, interval: float = 0.25, report_seconds: float = 120.0, progress_path: Path | None = None) -> None:
        self.root = root
        self.records: list[dict[str, Any]] = []
        self.peaks = {"rss_bytes": 0, "disk_added_bytes": 0, "vram_allocated_bytes": 0, "vram_reserved_bytes": 0}
        self._open: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._origin = time.monotonic()
        self._disk_origin = shutil.disk_usage(root).used
        self._interval = interval
        self._report_seconds = report_seconds
        self._progress_path = progress_path
        self._stopped = threading.Event()
        self._sampler = threading.Thread(target=self._sample, daemon=True)
        self._sampler.start()

    def elapsed(self) -> float:
        return time.monotonic() - self._origin

    def close(self) -> None:
        self._stopped.set()
        self._sampler.join()

    def open_stages(self) -> list[str]:
        with self._lock:
            return [record["name"] for record in self._open]

    def failed_stages(self) -> list[str]:
        """The stages an exception passed through, outermost first."""
        with self._lock:
            return [record["name"] for record in self.records if "failed" in record]

    def write_progress(self) -> None:
        if self._progress_path is None:
            return
        with self._lock:
            progress = {"elapsed_seconds": round(self.elapsed(), 1), "open_stages": [record["name"] for record in self._open], "peaks": dict(self.peaks), "stages": list(self.records)}
        write_json(self._progress_path, progress)

    def _sample(self) -> None:
        last_report = time.monotonic()
        while not self._stopped.wait(self._interval):
            self.observe()
            if time.monotonic() - last_report >= self._report_seconds:
                last_report = time.monotonic()
                names = " > ".join(self.open_stages()) or "between stages"
                print(f"[{self.elapsed():9.1f} s] {names}: {resident_bytes() / 2**30:.1f} GiB resident, {self.peaks['rss_bytes'] / 2**30:.1f} GiB peak", flush=True)
                self.write_progress()

    def observe(self) -> None:
        """Credit the current resident memory and disk growth to every open stage."""
        resident = resident_bytes()
        added = max(shutil.disk_usage(self.root).used - self._disk_origin, 0)
        with self._lock:
            self.peaks["rss_bytes"] = max(self.peaks["rss_bytes"], resident)
            self.peaks["disk_added_bytes"] = max(self.peaks["disk_added_bytes"], added)
            for record in self._open:
                record["rss_peak_bytes"] = max(record["rss_peak_bytes"], resident)
                record["disk_added_peak_bytes"] = max(record["disk_added_peak_bytes"], added)

    def _credit_device(self) -> None:
        if not torch.cuda.is_available():
            return
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        with self._lock:
            self.peaks["vram_allocated_bytes"] = max(self.peaks["vram_allocated_bytes"], allocated)
            self.peaks["vram_reserved_bytes"] = max(self.peaks["vram_reserved_bytes"], reserved)
            for record in self._open:
                record["vram_allocated_peak_bytes"] = max(record["vram_allocated_peak_bytes"], allocated)
                record["vram_reserved_peak_bytes"] = max(record["vram_reserved_peak_bytes"], reserved)
        torch.cuda.reset_peak_memory_stats()

    @contextlib.contextmanager
    def stage(self, name: str, **facts: Any) -> Iterator[dict[str, Any]]:
        self._credit_device()
        resident = resident_bytes()
        record = {
            "name": name, "depth": len(self._open), "started_seconds": round(self.elapsed(), 3), "rss_start_bytes": resident,
            "rss_peak_bytes": resident, "disk_added_peak_bytes": 0, "vram_allocated_peak_bytes": 0, "vram_reserved_peak_bytes": 0, **facts,
        }
        with self._lock:
            self.records.append(record)
            self._open.append(record)
        started = time.perf_counter()
        print(f"[{self.elapsed():9.1f} s] {'  ' * record['depth']}start {name}", flush=True)
        try:
            yield record
        except BaseException as error:
            record["failed"] = type(error).__name__
            raise
        finally:
            record["seconds"] = round(time.perf_counter() - started, 3)
            self.observe()
            self._credit_device()
            record["rss_end_bytes"] = resident_bytes()
            with self._lock:
                closed = self._open.pop()
            assert closed is record, "stages must close in the order they opened"
            if not record["depth"]:
                record["tree_bytes"] = tree_bytes(self.root)
            print(f"[{self.elapsed():9.1f} s] {'  ' * record['depth']}end {name}: {record['seconds']:.1f} s, peak {record['rss_peak_bytes'] / 2**30:.2f} GiB", flush=True)
            self.write_progress()

    def calls(self, name: str) -> int:
        return sum(1 for record in self.records if record["name"] == name)

    def first_seconds(self, name: str) -> float:
        return next((record["seconds"] for record in self.records if record["name"] == name and "seconds" in record), 0.0)


def timed(recorder: StageRecorder, stage: str, function: Callable[..., Any]) -> Callable[..., Any]:
    """`function`, each call of which runs inside the named stage; a class becomes a function that builds its instances."""
    def call(*arguments: Any, **options: Any) -> Any:
        with recorder.stage(stage):
            return function(*arguments, **options)

    return call


@contextlib.contextmanager
def instrumented(recorder: StageRecorder, module: Any, stages: Mapping[str, str]) -> Iterator[None]:
    """`module` with each function `stages` names replaced by one that runs inside its stage, restored on exit."""
    originals = {name: getattr(module, name) for name in stages}
    try:
        for name, stage in stages.items():
            setattr(module, name, timed(recorder, stage, originals[name]))
        yield
    finally:
        for name, function in originals.items():
            setattr(module, name, function)


def cell_mapping(root: Path, name: str, settings: Mapping[str, Any], manifest: Path, digest: str, *, mode: str, epochs: int | None) -> dict[str, Any]:
    """The cell of `settings` (a production cell's `protify_args` and `uncertainty`) over the release at `manifest`."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    paths = {"embedding_save_dir": "embeddings", "log_dir": "logs", "results_dir": "results", "plots_dir": "plots", "model_save_dir": "weights"}
    protify_args = {**settings["protify_args"], **{key: str(root / directory) for key, directory in paths.items()}}
    if epochs is not None:
        protify_args["num_epochs"] = epochs
    uncertainty = {
        **settings["uncertainty"], "dataset_manifest": str(manifest), "dataset_sha256": digest, "evaluation_mode": mode,
        "save_heads": True, "require_cached_embeddings": True,
    }
    return {"cell_id": name, "experiment_id": "suite_capacity", "protify_args": protify_args, "uncertainty": uncertainty, "output_dir": str(root / "runs" / name)}


def store_descriptors(cell: Any) -> dict[str, dict[str, Any]]:
    """The descriptors of the cell's two stores, named as the run itself names them: the sparse branch and the dense one."""
    specs = [parse_branch_spec(token) for token in cell.uncertainty.branch_specs]
    arguments = vars(workflow.cell_arguments(cell, specs))
    descriptors = {"sae" if spec.sparse else "dense": branch_embedding_arguments(spec, arguments).cache_descriptor(spec.model_name, False) for spec in specs}
    if len(specs) != 2 or set(descriptors) != {"sae", "dense"}:
        raise ValueError("the capacity path takes one sparse SAE branch and one dense branch")
    return descriptors


def shipment_directory() -> Path | None:
    """The `ws remote` run directory this pilot was shipped into, which holds `run.json` and `receipt.json`; None when run in place."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "run.json").is_file() and (parent / "receipt.json").is_file():
            return parent
    return None


def workspace_path(path: Path, shipment: Path | None) -> str | None:
    """A source file's path in the workspace, whether it runs from the workspace or from a shipment; None for any other file.

    A shipment holds the workspace's files under `tree/`, and `foundry` under `import-roots/`.
    """
    resolved = path.resolve()
    if shipment is not None:
        for base in (shipment / "tree", shipment / "import-roots"):
            if resolved.is_relative_to(base):
                return resolved.relative_to(base).as_posix()
        return None
    for parent in resolved.parents:
        if (parent / "AGENTS.md").is_file() and (parent / "projects").is_dir():
            return resolved.relative_to(parent).as_posix()
    return None


def loaded_source_digests() -> dict[str, str]:
    """The SHA-256 of every workspace source file this process has loaded, by workspace path: the code that ran."""
    shipment, digests = shipment_directory(), {}
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file and file.endswith(".py") and Path(file).is_file() and (relative := workspace_path(Path(file), shipment)) is not None:
            digests[relative] = sha256_file(file)
    return dict(sorted(digests.items()))


def host_facts() -> dict[str, Any]:
    facts: dict[str, Any] = {
        "platform": platform.platform(), "python": platform.python_version(), "linux": sys.platform.startswith("linux"),
        "cpu_count": os.cpu_count(), "packages": {},
    }
    for name in ("torch", "numpy", "pyarrow", "datasets", "transformers", "scipy", "scikit-learn"):
        try:
            facts["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            facts["packages"][name] = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        facts["memory_total_bytes"] = int(next(line for line in meminfo.read_text().splitlines() if line.startswith("MemTotal:")).split()[1]) * 1024
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        facts["gpu"] = {"name": properties.name, "total_memory_bytes": properties.total_memory, "capability": list(torch.cuda.get_device_capability(0)), "count": torch.cuda.device_count()}
    return facts


def shipment_identity() -> dict[str, Any] | None:
    """`tree_hash` and origin of the `ws remote` shipment this run came in, when it came in one."""
    if (parent := shipment_directory()) is None:
        return None
    record = json.loads((parent / "run.json").read_text(encoding="utf-8"))
    return {key: record.get(key) for key in ("entity", "tree_hash", "sent_from", "sent")} | {"receipt_sha256": sha256_file(parent / "receipt.json")}


def seed_records(cell: Any) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(cell.output_dir).glob("seed_*.json"))]


def agreement(first: Any, second: Any, seed: int) -> dict[str, bool]:
    """Whether two cells of one truth saved the same outputs, labels, masks, and observation ids for every role."""
    a = load_seed_predictions(str(Path(first.output_dir) / f"seed_{seed}.npz"))
    b = load_seed_predictions(str(Path(second.output_dir) / f"seed_{seed}.npz"))
    return {
        f"{role}.{field}": bool(np.array_equal(np.asarray(getattr(a[role], field)), np.asarray(getattr(b[role], field))))
        for role in a for field in ("outputs", "labels", "label_mask", "observation_ids")
    }


def consumed_rows(split: Mapping[str, int | None], release_rows: Mapping[str, int]) -> dict[str, Any]:
    """The rows the cell read from each split beside the rows its release holds, so a dropped row cannot pass as a saving.

    A validation-only fit never loads the calibration or test split, which the report records as None.
    """
    roles = {"train": "train", "v_fit": "validation", "v_cal": "calibration", "test": "test"}
    counted = {role: split[role] for role in roles}
    return {"split": counted, "release": {role: release_rows[name] for role, name in roles.items()},
            "matches_release": all(rows is None or rows == release_rows[roles[role]] for role, rows in counted.items())}


def prediction_facts(cell: Any, seed: int) -> dict[str, dict[str, Any]]:
    """The dtype and shape of each role's saved outputs, read from the mapped arrays without loading them."""
    saved = load_seed_predictions(str(Path(cell.output_dir) / f"seed_{seed}.npz"))
    return {role: {"dtype": str(part.outputs.dtype), "shape": list(part.outputs.shape)} for role, part in saved.items()}


def first_sequences(path: Path, count: int) -> list[str]:
    batch = next(pq.ParquetFile(path).iter_batches(batch_size=count, columns=["seqs"]))
    return batch.column("seqs").to_pylist()


def throughput(dims: Dimensions, recorder: StageRecorder, mode: str, seeds: list[dict[str, Any]], batch_size: int) -> dict[str, float | None]:
    """Rows per second of each stage that scales with rows, from the first run of the cell."""
    rows = dims.total_rows if mode == "full" else dims.train_rows + dims.scored_rows

    def per_second(name: str) -> float | None:
        seconds = recorder.first_seconds(name)
        return round(rows / seconds, 1) if seconds else None

    outcome = (seeds[0]["training_outcome"] or {}) if seeds else {}
    train_seconds = seeds[0]["train_seconds"] if seeds else None
    steps = outcome.get("global_step")
    return {
        "release_rows_validated_per_second": per_second("cell.load_release"),
        "feature_rows_joined_per_second": per_second("cell.load_features"),
        "training_steps_per_second": round(steps / train_seconds, 3) if steps and train_seconds else None,
        "training_rows_per_second": round(steps * batch_size / train_seconds, 1) if steps and train_seconds else None,
    }


def run_pilot(
    root: Path, dims: Dimensions, settings: Mapping[str, Any], *, mode: str, epochs: int | None, tier: str, parity: bool = False,
    package_rows: int = PACKAGE_PREDICTION_ROWS, recorder: StageRecorder | None = None,
) -> dict[str, Any]:
    """Write the fixture under `root`, run the path over it, and return the receipt, whole or as far as it got."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode == "validation_only" and parity:
        raise ValueError("parity compares saved test predictions, which a validation-only fit never makes")
    root.mkdir(parents=True, exist_ok=True)
    recorder = recorder or StageRecorder(root)
    main_module = sys.modules[workflow.parse_arguments.__module__]
    load_secrets, real_join = main_module.load_secrets_env, workflow.join_branch_embeddings
    main_module.load_secrets_env = lambda: []  # the pilot reads no credential file, however it is started
    receipt: dict[str, Any] = {"schema": RECEIPT_SCHEMA, "kind": "procedural_software_fixture", "tier": tier, "mode": mode, "dimensions": dims.record(), "failure": None}
    package_features: list[np.ndarray] = []

    def join_and_keep_package_rows(*arguments: Any, **options: Any) -> Any:
        """The workflow's join, keeping the first `package_rows` test rows' features from the first join for the package."""
        joined, widths = real_join(*arguments, **options)
        if mode == "full" and not package_features:
            package_features.append(joined.block([joined.rows[sequence] for sequence in package_sequences]))  # (r, w)
        return joined, widths

    try:
        with recorder.stage("fixture.release"):
            manifest_path, digest = write_release(root / "release", dims)
        release = json.loads(manifest_path.read_text(encoding="utf-8"))
        receipt["release"] = {
            "manifest_sha256": digest, "splits": {name: record["rows"] for name, record in release["splits"].items()},
            "artifacts": release["splits"], "parquet_bytes": sum((root / "release" / record["path"]).stat().st_size for record in release["splits"].values()),
            "label_entries": dims.label_entries(), "scored_observed_targets": dims.scored_observed_targets(),
        }
        cell = cell_from_mapping(cell_mapping(root, f"capacity_{tier}_{mode}", settings, manifest_path, digest, mode=mode, epochs=epochs))
        receipt["cell"] = {"protify_args": dict(cell.protify_args), "uncertainty": asdict(cell.uncertainty)}
        descriptors = store_descriptors(cell)
        with recorder.stage("fixture.features"):
            stores = write_feature_stores(dims, descriptors, root / "embeddings")
        for name, store in stores.items():
            store["bytes"], store["descriptor"] = tree_bytes(Path(store["directory"])), descriptors[name]
            store["directory"] = Path(store["directory"]).name
        receipt["features"] = stores
        package_sequences = first_sequences(root / "release" / release["splits"]["test"]["path"], min(package_rows, dims.scored_rows))
        workflow.join_branch_embeddings = join_and_keep_package_rows
        with instrumented(recorder, workflow, WORKFLOW_STAGES):
            with recorder.stage("cell.run"):
                report = workflow.run_uncertainty_cell(cell)
            trained = recorder.calls("seed")
            with recorder.stage("cell.resume"):
                resumed = workflow.run_uncertainty_cell(cell)
        receipt["resume"] = {"seconds": recorder.first_seconds("cell.resume"), "seeds_trained_again": recorder.calls("seed") - trained, "report_identical": resumed == report}
        receipt["seeds"] = [
            {key: record.get(key) for key in ("seed", "train_seconds", "v_fit_primary", "v_fit_nll", "inference_seconds_per_1000_rows", "scored_splits")}
            | {"training_outcome": record.get("training", {}).get("outcome")}
            for record in seed_records(cell)
        ]
        receipt["report"] = {"split": report["split"], "evaluation_mode": report["evaluation_mode"], "methods": sorted(report["methods"]), "input_size": report["input_size"]}
        receipt["consumed"] = consumed_rows(report["split"], receipt["release"]["splits"])
        receipt["predictions"] = prediction_facts(cell, int(cell.protify_args["seed"]))
        if mode == "full":
            with recorder.stage("package.build"):
                directory = build_package(cell.output_dir, "temperature", str(root / "package"))
            with recorder.stage("package.load"):
                package = load_package(directory)
            with recorder.stage("package.predict"):
                probabilities = np.asarray(package.predict(package_features[0], batch_size=PACKAGE_BATCH_SIZE, device="cuda" if torch.cuda.is_available() else "cpu").probs)  # (r, c)
            receipt["package"] = {"rows": probabilities.shape[0], "shape": list(probabilities.shape), "finite": bool(np.isfinite(probabilities).all()), "bytes": tree_bytes(Path(directory))}
        if parity:
            dense_path, dense_digest = write_release(root / "release_dense", dims, compact=False)
            dense_cell = cell_from_mapping(cell_mapping(root, f"capacity_{tier}_dense", settings, dense_path, dense_digest, mode=mode, epochs=epochs))
            dense_report = workflow.run_uncertainty_cell(dense_cell)
            arrays = agreement(cell, dense_cell, int(cell.protify_args["seed"]))
            receipt["parity"] = {"reports_equal": dense_report["methods"] == report["methods"], "arrays_equal": arrays}
            receipt["parity"]["agree"] = receipt["parity"]["reports_equal"] and all(arrays.values())
    except Exception as error:  # noqa: broad-except the pilot records any stage failure in its receipt, which is the evidence it exists to produce
        receipt["failure"] = {"stages": recorder.failed_stages(), "exception": type(error).__name__, "message": str(error)[:600], "traceback": traceback.format_exc()[-6000:]}
        print(f"failed in {receipt['failure']['stages']}: {type(error).__name__}: {error}", flush=True)
    finally:
        workflow.join_branch_embeddings = real_join
        main_module.load_secrets_env = load_secrets
        recorder.close()
    receipt["stages"] = recorder.records
    receipt["peaks"] = dict(recorder.peaks)
    receipt["peak_stage"] = max(recorder.records, key=lambda record: record["rss_peak_bytes"])["name"] if recorder.peaks["rss_bytes"] else None
    receipt["throughput"] = throughput(dims, recorder, mode, receipt.get("seeds", []), int(receipt.get("cell", {}).get("protify_args", {}).get(TRAINING_BATCH_ARGUMENT, 0)) or 1)
    receipt["host"] = host_facts()
    receipt["shipment"] = shipment_identity()
    receipt["code"] = loaded_source_digests()
    resume = receipt.get("resume", {})
    receipt["complete"] = bool(
        receipt["failure"] is None and resume.get("report_identical") and resume.get("seeds_trained_again") == 0
        and receipt.get("consumed", {}).get("matches_release") and (mode == "validation_only" or receipt["package"]["finite"])
        and receipt.get("parity", {"agree": True})["agree"]
    )
    return receipt


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Write `document` whole or not at all: the file a reader finds is never half written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(document, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="a fresh directory on the disk that holds the run")
    parser.add_argument("--settings", type=Path, required=True, help="JSON with the production cell's protify_args and uncertainty")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--epochs", type=int, default=None, help="overrides num_epochs; the receipt records it")
    parser.add_argument("--parity", action="store_true")
    for name in ("train_rows", "scored_rows", "labels", "positives", "dense_width", "sae_width", "sae_nonzeros"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, required=True, dest=name)
    parser.add_argument("--observed-fraction", type=float, required=True, help="the share of a training row's labels whose truth is observed")
    parser.add_argument("--scored-observed-fraction", type=float, default=None, help="the same for validation, test, and calibration rows; default the training share")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.root.exists() and any(arguments.root.iterdir()):
        raise SystemExit("--root must be empty or absent: a pilot measures a fresh disk")
    fields = ("train_rows", "scored_rows", "labels", "positives", "observed_fraction", "scored_observed_fraction", "dense_width", "sae_width", "sae_nonzeros", "seed")
    dims = Dimensions(**{name: getattr(arguments, name) for name in fields})
    settings = json.loads(arguments.settings.read_text(encoding="utf-8"))
    started = time.monotonic()
    arguments.root.mkdir(parents=True, exist_ok=True)
    recorder = StageRecorder(arguments.root, progress_path=arguments.receipt.with_name(arguments.receipt.stem + ".progress.json"))
    receipt = run_pilot(arguments.root, dims, settings, mode=arguments.mode, epochs=arguments.epochs, tier=arguments.tier, parity=arguments.parity, recorder=recorder)
    receipt["total_seconds"] = round(time.monotonic() - started, 1)
    write_json(arguments.receipt, receipt)
    print(json.dumps({key: receipt[key] for key in ("tier", "mode", "complete", "failure", "peaks", "peak_stage", "resume", "throughput", "total_seconds") if key in receipt}, sort_keys=True, indent=1), flush=True)
    return 0 if receipt["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
