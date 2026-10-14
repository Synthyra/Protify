"""Bind calibration-only cross-fits to frozen source files and resumable output bytes.

This is a local integrity contract. Candidate selection, homology qualification, and the
later locked-test release gate belong to the suite that invokes the cross-fit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from foundry.digests import sha256_file
from .cell import UncertaintyCell
from .report import read_json
from .workflow import CELL_RECORD, REPORT, SPLIT_RECORD, STANDARDIZER_RECORD, head_directory, head_is_saved
from ..data.branch_features import BranchSpec, branch_cache_path
from ..data.feature_inputs import FrozenFeatureInput
from ..feature_cache import require_feature, store_fingerprint


CALIBRATION_SCOPE = "calibration_only"


def calibration_source(
    cell: UncertaintyCell, seed: int, specs: Sequence[BranchSpec], arguments: Mapping[str, object],
) -> dict[str, Any]:
    """Check a finished single-head validation fit and hash its files and frozen caches."""
    root = Path(cell.output_dir)
    report = read_json(str(root / REPORT))
    seed_record = read_json(str(root / f"seed_{seed}.json"))
    for record in (report, seed_record):
        if record.get("evaluation_mode") != "validation_only" or record.get("scored_splits") != ["v_fit"]:
            raise ValueError("calibration-only cross-fitting needs a finished validation-only fit")
    declared_seeds = list(range(cell.protify_args["seed"], cell.protify_args["seed"] + cell.protify_args.get("num_runs", 1)))
    if report.get("seeds") != declared_seeds or seed not in declared_seeds or seed_record.get("seed") != seed or not head_is_saved(str(root), seed):
        raise ValueError("calibration-only cross-fitting needs one complete saved head and its seed record")
    release = read_json(str(root / "dataset_release.json"))
    if release != report.get("dataset_release") or release.get("manifest_sha256") != cell.uncertainty.dataset_sha256:
        raise ValueError("calibration source disagrees on its declared dataset release")
    names = [CELL_RECORD, REPORT, SPLIT_RECORD, STANDARDIZER_RECORD, "dataset_release.json", f"seed_{seed}.json", f"seed_{seed}.npz"]
    names.extend(path.relative_to(root).as_posix() for path in sorted(Path(head_directory(str(root), seed)).rglob("*")) if path.is_file())
    files = {name: sha256_file(root / name) for name in names}
    if cell.uncertainty.feature_manifest is not None:
        settings = cell.uncertainty
        inputs = FrozenFeatureInput(settings.feature_manifest, settings.feature_sha256,
                                    dataset_sha256=settings.dataset_sha256, branch_specs=settings.branch_specs)
        inputs.verify_files()
        features = inputs.files
    else:
        # A store commits whole segments, so a writer still running leaves committed content unchanged
        # and the fingerprint names exactly the parts a reader can see. It refuses a part that differs
        # from its commit marker.
        features = {
            spec.slug: store_fingerprint(require_feature(branch_cache_path(spec, arguments)))
            for spec in specs
        }
    package = Path(__file__).resolve().parents[1]
    implementation = [
        "uncertainty/crossfit.py", "uncertainty/crossfit_artifacts.py", "uncertainty/workflow.py",
        "uncertainty/correctness.py", "uncertainty/methods.py", "uncertainty/latent_geometry.py", "uncertainty/validation_head.py",
        "data/branch_features.py", "data/feature_inputs.py", "data/cluster_sampling.py", "probes/trainers.py",
    ]
    return {
        "files": files, "features": features, "dataset_manifest_sha256": cell.uncertainty.dataset_sha256,
        "implementation": {name: sha256_file(package / name) for name in implementation},
    }


def verify_calibration_artifacts(directory: Path, record: Mapping[str, Any], *, complete: bool = False) -> None:
    """Refuse altered or missing committed outputs; an interrupted uncommitted output is rebuilt."""
    if record.get("evaluation_scope") != CALIBRATION_SCOPE:
        raise ValueError("this directory is not a calibration-only cross-fit")
    if record.get("status") not in {"running", "complete"}:
        raise ValueError("calibration cross-fit has an invalid completion status")
    if complete and record.get("status") != "complete":
        raise ValueError("finish the calibration-only cross-fit before fitting correctness")
    artifacts = record.get("artifact_sha256")
    if not isinstance(artifacts, dict):
        raise ValueError("calibration-only cross-fit lacks its artifact digests")
    if record.get("status") == "complete":
        required = {"served.npz", "served_correctness.npz", "reference/reference.npz", "reference/reference.json"}
        required.update(f"{name}.npz" for name in record["folds"])
        if len(record["folds"]) != 5 or not required <= artifacts.keys():
            raise ValueError("completed calibration cross-fit lacks committed fold or served artifacts")
    for name, digest in artifacts.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"calibration cross-fit artifact digest mismatch: {name}")


def record_calibration_artifacts(directory: Path, record: dict[str, Any], paths: Sequence[Path]) -> None:
    """Add completed output files to the receipt before its atomic write."""
    record.setdefault("artifact_sha256", {}).update({path.relative_to(directory).as_posix(): sha256_file(path) for path in paths})
