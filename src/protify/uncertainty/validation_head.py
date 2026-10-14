"""Reproduce frozen validation-only heads without opening calibration or test payloads."""

from __future__ import annotations

import os
import numpy as np
import torch

from typing import Any

from .calibration_split import row_digests
from .cell import UncertaintyCell
from .predictions import collect_predictions
from .report import read_json
from .workflow import (
    CELL_RECORD,
    REPORT,
    STANDARDIZER_RECORD,
    _record_cell,
    cell_arguments,
    head_directory,
    load_cell_branches,
    load_split_cell_data,
    load_standardizer_record,
)
from ..data.branch_features import JoinedBranchCollator, JoinedBranchDataset, parse_branch_spec
from ..probes.projected_probe import ProjectedProbe


def verify_validation_head(
    cell_dir: str, seed: int, rows: JoinedBranchDataset, joined: Any,
    *, kind: str, batch_size: int, device: torch.device,
) -> float:
    """Reproduce saved V_fit outputs and truth in the declared observation order."""
    standardizer, _ = load_standardizer_record(os.path.join(cell_dir, STANDARDIZER_RECORD))
    collator = JoinedBranchCollator(joined, False, standardizer)
    model = ProjectedProbe.from_pretrained(head_directory(cell_dir, seed))
    predictions, _ = collect_predictions(model, rows, collator, task_kind=kind, batch_size=batch_size, device=device)
    with np.load(os.path.join(cell_dir, f"seed_{seed}.npz"), allow_pickle=False) as archive:
        if any(name.startswith(("test__", "v_cal__")) for name in archive.files):
            raise ValueError("validation-only source predictions must contain V_fit alone")
        for name in ("labels", "label_mask", "observation_ids"):
            value = getattr(predictions, name)
            if value is None or not np.array_equal(archive[f"v_fit__{name}"], value):
                raise ValueError(f"saved V_fit {name} differ from the declared validation rows")
        saved = archive["v_fit__outputs"]  # (n_fit, c)
        if saved.shape != predictions.outputs.shape:
            raise ValueError("saved V_fit output shape differs from the reloaded head")
        worst = float(np.abs(predictions.outputs - saved).max())
    if not np.isfinite(worst) or worst > 1e-3:
        raise ValueError(f"the served head's reloaded V_fit outputs differ by up to {worst:.2e}")
    return worst


def verify_validation_cell(cell: UncertaintyCell) -> dict[str, Any]:
    """Check every declared seed using train/validation only; fitting artifacts stay unchanged.

    This verifies prediction parity, not suite selection or a completed calibration. The
    caller must authenticate the selected cell and bind source bytes before invoking it.
    """
    settings = cell.uncertainty
    if (settings.evaluation_mode != "validation_only" or settings.dataset_manifest is None
            or settings.head != "projected" or not settings.save_heads or not settings.require_cached_embeddings):
        raise ValueError("validation preflight requires a declared validation-only projected cell with frozen caches and saved heads")
    record_path = os.path.join(cell.output_dir, CELL_RECORD)
    read_json(record_path)  # Refuse a missing record before the resume checker could create one.
    _record_cell(cell)
    report = read_json(os.path.join(cell.output_dir, REPORT))
    seeds = list(range(cell.protify_args["seed"], cell.protify_args["seed"] + cell.protify_args["num_runs"]))
    if (report.get("evaluation_mode") != "validation_only" or report.get("scored_splits") != ["v_fit"]
            or report.get("methods") != {} or report.get("seeds") != seeds):
        raise ValueError("validation preflight requires a complete validation-only report and its declared seeds")
    specs = [parse_branch_spec(token) for token in settings.branch_specs]
    cell_data, _ = load_split_cell_data(
        cell, cell_arguments(cell, specs), None, include_test=False, include_calibration=False, file_logging=False,
    )
    if cell_data.release != report.get("dataset_release") or cell_data.release != read_json(os.path.join(cell.output_dir, "dataset_release.json")):
        raise ValueError("validation preflight dataset differs from the frozen fit release")
    joined, _ = load_cell_branches(cell, cell_data, specs, cached_only=True)
    rows = JoinedBranchDataset(cell_data.valid, joined, cell_data.label_type, False, False, frozen_release=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agreement = {str(seed): verify_validation_head(
        cell.output_dir, seed, rows, joined, kind=cell_data.kind, batch_size=cell_data.main.trainer_args.probe_batch_size, device=device,
    ) for seed in seeds}
    return {
        "evaluation_scope": "validation_preflight", "loaded_splits": list(cell_data.loaded_splits),
        "excluded_splits": ["calibration", "test"], "seeds": seeds, "validation_rows": len(rows),
        "validation_families": len(set(cell_data.valid["cluster_id"])), "validation_row_digests": row_digests(cell_data.valid),
        "maximum_absolute_output_difference": agreement, "absolute_tolerance": 1e-3,
    }
