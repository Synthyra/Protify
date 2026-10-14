"""Cross-fit a finished cell's head by protein cluster, and record signals that are not its confidence.

A cross-fit plan assigns every training protein, at each of several identity levels, to one of `k`
folds by its cluster at that level. A fold head trains on the other folds, early-stops on V_fit as
the cell's head did, and predicts the held-out rows. Holding out whole clusters at level L leaves
each held-out row below about L identity to what its fold head trained on, so the levels together
give out-of-fold predictions at every identity to training that a test row can have, from many more
protein families than V_cal holds. The fold heads are for calibration only: the served head is
still the cell's own, trained on all of train, loaded from `heads/seed_<seed>`.

For every out-of-fold row under its fold head, and for every V_fit, V_cal, and test row under the
served head, `signal_arrays` records the head's outputs and signals read from its training rows
rather than from its confidence:
- the `NEIGHBORS` nearest training rows, as indices into train and Euclidean distances, in the
  head's trunk space and in its clamped, standardized input, from which neighbor labels, trust
  scores, and density follow;
- Gaussian scores of the trunk features (Lee et al. 2018; Ren et al. 2021; Mukhoti et al. 2023):
  the squared Mahalanobis distance to the training mean, and for single-label tasks to each class
  mean under the shared covariance, and each class's Gaussian log density under its own.

`crossfit_identity` searches every pooled sequence against every other once, then gives each
out-of-fold row its highest identity to its fold's training proteins, and each V_fit, V_cal, and
test row its highest identity to train.

    python -m protify.uncertainty crossfit <cell dir> --plan <plan.json> --output <dir>
    python -m protify.uncertainty crossfit-identity <cell dir> --plan <plan.json> --output <dir>
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import numpy as np
import torch

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from torch import Tensor

from foundry.datasets.homology import DEFAULT_SEARCH, search_command
from foundry.datasets.mmseqs import fasta_bytes, run_command
from .answering import decode, requires_a_label
from .calibration_split import resolve_validation_roles, sequence_columns, take_rows
from .cell import UncertaintyCell, cell_from_mapping
from .correctness import (
    LEVEL,
    OUT_OF_FOLD_ROWS,
    REFERENCE_ROWS,
    dataset_batches,
    fit_reference,
    head_features,
    out_of_fold_columns,
    reference_sample,
)
from .crossfit_artifacts import CALIBRATION_SCOPE, calibration_source, record_calibration_artifacts, verify_calibration_artifacts
from .identity_bins import sequence_digest
from .latent_geometry import gaussian_scores, nearest_rows
from .methods import probabilities
from .predictions import collect_predictions, trunk_features
from .report import read_json, write_json
from .validation_head import verify_validation_head
from .workflow import (
    CELL_RECORD,
    STANDARDIZER_RECORD,
    _train_head,
    cell_arguments,
    head_directory,
    load_cell_branches,
    load_split_cell_data,
    load_standardizer_record,
)
from ..data.branch_features import (
    JoinedBranchCollator,
    JoinedBranchDataset,
    display_name,
    fit_joined_standardizer,
    parse_branch_spec,
)
from ..data.cluster_sampling import validate_sampling_receipt
from ..data.label_batches import SparseObservedTargets, observed_row_counts
from ..probes.projected_probe import ProjectedProbe


PLAN_SCHEME = "cluster_levels"
NEIGHBORS = 32
SERVED = "served"
EVALUATION_SPLITS = ("v_fit", "v_cal", "test")
RECORD = "crossfit.json"
SERVED_CORRECTNESS = "served_correctness.npz"
REFERENCE_DIRECTORY = "reference"
# Test rows whose raw features go beside the served signals, for a package to be checked against.
CHECK_ROWS = 64


@dataclass(frozen=True)
class CrossfitPlan:
    """Each training protein's fold at each identity level, keyed by sequence digest."""

    levels: tuple[str, ...]
    folds: int
    seed: int
    assignments: Mapping[str, Mapping[str, int]]  # level -> protein -> fold

    def fold_names(self) -> list[str]:
        return [fold_name(level, fold) for level in self.levels for fold in range(self.folds)]


def fold_name(level: str, fold: int) -> str:
    return f"l{level.replace('.', '')}_f{fold}"


def read_plan(path: str) -> CrossfitPlan:
    document = read_json(path)
    if document.get("scheme") != PLAN_SCHEME:
        raise ValueError(f"{path} is not a {PLAN_SCHEME} cross-fit plan.")
    levels = tuple(document["levels"])
    assignments = {level: dict(document["assignments"][level]) for level in levels}
    for level, folds in assignments.items():
        bad = sorted({fold for fold in folds.values() if not 0 <= fold < document["folds"]})
        if bad:
            raise ValueError(f"{path} assigns folds {bad} at level {level}, outside 0..{document['folds'] - 1}.")
    return CrossfitPlan(levels, int(document["folds"]), int(document["seed"]), assignments)


def held_out_rows(plan: CrossfitPlan, level: str, fold: int, train_proteins: Sequence[str]) -> np.ndarray:
    """The training rows whose protein the plan puts in `fold` at `level`; every protein must be assigned."""
    assignment = plan.assignments[level]
    missing = sorted({protein for protein in train_proteins if protein not in assignment})
    if missing:
        raise ValueError(f"the plan assigns no fold at level {level} to {len(missing)} training proteins, such as {missing[:3]}.")
    return np.flatnonzero(np.array([assignment[protein] == fold for protein in train_proteins]))  # (h,)


@torch.no_grad()
def clamped_inputs(dataset: JoinedBranchDataset, collator: JoinedBranchCollator, clip: float | None, batch_size: int, device: torch.device) -> Tensor:
    """The standardized, clamped rows the head reads, in float16 on `device`, `(n, d_in)`."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    blocks = []
    for batch in loader:
        rows = batch["embeddings"].to(device)  # (b, d_in)
        if clip is not None:
            rows = rows.clamp(-clip, clip)
        blocks.append(rows.half())
    return torch.cat(blocks)  # (n, d_in)


def signal_arrays(
    model: ProjectedProbe,
    reference: JoinedBranchDataset,
    reference_labels: np.ndarray,
    queries: Mapping[str, JoinedBranchDataset],
    collator: JoinedBranchCollator,
    kind: str,
    num_labels: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """The head's outputs and every signal above, for each named query split, as `<split>__<name>` arrays."""
    # reference_labels: (r,) or (r, c)
    for split, dataset in [("reference", reference), *queries.items()]:
        _require_complete_targets(dataset.labels, dataset.label_masks, split)
    arrays: dict[str, np.ndarray] = {}
    reference_trunk = trunk_features(model, reference, collator, batch_size=batch_size, device=device)  # (r, p)
    reference_inputs = clamped_inputs(reference, collator, model.input_clip, batch_size, device)  # (r, d_in)
    single_label = kind in ("binary", "multiclass")
    for split, dataset in queries.items():
        predictions, _ = collect_predictions(model, dataset, collator, task_kind=kind, batch_size=batch_size, device=device)
        arrays[f"{split}__outputs"] = predictions.outputs.astype(np.float32)  # (n, c) or (n,)
        arrays[f"{split}__labels"] = predictions.labels
        query_trunk = trunk_features(model, dataset, collator, batch_size=batch_size, device=device)  # (n, p)
        indices, distances = nearest_rows(query_trunk, reference_trunk, NEIGHBORS)
        arrays[f"{split}__trunk_neighbors"], arrays[f"{split}__trunk_distances"] = indices, distances
        query_inputs = clamped_inputs(dataset, collator, model.input_clip, batch_size, device)  # (n, d_in)
        indices, distances = nearest_rows(query_inputs.float(), reference_inputs.float(), NEIGHBORS)
        arrays[f"{split}__input_neighbors"], arrays[f"{split}__input_distances"] = indices, distances
        gaussians = gaussian_scores(
            reference_trunk, reference_labels if single_label else None, query_trunk, num_labels if single_label else None
        )
        arrays.update({f"{split}__{name}": value for name, value in gaussians.items()})
        del query_trunk, query_inputs
    return arrays  # each array by name, such as <split>__outputs (n, c) or (n,)


def _require_complete_targets(labels: Sequence[Any], masks: Sequence[Any] | None, name: str) -> None:
    """Do not turn missing truth into correctness labels or class-conditioned references."""
    if masks is None:
        return
    if len(labels) != len(masks):
        raise ValueError(f"cross-fit {name} labels and masks have different row counts")
    for label, mask in zip(labels, masks, strict=True):
        _, observed, total = observed_row_counts(label, mask)
        if observed != total:
            raise ValueError(f"cross-fit {name} has partial labels; correctness and reference signals require complete observed truth")


def _save_arrays(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    # arrays: (...) per name, any shape
    temporary = path.with_name(path.name + ".partial")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _finished_cell(cell_dir: str) -> tuple[UncertaintyCell, int]:
    """The finished cell in `cell_dir`, whose head `save_heads` kept, and its first seed."""
    record = read_json(os.path.join(cell_dir, CELL_RECORD))
    cell = cell_from_mapping({**record, "output_dir": cell_dir})
    seed = int(cell.protify_args["seed"])
    if not os.path.isdir(head_directory(cell_dir, seed)):
        raise ValueError(f"{cell_dir} kept no head for seed {seed}; cross-fitting needs a cell run with save_heads.")
    return cell, seed


def _sae_columns(specs: Sequence[Any], widths: Sequence[int]) -> list[int]:
    """The input columns of every SAE branch, in the joined order."""
    columns, start = [], 0
    for spec, width in zip(specs, widths, strict=True):
        if spec.sparse:
            columns.extend(range(start, start + width))
        start += width
    return columns


def _held_sample(held: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """At most `limit` of a fold's held-out rows, ascending, a seeded sample where there are more."""
    # held: (h,)
    if held.size <= limit:
        return held  # (h,)
    return np.sort(np.random.default_rng(seed).choice(held, limit, replace=False))  # (limit,)


def crossfit_cell(
    cell_dir: str, plan_path: str, output_dir: str,
    *, m8_signals: bool = True, correctness: bool = True, calibration_only: bool = False, served_seed: int | None = None,
) -> dict[str, Any]:
    """Every fold head of the plan, then the served head's signals, each resumable by its file.

    `m8_signals` records `signal_arrays` for every row, as the family cross-fitting rounds read
    them; it holds every training row's input on the device, so the suite, which reads only the
    correctness model, turns it off. `correctness` records the correctness model's columns
    (`correctness.py`) for a sample of each fold's held-out rows, as `correctness__<name>`, and
    for the served head writes its reference to `reference/` and its V_fit, V_cal, and test
    signals to `served_correctness.npz`, with a few test rows' raw features to check a package by.
    Regression has no correctness model. `calibration_only` takes a declared validation-only
    classification cell and the accepted five-fold compact plan. It opens no test payload,
    regenerates V_cal from the frozen served head, and records V_cal package-check rows.
    This produces cross-fit inputs, not a completed suite calibration or a shippable package.
    """
    cell, seed = _finished_cell(cell_dir)
    if served_seed is not None:
        expected_seeds = range(seed, seed + cell.protify_args.get("num_runs", 1))
        if not calibration_only or type(served_seed) is not int or served_seed not in expected_seeds:
            raise ValueError("an explicit served seed must belong to a calibration-only cell's declared seeds")
        seed = served_seed
    plan = read_plan(plan_path)
    if calibration_only:
        if (cell.uncertainty.evaluation_mode != "validation_only" or cell.uncertainty.dataset_manifest is None
                or cell.uncertainty.head != "projected" or (cell.protify_args.get("num_runs", 1) != 1 and served_seed is None)):
            raise ValueError("calibration-only cross-fitting requires one declared validation-only projected head")
        if m8_signals or not correctness or plan.levels != (LEVEL,) or plan.folds != 5:
            raise ValueError("calibration-only cross-fitting requires correctness-only and five folds at level 0.3")
    elif getattr(cell.uncertainty, "evaluation_mode", "full") != "full":
        raise ValueError("a validation-only cell requires calibration-only cross-fitting")
    specs = [parse_branch_spec(token) for token in cell.uncertainty.branch_specs]
    arguments = cell_arguments(cell, specs)
    if calibration_only:
        cell_data, explicit_rows = load_split_cell_data(cell, arguments, None, include_test=False, file_logging=False)
        if cell_data.kind == "regression":
            raise ValueError("regression uses split conformal; calibration-only cross-fitting is for classification")
    else:
        cell_data, explicit_rows = load_split_cell_data(cell, arguments, None)
    loaded = [("train", cell_data.train), ("validation", cell_data.valid)]
    if not calibration_only:
        loaded.append(("test", cell_data.test))
    for name, rows in loaded:
        columns = getattr(rows, "column_names", rows)
        _require_complete_targets(rows["labels"], rows["label_mask"] if "label_mask" in columns else None, name)
    if cell_data.ppi:
        raise ValueError("cross-fitting by protein cluster takes single-sequence datasets only.")
    main = cell_data.main
    roles = resolve_validation_roles(
        list(cell_data.valid["labels"]), cell_data.label_type,
        fraction=cell.uncertainty.calibration_fraction, seed=cell.uncertainty.split_seed, explicit_rows=explicit_rows,
    )
    v_fit, v_cal = take_rows(cell_data.valid, roles.fit), take_rows(cell_data.valid, roles.cal)

    run_arguments = dict(vars(main.full_args))
    source = calibration_source(cell, seed, specs, run_arguments) if calibration_only else None
    joined, widths = load_cell_branches(cell, cell_data, specs, cached_only=True)
    main.probe_args.input_size = joined.width
    main.probe_args.branch_widths = list(widths)
    if cell.uncertainty.branch_projection_sizes is not None:
        main.probe_args.branch_sizes = list(cell.uncertainty.branch_projection_sizes)
    main.probe_args.num_labels = cell_data.num_labels
    main.probe_args.task_type = cell_data.label_type
    main.trainer_args.task_type = cell_data.label_type
    kind = cell_data.kind
    batch_size = main.trainer_args.probe_batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def dataset(rows: Any, train: bool = False) -> JoinedBranchDataset:
        return JoinedBranchDataset(
            rows, joined, cell_data.label_type, False, False, train=train, frozen_release=calibration_only,
            cluster_draws_per_epoch=cell.uncertainty.cluster_draws_per_epoch if calibration_only else None,
        )

    if isinstance(cell_data.train["labels"][0], Mapping):
        # Cross-fit reference fitting still consumes a full (n, c) matrix. Its memory gate
        # remains separate from compact source loading and batchwise head training.
        train_labels = SparseObservedTargets.from_rows(cell_data.train["labels"], cell_data.train["label_mask"]).values.toarray()
    else:
        train_labels = np.asarray(list(cell_data.train["labels"]))  # (n,) or (n, c)
    train_proteins = list(cell_data.train["protein_id"]) if calibration_only else [sequence_digest(sequence) for sequence in cell_data.train["seqs"]]
    evaluation = {"v_fit": dataset(v_fit), "v_cal": dataset(v_cal)}
    if not calibration_only:
        evaluation["test"] = dataset(cell_data.test)
    correctness = correctness and kind != "regression"
    sae_columns = _sae_columns(specs, widths)
    if kind == "multilabel" and isinstance(v_fit["labels"][0], Mapping):
        at_least_one_label = all(bool(np.all(evaluation[name].sparse_targets.values.getnnz(axis=1) > 0)) for name in ("v_fit", "v_cal"))
    else:
        at_least_one_label = kind == "multilabel" and requires_a_label(
            [np.asarray(list(v_fit["labels"])), np.asarray(list(v_cal["labels"]))]
        )
    # The served head reads up to REFERENCE_ROWS of train; a fold head keeps the same share of its own.
    served_reference = min(len(train_proteins), REFERENCE_ROWS)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / RECORD
    if calibration_only and not record_path.exists() and any(output.iterdir()):
        raise ValueError("calibration-only cross-fitting requires an empty output or a matching receipt")
    record = read_json(str(record_path)) if record_path.exists() else {
        "cell_dir": cell_dir,
        "plan_sha256": hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
        "train_rows": len(train_proteins),
        "plan": {"levels": list(plan.levels), "folds": plan.folds},
        "served_seed": seed,
        "folds": {},
        "evaluation_scope": CALIBRATION_SCOPE if calibration_only else "full",
        "evaluation_splits": list(evaluation),
    }
    if record.get("evaluation_scope", "full") != (CALIBRATION_SCOPE if calibration_only else "full"):
        raise ValueError("cross-fit evaluation scope changed; use a new output directory")
    if record["plan_sha256"] != hashlib.sha256(Path(plan_path).read_bytes()).hexdigest():
        raise ValueError(f"{output_dir} was cross-fit under another plan; use a new output directory.")
    if calibration_only:
        if record_path.exists():
            if record.get("source") != source:
                raise ValueError("calibration source files or feature caches changed; use a new output directory")
            verify_calibration_artifacts(output, record)
        else:
            record.update(source=source, artifact_sha256={}, status="running", task_type=kind, num_labels=cell_data.num_labels, split_access={
                "loaded": list(cell_data.loaded_splits), "excluded": ["test"], "membership_checks": "loaded_splits_only",
            })
        for level in plan.levels:
            assignment = plan.assignments[level]
            if set(assignment) != set(train_proteins):
                raise ValueError("calibration cross-fit plan must assign exactly the declared training proteins")
            if set(assignment.values()) != set(range(plan.folds)):
                raise ValueError("calibration cross-fit plan has an empty fold")
            cluster_folds = {}
            for protein, cluster in zip(train_proteins, cell_data.train["cluster_id"], strict=True):
                if cluster_folds.setdefault(cluster, assignment[protein]) != assignment[protein]:
                    raise ValueError("calibration cross-fit plan splits a declared protein family")
        record["validation_head_agreement"] = verify_validation_head(
            cell_dir, seed, evaluation["v_fit"], joined, kind=kind, batch_size=batch_size, device=device,
        )
        write_json(str(record_path), record)

    for level in plan.levels:
        for fold in range(plan.folds):
            name = fold_name(level, fold)
            path = output / f"{name}.npz"
            if path.exists() and (not calibration_only or path.name in record["artifact_sha256"]):
                with np.load(path) as saved:
                    if correctness and "correctness__rows" not in saved.files:
                        raise ValueError(f"{path} was cross-fit without the correctness columns; use a new output directory.")
                continue
            started = time.perf_counter()
            held = held_out_rows(plan, level, fold, train_proteins)  # (h,)
            kept = np.setdiff1d(np.arange(len(train_proteins)), held)  # (r,)
            fold_train = take_rows(cell_data.train, kept)
            standardizer = fit_joined_standardizer(joined, fold_train, False) if main.embedding_args.embedding_scaler else None
            collator = JoinedBranchCollator(joined, False, standardizer)
            fold_data_name = f"{cell_data.data_name}__crossfit"
            if kind == "regression":
                main._compute_balanced_weights_for(fold_data_name, fold_train, v_fit, v_fit)
            train_dataset = dataset(fold_train, train=True)
            model, v_fit_metrics, _ = _train_head(
                main, seed, train_dataset, evaluation["v_fit"], evaluation["v_fit"], collator,
                f"{cell.cell_id}_{name}", display_name(specs), fold_data_name, False,
                **({"evaluate_test": False} if calibration_only else {}),
            )
            arrays = {}
            if m8_signals:
                arrays = signal_arrays(
                    model, dataset(fold_train), train_labels[kept], {"held": dataset(take_rows(cell_data.train, held)), **evaluation},
                    collator, kind, cell_data.num_labels, batch_size, device,
                )
                # Neighbor indices point into the fold's training rows; store them as rows of train.
                for key in [key for key in arrays if key.endswith("_neighbors")]:
                    arrays[key] = kept[arrays[key]].astype(np.int32)
            if correctness:
                limit = round(served_reference * kept.size / len(train_proteins))
                reference_rows = kept[reference_sample(train_labels[kept], kind, limit, plan.seed)]  # rows of train
                sampled = _held_sample(held, -(-OUT_OF_FOLD_ROWS // plan.folds), plan.seed + fold)  # rows of train
                columns = out_of_fold_columns(
                    model, dataset(take_rows(cell_data.train, reference_rows)), train_labels[reference_rows],
                    dataset(take_rows(cell_data.train, sampled)), evaluation["v_fit"], collator,
                    kind=kind, num_labels=cell_data.num_labels, sae_columns=sae_columns, at_least_one_label=at_least_one_label,
                    batch_size=batch_size, device=device,
                )
                arrays.update({f"correctness__{column}": value for column, value in columns.items()})
                arrays["correctness__rows"] = sampled.astype(np.int32)
                arrays["correctness__reference_rows"] = np.asarray(reference_rows.size, dtype=np.int64)
            arrays["held__rows"] = held.astype(np.int32)
            _save_arrays(path, arrays)
            record["folds"][name] = {
                "held_out_rows": int(held.size),
                "training_rows": int(kept.size),
                "seconds": time.perf_counter() - started,
                "v_fit_metrics": v_fit_metrics,
            }
            if calibration_only:
                validate_sampling_receipt(train_dataset.sampling_plan, train_dataset.sampling_receipt, seed)
                record["folds"][name].update(
                    sampling=train_dataset.sampling_receipt,
                    training={"recipe": train_dataset.training_recipe, "outcome": train_dataset.training_outcome, "seed": seed},
                )
                record_calibration_artifacts(output, record, [path])
            write_json(str(record_path), record)
            del model

    served_path = output / f"{SERVED}.npz"
    served_correctness_path = output / SERVED_CORRECTNESS
    if (served_path.exists() and (served_correctness_path.exists() or not correctness)
            and (not calibration_only or record.get("status") == "complete")):
        return record

    standardizer, _ = load_standardizer_record(os.path.join(cell_dir, STANDARDIZER_RECORD))
    collator = JoinedBranchCollator(joined, False, standardizer)
    model = ProjectedProbe.from_pretrained(head_directory(cell_dir, seed))
    saved_splits = ("v_fit",) if calibration_only else EVALUATION_SPLITS
    with np.load(os.path.join(cell_dir, f"seed_{seed}.npz"), allow_pickle=False) as archive:
        saved = {f"{split}__outputs": archive[f"{split}__outputs"] for split in saved_splits}
        if calibration_only and any(name.startswith(("test__", "v_cal__")) for name in archive.files):
            raise ValueError("validation-only source predictions must contain V_fit alone")
    if not served_path.exists() or (calibration_only and served_path.name not in record["artifact_sha256"]):
        if m8_signals:
            arrays = signal_arrays(model, dataset(cell_data.train), train_labels, evaluation, collator, kind, cell_data.num_labels, batch_size, device)
        else:
            arrays = {}
            for split, rows in evaluation.items():
                predictions, _ = collect_predictions(model, rows, collator, task_kind=kind, batch_size=batch_size, device=device)
                arrays[f"{split}__outputs"], arrays[f"{split}__labels"] = predictions.outputs.astype(np.float32), predictions.labels
        worst = max(float(np.abs(arrays[f"{split}__outputs"] - saved[f"{split}__outputs"]).max()) for split in saved_splits)
        if not np.isfinite(worst) or worst > 1e-3:
            raise ValueError(f"the served head's reloaded outputs differ from seed_{seed}.npz by up to {worst:.2e}.")
        arrays["train__labels"] = train_labels
        _save_arrays(served_path, arrays)
        record["served_output_agreement"] = worst
        if calibration_only:
            record["agreement_splits"] = list(saved_splits)
            record_calibration_artifacts(output, record, [served_path])
        write_json(str(record_path), record)
    if calibration_only:
        with np.load(served_path, allow_pickle=False) as archive:
            saved["v_cal__outputs"] = archive["v_cal__outputs"]
    if correctness:
        record["served_correctness"] = served_correctness(
            model, dataset, cell_data.train, train_labels, evaluation, saved, collator, joined, output,
            kind=kind, num_labels=cell_data.num_labels, sae_columns=sae_columns, at_least_one_label=at_least_one_label,
            reference_rows=reference_sample(train_labels, kind, served_reference, plan.seed),
            batch_size=batch_size, device=device,
            check_split="v_cal" if calibration_only else "test",
        )
        if calibration_only:
            paths = [served_correctness_path, *(output / REFERENCE_DIRECTORY).iterdir()]
            record_calibration_artifacts(output, record, paths)
        write_json(str(record_path), record)
    if calibration_only:
        if calibration_source(cell, seed, specs, run_arguments) != source:
            raise ValueError("calibration source changed during cross-fitting")
        record["status"] = "complete"
        write_json(str(record_path), record)
    return record


def served_correctness(
    model: ProjectedProbe,
    dataset: Callable[[Any], JoinedBranchDataset],
    train: Any,
    train_labels: np.ndarray,
    evaluation: Mapping[str, JoinedBranchDataset],
    saved: Mapping[str, np.ndarray],
    collator: JoinedBranchCollator,
    joined: Any,
    output: Path,
    *,
    kind: str,
    num_labels: int,
    sae_columns: Sequence[int],
    at_least_one_label: bool,
    reference_rows: np.ndarray,
    batch_size: int,
    device: torch.device,
    check_split: str = "test",
) -> dict[str, Any]:
    """The served head's reference, written to `reference/`, and its signals on V_fit, V_cal, and test.

    Each split's answers are decoded from the outputs the cell saved for the head, `saved`, which a
    package reads too: a temperature changes neither an argmax nor which side of 0.5 a sigmoid
    falls, so they are the answers the package gives. The first
    `CHECK_ROWS` rows of `check_split` carry raw joined features for reload parity. The
    calibration-only path uses V_cal, and the full research path uses test.
    """
    # train_labels: (n_train,) or (n_train, c), saved: (...) per name, any shape, reference_rows: (r,)
    reference = fit_reference(
        model, dataset(take_rows(train, reference_rows)), train_labels[reference_rows], evaluation["v_fit"], collator,
        kind=kind, num_labels=num_labels, sae_columns=sae_columns, batch_size=batch_size, device=device,
    )
    directory = output / REFERENCE_DIRECTORY
    staged = output / f".{REFERENCE_DIRECTORY}.partial"
    shutil.rmtree(staged, ignore_errors=True)
    staged.mkdir(parents=True)
    reference.save(staged)
    write_json(str(staged / "reference.json"), {"scales": reference.scales, "rows": reference_rows.tolist()})
    shutil.rmtree(directory, ignore_errors=True)
    os.replace(staged, directory)

    arrays = {}
    for split, rows in evaluation.items():
        answers = decode(probabilities(saved[f"{split}__outputs"].astype(np.float64), kind), kind, at_least_one_label)
        trunk, inputs = head_features(model, dataset_batches(rows, collator, batch_size), device)
        for name, value in reference.signals(reference.space.rows(trunk, inputs), answers).items():
            arrays[f"{split}__{name}"] = value
        arrays[f"{split}__answers"] = answers
    check = np.arange(min(CHECK_ROWS, len(evaluation[check_split])))
    arrays["check__rows"] = check
    arrays["check__features"] = joined.block(evaluation[check_split].rows[check, 0])  # (m, W), before standardization
    _save_arrays(output / SERVED_CORRECTNESS, arrays)
    return {"reference_rows": int(reference_rows.size), "scales": reference.scales, "at_least_one_label": at_least_one_label, "check_split": check_split}


def _hit_arrays(hits_path: Path, index: Mapping[str, int], evalue_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Query and target indices and identities of every hit at or below `evalue_max`, self hits dropped."""
    queries, targets, identities = [], [], []
    with hits_path.open(encoding="utf-8") as handle:
        for line in handle:
            query, target, identity, _, evalue, _ = line.rstrip("\n").split("\t")
            if query == target or float(evalue) > evalue_max:
                continue
            queries.append(index[query])
            targets.append(index[target])
            identities.append(float(identity))
    return np.array(queries, dtype=np.int64), np.array(targets, dtype=np.int64), np.array(identities, dtype=np.float32)  # (h,), (h,), (h,)


def best_identity(queries: np.ndarray, targets: np.ndarray, identities: np.ndarray, allowed: np.ndarray, count: int) -> np.ndarray:
    """Each protein's highest identity to a protein `allowed` marks, 0 where it has no such hit, `(count,)`."""
    # queries: (h,), targets: (h,), identities: (h,), allowed: (n_proteins,)
    keep = allowed[targets]  # (h,)
    best = np.zeros(count, dtype=np.float32)  # (count,)
    np.maximum.at(best, queries[keep], identities[keep])
    return best  # (count,)


def crossfit_identity(cell_dir: str, plan_path: str, output_dir: str, *, threads: int | None = None) -> dict[str, Any]:
    """One all-against-all search of the pooled proteins, then each fold's and the served head's identities."""
    cell, _ = _finished_cell(cell_dir)
    plan = read_plan(plan_path)
    specs = [parse_branch_spec(token) for token in cell.uncertainty.branch_specs]
    cell_data, explicit_rows = load_split_cell_data(cell, cell_arguments(cell, specs), None)
    roles = resolve_validation_roles(
        list(cell_data.valid["labels"]), cell_data.label_type,
        fraction=cell.uncertainty.calibration_fraction, seed=cell.uncertainty.split_seed, explicit_rows=explicit_rows,
    )
    splits = {"train": cell_data.train, "v_fit": take_rows(cell_data.valid, roles.fit), "v_cal": take_rows(cell_data.valid, roles.cal), "test": cell_data.test}
    if any(sequence_columns(rows) != ("seqs",) for rows in splits.values()):
        raise ValueError("cross-fit identities take single-sequence datasets only.")
    proteins = {name: [sequence_digest(sequence) for sequence in rows["seqs"]] for name, rows in splits.items()}
    sequences = {sequence_digest(sequence): sequence for rows in splits.values() for sequence in rows["seqs"]}
    order = sorted(sequences)
    index = {protein: position for position, protein in enumerate(order)}

    output = Path(output_dir)
    stage = output / "search"
    stage.mkdir(parents=True, exist_ok=True)
    hits_path = stage / "hits.tsv"
    config = DEFAULT_SEARCH if threads is None else replace(DEFAULT_SEARCH, threads=threads)
    if not hits_path.exists():
        fasta = stage / "pooled.fasta"
        fasta.write_bytes(fasta_bytes({protein: sequences[protein] for protein in order}))
        partial = stage / "hits.partial.tsv"
        run_command(search_command(fasta, fasta, partial, stage / "tmp", config))
        os.replace(partial, hits_path)
        shutil.rmtree(stage / "tmp", ignore_errors=True)
    queries, targets, identities = _hit_arrays(hits_path, index, config.evalue_max)

    train_rows = np.array([index[protein] for protein in proteins["train"]])  # (n_train,)
    in_train = np.zeros(len(order), dtype=bool)
    in_train[train_rows] = True
    served = best_identity(queries, targets, identities, in_train, len(order))  # (proteins,)
    arrays = {f"{split}__identity": served[[index[protein] for protein in proteins[split]]] for split in EVALUATION_SPLITS}
    for level in plan.levels:
        for fold in range(plan.folds):
            held = held_out_rows(plan, level, fold, proteins["train"])  # (h,)
            allowed = in_train.copy()
            allowed[train_rows[held]] = False
            fold_identity = best_identity(queries, targets, identities, allowed, len(order))
            arrays[f"{fold_name(level, fold)}__identity"] = fold_identity[train_rows[held]]  # (h,)
            arrays[f"{fold_name(level, fold)}__rows"] = held.astype(np.int32)
    _save_arrays(output / "identity.npz", arrays)
    summary = {
        "proteins": len(order),
        "hits": int(queries.size),
        "search": config.result_parameters(),
        "plan_sha256": hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
    }
    write_json(str(output / "identity.json"), summary)
    return summary


def plan_document(levels: Sequence[str], folds: int, seed: int, assignments: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    """A plan as `read_plan` reads it."""
    return {"scheme": PLAN_SCHEME, "levels": list(levels), "folds": folds, "seed": seed, "assignments": {level: dict(assignments[level]) for level in levels}}


def assign_folds(cluster_of: Mapping[str, str], proteins: Sequence[str], folds: int, seed: int) -> dict[str, int]:
    """Each protein's fold, whole clusters together, clusters dealt to folds in a seeded random order."""
    clusters = sorted({cluster_of[protein] for protein in proteins})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clusters))
    fold_of_cluster = {clusters[position]: rank % folds for rank, position in enumerate(order)}
    return {protein: int(fold_of_cluster[cluster_of[protein]]) for protein in proteins}


def write_plan(path: str, document: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
