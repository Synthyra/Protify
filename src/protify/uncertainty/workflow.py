"""Run one uncertainty cell, and score its dataset's identity to training.

`run_uncertainty_cell` follows one visible sequence:
1. Protify's parsed defaults, with the cell's `protify_args` over them, loaded by MainProcess
   with the data seed fixed, because a per-seed data seed would carve different splits.
2. V_fit and V_cal resolved once from the validation split, and recorded in `split.json`. Under
   `split_manifest` the rows are first split again by identity strata (`stratified_split.py`),
   and V_fit and V_cal are the manifest's validation and calibration splits.
3. Each branch's ordinary cache completed through the Embedder, or under
   `require_cached_embeddings` checked complete without writing, then joined in memory.
4. Seeds `seed`, `seed + 1`, ... trained one after another through Protify's `_train`, each
   writing `seed_<seed>.npz` and `seed_<seed>.json`, so a rerun resumes at the first missing
   seed. Protify's multi-run path keeps the run with the lowest test loss; this keeps them all.
   A seed's files carry nothing cell-specific, so a directory seeded with another cell's
   `seed_<seed>` files and `split.json`, under the same training settings, trains only the rest.
   Under `save_heads` each seed also keeps its trained head in `heads/seed_<seed>/`, which a
   seeded directory must then carry too, and the cell keeps its embedding standardizer in
   `standardizer.npz`, for `package.py`. Under `save_trunk_features` each seed's npz also holds
   every V_fit, V_cal, and test row's trunk features, for `calibration_neighbors.py`.
   Under `laplace` a seed also saves a last-layer Laplace variance (`laplace.py`), and under
   `loss_head_folds` a cross-fitted loss prediction (`loss_head.py`), both after its own
   predictions, so neither changes what the task head trains or predicts.
5. Every method built from the saved predictions, calibrated on V_fit, conformalized on V_cal,
   scored on test, and written to `report.json`.

`evaluation_mode: validation_only` instead saves only V_fit predictions and the selected
head. It skips calibration and test prediction entirely; a separate frozen-checkpoint stage
must provide those artifacts before packaging. The recorded scope cannot change on resume.
It also never loads the calibration or test split of its declared release: those payloads are
not opened, so training cannot read their labels, V_fit is the whole validation split, and
`split.json` records V_cal as empty.

`score_identity` writes each test, validation, V_fit, and V_cal row's highest identity to
training, which `run_uncertainty_cell` reads back for its identity bins. `split_cell` writes the
identity-stratified split manifest that both then read.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
import numpy as np
import torch

from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any
from datasets import concatenate_datasets
from transformers.utils import CONFIG_NAME, SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from foundry.datasets.homology import DEFAULT_SEARCH, max_identity_to_reference
from .calibration_split import (
    ValidationRoles,
    check_manifest,
    resolve_validation_roles,
    roles_manifest,
    row_digests,
    sequence_columns,
    take_rows,
)
from .cell import CLIP_QUANTILES, HEAD_RECIPES, UncertaintyCell, UncertaintySettings
from .identity_bins import bin_counts, bin_rows, row_identity, sequence_digest
from .laplace import fit_kfac, latent_variance
from .loss_head import cross_fitted_losses, log_loss_targets, per_row_losses, stratified_folds
from .methods import LAPLACE_PRIOR_PRECISIONS, build_methods, probabilities
from .metric_contracts import SELECTION_METRICS, canonical_task_metrics, metric_contract
from .metrics import PRIMARY_METRICS, gaussian_nll, negative_log_likelihood, task_kind, task_metrics
from .predictions import (
    SPLITS,
    SplitPredictions,
    collect_predictions,
    load_seed_predictions,
    nearest_neighbor_distance,
    save_seed_predictions,
    trunk_features,
)
from .quantile_clip import ClippedStandardizer, fit_clipped_standardizer, standardized_rows
from .report import read_json, score_method, write_json
from .stratified_split import apply_split_manifest, build_split_manifest
from .verdicts import json_ready
from ..data.branch_features import (
    BranchSpec,
    IndexedBranchEmbeddings,
    JoinedBranchCollator,
    JoinedBranchDataset,
    JoinedBranchEmbeddings,
    display_name,
    embed_missing_branches,
    fit_joined_standardizer,
    join_branch_embeddings,
    parse_branch_spec,
    require_cached_branches,
)
from ..data.cluster_sampling import ClusterSamplingPlan, validate_sampling_receipt
from ..data.dataset_classes import EmbeddingStandardizer
from ..data.declared_dataset import SPLIT_NAMES, load_declared_dataset
from ..data.label_batches import observed_row_counts
from ..data.probe_features import open_probe_features
from ..main import MainProcess, parse_arguments, set_global_seed
from ..metrics import validate_observed_labels
from ..probes.get_probe import get_probe
from ..probes.training_recipe import resolved_training_recipe, validate_training_receipt


# Arguments the cell sets itself, and the uncertainty key that sets each.
CELL_OWNED_ARGUMENTS = {
    "probe_type": "uncertainty.head",
    "projection_size": "uncertainty.projection_size",
    "model_names": "uncertainty.branch_specs",
    "model_paths": "uncertainty.branch_specs",
    "model_types": "uncertainty.branch_specs",
    "embedding_pooling_types": "uncertainty.branch_specs",
    "embedding_hidden_state_index": "uncertainty.branch_specs",
}
# Protify modes this workflow does not run: each must stay off.
UNSUPPORTED_ARGUMENTS = (
    "matrix_embed",
    "multi_column",
    "tokenwise",
    "full_finetuning",
    "hybrid_probe",
    "use_scikit",
    "use_wandb_hyperopt",
    "parallel_probe_runs",
    "feature_selection",
    "save_model",
    "push_raw_probe",
    "proteingym",
    "cloud",
    "yaml_path",
    "replay_path",
)
# Cell settings that change what a seed trains or saves. A resumed run must match them.
TRAINING_SETTINGS = (
    "branch_specs",
    "head",
    "projection_size",
    "branch_projection_sizes",
    "learning_rate",
    "dropout",
    "gp_lengthscale_squared",
    "calibration_fraction",
    "split_seed",
    "mc_dropout_samples",
    "save_heads",
    "save_trunk_features",
    "weight_decay",
    "laplace",
    "loss_head_folds",
    "split_manifest",
    "dataset_manifest",
    "dataset_sha256",
    "feature_manifest",
    "feature_sha256",
    "probe_features",
    "cluster_draws_per_epoch",
    "cluster_class_balance",
    "checkpoint_selection",
    "evaluation_mode",
    "unlisted_labels",
    "selection_metric",
)
# A training setting added after a directory's cell.json was written reads there as its
# default, the value that run trained under, so the directory still resumes.
TRAINING_SETTING_DEFAULTS = {
    setting.name: setting.default
    for setting in fields(UncertaintySettings)
    if setting.name in TRAINING_SETTINGS and setting.default is not MISSING
}
CELL_RECORD = "cell.json"
SPLIT_RECORD = "split.json"
OBSERVATIONS_RECORD = "label_observations.json"
REPORT = "report.json"
# Written only under `save_heads`: each seed's trained head, and the cell's standardizer.
HEADS_DIRECTORY = "heads"
STANDARDIZER_RECORD = "standardizer.npz"


@dataclass
class CellData:
    """The one dataset a cell reads, as MainProcess loaded it."""

    main: MainProcess
    data_name: str
    train: Any
    valid: Any
    test: Any
    num_labels: int
    label_type: str
    ppi: bool
    release: dict[str, Any] | None = None
    loaded_splits: tuple[str, ...] | None = None

    @property
    def kind(self) -> str:
        if self.release is not None:
            return self.release["task_type"]
        return task_kind(self.label_type, self.num_labels)


def protify_arguments(overrides: Mapping[str, Any]) -> argparse.Namespace:
    """Protify's parsed defaults with `overrides` set over them; an unknown name is refused."""
    saved_argv = sys.argv
    sys.argv = saved_argv[:1] or ["protify"]
    try:
        arguments = parse_arguments()
    finally:
        sys.argv = saved_argv

    unknown = sorted(set(overrides) - set(vars(arguments)))
    if unknown:
        raise ValueError(f"protify_args has unknown keys {unknown}; use Protify argument names as main.py spells them.")
    owned = {name: CELL_OWNED_ARGUMENTS[name] for name in overrides if name in CELL_OWNED_ARGUMENTS}
    if owned:
        raise ValueError(f"protify_args sets {sorted(owned)}, which the cell sets itself through {owned}.")
    for name, value in overrides.items():
        setattr(arguments, name, value)
    return arguments


def _check_hf_home(hf_home: str | None) -> None:
    """Refuse an `hf_home` the Hugging Face libraries can no longer honor.

    They read their cache locations once, on import, and importing the protify package loads
    them, so by the time a cell is read it is too late to move the caches. `main.py` sets them
    before its imports; here the environment must already name them.
    """
    if hf_home is None:
        return
    expected = os.path.normpath(os.path.join(hf_home, "hf_cache"))
    if os.path.normpath(os.environ.get("HF_HOME", "")) != expected:
        raise ValueError(
            f"hf_home is read before Hugging Face loads, which happens on importing protify. "
            f"Set HF_HOME={expected}, HF_HUB_CACHE={os.path.join(expected, 'hub')}, and "
            f"HF_DATASETS_CACHE={os.path.join(expected, 'datasets')} in the environment instead, "
            f"or leave hf_home unset."
        )


def cell_arguments(cell: UncertaintyCell, specs: Sequence[BranchSpec]) -> argparse.Namespace:
    """The Protify arguments one cell runs with."""
    arguments = protify_arguments(cell.protify_args)
    enabled = [name for name in UNSUPPORTED_ARGUMENTS if getattr(arguments, name, None)]
    if enabled:
        raise ValueError(f"the uncertainty workflow trains pooled probes only; turn off {enabled}.")
    if arguments.seed is None:
        raise ValueError("protify_args.seed must be set: a time-based seed carves different splits on every resume.")
    if arguments.num_runs < 1:
        raise ValueError("protify_args.num_runs must be at least 1.")
    _check_hf_home(arguments.hf_home)

    settings = cell.uncertainty
    arguments.probe_type = HEAD_RECIPES[settings.head].probe_type
    arguments.projection_size = settings.projection_size
    arguments.model_names = list(dict.fromkeys(spec.model_name for spec in specs))
    if settings.learning_rate is not None:
        arguments.probe_lr = settings.learning_rate
    if settings.weight_decay is not None:
        arguments.weight_decay = settings.weight_decay
    if settings.dropout is not None:
        arguments.dropout = settings.dropout
    return arguments


def load_cell_data(arguments: argparse.Namespace) -> CellData:
    """Load the cell's one dataset exactly as `main.py` would, data seed first."""
    set_global_seed(arguments.seed)
    main = MainProcess(arguments, GUI=False)
    main.apply_current_settings()
    main.get_datasets()
    if len(main.datasets) != 1:
        raise ValueError(f"a cell reads one dataset; these arguments load {sorted(main.datasets)}.")

    (data_name, (train, valid, test, num_labels, label_type, ppi)), = main.datasets.items()
    return CellData(main, data_name, train, valid, test, num_labels, label_type, ppi)


def load_split_cell_data(
    cell: UncertaintyCell,
    arguments: argparse.Namespace,
    explicit_rows: tuple[Sequence[int], Sequence[int]] | None,
    *,
    include_test: bool = True,
    include_calibration: bool = True,
    file_logging: bool = True,
) -> tuple[CellData, tuple[Sequence[int], Sequence[int]] | None]:
    """The cell's data, re-split by its split manifest when it names one, and the explicit V_fit and V_cal rows."""
    if getattr(cell.uncertainty, "dataset_manifest", None) is not None:
        if explicit_rows is not None or cell.uncertainty.split_manifest is not None:
            raise ValueError("a declared dataset fixes validation/calibration roles; overrides and re-splitting are forbidden")
        return _load_declared_cell_data(cell, arguments, include_test=include_test, include_calibration=include_calibration, file_logging=file_logging)
    if not include_test or not include_calibration:
        raise ValueError("excluding held-out splits requires a declared four-way dataset")
    cell_data = load_cell_data(arguments)
    path = cell.uncertainty.split_manifest
    if path is None:
        return cell_data, explicit_rows
    if explicit_rows is not None:
        raise ValueError("a cell with a split manifest takes V_fit and V_cal from it; explicit rows cannot be given too.")
    return apply_split_manifest(cell_data, read_json(path))


def _load_declared_cell_data(
    cell: UncertaintyCell, arguments: argparse.Namespace, *, include_test: bool = True, include_calibration: bool = True, file_logging: bool = True,
) -> tuple[CellData, tuple[np.ndarray, np.ndarray]]:
    """Use a frozen release directly; MainProcess supplies only model/trainer settings.

    Only the splits a caller may read are opened. Excluding calibration leaves V_fit as the whole
    validation split (V_cal is empty), and excluding test leaves `CellData.test` unset.
    """
    changing = (
        "data_names", "data_dirs", "data_revisions", "max_train_rows", "max_valid_rows", "max_test_rows",
        "trim", "aa_to_dna", "aa_to_rna", "dna_to_aa", "rna_to_aa", "codon_to_aa", "aa_to_codon",
    )
    enabled = [name for name in changing if getattr(arguments, name, None)]
    if enabled:
        raise ValueError(f"declared datasets forbid alternate sources, row caps, and sequence changes: {enabled}")
    if include_test and not include_calibration:
        raise ValueError("validation preflight must exclude both calibration and test")
    set_global_seed(arguments.seed)
    # MainProcess's GUI constructor omits file logging. A private handlerless logger lets
    # a caller that only reads (preflight, calibration) reuse its settings without appending
    # to the original fit's global logger.
    main = MainProcess(arguments, GUI=not file_logging)
    if not file_logging:
        main.logger = logging.Logger("protify.frozen_head_evaluation")
        main.logger.addHandler(logging.NullHandler())
    main.apply_current_settings()
    split_names = SPLIT_NAMES if include_test else ("train", "validation", "calibration") if include_calibration else ("train", "validation")
    release = load_declared_dataset(
        cell.uncertainty.dataset_manifest, cell.uncertainty.dataset_sha256,
        # Native inputs bind the original protein to an explicit retained-residue prefix.
        # Preserve full registry text; the frozen feature contract enforces its own budget.
        max_sequence_length=None if reads_native_features(cell.uncertainty) else main._data_max_length,
        split_names=split_names,
    )
    if release.label_type == "regression" and arguments.balanced_regression_metrics:
        raise ValueError("declared regression masks require balanced_regression_metrics=false")
    if cell.uncertainty.unlisted_labels == "negative":
        release = release.read_unlisted_as_negative()
    train, fit = (release.splits[name] for name in ("train", "validation"))
    test = release.splits.get("test")
    valid = concatenate_datasets([fit, release.splits["calibration"]]) if include_calibration else fit
    name = release.manifest["dataset_id"]
    main.datasets = {name: (train, valid, test, release.num_labels, release.label_type, False)}
    main.all_seqs = sorted({sequence for split in release.splits.values() for sequence in split["seqs"]}, key=lambda sequence: (-len(sequence), sequence))
    cell_data = CellData(main, name, train, valid, test, release.num_labels, release.label_type, False, release.training_record(), tuple(release.splits))
    return cell_data, (np.arange(len(fit)), np.arange(len(fit), len(valid)))  # fit rows (n_f,) and calibration rows (n_c,), as indices into valid


def _training_settings(cell: UncertaintyCell) -> dict[str, Any]:
    settings = asdict(cell.uncertainty)
    uncertainty = {name: settings[name] for name in TRAINING_SETTINGS}
    if reads_native_features(cell.uncertainty):
        # Native inputs forbid embedding. Saved cells must retain that requirement when
        # cross-fitting or packaging reconstructs their configuration.
        uncertainty["require_cached_embeddings"] = True
    return {
        "protify_args": dict(cell.protify_args),
        "uncertainty": uncertainty,
    }


def _record_cell(cell: UncertaintyCell) -> None:
    """Record the cell, or refuse to resume under settings that change what a seed trains."""
    path = os.path.join(cell.output_dir, CELL_RECORD)
    current = {"cell_id": cell.cell_id, "experiment_id": cell.experiment_id, **_training_settings(cell)}
    # Round-tripped through JSON so tuples and lists compare equal.
    current = json.loads(json.dumps(current))
    if os.path.exists(path):
        saved = read_json(path)
        saved["uncertainty"] = {**TRAINING_SETTING_DEFAULTS, **saved.get("uncertainty", {})}
        if {key: saved.get(key) for key in current} != current:
            raise ValueError(
                f"{cell.output_dir} was run with different training settings; use a new output_dir."
            )
        return
    write_json(path, current)


def _record_roles(output_dir: str, manifest: Mapping[str, Any]) -> None:
    path = os.path.join(output_dir, SPLIT_RECORD)
    if os.path.exists(path):
        check_manifest(read_json(path), manifest)
        return
    write_json(path, manifest)


def _record_observations(output_dir: str, cell_data: CellData) -> bool:
    """Bind masked runs to ordered rows, targets, masks, and the selection rule.

    This is a resume identity check, not authentication of source assays. Historical
    unmasked runs retain their existing resume behavior. Return whether any truth is partial.
    Only the splits the cell loaded are recorded: a fit that never opened the test split
    names no test rows.
    """
    splits = {"train": cell_data.train, "valid": cell_data.valid}
    if cell_data.test is not None:
        splits["test"] = cell_data.test
    masked = ["label_mask" in getattr(split, "column_names", split) for split in splits.values()]
    path = os.path.join(output_dir, OBSERVATIONS_RECORD)
    if not any(masked) and not os.path.exists(path):
        return False
    if not all(masked):
        raise ValueError(f"explicit label masks must be present in every loaded split ({', '.join(splits)})")
    current: dict[str, Any] = {"schema_version": 1, "selection_metric": "eval_observed_loss", "splits": {}}
    for name, split in splits.items():
        observed_count, target_count, target_shape = 0, 0, None
        for label, observation in zip(split["labels"], split["label_mask"], strict=True):
            shape, observed, total = observed_row_counts(label, observation)
            if target_shape is not None and target_shape != shape:
                raise ValueError(f"{name} labels must have one target shape")
            target_shape = shape
            observed_count += observed
            target_count += total
        if not observed_count:
            raise ValueError(f"{name} requires at least one observed label")
        digests = row_digests(split)
        current["splits"][name] = {
            "rows": len(digests), "observed_labels": observed_count,
            "unobserved_labels": int(target_count - observed_count),
            "ordered_row_digests_sha256": hashlib.sha256("\n".join(digests).encode()).hexdigest(),
        }
    partial = any(record["unobserved_labels"] for record in current["splits"].values())
    if os.path.exists(path):
        if read_json(path) != current:
            raise ValueError("label observations changed since this run; use a new output_dir")
        return partial
    if any(Path(output_dir).glob("seed_*.npz")) or any(Path(output_dir).glob("seed_*.json")):
        raise ValueError("masked seeds have no label observation record; use a new output_dir")
    write_json(path, current)
    return partial


def _primary(split: SplitPredictions, kind: str, *, canonical: bool = False, selection: str | None = None) -> float | None:
    """The V_fit score a seed, a learning rate and a calibration seed are chosen on: `selection`, else the kind's primary."""
    predictions = {"mean": split.outputs} if kind == "regression" else {"probs": probabilities(split.outputs, kind)}
    if canonical:
        # Only the primary: the report scores every other metric, and ranking each label of a wide vocabulary costs minutes.
        metrics = canonical_task_metrics(kind, split.labels, **predictions, label_mask=split.label_mask, primary_only=True, selection=selection)
    else:
        metrics = task_metrics(kind, split.labels, **predictions, label_mask=split.label_mask)
    return metrics[selection or PRIMARY_METRICS[kind]]


def _head_nll(split: SplitPredictions, kind: str) -> float:
    """The head's mean negative log-likelihood of a split's labels, as it predicts them.

    Classification reads the softmax or per-label sigmoid of the outputs, mean-field logits
    for SNGP. Regression is Gaussian: SNGP with its predictive variance, the plain head with
    one variance for every row, its mean squared error on these rows, which is the
    maximum-likelihood constant.
    """
    if kind != "regression":
        return negative_log_likelihood(kind, split.labels, probabilities(split.outputs, kind), split.label_mask)

    # labels: (n,); outputs: (n,)
    variance = split.variance  # (n,) or None
    if variance is None:
        observed = slice(None) if split.label_mask is None else split.label_mask.reshape(-1)
        squared_error = np.mean((split.labels.reshape(-1)[observed] - split.outputs[observed]) ** 2, dtype=np.float64)  # ()
        variance = np.full(split.outputs.shape, squared_error)  # (n,)
    return gaussian_nll(split.labels, split.outputs, variance, split.label_mask)


def _seed_paths(output_dir: str, seed: int) -> tuple[str, str]:
    stem = os.path.join(output_dir, f"seed_{seed}")
    return f"{stem}.npz", f"{stem}.json"


def seed_is_complete(output_dir: str, seed: int) -> bool:
    """Whether a seed's predictions and its record, the file written last, are both on disk."""
    return all(os.path.exists(path) for path in _seed_paths(output_dir, seed))


def _record_sampling(output_dir: str, plan: ClusterSamplingPlan, seeds: Sequence[int]) -> None:
    path = os.path.join(output_dir, "sampling.json")
    expected = plan.record()
    if os.path.exists(path):
        if read_json(path) != expected:
            raise ValueError("cluster sampling plan changed; use a new output_dir")
    elif any(seed_is_complete(output_dir, seed) for seed in seeds):
        raise ValueError("existing seeds lack their cluster sampling plan")
    else:
        write_json(path, expected)
    for seed in seeds:
        if seed_is_complete(output_dir, seed):
            record = read_json(_seed_paths(output_dir, seed)[1])
            validate_sampling_receipt(plan, record.get("sampling"), seed)


def _record_metric_contract(output_dir: str, release: Mapping[str, Any], seeds: Sequence[int]) -> None:
    """Never reuse validation selection made under another metric definition."""
    path = os.path.join(output_dir, "metric_contract.json")
    expected = metric_contract(release)
    if os.path.exists(path):
        if read_json(path) != expected:
            raise ValueError("declared metric contract changed; use a new output_dir")
    elif any(seed_is_complete(output_dir, seed) for seed in seeds):
        raise ValueError("finished seeds lack their declared metric contract; use a new output_dir")
    else:
        write_json(path, expected)
    for seed in seeds:
        if seed_is_complete(output_dir, seed) and read_json(_seed_paths(output_dir, seed)[1]).get("metric_contract") != expected:
            raise ValueError("finished seed metric contract differs from the declared endpoint")


def _record_training_recipe(output_dir: str, recipe: Mapping[str, Any], seeds: Sequence[int]) -> None:
    """Bind execution defaults before a finished-seed shortcut, without adopting old fits."""
    path = os.path.join(output_dir, "training_recipe.json")
    if os.path.exists(path):
        if read_json(path) != recipe:
            raise ValueError("resolved training recipe changed; use a new output_dir")
    elif any(seed_is_complete(output_dir, seed) for seed in seeds):
        raise ValueError("finished seeds lack their training recipe; use a new output_dir")
    else:
        write_json(path, dict(recipe))
    for seed in seeds:
        if seed_is_complete(output_dir, seed):
            validate_training_receipt(recipe, read_json(_seed_paths(output_dir, seed)[1]).get("training"), seed)


def load_seeds(output_dir: str, seeds: Sequence[int]) -> dict[int, dict[str, SplitPredictions]]:
    """Every seed's saved predictions, by seed."""
    mode = "r" if Path(output_dir, "dataset_release.json").exists() else None
    return {seed: load_seed_predictions(_seed_paths(output_dir, seed)[0], mmap_mode=mode) for seed in seeds}


def agreed_labels(
    seed_predictions: Mapping[int, Mapping[str, SplitPredictions]], seeds: Sequence[int]
) -> dict[str, np.ndarray]:
    """Each split's labels, refused unless every seed saved the same ones."""
    first = seed_predictions[seeds[0]]
    labels = {split: first[split].labels for split in SPLITS}
    for seed in seeds[1:]:
        for split in SPLITS:
            mask, other_mask = first[split].label_mask, seed_predictions[seed][split].label_mask
            if (mask is None) != (other_mask is None) or (mask is not None and not np.array_equal(mask, other_mask)):
                raise ValueError(f"seed {seed} saved different {split} observation masks from seed {seeds[0]}.")
            observed = slice(None) if mask is None else mask
            if not np.array_equal(seed_predictions[seed][split].labels[observed], labels[split][observed]):
                raise ValueError(f"seed {seed} saved different {split} labels from seed {seeds[0]}.")
    return labels  # labels by split name, each (n,) or (n, c)


def head_directory(output_dir: str, seed: int) -> str:
    """Where `save_heads` keeps one seed's trained head, as `save_pretrained` writes it."""
    return os.path.join(output_dir, HEADS_DIRECTORY, f"seed_{seed}")


def head_is_saved(output_dir: str, seed: int) -> bool:
    """Whether a seed's head directory holds a config and the weights it names."""
    directory = head_directory(output_dir, seed)
    weights = [os.path.join(directory, name) for name in (SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME)]
    return os.path.isfile(os.path.join(directory, CONFIG_NAME)) and any(os.path.isfile(path) for path in weights)


def _refuse_finished_seeds_without_heads(output_dir: str, seeds: Sequence[int]) -> None:
    """Refuse a `save_heads` directory holding a finished seed whose head was never saved.

    A finished seed never trains again, so its head could not be recovered. This happens when
    a directory is seeded with the files of a cell that ran without `save_heads`.
    """
    headless = [seed for seed in seeds if seed_is_complete(output_dir, seed) and not head_is_saved(output_dir, seed)]
    if headless:
        raise ValueError(
            f"{output_dir} holds finished seeds {headless} without a saved head, and save_heads is on. "
            f"Copy each one's {HEADS_DIRECTORY}/seed_<seed> directory in beside its seed files, or move "
            f"its seed_<seed>.npz and seed_<seed>.json aside so that the seed trains again."
        )


def _train_label_std(cell_data: CellData) -> float:
    """The training labels' standard deviation for regression, which sets the conformal floor; else 0."""
    if cell_data.kind != "regression":
        return 0.0
    labels = np.asarray(list(cell_data.train["labels"]), dtype=np.float64)  # (n,) or (n, 1)
    columns = getattr(cell_data.train, "column_names", cell_data.train)
    if "label_mask" in columns:
        mask = np.asarray(list(cell_data.train["label_mask"]))  # labels shape
        validate_observed_labels(labels, mask)
        labels = labels[mask]  # (m,)
        if not labels.size:
            raise ValueError("regression calibration requires observed training targets")
    return float(np.std(labels))


def save_standardizer_record(path: str, standardizer: EmbeddingStandardizer | None, train_label_std: float) -> None:
    """Write the cell's embedding standardizer, or that none was fitted, atomically.

    The training labels' standard deviation is kept beside it: the regression conformal floor
    reads it, and no other file records it. A `projected_qclip` standardizer also keeps its
    clip bounds.
    """
    arrays = {
        "fitted": np.asarray(standardizer is not None),  # ()
        "train_label_std": np.asarray(train_label_std, dtype=np.float64),  # ()
    }
    if standardizer is not None:
        arrays["mean"] = standardizer.mean.numpy()  # (d_in,)
        arrays["scale"] = standardizer.scale.numpy()  # (d_in,)
    if isinstance(standardizer, ClippedStandardizer):
        arrays["clip_lower"] = standardizer.lower.numpy()  # (d_in,)
        arrays["clip_upper"] = standardizer.upper.numpy()  # (d_in,)
    temporary = f"{path}.partial"
    with open(temporary, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def load_standardizer_record(path: str) -> tuple[EmbeddingStandardizer | None, float]:
    """The standardizer a `save_heads` cell trained its heads with, or None, and its label spread."""
    with np.load(path, allow_pickle=False) as archive:
        train_label_std = float(archive["train_label_std"])
        if not bool(archive["fitted"]):
            return None, train_label_std
        if "clip_lower" in archive.files:
            clipped = ClippedStandardizer(archive["mean"], archive["scale"], archive["clip_lower"], archive["clip_upper"])
            return clipped, train_label_std
        return EmbeddingStandardizer(archive["mean"], archive["scale"]), train_label_std


def _record_standardizer(output_dir: str, standardizer: EmbeddingStandardizer | None, train_label_std: float) -> None:
    """Record the standardizer the heads train with, or refuse a resume that fits another.

    The saved heads read features through the recorded standardizer, so a resume whose refit
    differs, as it would after a cache or the training split changed, would pair earlier heads
    with the wrong one. The comparison allows float32 rounding, since a refit on another machine
    can sum in another order.
    """
    path = os.path.join(output_dir, STANDARDIZER_RECORD)
    if not os.path.exists(path):
        save_standardizer_record(path, standardizer, train_label_std)
        return

    saved, saved_label_std = load_standardizer_record(path)
    same = (saved is None) == (standardizer is None) and np.isclose(saved_label_std, train_label_std, rtol=1e-5)
    same = same and isinstance(saved, ClippedStandardizer) == isinstance(standardizer, ClippedStandardizer)
    if same and saved is not None:
        pairs = [(saved.mean, standardizer.mean), (saved.scale, standardizer.scale)]
        if isinstance(saved, ClippedStandardizer):
            pairs += [(saved.lower, standardizer.lower), (saved.upper, standardizer.upper)]
        same = all(
            recorded.shape == refit.shape and np.allclose(recorded.numpy(), refit.numpy(), rtol=1e-5, atol=1e-6)
            for recorded, refit in pairs
        )
    if not same:
        raise ValueError(
            f"{path} records another standardizer than this run fits, so a cache or the training split "
            f"changed since its heads trained; use a new output_dir."
        )


DatasetMaker = Callable[..., JoinedBranchDataset]


def _train_head(
    main: MainProcess,
    seed: int,
    train_dataset: JoinedBranchDataset,
    valid_dataset: JoinedBranchDataset,
    test_dataset: JoinedBranchDataset | None,
    collator: JoinedBranchCollator,
    log_id: str,
    model_name: str,
    data_name: str,
    ppi: bool,
    *,
    evaluate_test: bool = True,
    checkpoint_selection: str | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """One head built from `main.probe_args` and trained through Protify's `_train` under `seed`.

    Returns the trained model with the weights early stopping restored, and its validation
    and test metrics. With evaluate_test false the third mapping contains training time only,
    and `test_dataset` may be None: a fit that never loaded the test split passes none.
    """
    main.trainer_args.seed = seed
    set_global_seed(seed)
    probe = get_probe(main.probe_args)
    # Hugging Face writes checkpoints under the working directory, so each head trains in a
    # temporary one and leaves nothing behind.
    with (
        tempfile.TemporaryDirectory(prefix="protify_uncertainty_", ignore_cleanup_errors=True) as workdir,
        contextlib.chdir(workdir),
    ):
        model, valid_metrics, test_metrics, _, _ = main._train(
            model=probe,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            data_collator=collator,
            tokenizer=None,
            log_id=log_id,
            model_name=model_name,
            data_name=data_name,
            ppi=ppi,
            probe=True,
            skip_plot=True,
            **({"evaluate_test": False} if not evaluate_test else {}),
            **({"checkpoint_selection": checkpoint_selection} if checkpoint_selection is not None else {}),
        )
    return model, valid_metrics, test_metrics


def _score_fold(
    model: Any, dataset: JoinedBranchDataset, collator: JoinedBranchCollator, kind: str, batch_size: int, device: torch.device
) -> np.ndarray:
    """A fold head's per-row training loss on the rows of `dataset`, the fold it never trained on."""
    predictions, _ = collect_predictions(model, dataset, collator, task_kind=kind, batch_size=batch_size, device=device)
    return per_row_losses(kind, predictions.labels, predictions.outputs)  # (n,)


def _with_labels(hf_dataset: Any, labels: np.ndarray) -> dict[str, list[Any]]:
    """The dataset's sequence columns with `labels` in place of its own."""
    # labels: (n,) or (n, c)
    columns = {name: list(hf_dataset[name]) for name in sequence_columns(hf_dataset)}
    return {**columns, "labels": [float(label) for label in labels]}


@contextlib.contextmanager
def _loss_predictor_arguments(main: MainProcess) -> Any:
    """`main`'s probe and trainer arguments switched to a one-output `projected` regression head."""
    probe_saved = (main.probe_args.probe_type, main.probe_args.task_type, main.probe_args.num_labels)
    task_saved = main.trainer_args.task_type
    main.probe_args.probe_type = "projected"
    main.probe_args.task_type = "regression"
    main.probe_args.num_labels = 1
    main.trainer_args.task_type = "regression"
    try:
        yield
    finally:
        main.probe_args.probe_type, main.probe_args.task_type, main.probe_args.num_labels = probe_saved
        main.trainer_args.task_type = task_saved


def _loss_predictions(
    cell: UncertaintyCell,
    cell_data: CellData,
    seed: int,
    datasets: Mapping[str, JoinedBranchDataset],
    collator: JoinedBranchCollator,
    make_dataset: DatasetMaker,
    model_name: str,
    v_fit_rows: Any,
    v_fit: SplitPredictions,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Cross-fit the task head's training losses, train the loss predictor, and predict every split.

    Fold heads early-stop on V_fit, as the task head does. The predictor trains on every
    training row's cross-fitted `log(loss + 1e-6)` and early-stops on V_fit's, scored against
    the task head itself, so it learns the loss of a head that never saw the row.
    """
    main = cell_data.main
    batch_size = main.trainer_args.probe_batch_size
    folds = stratified_folds(list(cell_data.train["labels"]), cell_data.label_type, cell.uncertainty.loss_head_folds, seed)  # (n,)

    def fit(rows: np.ndarray) -> Any:
        # Its own data name: Protify's balanced regression weights, keyed by the dataset's, are
        # sized for the test split, which a fold head never reads.
        # rows: (m,)
        train_dataset = make_dataset(take_rows(cell_data.train, rows), train=True)
        model, _, _ = _train_head(
            main, seed, train_dataset, datasets["v_fit"], datasets["v_fit"], collator,
            f"{cell.cell_id}_seed{seed}_fold", model_name, f"{cell_data.data_name}__fold", cell_data.ppi,
        )
        return model

    def score(model: Any, rows: np.ndarray) -> np.ndarray:
        # rows: (m,)
        return _score_fold(model, make_dataset(take_rows(cell_data.train, rows)), collator, cell_data.kind, batch_size, device)  # (m,) per-row losses

    train_targets = log_loss_targets(cross_fitted_losses(folds, fit, score))  # (n,)
    v_fit_targets = log_loss_targets(per_row_losses(cell_data.kind, v_fit.labels, v_fit.outputs))  # (n_fit,)
    v_fit_dataset = make_dataset(_with_labels(v_fit_rows, v_fit_targets), label_type="regression")
    with _loss_predictor_arguments(main):
        predictor, _, _ = _train_head(
            main,
            seed,
            make_dataset(_with_labels(cell_data.train, train_targets), label_type="regression", train=True),
            v_fit_dataset,
            v_fit_dataset,
            collator,
            f"{cell.cell_id}_seed{seed}_loss_predictor",
            model_name,
            f"{cell_data.data_name}__loss_predictor",
            cell_data.ppi,
        )
    predictions = {}
    for split in SPLITS:
        split_predictions, _ = collect_predictions(
            predictor, datasets[split], collator, task_kind="regression", batch_size=batch_size, device=device
        )
        predictions[split] = split_predictions.outputs.astype(np.float32)  # (n,)
    return predictions  # by split name, each (n,)


def _train_seed(
    cell: UncertaintyCell,
    cell_data: CellData,
    seed: int,
    datasets: Mapping[str, JoinedBranchDataset],
    collator: JoinedBranchCollator,
    model_name: str,
    test_digests: Sequence[str],
    make_dataset: DatasetMaker,
    v_fit_rows: Any,
) -> None:
    """Train one seed through Protify's `_train`, then save its predictions and record."""
    main = cell_data.main
    validation_only = cell.uncertainty.evaluation_mode == "validation_only"
    scored_splits = ("v_fit",) if validation_only else SPLITS
    # A declared release's held-out scores come from the predictions saved below, so the Trainer never predicts
    # its validation and test splits a second time: each prediction is a full (rows, labels) array to hold and score.
    model, v_fit_metrics, test_metrics = _train_head(
        main, seed, datasets["train"], datasets["v_fit"], None if validation_only else datasets["test"], collator,
        f"{cell.cell_id}_seed{seed}", model_name, cell_data.data_name, cell_data.ppi,
        **({"evaluate_test": False} if validation_only or cell_data.release is not None else {}),
        **({"checkpoint_selection": cell.uncertainty.checkpoint_selection} if cell.uncertainty.checkpoint_selection is not None else {}),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = main.trainer_args.probe_batch_size
    mc_samples = cell.uncertainty.mc_dropout_samples if cell.uncertainty.head == "projected" else 0
    torch.manual_seed(seed)
    predictions: dict[str, SplitPredictions] = {}
    inference_seconds = {}
    for split in scored_splits:
        predictions[split], inference_seconds[split] = collect_predictions(
            model,
            datasets[split],
            collator,
            task_kind=cell_data.kind,
            batch_size=batch_size,
            device=device,
            mc_samples=mc_samples,
            disk_backed=cell_data.release is not None,
        )

    reference = trunk_features(model, datasets["train_reference"], collator, batch_size=batch_size, device=device)  # (r, p)
    for split in scored_splits:
        queries = trunk_features(model, datasets[split], collator, batch_size=batch_size, device=device)  # (n, p)
        predictions[split].nn_distance = nearest_neighbor_distance(queries, reference)
        if cell.uncertainty.save_trunk_features:
            predictions[split].trunk = queries.cpu().numpy().astype(np.float32)  # (n, p)
    if not validation_only:
        predictions["test"].digests = np.asarray(test_digests)
    del reference, queries

    # The extras run after every prediction of the task head is collected, so they cannot
    # change one. Their seconds go to the record, for the cost comparison.
    extra_seconds: dict[str, float] = {}
    if cell.uncertainty.laplace:
        started = time.perf_counter()
        factors = fit_kfac(model, datasets["train_reference"], collator, kind=cell_data.kind, batch_size=batch_size, device=device)
        for split in SPLITS:
            predictions[split].laplace_variance = latent_variance(
                model, factors, datasets[split], collator, LAPLACE_PRIOR_PRECISIONS,
                kind=cell_data.kind, batch_size=batch_size, device=device,
            )  # (n, L, c) or (n, L)
        del factors
        extra_seconds["laplace_seconds"] = time.perf_counter() - started
    if cell.uncertainty.loss_head_folds:
        started = time.perf_counter()
        loss_predictions = _loss_predictions(
            cell, cell_data, seed, datasets, collator, make_dataset, model_name, v_fit_rows, predictions["v_fit"], device
        )
        for split in SPLITS:
            predictions[split].loss_prediction = loss_predictions[split]  # (n,)
        extra_seconds["loss_head_seconds"] = time.perf_counter() - started
    model.cpu()
    if cell.uncertainty.save_heads:
        # The weights early stopping restored, with the SNGP posterior `_train` fitted to them.
        # Saved before the seed's files, because the record marks the seed finished: a crash in
        # between retrains the seed rather than leaving it finished without a head.
        model.save_pretrained(head_directory(cell.output_dir, seed))

    npz_path, record_path = _seed_paths(cell.output_dir, seed)
    save_seed_predictions(npz_path, predictions)
    timed_split = "v_fit" if validation_only else "test"
    timed_rows = predictions[timed_split].labels.shape[0]
    record = {
        "seed": seed,
        "v_fit_primary": _primary(predictions["v_fit"], cell_data.kind, canonical=cell_data.release is not None, selection=cell.uncertainty.selection_metric),
        "v_fit_nll": _head_nll(predictions["v_fit"], cell_data.kind),
        "train_seconds": test_metrics.get("training_time_seconds"),
        "inference_seconds_per_1000_rows": 1000.0 * inference_seconds[timed_split] / timed_rows,
        "inference_split": timed_split,
        "evaluation_mode": cell.uncertainty.evaluation_mode,
        "scored_splits": list(scored_splits),
        "protify_v_fit_metrics": v_fit_metrics,
        **extra_seconds,
    }
    if cell_data.release is not None:
        record["metric_contract"] = metric_contract(cell_data.release)
    if datasets["train"].sampling_plan is not None:
        record["sampling"] = datasets["train"].sampling_receipt
        validate_sampling_receipt(datasets["train"].sampling_plan, record["sampling"], seed)
    if cell_data.release is not None or main.trainer_args.record_training_recipe:
        record["training"] = {
            "recipe": datasets["train"].training_recipe, "seed": seed,
            "outcome": datasets["train"].training_outcome,
        }
        expected = read_json(os.path.join(cell.output_dir, "training_recipe.json"))
        validate_training_receipt(expected, record["training"], seed)
    # Written last: its presence marks the seed complete.
    write_json(record_path, json_ready(record))


def _identity_bins(cell: UncertaintyCell, cell_data: CellData) -> tuple[dict[str, np.ndarray] | None, dict[str, int] | None]:
    path = cell.uncertainty.identity_path
    if path is None or not os.path.exists(path):
        if path is not None:
            print(f"No identity file at {path}; run `python -m protify.uncertainty identity` for identity bins.")
        return None, None  # None, None: no (n,) identity bins

    sequence_identity = read_json(path)["sequences"]
    columns = [list(cell_data.test[name]) for name in sequence_columns(cell_data.test)]
    missing = [sequence for column in columns for sequence in column if sequence_digest(sequence) not in sequence_identity]
    if missing:
        raise ValueError(f"{path} lacks {len(missing)} test sequences; rerun the identity command for this dataset.")
    identity = row_identity(sequence_identity, columns)  # (n_test,)
    return bin_rows(identity), bin_counts(identity)  # each bin name -> (m,) row indices, then each name -> a count


def _report(
    cell: UncertaintyCell,
    cell_data: CellData,
    seeds: Sequence[int],
    specs: Sequence[BranchSpec],
    widths: Sequence[int],
    roles: ValidationRoles,
) -> dict[str, Any]:
    kind = cell_data.kind
    validation_only = cell.uncertainty.evaluation_mode == "validation_only"
    scored_splits = ("v_fit",) if validation_only else SPLITS
    seed_predictions = load_seeds(cell.output_dir, seeds)
    records = [read_json(_seed_paths(cell.output_dir, seed)[1]) for seed in seeds]
    if cell_data.release is not None:
        for seed, record in zip(seeds, records, strict=True):
            if (set(seed_predictions[seed]) != set(scored_splits)
                    or record.get("evaluation_mode", "full") != cell.uncertainty.evaluation_mode
                    or record.get("scored_splits", list(SPLITS)) != list(scored_splits)):
                raise ValueError("declared seed evaluation scope differs from its cell")
            expected_primary = _primary(seed_predictions[seed]["v_fit"], kind, canonical=True, selection=cell.uncertainty.selection_metric)
            if record.get("v_fit_primary") != expected_primary:
                raise ValueError("declared seed validation primary differs from its saved predictions")
    for record in records:
        # A seed recorded before `v_fit_nll` existed has none.
        record.setdefault("v_fit_nll", None)
    methods: dict[str, Any] = {}
    method_bins: dict[str, Any] = {}
    test_bins, counts = None, None
    if not validation_only:
        labels = agreed_labels(seed_predictions, seeds)
        label_masks = {split: seed_predictions[seeds[0]][split].label_mask for split in SPLITS}
        train_label_std = _train_label_std(cell_data)
        levels = cell.uncertainty.conformal_levels
        test_bins, counts = _identity_bins(cell, cell_data)
        for method in build_methods(seed_predictions, cell.uncertainty.head, kind, pnorm=cell.uncertainty.pnorm_uncertainty):
            summary = score_method(method, kind, labels, levels, train_label_std, test_bins, label_masks, canonical=cell_data.release is not None)
            if "identity_bins" in summary:
                method_bins[method.name] = summary.pop("identity_bins")
            methods[method.name] = summary

    input_size = cell_data.main.probe_args.input_size
    report: dict[str, Any] = {
        "cell_id": cell.cell_id,
        "experiment_id": cell.experiment_id,
        "tags": dict(cell.tags),
        "data_name": cell_data.data_name,
        "task_type": kind,
        "label_type": cell_data.label_type,
        "num_labels": cell_data.num_labels,
        "ppi": cell_data.ppi,
        "head": cell.uncertainty.head,
        "branch_specs": list(cell.uncertainty.branch_specs),
        "display_name": display_name(specs),
        "branches": [
            {**asdict(spec), "width": width, "sparse": spec.sparse} for spec, width in zip(specs, widths, strict=True)
        ],
        "input_size": input_size,
        "primary_metric": PRIMARY_METRICS[kind],
        **({"selection_metric": cell.uncertainty.selection_metric} if cell.uncertainty.selection_metric is not None else {}),
        "data_seed": seeds[0],
        "seeds": list(seeds),
        "evaluation_mode": cell.uncertainty.evaluation_mode,
        "scored_splits": list(scored_splits),
        "split": {
            "source": roles.source,
            "calibration_fraction": roles.fraction,
            "split_seed": roles.seed,
            "train": len(cell_data.train),
            "v_fit": int(roles.fit.size),
            # A validation-only fit never loads these two splits, so it has no count for them.
            "v_cal": None if validation_only else int(roles.cal.size),
            "test": None if validation_only else len(cell_data.test),
        },
        "v_fit_primary": _mean_defined([record["v_fit_primary"] for record in records]),
        "v_fit_nll": _mean_defined([record["v_fit_nll"] for record in records]),
        "per_seed": records,
        "methods": methods,
        "cost": {
            "head_parameters": sum(parameter.numel() for parameter in get_probe(cell_data.main.probe_args).parameters()),
            "inference_seconds_per_1000_rows": _mean_defined(
                [record["inference_seconds_per_1000_rows"] for record in records]
            ),
        },
    }
    if cell_data.release is not None:
        report["dataset_release"] = cell_data.release
        report["metric_contract"] = metric_contract(cell_data.release)
        report["sampling"] = read_json(os.path.join(cell.output_dir, "sampling.json"))
        if validation_only:
            report["split_access"] = {
                "loaded": list(cell_data.loaded_splits),
                "excluded": [name for name in SPLIT_NAMES if name not in cell_data.loaded_splits],
            }
    if cell_data.release is not None or cell_data.main.trainer_args.record_training_recipe:
        report["training_recipe"] = read_json(os.path.join(cell.output_dir, "training_recipe.json"))
    if test_bins is not None:
        report["identity_bins"] = {
            "path": cell.uncertainty.identity_path,
            "counts": counts,
            "methods": method_bins,
        }
    return json_ready(report)


def _mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def reads_native_features(settings: UncertaintySettings) -> bool:
    """Whether the cell reads a pinned native input or a probe input, neither of which embeds or truncates."""
    return settings.feature_manifest is not None or settings.probe_features is not None


def load_cell_branches(
    cell: UncertaintyCell, cell_data: CellData, specs: Sequence[BranchSpec], *, cached_only: bool = False,
) -> tuple[JoinedBranchEmbeddings | IndexedBranchEmbeddings, tuple[int, ...]]:
    """Select the cell's pinned native reader or its existing legacy cache path."""
    settings, main = cell.uncertainty, cell_data.main
    if settings.probe_features is not None:
        if cell_data.release is None or cell_data.ppi:
            raise ValueError("probe inputs require validated single-protein declared rows")
        joined = open_probe_features(settings.probe_features, settings.branch_specs, main.all_seqs, batch_size=main.trainer_args.probe_batch_size)
        return joined, joined.widths
    if settings.feature_manifest is not None:
        from ..data.feature_inputs import FrozenFeatureInput

        if cell_data.release is None or cell_data.ppi:
            raise ValueError("native Oracle inputs require validated single-protein declared rows")
        inputs = FrozenFeatureInput(settings.feature_manifest, settings.feature_sha256,
                                    dataset_sha256=settings.dataset_sha256, branch_specs=settings.branch_specs)
        joined = inputs.joined(main.all_seqs, batch_size=main.trainer_args.probe_batch_size,
                               workers=main.trainer_args.num_workers)
        return joined, joined.widths
    run_arguments = dict(vars(main.full_args))
    if cached_only or settings.require_cached_embeddings:
        require_cached_branches(specs, main.all_seqs, run_arguments)
    else:
        embed_missing_branches(specs, main.all_seqs, run_arguments)
    return join_branch_embeddings(specs, main.all_seqs, run_arguments)


def run_uncertainty_cell(
    cell: UncertaintyCell, *, explicit_rows: tuple[Sequence[int], Sequence[int]] | None = None,
    branch_loader: Any = None,
) -> dict[str, Any]:
    """Train the cell's seeds, resuming past finished ones, and write `report.json`.

    `explicit_rows` names V_fit and V_cal as validation row indices instead of the stratified
    carve, as a split manifest with a calibration split would.
    """
    settings = cell.uncertainty
    if settings.evaluation_mode not in ("full", "validation_only"):
        raise ValueError("evaluation_mode must be full or validation_only")
    if settings.evaluation_mode == "validation_only" and (
        not settings.dataset_manifest or not settings.save_heads or settings.laplace or settings.loss_head_folds or settings.pnorm_uncertainty
    ):
        raise ValueError("validation_only requires a declared dataset, saved heads, and no calibration extras")
    specs = [parse_branch_spec(token) for token in settings.branch_specs]
    arguments = cell_arguments(cell, specs)
    os.makedirs(cell.output_dir, exist_ok=True)
    _record_cell(cell)
    seeds = [arguments.seed + index for index in range(arguments.num_runs)]
    if settings.save_heads:
        # Before loading anything, since no amount of training can recover a finished seed's head.
        _refuse_finished_seeds_without_heads(cell.output_dir, seeds)

    # A fit opens no payload it may not read: its calibration and test labels stay on disk.
    validation_only = settings.evaluation_mode == "validation_only"
    cell_data, explicit_rows = load_split_cell_data(
        cell, arguments, explicit_rows, include_test=not validation_only, include_calibration=not validation_only,
    )
    if settings.selection_metric is not None and settings.selection_metric not in SELECTION_METRICS[cell_data.kind]:
        raise ValueError(f"a {cell_data.kind} task selects on one of {SELECTION_METRICS[cell_data.kind]}, not {settings.selection_metric!r}")
    if cell_data.release is not None:
        release_path = os.path.join(cell.output_dir, "dataset_release.json")
        if os.path.exists(release_path):
            if read_json(release_path) != cell_data.release:
                raise ValueError("the declared dataset release changed; use a new output_dir")
        elif any(Path(cell.output_dir).glob("seed_*.npz")):
            raise ValueError("existing seeds lack their declared dataset release record")
        else:
            write_json(release_path, cell_data.release)
    partial = _record_observations(cell.output_dir, cell_data)
    if cell_data.release is not None:
        _record_metric_contract(cell.output_dir, cell_data.release, seeds)
    if partial and explicit_rows is None:
        raise ValueError("partial labels require explicit validation/calibration rows; filler targets cannot define strata")
    if partial and (settings.laplace or settings.loss_head_folds or settings.pnorm_uncertainty):
        raise ValueError("partial labels require mask-aware fitting for optional Laplace, loss-head, or pnorm methods")
    main = cell_data.main
    recipe = HEAD_RECIPES[settings.head]
    if recipe.regression_only and cell_data.kind != "regression":
        raise ValueError(f"head {settings.head!r} predicts a Gaussian variance, so it takes regression only; {cell_data.data_name} is {cell_data.kind}.")
    roles = resolve_validation_roles(
        cell_data.valid["labels"] if explicit_rows is not None else list(cell_data.valid["labels"]),
        cell_data.label_type,
        fraction=settings.calibration_fraction,
        seed=settings.split_seed,
        explicit_rows=explicit_rows,
        require_calibration=not validation_only,
    )
    _record_roles(cell.output_dir, roles_manifest(roles, row_digests(cell_data.valid), cell_data.label_type))
    v_fit = take_rows(cell_data.valid, roles.fit)
    v_cal = None if validation_only else take_rows(cell_data.valid, roles.cal)

    joined, widths = (load_cell_branches if branch_loader is None else branch_loader)(cell, cell_data, specs)
    flip_pairs = main._augment_pair_order()
    standardizer = None
    if main.embedding_args.embedding_scaler:
        standardizer = fit_joined_standardizer(joined, cell_data.train, cell_data.ppi, main._pair_feature_mode, flip_pairs)
    if recipe.quantile_clip:
        if standardizer is None:
            raise ValueError(f"head {settings.head!r} clips standardized features, so the embedding scaler must stay on.")
        # The bounds read the training rows alone, standardized as the probe reads them.
        unclipped = JoinedBranchCollator(joined, cell_data.ppi, standardizer, main._pair_feature_mode)
        train_rows = standardized_rows(JoinedBranchDataset(cell_data.train, joined, cell_data.label_type, cell_data.ppi), unclipped)  # (n, d_in)
        standardizer = fit_clipped_standardizer(standardizer, train_rows, CLIP_QUANTILES)
        del train_rows
    if settings.save_heads:
        _record_standardizer(cell.output_dir, standardizer, _train_label_std(cell_data))
    collator = JoinedBranchCollator(joined, cell_data.ppi, standardizer, main._pair_feature_mode)

    def make_dataset(hf_dataset: Any, label_type: str = cell_data.label_type, train: bool = False) -> JoinedBranchDataset:
        """Probe rows over the joined branches; training rows swap pairs as the task head's do."""
        return JoinedBranchDataset(hf_dataset, joined, label_type, cell_data.ppi, flip_pairs if train else False, train=train,
                                   frozen_release=cell_data.release is not None, cluster_draws_per_epoch=settings.cluster_draws_per_epoch,
                                   cluster_class_balance=settings.cluster_class_balance if train else False)

    datasets = {
        "train": make_dataset(cell_data.train, train=True),
        "train_reference": make_dataset(cell_data.train),
        "v_fit": make_dataset(v_fit),
    }
    if not validation_only:
        datasets["v_cal"] = make_dataset(v_cal)
        datasets["test"] = make_dataset(cell_data.test)
    if datasets["train"].sampling_plan is not None:
        _record_sampling(cell.output_dir, datasets["train"].sampling_plan, seeds)

    main.probe_args.input_size = joined.width * (2 if cell_data.ppi else 1)
    main.probe_args.branch_widths = list(widths)
    if settings.branch_projection_sizes is not None:
        main.probe_args.branch_sizes = list(settings.branch_projection_sizes)
    if recipe.quantile_clip:
        # The standardizer already clipped every feature, so the probe's own clamp is off.
        main.probe_args.input_clip = None
    if settings.gp_lengthscale_squared is not None:
        # main.py has no flag for it; get_probe hands every probe argument to the config.
        main.probe_args.gp_lengthscale_squared = settings.gp_lengthscale_squared
    main.probe_args.num_labels = cell_data.num_labels
    main.probe_args.task_type = cell_data.label_type
    main.trainer_args.task_type = cell_data.label_type
    if cell_data.kind == "regression" and not validation_only:
        # Protify weighted its balanced regression metrics for the whole validation split;
        # the trainer evaluates V_fit instead. Its weights read test labels, which a
        # validation-only fit never loads, so that mode leaves them off (a declared
        # regression release already requires balanced_regression_metrics false).
        main._compute_balanced_weights_for(cell_data.data_name, cell_data.train, v_fit, cell_data.test)

    if cell_data.release is not None or main.trainer_args.record_training_recipe:
        effective = main.trainer_args.for_datasets(main.probe_args, datasets["train"], datasets["v_fit"], datasets.get("test"))
        resolved = resolved_training_recipe(effective, main.probe_args, main.trainer_args.early_stopping(), datasets["train"].sampling_plan,
                                            checkpoint_selection=settings.checkpoint_selection)
        _record_training_recipe(cell.output_dir, resolved, seeds)

    test_digests = () if validation_only else row_digests(cell_data.test)
    for seed in seeds:
        if seed_is_complete(cell.output_dir, seed):
            print(f"Seed {seed} of {cell.cell_id} is complete; resuming past it.")
            continue
        _train_seed(cell, cell_data, seed, datasets, collator, display_name(specs), test_digests, make_dataset, v_fit)

    report = _report(cell, cell_data, seeds, specs, widths, roles)
    write_json(os.path.join(cell.output_dir, REPORT), report)
    return report


def split_cell(cell: UncertaintyCell, *, threads: int | None = None) -> dict[str, Any]:
    """Write the identity-stratified split manifest `uncertainty.split_manifest` names, seeded by `split_seed`."""
    if getattr(cell.uncertainty, "dataset_manifest", None) is not None:
        raise ValueError("declared datasets already have frozen splits and cannot be re-split")
    path = cell.uncertainty.split_manifest
    if path is None:
        raise ValueError("uncertainty.split_manifest must name where the split manifest goes.")

    specs = [parse_branch_spec(token) for token in cell.uncertainty.branch_specs]
    cell_data = load_cell_data(cell_arguments(cell, specs))
    return build_split_manifest(
        cell_data.data_name, [cell_data.train, cell_data.valid, cell_data.test], cell_data.label_type, path, seed=cell.uncertainty.split_seed, threads=threads
    )


def score_identity(
    cell: UncertaintyCell, *, explicit_rows: tuple[Sequence[int], Sequence[int]] | None = None
) -> dict[str, Any]:
    """Write each test, validation, V_fit, and V_cal row's highest identity to training.

    MMseqs searches every distinct test and validation sequence against the distinct training
    sequences once. The result goes to `uncertainty.identity_path`, which `run` reads back.
    """
    path = cell.uncertainty.identity_path
    if path is None:
        raise ValueError("uncertainty.identity_path must name where the identity file goes.")

    specs = [parse_branch_spec(token) for token in cell.uncertainty.branch_specs]
    cell_data, explicit_rows = load_split_cell_data(cell, cell_arguments(cell, specs), explicit_rows)
    roles = resolve_validation_roles(
        cell_data.valid["labels"] if explicit_rows is not None else list(cell_data.valid["labels"]),
        cell_data.label_type,
        fraction=cell.uncertainty.calibration_fraction,
        seed=cell.uncertainty.split_seed,
        explicit_rows=explicit_rows,
    )
    names = sequence_columns(cell_data.train)
    references = {sequence_digest(sequence): sequence for name in names for sequence in cell_data.train[name]}
    queries = {
        sequence_digest(sequence): sequence
        for split in (cell_data.valid, cell_data.test)
        for name in names
        for sequence in split[name]
    }
    # The search stage is kept beside the identity file, so a rerun reads its finished hits.
    search_directory = Path(path).parent / f"{Path(path).stem}_search"
    sequence_identity = max_identity_to_reference(queries, references, search_directory)

    valid_identity = row_identity(sequence_identity, [list(cell_data.valid[name]) for name in names])  # (n_valid,)
    rows = {
        "test": row_identity(sequence_identity, [list(cell_data.test[name]) for name in names]),
        "valid": valid_identity,
        "v_fit": valid_identity[roles.fit],
        "v_cal": valid_identity[roles.cal],
    }
    payload = {
        "cell_id": cell.cell_id,
        "data_name": cell_data.data_name,
        "search": DEFAULT_SEARCH.result_parameters(),
        "query_sequences": len(queries),
        "reference_sequences": len(references),
        "validation_roles": {"source": roles.source, "fraction": roles.fraction, "seed": roles.seed},
        "bin_counts": {split: bin_counts(identity) for split, identity in rows.items()},
        "rows": {split: identity.tolist() for split, identity in rows.items()},
        "sequences": sequence_identity,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_json(path, json_ready(payload))
    return payload
