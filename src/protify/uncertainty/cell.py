"""The cell YAML: one dataset, one representation, one head, and its seeds.

```yaml
cell_id: deeplocbinary_both_projected
experiment_id: 2026-09-24_sae_dense_ablation
tags: {arm: both, group: deeploc-2.both, role: final}   # optional, strings only; copied into report.json
protify_args:            # overrides onto Protify's parsed defaults; an unknown key is refused
  data_names: [DeepLoc-2]
  sql: true
  seed: 42               # the data seed, fixed; run seeds are seed, seed + 1, ...
  num_runs: 5
uncertainty:
  branch_specs: ["ESMC-600-SAE-l27-k64-c16384:max", "ESMC-600:mean,var"]
  head: projected        # one of HEADS; see HEAD_RECIPES
  projection_size: 1024
  # branch_projection_sizes: [512, 512, 1024]   # optional; one width per branch_specs entry, summing to projection_size, which then reads 2048
  learning_rate: 0.0003  # optional; Protify's probe learning rate otherwise
  weight_decay: 0.01     # optional; Protify's AdamW weight decay (0.0) otherwise
  dropout: 0.1           # optional; Protify's dropout otherwise
  # gp_lengthscale_squared: 256   # sngp only, optional; ProjectedProbeConfig's 1024 otherwise
  laplace: false         # true: each seed also saves a last-layer KFAC Laplace variance
  loss_head_folds: 0     # k >= 2: each seed also trains a cross-fitted loss predictor
  pnorm_uncertainty: false   # true: the report adds temperature_pnorm (multiclass)
  calibration_fraction: 0.5
  split_seed: 0
  mc_dropout_samples: 30
  conformal_levels: [0.8, 0.9, 0.95]
  identity_path: /vol/artifacts/<experiment>/identity/<dataset>.json   # optional
  split_manifest: /vol/artifacts/<experiment>/splits/<dataset>.json   # optional; see stratified_split.py
  require_cached_embeddings: false   # true: raise on a sequence a branch cache lacks, never embed
  save_heads: false      # true: keep each seed's trained head, for `python -m protify.uncertainty package`
  save_trunk_features: false   # true: each seed's npz also holds every V_fit, V_cal, and test row's trunk features
  unlisted_labels: unknown     # negative: a declared multilabel release is a closed world, every unlisted label a negative (GO)
output_dir: /vol/artifacts/<experiment>/runs/deeplocbinary_both_projected
```

`gp_lengthscale_squared` changes what a seed trains, so a resumed directory must match it, as
must `weight_decay`, `laplace`, `loss_head_folds`, and `split_manifest`, which re-splits the
rows. `save_heads` changes what a seed saves, so it must match too: a head that was never saved
cannot be recovered without training the seed again, and so must `save_trunk_features`, for the
same reason. `require_cached_embeddings`, `pnorm_uncertainty`, and `tags` do not: the first
only decides whether a missing cache row is embedded or refused, the second only adds a
method the report computes from the saved seed files, and the third only labels the report.

The heads beyond `projected` and `sngp` serve the single-probe uncertainty experiment:
`linear` is one Linear layer on the clamped joined input, `faithful` and `beta_nll` add a
variance output for regression, and `projected_qclip` is `projected` with each standardized
feature clipped to its training quantiles instead of plus or minus 10.
"""

from __future__ import annotations

import math
import re
import yaml

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .metric_contracts import SELECTION_METRICS
from ..data.declared_dataset import checked_digest
from ..data.probe_features import validate_probe_features


@dataclass(frozen=True)
class HeadRecipe:
    """How a cell head is built: the Protify probe type, and whether the standardizer clips.

    `quantile_clip` clips each standardized feature to its `CLIP_QUANTILES` over the training
    rows and turns the probe's own clamp off. `regression_only` heads predict a Gaussian
    variance. `laplace_ready` heads end in one Linear layer, the layer a last-layer Laplace
    posterior and the loss predictor's fold heads are defined for.
    """

    probe_type: str
    quantile_clip: bool = False
    regression_only: bool = False
    laplace_ready: bool = False


HEAD_RECIPES = {
    "projected": HeadRecipe("projected", laplace_ready=True),
    "sngp": HeadRecipe("sngp"),
    "linear": HeadRecipe("projected_linear", laplace_ready=True),
    "faithful": HeadRecipe("projected_faithful", regression_only=True),
    "beta_nll": HeadRecipe("projected_beta_nll", regression_only=True),
    "projected_qclip": HeadRecipe("projected", quantile_clip=True, laplace_ready=True),
}
HEADS = tuple(HEAD_RECIPES)
# The training quantiles `projected_qclip` clips each standardized feature to.
CLIP_QUANTILES = (0.001, 0.999)
CELL_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
CELL_KEYS = frozenset({"cell_id", "experiment_id", "tags", "protify_args", "uncertainty", "output_dir"})


@dataclass(frozen=True)
class UncertaintySettings:
    branch_specs: tuple[str, ...]
    head: str = "projected"
    projection_size: int = 1024
    # One projected width per branch, summing to `projection_size`; None splits `projection_size` evenly, as every cell before it did.
    branch_projection_sizes: tuple[int, ...] | None = None
    learning_rate: float | None = None
    dropout: float | None = None
    # SNGP's squared RBF lengthscale; None keeps ProjectedProbeConfig's default.
    gp_lengthscale_squared: float | None = None
    calibration_fraction: float = 0.5
    split_seed: int = 0
    mc_dropout_samples: int = 30
    conformal_levels: tuple[float, ...] = (0.8, 0.9, 0.95)
    identity_path: str | None = None
    # An identity-stratified split's manifest (stratified_split.py): the cell's rows pooled and
    # split again, with validation as V_fit and calibration as V_cal. None keeps the published
    # splits and carves V_fit and V_cal from validation. `split_seed` seeds the manifest's split.
    split_manifest: str | None = None
    # A typed four-way Parquet release, pinned independently of its contents. It bypasses
    # legacy inference and the split-manifest pooling path entirely.
    dataset_manifest: str | None = None
    dataset_sha256: str | None = None
    # Native feature-store inputs, pinned separately from the declared dataset.
    feature_manifest: str | None = None
    feature_sha256: str | None = None
    # The probe matrix's inputs: a base's canonical store, or a generated control (data/probe_features.py).
    # Unlike `feature_manifest` it pins no file digest; the engine verifies each part it reads.
    probe_features: dict[str, Any] | None = None
    # Declared releases draw clusters, proteins, then eligible assays uniformly. None
    # resolves to one draw per eligible cluster per epoch, with replacement.
    cluster_draws_per_epoch: int | None = None
    cluster_class_balance: bool = False
    checkpoint_selection: str | None = None
    # Concurrent training containers read one shared cache and must never write it, so this
    # raises on a sequence a branch cache lacks instead of embedding it.
    require_cached_embeddings: bool = False
    # Keep each seed's trained head under `heads/seed_<seed>/`, and the embedding standardizer,
    # so the cell can be packaged. Off, a run writes the seed files and report it always has;
    # only cell.json gains the setting, which an older cell.json lacks and reads as false.
    save_heads: bool = False
    # Keep every V_fit, V_cal, and test row's trunk features, the vectors the output layer reads,
    # in each seed's npz, for `python -m protify.uncertainty neighbors`. An older cell.json lacks
    # the setting and reads as false.
    save_trunk_features: bool = False
    # AdamW's weight decay; None keeps Protify's, 0.0, so an older cell trains as it did.
    weight_decay: float | None = None
    # Fit a last-layer KFAC Laplace posterior after training and save its latent variance.
    laplace: bool = False
    # Cross-fit a loss predictor over this many stratified folds of train; 0 is off.
    loss_head_folds: int = 0
    # Add temperature_pnorm, temperature's probabilities scored by -max z / ||z||_p of the centered
    # logits z, on multiclass tasks.
    pnorm_uncertainty: bool = False
    # Canonical fitting collects validation predictions only. Calibration and test evaluation
    # are separate stages after checkpoint and seed selection have been frozen.
    evaluation_mode: str = "full"
    # How a multilabel release's unlisted labels read. "unknown" keeps the release's observation mask as written. "negative" is
    # a closed world: a label a row does not carry is a negative, so every label counts as observed in training, calibration and
    # scoring (the GO Oracle, whose release already carries each term's ancestors). It needs a declared multilabel release.
    unlisted_labels: str = "unknown"
    # The V_fit score that picks a seed (and, in a suite, the learning rate): one of metric_contracts.SELECTION_METRICS for the
    # task kind, None for the kind's primary. It needs a declared release, whose canonical metrics define it.
    selection_metric: str | None = None


UNLISTED_LABEL_READINGS = ("unknown", "negative")
UNCERTAINTY_KEYS = frozenset(UncertaintySettings.__dataclass_fields__)


@dataclass(frozen=True)
class UncertaintyCell:
    cell_id: str
    experiment_id: str
    uncertainty: UncertaintySettings
    output_dir: str
    protify_args: Mapping[str, Any] = field(default_factory=dict)
    # Labels an orchestrator reads back from report.json, such as the arm and the tuning role.
    tags: Mapping[str, str] = field(default_factory=dict)


def _refuse_unknown(section: str, keys: set[str], known: frozenset[str]) -> None:
    unknown = sorted(keys - known)
    if unknown:
        raise ValueError(f"{section} has unknown keys {unknown}; known keys are {sorted(known)}.")


def _settings(mapping: Mapping[str, Any]) -> UncertaintySettings:
    _refuse_unknown("uncertainty", set(mapping), UNCERTAINTY_KEYS)
    dataset_path, dataset_digest = mapping.get("dataset_manifest"), mapping.get("dataset_sha256")
    draws = mapping.get("cluster_draws_per_epoch")
    balance = mapping.get("cluster_class_balance", False)
    if type(balance) is not bool or (balance and dataset_path is None):
        raise ValueError("cluster_class_balance requires a boolean and a declared dataset")
    selection = mapping.get("checkpoint_selection")
    if selection is not None:
        if selection != "validation_mcc_then_observed_loss_initial_v1" or dataset_path is None:
            raise ValueError("checkpoint_selection requires the supported policy and a declared dataset")
        if mapping.get("selection_metric") not in (None, "mcc"):
            raise ValueError("initial MCC checkpoint selection requires selection_metric=mcc")
    if draws is not None and (type(draws) is not int or draws < 1 or dataset_path is None):
        raise ValueError("cluster_draws_per_epoch requires a declared dataset and a positive integer")
    if dataset_path is not None and mapping.get("loss_head_folds", 0):
        raise ValueError("declared cluster sampling requires a cluster-aware loss-head fold protocol before cross-fitting")
    if (dataset_path is None) != (dataset_digest is None):
        raise ValueError("dataset_manifest and dataset_sha256 must be supplied together")
    if dataset_path is not None:
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            raise ValueError("dataset_manifest must be a nonempty path")
        checked_digest(dataset_digest)
        if mapping.get("split_manifest") is not None:
            raise ValueError("a declared dataset cannot be re-split with split_manifest")
    branch_specs = mapping.get("branch_specs")
    if not branch_specs or not all(isinstance(token, str) for token in branch_specs):
        raise ValueError("uncertainty.branch_specs must be a non-empty list of MODEL:POOL[,POOL][@HS] strings.")

    feature_path, feature_digest = mapping.get("feature_manifest"), mapping.get("feature_sha256")
    if (feature_path is None) != (feature_digest is None):
        raise ValueError("feature_manifest and feature_sha256 must be supplied together")
    if feature_path is not None:
        if not isinstance(feature_path, str) or not feature_path.strip():
            raise ValueError("feature_manifest must be a nonempty path")
        checked_digest(feature_digest)
        if dataset_path is None or mapping.get("require_cached_embeddings") is not True or mapping.get("head", "projected") != "projected":
            raise ValueError("native feature inputs require a declared dataset, cached embeddings and a projected head")

    probe_features = mapping.get("probe_features")
    if probe_features is not None:
        validate_probe_features(probe_features)
        if feature_path is not None or dataset_path is None or mapping.get("require_cached_embeddings") is not True or mapping.get("head", "projected") != "projected":
            raise ValueError("probe_features replaces feature_manifest and requires a declared dataset, cached embeddings and a projected head")

    lengthscale_squared = mapping.get("gp_lengthscale_squared")
    branch_sizes = mapping.get("branch_projection_sizes")
    if branch_sizes is not None:
        if (not isinstance(branch_sizes, list | tuple) or len(branch_sizes) != len(branch_specs)
                or not all(type(size) is int and size > 0 for size in branch_sizes)):
            raise ValueError("uncertainty.branch_projection_sizes must list one positive integer for each of the branch_specs.")
        if sum(branch_sizes) != mapping.get("projection_size", 1024):
            raise ValueError(
                f"uncertainty.branch_projection_sizes sum to {sum(branch_sizes)}, so projection_size must be {sum(branch_sizes)}, "
                f"not {mapping.get('projection_size', 1024)}."
            )
    settings = UncertaintySettings(**{
        **mapping,
        "branch_specs": tuple(branch_specs),
        "branch_projection_sizes": None if branch_sizes is None else tuple(branch_sizes),
        "gp_lengthscale_squared": None if lengthscale_squared is None else float(lengthscale_squared),
        "conformal_levels": tuple(float(level) for level in mapping.get("conformal_levels", (0.8, 0.9, 0.95))),
    })
    if settings.head not in HEADS:
        raise ValueError(f"uncertainty.head must be one of {HEADS}; got {settings.head!r}.")
    if not isinstance(settings.evaluation_mode, str) or settings.evaluation_mode not in {"full", "validation_only"}:
        raise ValueError("evaluation_mode must be full or validation_only")
    if settings.unlisted_labels not in UNLISTED_LABEL_READINGS:
        raise ValueError(f"unlisted_labels must be one of {UNLISTED_LABEL_READINGS}")
    if settings.unlisted_labels == "negative" and dataset_path is None:
        raise ValueError("unlisted_labels negative reads a declared multilabel release, so it needs dataset_manifest")
    if settings.selection_metric is not None and (
        dataset_path is None or settings.selection_metric not in {name for names in SELECTION_METRICS.values() for name in names}
    ):
        raise ValueError(f"selection_metric must name a metric of SELECTION_METRICS and needs dataset_manifest: {SELECTION_METRICS}")
    if settings.evaluation_mode == "validation_only" and (
        dataset_path is None or not settings.save_heads or settings.laplace or settings.loss_head_folds or settings.pnorm_uncertainty
    ):
        raise ValueError("validation_only requires a declared dataset, saved heads, and no calibration extras")
    if settings.projection_size < 1:
        raise ValueError("uncertainty.projection_size must be positive.")
    if settings.learning_rate is not None and settings.learning_rate <= 0:
        raise ValueError("uncertainty.learning_rate must be positive.")
    if settings.dropout is not None and not 0.0 <= settings.dropout < 1.0:
        raise ValueError("uncertainty.dropout must lie in [0, 1).")
    if settings.gp_lengthscale_squared is not None:
        if settings.head != "sngp":
            raise ValueError(
                f"uncertainty.gp_lengthscale_squared sets the SNGP kernel, but head is {settings.head!r}."
            )
        if not 0.0 < settings.gp_lengthscale_squared < math.inf:
            raise ValueError("uncertainty.gp_lengthscale_squared must be positive and finite.")
    if not isinstance(settings.require_cached_embeddings, bool):
        raise ValueError("uncertainty.require_cached_embeddings must be true or false.")
    for flag in ("save_heads", "save_trunk_features"):
        if not isinstance(getattr(settings, flag), bool):
            raise ValueError(f"uncertainty.{flag} must be true or false.")
    if settings.weight_decay is not None and not 0.0 <= settings.weight_decay < math.inf:
        raise ValueError("uncertainty.weight_decay must be non-negative and finite.")
    for flag in ("laplace", "pnorm_uncertainty"):
        if not isinstance(getattr(settings, flag), bool):
            raise ValueError(f"uncertainty.{flag} must be true or false.")
    folds = settings.loss_head_folds
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 0 or folds == 1:
        raise ValueError("uncertainty.loss_head_folds must be 0 (off) or an integer of at least 2.")
    recipe = HEAD_RECIPES[settings.head]
    if (settings.laplace or folds) and not recipe.laplace_ready:
        ready = sorted(head for head, candidate in HEAD_RECIPES.items() if candidate.laplace_ready)
        raise ValueError(
            f"uncertainty.laplace and uncertainty.loss_head_folds need a head that ends in one Linear "
            f"layer, one of {ready}; got head {settings.head!r}."
        )
    if not 0.0 < settings.calibration_fraction < 1.0:
        raise ValueError("uncertainty.calibration_fraction must lie strictly between 0 and 1.")
    if settings.mc_dropout_samples < 0:
        raise ValueError("uncertainty.mc_dropout_samples cannot be negative.")
    levels = settings.conformal_levels
    if not levels or any(not 0.0 < level < 1.0 for level in levels) or len(set(levels)) != len(levels):
        raise ValueError("uncertainty.conformal_levels must be distinct coverages strictly between 0 and 1.")
    return settings


def _tags(value: object) -> dict[str, str]:
    """The cell's tags, refused unless every key and value is a string.

    YAML reads an unquoted `1` or `yes` as a number or a boolean, which would reach the report
    as something other than the label written.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(tag, str) for key, tag in value.items()
    ):
        raise ValueError(
            f"tags must map strings to strings, quoting any value YAML reads as a number or boolean; got {value!r}."
        )
    return dict(value)


def cell_from_mapping(mapping: Mapping[str, Any]) -> UncertaintyCell:
    _refuse_unknown("the cell", set(mapping), CELL_KEYS)
    missing = sorted({"cell_id", "experiment_id", "uncertainty", "output_dir"} - set(mapping))
    if missing:
        raise ValueError(f"the cell is missing {missing}.")
    if not CELL_ID_PATTERN.match(str(mapping["cell_id"])):
        raise ValueError(f"cell_id {mapping['cell_id']!r} must be usable as a file name.")
    protify_args = mapping.get("protify_args") or {}
    if not isinstance(protify_args, Mapping):
        raise ValueError("protify_args must be a mapping of Protify argument names to values.")

    return UncertaintyCell(
        cell_id=str(mapping["cell_id"]),
        experiment_id=str(mapping["experiment_id"]),
        uncertainty=_settings(mapping["uncertainty"]),
        output_dir=str(mapping["output_dir"]),
        protify_args=dict(protify_args),
        tags=_tags(mapping.get("tags")),
    )


def load_cell(path: str) -> UncertaintyCell:
    with open(path, encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle)
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{path} does not hold a cell mapping.")
    return cell_from_mapping(mapping)
