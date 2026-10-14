"""Package one finished cell for serving: `python -m protify.uncertainty package`.

    python -m protify.uncertainty package <cell output_dir> --method temperature --output <package dir>
        [--levels 0.8 0.9] [--revisions revisions.json]

The cell must have run with `uncertainty.save_heads: true`. The method is rebuilt from the
cell's saved per-seed predictions by `build_methods`, the function its report was scored with,
so a package ships the calibration the report measured and never fits one again:

- An ensemble ships every seed. A per-seed method ships the seed whose head scored best on
  V_fit by the primary metric, ties to the lowest seed; test is never read to choose.
- `calibration.json` holds the shipped run's fitted temperature or variance map, the
  conformal quantiles of that calibrated prediction on V_cal at each level, and, for
  classification, the answer rule: V_cal's wrong answers' confidences, which conformal
  selection reads (`answering.py`). It also holds the novelty cutoff and support table
  (`applicability`) and the answer-or-abstain policy that pins them (`decision_policy`),
  both from `applicability.py`.
- `metrics.json` scores the shipped run on test through `report.py`'s scorers, and the answer
  rule at its default target; with a correctness model, the test rows' novelty and the
  policy's decisions on them, as one request and one row per request.
- `features.json` says how to build the rows the heads read, `origin.json` where the cell came
  from, and `README.md` is the model card.

Before writing, the package is checked three ways, and any disagreement is refused: its
calibration, applied by `packaged.py` to the saved predictions, must reproduce the method's
prediction on every split exactly; its quantiles must reproduce `conformal_sets`' test
coverage exactly; and its test scores and fitted parameters must match the report's for that
run. The package is built in a temporary sibling directory and renamed into place, so no
reader sees a partial one.

Shapes: `n` rows of a split, `c` classes or labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import numpy as np

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from foundry.logging.revision import source_revision
from .answering import (
    answer_confidence,
    answer_rule_record,
    conformal_p_values,
    decode,
    requires_a_label,
    select_answers,
)
from .applicability import (
    APPLICABILITY_UNKNOWN_MESSAGES,
    NOT_ASSESSED_MESSAGES,
    decision_policy,
    unavailable_applicability,
)
from .conformal_sets import SET_SCORES, regression_quantile, regression_spread_floor, set_quantile
from .methods import LAPLACE_METHOD, LOSS_HEAD_METHOD, PNORM_METHOD, Method, MethodRun, build_methods
from .metric_contracts import metric_contract
from .metrics import CLASSIFICATION_KINDS, PRIMARY_METRICS
from .observation_contract import answer_truth_bounds, calibration_observations, partial_answer_unavailable
from .package_correctness import PackagedCorrectness, check_shipped_correctness, package_correctness, write_correctness
from .packaged import (
    CALIBRATION_RECORD,
    CORRECTNESS_DIRECTORY,
    FEATURES_RECORD,
    HEADS_DIRECTORY,
    METHOD_RECIPES,
    METRICS_RECORD,
    MODEL_CARD,
    ORIGIN_RECORD,
    PACKAGE_SCHEME,
    STANDARDIZER_FILE,
    calibrated_predictive,
    calibration_kind,
    conformal_prediction,
    load_package,
)
from .prediction_arrays import prediction_mask_is_complete
from .predictions import SPLITS, SplitPredictions
from .report import conformal_rows, label_support, read_json, score_rows, write_json
from .verdicts import json_ready
from .workflow import (
    CELL_RECORD,
    REPORT,
    SPLIT_RECORD,
    STANDARDIZER_RECORD,
    _primary,
    agreed_labels,
    head_directory,
    head_is_saved,
    load_seeds,
    load_standardizer_record,
    protify_arguments,
    seed_is_complete,
)
from ..base_models.control_bases import CONTROL_SCHEMA, is_control_model
from ..base_models.esmc_sae import ESMC_SAE_COVERAGE, parse_sae_model_name
from ..base_models.supported_models import all_presets_with_paths
from ..data.branch_features import BranchSpec
from ..data.dataset_classes import EmbeddingStandardizer
from ..data.declared_dataset import validate_declared_manifest
from ..data.probe_features import probe_provenance
from ..metrics import validate_observed_labels
from ..probes.projected_probe import ProjectedProbeConfig
from ..probes.training_recipe import validate_training_receipt


# Protify's project root, whose `src` the Modal grid hashes for a run's revision.
PROTIFY_ROOT = Path(__file__).resolve().parents[3]
# Methods that rank rows by another score and predict as their head.
SCORE_ONLY_SUFFIXES = ("_entropy", "_nn_distance", "_variance")
# The heads a package rebuilds, and the methods built from per-seed extras it cannot ship yet.
PACKAGED_HEADS = frozenset({"projected", "sngp"})
UNPACKAGED_METHODS = frozenset({LAPLACE_METHOD, LOSS_HEAD_METHOD, PNORM_METHOD})
# A probe input crops at 2,046 residues, so a row is at most 2,048 tokens with CLS and EOS.
PROBE_MAX_LENGTH = 2048
# Embedding dtype names as `MainProcess.apply_current_settings` reads them.
EMBEDDING_DTYPES = {
    "fp32": "float32",
    "float32": "float32",
    "fp16": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}
# How far a rebuilt fitted parameter or test score may sit from the report's: summation order
# can differ between the machine that wrote the report and this one, and nothing else may.
REPORT_TOLERANCE = {"rtol": 1e-6, "atol": 1e-9}
# The conformal set scores a package ships. The report also measures APS, which is never shipped:
# on a binary task at high levels its threshold sits at a cumulative probability of 1.0, where
# floating-point rounding, which differs between machines, decides whether a set holds one class
# or two. Its sets then move by up to about 0.01 in coverage from one machine to the next.
SHIPPED_SET_SCORES = ("lac",)


@dataclass(frozen=True)
class FinishedCell:
    """What a finished `save_heads` cell directory holds, read once."""

    directory: str
    record: dict[str, Any]  # cell.json
    report: dict[str, Any]  # report.json
    split: dict[str, Any]  # split.json
    split_sha256: str
    seed_predictions: dict[int, dict[str, SplitPredictions]]
    labels: dict[str, np.ndarray]
    standardizer: EmbeddingStandardizer | None
    train_label_std: float

    @property
    def kind(self) -> str:
        return self.report["task_type"]

    @property
    def seeds(self) -> list[int]:
        return list(self.report["seeds"])

    @property
    def label_masks(self) -> dict[str, np.ndarray | None]:
        """Masks agreed across seeds by `agreed_labels`, including explicit all-observed masks."""
        return {role: split.label_mask for role, split in self.seed_predictions[self.seeds[0]].items()}  # each (n,) or (n, c) by role, or None

    @cached_property
    def observation_contract(self) -> dict[str, Any] | None:
        return calibration_observations(self.seed_predictions[self.seeds[0]], self.kind)


@dataclass(frozen=True)
class ShippedRun:
    """The run a package ships: its method, its seeds, and how those seeds were chosen."""

    method: Method
    run: MethodRun
    seeds: list[int]
    seed_choice: dict[str, Any]


def read_finished_cell(directory: str) -> FinishedCell:
    """Read a finished cell, refusing one that ran without `save_heads` or lacks a head."""
    for name in (CELL_RECORD, SPLIT_RECORD, REPORT):
        if not os.path.exists(os.path.join(directory, name)):
            raise ValueError(f"{directory} lacks {name}; package reads a cell that `run` has finished.")

    record = read_json(os.path.join(directory, CELL_RECORD))
    if not record["uncertainty"].get("save_heads", False):
        raise ValueError(
            f"{directory} ran without uncertainty.save_heads, so its trained heads were not kept. "
            f"Run the cell again with save_heads: true in a new output_dir."
        )

    report = read_json(os.path.join(directory, REPORT))
    if record["uncertainty"].get("evaluation_mode", "full") != "full" or report.get("evaluation_mode", "full") != "full":
        raise ValueError("validation-only fitting cannot be packaged before a separate frozen-checkpoint calibration and test stage")
    release_digest = record["uncertainty"].get("dataset_sha256")
    if (release_digest is None) != (report.get("dataset_release") is None):
        raise ValueError("cell and report disagree on whether a declared dataset release was used")
    if release_digest is not None:
        release = read_json(os.path.join(directory, "dataset_release.json"))
        if release != report.get("dataset_release") or release.get("manifest_sha256") != release_digest:
            raise ValueError("declared dataset release metadata differs between the cell, report, and release record")
        validate_declared_manifest({key: value for key, value in release.items() if key != "manifest_sha256"})
    if report["head"] not in PACKAGED_HEADS:
        raise ValueError(
            f"{directory} trained a {report['head']!r} head, and packaging one is not built yet: a package "
            f"ships the {sorted(PACKAGED_HEADS)} heads, whose features it rebuilds with a plain standardizer "
            f"and the probe's own clamp. See docs/uncertainty_packages.md."
        )
    seeds = list(report["seeds"])
    unfinished = [seed for seed in seeds if not seed_is_complete(directory, seed)]
    headless = [seed for seed in seeds if not head_is_saved(directory, seed)]
    if unfinished or headless:
        raise ValueError(f"{directory} has unfinished seeds {unfinished} and seeds without a saved head {headless}.")
    if release_digest is not None:
        expected_metrics = metric_contract(release)
        metric_path = os.path.join(directory, "metric_contract.json")
        if not os.path.isfile(metric_path) or read_json(metric_path) != expected_metrics or report.get("metric_contract") != expected_metrics:
            raise ValueError("declared package lacks its matching metric contract")
        for seed in seeds:
            if read_json(os.path.join(directory, f"seed_{seed}.json")).get("metric_contract") != expected_metrics:
                raise ValueError("declared package seed metric contract differs from its endpoint")
        sampling_path = os.path.join(directory, "sampling.json")
        if not os.path.isfile(sampling_path) or read_json(sampling_path) != report.get("sampling"):
            raise ValueError("declared package lacks its matching cluster sampling plan")
        reported_seeds = report.get("per_seed", [])
        if [entry.get("seed") for entry in reported_seeds] != seeds:
            raise ValueError("declared package sampling report has different seeds")
        for seed, entry in zip(seeds, reported_seeds, strict=True):
            receipt = read_json(os.path.join(directory, f"seed_{seed}.json")).get("sampling")
            if not isinstance(receipt, dict) or receipt != entry.get("sampling") or receipt.get("plan") != report["sampling"] or receipt.get("seed") != seed:
                raise ValueError("declared package sampling receipt differs from its seed or plan")
    if release_digest is not None or record["protify_args"].get("record_training_recipe", False):
        recipe_path = os.path.join(directory, "training_recipe.json")
        if not os.path.isfile(recipe_path) or read_json(recipe_path) != report.get("training_recipe"):
            raise ValueError("package lacks its matching training recipe")
        reported_seeds = report.get("per_seed", [])
        if [entry.get("seed") for entry in reported_seeds] != seeds:
            raise ValueError("package training recipe report has different seeds")
        for seed, entry in zip(seeds, reported_seeds, strict=True):
            receipt = read_json(os.path.join(directory, f"seed_{seed}.json")).get("training")
            if receipt != entry.get("training"):
                raise ValueError("package training recipe receipt differs from its seed record")
            validate_training_receipt(report["training_recipe"], receipt, seed)
    standardizer_path = os.path.join(directory, STANDARDIZER_RECORD)
    if not os.path.exists(standardizer_path):
        raise ValueError(f"{directory} lacks {STANDARDIZER_RECORD}, which a save_heads run writes before its first seed.")

    split_path = Path(directory) / SPLIT_RECORD
    split = read_json(str(split_path))
    seed_predictions = load_seeds(directory, seeds)
    if release_digest is not None:
        for seed, entry in zip(seeds, report["per_seed"], strict=True):
            expected_primary = _primary(seed_predictions[seed]["v_fit"], report["task_type"], canonical=True, selection=report.get("selection_metric"))
            recorded_primary = read_json(os.path.join(directory, f"seed_{seed}.json")).get("v_fit_primary")
            if entry.get("v_fit_primary") != expected_primary or recorded_primary != expected_primary:
                raise ValueError("declared package validation primary differs from its saved predictions")
    labels = agreed_labels(seed_predictions, seeds)
    named = {role: len(split[f"{role}_rows"]) for role in ("v_fit", "v_cal")}
    saved = {role: int(labels[role].shape[0]) for role in ("v_fit", "v_cal")}
    if named != saved:
        raise ValueError(f"{directory}'s seeds saved {saved} rows, but its split.json names {named}.")

    standardizer, train_label_std = load_standardizer_record(standardizer_path)
    return FinishedCell(
        directory=directory,
        record=record,
        report=report,
        split=split,
        split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        seed_predictions=seed_predictions,
        labels=labels,
        standardizer=standardizer,
        train_label_std=train_label_std,
    )


def best_v_fit_seed(per_seed: Sequence[Mapping[str, Any]]) -> int:
    """The seed whose head scored best on V_fit by the primary metric, ties to the lowest.

    Every primary metric is higher-is-better. A seed whose V_fit metric is undefined is
    passed over, and test is never read.
    """
    scored = [(record["v_fit_primary"], record["seed"]) for record in per_seed if record.get("v_fit_primary") is not None]
    if not scored:
        raise ValueError("no seed recorded a V_fit primary metric, so none can be chosen to ship.")
    best = max(value for value, _ in scored)
    return min(seed for value, seed in scored if value == best)


def select_method(cell: FinishedCell, name: str) -> Method:
    """The named method, as `build_methods` builds it from the cell's saved predictions."""
    if name.endswith(SCORE_ONLY_SUFFIXES):
        raise ValueError(
            f"{name!r} only ranks rows by another score and predicts as its head, so it cannot ship; "
            f"package the head's own method instead."
        )
    if name in UNPACKAGED_METHODS:
        raise ValueError(
            f"packaging {name!r} is not built yet: a package ships a head's outputs and one fitted map, and "
            f"{name} also needs its Laplace posterior, loss predictor, or chosen norm order at inference. "
            f"See docs/uncertainty_packages.md."
        )

    methods = {method.name: method for method in build_methods(cell.seed_predictions, cell.report["head"], cell.kind)}
    shippable = [method.name for method in methods.values() if not method.score_only and method.name in METHOD_RECIPES]
    if name not in shippable:
        raise ValueError(f"{cell.directory} has no shippable method {name!r}; it has {shippable}.")
    return methods[name]


def shipped_run(cell: FinishedCell, method: Method) -> ShippedRun:
    """Every seed for an ensemble; otherwise the seed `best_v_fit_seed` picks, and its run."""
    if len(method.runs) == 1 and method.runs[0].seed is None:
        return ShippedRun(method, method.runs[0], sorted(cell.seed_predictions), {"rule": "every_seed"})

    per_seed = [{"seed": record["seed"], "v_fit_primary": record.get("v_fit_primary")} for record in cell.report["per_seed"]]
    seed = best_v_fit_seed(per_seed)
    run = next(run for run in method.runs if run.seed == seed)
    choice = {"rule": "best_v_fit_primary", "primary_metric": cell.report.get("selection_metric", cell.report["primary_metric"]), "per_seed": per_seed}
    return ShippedRun(method, run, [seed], choice)


def report_levels(cell: FinishedCell, name: str) -> tuple[float, ...]:
    """The conformal levels the report scored this method at, read off its `@level` scores."""
    return tuple(sorted({float(score.rsplit("@", 1)[1]) for score in cell.report["methods"][name]["test"] if "@" in score}))


def _check_levels(levels: Sequence[float]) -> tuple[float, ...]:
    if any(not 0.0 < level < 1.0 for level in levels) or len(set(levels)) != len(levels):
        raise ValueError(f"--levels must be distinct coverages strictly between 0 and 1; got {list(levels)}.")
    return tuple(levels)


def shipped_conformal_rows(rows: Mapping[str, np.ndarray] | None) -> dict[str, np.ndarray]:
    """`conformal_rows`' per-row columns, without those of a set score no package ships."""
    # rows: (n,) per column name or None
    unshipped = tuple(f"{score}_" for score in SET_SCORES if score not in SHIPPED_SET_SCORES)
    return {name: values for name, values in (rows or {}).items() if not name.startswith(unshipped)}  # each (n,) by column name


def conformal_record(cell: FinishedCell, run: MethodRun, levels: Sequence[float]) -> dict[str, Any] | None:
    """The completed cell's shipped quantiles; the numerical fit only consumes V_cal."""
    return fitted_conformal_record(run, cell.kind, cell.labels["v_cal"], levels, cell.train_label_std, cell.label_masks["v_cal"])


def fitted_conformal_record(
    run: MethodRun, kind: str, labels: np.ndarray, levels: Sequence[float], train_label_std: float,
    label_mask: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """The shipped prediction's conformal quantiles on V_cal at each level; None for multilabel.

    Binary and multiclass get one quantile per shipped set score (LAC) and level, as
    `prediction_sets` computes it. Regression gets the quantile of `|y - mu| / (sigma + floor)`
    with the floor of `regression_conformal_rows`, or of the absolute residual for a method with
    no sigma.
    """
    # label_mask: (n,) or (n, c) or None
    calibration = run.splits["v_cal"]
    # labels: (n,) for scalar tasks, (n, c) for multilabel.
    if kind == "multilabel":
        return None
    if label_mask is not None:
        validate_observed_labels(labels, label_mask)
        if not label_mask.any():
            raise ValueError("conformal calibration requires observed calibration targets")
    observed = slice(None) if label_mask is None else label_mask.reshape(-1)
    labels = labels[observed].reshape(-1)  # (m,), observed calibration targets only
    if kind in CLASSIFICATION_KINDS:
        quantiles = [
            {"level": level, **{score: set_quantile(calibration.probs[observed], labels, level, score) for score in SHIPPED_SET_SCORES}}
            for level in levels
        ]
        record = {"levels": list(levels), "set_scores": list(SHIPPED_SET_SCORES), "quantiles": quantiles}
    elif kind == "regression":
        sigma = None if calibration.variance is None else np.sqrt(calibration.variance[observed])  # (m,) or None
        floor = regression_spread_floor(sigma is not None, train_label_std)
        quantiles = [
            {"level": level, "quantile": regression_quantile(calibration.mean[observed], labels, sigma, level, floor)} for level in levels
        ]
        record = {
            "levels": list(levels),
            "spread": "none" if sigma is None else "sigma",
            "spread_floor": floor,
            "train_label_std": train_label_std,
            "quantiles": quantiles,
        }
    else:
        return None

    rows = labels.shape[0]
    infinite = [entry["level"] for entry in quantiles if not all(math.isfinite(value) for value in entry.values())]
    if infinite:
        raise ValueError(
            f"V_cal's {rows} rows cannot calibrate levels {infinite}: the highest they can is "
            f"{rows}/{rows + 1} = {rows / (rows + 1):.4f}. Pass --levels below it."
        )
    return record


def _differences(rebuilt: Mapping[str, Any], reported: Mapping[str, Any]) -> list[str]:
    """Names whose values differ beyond `REPORT_TOLERANCE`, over the names both hold."""
    differ = []
    for name in sorted(set(rebuilt) & set(reported)):
        first, second = rebuilt[name], reported[name]
        if first is None or second is None:
            if first is not second:
                differ.append(name)
        elif not math.isclose(first, second, rel_tol=REPORT_TOLERANCE["rtol"], abs_tol=REPORT_TOLERANCE["atol"]):
            differ.append(name)
    return differ


def check_packaged_arithmetic(cell: FinishedCell, shipped: ShippedRun, calibration: Mapping[str, Any]) -> None:
    """Refuse unless the package's own arithmetic reproduces the method on every split.

    `calibrated_predictive` applies the shipped parameters to the saved head outputs, as a
    package does to new rows, and must give exactly what `build_methods` gave. Its quantiles
    must then reproduce `conformal_rows`' test coverage and set size or interval width exactly.
    """
    fields = ("probs", "mean", "variance", "uncertainty")
    for split in SPLITS:
        per_seed = {seed: cell.seed_predictions[seed][split] for seed in shipped.seeds}
        rebuilt = calibrated_predictive(
            per_seed, calibration["predictive"], calibration["calibration"], calibration["fitted"], cell.kind
        )
        expected = shipped.run.splits[split]
        for name in fields:
            first, second = getattr(rebuilt, name), getattr(expected, name)
            if (first is None) != (second is None) or (first is not None and not np.array_equal(first, second)):
                raise ValueError(f"the packaged {shipped.method.name} does not reproduce build_methods' {name} on {split}.")

    levels = () if calibration["conformal"] is None else tuple(calibration["conformal"]["levels"])
    rows = shipped_conformal_rows(conformal_rows(shipped.run, cell.kind, cell.labels, levels, cell.train_label_std, cell.label_masks))
    sets, intervals = conformal_prediction(shipped.run.splits["test"], calibration["conformal"], cell.kind)
    labels = cell.labels["test"]  # (n,)
    mask = cell.label_masks["test"]
    if mask is not None and cell.kind != "multilabel":
        labels = np.where(mask, labels, 0)  # unknown truth never reaches indexing or residuals
    if cell.kind != "multilabel":
        labels = labels.reshape(-1)  # (n,)
    rebuilt_rows: dict[str, np.ndarray] = {}
    for score, by_level in sets.items():
        for level, members in by_level.items():
            rebuilt_rows[f"{score}_coverage@{level}"] = members[np.arange(labels.shape[0]), labels].astype(np.float64)  # (n,)
            rebuilt_rows[f"{score}_size@{level}"] = members.sum(axis=1).astype(np.float64)  # (n,)
    for level, interval in intervals.items():
        rebuilt_rows[f"conformal_coverage@{level}"] = interval.covers(labels).astype(np.float64)  # (n,)
        rebuilt_rows[f"conformal_width@{level}"] = interval.width  # (n,)
    if mask is not None and cell.kind != "multilabel":
        for name, values in rebuilt_rows.items():
            if "coverage@" in name:
                values[~mask.reshape(-1)] = np.nan
    if rebuilt_rows.keys() != rows.keys() or any(not np.array_equal(rebuilt_rows[name], rows[name], equal_nan=True) for name in rows):
        raise ValueError(f"the packaged quantiles do not reproduce conformal_sets' test rows for {shipped.method.name}.")


def check_against_report(cell: FinishedCell, shipped: ShippedRun, test_scores: Mapping[str, Any]) -> None:
    """Refuse unless the rebuilt run's fitted parameters and test scores are the report's."""
    reported = next(
        entry for entry in cell.report["methods"][shipped.method.name]["per_run"] if entry["seed"] == shipped.run.seed
    )
    differ = _differences(dict(shipped.run.fitted), reported["fitted"]) + _differences(test_scores, reported["test"])
    if differ:
        raise ValueError(
            f"{shipped.method.name} rebuilt from {cell.directory}'s seed files differs from its report.json on "
            f"{differ}: the seed files are not the ones the report scored, or a method or scorer changed since. "
            f"Run the cell again, which rewrites report.json from its seed files without training, then package it."
        )


def _dataset_revision(arguments: argparse.Namespace) -> str | None:
    # A cell reads one dataset, so its revisions, when given, are one.
    revisions = arguments.data_revisions or []
    return revisions[0] if revisions else None


def _embedding_dtype(arguments: argparse.Namespace) -> str:
    # With no embed_dtype, embeddings are cached in the model dtype.
    name = arguments.embed_dtype or arguments.model_dtype
    if name not in EMBEDDING_DTYPES:
        raise ValueError(f"cannot name the embedding dtype {name!r}; expected one of {sorted(EMBEDDING_DTYPES)}.")
    return EMBEDDING_DTYPES[name]


def read_revisions(path: str | None) -> dict[str, str] | None:
    """The repository-to-revision map the embed stage wrote, or None without one."""
    if path is None:
        return None
    with open(path, encoding="utf-8") as handle:
        revisions = json.load(handle)
    if not isinstance(revisions, dict) or not all(
        isinstance(repository, str) and isinstance(revision, str) for repository, revision in revisions.items()
    ):
        raise ValueError(f"{path} must map each repository to the revision it was loaded at, both strings.")
    return revisions


def _projection_text(features: Mapping[str, Any]) -> str:
    """How a card says the branches are projected: even, or each to its own width and joined to the trunk width."""
    sizes = features.get("branch_sizes")
    if sizes is None:
        return f"{features['projection_size']} columns"
    return f"{', '.join(str(size) for size in sizes)} columns, joined to {features['projection_size']}"


def backbones(specs: Sequence[BranchSpec], revisions: Mapping[str, str] | None) -> list[dict[str, Any]]:
    """Each repository a branch's features were embedded with, and its revision.

    An SAE branch reads two: the ESM++ backbone and the Biohub codebook. A generated control
    (random vectors, one-hot) loads nothing: it records the generator that made its rows and no
    repository. A revisions file must name every repository; without one, each revision is
    recorded as unknown.
    """
    entries: list[dict[str, Any]] = []
    for spec in specs:
        if is_control_model(spec.model_name):
            entry = {"model_name": spec.model_name, "component": "generated_rows", "repository": None, "revision": None,
                     "generator": CONTROL_SCHEMA}
            if entry not in entries:
                entries.append(entry)
            continue
        if spec.profile:
            # The profile is read from a store whose manifest the package's `source` records, so it names no repository here.
            entry = {"model_name": spec.model_name, "component": "atlas_proteome_profile", "repository": None, "revision": None}
            if entry not in entries:
                entries.append(entry)
            continue
        if spec.sparse:
            selection = parse_sae_model_name(spec.model_name)
            components = [
                ("backbone", ESMC_SAE_COVERAGE[selection.backbone].model_path),
                ("sparse_autoencoder", selection.repository),
            ]
        else:
            components = [("backbone", all_presets_with_paths.get(spec.model_name))]
        for component, repository in components:
            entry = {"model_name": spec.model_name, "component": component, "repository": repository, "revision": None}
            if entry not in entries:
                entries.append(entry)

    loaded = [entry for entry in entries if entry["repository"] is not None]
    if revisions is not None:
        missing = sorted({entry["repository"] for entry in loaded if entry["repository"] not in revisions})
        if missing:
            raise ValueError(f"the revisions file names no revision for {missing}.")
        for entry in loaded:
            entry["revision"] = revisions[entry["repository"]]
    return entries


def features_record(
    cell: FinishedCell, arguments: argparse.Namespace, config: ProjectedProbeConfig, revisions: Mapping[str, str] | None
) -> dict[str, Any]:
    """How to build the rows the heads read, refused where the cell's records disagree."""
    report = cell.report
    specs = [
        BranchSpec(branch["model_name"], tuple(branch["pooling_types"]), branch["hidden_state_index"])
        for branch in report["branches"]
    ]
    width = sum(branch["width"] for branch in report["branches"])
    input_size = report["input_size"]
    # A pair row is two proteins' rows side by side.
    branch_input = width * (2 if report["ppi"] else 1)
    standardizer_width = None if cell.standardizer is None else cell.standardizer.mean.shape[0]
    if config.input_size != input_size or branch_input != input_size or standardizer_width not in (None, input_size):
        raise ValueError(
            f"{cell.directory}'s records disagree on the input width: branches {branch_input}, report {input_size}, "
            f"head {config.input_size}, standardizer {standardizer_width}."
        )

    branches = [
        {**branch, "spec": token} for token, branch in zip(cell.record["uncertainty"]["branch_specs"], report["branches"], strict=True)
    ]
    probe = cell.record["uncertainty"].get("probe_features")
    record = {
        "branches": branches,
        "width": width,
        "pair": report["ppi"],
        "pair_feature_mode": arguments.pair_feature_mode,
        "input_size": input_size,
        # A probe input reads the engine's pooled float32 streams, cut at 2,046 residues plus CLS and EOS.
        "max_length": PROBE_MAX_LENGTH if probe else arguments.max_length,
        "embedding_dtype": "float32" if probe else _embedding_dtype(arguments),
        "standardizer": None if cell.standardizer is None else STANDARDIZER_FILE,
        "input_clip": config.input_clip,
        "projection_size": config.projection_size,
        "head": report["head"],
        "backbones": backbones(specs, revisions),
    }
    if getattr(config, "branch_sizes", None) is not None:
        record["branch_sizes"] = list(config.branch_sizes)
    if probe:
        record["source"] = probe_provenance(probe, cell.record["uncertainty"]["branch_specs"])
        initialization = record["source"].get("random_init")
        if initialization:
            # A random-init control has no released checkpoint: its backbone is the architecture and the seed that made it.
            record["backbones"] = [
                {"model_name": name, "component": "random_init_backbone", "repository": None, "revision": None,
                 "generator": f"random init, seed {initialization['seed']}, {initialization['architecture']}"}
                for name in dict.fromkeys(spec.model_name for spec in specs)
            ]
    return record


def calibration_record(cell: FinishedCell, shipped: ShippedRun, levels: Sequence[float]) -> dict[str, Any]:
    predictive, calibrated = METHOD_RECIPES[shipped.method.name]
    observations = cell.observation_contract
    mc_dropout_samples = cell.record["uncertainty"]["mc_dropout_samples"] if predictive == "mc_dropout" else None
    record = {
        "scheme": PACKAGE_SCHEME,
        "method": shipped.method.name,
        "task_type": cell.kind,
        "label_type": cell.report["label_type"],
        "num_labels": cell.report["num_labels"],
        "head": cell.report["head"],
        "predictive": predictive,
        "calibration": calibration_kind(predictive, calibrated, cell.kind),
        "seeds": shipped.seeds,
        "seed_choice": shipped.seed_choice,
        "fitted": dict(shipped.run.fitted),
        "mc_dropout_samples": mc_dropout_samples,
        "rows": {"v_fit": int(cell.labels["v_fit"].shape[0]), "v_cal": int(cell.labels["v_cal"].shape[0])},
        "split_sha256": cell.split_sha256,
        "conformal": conformal_record(cell, shipped.run, levels),
        "answer_rule": None if partial_answer_unavailable(observations) else answer_rule_record(
            shipped.run.splits["v_cal"].probs, cell.labels["v_cal"], cell.kind, _at_least_one_label(cell)
        ),
    }
    if observations is not None:
        record["observation_contract"] = observations
    if "dataset_release" in cell.report:
        release = cell.report["dataset_release"]
        record["target_contract"] = {name: release[name] for name in ("manifest_sha256", "endpoint", "task_type", "units", "target_transform", "label_vocabulary")}
    return record


def shipped_correctness(
    cell: FinishedCell, shipped: ShippedRun, crossfit_dir: str, calibration: Mapping[str, Any]
) -> PackagedCorrectness:
    """The correctness model for a classification method that ships one calibrated head."""
    if cell.kind not in (*CLASSIFICATION_KINDS, "multilabel"):
        raise ValueError(f"a correctness model scores answers, which a {cell.kind} package does not give.")
    if any(mask is not None and not prediction_mask_is_complete(mask) for mask in cell.label_masks.values()):
        raise ValueError("partial truth cannot fit or evaluate whole-answer correctness; the M8 package requires complete labels")
    if calibration["predictive"] != "head" or calibration["calibration"] is None:
        raise ValueError(
            f"a correctness model reads one head's latent space beside its temperature-scaled confidence; "
            f"{shipped.method.name} ships {calibration['predictive']!r} with {calibration['calibration']!r}."
        )
    splits = ("v_cal", "test")
    return package_correctness(
        crossfit_dir,
        kind=cell.kind,
        num_labels=cell.report["num_labels"],
        shipped_seeds=shipped.seeds,
        probs={split: shipped.run.splits[split].probs for split in splits},
        labels={split: cell.labels[split] for split in splits},
        at_least_one_label=_at_least_one_label(cell),
    )


def decision_records(
    cell: FinishedCell, rule: Mapping[str, Any] | None, correctness: PackagedCorrectness | None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The package's novelty record and its decision policy; the policy is None where there is no answer rule.

    Only a correctness model reads the latent space, so only a package that ships one can tell a novel row from a familiar one.
    """
    if correctness is not None:
        return correctness.applicability, correctness.policy
    reason = "partial_calibration_truth" if partial_answer_unavailable(cell.observation_contract) else "no_latent_reference"
    applicability = unavailable_applicability(reason)
    return applicability, None if rule is None else decision_policy(rule, applicability)


def _at_least_one_label(cell: FinishedCell) -> bool | None:
    """For multilabel, whether every V_fit and V_cal row carries a label; test is never read."""
    if cell.kind != "multilabel":
        return None
    if partial_answer_unavailable(cell.observation_contract):
        return False
    return requires_a_label([cell.labels["v_fit"], cell.labels["v_cal"]])


def answer_outcome(
    cell: FinishedCell, shipped: ShippedRun, rule: Mapping[str, Any] | None, reliability: np.ndarray | None = None
) -> dict[str, Any] | None:
    """The answer rule on test at its default target: the share answered and their error.

    A rule ranked by a correctness model reads its test `reliability`, `(n,)`.
    """
    # reliability: (n,) or None
    if cell.kind == "regression" or (rule is None and not partial_answer_unavailable(cell.observation_contract)):
        return None
    probs = shipped.run.splits["test"].probs  # (n, c)
    at_least_one_label = False if rule is None else bool(rule["at_least_one_label"])
    answers = decode(probs, cell.kind, at_least_one_label)
    certain, possible = answer_truth_bounds(answers, cell.labels["test"], cell.kind, cell.label_masks["test"])  # (n,), (n,)
    answered = np.zeros(len(answers), dtype=bool)  # (n,), partial calibration cannot select answers
    if rule is not None:
        scores = reliability if rule.get("score") == "correctness" else answer_confidence(probs, answers, cell.kind, at_least_one_label)
        answered = select_answers(conformal_p_values(scores, rule), rule["default_target"])  # (n,)
    identified = certain == possible  # (n,), observed disagreement establishes an error
    record = {
        "target": None if rule is None else rule["default_target"],
        "accuracy": float(certain.mean()) if identified.all() else None,
        "answered": float(answered.mean()),
        "answered_error": float(1.0 - certain[answered].mean()) if answered.any() and identified[answered].all() else None,
    }
    if cell.label_masks["test"] is not None or rule is None:
        record.update({
            "truth_scope": "observed_targets", "certainly_correct": int(certain.sum()),
            "certainly_wrong": int((~possible).sum()), "unknown_correctness": int((~identified).sum()),
            "accuracy_bounds": [float(certain.mean()), float(possible.mean())],
            "answered_error_bounds": [float(1 - possible[answered].mean()), float(1 - certain[answered].mean())] if answered.any() else None,
            "selection_status": "unavailable_partial_calibration" if rule is None else "calibrated",
        })
    return record


def metrics_record(
    cell: FinishedCell,
    shipped: ShippedRun,
    levels: Sequence[float],
    rule: Mapping[str, Any] | None,
    correctness: PackagedCorrectness | None = None,
) -> dict[str, Any]:
    """The shipped run on test, scored by the report's own scorers, with the sets it ships, its answer
    rule, and its correctness model's scores beside the confidence's."""
    conformal = shipped_conformal_rows(conformal_rows(shipped.run, cell.kind, cell.labels, tuple(levels), cell.train_label_std, cell.label_masks))
    test_scores = json_ready(score_rows(cell.kind, shipped.run.splits["test"], cell.labels["test"], tuple(levels), None, conformal,
                                       label_mask=cell.label_masks["test"], canonical="dataset_release" in cell.report))
    primary = PRIMARY_METRICS[cell.kind]
    return {
        "method": shipped.method.name,
        "seeds": shipped.seeds,
        "split": "test",
        "levels": list(levels),
        "primary_metric": primary,
        "primary": test_scores.get(primary),
        **({"metric_contract": cell.report["metric_contract"]} if "metric_contract" in cell.report else {}),
        "test": test_scores,
        **({"test_label_support": label_support(cell.kind, cell.labels["test"], cell.label_masks["test"])} if cell.label_masks["test"] is not None else {}),
        "answer_rule": answer_outcome(cell, shipped, rule, None if correctness is None else correctness.reliability["test"]),
        **({"correctness": correctness.metrics} if correctness is not None else {}),
        **({"decision_policy": correctness.evidence} if correctness is not None and correctness.evidence else {}),
    }


def origin_record(cell: FinishedCell, shipped: ShippedRun, arguments: argparse.Namespace) -> dict[str, Any]:
    report = cell.report
    return {
        "cell_id": report["cell_id"],
        "experiment_id": report["experiment_id"],
        "tags": report["tags"],
        "dataset": {"name": report["data_name"], "revision": _dataset_revision(arguments), **({"release": report["dataset_release"]} if "dataset_release" in report else {})},
        "split": report["split"],
        "data_seed": report["data_seed"],
        "seeds": report["seeds"],
        "shipped_seeds": shipped.seeds,
        **({"observation_contract": cell.observation_contract} if cell.observation_contract is not None else {}),
        **({"metric_contract": report["metric_contract"]} if "metric_contract" in report else {}),
        **({"training": {
            str(record["seed"]): record["training"] for record in report["per_seed"] if record["seed"] in shipped.seeds
        }} if "training_recipe" in report else {}),
        **({"sampling": {"plan": report["sampling"], "seeds": {
            str(record["seed"]): record["sampling"] for record in report["per_seed"] if record["seed"] in shipped.seeds
        }}} if "sampling" in report else {}),
        "protify_revision": source_revision(PROTIFY_ROOT, include=("src",)).to_dict(),
    }


def _format(value: Any) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


# What a decision reads and does not read, said once for every package whose policy selects.
REQUEST_LIMIT = (
    "Selection reads the whole request: its threshold tightens with every row in it that the package is unsure of, so a row "
    "answered alone can abstain beside many such rows, and never the reverse. The target bounds the expected share of wrong "
    "answers among one request's answers, a statement a request of one row does not carry."
)


def held_out_decision_lines(metrics: Mapping[str, Any]) -> list[str]:
    """What the policy did with the test rows, decided as one request and again one row per request."""
    evidence = metrics.get("decision_policy", {})
    if "decisions" not in evidence:
        return []

    def error(outcome: Mapping[str, Any]) -> str:
        return "no row was answered" if outcome["answered_error"] is None else f"error {outcome['answered_error']:.4g} among them"

    together, alone = evidence["decisions"], evidence["single_row_decisions"]
    return [
        "",
        f"On the {together['rows']} test rows decided as one request the policy answered {together['answered']}, with {error(together)}, "
        f"abstained from {together['abstained']}, and left {together['not_assessed']} unassessed.",
        "",
        f"Decided one row per request, it answers {alone['answered']} of them, with {error(alone)}: a request of one row is "
        "answered whenever its p-value is at most the target, so this includes every row answered above.",
    ]


def applicability_lines(calibration: Mapping[str, Any], metrics: Mapping[str, Any]) -> list[str]:
    """The model card's section on novelty, abstention, and the limits of what the package's uncertainty says."""
    record, policy = calibration.get("applicability"), calibration.get("decision_policy")
    if record is None:
        return []

    lines = ["", "## Applicability and abstention", ""]
    if calibration["task_type"] == "regression":
        return lines + [
            "A regression package ships no novelty indicator and no answer rule, so `head.decide` leaves every row unassessed. "
            "Its conformal interval holds the truth for the stated share of proteins drawn like V_cal, on average over them. "
            "That is a marginal statement: it says nothing about one protein, and nothing about a protein unlike V_cal.",
        ]

    if record["status"] == "unavailable":
        lines.append(
            f"This package cannot say whether a row lies far from its training proteins. {APPLICABILITY_UNKNOWN_MESSAGES[record['reason_code']]}"
        )
        if policy is None:
            reason = "partial_calibration_truth" if record["reason_code"] == "partial_calibration_truth" else "no_answer_rule"
            return lines + ["", f"`head.decide` leaves every row unassessed. {NOT_ASSESSED_MESSAGES[reason]}"]
        return lines + [
            "",
            f"`head.decide` (policy `{policy['sha256']}`) therefore selects among every row of a request at the answer rule's "
            f"target, {policy['selection']['target']:.4g}, and marks each answer `applicability_unassessed`.",
            *held_out_decision_lines(metrics),
            "",
            REQUEST_LIMIT,
        ]

    strata = record["strata"]
    lines += [
        f"`head.novelty(prediction)` marks a row novel when its `{record['signal']}` exceeds {record['cutoff']:.4g}, the "
        f"{record['level']:.0%} conformal quantile of the {record['calibration_rows']} V_cal rows, so {record['novel_share']:.1%} "
        f"of V_cal is novel. {record['meaning']}",
        "",
        f"V_cal's rows by novelty and by band of the {strata['score']} the answer rule ranks by. A cell of fewer than "
        f"{strata['minimum_rows']} rows states no rate.",
        "",
        "| Novelty | Score band | V_cal rows | Accuracy | Error upper bound (95%) |",
        "|---|---|---|---|---|",
    ]
    for cell in strata["cells"]:
        accuracy = "not stated" if cell["accuracy"] is None else f"{cell['accuracy']:.4g}"
        bound = "not stated" if cell["error_upper_95"] is None else f"{cell['error_upper_95']:.4g}"
        lines.append(f"| {cell['applicability']} | {cell['lower']:g} to {cell['upper']:g} | {cell['rows']} | {accuracy} | {bound} |")

    lines += [
        "",
        f"`head.decide(prediction)` follows policy `{policy['scheme']}` (`{policy['sha256']}`). A novel row, or a row in a "
        f"cell of fewer than {strata['minimum_rows']} V_cal rows, abstains. Conformal selection at target "
        f"{policy['selection']['target']:.4g} then runs over the remaining rows of one request, and a row it does not select "
        f"abstains. Abstaining says the package will not vouch for the answer, not that the answer is wrong. No larger model "
        f"is consulted, and no percentage or confidence band is defined.",
    ]
    lines += held_out_decision_lines(metrics)
    lines += [
        "",
        "The false-answer-rate statement of conformal selection holds for rows exchangeable with V_cal. The two screens make "
        "the answered rows a subset chosen from features alone, so the statement is not restated for that subset; the "
        f"test error above and the table are the evidence. {REQUEST_LIMIT} Nothing here says how a protein far from every "
        "training protein will behave, and a row inside the cutoff is not thereby shown to be in distribution.",
    ]
    return lines


def model_card(
    calibration: Mapping[str, Any], features: Mapping[str, Any], metrics: Mapping[str, Any], origin: Mapping[str, Any]
) -> str:
    """The package's README, built from its records."""
    method, kind = calibration["method"], calibration["task_type"]
    dataset = origin["dataset"]
    lines = [
        f"# {origin['cell_id']}: {method}",
        "",
        f"A `{features['head']}` probe head for {dataset['name']} ({kind}, {calibration['num_labels']} outputs), "
        f"shipped as the uncertainty method `{method}`. It was packaged from cell `{origin['cell_id']}` of "
        f"experiment `{origin['experiment_id']}` by `python -m protify.uncertainty package`.",
        "",
        "## Inputs",
        "",
        f"Each protein is one row of {features['width']} columns: its pooled branches, joined in this order.",
        "",
        "| Branch | Model | Pooling | Hidden state | Width |",
        "|---|---|---|---|---|",
    ]
    for branch in features["branches"]:
        lines.append(
            f"| `{branch['spec']}` | {branch['model_name']} | {', '.join(branch['pooling_types'])} | "
            f"{branch['hidden_state_index']} | {branch['width']} |"
        )
    pair = (
        f" A pair row joins two proteins' rows by the `{features['pair_feature_mode']}` rule, {features['input_size']} columns in all."
        if features["pair"] else ""
    )
    standardized = (
        "The package standardizes the rows with the training split's mean and scale"
        if features["standardizer"] else "The cell fit no standardizer, so rows reach the head as given"
    )
    source = features.get("source")
    if source:
        lines += ["", f"Rows are the embedding engine's pooled streams of the {source['kind']} base `{source['base']}`, over all residues and the CLS and EOS rows."]
    lines += [
        "",
        f"Embeddings were cached at max_length {features['max_length']} in {features['embedding_dtype']}.{pair} "
        f"{standardized}, and the head clips each standardized value to plus or minus {features['input_clip']} "
        f"before projecting its branches to {_projection_text(features)}.",
        "",
        "| Model | Component | Repository | Revision |",
        "|---|---|---|---|",
    ]
    for entry in features["backbones"]:
        repository, revision = entry["repository"], entry["revision"]
        shown = repository or (f"generated by {entry['generator']}" if entry.get("generator") else "none")
        lines.append(
            f"| {entry['model_name']} | {entry['component']} | {shown} | "
            f"{revision or ('unknown' if repository else 'none')} |"
        )
    unknown = [entry["repository"] for entry in features["backbones"] if entry["repository"] and entry["revision"] is None]
    if unknown:
        lines += [
            "",
            f"Warning: the revision of {', '.join(unknown)} is unknown, because no revisions file was given. "
            f"Features embedded at another revision may not be the ones these heads were trained on.",
        ]

    choice = calibration["seed_choice"]
    seeds = ", ".join(str(seed) for seed in calibration["seeds"])
    shipped = (
        f"It ships every seed, {seeds}, and averages them."
        if choice["rule"] == "every_seed"
        else f"It ships seed {seeds}, the seed whose head scored the best V_fit {choice['primary_metric']}."
    )
    fitted = ", ".join(f"{name} = {value:.6g}" for name, value in calibration["fitted"].items())
    calibrated = (
        f"Its {calibration['calibration'].replace('_', ' ')} ({fitted}) was fitted on the {calibration['rows']['v_fit']} V_fit rows."
        if calibration["calibration"] else "It applies no calibration."
    )
    lines += ["", "## Calibration", "", f"{shipped} {calibrated}"]
    conformal = calibration["conformal"]
    observations = calibration.get("observation_contract")
    calibration_rows = calibration["rows"]["v_cal"] if observations is None else observations["support"]["v_cal"]["observed"]
    if observations is not None:
        lines += [
            "", "## Observed truth", "",
            "Temperature and conformal calibration use measured targets only. Unknown annotations are not negatives. "
            "Any stated coverage concerns the measured population under the conformal method's exchangeability assumptions; "
            "coverage for unobserved truth or distant protein families is not established.", "",
            "| Role | Observed targets | Unknown targets |", "|---|---|---|",
            *[f"| {role} | {support['observed']} | {support['unknown']} |" for role, support in observations["support"].items()],
            "", "Per-label support and insufficient-class flags are in `calibration.json`; test support is in `metrics.json`.",
        ]
        if partial_answer_unavailable(observations):
            lines += [
                "", "This development package exposes point predictions and observed-target scores. Whole-answer confidence, "
                "P(right), and answer-rule p-values are unavailable (`None`), and `head.select` abstains on every row. "
                "This is not a qualified public Oracle package; the M8 correctness requirement remains in force.",
            ]
    if conformal is None:
        lines += ["", "Multilabel predictions carry no conformal sets."]
    elif kind in CLASSIFICATION_KINDS:
        lines += [
            "",
            f"Conformal set thresholds were fitted on the {calibration_rows} observed V_cal rows. A class joins "
            f"a row's set when its LAC score, 1 - p, is at most the threshold.",
            "",
            "| Level | LAC |",
            "|---|---|",
        ]
        lines += [f"| {entry['level']} | {entry['lac']:.6g} |" for entry in conformal["quantiles"]]
    else:
        spread = "sigma + floor" if conformal["spread"] == "sigma" else "1, since the method has no sigma"
        lines += [
            "",
            f"Conformal interval quantiles were fitted on the {calibration_rows} observed V_cal rows. A row's "
            f"interval is its mean plus or minus the quantile times {spread}, with floor {conformal['spread_floor']:.6g}.",
            "",
            "| Level | Quantile |",
            "|---|---|",
        ]
        lines += [f"| {entry['level']} | {entry['quantile']:.6g} |" for entry in conformal["quantiles"]]

    rule = calibration.get("answer_rule")
    if rule is not None:
        answer = "its top class" if rule["answer"] == "top_class" else "its label set, every label at 0.5 or above"
        if rule["at_least_one_label"]:
            answer += (
                ", or its single most probable label when none reaches 0.5, since every V_fit and V_cal row "
                "carries a label"
            )
        outcome = metrics.get("answer_rule")
        tested = ""
        if outcome is not None:
            error = "error not estimable" if outcome["answered_error"] is None else f"error {outcome['answered_error']:.4g}"
            tested = f" On test it answered {outcome['answered']:.4g} of rows, with {error}."
        lines += [
            "",
            "## Answering",
            "",
            f"Each row's answer is {answer}. Its confidence comes from the calibrated head scores, "
            f"and `head.select(prediction, target)` answers the rows whose conformal p-values pass Benjamini-Hochberg "
            f"at the target, the expected share of wrong answers among those answered. The p-values count the "
            f"{rule['calibration_rows']} V_cal rows' wrong answers, V_cal's error rate is {rule['calibration_error']:.4g}, "
            f"and the default target is half of it, {rule['default_target']:.4g}.{tested}",
        ]
    if "correctness" in calibration:
        scores = metrics["correctness"]["test"]
        lines += [
            "",
            "## Reliability",
            "",
            "Each answer also carries its reliability: the probability that it is right, from a logistic regression "
            "on the confidence and on where the protein sits in the head's latent space, among the training proteins "
            "in `correctness/reference.npz` (`protify.uncertainty.correctness`). The regression was fitted on "
            "proteins each held out, cluster by cluster, from a head trained on the rest, so it learned how far "
            "the confidence holds for families the head never saw. The answer rule's p-values rank answers by "
            "reliability rather than confidence.",
            "",
            "| On test | Reliability | Confidence |",
            "|---|---|---|",
        ]
        lines += [
            f"| {label} | {scores[f'reliability_{name}']:.4g} | {scores[f'confidence_{name}']:.4g} |"
            for name, label in (("brier", "Brier"), ("nll", "NLL"), ("ece", "ECE"), ("aurc", "AURC"))
        ]
    lines += applicability_lines(calibration, metrics)

    lines += [
        "",
        "## Test metrics",
        "",
        f"The shipped predictor on the {origin['split']['test']} test rows, scored as the cell's report scores it. "
        f"The primary metric is {metrics['primary_metric']}.",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {name} | {_format(value)} |" for name, value in metrics["test"].items()]

    split = origin["split"]
    revision = origin["protify_revision"]
    call = "head.predict(features_a, features_b)" if features["pair"] else "head.predict(features)"
    partition_description = (
        "V_fit and V_cal are the release's frozen validation and calibration partitions."
        if "release" in dataset else f"V_fit and V_cal carved from validation by {split['source']} roles."
    )
    target = calibration.get("target_contract")
    if target is not None:
        transform = target["target_transform"]
        lines += [
            "", "## Target contract", "",
            (f"Endpoint `{target['endpoint']}`; model outputs use `{target['units']}`. The target transformation is "
             f"`{transform['name']}` from `{transform['source_units']}`, with inverse `{transform['inverse']}`."),
            (f"The frozen vocabulary has {len(target['label_vocabulary'])} entries, in `calibration.json` label order. "
             f"The release manifest SHA-256 is `{target['manifest_sha256']}`."),
            "Predictions remain in model target units; the package does not silently invert point estimates or uncertainty intervals.",
        ]
    lines += [
        "",
        "## Origin",
        "",
        f"Dataset {dataset['name']} at revision {dataset['revision'] or 'unrecorded'}: {split['train']} training, "
        f"{split['v_fit']} V_fit, {split['v_cal']} V_cal, and {split['test']} test rows. {partition_description} "
        f"Seeds {', '.join(str(seed) for seed in origin['seeds'])} trained with "
        f"data seed {origin['data_seed']}. Packaged by Protify {revision['kind']} {revision['value']}.",
        "",
        "## Use",
        "",
        "```python",
        "from protify.uncertainty.packaged import load_package",
        "",
        "head = load_package(\"<this directory>\")",
        f"prediction = {call}  # each (rows, {features['width']}) float32, the joined branches above",
        "```",
        "",
        "The heads give the cell's own outputs back to float rounding on the device type the cell predicted on. "
        "Protify runs CUDA matmuls in TF32, so a GPU and a CPU differ at that precision.",
        "",
    ]
    return "\n".join(lines)


def _write_package(
    staged: Path,
    cell: FinishedCell,
    shipped: ShippedRun,
    records: Mapping[str, Mapping[str, Any]],
    correctness: PackagedCorrectness | None = None,
) -> None:
    staged.mkdir()
    for seed in shipped.seeds:
        shutil.copytree(head_directory(cell.directory, seed), staged / HEADS_DIRECTORY / f"seed_{seed}")
    if cell.standardizer is not None:
        with open(staged / STANDARDIZER_FILE, "wb") as handle:
            np.savez(handle, mean=cell.standardizer.mean.numpy(), scale=cell.standardizer.scale.numpy())  # (d_in,) each
    if correctness is not None:
        write_correctness(staged / CORRECTNESS_DIRECTORY, correctness)
    for name, record in records.items():
        write_json(str(staged / name), record)
    card = model_card(records[CALIBRATION_RECORD], records[FEATURES_RECORD], records[METRICS_RECORD], records[ORIGIN_RECORD])
    (staged / MODEL_CARD).write_text(card, encoding="utf-8", newline="\n")


def build_package(
    cell_directory: str,
    method_name: str,
    output: str,
    *,
    levels: Sequence[float] | None = None,
    revisions_path: str | None = None,
    crossfit_dir: str | None = None,
) -> str:
    """Package one method of a finished `save_heads` cell into `output`, which must not exist yet.

    With `crossfit_dir`, the cell's `crossfit --correctness-only` output, the package also ships
    the correctness model fitted on it, and its answer rule ranks answers by that model.
    """
    target = Path(output).resolve()
    if target.exists():
        raise ValueError(f"{target} already exists; a package is written once, to a new directory.")

    revisions = read_revisions(revisions_path)
    cell = read_finished_cell(cell_directory)
    method = select_method(cell, method_name)
    shipped = shipped_run(cell, method)
    levels = report_levels(cell, method_name) if levels is None else _check_levels(levels)
    arguments = protify_arguments(cell.record["protify_args"])
    config = ProjectedProbeConfig.from_pretrained(head_directory(cell.directory, shipped.seeds[0]))

    calibration = calibration_record(cell, shipped, levels)
    check_packaged_arithmetic(cell, shipped, calibration)
    correctness = None
    if crossfit_dir is not None:
        correctness = shipped_correctness(cell, shipped, crossfit_dir, calibration)
        calibration["answer_rule"] = correctness.rule
        calibration["correctness"] = {"directory": CORRECTNESS_DIRECTORY, "scheme": correctness.record["scheme"]}
    calibration["applicability"], calibration["decision_policy"] = decision_records(cell, calibration["answer_rule"], correctness)
    metrics = metrics_record(cell, shipped, levels, calibration["answer_rule"], correctness)
    check_against_report(cell, shipped, metrics["test"])
    records = {
        CALIBRATION_RECORD: calibration,
        FEATURES_RECORD: features_record(cell, arguments, config, revisions),
        METRICS_RECORD: metrics,
        ORIGIN_RECORD: origin_record(cell, shipped, arguments),
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.", dir=target.parent) as staging:
        staged = Path(staging) / target.name
        _write_package(staged, cell, shipped, records, correctness)
        loaded = load_package(staged)
        if correctness is not None:
            metrics["correctness"]["check"] = check_shipped_correctness(loaded, correctness)
            write_json(str(staged / METRICS_RECORD), metrics)
        os.replace(staged, target)
    print(f"Packaged {method_name} of {cell.report['cell_id']}, seeds {shipped.seeds}, into {target}.")
    return str(target)
