"""Versioned declared-endpoint metrics: explicit masks, fixed thresholds, null undefined scores."""

from __future__ import annotations

import numpy as np

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from scipy.stats import spearmanr

from foundry.metrics.ranking import binary_ranking_metrics
from ..metrics import validate_observed_labels


PRIMARY = {"binary": "mcc", "multiclass": "mcc", "multilabel": "f1", "regression": "spearman"}
RANK_METRICS = ("average_precision", "pr_auc_trapezoidal", "auroc")
# A term is scored in a split only when the split holds this many of its positives: an average precision over fewer is mostly noise.
MIN_SCORED_POSITIVES = 10
# What a cell may select its learning rate and seed on, V_fit's primary, by task kind. The first is `PRIMARY` and the default.
SELECTION_METRICS = {"binary": ("mcc",), "multiclass": ("mcc",), "multilabel": ("f1", "fmax"), "regression": ("spearman",)}
# CAFA's decision thresholds for Fmax, 0.01 to 0.99; a protein predicts a term at or above one.
FMAX_THRESHOLDS = np.arange(1, 100) / 100  # (q,)


def metric_contract(release: Mapping[str, Any]) -> dict[str, Any]:
    """Bind definitions and units to the endpoint's independently pinned manifest."""
    kind = release["task_type"]
    if kind not in PRIMARY:
        raise ValueError(f"unsupported declared metric task type {kind!r}")
    contract: dict[str, Any] = {
        "schema_version": 1, "scheme": "declared_oracle_metrics_v1", "task_type": kind,
        "primary_metric": PRIMARY[kind], "dataset_sha256": release["manifest_sha256"],
        "endpoint": release["endpoint"], "units": release["units"],
        "target_transform": release["target_transform"], "label_vocabulary": release["label_vocabulary"],
        "unknown_targets": "excluded_by_explicit_mask", "undefined": "null; never_zero_filled",
        "classification_decision": "argmax_first_class_on_ties",
        "multilabel_decision": "probability_strictly_greater_than_0.5",
        "multilabel_f1": "f1_is_micro_f1_at_the_fixed_threshold",
        "macro": "unweighted_mean_of_defined_label_metrics; denominator_reported_per_metric",
        "average_precision": "sum_of_recall_increments_times_precision_at_each_distinct_score",
        "pr_auc_trapezoidal": "trapezoidal_area_of_threshold_curve_including_recall_zero_precision_one",
        "auroc": "probability_positive_ranks_above_negative; ties_count_one_half",
        "calibration": {"ece": "15_equal_mass_bins; stable_score_order", "nll_log_base": "e",
                        "probability_clip": 1e-12, "brier": "binary_positive_class; multiclass_sum; multilabel_observed_entry_mean"},
        "regression_errors": "model_target_units; no_automatic_inverse_transform",
        "metric_units": ({"mae": release["units"], "rmse": release["units"], "mse": f"({release['units']})^2",
                          "spearman": "dimensionless", "r2": "dimensionless", "interval_width": release["units"]}
                         if kind == "regression" else {"discrimination_and_calibration": "dimensionless", "nll": "nats"}),
    }
    if kind == "multilabel":
        contract["scored_terms"] = f"terms_with_at_least_{MIN_SCORED_POSITIVES}_observed_positives_in_the_scored_split; a_term_below_is_never_imputed"
        contract["fmax"] = ("cafa_protein_centric_best_f_over_thresholds_0.01_to_0.99_at_or_above; precision_over_proteins_with_a_prediction; "
                            "recall_over_proteins_with_a_scored_positive; scored_terms_only")
        contract["scored_aupr"] = "average_precision_of_the_flattened_scored_entries_and_mean_over_scored_terms; scored_terms_only"
    return contract


def classification_mcc(labels: np.ndarray, predicted: np.ndarray, classes: int) -> float | None:
    """MCC from a multiclass confusion matrix; null when its denominator vanishes."""
    # labels, predicted: (m,) class indices of the observed rows
    confusion = np.bincount(labels * classes + predicted, minlength=classes * classes).reshape(classes, classes)  # (c, c)
    true_counts = confusion.sum(axis=1).astype(np.float64)  # (c,)
    predicted_counts = confusion.sum(axis=0).astype(np.float64)  # (c,)
    total = float(confusion.sum())
    numerator = float(np.trace(confusion)) * total - true_counts @ predicted_counts
    denominator = np.sqrt((total ** 2 - predicted_counts @ predicted_counts) * (total ** 2 - true_counts @ true_counts))
    return float(numerator / denominator) if denominator > 0 else None


def _threshold_metrics(labels: np.ndarray, scores: np.ndarray, decisions: np.ndarray | None = None) -> dict[str, float | None]:
    """F1, precision and recall from the fixed-threshold counts; a vanishing denominator stays null."""
    # labels, scores, decisions: (m,), each observed entry; decisions is None where the fixed threshold decides
    positive = labels == 1  # (m,), each observed entry
    predicted = scores > 0.5 if decisions is None else decisions  # (m,)
    tp = int(np.count_nonzero(positive & predicted))
    fp = int(np.count_nonzero(~positive & predicted))
    fn = int(np.count_nonzero(positive & ~predicted))
    return {
        "f1": 2.0 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def _binary_metrics(labels: np.ndarray, scores: np.ndarray, *, decisions: np.ndarray | None = None) -> dict[str, float | None]:
    """Binary rankings and fixed-threshold counts; empty and unsupported scores stay null."""
    # labels, scores, decisions: (m,), each observed entry; decisions is None where the fixed threshold decides
    return {**_threshold_metrics(labels, scores, decisions), **binary_ranking_metrics(labels, scores)}


def _macro(columns: list[dict[str, float | None]], names: tuple[str, ...]) -> dict[str, float | None]:
    macro_metrics = {}
    for name in names:
        defined = [column[name] for column in columns if column[name] is not None]
        macro_metrics[f"macro_{name}"] = float(np.mean(defined)) if defined else None
        macro_metrics[f"macro_{name}_defined_labels"] = float(len(defined))
    return macro_metrics


@dataclass(frozen=True, slots=True)
class FmaxCounts:
    """Each protein's counts at every Fmax threshold, which a weighted sum over proteins turns into Fmax."""

    predicted: np.ndarray  # (n, q) scored terms the protein predicts at each threshold
    hits: np.ndarray  # (n, q) of them that are observed positives
    positives: np.ndarray  # (n,) scored terms the protein holds as observed positives


def scored_term_columns(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The terms of a split that hold at least `MIN_SCORED_POSITIVES` observed positives, as a `(c,)` boolean."""
    # labels, mask: (n, c)
    return np.count_nonzero((labels == 1) & mask, axis=0) >= MIN_SCORED_POSITIVES  # (c,)


def fmax_counts(labels: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> FmaxCounts:
    """Per-protein prediction and hit counts over the scored terms, at each of the `q` thresholds."""
    # labels, probability, mask: (n, s), the scored terms only
    positive = (labels == 1) & mask  # (n, s)
    predicted = np.empty((labels.shape[0], FMAX_THRESHOLDS.size), dtype=np.int64)  # (n, q)
    hits = np.empty_like(predicted)  # (n, q)
    for index, threshold in enumerate(FMAX_THRESHOLDS):
        called = (probability >= threshold) & mask  # (n, s)
        predicted[:, index] = np.count_nonzero(called, axis=1)
        hits[:, index] = np.count_nonzero(called & positive, axis=1)
    return FmaxCounts(predicted, hits, np.count_nonzero(positive, axis=1))


def fmax_from_counts(counts: FmaxCounts, weights: np.ndarray | None = None) -> float | None:
    """CAFA's protein-centric Fmax; `weights` `(n,)` are how many times each protein is drawn, ones for the split itself.

    A protein with no scored positive is not in the benchmark. Precision averages over the proteins with a prediction at the
    threshold and recall over all benchmark proteins, so a threshold at which nothing is predicted has no precision and is skipped.
    """
    # weights: (n,) or None
    counted = counts.positives > 0  # (n,)
    draws = np.ones(counts.positives.shape) if weights is None else np.asarray(weights, dtype=np.float64)  # (n,)
    benchmark = np.where(counted, draws, 0.0)  # (n,)
    if benchmark.sum() == 0:
        return None
    safe_positives = np.maximum(counts.positives, 1)  # (n,)
    recall = benchmark @ (counts.hits / safe_positives[:, None]) / benchmark.sum()  # (q,)
    answered = np.where(counts.predicted > 0, benchmark[:, None], 0.0)  # (n, q), the benchmark proteins with a prediction
    precision_sum = (answered * counts.hits / np.maximum(counts.predicted, 1)).sum(axis=0)  # (q,)
    answered_total = answered.sum(axis=0)  # (q,)
    defined = answered_total > 0  # (q,)
    precision = precision_sum[defined] / answered_total[defined]  # (u,), u thresholds with a prediction
    if not defined.any():
        return None
    both = precision + recall[defined]  # (u,)
    best = np.where(both > 0, 2.0 * precision * recall[defined] / np.maximum(both, np.finfo(np.float64).tiny), 0.0)  # (u,)
    return float(best.max())


def selection_fmax(labels: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> float | None:
    """Fmax over every term with an observed positive, for choosing a learning rate and seed on V_fit.

    V_fit holds half the validation clusters, so few terms reach `MIN_SCORED_POSITIVES` there; the reported Fmax keeps that minimum.
    """
    # labels, probability, mask: (n, c)
    present = np.count_nonzero((labels == 1) & mask, axis=0) > 0  # (c,)
    if not present.any():
        return None
    return fmax_from_counts(fmax_counts(labels[:, present], probability[:, present], mask[:, present]))


def scored_term_metrics(labels: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    """Fmax and the micro and macro average precision over the terms with enough observed positives; null when no term has."""
    # labels, probability, mask: (n, c)
    scored = scored_term_columns(labels, mask)  # (c,)
    term_metrics: dict[str, float | None] = {"fmax": None, "scored_micro_aupr": None, "scored_macro_aupr": None, "scored_terms": float(scored.sum())}
    if not scored.any():
        return term_metrics
    kept_labels, kept_probability, kept_mask = labels[:, scored], probability[:, scored], mask[:, scored]  # (n, s) each; s scored terms
    term_metrics["fmax"] = fmax_from_counts(fmax_counts(kept_labels, kept_probability, kept_mask))
    term_metrics["scored_micro_aupr"] = binary_ranking_metrics(kept_labels[kept_mask], kept_probability[kept_mask])["average_precision"]
    columns = [binary_ranking_metrics(kept_labels[kept_mask[:, index], index], kept_probability[kept_mask[:, index], index])["average_precision"]
               for index in range(kept_labels.shape[1])]
    term_metrics["scored_macro_aupr"] = float(np.mean(columns))  # every scored term holds a positive, so each average precision is defined
    return term_metrics


def _regression(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    # truth, prediction: (m,), in declared model target units.
    if not np.isfinite(prediction).all():
        raise ValueError("observed predictions must be finite")
    with np.errstate(over="raise", invalid="raise"):
        residual = prediction - truth  # (m,)
        mse = float(np.mean(residual ** 2))
        total = float(np.sum((truth - truth.mean()) ** 2))
        return {
            "spearman": float(spearmanr(truth, prediction).statistic) if truth.size > 1 and np.ptp(truth) > 0 and np.ptp(prediction) > 0 else None,
            "r2": 1.0 - float(np.sum(residual ** 2)) / total if truth.size > 1 and total > 0 else None,
            "mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(residual))),
        }


def canonical_task_metrics(
    kind: str, labels: np.ndarray, probs: np.ndarray | None = None, mean: np.ndarray | None = None,
    label_mask: np.ndarray | None = None, *, primary_only: bool = False, selection: str | None = None,
) -> dict[str, float | None]:
    """Discrimination/errors for scalar (n,) or (n, 1), or multilabel (n, c) targets.

    Validation is the same either way. With `primary_only`, the result holds the value of `PRIMARY[kind]`, equal to
    the full result's, without the per-label and micro ranking metrics: those sort every observed score, which costs
    minutes at 9,951 labels and tens of thousands of rows, and only the report reads them. `selection` names another
    selectable primary of `SELECTION_METRICS[kind]` for `primary_only` to return; None keeps `PRIMARY[kind]`.
    """
    # labels, label_mask: (n,) or (n, c); probs: (n, c); mean: (n,) or (n, 1); each None where its kind does not read it
    if kind not in PRIMARY:
        raise ValueError(f"unknown canonical metric task type {kind!r}")
    if selection is not None and selection not in SELECTION_METRICS[kind]:
        raise ValueError(f"a {kind} task selects on one of {SELECTION_METRICS[kind]}, not {selection!r}")
    labels = np.asarray(labels)
    if labels.ndim not in (1, 2):
        raise ValueError("canonical targets must be a row vector or matrix")
    mask = np.ones(labels.shape, dtype=bool) if label_mask is None else np.asarray(label_mask)
    validate_observed_labels(labels, mask)
    if not mask.any():
        raise ValueError("canonical scoring requires observed targets")
    if kind == "regression":
        if mean is None or labels.size != labels.shape[0] or np.asarray(mean).shape not in {(labels.shape[0],), (labels.shape[0], 1)}:
            raise ValueError("regression predictions must match scalar targets")
        return _regression(labels[mask].astype(np.float64), np.asarray(mean).reshape(labels.shape)[mask].astype(np.float64))
    probability = np.asarray(probs, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[0] != labels.shape[0]:
        raise ValueError("classification probabilities require shape (rows, classes)")
    if kind == "multilabel":
        if probability.shape != labels.shape:
            raise ValueError("multilabel probabilities must match the target matrix")
        truth, selected = labels[mask], probability[mask]  # (m,)
    else:
        if labels.size != labels.shape[0] or probability.shape[1] < 2 or (kind == "binary" and probability.shape[1] != 2):
            raise ValueError("single-label target or probability shape is invalid")
        truth, selected = labels[mask], probability[mask.reshape(-1)]  # (m,), (m, c)
        if (truth < 0).any() or (truth >= selected.shape[1]).any() or not np.all(truth == np.floor(truth)):
            raise ValueError("class indices must lie in the declared output space")
        if not np.allclose(selected.sum(axis=1), 1.0, rtol=0, atol=1e-6):
            raise ValueError("single-label probabilities must sum to one")
    if not np.isfinite(selected).all() or (selected < 0).any() or (selected > 1).any():
        raise ValueError("observed probabilities must be finite and lie in [0, 1]")
    if kind == "multilabel":
        if not np.isin(truth, (0, 1)).all():
            raise ValueError("multilabel observed targets must be binary")
        if primary_only and selection == "fmax":
            return {"fmax": selection_fmax(labels, probability, mask)}
        if primary_only:
            return {"f1": _threshold_metrics(truth, selected)["f1"]}
        micro = _binary_metrics(truth, selected)
        columns = [_binary_metrics(labels[mask[:, index], index], probability[mask[:, index], index]) for index in range(labels.shape[1])]
        return {
            "f1": micro["f1"], **{f"micro_{name}": value for name, value in micro.items()}, **_macro(columns, ("f1", *RANK_METRICS)),
            **scored_term_metrics(labels, probability, mask),
        }
    truth = truth.astype(np.int64)
    predicted = selected.argmax(axis=1)  # (m,)
    base_metrics = {"mcc": classification_mcc(truth, predicted, selected.shape[1]), "accuracy": float(np.mean(predicted == truth))}
    if primary_only:
        return {"mcc": base_metrics["mcc"]}
    if kind == "binary":
        return {**base_metrics, **_binary_metrics(truth, selected[:, 1], decisions=predicted == 1)}
    columns = []
    for index in range(selected.shape[1]):
        column = _binary_metrics((truth == index).astype(np.int64), selected[:, index])
        # Multiclass decisions use argmax, not independent probability thresholds.
        tp = int(np.count_nonzero((truth == index) & (predicted == index)))
        denominator = int(np.count_nonzero(truth == index) + np.count_nonzero(predicted == index))
        column["f1"] = 2.0 * tp / denominator if denominator else None
        columns.append(column)
    return {**base_metrics, **_macro(columns, ("f1", *RANK_METRICS))}
