"""Scores for one method's predictions on one set of rows.

Task metrics reuse Protify's own definitions, so a workflow MCC or multilabel F1 is the number
a plain Protify run reports. Calibration, interval, and selective-prediction scores are added
here, the calibration error and the risk-coverage area through `foundry.metrics`. AURC gives
rows tied in uncertainty their mean loss first (`tie_averaged_losses`), so it never depends on
the order the rows are stored in.

Task kinds: `binary` and `multiclass` (single-label, softmax over `c` classes), `multilabel`
(`c` independent sigmoids), and `regression`. Shapes: `n` rows, `c` classes or labels, `u` tie
groups of equal uncertainty.
"""

from __future__ import annotations

import math
import numpy as np
import torch

from collections.abc import Callable
from scipy.special import ndtri
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

from foundry.metrics import area_under_risk_coverage, expected_calibration_error
from .metric_contracts import canonical_task_metrics, classification_mcc
from .prediction_arrays import DiskPredictionArray, prediction_row_blocks, transform_prediction_rows
from ..metrics import calculate_robust_pr_auc_multilabel, calculate_robust_roc_auc_multiclass, max_metrics, validate_observed_labels


TASK_KINDS = ("binary", "multiclass", "multilabel", "regression")
CLASSIFICATION_KINDS = ("binary", "multiclass")
PRIMARY_METRICS = {"binary": "mcc", "multiclass": "mcc", "multilabel": "f1", "regression": "spearman"}
ECE_BINS = 15
# Probabilities are clipped this far from 0 and 1 before a logarithm.
PROBABILITY_FLOOR = 1e-12
# Protify's robust AUC helpers return this when a score is undefined.
PROTIFY_UNDEFINED = -100.0


def task_kind(label_type: str, num_labels: int) -> str:
    """The task kind a Protify label type and label count describe."""
    if label_type == "singlelabel":
        return "binary" if num_labels == 2 else "multiclass"
    if label_type == "multilabel":
        return "multilabel"
    if label_type in ("regression", "sigmoid_regression"):
        return "regression"
    raise ValueError(f"the uncertainty workflow takes pooled single-label, multilabel, or regression tasks, not {label_type!r}.")


def _defined(value: float | None) -> float | None:
    """A finite float, or None for a score the rows cannot define."""
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value == PROTIFY_UNDEFINED:
        return None
    return value


def matthews_correlation(labels: np.ndarray, predicted: np.ndarray, classes: int) -> float:
    """Multiclass MCC from the confusion matrix, 0 where undefined: scikit-learn's value.

    Computed here rather than by `matthews_corrcoef` so that identity bins and bootstrap
    replicates holding a single class score without a warning apiece.
    """
    # labels: (n,), predicted: (n,)
    value = classification_mcc(labels, predicted, classes)
    return value if value is not None else 0.0


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    """Spearman's rho, or None when either side is constant or has fewer than two rows."""
    # first: (n,), second: (n,)
    if first.shape[0] < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return None
    return _defined(spearmanr(first, second).statistic)


def _observed_entries(kind: str, labels: np.ndarray, predictions: np.ndarray, label_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select target entries for multilabel tasks, or complete rows for scalar targets."""
    # labels, label_mask: (n,), (n, 1), or (n, c); predictions: (n,) or (n, c)
    validate_observed_labels(labels, label_mask)
    if not label_mask.any():
        raise ValueError("scoring requires at least one observed target")
    if kind == "multilabel":
        selected = predictions[label_mask]  # (m,)
    else:
        selected = predictions[label_mask.reshape(-1)]  # (m,) or (m, c)
    if not np.isfinite(selected).all():
        raise ValueError("observed predictions must be finite")
    return labels[label_mask], selected  # (m,), then (m,) or (m, c)


def task_metrics(kind: str, labels: np.ndarray, probs: np.ndarray | None = None, mean: np.ndarray | None = None, label_mask: np.ndarray | None = None, *, canonical: bool = False) -> dict[str, float | None]:
    """binary: MCC, AUROC; multiclass: MCC, accuracy; multilabel: F1, AUPRC; regression: Spearman, R2, RMSE."""
    # label_mask: (n,) or (n, c), true where the label is observed
    if canonical:
        return canonical_task_metrics(kind, labels, probs, mean, label_mask)
    if label_mask is not None and kind != "multilabel":
        labels, selected = _observed_entries(kind, labels, mean if kind == "regression" else probs, label_mask)
        if kind == "regression":
            mean = selected
        else:
            probs = selected
    if kind in CLASSIFICATION_KINDS:
        # labels: (n,); probs: (n, c)
        predicted = probs.argmax(axis=1)  # (n,)
        metrics = {"mcc": _defined(matthews_correlation(labels.astype(np.int64), predicted, probs.shape[1]))}
        if kind == "binary":
            metrics["auroc"] = _defined(calculate_robust_roc_auc_multiclass(labels, probs))
        else:
            metrics["accuracy"] = _defined(accuracy_score(labels, predicted))
        return metrics

    if kind == "multilabel":
        # labels: (n, c); probs: (n, c). Protify's F1 is the best over thresholds on flattened entries.
        truth, predicted = (labels.reshape(-1), probs.reshape(-1)) if label_mask is None else _observed_entries(kind, labels, probs, label_mask)  # (m,), (m,)
        f1, _, _, _ = max_metrics(torch.from_numpy(predicted), torch.from_numpy(truth))
        return {
            "f1": _defined(f1),
            "auprc": _defined(calculate_robust_pr_auc_multilabel(labels, probs, label_mask)),
        }

    # labels: (n,); mean: (n,)
    return {
        "spearman": rank_correlation(mean, labels),
        "r2": _defined(r2_score(labels, mean)) if labels.shape[0] >= 2 and np.ptp(labels) > 0 else None,
        "rmse": _defined(np.sqrt(np.mean((labels - mean) ** 2))),
    }


def _mapped_mean(score: Callable[..., float], labels: np.ndarray, *predictions: np.ndarray, label_mask: np.ndarray | None) -> float:
    """Weight block means by observed entries, never by rows with different label support."""
    # labels: (n,) or (n, c), predictions: (n,) or (n, c), label_mask: (n,) or (n, c)
    inputs = (labels, *predictions) + (() if label_mask is None else (label_mask,))
    total = 0
    weighted = []
    for blocks in prediction_row_blocks(*inputs):
        if label_mask is None:
            count = blocks[0].size
        else:
            validate_observed_labels(blocks[0], blocks[-1])
            count = int(blocks[-1].sum())
        if count:
            weighted.append(score(*blocks) * count)
            total += count
    if not total:
        raise ValueError("scoring requires at least one observed target")
    return math.fsum(weighted) / total


def brier_score(kind: str, labels: np.ndarray, probs: np.ndarray, label_mask: np.ndarray | None = None) -> float:
    """Binary: `(p_1 - y)^2`. Multiclass: `sum_k (p_k - [y = k])^2`. Multilabel: per entry. Row mean."""
    # labels: (n,) or (n, c), probs: (n, c), label_mask: (n,) or (n, c)
    if any(isinstance(values, DiskPredictionArray) for values in (labels, probs, label_mask)):
        return _mapped_mean(lambda *blocks: brier_score(kind, *blocks), labels, probs, label_mask=label_mask)
    if label_mask is not None:
        labels, probs = _observed_entries(kind, labels, probs, label_mask)
    if kind == "binary":
        return float(np.mean((probs[:, 1] - labels) ** 2))
    if kind == "multiclass":
        onehot = np.eye(probs.shape[1])[labels]  # (n, c)
        return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    return float(np.mean((probs - labels) ** 2))


def negative_log_likelihood(kind: str, labels: np.ndarray, probs: np.ndarray, label_mask: np.ndarray | None = None) -> float:
    """Cross-entropy of the true class, or binary cross-entropy averaged over every label."""
    # labels: (n,) or (n, c), probs: (n, c), label_mask: (n,) or (n, c)
    if any(isinstance(values, DiskPredictionArray) for values in (labels, probs, label_mask)):
        return _mapped_mean(lambda *blocks: negative_log_likelihood(kind, *blocks), labels, probs, label_mask=label_mask)
    if label_mask is not None:
        labels, probs = _observed_entries(kind, labels, probs, label_mask)
    if kind in CLASSIFICATION_KINDS:
        clipped = np.clip(probs, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)  # (n, c)
        return float(-np.mean(np.log(clipped[np.arange(labels.shape[0]), labels])))
    # Clipped in float64: in float32, 1 - PROBABILITY_FLOOR rounds to 1.0, and log(1 - p) of a
    # probability that reached 1.0 is infinite.
    clipped = np.clip(np.asarray(probs, dtype=np.float64), PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)  # (n, c)
    return float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))


def calibration_error(kind: str, labels: np.ndarray, probs: np.ndarray, bins: int = ECE_BINS, label_mask: np.ndarray | None = None) -> float:
    """Equal-mass ECE: of p_1 for binary, of the top label for multiclass, per entry for multilabel."""
    # labels: (n,) or (n, c), probs: (n, c), label_mask: (n,) or (n, c)
    if label_mask is not None:
        labels, probs = _observed_entries(kind, labels, probs, label_mask)
    if kind == "binary":
        return expected_calibration_error(labels, probs[:, 1], bins)
    if kind == "multiclass":
        return expected_calibration_error(probs.argmax(axis=1) == labels, probs.max(axis=1), bins)
    return expected_calibration_error(labels.reshape(-1), probs.reshape(-1), bins)


def gaussian_nll(labels: np.ndarray, mean: np.ndarray, variance: np.ndarray, label_mask: np.ndarray | None = None) -> float:
    """Mean negative log density of the labels under `N(mean, variance)`."""
    # labels: (n,), mean: (n,), variance: (n,), label_mask: (n,)
    if any(isinstance(values, DiskPredictionArray) for values in (labels, mean, variance, label_mask)):
        return _mapped_mean(gaussian_nll, labels, mean, variance, label_mask=label_mask)
    if label_mask is not None:
        labels, mean = _observed_entries("regression", labels, mean, label_mask)
        variance = variance[label_mask.reshape(-1)]  # (m,)
    variance = np.maximum(variance, PROBABILITY_FLOOR)  # (n,)
    return float(np.mean(0.5 * np.log(2.0 * np.pi * variance) + (labels - mean) ** 2 / (2.0 * variance)))


def gaussian_interval(labels: np.ndarray, mean: np.ndarray, variance: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Per row, whether the central `level` Gaussian interval holds the label, and its width."""
    # labels: (n,), mean: (n,), variance: (n,)
    half_width = ndtri(0.5 + level / 2.0) * np.sqrt(np.maximum(variance, 0.0))  # (n,)
    return np.abs(labels - mean) <= half_width, 2.0 * half_width  # (n,), (n,)


def row_losses(kind: str, labels: np.ndarray, probs: np.ndarray | None = None, mean: np.ndarray | None = None, label_mask: np.ndarray | None = None) -> np.ndarray:
    """The per-row loss selective prediction ranks: misclassification, Hamming error, or |error|."""
    # labels: (n,) or (n, c), probs: (n, c), mean: (n,), label_mask: (n,) or (n, c)
    arrays = {name: values for name, values in {"labels": labels, "probs": probs, "mean": mean, "label_mask": label_mask}.items() if values is not None}
    if any(isinstance(values, DiskPredictionArray) for values in arrays.values()):
        return transform_prediction_rows(
            lambda *blocks: {"losses": row_losses(kind, **dict(zip(arrays, blocks, strict=True)))}, *arrays.values(),
        )["losses"]  # (n,), mapped
    if label_mask is not None:
        validate_observed_labels(labels, label_mask)
        prediction = mean if kind == "regression" else probs
        observed_prediction = prediction[label_mask] if kind == "multilabel" else prediction[label_mask.reshape(-1)]
        if not np.isfinite(observed_prediction).all():
            raise ValueError("observed predictions must be finite")
        if kind == "multilabel":
            counts = label_mask.sum(axis=1)  # (n,)
            errors = (((probs > 0.5) != (labels > 0.5)) & label_mask).sum(axis=1)  # (n,)
            return np.divide(errors, counts, out=np.full(len(labels), np.nan), where=counts > 0)  # (n,)
        observed = label_mask.reshape(-1)  # (n,)
        losses = np.full(labels.size, np.nan)  # (n,), undefined on unknown targets
        losses[observed] = row_losses(kind, labels.reshape(-1)[observed], None if probs is None else probs[observed], None if mean is None else mean[observed])
        return losses  # (n,)
    if kind in CLASSIFICATION_KINDS:
        return (probs.argmax(axis=1) != labels).astype(np.float64)  # (n,)
    if kind == "multilabel":
        return np.mean((probs > 0.5) != (labels > 0.5), axis=1).astype(np.float64)  # (n,)
    return np.abs(labels - mean).astype(np.float64)  # (n,)


def row_errors(kind: str, losses: np.ndarray) -> np.ndarray:
    """Which rows count as errors for error detection.

    Classification: a wrong argmax. Multilabel: any wrong label at 0.5. Regression: an absolute
    error above the median of these rows, so half the rows are errors by construction.
    """
    # losses: (n,)
    if kind == "regression":
        return losses > np.median(losses)  # (n,)
    return losses > 0.0  # (n,)


def tie_averaged_losses(losses: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    """Each row's loss replaced by the mean loss of the rows that share its uncertainty.

    `foundry.metrics.area_under_risk_coverage` ranks rows by ascending uncertainty and keeps tied
    rows in input order, so tied rows would enter the risk-coverage curve in the order the file
    stores them: a plain regression head's temperature map gives every row one sigma, and its
    AURC would follow row order alone. AURC is linear in the losses taken in ranked order, and a
    uniformly random order of a tie group puts the group's mean loss at each of its positions in
    expectation, so the AURC of these losses is the expectation over every order of the tied
    rows. The mean is taken as an offset from the group's first loss, so a group whose losses are
    equal, a lone row among them, keeps its loss exactly. NaN never equals itself, so rows of NaN
    uncertainty stay untied.
    """
    # losses: (n,); uncertainty: (n,)
    order = np.argsort(uncertainty, kind="stable")  # (n,)
    sorted_uncertainty = uncertainty[order]  # (n,)
    starts = np.ones(order.shape[0], dtype=bool)  # (n,): where each tie group begins in ranked order
    starts[1:] = sorted_uncertainty[1:] != sorted_uncertainty[:-1]
    group = np.cumsum(starts) - 1  # (n,): each ranked row's tie group
    sorted_losses = losses[order].astype(np.float64)  # (n,)
    first = sorted_losses[starts]  # (u,)
    mean_offset = np.bincount(group, weights=sorted_losses - first[group]) / np.bincount(group)  # (u,)
    averaged = np.empty_like(sorted_losses)  # (n,)
    averaged[order] = (first + mean_offset)[group]
    return averaged  # (n,)


def selective_metrics(kind: str, losses: np.ndarray, uncertainty: np.ndarray) -> dict[str, float | None]:
    """AURC over every coverage, error-detection AUROC, and the mean uncertainty.

    AURC is the expectation over every order of rows tied in uncertainty; see `tie_averaged_losses`.
    """
    # losses: (n,), uncertainty: (n,)
    metrics: dict[str, float | None] = {"mean_uncertainty": _defined(np.mean(uncertainty))}
    if losses.shape[0] >= 2:
        metrics["aurc"] = _defined(area_under_risk_coverage(tie_averaged_losses(losses, uncertainty), uncertainty))
    else:
        metrics["aurc"] = None
    errors = row_errors(kind, losses)  # (n,)
    if errors.all() or not errors.any():
        metrics["error_auroc"] = None
    else:
        metrics["error_auroc"] = _defined(roc_auc_score(errors, uncertainty))
    if kind == "regression":
        metrics["uncertainty_error_spearman"] = rank_correlation(uncertainty, losses)
    return metrics


def primary_value(kind: str, metrics: dict[str, float | None]) -> float | None:
    return metrics.get(PRIMARY_METRICS[kind])
