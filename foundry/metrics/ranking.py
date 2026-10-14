"""Binary ranking definitions shared by Oracle and retrieval evaluation.

Scores may be arbitrary finite real values; they need not be probabilities. Unknown
targets must be excluded by the caller's explicit observation mask before this function.
"""

from __future__ import annotations

import numpy as np

from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score


def binary_ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    """AP, trapezoidal PR area, and AUROC, with null for an unsupported denominator.

    AP and PR area require at least one positive. AUROC requires both classes. Empty inputs
    return three nulls. Equal scores form threshold groups; ROC ties count one half.
    """
    # labels, scores: (n,), one entry per observed row
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)  # (n,), (n,)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("binary rankings require aligned one-dimensional labels and scores")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("binary rankings require observed binary labels and finite scores")

    metrics: dict[str, float | None] = {"average_precision": None, "pr_auc_trapezoidal": None, "auroc": None}
    positive = labels == 1  # (n,)
    if positive.any():
        # One point per distinct score threshold, plus recall zero at precision one.
        precision, recall, _ = precision_recall_curve(labels, scores)  # (n_thresholds + 1,) each
        metrics["average_precision"] = float(average_precision_score(labels, scores))
        metrics["pr_auc_trapezoidal"] = float(auc(recall, precision))
    if positive.any() and (~positive).any():
        metrics["auroc"] = float(roc_auc_score(labels, scores))
    return metrics
