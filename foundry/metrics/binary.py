"""Binary classification summaries from a confusion matrix's counts and from ranked scores.

Promoted from Atlas's `utils/metrics.py`, which synth's `synth/utilities/metrics.py` copied
token for token. Atlas's trainer reduces MCC from counts gathered across processes, and both
report top-k accuracy beside the thresholded metrics.
"""

from __future__ import annotations

import math
import numpy as np

from collections.abc import Iterable


def compute_metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """Precision, recall, accuracy, F1, and MCC of the positive class, each 0.0 where undefined."""
    precision = 0.0
    recall = 0.0
    accuracy = 0.0
    f1 = 0.0
    mcc = 0.0

    if tp + fp > 0:
        precision = tp / (tp + fp)
    if tp + fn > 0:
        recall = tp / (tp + fn)

    total = tp + fp + fn + tn
    if total > 0:
        accuracy = (tp + tn) / total
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    mcc_denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if mcc_denominator > 0:
        mcc = (tp * tn - fp * fn) / math.sqrt(float(mcc_denominator))

    return {
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "mcc": float(mcc),
    }


def calculate_topk_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    k_values: Iterable[int] = (10, 100, 1000),
) -> dict[str, float]:
    """Top-k accuracy: the positive share of the `k` highest scores and the negative share of the `k` lowest.

    Rows with a non-finite label or score are left out, and `k` is capped at the rows left. Each
    `k` gives `topk_pred_pos_acc_{k}` and `topk_pred_neg_acc_{k}`; no rows give no keys.
    """
    # labels, probs: (...), n elements each, read flattened
    flat_labels = np.asarray(labels).astype(float).flatten()  # (n,)
    flat_probs = np.asarray(probs).astype(float).flatten()  # (n,)

    finite = np.isfinite(flat_labels) & np.isfinite(flat_probs)  # (n,)
    flat_labels = flat_labels[finite].astype(int)  # (m,)
    flat_probs = flat_probs[finite]  # (m,)

    n_valid = int(flat_labels.shape[0])  # m
    order = np.argsort(flat_probs)  # (m,), ascending score
    metrics: dict[str, float] = {}

    for k in k_values:
        k_kept = min(k, n_valid)
        if k_kept <= 0:
            continue
        top = order[-k_kept:]  # (k_kept,)
        bottom = order[:k_kept]  # (k_kept,)
        metrics[f"topk_pred_pos_acc_{k}"] = float((flat_labels[top] == 1).mean())
        metrics[f"topk_pred_neg_acc_{k}"] = float((flat_labels[bottom] == 0).mean())

    return metrics
