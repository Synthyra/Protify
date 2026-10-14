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


def balanced_subsamples(labels: np.ndarray, *, seed: int = 0, subsamples: int = 10) -> list[np.ndarray]:
    """Index sets of class-balanced views: the majority class downsampled without replacement to the minority's size.

    View i draws with seed ``seed + 11 * (i + 1)``, the seeds Atlas's balanced metrics have always used, so callers see the
    same views. An input already balanced is its own single view, in its own order.
    """
    # labels: (n,) of 0 and 1.
    labels = np.asarray(labels).ravel()
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)  # (n_pos,), (n_neg,)
    if not len(positive) or not len(negative) or len(positive) + len(negative) != len(labels):
        raise ValueError("balanced views need binary labels with both classes present")
    if len(positive) == len(negative):
        return [np.arange(len(labels))]
    if subsamples < 1:
        raise ValueError("balanced views need at least one subsample")
    majority, minority = (negative, positive) if len(negative) > len(positive) else (positive, negative)
    views = []
    for view in range(subsamples):
        generator = np.random.default_rng(seed + 11 * (view + 1))
        indices = np.concatenate([minority, generator.choice(majority, size=len(minority), replace=False)])  # (2 n_min,)
        generator.shuffle(indices)
        views.append(indices)
    return views


def balanced_mcc(labels: np.ndarray, scores: np.ndarray, *, seed: int = 0, subsamples: int = 10) -> dict[str, float]:
    """Balanced MCC: MCC at the threshold that maximizes its mean over the balanced views of `balanced_subsamples`.

    A score at or above the threshold predicts positive. Candidate thresholds are every distinct score in any view, swept
    in ascending order so a tie goes to the lowest, and a view predicted all one class scores 0. Counts come from sorted
    scores, one pass per view, so a sweep over millions of pairs is cheap. Returns the mean MCC, its population standard
    deviation across views at that threshold, the threshold, and the number of views.
    """
    # labels, scores: (n,).
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("balanced MCC needs one finite score per label")
    views = balanced_subsamples(labels, seed=seed, subsamples=subsamples)
    candidates = np.unique(np.concatenate([scores[view] for view in views]))  # (t,), ascending
    mcc = np.zeros((len(views), len(candidates)))  # (s, t)
    for row, view in enumerate(views):
        view_labels, view_scores = labels[view], scores[view]  # Each (m,).
        positive = np.sort(view_scores[view_labels == 1])  # (m_pos,)
        negative = np.sort(view_scores[view_labels == 0])  # (m_neg,)
        tp = (len(positive) - np.searchsorted(positive, candidates, side="left")).astype(np.float64)  # (t,)
        fp = (len(negative) - np.searchsorted(negative, candidates, side="left")).astype(np.float64)  # (t,)
        fn, tn = len(positive) - tp, len(negative) - fp  # Each (t,).
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))  # (t,)
        np.divide(tp * tn - fp * fn, denominator, out=mcc[row], where=denominator > 0)
    mean = mcc.mean(axis=0)  # (t,)
    best = int(np.argmax(mean))
    return {"balanced_mcc": float(mean[best]), "balanced_mcc_std": float(mcc[:, best].std()),
            "threshold": float(candidates[best]), "subsamples": float(len(views))}
