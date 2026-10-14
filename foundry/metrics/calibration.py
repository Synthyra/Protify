"""Calibration error of predicted probabilities against the outcomes they predict.

Promoted from sae_xgboost_ppi's `gp_analysis.py`. Its private copy stays until its Modal image
carries foundry, and its suite holds the two equal.
"""

from __future__ import annotations

import numpy as np


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, bins: int) -> float:
    """Equal-mass expected calibration error of `probs` against `labels`.

    Rows are sorted by probability, ties kept in input order, and cut into `bins` groups whose
    sizes differ by at most one row. Each group adds its share of the rows times the gap between
    its observed rate and its mean probability. More bins than rows leaves each row alone.

    Binary: `labels` in {0, 1} and `probs` the probability of 1. Top-label ECE of a multiclass
    model is the same call on whether each argmax is correct and on its probability:
    `expected_calibration_error(probs.argmax(1) == labels, probs.max(1), bins)`.
    """
    # labels, probs: (n,)
    labels = np.asarray(labels)  # (n,)
    probs = np.asarray(probs)  # (n,)
    if labels.ndim != 1 or labels.shape != probs.shape or labels.size == 0:
        raise ValueError(f"labels and probs must be non-empty vectors of one length, got {labels.shape} and {probs.shape}")
    if bins < 1:
        raise ValueError(f"bins must be positive, got {bins}")

    order = np.argsort(probs, kind="mergesort")  # (n,)
    total = 0.0
    # n_b rows per bin, n // bins or one more.
    for rows in np.array_split(order, bins):  # (n_b,)
        if len(rows):
            total += len(rows) * abs(labels[rows].mean() - probs[rows].mean())

    return float(total / len(labels))
