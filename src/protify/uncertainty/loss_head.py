"""A cross-fitted loss predictor: a second head that learns where the task head is wrong.

Training rows are split into `k` stratified folds seeded by the run seed. For each fold, a head
with the cell's settings trains on the other `k - 1` folds and scores the held fold, so every
training row gets a loss from a head that never trained on it. A `projected` regression head
then learns `log(loss + 1e-6)` from the rows' features, and its prediction on V_fit, V_cal, and
test is saved as `loss_prediction`. The task head trains before any of this and reads nothing
from it.

The per-row loss is the task's training loss: cross-entropy of the true class, binary
cross-entropy averaged over the labels, or the squared error.

Shapes: `n` rows, `c` classes or labels.
"""

from __future__ import annotations

import numpy as np

from collections.abc import Callable, Sequence
from typing import Any
from scipy.special import log_softmax

from .calibration_split import stratum_codes
from .methods import LOSS_OFFSET
from .metrics import CLASSIFICATION_KINDS


def stratified_folds(labels: Sequence[Any], label_type: str, folds: int, seed: int) -> np.ndarray:
    """Each row's fold, `0 .. folds - 1`, dealt in turn within each stratum in a seeded random order."""
    if folds < 2:
        raise ValueError(f"cross-fitting needs at least 2 folds; got {folds}.")
    codes = stratum_codes(labels, label_type)  # (n,)
    if codes.shape[0] < folds:
        raise ValueError(f"{codes.shape[0]} training rows cannot fill {folds} folds.")
    keys = np.random.default_rng(seed).random(codes.shape[0])  # (n,)
    order = np.lexsort((keys, codes))  # (n,): by stratum, randomly within each
    assignment = np.empty(codes.shape[0], dtype=np.int64)  # (n,)
    assignment[order] = np.arange(codes.shape[0]) % folds
    return assignment  # (n,)


def per_row_losses(kind: str, labels: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """Each row's training loss under the head's outputs, in float64."""
    # labels: (n,) or (n, c); outputs: (n, c) logits, or (n,) predictions for regression
    logits = np.asarray(outputs, dtype=np.float64)  # (n, c) or (n,)
    if kind in CLASSIFICATION_KINDS:
        rows = np.arange(logits.shape[0])  # (n,)
        return -log_softmax(logits, axis=1)[rows, np.asarray(labels, dtype=np.int64)]  # (n,)
    if kind == "multilabel":
        targets = np.asarray(labels, dtype=np.float64)  # (n, c)
        # log(1 + e^z) - y z is the binary cross-entropy of a logit z.
        return np.mean(np.logaddexp(0.0, logits) - targets * logits, axis=1)  # (n,)
    return (logits.reshape(-1) - np.asarray(labels, dtype=np.float64).reshape(-1)) ** 2  # (n,)


def log_loss_targets(losses: np.ndarray) -> np.ndarray:
    """The loss predictor's regression target, `log(loss + LOSS_OFFSET)`."""
    # losses: (n,)
    return np.log(np.asarray(losses, dtype=np.float64) + LOSS_OFFSET)  # (n,)


def cross_fitted_losses(
    folds: np.ndarray,
    fit: Callable[[np.ndarray], Any],
    score: Callable[[Any, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Every row's loss from a head trained without its fold.

    `fit(rows)` trains a head on the given training rows; `score(head, rows)` returns that
    head's loss on each given row. A fold's rows go to `score` only, never to the `fit` whose
    head scores them.
    """
    # folds: (n,) each row's fold
    # fit: (n - h,) rows in, score: (h,) rows in and (h,) losses out
    losses = np.full(folds.shape[0], np.nan)  # (n,)
    for fold in np.unique(folds):
        held = np.flatnonzero(folds == fold)  # (h,)
        kept = np.flatnonzero(folds != fold)  # (n - h,)
        head = fit(kept)
        losses[held] = score(head, held)
    if not np.all(np.isfinite(losses)):
        raise ValueError("a cross-fitted loss is missing or not finite.")
    return losses  # (n,)
