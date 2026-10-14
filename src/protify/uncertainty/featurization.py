"""Question 1's scores: a cell's seed-mean primary metric on its test rows and on any draw of them.

A draw is how many times each test row was drawn, so one call scores a thousand bootstrap
replicates. Each seed is scored as `metrics.task_metrics` scores it, from the head's own
outputs: MCC of the argmax of the softmax, Protify's multilabel F1 of the sigmoid, or Spearman's
rho of the regression mean. A seed whose score is undefined on a draw is left out of that draw's
mean, as `report.summarize` leaves out a seed whose score is None.

Shapes: `B` draws, `n` test rows, `c` classes or labels, `S` seeds.
"""

from __future__ import annotations

import numpy as np

from collections.abc import Sequence

from .methods import probabilities
from .predictions import SplitPredictions
from .weighted_metrics import weighted_mcc, weighted_protify_f1, weighted_spearman


# Draws scored at once, which bounds the memory a (draws, rows) weight matrix takes.
DRAW_BLOCK = 250


def mean_over_seeds(per_seed: np.ndarray) -> np.ndarray:
    """The mean over seeds of each draw's defined scores, NaN where no seed is defined."""
    # per_seed: (S, B)
    defined = np.isfinite(per_seed)  # (S, B)
    count = defined.sum(axis=0)  # (B,)
    total = np.where(defined, per_seed, 0.0).sum(axis=0)  # (B,)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)  # (B,)


class PrimaryScorer:
    """One cell's seed-mean primary metric on any draw of its test rows."""

    def __init__(self, kind: str, seeds: Sequence[SplitPredictions]) -> None:
        self.kind = kind
        self.labels = seeds[0].labels  # (n,) or (n, c)
        self.rows = self.labels.shape[0]
        if kind == "regression":
            self.members = [split.outputs for split in seeds]  # S of (n,)
        elif kind == "multilabel":
            self.members = [probabilities(split.outputs, kind) for split in seeds]  # S of (n, c)
        else:
            # task_metrics takes the argmax of the probabilities, which can tie where the logits do not.
            self.members = [probabilities(split.outputs, kind).argmax(axis=1) for split in seeds]  # S of (n,)
            self.classes = seeds[0].outputs.shape[1]

    def _member_draws(self, member: np.ndarray, counts: np.ndarray) -> np.ndarray:
        # member: (n,) or (n, c); counts: (B, n)
        if self.kind == "regression":
            return weighted_spearman(counts, member, self.labels)  # (B,)
        if self.kind == "multilabel":
            return weighted_protify_f1(counts, member, self.labels)  # (B,)
        return weighted_mcc(counts, self.labels.astype(np.int64), member, self.classes)  # (B,)

    def score_draws(self, counts: np.ndarray) -> np.ndarray:
        """The seed-mean primary metric of each draw."""
        # counts: (B, n)
        blocks = []
        for start in range(0, counts.shape[0], DRAW_BLOCK):
            block = counts[start:start + DRAW_BLOCK]  # (b, n)
            blocks.append(mean_over_seeds(np.stack([self._member_draws(member, block) for member in self.members])))  # (b,)
        return np.concatenate(blocks)  # (B,)

    def score(self, rows: np.ndarray) -> float:
        """The seed-mean primary metric of the given row indices, repeats allowed."""
        # rows: (m,)
        counts = np.bincount(rows, minlength=self.rows)[None, :]  # (1, n)
        return float(self.score_draws(counts)[0])
