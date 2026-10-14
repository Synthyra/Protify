"""Deterministic binary majority downsampling within an existing partition.

Shapes: n source observations, k retained observations.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BalancedSelection:
    indices: np.ndarray  # (k,), ascending positions in the original partition
    counts_before: tuple[int, int]
    counts_after: tuple[int, int]


def balanced_indices(observations: Sequence[str], labels: Sequence[float], masks: Sequence[bool],
                     *, split: str, seed: int) -> BalancedSelection:
    """Retain every minority row and hash-select the same number from the majority.

    Selection depends on immutable observation identity, split and seed, never feature
    values or model predictions. Output keeps the original relative row order.
    """
    if type(seed) is not int or seed < 0 or not split:
        raise ValueError("selection requires a nonnegative integer seed and a named partition")
    if not observations or len(set(observations)) != len(observations):
        raise ValueError("selection requires unique nonempty observation identities")
    if not len(observations) == len(labels) == len(masks):
        raise ValueError("selection columns have different lengths")
    targets = np.asarray(labels)  # (n,)
    observed = np.asarray(masks)  # (n,)
    if targets.ndim != 1 or observed.dtype != np.bool_ or observed.shape != targets.shape or not observed.all():
        raise ValueError("selection requires fully observed scalar binary labels")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("selection requires binary labels with their existing orientation")
    classes = [np.flatnonzero(targets == label) for label in (0, 1)]  # two (n_c,) arrays
    counts = (len(classes[0]), len(classes[1]))
    minority = min(counts)
    if minority == 0:
        raise ValueError("both classes must be present before balancing")
    chosen: list[int] = []
    for rows in classes:
        ranked = sorted(map(int, rows), key=lambda row: (
            hashlib.sha256(json.dumps((seed, split, observations[row]), separators=(",", ":")).encode()).digest(), observations[row]))
        chosen.extend(ranked[:minority])
    indices = np.asarray(sorted(chosen), dtype=np.int64)  # (k,)
    return BalancedSelection(indices, counts, (minority, minority))
