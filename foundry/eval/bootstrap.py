"""Stratified bootstrap replicates, which hold every label's count fixed.

Resampling with replacement inside each label, rather than across all rows, keeps each
replicate's class balance equal to the data's, so a metric that depends on prevalence, such as
AUPRC, varies only through which rows are drawn. Scoring several models on the same replicate
indices gives a paired bootstrap of their differences.

Promoted from sae_xgboost_ppi's `gp_analysis._replicate`, which resampled the negatives and then
the positives of a binary test set.
"""

from __future__ import annotations

import numpy as np


def stratified_bootstrap_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one replicate: each label's rows resampled with replacement to its own count.

    Labels are grouped in `np.unique` order and the groups concatenated in that order, so for
    binary labels in {0, 1} a seed gives sae_xgboost_ppi's replicate exactly. Labels must be
    discrete, such as classes: a continuous target puts every row in a group of its own, and
    the replicate never resamples.
    """
    # labels: (n,)
    labels = np.asarray(labels)  # (n,)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(f"labels must be a non-empty vector, got shape {labels.shape}")

    # n_j rows hold the j-th distinct label.
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]  # (n_j,) each

    return np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])  # (n,)
