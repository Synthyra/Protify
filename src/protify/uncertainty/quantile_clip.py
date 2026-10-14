"""The `projected_qclip` head's input clip: each standardized feature held to its training quantiles.

The projected head clamps every standardized feature to plus or minus 10. `projected_qclip`
instead clips feature `j` to `[q_lo_j, q_hi_j]`, its `cell.CLIP_QUANTILES` (0.1% and 99.9%)
over the training rows after standardization, and turns the probe's own clamp off. The bounds
are fitted on the training rows alone, as the standardizer is, and saved with the standardizer
record, so a package that reads the record applies them too.

Pair rows are read in stored order, as the probe reads them outside training, even where
`random_pair_flipping` swaps them in training.

Shapes: `n` training rows, `b` rows in a batch, `d` joined features, `w` a block of features.
"""

from __future__ import annotations

import numpy as np
import torch

from collections.abc import Sequence
from torch.utils.data import DataLoader, Dataset

from ..data.dataset_classes import EmbeddingStandardizer


# Features whose quantiles are taken at once, bounding the copy np.quantile makes.
QUANTILE_BLOCK = 1024


class ClippedStandardizer(EmbeddingStandardizer):
    """A standardizer that clips each standardized feature to its training bounds."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
        # mean: (d,), scale: (d,)
        super().__init__(mean, scale)
        # lower, upper: (d,)
        if lower.shape != mean.shape or upper.shape != mean.shape:
            raise ValueError(f"clip bounds {lower.shape} and {upper.shape} must match the standardizer's {mean.shape}.")
        if np.any(lower > upper):
            raise ValueError("every lower clip bound must be at most its upper bound.")
        self.lower = torch.from_numpy(lower.astype(np.float32))  # (d,)
        self.upper = torch.from_numpy(upper.astype(np.float32))  # (d,)

    def transform_tensor(self, embedding: torch.Tensor) -> torch.Tensor:
        # embedding: (d,) or (1, d)
        shape = embedding.shape
        standardized = super().transform_tensor(embedding).reshape(1, -1)  # (1, d)
        return torch.minimum(torch.maximum(standardized, self.lower), self.upper).reshape(shape)  # (d,) or (1, d), the input shape

    def transform_numpy(self, features: np.ndarray) -> np.ndarray:
        # features: (b, d)
        standardized = super().transform_numpy(features)  # (b, d)
        return np.clip(standardized, self.lower.numpy(), self.upper.numpy()).astype(np.float32)  # (b, d)

    def select_features(self, indices: np.ndarray) -> ClippedStandardizer:
        # indices: (s,)
        return ClippedStandardizer(
            self.mean.numpy()[indices],  # (s,)
            self.scale.numpy()[indices],  # (s,)
            self.lower.numpy()[indices],  # (s,)
            self.upper.numpy()[indices],  # (s,)
        )


def standardized_rows(dataset: Dataset, collator: object, batch_size: int = 4096) -> np.ndarray:
    """Every row of `dataset` as `collator` hands it to the probe, stacked `(n, d)` in float32."""
    blocks = [batch["embeddings"].numpy() for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)]
    return np.concatenate(blocks).astype(np.float32, copy=False)  # (n, d)


def fit_clipped_standardizer(
    standardizer: EmbeddingStandardizer, train_rows: np.ndarray, quantiles: Sequence[float]
) -> ClippedStandardizer:
    """`standardizer` with each feature's `quantiles` over `train_rows`, already standardized, as its bounds."""
    # train_rows: (n, d), the training rows after `standardizer`
    if train_rows.ndim != 2 or train_rows.shape[0] == 0:
        raise ValueError(f"the quantile clip needs a non-empty (rows, features) matrix; got shape {train_rows.shape}.")
    low, high = quantiles
    lower = np.empty(train_rows.shape[1], dtype=np.float32)  # (d,)
    upper = np.empty(train_rows.shape[1], dtype=np.float32)  # (d,)
    for start in range(0, train_rows.shape[1], QUANTILE_BLOCK):
        block = train_rows[:, start:start + QUANTILE_BLOCK]  # (n, w)
        bounds = np.quantile(block, [low, high], axis=0)  # (2, w)
        lower[start:start + block.shape[1]] = bounds[0]
        upper[start:start + block.shape[1]] = bounds[1]
    return ClippedStandardizer(standardizer.mean.numpy(), standardizer.scale.numpy(), lower, upper)
