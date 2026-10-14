"""How the two proteins of a pair become one feature row.

Whether A binds B does not depend on which protein the dataset happened to list first,
but concatenation does. A probe reading [A | B] therefore has to learn the same relation
twice, once in each column block, and `--random_pair_flipping` exists to show it both
orderings during training.

`minmax` removes the asymmetry instead of augmenting around it. Sorting each feature
across the two partners gives [min(a, b) | max(a, b)], which a swap leaves unchanged, so
the two orderings are literally the same row. For an axis-aligned learner the difference
is structural rather than cosmetic: "both partners express concept j" is a single split
on min(a, b)[j], where under concatenation it needs one split per block on two columns
that column subsampling may never place in the same tree.

Both modes return two blocks of the pooled width, so a pair row stays 2 * d wide and no
shape downstream changes.
"""

from __future__ import annotations

import numpy as np
import torch

from typing import TypeVar


PAIR_FEATURE_MODES = ("concat", "minmax")
DEFAULT_PAIR_FEATURE_MODE = "concat"

# One pooled vector, at whatever leading rank the caller holds it in.
PairVector = TypeVar("PairVector", np.ndarray, torch.Tensor)


def check_pair_feature_mode(mode: str) -> str:
    if mode not in PAIR_FEATURE_MODES:
        raise ValueError(
            f"pair_feature_mode must be one of {PAIR_FEATURE_MODES}; got {mode!r}."
        )
    return mode


def pair_order_matters(mode: str) -> bool:
    """Whether swapping the two proteins changes the row this mode builds.

    False makes `--random_pair_flipping` a no-op, since the augmentation would emit the
    row it already emitted.
    """
    return check_pair_feature_mode(mode) == "concat"


def pair_vector_halves(
        embedding_a: PairVector,
        embedding_b: PairVector,
        mode: str,
    ) -> tuple[PairVector, PairVector]:
    """The two blocks one pair row is built from, in the order they are joined.

    Returned as a pair rather than already joined, because each caller joins at a
    different point: the neural path in its collator, the estimator path per row as it
    reads the cache, and the selector fit per sampled row.
    """
    # embedding_a, embedding_b: both (..., d) -> two of (..., d)
    if check_pair_feature_mode(mode) == "concat":
        return embedding_a, embedding_b

    if embedding_a.shape != embedding_b.shape:
        raise ValueError(
            f"minmax pair features need both proteins pooled to the same width; got "
            f"{tuple(embedding_a.shape)} and {tuple(embedding_b.shape)}. Per-residue "
            f"matrix embeddings have a per-protein length and cannot be sorted this way."
        )

    if isinstance(embedding_a, torch.Tensor):
        return (
            torch.minimum(embedding_a, embedding_b),  # (..., d)
            torch.maximum(embedding_a, embedding_b),  # (..., d)
        )

    return (
        np.minimum(embedding_a, embedding_b),  # (..., d)
        np.maximum(embedding_a, embedding_b),  # (..., d)
    )


def check_pair_features_supported(mode: str, full: bool) -> None:
    """Reject a pair mode the stored embedding layout cannot express."""
    if check_pair_feature_mode(mode) == "minmax" and full:
        raise ValueError(
            "--pair_feature_mode minmax sorts two pooled vectors feature by feature and "
            "needs one vector per protein, but --matrix_embed stores one row per residue "
            "and the two proteins have different lengths. Pool the embeddings, or use "
            "--pair_feature_mode concat."
        )
