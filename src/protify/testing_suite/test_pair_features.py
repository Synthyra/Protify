"""How two pooled protein vectors become one pair row.

The property that matters is symmetry. Under `minmax` the row a pair produces must not
depend on which protein the dataset listed first, because that is the whole reason the
mode exists: it removes the asymmetry that `--random_pair_flipping` was added to augment
around. A regression here would not raise, it would quietly reintroduce order dependence
and cost accuracy on every pair dataset.
"""

import numpy as np
import pytest
import torch


try:
    from src.protify.data.pair_features import (
        PAIR_FEATURE_MODES,
        check_pair_features_supported,
        pair_order_matters,
        pair_vector_halves,
    )
except ImportError:
    from protify.data.pair_features import (
        PAIR_FEATURE_MODES,
        check_pair_features_supported,
        pair_order_matters,
        pair_vector_halves,
    )


FEATURE_WIDTH = 6


def _joined(embedding_a, embedding_b, mode):
    """The pair row itself, joined the way every caller joins it."""
    half_a, half_b = pair_vector_halves(embedding_a, embedding_b, mode)  # two of (1, f)
    if isinstance(half_a, torch.Tensor):
        return torch.cat([half_a, half_b], dim=-1)  # (1, 2 * f)

    return np.concatenate([half_a, half_b], axis=-1)  # (1, 2 * f)


@pytest.fixture(params=["torch", "numpy"])
def pooled_pair(request):
    """One pair of pooled vectors, in each array library a caller uses."""
    values_a = [0.0, 3.0, -1.0, 7.5, 0.0, 2.0]
    values_b = [1.0, 2.0, -4.0, 7.5, 0.0, 0.0]
    if request.param == "torch":
        return (
            torch.tensor([values_a], dtype=torch.float32),  # (1, f)
            torch.tensor([values_b], dtype=torch.float32),  # (1, f)
        )

    return (
        np.array([values_a], dtype=np.float32),  # (1, f)
        np.array([values_b], dtype=np.float32),  # (1, f)
    )


def test_minmax_row_is_unchanged_by_swapping_the_pair(pooled_pair):
    embedding_a, embedding_b = pooled_pair
    forward = np.asarray(_joined(embedding_a, embedding_b, "minmax"))  # (1, 2 * f)
    reversed_order = np.asarray(_joined(embedding_b, embedding_a, "minmax"))  # (1, 2 * f)
    np.testing.assert_array_equal(forward, reversed_order)


def test_concat_row_does_change_when_the_pair_is_swapped(pooled_pair):
    embedding_a, embedding_b = pooled_pair
    forward = np.asarray(_joined(embedding_a, embedding_b, "concat"))  # (1, 2 * f)
    reversed_order = np.asarray(_joined(embedding_b, embedding_a, "concat"))  # (1, 2 * f)
    assert not np.array_equal(forward, reversed_order)


def test_minmax_halves_are_the_sorted_pair(pooled_pair):
    embedding_a, embedding_b = pooled_pair
    lower, upper = pair_vector_halves(embedding_a, embedding_b, "minmax")  # two of (1, f)
    stacked = np.stack([np.asarray(embedding_a), np.asarray(embedding_b)])  # (2, 1, f)
    np.testing.assert_array_equal(np.asarray(lower), stacked.min(axis=0))
    np.testing.assert_array_equal(np.asarray(upper), stacked.max(axis=0))


def test_concat_returns_the_two_proteins_untouched(pooled_pair):
    embedding_a, embedding_b = pooled_pair
    half_a, half_b = pair_vector_halves(embedding_a, embedding_b, "concat")
    assert half_a is embedding_a
    assert half_b is embedding_b


@pytest.mark.parametrize("mode", PAIR_FEATURE_MODES)
def test_every_mode_keeps_the_pair_row_width(pooled_pair, mode):
    embedding_a, embedding_b = pooled_pair
    assert np.asarray(_joined(embedding_a, embedding_b, mode)).shape == (1, 2 * FEATURE_WIDTH)


def test_only_concat_needs_the_order_augmentation():
    assert pair_order_matters("concat")
    assert not pair_order_matters("minmax")


def test_an_unknown_mode_is_rejected(pooled_pair):
    embedding_a, embedding_b = pooled_pair
    with pytest.raises(ValueError, match="pair_feature_mode"):
        pair_vector_halves(embedding_a, embedding_b, "average")


def test_minmax_rejects_two_proteins_of_different_length():
    """Per-residue matrices have a per-protein row count and cannot be sorted together."""
    matrix_a = torch.zeros(11, FEATURE_WIDTH)  # (l_a, f)
    matrix_b = torch.zeros(7, FEATURE_WIDTH)  # (l_b, f)
    with pytest.raises(ValueError, match="same width"):
        pair_vector_halves(matrix_a, matrix_b, "minmax")


def test_matrix_embeddings_are_refused_before_a_run_starts():
    check_pair_features_supported("minmax", full=False)
    check_pair_features_supported("concat", full=True)
    with pytest.raises(ValueError, match="matrix_embed"):
        check_pair_features_supported("minmax", full=True)
