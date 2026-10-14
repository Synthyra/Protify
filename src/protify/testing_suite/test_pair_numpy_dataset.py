"""Pair ordering in the numpy dataset the estimator and scikit probes consume.

Interaction is symmetric, but concatenating A with B is not, so swapping the two proteins
is a training augmentation. Doing it on validation or test instead makes a reported score
depend on the random state, which these tests pin down.
"""

import random
import numpy as np
import pytest
import torch

from protify.data.data_mixin import DataMixin
from protify.testing_suite.embedding_store_builder import build_store_for


POOLING_TYPES = ['max']
HIDDEN_STATE_INDEX = -1

SEQ_A = 'AAAA'
SEQ_B = 'CCCC'


@pytest.fixture
def mixin(tmp_path):
    """A DataMixin whose only embedding source is a two-protein feature store."""
    from protify.embedder import EmbeddingArguments

    # One distinguishing value per protein, so a swap is visible in the concatenated row.
    embeddings = {
        SEQ_A: torch.tensor([1.0, 2.0]),  # (d,), d = 2
        SEQ_B: torch.tensor([30.0, 40.0]),  # (d,)
    }
    # The real config object names the store, so the fixture cannot drift from the cache
    # identity the reader builds.
    embedding_args = EmbeddingArguments(
        embedding_save_dir=str(tmp_path),
        embedding_pooling_types=POOLING_TYPES,
        embedding_hidden_state_index=HIDDEN_STATE_INDEX,
        matrix_embed=False,
    )
    build_store_for(embedding_args, 'test-model', embeddings)

    mixin = DataMixin()
    mixin._sql = False
    mixin._full = False
    mixin.embedding_args = embedding_args
    return mixin


def build(mixin, flip_train_pairs, pair_count=64):
    """Build train, validation, and test matrices from the same repeated pair."""
    seqs_a = [SEQ_A] * pair_count
    seqs_b = [SEQ_B] * pair_count
    return mixin.build_pair_vector_numpy_dataset_from_embeddings(
        'test-model',
        seqs_a, seqs_b,
        seqs_a, seqs_b,
        seqs_a, seqs_b,
        flip_train_pairs=flip_train_pairs,
    )  # three arrays, each (pair_count, 2 * d)


def test_evaluation_splits_keep_dataset_pair_order_when_flipping(mixin):
    random.seed(0)
    _, X_valid, X_test = build(mixin, flip_train_pairs=True)  # (n, 2 * d) each

    forward = np.array([1.0, 2.0, 30.0, 40.0])  # (2 * d,)
    assert np.array_equal(X_valid, np.tile(forward, (X_valid.shape[0], 1)))
    assert np.array_equal(X_test, np.tile(forward, (X_test.shape[0], 1)))


def test_evaluation_splits_are_independent_of_the_random_state(mixin):
    random.seed(0)
    _, first_valid, first_test = build(mixin, flip_train_pairs=True)
    random.seed(12345)
    _, second_valid, second_test = build(mixin, flip_train_pairs=True)

    assert np.array_equal(first_valid, second_valid)
    assert np.array_equal(first_test, second_test)


def test_training_pairs_swap_only_when_flipping_is_requested(mixin):
    random.seed(0)
    flipped_train, _, _ = build(mixin, flip_train_pairs=True)  # (n, 2 * d)
    reversed_row = np.array([30.0, 40.0, 1.0, 2.0])  # (2 * d,)
    swapped = (flipped_train == reversed_row).all(axis=1)  # (n,)
    assert swapped.any(), 'flipping requested but no training pair was swapped'
    assert not swapped.all(), 'flipping should leave some training pairs in dataset order'

    unflipped_train, _, _ = build(mixin, flip_train_pairs=False)
    forward = np.array([1.0, 2.0, 30.0, 40.0])  # (2 * d,)
    assert np.array_equal(unflipped_train, np.tile(forward, (unflipped_train.shape[0], 1)))


def test_flipping_defaults_to_off(mixin):
    random.seed(0)
    seqs_a, seqs_b = [SEQ_A] * 64, [SEQ_B] * 64
    X_train, _, _ = mixin.build_pair_vector_numpy_dataset_from_embeddings(
        'test-model', seqs_a, seqs_b, seqs_a, seqs_b, seqs_a, seqs_b,
    )  # (n, 2 * d)

    forward = np.array([1.0, 2.0, 30.0, 40.0])  # (2 * d,)
    assert np.array_equal(X_train, np.tile(forward, (X_train.shape[0], 1)))


def test_minmax_builds_the_same_row_from_either_ordering(mixin):
    """The estimator path's pair join, pinned where it is actually built.

    Under `minmax` a pair row is the two proteins sorted feature by feature, so listing B
    first cannot change it. This is what lets the augmentation be switched off rather than
    doubling the training split.
    """
    mixin._pair_feature_mode = 'minmax'
    seqs_a, seqs_b = [SEQ_A] * 8, [SEQ_B] * 8

    forward, _, _ = mixin.build_pair_vector_numpy_dataset_from_embeddings(
        'test-model', seqs_a, seqs_b, seqs_a, seqs_b, seqs_a, seqs_b,
    )  # (n, 2 * d)
    reversed_order, _, _ = mixin.build_pair_vector_numpy_dataset_from_embeddings(
        'test-model', seqs_b, seqs_a, seqs_b, seqs_a, seqs_b, seqs_a,
    )  # (n, 2 * d)

    sorted_row = np.array([1.0, 2.0, 30.0, 40.0])  # (2 * d,); min block then max block
    assert np.array_equal(forward, np.tile(sorted_row, (forward.shape[0], 1)))
    assert np.array_equal(forward, reversed_order)


def test_minmax_leaves_the_row_count_alone_when_flipping_is_requested(mixin):
    """`--random_pair_flipping` under a symmetric mode must not double or perturb anything."""
    mixin._pair_feature_mode = 'minmax'
    seqs_a, seqs_b = [SEQ_A] * 8, [SEQ_B] * 8

    random.seed(0)
    flipped, _, _ = mixin.build_pair_vector_numpy_dataset_from_embeddings(
        'test-model', seqs_a, seqs_b, seqs_a, seqs_b, seqs_a, seqs_b, flip_train_pairs=True,
    )  # (n, 2 * d)
    unflipped, _, _ = mixin.build_pair_vector_numpy_dataset_from_embeddings(
        'test-model', seqs_a, seqs_b, seqs_a, seqs_b, seqs_a, seqs_b, flip_train_pairs=False,
    )  # (n, 2 * d)

    assert flipped.shape == unflipped.shape
    assert np.array_equal(flipped, unflipped)


def test_a_mostly_zero_pair_split_comes_back_compressed(mixin, tmp_path):
    """The layout follows the data, so a sparse codebook does not have to be materialized."""
    from scipy import sparse
    from protify.embedder import EmbeddingArguments

    width = 40  # d
    sparse_a, sparse_b = torch.zeros(width), torch.zeros(width)  # each (d,)
    sparse_a[3] = 1.0
    sparse_b[11] = 2.0
    embedding_args = EmbeddingArguments(
        embedding_save_dir=str(tmp_path),
        embedding_pooling_types=POOLING_TYPES,
        embedding_hidden_state_index=HIDDEN_STATE_INDEX,
        matrix_embed=False,
    )
    build_store_for(embedding_args, 'sparse-model', {SEQ_A: sparse_a, SEQ_B: sparse_b})
    mixin.embedding_args = embedding_args

    seqs_a, seqs_b = [SEQ_A] * 8, [SEQ_B] * 8
    X_train, X_valid, X_test = mixin.build_pair_vector_numpy_dataset_from_embeddings(
        'sparse-model', seqs_a, seqs_b, seqs_a, seqs_b, seqs_a, seqs_b,
    )  # (n, 2 * d) each

    assert all(sparse.issparse(split) for split in (X_train, X_valid, X_test))
    expected = np.concatenate([sparse_a.numpy(), sparse_b.numpy()])  # (2 * d,)
    np.testing.assert_allclose(X_train.toarray(), np.tile(expected, (8, 1)))
