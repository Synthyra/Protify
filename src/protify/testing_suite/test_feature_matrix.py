"""Stacking pooled rows into the matrix an estimator probe fits.

Two things have to hold. The compressed and dense stacks must contain the same numbers,
because the layout is chosen from the data and a run should not depend on which branch it
took. And the layout must be pinned across splits: a train matrix that came back sparse
while validation came back dense would hand an estimator two different readings of the
same zero.
"""

import numpy as np
import pytest

from scipy import sparse


try:
    from src.protify.data import feature_matrix
    from src.protify.data.feature_matrix import (
        LAYOUT_PROBE_ROWS,
        live_column_indices,
        matrix_density,
        stack_pooled_rows,
        take_columns,
    )
except ImportError:
    from protify.data import feature_matrix
    from protify.data.feature_matrix import (
        LAYOUT_PROBE_ROWS,
        live_column_indices,
        matrix_density,
        stack_pooled_rows,
        take_columns,
    )


FEATURE_WIDTH = 12


def _rows(matrix: np.ndarray):
    """The (1, f) rows a builder would hand the stacker, one at a time."""
    # matrix: (n, f)
    return (matrix[index: index + 1] for index in range(matrix.shape[0]))


def _mostly_zero(n_rows: int, active_per_row: int = 2) -> np.ndarray:
    """A matrix sparse enough that the layout probe should choose CSR."""
    generator = np.random.default_rng(0)
    matrix = np.zeros((n_rows, FEATURE_WIDTH), dtype=np.float32)  # (n, f)
    for index in range(n_rows):
        columns = generator.choice(FEATURE_WIDTH, size=active_per_row, replace=False)
        matrix[index, columns] = generator.random(active_per_row).astype(np.float32) + 0.1
    return matrix  # (n, f)


def test_a_sparse_split_stacks_compressed_and_keeps_its_values():
    matrix = _mostly_zero(LAYOUT_PROBE_ROWS + 37)  # (n, f)
    stacked, layout = stack_pooled_rows(_rows(matrix))  # (n, f)
    assert layout is True
    assert sparse.issparse(stacked)
    np.testing.assert_allclose(stacked.toarray(), matrix)


def test_a_dense_split_stacks_dense_and_keeps_its_values():
    matrix = np.random.default_rng(1).random((LAYOUT_PROBE_ROWS + 5, FEATURE_WIDTH)).astype(np.float32)
    stacked, layout = stack_pooled_rows(_rows(matrix))  # (n, f)
    assert layout is False
    assert isinstance(stacked, np.ndarray)
    np.testing.assert_allclose(stacked, matrix)


@pytest.mark.parametrize("pinned", [True, False])
def test_a_pinned_layout_is_used_whatever_the_data_looks_like(pinned):
    """A later split reuses the first split's layout rather than measuring its own."""
    matrix = _mostly_zero(8)  # (n, f); sparse, so an unpinned call would choose CSR
    stacked, layout = stack_pooled_rows(_rows(matrix), sparse_layout=pinned)
    assert layout is pinned
    assert sparse.issparse(stacked) is pinned
    np.testing.assert_allclose(
        stacked.toarray() if pinned else stacked, matrix
    )


def test_chunked_compression_reassembles_the_split_in_order(monkeypatch):
    """More rows than one staging chunk holds, so the vstack path is what is checked."""
    # A chunk of one element forces one block per row, the smallest chunking there is.
    monkeypatch.setattr(feature_matrix, "SPARSE_CHUNK_ELEMENTS", 1)
    matrix = _mostly_zero(LAYOUT_PROBE_ROWS + 11)  # (n, f)
    stacked, layout = stack_pooled_rows(_rows(matrix))
    assert layout is True
    np.testing.assert_allclose(stacked.toarray(), matrix)


def test_zero_rows_is_an_error_not_an_empty_matrix():
    with pytest.raises(ValueError, match="zero rows"):
        stack_pooled_rows(iter([]))


def test_density_is_reported_the_same_way_for_both_layouts():
    matrix = _mostly_zero(LAYOUT_PROBE_ROWS + 3)  # (n, f)
    dense, _ = stack_pooled_rows(_rows(matrix), sparse_layout=False)
    compressed, _ = stack_pooled_rows(_rows(matrix), sparse_layout=True)
    assert matrix_density(dense) == pytest.approx(matrix_density(compressed))


@pytest.mark.parametrize("as_sparse", [True, False])
def test_columns_that_are_zero_for_every_row_are_the_ones_dropped(as_sparse):
    matrix = np.zeros((5, FEATURE_WIDTH), dtype=np.float32)  # (n, f)
    matrix[0, 2] = 1.0
    matrix[3, 7] = -2.0
    matrix[4, 7] = 0.5
    features = sparse.csr_matrix(matrix) if as_sparse else matrix

    live = live_column_indices(features)  # (s,)
    np.testing.assert_array_equal(live, np.array([2, 7]))

    narrowed = take_columns(features, live)  # (n, s)
    assert sparse.issparse(narrowed) is as_sparse
    np.testing.assert_allclose(
        narrowed.toarray() if as_sparse else narrowed, matrix[:, [2, 7]]
    )


def test_a_column_that_is_only_ever_negative_counts_as_live():
    """Live means non-zero, not positive. Dense embeddings are signed."""
    matrix = np.zeros((3, FEATURE_WIDTH), dtype=np.float32)  # (n, f)
    matrix[:, 4] = -1.0
    np.testing.assert_array_equal(live_column_indices(matrix), np.array([4]))
    np.testing.assert_array_equal(live_column_indices(sparse.csr_matrix(matrix)), np.array([4]))


def test_an_explicitly_stored_zero_is_not_a_live_column():
    """CSR can carry a stored zero, and a column holding only those is still dead."""
    # Built from the triplets directly, so column 6 is stored holding a zero rather than
    # dropped the way an assignment into a dense array would drop it.
    matrix = sparse.coo_matrix(
        (np.array([1.0, 0.0], dtype=np.float32), (np.array([0, 1]), np.array([5, 6]))),
        shape=(3, FEATURE_WIDTH),
    ).tocsr()  # (n, f)
    np.testing.assert_array_equal(live_column_indices(matrix), np.array([5]))


def test_live_columns_agree_between_the_two_layouts():
    matrix = _mostly_zero(LAYOUT_PROBE_ROWS + 9)  # (n, f)
    np.testing.assert_array_equal(
        live_column_indices(matrix), live_column_indices(sparse.csr_matrix(matrix))
    )
