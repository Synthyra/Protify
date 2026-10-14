"""Selecting a subsample of rows for the feature-path selector.

The selector fits on a subsample, so the matrix builder is handed `keep_rows` rather than a
whole split. It used to reach those rows by streaming every embedding in the split and
discarding the ones it did not want, which on a large pair dataset meant deserializing
roughly 222000 blobs to keep 40000.

What has to hold for the row-addressed read that replaced it: the same rows, in the same
order, from both the feature store and the in-memory dict, for single-sequence and pair
datasets alike. Row order is the dangerous one. The caller pairs this matrix with
`labels[keep_rows]`, so a permutation here mislabels every row and raises nothing.
"""

import numpy as np
import pytest
import torch

from scipy import sparse

from protify.data.data_mixin import DataMixin
from protify.probes.trainers import TrainerMixin
from protify.testing_suite.embedding_store_builder import build_store


FEATURE_WIDTH = 8


class _EmbeddingArgs:
    def __init__(self, sql: bool) -> None:
        self.sql = sql


class _Reader(TrainerMixin, DataMixin):
    """The two row-reading methods under test, with the state they actually touch.

    Both mixins, because the readers live on TrainerMixin but reach the cache through
    DataMixin's `_select_from_store`, and the real pipeline class inherits the pair.
    Subclassing rather than rebinding, so the methods are exercised exactly as they are
    defined and a signature change fails the test rather than silently passing.
    """

    def __init__(self, sql: bool, pair_feature_mode: str = "concat") -> None:
        self.embedding_args = _EmbeddingArgs(sql)
        self._full = False
        self._pair_feature_mode = pair_feature_mode

    def stream_all(self, dataset, emb_dict, db_path, ppi, use_multi):
        """The pre-change behaviour: stream the split and filter by position."""
        return self._iter_vector_embeddings(
            dataset, emb_dict, db_path, ppi, use_multi, flip_pairs=False
        )


def _vector(index: int) -> torch.Tensor:
    """A distinct, order-revealing vector, so a permuted row cannot pass by coincidence."""
    return torch.arange(FEATURE_WIDTH, dtype=torch.float32) + index * 100.0  # (f,)


def _build(tmp_path, ppi: bool, n_rows: int):
    """One dataset plus both backends holding the same vectors."""
    sequences = [f"SEQ{index}" for index in range(n_rows * (2 if ppi else 1))]
    vectors = {name: _vector(position) for position, name in enumerate(sequences)}

    db_path = str(build_store(tmp_path, vectors))  # the feature store directory

    if ppi:
        dataset = {
            "SeqA": [sequences[2 * row] for row in range(n_rows)],
            "SeqB": [sequences[2 * row + 1] for row in range(n_rows)],
        }
    else:
        dataset = {"seqs": sequences}

    return dataset, vectors, db_path


def _expected(dataset, vectors, ppi: bool, keep_rows: np.ndarray) -> np.ndarray:
    """The matrix rows spelled out directly from the dataset, independent of the reader."""
    # keep_rows: (n_keep,)
    rows = []
    for row in keep_rows:
        if ppi:
            names = [dataset["SeqA"][int(row)], dataset["SeqB"][int(row)]]
        else:
            names = [dataset["seqs"][int(row)]]
        rows.append(torch.cat([vectors[name] for name in names]).numpy())  # (p * f,)
    return np.stack(rows)  # (n_keep, p * f)


@pytest.mark.parametrize("ppi", [False, True])
@pytest.mark.parametrize("sql", [True, False])
def test_selected_rows_match_a_full_scan_of_the_same_split(tmp_path, ppi, sql):
    """The row-addressed read must agree with streaming everything and filtering."""
    dataset, vectors, db_path = _build(tmp_path, ppi, n_rows=12)
    emb_dict = None if sql else vectors
    keep_rows = np.array([0, 3, 4, 9, 11])  # (n_keep,) sparse and non-contiguous

    reader = _Reader(sql=sql)
    selected = reader._split_feature_matrix(
        dataset, emb_dict, db_path, ppi, None, keep_rows
    ).toarray()  # (n_keep, p * f)

    wanted = set(int(row) for row in keep_rows)
    scanned = np.stack([
        embedding.reshape(-1).float().numpy()
        for index, embedding in enumerate(
            reader.stream_all(dataset, emb_dict, db_path, ppi, None)
        )
        if index in wanted
    ])  # (n_keep, p * f)

    assert selected.shape == scanned.shape
    assert np.array_equal(selected, scanned)


@pytest.mark.parametrize("ppi", [False, True])
@pytest.mark.parametrize("sql", [True, False])
def test_row_order_follows_keep_rows_not_storage_order(tmp_path, ppi, sql):
    """A permutation here would mislabel every row without raising, so pin it directly."""
    dataset, vectors, db_path = _build(tmp_path, ppi, n_rows=10)
    emb_dict = None if sql else vectors
    keep_rows = np.array([1, 2, 5, 8])  # (n_keep,)

    matrix = _Reader(sql=sql)._split_feature_matrix(
        dataset, emb_dict, db_path, ppi, None, keep_rows
    ).toarray()  # (n_keep, p * f)

    assert np.array_equal(matrix, _expected(dataset, vectors, ppi, keep_rows))


@pytest.mark.parametrize("sql", [True, False])
def test_a_pair_row_reads_both_of_its_sequences_unswapped(tmp_path, sql):
    """Selection must not see the training-time A/B augmentation.

    The streaming sibling takes `flip_pairs` and doubles a pair row when it is set. Rows
    chosen here line up one for one with labels, so the pair must stay in stored order and
    the row count must equal the number of rows kept.
    """
    dataset, vectors, db_path = _build(tmp_path, ppi=True, n_rows=6)
    emb_dict = None if sql else vectors
    keep_rows = np.array([0, 2])  # (n_keep,)

    matrix = _Reader(sql=sql)._split_feature_matrix(
        dataset, emb_dict, db_path, True, None, keep_rows
    ).toarray()  # (n_keep, 2 * f)

    assert matrix.shape == (2, 2 * FEATURE_WIDTH)
    for position, row in enumerate(keep_rows):
        first = vectors[dataset["SeqA"][int(row)]].numpy()  # (f,)
        second = vectors[dataset["SeqB"][int(row)]].numpy()  # (f,)
        assert np.array_equal(matrix[position, :FEATURE_WIDTH], first)
        assert np.array_equal(matrix[position, FEATURE_WIDTH:], second)


@pytest.mark.parametrize("sql", [True, False])
def test_only_the_kept_rows_are_read(tmp_path, sql):
    """The point of the change: cost scales with rows kept, not rows in the split."""
    dataset, vectors, db_path = _build(tmp_path, ppi=False, n_rows=50)
    emb_dict = None if sql else vectors
    keep_rows = np.array([7, 21, 40])  # (n_keep,)

    reader = _Reader(sql=sql)
    reads = []
    if sql:
        original = reader._select_from_store

        def counting(store_reader, sequence, cast_to_torch=True):
            reads.append(sequence)
            return original(store_reader, sequence, cast_to_torch=cast_to_torch)

        reader._select_from_store = counting
    else:
        emb_dict = _CountingDict(vectors, reads)

    reader._split_feature_matrix(dataset, emb_dict, db_path, False, None, keep_rows)

    assert reads == [dataset["seqs"][int(row)] for row in keep_rows]


class _CountingDict(dict):
    """Records every lookup, so the dict path can assert the same read count as SQL."""

    def __init__(self, source, reads) -> None:
        super().__init__(source)
        self._reads = reads

    def __getitem__(self, key):
        self._reads.append(key)
        return super().__getitem__(key)


def test_the_matrix_stays_csr(tmp_path):
    """The selector relies on CSR; a dense return would not fit a real codebook."""
    dataset, vectors, db_path = _build(tmp_path, ppi=False, n_rows=8)

    matrix = _Reader(sql=True)._split_feature_matrix(
        dataset, None, db_path, False, None, np.array([1, 4])
    )

    assert sparse.isspmatrix_csr(matrix)
    assert matrix.shape == (2, FEATURE_WIDTH)


@pytest.mark.parametrize("sql", [True, False])
def test_a_minmax_pair_row_is_sorted_and_order_free(tmp_path, sql):
    """The selector fits on the row the probe reads, so it has to join the same way.

    A selector fit on [A | B] while the probe reads [min | max] would rank columns of a
    representation nothing is ever trained on, and nothing would raise.
    """
    dataset, vectors, db_path = _build(tmp_path, ppi=True, n_rows=6)
    emb_dict = None if sql else vectors
    keep_rows = np.array([0, 2, 5])  # (n_keep,)

    matrix = _Reader(sql=sql, pair_feature_mode="minmax")._split_feature_matrix(
        dataset, emb_dict, db_path, True, None, keep_rows
    ).toarray()  # (n_keep, 2 * f)

    swapped = {"SeqA": dataset["SeqB"], "SeqB": dataset["SeqA"]}
    swapped_matrix = _Reader(sql=sql, pair_feature_mode="minmax")._split_feature_matrix(
        swapped, emb_dict, db_path, True, None, keep_rows
    ).toarray()  # (n_keep, 2 * f)
    assert np.array_equal(matrix, swapped_matrix)

    for position, row in enumerate(keep_rows):
        first = vectors[dataset["SeqA"][int(row)]].numpy()  # (f,)
        second = vectors[dataset["SeqB"][int(row)]].numpy()  # (f,)
        assert np.array_equal(matrix[position, :FEATURE_WIDTH], np.minimum(first, second))
        assert np.array_equal(matrix[position, FEATURE_WIDTH:], np.maximum(first, second))
