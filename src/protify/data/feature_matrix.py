"""Stacking pooled embedding rows into the matrix an estimator probe fits.

A max-pooled sparse autoencoder row is mostly zero, and at the published codebook widths
the dense stack does not exist to be built: 163192 training pairs of 262144 float32
columns is 171 GB, which is where a full-width pair run dies. Compressed sparse rows hold
only the features that fired, and XGBoost, LightGBM, and scikit-learn's forests all fit
CSR directly.

Dense embeddings have nothing to drop, so CSR would cost roughly twice their memory and
buy nothing. The layout is therefore measured from a prefix of the rows rather than
assumed from the model, and pinned across the splits of one run so that train,
validation, and test are stored the same way.

XGBoost reads an entry a CSR does not store as missing rather than as zero, and learns a
default branch direction for it. For a max-pooled codebook that is the reading we want:
an absent feature is a concept the protein never expressed. This is the case the sparsity
aware split finding was written for, and where its speedup comes from.
"""

from __future__ import annotations

import numpy as np

from collections.abc import Iterable
from itertools import chain, islice
from scipy import sparse

from ..console_output import print_message


FeatureMatrix = np.ndarray | sparse.csr_matrix

# Below this fraction of non-zero entries, CSR costs less memory than a dense array and
# hands the tree learners their sparse path. A float32 dense entry is 4 bytes against 8
# for a stored CSR entry plus its column index, so the break-even is 50%; half that
# leaves room for the prefix estimate to be optimistic.
SPARSE_LAYOUT_MAX_DENSITY = 0.25

# Rows read before the layout is decided. Enough to average over sequence length, which
# is what drives how many codebook features a max pool unions.
LAYOUT_PROBE_ROWS = 256

# Dense elements staged before a chunk is compressed, about 64 MB of float32. Converting
# per row would call scipy once per protein; converting the whole split at once would
# need the dense matrix this module exists to avoid.
SPARSE_CHUNK_ELEMENTS = 1 << 24


def stack_pooled_rows(
        rows: Iterable[np.ndarray],
        sparse_layout: bool | None = None,
    ) -> tuple[FeatureMatrix, bool]:
    """Stack (1, d) pooled rows into one (n, d) matrix, and report the layout used.

    Pass the `sparse_layout` a previous split returned to pin this split to it. Leave it
    None to measure from the first `LAYOUT_PROBE_ROWS` rows.
    """
    # rows: (1, d) per row
    row_iterator = iter(rows)
    probe = list(islice(row_iterator, LAYOUT_PROBE_ROWS))  # up to 256 of (1, d)
    if not probe:
        raise ValueError("Cannot stack a feature matrix from zero rows.")

    if sparse_layout is None:
        stored = sum(int(np.count_nonzero(row)) for row in probe)
        total = sum(int(row.size) for row in probe)
        density = stored / total
        sparse_layout = density < SPARSE_LAYOUT_MAX_DENSITY
        print_message(
            f"Feature matrix layout: {'sparse CSR' if sparse_layout else 'dense'} "
            f"({100 * density:.1f}% non-zero over the first {len(probe)} rows)"
        )

    if not sparse_layout:
        return np.concatenate(probe + list(row_iterator), axis=0), False  # (n, d)

    width = int(probe[0].shape[-1])  # d
    chunk_rows = max(1, SPARSE_CHUNK_ELEMENTS // width)
    blocks, staged = [], []
    for row in chain(probe, row_iterator):
        staged.append(row)
        if len(staged) >= chunk_rows:
            blocks.append(sparse.csr_matrix(np.concatenate(staged, axis=0)))  # (chunk_rows, d)
            staged = []

    if staged:
        blocks.append(sparse.csr_matrix(np.concatenate(staged, axis=0)))  # (<= chunk_rows, d)

    return sparse.vstack(blocks, format="csr"), True  # (n, d)


def matrix_density(features: FeatureMatrix) -> float:
    """Fraction of entries a matrix actually stores, for logging a fit."""
    stored = features.nnz if sparse.issparse(features) else int(np.count_nonzero(features))
    return stored / (features.shape[0] * features.shape[1])


def live_column_indices(features: FeatureMatrix) -> np.ndarray:
    """Columns of a training matrix that are non-zero for at least one row.

    A column that never leaves zero on train cannot produce a split and cannot change a
    linear map, so dropping it is lossless. It is worth doing because it is not free to
    keep: `colsample_bytree` draws columns uniformly, so dead columns dilute every tree's
    sample. At a 131072-wide codebook joined into a pair row, most columns are dead.
    """
    # features: (n, f) -> (s,) ascending
    if sparse.issparse(features):
        # Read the column of every stored non-zero rather than transposing to CSC. At the
        # widths this exists for, that transpose is another copy of the whole matrix.
        stored = features.tocsr()
        columns = stored.indices[stored.data != 0]  # (kept,)
        return np.unique(columns).astype(np.int64)  # (s,)

    return np.flatnonzero(np.any(features != 0, axis=0)).astype(np.int64)  # (s,)


def take_columns(features: FeatureMatrix, indices: np.ndarray) -> FeatureMatrix:
    """Narrow a matrix to `indices`, keeping its layout."""
    # features: (n, f); indices: (s,) -> (n, s)
    if sparse.issparse(features):
        return features.tocsc()[:, indices].tocsr()

    return features[:, indices]
