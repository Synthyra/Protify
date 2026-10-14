"""The dimensionality-reduction plots read their embeddings from the feature store.

`DimensionalityReducer.load_embeddings` asks the Embedder for the sequences the store lacks, which
converts an old cache it finds and embeds nothing when the store is complete, then reads the rows back
one at a time so a large store never sits in memory at once. Here every store is complete or converted,
so no model loads.
"""

import numpy as np
import pytest
import torch


pytest.importorskip("umap")

try:
    from src.protify.testing_suite.embedding_store_builder import build_store
    from src.protify.testing_suite.legacy_cache_writer import write_sqlite_cache
    from src.protify.visualization.reduce_dim import PCA, VisualizationArguments
except ImportError:
    from protify.testing_suite.embedding_store_builder import build_store
    from protify.testing_suite.legacy_cache_writer import write_sqlite_cache
    from protify.visualization.reduce_dim import PCA, VisualizationArguments


pytestmark = pytest.mark.filterwarnings("ignore:Downloading embeddings is recommended")

SEQUENCES = ["MKTAYIAKQR", "MKTAYIAKQRQISFVKSHFSRQ", "GGGGSGGGGS", "ACDEFGHIKL", "WWWWWWWWWWWWWW"]


def arguments(tmp_path, **overrides):
    settings = dict(embedding_save_dir=str(tmp_path / "embeddings"), fig_dir=str(tmp_path / "figures"), save_fig=False)
    settings.update(overrides)
    return VisualizationArguments(**settings)


def test_a_complete_store_is_read_row_by_row_and_nothing_is_embedded(tmp_path):
    generator = torch.Generator().manual_seed(0)
    rows = {sequence: torch.randn(6, generator=generator) for sequence in SEQUENCES}
    build_store(tmp_path / "embeddings", rows)
    reducer = PCA(arguments(tmp_path))

    reducer.load_embeddings(SEQUENCES, labels=[0, 1, 0, 1, 0])

    np.testing.assert_array_equal(reducer.embeddings, np.stack([rows[sequence].numpy() for sequence in SEQUENCES]))
    assert reducer.labels.tolist() == [0, 1, 0, 1, 0]
    assert reducer.fit_transform().shape == (len(SEQUENCES), 2)


def test_a_per_token_store_is_averaged_over_its_tokens(tmp_path):
    matrices = {sequence: torch.arange((len(sequence) + 2) * 3, dtype=torch.float32).reshape(-1, 3) for sequence in SEQUENCES}
    build_store(tmp_path / "embeddings", matrices, matrix=True)
    reducer = PCA(arguments(tmp_path, matrix_embed=True))

    reducer.load_embeddings(SEQUENCES)

    expected = np.stack([matrices[sequence].mean(dim=0).numpy() for sequence in SEQUENCES])
    np.testing.assert_allclose(reducer.embeddings, expected)
    assert reducer.labels is None


def test_an_old_cache_of_the_representation_is_converted_first(tmp_path):
    generator = torch.Generator().manual_seed(1)
    rows = {sequence: torch.randn(6, generator=generator) for sequence in SEQUENCES}
    root = tmp_path / "embeddings"
    root.mkdir()
    old = write_sqlite_cache(root / "ESM2-8_False_len2048_mean.db", rows, wal=False)
    before = old.read_bytes()
    reducer = PCA(arguments(tmp_path))

    reducer.load_embeddings(SEQUENCES)

    np.testing.assert_array_equal(reducer.embeddings, np.stack([rows[sequence].numpy() for sequence in SEQUENCES]))
    assert old.read_bytes() == before, "the old cache is converted, never changed"
    assert sorted(path.suffix for path in root.iterdir() if path.is_file()) == [".db"]
