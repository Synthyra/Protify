"""Applying a fitted feature selection inside the probe datasets.

The selection narrows what a probe reads, so the dataset classes must apply it before the
standardizer and must keep doing so on validation and test. These tests cover the wiring
rather than the fit; `test_feature_path_selector.py` covers the fit.
"""

import numpy as np
import torch


try:
    from src.protify.data.dataset_classes import (
        EmbeddingStandardizer,
        EmbedsLabelsDataset,
        PairEmbedsLabelsDataset,
        apply_embedding_transforms,
    )
    from src.protify.probes.feature_path_selector import FeatureSelection
    from src.protify.testing_suite.embedding_store_builder import build_store_for
except ImportError:
    from protify.data.dataset_classes import (
        EmbeddingStandardizer,
        EmbedsLabelsDataset,
        PairEmbedsLabelsDataset,
        apply_embedding_transforms,
    )
    from protify.probes.feature_path_selector import FeatureSelection
    from protify.testing_suite.embedding_store_builder import build_store_for


SEQ_A = "AAAA"
SEQ_B = "CCCC"


def standardizer(mean, scale) -> EmbeddingStandardizer:
    return EmbeddingStandardizer(np.array(mean, dtype=np.float32), np.array(scale, dtype=np.float32))


def test_selection_runs_before_standardization():
    """Order matters: the scaler statistics have to describe the columns that survive."""
    selection = FeatureSelection(indices=np.array([0, 2], dtype=np.int64), input_size=4)
    # Statistics for columns 0 and 2 only, which is what select_features would hand back.
    scaler = standardizer([1.0, 3.0], [1.0, 1.0])
    embedding = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # (1, 4)

    transformed = apply_embedding_transforms(embedding, selection, scaler)

    assert transformed.tolist() == [[9.0, 27.0]]


def test_transforms_are_optional_and_independent():
    embedding = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # (1, 4)
    selection = FeatureSelection(indices=np.array([1, 3], dtype=np.int64), input_size=4)

    assert apply_embedding_transforms(embedding, None, None).tolist() == embedding.tolist()
    assert apply_embedding_transforms(embedding, selection, None).tolist() == [[20.0, 40.0]]


def test_subsetting_a_scaler_matches_fitting_one_on_those_columns():
    """StandardScaler is per column, so subsetting must equal a refit exactly."""
    features = np.array(
        [[1.0, 10.0, 100.0, 5.0], [3.0, 14.0, 120.0, 9.0], [5.0, 30.0, 160.0, 1.0]],
        dtype=np.float32,
    )  # (3, 4)
    columns = np.array([0, 2], dtype=np.int64)  # (2,)

    subset = EmbeddingStandardizer.fit_numpy(features).select_features(columns)
    refit = EmbeddingStandardizer.fit_numpy(features[:, columns])

    assert np.allclose(subset.mean.numpy(), refit.mean.numpy())
    assert np.allclose(subset.scale.numpy(), refit.scale.numpy())


def test_a_single_sequence_dataset_yields_the_narrowed_width():
    emb_dict = {SEQ_A: torch.tensor([1.0, 2.0, 3.0, 4.0])}  # (4,)
    selection = FeatureSelection(indices=np.array([1, 3], dtype=np.int64), input_size=4)
    dataset = EmbedsLabelsDataset(
        hf_dataset={"seqs": [SEQ_A], "labels": [0]},
        emb_dict=emb_dict,
        input_size=4,
        task_type="singlelabel",
        full=False,
        feature_selection=selection,
    )

    embedding, _ = dataset[0]

    assert embedding.tolist() == [2.0, 4.0]


def test_a_pair_dataset_selects_across_the_joined_vector():
    """The path fit sees A concatenated with B, so the selection spans that joined width.

    Keeping columns 0 and 3 of a four-wide pair means one column from each protein, which
    the dataset must split back down the middle of the selected width, not the original.
    """
    emb_dict = {SEQ_A: torch.tensor([1.0, 2.0]), SEQ_B: torch.tensor([30.0, 40.0])}  # (2,) each
    # input_size on the dataset is one protein's width; the selection spans both.
    selection = FeatureSelection(indices=np.array([0, 3], dtype=np.int64), input_size=4)
    dataset = PairEmbedsLabelsDataset(
        hf_dataset={"SeqA": [SEQ_A], "SeqB": [SEQ_B], "labels": [0]},
        emb_dict=emb_dict,
        input_size=2,
        task_type="singlelabel",
        full=False,
        train=False,
        feature_selection=selection,
    )

    emb_a, emb_b, _ = dataset[0]

    assert emb_a.tolist() == [[1.0]]
    assert emb_b.tolist() == [[40.0]]


def test_a_pair_dataset_applies_the_selection_without_a_standardizer():
    """The concat-and-split block used to be gated on the scaler alone.

    With --embedding_scaler off and selection on, that gate would have skipped the
    selection entirely and handed the probe a full-width embedding.
    """
    emb_dict = {SEQ_A: torch.tensor([1.0, 2.0]), SEQ_B: torch.tensor([30.0, 40.0])}
    selection = FeatureSelection(indices=np.array([1, 2], dtype=np.int64), input_size=4)
    dataset = PairEmbedsLabelsDataset(
        hf_dataset={"SeqA": [SEQ_A], "SeqB": [SEQ_B], "labels": [0]},
        emb_dict=emb_dict,
        input_size=2,
        task_type="singlelabel",
        full=False,
        train=False,
        embedding_standardizer=None,
        feature_selection=selection,
    )

    emb_a, emb_b, _ = dataset[0]

    assert emb_a.tolist() == [[2.0]]
    assert emb_b.tolist() == [[30.0]]


def test_the_scikit_builder_narrows_rows_as_it_reads_them(tmp_path):
    """The estimator path must never materialize the full width, only the selected one."""
    try:
        from src.protify.data.data_mixin import DataMixin
        from src.protify.embedder import EmbeddingArguments
    except ImportError:
        from protify.data.data_mixin import DataMixin
        from protify.embedder import EmbeddingArguments

    embeddings = {SEQ_A: torch.tensor([1.0, 2.0, 3.0, 4.0]), SEQ_B: torch.tensor([5.0, 6.0, 7.0, 8.0])}
    embedding_args = EmbeddingArguments(
        embedding_save_dir=str(tmp_path),
        embedding_pooling_types=["max"],
        matrix_embed=False,
    )
    build_store_for(embedding_args, "test-model", embeddings)

    mixin = DataMixin()
    mixin._sql = False
    mixin._full = False
    mixin.embedding_args = embedding_args
    selection = FeatureSelection(indices=np.array([0, 3], dtype=np.int64), input_size=4)

    X_train, X_valid, X_test = mixin.build_vector_numpy_dataset_from_embeddings(
        "test-model", [SEQ_A], [SEQ_B], [SEQ_A], feature_selection=selection
    )

    assert X_train.tolist() == [[1.0, 4.0]]
    assert X_valid.tolist() == [[5.0, 8.0]]
    assert X_test.tolist() == [[1.0, 4.0]]


def test_the_scikit_builder_is_unchanged_without_a_selection(tmp_path):
    try:
        from src.protify.data.data_mixin import DataMixin
        from src.protify.embedder import EmbeddingArguments
    except ImportError:
        from protify.data.data_mixin import DataMixin
        from protify.embedder import EmbeddingArguments

    embeddings = {SEQ_A: torch.tensor([1.0, 2.0, 3.0, 4.0])}
    embedding_args = EmbeddingArguments(
        embedding_save_dir=str(tmp_path),
        embedding_pooling_types=["max"],
        matrix_embed=False,
    )
    build_store_for(embedding_args, "test-model", embeddings)

    mixin = DataMixin()
    mixin._sql = False
    mixin._full = False
    mixin.embedding_args = embedding_args

    X_train, _, _ = mixin.build_vector_numpy_dataset_from_embeddings(
        "test-model", [SEQ_A], [SEQ_A], [SEQ_A]
    )

    assert X_train.tolist() == [[1.0, 2.0, 3.0, 4.0]]
