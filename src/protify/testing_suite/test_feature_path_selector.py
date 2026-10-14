"""Selecting embedding columns by an L1 regularization path."""

import numpy as np
import pytest
import torch

from dataclasses import replace
from scipy import sparse

from protify.probes.feature_path_selector import (
    FeaturePathConfig,
    FeatureSelection,
    _candidate_widths,
    _feature_support,
    _nearest_power_of_two,
    fit_feature_path_selection,
    path_importance,
)


# Small enough to run on CPU in a few seconds, wide enough that a selection is a real
# choice rather than a formality.
FAST_CONFIG = FeaturePathConfig(
    hidden_size=64,
    batch_size=64,
    n_alphas=8,
    min_output_size=4,
    max_fit_rows=4000,
    refit_epochs=30,
    device="cpu",
)

INFORMATIVE_COLUMNS = (3, 11, 29, 47)


def planted_regression(n_rows: int = 2000, n_features: int = 64, seed: int = 0):
    """A sparse design where four known columns carry the whole signal.

    Two of those four enter only as a product, so their effect is invisible to any
    statistic that looks at one column at a time. That is the case this module exists
    for, and `test_the_path_beats_univariate_correlation` pins the difference.
    """
    generator = np.random.default_rng(seed)
    X = (generator.random((n_rows, n_features)) < 0.6).astype(np.float32)  # (n, f)
    X *= generator.normal(size=(n_rows, n_features)).astype(np.float32)  # (n, f)
    y = (
        2.0 * X[:, 3]
        + 1.5 * X[:, 11]
        + 3.0 * X[:, 29] * X[:, 47]
        + 0.05 * generator.normal(size=n_rows)
    ).astype(np.float32)  # (n,)
    return X, y


def split(X, y, n_train: int = 1400):
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def univariate_top_columns(X: np.ndarray, y: np.ndarray, k: int) -> set:
    """The k columns an absolute-Pearson ranking would keep."""
    # X: (n, f); y: (n,)
    correlation = np.abs(np.corrcoef(np.vstack([X.T, y[None, :]]))[-1, :-1])  # (f,)
    return set(np.argsort(-correlation)[:k].tolist())


def test_nearest_power_of_two_rounds_in_log_space():
    """Rounding down would halve a survivor count that lands just under a power of two."""
    assert _nearest_power_of_two(1000) == 1024
    assert _nearest_power_of_two(600) == 512
    assert _nearest_power_of_two(1) == 1
    assert _nearest_power_of_two(0) == 1


def test_support_prune_finds_only_nonzero_columns():
    X = np.zeros((5, 6), dtype=np.float32)  # (5, 6)
    X[0, 1] = 1.0
    X[3, 4] = -2.0

    assert _feature_support(X).tolist() == [1, 4]
    assert _feature_support(sparse.csr_matrix(X)).tolist() == [1, 4]


def test_importance_reproduces_entry_order_on_an_exact_path():
    """With exact zeros the trajectory measure must agree with the classic path ordering.

    Rows run from strong penalty to weak. Feature 0 is alive throughout, feature 1 enters
    halfway, feature 2 only at the weakest penalty, so that is the order.
    """
    path_norms = np.array(
        [  # (a=4, f=3)
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [1.0, 0.8, 0.3],
        ]
    )

    assert np.argsort(-path_importance(path_norms)).tolist() == [0, 1, 2]


def test_importance_ranks_a_shrinking_feature_below_a_persistent_one():
    """Gradient training never reaches exact zero, so the ranking cannot depend on it.

    Feature 1 is merely small rather than dead, which under a hard-zero test would count
    as alive at every alpha and tie with feature 0.
    """
    path_norms = np.array(
        [  # (a=3, f=2)
            [1.0, 1e-4],
            [1.0, 1e-3],
            [1.0, 0.9],
        ]
    )

    importance = path_importance(path_norms)

    assert importance[0] > importance[1]


def test_importance_is_invariant_to_the_overall_weight_scale():
    """Per-alpha normalization is what makes early and late alphas comparable at all."""
    path_norms = np.array([[1.0, 0.25], [2.0, 1.0]])  # (a=2, f=2)

    assert np.allclose(path_importance(path_norms), path_importance(path_norms * 1000.0))


def test_the_path_recovers_planted_interacting_features():
    X_train, y_train, X_valid, y_valid = split(*planted_regression())
    selection = fit_feature_path_selection(
        X_train, y_train, X_valid, y_valid, "regression", FAST_CONFIG
    )

    assert selection.input_size == 64
    assert selection.output_size < 64, "Selection must actually drop columns."
    kept = set(selection.indices.tolist())
    assert set(INFORMATIVE_COLUMNS).issubset(kept), (
        f"Kept {sorted(kept)}, which misses some of {list(INFORMATIVE_COLUMNS)}."
    )


def test_the_path_beats_univariate_correlation():
    """The whole reason this module is not a variance or correlation filter.

    Columns 29 and 47 matter only through their product, so each one on its own barely
    correlates with the target. A univariate ranking of the same width misses them; the
    path fit, which sees them together inside a hidden layer, does not.
    """
    X_train, y_train, X_valid, y_valid = split(*planted_regression())
    selection = fit_feature_path_selection(
        X_train, y_train, X_valid, y_valid, "regression", FAST_CONFIG
    )

    kept = set(selection.indices.tolist())
    univariate = univariate_top_columns(X_train, y_train, selection.output_size)
    assert set(INFORMATIVE_COLUMNS).issubset(kept)
    assert not set(INFORMATIVE_COLUMNS).issubset(univariate), (
        "The univariate baseline found every planted column, so this case no longer "
        "distinguishes the two approaches."
    )


def test_a_csr_matrix_selects_the_same_columns_as_its_dense_form():
    """Storage format must not change the result, only the memory it takes to get there."""
    X_train, y_train, X_valid, y_valid = split(*planted_regression())

    dense = fit_feature_path_selection(
        X_train, y_train, X_valid, y_valid, "regression", FAST_CONFIG
    )
    csr = fit_feature_path_selection(
        sparse.csr_matrix(X_train),
        y_train,
        sparse.csr_matrix(X_valid),
        y_valid,
        "regression",
        FAST_CONFIG,
    )

    assert dense.indices.tolist() == csr.indices.tolist()


def test_all_zero_columns_never_reach_the_selection():
    X, y = planted_regression()
    X[:, 5] = 0.0
    X[:, 17] = 0.0
    X_train, y_train, X_valid, y_valid = split(X, y)

    selection = fit_feature_path_selection(
        X_train, y_train, X_valid, y_valid, "regression", FAST_CONFIG
    )

    assert 5 not in selection.indices.tolist()
    assert 17 not in selection.indices.tolist()


def test_a_non_finite_input_is_named_rather_than_diverging_later():
    """A corrupt embedding must fail with the array named, not as a NaN deep in the path."""
    X_train, y_train, X_valid, y_valid = split(*planted_regression())
    X_train = X_train.copy()
    X_train[0, 0] = np.nan

    with pytest.raises(ValueError, match="train_features contains NaN"):
        fit_feature_path_selection(X_train, y_train, X_valid, y_valid, "regression", FAST_CONFIG)


def test_non_finite_labels_are_named_too():
    X_train, y_train, X_valid, y_valid = split(*planted_regression())
    y_valid = y_valid.copy()
    y_valid[3] = np.inf

    with pytest.raises(ValueError, match="valid_labels contains NaN"):
        fit_feature_path_selection(X_train, y_train, X_valid, y_valid, "regression", FAST_CONFIG)


def test_candidate_widths_are_powers_of_two_up_to_the_ranked_set():
    assert _candidate_widths(64, 4) == [4, 8, 16, 32, 64]
    assert _candidate_widths(100, 4) == [4, 8, 16, 32, 64]
    # A support narrower than the floor still gets one candidate rather than none.
    assert _candidate_widths(64, 256) == [64]


def test_a_singlelabel_task_selects_over_class_logits():
    generator = np.random.default_rng(3)
    X = generator.normal(size=(400, 32)).astype(np.float32)  # (400, 32)
    labels = (X[:, 7] + X[:, 19] > 0).astype(np.int64)  # (400,)

    selection = fit_feature_path_selection(
        X[:300], labels[:300], X[300:], labels[300:], "singlelabel", FAST_CONFIG
    )

    kept = set(selection.indices.tolist())
    assert {7, 19}.issubset(kept), f"Kept {sorted(kept)}, which misses 7 or 19."


def test_a_multilabel_task_selects_over_every_label():
    generator = np.random.default_rng(5)
    X = generator.normal(size=(400, 32)).astype(np.float32)  # (400, 32)
    labels = np.stack(
        [(X[:, 2] > 0), (X[:, 25] > 0)], axis=1
    ).astype(np.float32)  # (400, 2)

    selection = fit_feature_path_selection(
        X[:300], labels[:300], X[300:], labels[300:], "multilabel", FAST_CONFIG
    )

    kept = set(selection.indices.tolist())
    assert {2, 25}.issubset(kept), f"Kept {sorted(kept)}, which misses 2 or 25."


def test_selection_round_trips_through_disk(tmp_path):
    """The index array is the whole artifact, so a reload has to reproduce it exactly."""
    selection = FeatureSelection(indices=np.array([0, 4, 9], dtype=np.int64), input_size=10)
    path = tmp_path / "selection.npz"

    selection.save(path)
    reloaded = FeatureSelection.load(path)

    assert reloaded.indices.tolist() == selection.indices.tolist()
    assert reloaded.input_size == selection.input_size


def test_transform_keeps_the_selected_columns_in_order():
    selection = FeatureSelection(indices=np.array([0, 2], dtype=np.int64), input_size=4)
    features = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)  # (2, 4)

    assert selection.transform_numpy(features).tolist() == [[1.0, 3.0], [5.0, 7.0]]
    assert selection.transform_tensor(torch.from_numpy(features)).tolist() == [
        [1.0, 3.0],
        [5.0, 7.0],
    ]


def test_a_width_mismatch_raises_instead_of_silently_reindexing():
    selection = FeatureSelection(indices=np.array([0, 2], dtype=np.int64), input_size=4)

    with pytest.raises(ValueError, match="does not match the selected width"):
        selection.transform_numpy(np.zeros((2, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="does not match the selected width"):
        selection.transform_tensor(torch.zeros(8))


def test_an_index_past_the_embedding_width_is_rejected_at_construction():
    with pytest.raises(ValueError, match="outside an embedding of width"):
        FeatureSelection(indices=np.array([0, 9], dtype=np.int64), input_size=4)


def test_mismatched_train_and_valid_widths_raise():
    with pytest.raises(ValueError, match="does not match valid width"):
        fit_feature_path_selection(
            np.ones((4, 8), dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.ones((4, 6), dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            "regression",
            FAST_CONFIG,
        )


def test_an_all_zero_training_matrix_raises():
    with pytest.raises(ValueError, match="nothing can be selected"):
        fit_feature_path_selection(
            np.zeros((8, 8), dtype=np.float32),
            np.zeros(8, dtype=np.float32),
            np.zeros((8, 8), dtype=np.float32),
            np.zeros(8, dtype=np.float32),
            "regression",
            FAST_CONFIG,
        )


def test_the_width_search_averages_over_several_refits():
    """One refit per width lets a single lucky fit set the probe's input width.

    The search compares about ten candidates on one validation split, so the best of them
    is partly a draw. On real ESMC-6B data that produced a 256-of-131072 selection whose
    validation loss was the best of the candidates and whose test score was 0.03 Spearman
    worse than keeping every column, which is the failure this averaging targets.
    """
    assert FeaturePathConfig().refit_seeds > 1


def test_refit_seeds_of_one_reproduces_a_single_fit():
    """The averaging must collapse cleanly, so a cheap run stays available."""
    X_train, y_train, X_valid, y_valid = split(*planted_regression())
    single = replace(FAST_CONFIG, refit_seeds=1)
    repeated = replace(FAST_CONFIG, refit_seeds=1)

    first = fit_feature_path_selection(X_train, y_train, X_valid, y_valid, "regression", single)
    second = fit_feature_path_selection(X_train, y_train, X_valid, y_valid, "regression", repeated)

    assert first.indices.tolist() == second.indices.tolist()
