"""Saving a fitted estimator probe and scoring with it again.

A saved probe is only useful if the row it is handed at inference is the row it was fit
on. Three things decide that and none of them can be inferred from the estimator itself:
the embedding recipe, the pair rule, and the column selection. These tests pin that all
three survive a save and load, and that the composed column selection indexes the original
full-width vector rather than some intermediate one.
"""

import json
import numpy as np
import pytest


try:
    from src.protify.probes.estimator_artifact import (
        EstimatorProbeRecipe,
        build_feature_rows,
        predict_features,
        load_estimator_probe,
        save_estimator_probe,
    )
    from src.protify.probes.feature_path_selector import FeatureSelection
except ImportError:
    from protify.probes.estimator_artifact import (
        EstimatorProbeRecipe,
        build_feature_rows,
        predict_features,
        load_estimator_probe,
        save_estimator_probe,
    )
    from protify.probes.feature_path_selector import FeatureSelection


pytest.importorskip("joblib")


FULL_WIDTH = 10


class _ConstantEstimator:
    """Stands in for a fitted booster, so a round trip needs no learner library."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def predict(self, X):
        return np.full(X.shape[0], len(self.tag), dtype=np.float64)


def _recipe(**overrides) -> EstimatorProbeRecipe:
    fields = dict(
        probe_type="xgboost",
        task_type="singlelabel",
        num_labels=2,
        model_name="ESMC-6B-SAE-l60-k16-c131072",
        pooling_types=["max"],
        hidden_state_index=-1,
        max_length=2048,
        ppi=False,
        pair_feature_mode="concat",
        hyperparameters={"max_depth": 6},
    )
    fields.update(overrides)
    return EstimatorProbeRecipe(**fields)


def test_a_saved_probe_reloads_with_its_recipe_and_selection(tmp_path):
    selection = FeatureSelection(indices=np.array([1, 4, 7], dtype=np.int64), input_size=FULL_WIDTH)
    save_estimator_probe(tmp_path / "probe", _ConstantEstimator("abc"), _recipe(), selection)

    estimator, recipe, loaded = load_estimator_probe(tmp_path / "probe")
    assert estimator.tag == "abc"
    assert recipe == _recipe()
    np.testing.assert_array_equal(loaded.indices, selection.indices)
    assert loaded.input_size == FULL_WIDTH


def test_a_probe_with_no_selection_reloads_without_one(tmp_path):
    save_estimator_probe(tmp_path / "probe", _ConstantEstimator("a"), _recipe(), None)
    _, _, loaded = load_estimator_probe(tmp_path / "probe")
    assert loaded is None


def test_the_recipe_is_readable_without_importing_a_model(tmp_path):
    """The configuration is plain JSON, so a reader does not need FastPLMs to inspect it."""
    save_estimator_probe(tmp_path / "probe", _ConstantEstimator("a"), _recipe(), None)
    stored = json.loads((tmp_path / "probe" / "probe.json").read_text(encoding="utf-8"))
    assert stored["model_name"] == "ESMC-6B-SAE-l60-k16-c131072"
    assert stored["pooling_types"] == ["max"]
    assert stored["pair_feature_mode"] == "concat"


def test_a_pair_probe_joins_at_inference_the_way_it_was_trained():
    pooled_a = np.array([[0.0, 3.0, 1.0]])  # (1, d)
    pooled_b = np.array([[2.0, 1.0, 1.0]])  # (1, d)

    concat = build_feature_rows(pooled_a, _recipe(ppi=True), None, pooled_b)  # (1, 2 * d)
    np.testing.assert_array_equal(concat, np.array([[0.0, 3.0, 1.0, 2.0, 1.0, 1.0]]))

    minmax = build_feature_rows(
        pooled_a, _recipe(ppi=True, pair_feature_mode="minmax"), None, pooled_b
    )  # (1, 2 * d)
    np.testing.assert_array_equal(minmax, np.array([[0.0, 1.0, 1.0, 2.0, 3.0, 1.0]]))
    swapped = build_feature_rows(
        pooled_b, _recipe(ppi=True, pair_feature_mode="minmax"), None, pooled_a
    )
    np.testing.assert_array_equal(minmax, swapped)


def test_a_pair_probe_refuses_a_single_protein():
    with pytest.raises(ValueError, match="both"):
        build_feature_rows(np.zeros((1, 3)), _recipe(ppi=True), None)


def test_a_single_protein_probe_refuses_a_partner():
    with pytest.raises(ValueError, match="no meaning"):
        build_feature_rows(np.zeros((1, 3)), _recipe(), None, np.zeros((1, 3)))


def test_two_narrowing_steps_compose_into_one_index_over_the_original_width():
    """The saved selection has to index the full pooled vector, not the first step's output.

    Replaying the steps in the wrong space would read real columns at wrong positions,
    which produces a plausible prediction from the wrong features and raises nothing.
    """
    values = np.arange(FULL_WIDTH, dtype=np.float64).reshape(1, -1)  # (1, f)
    first = FeatureSelection(indices=np.array([2, 3, 5, 8], dtype=np.int64), input_size=FULL_WIDTH)
    after_first = first.transform_numpy(values)  # (1, 4)

    live = np.array([1, 3])  # positions within the first step's output
    composed = first.narrowed_by(live)
    assert composed.input_size == FULL_WIDTH
    np.testing.assert_array_equal(composed.indices, np.array([3, 8]))
    np.testing.assert_array_equal(composed.transform_numpy(values), after_first[:, live])


def test_a_first_selection_that_is_absent_leaves_the_live_columns_as_the_selection():
    live = np.array([0, 4, 9], dtype=np.int64)
    selection = FeatureSelection(indices=live, input_size=FULL_WIDTH)
    values = np.arange(FULL_WIDTH, dtype=np.float64).reshape(1, -1)  # (1, f)
    np.testing.assert_array_equal(selection.transform_numpy(values), values[:, live])


class _ProbabilityEstimator:
    """Stands in for a fitted classifier, so the scoring contract needs no learner."""

    classes_ = [np.array([0, 1]), np.array([0, 1])]

    def predict(self, X):
        return np.full(X.shape[0], 2.5)

    def predict_proba(self, X):
        block = np.tile(np.array([[0.3, 0.7]]), (X.shape[0], 1))  # (n, 2)
        return block


class _MultiLabelEstimator(_ProbabilityEstimator):
    def predict_proba(self, X):
        return [
            np.tile(np.array([[0.9, 0.1]]), (X.shape[0], 1)),
            np.tile(np.array([[0.2, 0.8]]), (X.shape[0], 1)),
        ]


def test_regression_scores_are_values_not_labels():
    features = np.zeros((3, 4))
    np.testing.assert_allclose(
        predict_features(_ProbabilityEstimator(), features, "regression"),
        np.full(3, 2.5),
    )


def test_classification_scores_are_probabilities_not_labels_or_logs():
    """A caller scoring new proteins wants a probability, which is neither what the metric
    functions read nor what `predict` returns."""
    features = np.zeros((3, 4))
    scores = predict_features(_ProbabilityEstimator(), features, "singlelabel")  # (n, c)
    np.testing.assert_allclose(scores, np.tile(np.array([[0.3, 0.7]]), (3, 1)))
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_multilabel_scores_are_one_positive_probability_per_label():
    features = np.zeros((3, 4))
    scores = predict_features(_MultiLabelEstimator(), features, "multilabel")  # (n, c)
    np.testing.assert_allclose(scores, np.tile(np.array([[0.1, 0.8]]), (3, 1)))
