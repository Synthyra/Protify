"""Searching estimator probe hyperparameters against the validation split.

The two properties worth pinning are discipline and reproducibility. A search must never
read the test split, because a configuration chosen against test is not a result. And the
overrides it returns must be exactly what the final fit receives, or the reported score
belongs to a model nobody trained.
"""

import numpy as np
import pytest

from scipy import sparse

from protify.probes import estimator_tuning
from protify.probes.estimator_probe import EstimatorProbeArguments
from protify.probes.estimator_tuning import (
    MINIMIZED_METRICS,
    SEARCH_SPACES,
    LIBRARY_DEFAULTS,
    EstimatorSearchArguments,
    default_trial_params,
    default_tune_metric,
    search_space_names,
    tune_estimator_probe,
)


FEATURE_WIDTH = 6


def _probe_args(task_type="singlelabel", overrides=None):
    return EstimatorProbeArguments(
        probe_type="xgboost",
        task_type=task_type,
        num_labels=2,
        seed=7,
        n_jobs=1,
        device="cpu",
        overrides={} if overrides is None else overrides,
    )


def _split(n_rows: int = 24):
    generator = np.random.default_rng(3)
    X = generator.random((n_rows, FEATURE_WIDTH)).astype(np.float32)  # (n, f)
    y = (X[:, 0] > 0.5).astype(np.int64)  # (n,)
    return X, y


@pytest.fixture
def recorded_fits(monkeypatch):
    """Replace the fit with a recorder, so a search runs without an estimator library."""
    calls = []

    def fake_fit(args, X_train, y_train, X_valid, y_valid):
        calls.append({"overrides": dict(args.overrides), "seed": args.seed})
        # Scores rise with max_depth, so the best trial is knowable in advance.
        return object(), {"mcc": 0.5 + 0.01 * args.overrides["max_depth"], "pr_auc": 0.1}

    monkeypatch.setattr(estimator_tuning, "fit_and_score_valid", fake_fit)
    return calls


def test_the_search_never_receives_the_test_split():
    """`tune_estimator_probe` takes four arrays. Test cannot leak through an argument it
    does not have, so this pins the signature that guarantees it."""
    import inspect

    parameters = list(inspect.signature(tune_estimator_probe).parameters)
    assert not any("test" in name for name in parameters)
    assert parameters[2:] == ["X_train", "y_train", "X_valid", "y_valid"]


def test_the_returned_overrides_are_the_best_trial_seen(recorded_fits):
    X, y = _split()
    best = tune_estimator_probe(
        _probe_args(), EstimatorSearchArguments(n_trials=6, seed=1), X, y, X, y
    )
    depths = [call["overrides"]["max_depth"] for call in recorded_fits]
    assert best["max_depth"] == max(depths)


def test_a_pinned_override_survives_the_search(recorded_fits):
    """--scikit_model_args sets the run-wide default; only the search space overrides it."""
    X, y = _split()
    tune_estimator_probe(
        _probe_args(overrides={"tree_method": "exact", "max_depth": 99}),
        EstimatorSearchArguments(n_trials=3, seed=1),
        X, y, X, y,
    )
    assert all(call["overrides"]["tree_method"] == "exact" for call in recorded_fits)
    # max_depth is in the search space, so the search is what sets it, not the pin.
    assert all(call["overrides"]["max_depth"] != 99 for call in recorded_fits)


def test_every_trial_fits_at_the_same_seed(recorded_fits):
    """Otherwise a trial could win on its seed rather than on its hyperparameters."""
    X, y = _split()
    tune_estimator_probe(
        _probe_args(), EstimatorSearchArguments(n_trials=4, seed=1), X, y, X, y
    )
    assert {call["seed"] for call in recorded_fits} == {7}


def test_the_number_of_trials_is_what_was_asked_for(recorded_fits):
    X, y = _split()
    tune_estimator_probe(
        _probe_args(), EstimatorSearchArguments(n_trials=5, seed=1), X, y, X, y
    )
    assert len(recorded_fits) == 5


def test_a_metric_the_task_does_not_report_names_itself(recorded_fits):
    X, y = _split()
    with pytest.raises(KeyError, match="spearman_rho"):
        tune_estimator_probe(
            _probe_args(),
            EstimatorSearchArguments(n_trials=1, metric="spearman_rho", seed=1),
            X, y, X, y,
        )


def test_an_error_metric_is_minimized_and_the_rest_maximized(monkeypatch):
    """Direction is read from the metric name, so mse must not be maximized."""
    scores = iter([9.0, 1.0, 5.0])

    def fake_fit(args, X_train, y_train, X_valid, y_valid):
        return object(), {"mse": next(scores)}

    monkeypatch.setattr(estimator_tuning, "fit_and_score_valid", fake_fit)
    recorded = {}
    monkeypatch.setattr(
        estimator_tuning, "print_message",
        lambda message: recorded.setdefault("messages", []).append(message),
    )
    X, y = _split()
    tune_estimator_probe(
        _probe_args(task_type="regression"),
        EstimatorSearchArguments(n_trials=3, metric="mse", seed=1),
        X, y, X, y,
    )
    assert any("minimize validation mse" in message for message in recorded["messages"])
    assert "mse" in MINIMIZED_METRICS


def test_the_default_objective_is_the_metric_the_results_table_reports():
    """A search scored on one metric and reported on another can move the reported one
    backwards, which is what pr_auc did on a balanced pair dataset."""
    assert default_tune_metric("singlelabel") == "mcc"
    assert default_tune_metric("multilabel") == "mcc"
    assert default_tune_metric("regression") == "spearman_rho"
    assert default_tune_metric("sigmoid_regression") == "spearman_rho"


def test_a_probe_type_with_no_space_is_refused():
    args = _probe_args()
    args.probe_type = "mlp"
    X, y = _split()
    with pytest.raises(ValueError, match="No search space"):
        tune_estimator_probe(args, EstimatorSearchArguments(n_trials=1), X, y, X, y)


def test_every_estimator_probe_type_has_a_search_space():
    from protify.probes.estimator_probe import ESTIMATOR_PROBE_TYPES

    assert set(SEARCH_SPACES) == set(ESTIMATOR_PROBE_TYPES)


def test_a_compressed_feature_matrix_reaches_the_fit_unchanged(recorded_fits, monkeypatch):
    """The estimator path hands the search CSR, not a dense array."""
    seen = {}

    def fake_fit(args, X_train, y_train, X_valid, y_valid):
        seen["sparse"] = sparse.issparse(X_train)
        return object(), {"mcc": 0.5 + 0.01 * args.overrides["max_depth"]}

    monkeypatch.setattr(estimator_tuning, "fit_and_score_valid", fake_fit)
    X, y = _split()
    compressed = sparse.csr_matrix(X)  # (n, f)
    tune_estimator_probe(
        _probe_args(), EstimatorSearchArguments(n_trials=2, seed=1), compressed, y, compressed, y
    )
    assert seen["sparse"] is True


def test_the_first_trial_is_the_shipped_defaults(recorded_fits):
    """The search starts where an untuned run lands, so it can only move away by scoring
    better on validation."""
    from protify.probes.estimator_probe import ESTIMATOR_DEFAULTS

    X, y = _split()
    tune_estimator_probe(
        _probe_args(), EstimatorSearchArguments(n_trials=3, seed=1), X, y, X, y
    )
    first = recorded_fits[0]["overrides"]
    for name in ("learning_rate", "max_depth", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda"):
        assert first[name] == pytest.approx(ESTIMATOR_DEFAULTS["xgboost"][name]), name
    # The parameters the shipped defaults leave to XGBoost sit at XGBoost's own values.
    assert first["reg_alpha"] == 0.0
    assert first["scale_pos_weight"] == 1.0


@pytest.mark.parametrize("probe_type", ["xgboost", "lightgbm", "random_forest"])
@pytest.mark.parametrize("task_type,num_labels", [("singlelabel", 2), ("singlelabel", 5), ("regression", 1)])
def test_the_warm_start_names_every_parameter_the_space_asks_for(probe_type, task_type, num_labels):
    """A searched parameter left out of the first trial would be sampled into it, so the first trial would not be the untuned run."""
    args = _probe_args()
    args.probe_type, args.task_type, args.num_labels = probe_type, task_type, num_labels
    assert list(default_trial_params(args)) == search_space_names(args)


def test_the_warm_start_matches_the_libraries_defaults():
    """Where the shipped defaults say nothing, the first trial uses what an untuned fit uses."""
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier().get_params()
    assert {name: forest[name] for name in LIBRARY_DEFAULTS["random_forest"]} == LIBRARY_DEFAULTS["random_forest"]
    assert LGBMClassifier().get_params()["subsample"] == LIBRARY_DEFAULTS["lightgbm"]["subsample"]


def test_tuning_returns_estimator_parameters_not_the_trials_switches(recorded_fits):
    X, y = _split()
    best = tune_estimator_probe(_probe_args(), EstimatorSearchArguments(n_trials=4, seed=1), X, y, X, y)
    assert "l1_penalty" not in best
    assert "reg_alpha" in best and best["reg_alpha"] >= 0.0


@pytest.mark.parametrize("probe_type", ["xgboost", "lightgbm", "random_forest"])
def test_the_recorded_names_are_the_names_a_trial_is_asked_for(probe_type, monkeypatch):
    """The recorder reads the space itself, so a new parameter cannot go unenqueued."""
    asked = []

    class _Watching:
        def suggest_float(self, name, low, high, **kwargs):
            asked.append(name)
            return low

        def suggest_int(self, name, low, high, **kwargs):
            asked.append(name)
            return low

        def suggest_categorical(self, name, choices):
            asked.append(name)
            return choices[0]

    args = _probe_args()
    args.probe_type = probe_type
    SEARCH_SPACES[probe_type](_Watching(), args)
    assert asked == search_space_names(args)


def test_a_positive_class_weight_is_searched_only_for_binary_classification():
    """It is defined against one label taking two values, and is inert elsewhere."""
    binary = _probe_args()
    assert "scale_pos_weight" in search_space_names(binary)

    multiclass = _probe_args()
    multiclass.num_labels = 13
    assert "scale_pos_weight" not in search_space_names(multiclass)

    regression = _probe_args(task_type="regression")
    assert "scale_pos_weight" not in search_space_names(regression)

    multilabel = _probe_args(task_type="multilabel")
    multilabel.num_labels = 685
    assert "scale_pos_weight" not in search_space_names(multilabel)


def test_a_forest_weights_classes_but_not_regression_targets():
    forest = _probe_args()
    forest.probe_type = "random_forest"
    assert "class_weight" in search_space_names(forest)

    forest_regression = _probe_args(task_type="regression")
    forest_regression.probe_type = "random_forest"
    assert "class_weight" not in search_space_names(forest_regression)


def test_a_timeout_that_beats_the_first_fit_says_so(monkeypatch):
    """`study.best_value` raises an opaque message when nothing completed, so name it."""
    def slow_fit(args, X_train, y_train, X_valid, y_valid):
        raise AssertionError("no trial should have been allowed to finish")

    monkeypatch.setattr(estimator_tuning, "fit_and_score_valid", slow_fit)
    X, y = _split()
    with pytest.raises(RuntimeError, match="before the first"):
        tune_estimator_probe(
            _probe_args(),
            EstimatorSearchArguments(n_trials=2, timeout=0.0, seed=1),
            X, y, X, y,
        )


def test_several_fits_per_trial_are_averaged_into_the_objective(monkeypatch):
    """A configuration that wins one fit on subsampling luck should not win the search."""
    per_seed = {7: 0.9, 8: 0.1}  # one lucky seed, one unlucky

    def fake_fit(args, X_train, y_train, X_valid, y_valid):
        return object(), {"mcc": per_seed[args.seed]}

    monkeypatch.setattr(estimator_tuning, "fit_and_score_valid", fake_fit)
    seen = []
    monkeypatch.setattr(
        estimator_tuning, "print_message", lambda message: seen.append(message)
    )
    X, y = _split()
    tune_estimator_probe(
        _probe_args(),
        EstimatorSearchArguments(n_trials=1, seed=1, n_seeds=2),
        X, y, X, y,
    )
    assert any("Best validation mcc: 0.50000" in message for message in seen)


def test_every_trial_sees_the_same_seeds(monkeypatch):
    """Otherwise a trial could win on which seeds it drew rather than on its settings."""
    seeds_per_trial = []

    def fake_fit(args, X_train, y_train, X_valid, y_valid):
        seeds_per_trial.append(args.seed)
        return object(), {"mcc": 0.5}

    monkeypatch.setattr(estimator_tuning, "fit_and_score_valid", fake_fit)
    X, y = _split()
    tune_estimator_probe(
        _probe_args(), EstimatorSearchArguments(n_trials=3, seed=1, n_seeds=3), X, y, X, y
    )
    assert seeds_per_trial == [7, 8, 9] * 3


def test_the_search_can_be_pointed_at_a_ranking_metric_instead(monkeypatch):
    """`pr_auc` is still right when the positive class is rare, so --tune_metric keeps it."""
    def fake_fit(args, X_train, y_train, X_valid, y_valid):
        return object(), {"mcc": 0.1, "pr_auc": 0.5 + 0.01 * args.overrides["max_depth"]}

    monkeypatch.setattr(estimator_tuning, "fit_and_score_valid", fake_fit)
    seen = []
    monkeypatch.setattr(estimator_tuning, "print_message", lambda m: seen.append(m))
    X, y = _split()
    tune_estimator_probe(
        _probe_args(), EstimatorSearchArguments(n_trials=2, metric="pr_auc", seed=1), X, y, X, y
    )
    assert any("maximize validation pr_auc" in message for message in seen)
