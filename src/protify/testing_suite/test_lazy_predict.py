"""
Tests for LazyClassifier and LazyRegressor from probes.lazy_predict.

Run as script for a full "all models" smoke with verbose output:
    python -m src.protify.testing_suite.test_lazy_predict --verbose 1
    python -m src.protify.testing_suite.test_lazy_predict --verbose 0

Run with pytest for fast, rigorous unit tests (subset of models):
    pytest src/protify/testing_suite/test_lazy_predict.py -v
"""

import argparse
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeClassifier

try:
    from src.protify.probes.lazy_predict import LazyClassifier, LazyRegressor
except ImportError:
    try:
        from protify.probes.lazy_predict import LazyClassifier, LazyRegressor
    except ImportError:
        from ..probes.lazy_predict import LazyClassifier, LazyRegressor


# Subset of models for fast pytest runs (avoids "all" which is slow in Docker)
FAST_CLASSIFIERS = [
    ("RidgeClassifier", RidgeClassifier),
    ("RandomForestClassifier", RandomForestClassifier),
]
FAST_REGRESSORS = [
    ("Ridge", Ridge),
    ("RandomForestRegressor", RandomForestRegressor),
]


# Synthetic data used by tests and by __main__ script
def _make_classification_data(n_samples=100, n_features=10, random_state=42):
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)
    return X, y


def _make_regression_data(n_samples=100, n_features=10, random_state=42):
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))
    y = rng.standard_normal(n_samples)
    return X, y


def _train_test_split(X, y, train_frac=0.8, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(y)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    return (
        X[train_idx], X[test_idx],
        y[train_idx], y[test_idx],
    )


# -------- Pytest tests (no argparse at import time) --------


def test_lazy_classifier_fit_returns_dataframe():
    """LazyClassifier.fit returns a DataFrame with expected columns and index."""
    X, y = _make_classification_data(n_samples=80, n_features=5)
    X_train, X_test, y_train, y_test = _train_test_split(X, y, train_frac=0.8)
    clf = LazyClassifier(classifiers=FAST_CLASSIFIERS, verbose=0)
    scores = clf.fit(X_train, X_test, y_train, y_test)
    assert scores is not None
    assert hasattr(scores, "index") and hasattr(scores, "columns")
    assert len(scores) == len(FAST_CLASSIFIERS)
    assert "Accuracy" in scores.columns
    assert "Balanced Accuracy" in scores.columns
    assert "F1 Score" in scores.columns
    assert "Time Taken" in scores.columns
    for col in ("Accuracy", "Balanced Accuracy", "F1 Score"):
        assert np.issubdtype(scores[col].dtype, np.floating)
    assert (scores["Accuracy"] >= 0).all() and (scores["Accuracy"] <= 1).all()
    assert (scores["Balanced Accuracy"] >= 0).all() and (scores["Balanced Accuracy"] <= 1).all()


def test_lazy_classifier_models_populated():
    """LazyClassifier stores fitted models in .models."""
    X, y = _make_classification_data(n_samples=80, n_features=5)
    X_train, X_test, y_train, y_test = _train_test_split(X, y, train_frac=0.8)
    clf = LazyClassifier(classifiers=FAST_CLASSIFIERS, verbose=0)
    clf.fit(X_train, X_test, y_train, y_test)
    assert len(clf.models) == len(FAST_CLASSIFIERS)
    for name, _ in FAST_CLASSIFIERS:
        assert name in clf.models
        assert hasattr(clf.models[name], "predict")


def test_lazy_classifier_passes_n_jobs_to_estimators():
    X, y = _make_classification_data(n_samples=80, n_features=5)
    X_train, X_test, y_train, y_test = _train_test_split(X, y, train_frac=0.8)
    clf = LazyClassifier(
        classifiers=[RandomForestClassifier],
        verbose=0,
        n_jobs=2,
    )
    clf.fit(X_train, X_test, y_train, y_test)
    assert clf.models["RandomForestClassifier"].n_jobs == 2


def test_lazy_classifier_predictions_true_returns_tuple():
    """LazyClassifier with predictions=True returns (scores, predictions_df)."""
    X, y = _make_classification_data(n_samples=80, n_features=5)
    X_train, X_test, y_train, y_test = _train_test_split(X, y, train_frac=0.8)
    clf = LazyClassifier(classifiers=FAST_CLASSIFIERS, verbose=0, predictions=True)
    result = clf.fit(X_train, X_test, y_train, y_test)
    assert isinstance(result, tuple)
    assert len(result) == 2
    scores, preds_df = result
    assert len(preds_df) == len(y_test)
    assert list(preds_df.columns) == [name for name, _ in FAST_CLASSIFIERS]


def test_lazy_regressor_fit_returns_dataframe():
    """LazyRegressor.fit returns a DataFrame with expected columns."""
    X, y = _make_regression_data(n_samples=80, n_features=5)
    X_train, X_test, y_train, y_test = _train_test_split(X, y, train_frac=0.8)
    rg = LazyRegressor(regressors=FAST_REGRESSORS, verbose=0)
    scores = rg.fit(X_train, X_test, y_train, y_test)
    assert scores is not None
    assert len(scores) == len(FAST_REGRESSORS)
    assert "R-Squared" in scores.columns
    assert "Adjusted R-Squared" in scores.columns
    assert "RMSE" in scores.columns
    assert "Time Taken" in scores.columns
    assert (scores["RMSE"] >= 0).all()


def test_lazy_regressor_models_populated():
    """LazyRegressor stores fitted models in .models."""
    X, y = _make_regression_data(n_samples=80, n_features=5)
    X_train, X_test, y_train, y_test = _train_test_split(X, y, train_frac=0.8)
    rg = LazyRegressor(regressors=FAST_REGRESSORS, verbose=0)
    rg.fit(X_train, X_test, y_train, y_test)
    assert len(rg.models) == len(FAST_REGRESSORS)
    for name, _ in FAST_REGRESSORS:
        assert name in rg.models
        assert hasattr(rg.models[name], "predict")


def test_lazy_regressor_passes_n_jobs_to_estimators():
    X, y = _make_regression_data(n_samples=80, n_features=5)
    X_train, X_test, y_train, y_test = _train_test_split(X, y, train_frac=0.8)
    rg = LazyRegressor(
        regressors=[RandomForestRegressor],
        verbose=0,
        n_jobs=2,
    )
    rg.fit(X_train, X_test, y_train, y_test)
    assert rg.models["RandomForestRegressor"].n_jobs == 2


def test_lazy_regressor_predictions_true_returns_tuple():
    """LazyRegressor with predictions=True returns (scores, predictions_df)."""
    X, y = _make_regression_data(n_samples=80, n_features=5)
    X_train, X_test, y_train, y_test = _train_test_split(X, y, train_frac=0.8)
    rg = LazyRegressor(regressors=FAST_REGRESSORS, verbose=0, predictions=True)
    result = rg.fit(X_train, X_test, y_train, y_test)
    assert isinstance(result, tuple)
    assert len(result) == 2
    scores, preds_df = result
    assert len(preds_df) == len(y_test)
    assert list(preds_df.columns) == [name for name, _ in FAST_REGRESSORS]


def test_lazy_classifier_provide_models_returns_fitted_models():
    """LazyClassifier.provide_models returns the same models as after fit."""
    X, y = _make_classification_data(n_samples=80, n_features=5)
    X_train, X_test, y_train, y_test = _train_test_split(X, y, train_frac=0.8)
    clf = LazyClassifier(classifiers=FAST_CLASSIFIERS, verbose=0)
    clf.fit(X_train, X_test, y_train, y_test)
    models = clf.provide_models(X_train, X_test, y_train, y_test)
    assert models is clf.models
    assert len(models) == len(FAST_CLASSIFIERS)


def test_lazy_regressor_provide_models_calls_fit_if_empty():
    """LazyRegressor.provide_models calls fit when models are empty."""
    X, y = _make_regression_data(n_samples=80, n_features=5)
    X_train, X_test, y_train, y_test = _train_test_split(X, y, train_frac=0.8)
    rg = LazyRegressor(regressors=FAST_REGRESSORS, verbose=0)
    assert len(rg.models) == 0
    models = rg.provide_models(X_train, X_test, y_train, y_test)
    assert len(models) == len(FAST_REGRESSORS)
    assert models is rg.models


# -------- Script entrypoint (full "all" models, verbose control) --------


def _run_full_suite(verbose: int = 0) -> None:
    """Run full LazyClassifier + LazyRegressor with classifiers/regressors='all'."""
    X_clf, y_clf = _make_classification_data()
    X_reg, y_reg = _make_regression_data()
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = _train_test_split(X_clf, y_clf)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = _train_test_split(X_reg, y_reg)

    clf = LazyClassifier(classifiers="all", verbose=verbose)
    clf_scores = clf.fit(X_clf_train, X_clf_test, y_clf_train, y_clf_test)
    assert clf_scores is not None and len(clf_scores) > 0

    rg = LazyRegressor(regressors="all", verbose=verbose)
    rg_scores = rg.fit(X_reg_train, X_reg_test, y_reg_train, y_reg_test)
    assert rg_scores is not None and len(rg_scores) > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", type=int, default=0, help="0=summary, 1=full table")
    args = parser.parse_args()
    _run_full_suite(verbose=args.verbose)
