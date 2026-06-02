import numpy as np
from sklearn.linear_model import Ridge

try:
    from src.protify.probes import scikit_classes
    from src.protify.probes.scikit_classes import ScikitArguments, ScikitProbe
except ImportError:
    try:
        from protify.probes import scikit_classes
        from protify.probes.scikit_classes import ScikitArguments, ScikitProbe
    except ImportError:
        from ..probes import scikit_classes
        from ..probes.scikit_classes import ScikitArguments, ScikitProbe


def test_scikit_probe_passes_n_jobs_to_random_search(monkeypatch) -> None:
    captured = {}

    class FakeRandomizedSearchCV:
        def __init__(
            self,
            estimator,
            param_distributions,
            n_iter,
            scoring,
            cv,
            random_state,
            n_jobs,
            verbose,
        ):
            captured["n_jobs"] = n_jobs
            self.best_estimator_ = estimator
            self.best_params_ = {"alpha": 1.0}
            self.best_score_ = 0.5

        def fit(self, X_train, y_train) -> None:
            captured["n_samples"] = X_train.shape[0]

    monkeypatch.setattr(scikit_classes, "RandomizedSearchCV", FakeRandomizedSearchCV)
    monkeypatch.setitem(
        scikit_classes.HYPERPARAMETER_DISTRIBUTIONS,
        "Ridge",
        {"alpha": [0.1, 1.0]},
    )

    probe = ScikitProbe(ScikitArguments(n_jobs=4, n_iter=2, cv=2))
    X_train = np.ones((8, 3))
    y_train = np.arange(8, dtype=float)
    best_model, best_params = probe._tune_hyperparameters(
        Ridge,
        "Ridge",
        X_train,
        y_train,
        custom_scorer=None,
    )

    assert captured["n_jobs"] == 4
    assert captured["n_samples"] == 8
    assert isinstance(best_model, Ridge)
    assert best_params == {"alpha": 1.0}
