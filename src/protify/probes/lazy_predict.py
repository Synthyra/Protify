import time
import warnings
import lightgbm
### Modified version of lazy predict from https://github.com/shankarpandala/lazypredict
import numpy as np
import pandas as pd
import xgboost

from collections.abc import Callable, Sequence
from typing import Any
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    r2_score,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.utils import all_estimators
from tqdm import tqdm

from ..console_output import print_message
from ..seed_utils import get_global_seed


warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)


removed_classifiers = [
    "ClassifierChain",
    "ComplementNB",
    "GradientBoostingClassifier",
    "GaussianProcessClassifier",
    "HistGradientBoostingClassifier",
    "MLPClassifier",
    "LogisticRegressionCV", 
    "MultiOutputClassifier", 
    "MultinomialNB", 
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "RadiusNeighborsClassifier",
    "VotingClassifier",
    "LogisticRegressionCV",
    "CalibratedClassifierCV",
    "RidgeClassifierCV",
    "LinearSVC",
    "Perceptron",
    "MLPClassifier",
    "SGDClassifier",
    # O(n²) memory models - too slow for large datasets
    "LabelPropagation",
    "LabelSpreading", 
    "SVC",
    "NuSVC",
    # Sequential ensemble models - slow for large datasets
    "AdaBoostClassifier",
    "BaggingClassifier",
    # O(n×m) prediction time - slow for large test sets
    "KNeighborsClassifier",
    # Unbounded tree depth - very slow on high-dim data
    "DecisionTreeClassifier",
    "ExtraTreeClassifier",
    "ExtraTreesClassifier",
    # Fails on negative values after StandardScaler
    "CategoricalNB",
    # O(d²) or O(d³) - slow on high-dimensional data (4608 features)
    "LinearDiscriminantAnalysis",
    "QuadraticDiscriminantAnalysis",
    # Requires estimator argument
    "FixedThresholdClassifier",
    "TunedThresholdClassifierCV",
]

removed_regressors = [
    "TheilSenRegressor",
    "ARDRegression", 
    "CCA", 
    "IsotonicRegression", 
    "StackingRegressor",
    "MultiOutputRegressor", 
    "MultiTaskElasticNet", 
    "MultiTaskElasticNetCV", 
    "MultiTaskLasso", 
    "MultiTaskLassoCV", 
    "PLSCanonical", 
    "PLSRegression", 
    "RadiusNeighborsRegressor", 
    "RegressorChain", 
    "VotingRegressor",
    "OrthogonalMatchingPursuitCV",
    "LassoLars",
    "LarsCV",
    "LassoCV",
    "RidgeCV",
    "LassoLarsCV",
    "ElasticNetCV",
    "LinearSVR",
    "LassoLarsIC",
    # Sequential ensemble models - slow for large datasets
    "AdaBoostRegressor",
    "BaggingRegressor",
    # O(n×m) prediction time - slow for large test sets
    "KNeighborsRegressor",
    # Unbounded tree depth - very slow on high-dim data
    "DecisionTreeRegressor",
    "ExtraTreeRegressor",
    "ExtraTreesRegressor",
]

# Tuple of (name, class)
CLASSIFIERS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], ClassifierMixin) and (est[0] not in removed_classifiers))
]
CLASSIFIER_DICT = {model[0]: model[1] for model in CLASSIFIERS}

"""
CLASSIFIERS = [
    'LogisticRegression',
    'SVC', 
    'PassiveAggressiveClassifier',
    'LabelPropagation',
    'LabelSpreading',
    'RandomForestClassifier',
    'GradientBoostingClassifier', 
    'QuadraticDiscriminantAnalysis',
    'HistGradientBoostingClassifier',
    'RidgeClassifier',
    'AdaBoostClassifier',
    'ExtraTreesClassifier',
    'KNeighborsClassifier',
    'BaggingClassifier',
    'BernoulliNB',
    'LinearDiscriminantAnalysis',
    'GaussianNB',
    'NuSVC',
    'DecisionTreeClassifier',
    'NearestCentroid',
    'ExtraTreeClassifier',
    'CheckingClassifier',
    'DummyClassifier'
]
"""

REGRESSORS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], RegressorMixin) and (est[0] not in removed_regressors))
]
REGRESSOR_DICT = {model[0]: model[1] for model in REGRESSORS}

ALL_MODELS = CLASSIFIERS + REGRESSORS
ALL_MODEL_DICT = {model[0]: model[1] for model in ALL_MODELS}

"""
REGRESSORS = [
    'ExtraTreesRegressor',
    'Lasso',
    'PassiveAggressiveRegressor',
    'SGDRegressor',
    'Ridge',
    'BayesianRidge',
    'TransformedTargetRegressor',
    'LinearRegression',
    'Lars',
    'HuberRegressor',
    'RandomForestRegressor',
    'AdaBoostRegressor',
    'LGBMRegressor',
    'HistGradientBoostingRegressor',
    'PoissonRegressor',
    'ElasticNet',
    'KNeighborsRegressor',
    'OrthogonalMatchingPursuit',
    'BaggingRegressor',
    'GradientBoostingRegressor',
    'TweedieRegressor',
    'XGBRegressor',
    'GammaRegressor',
    'RANSACRegressor',
    'ExtraTreeRegressor',
    'NuSVR',
    'SVR',
    'DummyRegressor',
    'DecisionTreeRegressor',
    'GaussianProcessRegressor',
    'MLPRegressor',
    'KernelRidge'
]
"""

REGRESSORS.append(("XGBRegressor", xgboost.XGBRegressor))
REGRESSORS.append(("LGBMRegressor", lightgbm.LGBMRegressor))
# REGRESSORS.append(('CatBoostRegressor',catboost.CatBoostRegressor))

CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))
CLASSIFIERS.append(("LGBMClassifier", lightgbm.LGBMClassifier))
# CLASSIFIERS.append(('CatBoostClassifier',catboost.CatBoostClassifier))

# Update dicts with XGB and LGBM
CLASSIFIER_DICT["XGBClassifier"] = xgboost.XGBClassifier
CLASSIFIER_DICT["LGBMClassifier"] = lightgbm.LGBMClassifier
REGRESSOR_DICT["XGBRegressor"] = xgboost.XGBRegressor
REGRESSOR_DICT["LGBMRegressor"] = lightgbm.LGBMRegressor
ALL_MODEL_DICT["XGBClassifier"] = xgboost.XGBClassifier
ALL_MODEL_DICT["LGBMClassifier"] = lightgbm.LGBMClassifier
ALL_MODEL_DICT["XGBRegressor"] = xgboost.XGBRegressor
ALL_MODEL_DICT["LGBMRegressor"] = lightgbm.LGBMRegressor

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_transformer_low = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

categorical_transformer_high = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        # 'OrdianlEncoder' Raise a ValueError when encounters an unknown value. Check https://github.com/scikit-learn/scikit-learn/pull/13423
        ("encoding", OrdinalEncoder()),
    ]
)


# Helper function
def get_card_split(df: pd.DataFrame, cols: pd.Index, n: int=11) -> tuple[pd.Index, pd.Index]:
    """
    Splits categorical columns into 2 lists based on cardinality (i.e # of unique values)
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame from which the cardinality of the columns is calculated.
    cols : list-like
        Categorical columns to list
    n : int, optional (default=11)
        The value of 'n' will be used to split columns.
    Returns
    -------
    card_low : list-like
        Columns with cardinality < n
    card_high : list-like
        Columns with cardinality >= n
    """
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high


# Helper class for performing classification
class LazyClassifier:
    """
    This module helps in fitting to all the classification algorithms that are available in Scikit-learn
    Parameters
    ----------
    verbose : int, optional (default=0)
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    ignore_warnings : bool, optional (default=True)
        When set to True, the warning related to algorigms that are not able to run are ignored.
    custom_metric : function, optional (default=None)
        When function is provided, models are evaluated based on the custom evaluation metric provided.
    prediction : bool, optional (default=False)
        When set to True, the predictions of all the models models are returned as dataframe.
    classifiers : list, optional (default="all")
        When function is provided, trains the chosen classifier(s).
    """

    def __init__(
        self,
        verbose: int=0,
        ignore_warnings: bool=True,
        custom_metric: Callable[[np.ndarray, np.ndarray], float] | None=None,
        predictions: bool=False,
        random_state: int | None=None,
        n_jobs: int=1,
        classifiers: str="all",
    ) -> None:
        # custom_metric: (n,) y_true and (n,) y_pred in, a scalar out
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state or get_global_seed()
        self.n_jobs = n_jobs
        self.classifiers = classifiers

    def fit(self, X_train: np.ndarray | pd.DataFrame, X_test: np.ndarray | pd.DataFrame, y_train: np.ndarray, y_test: np.ndarray) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Fit Classification algorithms to X_train and y_train, predict and score on X_test, y_test.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        scores : Pandas DataFrame
            Returns metrics of all the models in a Pandas DataFrame.
        predictions : Pandas DataFrame
            Returns predictions of all the models in a Pandas DataFrame.
        """
        # X_train: (n_train, f), X_test: (n_test, f), y_train: (n_train,), y_test: (n_test,)
        accuracy_scores = []
        balanced_accuracy_scores = []
        roc_auc_scores = []
        f1_scores = []
        names = []
        fit_times = []
        predictions = {}

        if self.custom_metric is not None:
            custom_metric_scores = []

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = get_card_split(
            X_train, categorical_features
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )
        
        # Precompute preprocessing once for all models (major optimization for large datasets)
        print_message("Preprocessing data once for all models...")
        preprocess_start = time.time()
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        print_message(f"Preprocessing completed in {time.time() - preprocess_start:.1f}s")

        if self.classifiers == "all":
            self.classifiers = CLASSIFIERS
        else:
            try:
                temp_list = []
                for classifier in self.classifiers:
                    full_name = (classifier.__name__, classifier)
                    temp_list.append(full_name)
                self.classifiers = temp_list
            except (TypeError, AttributeError) as exception:
                print_message(exception)
                print_message("Invalid Classifier(s)")

        # Track failed models
        failed_models = []
        total_start = time.time()

        for name, model in tqdm(self.classifiers, desc="Training classifiers"):
            print_message(f"Starting {name}...")
            start = time.time()
            try:
                # Build model kwargs
                model_kwargs = {}
                if "random_state" in model().get_params():
                    model_kwargs["random_state"] = self.random_state
                # Enable parallelization for models that support it
                if "n_jobs" in model().get_params():
                    model_kwargs["n_jobs"] = self.n_jobs
                # Enable verbose for boosting models to show iteration progress
                if name in ("XGBClassifier", "LGBMClassifier"):
                    model_kwargs["verbose"] = 1
                
                # Train directly on preprocessed data (no Pipeline needed)
                clf = model(**model_kwargs)
                clf.fit(X_train_transformed, y_train)
                self.models[name] = clf
                y_pred = clf.predict(X_test_transformed)
                accuracy = accuracy_score(y_test, y_pred, normalize=True)
                b_accuracy = balanced_accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                try:
                    roc_auc = roc_auc_score(y_test, y_pred)
                except Exception as exception:  # noqa: broad-except any estimator may fail to fit or score; the failure is reported and the other models still run
                    roc_auc = None
                    if self.ignore_warnings is False:
                        print_message("ROC AUC couldn't be calculated for " + name)
                        print_message(exception)
                fit_time = time.time() - start
                names.append(name)
                accuracy_scores.append(accuracy)
                balanced_accuracy_scores.append(b_accuracy)
                roc_auc_scores.append(roc_auc)
                f1_scores.append(f1)
                fit_times.append(fit_time)
                
                print_message(f"  {name} completed in {fit_time:.1f}s | Acc: {accuracy:.3f} | F1: {f1:.3f}")

                if self.custom_metric is not None:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    custom_metric_scores.append(custom_metric)

                if self.predictions:
                    predictions[name] = y_pred

            except Exception as exception:  # noqa: broad-except any estimator may fail to fit or score; the failure is reported and the other models still run
                failed_models.append(name)
                if self.ignore_warnings is False:
                    print_message(f'\n{name} model failed to execute')
                    print_message(exception)

        if self.custom_metric is None:
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": accuracy_scores,
                    "Balanced Accuracy": balanced_accuracy_scores,
                    "ROC AUC": roc_auc_scores,
                    "F1 Score": f1_scores,
                    "Time Taken": fit_times,
                }
            )
        else:
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": accuracy_scores,
                    "Balanced Accuracy": balanced_accuracy_scores,
                    "ROC AUC": roc_auc_scores,
                    "F1 Score": f1_scores,
                    "Custom Metric": custom_metric_scores,
                    "Time Taken": fit_times,
                }
            )
        scores = scores.sort_values(by="Balanced Accuracy", ascending=False).set_index(
            "Model"
        )

        # Print summary
        total_time = time.time() - total_start
        n_success = len(names)
        n_failed = len(failed_models)
        best_model = scores.index[0] if len(scores) > 0 else "N/A"
        best_score = scores["Balanced Accuracy"].iloc[0] if len(scores) > 0 else 0

        if self.verbose > 0:
            # Full table + failed models
            summary = f"\nLazyClassifier Results ({n_success} succeeded, {n_failed} failed, {total_time:.1f}s)\n"
            summary += scores.to_string()
            if failed_models:
                summary += f"\n\nFailed: {', '.join(failed_models)}"
            print_message(summary)
        else:
            # 1-line summary
            print_message(f"Completed {n_success + n_failed} classifiers in {total_time:.1f}s | {n_success} succeeded, {n_failed} failed | Best: {best_model} ({best_score:.2f})")

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions)
            return scores, predictions_df
        return scores

    def provide_models(self, X_train: np.ndarray | pd.DataFrame, X_test: np.ndarray | pd.DataFrame, y_train: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
        """
        This function returns all the model objects trained in fit function.
        If fit is not called already, then we call fit and then return the models.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        models: dict-object,
            Returns a dictionary with each model pipeline as value 
            with key as name of models.
        """
        # X_train: (n_train, f), X_test: (n_test, f), y_train: (n_train,), y_test: (n_test,)
        if len(self.models.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test)

        return self.models


def adjusted_rsquared(r2: float, n: int, p: int) -> float:
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))


# Helper class for performing classification
class LazyRegressor:
    """
    This module helps in fitting regression models that are available in Scikit-learn
    Parameters
    ----------
    verbose : int, optional (default=0)
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    ignore_warnings : bool, optional (default=True)
        When set to True, the warning related to algorigms that are not able to run are ignored.
    custom_metric : function, optional (default=None)
        When function is provided, models are evaluated based on the custom evaluation metric provided.
    prediction : bool, optional (default=False)
        When set to True, the predictions of all the models models are returned as dataframe.
    regressors : list, optional (default="all")
        When function is provided, trains the chosen regressor(s).
    """

    def __init__(
        self,
        verbose: int=0,
        ignore_warnings: bool=True,
        custom_metric: Callable[[np.ndarray, np.ndarray], float] | None=None,
        predictions: bool=False,
        random_state: int | None=None,
        n_jobs: int=1,
        regressors: str | Sequence[type[BaseEstimator]]="all",
    ) -> None:
        # custom_metric: (n,) y_true and (n,) y_pred in, a scalar out
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state or get_global_seed()
        self.n_jobs = n_jobs
        self.regressors = regressors

    def fit(self, X_train: np.ndarray | pd.DataFrame, X_test: np.ndarray | pd.DataFrame, y_train: np.ndarray, y_test: np.ndarray) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Fit Regression algorithms to X_train and y_train, predict and score on X_test, y_test.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        scores : Pandas DataFrame
            Returns metrics of all the models in a Pandas DataFrame.
        predictions : Pandas DataFrame
            Returns predictions of all the models in a Pandas DataFrame.
        """
        # X_train: (n_train, f), X_test: (n_test, f), y_train: (n_train,), y_test: (n_test,)
        r_squared_scores = []
        adjusted_r_squared_scores = []
        rmse_scores = []
        # WIN = []
        names = []
        fit_times = []
        predictions = {}

        if self.custom_metric:
            custom_metric_scores = []

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = get_card_split(
            X_train, categorical_features
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )
        
        # Precompute preprocessing once for all models (major optimization for large datasets)
        print_message("Preprocessing data once for all models...")
        preprocess_start = time.time()
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        print_message(f"Preprocessing completed in {time.time() - preprocess_start:.1f}s")

        if self.regressors == "all":
            self.regressors = REGRESSORS
        else:
            try:
                temp_list = []
                for regressor in self.regressors:
                    full_name = (regressor.__name__, regressor)
                    temp_list.append(full_name)
                self.regressors = temp_list
            except (TypeError, AttributeError) as exception:
                print_message(exception)
                print_message("Invalid Regressor(s)")

        # Track failed models
        failed_models = []
        total_start = time.time()

        for name, model in tqdm(self.regressors, desc="Training regressors"):
            print_message(f"Starting {name}...")
            start = time.time()
            try:
                # Build model kwargs
                model_kwargs = {}
                if "random_state" in model().get_params():
                    model_kwargs["random_state"] = self.random_state
                # Enable parallelization for models that support it
                if "n_jobs" in model().get_params():
                    model_kwargs["n_jobs"] = self.n_jobs
                # Enable verbose for boosting models to show iteration progress
                if name in ("XGBRegressor", "LGBMRegressor"):
                    model_kwargs["verbose"] = 1
                
                # Train directly on preprocessed data (no Pipeline needed)
                reg = model(**model_kwargs)
                reg.fit(X_train_transformed, y_train)
                self.models[name] = reg
                y_pred = reg.predict(X_test_transformed)

                r_squared = r2_score(y_test, y_pred)
                adj_rsquared = adjusted_rsquared(
                    r_squared, X_test.shape[0], X_test.shape[1]
                )
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                fit_time = time.time() - start
                names.append(name)
                r_squared_scores.append(r_squared)
                adjusted_r_squared_scores.append(adj_rsquared)
                rmse_scores.append(rmse)
                fit_times.append(fit_time)
                
                print_message(f"  {name} completed in {fit_time:.1f}s | R²: {r_squared:.3f} | RMSE: {rmse:.3f}")

                if self.custom_metric:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    custom_metric_scores.append(custom_metric)

                if self.predictions:
                    predictions[name] = y_pred

            except Exception as exception:  # noqa: broad-except any estimator may fail to fit or score; the failure is reported and the other models still run
                failed_models.append(name)
                if self.ignore_warnings is False:
                    print_message(f'\n{name} model failed to execute')
                    print_message(exception)

        scores = {
            "Model": names,
            "Adjusted R-Squared": adjusted_r_squared_scores,
            "R-Squared": r_squared_scores,
            "RMSE": rmse_scores,
            "Time Taken": fit_times,
        }

        if self.custom_metric:
            scores[self.custom_metric.__name__] = custom_metric_scores

        scores = pd.DataFrame(scores)
        scores = scores.sort_values(by="Adjusted R-Squared", ascending=False).set_index(
            "Model"
        )

        # Print summary
        total_time = time.time() - total_start
        n_success = len(names)
        n_failed = len(failed_models)
        best_model = scores.index[0] if len(scores) > 0 else "N/A"
        best_score = scores["Adjusted R-Squared"].iloc[0] if len(scores) > 0 else 0

        if self.verbose > 0:
            # Full table + failed models
            summary = f"\nLazyRegressor Results ({n_success} succeeded, {n_failed} failed, {total_time:.1f}s)\n"
            summary += scores.to_string()
            if failed_models:
                summary += f"\n\nFailed: {', '.join(failed_models)}"
            print_message(summary)
        else:
            # 1-line summary
            print_message(f"Completed {n_success + n_failed} regressors in {total_time:.1f}s | {n_success} succeeded, {n_failed} failed | Best: {best_model} ({best_score:.2f})")

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions)
            return scores, predictions_df
        return scores

    def provide_models(self, X_train: np.ndarray | pd.DataFrame, X_test: np.ndarray | pd.DataFrame, y_train: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
        """
        This function returns all the model objects trained in fit function.
        If fit is not called already, then we call fit and then return the models.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        models: dict-object,
            Returns a dictionary with each model pipeline as value 
            with key as name of models.
        """
        # X_train: (n_train, f), X_test: (n_test, f), y_train: (n_train,), y_test: (n_test,)
        if len(self.models.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test)

        return self.models
