"""Choosing estimator probe hyperparameters against the validation split.

The wandb sweep in `hyperopt_utils` searches every probe family and needs a project, an
account, and a network round trip for each trial. An estimator probe needs none of that:
its feature matrices are already in memory, one fit is seconds to minutes, and the search
is over half a dozen numbers. Optuna's TPE sampler runs the same search in process and
writes nothing outside the run.

Tuning matters more here than the defaults suggest. `ESTIMATOR_DEFAULTS` was selected on
an 8192-wide solubility task, and carrying it unchanged to a pair row sixteen to thirty
two times wider was measured to cost most of the gap between a sparse autoencoder feature
matrix and a dense one.

The search never reads the test split. It refits through `fit_and_score_valid`, which is
the same fit the final model uses, so a chosen configuration reproduces exactly.
"""

from __future__ import annotations

import numpy as np

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .estimator_probe import (
    ESTIMATOR_DEFAULTS,
    ESTIMATOR_PROBE_TYPES,
    REGRESSION_TASK_TYPES,
    EstimatorProbeArguments,
    fit_and_score_valid,
)
from ..console_output import print_message


# The default objective is the metric the results table reports, because a search scored on
# one metric and reported on another can move the reported one backwards.
#
# This started as `pr_auc`, on the reasoning that MCC reads one fixed 0.5 threshold and so
# hides a ranking improvement that has not yet crossed it. That reasoning is right about
# MCC and wrong about the search: `scale_pos_weight` shifts every predicted probability,
# which a threshold-free metric cannot see and a thresholded one is punished by. On
# bernett_gold_ppi, which is 50.0% positive, twelve trials against validation AUPRC chose
# `scale_pos_weight` 0.33 and took test MCC from 0.3000 down to 0.2227, below the untuned
# defaults it started from.
#
# `pr_auc` is still the right objective when the positive class is rare and ranking is what
# you ship. Ask for it with --tune_metric.
DEFAULT_TUNE_METRIC_CLS = "mcc"
DEFAULT_TUNE_METRIC_REG = "spearman_rho"

# Error metrics and p-values improve downward; every other reported metric improves
# upward. Naming the small set is safer than naming the large one, because a metric added
# to `metrics.py` later is almost certainly one to maximize.
MINIMIZED_METRICS = frozenset({"mse", "mae", "rmse", "spear_pval", "pear_pval", "loss"})


@dataclass
class EstimatorSearchArguments:
    """What one hyperparameter search is allowed to do."""

    n_trials: int = 40
    metric: str = ""  # empty selects the task default
    timeout: float | None = None
    seed: int = 42
    # Fits per trial, averaged into the objective. One is cheapest and is what a search
    # over a large validation split needs. More than one is for a small one, where a
    # configuration can win a trial on subsampling luck rather than on merit, and the
    # search then reports a validation score the test split does not reproduce.
    n_seeds: int = 1

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError(f"--tune_trials must be at least 1; got {self.n_trials}.")
        if self.n_seeds < 1:
            raise ValueError(f"--tune_seeds must be at least 1; got {self.n_seeds}.")


def default_tune_metric(task_type: str) -> str:
    return DEFAULT_TUNE_METRIC_REG if task_type in REGRESSION_TASK_TYPES else DEFAULT_TUNE_METRIC_CLS


def is_binary_classification(args: EstimatorProbeArguments) -> bool:
    """Whether the task has one label taking two values, which is what a positive-class
    weight is defined against."""
    return args.task_type == "singlelabel" and args.num_labels <= 2


def _xgboost_space(trial: Any, args: EstimatorProbeArguments) -> dict[str, Any]:
    """XGBoost hyperparameters worth searching on wide, sparse, non-negative features.

    `n_estimators` is absent on purpose: early stopping against the validation split
    already picks the count, and searching it would spend trials rediscovering that.
    `max_bin` is absent because it is a memory decision the machine makes, not a quality
    decision, and changing it rebins the whole matrix on every trial.
    """
    space = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 32, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        # Wide codebooks tolerate very low column sampling, and it is the strongest lever
        # on fit time, so the range reaches further down than a dense embedding needs.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.02, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100.0, log=True),
        # The untuned model has no L1 penalty, which a log range cannot hold, so the penalty is
        # switched on before its strength is searched; the warm start switches it off.
        "reg_alpha": (
            trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)
            if trial.suggest_categorical("l1_penalty", [False, True]) else 0.0
        ),
    }
    if is_binary_classification(args):
        # Only binary classification reads it, and the objective this searches against,
        # area under the precision-recall curve, is the metric it moves most when the
        # positive class is rare. A balanced task simply lands near 1.
        space["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 0.1, 10.0, log=True)

    return space


def _lightgbm_space(trial: Any, args: EstimatorProbeArguments) -> dict[str, Any]:
    space = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 511, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 64, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.02, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        # LightGBM bags rows only every `subsample_freq` iterations and never by default, so
        # without this the search spent trials on a subsample nothing read. At subsample 1.0,
        # the untuned value, every iteration still reads every row.
        "subsample_freq": 1,
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100.0, log=True),
    }
    if is_binary_classification(args):
        space["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 0.1, 10.0, log=True)

    return space


def _random_forest_space(trial: Any, args: EstimatorProbeArguments) -> dict[str, Any]:
    """A forest has no early stopping, so its tree count is part of the search."""
    space = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, log=True),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.1]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 16, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [None, 16, 32]),
    }
    if args.task_type not in REGRESSION_TASK_TYPES:
        # A forest has no positive-class weight, so the equivalent knob is the per-class
        # sample weighting scikit-learn applies before splitting.
        space["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])

    return space


SEARCH_SPACES: dict[str, Callable[[Any, EstimatorProbeArguments], dict[str, Any]]] = {
    "xgboost": _xgboost_space,
    "lightgbm": _lightgbm_space,
    "random_forest": _random_forest_space,
}


class _NameRecorder:
    """A trial stand-in that reports which parameters a search space asks for.

    Reading the names off the space itself keeps them from drifting out of step with a
    second hand-written list. Every method returns the low end of its range, which is
    never used: the recorder is run once and its return values discarded.
    """

    def __init__(self) -> None:
        self.names: list[str] = []

    def suggest_float(self, name: str, low: float, high: float, **kwargs: Any) -> float:
        self.names.append(name)
        return low

    def suggest_int(self, name: str, low: int, high: int, **kwargs: Any) -> int:
        self.names.append(name)
        return low

    def suggest_categorical(self, name: str, choices: Any) -> Any:
        self.names.append(name)
        return choices[0]


def search_space_names(args: EstimatorProbeArguments) -> list[str]:
    """Every parameter this task's search space suggests, in the order it asks.

    The space depends on the task as well as the probe type, because a positive-class
    weight is only defined for binary classification.
    """
    recorder = _NameRecorder()
    SEARCH_SPACES[args.probe_type](recorder, args)
    return recorder.names


# What an untuned fit uses for each searched parameter `ESTIMATOR_DEFAULTS` leaves to the library:
# the library's own default, or for `l1_penalty` the switch that leaves XGBoost's `reg_alpha` at its
# default of 0.
LIBRARY_DEFAULTS: dict[str, dict[str, Any]] = {
    "xgboost": {"l1_penalty": False, "scale_pos_weight": 1.0},
    "lightgbm": {"subsample": 1.0, "scale_pos_weight": 1.0},
    "random_forest": {"max_depth": None, "class_weight": None},
}


def default_trial_params(args: EstimatorProbeArguments) -> dict[str, Any]:
    """The untuned configuration, as values for every parameter this task's search space asks for.

    Enqueued as the first trial so the search starts exactly where an untuned run lands and
    moves away from it only by scoring better on validation. A searched parameter with no
    untuned value raises rather than letting Optuna sample it into the first trial.
    """
    untuned = {**ESTIMATOR_DEFAULTS[args.probe_type], **LIBRARY_DEFAULTS[args.probe_type]}
    names = search_space_names(args)
    unknown = [name for name in names if name not in untuned]
    if unknown:
        raise ValueError(f"{args.probe_type} searches {unknown} with no untuned value; add it to LIBRARY_DEFAULTS.")
    return {name: untuned[name] for name in names}


def tune_estimator_probe(
    args: EstimatorProbeArguments,
    search: EstimatorSearchArguments,
    X_train: Any,
    y_train: np.ndarray,
    X_valid: Any,
    y_valid: np.ndarray,
) -> dict[str, Any]:
    """Search hyperparameters on validation and return the best overrides found.

    The returned dict layers on top of whatever `args.overrides` already carried, so a
    value pinned through --scikit_model_args stays pinned unless the search space names
    it too.
    """
    # X_*: (n_split, d), dense or CSR; y_*: (n_split,) or (n_split, c)
    # y_train: (n_train,) or (n_train, c), y_valid: (n_valid,) or (n_valid, c)
    import optuna

    if args.probe_type not in SEARCH_SPACES:
        raise ValueError(
            f"No search space for probe type {args.probe_type!r}; expected one of "
            f"{ESTIMATOR_PROBE_TYPES}."
        )

    metric = search.metric or default_tune_metric(args.task_type)
    suggest = SEARCH_SPACES[args.probe_type]

    def objective(trial: Any) -> float:
        suggested = suggest(trial, args)
        scores = []
        for offset in range(search.n_seeds):
            trial_args = EstimatorProbeArguments(
                probe_type=args.probe_type,
                task_type=args.task_type,
                num_labels=args.num_labels,
                # The same seeds for every trial, so a configuration cannot win on which
                # seeds it happened to be given.
                seed=args.seed + offset,
                n_jobs=args.n_jobs,
                device=args.device,
                overrides={**args.overrides, **suggested},
            )
            _, valid_metrics = fit_and_score_valid(trial_args, X_train, y_train, X_valid, y_valid)
            if metric not in valid_metrics:
                raise KeyError(
                    f"Tuning metric {metric!r} is not among the validation metrics "
                    f"{sorted(valid_metrics)} this task reports."
                )
            scores.append(float(valid_metrics[metric]))

        return sum(scores) / len(scores)

    # Optuna logs one INFO line per trial by default, which buries the fit output the
    # trials themselves print.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    direction = "minimize" if metric in MINIMIZED_METRICS else "maximize"
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=search.seed),
    )
    study.enqueue_trial(default_trial_params(args), skip_if_exists=True)
    print_message(
        f"Tuning {args.probe_type} over {search.n_trials} trials to {direction} "
        f"validation {metric}"
    )
    study.optimize(objective, n_trials=search.n_trials, timeout=search.timeout)

    completed = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        raise RuntimeError(
            f"--tune_timeout {search.timeout} stopped the search before the first "
            f"{args.probe_type} fit finished, so there is nothing to choose from. Raise "
            f"the timeout, or lower --tune_trials and drop it."
        )

    print_message(f"Searched {len(completed)} of {search.n_trials} trials")
    print_message(f"Best validation {metric}: {study.best_value:.5f}")
    # The space turns the trial's choices, such as the L1 switch, into the estimator's own parameters.
    best = suggest(optuna.trial.FixedTrial(study.best_params), args)
    print_message(f"Best {args.probe_type} hyperparameters: {best}")
    return best
