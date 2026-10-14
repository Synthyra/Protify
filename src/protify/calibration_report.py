"""A small calibration analysis, for a dataset that ships a calibration split.

A probe trained on train and selected on validation predicts the calibration split, and the
test split shows what calibrating on it buys:

  single-label classification   one temperature fitted on calibration, and test's top-label ECE
                                and NLL before and after it; then split-conformal prediction sets
                                at 90%, scored 1 - p of the true class on the scaled calibration
                                rows, with their coverage and mean size on test
  regression                    split-conformal intervals at 90% on the absolute residual, with
                                their coverage and mean width on test

Multilabel and tokenwise tasks get none, and a dataset without the split runs as it always did.
Nothing here is fitted on test, and nothing changes the probe or the metrics test reports.
Every number comes from `foundry.eval` and `foundry.metrics`.
"""

from __future__ import annotations

import math
import numpy as np

from foundry.eval.conformal import conformal_quantile, empirical_coverage, split_conformal
from foundry.eval.temperature import fit_temperature
from foundry.metrics import expected_calibration_error
from .metrics import softmax


CONFORMAL_LEVEL = 0.9
ECE_BINS = 15
# A true-class probability is clipped this far from zero before its logarithm.
PROBABILITY_FLOOR = 1e-12
REGRESSION_TASKS = ("regression", "sigmoid_regression")


def supports_calibration_report(task_type: str) -> bool:
    """Whether a task gets the report: single-label classification and regression do."""
    return task_type == "singlelabel" or task_type in REGRESSION_TASKS


def calibration_report(
    task_type: str,
    calibration_outputs: np.ndarray,
    calibration_labels: np.ndarray,
    test_outputs: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, float]:
    """The report's metrics, each named `calibration_*`, from the probe's raw outputs on both splits.

    Outputs are logits, `(n, c)`, for classification and predicted values, `(n,)` or `(n, 1)`, for
    regression. A conformal quantile needs at least 9 calibration rows at 90%; with fewer, the
    sets or intervals are left out rather than reported infinite.
    """
    if task_type == "singlelabel":
        return _classification_report(calibration_outputs, calibration_labels, test_outputs, test_labels)
    if task_type in REGRESSION_TASKS:
        return _regression_report(calibration_outputs, calibration_labels, test_outputs, test_labels)
    raise ValueError(f"the calibration report covers single-label classification and regression, not {task_type!r}")


def _top_label_ece(probs: np.ndarray, labels: np.ndarray) -> float:
    # probs: (n, c); labels: (n,)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)  # (n,)
    return expected_calibration_error(correct, probs.max(axis=1), ECE_BINS)


def _negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    # probs: (n, c); labels: (n,)
    true_class = np.clip(probs[np.arange(labels.shape[0]), labels], PROBABILITY_FLOOR, 1.0)  # (n,)
    return float(-np.log(true_class).mean())


def _classification_report(
    calibration_logits: np.ndarray, calibration_labels: np.ndarray, test_logits: np.ndarray, test_labels: np.ndarray,
) -> dict[str, float]:
    calibration_logits = np.asarray(calibration_logits, dtype=np.float64)  # (n_cal, c)
    calibration_labels = np.asarray(calibration_labels).reshape(-1).astype(np.int64)  # (n_cal,)
    test_logits = np.asarray(test_logits, dtype=np.float64)  # (n_test, c)
    test_labels = np.asarray(test_labels).reshape(-1).astype(np.int64)  # (n_test,)

    temperature = fit_temperature(calibration_logits, calibration_labels, multiclass=True)
    test_probs = softmax(test_logits)  # (n_test, c)
    scaled_test = softmax(test_logits / temperature)  # (n_test, c)
    report = {
        "calibration_rows": float(calibration_labels.shape[0]),
        "calibration_temperature": float(temperature),
        "calibration_ece": _top_label_ece(test_probs, test_labels),
        "calibration_ece_scaled": _top_label_ece(scaled_test, test_labels),
        "calibration_nll": _negative_log_likelihood(test_probs, test_labels),
        "calibration_nll_scaled": _negative_log_likelihood(scaled_test, test_labels),
    }

    # A class joins a row's set when 1 - p is at most the calibration rows' conformal quantile of it.
    scaled_calibration = softmax(calibration_logits / temperature)  # (n_cal, c)
    scores = 1.0 - scaled_calibration[np.arange(calibration_labels.shape[0]), calibration_labels]  # (n_cal,)
    threshold = conformal_quantile(scores, 1.0 - CONFORMAL_LEVEL)
    if math.isfinite(threshold):
        sets = (1.0 - scaled_test) <= threshold  # (n_test, c)
        report["calibration_set_coverage_90"] = float(sets[np.arange(test_labels.shape[0]), test_labels].mean())
        report["calibration_set_size_90"] = float(sets.sum(axis=1).mean())
    return report


def _regression_report(
    calibration_predictions: np.ndarray, calibration_labels: np.ndarray, test_predictions: np.ndarray, test_labels: np.ndarray,
) -> dict[str, float]:
    calibration_predictions = np.asarray(calibration_predictions, dtype=np.float64).reshape(-1)  # (n_cal,)
    calibration_labels = np.asarray(calibration_labels, dtype=np.float64).reshape(-1)  # (n_cal,)
    test_predictions = np.asarray(test_predictions, dtype=np.float64).reshape(-1)  # (n_test,)
    test_labels = np.asarray(test_labels, dtype=np.float64).reshape(-1)  # (n_test,)

    report = {"calibration_rows": float(calibration_labels.shape[0])}
    # A zero spread with a floor of one makes the score the absolute residual, so every interval has one width.
    interval = split_conformal(
        calibration_predictions, calibration_labels, np.zeros_like(calibration_predictions),
        test_predictions, np.zeros_like(test_predictions), alpha=1.0 - CONFORMAL_LEVEL, spread_floor=1.0,
    )
    if np.isfinite(interval.width).all():
        report["calibration_interval_coverage_90"] = empirical_coverage(interval, test_labels)
        report["calibration_interval_width_90"] = float(interval.width.mean())
    return report
