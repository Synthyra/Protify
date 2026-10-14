"""Split conformal prediction sets and intervals, calibrated on V_cal alone.

Classification builds two kinds of set from one method's probabilities. LAC scores a class by
`1 - p_k` and keeps every class under the calibrated threshold, which gives the smallest sets
on average. APS scores a class by the total probability of every class at least as likely,
which adapts the set size to how spread the prediction is; this is its non-randomized form,
so sets can over-cover. Either way the set is `{k : score(k) <= q}` with `q` the conformal
quantile of the V_cal rows' true-class scores, so exchangeable test rows are covered with
probability at least `1 - alpha`.

Regression scores `|y - mu| / (sigma + floor)` through `foundry.eval.conformal.split_conformal`,
with the floor `1e-3` of the training labels' standard deviation, and a method with no sigma
scores the absolute residual. Levels are coverages `1 - alpha`, such as 0.9.

`set_quantile`, `regression_quantile`, and `regression_interval` expose the thresholds these
rows are built from, so a package can ship them and apply them to rows it has never seen.

Shapes: `n` V_cal rows, `m` test rows, `c` classes.
"""

from __future__ import annotations

import numpy as np

from foundry.eval.conformal import Interval, conformal_quantile, split_conformal


SET_SCORES = ("lac", "aps")
# The regression spread floor, as a fraction of the training labels' standard deviation.
SPREAD_FLOOR_FRACTION = 1e-3


def class_scores(probs: np.ndarray, score: str) -> np.ndarray:
    """Every class's nonconformity score, `(rows, c)`."""
    # probs: (rows, c)
    if score == "lac":
        return 1.0 - probs  # (rows, c)
    if score == "aps":
        # Total probability of the classes at least as likely as class k, per row.
        at_least_as_likely = probs[:, None, :] >= probs[:, :, None]  # (rows, c, c)
        return np.sum(probs[:, None, :] * at_least_as_likely, axis=2)  # (rows, c)
    raise ValueError(f"unknown conformal set score {score!r}; expected one of {SET_SCORES}.")


def true_class_scores(probs: np.ndarray, labels: np.ndarray, score: str) -> np.ndarray:
    # probs: (rows, c), labels: (rows,)
    return class_scores(probs, score)[np.arange(labels.shape[0]), labels]  # (rows,)


def set_quantile(calibration_probs: np.ndarray, calibration_labels: np.ndarray, level: float, score: str) -> float:
    """The threshold on `score` that a class must not exceed to join a set at coverage `level`.

    Infinite when V_cal has too few rows to express the level, and then every set holds every class.
    """
    # calibration_probs: (n, c); calibration_labels: (n,)
    return conformal_quantile(true_class_scores(calibration_probs, calibration_labels, score), 1.0 - level)


def prediction_sets(
    calibration_probs: np.ndarray,
    calibration_labels: np.ndarray,
    test_probs: np.ndarray,
    level: float,
    score: str,
) -> np.ndarray:
    """Membership of every class in each test row's set at coverage `level`, `(m, c)` bool."""
    # calibration_probs: (r, c), calibration_labels: (r,), test_probs: (m, c)
    quantile = set_quantile(calibration_probs, calibration_labels, level, score)
    return class_scores(test_probs, score) <= quantile  # (m, c)


def classification_conformal_rows(
    calibration_probs: np.ndarray,
    calibration_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    levels: tuple[float, ...],
) -> dict[str, np.ndarray]:
    """Per test row, whether each set holds the true class and how many classes it holds."""
    # calibration_probs: (r, c), calibration_labels: (r,), test_probs: (m, c), test_labels: (m,)
    rows: dict[str, np.ndarray] = {}
    for score in SET_SCORES:
        for level in levels:
            sets = prediction_sets(calibration_probs, calibration_labels, test_probs, level, score)  # (m, c)
            rows[f"{score}_coverage@{level}"] = sets[np.arange(test_labels.shape[0]), test_labels].astype(np.float64)  # (m,)
            rows[f"{score}_size@{level}"] = sets.sum(axis=1).astype(np.float64)  # (m,)
    return rows  # each (m,) by metric name


def regression_spread_floor(has_sigma: bool, train_label_std: float) -> float:
    """What is added to every sigma before it divides a residual.

    A method with no sigma takes a zero spread and a unit floor, which scores the absolute
    residual itself.
    """
    return SPREAD_FLOOR_FRACTION * train_label_std if has_sigma else 1.0


def regression_quantile(
    calibration_mean: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_sigma: np.ndarray | None,
    level: float,
    floor: float,
) -> float:
    """The conformal quantile of `|y - mu| / (sigma + floor)` on V_cal, scored as `split_conformal` scores it.

    Infinite when V_cal has too few rows to express the level.
    """
    # calibration_mean: (n,); calibration_labels: (n,); calibration_sigma: (n,) or None
    spread = np.zeros_like(calibration_mean) if calibration_sigma is None else calibration_sigma  # (n,)
    scale = np.asarray(spread, dtype=np.float64) + floor  # (n,)
    scores = np.abs(calibration_labels - calibration_mean) / scale  # (n,)
    return conformal_quantile(scores, 1.0 - level)


def regression_interval(
    test_mean: np.ndarray, test_sigma: np.ndarray | None, quantile: float, floor: float, level: float
) -> Interval:
    """Each row's interval at coverage `level`: the mean plus or minus `quantile * (sigma + floor)`."""
    # test_mean: (m,); test_sigma: (m,) or None
    spread = np.zeros_like(test_mean) if test_sigma is None else test_sigma  # (m,)
    half_width = quantile * (np.asarray(spread, dtype=np.float64) + floor)  # (m,)
    return Interval(lower=test_mean - half_width, upper=test_mean + half_width, alpha=1.0 - level)  # (m,) bounds


def regression_conformal_rows(
    calibration_mean: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_sigma: np.ndarray | None,
    test_mean: np.ndarray,
    test_labels: np.ndarray,
    test_sigma: np.ndarray | None,
    levels: tuple[float, ...],
    train_label_std: float,
) -> dict[str, np.ndarray]:
    """Per test row, whether each interval holds the label and how wide it is."""
    # calibration_mean: (r,), calibration_labels: (r,), calibration_sigma: (r,) or None, test_mean: (m,), test_labels: (m,), test_sigma: (m,) or None
    if calibration_sigma is None:
        # No spread: a zero sigma with a unit floor scores the absolute residual itself.
        calibration_spread, test_spread = np.zeros_like(calibration_mean), np.zeros_like(test_mean)
    else:
        calibration_spread, test_spread = calibration_sigma, test_sigma
    floor = regression_spread_floor(calibration_sigma is not None, train_label_std)

    rows: dict[str, np.ndarray] = {}
    for level in levels:
        interval = split_conformal(
            calibration_mean,
            calibration_labels,
            calibration_spread,
            test_mean,
            test_spread,
            alpha=1.0 - level,
            spread_floor=floor,
        )
        rows[f"conformal_coverage@{level}"] = interval.covers(test_labels).astype(np.float64)  # (m,)
        rows[f"conformal_width@{level}"] = interval.width  # (m,)
    return rows  # each (m,) by metric name
