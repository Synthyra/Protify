"""Split conformal predictive intervals, optionally reweighted for a covariate shift.

A model gives each row a prediction and a predictive spread, such as an ensemble's standard
deviation or a head's predicted sigma. Split conformal scores held-out calibration rows by their
residual divided by the spread, takes a finite-sample quantile of those scores, and scales each
test row's spread by it. The interval width therefore tracks where the model is unsure rather
than being constant, and when calibration and test rows are exchangeable, each interval holds
the truth with probability at least `1 - alpha`.

The guarantee is marginal over rows, not simultaneous over a set of them. A level needs at least
`ceil(1 / alpha) - 1` calibration rows; below that the quantile, and so every interval, is
infinite.

Promoted from enzyme_loop's `uncertainty/conformal.py`, which now imports these and keeps its
design-specific weights and levels.
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    """Lower and upper predictive bounds, plus the level they were built at."""

    lower: np.ndarray  # (m,)
    upper: np.ndarray  # (m,)
    alpha: float

    def __post_init__(self) -> None:
        if self.lower.shape != self.upper.shape:
            raise ValueError("interval bounds must share a shape.")

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower  # (m,)

    def covers(self, truth: np.ndarray) -> np.ndarray:
        """Whether each interval contains the given value, shape ``(m,)``."""
        # truth: (m,)
        return (truth >= self.lower) & (truth <= self.upper)  # (m,)


def empirical_coverage(interval: Interval, truth: np.ndarray) -> float:
    """Fraction of intervals containing the truth, the only honest check of a nominal level."""
    # truth: (m,)
    return float(interval.covers(truth).mean())


def conformal_quantile(scores: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> float:
    """The ``(1 - alpha)`` conformal quantile of calibration scores, optionally weighted.

    Unweighted, this is the ``ceil((n + 1)(1 - alpha))``-th order statistic, which is what makes
    split conformal finite-sample valid rather than merely asymptotic, and it is infinite when
    that rank exceeds ``n``. Weighted, it is the smallest score whose normalized cumulative
    weight reaches ``1 - alpha``, as for calibration rows reweighted toward a shifted test
    distribution. The weights are normalized over the calibration rows alone, with no mass for
    the test row, so equal weights give the empirical quantile, which can sit one order
    statistic below the unweighted one.
    """
    # scores: (...), read flattened as (n,); weights: (...), the same n entries
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)  # (n,)
    n = scores.shape[0]
    if n == 0:
        return float("inf")

    if weights is None:
        rank = int(np.ceil((n + 1) * (1.0 - alpha)))
        if rank > n:
            return float("inf")

        return float(np.sort(scores)[rank - 1])

    weights = np.asarray(weights, dtype=np.float64).reshape(-1)  # (n,)
    if weights.shape != scores.shape:
        raise ValueError("weights must align with scores.")
    if float(weights.sum()) <= 0.0:
        return float("inf")

    order = np.argsort(scores)  # (n,)
    normalized = weights[order] / weights.sum()  # (n,)
    cumulative = np.cumsum(normalized)  # (n,)
    reached = np.nonzero(cumulative >= 1.0 - alpha)[0]  # (n_reached,), sorted positions at or past 1 - alpha
    if reached.size == 0:
        return float("inf")

    return float(scores[order][reached[0]])


def split_conformal(
    calibration_mean: np.ndarray,  # (n_cal,)
    calibration_truth: np.ndarray,  # (n_cal,)
    calibration_spread: np.ndarray,  # (n_cal,)
    test_mean: np.ndarray,  # (m,)
    test_spread: np.ndarray,  # (m,)
    alpha: float = 0.1,
    spread_floor: float = 0.05,
    calibration_weights: np.ndarray | None = None,  # (n_cal,)
) -> Interval:
    """Locally weighted split conformal, optionally reweighted for a covariate shift.

    Dividing the residual by the predictive spread lets the interval widen where the model is
    unsure. The floor keeps a collapsed spread from producing an unbounded score.
    """
    scale = np.asarray(calibration_spread, dtype=np.float64) + spread_floor  # (n_cal,)
    scores = np.abs(calibration_truth - calibration_mean) / scale  # (n_cal,)
    quantile = conformal_quantile(scores, alpha, calibration_weights)

    half_width = quantile * (np.asarray(test_spread, dtype=np.float64) + spread_floor)  # (m,)

    return Interval(lower=test_mean - half_width, upper=test_mean + half_width, alpha=alpha)  # bounds (m,)


def calibration_size_for_alpha(alpha: float) -> int:
    """Smallest calibration set that can express a level, since below it the quantile is infinite."""

    return int(np.ceil(1.0 / alpha)) - 1
