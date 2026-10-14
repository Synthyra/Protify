"""Temperature and variance maps fitted on many calibration sets drawn from one set of rows.

Scheme T refits every calibrated method on each repeat's own calibration rows, a subset of the
test rows. These fits minimize the objectives `methods.py` minimizes once per seed:

- temperature: the mean cross-entropy of `logits / T`, softmax over classes, or the mean binary
  cross-entropy over every entry, as `foundry.eval.temperature.fit_temperature` defines it;
- variance: the mean Gaussian negative log-likelihood of `N(mean, a v + b)`, as
  `methods.fit_variance_calibration` defines it, with `a = 0` and `b` the mean squared residual
  when there is no variance.

Every fit reads rows of one set, so work that depends only on a row is done once per row rather
than once per fit. The temperature objective is convex in `beta = 1 / T`, and its derivative is
the mean over a fit's rows of per-row slopes. Each row's slope and curvature, evaluated on a grid
of `beta` shared by every fit, therefore give every fit's derivative on that grid through one
matrix product. A coarse grid brackets each fit's root, a finer grid inside that bracket narrows
it, and a cubic Hermite interpolant through the derivative and its slope at the two nearest fine
points places it. The variance objective has a closed-form `b` for each ratio `rho = a / b`,
which leaves one dimension: a shared grid over `log rho` finds each fit's basin, a safeguarded
Newton iteration on the fit's own rows finds its minimum, and `rho = 0`, the constant-variance
model the grid only approaches, is kept where it is lower.

Shapes: `r` fits, `n` rows, `m` calibration rows per fit, `c` classes or labels, `g` grid points.
"""

from __future__ import annotations

import numpy as np

from scipy.special import expit, softmax

from .methods import VARIANCE_FLOOR


# Temperatures stay between 1e-4 and 1e4, so log(beta) stays within these bounds.
LOG_BETA_BOUNDS = (float(np.log(1e-4)), float(np.log(1e4)))
# The coarse grid's spacing in log(beta), its initial half-width in steps, and how far it grows at a time.
COARSE_STEP = 0.25
COARSE_HALF_WIDTH = 4
COARSE_GROWTH = 4
# Fine points per coarse interval: a fine spacing of 0.0125 in log(beta).
FINE_POINTS = 20
# Grid points whose row slopes are held in memory at once.
GRID_BLOCK = 8
ROOT_ITERATIONS = 60
# log(rho * mean variance) is searched over this grid, then refined within one step of its minimum.
LOG_RATIO_GRID = np.arange(-20.0, 20.0 + 0.25, 0.5)  # (g,)
NEWTON_ITERATIONS = 60


def row_weights(fit_rows: np.ndarray, rows: int) -> np.ndarray:
    """How many times each fit reads each row: its calibration rows as a weight matrix."""
    # fit_rows: (r, m) row indices
    fits = fit_rows.shape[0]
    offsets = np.arange(fits)[:, None] * rows  # (r, 1)
    counts = np.bincount((offsets + fit_rows).ravel(), minlength=fits * rows)  # (r * n,)
    return counts.reshape(fits, rows).astype(np.float64)  # (r, n)


class _TemperatureRows:
    """Per-row terms of the temperature objective, at any `beta`."""

    def __init__(self, logits: np.ndarray, labels: np.ndarray, multiclass: bool) -> None:
        # logits: (n, c); labels: (n,) classes or (n, c) in {0, 1}
        self.logits = np.asarray(logits, dtype=np.float64)  # (n, c)
        self.multiclass = multiclass
        if multiclass:
            classes = np.asarray(labels, dtype=np.int64)  # (n,)
            self.true_logit = np.take_along_axis(self.logits, classes[:, None], axis=1)[:, 0]  # (n,)
        else:
            self.targets = np.asarray(labels, dtype=np.float64)  # (n, c)

    def slopes(self, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each row's loss derivative and second derivative in `beta`, at every `beta` given."""
        # beta: (g,)
        scaled = beta[:, None, None] * self.logits[None]  # (g, n, c)
        if self.multiclass:
            probs = softmax(scaled, axis=2)  # (g, n, c)
            expected = np.sum(probs * self.logits[None], axis=2)  # (g, n)
            slope = expected - self.true_logit[None]  # (g, n)
            curvature = np.sum(probs * (self.logits[None] - expected[..., None]) ** 2, axis=2)  # (g, n)
            return slope, curvature  # (g, n), (g, n)

        probs = expit(scaled)  # (g, n, c)
        slope = np.mean((probs - self.targets[None]) * self.logits[None], axis=2)  # (g, n)
        curvature = np.mean(probs * (1.0 - probs) * self.logits[None] ** 2, axis=2)  # (g, n)
        return slope, curvature  # (g, n), (g, n)

    def fit_slopes(self, weights: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Every fit's objective derivative and second derivative in `beta` at each grid point."""
        # weights: (r, n); beta: (g,)
        totals = weights.sum(axis=1, keepdims=True)  # (r, 1)
        first, second = [], []
        for start in range(0, beta.shape[0], GRID_BLOCK):
            slope, curvature = self.slopes(beta[start:start + GRID_BLOCK])  # (g_block, n) each
            first.append(weights @ slope.T)  # (r, g_block)
            second.append(weights @ curvature.T)  # (r, g_block)
        return np.concatenate(first, axis=1) / totals, np.concatenate(second, axis=1) / totals  # (r, g), (r, g)


def _pooled_log_beta(problem: _TemperatureRows) -> float:
    """log(beta) minimizing the objective over every row: the center of the shared grid.

    Newton's method in log(beta), bisecting whenever a step leaves the bracket the derivative's
    sign has established. Only the grid's position depends on it, never a fit's result.
    """
    lower, upper = LOG_BETA_BOUNDS
    log_beta = 0.0
    for _ in range(NEWTON_ITERATIONS):
        beta = np.exp(log_beta)
        slope, curvature = problem.slopes(np.array([beta]))  # (1, n) each
        derivative = beta * float(slope.mean())
        second = derivative + beta ** 2 * float(curvature.mean())
        if derivative < 0.0:
            lower = log_beta
        else:
            upper = log_beta
        step = -derivative / second if second > 0.0 else np.inf
        proposal = log_beta + step
        if not lower <= proposal <= upper:
            proposal = (lower + upper) / 2.0
        if abs(proposal - log_beta) <= 1e-10:
            return proposal
        log_beta = proposal
    return log_beta


def _hermite_roots(
    left: np.ndarray, right: np.ndarray, value: tuple[np.ndarray, np.ndarray], slope: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """The root, in each interval, of the cubic Hermite interpolant through two values and slopes.

    Each interval's left value is negative and its right value is not, so the cubic has a root
    inside it, found by Newton's method on the cubic with bisection as the safeguard. An
    objective flat to rounding, as for calibration rows that a sharp enough temperature
    classifies without error, can leave both values at zero; the search then starts from the
    middle and stays inside the interval.
    """
    # left: (r,), right: (r,), value: (r,) at each end, slope: (r,) at each end
    width = right - left  # (r,)
    left_value, right_value = value  # (r,), (r,)
    left_slope, right_slope = slope[0] * width, slope[1] * width  # (r,), (r,)

    def cubic(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # t: (r,)
        t2, t3 = t * t, t * t * t  # (r,), (r,)
        at = (2 * t3 - 3 * t2 + 1) * left_value + (t3 - 2 * t2 + t) * left_slope + (-2 * t3 + 3 * t2) * right_value + (t3 - t2) * right_slope  # (r,)
        derivative = (6 * t2 - 6 * t) * left_value + (3 * t2 - 4 * t + 1) * left_slope + (6 * t - 6 * t2) * right_value + (3 * t2 - 2 * t) * right_slope  # (r,)
        return at, derivative  # (r,), (r,)

    low, high = np.zeros_like(width), np.ones_like(width)  # (r,), (r,)
    with np.errstate(divide="ignore", invalid="ignore"):
        secant = -left_value / (right_value - left_value)  # (r,)
    t = np.where(np.isfinite(secant), np.clip(secant, 0.0, 1.0), 0.5)  # (r,)
    for _ in range(ROOT_ITERATIONS):
        at, derivative = cubic(t)  # (r,), (r,)
        low = np.where(at < 0.0, t, low)
        high = np.where(at < 0.0, high, t)
        with np.errstate(divide="ignore", invalid="ignore"):
            proposal = t - at / derivative  # (r,)
        proposal = np.where((proposal >= low) & (proposal <= high), proposal, (low + high) / 2.0)  # (r,)
        if np.all(np.abs(proposal - t) <= 1e-15):
            break
        t = proposal
    return left + t * width  # (r,)


def fit_temperatures(logits: np.ndarray, labels: np.ndarray, multiclass: bool, fit_rows: np.ndarray) -> np.ndarray:
    """Each fit's temperature: the `T` whose `logits / T` best explain its rows' labels."""
    # logits: (n, c); labels: (n,) classes or (n, c) in {0, 1}; fit_rows: (r, m) row indices
    problem = _TemperatureRows(logits, labels, multiclass)
    weights = row_weights(fit_rows, problem.logits.shape[0])  # (r, n)
    lower, upper = LOG_BETA_BOUNDS

    # The coarse grid grows toward any fit whose derivative has not changed sign within it.
    center = _pooled_log_beta(problem)
    grid = np.clip(center + COARSE_STEP * np.arange(-COARSE_HALF_WIDTH, COARSE_HALF_WIDTH + 1), lower, upper)  # (g,)
    grid = np.unique(grid)  # (g,)
    derivative, _ = problem.fit_slopes(weights, np.exp(grid))  # (r, g)
    while True:
        grow_left = bool(np.any(derivative[:, 0] > 0.0)) and grid[0] > lower
        grow_right = bool(np.any(derivative[:, -1] < 0.0)) and grid[-1] < upper
        if not (grow_left or grow_right):
            break
        if grow_left:
            points = np.unique(np.clip(grid[0] - COARSE_STEP * np.arange(COARSE_GROWTH, 0, -1), lower, upper))  # (g_new,)
            grid = np.concatenate([points, grid])
            derivative = np.concatenate([problem.fit_slopes(weights, np.exp(points))[0], derivative], axis=1)
        if grow_right:
            points = np.unique(np.clip(grid[-1] + COARSE_STEP * np.arange(1, COARSE_GROWTH + 1), lower, upper))  # (g_new,)
            grid = np.concatenate([grid, points])
            derivative = np.concatenate([derivative, problem.fit_slopes(weights, np.exp(points))[0]], axis=1)

    # A fit whose derivative keeps one sign over the whole grid has its minimum at a bound.
    below = np.sum(derivative < 0.0, axis=1)  # (r,)
    log_beta = np.where(below == 0, grid[0], grid[-1])  # (r,)
    bracketed = (below > 0) & (below < grid.shape[0])  # (r,)
    for interval in np.unique(below[bracketed] - 1):
        fits = np.flatnonzero(bracketed & (below - 1 == interval))  # (r_i,)
        fine = np.linspace(grid[interval], grid[interval + 1], FINE_POINTS + 1)  # (g_fine,)
        fine_derivative, fine_curvature = problem.fit_slopes(weights[fits], np.exp(fine))  # (r_i, g_fine) each
        # The last fine point with a negative derivative starts each fit's final interval.
        start = np.clip(np.sum(fine_derivative < 0.0, axis=1) - 1, 0, FINE_POINTS - 1)  # (r_i,)
        take = np.arange(fits.shape[0])  # (r_i,)
        beta = _hermite_roots(
            np.exp(fine[start]),
            np.exp(fine[start + 1]),
            (fine_derivative[take, start], fine_derivative[take, start + 1]),
            (fine_curvature[take, start], fine_curvature[take, start + 1]),
        )  # (r_i,)
        log_beta[fits] = np.log(beta)
    return np.exp(-np.clip(log_beta, lower, upper))  # (r,)


def temperature_objective(
    logits: np.ndarray, labels: np.ndarray, multiclass: bool, fit_rows: np.ndarray, temperature: np.ndarray
) -> np.ndarray:
    """Each fit's mean negative log-likelihood of its rows' labels at its temperature."""
    # logits: (n, c); labels: (n,) or (n, c); fit_rows: (r, m); temperature: (r,)
    logits = np.asarray(logits, dtype=np.float64)  # (n, c)
    values = np.empty(fit_rows.shape[0])  # (r,)
    for fit, rows in enumerate(fit_rows):
        scaled = logits[rows] / temperature[fit]  # (m, c)
        if multiclass:
            peak = scaled.max(axis=1, keepdims=True)  # (m, 1)
            log_normalizer = peak[:, 0] + np.log(np.sum(np.exp(scaled - peak), axis=1))  # (m,)
            true_logit = np.take_along_axis(scaled, np.asarray(labels)[rows].astype(np.int64)[:, None], axis=1)[:, 0]  # (m,)
            values[fit] = np.mean(log_normalizer - true_logit)
        else:
            values[fit] = np.mean(np.logaddexp(0.0, scaled) - np.asarray(labels, dtype=np.float64)[rows] * scaled)
    return values  # (r,)


def variance_objective(
    mean: np.ndarray, variance: np.ndarray | None, labels: np.ndarray, fit_rows: np.ndarray, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Each fit's `0.5 * mean(log s + e / s)` over its rows, `s = a v + b` and `e` the squared residual."""
    # mean, labels: (n,); variance: (n,) or None; fit_rows: (r, m); a, b: (r,)
    squared_residuals = ((np.asarray(labels, dtype=np.float64) - mean) ** 2)[fit_rows]  # (r, m)
    raw = np.zeros_like(squared_residuals) if variance is None else np.asarray(variance, dtype=np.float64)[fit_rows]  # (r, m)
    scale = np.asarray(a, dtype=np.float64)[:, None] * raw + np.asarray(b, dtype=np.float64)[:, None]  # (r, m)
    return 0.5 * np.mean(np.log(scale) + squared_residuals / scale, axis=1)  # (r,)


def _profile(squared_residuals: np.ndarray, variance: np.ndarray, ratio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each fit's objective with `b` at its optimum for the ratio `rho = a / b`, and that `b`."""
    # squared_residuals, variance: (r, m); ratio: (r,)
    spread = 1.0 + ratio[:, None] * variance  # (r, m)
    best_b = np.mean(squared_residuals / spread, axis=1)  # (r,)
    with np.errstate(divide="ignore"):
        value = 0.5 * (np.log(best_b) + np.mean(np.log(spread), axis=1) + 1.0)  # (r,)
    return value, best_b  # (r,), (r,)


def _profile_slopes(
    squared_residuals: np.ndarray, variance: np.ndarray, log_ratio: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The profile's first and second derivatives in `log rho`, per fit."""
    # squared_residuals, variance: (r, m); log_ratio: (r,)
    ratio = np.exp(log_ratio)  # (r,)
    inverse = 1.0 / (1.0 + ratio[:, None] * variance)  # (r, m)
    scaled_variance = variance * inverse  # (r, m)
    scaled_residuals = squared_residuals * inverse  # (r, m)
    residual_moment = scaled_residuals * scaled_variance  # (r, m)
    level = np.mean(scaled_residuals, axis=1)  # (r,)
    level_slope = -np.mean(residual_moment, axis=1)  # (r,)
    level_curvature = 2.0 * np.mean(residual_moment * scaled_variance, axis=1)  # (r,)
    spread_slope = np.mean(scaled_variance, axis=1)  # (r,)
    spread_curvature = -np.mean(scaled_variance * scaled_variance, axis=1)  # (r,)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = 0.5 * (level_slope / level + spread_slope)  # (r,): derivative in rho
        curvature = 0.5 * (level_curvature / level - (level_slope / level) ** 2 + spread_curvature)  # (r,)
    return ratio * slope, ratio * slope + ratio ** 2 * curvature  # (r,), (r,)


def fit_variance_maps(
    mean: np.ndarray, variance: np.ndarray | None, labels: np.ndarray, fit_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each fit's `(a, b)` for `N(mean, a v + b)`, with `b` floored as `fit_variance_calibration` floors it."""
    # mean, labels: (n,); variance: (n,) or None; fit_rows: (r, m) row indices
    all_squared_residuals = (np.asarray(labels, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) ** 2  # (n,)
    squared_residuals = all_squared_residuals[fit_rows]  # (r, m)
    constant_b = np.maximum(squared_residuals.mean(axis=1), VARIANCE_FLOOR)  # (r,)
    fits = fit_rows.shape[0]
    if variance is None:
        return np.zeros(fits), constant_b  # (r,), (r,)

    all_variance = np.asarray(variance, dtype=np.float64)  # (n,)
    fit_variance = all_variance[fit_rows]  # (r, m)
    informative = np.any(fit_variance > 0, axis=1)  # (r,)
    # The grid is over log(rho * mean variance), so it is centered whatever the variance's units.
    positive = all_variance[all_variance > 0]  # (n_pos,)
    unit = float(positive.mean()) if positive.size else 1.0

    # Every fit's profile on the shared grid, from per-row terms and one matrix product.
    weights = row_weights(fit_rows, all_variance.shape[0])  # (r, n)
    totals = weights.sum(axis=1, keepdims=True)  # (r, 1)
    ratio_grid = np.exp(LOG_RATIO_GRID) / unit  # (g,)
    spread = 1.0 + all_variance[:, None] * ratio_grid[None, :]  # (n, g)
    level = weights @ (all_squared_residuals[:, None] / spread) / totals  # (r, g)
    log_spread = weights @ np.log(spread) / totals  # (r, g)
    with np.errstate(divide="ignore"):
        grid_values = 0.5 * (np.log(level) + log_spread + 1.0)  # (r, g)
    grid_values = np.where(np.isnan(grid_values), np.inf, grid_values)  # (r, g)
    best = np.argmin(grid_values, axis=1)  # (r,)

    # Newton's method in log(rho) within one grid step of each fit's grid minimum, from the vertex
    # of the parabola through the three grid values around it, bisecting when a step leaves the bracket.
    take = np.arange(fits)  # (r,)
    last = LOG_RATIO_GRID.shape[0] - 1
    before, after = np.maximum(best - 1, 0), np.minimum(best + 1, last)  # (r,), (r,)
    offset = np.log(unit)
    low, high = LOG_RATIO_GRID[before] - offset, LOG_RATIO_GRID[after] - offset  # (r,), (r,)
    left, middle, right = grid_values[take, before], grid_values[take, best], grid_values[take, after]  # (r,) each
    step = LOG_RATIO_GRID[1] - LOG_RATIO_GRID[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        vertex = LOG_RATIO_GRID[best] - offset + 0.5 * step * (left - right) / (left - 2.0 * middle + right)  # (r,)
    log_ratio = np.where(np.isfinite(vertex) & (vertex > low) & (vertex < high), vertex, LOG_RATIO_GRID[best] - offset)  # (r,)
    fitted_rows = np.flatnonzero(informative)  # (r_i,)
    fit_squared, fit_spread_source = squared_residuals[fitted_rows], fit_variance[fitted_rows]  # (r_i, m) each
    current, low, high = log_ratio[fitted_rows], low[fitted_rows], high[fitted_rows]  # (r_i,) each
    converged = np.zeros(fitted_rows.shape[0], dtype=bool)  # (r_i,)
    for _ in range(NEWTON_ITERATIONS):
        if converged.all():
            break
        slope, curvature = _profile_slopes(fit_squared, fit_spread_source, current)  # (r_i,) each
        low = np.where(converged | (slope >= 0.0), low, current)
        high = np.where(converged | (slope < 0.0), high, current)
        with np.errstate(divide="ignore", invalid="ignore"):
            proposal = current - slope / curvature  # (r_i,)
        # Inclusive, so a zero step at the root is taken rather than bisected away from it.
        inside = (curvature > 0.0) & (proposal >= low) & (proposal <= high)  # (r_i,)
        proposal = np.where(converged, current, np.where(inside, proposal, (low + high) / 2.0))  # (r_i,)
        converged |= np.abs(proposal - current) <= 1e-12
        current = proposal
    log_ratio[fitted_rows] = current

    ratio = np.exp(log_ratio)  # (r,)
    interior_value, interior_b = _profile(squared_residuals, fit_variance, ratio)  # (r,), (r,)
    # rho = 0 is the constant-variance model, the boundary the grid only approaches.
    zero_value, _ = _profile(squared_residuals, fit_variance, np.zeros(fits))  # (r,)
    use_zero = ~informative | ~(interior_value < zero_value)  # (r,)
    a = np.where(use_zero, 0.0, ratio * interior_b)  # (r,)
    b = np.where(use_zero, constant_b, np.maximum(interior_b, VARIANCE_FLOOR))  # (r,)
    return a, b  # (r,), (r,)
