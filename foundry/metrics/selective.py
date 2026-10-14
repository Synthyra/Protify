"""Selective prediction: the loss left when a model keeps only the rows it is most certain about.

Ranking rows by an uncertainty score and keeping the least uncertain fraction, the coverage,
leaves a mean loss over the kept rows, the risk. Sweeping the coverage traces the risk-coverage
curve. The area under it, divided by the coverage it spans, is AURC: lower means the score puts
the costly rows last.

Promoted from sae_xgboost_ppi's `gp_analysis.py` (`_risk_coverage`, `_aurc_only`), which fixed a
grid of 19 coverages from 0.1 to 1.0 and scored misclassification. Given that grid and that
loss, these reproduce its numbers exactly. Its private copies stay until its Modal image carries
foundry, and its suite holds them equal.
"""

from __future__ import annotations

import numpy as np


def risk_coverage_curve(
    losses: np.ndarray, uncertainty: np.ndarray, coverage: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mean loss over the least uncertain rows at each coverage, as `(coverage, risk)`.

    Rows are ordered by ascending `uncertainty`, ties kept in input order. At coverage `c`, the
    first `max(1, round(c * n))` rows are kept, rounding half to even. Without a grid, every
    retained count `k = 1..n` is a level, at coverage `k / n`. A grid must increase strictly
    within (0, 1].

    `losses` is any per-row loss: misclassification (`predicted != label`) for a classifier, an
    absolute or squared error for a regressor.
    """
    # losses, uncertainty: (n,); coverage: (g,)
    losses = np.asarray(losses, dtype=np.float64)  # (n,)
    uncertainty = np.asarray(uncertainty)  # (n,)
    if losses.ndim != 1 or losses.shape != uncertainty.shape or losses.size == 0:
        raise ValueError(
            f"losses and uncertainty must be non-empty vectors of one length, got {losses.shape} and {uncertainty.shape}"
        )

    n = losses.shape[0]
    order = np.argsort(uncertainty, kind="mergesort")  # (n,)
    # Total loss of the k least uncertain rows, for k = 1..n.
    cumulative = np.cumsum(losses[order])  # (n,)
    # g coverage levels: the grid's, or n without one.
    if coverage is None:
        counts = np.arange(1, n + 1)  # (g,)
        levels = counts / n  # (g,)
    else:
        levels = _checked_grid(coverage)  # (g,)
        counts = np.maximum(1, np.round(levels * n).astype(int))  # (g,)
    risk = cumulative[counts - 1] / counts  # (g,)

    return levels, risk  # (g,), (g,)


def area_under_risk_coverage(
    losses: np.ndarray, uncertainty: np.ndarray, coverage: np.ndarray | None = None
) -> float:
    """AURC: the trapezoidal area under `risk_coverage_curve`, divided by the coverage it spans.

    The division makes a constant risk `r` score `r` on any grid. Without a grid the levels run
    from `1 / n` to 1, so at least two rows are needed.
    """
    # losses, uncertainty: (n,); coverage: (g,)
    levels, risk = risk_coverage_curve(losses, uncertainty, coverage)  # (g,), (g,)
    if levels.shape[0] < 2:
        raise ValueError("an area needs at least two coverage levels, and one row gives one")

    return float(np.trapezoid(risk, levels) / (levels[-1] - levels[0]))


def _checked_grid(coverage: np.ndarray) -> np.ndarray:
    """The coverage grid as float64, refused unless it increases strictly within (0, 1]."""
    # coverage: (g,)
    levels = np.array(coverage, dtype=np.float64)  # (g,)
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError(f"coverage must be a non-empty vector, got shape {levels.shape}")
    if not (levels[0] > 0.0 and levels[-1] <= 1.0 and bool(np.all(np.diff(levels) > 0.0))):
        raise ValueError(f"coverage must increase strictly within (0, 1], got {levels.tolist()}")

    return levels  # (g,)
