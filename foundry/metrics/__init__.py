"""Calibration, selective-prediction, and binary classification metrics for predicted probabilities."""

from .binary import calculate_topk_metrics, compute_metrics_from_counts
from .calibration import expected_calibration_error
from .selective import area_under_risk_coverage, risk_coverage_curve


__all__ = [
    "area_under_risk_coverage",
    "calculate_topk_metrics",
    "compute_metrics_from_counts",
    "expected_calibration_error",
    "risk_coverage_curve",
]
