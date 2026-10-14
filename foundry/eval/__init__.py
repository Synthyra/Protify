"""Evaluation procedures: temperature scaling, split conformal intervals, stratified bootstrap, and all-vs-all pair scoring.

`fit_temperature`, `gather_pair_scores` and `dual_tower_blocks` need torch and load it on first use,
so conformal intervals and bootstrap replicates need only NumPy.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from .bootstrap import stratified_bootstrap_indices
from .conformal import (
    Interval,
    calibration_size_for_alpha,
    conformal_quantile,
    empirical_coverage,
    split_conformal,
)


if TYPE_CHECKING:
    from .pair_scores import dual_tower_blocks, gather_pair_scores
    from .temperature import fit_temperature


TORCH_MODULES = {"fit_temperature": ".temperature", "gather_pair_scores": ".pair_scores", "dual_tower_blocks": ".pair_scores"}


def __getattr__(name: str) -> object:
    if name in TORCH_MODULES:
        return getattr(import_module(TORCH_MODULES[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Interval",
    "calibration_size_for_alpha",
    "conformal_quantile",
    "dual_tower_blocks",
    "empirical_coverage",
    "fit_temperature",
    "gather_pair_scores",
    "split_conformal",
    "stratified_bootstrap_indices",
]
