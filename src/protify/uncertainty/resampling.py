"""Seeded draws of one dataset's test rows, the same for every cell of that dataset.

Every cell of a dataset scored the same test rows, so arms and heads are compared on the same
draws: a draw depends only on the analysis seed, the dataset's name, and its test labels, never
on the cell. Each worker process can therefore rebuild a dataset's draws on its own.

- Question 1 draws `replicates` stratified bootstrap resamples of the test rows.
- Scheme T, question 2's exchangeable scheme, draws `repeats` splits of the test rows. Each
  divides them, stratified, into halves T_cal and T_eval, and T_cal again into T_fit, which
  plays V_fit's role and fits the calibration maps, and T_conformal, which plays V_cal's and
  fits the conformal quantiles. T_eval is bootstrap-resampled once, stratified.

Halves are carved by `calibration_split.carve_validation`, the carve that divides validation
into V_fit and V_cal, so every repeat has the same sizes. Strata are
`calibration_split.stratum_codes`: the class, the label decile, or the rarest positive label,
recomputed within the rows being divided or resampled.

Shapes: `B` bootstrap replicates, `R` repeats, `n` test rows, `m_fit` and `m_conf` rows of T_fit
and T_conformal.
"""

from __future__ import annotations

import zlib
import numpy as np

from dataclasses import dataclass

from foundry.eval.bootstrap import stratified_bootstrap_indices
from .calibration_split import carve_validation, stratum_codes


# Each use of a dataset's seed draws from its own stream, so adding one use never moves another.
BOOTSTRAP_STREAM = 1
EXCHANGEABLE_STREAM = 2
CALIBRATION_HALF = 0.5


@dataclass(frozen=True)
class ExchangeableDesign:
    """Scheme T's rows for every repeat: which fit the maps, which the quantiles, which are scored."""

    fit_rows: np.ndarray  # (R, m_fit) ascending row indices of T_fit
    conformal_rows: np.ndarray  # (R, m_conf) ascending row indices of T_conformal
    eval_counts: np.ndarray  # (R, n) how many times the resample of T_eval drew each row

    @property
    def repeats(self) -> int:
        return self.eval_counts.shape[0]


def dataset_generator(seed: int, data_name: str, stream: int) -> np.random.Generator:
    """A generator that depends only on the analysis seed, the dataset, and the use it serves."""
    return np.random.default_rng([seed, zlib.crc32(data_name.encode("utf-8")), stream])


def bootstrap_counts(labels: np.ndarray, label_type: str, data_name: str, replicates: int, seed: int) -> np.ndarray:
    """How many times each test row is drawn in each stratified bootstrap replicate, `(B, n)`."""
    # labels: (n,) or (n, c)
    strata = stratum_codes(labels, label_type)  # (n,)
    rows = strata.shape[0]
    rng = dataset_generator(seed, data_name, BOOTSTRAP_STREAM)
    counts = np.empty((replicates, rows), dtype=np.int32)  # (B, n)
    for replicate in range(replicates):
        counts[replicate] = np.bincount(stratified_bootstrap_indices(strata, rng), minlength=rows)
    return counts  # (B, n)


def exchangeable_design(labels: np.ndarray, label_type: str, data_name: str, repeats: int, seed: int) -> ExchangeableDesign:
    """Scheme T's T_fit, T_conformal, and resampled T_eval rows for each repeat."""
    # labels: (n,) or (n, c)
    rows = labels.shape[0]
    rng = dataset_generator(seed, data_name, EXCHANGEABLE_STREAM)
    fit_rows, conformal_rows = [], []
    eval_counts = np.zeros((repeats, rows), dtype=np.int32)  # (R, n)
    for repeat in range(repeats):
        outer_seed, inner_seed = (int(value) for value in rng.integers(0, 2 ** 32, size=2))
        evaluation, calibration = carve_validation(labels, label_type, CALIBRATION_HALF, outer_seed)  # (n - n // 2,), (n // 2,)
        fit_positions, conformal_positions = carve_validation(labels[calibration], label_type, CALIBRATION_HALF, inner_seed)
        fit_rows.append(calibration[fit_positions])  # (m_fit,)
        conformal_rows.append(calibration[conformal_positions])  # (m_conf,)
        drawn = evaluation[stratified_bootstrap_indices(stratum_codes(labels[evaluation], label_type), rng)]  # (n - n // 2,)
        eval_counts[repeat] = np.bincount(drawn, minlength=rows)
    return ExchangeableDesign(np.stack(fit_rows), np.stack(conformal_rows), eval_counts)
