"""Scheme T for one cell: every method recalibrated and rescored on exchangeable halves of test.

Published validation splits are not exchangeable with their test splits, so question 2 reads a
second scheme. In each repeat (`resampling.exchangeable_design`), T_fit fits every calibration
map, T_conformal fits every conformal quantile, and a resample of T_eval is scored, as V_fit,
V_cal, and test are under scheme V.

Methods come from `methods.build_methods`, unchanged, called once on four splits sliced from the
test rows: repeat 0's T_fit, T_conformal, and resample of T_eval as V_fit, V_cal, and test, and
every test row as a fourth split. A run with nothing fitted, such as a head or an ensemble,
predicts every repeat as it predicts that fourth split. A calibrated run, one whose `fitted`
holds a temperature or a variance map, is refitted in every repeat by `batched_calibration` on
what it calibrates: the head's logits for the head's own `temperature` or `sngp_temperature`, as
`methods._calibrated_run` reads them, and otherwise the method its name extends by
`_temperature`, as `methods._recalibrated_run` reads it. Every repeat is then scored from the
resample's draw counts by `weighted_metrics`, in the arithmetic `metrics.py` uses.

The methods built from the per-seed extras refit every choice they make on V_fit, in every
repeat, on T_fit: `laplace_temperature` fits a map for each prior precision and keeps the one
whose objective is lowest, the first on a tie; `temperature_pnorm` refits its temperature and
keeps the norm order whose T_fit AURC is lowest, by `methods.pnorm_aurc`; and `loss_head`
refits the temperature under the predicted loss, or the variance map over it. Their candidate
logits, variances, and rankings come from the functions `build_methods` uses.

Repeat 0 is also scored the unchanged way, by `report.score_method` on that build, which checks
this path three ways: each calibrated run, given the map `build_methods` fitted, predicts every
test row exactly as that build does; the maps fitted here reach the same objective as those
maps; and repeat 0 scored here with those maps matches `score_method`.

Beyond the report's metrics, each repeat adds the shift AUROC, with which a method's uncertainty
separates test proteins below 0.3 identity to training, no hit included, from those at 0.5 or
above; every conformal coverage and size or width per identity bin; and Mondrian conformal by
identity bin, with adjacent bins merged, in identity order, until each group holds at least 100
T_conformal rows, a last group left short joining the one before it.

Departures from `metrics.py`, each visible in the checks:
- Where tied probabilities straddle the end of a calibration-error run, their order can differ
  from the reference's, as `weighted_metrics.weighted_ece` describes.
- The regression conformal spread floor reads the given `train_label_std`, which report.json
  does not record; `compare` documents the stand-in it passes.

Shapes: `R` repeats, `r` repeats in a chunk (or 1 for predictions every repeat shares), `n` test
rows, `c` classes or labels, `m_fit` and `m_conf` rows of T_fit and T_conformal, `G` identity
bins, `K` candidates a map chooses among.
"""

from __future__ import annotations

import numpy as np

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .batched_calibration import fit_temperatures, fit_variance_maps, temperature_objective, variance_objective
from .conformal_sets import SET_SCORES, SPREAD_FLOOR_FRACTION, class_scores
from .identity_bins import IDENTITY_BIN_NAMES, bin_rows, identity_codes
from .methods import (
    HEAD_NAMES,
    LAPLACE_METHOD,
    LAPLACE_PRIOR_PRECISIONS,
    LOSS_HEAD_METHOD,
    PNORM_METHOD,
    PNORM_ORDERS,
    VARIANCE_FLOOR,
    Method,
    MethodRun,
    Predictive,
    build_methods,
    classification_predictive,
    laplace_logits,
    loss_head_variance,
    pnorm_aurc,
    pnorm_uncertainty,
    probabilities,
    probability_logits,
)
from .metrics import CLASSIFICATION_KINDS, ECE_BINS, PROBABILITY_FLOOR, gaussian_interval
from .predictions import SplitPredictions
from .report import score_method
from .resampling import ExchangeableDesign
from .weighted_metrics import (
    multilabel_draw_scores,
    weighted_aurc,
    weighted_auroc,
    weighted_ece,
    weighted_mcc,
    weighted_median,
    weighted_spearman,
)


# The split every test row forms in the one build of the methods.
ALL_TEST = "all_test"
# Test proteins below this identity to training are novel, those at or above the second familiar.
NOVEL_BELOW = 0.3
FAMILIAR_FROM = 0.5
# Mondrian conformal merges identity bins until each group holds this many T_conformal rows.
MONDRIAN_MINIMUM_ROWS = 100
# Bytes of per-repeat prediction arrays a calibrated run holds at once, allowing for those scoring derives.
CHUNK_BYTES = 256 * 2 ** 20
# How far repeat 0 scored here may sit from score_method, relative to the larger of 1 and its value.
CHECK_TOLERANCE = 1e-5
# Methods built from the per-seed extras, each refitted by its own plan rather than by its name.
EXTRA_METHODS = (LAPLACE_METHOD, PNORM_METHOD, LOSS_HEAD_METHOD)

SetMeasure = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class RunPlan:
    """One method run as scheme T predicts it in each repeat."""

    method: str
    seed: int | None
    score_only: bool
    # The run's predictions on every test row, which a run with nothing fitted keeps in every repeat.
    fixed: Predictive
    # A calibrated run's map, as `build_methods` fitted it on repeat 0's T_fit.
    fitted: Mapping[str, float] = field(default_factory=dict)
    # What a calibrated run's map applies to: logits `(n, c)` to divide, or the regression prediction.
    logits: np.ndarray | None = None
    base: Predictive | None = None
    # Multilabel entries in ascending logits `(n * c,)`, which sorts the probabilities at any temperature.
    entry_order: np.ndarray | None = None
    # A map that also chooses among candidates: the `fitted` key naming the choice, and its values.
    choice: str | None = None
    grid: tuple[float, ...] = ()
    # Each candidate's logits `(K, n, c)`, or each candidate's variance `(K, n)` for regression.
    candidates: np.ndarray | None = None
    # Each candidate's uncertainty `(K, n)`, for a choice among rankings.
    rankings: np.ndarray | None = None
    # An uncertainty `(n,)` every map keeps, in place of the one its probabilities give.
    uncertainty: np.ndarray | None = None

    @property
    def calibrated(self) -> bool:
        return bool(self.fitted)


def _choice_index(plan: RunPlan, values: np.ndarray) -> np.ndarray:
    """Which candidate each repeat's chosen value names in `plan.grid`."""
    # values: (r,)
    matches = np.asarray(values, dtype=np.float64)[:, None] == np.asarray(plan.grid, dtype=np.float64)[None, :]  # (r, K)
    if not matches.any(axis=1).all():
        raise ValueError(f"{plan.method} chose a {plan.choice} outside its grid {plan.grid}.")
    return matches.argmax(axis=1)  # (r,)


@dataclass(frozen=True)
class RepeatPredictive:
    """A run's predictions on every test row in a chunk of repeats, or in every repeat alike."""

    probs: np.ndarray | None = None  # (r, n, c)
    mean: np.ndarray | None = None  # (n,)
    variance: np.ndarray | None = None  # (r, n)
    uncertainty: np.ndarray | None = None  # (r, n)
    entry_order: np.ndarray | None = None  # (n * c,)


@dataclass
class RunScores:
    """Every score of one run in each repeat, overall and per identity bin."""

    overall: dict[str, np.ndarray] = field(default_factory=dict)  # metric -> (r,)
    by_bin: dict[str, np.ndarray] = field(default_factory=dict)  # metric -> (r, G)


@dataclass(frozen=True)
class MethodScores:
    """One method's scores in each repeat: its runs' mean, a run left out where its score is undefined."""

    name: str
    score_only: bool
    overall: dict[str, np.ndarray]  # metric -> (R,)
    by_bin: dict[str, np.ndarray]  # metric -> (R, G)


@dataclass(frozen=True)
class IdentityLayout:
    """Where each test row sits by identity, and each repeat's Mondrian groups."""

    bin_indicator: np.ndarray  # (n, G) one-hot identity bin
    shift_rows: np.ndarray  # (n,) novel or familiar, the rows the shift AUROC reads
    novel: np.ndarray  # (n,)
    bin_groups: np.ndarray  # (R, G) each bin's Mondrian group, named by the group's first bin
    row_groups: np.ndarray  # (R, n) each row's Mondrian group

    def window(self, repeats: slice) -> IdentityLayout:
        return replace(self, bin_groups=self.bin_groups[repeats], row_groups=self.row_groups[repeats])


@dataclass(frozen=True)
class ExchangeableResult:
    methods: list[MethodScores]
    checks: list[dict[str, Any]]
    # T_conformal rows in every repeat, the calibration size of scheme T's conformal band.
    calibration_rows: int
    # Each distinct Mondrian grouping of the identity bins and how many repeats used it.
    mondrian_groupings: list[dict[str, Any]] | None


def calibrated_base(method_name: str, head: str) -> str:
    """The method a calibrated method's map is applied to: the head for `temperature`, else its name less `_temperature`."""
    if method_name == "temperature":
        return HEAD_NAMES[head]
    if method_name.endswith("_temperature"):
        return method_name.removesuffix("_temperature")
    raise ValueError(f"method {method_name!r} fits a map, but its name does not say what the map calibrates.")


def slice_split(split: SplitPredictions, rows: np.ndarray) -> SplitPredictions:
    """The given rows of one split, every per-row field sliced alike."""
    # rows: (m,)
    return SplitPredictions(**{name: None if value is None else value[rows] for name, value in vars(split).items()})


def reference_splits(
    test_splits: Mapping[int, SplitPredictions], design: ExchangeableDesign, repeat: int
) -> dict[int, dict[str, SplitPredictions]]:
    """Each seed's test predictions sliced into one repeat's V_fit, V_cal, and test roles, and every row."""
    rows = next(iter(test_splits.values())).labels.shape[0]
    evaluation = np.repeat(np.arange(rows), design.eval_counts[repeat])  # (n - n // 2,), ascending
    return {
        seed: {
            "v_fit": slice_split(split, design.fit_rows[repeat]),
            "v_cal": slice_split(split, design.conformal_rows[repeat]),
            "test": slice_split(split, evaluation),
            ALL_TEST: split,
        }
        for seed, split in test_splits.items()
    }


def plan_runs(
    methods: Sequence[Method], splits: Mapping[int, Mapping[str, SplitPredictions]], head: str, kind: str
) -> list[RunPlan]:
    """Every run of every method, with what a calibrated run's map applies to."""
    by_name = {method.name: method for method in methods}
    plans = []
    for method in methods:
        for run in method.runs:
            if not run.fitted:
                plans.append(RunPlan(method.name, run.seed, method.score_only, run.splits[ALL_TEST]))
                continue
            if method.name in EXTRA_METHODS:
                plans.append(_extra_plan(method, run, splits[run.seed][ALL_TEST], kind))
                continue

            base_name = calibrated_base(method.name, head)
            if base_name not in by_name:
                raise ValueError(f"{method.name} calibrates {base_name}, which build_methods did not build.")
            base_runs = [candidate for candidate in by_name[base_name].runs if candidate.seed == run.seed]
            if len(base_runs) != 1:
                raise ValueError(f"{method.name} seed {run.seed} has {len(base_runs)} runs of {base_name} to calibrate.")
            base = base_runs[0].splits[ALL_TEST]
            logits, entry_order = None, None
            if kind != "regression":
                # The head calibrates its own logits; every other method the logits of its probabilities.
                own_head = base_name == HEAD_NAMES[head]
                logits = splits[run.seed][ALL_TEST].outputs if own_head else probability_logits(base.probs, kind)  # (n, c)
                if kind == "multilabel":
                    entry_order = np.argsort(logits.reshape(-1), kind="stable")  # (n * c,)
            plans.append(RunPlan(method.name, run.seed, False, run.splits[ALL_TEST], dict(run.fitted), logits, base, entry_order))
    return plans


def _extra_plan(method: Method, run: MethodRun, split: SplitPredictions, kind: str) -> RunPlan:
    """How scheme T refits a method built from the per-seed extras, from one seed's every test row."""
    fixed = run.splits[ALL_TEST]
    fitted = dict(run.fitted)
    entry_order = None
    if kind == "multilabel":
        entry_order = np.argsort(split.outputs.reshape(-1), kind="stable")  # (n * c,)

    if method.name == LAPLACE_METHOD:
        if kind == "regression":
            candidates = np.ascontiguousarray(split.laplace_variance.T)  # (K, n)
            return RunPlan(
                method.name, run.seed, False, fixed, fitted, base=Predictive(mean=split.outputs),
                choice="prior_precision", grid=LAPLACE_PRIOR_PRECISIONS, candidates=candidates,
            )
        return RunPlan(
            method.name, run.seed, False, fixed, fitted, choice="prior_precision", grid=LAPLACE_PRIOR_PRECISIONS,
            candidates=laplace_logits(split.outputs, split.laplace_variance, kind),  # (K, n, c)
        )

    if method.name == PNORM_METHOD:
        rankings = np.stack([pnorm_uncertainty(split.outputs, order) for order in PNORM_ORDERS])  # (K, n)
        return RunPlan(
            method.name, run.seed, method.score_only, fixed, fitted, split.outputs,
            choice="p", grid=PNORM_ORDERS, rankings=rankings,
        )

    if kind == "regression":
        base = Predictive(mean=split.outputs, variance=loss_head_variance(split.loss_prediction))
        return RunPlan(method.name, run.seed, False, fixed, fitted, base=base)
    return RunPlan(
        method.name, run.seed, False, fixed, fitted, split.outputs, entry_order=entry_order,
        uncertainty=np.asarray(split.loss_prediction, dtype=np.float64),
    )


def _fit_choice(plan: RunPlan, kind: str, labels: np.ndarray, fit_rows: np.ndarray) -> dict[str, np.ndarray]:
    """A map fitted for every candidate in every repeat, and each repeat's lowest-objective candidate."""
    # labels: (n,) or (n, c); fit_rows: (R, m_fit)
    multiclass = kind in CLASSIFICATION_KINDS
    fitted: dict[str, list[np.ndarray]] = {}
    objectives = []
    for candidate in plan.candidates:
        if kind == "regression":
            a, b = fit_variance_maps(plan.base.mean, candidate, labels, fit_rows)  # (R,), (R,)
            maps = {"a": a, "b": b}
            objectives.append(variance_objective(plan.base.mean, candidate, labels, fit_rows, a, b))  # (R,)
        else:
            temperature = fit_temperatures(candidate, labels, multiclass, fit_rows)  # (R,)
            maps = {"temperature": temperature}
            objectives.append(temperature_objective(candidate, labels, multiclass, fit_rows, temperature))  # (R,)
        for key, values in maps.items():
            fitted.setdefault(key, []).append(values)
    index = np.argmin(np.stack(objectives), axis=0)  # (R,), the first candidate on a tie
    repeats = np.arange(fit_rows.shape[0])  # (R,)
    chosen = {key: np.stack(values)[index, repeats] for key, values in fitted.items()}  # key -> (R,)
    return {**chosen, plan.choice: np.asarray(plan.grid, dtype=np.float64)[index]}  # each (R,)


def _pnorm_choices(plan: RunPlan, labels: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    """Each repeat's norm order, the one whose AURC over its T_fit rows is lowest, the first on a tie."""
    # labels: (n,); fit_rows: (R, m_fit)
    index = np.empty(fit_rows.shape[0], dtype=np.int64)  # (R,)
    for repeat, rows in enumerate(fit_rows):
        values = [pnorm_aurc(plan.logits[rows], labels[rows], order) for order in plan.grid]
        index[repeat] = int(np.argmin(np.asarray(values, dtype=np.float64)))
    return np.asarray(plan.grid, dtype=np.float64)[index]  # (R,)


def fit_maps(plan: RunPlan, kind: str, labels: np.ndarray, fit_rows: np.ndarray) -> dict[str, np.ndarray]:
    """A calibrated run's map in every repeat, fitted on that repeat's T_fit rows."""
    # labels: (n,) or (n, c); fit_rows: (R, m_fit)
    if plan.candidates is not None:
        return _fit_choice(plan, kind, labels, fit_rows)  # each (R,)
    if plan.rankings is not None:
        temperature = fit_temperatures(plan.logits, labels, True, fit_rows)  # (R,)
        return {"temperature": temperature, plan.choice: _pnorm_choices(plan, labels, fit_rows)}  # each (R,)
    if kind == "regression":
        a, b = fit_variance_maps(plan.base.mean, plan.base.variance, labels, fit_rows)  # (R,), (R,)
        return {"a": a, "b": b}  # each (R,)
    return {"temperature": fit_temperatures(plan.logits, labels, kind in CLASSIFICATION_KINDS, fit_rows)}  # (R,)


def fixed_predictive(predictive: Predictive) -> RepeatPredictive:
    """Predictions every repeat shares, with a leading axis of one."""
    return RepeatPredictive(
        probs=None if predictive.probs is None else predictive.probs[None],  # (1, n, c)
        mean=predictive.mean,  # (n,)
        variance=None if predictive.variance is None else predictive.variance[None],  # (1, n)
        uncertainty=None if predictive.uncertainty is None else predictive.uncertainty[None],  # (1, n)
    )


def calibrated_predictive(plan: RunPlan, kind: str, maps: Mapping[str, np.ndarray]) -> RepeatPredictive:
    """A calibrated run's predictions under each of the given maps, in the arithmetic `methods.py` uses.

    Classification temperature scaling uses float64, as `methods.probabilities` does.
    Regression casts each map to the variance's dtype to reproduce its scalar arithmetic.
    """
    # maps: (r,) per parameter
    choice = None if plan.choice is None else _choice_index(plan, maps[plan.choice])  # (r,)
    if kind == "regression":
        if plan.candidates is not None:
            raw = plan.candidates[choice]  # (r, n)
        elif plan.base.variance is None:
            raw = np.zeros(plan.base.mean.shape[0])[None]  # (1, n)
        else:
            raw = plan.base.variance[None]  # (1, n)
        a = maps["a"].astype(raw.dtype)[:, None]  # (r, 1)
        b = maps["b"].astype(raw.dtype)[:, None]  # (r, 1)
        variance = np.maximum(a * raw + b, VARIANCE_FLOOR)  # (r, n)
        return RepeatPredictive(mean=plan.base.mean, variance=variance, uncertainty=np.sqrt(variance))

    entry_order = plan.entry_order
    if plan.candidates is not None:
        logits = plan.candidates[choice]  # (r, n, c)
        entry_order = None
        if kind == "multilabel" and np.all(choice == choice[0]):
            entry_order = np.argsort(plan.candidates[choice[0]].reshape(-1), kind="stable")  # (n * c,)
    else:
        logits = plan.logits[None]  # (1, n, c)
    # Match methods.probabilities(..., temperature=T): float32 scaling can create
    # answer-changing probability ties at a large temperature.
    temperatures = maps["temperature"].astype(np.float64)[:, None, None]  # (r, 1, 1)
    repeats = temperatures.shape[0]
    rows, classes = logits.shape[1:]
    scaled = (np.asarray(logits, dtype=np.float64) / temperatures).reshape(repeats * rows, classes)  # (r * n, c)
    predictive = classification_predictive(probabilities(scaled, kind), kind)
    uncertainty = predictive.uncertainty.reshape(repeats, rows)  # (r, n)
    if plan.rankings is not None:
        uncertainty = plan.rankings[choice]  # (r, n)
    elif plan.uncertainty is not None:
        uncertainty = np.broadcast_to(plan.uncertainty[None], (repeats, rows))  # (r, n)
    return RepeatPredictive(
        probs=predictive.probs.reshape(repeats, rows, classes),  # (r, n, c)
        uncertainty=uncertainty,  # (r, n)
        entry_order=entry_order,
    )


def mondrian_bin_groups(codes: np.ndarray, conformal_rows: np.ndarray, minimum: int = MONDRIAN_MINIMUM_ROWS) -> np.ndarray:
    """Each identity bin's Mondrian group in each repeat, named by the group's first bin.

    Bins join the open group in identity order until it holds `minimum` T_conformal rows; a last
    group left short joins the one before it.
    """
    # codes: (n,) identity bin per test row; conformal_rows: (R, m_conf)
    bins = len(IDENTITY_BIN_NAMES)
    groups = np.empty((conformal_rows.shape[0], bins), dtype=np.int64)  # (R, G)
    for repeat, rows in enumerate(conformal_rows):
        held_per_bin = np.bincount(codes[rows], minlength=bins)  # (G,)
        first, held, closed = 0, 0, []
        for code in range(bins):
            groups[repeat, code] = first
            held += held_per_bin[code]
            if held >= minimum and code < bins - 1:
                closed.append(first)
                first, held = code + 1, 0
        if held < minimum and closed:
            groups[repeat, groups[repeat] == first] = closed[-1]
    return groups  # (R, G)


def identity_layout(identity: np.ndarray, conformal_rows: np.ndarray) -> IdentityLayout:
    """The identity bins, shift rows, and Mondrian groups of every test row."""
    # identity: (n,) highest identity to training; conformal_rows: (R, m_conf)
    codes = identity_codes(identity)  # (n,)
    novel = identity < NOVEL_BELOW  # (n,)
    bin_groups = mondrian_bin_groups(codes, conformal_rows)  # (R, G)
    return IdentityLayout(
        bin_indicator=np.eye(len(IDENTITY_BIN_NAMES))[codes],  # (n, G)
        shift_rows=novel | (identity >= FAMILIAR_FROM),  # (n,)
        novel=novel,
        bin_groups=bin_groups,
        row_groups=bin_groups[:, codes],  # (R, n)
    )


def mondrian_groupings(bin_groups: np.ndarray) -> list[dict[str, Any]]:
    """Each distinct grouping of the identity bins, most used first, and how many repeats used it."""
    # bin_groups: (R, G)
    groupings = []
    for grouping, repeats in Counter(tuple(row) for row in bin_groups.tolist()).most_common():
        groups: dict[int, list[str]] = {}
        for code, group in enumerate(grouping):
            groups.setdefault(group, []).append(IDENTITY_BIN_NAMES[code])
        groupings.append({"groups": list(groups.values()), "repeats": repeats})
    return groupings


@dataclass(frozen=True)
class Draws:
    """A window of resamples: how many times each draw holds each row, and what every mean reuses."""

    counts: np.ndarray  # (r, n)
    weights: np.ndarray  # (r, n) float64
    totals: np.ndarray  # (r,)
    bin_indicator: np.ndarray | None = None  # (n, G)
    bin_totals: np.ndarray | None = None  # (r, G)

    @classmethod
    def of(cls, counts: np.ndarray, layout: IdentityLayout | None) -> Draws:
        # counts: (r, n)
        weights = counts.astype(np.float64)  # (r, n)
        if layout is None:
            return cls(counts, weights, weights.sum(axis=1))
        return cls(counts, weights, weights.sum(axis=1), layout.bin_indicator, weights @ layout.bin_indicator)

    def _weighted(self, values: np.ndarray) -> np.ndarray:
        """Each draw's weight times each row's value, a row no draw holds adding nothing even where infinite."""
        # values: (r, n) or (1, n)
        if not np.isfinite(values).all():
            values = np.where(self.counts > 0, values, 0.0)  # (r, n)
        return self.weights * values  # (r, n)

    def mean(self, values: np.ndarray) -> np.ndarray:
        """`weighted_metrics.weighted_mean`: each draw's mean of `values`, NaN for a draw with no rows."""
        # values: (r, n) or (1, n)
        if values.shape[0] == 1 and np.isfinite(values).all():
            sums = self.weights @ values[0].astype(np.float64)  # (r,)
        else:
            sums = self._weighted(values).sum(axis=1)  # (r,)
        return np.divide(sums, self.totals, out=np.full(self.totals.shape, np.nan), where=self.totals > 0)  # (r,)

    def bin_means(self, values: np.ndarray) -> np.ndarray:
        """Each draw's mean of `values` over its rows in each identity bin, NaN for a bin it holds no rows of."""
        # values: (r, n) or (1, n)
        if values.shape[0] == 1 and np.isfinite(values).all():
            sums = self.weights @ (values[0].astype(np.float64)[:, None] * self.bin_indicator)  # (r, G)
        else:
            sums = self._weighted(values) @ self.bin_indicator  # (r, G)
        return np.divide(sums, self.bin_totals, out=np.full(self.bin_totals.shape, np.nan), where=self.bin_totals > 0)  # (r, G)


def order_statistics(ranked: np.ndarray, held: np.ndarray, level: float) -> np.ndarray:
    """`foundry.eval.conformal.conformal_quantile` of each repeat's sorted calibration scores.

    `ranked` holds each repeat's scores in ascending order, its first `held` real and the rest
    padding; the quantile is the `ceil((held + 1) level)`-th, and infinite past `held`.
    """
    # ranked: (r, m); held: (r,)
    alpha = 1.0 - level
    rank = np.ceil((held + 1) * (1.0 - alpha)).astype(np.int64)  # (r,)
    valid = (held > 0) & (rank <= held)  # (r,)
    picked = np.take_along_axis(ranked, np.clip(rank - 1, 0, ranked.shape[1] - 1)[:, None], axis=1)[:, 0]  # (r,)
    return np.where(valid, picked, np.inf)  # (r,)


def _record(scores: RunScores, name: str, draws: Draws, values: np.ndarray) -> None:
    """Add one per-row score, overall and, where identity is known, per identity bin."""
    # values: (r, n) or (1, n)
    scores.overall[name] = draws.mean(values)  # (r,)
    if draws.bin_indicator is not None:
        scores.by_bin[name] = draws.bin_means(values)  # (r, G)


def _conformal_scores(
    scores: RunScores,
    draws: Draws,
    conformal_rows: np.ndarray,
    levels: Sequence[float],
    names: tuple[str, str],
    true_scores: np.ndarray,
    set_measure: SetMeasure,
    layout: IdentityLayout | None,
) -> None:
    """Coverage and set size or width at every level, marginal and Mondrian.

    `true_scores` is each row's nonconformity score of its label, and `set_measure(q)` each row's
    coverage and size or width when every row reads the quantile `q` of its own row. `names` are
    the coverage and size metrics' stems, such as `lac_coverage` and `lac_size`.
    """
    # conformal_rows: (r, m_conf); true_scores: (r or 1, n)
    repeats, rows = draws.counts.shape
    calibration = np.take_along_axis(np.broadcast_to(true_scores, (repeats, rows)), conformal_rows, axis=1)  # (r, m_conf)
    calibration = calibration.astype(np.float64)  # (r, m_conf), as conformal_quantile casts
    ranked = np.sort(calibration, axis=1)  # (r, m_conf)
    held = np.full(repeats, calibration.shape[1])  # (r,)
    grouped = {}
    if layout is not None:
        calibration_groups = np.take_along_axis(layout.row_groups, conformal_rows, axis=1)  # (r, m_conf)
        for group in np.unique(calibration_groups):
            member = calibration_groups == group  # (r, m_conf)
            grouped[int(group)] = (np.sort(np.where(member, calibration, np.inf), axis=1), member.sum(axis=1))  # (r, m_conf), (r,)

    coverage_name, size_name = names
    for level in levels:
        quantile = order_statistics(ranked, held, level)  # (r,)
        covered, size = set_measure(quantile[:, None])  # (r, n), (r, n)
        _record(scores, f"{coverage_name}@{level}", draws, covered)
        _record(scores, f"{size_name}@{level}", draws, size)
        if layout is None:
            continue

        group_quantiles = np.full((repeats, len(IDENTITY_BIN_NAMES)), np.inf)  # (r, G)
        for group, (group_ranked, group_held) in grouped.items():
            group_quantiles[:, group] = order_statistics(group_ranked, group_held, level)
        covered, size = set_measure(np.take_along_axis(group_quantiles, layout.row_groups, axis=1))  # (r, n), (r, n)
        _record(scores, f"mondrian_{coverage_name}@{level}", draws, covered)
        _record(scores, f"mondrian_{size_name}@{level}", draws, size)


def _classification_conformal(
    scores: RunScores,
    probs: np.ndarray,
    labels: np.ndarray,
    draws: Draws,
    conformal_rows: np.ndarray,
    levels: Sequence[float],
    layout: IdentityLayout | None,
) -> None:
    """LAC and APS sets, as `conformal_sets.classification_conformal_rows` builds them."""
    # probs: (r or 1, n, c); labels: (n,)
    # conformal_rows: (r, m_conf)
    repeats, rows, classes = probs.shape
    for score in SET_SCORES:
        every_class = class_scores(probs.reshape(repeats * rows, classes), score).reshape(repeats, rows, classes)  # (r or 1, n, c)
        true_scores = np.take_along_axis(every_class, labels[None, :, None], axis=2)[..., 0]  # (r or 1, n)

        def set_measure(quantile: np.ndarray, every_class: np.ndarray = every_class, true_scores: np.ndarray = true_scores) -> tuple[np.ndarray, np.ndarray]:
            # quantile: (r, 1) or (r, n)
            # every_class: (r or 1, n, c), true_scores: (r or 1, n)
            covered = true_scores <= quantile  # (r, n)
            size = np.zeros(covered.shape)  # (r, n)
            for index in range(every_class.shape[2]):
                size += every_class[..., index] <= quantile
            return covered, size  # (r, n), (r, n)

        _conformal_scores(scores, draws, conformal_rows, levels, (f"{score}_coverage", f"{score}_size"), true_scores, set_measure, layout)


def _regression_conformal(
    scores: RunScores,
    prediction: RepeatPredictive,
    labels: np.ndarray,
    draws: Draws,
    conformal_rows: np.ndarray,
    levels: Sequence[float],
    spread_floor: float,
    layout: IdentityLayout | None,
) -> None:
    """Normalized residual intervals, as `conformal_sets.regression_conformal_rows` and `split_conformal` build them."""
    # labels: (n,)
    # conformal_rows: (r, m_conf)
    mean = prediction.mean  # (n,)
    if prediction.variance is None:
        # No spread: a zero sigma with a unit floor scores the absolute residual itself.
        scale = np.zeros(mean.shape)[None] + 1.0  # (1, n)
    else:
        scale = np.sqrt(prediction.variance).astype(np.float64) + spread_floor  # (r or 1, n)
    true_scores = np.abs(labels - mean) / scale  # (r or 1, n)

    def set_measure(quantile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # quantile: (r, 1) or (r, n)
        half_width = quantile * scale  # (r, n)
        lower, upper = mean - half_width, mean + half_width  # (r, n), (r, n)
        return (labels >= lower) & (labels <= upper), upper - lower  # (r, n), (r, n)

    _conformal_scores(scores, draws, conformal_rows, levels, ("conformal_coverage", "conformal_width"), true_scores, set_measure, layout)


def _selective(
    scores: RunScores, draws: Draws, losses: np.ndarray, errors: np.ndarray, uncertainty: np.ndarray, layout: IdentityLayout | None
) -> None:
    """AURC, error-detection AUROC, mean uncertainty, and the shift AUROC."""
    # losses, errors: (r or 1, n); uncertainty: (r or 1, n)
    counts = draws.counts  # (r, n)
    scores.overall["mean_uncertainty"] = draws.mean(uncertainty)  # (r,)
    scores.overall["aurc"] = weighted_aurc(counts, losses, uncertainty)  # (r,)
    scores.overall["error_auroc"] = weighted_auroc(counts, errors, uncertainty)  # (r,)
    if layout is not None:
        scores.overall["shift_auroc"] = weighted_auroc(counts * layout.shift_rows, layout.novel, uncertainty)  # (r,)


def score_repeats(
    prediction: RepeatPredictive,
    kind: str,
    score_only: bool,
    labels: np.ndarray,
    counts: np.ndarray,
    conformal_rows: np.ndarray,
    levels: Sequence[float],
    spread_floor: float,
    layout: IdentityLayout | None,
) -> RunScores:
    """Every score of one run's predictions on each repeat's resample of T_eval."""
    # labels: (n,) or (n, c); counts: (r, n); conformal_rows: (r, m_conf)
    scores = RunScores()
    draws = Draws.of(counts, layout)
    uncertainty = prediction.uncertainty  # (r or 1, n) or None

    if kind == "regression":
        mean = prediction.mean  # (n,)
        if not score_only:
            scores.overall["spearman"] = weighted_spearman(counts, mean, labels)  # (r,)
            if prediction.variance is not None:
                variance = np.maximum(prediction.variance, PROBABILITY_FLOOR)  # (r or 1, n)
                nll = 0.5 * np.log(2.0 * np.pi * variance) + (labels - mean) ** 2 / (2.0 * variance)  # (r or 1, n)
                scores.overall["gaussian_nll"] = draws.mean(nll)  # (r,)
                for level in levels:
                    covered, width = gaussian_interval(labels, mean, prediction.variance, level)  # (r or 1, n) each
                    scores.overall[f"interval_coverage@{level}"] = draws.mean(covered)  # (r,)
                    scores.overall[f"interval_width@{level}"] = draws.mean(width)  # (r,)
            _regression_conformal(scores, prediction, labels, draws, conformal_rows, levels, spread_floor, layout)
        if uncertainty is not None:
            losses = np.abs(labels - mean).astype(np.float64)[None]  # (1, n)
            errors = losses > weighted_median(counts, losses)[:, None]  # (r, n)
            _selective(scores, draws, losses, errors, uncertainty, layout)
            scores.overall["uncertainty_error_spearman"] = weighted_spearman(counts, uncertainty, losses)  # (r,)
        return scores

    probs = prediction.probs  # (r or 1, n, c)
    if kind == "multilabel":
        if not score_only:
            if probs.shape[0] == 1:
                draw_scores = multilabel_draw_scores(counts, probs[0], labels, ECE_BINS)
            else:
                draw_scores = multilabel_draw_scores(counts, probs, labels, ECE_BINS, entry_order=prediction.entry_order)
            scores.overall["f1"], scores.overall["ece"] = draw_scores  # (r,), (r,)
            # Clipped in float64, as metrics.negative_log_likelihood clips it.
            clipped = np.clip(probs.astype(np.float64), PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)  # (r or 1, n, c)
            nll = -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped), axis=2)  # (r or 1, n)
            scores.overall["nll"] = draws.mean(nll)  # (r,)
            scores.overall["brier"] = draws.mean(np.mean((probs - labels) ** 2, axis=2))  # (r,)
        losses = np.mean((probs > 0.5) != (labels > 0.5), axis=2).astype(np.float64)  # (r or 1, n)
        _selective(scores, draws, losses, losses > 0.0, uncertainty, layout)
        return scores

    classes = probs.shape[2]
    predicted = probs.argmax(axis=2)  # (r or 1, n)
    losses = (predicted != labels).astype(np.float64)  # (r or 1, n)
    if not score_only:
        scores.overall["mcc"] = weighted_mcc(counts, labels, predicted, classes)  # (r,)
        true_probs = np.take_along_axis(probs, labels[None, :, None], axis=2)[..., 0]  # (r or 1, n)
        clipped = np.clip(true_probs, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)  # (r or 1, n)
        scores.overall["nll"] = draws.mean(-np.log(clipped))  # (r,)
        if kind == "binary":
            scores.overall["brier"] = draws.mean((probs[..., 1] - labels) ** 2)  # (r,)
            scores.overall["ece"] = weighted_ece(counts, labels, probs[..., 1], ECE_BINS)  # (r,)
        else:
            onehot = np.eye(classes)[labels]  # (n, c)
            scores.overall["brier"] = draws.mean(np.sum((probs - onehot) ** 2, axis=2))  # (r,)
            scores.overall["ece"] = weighted_ece(counts, predicted == labels, probs.max(axis=2), ECE_BINS)  # (r,)
        _classification_conformal(scores, probs, labels, draws, conformal_rows, levels, layout)
    _selective(scores, draws, losses, losses > 0.0, uncertainty, layout)
    return scores


def _chunk_size(plan: RunPlan) -> int:
    """Repeats whose calibrated predictions, and the arrays scoring derives from them, fit in `CHUNK_BYTES`."""
    per_repeat = plan.fixed.probs.size if plan.fixed.probs is not None else plan.fixed.mean.size
    return max(1, CHUNK_BYTES // (8 * 8 * per_repeat))


def score_plan(
    plan: RunPlan,
    kind: str,
    labels: np.ndarray,
    design: ExchangeableDesign,
    repeats: slice,
    maps: Mapping[str, np.ndarray] | None,
    levels: Sequence[float],
    spread_floor: float,
    layout: IdentityLayout | None,
) -> RunScores:
    """One run's scores in a window of repeats: its fixed predictions, or its map in each repeat.

    `maps`, for a calibrated run, holds each map parameter for every repeat in the window.
    """
    # labels: (n,) or (n, c), maps: (r,) per parameter
    if not plan.calibrated:
        return score_repeats(
            fixed_predictive(plan.fixed), kind, plan.score_only, labels, design.eval_counts[repeats],
            design.conformal_rows[repeats], levels, spread_floor, None if layout is None else layout.window(repeats),
        )

    count = repeats.stop - repeats.start
    size = _chunk_size(plan)
    chunks = []
    for offset in range(0, count, size):
        window = slice(repeats.start + offset, repeats.start + min(offset + size, count))
        chunk_maps = {key: values[offset:offset + size] for key, values in maps.items()}  # key -> (r,)
        chunks.append(score_repeats(
            calibrated_predictive(plan, kind, chunk_maps), kind, plan.score_only, labels, design.eval_counts[window],
            design.conformal_rows[window], levels, spread_floor, None if layout is None else layout.window(window),
        ))
    return RunScores(
        overall={name: np.concatenate([chunk.overall[name] for chunk in chunks]) for name in chunks[0].overall},
        by_bin={name: np.concatenate([chunk.by_bin[name] for chunk in chunks]) for name in chunks[0].by_bin},
    )


def _mean_over_runs(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """The mean over runs of each finite value, NaN where none is, as `report.summarize` skips None."""
    # arrays: (...) each
    stacked = np.stack(arrays)  # (runs, ...)
    defined = np.isfinite(stacked)  # (runs, ...)
    count = defined.sum(axis=0)  # (...)
    total = np.where(defined, stacked, 0.0).sum(axis=0)  # (...)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)  # (...)


def method_scores(plans: Sequence[RunPlan], run_scores: Sequence[RunScores]) -> list[MethodScores]:
    """Each method's scores in each repeat, the mean over its runs."""
    methods: dict[str, list[tuple[RunPlan, RunScores]]] = {}
    for plan, scores in zip(plans, run_scores, strict=True):
        methods.setdefault(plan.method, []).append((plan, scores))
    summaries = []
    for name, runs in methods.items():
        overall = list(dict.fromkeys(metric for _, scores in runs for metric in scores.overall))
        by_bin = list(dict.fromkeys(metric for _, scores in runs for metric in scores.by_bin))
        summaries.append(MethodScores(
            name=name,
            score_only=runs[0][0].score_only,
            overall={metric: _mean_over_runs([scores.overall[metric] for _, scores in runs if metric in scores.overall]) for metric in overall},
            by_bin={metric: _mean_over_runs([scores.by_bin[metric] for _, scores in runs if metric in scores.by_bin]) for metric in by_bin},
        ))
    return summaries


def _relative_gap(mine: float, reference: float | None) -> float | None:
    """How far a score here sits from the reference's, relative to the larger of 1 and the reference.

    0 when neither is defined, None when only one is.
    """
    mine_defined = bool(np.isfinite(mine))
    reference_defined = reference is not None and bool(np.isfinite(reference))
    if not mine_defined and not reference_defined:
        return 0.0
    if mine_defined != reference_defined:
        return None
    return abs(mine - reference) / max(1.0, abs(reference))


def _compare_scores(method: MethodScores, reference: Mapping[str, Any], bins: Mapping[str, Any] | None) -> dict[str, Any]:
    """Every score repeat 0 has both here and in `score_method`, and the widest gap between them."""
    gaps: dict[str, float | None] = {}
    for metric, values in method.overall.items():
        if metric in reference:
            gaps[metric] = _relative_gap(float(values[0]), reference[metric])
    for metric, values in method.by_bin.items():
        for index, bin_name in enumerate(IDENTITY_BIN_NAMES):
            if bins is not None and metric in bins.get(bin_name, {}):
                gaps[f"{metric}[{bin_name}]"] = _relative_gap(float(values[0, index]), bins[bin_name][metric])
    one_sided = sorted(name for name, gap in gaps.items() if gap is None)
    measured = {name: gap for name, gap in gaps.items() if gap is not None}
    worst = max(measured, key=measured.get) if measured else None
    return {
        "compared": len(gaps),
        "largest_relative_gap": None if worst is None else measured[worst],
        "largest_gap_metric": worst,
        "defined_on_one_side": one_sided,
        "passed": not one_sided and (worst is None or measured[worst] <= CHECK_TOLERANCE),
    }


def _check_maps(plan: RunPlan, kind: str, labels: np.ndarray, design: ExchangeableDesign, maps: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Whether the reference map reproduces `build_methods`' predictions, and how the map fitted here compares.

    The objective gap is the repeat-0 fit's objective here minus the reference map's, on the
    same T_fit rows, so a value at or below 0 means this fit is at least as good.
    """
    # labels: (n,) or (n, c), maps: (R,) per parameter
    reference = {key: np.array([value]) for key, value in plan.fitted.items()}
    rebuilt = calibrated_predictive(plan, kind, reference)
    if kind == "regression":
        identical = np.array_equal(rebuilt.variance[0], plan.fixed.variance)
    else:
        identical = np.array_equal(rebuilt.probs[0], plan.fixed.probs)
        if plan.rankings is not None or plan.uncertainty is not None:
            identical = identical and np.array_equal(rebuilt.uncertainty[0], plan.fixed.uncertainty)
    objectives = _map_objectives(plan, kind, labels, design.fit_rows[:1], maps, reference)  # (2,)
    gaps = {key: abs(float(maps[key][0]) - value) / max(abs(value), 1e-12) for key, value in plan.fitted.items()}
    objective_gap = float(objectives[0] - objectives[1])
    passed = bool(identical) and objective_gap <= CHECK_TOLERANCE * max(1.0, abs(float(objectives[1])))
    check = {
        "predictions_identical": bool(identical),
        "map_relative_gap": max(gaps.values()),
        "objective_gap": objective_gap,
    }
    if plan.rankings is not None:
        # The norm order is chosen by T_fit AURC, which the temperature does not move.
        rows = design.fit_rows[0]  # (m_fit,)
        here, fitted = (
            pnorm_aurc(plan.logits[rows], labels[rows], float(chosen[plan.choice][0])) for chosen in (maps, reference)
        )
        check["aurc_gap"] = here - fitted
        passed = passed and here - fitted <= CHECK_TOLERANCE * max(1.0, abs(fitted))
    return {**check, "passed": passed}


def _map_objectives(
    plan: RunPlan,
    kind: str,
    labels: np.ndarray,
    fit_rows: np.ndarray,
    maps: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
) -> np.ndarray:
    """The objective on one repeat's T_fit rows of the first map fitted here and of the reference map."""
    # fit_rows: (1, m_fit)
    # labels: (n,) or (n, c), maps: (R,) per parameter, reference: (1,) per parameter
    values = []
    for chosen in (maps, reference):
        if plan.candidates is not None:
            index = int(_choice_index(plan, chosen[plan.choice][:1])[0])
            candidate = plan.candidates[index]  # (n, c) or (n,)
        else:
            candidate = plan.base.variance if kind == "regression" else plan.logits
        if kind == "regression":
            value = variance_objective(plan.base.mean, candidate, labels, fit_rows, chosen["a"][:1], chosen["b"][:1])  # (1,)
        else:
            value = temperature_objective(candidate, labels, kind in CLASSIFICATION_KINDS, fit_rows, chosen["temperature"][:1])  # (1,)
        values.append(float(value[0]))
    return np.array(values)  # (2,)


def exchangeable_scores(
    test_splits: Mapping[int, SplitPredictions],
    head: str,
    kind: str,
    design: ExchangeableDesign,
    levels: Sequence[float],
    train_label_std: float,
    identity: np.ndarray | None,
    pnorm: bool = False,
) -> ExchangeableResult:
    """Every method's scheme T scores in every repeat, and the checks of this path against repeat 0.

    `test_splits` holds each seed's test predictions; `identity`, when known, each test row's
    highest identity to training. `pnorm` builds `temperature_pnorm`, as the cell's report did.
    """
    # identity: (n,)
    first = next(iter(test_splits.values()))
    labels = first.labels.astype(np.int64) if kind in CLASSIFICATION_KINDS else first.labels  # (n,) or (n, c)
    spread_floor = SPREAD_FLOOR_FRACTION * train_label_std
    splits = reference_splits(test_splits, design, 0)
    methods = build_methods(splits, head, kind, pnorm=pnorm)
    plans = plan_runs(methods, splits, head, kind)
    layout = None if identity is None else identity_layout(identity, design.conformal_rows)

    run_scores, checks = [], []
    for plan in plans:
        maps = None
        if plan.calibrated:
            maps = fit_maps(plan, kind, labels, design.fit_rows)
            checks.append({"check": "calibration map", "method": plan.method, "seed": plan.seed, **_check_maps(plan, kind, labels, design, maps)})
        run_scores.append(score_plan(plan, kind, labels, design, slice(0, design.repeats), maps, levels, spread_floor, layout))

    checks += _check_repeat_zero(methods, plans, kind, splits, design, levels, spread_floor, layout, train_label_std, identity)
    return ExchangeableResult(
        methods=method_scores(plans, run_scores),
        checks=checks,
        calibration_rows=int(design.conformal_rows.shape[1]),
        mondrian_groupings=None if layout is None else mondrian_groupings(layout.bin_groups),
    )


def _check_repeat_zero(
    methods: Sequence[Method],
    plans: Sequence[RunPlan],
    kind: str,
    splits: Mapping[int, Mapping[str, SplitPredictions]],
    design: ExchangeableDesign,
    levels: Sequence[float],
    spread_floor: float,
    layout: IdentityLayout | None,
    train_label_std: float,
    identity: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Repeat 0 scored here, with the maps `build_methods` fitted, against `report.score_method`.

    Each method's first run is checked, since every run of a method takes one path. Per-bin
    scores are checked where this path adds them, which is conformal coverage and size, so not
    for multilabel tasks, whose per-bin reference AUPRC would cost more than the whole scheme.
    """
    # identity: (n,)
    roles = next(iter(splits.values()))
    labels = {role: roles[role].labels for role in ("v_fit", "v_cal", "test")}
    test_labels = roles[ALL_TEST].labels
    test_labels = test_labels.astype(np.int64) if kind in CLASSIFICATION_KINDS else test_labels  # (n,) or (n, c)
    test_bins = None
    if identity is not None and kind != "multilabel":
        test_bins = bin_rows(identity[np.repeat(np.arange(identity.shape[0]), design.eval_counts[0])])

    checks = []
    for method in methods:
        first_run = method.runs[0]
        checked = [plan for plan in plans if plan.method == method.name and plan.seed == first_run.seed]
        run_scores = [
            score_plan(
                plan, kind, test_labels, design, slice(0, 1),
                {key: np.array([value]) for key, value in plan.fitted.items()} if plan.calibrated else None,
                levels, spread_floor, layout,
            )
            for plan in checked
        ]
        mine = method_scores(checked, run_scores)[0]
        reference = score_method(replace(method, runs=(first_run,)), kind, labels, tuple(levels), train_label_std, test_bins)
        checks.append({
            "check": "repeat 0 against score_method", "method": method.name, "seed": first_run.seed,
            **_compare_scores(mine, reference["test"], reference.get("identity_bins")),
        })
    return checks
