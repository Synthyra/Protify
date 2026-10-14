"""Uncertainty from the calibration set alone: how like each V_cal row a query is, and how right the model was there.

Every row has a correctness `e`: the argmax right for a single-label task, the whole predicted
label set right for multilabel, and the absolute error within `tau` for regression, where `tau`
is V_cal's median absolute error. The model's own claim of it is `p`: the temperature-scaled
probability of the argmax, or of the predicted label set (the product over labels of the chosen
side's sigmoid). A regression head states no per-row confidence, so its claim is V_cal's share
within `tau` on every row.

For a query x and a V_cal row i, with z the trunk features the output layer reads:

    s_i = cos(z(x), z_i),    w_i = exp((s_i - 1) / h)

w_i is 1 for a row pointing the way x does and falls with the angle between them, at a rate the
bandwidth h sets. Each estimator gives q(x), an estimate of P(correct | x):

- `model`: p(x), the head's own claim.
- `knn`: the share correct of the k V_cal rows most similar to x.
- `joint`: kernel regression of e on latent similarity and the model's confidence together, the
  weight w_i * exp(-(l_i - l(x))^2 / (2 g^2)) with l = logit p, as LoRe (Luo et al. 2022), shrunk
  toward p(x) by m pseudo-rows so that a query with no weighted row keeps the model's claim.
- `residual`: q(x) = p(x) + sum_i w_i (e_i - p_i) / (sum_i w_i + m). The kernel-weighted mean of
  how far the V_cal rows near x fell short of, or beat, what the model promised them, shrunk
  toward zero by m pseudo-rows. With no V_cal row near x, q is p; near rows the model was
  overconfident on pull q down, and near rows it was underconfident on push q up.

Each estimator's settings are chosen by the leave-one-out log-likelihood of V_cal's correctness,
every V_cal row scored from all the others, so no test row is read. The same leave-one-out scores
choose the answer rule's threshold (`answer_threshold`).

The evidence behind an estimate is W = sum_i w_i, and n_eff = W^2 / sum_i w_i^2 counts the V_cal
rows it rests on. Regression also gets a local scale, sigma(x)^2 = (sum_i w_i r_i^2 + m s^2) /
(W + m), with r the V_cal residuals and s^2 their mean square, for normalized split conformal
intervals mu +- c sigma(x).

Shapes: `n` queries, `r` V_cal rows, `p` trunk width, `b` queries in a block, `c` classes or labels.
"""

from __future__ import annotations

import itertools
import numpy as np

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from scipy.special import logit
from scipy.stats import beta

from foundry.eval.conformal import conformal_quantile
from .metrics import CLASSIFICATION_KINDS


# Kernel bandwidths h on 1 - cos, from nearly exact matches to nearly uniform weights.
BANDWIDTHS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
# Pseudo-rows m of prior at the model's claim.
PRIOR_ROWS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
# The k of `knn`.
NEIGHBOR_COUNTS = (5, 10, 20, 50, 100)
# The confidence bandwidths g of `joint`, in logit units.
CONFIDENCE_BANDWIDTHS = (0.25, 0.5, 1.0, 2.0)
ESTIMATORS = ("model", "knn", "joint", "residual")
# q is kept this far inside (0, 1), so a log-likelihood stays finite.
Q_FLOOR = 1e-4
# Queries scored per block, which bounds the (b, r) kernel matrices held at once.
QUERY_BLOCK = 2048
# The answer rule's coverage grid and the confidence of its error bound.
ANSWER_COVERAGES = tuple(round(0.05 * step, 2) for step in range(1, 21))
ANSWER_CONFIDENCE = 0.9


@dataclass(frozen=True)
class CalibrationRows:
    """What the estimators read of V_cal: its trunk directions, correctness, and the model's claims.

    `residuals` is `y - mu` for regression and None otherwise.
    """

    unit: np.ndarray  # (r, p)
    correct: np.ndarray  # (r,)
    claimed: np.ndarray  # (r,)
    residuals: np.ndarray | None = None


@dataclass(frozen=True)
class Estimate:
    """One estimator on the queries and, left out one at a time, on V_cal itself."""

    name: str
    settings: Mapping[str, float]
    q: np.ndarray  # (n,)
    calibration_q: np.ndarray  # (r,)
    loo_log_likelihood: float
    # Every setting tried, as its key and V_cal's leave-one-out mean log-likelihood.
    tried: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalScale:
    """Regression's local residual scale on the queries and, left out one at a time, on V_cal."""

    settings: Mapping[str, float]
    sigma: np.ndarray  # (n,)
    calibration_sigma: np.ndarray  # (r,)
    loo_gaussian_nll: float


@dataclass(frozen=True)
class AnswerRule:
    """Answer a query when its q reaches `threshold`; infinite when V_cal supports no coverage.

    Chosen on V_cal's leave-one-out q as the largest grid coverage whose error's Clopper-Pearson
    upper bound, at `ANSWER_CONFIDENCE`, stays within `target_error`.
    """

    target_error: float
    threshold: float
    calibration_coverage: float
    calibration_error: float | None
    error_bound: float | None


def unit_rows(features: np.ndarray) -> np.ndarray:
    """Each row scaled to unit length, in float32; a zero row stays zero."""
    # features: (n, p)
    features = np.asarray(features, dtype=np.float32)  # (n, p)
    norms = np.linalg.norm(features, axis=1, keepdims=True)  # (n, 1)
    return features / np.maximum(norms, np.float32(1e-12))  # (n, p)


def cosine_similarities(queries: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Cosine similarity of every unit query row to every unit reference row, `(n, r)`, in float32."""
    # queries: (n, p), reference: (r, p)
    return np.clip(queries @ reference.T, -1.0, 1.0)  # (n, r)


def kernel_weights(similarities: np.ndarray, bandwidth: float) -> np.ndarray:
    """`exp((s - 1) / h)` in float64: 1 at the same direction, falling with the angle."""
    # similarities: (b, r)
    return np.exp((similarities.astype(np.float64) - 1.0) / bandwidth)  # (b, r)


def _without_self(weights: np.ndarray, start: int | None) -> np.ndarray:
    """Zero each block row's own column, when the queries are V_cal rows from `start` on."""
    # weights: (b, r)
    if start is not None:
        rows = np.arange(weights.shape[0])  # (b,)
        weights[rows, start + rows] = 0.0
    return weights  # (b, r)


def claimed_correctness(kind: str, outputs: np.ndarray) -> np.ndarray:
    """The head's own probability that its prediction is right, from temperature-scaled outputs.

    Single-label: the top class's softmax probability. Multilabel: the probability of the whole
    predicted label set under independent sigmoids, each label's chosen side multiplied, computed
    in log space.
    """
    # outputs: (n, c)
    if kind in CLASSIFICATION_KINDS:
        shifted = outputs - outputs.max(axis=1, keepdims=True)  # (n, c)
        return 1.0 / np.exp(shifted).sum(axis=1)  # (n,)
    if kind == "multilabel":
        # log sigmoid(|z|) is the log probability of each label's chosen side.
        return np.exp(-np.logaddexp(0.0, -np.abs(outputs.astype(np.float64))).sum(axis=1))  # (n,)
    raise ValueError(f"a regression head claims no per-row correctness; kind {kind!r}")


def correctness(kind: str, labels: np.ndarray, outputs: np.ndarray, tolerance: float | None = None) -> np.ndarray:
    """Each row's correctness as 0 or 1: the argmax, the whole label set at 0.5, or |error| <= tolerance."""
    # labels: (n,) or (n, c), outputs: (n,) or (n, c)
    if kind in CLASSIFICATION_KINDS:
        return (outputs.argmax(axis=1) == labels).astype(np.float64)  # (n,)
    if kind == "multilabel":
        return np.all((outputs > 0.0) == (labels > 0.5), axis=1).astype(np.float64)  # (n,)
    if tolerance is None:
        raise ValueError("regression correctness needs a tolerance")
    return (np.abs(labels - outputs) <= tolerance).astype(np.float64)  # (n,)


def _clipped(q: np.ndarray) -> np.ndarray:
    # q: (n,)
    return np.clip(q, Q_FLOOR, 1.0 - Q_FLOOR)  # (n,)


def log_likelihood(q: np.ndarray, correct: np.ndarray) -> float:
    """Mean Bernoulli log-likelihood of the outcomes under q."""
    # correct: (n,), q: (n,)
    q = _clipped(q)  # (n,)
    return float(np.mean(correct * np.log(q) + (1.0 - correct) * np.log1p(-q)))


def _blocks(similarities: np.ndarray, calibration: bool) -> list[tuple[np.ndarray, int | None, slice]]:
    # similarities: (n, r)
    return [  # (b, r) blocks of the (n, r) similarities
        (similarities[start:start + QUERY_BLOCK], start if calibration else None, slice(start, start + QUERY_BLOCK))
        for start in range(0, similarities.shape[0], QUERY_BLOCK)
    ]


def _knn(similarities: np.ndarray, rows: CalibrationRows, claimed: np.ndarray, calibration: bool, k: int) -> np.ndarray:
    # claimed: (n,), similarities: (n, r)
    del claimed
    q = np.empty(similarities.shape[0])  # (n,)
    for block, start, span in _blocks(similarities, calibration):
        block = block.astype(np.float64)  # (b, r)
        if start is not None:
            # A V_cal row is never its own neighbor.
            rows_in_block = np.arange(block.shape[0])  # (b,)
            block[rows_in_block, start + rows_in_block] = -np.inf
        count = min(k, block.shape[1] - (1 if calibration else 0))
        nearest = np.argpartition(-block, count - 1, axis=1)[:, :count]  # (b, k)
        q[span] = rows.correct[nearest].mean(axis=1)
    return q  # (n,)


def _residual_sums(similarities: np.ndarray, rows: CalibrationRows, calibration: bool, bandwidth: float) -> tuple[np.ndarray, np.ndarray]:
    """Per query, W = sum w and sum w (e - p) at one bandwidth."""
    # similarities: (n, r)
    shortfall = rows.correct - rows.claimed  # (r,)
    total = np.empty(similarities.shape[0])  # (n,)
    weighted = np.empty(similarities.shape[0])  # (n,)
    for block, start, span in _blocks(similarities, calibration):
        weights = _without_self(kernel_weights(block, bandwidth), start)  # (b, r)
        total[span] = weights.sum(axis=1)
        weighted[span] = weights @ shortfall
    return total, weighted  # (n,), (n,)


def _joint_sums(
    similarities: np.ndarray, rows: CalibrationRows, claimed: np.ndarray, calibration: bool, bandwidth: float, confidence_bandwidth: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per query, the joint kernel's total weight and its weighted count correct."""
    # claimed: (n,), similarities: (n, r)
    reference = logit(_clipped(rows.claimed))  # (r,)
    queries = logit(_clipped(claimed))  # (n,)
    total = np.empty(similarities.shape[0])  # (n,)
    weighted = np.empty(similarities.shape[0])  # (n,)
    for block, start, span in _blocks(similarities, calibration):
        gap = queries[span, None] - reference[None, :]  # (b, r)
        weights = kernel_weights(block, bandwidth) * np.exp(-0.5 * (gap / confidence_bandwidth) ** 2)  # (b, r)
        weights = _without_self(weights, start)  # (b, r)
        total[span] = weights.sum(axis=1)
        weighted[span] = weights @ rows.correct
    return total, weighted  # (n,), (n,)


def _settings_key(settings: Mapping[str, float]) -> str:
    return ",".join(f"{name}={value:g}" for name, value in settings.items())


def _best(tried: Mapping[str, float]) -> str:
    """The setting of highest log-likelihood, the earliest tried on a tie."""
    return max(tried, key=lambda key: (tried[key], -list(tried).index(key)))


def _chosen(
    name: str,
    grid: Sequence[Mapping[str, float]],
    score: Callable[[np.ndarray, np.ndarray, bool, Mapping[str, float]], np.ndarray],
    rows: CalibrationRows,
    calibration_similarities: np.ndarray,
    query_similarities: np.ndarray,
    query_claimed: np.ndarray,
) -> Estimate:
    """Try every setting on V_cal left out one at a time, keep the best, and score the queries with it."""
    # score: (n,) output, from (n, r) and (n,) inputs, calibration_similarities: (r, r), query_similarities: (n, r), query_claimed: (n,)
    tried, scores = {}, {}
    for settings in grid:
        key = _settings_key(settings)
        scores[key] = _clipped(score(calibration_similarities, rows.claimed, True, settings))  # (r,)
        tried[key] = log_likelihood(scores[key], rows.correct)
    key = _best(tried)
    settings = grid[list(tried).index(key)]
    q = _clipped(score(query_similarities, query_claimed, False, settings))  # (n,)
    return Estimate(name, dict(settings), q, scores[key], tried[key], tried)


def estimate(
    name: str, rows: CalibrationRows, calibration_similarities: np.ndarray, query_similarities: np.ndarray, query_claimed: np.ndarray
) -> Estimate:
    """One estimator of P(correct), its settings chosen on V_cal, applied to the queries.

    `calibration_similarities` is V_cal against itself `(r, r)`, `query_similarities` the queries
    against V_cal `(n, r)`, and `query_claimed` the model's claim on each query.
    """
    # calibration_similarities: (r, r), query_similarities: (n, r), query_claimed: (n,)
    if name == "model":
        calibration_q = _clipped(rows.claimed)  # (r,)
        return Estimate(name, {}, _clipped(query_claimed), calibration_q, log_likelihood(calibration_q, rows.correct))

    if name == "knn":
        grid = [{"k": float(k)} for k in NEIGHBOR_COUNTS if k < rows.correct.shape[0]]

        def score(similarities: np.ndarray, claimed: np.ndarray, calibration: bool, settings: Mapping[str, float]) -> np.ndarray:
            # similarities: (n, r), claimed: (n,)
            return _knn(similarities, rows, claimed, calibration, int(settings["k"]))  # (n,)

        return _chosen(name, grid, score, rows, calibration_similarities, query_similarities, query_claimed)

    if name == "residual":
        return _residual_estimate(rows, calibration_similarities, query_similarities, query_claimed)

    if name == "joint":
        grid = [
            {"h": h, "g": g, "m": m} for h, g, m in itertools.product(BANDWIDTHS, CONFIDENCE_BANDWIDTHS, PRIOR_ROWS)
        ]
        cache: dict[tuple[float, float, bool], tuple[np.ndarray, np.ndarray]] = {}

        def score(similarities: np.ndarray, claimed: np.ndarray, calibration: bool, settings: Mapping[str, float]) -> np.ndarray:
            # The kernel sums depend on h and g alone; each m reuses them.
            # similarities: (n, r), claimed: (n,)
            key = (settings["h"], settings["g"], calibration)
            if key not in cache:
                cache[key] = _joint_sums(similarities, rows, claimed, calibration, settings["h"], settings["g"])
            total, weighted = cache[key]
            return (weighted + settings["m"] * claimed) / (total + settings["m"])  # (n,)

        return _chosen(name, grid, score, rows, calibration_similarities, query_similarities, query_claimed)

    raise ValueError(f"estimator must be one of {ESTIMATORS}, not {name!r}")


def _residual_estimate(
    rows: CalibrationRows, calibration_similarities: np.ndarray, query_similarities: np.ndarray, query_claimed: np.ndarray
) -> Estimate:
    # calibration_similarities: (r, r), query_similarities: (n, r), query_claimed: (n,)
    grid = [{"h": h, "m": m} for h, m in itertools.product(BANDWIDTHS, PRIOR_ROWS)]
    cache: dict[tuple[float, bool], tuple[np.ndarray, np.ndarray]] = {}

    def score(similarities: np.ndarray, claimed: np.ndarray, calibration: bool, settings: Mapping[str, float]) -> np.ndarray:
        # similarities: (n, r), claimed: (n,)
        key = (settings["h"], calibration)
        if key not in cache:
            cache[key] = _residual_sums(similarities, rows, calibration, settings["h"])
        total, weighted = cache[key]
        return claimed + weighted / (total + settings["m"])  # (n,)

    return _chosen("residual", grid, score, rows, calibration_similarities, query_similarities, query_claimed)


def evidence(similarities: np.ndarray, bandwidth: float) -> tuple[np.ndarray, np.ndarray]:
    """Per query, the total kernel weight W on V_cal and the effective count n_eff = W^2 / sum w^2."""
    # similarities: (n, r)
    total = np.empty(similarities.shape[0])  # (n,)
    effective = np.empty(similarities.shape[0])  # (n,)
    for block, _, span in _blocks(similarities, False):
        weights = kernel_weights(block, bandwidth)  # (b, r)
        total[span] = weights.sum(axis=1)
        squares = (weights ** 2).sum(axis=1)  # (b,)
        effective[span] = np.where(squares > 0.0, total[span] ** 2 / np.maximum(squares, 1e-300), 0.0)
    return total, effective  # (n,), (n,)


def local_scale(residuals: np.ndarray, calibration_similarities: np.ndarray, query_similarities: np.ndarray) -> LocalScale:
    """sigma(x)^2 = (sum w r^2 + m s^2) / (W + m), its h and m chosen by V_cal's leave-one-out Gaussian NLL."""
    # residuals: (r,), calibration_similarities: (r, r), query_similarities: (n, r)
    squares = residuals.astype(np.float64) ** 2  # (r,)
    prior = float(squares.mean())

    def sigma(similarities: np.ndarray, calibration: bool, bandwidth: float, rows: float) -> np.ndarray:
        # similarities: (n, r)
        values = np.empty(similarities.shape[0])  # (n,)
        for block, start, span in _blocks(similarities, calibration):
            weights = _without_self(kernel_weights(block, bandwidth), start)  # (b, r)
            values[span] = (weights @ squares + rows * prior) / (weights.sum(axis=1) + rows)
        return np.sqrt(np.maximum(values, 1e-12))  # (n,)

    tried, scales = {}, {}
    for h, m in itertools.product(BANDWIDTHS, PRIOR_ROWS):
        key = _settings_key({"h": h, "m": m})
        scales[key] = sigma(calibration_similarities, True, h, m)  # (r,)
        variance = scales[key] ** 2  # (r,)
        tried[key] = -float(np.mean(0.5 * np.log(2.0 * np.pi * variance) + squares / (2.0 * variance)))
    key = _best(tried)
    h, m = (float(part.split("=")[1]) for part in key.split(","))
    return LocalScale({"h": h, "m": m}, sigma(query_similarities, False, h, m), scales[key], -tried[key])


def normalized_interval_half_widths(
    residuals: np.ndarray, calibration_sigma: np.ndarray, query_sigma: np.ndarray, level: float
) -> np.ndarray:
    """Split conformal half-widths c * sigma(x), c the conformal quantile of V_cal's |r| / sigma."""
    # residuals: (r,), calibration_sigma: (r,), query_sigma: (n,)
    scores = np.abs(residuals) / calibration_sigma  # (r,)
    return conformal_quantile(scores, 1.0 - level) * query_sigma  # (n,)


def clopper_pearson_upper(errors: int, rows: int, confidence: float = ANSWER_CONFIDENCE) -> float:
    """The one-sided Clopper-Pearson upper bound on an error rate from `errors` of `rows`."""
    if rows == 0:
        return 1.0
    if errors >= rows:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, rows - errors))


def answer_threshold(
    calibration_q: np.ndarray, calibration_correct: np.ndarray, target_error: float, coverages: Sequence[float] = ANSWER_COVERAGES
) -> AnswerRule:
    """The lowest q threshold whose answered V_cal rows' error bound stays within the target.

    Each grid coverage c answers the V_cal rows whose leave-one-out q reaches the round(c r)-th
    highest q, ties included. The rule keeps the largest coverage whose error's Clopper-Pearson
    upper bound is at most `target_error`; with none, it answers nothing.
    """
    # calibration_correct: (r,), calibration_q: (r,)
    order = np.sort(calibration_q)[::-1]  # (r,)
    rows = calibration_q.shape[0]
    best = AnswerRule(target_error, float("inf"), 0.0, None, None)
    for coverage in coverages:
        threshold = float(order[max(1, round(coverage * rows)) - 1])
        answered = calibration_q >= threshold  # (r,)
        count = int(answered.sum())
        errors = int(count - calibration_correct[answered].sum())
        bound = clopper_pearson_upper(errors, count)
        if bound <= target_error:
            best = AnswerRule(target_error, threshold, count / rows, errors / count, bound)
    return best
