"""Uncertainty methods, built from the saved per-seed predictions.

Every method is a prediction per row plus an uncertainty score per row, on V_fit, V_cal, and
test. A method either follows one seed, scored per seed and averaged, or pools the seeds.

| Method | Head | Prediction | Uncertainty |
|---|---|---|---|
| `plain` / `sngp` | its own | softmax, sigmoid, or the regression mean; SNGP uses mean-field logits and its predictive variance | 1 - top probability, mean binary entropy for multilabel, or sigma |
| `<head>_entropy` | both | as the head | predictive entropy |
| `<head>_nn_distance` | both | as the head | distance to the nearest training row in trunk space |
| `sngp_variance` | sngp | as the head | the Gaussian process's latent variance |
| `temperature` / `sngp_temperature` | its own | logits / T, or sigma^2 = a v + b, fitted on V_fit | as the head |
| `mc_dropout` | projected | the mean of the dropout passes | as the head, from those passes |
| `mc_dropout_temperature` | projected | `mc_dropout` calibrated on V_fit | as the head |
| `ensemble` / `sngp_ensemble` | its own | the seeds' mean probability, or their Gaussian mixture | as the head |
| `ensemble_temperature` / `sngp_ensemble_temperature` | its own | the ensemble calibrated on V_fit | as the head |
| `temperature_pnorm` | any, multiclass, when asked | `temperature`'s | `-max z / ||z||_p` of the centered logits `z`, `p` chosen on V_fit by AURC |
| `laplace_temperature` | any, with a Laplace variance | logits scaled by `1 / sqrt(1 + pi v / 8)`, then T; or `N(mean, a v + b)` | as the head |
| `loss_head` | any, with a loss prediction | `temperature`'s; or `N(mean, a exp(l) + b)` | the predicted log loss `l`; or sigma |

The heads `linear` and `projected_qclip` name their methods as `projected` does (`plain`,
`temperature`, `ensemble`), and `faithful` and `beta_nll` by their own name (`faithful`,
`faithful_temperature`, ...). A report holds one head, so the names never meet in one report.

Methods whose name ends in `_entropy`, `_nn_distance`, or `_variance` share their head's
predictions and differ only in how they rank rows, so only their ranking is scored, as is
`temperature_pnorm`, which shares `temperature`'s predictions.

`laplace_temperature` chooses its prior precision from `LAPLACE_PRIOR_PRECISIONS` jointly with
its map: for each precision it fits T, or `(a, b)`, on V_fit, and keeps the precision whose
fitted map reaches the lowest V_fit negative log-likelihood, the smallest precision on a tie.
`temperature_pnorm` keeps the order in `PNORM_ORDERS` whose ranking reaches the lowest V_fit
AURC, the smallest order on a tie; its ranking does not depend on T. It centers each row's
logits first, as MaxLogit-pNorm does (Cattelan and Silva, UAI 2024), since softmax ignores a
shift of the logits and the score must too. Centered, two logits are always `(-d, d)`, whose
score is the constant `-2^(-1/p)`, so the method is built for multiclass tasks only.

A name ending in `temperature` is its method calibrated on V_fit, so the head, MC dropout, and
the ensembles are each compared calibrated as well as raw: classification divides logits by a
temperature T, and regression maps the variance to sigma^2 = a v + b by Gaussian likelihood.
The head calibrates its own logits. MC dropout and the ensembles keep only mean probabilities, so their temperature
divides the log of those probabilities, or their log odds for multilabel, which T = 1 returns
unchanged. For regression the plain head has no variance, so its map fits a constant
sigma^2 = b; the SNGP one rescales the Gaussian process variance, MC dropout's the spread of
its passes, and an ensemble's its mixture variance. Temperature and variance maps read V_fit
and nothing else.

Shapes: `n` rows, `c` classes or labels, `S` seeds, `L` prior precisions, `K` candidates.
"""

from __future__ import annotations

import numpy as np

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from scipy.optimize import minimize
from scipy.special import expit, log_softmax, softmax

from foundry.eval.temperature import fit_temperature
from foundry.metrics import area_under_risk_coverage
from .metrics import CLASSIFICATION_KINDS, PROBABILITY_FLOOR, tie_averaged_losses
from .prediction_arrays import prediction_mask_is_complete, release_prediction_pages, transform_prediction_rows
from .predictions import SplitPredictions
from ..metrics import validate_observed_labels
from ..probes.sngp import MEAN_FIELD_FACTOR


HEAD_NAMES = {
    "projected": "plain",
    "sngp": "sngp",
    "linear": "plain",
    "projected_qclip": "plain",
    "faithful": "faithful",
    "beta_nll": "beta_nll",
}
# Smallest variance a calibrated Gaussian may report.
VARIANCE_FLOOR = 1e-12
# The last-layer Laplace posterior's prior precisions, log-spaced: one latent variance per entry.
LAPLACE_PRIOR_PRECISIONS = tuple(float(value) for value in np.logspace(-4.0, 8.0, 13))
# The norm orders `temperature_pnorm` chooses among.
PNORM_ORDERS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
# The task kinds `temperature_pnorm` is defined for: a binary task's centered score is constant.
PNORM_KINDS = ("multiclass",)
# Methods built from the per-seed extras, beyond the ablation's.
LAPLACE_METHOD = "laplace_temperature"
LOSS_HEAD_METHOD = "loss_head"
PNORM_METHOD = "temperature_pnorm"
# The offset inside the loss predictor's target, log(loss + LOSS_OFFSET).
LOSS_OFFSET = 1e-6


@dataclass(frozen=True)
class Predictive:
    """One method's prediction on one split: probabilities, or a mean and variance."""

    probs: np.ndarray | None = None  # (n, c)
    mean: np.ndarray | None = None  # (n,)
    variance: np.ndarray | None = None  # (n,)
    uncertainty: np.ndarray | None = None  # (n,), higher is less certain


@dataclass(frozen=True)
class MethodRun:
    """A method on every split, from one seed, or from all of them when `seed` is None."""

    seed: int | None
    splits: Mapping[str, Predictive]
    fitted: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Method:
    name: str
    runs: tuple[MethodRun, ...]
    # Shares its predictions with another method, so only its ranking is scored.
    score_only: bool = False


def probabilities(outputs: np.ndarray, kind: str, *, temperature: float | None = None) -> np.ndarray:
    """Softmax over classes, or a sigmoid per label; temperature scaling uses float64.

    A large fitted temperature can round float32 probabilities to equal values, changing
    the decoded answer even though positive scaling preserves logit order and sign.
    Convert inside each block so mapped inputs retain bounded intermediates.
    """
    # outputs: (n, c)
    def transform(block: np.ndarray) -> dict[str, np.ndarray]:
        # block: (b, c)
        scaled = block if temperature is None else np.asarray(block, dtype=np.float64) / temperature  # (b, c)
        return {"probs": expit(scaled) if kind == "multilabel" else softmax(scaled, axis=1)}  # (b, c) probs
    return transform_prediction_rows(transform, outputs)["probs"]  # (n, c)


def probability_logits(probs: np.ndarray, kind: str) -> np.ndarray:
    """Logits whose `probabilities` are `probs`: log p for a softmax, log odds for sigmoids.

    A softmax ignores a constant added to a row, so log p serves as its logits. Probabilities
    are clipped off 0 and 1 in float64, where a float32 1.0 would give infinite log odds.
    """
    # probs: (n, c)
    def transform(block: np.ndarray) -> dict[str, np.ndarray]:
        # block: (b, c)
        clipped = np.clip(np.asarray(block, dtype=np.float64), PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)  # (b, c)
        return {"logits": np.log(clipped) - np.log1p(-clipped) if kind == "multilabel" else np.log(clipped)}  # (b, c) logits
    return transform_prediction_rows(transform, probs)["logits"]  # (n, c)


def calibrated_probabilities(probs: np.ndarray, kind: str, temperature: float) -> np.ndarray:
    """Temperature on probability logits, fused within a block to avoid a complete logit copy."""
    # probs: (n, c)
    def transform(block: np.ndarray) -> dict[str, np.ndarray]:
        # block: (b, c)
        return {"probs": probabilities(probability_logits(block, kind), kind, temperature=temperature)}  # (b, c)
    return transform_prediction_rows(transform, probs)["probs"]  # (n, c)


def top_label_uncertainty(probs: np.ndarray) -> np.ndarray:
    # probs: (n, c)
    return transform_prediction_rows(lambda block: {"uncertainty": 1.0 - block.max(axis=1)}, probs)["uncertainty"]  # (n,)


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    # probs: (n, c)
    def transform(block: np.ndarray) -> dict[str, np.ndarray]:
        # block: (b, c)
        clipped = np.clip(block, PROBABILITY_FLOOR, 1.0)  # (b, c)
        return {"entropy": -np.sum(clipped * np.log(clipped), axis=1)}  # (b,)
    return transform_prediction_rows(transform, probs)["entropy"]  # (n,)


def mean_binary_entropy(probs: np.ndarray) -> np.ndarray:
    # Clipped in float64: in float32, 1 - PROBABILITY_FLOOR rounds to 1.0, so a probability that
    # reached 1.0, as an MC dropout mean of saturated sigmoids does, would give 0 * log(0) = NaN.
    # probs: (n, c)
    def transform(block: np.ndarray) -> dict[str, np.ndarray]:
        # block: (b, c)
        clipped = np.clip(np.asarray(block, dtype=np.float64), PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)  # (b, c)
        return {"entropy": -np.mean(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped), axis=1)}  # (b,)
    return transform_prediction_rows(transform, probs)["entropy"]  # (n,)


def classification_predictive(probs: np.ndarray, kind: str) -> Predictive:
    # probs: (n, c)
    uncertainty = top_label_uncertainty(probs) if kind in CLASSIFICATION_KINDS else mean_binary_entropy(probs)
    return Predictive(probs=probs, uncertainty=uncertainty)


def regression_predictive(mean: np.ndarray, variance: np.ndarray | None) -> Predictive:
    # mean: (n,), variance: (n,) or None
    uncertainty = None if variance is None else transform_prediction_rows(lambda block: {"sigma": np.sqrt(block)}, variance)["sigma"]  # (n,)
    return Predictive(mean=mean, variance=variance, uncertainty=uncertainty)


def fit_variance_calibration(
    mean: np.ndarray, variance: np.ndarray | None, labels: np.ndarray
) -> tuple[float, float]:
    """The `(a, b)` whose Gaussian `N(mean, a * variance + b)` best explains `labels`.

    Minimizes the mean Gaussian negative log-likelihood over `log a` and `log b`, which keeps
    both positive. With no variance, or an all-zero one, only `b` is identifiable, and its
    maximum-likelihood value is the mean squared residual.
    """
    # mean: (n,), variance: (n,) or None, labels: (n,)
    squared_residuals = (np.asarray(labels, dtype=np.float64) - mean) ** 2  # (n,)
    if variance is None or not np.any(variance > 0):
        return 0.0, float(max(squared_residuals.mean(), VARIANCE_FLOOR))

    variance = np.asarray(variance, dtype=np.float64)  # (n,)
    count = squared_residuals.shape[0]

    def negative_log_likelihood(log_parameters: np.ndarray) -> tuple[float, np.ndarray]:
        # log_parameters: (2,)
        a, b = np.exp(log_parameters)
        scale = a * variance + b  # (n,)
        value = 0.5 * np.mean(np.log(scale) + squared_residuals / scale)
        per_row = 0.5 * (1.0 / scale - squared_residuals / scale ** 2) / count  # (n,)
        return float(value), np.array([np.sum(per_row * variance) * a, np.sum(per_row) * b])  # (), then (2,)

    start_b = max(squared_residuals.mean() - variance.mean(), 0.1 * squared_residuals.mean(), VARIANCE_FLOOR)
    optimum = minimize(
        negative_log_likelihood, np.log([1.0, start_b]), jac=True, method="L-BFGS-B"
    )
    a, b = np.exp(optimum.x)
    return float(a), float(max(b, VARIANCE_FLOOR))


def calibrated_variance(variance: np.ndarray | None, a: float, b: float, count: int) -> np.ndarray:
    # variance: (n,) or None
    raw = np.zeros(count) if variance is None else variance  # (n,)
    return transform_prediction_rows(lambda block: {"variance": np.maximum(a * block + b, VARIANCE_FLOOR)}, raw)["variance"]  # (n,)


def head_run(seed: int, splits: Mapping[str, SplitPredictions], kind: str) -> MethodRun:
    if kind == "regression":
        predictive = {name: regression_predictive(split.outputs, split.variance) for name, split in splits.items()}
    else:
        predictive = {
            name: classification_predictive(probabilities(split.outputs, kind), kind) for name, split in splits.items()
        }
    return MethodRun(seed, predictive)


def _rescored(run: MethodRun, uncertainty: Mapping[str, np.ndarray]) -> MethodRun:
    # uncertainty: (n,) per split name
    return MethodRun(run.seed, {name: replace(split, uncertainty=uncertainty[name]) for name, split in run.splits.items()})


def _fit_observed_temperature(outputs: np.ndarray, labels: np.ndarray, kind: str, label_mask: np.ndarray | None) -> float:
    """Fit on observed validation targets in bounded blocks, retaining mapped input storage."""
    # outputs: (n, c); labels, label_mask: (n,), scalar columns (n, 1), or multilabel (n, c)
    if kind in CLASSIFICATION_KINDS and labels.ndim == 2 and labels.shape[1] == 1:
        if label_mask is not None and label_mask.shape != labels.shape:
            raise ValueError("label_mask must be boolean with exactly the labels shape")
        labels = labels.reshape(-1)  # (n,)
        label_mask = None if label_mask is None else label_mask.reshape(-1)  # (n,)

    def release_inputs() -> None:
        for values in (outputs, labels, label_mask):
            if values is not None:
                release_prediction_pages(values)

    return fit_temperature(outputs, labels, multiclass=kind in CLASSIFICATION_KINDS,
                           label_mask=label_mask, release_batch=release_inputs)


def _recalibrated_run(run: MethodRun, fit_labels: np.ndarray, kind: str, label_mask: np.ndarray | None = None) -> MethodRun:
    """`run` calibrated on its own V_fit predictions and V_fit's labels.

    Classification: a temperature on `probability_logits` of its probabilities. Regression:
    `sigma^2 = a * v + b` on its variance.
    """
    # fit_labels: (n,) or (n, c), label_mask: (n,) or (n, c)
    fit = run.splits["v_fit"]
    if kind == "regression":
        if label_mask is not None:
            validate_observed_labels(fit_labels, label_mask)
            if not label_mask.any():
                raise ValueError("variance calibration requires observed validation labels")
        observed = slice(None) if label_mask is None else label_mask.reshape(-1)
        a, b = fit_variance_calibration(fit.mean[observed], None if fit.variance is None else fit.variance[observed], fit_labels.reshape(-1)[observed])
        predictive = {
            name: regression_predictive(split.mean, calibrated_variance(split.variance, a, b, split.mean.shape[0]))
            for name, split in run.splits.items()
        }
        return MethodRun(run.seed, predictive, {"a": a, "b": b})

    temperature = _fit_observed_temperature(probability_logits(fit.probs, kind), fit_labels, kind, label_mask)
    predictive = {
        name: classification_predictive(calibrated_probabilities(split.probs, kind, temperature), kind)
        for name, split in run.splits.items()
    }
    return MethodRun(run.seed, predictive, {"temperature": temperature})


def _calibrated_run(seed: int, splits: Mapping[str, SplitPredictions], kind: str) -> MethodRun:
    """Temperature on the head's logits, or `a * v + b` for regression, fitted on V_fit alone."""
    fit = splits["v_fit"]
    if kind == "regression":
        return _recalibrated_run(head_run(seed, splits, kind), fit.labels, kind, fit.label_mask)

    temperature = _fit_observed_temperature(fit.outputs, fit.labels, kind, fit.label_mask)
    predictive = {
        name: classification_predictive(probabilities(split.outputs, kind, temperature=temperature), kind)
        for name, split in splits.items()
    }
    return MethodRun(seed, predictive, {"temperature": temperature})


def mc_dropout_run(seed: int, splits: Mapping[str, SplitPredictions], kind: str) -> MethodRun:
    if kind == "regression":
        predictive = {name: regression_predictive(split.mc_outputs, split.mc_variance) for name, split in splits.items()}
    else:
        predictive = {name: classification_predictive(split.mc_outputs, kind) for name, split in splits.items()}
    return MethodRun(seed, predictive)


def ensemble_run(runs: tuple[MethodRun, ...], kind: str) -> MethodRun:
    """Mean probabilities, or the Gaussian mixture of the seeds' predictions."""
    predictive = {}
    for name in runs[0].splits:
        members = [run.splits[name] for run in runs]
        if kind != "regression":
            averaged = transform_prediction_rows(lambda *blocks: {"probs": np.mean(blocks, axis=0)},
                                                 *(member.probs for member in members))["probs"]  # (n, c)
            predictive[name] = classification_predictive(averaged, kind)
            continue
        def mixture(*blocks: np.ndarray, members: list[Predictive] = members) -> dict[str, np.ndarray]:
            # blocks: (b,) per member mean, then (b,) per supplied variance
            means = np.stack(blocks[:len(members)])  # (S, b)
            supplied = iter(blocks[len(members):])
            variances = np.stack([np.zeros_like(means[index]) if member.variance is None else next(supplied)
                                  for index, member in enumerate(members)])  # (S, b)
            mean = means.mean(axis=0)  # (b,)
            variance = np.maximum((variances + means ** 2).mean(axis=0) - mean ** 2, 0.0)  # (b,)
            return {"mean": mean, "variance": variance}  # (b,), (b,)
        combined = transform_prediction_rows(mixture, *(member.mean for member in members),
                                              *(member.variance for member in members if member.variance is not None))
        predictive[name] = regression_predictive(combined["mean"], combined["variance"])
    return MethodRun(None, predictive)


def temperature_nll(logits: np.ndarray, labels: np.ndarray, kind: str, temperature: float) -> float:
    """The mean negative log-likelihood of `labels` under `logits / temperature`, in float64.

    Cross-entropy for single-label tasks, binary cross-entropy averaged over every entry for
    multilabel, as `fit_temperature` defines the objective it minimizes.
    """
    # logits: (n, c); labels: (n,) or (n, c)
    scaled = np.asarray(logits, dtype=np.float64) / temperature  # (n, c)
    if kind in CLASSIFICATION_KINDS:
        rows = np.arange(scaled.shape[0])  # (n,)
        return float(-np.mean(log_softmax(scaled, axis=1)[rows, np.asarray(labels, dtype=np.int64)]))
    targets = np.asarray(labels, dtype=np.float64)  # (n, c)
    # log(1 + e^x) - y x is the binary cross-entropy of a logit x.
    return float(np.mean(np.logaddexp(0.0, scaled) - targets * scaled))


def variance_nll(mean: np.ndarray, variance: np.ndarray, labels: np.ndarray, a: float, b: float) -> float:
    """`0.5 * mean(log s + e / s)` for `s = a v + b` and `e` the squared residual, in float64."""
    # mean, variance, labels: (n,)
    scale = a * np.asarray(variance, dtype=np.float64) + b  # (n,)
    squared_residuals = (np.asarray(labels, dtype=np.float64) - mean) ** 2  # (n,)
    return float(0.5 * np.mean(np.log(scale) + squared_residuals / scale))


def laplace_logits(outputs: np.ndarray, laplace_variance: np.ndarray, kind: str) -> np.ndarray:
    """The head's logits under each prior precision's probit approximation, `z / sqrt(1 + pi v / 8)`.

    Single-label variance is per class `(n, L, c)`; multilabel shares one per row `(n, L)`
    across labels, because its curvature is one scalar shared by every label.
    """
    # outputs: (n, c)
    # laplace_variance: (n, L, c) single-label or (n, L) multilabel
    variance = laplace_variance if laplace_variance.ndim == 3 else laplace_variance[:, :, None]  # (n, L, c) or (n, L, 1)
    scale = np.sqrt(1.0 + MEAN_FIELD_FACTOR * variance.transpose(1, 0, 2))  # (L, n, c) or (L, n, 1)
    return (outputs[None] / scale).astype(outputs.dtype)  # (L, n, c)


def pnorm_uncertainty(outputs: np.ndarray, order: float) -> np.ndarray:
    """`-max_j z_j / ||z||_p` of each row's centered logits `z`: higher is less certain, and T cancels."""
    # outputs: (n, c)
    logits = np.asarray(outputs, dtype=np.float64)  # (n, c)
    logits = logits - logits.mean(axis=1, keepdims=True)  # (n, c)
    norm = np.maximum(np.linalg.norm(logits, ord=order, axis=1), PROBABILITY_FLOOR)  # (n,)
    return -logits.max(axis=1) / norm  # (n,)


def pnorm_aurc(outputs: np.ndarray, labels: np.ndarray, order: float) -> float:
    """The AURC of misclassification ranked by `pnorm_uncertainty`, ties averaged as `metrics.py` does."""
    # outputs: (n, c); labels: (n,)
    losses = (outputs.argmax(axis=1) != labels).astype(np.float64)  # (n,)
    uncertainty = pnorm_uncertainty(outputs, order)  # (n,)
    return area_under_risk_coverage(tie_averaged_losses(losses, uncertainty), uncertainty)


def loss_head_variance(loss_prediction: np.ndarray) -> np.ndarray:
    """The predicted squared error, `exp` of the predicted log loss, as a regression variance."""
    # loss_prediction: (n,)
    return np.exp(np.asarray(loss_prediction, dtype=np.float64))  # (n,)


def _earliest_minimum(values: list[float]) -> int:
    """The index of the smallest value, the first on a tie."""
    return int(np.argmin(np.asarray(values, dtype=np.float64)))


def _pnorm_run(calibrated: MethodRun, splits: Mapping[str, SplitPredictions]) -> MethodRun:
    """`temperature`'s predictions ranked by the p-norm score whose V_fit AURC is lowest."""
    fit = splits["v_fit"]
    index = _earliest_minimum([pnorm_aurc(fit.outputs, fit.labels, order) for order in PNORM_ORDERS])
    order = PNORM_ORDERS[index]
    uncertainty = {name: pnorm_uncertainty(split.outputs, order) for name, split in splits.items()}
    rescored = _rescored(calibrated, uncertainty)
    return MethodRun(rescored.seed, rescored.splits, {**calibrated.fitted, "p": order})


def _laplace_run(seed: int, splits: Mapping[str, SplitPredictions], kind: str) -> MethodRun:
    """The head's Laplace predictive at the prior precision whose calibrated V_fit NLL is lowest."""
    fit = splits["v_fit"]
    if kind == "regression":
        candidates = []
        for index in range(len(LAPLACE_PRIOR_PRECISIONS)):
            variance = fit.laplace_variance[:, index]  # (n,)
            a, b = fit_variance_calibration(fit.outputs, variance, fit.labels)
            candidates.append((variance_nll(fit.outputs, variance, fit.labels, a, b), a, b))
        index = _earliest_minimum([objective for objective, _, _ in candidates])
        _, a, b = candidates[index]
        predictive = {
            name: regression_predictive(
                split.outputs, calibrated_variance(split.laplace_variance[:, index], a, b, split.outputs.shape[0])
            )
            for name, split in splits.items()
        }
        return MethodRun(seed, predictive, {"a": a, "b": b, "prior_precision": LAPLACE_PRIOR_PRECISIONS[index]})

    fit_logits = laplace_logits(fit.outputs, fit.laplace_variance, kind)  # (L, n, c)
    candidates = []
    for index in range(len(LAPLACE_PRIOR_PRECISIONS)):
        temperature = fit_temperature(fit_logits[index], fit.labels, multiclass=kind in CLASSIFICATION_KINDS)
        candidates.append((temperature_nll(fit_logits[index], fit.labels, kind, temperature), temperature))
    index = _earliest_minimum([objective for objective, _ in candidates])
    temperature = candidates[index][1]
    predictive = {
        name: classification_predictive(
            probabilities(laplace_logits(split.outputs, split.laplace_variance, kind)[index], kind, temperature=temperature), kind
        )
        for name, split in splits.items()
    }
    return MethodRun(seed, predictive, {"temperature": temperature, "prior_precision": LAPLACE_PRIOR_PRECISIONS[index]})


def _loss_head_run(
    seed: int, splits: Mapping[str, SplitPredictions], kind: str, calibrated: MethodRun
) -> MethodRun:
    """The predicted loss as the uncertainty: over `temperature`, or as the variance a map rescales."""
    if kind != "regression":
        # In float64, so a mean over rows of predicted log losses near 0 does not cancel in float32.
        uncertainty = {name: np.asarray(split.loss_prediction, dtype=np.float64) for name, split in splits.items()}
        rescored = _rescored(calibrated, uncertainty)
        return MethodRun(seed, rescored.splits, dict(calibrated.fitted))

    base = MethodRun(
        seed,
        {name: regression_predictive(split.outputs, loss_head_variance(split.loss_prediction)) for name, split in splits.items()},
    )
    return _recalibrated_run(base, splits["v_fit"].labels, kind)


def build_methods(
    seed_predictions: Mapping[int, Mapping[str, SplitPredictions]], head: str, kind: str, pnorm: bool = False
) -> list[Method]:
    """Every method the head supports, in the order the report lists them.

    The methods the per-seed extras support follow the ablation's: `temperature_pnorm` under
    `pnorm` for multiclass tasks, `laplace_temperature` when every seed saved a Laplace
    variance, and `loss_head` when every seed saved a loss prediction.
    """
    base = HEAD_NAMES[head]
    seeds = sorted(seed_predictions)
    partial = any(split.label_mask is not None and not prediction_mask_is_complete(split.label_mask) for splits in seed_predictions.values() for split in splits.values())
    if partial and (pnorm or any(split.laplace_variance is not None or split.loss_prediction is not None for splits in seed_predictions.values() for split in splits.values())):
        raise ValueError("partial labels require mask-aware pnorm, Laplace, and loss-head fitting before those optional methods can run")
    head_runs = tuple(head_run(seed, seed_predictions[seed], kind) for seed in seeds)
    methods = [Method(base, head_runs)]

    if kind in CLASSIFICATION_KINDS:
        entropy = tuple(
            _rescored(run, {name: predictive_entropy(split.probs) for name, split in run.splits.items()})
            for run in head_runs
        )
        methods.append(Method(f"{base}_entropy", entropy, score_only=True))

    if all(split.nn_distance is not None for seed in seeds for split in seed_predictions[seed].values()):
        distance = tuple(
            _rescored(run, {name: seed_predictions[run.seed][name].nn_distance for name in run.splits})
            for run in head_runs
        )
        methods.append(Method(f"{base}_nn_distance", distance, score_only=True))

    if head == "sngp" and kind in CLASSIFICATION_KINDS:
        variance = tuple(
            _rescored(run, {name: seed_predictions[run.seed][name].variance for name in run.splits})
            for run in head_runs
        )
        methods.append(Method("sngp_variance", variance, score_only=True))

    calibrated = tuple(_calibrated_run(seed, seed_predictions[seed], kind) for seed in seeds)
    methods.append(Method("temperature" if base == "plain" else f"{base}_temperature", calibrated))

    if head == "projected" and all(
        split.mc_outputs is not None for seed in seeds for split in seed_predictions[seed].values()
    ):
        mc_runs = tuple(mc_dropout_run(seed, seed_predictions[seed], kind) for seed in seeds)
        mc_calibrated = tuple(
            _recalibrated_run(run, seed_predictions[run.seed]["v_fit"].labels, kind, seed_predictions[run.seed]["v_fit"].label_mask) for run in mc_runs
        )
        methods += [Method("mc_dropout", mc_runs), Method("mc_dropout_temperature", mc_calibrated)]

    if len(seeds) >= 2:
        ensemble = ensemble_run(head_runs, kind)
        # The workflow refuses seeds whose labels differ, so any seed's V_fit labels are the ensemble's.
        ensemble_calibrated = _recalibrated_run(ensemble, seed_predictions[seeds[0]]["v_fit"].labels, kind, seed_predictions[seeds[0]]["v_fit"].label_mask)
        name = "ensemble" if base == "plain" else f"{base}_ensemble"
        methods += [Method(name, (ensemble,)), Method(f"{name}_temperature", (ensemble_calibrated,))]

    if pnorm and kind in PNORM_KINDS:
        pnorm_runs = tuple(_pnorm_run(run, seed_predictions[run.seed]) for run in calibrated)
        methods.append(Method(PNORM_METHOD, pnorm_runs, score_only=True))

    if all(split.laplace_variance is not None for seed in seeds for split in seed_predictions[seed].values()):
        methods.append(Method(LAPLACE_METHOD, tuple(_laplace_run(seed, seed_predictions[seed], kind) for seed in seeds)))

    if all(split.loss_prediction is not None for seed in seeds for split in seed_predictions[seed].values()):
        loss_runs = tuple(
            _loss_head_run(run.seed, seed_predictions[run.seed], kind, run) for run in calibrated
        )
        methods.append(Method(LOSS_HEAD_METHOD, loss_runs))

    return methods
