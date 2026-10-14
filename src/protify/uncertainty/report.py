"""Scoring every method on test, overall and per identity bin, into `report.json`.

A per-seed method is scored once per seed and reported as the mean and standard deviation over
seeds; an ensemble is scored once. Conformal sets and intervals are calibrated on V_cal and
evaluated on test, and task, calibration, interval, and selective-prediction scores are
computed on test rows only.

Shapes: `n` rows, `c` classes or labels.
"""

from __future__ import annotations

import json
import os
import numpy as np

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .conformal_sets import classification_conformal_rows, regression_conformal_rows
from .methods import Method, MethodRun, Predictive
from .metrics import (
    CLASSIFICATION_KINDS,
    brier_score,
    calibration_error,
    gaussian_interval,
    gaussian_nll,
    negative_log_likelihood,
    row_losses,
    selective_metrics,
    task_metrics,
)
from .prediction_arrays import prediction_row_blocks, select_prediction_rows
from ..metrics import validate_observed_labels


Scores = dict[str, float | None]


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def conformal_rows(
    run: MethodRun, kind: str, labels: Mapping[str, np.ndarray], levels: tuple[float, ...], train_label_std: float,
    label_masks: Mapping[str, np.ndarray | None] | None = None,
) -> dict[str, np.ndarray] | None:
    """Per test row coverage and size or width, calibrated on this run's V_cal predictions."""
    # labels: (n,) or (n, c) per split name, label_masks: (n,) or (n, c) per split name or None
    calibration, test = run.splits["v_cal"], run.splits["test"]
    mask = None if label_masks is None else label_masks.get("v_cal")
    if mask is not None and kind != "multilabel":
        validate_observed_labels(labels["v_cal"], mask)
        if not mask.any():
            raise ValueError("conformal calibration requires observed calibration targets")
    observed = slice(None) if mask is None or kind == "multilabel" else mask.reshape(-1)
    calibration_labels = labels["v_cal"][observed]  # (m,) or (n, c)
    test_mask = None if label_masks is None else label_masks.get("test")
    test_labels = labels["test"]
    if test_mask is not None and kind != "multilabel":
        validate_observed_labels(test_labels, test_mask)
        # Unknown targets never reach class indexing or residual arithmetic. Coverage
        # for these rows is undefined; prediction-set size still needs no truth.
        test_labels = np.where(test_mask, test_labels, 0)
    if kind != "multilabel":
        calibration_labels, test_labels = calibration_labels.reshape(-1), test_labels.reshape(-1)
    if kind in CLASSIFICATION_KINDS:
        rows = classification_conformal_rows(calibration.probs[observed], calibration_labels, test.probs, test_labels, levels)
    elif kind == "regression":
        rows = regression_conformal_rows(
            calibration.mean[observed],
            calibration_labels,
            None if calibration.variance is None else np.sqrt(calibration.variance[observed]),
            test.mean,
            test_labels,
            None if test.variance is None else np.sqrt(test.variance),
            levels,
            train_label_std,
        )
    else:
        return None
    if test_mask is not None:
        for name, values in rows.items():
            if "coverage@" in name:
                values[~test_mask.reshape(-1)] = np.nan
    return rows  # each (m,) by metric name


def label_support(kind: str, labels: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Describe measured support without counting an unknown entry as a negative."""
    # labels: (n,) or (n, c), mask: (n,) or (n, c)
    observed = 0
    class_counts: Counter[int] = Counter()
    if kind == "multilabel":
        positives = np.zeros(labels.shape[1], dtype=np.int64)  # (c,)
        negatives = np.zeros_like(positives)  # (c,)
        known = np.zeros_like(positives)  # (c,)
    for truth, flags in prediction_row_blocks(labels, mask):  # (b, c) or scalar rows
        validate_observed_labels(truth, flags)
        observed += int(flags.sum())
        if kind == "multilabel":
            positives += ((truth == 1) & flags).sum(axis=0)
            negatives += ((truth == 0) & flags).sum(axis=0)
            known += flags.sum(axis=0)
        elif kind in CLASSIFICATION_KINDS:
            classes, counts = np.unique(truth[flags], return_counts=True)  # (k,), (k,)
            for label, count in zip(classes, counts, strict=True):
                class_counts[int(label)] += int(count)
    support: dict[str, Any] = {"scope": "observed_targets", "observed": observed, "unknown": int(mask.size - observed)}
    if kind == "multilabel":
        support["labels"] = [
            {
                "index": index, "positive": int(positive), "negative": int(negative),
                "unknown": int(len(labels) - known[index]),
                "discrimination_status": "supported" if positive and negative else "insufficient_observed_classes",
            }
            for index, (positive, negative) in enumerate(zip(positives, negatives, strict=True))
        ]
    elif kind in CLASSIFICATION_KINDS:
        support["class_counts"] = {str(label): count for label, count in sorted(class_counts.items())}
        support["discrimination_status"] = "supported" if len(class_counts) > 1 else "insufficient_observed_classes"
    return support


def score_rows(
    kind: str,
    predictive: Predictive,
    labels: np.ndarray,
    levels: tuple[float, ...],
    rows: np.ndarray | None = None,
    conformal: Mapping[str, np.ndarray] | None = None,
    score_only: bool = False,
    label_mask: np.ndarray | None = None,
    canonical: bool = False,
) -> Scores:
    """Every score a method has on the given rows, all rows when `rows` is None."""
    # labels: (n,) or (n, c), rows: (m,) or None, conformal: (m,) per metric name or None, label_mask: (n,) or (n, c) or None
    select = slice(None) if rows is None else rows
    selected = np.arange(labels.shape[0])[select]  # (n_selected,), independent of label width
    scores: Scores = {"count": float(len(selected))}
    if label_mask is not None:
        for truth_block, mask_block in prediction_row_blocks(labels, label_mask):
            validate_observed_labels(truth_block, mask_block)
        mask = select_prediction_rows(label_mask, select)  # (n_selected, c) or scalar rows
        observed_rows = np.empty(len(selected), dtype=bool)  # (n_selected,)
        observed, cursor = 0, 0
        for (flags,) in prediction_row_blocks(mask):
            observed_rows[cursor:cursor + len(flags)] = flags.any(axis=1) if kind == "multilabel" else flags.reshape(-1)
            observed += int(flags.sum())
            cursor += len(flags)
        scores.update(observed_labels=float(observed), unobserved_labels=float(mask.size - observed), observed_rows=float(observed_rows.sum()))
        if not observed_rows.all():
            select = selected[observed_rows]  # (m,)
        label_mask = select_prediction_rows(label_mask, select)  # (m, c) or scalar rows
    truth = select_prediction_rows(labels, select)  # (m, c) or scalar rows
    if truth.shape[0] == 0:
        return scores
    if kind != "multilabel":
        truth = truth.reshape(-1)  # (m,), including scalar targets stored as (m, 1)
        label_mask = None if label_mask is None else label_mask.reshape(-1)

    probs = None if predictive.probs is None else select_prediction_rows(predictive.probs, select)  # (m, c)
    mean = None if predictive.mean is None else select_prediction_rows(predictive.mean, select)  # (m,)
    variance = None if predictive.variance is None else select_prediction_rows(predictive.variance, select)  # (m,)
    if not score_only:
        scores.update(task_metrics(kind, truth, probs=probs, mean=mean, label_mask=label_mask, canonical=canonical))
        if kind != "regression":
            scores["ece"] = calibration_error(kind, truth, probs, label_mask=label_mask)
            scores["nll"] = negative_log_likelihood(kind, truth, probs, label_mask)
            scores["brier"] = brier_score(kind, truth, probs, label_mask)
        elif variance is not None:
            scores["gaussian_nll"] = gaussian_nll(truth, mean, variance)
            for level in levels:
                covered, width = gaussian_interval(truth, mean, variance, level)  # (n,), (n,)
                scores[f"interval_coverage@{level}"] = float(covered.mean())
                scores[f"interval_width@{level}"] = float(width.mean())
        for name, values in (conformal or {}).items():
            value = float(values[select].mean())
            scores[name] = value if np.isfinite(value) else None

    if predictive.uncertainty is not None:
        losses = row_losses(kind, truth, probs=probs, mean=mean, label_mask=label_mask)  # (n,)
        scores.update(selective_metrics(kind, losses, predictive.uncertainty[select]))
    return scores


def summarize(runs: Sequence[Scores]) -> tuple[Scores, Scores]:
    """The mean and standard deviation of each score over runs, skipping undefined values."""
    names = list(dict.fromkeys(name for scores in runs for name in scores))
    means: Scores = {}
    deviations: Scores = {}
    for name in names:
        values = [scores[name] for scores in runs if scores.get(name) is not None]
        means[name] = _mean(values)
        deviations[name] = float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None)
    return means, deviations


def score_method(
    method: Method,
    kind: str,
    labels: Mapping[str, np.ndarray],
    levels: tuple[float, ...],
    train_label_std: float,
    test_bins: Mapping[str, np.ndarray] | None,
    label_masks: Mapping[str, np.ndarray | None] | None = None,
    canonical: bool = False,
) -> dict[str, Any]:
    """One method's test scores, over seeds, with its per-bin scores when identity is known."""
    # labels: (n,) or (n, c) per split name, test_bins: (m,) per bin name or None, label_masks: (n,) or (n, c) per split name or None
    run_scores, run_bins = [], []
    for run in method.runs:
        conformal = None if method.score_only else conformal_rows(run, kind, labels, levels, train_label_std, label_masks)
        test = run.splits["test"]
        mask = None if label_masks is None else label_masks.get("test")
        run_scores.append(score_rows(kind, test, labels["test"], levels, None, conformal, method.score_only, mask, canonical))
        if test_bins is not None:
            run_bins.append({
                name: score_rows(kind, test, labels["test"], levels, rows, conformal, method.score_only, mask, canonical)
                for name, rows in test_bins.items()
            })

    means, deviations = summarize(run_scores)
    summary: dict[str, Any] = {
        "seeds": [run.seed for run in method.runs],
        "score_only": method.score_only,
        "test": means,
        "test_std": deviations,
        "per_run": [
            {"seed": run.seed, "fitted": dict(run.fitted), "test": scores}
            for run, scores in zip(method.runs, run_scores, strict=True)
        ],
    }
    if label_masks is not None and label_masks.get("test") is not None:
        summary["test_label_support"] = label_support(kind, labels["test"], label_masks["test"])
    if test_bins is not None:
        summary["identity_bins"] = {name: summarize([bins[name] for bins in run_bins])[0] for name in test_bins}
    return summary


def write_json(path: str, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically; a NaN anywhere is a bug upstream, so it raises."""
    temporary = f"{path}.partial"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
