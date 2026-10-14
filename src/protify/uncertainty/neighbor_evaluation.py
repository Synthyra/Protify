"""How well calibration-set neighbors say when a head's answer can be trusted, per the experiment's rules.

    python -m protify.uncertainty neighbors --runs <fit dir> --pattern "*/*/final/report.json" --output <dir>

Each finished cell must have run with `save_trunk_features`, so its npz holds every V_fit, V_cal,
and test row's trunk features. For each cell's first seed:
1. The head's logits are temperature-scaled on V_fit, as the adopted production method does.
2. Every row gets its correctness and the model's claim of it (`calibration_neighbors`).
3. Test and V_cal trunk features are L2-normalized, and every test row's cosine similarity to
   every V_cal row, and every V_cal row's to every other, is taken.
4. Each estimator of P(correct) is fitted on V_cal alone and scored on test: how well it ranks
   right answers above wrong ones (AUROC, AURC), whether it is calibrated (log loss, Brier,
   ECE), and whether an answer rule chosen on V_cal holds its error target on test.
5. Per identity bin to training, its mean estimate against the actual accuracy, and the evidence
   n_eff behind the residual estimator.

Regression also gets local-scale split conformal intervals against the absolute-residual ones
production ships. Differences from `model` are bootstrapped over test rows, paired, and pooled
within each family as the mean over datasets.

Shapes: `n` test rows, `r` V_cal rows, `B` bootstrap replicates.
"""

from __future__ import annotations

import csv
import math
import numpy as np

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from scipy.stats import rankdata, spearmanr

from foundry.eval.temperature import fit_temperature
from foundry.metrics import area_under_risk_coverage, expected_calibration_error
from .calibration_neighbors import (
    ESTIMATORS,
    AnswerRule,
    CalibrationRows,
    answer_threshold,
    claimed_correctness,
    correctness,
    cosine_similarities,
    estimate,
    evidence,
    local_scale,
    log_likelihood,
    normalized_interval_half_widths,
    unit_rows,
)
from .compare import read_identity
from .exchangeable import FAMILIAR_FROM, NOVEL_BELOW
from .identity_bins import IDENTITY_BIN_NAMES, bin_rows
from .metrics import CLASSIFICATION_KINDS, ECE_BINS, tie_averaged_losses
from .predictions import load_seed_predictions
from .report import read_json, write_json
from .verdicts import interval, json_ready


DEFAULT_PATTERN = "*/*/final/report.json"
REPLICATES = 2000
PRIMARY = "residual"
# The answer rule's error targets, as fractions of V_cal's error.
TARGET_FRACTIONS = (0.5, 0.25)
LEVELS = (0.8, 0.9, 0.95)
# A bin's accuracy and mean estimate are reported from this many test rows.
MINIMUM_BIN_ROWS = 30
# The pre-registered tolerances (the experiment README's decision rules).
INTERVAL_TOLERANCE = 0.01
RULE_TOLERANCE = 0.02
RULE_LEVEL = 0.9
RULE_DATASETS = {"classification": 5, "regression": 2}
BOOTSTRAP_METRICS = ("aurc", "log_loss", "error_auroc", "ece")
# One color per estimator in every panel, since regression has no `joint`.
COLORS = {"model": "tab:blue", "knn": "tab:orange", "joint": "tab:green", "residual": "tab:red"}


@dataclass
class CellScores:
    """One cell's per-row outcomes, which the pooled bootstrap resamples, and its summary."""

    data_name: str
    kind: str
    correct: np.ndarray  # (n,)
    q: dict[str, np.ndarray]  # estimator -> (n,)
    absolute_error: np.ndarray | None = None  # (n,)
    sigma: np.ndarray | None = None  # (n,)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def family(self) -> str:
        return "regression" if self.kind == "regression" else "classification"


def auroc(correct: np.ndarray, q: np.ndarray) -> float:
    """P(q of a right answer > q of a wrong one), ties counted half; NaN with one outcome only."""
    # correct: (n,), q: (n,)
    positives = int(correct.sum())
    negatives = correct.shape[0] - positives
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = rankdata(q)  # (n,)
    return float((ranks[correct > 0.5].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def aurc(correct: np.ndarray, q: np.ndarray) -> float:
    """Area under the risk-coverage curve of 0-1 error, rows answered in falling q, ties averaged."""
    # correct: (n,), q: (n,)
    uncertainty = -q  # (n,)
    return area_under_risk_coverage(tie_averaged_losses(1.0 - correct, uncertainty), uncertainty)


def correctness_metrics(correct: np.ndarray, q: np.ndarray) -> dict[str, float]:
    # correct: (n,), q: (n,)
    return {
        "error_auroc": auroc(correct, q),
        "aurc": aurc(correct, q),
        "log_loss": -log_likelihood(q, correct),
        "brier": float(np.mean((q - correct) ** 2)),
        "ece": expected_calibration_error(correct, q, ECE_BINS),
        "mean_q": float(q.mean()),
        "accuracy": float(correct.mean()),
    }


def coverage_at_error(correct: np.ndarray, q: np.ndarray, target: float) -> float:
    """The largest share of test rows, taken in falling q with ties averaged, whose error stays within the target."""
    # correct: (n,), q: (n,)
    order = np.argsort(-q, kind="stable")  # (n,)
    losses = tie_averaged_losses(1.0 - correct, -q)[order]  # (n,)
    risk = np.cumsum(losses) / np.arange(1, losses.shape[0] + 1)  # (n,)
    within = np.flatnonzero(risk <= target + 1e-12)
    return 0.0 if within.size == 0 else float((within[-1] + 1) / losses.shape[0])


def rule_outcome(rule: AnswerRule, correct: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    # correct: (n,), q: (n,)
    answered = q >= rule.threshold  # (n,)
    count = int(answered.sum())
    error = None if count == 0 else float(1.0 - correct[answered].mean())
    return {
        "target_error": rule.target_error,
        "threshold": rule.threshold,
        "calibration_coverage": rule.calibration_coverage,
        "calibration_error": rule.calibration_error,
        "calibration_error_bound": rule.error_bound,
        "test_coverage": count / correct.shape[0],
        "test_error": error,
        # A rule that answers nothing holds its target vacuously, and counts as not holding.
        "held": error is not None and error <= rule.target_error + RULE_TOLERANCE,
    }


def _identity_split(report: Mapping[str, Any], rows: int, runs_dir: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Test and V_cal rows' identity to training, from the file the report names."""
    identity, check = read_identity(report, rows, runs_dir)
    if identity is None:
        return None, None  # no identity: None, not (n,) or (r,) arrays
    v_cal = read_json(check["read"])["rows"].get("v_cal")
    return identity, None if v_cal is None else np.asarray(v_cal, dtype=np.float64)  # (n,), (r,) or None


def _bin_summary(
    identity: np.ndarray | None,
    calibration_identity: np.ndarray | None,
    correct: np.ndarray,
    q: Mapping[str, np.ndarray],
    effective: np.ndarray,
    nearest: np.ndarray,
) -> dict[str, Any]:
    # identity: (n,), calibration_identity: (r,), correct: (n,), q: (n,) per estimator, effective: (n,), nearest: (n,)
    if identity is None:
        return {}
    calibration_bins = None if calibration_identity is None else bin_rows(calibration_identity)
    summary = {}
    for name, rows in bin_rows(identity).items():
        entry: dict[str, Any] = {"rows": int(rows.shape[0])}
        if calibration_bins is not None:
            entry["v_cal_rows"] = int(calibration_bins[name].shape[0])
        if rows.shape[0] >= MINIMUM_BIN_ROWS:
            accuracy = float(correct[rows].mean())
            entry.update({
                "accuracy": accuracy,
                "mean_q": {estimator: float(values[rows].mean()) for estimator, values in q.items()},
                "gap": {estimator: float(values[rows].mean()) - accuracy for estimator, values in q.items()},
                "median_n_eff": float(np.median(effective[rows])),
                "median_nearest_similarity": float(np.median(nearest[rows])),
            })
        summary[name] = entry
    return summary


def _novelty_auroc(identity: np.ndarray | None, total: np.ndarray) -> float | None:
    """How well low evidence W separates test rows below NOVEL_BELOW identity from those at FAMILIAR_FROM or above."""
    # identity: (n,), total: (n,)
    if identity is None:
        return None
    novel, familiar = identity < NOVEL_BELOW, identity >= FAMILIAR_FROM  # (n,), (n,)
    if not novel.any() or not familiar.any():
        return None
    keep = novel | familiar  # (n,)
    return auroc(novel[keep].astype(np.float64), -total[keep])


def _intervals(
    residuals: np.ndarray,
    test_error: np.ndarray,
    scale_calibration: np.ndarray,
    scale_test: np.ndarray,
    identity: np.ndarray | None,
) -> dict[str, Any]:
    """Absolute-residual and local-scale split conformal intervals at each level, overall and per identity bin."""
    # residuals: (r,), test_error: (n,), scale_calibration: (r,), scale_test: (n,), identity: (n,)
    bins = {} if identity is None else bin_rows(identity)
    summary: dict[str, Any] = {}
    for level in LEVELS:
        half_widths = {
            "absolute": normalized_interval_half_widths(residuals, np.ones_like(residuals), np.ones_like(test_error), level),
            "local": normalized_interval_half_widths(residuals, scale_calibration, scale_test, level),
        }
        entry = {}
        for name, half_width in half_widths.items():
            covered = test_error <= half_width  # (n,)
            entry[name] = {
                "coverage": float(covered.mean()),
                "mean_width": float(2.0 * half_width.mean()),
                "bins": {
                    bin_name: {"rows": int(rows.shape[0]), "coverage": float(covered[rows].mean()), "mean_width": float(2.0 * half_width[rows].mean())}
                    for bin_name, rows in bins.items()
                    if rows.shape[0] >= MINIMUM_BIN_ROWS
                },
            }
        summary[f"{level:g}"] = entry
    return summary


def score_cell(report_path: Path, runs_dir: str) -> CellScores:
    """Every estimator on one finished cell's first seed."""
    directory = report_path.parent
    report = read_json(str(report_path))
    kind, data_name, seed = report["task_type"], report["data_name"], report["seeds"][0]
    splits = load_seed_predictions(str(directory / f"seed_{seed}.npz"))
    missing = [split for split in ("v_fit", "v_cal", "test") if splits[split].trunk is None]
    if missing:
        raise ValueError(f"{directory} saved no trunk features for {missing}; run the cell with uncertainty.save_trunk_features")
    fit, calibration, test = splits["v_fit"], splits["v_cal"], splits["test"]

    summary: dict[str, Any] = {"data_name": data_name, "kind": kind, "seed": seed, "cell": str(directory)}
    if kind == "regression":
        residuals = (calibration.labels - calibration.outputs).astype(np.float64)  # (r,)
        tolerance = float(np.median(np.abs(residuals)))
        calibration_correct = correctness(kind, calibration.labels, calibration.outputs, tolerance)  # (r,)
        test_correct = correctness(kind, test.labels, test.outputs, tolerance)  # (n,)
        share = float(calibration_correct.mean())
        calibration_claimed = np.full(calibration_correct.shape[0], share)  # (r,)
        test_claimed = np.full(test_correct.shape[0], share)  # (n,)
        summary["tolerance"] = tolerance
    else:
        residuals = None
        temperature = fit_temperature(fit.outputs, fit.labels, multiclass=kind in CLASSIFICATION_KINDS)
        calibration_logits, test_logits = calibration.outputs / temperature, test.outputs / temperature  # (r, c), (n, c)
        calibration_correct = correctness(kind, calibration.labels, calibration_logits)  # (r,)
        test_correct = correctness(kind, test.labels, test_logits)  # (n,)
        calibration_claimed = claimed_correctness(kind, calibration_logits)  # (r,)
        test_claimed = claimed_correctness(kind, test_logits)  # (n,)
        summary["temperature"] = float(temperature)

    calibration_unit, test_unit = unit_rows(calibration.trunk), unit_rows(test.trunk)  # (r, p), (n, p)
    calibration_similarities = cosine_similarities(calibration_unit, calibration_unit)  # (r, r)
    test_similarities = cosine_similarities(test_unit, calibration_unit)  # (n, r)
    rows = CalibrationRows(calibration_unit, calibration_correct, calibration_claimed, residuals)
    calibration_error = float(1.0 - calibration_correct.mean())
    summary.update({
        "test_rows": int(test_correct.shape[0]),
        "v_cal_rows": int(calibration_correct.shape[0]),
        "trunk_width": int(test_unit.shape[1]),
        "v_cal_error": calibration_error,
        "test_error": float(1.0 - test_correct.mean()),
    })

    # `joint` weighs confidence alongside similarity, and a regression claim is one constant.
    names = [name for name in ESTIMATORS if not (kind == "regression" and name == "joint")]
    estimates = {name: estimate(name, rows, calibration_similarities, test_similarities, test_claimed) for name in names}
    q = {name: fitted.q for name, fitted in estimates.items()}
    summary["estimators"] = {}
    for name, fitted in estimates.items():
        rules = [
            rule_outcome(answer_threshold(fitted.calibration_q, calibration_correct, fraction * calibration_error), test_correct, fitted.q)
            for fraction in TARGET_FRACTIONS
        ]
        summary["estimators"][name] = {
            "settings": dict(fitted.settings),
            "loo_log_likelihood": fitted.loo_log_likelihood,
            "test": correctness_metrics(test_correct, fitted.q),
            "coverage_at_error": {f"{fraction:g}": coverage_at_error(test_correct, fitted.q, fraction * calibration_error) for fraction in TARGET_FRACTIONS},
            "rules": rules,
        }

    bandwidth = estimates[PRIMARY].settings["h"]
    total, effective = evidence(test_similarities, bandwidth)  # (n,), (n,)
    nearest = test_similarities.max(axis=1)  # (n,)
    identity, calibration_identity = _identity_split(report, test_correct.shape[0], runs_dir)
    summary["evidence"] = {
        "bandwidth": bandwidth,
        "median_n_eff": float(np.median(effective)),
        "novelty_auroc": _novelty_auroc(identity, total),
        "median_nearest_similarity": float(np.median(nearest)),
    }
    summary["identity_bins"] = _bin_summary(identity, calibration_identity, test_correct, q, effective, nearest)

    scores = CellScores(data_name, kind, test_correct, q, summary=summary)
    if residuals is not None:
        scale = local_scale(residuals, calibration_similarities, test_similarities)
        scores.absolute_error = np.abs(test.labels - test.outputs).astype(np.float64)  # (n,)
        scores.sigma = scale.sigma
        summary["local_scale"] = {
            "settings": dict(scale.settings),
            "loo_gaussian_nll": scale.loo_gaussian_nll,
            "sigma_error_spearman": float(spearmanr(scale.sigma, scores.absolute_error).statistic),
        }
        summary["intervals"] = _intervals(residuals, scores.absolute_error, scale.calibration_sigma, scale.sigma, identity)
    return scores


def _metric(name: str, correct: np.ndarray, q: np.ndarray) -> float:
    # correct: (n,), q: (n,)
    if name == "aurc":
        return aurc(correct, q)
    if name == "log_loss":
        return -log_likelihood(q, correct)
    if name == "error_auroc":
        return auroc(correct, q)
    if name == "ece":
        return expected_calibration_error(correct, q, ECE_BINS)
    raise ValueError(name)


def _interval(values: np.ndarray) -> dict[str, float]:
    # values: (B,)
    lower, upper = interval(values)
    return {"lower": lower, "upper": upper}


def _replicate_draws(cell: CellScores, rows: np.ndarray, names: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    """Every estimator's every bootstrap metric on each replicate's resampled test rows, `(B,)` each."""
    # rows: (B, n)
    values = {name: {metric: np.empty(rows.shape[0]) for metric in BOOTSTRAP_METRICS} for name in names}
    for replicate, indices in enumerate(rows):
        correct = cell.correct[indices]  # (n,)
        for name in names:
            q = cell.q[name][indices]  # (n,)
            for metric in BOOTSTRAP_METRICS:
                values[name][metric][replicate] = _metric(metric, correct, q)
    return values  # (B,) per estimator and metric


def pooled_differences(cells: Sequence[CellScores], replicates: int, seed: int) -> dict[str, Any]:
    """Each estimator minus `model` on each metric: per dataset, and pooled within each family.

    Each replicate resamples a dataset's test rows with replacement, the same rows for every
    estimator, and the pooled value is the mean over the family's datasets. Each dataset draws
    from its own stream of `seed`, so its draws do not depend on which datasets run beside it.
    """
    pooled: dict[str, Any] = {}
    for family in ("classification", "regression"):
        members = [(index, cell) for index, cell in enumerate(cells) if cell.family == family]
        if not members:
            continue
        estimators = [name for name in members[0][1].q if name != "model" and all(name in cell.q for _, cell in members)]
        points: dict[str, dict[str, list[float]]] = {name: {metric: [] for metric in BOOTSTRAP_METRICS} for name in estimators}
        draws: dict[str, dict[str, list[np.ndarray]]] = {name: {metric: [] for metric in BOOTSTRAP_METRICS} for name in estimators}
        spearman_points, spearman_draws = [], []
        for index, cell in members:
            rng = np.random.default_rng([seed, index])
            rows = rng.integers(0, cell.correct.shape[0], size=(replicates, cell.correct.shape[0]), dtype=np.int32)  # (B, n)
            values = _replicate_draws(cell, rows, ["model", *estimators])
            for name in estimators:
                for metric in BOOTSTRAP_METRICS:
                    points[name][metric].append(_metric(metric, cell.correct, cell.q[name]) - _metric(metric, cell.correct, cell.q["model"]))
                    draws[name][metric].append(values[name][metric] - values["model"][metric])
            if family == "regression":
                spearman_points.append(spearmanr(cell.sigma, cell.absolute_error).statistic)
                spearman_draws.append(np.array([spearmanr(cell.sigma[indices], cell.absolute_error[indices]).statistic for indices in rows]))
            del rows

        family_result: dict[str, Any] = {}
        for name in estimators:
            family_result[name] = {}
            for metric in BOOTSTRAP_METRICS:
                point, replicate_values = np.array(points[name][metric]), np.stack(draws[name][metric])  # (datasets,), (datasets, B)
                family_result[name][metric] = {
                    "pooled": {"mean": float(np.nanmean(point)), **_interval(np.nanmean(replicate_values, axis=0))},
                    "datasets": {
                        cell.data_name: {"mean": float(value), **_interval(values)}
                        for (_, cell), value, values in zip(members, point, replicate_values, strict=True)
                    },
                }
        if family == "regression":
            family_result["sigma_error_spearman"] = {
                "mean": float(np.mean(spearman_points)),
                **_interval(np.nanmean(np.stack(spearman_draws), axis=0)),
            }
        pooled[family] = family_result
    return pooled


def verdicts(cells: Sequence[CellScores], differences: Mapping[str, Any]) -> dict[str, Any]:
    """The experiment README's rules R1 to R5, each with what it read."""
    outcomes: dict[str, Any] = {}
    classification = differences.get("classification", {}).get(PRIMARY)
    if classification is not None:
        r1 = classification["aurc"]["pooled"]
        r2 = classification["log_loss"]["pooled"]
        outcomes["R1_aurc"] = {**r1, "passed": r1["upper"] < 0.0}
        outcomes["R2_log_loss"] = {**r2, "passed": r2["upper"] < 0.0}
        outcomes["adopt_classification"] = outcomes["R1_aurc"]["passed"] and outcomes["R2_log_loss"]["passed"]

    regression = differences.get("regression", {})
    if PRIMARY in regression:
        r3 = regression[PRIMARY]["error_auroc"]["pooled"]
        outcomes["R3_error_auroc"] = {**r3, "passed": r3["lower"] > 0.0}
        level = f"{RULE_LEVEL:g}"
        coverage = {
            cell.data_name: {
                "local": cell.summary["intervals"][level]["local"]["coverage"],
                "absolute": cell.summary["intervals"][level]["absolute"]["coverage"],
            }
            for cell in cells
            if cell.family == "regression"
        }
        spearman = regression["sigma_error_spearman"]
        held = all(entry["local"] >= entry["absolute"] - INTERVAL_TOLERANCE for entry in coverage.values())
        outcomes["R4_intervals"] = {"coverage": coverage, "sigma_error_spearman": spearman, "passed": held and spearman["lower"] > 0.0}
        outcomes["adopt_regression"] = outcomes["R3_error_auroc"]["passed"] and outcomes["R4_intervals"]["passed"]

    for family, needed in RULE_DATASETS.items():
        held = {
            cell.data_name: cell.summary["estimators"][PRIMARY]["rules"][0]["held"]
            for cell in cells
            if cell.family == family
        }
        if held:
            outcomes[f"R5_answer_rule_{family}"] = {"held": held, "needed": needed, "passed": sum(held.values()) >= needed}
    return outcomes


def _table_rows(cells: Sequence[CellScores]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    estimators, rules, bins = [], [], []
    for cell in cells:
        for name, entry in cell.summary["estimators"].items():
            estimators.append({
                "data_name": cell.data_name, "kind": cell.kind, "estimator": name,
                "settings": ";".join(f"{key}={value:g}" for key, value in entry["settings"].items()),
                "loo_log_likelihood": entry["loo_log_likelihood"], **entry["test"],
                **{f"coverage_at_{key}_error": value for key, value in entry["coverage_at_error"].items()},
            })
            for fraction, outcome in zip(TARGET_FRACTIONS, entry["rules"], strict=True):
                rules.append({"data_name": cell.data_name, "estimator": name, "target_fraction": fraction, **outcome})
        for bin_name, entry in cell.summary["identity_bins"].items():
            row = {"data_name": cell.data_name, "bin": bin_name, "rows": entry["rows"], "v_cal_rows": entry.get("v_cal_rows"), "accuracy": entry.get("accuracy")}
            for name, value in (entry.get("mean_q") or {}).items():
                row[f"mean_q_{name}"] = value
            row["median_n_eff"] = entry.get("median_n_eff")
            row["median_nearest_similarity"] = entry.get("median_nearest_similarity")
            bins.append(row)
    return estimators, rules, bins


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "n/a" if not math.isfinite(value) else f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, cells: Sequence[CellScores], differences: Mapping[str, Any], rules: Mapping[str, Any]) -> None:
    lines = ["# Calibration-neighbor uncertainty", "", "## Rules", ""]
    for name, entry in rules.items():
        if isinstance(entry, Mapping):
            shown = {key: value for key, value in entry.items() if key in ("mean", "lower", "upper", "passed", "held")}
            lines.append(f"- **{name}**: {shown}")
        else:
            lines.append(f"- **{name}**: {entry}")
    lines += ["", "## Pooled differences from `model` (95 percent bootstrap intervals)", ""]
    lines += ["| family | estimator | metric | mean | lower | upper |", "|---|---|---|---:|---:|---:|"]
    for family, entries in differences.items():
        for name, metrics in entries.items():
            if name == "sigma_error_spearman":
                lines.append(f"| {family} | local scale | Spearman(sigma, abs error) | {_format(metrics['mean'])} | {_format(metrics['lower'])} | {_format(metrics['upper'])} |")
                continue
            for metric, entry in metrics.items():
                pooled = entry["pooled"]
                lines.append(f"| {family} | {name} | {metric} | {_format(pooled['mean'])} | {_format(pooled['lower'])} | {_format(pooled['upper'])} |")
    lines += ["", "## Per dataset, on test", ""]
    lines += ["| dataset | estimator | AUROC | AURC | log loss | ECE | rule coverage | rule error | target |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in cells:
        for name, entry in cell.summary["estimators"].items():
            test, rule = entry["test"], entry["rules"][0]
            lines.append(
                f"| {cell.data_name} | {name} | {_format(test['error_auroc'])} | {_format(test['aurc'])} | {_format(test['log_loss'])} | "
                f"{_format(test['ece'])} | {_format(rule['test_coverage'])} | {_format(rule['test_error'])} | {_format(rule['target_error'])} |"
            )
    lines += ["", "## Mean estimate minus accuracy, per identity bin", ""]
    lines += ["| dataset | bin | rows | accuracy | " + " | ".join(ESTIMATORS) + " | n_eff |", "|---|---|---:|---:|" + "---:|" * (len(ESTIMATORS) + 1)]
    for cell in cells:
        for bin_name in IDENTITY_BIN_NAMES:
            entry = cell.summary["identity_bins"].get(bin_name)
            if not entry or "accuracy" not in entry:
                continue
            gaps = " | ".join(_format(entry["gap"].get(name)) for name in ESTIMATORS)
            lines.append(f"| {cell.data_name} | {bin_name} | {entry['rows']} | {_format(entry['accuracy'])} | {gaps} | {_format(entry['median_n_eff'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def plot_risk_coverage(path: Path, cells: Sequence[CellScores]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 3
    rows = math.ceil(len(cells) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.2 * rows), squeeze=False)
    for axis, cell in zip(axes.flat, cells, strict=False):
        for name, q in cell.q.items():
            order = np.argsort(-q, kind="stable")  # (n,)
            losses = tie_averaged_losses(1.0 - cell.correct, -q)[order]  # (n,)
            coverage = np.arange(1, losses.shape[0] + 1) / losses.shape[0]  # (n,)
            axis.plot(coverage, np.cumsum(losses) / np.arange(1, losses.shape[0] + 1), label=name, color=COLORS[name], linewidth=1.2)
        axis.set_title(cell.data_name, fontsize=9)
        axis.set_xlabel("coverage", fontsize=8)
        axis.set_ylabel("error of answered rows", fontsize=8)
        axis.tick_params(labelsize=7)
    for axis in list(axes.flat)[len(cells):]:
        axis.axis("off")
    axes.flat[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def plot_identity_gap(path: Path, cells: Sequence[CellScores]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 3
    rows = math.ceil(len(cells) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.2 * rows), squeeze=False)
    for axis, cell in zip(axes.flat, cells, strict=False):
        bins = [name for name in IDENTITY_BIN_NAMES if "accuracy" in cell.summary["identity_bins"].get(name, {})]
        positions = np.arange(len(bins))
        axis.plot(positions, [cell.summary["identity_bins"][name]["accuracy"] for name in bins], "k-o", label="accuracy", markersize=3)
        for estimator in cell.q:
            axis.plot(positions, [cell.summary["identity_bins"][name]["mean_q"][estimator] for name in bins], "--", label=estimator, color=COLORS[estimator], linewidth=1.0)
        axis.set_xticks(positions, bins, rotation=45, fontsize=6)
        axis.set_title(cell.data_name, fontsize=9)
        axis.set_ylabel("share correct", fontsize=8)
        axis.tick_params(axis="y", labelsize=7)
    for axis in list(axes.flat)[len(cells):]:
        axis.axis("off")
    axes.flat[0].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def evaluate_neighbors(runs_dir: str, output_dir: str, *, pattern: str = DEFAULT_PATTERN, replicates: int = REPLICATES, seed: int = 0) -> dict[str, Any]:
    """Score every matching finished cell, pool, apply the rules, and write every output."""
    paths = sorted(Path(runs_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no report.json matches {pattern!r} under {runs_dir}")
    cells = []
    for path in paths:
        print(f"scoring {path.parent}", flush=True)
        cells.append(score_cell(path, runs_dir))
    differences = pooled_differences(cells, replicates, seed)
    rules = verdicts(cells, differences)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {"replicates": replicates, "seed": seed, "rules": rules, "differences": differences, "cells": [cell.summary for cell in cells]}
    write_json(str(output / "neighbors.json"), json_ready(payload))
    estimator_rows, rule_rows, bin_rows_ = _table_rows(cells)
    _write_csv(output / "estimators.csv", estimator_rows)
    _write_csv(output / "answer_rules.csv", rule_rows)
    _write_csv(output / "identity_bins.csv", bin_rows_)
    write_markdown(output / "neighbors.md", cells, differences, rules)
    plot_risk_coverage(output / "risk_coverage.png", cells)
    plot_identity_gap(output / "identity_gap.png", cells)
    return payload
