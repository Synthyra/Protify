"""The analysis's files: the answers in JSON and Markdown, and a CSV per table.

`write_outputs` writes into one directory:
- `verdicts.json`: the parameters, every cell, both questions' numbers, intervals, and verdicts,
  scheme V's answer to question 2 beside scheme T's, the shift AUROCs, the conformal rows, and
  a summary of the checks. It is strict JSON, so an undefined number is null.
- `q1_featurization.md` and `q2_uncertainty.md`: per-dataset rows, pooled rows, and the verdicts
  in words.
- `q1_scores.csv`, `q1_pairs.csv`, `q1_sensitivity.csv`, `q2_scheme_t.csv`, `q2_pooled.csv`,
  `q2_shift.csv`, `q2_identity_coverage.csv`, `conformal_bands.csv`, and `checks.csv`.
- The tables `compare` wrote before the decision rules, in the same columns: `arms.csv`,
  `methods.csv`, `identity_bins.csv`, `paired_bootstrap.csv`, and `summary.md`.

Nothing written depends on timing or on the number of worker processes, so one seed writes the
same bytes. A CSV leaves an undefined number empty and writes a float at full precision.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import numpy as np

from collections.abc import Mapping, Sequence
from itertools import permutations
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .methods import HEAD_NAMES
from .report import write_json
from .verdicts import FALLBACK_ADOPTION, DecisionRules, conformal_band, interval, json_ready, seed_spread, verdict_in_words


if TYPE_CHECKING:
    from .compare import CellResult, CellRun


CELL_COLUMNS = ("data_name", "cell_id", "head", "display_name")
PAIRED_COLUMNS = ("data_name", "head", "cell_a", "cell_b", "primary_metric", "difference", "lower_95", "upper_95", "replicates")
CELL_KEY_COLUMNS = ("data_name", "arm", "head")
# A seed spread is the standard deviation over seeds with ddof = 1, and its column says so.
SCORE_COLUMNS = (
    *CELL_KEY_COLUMNS, "cell_id", "primary_metric", "test_primary", "test_primary_lower_95", "test_primary_upper_95",
    "test_primary_seed_std_ddof1", "v_fit_primary", "report_test_primary",
)
PAIR_COLUMNS = (
    "section", "comparison", "data_name", "first", "second", "difference", "lower_95", "upper_95", "seed_std_ddof1",
    "datasets", "positive_datasets", "verdict",
)
SENSITIVITY_COLUMNS = ("excluded_datasets", *PAIR_COLUMNS)
SCHEME_T_COLUMNS = (*CELL_KEY_COLUMNS, "method", "score_only", "metric", "mean", "lower_95", "upper_95", "defined", "repeats")
POOLED_COLUMNS = ("scheme", "arm", "method", "role", "data_name", "quantity", "mean", "lower_95", "upper_95", "defined", "improves", "failed")
SHIFT_COLUMNS = ("level", "data_name", "arm", "head", "method", "mean", "lower_95", "upper_95", "defined", "datasets", "excluded_datasets")
IDENTITY_COLUMNS = (*CELL_KEY_COLUMNS, "method", "set", "level", "bin", "coverage", "coverage_lower_95", "coverage_upper_95", "defined", "size_or_width")
CONFORMAL_COLUMNS = (
    "scheme", *CELL_KEY_COLUMNS, "method", "set", "level", "coverage", "coverage_lower_95", "coverage_upper_95", "size_or_width",
    "calibration_rows", "rank", "marginal", "marginal_lower", "marginal_upper", "beta_lower", "beta_upper",
)
CHECK_COLUMNS = (*CELL_KEY_COLUMNS, "cell_id", "check", "method", "seed", "passed", "detail")
QUANTITIES = ("relative_nll", "relative_aurc", "primary_difference")
QUANTITY_LABELS = {"relative_nll": "relative NLL", "relative_aurc": "relative AURC", "primary_difference": "primary - plain"}
# A conformal metric: Mondrian or marginal, the set score, what it measures, and the level.
CONFORMAL_METRIC = re.compile(r"^(mondrian_)?(lac|aps|conformal)_(coverage|size|width)@([0-9.eE+-]+)$")


def _csv_value(value: Any) -> Any:
    """A CSV cell: empty for a missing or undefined number, a float at full precision."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value)) if math.isfinite(value) else ""
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    return value


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text with LF line ends, atomically."""
    temporary = path.with_name(f"{path.name}.partial")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows({name: _csv_value(row.get(name)) for name in columns} for row in rows)
    _write_text(path, buffer.getvalue())


def _number(value: float | None) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.4f}"


def _signed(value: float | None) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:+.4f}"


def _span(lower: float | None, upper: float | None, signed: bool = True) -> str:
    if lower is None or upper is None or not (math.isfinite(lower) and math.isfinite(upper)):
        return "n/a"
    return f"[{lower:+.4f}, {upper:+.4f}]" if signed else f"[{lower:.4f}, {upper:.4f}]"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *("| " + " | ".join(row) + " |" for row in rows)]


def _legacy_format(value: Any) -> str:
    """summary.md's number format, as compare wrote it before the decision rules."""
    return "" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def _cell_key(cell: CellRun) -> dict[str, str]:
    data_name, arm, head = cell.key
    return {"data_name": data_name, "arm": arm, "head": head}


def report_tables(cells: Sequence[CellRun]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """The arm, method, and identity-bin rows the reports hold, as `arms.csv`, `methods.csv`, and `identity_bins.csv`."""
    arms, methods, bins = [], [], []
    for cell in cells:
        report = cell.report
        columns = {name: report[name] for name in CELL_COLUMNS}
        base = report["methods"][HEAD_NAMES[report["head"]]]
        primary = report["primary_metric"]
        arms.append({
            **columns,
            "task_type": report["task_type"],
            "primary_metric": primary,
            "seeds": len(report["seeds"]),
            "v_fit_primary": report["v_fit_primary"],
            "test_primary": base["test"].get(primary),
            "test_primary_std": base["test_std"].get(primary),
        })
        for name, summary in report["methods"].items():
            for metric, value in summary["test"].items():
                methods.append({**columns, "method": name, "metric": metric, "mean": value, "std": summary["test_std"].get(metric)})
        for name, per_bin in report.get("identity_bins", {}).get("methods", {}).items():
            for bin_name, scores in per_bin.items():
                for metric, value in scores.items():
                    bins.append({**columns, "method": name, "bin": bin_name, "metric": metric, "value": value})
    return arms, methods, bins


def paired_rows(cells: Sequence[CellRun], results: Sequence[CellResult], replicates: int) -> list[dict[str, Any]]:
    """Every ordered pair of cells of one dataset and head: the difference and its 95% interval."""
    groups: dict[tuple[str, str], list[int]] = {}
    for index, cell in enumerate(cells):
        groups.setdefault((cell.report["data_name"], cell.report["head"]), []).append(index)
    rows = []
    for (data_name, head), members in groups.items():
        for first, second in permutations(members, 2):
            lower, upper = interval(results[first].draws - results[second].draws)
            rows.append({
                "data_name": data_name,
                "head": head,
                "cell_a": cells[first].cell_id,
                "cell_b": cells[second].cell_id,
                "primary_metric": cells[first].report["primary_metric"],
                "difference": results[first].point - results[second].point,
                "lower_95": lower,
                "upper_95": upper,
                "replicates": replicates,
            })
    return rows


def _size_metric(metric: str, score: str) -> str:
    return metric.replace("_coverage@", "_width@" if score == "conformal" else "_size@")


def conformal_rows(cells: Sequence[CellRun], results: Sequence[CellResult]) -> list[dict[str, Any]]:
    """Conformal coverage and size or width under both schemes, beside the band their calibration size allows.

    Scheme V is each report's seed-mean test coverage, calibrated on V_cal; scheme T the mean
    and interval over repeats, calibrated on T_conformal. A Mondrian set has no band, since its
    groups' calibration sizes vary.
    """
    rows = []
    for cell, cell_result in zip(cells, results, strict=True):
        v_cal = (cell.report.get("split") or {}).get("v_cal")
        for method, record in cell.report["methods"].items():
            means = record["test"]
            for metric, coverage in means.items():
                match = CONFORMAL_METRIC.match(metric)
                if match is None or match.group(3) != "coverage" or coverage is None:
                    continue
                mondrian, score, _, level = match.groups()
                band = conformal_band(float(level), v_cal) if v_cal and not mondrian else {}
                rows.append({
                    "scheme": "V", **_cell_key(cell), "method": method, "set": f"{mondrian or ''}{score}", "level": float(level),
                    "coverage": coverage, "size_or_width": means.get(_size_metric(metric, score)), **band,
                })

        overall = {(row["method"], row["metric"]): row for row in cell_result.scheme_t if not row["bin"]}
        for (method, metric), row in overall.items():
            match = CONFORMAL_METRIC.match(metric)
            if match is None or match.group(3) != "coverage":
                continue
            mondrian, score, _, level = match.groups()
            band = {} if mondrian else conformal_band(float(level), cell_result.calibration_rows)
            rows.append({
                "scheme": "T", **_cell_key(cell), "method": method, "set": f"{mondrian or ''}{score}", "level": float(level),
                "coverage": row["mean"], "coverage_lower_95": row["lower"], "coverage_upper_95": row["upper"],
                "size_or_width": overall.get((method, _size_metric(metric, score)), {}).get("mean"), **band,
            })
    return rows


def cell_records(cells: Sequence[CellRun], results: Sequence[CellResult]) -> list[dict[str, Any]]:
    """What verdicts.json and `q1_scores.csv` say about each cell."""
    records = []
    for cell, cell_result in zip(cells, results, strict=True):
        lower, upper = interval(cell_result.draws)
        defined_seeds = [value for value in cell_result.seed_points.values() if math.isfinite(value)]
        records.append({
            **_cell_key(cell),
            "cell_id": cell.cell_id,
            "directory": str(cell.directory),
            "task_type": cell.report["task_type"],
            "primary_metric": cell.report["primary_metric"],
            "seeds": list(cell.report["seeds"]),
            "test_primary": cell_result.point,
            "test_primary_lower_95": lower,
            "test_primary_upper_95": upper,
            "test_primary_per_seed": dict(cell_result.seed_points),
            "test_primary_seed_std_ddof1": seed_spread(defined_seeds),
            "v_fit_primary": cell.report.get("v_fit_primary"),
            **cell_result.notes,
            "scheme_t_calibration_rows": cell_result.calibration_rows,
            "mondrian_groupings": cell_result.mondrian_groupings,
        })
    return records


def scheme_t_rows(cells: Sequence[CellRun], results: Sequence[CellResult]) -> list[dict[str, Any]]:
    """Every method's scheme T score on every cell: the mean and the 2.5 and 97.5 percentiles over repeats."""
    return [
        {
            **_cell_key(cell), "method": row["method"], "score_only": row["score_only"], "metric": row["metric"],
            "mean": row["mean"], "lower_95": row["lower"], "upper_95": row["upper"], "defined": row["defined"], "repeats": row["draws"],
        }
        for cell, cell_result in zip(cells, results, strict=True)
        for row in cell_result.scheme_t
        if not row["bin"]
    ]


def identity_coverage_rows(cells: Sequence[CellRun], results: Sequence[CellResult]) -> list[dict[str, Any]]:
    """Scheme T conformal coverage and size or width in each identity bin, marginal and Mondrian."""
    rows = []
    for cell, cell_result in zip(cells, results, strict=True):
        merged: dict[tuple[str, str, float, str], dict[str, Any]] = {}
        for row in cell_result.scheme_t:
            match = CONFORMAL_METRIC.match(row["metric"])
            if not row["bin"] or match is None:
                continue
            mondrian, score, measure, level = match.groups()
            key = (row["method"], f"{mondrian or ''}{score}", float(level), row["bin"])
            entry = merged.setdefault(key, {**_cell_key(cell), "method": key[0], "set": key[1], "level": key[2], "bin": key[3]})
            if measure == "coverage":
                entry.update(coverage=row["mean"], coverage_lower_95=row["lower"], coverage_upper_95=row["upper"], defined=row["defined"])
            else:
                entry["size_or_width"] = row["mean"]
        rows += merged.values()
    return rows


def _comparison_rows(section: str, record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One comparison's per-dataset rows, then its pooled row with the verdict."""
    comparison = f"{record['first']} - {record['second']}"
    rows = [
        {
            "section": section, "comparison": comparison, "data_name": row["data_name"], "first": row["first"], "second": row["second"],
            "difference": row["difference"], "lower_95": row["lower"], "upper_95": row["upper"], "seed_std_ddof1": row["seed_std"],
            "datasets": 1,
        }
        for row in record["per_dataset"]
    ]
    pooled = record["pooled"]
    rows.append({
        "section": section, "comparison": comparison, "data_name": "pooled", "first": record["first"], "second": record["second"],
        "difference": pooled["difference"], "lower_95": pooled["lower"], "upper_95": pooled["upper"], "seed_std_ddof1": pooled["seed_std"],
        "datasets": pooled["datasets"], "positive_datasets": pooled["positive_datasets"], "verdict": record["verdict"],
    })
    return rows


def _recommendation_rows(answer: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each real arm against the best, in cost order, from question 1 or its sensitivity block."""
    return [
        {
            "section": "recommendation", "comparison": f"{row['arm']} - {answer['best_arm']}", "data_name": "pooled",
            "first": row["arm"], "second": answer["best_arm"], "difference": row["difference"], "lower_95": row["lower"],
            "upper_95": row["upper"], "seed_std_ddof1": row["seed_std"], "datasets": len(answer["common_datasets"]),
            "verdict": "qualifies" if row["qualifies"] else "does not qualify",
        }
        for row in answer["recommendation_rows"]
    ]


def q1_pair_rows(q1: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every question 1 comparison, per dataset and pooled, and the recommendation's rows."""
    rows = []
    for record in q1["pairs"]:
        rows += _comparison_rows("pair", record)
    for record in q1["controls"]["pairs"]:
        rows += _comparison_rows("control", record)
    return rows + _recommendation_rows(q1)


def q1_sensitivity_rows(q1: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The pairs and the recommendation without the datasets where a control reaches a real arm."""
    sensitivity = q1["sensitivity"]
    rows = []
    for record in sensitivity["pairs"]:
        rows += _comparison_rows("pair", record)
    rows += _recommendation_rows(sensitivity)
    return [{"excluded_datasets": sensitivity["excluded_datasets"], **row} for row in rows]


def q2_pooled_rows(q2: Mapping[str, Any], scheme: str) -> list[dict[str, Any]]:
    """Every candidate's and the reference's quantities per dataset and pooled, under one scheme."""
    rows = []
    for arm, record in q2["per_arm"].items():
        assessed = [("candidate", candidate) for candidate in record["candidates"]] + [("reference", record["reference"])]
        for role, candidate in assessed:
            base = {"scheme": scheme, "arm": arm, "method": candidate["method"], "role": role}
            for per_dataset in candidate["per_dataset"]:
                for quantity in QUANTITIES:
                    values = per_dataset[quantity]
                    rows.append({
                        **base, "data_name": per_dataset["data_name"], "quantity": quantity, "mean": values["mean"],
                        "lower_95": values["lower"], "upper_95": values["upper"], "defined": values["defined"],
                    })
            if candidate["pooled"] is None:
                continue
            for quantity in QUANTITIES:
                values = candidate["pooled"][quantity]
                rows.append({
                    **base, "data_name": "pooled", "quantity": quantity, "mean": values["mean"], "lower_95": values["lower"],
                    "upper_95": values["upper"], "defined": values["defined"], "improves": candidate["improves"],
                    "failed": "; ".join(candidate["failed"]),
                })
    return rows


def shift_rows(scheme_t: Sequence[Mapping[str, Any]], shift: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each method's shift AUROC per cell, then pooled over the datasets that define it."""
    rows = [
        {
            "level": "dataset", "data_name": row["data_name"], "arm": row["arm"], "head": row["head"], "method": row["method"],
            "mean": row["mean"], "lower_95": row["lower_95"], "upper_95": row["upper_95"], "defined": row["defined"], "datasets": 1,
        }
        for row in scheme_t
        if row["metric"] == "shift_auroc"
    ]
    for record in shift:
        pooled = record["pooled"] or {}
        rows.append({
            "level": "pooled", "data_name": "pooled", "arm": record["arm"], "head": record["head"], "method": record["method"],
            "mean": pooled.get("mean"), "lower_95": pooled.get("lower"), "upper_95": pooled.get("upper"), "defined": pooled.get("defined"),
            "datasets": len(record["datasets"]), "excluded_datasets": record["excluded_datasets"],
        })
    return rows


def check_rows(cells: Sequence[CellRun], results: Sequence[CellResult]) -> list[dict[str, Any]]:
    """Every check of every cell, with what it measured as JSON."""
    rows = []
    for cell, cell_result in zip(cells, results, strict=True):
        for check in cell_result.checks:
            detail = {key: value for key, value in check.items() if key not in {"check", "method", "seed", "passed"}}
            rows.append({
                **_cell_key(cell), "cell_id": cell.cell_id, "check": check["check"], "method": check.get("method"),
                "seed": check.get("seed"), "passed": bool(check.get("passed", True)),
                "detail": json.dumps(json_ready(detail), sort_keys=True, allow_nan=False),
            })
    return rows


def checks_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_check: dict[str, dict[str, int]] = {}
    for row in rows:
        counts = by_check.setdefault(row["check"], {"total": 0, "failed": 0})
        counts["total"] += 1
        counts["failed"] += int(not row["passed"])
    failed = [{**row, "detail": json.loads(row["detail"])} for row in rows if not row["passed"]]
    return {"total": len(rows), "failed": len(failed), "by_check": by_check, "failed_checks": failed}


def investigation_clause(q1: Mapping[str, Any]) -> str:
    """The clause `, provisional until <datasets> are investigated`, empty when no control reaches a real arm."""
    pending = q1["sensitivity"]["excluded_datasets"]
    if not pending:
        return ""
    return f", provisional until {', '.join(pending)} {'is' if len(pending) == 1 else 'are'} investigated"


def _with_spread(value: float | None, spread: float | None) -> str:
    """A seed-mean score and its seed spread as `mean ± sd`, or the mean alone where the spread is undefined."""
    if value is None or spread is None or not (math.isfinite(value) and math.isfinite(spread)):
        return _number(value)
    return f"{_number(value)} ± {_number(spread)}"


def _recommendation_table(answer: Mapping[str, Any], rules: DecisionRules) -> list[str]:
    """Each real arm against the best arm, in cost order, from question 1 or its sensitivity block."""
    best = answer["best_arm"]
    return _table(
        ["arm", f"pooled difference from {best}", "95% interval", "seed sd", f"lower bound above -{rules.margin:g}"],
        [
            [row["arm"], _signed(row["difference"]), _span(row["lower"], row["upper"]), _number(row["seed_std"]), "yes" if row["qualifies"] else "no"]
            for row in answer["recommendation_rows"]
        ],
    )


def _recommendation_lines(q1: Mapping[str, Any], rules: DecisionRules) -> list[str]:
    if q1["recommendation"] is None:
        return ["**No recommendation**: no dataset has a projected cell of every real arm present."]
    best = q1["best_arm"]
    lines = [
        f"**Recommendation: {q1['recommendation']}{investigation_clause(q1)}.** The best real arm by pooled mean over "
        f"{len(q1['common_datasets'])} datasets is {best}, at {_number(q1['pooled_means'][best])}. The recommendation is "
        f"the cheapest real arm, in the order {', '.join(rules.cost_order)}, whose pooled difference from {best} has "
        f"a lower bound above -{rules.margin:g}.",
    ]
    sensitivity = q1["sensitivity"]
    pending = sensitivity["excluded_datasets"]
    if pending:
        pronoun = "it" if len(pending) == 1 else "them"
        lines += [
            "",
            f"On {', '.join(pending)}, a control's point estimate reaches a real arm's, and the README counts such a "
            f"dataset only once it is investigated. Without {pronoun}, the recommendation is "
            f"{sensitivity['recommendation'] or 'undefined'}; the last section gives every pair without {pronoun}.",
        ]
    if q1["missing_real_arms"]:
        lines += ["", f"Real arms without a projected cell: {', '.join(q1['missing_real_arms'])}."]
    controls = q1["controls"]
    if controls["passed"]:
        lines += ["", "**Controls: passed.** Every real arm's pooled advantage over every control has its interval above 0."]
    else:
        failing = [f"{record['first']} over {record['second']}" for record in controls["pairs"] if not record["above_zero"]]
        missing = [f"{pair['real_arm']} over {pair['control']}" for pair in controls["unmeasured"]]
        reasons = []
        if failing:
            reasons.append(f"pooled advantages whose interval does not lie above 0: {', '.join(failing)}")
        if missing:
            reasons.append(f"pairs with no dataset in common: {', '.join(missing)}")
        text = "; ".join(reasons) or "no control was compared"
        lines += ["", f"**Controls: failed.** {text[:1].upper()}{text[1:]}."]
    return lines


def q1_markdown(q1: Mapping[str, Any], records: Sequence[Mapping[str, Any]], rules: DecisionRules, replicates: int) -> str:
    projected = [record for record in records if record["head"] == "projected"]
    datasets = sorted({record["data_name"] for record in projected})
    present = {record["arm"] for record in projected}
    known = (*rules.real_arms, *rules.control_arms)
    arms = [arm for arm in known if arm in present] + sorted(present - set(known))
    point = {(record["data_name"], record["arm"]): record["test_primary"] for record in projected}
    spread = {(record["data_name"], record["arm"]): record["test_primary_seed_std_ddof1"] for record in projected}
    metric = {record["data_name"]: record["primary_metric"] for record in projected}

    lines = [
        "# Question 1: featurization",
        "",
        "The projected head's test primary metric, the mean over seeds: MCC, F1 for multilabel tasks, and",
        "Spearman for regression. A difference's 95% interval is the 2.5 and 97.5 percentiles over",
        f"{replicates} bootstrap replicates of the test rows, stratified by label, each replicate drawing",
        "the same rows for both arms. A pooled row is the mean of the per-dataset differences, and",
        "replicate b of the pool averages replicate b of every dataset. The verdict, in precedence order:",
        "the first arm is better if the pooled interval lies above 0, the second if it lies below 0, the",
        f"two are equivalent if it lies inside (-{rules.margin:g}, +{rules.margin:g}), and it is unresolved otherwise.",
        "",
        "The seed spread sits beside every score and interval. A score's is its standard deviation over",
        "seeds (ddof = 1), written mean ± sd, which is not an interval. A difference's seed sd is the",
        "standard deviation over seeds (ddof = 1) of the difference between the two arms' runs of one",
        "seed; a pooled row first averages each seed's difference over its datasets.",
        "",
        *_recommendation_lines(q1, rules),
        "",
        "## Scores",
        "",
        *_table(
            ["dataset", "metric", *arms],
            [[name, metric[name], *(_with_spread(point.get((name, arm)), spread.get((name, arm))) for arm in arms)] for name in datasets]
            + [["pooled mean, common datasets", "", *(_number(q1["pooled_means"].get(arm)) if arm in q1["pooled_means"] else "" for arm in arms)]],
        ),
    ]

    for record in q1["pairs"]:
        combined = "chosen_on" in record
        title = f"## {record['first']} minus {record['second']}" + (", the single arm chosen per dataset on V_fit" if combined else "")
        header = ["dataset", *(["best single"] if combined else []), "difference", "95% interval", "seed sd"]
        rows = [
            [
                row["data_name"], *([row["second"]] if combined else []), _signed(row["difference"]), _span(row["lower"], row["upper"]),
                _number(row["seed_std"]),
            ]
            for row in record["per_dataset"]
        ]
        pooled = record["pooled"]
        rows.append([
            f"pooled, {pooled['datasets']} datasets, {pooled['positive_datasets']} positive", *([""] if combined else []),
            _signed(pooled["difference"]), _span(pooled["lower"], pooled["upper"]), _number(pooled["seed_std"]),
        ])
        lines += ["", title, "", *_table(header, rows), "", verdict_in_words(record, rules.margin)]
        if combined and record["datasets_without_v_fit"]:
            lines += ["", f"Left out, since neither single arm has a V_fit primary: {', '.join(record['datasets_without_v_fit'])}."]

    if q1["recommendation_rows"]:
        lines += ["", "## Recommendation, in cost order", "", *_recommendation_table(q1, rules)]

    controls = q1["controls"]
    lines += ["", "## Controls", ""]
    lines += _table(
        ["real arm", "control", "pooled difference", "95% interval", "seed sd", "datasets", "above 0"],
        [
            [record["first"], record["second"], _signed(record["pooled"]["difference"]), _span(record["pooled"]["lower"], record["pooled"]["upper"]),
             _number(record["pooled"]["seed_std"]), str(record["pooled"]["datasets"]), "yes" if record["above_zero"] else "no"]
            for record in controls["pairs"]
        ],
    )
    if controls["reached"]:
        lines += ["", "Datasets where a control's point estimate reaches a real arm's:", ""]
        lines += _table(
            ["dataset", "control", "real arm", "control score", "real arm score"],
            [[row["data_name"], row["control"], row["real_arm"], _number(row["control_score"]), _number(row["real_score"])] for row in controls["reached"]],
        )
    else:
        lines += ["", "No control's point estimate reaches a real arm's on any dataset."]
    lines += _sensitivity_lines(q1, rules)
    return "\n".join(lines) + "\n"


def _sensitivity_lines(q1: Mapping[str, Any], rules: DecisionRules) -> list[str]:
    """Question 1's last section: the pairs and the recommendation without the datasets to investigate."""
    sensitivity = q1["sensitivity"]
    pending = sensitivity["excluded_datasets"]
    lines = ["", "## Without the datasets to investigate", ""]
    if not pending:
        return lines + ["No control's point estimate reaches a real arm's, so no dataset waits on an investigation."]

    pronoun = "it" if len(pending) == 1 else "them"
    lines += [
        f"The pairs and the recommendation again without {', '.join(pending)}, where a control's point estimate",
        "reaches a real arm's. Each pooled row averages the same bootstrap replicates over the datasets that remain.",
        "",
        *_table(
            ["comparison", "datasets", "pooled difference", "95% interval", "seed sd", "verdict"],
            [
                [f"{record['first']} minus {record['second']}", str(record["pooled"]["datasets"]), _signed(record["pooled"]["difference"]),
                 _span(record["pooled"]["lower"], record["pooled"]["upper"]), _number(record["pooled"]["seed_std"]), record["verdict"]]
                for record in sensitivity["pairs"]
            ],
        ),
        "",
    ]
    if sensitivity["recommendation"] is None:
        return lines + [f"No recommendation without {pronoun}: no remaining dataset has a projected cell of every real arm present."]
    best = sensitivity["best_arm"]
    return lines + [
        f"Without {pronoun}, the best real arm by pooled mean over {len(sensitivity['common_datasets'])} datasets is {best}, at "
        f"{_number(sensitivity['pooled_means'][best])}, and the recommendation is **{sensitivity['recommendation']}**.",
        "",
        *_recommendation_table(sensitivity, rules),
    ]


def _summary_cell(values: Mapping[str, Any] | None, with_interval: bool = True, signed: bool = True) -> str:
    """A summary's mean, and its interval unless the summary is one draw."""
    if not values:
        return "n/a"
    mean = _signed(values["mean"]) if signed else _number(values["mean"])
    return f"{mean} {_span(values['lower'], values['upper'], signed)}" if with_interval else mean


def _improvement_in_words(candidate: Mapping[str, Any]) -> str:
    if candidate["pooled"] is None:
        return f"{candidate['method']}: not assessed, since no dataset has it, the reference, and plain."
    if candidate["improves"]:
        return f"{candidate['method']} improves on the reference: every condition holds."
    return f"{candidate['method']} does not improve on the reference; it fails: {'; '.join(candidate['failed'])}."


def _pooled_table(q2: Mapping[str, Any], with_interval: bool) -> list[str]:
    """Every candidate on every arm, pooled over datasets, then what each arm adopts."""
    rows = []
    for arm, record in q2["per_arm"].items():
        for candidate in record["candidates"]:
            pooled = candidate["pooled"] or {}
            rows.append([
                arm, candidate["method"], str(len(candidate["datasets"])),
                *(_summary_cell(pooled.get(quantity), with_interval) for quantity in QUANTITIES),
                "yes" if candidate["improves"] else "no", "; ".join(candidate["failed"]),
            ])
    header = ["arm", "candidate", "datasets", *(QUANTITY_LABELS[quantity] for quantity in QUANTITIES), "improves", "fails"]
    adopted = "; ".join(f"{arm}, {record['adopted']}" for arm, record in q2["per_arm"].items())
    return [*_table(header, rows), "", f"Adopted per arm: {adopted or 'no arm has scores'}.", *_coverage_lines(q2)]


def _coverage_lines(q2: Mapping[str, Any]) -> list[str]:
    """The datasets each candidate lacks, per arm, and what the adopted candidate's gaps would take."""
    lacking = [(arm, row) for arm, record in q2["per_arm"].items() for row in record["coverage"]["lacking"]]
    if not lacking:
        return []
    lines = [
        "",
        "Candidates pool over different datasets, since each is pooled over the datasets that have it, the",
        "reference, and plain:",
        "",
        *(f"- {arm}: {row['method']} lacks {row['data_name']} ({row['task_type']}; {row['reason']})." for arm, row in lacking),
    ]
    substitutes = [(arm, row) for arm, record in q2["per_arm"].items() for row in record["coverage"]["substitutes"]]
    if substitutes:
        lines += [
            "",
            "Where an arm's adopted candidate lacks a dataset, that dataset would take the first later candidate",
            "in the adoption order that improves on the arm, else the fallback. This is reported, not applied:",
            "",
            *(f"- {arm}: {row['data_name']} would take {row['takes']}." for arm, row in substitutes),
        ]
    return lines


def _adoption_lines(q2: Mapping[str, Any], rules: DecisionRules) -> list[str]:
    recommended, adopted = q2["recommended_arm"], q2["adopted"]
    if recommended is None or adopted is None:
        return ["**No adoption**: question 1 recommends no arm with scheme T scores."]
    if adopted == FALLBACK_ADOPTION:
        lines = [f"**On {recommended}, question 1's arm, no candidate improves on {rules.reference_method}: adopt {FALLBACK_ADOPTION}.**"]
    else:
        lines = [f"**On {recommended}, question 1's arm, adopt {adopted}**, the cheapest candidate that improves on {rules.reference_method}."]
    for row in q2["per_arm"][recommended]["coverage"]["substitutes"]:
        lines += [
            "",
            f"{adopted} has no scores on {row['data_name']} ({row['task_type']}). Under the same adoption order that dataset "
            f"would take {row['takes']}; this is reported, not applied.",
        ]
    differing = q2["arms_differing"]
    if differing:
        lines += ["", "Real arms whose adoption differs: " + ", ".join(f"{arm} ({q2['per_arm'][arm]['adopted']})" for arm in differing) + "."]
    else:
        lines += ["", "Every real arm with scheme T scores adopts the same."]
    return lines


def _conformal_lines(conformal: Sequence[Mapping[str, Any]], arm: str, methods: Sequence[str]) -> list[str]:
    """Scheme T against scheme V for the given methods on one arm, beside each scheme's band."""
    by_key = {(row["scheme"], row["data_name"], row["head"], row["method"], row["set"], row["level"]): row for row in conformal if row["arm"] == arm}
    rows = []
    for (scheme, data_name, head, method, set_name, level), row in by_key.items():
        if scheme != "T" or method not in methods or set_name.startswith("mondrian_"):
            continue
        other = by_key.get(("V", data_name, head, method, set_name, level), {})
        rows.append([
            data_name, head, method, set_name, f"{level:g}",
            f"{_number(row['coverage'])} {_span(row.get('coverage_lower_95'), row.get('coverage_upper_95'), signed=False)}",
            _number(row["size_or_width"]),
            f"[{_number(row.get('marginal_lower'))}, {_number(row.get('marginal_upper'))})",
            _number(other.get("coverage")), _number(other.get("size_or_width")),
            f"[{_number(other.get('beta_lower'))}, {_number(other.get('beta_upper'))}], n = {other.get('calibration_rows', 'n/a')}",
        ])
    if not rows:
        return ["No conformal rows for this arm."]
    header = [
        "dataset", "head", "method", "set", "level", "T coverage, 95% over repeats", "T size or width", "T marginal band",
        "V coverage", "V size or width", "V one-calibration-set band",
    ]
    return _table(header, rows)


def _identity_lines(identity_rows: Sequence[Mapping[str, Any]], arm: str, method: str) -> list[str]:
    """Scheme T coverage per identity bin, the mean over datasets, marginal beside Mondrian."""
    pooled: dict[tuple[str, float, str], dict[str, list[float]]] = {}
    for row in identity_rows:
        coverage = row.get("coverage")
        if row["arm"] != arm or row["method"] != method or coverage is None or not math.isfinite(coverage):
            continue
        mondrian = row["set"].startswith("mondrian_")
        key = (row["set"].removeprefix("mondrian_"), row["level"], row["bin"])
        pooled.setdefault(key, {"marginal": [], "mondrian": []})["mondrian" if mondrian else "marginal"].append(coverage)
    if not pooled:
        return [f"No identity-bin rows for {method} on {arm}; identity files may be absent."]
    rows = [
        [set_name, f"{level:g}", bin_name, _number(float(np.mean(values["marginal"])) if values["marginal"] else None),
         _number(float(np.mean(values["mondrian"])) if values["mondrian"] else None), str(len(values["marginal"]))]
        for (set_name, level, bin_name), values in pooled.items()
    ]
    return _table(["set", "level", "identity bin", "marginal coverage", "Mondrian coverage", "datasets"], rows)


def q2_markdown(
    q2: Mapping[str, Any],
    conformal: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    rules: DecisionRules,
    repeats: int,
) -> str:
    recommended = q2["recommended_arm"]
    candidates = ", ".join(f"{method} ({passes} forward pass{'' if passes == 1 else 'es'})" for method, passes in rules.candidates)
    lines = [
        "# Question 2: uncertainty",
        "",
        f"Scheme T decides. In each of {repeats} repeats, the test rows are split in half, stratified by label",
        "(by label decile for regression): one half of T_cal fits every calibration map, the other every",
        "conformal quantile, and a stratified bootstrap resample of T_eval is scored. A number is the mean",
        "over repeats, and its interval the 2.5 and 97.5 percentiles. A candidate's relative NLL (Gaussian",
        f"NLL for regression) and relative AURC are (candidate - {rules.reference_method}) / |{rules.reference_method}|, and its primary",
        f"difference is candidate - {rules.plain_method}; each is averaged over datasets within a repeat. AURC gives rows tied in",
        "uncertainty their mean loss, the expectation over every order of the tied rows, so it does not depend",
        f"on the order the rows are stored in; on a plain regression head, {rules.reference_method} gives every row one sigma.",
        "",
        "A candidate improves when its pooled relative NLL interval lies below 0, its pooled relative AURC",
        f"upper bound is below +{rules.aurc_bound:g}, and its pooled primary difference has a lower bound above -{rules.primary_margin:g}.",
        f"The cheapest improving candidate at inference is adopted, in the order {candidates}.",
        "",
        *_adoption_lines(q2, rules),
        "",
        "## Pooled over datasets, scheme T",
        "",
        *_pooled_table(q2, with_interval=True),
    ]

    if recommended in q2["per_arm"]:
        lines += ["", f"## {recommended}, per dataset, scheme T"]
        for candidate in q2["per_arm"][recommended]["candidates"]:
            rows = [
                [row["data_name"], *(_summary_cell(row[quantity]) for quantity in QUANTITIES)]
                for row in candidate["per_dataset"]
            ]
            if candidate["pooled"] is not None:
                rows.append([f"pooled, {len(candidate['datasets'])} datasets", *(_summary_cell(candidate["pooled"][quantity]) for quantity in QUANTITIES)])
            lines += ["", f"### {candidate['method']}", "", *_table(["dataset", *(QUANTITY_LABELS[quantity] for quantity in QUANTITIES)], rows)]
            lines += ["", _improvement_in_words(candidate)]

    scheme_v = q2["scheme_v"]
    lines += [
        "",
        "## Scheme V, beside it",
        "",
        "Scheme V reads each report's seed-mean test scores, calibrated on V_fit and V_cal, one value per",
        "dataset, so it has no interval and its conditions compare points.",
        "",
        *_adoption_lines(scheme_v, rules),
        "",
        *_pooled_table(scheme_v, with_interval=False),
    ]

    focus = [recommended] if recommended in q2["per_arm"] else list(q2["per_arm"])
    shift = [record for record in q2["shift"] if record["arm"] in focus]
    lines += [
        "",
        "## Shift",
        "",
        "The AUROC with which each method's uncertainty separates test proteins below 0.3 identity to",
        "training, no hit included, from those at 0.5 or above, pooled over the datasets where every repeat",
        "defines it.",
        "",
    ]
    if shift:
        lines += _table(
            ["arm", "head", "method", "shift AUROC", "datasets", "left out"],
            [
                [record["arm"], record["head"], record["method"], _summary_cell(record["pooled"], signed=False),
                 str(len(record["datasets"])), ", ".join(record["excluded_datasets"])]
                for record in shift
            ],
        )
    else:
        lines += ["No shift AUROC: no cell of these arms has an identity file."]

    adopted = q2["adopted"]
    methods = [rules.reference_method] + ([adopted] if adopted and adopted != FALLBACK_ADOPTION else [])
    for arm in focus:
        lines += [
            "",
            f"## Conformal coverage on {arm}",
            "",
            "Scheme T's mean coverage over repeats has the marginal band [level, level + 1 / (n + 1)) for its",
            "T_conformal size n; one calibration set's coverage has the 95% band of Beta(k, n + 1 - k), shown",
            "for scheme V's V_cal size, with k = ceil((n + 1) level).",
            "",
            *_conformal_lines(conformal, arm, methods),
            "",
            f"### Coverage per identity bin, {rules.reference_method}, scheme T, the mean over datasets",
            "",
            *_identity_lines(identity_rows, arm, rules.reference_method),
        ]
    return "\n".join(lines) + "\n"


def summary_markdown(arms: Sequence[Mapping[str, Any]], paired: Sequence[Mapping[str, Any]], q1: Mapping[str, Any], q2: Mapping[str, Any]) -> str:
    """summary.md as compare wrote it before, under a line naming both answers."""
    lines = [
        "# Uncertainty cells",
        "",
        f"Question 1 recommends {q1['recommendation'] or 'no arm'}{investigation_clause(q1)}; question 2 adopts {q2['adopted'] or 'nothing'}. "
        "See q1_featurization.md, q2_uncertainty.md, and verdicts.json.",
        "",
        "## Arms",
        "",
        "| dataset | cell | head | primary | V_fit | test | std |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {row['data_name']} | {row['cell_id']} | {row['head']} | {row['primary_metric']} | "
        f"{_legacy_format(row['v_fit_primary'])} | {_legacy_format(row['test_primary'])} | {_legacy_format(row['test_primary_std'])} |"
        for row in arms
    ]
    lines += ["", "## Paired bootstrap, A minus B", "", "| dataset | head | A | B | difference | 95% interval |", "|---|---|---|---|---|---|"]
    lines += [
        f"| {row['data_name']} | {row['head']} | {row['cell_a']} | {row['cell_b']} | {_legacy_format(row['difference'])} | "
        f"[{_legacy_format(row['lower_95'])}, {_legacy_format(row['upper_95'])}] |"
        for row in paired
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    *,
    cells: Sequence[CellRun],
    results: Sequence[CellResult],
    q1: Mapping[str, Any],
    q2: Mapping[str, Any],
    parameters: Mapping[str, Any],
    rules: DecisionRules,
) -> dict[str, Any]:
    """Write every file into `output_dir` and return what verdicts.json holds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = cell_records(cells, results)
    arms, methods, bins = report_tables(cells)
    paired = paired_rows(cells, results, parameters["replicates"])
    conformal = conformal_rows(cells, results)
    scheme_t = scheme_t_rows(cells, results)
    identity = identity_coverage_rows(cells, results)
    checks = check_rows(cells, results)

    verdicts = json_ready({
        "parameters": parameters,
        "cells": records,
        "question_1": q1,
        "question_2": q2,
        "conformal": conformal,
        "checks": checks_summary(checks),
    })
    write_json(str(output_dir / "verdicts.json"), verdicts)
    _write_text(output_dir / "q1_featurization.md", q1_markdown(q1, records, rules, parameters["replicates"]))
    _write_text(output_dir / "q2_uncertainty.md", q2_markdown(q2, conformal, identity, rules, parameters["repeats"]))
    _write_text(output_dir / "summary.md", summary_markdown(arms, paired, q1, q2))

    write_csv(output_dir / "q1_scores.csv", records, SCORE_COLUMNS)
    write_csv(output_dir / "q1_pairs.csv", q1_pair_rows(q1), PAIR_COLUMNS)
    write_csv(output_dir / "q1_sensitivity.csv", q1_sensitivity_rows(q1), SENSITIVITY_COLUMNS)
    write_csv(output_dir / "q2_scheme_t.csv", scheme_t, SCHEME_T_COLUMNS)
    write_csv(output_dir / "q2_pooled.csv", q2_pooled_rows(q2, "T") + q2_pooled_rows(q2["scheme_v"], "V"), POOLED_COLUMNS)
    write_csv(output_dir / "q2_shift.csv", shift_rows(scheme_t, q2["shift"]), SHIFT_COLUMNS)
    write_csv(output_dir / "q2_identity_coverage.csv", identity, IDENTITY_COLUMNS)
    write_csv(output_dir / "conformal_bands.csv", conformal, CONFORMAL_COLUMNS)
    write_csv(output_dir / "checks.csv", checks, CHECK_COLUMNS)
    write_csv(output_dir / "arms.csv", arms, (*CELL_COLUMNS, "task_type", "primary_metric", "seeds", "v_fit_primary", "test_primary", "test_primary_std"))
    write_csv(output_dir / "methods.csv", methods, (*CELL_COLUMNS, "method", "metric", "mean", "std"))
    write_csv(output_dir / "identity_bins.csv", bins, (*CELL_COLUMNS, "method", "bin", "metric", "value"))
    write_csv(output_dir / "paired_bootstrap.csv", paired, PAIRED_COLUMNS)
    return verdicts
