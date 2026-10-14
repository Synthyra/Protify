"""The single-probe experiment's pre-registered decision rules, applied to scheme T scores.

The rules are those of "Decision rules, fixed before launch" in
`experiments/tooling/property_lowcode/2026-09-25_single_probe_uncertainty/README.md`, with every threshold a field of
`SingleProbeRules`. They read the same `RepeatSeries` the ablation's question 2 reads: each
(dataset, arm, head, method, metric) in each of `R` scheme T repeats, repeat `r` drawing the
same rows for every cell of a dataset. A quantity is computed per dataset and repeat, averaged
over the datasets that have it within a repeat, and its interval is the 2.5 and 97.5
percentiles over repeats (`verdicts.summary`). Every rule applies to each family separately:
classification (binary, multiclass, and multilabel datasets) and regression.

- The head: `linear` minus `projected` on the plain head's primary metric, worded as the
  ablation's question 1 words a pair (`verdicts.verdict`).
- A candidate is a (head, method). Against the reference, `temperature` on `projected`, it
  improves when (1) the lower bound of its primary metric minus the plain `projected` head's is
  above `-margin`, (2) its relative NLL has an interval below 0, and (3) its relative AURC has an
  upper bound below 0. `temperature_pnorm` shares `temperature`'s probabilities, so its primary
  metric and NLL are read from `temperature` on its head, and rule 2 is replaced by "its NLL is
  the reference's": it improves on rules 1 and 3 alone.
- Adoption: the cheapest tier holding an improving candidate, then the lowest pooled mean
  relative AURC within it, the earlier listed on a tie. With no improving candidate, the family
  falls back to `temperature` with split conformal, on `linear` when the family's head verdict
  is "linear better" or "equivalent" and on `projected` otherwise.
- The ensemble's gain kept: for each candidate, its pooled relative NLL and AURC over those of
  `ensemble_temperature` on `projected`, pooled over the candidate's datasets. The share is the
  ratio of the pooled means; its interval is the percentiles over repeats of each repeat's
  ratio, left undefined in a repeat where the ensemble's pooled value is 0.
- The clamp: `projected_qclip` minus `projected` on the plain primary metric, and the relative
  NLL and AURC of `temperature` on each. The quantile clip replaces plus or minus 10 only when
  its primary verdict is "projected_qclip better" and neither relative NLL nor relative AURC
  has an interval above 0.
- Shift: `verdicts.pooled_shift` over every head and method.

Shapes: `R` repeats.
"""

from __future__ import annotations

import json
import math
import os
import numpy as np

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .verdicts import RepeatSeries, json_ready, nll_metric, pooled_over_datasets, pooled_shift, summary, verdict


# Heads only the single-probe experiment trains; a comparison holding any of them gets its rules.
SINGLE_PROBE_HEADS = frozenset({"linear", "faithful", "beta_nll", "projected_qclip"})
FAMILIES = {"classification": ("binary", "multiclass", "multilabel"), "regression": ("regression",)}
QUANTITIES = ("primary_difference", "relative_nll", "relative_aurc")
OUTPUT_JSON = "single_probe.json"
OUTPUT_MARKDOWN = "single_probe.md"


@dataclass(frozen=True)
class SingleProbeRules:
    """Every threshold, head, and method the single-probe rules read."""

    margin: float = 0.01
    reference: tuple[str, str] = ("projected", "temperature")
    plain: tuple[str, str] = ("projected", "plain")
    ceiling: tuple[str, str] = ("projected", "ensemble_temperature")
    # (head, method, tier), in the README's order; tier 1 is cheapest.
    candidates: tuple[tuple[str, str, int], ...] = (
        ("linear", "temperature", 1),
        ("projected", "temperature_pnorm", 1),
        ("linear", "temperature_pnorm", 1),
        ("projected", "laplace_temperature", 1),
        ("linear", "laplace_temperature", 1),
        ("faithful", "faithful_temperature", 2),
        ("beta_nll", "beta_nll_temperature", 2),
        ("projected", "loss_head", 3),
        ("linear", "loss_head", 3),
    )
    # Methods whose probabilities are another method's on the same head: primary and NLL come from it.
    shares_predictions: tuple[tuple[str, str], ...] = (("temperature_pnorm", "temperature"),)
    # The heads the head verdict compares, first minus second.
    head_pair: tuple[str, str] = ("linear", "projected")
    clamp_pair: tuple[str, str] = ("projected_qclip", "projected")

    def as_dict(self) -> dict[str, Any]:
        return json_ready(asdict(self))


def _relative(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """`(candidate - reference) / |reference|`, NaN where the reference is 0 or either is undefined."""
    # candidate: (R,), reference: (R,)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = (candidate - reference) / np.abs(reference)  # (R,)
    return np.where(np.isfinite(relative), relative, np.nan)  # (R,)


def _series(series: RepeatSeries, data_name: str, arm: str, head: str, method: str, metric: str) -> np.ndarray | None:
    return series.values.get((data_name, arm, head, method, metric))  # (R,) or None


def family_datasets(series: RepeatSeries, family: str) -> list[str]:
    """The family's datasets, sorted."""
    return sorted(name for name, kind in series.kinds.items() if kind in FAMILIES[family])


def candidate_quantities(
    series: RepeatSeries, data_name: str, arm: str, head: str, method: str, rules: SingleProbeRules
) -> dict[str, np.ndarray] | None:
    """A (head, method)'s primary difference from plain `projected`, relative NLL, and relative AURC, per repeat.

    None when the dataset lacks the candidate, the reference, or the plain head.
    """
    kind, primary = series.kinds[data_name], series.primary[data_name]
    nll = nll_metric(kind)
    predictions = dict(rules.shares_predictions).get(method, method)
    found = {
        "candidate_primary": _series(series, data_name, arm, head, predictions, primary),
        "candidate_nll": _series(series, data_name, arm, head, predictions, nll),
        "candidate_aurc": _series(series, data_name, arm, head, method, "aurc"),
        "plain_primary": _series(series, data_name, arm, *rules.plain, primary),
        "reference_nll": _series(series, data_name, arm, *rules.reference, nll),
        "reference_aurc": _series(series, data_name, arm, *rules.reference, "aurc"),
    }
    if any(values is None for values in found.values()):
        return None
    return {
        "primary_difference": found["candidate_primary"] - found["plain_primary"],  # (R,)
        "relative_nll": _relative(found["candidate_nll"], found["reference_nll"]),  # (R,)
        "relative_aurc": _relative(found["candidate_aurc"], found["reference_aurc"]),  # (R,)
    }


def _pooled(series: RepeatSeries, arm: str, head: str, method: str, datasets: Sequence[str], rules: SingleProbeRules) -> tuple[list[str], dict[str, np.ndarray]]:
    """The datasets that have a candidate, and each quantity pooled over them per repeat."""
    present, inputs = [], {name: [] for name in QUANTITIES}
    for data_name in datasets:
        values = candidate_quantities(series, data_name, arm, head, method, rules)
        if values is None:
            continue
        present.append(data_name)
        for name in QUANTITIES:
            inputs[name].append(values[name])
    return present, {name: pooled_over_datasets(draws) for name, draws in inputs.items() if draws}  # names, (R,) per quantity


def assess(series: RepeatSeries, arm: str, head: str, method: str, tier: int, datasets: Sequence[str], rules: SingleProbeRules) -> dict[str, Any]:
    """One candidate in one family: its pooled quantities and whether it improves on the reference."""
    present, pooled = _pooled(series, arm, head, method, datasets, rules)
    record: dict[str, Any] = {
        "head": head, "method": method, "tier": tier, "datasets": present,
        "lacking": [name for name in datasets if name not in present],
    }
    if not present:
        record.update(pooled=None, improves=False, conditions={}, failed=["no dataset has this candidate, the reference, and plain projected"])
        return record

    summaries = {name: summary(draws) for name, draws in pooled.items()}
    conditions = {
        f"primary lower bound above -{rules.margin:g}": summaries["primary_difference"]["lower"] > -rules.margin,
        "relative NLL interval below 0": summaries["relative_nll"]["upper"] < 0.0,
        "relative AURC upper bound below 0": summaries["relative_aurc"]["upper"] < 0.0,
    }
    replaced = method in dict(rules.shares_predictions)
    if replaced:
        # Its probabilities are temperature's, so its NLL is the reference's rather than a test.
        conditions.pop("relative NLL interval below 0")
    record.update(
        pooled=summaries,
        conditions=conditions,
        nll_rule_replaced=replaced,
        improves=all(conditions.values()),
        failed=[name for name, held in conditions.items() if not held],
    )
    return record


def head_verdict(series: RepeatSeries, arm: str, datasets: Sequence[str], rules: SingleProbeRules) -> dict[str, Any]:
    """`linear` minus `projected` on the plain primary metric, pooled, and its verdict."""
    first, second = rules.head_pair
    return _difference_verdict(series, arm, datasets, (first, "plain"), (second, "plain"), first, second, rules)


def _difference_verdict(
    series: RepeatSeries,
    arm: str,
    datasets: Sequence[str],
    first: tuple[str, str],
    second: tuple[str, str],
    first_name: str,
    second_name: str,
    rules: SingleProbeRules,
) -> dict[str, Any]:
    """First minus second on each dataset's primary metric, pooled, with the question 1 verdict."""
    per_dataset, draws = [], []
    for data_name in datasets:
        primary = series.primary[data_name]
        first_values = _series(series, data_name, arm, *first, primary)
        second_values = _series(series, data_name, arm, *second, primary)
        if first_values is None or second_values is None:
            continue
        difference = first_values - second_values  # (R,)
        per_dataset.append({"data_name": data_name, **summary(difference)})
        draws.append(difference)
    if not draws:
        return {"first": first_name, "second": second_name, "per_dataset": [], "pooled": None, "verdict": "undefined"}
    pooled = summary(pooled_over_datasets(draws))
    return {
        "first": first_name, "second": second_name, "per_dataset": per_dataset, "pooled": pooled,
        "verdict": verdict(pooled["lower"], pooled["upper"], rules.margin, first_name, second_name),
    }


def adoption(candidates: Sequence[Mapping[str, Any]], head_record: Mapping[str, Any], rules: SingleProbeRules) -> dict[str, Any]:
    """The cheapest tier's improving candidate with the lowest pooled relative AURC, or the fallback."""
    improving = [record for record in candidates if record["improves"]]
    if improving:
        tier = min(record["tier"] for record in improving)
        in_tier = [record for record in improving if record["tier"] == tier]
        best = min(in_tier, key=lambda record: record["pooled"]["relative_aurc"]["mean"])
        return {"head": best["head"], "method": best["method"], "tier": tier, "fallback": False}

    first, second = rules.head_pair
    head = first if head_record["verdict"] in (f"{first} better", "equivalent") else second
    return {"head": head, "method": "temperature", "tier": None, "fallback": True, "with": "split conformal on V_cal"}


def ensemble_share(series: RepeatSeries, arm: str, record: Mapping[str, Any], rules: SingleProbeRules) -> dict[str, Any] | None:
    """The share of the ensemble's pooled relative NLL and AURC gain a candidate keeps, over its datasets."""
    if not record["datasets"]:
        return None
    _, candidate = _pooled(series, arm, record["head"], record["method"], record["datasets"], rules)
    present, ceiling = _pooled(series, arm, *rules.ceiling, record["datasets"], rules)
    if present != record["datasets"]:
        return {"note": f"the ceiling lacks {sorted(set(record['datasets']) - set(present))}"}
    shares = {}
    for quantity in ("relative_nll", "relative_aurc"):
        mine, theirs = candidate[quantity], ceiling[quantity]  # (R,), (R,)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_repeat = np.where(theirs != 0.0, mine / theirs, np.nan)  # (R,)
        defined_mine, defined_theirs = mine[np.isfinite(mine)], theirs[np.isfinite(theirs)]
        ceiling_mean = float(defined_theirs.mean()) if defined_theirs.size else math.nan
        point = float(defined_mine.mean()) / ceiling_mean if defined_mine.size and ceiling_mean != 0.0 else math.nan
        shares[quantity] = {**summary(per_repeat), "share": point, "ensemble_mean": ceiling_mean}
    return shares


def clamp_verdict(series: RepeatSeries, arm: str, datasets: Sequence[str], rules: SingleProbeRules) -> dict[str, Any]:
    """Whether the quantile clip replaces plus or minus 10 in one family."""
    clipped, clamped = rules.clamp_pair
    primary = _difference_verdict(series, arm, datasets, (clipped, "plain"), (clamped, "plain"), clipped, clamped, rules)
    relative: dict[str, Any] = {}
    for quantity, metric_of in (("relative_nll", nll_metric), ("relative_aurc", lambda kind: "aurc")):
        draws = []
        for data_name in datasets:
            metric = metric_of(series.kinds[data_name])
            mine = _series(series, data_name, arm, clipped, "temperature", metric)
            theirs = _series(series, data_name, arm, clamped, "temperature", metric)
            if mine is not None and theirs is not None:
                draws.append(_relative(mine, theirs))
        relative[quantity] = summary(pooled_over_datasets(draws)) if draws else None
    worse = {
        quantity: values is not None and values["lower"] > 0.0 for quantity, values in relative.items()
    }
    replaces = primary["verdict"] == f"{clipped} better" and not any(worse.values())
    return {"primary": primary, **relative, "worse": worse, "replaces_clamp": replaces}


def single_probe_verdicts(series: RepeatSeries, arms: Sequence[str], rules: SingleProbeRules | None = None) -> dict[str, Any]:
    """Every family's head verdict, candidates, adoption, ensemble share, and clamp verdict, per arm."""
    rules = rules or SingleProbeRules()
    per_arm = {}
    for arm in arms:
        families = {}
        for family in FAMILIES:
            datasets = family_datasets(series, family)
            if not datasets:
                continue
            head_record = head_verdict(series, arm, datasets, rules)
            candidates = [assess(series, arm, head, method, tier, datasets, rules) for head, method, tier in rules.candidates]
            reference = assess(series, arm, *rules.reference, 0, datasets, rules)
            ceiling = assess(series, arm, *rules.ceiling, 0, datasets, rules)
            for record in candidates:
                record["ensemble_share"] = ensemble_share(series, arm, record, rules)
            adopted = adoption(candidates, head_record, rules)
            families[family] = {
                "datasets": datasets,
                "head": head_record,
                "reference": reference,
                "ceiling": ceiling,
                "candidates": candidates,
                "adopted": adopted,
                "clamp": clamp_verdict(series, arm, datasets, rules),
            }
        per_arm[arm] = families
    return {"rules": rules.as_dict(), "per_arm": per_arm, "shift": pooled_shift(series)}


def _defined(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _cell(values: Mapping[str, Any] | None) -> str:
    """A pooled mean and its 95% interval, from JSON-ready values in which undefined is None."""
    if values is None or not _defined(values.get("mean")):
        return "undefined"
    span = "[undefined]"
    if _defined(values.get("lower")) and _defined(values.get("upper")):
        span = f"[{values['lower']:+.4f}, {values['upper']:+.4f}]"
    return f"{values['mean']:+.4f} {span}"


def verdicts_markdown(verdicts: Mapping[str, Any]) -> str:
    """The head verdict, the candidates, the adoption, and the clamp, per arm and family."""
    lines = ["# Single-probe uncertainty: decision rules on scheme T", ""]
    for arm, families in verdicts["per_arm"].items():
        for family, record in families.items():
            head = record["head"]
            adopted = record["adopted"]
            lines += [
                f"## {arm}, {family} ({', '.join(record['datasets'])})",
                "",
                f"Head: {head['verdict']}, pooled {_cell(head['pooled'])}.",
                "",
                "| head | method | tier | datasets | primary minus plain projected | relative NLL | relative AURC | NLL share of ensemble | AURC share of ensemble | improves | fails |",
                "|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            for candidate in record["candidates"]:
                pooled = candidate["pooled"] or {}
                share = candidate.get("ensemble_share") or {}
                nll_cell = "its NLL is temperature's" if candidate.get("nll_rule_replaced") else _cell(pooled.get("relative_nll"))
                lines.append(
                    f"| {candidate['head']} | {candidate['method']} | {candidate['tier']} | {len(candidate['datasets'])} | "
                    f"{_cell(pooled.get('primary_difference'))} | {nll_cell} | {_cell(pooled.get('relative_aurc'))} | "
                    f"{_share(share.get('relative_nll'))} | {_share(share.get('relative_aurc'))} | "
                    f"{'yes' if candidate['improves'] else 'no'} | {'; '.join(candidate['failed'])} |"
                )
            if adopted["fallback"]:
                lines += ["", f"Adopted: no candidate improves, so temperature with split conformal on {adopted['head']}."]
            else:
                lines += ["", f"Adopted: {adopted['method']} on {adopted['head']} (tier {adopted['tier']})."]
            clamp = record["clamp"]
            lines += [
                "",
                f"Clamp: projected_qclip minus projected, primary {clamp['primary']['verdict']} "
                f"{_cell(clamp['primary']['pooled'])}, temperature relative NLL {_cell(clamp['relative_nll'])}, "
                f"relative AURC {_cell(clamp['relative_aurc'])}; the quantile clip "
                f"{'replaces' if clamp['replaces_clamp'] else 'does not replace'} plus or minus 10.",
                "",
            ]
    return "\n".join(lines)


def _share(values: Mapping[str, Any] | None) -> str:
    if values is None or not _defined(values.get("share")):
        return "undefined"
    span = "[undefined]"
    if _defined(values.get("lower")) and _defined(values.get("upper")):
        span = f"[{values['lower']:.2f}, {values['upper']:.2f}]"
    return f"{values['share']:.2f} {span}"


def write_single_probe(output_dir: Path, verdicts: Mapping[str, Any]) -> None:
    """Write the verdicts as JSON and Markdown beside the comparison's other outputs, atomically."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ready = json_ready(verdicts)
    text = json.dumps(ready, indent=2, sort_keys=True, allow_nan=False) + "\n"
    for name, content in ((OUTPUT_JSON, text), (OUTPUT_MARKDOWN, verdicts_markdown(ready) + "\n")):
        temporary = output_dir / f"{name}.partial"
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temporary, output_dir / name)
