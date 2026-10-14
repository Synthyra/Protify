"""The ablation's pre-registered decision rules, applied to scores gathered across cells.

The rules are those of the experiment README's "Decision rules, fixed before launch"
(`experiments/tooling/property_lowcode/2026-09-24_esmc600_representation_uncertainty`), with every threshold a field of
`DecisionRules`.

Question 1, featurization, reads the projected head. Each (dataset, arm) has the seed-mean test
primary metric on every test row and on each of `B` stratified bootstrap replicates, replicate
`b` drawing the same rows for every arm of that dataset. A pair's difference on a dataset has
the replicates' 2.5 and 97.5 percentiles as its 95% interval. Its pooled difference is the mean
of the per-dataset differences, and replicate `b` of the pool averages replicate `b` of every
dataset; since each dataset draws its replicates independently, that is a joint bootstrap. The
verdict, in precedence order: the first arm better if the pooled interval lies above 0, the
second better if below 0, equivalent if inside (-margin, +margin), unresolved otherwise. Beside
each interval sits the seed spread: the standard deviation, with ddof = 1, of the difference
between the two arms' runs of one seed. The README counts a dataset where a control reaches a
real arm only once it is investigated, so the pairs and the recommendation are also given
without every such dataset.

Question 2, uncertainty, reads scheme T. Each (dataset, arm, method) has a score in each of `R`
repeats, repeat `r` drawing the same rows for every cell of that dataset. A candidate's relative
NLL (Gaussian NLL for regression) and relative AURC are `(candidate - reference) / |reference|`
per dataset and repeat, and its primary difference is `candidate - plain`; each is averaged over
datasets within a repeat, and its interval is the 2.5 and 97.5 percentiles over repeats. A
quantity undefined on any of its datasets in a repeat leaves that repeat out, and the count left
out is reported. A candidate is pooled over the datasets that have it, so candidates can pool
over different datasets, and the datasets each one lacks are named.

Shapes: `B` bootstrap replicates, `R` repeats, `D` datasets.
"""

from __future__ import annotations

import math
import numpy as np

from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any
from scipy.stats import beta


INTERVAL_PERCENTILES = (2.5, 97.5)
# What is adopted when no candidate improves on the reference.
FALLBACK_ADOPTION = "temperature scaling plus split conformal on an exchangeable calibration split"


@dataclass(frozen=True)
class DecisionRules:
    """Every threshold and arm list the decision rules read."""

    # Question 1's verdict margin, and the recommendation's lower-bound margin.
    margin: float = 0.01
    real_arms: tuple[str, ...] = ("sae", "last", "l27", "both")
    control_arms: tuple[str, ...] = ("onehot", "random", "random_init")
    # Real arms, cheapest backbone first.
    cost_order: tuple[str, ...] = ("l27", "sae", "last", "both")
    pairs: tuple[tuple[str, str], ...] = (("sae", "last"), ("l27", "last"), ("sae", "l27"))
    # The combined arm is compared with whichever single arm scores higher on V_fit, per dataset.
    combined_arm: str = "both"
    single_arms: tuple[str, ...] = ("sae", "last")
    reference_method: str = "temperature"
    plain_method: str = "plain"
    # Question 2's candidates and their forward passes at inference, the cost they are ordered by.
    candidates: tuple[tuple[str, int], ...] = (("sngp_temperature", 1), ("ensemble_temperature", 5), ("mc_dropout_temperature", 30))
    # A candidate's pooled relative AURC must stay below this upper bound.
    aurc_bound: float = 0.02
    # A candidate's pooled primary difference from plain must have its lower bound above minus this.
    primary_margin: float = 0.01

    def __post_init__(self) -> None:
        if sorted(self.cost_order) != sorted(self.real_arms):
            raise ValueError(f"the cost order {list(self.cost_order)} must name every real arm {list(self.real_arms)} once.")
        if self.margin <= 0.0:
            raise ValueError(f"the margin must be positive; got {self.margin}.")

    def as_dict(self) -> dict[str, Any]:
        return json_ready(asdict(self))


def json_ready(value: Any) -> Any:
    """Plain JSON values: tuples as lists, NumPy scalars unwrapped, non-finite floats as None."""
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def interval(draws: np.ndarray) -> tuple[float, float]:
    """The 2.5 and 97.5 percentiles of the defined draws, NaN where none is."""
    # draws: (B,) or (R,)
    defined = draws[np.isfinite(draws)]  # (k,)
    if defined.size == 0:
        return math.nan, math.nan
    lower, upper = np.percentile(defined, INTERVAL_PERCENTILES)
    return float(lower), float(upper)


def summary(draws: np.ndarray) -> dict[str, float | int]:
    """The mean and 95% interval of the defined draws, and how many are defined."""
    # draws: (B,) or (R,)
    defined = np.isfinite(draws)  # (B,)
    lower, upper = interval(draws)
    mean = float(draws[defined].mean()) if defined.any() else math.nan
    return {"mean": mean, "lower": lower, "upper": upper, "defined": int(defined.sum()), "draws": int(draws.shape[0])}


def verdict(lower: float, upper: float, margin: float, first: str, second: str) -> str:
    """Question 1's verdict on a pooled interval of `first` minus `second`."""
    if not (math.isfinite(lower) and math.isfinite(upper)):
        return "undefined"
    if lower > 0.0:
        return f"{first} better"
    if upper < 0.0:
        return f"{second} better"
    if lower > -margin and upper < margin:
        return "equivalent"
    return "unresolved"


def verdict_in_words(record: Mapping[str, Any], margin: float) -> str:
    """One sentence saying what a pair's pooled interval shows."""
    pooled = record["pooled"]
    lower, upper = pooled["lower"], pooled["upper"]
    span = f"[{lower:+.4f}, {upper:+.4f}]" if math.isfinite(lower) and math.isfinite(upper) else "undefined"
    outcome = record["verdict"]
    if outcome == "undefined":
        return "No verdict: the pooled interval is undefined."
    if outcome == "equivalent":
        return f"Equivalent: the pooled 95% interval {span} lies inside (-{margin:g}, +{margin:g})."
    if outcome == "unresolved":
        return f"Unresolved: the pooled 95% interval {span} contains 0 and reaches past the margin of {margin:g}."
    side = "above" if outcome == f"{record['first']} better" else "below"
    return f"{outcome}: the pooled 95% interval {span} lies {side} 0."


@dataclass(frozen=True)
class ArmScores:
    """Question 1's inputs, keyed by (dataset, arm), for the projected head."""

    point: Mapping[tuple[str, str], float]  # the seed-mean test primary on every test row
    draws: Mapping[tuple[str, str], np.ndarray]  # (B,) the same on each bootstrap replicate
    v_fit: Mapping[tuple[str, str], float | None]  # the report's seed-mean V_fit primary
    # Each seed's test primary on every test row, by seed; two arms' runs of one seed pair up.
    per_seed: Mapping[tuple[str, str], Mapping[int, float]] = field(default_factory=dict)

    def datasets_with(self, arms: Sequence[str]) -> list[str]:
        """Datasets scored under every one of `arms`, sorted."""
        names = sorted({data_name for data_name, _ in self.point})
        return [name for name in names if all((name, arm) in self.point for arm in arms)]

    def excluding(self, data_names: Collection[str]) -> ArmScores:
        """The same scores without the named datasets; every other dataset keeps its replicates."""
        return ArmScores(
            {key: value for key, value in self.point.items() if key[0] not in data_names},
            {key: value for key, value in self.draws.items() if key[0] not in data_names},
            {key: value for key, value in self.v_fit.items() if key[0] not in data_names},
            {key: value for key, value in self.per_seed.items() if key[0] not in data_names},
        )


def seed_spread(values: Sequence[float]) -> float:
    """The standard deviation over seeds, with ddof = 1; NaN below two seeds."""
    return float(np.std(values, ddof=1)) if len(values) >= 2 else math.nan


def seed_differences(scores: ArmScores, data_name: str, first_arm: str, second_arm: str) -> dict[int, float]:
    """First minus second on one dataset for each seed both arms ran, where both runs are defined."""
    first = scores.per_seed.get((data_name, first_arm), {})
    second = scores.per_seed.get((data_name, second_arm), {})
    differences = {seed: first[seed] - second[seed] for seed in sorted(set(first) & set(second))}
    return {seed: difference for seed, difference in differences.items() if math.isfinite(difference)}


def compare_arms(scores: ArmScores, rows: Sequence[tuple[str, str, str]], first: str, second: str, margin: float) -> dict[str, Any]:
    """First minus second on each (dataset, first arm, second arm) row, and pooled over the rows.

    Each difference carries its seed spread: the standard deviation over seeds of the per-seed
    difference, pairing the two arms' runs by seed. Seed `s` of the pool averages seed `s` of
    every dataset, over the seeds every dataset has.
    """
    per_dataset, points, draws, by_seed = [], [], [], []
    for data_name, first_arm, second_arm in rows:
        difference = scores.point[(data_name, first_arm)] - scores.point[(data_name, second_arm)]
        replicate_differences = scores.draws[(data_name, first_arm)] - scores.draws[(data_name, second_arm)]  # (B,)
        lower, upper = interval(replicate_differences)
        seed_rows = seed_differences(scores, data_name, first_arm, second_arm)
        per_dataset.append({
            "data_name": data_name, "first": first_arm, "second": second_arm,
            "difference": difference, "lower": lower, "upper": upper,
            "seed_std": seed_spread(list(seed_rows.values())), "seeds": len(seed_rows),
        })
        points.append(difference)
        draws.append(replicate_differences)
        by_seed.append(seed_rows)

    record: dict[str, Any] = {"first": first, "second": second, "per_dataset": per_dataset}
    if not rows:
        record["pooled"] = {
            "difference": math.nan, "lower": math.nan, "upper": math.nan, "datasets": 0, "positive_datasets": 0,
            "undefined_replicates": 0, "seed_std": math.nan, "seeds": 0,
        }
        record["verdict"] = "undefined"
        return record
    pooled_draws = np.mean(np.stack(draws), axis=0)  # (B,), NaN where any dataset's replicate is
    lower, upper = interval(pooled_draws)
    pooled_seeds = sorted(set.intersection(*(set(seed_rows) for seed_rows in by_seed)))
    pooled_by_seed = [float(np.mean([seed_rows[seed] for seed_rows in by_seed])) for seed in pooled_seeds]
    record["pooled"] = {
        "difference": float(np.mean(points)),
        "lower": lower,
        "upper": upper,
        "datasets": len(rows),
        "positive_datasets": int(np.sum(np.array(points) > 0.0)),
        "undefined_replicates": int(np.sum(~np.isfinite(pooled_draws))),
        "seed_std": seed_spread(pooled_by_seed),
        "seeds": len(pooled_seeds),
    }
    record["verdict"] = verdict(lower, upper, margin, first, second)
    return record


def better_single_arm(scores: ArmScores, data_name: str, rules: DecisionRules) -> str | None:
    """The single arm with the higher V_fit primary on a dataset, the cheaper on a tie; None if neither has one."""
    candidates = [
        arm for arm in rules.single_arms
        if (data_name, arm) in scores.point and scores.v_fit.get((data_name, arm)) is not None
        and math.isfinite(scores.v_fit[(data_name, arm)])
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda arm: (scores.v_fit[(data_name, arm)], -rules.cost_order.index(arm) if arm in rules.cost_order else 0))


def compare_pairs(scores: ArmScores, rules: DecisionRules) -> list[dict[str, Any]]:
    """Every fixed pair, then the combined arm against the single arm better on V_fit."""
    pairs = []
    for first, second in rules.pairs:
        rows = [(name, first, second) for name in scores.datasets_with((first, second))]
        pairs.append(compare_arms(scores, rows, first, second, rules.margin))

    # The combined arm against the better single arm on V_fit, chosen per dataset, never on test.
    combined_rows, skipped = [], []
    for data_name in scores.datasets_with((rules.combined_arm,)):
        single = better_single_arm(scores, data_name, rules)
        if single is None:
            skipped.append(data_name)
        else:
            combined_rows.append((data_name, rules.combined_arm, single))
    combined = compare_arms(scores, combined_rows, rules.combined_arm, "best single", rules.margin)
    combined["chosen_on"] = "v_fit_primary"
    combined["datasets_without_v_fit"] = skipped
    pairs.append(combined)
    return pairs


def without_datasets(scores: ArmScores, rules: DecisionRules, excluded: Sequence[str]) -> dict[str, Any]:
    """The pairs and the recommendation again without `excluded`, from the same replicates.

    Replicate `b` of a pool then averages replicate `b` of the datasets that remain.
    """
    kept = scores.excluding(excluded)
    return {"excluded_datasets": list(excluded), "pairs": compare_pairs(kept, rules), **recommendation(kept, rules)}


def question_one(scores: ArmScores, rules: DecisionRules) -> dict[str, Any]:
    """Every pair's verdict, the best and recommended arms, the controls check, and the answer without the datasets to investigate.

    The README counts a dataset where a control's point estimate reaches a real arm's only once it
    is investigated, so `sensitivity` gives the pairs and the recommendation without every such
    dataset.
    """
    pairs = compare_pairs(scores, rules)

    # A pair with no dataset in common cannot show its interval above 0, so it fails the check.
    control_pairs, unmeasured = [], []
    for real in rules.real_arms:
        for control in rules.control_arms:
            rows = [(name, real, control) for name in scores.datasets_with((real, control))]
            if not rows:
                unmeasured.append({"real_arm": real, "control": control})
                continue
            record = compare_arms(scores, rows, real, control, rules.margin)
            record["above_zero"] = bool(record["pooled"]["lower"] > 0.0)
            control_pairs.append(record)
    reached = [
        {
            "data_name": data_name, "control": control, "real_arm": real,
            "control_score": scores.point[(data_name, control)], "real_score": scores.point[(data_name, real)],
        }
        for real in rules.real_arms
        for control in rules.control_arms
        for data_name in scores.datasets_with((real, control))
        if scores.point[(data_name, control)] >= scores.point[(data_name, real)]
    ]

    return {
        "pairs": pairs,
        **recommendation(scores, rules),
        "controls": {
            "passed": bool(control_pairs) and not unmeasured and all(record["above_zero"] for record in control_pairs),
            "pairs": control_pairs,
            "unmeasured": unmeasured,
            "reached": reached,
        },
        "sensitivity": without_datasets(scores, rules, sorted({row["data_name"] for row in reached})),
    }


def recommendation(scores: ArmScores, rules: DecisionRules) -> dict[str, Any]:
    """The best real arm by pooled mean and the cheapest whose pooled difference from it clears the margin."""
    present = [arm for arm in rules.real_arms if any(key[1] == arm for key in scores.point)]
    missing = [arm for arm in rules.real_arms if arm not in present]
    common = scores.datasets_with(present)
    if not present or not common:
        return {
            "best_arm": None, "pooled_means": {}, "common_datasets": common, "missing_real_arms": missing,
            "recommendation": None, "recommendation_rows": [],
        }

    means = {arm: float(np.mean([scores.point[(name, arm)] for name in common])) for arm in present}
    best = max(present, key=lambda arm: (means[arm], -rules.cost_order.index(arm)))
    rows = []
    for arm in sorted(present, key=rules.cost_order.index):
        record = compare_arms(scores, [(name, arm, best) for name in common], arm, best, rules.margin)
        pooled = record["pooled"]
        rows.append({
            "arm": arm, "difference": pooled["difference"], "lower": pooled["lower"], "upper": pooled["upper"],
            "seed_std": pooled["seed_std"], "qualifies": bool(pooled["lower"] > -rules.margin) or arm == best,
        })
    recommended = next(row["arm"] for row in rows if row["qualifies"])
    return {
        "best_arm": best,
        "pooled_means": means,
        "common_datasets": common,
        "missing_real_arms": missing,
        "recommendation": recommended,
        "recommendation_rows": rows,
    }


@dataclass
class RepeatSeries:
    """Question 2's inputs: every (dataset, arm, head, method, metric) score in each scheme T repeat."""

    values: Mapping[tuple[str, str, str, str, str], np.ndarray]  # (R,)
    kinds: Mapping[str, str]  # dataset -> task kind
    primary: Mapping[str, str]  # dataset -> primary metric
    _by_method: dict[tuple[str, str, str, str], np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Method names differ between heads, so a method names its head.
        self._by_method = {(name, arm, method, metric): values for (name, arm, _, method, metric), values in self.values.items()}

    def find(self, data_name: str, arm: str, method: str, metric: str) -> np.ndarray | None:
        """A method's series on a dataset and arm, from whichever head builds it."""
        return self._by_method.get((data_name, arm, method, metric))  # (R,) or None

    def heads_scoring(self, method: str, arm: str | None = None) -> set[str]:
        """The heads whose cells score `method` on any dataset, on `arm`, or on any arm when it is None."""
        return {head for (_, key_arm, head, key_method, _) in self.values if key_method == method and arm in (None, key_arm)}

    def heads_on(self, data_name: str, arm: str) -> set[str]:
        """The heads with any score on one dataset and arm."""
        return {head for (name, key_arm, head, _, _) in self.values if name == data_name and key_arm == arm}


def nll_metric(kind: str) -> str:
    return "gaussian_nll" if kind == "regression" else "nll"


def _relative(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """`(candidate - reference) / |reference|`, NaN where the reference is 0 or either is undefined."""
    # candidate: (R,), reference: (R,)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = (candidate - reference) / np.abs(reference)  # (R,)
    return np.where(np.isfinite(relative), relative, np.nan)  # (R,)


def candidate_series(series: RepeatSeries, data_name: str, arm: str, method: str, rules: DecisionRules) -> dict[str, np.ndarray] | None:
    """A candidate's relative NLL, relative AURC, and primary difference from plain on one dataset, per repeat."""
    kind, primary = series.kinds[data_name], series.primary[data_name]
    nll = nll_metric(kind)
    needed = {
        "candidate_nll": (method, nll), "reference_nll": (rules.reference_method, nll),
        "candidate_aurc": (method, "aurc"), "reference_aurc": (rules.reference_method, "aurc"),
        "candidate_primary": (method, primary), "plain_primary": (rules.plain_method, primary),
    }
    found = {name: series.find(data_name, arm, *key) for name, key in needed.items()}
    if any(values is None for values in found.values()):
        return None
    return {  # each (R,) by name
        "relative_nll": _relative(found["candidate_nll"], found["reference_nll"]),
        "relative_aurc": _relative(found["candidate_aurc"], found["reference_aurc"]),
        "primary_difference": found["candidate_primary"] - found["plain_primary"],
    }


def pooled_over_datasets(per_dataset: Sequence[np.ndarray]) -> np.ndarray:
    """Each repeat's mean over datasets, NaN where any dataset's value is undefined."""
    # per_dataset: (R,) per dataset
    return np.mean(np.stack(per_dataset), axis=0)  # (R,)


def assess_candidate(series: RepeatSeries, arm: str, method: str, datasets: Sequence[str], rules: DecisionRules) -> dict[str, Any]:
    """One candidate on one arm: per-dataset and pooled summaries, and whether it improves on the reference."""
    per_dataset, pooled_inputs = [], {"relative_nll": [], "relative_aurc": [], "primary_difference": []}
    for data_name in datasets:
        values = candidate_series(series, data_name, arm, method, rules)
        if values is None:
            continue
        per_dataset.append({"data_name": data_name, **{name: summary(draws) for name, draws in values.items()}})
        for name, draws in values.items():
            pooled_inputs[name].append(draws)

    record: dict[str, Any] = {"method": method, "arm": arm, "datasets": [row["data_name"] for row in per_dataset], "per_dataset": per_dataset}
    if not per_dataset:
        record.update(pooled=None, improves=False, failed=["no dataset has this candidate, the reference, and plain"])
        return record
    pooled = {name: summary(pooled_over_datasets(draws)) for name, draws in pooled_inputs.items()}
    conditions = {
        "relative NLL interval below 0": pooled["relative_nll"]["upper"] < 0.0,
        f"relative AURC upper bound below +{rules.aurc_bound:g}": pooled["relative_aurc"]["upper"] < rules.aurc_bound,
        f"primary difference lower bound above -{rules.primary_margin:g}": pooled["primary_difference"]["lower"] > -rules.primary_margin,
    }
    record.update(pooled=pooled, improves=all(conditions.values()), failed=[name for name, held in conditions.items() if not held])
    return record


def adoption_order(rules: DecisionRules) -> list[str]:
    """The candidates, cheapest at inference first, the earlier listed on a tie."""
    passes = dict(rules.candidates)
    return sorted(passes, key=lambda method: (passes[method], list(passes).index(method)))


def adoption(candidates: Sequence[Mapping[str, Any]], rules: DecisionRules) -> str:
    """The cheapest improving candidate at inference, or the fallback when none improves."""
    improving = {record["method"] for record in candidates if record["improves"]}
    return next((method for method in adoption_order(rules) if method in improving), FALLBACK_ADOPTION)


def _lacking_reason(series: RepeatSeries, data_name: str, arm: str, method: str) -> str:
    """Why a candidate has no scores on a dataset: the head that scores it has no cell there, or that cell lacks it."""
    # The heads that score it on this arm's other datasets, or on any arm when this arm has none.
    heads = sorted(series.heads_scoring(method, arm) or series.heads_scoring(method))
    if heads and not set(heads) & series.heads_on(data_name, arm):
        return f"no {' or '.join(heads)} cell of {arm}"
    return f"no {method} scores"


def dataset_coverage(series: RepeatSeries, arm: str, record: Mapping[str, Any], rules: DecisionRules) -> dict[str, Any]:
    """The datasets each candidate lacks on one arm, and what the adopted candidate's gaps would take.

    A candidate is pooled over the datasets that have it, the reference, and plain, so candidates
    can pool over different datasets: EC is multilabel and runs no SNGP. Each dataset the arm's
    reference is scored on but a candidate is not is named, with its task type and why. A dataset
    the adopted candidate lacks is named with what it would take under the same order: the first
    later candidate that improves on this arm and has it, else the fallback. That is reported,
    not applied; the adoption rule is unchanged.
    """
    universe = record["reference"]["datasets"]
    lacking = [
        {
            "method": candidate["method"], "data_name": name, "task_type": series.kinds[name],
            "reason": _lacking_reason(series, name, arm, candidate["method"]),
        }
        for candidate in record["candidates"]
        for name in universe
        if name not in candidate["datasets"]
    ]
    adopted = record["adopted"]
    order = adoption_order(rules)
    later = order[order.index(adopted) + 1:] if adopted in order else []
    by_method = {candidate["method"]: candidate for candidate in record["candidates"]}
    substitutes = []
    for row in lacking:
        if row["method"] != adopted:
            continue
        takes = next(
            (method for method in later if by_method[method]["improves"] and row["data_name"] in by_method[method]["datasets"]),
            FALLBACK_ADOPTION,
        )
        substitutes.append({"data_name": row["data_name"], "task_type": row["task_type"], "takes": takes})
    return {"datasets": list(universe), "lacking": lacking, "substitutes": substitutes}


def question_two(series: RepeatSeries, rules: DecisionRules, recommended_arm: str | None, arms: Sequence[str]) -> dict[str, Any]:
    """Every candidate on each of `arms`, the adoption on the recommended arm, and the arms where it differs.

    Each arm's `coverage` names the datasets each candidate lacks; see `dataset_coverage`.
    """
    datasets = sorted(series.kinds)
    per_arm = {}
    for arm in arms:
        candidates = [assess_candidate(series, arm, method, datasets, rules) for method, _ in rules.candidates]
        reference = assess_candidate(series, arm, rules.reference_method, datasets, rules)
        per_arm[arm] = {"candidates": candidates, "reference": reference, "adopted": adoption(candidates, rules)}
        per_arm[arm]["coverage"] = dataset_coverage(series, arm, per_arm[arm], rules)
    adopted = per_arm[recommended_arm]["adopted"] if recommended_arm in per_arm else None
    return {
        "reference_method": rules.reference_method,
        "recommended_arm": recommended_arm,
        "adopted": adopted,
        "per_arm": per_arm,
        "arms_differing": [arm for arm, record in per_arm.items() if adopted is not None and record["adopted"] != adopted],
    }


def pooled_shift(series: RepeatSeries) -> list[dict[str, Any]]:
    """Each (arm, head, method)'s shift AUROC pooled over the datasets where every repeat defines it."""
    groups: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for (data_name, arm, head, method, metric), values in series.values.items():
        if metric == "shift_auroc":
            groups.setdefault((arm, head, method), {})[data_name] = values
    rows = []
    for (arm, head, method), by_dataset in sorted(groups.items()):
        included = sorted(name for name, values in by_dataset.items() if np.isfinite(values).all())
        excluded = sorted(set(by_dataset) - set(included))
        pooled = summary(pooled_over_datasets([by_dataset[name] for name in included])) if included else None
        rows.append({"arm": arm, "head": head, "method": method, "datasets": included, "excluded_datasets": excluded, "pooled": pooled})
    return rows


def conformal_band(level: float, calibration_rows: int) -> dict[str, float | int]:
    """What split conformal at `level` with `calibration_rows` calibration rows guarantees.

    Marginal coverage is `k / (n + 1)` with `k = ceil((n + 1) level)`, so it lies in
    `[level, level + 1 / (n + 1))`; coverage conditional on one calibration set is
    `Beta(k, n + 1 - k)`, whose 2.5 and 97.5 percentiles are given. Past `k = n` the quantile
    is infinite and every set holds everything.
    """
    rows = int(calibration_rows)
    # As `foundry.eval.conformal.conformal_quantile` computes it from alpha = 1 - level, so the
    # rank agrees where (n + 1) * level lands on an integer.
    rank = int(math.ceil((rows + 1) * (1.0 - (1.0 - level))))
    if rows < 1 or rank > rows:
        return {"calibration_rows": rows, "rank": rank, "marginal": 1.0, "marginal_lower": 1.0, "marginal_upper": 1.0, "beta_lower": 1.0, "beta_upper": 1.0}
    beta_lower, beta_upper = beta.ppf([0.025, 0.975], rank, rows + 1 - rank)
    return {
        "calibration_rows": rows,
        "rank": rank,
        "marginal": rank / (rows + 1),
        "marginal_lower": level,
        "marginal_upper": level + 1.0 / (rows + 1),
        "beta_lower": float(beta_lower),
        "beta_upper": float(beta_upper),
    }
