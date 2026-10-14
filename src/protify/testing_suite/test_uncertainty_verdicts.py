"""The ablation's decision rules, on scores built so that the right answer is known.

Question 1 reads bootstrap draws and question 2 reads repeat draws; here both are synthetic, so
each rule is checked on its own: the verdict's precedence, the recommendation's cost order, the
combined arm's pairing on V_fit, the controls, the answer without the datasets a control reaches,
the seed spread's pairing by seed, question 2's adoption, and the datasets a candidate lacks.
"""

import math
import numpy as np
import pytest

from dataclasses import replace
from scipy.stats import beta


try:
    from src.protify.uncertainty.analysis_tables import investigation_clause, q1_markdown, q1_sensitivity_rows
    from src.protify.uncertainty.verdicts import (
        FALLBACK_ADOPTION,
        ArmScores,
        DecisionRules,
        RepeatSeries,
        conformal_band,
        question_one,
        question_two,
        verdict,
    )
except ImportError:
    from protify.uncertainty.analysis_tables import investigation_clause, q1_markdown, q1_sensitivity_rows
    from protify.uncertainty.verdicts import (
        FALLBACK_ADOPTION,
        ArmScores,
        DecisionRules,
        RepeatSeries,
        conformal_band,
        question_one,
        question_two,
        verdict,
    )


DATASETS = ("alpha", "beta", "gamma")
REPLICATES = 400
REPEATS = 200
CONTROLS = {"onehot": 0.3, "random": 0.1, "random_init": 0.2}
SEEDS = (42, 43, 44, 45, 46)


def _arm_scores(means: dict[str, float | dict[str, float]], v_fit: dict[str, float] | None = None) -> ArmScores:
    """Each arm's score per dataset, with replicate noise shared by every arm of a dataset, as a paired bootstrap has."""
    rng = np.random.default_rng(0)
    point, draws, fit = {}, {}, {}
    for data_name in DATASETS:
        shared = 0.02 * rng.normal(size=REPLICATES)  # (B,)
        for arm, value in means.items():
            score = value[data_name] if isinstance(value, dict) else value
            point[(data_name, arm)] = score
            draws[(data_name, arm)] = score + shared + 0.004 * rng.normal(size=REPLICATES)  # (B,)
            fit[(data_name, arm)] = (v_fit or {}).get(arm, score)
    return ArmScores(point, draws, fit)


def _pair(q1: dict, first: str, second: str) -> dict:
    return next(record for record in q1["pairs"] if (record["first"], record["second"]) == (first, second))


def test_the_verdict_follows_its_precedence():
    assert verdict(0.001, 0.005, 0.01, "a", "b") == "a better"
    assert verdict(-0.005, -0.001, 0.01, "a", "b") == "b better"
    assert verdict(-0.005, 0.005, 0.01, "a", "b") == "equivalent"
    assert verdict(-0.02, 0.005, 0.01, "a", "b") == "unresolved"
    assert verdict(math.nan, 0.005, 0.01, "a", "b") == "undefined"


def test_an_arm_better_by_construction_is_found_better_and_recommended():
    q1 = question_one(_arm_scores({"sae": 0.80, "last": 0.70, "l27": 0.70, "both": 0.72, **CONTROLS}), DecisionRules())

    assert _pair(q1, "sae", "last")["verdict"] == "sae better"
    assert _pair(q1, "sae", "l27")["verdict"] == "sae better"
    assert _pair(q1, "sae", "last")["pooled"]["positive_datasets"] == len(DATASETS)
    assert q1["best_arm"] == "sae"
    # l27 is cheaper, but 0.10 below the best is far past the margin.
    assert q1["recommendation"] == "sae"
    assert q1["controls"]["passed"] and not q1["controls"]["reached"]
    # No dataset waits on an investigation, so the answer stands as it is.
    assert (q1["sensitivity"]["excluded_datasets"], q1["sensitivity"]["recommendation"]) == ([], "sae")
    assert investigation_clause(q1) == ""


def test_near_identical_arms_are_equivalent_and_the_cheaper_is_recommended():
    means = {"sae": 0.70, "last": 0.750, "l27": 0.749, "both": 0.70}
    q1 = question_one(_arm_scores({**means, **CONTROLS}), DecisionRules())

    assert _pair(q1, "l27", "last")["verdict"] == "equivalent"
    assert q1["best_arm"] == "last"
    assert q1["recommendation"] == "l27"
    rows = {row["arm"]: row for row in q1["recommendation_rows"]}
    assert rows["l27"]["qualifies"] and not rows["sae"]["qualifies"]

    # The cost order is a parameter, and moves the recommendation with it.
    rules = DecisionRules(cost_order=("last", "l27", "sae", "both"))
    without_controls = question_one(_arm_scores(means), rules)
    assert without_controls["recommendation"] == "last"
    # With no control cells, no real arm can show it beats them.
    assert not without_controls["controls"]["passed"]
    assert len(without_controls["controls"]["unmeasured"]) == 12


def test_both_is_compared_with_the_single_arm_better_on_v_fit_not_on_test():
    # sae is better on test, last on V_fit: the pairing must follow V_fit.
    scores = _arm_scores({"sae": 0.80, "last": 0.70, "l27": 0.65, "both": 0.75}, v_fit={"sae": 0.60, "last": 0.65})
    combined = _pair(question_one(scores, DecisionRules()), "both", "best single")

    assert combined["chosen_on"] == "v_fit_primary"
    assert [row["second"] for row in combined["per_dataset"]] == ["last"] * len(DATASETS)
    assert combined["pooled"]["difference"] == pytest.approx(0.05)
    assert combined["verdict"] == "both better"


def test_controls_that_reach_a_real_arm_are_listed_and_a_control_level_with_one_fails():
    means = {"sae": 0.80, "last": 0.78, "l27": 0.76, "both": 0.80, "onehot": {"alpha": 0.3, "beta": 0.9, "gamma": 0.3}, "random": 0.1, "random_init": 0.76}
    controls = question_one(_arm_scores(means), DecisionRules())["controls"]

    reached = {(row["data_name"], row["control"], row["real_arm"]) for row in controls["reached"]}
    assert {("beta", "onehot", arm) for arm in ("sae", "last", "l27", "both")} <= reached
    assert ("alpha", "random_init", "l27") in reached
    assert not controls["passed"]
    failing = {(record["first"], record["second"]) for record in controls["pairs"] if not record["above_zero"]}
    assert ("l27", "random_init") in failing
    assert ("sae", "random") not in failing


def test_without_the_datasets_a_control_reaches_the_answer_is_recomputed_from_the_same_replicates():
    # On beta a control reaches every real arm, and l27's lead there alone makes it the best arm.
    means = {
        "sae": 0.75, "last": 0.70, "l27": {"alpha": 0.70, "beta": 0.95, "gamma": 0.70}, "both": 0.72,
        "onehot": {"alpha": 0.3, "beta": 0.96, "gamma": 0.3}, "random": 0.1, "random_init": 0.2,
    }
    scores = _arm_scores(means)
    q1 = question_one(scores, DecisionRules())
    sensitivity = q1["sensitivity"]

    assert {row["data_name"] for row in q1["controls"]["reached"]} == {"beta"}
    assert sensitivity["excluded_datasets"] == ["beta"]
    assert sensitivity["common_datasets"] == ["alpha", "gamma"]
    assert (q1["recommendation"], sensitivity["recommendation"]) == ("l27", "sae")
    assert [(record["first"], record["second"]) for record in sensitivity["pairs"]] == [(record["first"], record["second"]) for record in q1["pairs"]]
    assert _pair(q1, "sae", "l27")["verdict"] == "l27 better"
    kept = _pair(sensitivity, "sae", "l27")
    assert kept["verdict"] == "sae better"

    # The datasets that remain keep their replicates, and the pool averages them.
    full = {row["data_name"]: row for row in _pair(q1, "sae", "l27")["per_dataset"]}
    assert [(row["data_name"], row["difference"], row["lower"], row["upper"]) for row in kept["per_dataset"]] == [
        (name, full[name]["difference"], full[name]["lower"], full[name]["upper"]) for name in ("alpha", "gamma")
    ]
    pooled_draws = np.mean([scores.draws[(name, "sae")] - scores.draws[(name, "l27")] for name in ("alpha", "gamma")], axis=0)  # (B,)
    assert (kept["pooled"]["lower"], kept["pooled"]["upper"]) == pytest.approx(tuple(np.percentile(pooled_draws, (2.5, 97.5))))

    # What q1_featurization.md and q1_sensitivity.csv say.
    assert investigation_clause(q1) == ", provisional until beta is investigated"
    records = [
        {"head": "projected", "data_name": name, "arm": arm, "primary_metric": "mcc", "test_primary": value, "test_primary_seed_std_ddof1": 0.01}
        for (name, arm), value in scores.point.items()
    ]
    markdown = q1_markdown(q1, records, DecisionRules(), REPLICATES)
    assert "**Recommendation: l27, provisional until beta is investigated.**" in markdown
    assert "Without it, the recommendation is sae" in markdown
    assert "## Without the datasets to investigate" in markdown
    assert "| 0.7500 ± 0.0100 |" in markdown
    rows = q1_sensitivity_rows(q1)
    assert {tuple(row["excluded_datasets"]) for row in rows} == {("beta",)}
    assert {row["section"] for row in rows} == {"pair", "recommendation"}


def test_the_seed_spread_pairs_two_arms_runs_by_seed():
    base = _arm_scores({"sae": 0.80, "last": 0.70, "l27": 0.70, "both": 0.72, **CONTROLS})
    # A seed moves every arm alike, and sae's lead over last varies a little from seed to seed.
    seed_effect = np.array([0.00, 0.10, -0.05, 0.08, -0.12])  # (5,)
    lead = {"alpha": [0.10, 0.11, 0.09, 0.10, 0.12], "beta": [0.08, 0.10, 0.10, 0.11, 0.09], "gamma": [0.12, 0.10, 0.10, 0.09, 0.11]}
    per_seed = {}
    for name in DATASETS:
        per_seed[(name, "last")] = dict(zip(SEEDS, 0.70 + seed_effect, strict=True))
        per_seed[(name, "sae")] = dict(zip(SEEDS, 0.70 + seed_effect + np.array(lead[name]), strict=True))
        # l27 has no run of the last seed on alpha.
        per_seed[(name, "l27")] = dict(zip(SEEDS[:4] if name == "alpha" else SEEDS, 0.70 + seed_effect, strict=False))
    q1 = question_one(replace(base, per_seed=per_seed), DecisionRules())

    pair = _pair(q1, "sae", "last")
    for row in pair["per_dataset"]:
        assert row["seeds"] == 5
        assert row["seed_std"] == pytest.approx(np.std(lead[row["data_name"]], ddof=1))
    # Unpaired, each arm's own spread is several times the paired one.
    assert np.std(0.70 + seed_effect, ddof=1) > 5 * max(row["seed_std"] for row in pair["per_dataset"])
    # The pool averages each seed's difference over the datasets first.
    assert pair["pooled"]["seeds"] == 5
    assert pair["pooled"]["seed_std"] == pytest.approx(np.std(np.mean([lead[name] for name in DATASETS], axis=0), ddof=1))
    recommendation_rows = {row["arm"]: row for row in q1["recommendation_rows"]}
    assert recommendation_rows["last"]["seed_std"] == pytest.approx(pair["pooled"]["seed_std"])

    # A seed one arm lacks leaves that dataset's pairing and the pool's.
    partial = _pair(q1, "sae", "l27")
    assert {row["data_name"]: row["seeds"] for row in partial["per_dataset"]} == {"alpha": 4, "beta": 5, "gamma": 5}
    assert partial["pooled"]["seeds"] == 4


def _series(effects: dict[tuple[str, str], dict[str, float]]) -> RepeatSeries:
    """Question 2's inputs: temperature and plain at fixed levels, each candidate at the given relative changes.

    `effects[(arm, method)]` holds the candidate's relative NLL and AURC change and its primary
    difference from plain, each around which the repeats scatter.
    """
    rng = np.random.default_rng(1)
    values = {}
    for data_name in ("alpha", "beta"):
        for arm in ("sae", "last", "l27"):
            nll, aurc, primary = 0.5, 0.2, 0.6
            values[(data_name, arm, "projected", "plain", "mcc")] = np.full(REPEATS, primary)
            values[(data_name, arm, "projected", "temperature", "nll")] = np.full(REPEATS, nll)
            values[(data_name, arm, "projected", "temperature", "aurc")] = np.full(REPEATS, aurc)
            values[(data_name, arm, "projected", "temperature", "mcc")] = np.full(REPEATS, primary)
            for method, head in (("sngp_temperature", "sngp"), ("ensemble_temperature", "projected"), ("mc_dropout_temperature", "projected")):
                effect = effects.get((arm, method), {"nll": 0.0, "aurc": 0.0, "primary": 0.0})
                values[(data_name, arm, head, method, "nll")] = nll * (1 + effect["nll"] + 0.01 * rng.normal(size=REPEATS))
                values[(data_name, arm, head, method, "aurc")] = aurc * (1 + effect["aurc"] + 0.005 * rng.normal(size=REPEATS))
                values[(data_name, arm, head, method, "mcc")] = primary + effect["primary"] + 0.002 * rng.normal(size=REPEATS)
    return RepeatSeries(values, {"alpha": "binary", "beta": "binary"}, {"alpha": "mcc", "beta": "mcc"})


def test_the_cheapest_improving_candidate_is_adopted_and_differing_arms_are_named():
    better = {"nll": -0.2, "aurc": -0.1, "primary": 0.0}
    series = _series({
        # sae: every candidate improves, so the cheapest, SNGP, is adopted.
        ("sae", "sngp_temperature"): better, ("sae", "ensemble_temperature"): better, ("sae", "mc_dropout_temperature"): better,
        # last: SNGP's AURC is worse and MC dropout costs its primary metric, so the ensemble is adopted.
        ("last", "sngp_temperature"): {"nll": -0.2, "aurc": 0.1, "primary": 0.0},
        ("last", "ensemble_temperature"): better,
        ("last", "mc_dropout_temperature"): {"nll": -0.2, "aurc": -0.1, "primary": -0.05},
        # l27: nothing improves, so the fallback is adopted.
    })
    q2 = question_two(series, DecisionRules(), "sae", ("sae", "last", "l27"))

    assert q2["adopted"] == "sngp_temperature"
    assert q2["per_arm"]["last"]["adopted"] == "ensemble_temperature"
    assert q2["per_arm"]["l27"]["adopted"] == FALLBACK_ADOPTION
    assert q2["arms_differing"] == ["last", "l27"]
    last = {record["method"]: record for record in q2["per_arm"]["last"]["candidates"]}
    assert last["sngp_temperature"]["failed"] == ["relative AURC upper bound below +0.02"]
    assert last["mc_dropout_temperature"]["failed"] == ["primary difference lower bound above -0.01"]
    assert last["ensemble_temperature"]["pooled"]["relative_nll"]["upper"] < 0.0
    assert all(not record["coverage"]["lacking"] for record in q2["per_arm"].values())


def test_candidates_pooled_over_fewer_datasets_name_what_they_lack_and_what_it_would_take():
    better = {"nll": -0.2, "aurc": -0.1, "primary": 0.0}
    full = _series({("sae", "sngp_temperature"): better, ("sae", "ensemble_temperature"): better, ("l27", "sngp_temperature"): better})
    # beta has no SNGP cell of sae or l27, as a multilabel dataset runs none, and last's cell there scored no MC dropout.
    values = {
        key: draws for key, draws in full.values.items()
        if not (key[0] == "beta" and key[2] == "sngp" and key[1] in ("sae", "l27"))
        and not (key[:2] == ("beta", "last") and key[3] == "mc_dropout_temperature")
    }
    q2 = question_two(RepeatSeries(values, full.kinds, full.primary), DecisionRules(), "sae", ("sae", "last", "l27"))

    sae = q2["per_arm"]["sae"]
    assert sae["adopted"] == "sngp_temperature"
    assert {record["method"]: record["datasets"] for record in sae["candidates"]}["sngp_temperature"] == ["alpha"]
    assert sae["coverage"]["datasets"] == ["alpha", "beta"]
    assert sae["coverage"]["lacking"] == [
        {"method": "sngp_temperature", "data_name": "beta", "task_type": "binary", "reason": "no sngp cell of sae"},
    ]
    # beta would take the next candidate in the order that improves on sae and has it; the rule is unchanged.
    assert sae["coverage"]["substitutes"] == [{"data_name": "beta", "task_type": "binary", "takes": "ensemble_temperature"}]
    assert q2["adopted"] == "sngp_temperature"
    # On l27 nothing later improves, so beta would take the fallback.
    assert q2["per_arm"]["l27"]["coverage"]["substitutes"] == [{"data_name": "beta", "task_type": "binary", "takes": FALLBACK_ADOPTION}]
    # last's cell on beta is there without MC dropout; last adopts the fallback, which lacks nothing.
    last = q2["per_arm"]["last"]
    assert last["coverage"]["lacking"] == [
        {"method": "mc_dropout_temperature", "data_name": "beta", "task_type": "binary", "reason": "no mc_dropout_temperature scores"},
    ]
    assert (last["adopted"], last["coverage"]["substitutes"]) == (FALLBACK_ADOPTION, [])


def test_the_conformal_band_is_split_conformals_finite_sample_guarantee():
    band = conformal_band(0.9, 19)
    assert (band["rank"], band["marginal"], band["marginal_upper"]) == (18, pytest.approx(0.9), pytest.approx(0.95))
    assert (band["beta_lower"], band["beta_upper"]) == pytest.approx(tuple(beta.ppf([0.025, 0.975], 18, 2)))

    # Exchangeable scores: the rank-18 of 19 calibration scores covers a new one 18 times in 20.
    rng = np.random.default_rng(2)
    scores = rng.normal(size=(20000, 20))  # (trials, n + 1)
    quantiles = np.sort(scores[:, :19], axis=1)[:, band["rank"] - 1]  # (trials,)
    assert np.mean(scores[:, 19] <= quantiles) == pytest.approx(band["marginal"], abs=0.01)

    # Past the calibration size the quantile is infinite and every set holds everything.
    assert conformal_band(0.95, 5)["marginal"] == 1.0
