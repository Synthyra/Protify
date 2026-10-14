"""The novelty cutoff, the support table, and the answer-or-abstain policy of a package."""

from __future__ import annotations

import copy
import json
import numpy as np
import pytest

from scipy.stats import beta

from foundry.eval.conformal import conformal_quantile
from protify.uncertainty.answering import conformal_p_values, scored_rule_record, select_answers
from protify.uncertainty.applicability import (
    ABSTAIN,
    ANSWER,
    APPLICABILITY_UNASSESSED,
    BAND_EDGES,
    INSUFFICIENT_STRATUM,
    MINIMUM_RULE_ERRORS,
    NON_FINITE_OUTPUT,
    NOT_ASSESSED,
    NOVEL,
    NOVEL_INPUT,
    SELECTION_FAILED,
    SELECTION_PASSED,
    SUPPORTED,
    Decisions,
    applicability_status,
    band_of,
    canonical_sha256,
    decide,
    decide_alone,
    decision_outcome,
    decision_policy,
    error_upper_bound,
    evaluate_applicability,
    fit_applicability,
    stratum_table,
    unassessed,
    unavailable_applicability,
    validate_applicability,
    validate_policy,
)
from protify.uncertainty.packaged import Prediction, PackagedHead, check_decision_records


REFERENCE = "a" * 64


def _calibration_rows(count: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V_cal rows whose reliability tracks their correctness, so every band and both statuses hold rows."""
    rng = np.random.default_rng(seed)
    signal = rng.gamma(shape=6.0, scale=0.2, size=count)  # (count,), distances near 1.2
    score = rng.uniform(0.0, 1.0, size=count)  # (count,)
    correct = rng.uniform(size=count) < score  # (count,) a row is right with its own reliability
    return signal, score, correct  # (count,), (count,), (count,)


def _fitted(count: int = 4000, seed: int = 0) -> dict:
    signal, score, correct = _calibration_rows(count, seed)
    return fit_applicability(signal, score, correct, score_name="correctness", reference_sha256=REFERENCE)


def _rule(score: np.ndarray, correct: np.ndarray) -> dict:
    # score: (n,), correct: (n,)
    return scored_rule_record(score, correct, "binary", None, "correctness")


def test_the_error_bound_is_the_clopper_pearson_upper_limit():
    assert error_upper_bound(0, 30) == pytest.approx(1.0 - 0.05 ** (1.0 / 30.0))
    assert error_upper_bound(0, 30) == pytest.approx(3.0 / 30.0, abs=0.006)  # the rule of three
    assert error_upper_bound(5, 40) == pytest.approx(beta.ppf(0.95, 6, 35))
    assert error_upper_bound(12, 12) == 1.0
    assert error_upper_bound(2, 100) > error_upper_bound(1, 100) > error_upper_bound(0, 100)
    assert error_upper_bound(3, 60) < error_upper_bound(3, 30)


def test_a_score_on_a_band_edge_belongs_to_the_band_above_and_one_belongs_to_the_last():
    scores = np.array([0.0, 0.4999, 0.5, 0.79, 0.8, 0.95, 1.0])  # (7,)
    assert band_of(scores, BAND_EDGES).tolist() == [0, 0, 1, 1, 2, 3, 3]


def test_the_cutoff_is_the_conformal_quantile_and_novelty_matches_it():
    signal = np.arange(200.0)  # (200,)
    score, correct = np.full(200, 0.9), np.ones(200, dtype=bool)
    record = fit_applicability(signal, score, correct, score_name="correctness", reference_sha256=REFERENCE)

    assert record["cutoff"] == conformal_quantile(signal, 0.05) == 190.0
    assert record["novel_share"] == pytest.approx(9 / 200)
    assert record["calibration_rows"] == 200
    status = applicability_status(record, np.array([0.0, 190.0, 190.5, 500.0, np.nan]))  # (5,)
    assert status.tolist() == [SUPPORTED, SUPPORTED, NOVEL, NOVEL, NOVEL]


def test_a_fresh_row_lies_past_the_cutoff_no_more_often_than_one_minus_the_level():
    calibration = np.random.default_rng(3).gamma(6.0, 0.2, size=400)  # (400,)
    record = fit_applicability(
        calibration, np.full(400, 0.9), np.ones(400, dtype=bool), score_name="confidence", reference_sha256=REFERENCE
    )
    fresh = np.random.default_rng(4).gamma(6.0, 0.2, size=200_000)  # (200000,) exchangeable with V_cal
    share = float((applicability_status(record, fresh) == NOVEL).mean())
    assert share <= 0.05 + 0.005


def test_a_small_v_cal_calibrates_no_cutoff_and_says_why():
    signal, score, correct = _calibration_rows(99)
    record = fit_applicability(signal, score, correct, score_name="correctness", reference_sha256=REFERENCE)
    assert record == unavailable_applicability("insufficient_calibration_rows")
    assert fit_applicability(*_calibration_rows(100), score_name="correctness", reference_sha256=REFERENCE)["status"] == "calibrated"


@pytest.mark.parametrize("change", ["length", "matrix", "nan_signal", "inf_score", "score_name", "level", "edges", "edge_order"])
def test_fitting_refuses_rows_it_cannot_read(change):
    signal, score, correct = _calibration_rows(500)
    options = {"score_name": "correctness", "reference_sha256": REFERENCE}
    if change == "length":
        correct = correct[:-1]
    elif change == "matrix":
        signal = signal[:, None]
    elif change == "nan_signal":
        signal[3] = np.nan
    elif change == "inf_score":
        score[3] = np.inf
    elif change == "score_name":
        options["score_name"] = "accuracy"
    elif change == "level":
        options["level"] = 1.0
    elif change == "edges":
        options["edges"] = (0.5, 1.0)
    else:
        options["edges"] = (0.8, 0.5)
    with pytest.raises(ValueError):
        fit_applicability(signal, score, correct, **options)


def test_the_support_table_covers_every_row_and_states_rates_only_where_it_has_support():
    record = _fitted()
    cells = record["strata"]["cells"]

    assert sum(cell["rows"] for cell in cells) == record["calibration_rows"]
    assert [(cell["applicability"], cell["band"]) for cell in cells] == [(name, band) for name in (SUPPORTED, NOVEL) for band in range(4)]
    for cell in cells:
        if cell["rows"] >= 30:
            assert cell["support"] == "sufficient"
            assert cell["accuracy"] == cell["correct"] / cell["rows"]
            assert cell["error_upper_95"] == pytest.approx(error_upper_bound(cell["rows"] - cell["correct"], cell["rows"]))
            assert 0.0 <= cell["mean_score"] <= 1.0
        else:
            assert cell["support"] == "insufficient"
            assert (cell["accuracy"], cell["error_upper_95"], cell["mean_score"]) == (None, None, None)
    assert any(cell["support"] == "insufficient" for cell in _fitted(count=300)["strata"]["cells"])


def test_a_higher_band_is_more_accurate_when_the_score_is_reliable():
    supported = [cell for cell in _fitted(count=20_000)["strata"]["cells"] if cell["applicability"] == SUPPORTED]
    accuracy = [cell["accuracy"] for cell in supported]
    assert accuracy == sorted(accuracy)
    for cell in supported:
        assert cell["lower"] - 0.03 <= cell["accuracy"] <= cell["upper"] + 0.03


def test_evaluation_on_other_rows_uses_the_recorded_cutoff_and_bands():
    record = _fitted()
    signal, score, correct = _calibration_rows(3000, seed=11)
    evaluated = evaluate_applicability(record, signal, score, correct)

    assert evaluated["rows"] == 3000
    assert evaluated["novel_share"] == pytest.approx(float((signal > record["cutoff"]).mean()))
    assert sum(cell["rows"] for cell in evaluated["cells"]) == 3000
    with pytest.raises(ValueError, match="finite"):
        evaluate_applicability(record, signal, np.where(np.arange(3000) == 0, np.nan, score), correct)


def test_a_stored_record_passes_and_survives_json():
    record = _fitted()
    validate_applicability(record, 4000, "correctness")
    validate_applicability(json.loads(json.dumps(record)), 4000, "correctness")
    validate_applicability(unavailable_applicability("no_latent_reference"), 4000, None)


def _cell(record: dict, status: str, band: int) -> dict:
    return next(cell for cell in record["strata"]["cells"] if (cell["applicability"], cell["band"]) == (status, band))


@pytest.mark.parametrize(
    "change",
    [
        "scheme", "reason", "extra_field", "rows", "score_name", "reference", "signal", "meaning", "cutoff", "novel_share",
        "support_flag", "rate_without_support", "rate_off", "bound_off", "count_off", "cell_missing", "cell_order", "edges",
    ],
)
def test_a_tampered_record_is_refused(change):
    record = copy.deepcopy(_fitted(count=300))
    thin = next(cell for cell in record["strata"]["cells"] if cell["support"] == "insufficient")
    thick = next(cell for cell in record["strata"]["cells"] if cell["support"] == "sufficient")
    rows, score_name = 300, "correctness"
    if change == "scheme":
        record["scheme"] = "other"
    elif change == "reason":
        record = {**unavailable_applicability("no_latent_reference"), "reason_code": "because"}
    elif change == "extra_field":
        record["threshold"] = 0.5
    elif change == "rows":
        rows = 301
    elif change == "score_name":
        score_name = "confidence"
    elif change == "reference":
        record["reference_sha256"] = "abc"
    elif change == "signal":
        record["signal"] = "trunk_d1"
    elif change == "meaning":
        record["meaning"] = "percent identity"
    elif change == "cutoff":
        record["cutoff"] = -1.0
    elif change == "novel_share":
        record["novel_share"] = 0.5
    elif change == "support_flag":
        thin["support"] = "sufficient"
    elif change == "rate_without_support":
        thin["accuracy"] = 0.99
    elif change == "rate_off":
        thick["accuracy"] = 0.123456
    elif change == "bound_off":
        thick["error_upper_95"] = 0.0
    elif change == "count_off":
        thick["correct"] = thick["rows"] + 1
    elif change == "cell_missing":
        record["strata"]["cells"].pop()
    elif change == "cell_order":
        record["strata"]["cells"].reverse()
    else:
        record["strata"]["edges"] = [0.5, 0.8, 0.9]
    with pytest.raises(ValueError):
        validate_applicability(record, rows, score_name)


def test_no_number_in_a_record_is_a_bare_boolean_or_non_finite():
    record = _fitted()
    for name in ("cutoff", "level", "novel_share"):
        for bad in (True, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                validate_applicability({**record, name: bad}, 4000, "correctness")


def test_the_policy_digest_pins_the_rule_and_the_record_and_survives_key_order():
    signal, score, correct = _calibration_rows(4000)
    record = _fitted()
    rule = _rule(score, correct)
    policy = decision_policy(rule, record)

    assert decision_policy(json.loads(json.dumps(rule)), json.loads(json.dumps(record))) == policy
    assert policy["sha256"] == canonical_sha256({key: value for key, value in policy.items() if key != "sha256"})
    assert policy["selection"]["calibration_errors"] == int((~correct).sum())
    assert policy["escalation"] == "none"
    validate_policy(policy, rule, record)

    other_rule = {**rule, "default_target": rule["default_target"] / 2}
    other_record = {**record, "cutoff": record["cutoff"] + 0.1}
    assert decision_policy(other_rule, record)["sha256"] != policy["sha256"]
    assert decision_policy(rule, other_record)["sha256"] != policy["sha256"]
    for stale, stale_rule, stale_record in (
        (policy, other_rule, record), (policy, rule, other_record),
        ({**policy, "escalation": "larger_base"}, rule, record), ({**policy, "sha256": "0" * 64}, rule, record),
    ):
        with pytest.raises(ValueError, match="differs"):
            validate_policy(stale, stale_rule, stale_record)


def test_the_policy_without_a_novelty_cutoff_selects_among_every_row():
    signal, score, correct = _calibration_rows(4000)
    policy = decision_policy(_rule(score, correct), unavailable_applicability("no_latent_reference"))
    assert policy["selection"]["pool"] == "rows_of_one_request"
    assert policy["novel"] is None and policy["insufficient_stratum"] is None
    assert decision_policy(_rule(score, correct), _fitted())["selection"]["pool"] == "supported_rows_of_one_request"


def _policy_fixture(count: int = 4000) -> tuple[dict, dict, dict]:
    signal, score, correct = _calibration_rows(count)
    record = fit_applicability(signal, score, correct, score_name="correctness", reference_sha256=REFERENCE)
    rule = _rule(score, correct)
    return rule, record, decision_policy(rule, record)


def _decide(rule, record, policy, status, score) -> Decisions:
    return decide(policy, record, status=status, score=score, p_values=conformal_p_values(score, rule))


def test_screened_rows_abstain_and_only_the_rest_are_selected():
    rule, record, policy = _policy_fixture()
    thin_cell = next(cell for cell in record["strata"]["cells"] if cell["support"] == "insufficient")
    assert thin_cell["applicability"] == NOVEL, "the fixture's novel cells hold the fewest rows"

    thin_score = (thin_cell["lower"] + thin_cell["upper"]) / 2.0
    score = np.array([0.99, 0.99, 0.99, thin_score, 0.01, 0.99])  # (6,)
    status = np.array([SUPPORTED, NOVEL, SUPPORTED, NOVEL, SUPPORTED, SUPPORTED])  # (6,)
    decisions = _decide(rule, record, policy, status, score)

    assert decisions.policy_sha256 == policy["sha256"]
    assert decisions.actions == (ANSWER, ABSTAIN, ANSWER, ABSTAIN, ABSTAIN, ANSWER)
    assert decisions.reasons[0] == decisions.reasons[2] == decisions.reasons[5] == (SELECTION_PASSED,)
    assert NOVEL_INPUT in decisions.reasons[1] and NOVEL_INPUT in decisions.reasons[3]
    assert INSUFFICIENT_STRATUM in decisions.reasons[3]
    assert decisions.reasons[4] == (SELECTION_FAILED,)


def test_a_supported_row_in_a_cell_v_cal_cannot_support_abstains_for_that_alone():
    rule, record, policy = _policy_fixture(count=300)
    thin = [cell for cell in record["strata"]["cells"] if cell["applicability"] == SUPPORTED and cell["support"] == "insufficient"]
    assert [cell["band"] for cell in thin] == [3]

    score = np.array([0.99, 0.6])  # (2,) the first falls in the thin band, the second in a supported one
    decisions = _decide(rule, record, policy, np.array([SUPPORTED, SUPPORTED]), score)
    assert decisions.actions[0] == ABSTAIN and decisions.reasons[0] == (INSUFFICIENT_STRATUM,)


def test_excluded_rows_do_not_change_which_of_the_rest_are_answered():
    rule, record, policy = _policy_fixture()
    rng = np.random.default_rng(5)
    score = rng.uniform(0.3, 1.0, size=300)  # (300,)
    status = np.where(rng.uniform(size=300) < 0.2, NOVEL, SUPPORTED)  # (300,)
    decisions = _decide(rule, record, policy, status, score)

    thin = {(cell["applicability"], cell["band"]) for cell in record["strata"]["cells"] if cell["support"] == "insufficient"}
    bands = band_of(score, record["strata"]["edges"])  # (300,)
    pool = np.array([name == SUPPORTED and (name, band) not in thin for name, band in zip(status, bands, strict=True)])  # (300,)
    expected = select_answers(conformal_p_values(score[pool], rule), rule["default_target"])
    actions = np.asarray(decisions.actions)  # (300,)
    assert (actions[pool] == ANSWER).tolist() == expected.tolist()
    assert (actions[~pool] == ABSTAIN).all()
    assert pool.sum() < 300 and expected.any()


def test_a_row_answered_in_a_request_is_answered_alone_and_not_the_reverse():
    rule, record, policy = _policy_fixture()
    rng = np.random.default_rng(9)
    score = rng.uniform(0.2, 1.0, size=400)  # (400,)
    status = np.where(rng.uniform(size=400) < 0.2, NOVEL, SUPPORTED)  # (400,)
    p_values = conformal_p_values(score, rule)  # (400,)
    together = decide(policy, record, status=status, score=score, p_values=p_values)
    alone = decide_alone(policy, record, status=status, score=score, p_values=p_values)

    answered_together = np.asarray(together.actions) == ANSWER  # (400,)
    answered_alone = np.asarray(alone.actions) == ANSWER  # (400,)
    assert alone.policy_sha256 == together.policy_sha256 == policy["sha256"]
    assert (answered_alone | ~answered_together).all()
    assert answered_alone.sum() > answered_together.sum() > 0
    for row in np.flatnonzero(answered_alone & ~answered_together):
        assert together.reasons[row] == (SELECTION_FAILED,), "only selection separates a request from one row"
    for row in np.flatnonzero(~answered_alone):
        assert alone.reasons[row] == together.reasons[row], "a screened or unselected row reads the same either way"


def test_rows_the_package_is_unsure_of_tighten_the_selection_of_the_others():
    rule, record, policy = _policy_fixture()
    target = policy["selection"]["target"]
    confident, borderline = target / 1000.0, 0.9 * target

    def decided(p_values: np.ndarray) -> Decisions:
        # p_values: (n,)
        return decide(policy, record, status=np.full(p_values.shape[0], SUPPORTED), score=np.full(p_values.shape[0], 0.6), p_values=p_values)

    beside_unsure = np.concatenate([np.full(50, confident), [borderline], np.full(49, 0.9)])  # (100,)
    beside_confident = np.concatenate([np.full(99, confident), [borderline]])  # (100,)
    assert decided(beside_unsure).actions[50] == ABSTAIN
    assert decided(beside_confident).actions[99] == ANSWER
    assert decided(np.array([borderline])).actions == (ANSWER,)
    assert decided(np.array([1.1 * target])).actions == (ABSTAIN,)
    assert decided(np.concatenate([np.full(99, confident), [1.1 * target]])).actions[99] == ABSTAIN, "a row past the target is never answered"


def test_deciding_one_row_at_a_time_keeps_the_screens_and_the_unassessed_reasons():
    rule, record, policy = _policy_fixture()
    score = np.array([0.99, np.nan, 0.99])  # (3,)
    p_values = np.array([0.001, np.nan, 0.001])  # (3,)
    status = np.array([SUPPORTED, SUPPORTED, NOVEL])  # (3,)
    alone = decide_alone(policy, record, status=status, score=score, p_values=p_values)
    assert alone.actions == (ANSWER, NOT_ASSESSED, ABSTAIN)
    assert alone.reasons[:2] == ((SELECTION_PASSED,), (NON_FINITE_OUTPUT,)) and NOVEL_INPUT in alone.reasons[2]
    assert decide_alone(policy, record, status=status[:0], score=score[:0], p_values=p_values[:0]) == Decisions(None, (), ())

    _, calibration_score, correct = _calibration_rows(2000)
    correct[:] = True
    correct[: MINIMUM_RULE_ERRORS - 1] = False
    thin_rule = _rule(calibration_score, correct)
    thin_policy = decision_policy(thin_rule, record)
    assert decide_alone(thin_policy, record, status=status, score=score, p_values=p_values) == unassessed(3, "insufficient_calibration_errors")


def test_every_abstention_and_answer_names_its_reason():
    rule, record, policy = _policy_fixture()
    rng = np.random.default_rng(6)
    score = rng.uniform(size=500)  # (500,)
    status = np.where(rng.uniform(size=500) < 0.3, NOVEL, SUPPORTED)  # (500,)
    decisions = _decide(rule, record, policy, status, score)

    assert len(decisions.actions) == len(decisions.reasons) == 500
    for action, codes in zip(decisions.actions, decisions.reasons, strict=True):
        assert codes and len(set(codes)) == len(codes) and list(codes) == sorted(codes)
        assert (action == ANSWER) == (SELECTION_PASSED in codes)
        if action == ABSTAIN:
            assert set(codes) <= {NOVEL_INPUT, INSUFFICIENT_STRATUM, SELECTION_FAILED}
    assert {ANSWER, ABSTAIN} <= set(decisions.actions)


def test_non_finite_rows_are_unassessed_and_never_select_anything():
    rule, record, policy = _policy_fixture()
    score = np.array([0.99, np.nan, 0.99, 0.99])  # (4,)
    p_values = np.array([0.001, np.nan, 0.001, np.inf])  # (4,)
    status = np.array([SUPPORTED, SUPPORTED, NOVEL, SUPPORTED])  # (4,)
    decisions = decide(policy, record, status=status, score=score, p_values=p_values)

    assert decisions.actions[1] == NOT_ASSESSED and decisions.reasons[1] == (NON_FINITE_OUTPUT,)
    assert decisions.actions[3] == NOT_ASSESSED
    assert decisions.actions[2] == ABSTAIN
    assert decisions.actions[0] == ANSWER


def test_a_package_without_a_cutoff_answers_by_selection_alone_and_says_so():
    signal, score, correct = _calibration_rows(4000)
    rule = _rule(score, correct)
    record = unavailable_applicability("no_latent_reference")
    policy = decision_policy(rule, record)
    fresh = np.array([0.99, 0.01, 0.99])  # (3,)
    decisions = decide(policy, record, status=None, score=fresh, p_values=conformal_p_values(fresh, rule))

    assert decisions.actions == (ANSWER, ABSTAIN, ANSWER)
    assert decisions.reasons[0] == (APPLICABILITY_UNASSESSED, SELECTION_PASSED)
    assert decisions.reasons[1] == (SELECTION_FAILED,)


def test_a_cutoff_needs_each_rows_status():
    rule, record, policy = _policy_fixture()
    with pytest.raises(ValueError, match="novelty status"):
        decide(policy, record, status=None, score=np.array([0.9]), p_values=np.array([0.01]))


def test_too_few_wrong_calibration_answers_leave_every_row_unassessed():
    rng = np.random.default_rng(8)
    score = rng.uniform(0.9, 1.0, size=2000)  # (2000,)
    correct = np.ones(2000, dtype=bool)
    correct[: MINIMUM_RULE_ERRORS - 1] = False
    rule = _rule(score, correct)
    record = fit_applicability(rng.gamma(6.0, 0.2, 2000), score, correct, score_name="correctness", reference_sha256=REFERENCE)
    policy = decision_policy(rule, record)

    fresh = np.array([0.99, 0.5])  # (2,)
    decisions = decide(policy, record, status=np.array([SUPPORTED, NOVEL]), score=fresh, p_values=conformal_p_values(fresh, rule))
    assert decisions == unassessed(2, "insufficient_calibration_errors")
    assert decisions.policy_sha256 is None

    correct[MINIMUM_RULE_ERRORS - 1] = False
    enough = decision_policy(_rule(score, correct), record)
    assert decide(enough, record, status=np.array([SUPPORTED]), score=np.array([0.99]), p_values=np.array([0.0001])).actions == (ANSWER,)


def test_a_package_with_no_errors_cannot_be_asked_to_select():
    score, correct = np.full(500, 0.99), np.ones(500, dtype=bool)
    rule = _rule(score, correct)
    assert rule["default_target"] == 0.0
    policy = decision_policy(rule, unavailable_applicability("no_latent_reference"))
    assert decide(policy, unavailable_applicability("no_latent_reference"), status=None, score=np.array([0.99]),
                  p_values=np.array([0.01])) == unassessed(1, "insufficient_calibration_errors")


def test_unassessed_names_only_reasons_the_responses_can_explain():
    assert unassessed(2, "regression_has_no_answer_rule").actions == (NOT_ASSESSED, NOT_ASSESSED)
    with pytest.raises(ValueError, match="unknown"):
        unassessed(1, "because")
    with pytest.raises(ValueError, match="unknown"):
        unavailable_applicability("because")


def test_the_outcome_reports_answered_error_and_every_reason_count():
    decisions = Decisions(
        "f" * 64, (ANSWER, ANSWER, ABSTAIN, ABSTAIN, NOT_ASSESSED),
        ((SELECTION_PASSED,), (SELECTION_PASSED,), (NOVEL_INPUT, INSUFFICIENT_STRATUM), (SELECTION_FAILED,), (NON_FINITE_OUTPUT,)),
    )
    outcome = decision_outcome(decisions, np.array([True, False, False, True, True]))

    assert outcome == {
        "rows": 5, "answered": 2, "abstained": 2, "not_assessed": 1, "answered_error": 0.5, "abstained_error": 0.5,
        "reasons": {INSUFFICIENT_STRATUM: 1, NON_FINITE_OUTPUT: 1, NOVEL_INPUT: 1, SELECTION_FAILED: 1, SELECTION_PASSED: 2},
    }
    nothing = decision_outcome(unassessed(3, "no_answer_rule"), np.ones(3, dtype=bool))
    assert nothing["answered_error"] is None and nothing["abstained_error"] is None and nothing["not_assessed"] == 3


def test_the_table_of_a_set_of_rows_counts_each_row_once():
    status = np.array([SUPPORTED, SUPPORTED, NOVEL, NOVEL, SUPPORTED])  # (5,)
    score = np.array([0.1, 0.6, 0.9, 0.99, 0.6])  # (5,)
    correct = np.array([False, True, True, False, True])  # (5,)
    cells = stratum_table(status, score, correct, BAND_EDGES, minimum_rows=2)

    by_cell = {(cell["applicability"], cell["band"]): cell for cell in cells}
    assert sum(cell["rows"] for cell in cells) == 5
    assert by_cell[(SUPPORTED, 1)]["rows"] == 2 and by_cell[(SUPPORTED, 1)]["support"] == "sufficient"
    assert by_cell[(SUPPORTED, 1)]["accuracy"] == 1.0 and by_cell[(SUPPORTED, 1)]["mean_score"] == pytest.approx(0.6)
    assert by_cell[(SUPPORTED, 0)]["support"] == "insufficient" and by_cell[(SUPPORTED, 0)]["accuracy"] is None
    assert by_cell[(NOVEL, 3)]["rows"] == 1 and by_cell[(NOVEL, 3)]["correct"] == 0


def _head(calibration: dict) -> PackagedHead:
    return PackagedHead("package", calibration, {}, {}, None)


def _package_calibration(*, with_cutoff: bool = True) -> dict:
    signal, score, correct = _calibration_rows(4000)
    record = _fitted() if with_cutoff else unavailable_applicability("no_latent_reference")
    rule = _rule(score, correct)
    return {
        "task_type": "binary", "answer_rule": rule, "applicability": record, "decision_policy": decision_policy(rule, record),
        "rows": {"v_fit": 4000, "v_cal": 4000},
    }


def _prediction(count: int, *, reliability: float = 0.99, signal: float = 1.0) -> Prediction:
    rule = _package_calibration()["answer_rule"]
    scores = np.full(count, reliability)
    return Prediction(
        probs=np.full((count, 2), 0.5), confidence=np.full(count, 0.9), p_values=conformal_p_values(scores, rule),
        reliability=scores, signals={"inpca_d1": np.full(count, signal)},
    )


def test_a_package_decides_a_request_from_its_own_record():
    head = _head(_package_calibration())
    prediction = _prediction(3)
    prediction.signals["inpca_d1"][:] = [1.0, 1e6, 1.0]

    assert head.novelty(prediction).tolist() == [SUPPORTED, NOVEL, SUPPORTED]
    decisions = head.decide(prediction)
    assert decisions.policy_sha256 == head.decision_policy["sha256"]
    assert decisions.actions == (ANSWER, ABSTAIN, ANSWER)
    assert NOVEL_INPUT in decisions.reasons[1]


def test_a_package_reads_the_score_its_rule_ranks_by():
    calibration = _package_calibration(with_cutoff=False)
    calibration["answer_rule"] = {**calibration["answer_rule"], "score": "confidence"}
    calibration["decision_policy"] = decision_policy(calibration["answer_rule"], calibration["applicability"])
    head = _head(calibration)
    prediction = Prediction(
        probs=np.full((2, 2), 0.5), confidence=np.array([0.99, 0.01]), reliability=np.array([0.01, 0.99]),
        p_values=conformal_p_values(np.array([0.99, 0.01]), calibration["answer_rule"]),
    )
    assert head.decide(prediction).actions == (ANSWER, ABSTAIN)


def test_a_package_without_a_cutoff_gives_no_novelty_and_still_decides():
    head = _head(_package_calibration(with_cutoff=False))
    prediction = _prediction(2)
    assert head.novelty(prediction) is None
    assert head.decide(prediction).reasons[0] == (APPLICABILITY_UNASSESSED, SELECTION_PASSED)


def test_a_cutoff_without_the_latent_signals_is_a_hard_error():
    head = _head(_package_calibration())
    with pytest.raises(ValueError, match="latent signals"):
        head.novelty(Prediction(probs=np.full((1, 2), 0.5)))


@pytest.mark.parametrize(
    ("task_type", "observation", "policy", "reason"),
    [
        ("regression", None, True, "regression_has_no_answer_rule"),
        ("binary", {"whole_answer": "unavailable_partial_calibration"}, True, "partial_calibration_truth"),
        ("binary", None, "no_rule", "no_answer_rule"),
        ("binary", None, "no_policy", "no_decision_policy"),
    ],
)
def test_a_package_that_cannot_decide_leaves_every_row_unassessed(task_type, observation, policy, reason):
    calibration = _package_calibration()
    calibration["task_type"] = task_type
    if observation is not None:
        calibration["observation_contract"] = observation
    if policy == "no_rule":
        calibration.update(answer_rule=None, decision_policy=None)
    elif policy == "no_policy":
        calibration["decision_policy"] = None
    prediction = _prediction(4) if task_type != "regression" else Prediction(mean=np.zeros(4))
    decisions = _head(calibration).decide(prediction)

    assert decisions == unassessed(4, reason)
    assert decisions.policy_sha256 is None


def test_a_package_from_before_the_novelty_indicator_says_so():
    calibration = _package_calibration()
    for name in ("applicability", "decision_policy"):
        calibration.pop(name)
    head = _head(calibration)

    assert head.applicability == unavailable_applicability("package_predates_applicability")
    assert head.novelty(_prediction(2)) is None
    assert head.decide(_prediction(2)) == unassessed(2, "no_decision_policy")
    check_decision_records(calibration, None)


class _Reference:
    def __init__(self, digest: str) -> None:
        self.digest = digest

    def content_sha256(self) -> str:
        return self.digest


class _Correctness:
    def __init__(self, digest: str) -> None:
        self.reference = _Reference(digest)


@pytest.mark.parametrize(
    "change",
    ["good", "policy_alone", "record_alone", "wrong_reference", "no_reference", "reference_without_cutoff", "cutoff_without_rule",
     "stale_policy", "stored_predates", "wrong_score"],
)
def test_a_package_is_refused_when_its_records_disagree(change):
    calibration = _package_calibration()
    correctness = _Correctness(REFERENCE)
    if change == "policy_alone":
        calibration.pop("applicability")
    elif change == "record_alone":
        calibration.pop("decision_policy")
    elif change == "wrong_reference":
        correctness = _Correctness("b" * 64)
    elif change == "no_reference":
        correctness = None
    elif change == "reference_without_cutoff":
        calibration["applicability"] = unavailable_applicability("no_latent_reference")
        calibration["decision_policy"] = decision_policy(calibration["answer_rule"], calibration["applicability"])
    elif change == "cutoff_without_rule":
        calibration.update(answer_rule=None, decision_policy=None)
    elif change == "stale_policy":
        calibration["decision_policy"] = {**calibration["decision_policy"], "escalation": "larger_base"}
    elif change == "stored_predates":
        calibration["applicability"] = unavailable_applicability("package_predates_applicability")
        calibration["decision_policy"] = decision_policy(calibration["answer_rule"], calibration["applicability"])
    elif change == "wrong_score":
        calibration["answer_rule"] = {**calibration["answer_rule"], "score": "confidence"}

    if change == "good":
        check_decision_records(calibration, correctness)
        return
    with pytest.raises(ValueError):
        check_decision_records(calibration, correctness)


def test_a_package_without_a_reference_may_only_say_it_has_none():
    calibration = _package_calibration(with_cutoff=False)
    check_decision_records(calibration, None)
    with pytest.raises(ValueError, match="contradicts"):
        check_decision_records(calibration, _Correctness(REFERENCE))

    small = {**calibration, "applicability": unavailable_applicability("insufficient_calibration_rows")}
    small["decision_policy"] = decision_policy(small["answer_rule"], small["applicability"])
    check_decision_records(small, _Correctness(REFERENCE))
    with pytest.raises(ValueError, match="contradicts"):
        check_decision_records(small, None)


def test_a_regression_package_records_no_reference_and_no_policy():
    calibration = {
        "task_type": "regression", "answer_rule": None, "rows": {"v_fit": 100, "v_cal": 100},
        "applicability": unavailable_applicability("no_latent_reference"), "decision_policy": None,
    }
    check_decision_records(calibration, None)
    with pytest.raises(ValueError, match="exactly when"):
        check_decision_records({**calibration, "decision_policy": {"scheme": "answer_or_abstain_v1"}}, None)


@pytest.mark.parametrize("kind, predictive, method, fixed", [
    ("regression", "head", "plain", True),
    ("regression", "head", "temperature", True),
    ("regression", "head", "sngp", False),
    ("regression", "head", "sngp_temperature", False),
    ("regression", "mc_dropout", "mc_dropout", False),
    ("regression", "mc_dropout", "mc_dropout_temperature", False),
    ("regression", "ensemble", "ensemble", False),
    ("regression", "ensemble", "sngp_ensemble_temperature", False),
    ("binary", "head", "temperature", False),
])
def test_only_a_regression_package_without_a_spread_of_its_own_has_a_fixed_width_interval(kind, predictive, method, fixed):
    head = _head({"task_type": kind, "predictive": predictive, "method": method})
    assert head.fixed_width_interval is fixed
