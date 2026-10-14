"""Where a request stands against the proteins a package was calibrated on, and what the package does about it.

A package's answer rule (`answering.py`) and its correctness model (`correctness.py`) speak for proteins that look like
V_cal. A protein far from every training protein is outside that population, so neither covers it. This module records
where that boundary is, and turns it, the answer rule, and the support behind each claim into one decision per row.

**Novelty.** The indicator is `inpca_d1`, the distance from a row's clamped, standardized input, whitened on the
reference's top principal components, to its nearest reference row, in units of the median such distance among V_fit rows
(`correctness.LatentReference`). Nearest-neighbor distance in a normalized representation is a strong out-of-distribution
score (Sun et al., ICML 2022, "Out-of-distribution detection with deep nearest neighbors"). Its cutoff is the conformal
quantile of V_cal's distances at `NOVELTY_LEVEL` (Bates et al., Annals of Statistics 2023, "Testing for outliers with
conformal p-values"): a new row exchangeable with V_cal lies past it with probability at most `1 - NOVELTY_LEVEL`. It is
a distance in a learned space and never a percent sequence identity, and a row inside the cutoff is not thereby shown to
be in distribution.

**Support.** V_cal's rows are tabulated by novelty and by band of the score the answer rule ranks by, with each cell's
right answers and a Clopper-Pearson upper bound on its error. A cell of fewer than `MINIMUM_STRATUM_ROWS` rows is
`insufficient`: with no wrong answer observed, the 95% upper bound on the error is 3 / n (the rule of three), so thirty
rows can at best bound it near 10%. The package states no rate for an insufficient cell, and answers no row that falls
in one.

**Decision.** `answer_or_abstain_v1` runs in this order. A row whose output is not finite is `not_assessed`. A novel row,
or a row in an insufficient cell, abstains. Benjamini-Hochberg selection at the answer rule's target (Jin and Candes,
arXiv 2210.01408) then runs over the remaining rows of one request together, and a row it does not select abstains.
A package whose V_cal holds fewer than `MINIMUM_RULE_ERRORS` wrong answers cannot support an error target, so it assesses
nothing; nor does a package with no answer rule, or with V_cal truth too partial to calibrate one.

The false-answer-rate statement of conformal selection holds for rows exchangeable with V_cal. The two screens make the
answered rows a subset chosen from features alone, and the p-values still count all of V_cal's wrong answers, so the
statement is not restated for that subset. `metrics.json` reports the error of the rows the policy answers on test, and the
support table reports V_cal's error by stratum. Abstaining says the package will not vouch for the answer; it does not say
the answer is wrong. Selection reads the rows of one request together: a row is answered when its p-value is at most
`target * k / m`, for `k` rows answered among the `m` selection tests, so a request that holds many rows the package is
unsure of is stricter for every row in it. A row answered in a request is answered alone, since the threshold never exceeds
the target and one row is answered when its p-value is at most the target; a row answered alone can abstain in a request.
The target bounds the expected share of wrong answers among one request's answers, a statement a request of one row does not
carry, so `decide_alone` reports what the same rows get one to a request. No policy here escalates a row to a larger base
model: that is a separate routing policy with its own validated release. No percentage or confidence band is defined either;
a display policy must read the support table.

Shapes: `n` rows of one request.
"""

from __future__ import annotations

import hashlib
import json
import math
import numpy as np

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from scipy.stats import beta

from foundry.eval.conformal import conformal_quantile
from .answering import select_answers


APPLICABILITY_SCHEME = "latent_distance_applicability_v1"
POLICY_SCHEME = "answer_or_abstain_v1"
NOVELTY_SIGNAL = "inpca_d1"
NOVELTY_LEVEL = 0.95
# V_cal's top `1 - level` share must hold at least this many rows, or its quantile says little about where novelty begins.
MINIMUM_ROWS_ABOVE_CUTOFF = 5
# Bands of the score the answer rule ranks by, as [0, 0.5), [0.5, 0.8), [0.8, 0.95), [0.95, 1].
BAND_EDGES = (0.5, 0.8, 0.95)
MINIMUM_STRATUM_ROWS = 30
# Fewer wrong V_cal answers than this leave an answer rule's error target unsupported: ten is a reporting screen, not a theorem.
MINIMUM_RULE_ERRORS = 10
ERROR_BOUND_LEVEL = 0.95
SUPPORTED, NOVEL = "supported", "novel"
SCORE_NAMES = ("correctness", "confidence")
MEANING = (
    "The distance from a row's clamped, standardized input, whitened on the reference's top principal components, to its "
    "nearest reference row, in units of the median such distance among V_fit rows. It is a distance in a learned space, "
    "not a percent sequence identity, and a row inside the cutoff is not thereby shown to be in distribution."
)

ANSWER, ABSTAIN, NOT_ASSESSED = "answer", "abstain", "not_assessed"
SELECTION_PASSED = "conformal_selection_passed"
APPLICABILITY_UNASSESSED = "applicability_unassessed"
NOVEL_INPUT = "novel_input"
INSUFFICIENT_STRATUM = "insufficient_stratum_support"
SELECTION_FAILED = "selection_not_passed"
NON_FINITE_OUTPUT = "non_finite_output"
# Why a package says nothing about a row, and the sentence a response gives for it.
NOT_ASSESSED_MESSAGES = {
    "regression_has_no_answer_rule": "A regression package claims no per-row answer confidence, so it has no answer or abstain rule.",
    "partial_calibration_truth": "Calibration truth is only partly observed, so whole-answer confidence and selection are unavailable.",
    "no_answer_rule": "This package was written before the answer rule existed.",
    "no_decision_policy": "This package was written before its decision policy existed.",
    "insufficient_calibration_errors": "Too few calibration answers were wrong to support an error target, so no answer is vouched for.",
    NON_FINITE_OUTPUT: "The head's output for this row is not finite, so nothing about it can be assessed.",
}
# Why a package has no novelty indicator, and the sentence a response gives for it.
APPLICABILITY_UNKNOWN_MESSAGES = {
    "no_latent_reference": "This package ships no latent reference, so it cannot tell whether a row lies far from its training proteins.",
    "package_predates_applicability": "This package was written before its novelty indicator existed.",
    "insufficient_calibration_rows": "V_cal is too small to calibrate a novelty cutoff.",
    "partial_calibration_truth": "Calibration truth is only partly observed, so no latent reference was fitted.",
}
CELL_FIELDS = (
    "applicability", "band", "lower", "upper", "rows", "correct", "accuracy", "error_upper_95", "mean_score", "support",
)


def canonical_sha256(record: Mapping[str, Any]) -> str:
    """The SHA-256 of a record's JSON with sorted keys, so the same record gives the same digest anywhere."""
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def error_upper_bound(errors: int, rows: int) -> float:
    """The Clopper-Pearson upper bound, one-sided at `ERROR_BOUND_LEVEL`, on the error rate of `rows` rows with `errors` wrong."""
    if errors == rows:
        return 1.0
    return float(beta.ppf(ERROR_BOUND_LEVEL, errors + 1, rows - errors))


def band_bounds(edges: Sequence[float]) -> list[tuple[float, float]]:
    """Each band's lower and upper score, the last one closed at 1."""
    bounds = [0.0, *edges, 1.0]
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def band_of(score: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Each row's band index, `(n,)`: the number of edges its score reaches."""
    # score: (n,)
    return np.searchsorted(np.asarray(edges, dtype=np.float64), score, side="right")  # (n,)


def stratum_table(
    status: np.ndarray, score: np.ndarray, correct: np.ndarray, edges: Sequence[float], minimum_rows: int
) -> list[dict[str, Any]]:
    """One cell per novelty status and score band: its rows and right answers, and, once it has support, its rates.

    A cell of fewer than `minimum_rows` rows keeps its counts and states no accuracy, error bound, or mean score.
    """
    # status: (n,) of SUPPORTED or NOVEL; score: (n,); correct: (n,) bool
    band = band_of(score, edges)  # (n,)
    cells = []
    for name in (SUPPORTED, NOVEL):
        for index, (lower, upper) in enumerate(band_bounds(edges)):
            members = (status == name) & (band == index)  # (n,)
            rows = int(members.sum())
            right = int(correct[members].sum())
            sufficient = rows >= minimum_rows
            cells.append({
                "applicability": name, "band": index, "lower": lower, "upper": upper, "rows": rows, "correct": right,
                "accuracy": right / rows if sufficient else None,
                "error_upper_95": error_upper_bound(rows - right, rows) if sufficient else None,
                "mean_score": float(score[members].mean()) if sufficient else None,
                "support": "sufficient" if sufficient else "insufficient",
            })
    return cells


def unavailable_applicability(reason_code: str) -> dict[str, Any]:
    """The record of a package with no novelty indicator, and why."""
    if reason_code not in APPLICABILITY_UNKNOWN_MESSAGES:
        raise ValueError(f"unknown applicability reason {reason_code!r}.")
    return {"scheme": APPLICABILITY_SCHEME, "status": "unavailable", "reason_code": reason_code}


def fit_applicability(
    signal: np.ndarray,
    score: np.ndarray,
    correct: np.ndarray,
    *,
    score_name: str,
    reference_sha256: str,
    level: float = NOVELTY_LEVEL,
    edges: Sequence[float] = BAND_EDGES,
    minimum_rows: int = MINIMUM_STRATUM_ROWS,
) -> dict[str, Any]:
    """The novelty cutoff and support table from V_cal's rows, or an unavailable record when V_cal is too small.

    `signal` is each V_cal row's `NOVELTY_SIGNAL`, `score` the score the answer rule ranks by (`score_name`), and
    `correct` whether its answer is right, `(n,)` each.
    """
    # signal: (n,), score: (n,), correct: (n,)
    signal, score, correct = np.asarray(signal, dtype=np.float64), np.asarray(score, dtype=np.float64), np.asarray(correct, dtype=bool)
    if not signal.shape == score.shape == correct.shape or signal.ndim != 1:
        raise ValueError("V_cal's novelty signal, score, and correctness must be vectors of one length.")
    if score_name not in SCORE_NAMES:
        raise ValueError(f"the support table ranks by {SCORE_NAMES}, not {score_name!r}.")
    if not 0.0 < level < 1.0 or list(edges) != sorted(set(edges)) or not all(0.0 < edge < 1.0 for edge in edges):
        raise ValueError("the novelty level and the band edges must lie strictly between 0 and 1, the edges ascending.")
    if not np.isfinite(signal).all() or not np.isfinite(score).all():
        raise ValueError("V_cal's novelty signal and score must be finite.")

    rows = int(signal.shape[0])
    if rows * (1.0 - level) < MINIMUM_ROWS_ABOVE_CUTOFF:
        return unavailable_applicability("insufficient_calibration_rows")
    cutoff = conformal_quantile(signal, 1.0 - level)
    status = np.where(signal > cutoff, NOVEL, SUPPORTED)  # (n,)
    return {
        "scheme": APPLICABILITY_SCHEME,
        "status": "calibrated",
        "signal": NOVELTY_SIGNAL,
        "meaning": MEANING,
        "level": level,
        "cutoff": cutoff,
        "calibration_rows": rows,
        "novel_share": float((status == NOVEL).mean()),
        "reference_sha256": reference_sha256,
        "strata": {
            "score": score_name, "edges": list(edges), "minimum_rows": minimum_rows,
            "cells": stratum_table(status, score, correct, edges, minimum_rows),
        },
    }


def applicability_status(record: Mapping[str, Any], signal: np.ndarray) -> np.ndarray:
    """Each row's `SUPPORTED` or `NOVEL`, `(n,)`. A row whose distance is not a number cannot be shown supported."""
    # signal: (n,)
    return np.where(signal <= record["cutoff"], SUPPORTED, NOVEL)  # (n,)


def evaluate_applicability(
    record: Mapping[str, Any], signal: np.ndarray, score: np.ndarray, correct: np.ndarray
) -> dict[str, Any]:
    """The share of rows a calibrated record calls novel, and its support table, on rows other than V_cal, such as test."""
    # signal: (n,), score: (n,), correct: (n,)
    if not np.isfinite(score).all():
        raise ValueError("the scores of rows to evaluate must be finite.")
    status = applicability_status(record, signal)  # (n,)
    strata = record["strata"]
    return {
        "rows": int(signal.shape[0]),
        "novel_share": float((status == NOVEL).mean()),
        "cells": stratum_table(status, score, correct, strata["edges"], strata["minimum_rows"]),
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_applicability(record: Any, v_cal_rows: int, score_name: str | None) -> None:
    """Refuse a record that is not one this module writes, or whose counts and rates disagree with each other or with V_cal.

    `score_name` is what the package's answer rule ranks by, or None when it has no rule.
    """
    if not isinstance(record, Mapping) or record.get("scheme") != APPLICABILITY_SCHEME:
        raise ValueError("invalid packaged applicability record")
    if record.get("status") == "unavailable":
        if set(record) != {"scheme", "status", "reason_code"} or record["reason_code"] not in APPLICABILITY_UNKNOWN_MESSAGES:
            raise ValueError("packaged applicability names an unknown reason")
        return

    fields = {"scheme", "status", "signal", "meaning", "level", "cutoff", "calibration_rows", "novel_share", "reference_sha256", "strata"}
    if record.get("status") != "calibrated" or set(record) != fields or record["signal"] != NOVELTY_SIGNAL or record["meaning"] != MEANING:
        raise ValueError("invalid packaged applicability record")
    if record["calibration_rows"] != v_cal_rows or not _is_number(record["level"]) or not 0.0 < record["level"] < 1.0:
        raise ValueError("packaged applicability disagrees with V_cal")
    if not _is_number(record["cutoff"]) or record["cutoff"] < 0.0 or not _is_number(record["novel_share"]) or not 0.0 <= record["novel_share"] <= 1.0:
        raise ValueError("packaged novelty cutoff must be a finite distance")
    if not isinstance(record["reference_sha256"], str) or len(record["reference_sha256"]) != 64:
        raise ValueError("packaged applicability must pin its latent reference")

    strata = record["strata"]
    if not isinstance(strata, Mapping) or set(strata) != {"score", "edges", "minimum_rows", "cells"}:
        raise ValueError("invalid packaged support table")
    if strata["score"] not in SCORE_NAMES or strata["score"] != score_name:
        raise ValueError("packaged support table ranks by a score other than the answer rule's")
    edges, cells = strata["edges"], strata["cells"]
    if not edges or list(edges) != sorted(set(edges)) or not all(_is_number(edge) and 0.0 < edge < 1.0 for edge in edges):
        raise ValueError("packaged support table has invalid band edges")
    expected = [(name, index) for name in (SUPPORTED, NOVEL) for index in range(len(edges) + 1)]
    if [(cell.get("applicability"), cell.get("band")) for cell in cells] != expected:
        raise ValueError("packaged support table must hold one cell per status and band")
    for cell, (lower, upper) in zip(cells, band_bounds(edges) * 2, strict=True):
        if set(cell) != set(CELL_FIELDS) or (cell["lower"], cell["upper"]) != (lower, upper):
            raise ValueError("packaged support table has a malformed cell")
        rows, right = cell["rows"], cell["correct"]
        if not (isinstance(rows, int) and isinstance(right, int) and 0 <= right <= rows):
            raise ValueError("packaged support table has impossible counts")
        sufficient = rows >= strata["minimum_rows"]
        if cell["support"] != ("sufficient" if sufficient else "insufficient"):
            raise ValueError("packaged support table overstates a cell's support")
        rates = (cell["accuracy"], cell["error_upper_95"], cell["mean_score"])
        if not sufficient:
            if any(rate is not None for rate in rates):
                raise ValueError("packaged support table states a rate for a cell without support")
        elif not all(_is_number(rate) for rate in rates) or cell["accuracy"] != right / rows or not math.isclose(
            cell["error_upper_95"], error_upper_bound(rows - right, rows), rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("packaged support table rates disagree with its counts")
    if sum(cell["rows"] for cell in cells) != v_cal_rows:
        raise ValueError("packaged support table does not cover V_cal's rows")
    novel = sum(cell["rows"] for cell in cells if cell["applicability"] == NOVEL)
    if not math.isclose(record["novel_share"], novel / v_cal_rows, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("packaged novel share disagrees with the support table")


def decision_policy(rule: Mapping[str, Any], applicability: Mapping[str, Any]) -> dict[str, Any]:
    """`answer_or_abstain_v1` for one package, with the digest that pins its inputs and, last, its own."""
    calibrated = applicability["status"] == "calibrated"
    policy = {
        "scheme": POLICY_SCHEME,
        "selection": {
            "scheme": rule["scheme"], "score": rule["score"], "target": rule["default_target"],
            "calibration_errors": len(rule["wrong_confidences"]), "minimum_errors": MINIMUM_RULE_ERRORS,
            "answer_rule_sha256": canonical_sha256(rule),
            "pool": "supported_rows_of_one_request" if calibrated else "rows_of_one_request",
        },
        "applicability_sha256": canonical_sha256(applicability),
        "novel": ABSTAIN if calibrated else None,
        "insufficient_stratum": ABSTAIN if calibrated else None,
        "escalation": "none",
    }
    return {**policy, "sha256": canonical_sha256(policy)}


def validate_policy(policy: Any, rule: Mapping[str, Any], applicability: Mapping[str, Any]) -> None:
    """Refuse a policy that is not the one these inputs give."""
    if policy != decision_policy(rule, applicability):
        raise ValueError("packaged decision policy differs from its answer rule and applicability record")


@dataclass(frozen=True)
class Decisions:
    """What a package's policy does with each row of one request."""

    policy_sha256: str | None  # None when nothing was assessed
    actions: tuple[str, ...]  # per row: ANSWER, ABSTAIN, or NOT_ASSESSED
    reasons: tuple[tuple[str, ...], ...]  # per row: sorted reason codes, exactly one for NOT_ASSESSED


def unassessed(count: int, reason_code: str) -> Decisions:
    """A package that assesses nothing: every row `NOT_ASSESSED` for one reason."""
    if reason_code not in NOT_ASSESSED_MESSAGES:
        raise ValueError(f"unknown not-assessed reason {reason_code!r}.")
    return Decisions(None, (NOT_ASSESSED,) * count, ((reason_code,),) * count)


def decide(
    policy: Mapping[str, Any],
    applicability: Mapping[str, Any],
    *,
    status: np.ndarray | None,
    score: np.ndarray,
    p_values: np.ndarray,
) -> Decisions:
    """Answer, abstain from, or leave unassessed each row of one request under `policy`.

    `status` is each row's `SUPPORTED` or `NOVEL`, `None` when the package has no novelty indicator; `score` is the score
    the answer rule ranks by and `p_values` its conformal p-values, `(n,)` each. Rows with a non-finite score or p-value are
    unassessed, since a p-value read from NaN would be the smallest there is.
    """
    # status: (n,) or None, score: (n,), p_values: (n,)
    count = int(p_values.shape[0])
    selection = policy["selection"]
    if selection["calibration_errors"] < selection["minimum_errors"]:
        return unassessed(count, "insufficient_calibration_errors")

    calibrated = applicability["status"] == "calibrated"
    if calibrated and status is None:
        raise ValueError("a package with a novelty cutoff needs each row's novelty status to decide.")
    finite = np.isfinite(score) & np.isfinite(p_values)  # (n,)
    pool = finite.copy()  # (n,) the rows selection may answer
    reasons: list[list[str]] = [[] for _ in range(count)]
    if calibrated:
        strata = applicability["strata"]
        band = band_of(np.where(finite, score, 0.0), strata["edges"])  # (n,)
        thin = np.zeros(count, dtype=bool)  # (n,) rows in a cell V_cal cannot support
        for cell in strata["cells"]:
            if cell["support"] == "insufficient":
                thin |= (status == cell["applicability"]) & (band == cell["band"])
        novel = status == NOVEL  # (n,)
        for row in np.flatnonzero(finite & novel):
            reasons[row].append(NOVEL_INPUT)
        for row in np.flatnonzero(finite & thin):
            reasons[row].append(INSUFFICIENT_STRATUM)
        pool &= ~(novel | thin)

    chosen = np.zeros(count, dtype=bool)  # (n,)
    chosen[pool] = select_answers(p_values[pool], selection["target"])
    for row in np.flatnonzero(pool & ~chosen):
        reasons[row].append(SELECTION_FAILED)

    actions, codes = [], []
    for row in range(count):
        if not finite[row]:
            actions.append(NOT_ASSESSED)
            codes.append((NON_FINITE_OUTPUT,))
        elif chosen[row]:
            actions.append(ANSWER)
            codes.append((SELECTION_PASSED,) if calibrated else (APPLICABILITY_UNASSESSED, SELECTION_PASSED))
        else:
            actions.append(ABSTAIN)
            codes.append(tuple(sorted(reasons[row])))
    return Decisions(policy["sha256"], tuple(actions), tuple(codes))


def decide_alone(
    policy: Mapping[str, Any],
    applicability: Mapping[str, Any],
    *,
    status: np.ndarray | None,
    score: np.ndarray,
    p_values: np.ndarray,
) -> Decisions:
    """What `decide` does with each row when every row is its own request, in the arguments' order.

    One row is answered when its p-value is at most the target, so these answers include every answer the same rows get
    together.
    """
    # status: (n,) or None, score: (n,), p_values: (n,)
    each = [
        decide(
            policy, applicability, status=None if status is None else status[row:row + 1],
            score=score[row:row + 1], p_values=p_values[row:row + 1],
        )
        for row in range(int(p_values.shape[0]))
    ]
    return Decisions(
        each[0].policy_sha256 if each else None, tuple(decisions.actions[0] for decisions in each), tuple(decisions.reasons[0] for decisions in each)
    )


def decision_outcome(decisions: Decisions, correct: np.ndarray) -> dict[str, Any]:
    """What a policy did on rows whose answers are known to be right or wrong: counts, reasons, and the error it let through."""
    # correct: (n,) bool
    actions = np.asarray(decisions.actions)  # (n,)
    answered = actions == ANSWER  # (n,)
    abstained = actions == ABSTAIN  # (n,)
    rows = int(actions.shape[0])
    reasons: dict[str, int] = {}
    for codes in decisions.reasons:
        for code in codes:
            reasons[code] = reasons.get(code, 0) + 1
    return {
        "rows": rows,
        "answered": int(answered.sum()),
        "abstained": int(abstained.sum()),
        "not_assessed": int((actions == NOT_ASSESSED).sum()),
        "answered_error": float(1.0 - correct[answered].mean()) if answered.any() else None,
        "abstained_error": float(1.0 - correct[abstained].mean()) if abstained.any() else None,
        "reasons": dict(sorted(reasons.items())),
    }
