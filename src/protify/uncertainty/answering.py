"""The answer a package gives each row, how confident it is that the whole answer is right, and when to answer.

**The answer.** Binary and multiclass answer the top class. Multilabel answers the label set of
every label at 0.5 or above, the most probable set when labels are independent. When every V_fit
and V_cal row carries at least one label, as every enzyme in EC carries an EC number, the empty set
is never right. A row whose every label falls below 0.5 then answers its single most probable
label: adding label k to a set multiplies its probability by p_k / (1 - p_k), which is below 1 for
every such label, so the top label alone is the most probable non-empty set.

**The confidence** is the calibrated probability that the answer is right: the top probability,
or the product over labels of each chosen side's probability. Under the non-empty rule, the set's
probability is divided by one minus the empty set's, its probability given that the set is not
empty. Without that rule, EC's head ranked empty sets, all wrong, among its most confident
answers (result 2026-09-25_calibration_logistic_uncertainty).

**When to answer: conformal selection** (Jin and Candes, arXiv 2210.01408). Each row's p-value
against "this answer is wrong" is `(1 + #{V_cal rows answered wrongly with confidence >= c}) /
(n + 1)`, from V_cal's `n` rows. Benjamini-Hochberg at a target over the rows asked about together
answers the rows whose p-values pass. When V_cal and those rows are exchangeable, the expected
share of wrong answers among the answered stays within the target. One row alone is answered when
its p-value is at most the target. Regression claims no per-row confidence, so it has no rule.

Shapes: `n` rows, `c` classes or labels.
"""

from __future__ import annotations

import numpy as np

from collections.abc import Mapping, Sequence
from typing import Any

from .metrics import CLASSIFICATION_KINDS, PROBABILITY_FLOOR


ANSWER_SCHEME = "conformal_selection"
# The default target: half of V_cal's error rate, the answer rule's development target.
DEFAULT_TARGET_SHARE = 0.5


def requires_a_label(label_sets: Sequence[np.ndarray]) -> bool:
    """Whether every multilabel row given carries at least one positive label."""
    # label_sets: (n, c) per set
    return all(bool(np.all((labels > 0.5).any(axis=1))) for labels in label_sets)  # each (n, c)


def decode(probs: np.ndarray, kind: str, at_least_one_label: bool) -> np.ndarray:
    """Each row's answer: its top class, `(n,)`, or its label set, `(n, c)` bool."""
    # probs: (n, c)
    if kind in CLASSIFICATION_KINDS:
        return probs.argmax(axis=1)  # (n,)
    if kind != "multilabel":
        raise ValueError(f"only classification tasks give an answer to select; got {kind!r}.")

    chosen = probs >= 0.5  # (n, c)
    if at_least_one_label:
        empty = ~chosen.any(axis=1)  # (n,)
        chosen[np.nonzero(empty)[0], probs[empty].argmax(axis=1)] = True
    return chosen  # (n, c)


def answer_confidence(probs: np.ndarray, answers: np.ndarray, kind: str, at_least_one_label: bool) -> np.ndarray:
    """The calibrated probability that each row's answer is right, `(n,)`."""
    # probs: (n, c), answers: (n,) or (n, c)
    if kind in CLASSIFICATION_KINDS:
        return probs[np.arange(probs.shape[0]), answers].astype(np.float64)  # (n,)

    # Clipped in float64, where a saturated float32 sigmoid of 1.0 would give log(0).
    clipped = np.clip(np.asarray(probs, dtype=np.float64), PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)  # (n, c)
    log_set = np.where(answers, np.log(clipped), np.log1p(-clipped)).sum(axis=1)  # (n,)
    if at_least_one_label:
        log_empty = np.log1p(-clipped).sum(axis=1)  # (n,)
        log_set = log_set - np.log(-np.expm1(log_empty))  # (n,), given a non-empty set
    return np.exp(log_set)  # (n,)


def answer_correct(answers: np.ndarray, labels: np.ndarray, kind: str) -> np.ndarray:
    """Whether each row's answer is right: its class, or its whole label set, `(n,)` bool."""
    # answers: (n,) or (n, c), labels: (n,) or (n, c)
    if kind in CLASSIFICATION_KINDS:
        return answers == labels  # (n,)
    return np.all(answers == (labels > 0.5), axis=1)  # (n,)


def answer_rule_record(probs: np.ndarray, labels: np.ndarray, kind: str, at_least_one_label: bool | None) -> dict[str, Any] | None:
    """What a package needs to answer by conformal selection, from the shipped prediction on V_cal.

    None for regression. `wrong_confidences` holds the confidence of every V_cal row answered
    wrongly, ascending, and is all a p-value reads besides the row count.
    """
    # probs: (n, c), labels: (n,) or (n, c)
    if kind == "regression":
        return None

    rule = bool(at_least_one_label) if kind == "multilabel" else False
    answers = decode(probs, kind, rule)
    confidence = answer_confidence(probs, answers, kind, rule)  # (n,)
    return scored_rule_record(confidence, answer_correct(answers, labels, kind), kind, at_least_one_label, "confidence")


def scored_rule_record(
    scores: np.ndarray, correct: np.ndarray, kind: str, at_least_one_label: bool | None, score: str
) -> dict[str, Any]:
    """The answer rule over any score that is higher for answers more likely right, from V_cal's rows.

    `score` names it: `confidence`, the calibrated probability, or `correctness`, a correctness
    model's (`correctness.py`). `wrong_confidences` holds the score of every row answered wrongly.
    """
    # scores, correct: (n,)
    error = float(1.0 - correct.mean())
    return {
        "scheme": ANSWER_SCHEME,
        "score": score,
        "answer": "top_class" if kind in CLASSIFICATION_KINDS else "label_set",
        "at_least_one_label": at_least_one_label if kind == "multilabel" else None,
        "calibration_rows": int(correct.shape[0]),
        "calibration_error": error,
        "default_target": DEFAULT_TARGET_SHARE * error,
        "wrong_confidences": np.sort(np.asarray(scores, dtype=np.float64)[~correct]).tolist(),
    }


def conformal_p_values(confidence: np.ndarray, rule: Mapping[str, Any]) -> np.ndarray:
    """Each row's p-value against "this answer is wrong", from V_cal's wrong answers, `(n,)`."""
    # confidence: (n,)
    wrong = np.asarray(rule["wrong_confidences"], dtype=np.float64)  # (w,), ascending
    at_least_as_confident = wrong.shape[0] - np.searchsorted(wrong, confidence, side="left")  # (n,)
    return (1.0 + at_least_as_confident) / (rule["calibration_rows"] + 1.0)  # (n,)


def select_answers(p_values: np.ndarray, target: float) -> np.ndarray:
    """Benjamini-Hochberg at `target` over the rows given: which to answer, `(n,)` bool."""
    # p_values: (n,)
    if not 0.0 < target < 1.0:
        raise ValueError(f"the target error must lie strictly between 0 and 1; got {target}.")
    count = p_values.shape[0]
    if count == 0:
        return np.zeros(0, dtype=bool)  # (0,)

    ordered = np.sort(p_values)  # (n,)
    passing = np.nonzero(ordered <= target * np.arange(1, count + 1) / count)[0]
    if passing.size == 0:
        return np.zeros(count, dtype=bool)  # (n,)
    return p_values <= ordered[passing[-1]]  # (n,)
