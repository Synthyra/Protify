"""The answer rule: decoding, the confidence that an answer is right, and conformal selection."""

from __future__ import annotations

import itertools
import numpy as np
import pytest

from protify.uncertainty.answering import (
    answer_confidence,
    answer_correct,
    answer_rule_record,
    conformal_p_values,
    decode,
    requires_a_label,
    select_answers,
)


def _set_probability(probs: np.ndarray, chosen: tuple[bool, ...]) -> float:
    # probs: (k,)
    return float(np.prod([p if pick else 1.0 - p for p, pick in zip(probs, chosen, strict=True)]))


def test_an_empty_set_answers_its_top_label_only_under_the_rule():
    probs = np.array([[0.2, 0.4, 0.1], [0.7, 0.2, 0.9]])  # (2, 3)

    assert decode(probs, "multilabel", False).tolist() == [[False, False, False], [True, False, True]]
    assert decode(probs, "multilabel", True).tolist() == [[False, True, False], [True, False, True]]
    assert decode(probs, "multiclass", False).tolist() == [1, 2]


def test_the_decoded_set_is_the_most_probable_non_empty_set():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.01, 0.99, size=(200, 4))  # (200, 4)
    probs[:50] *= 0.4  # rows whose every label falls below 0.5

    decoded = decode(probs, "multilabel", True)  # (200, 4)
    sets = [chosen for chosen in itertools.product((False, True), repeat=4) if any(chosen)]
    for row, answer in zip(probs, decoded, strict=True):
        best = max(sets, key=lambda chosen: _set_probability(row, chosen))
        assert tuple(answer) == best


def test_the_confidence_is_the_set_probability_given_a_non_empty_set():
    rng = np.random.default_rng(1)
    probs = rng.uniform(0.01, 0.99, size=(50, 3))  # (50, 3)

    for rule in (False, True):
        answers = decode(probs, "multilabel", rule)
        confidence = answer_confidence(probs, answers, "multilabel", rule)  # (50,)
        for row, answer, value in zip(probs, answers, confidence, strict=True):
            expected = _set_probability(row, tuple(answer))
            if rule:
                expected /= 1.0 - _set_probability(row, (False, False, False))
            assert value == pytest.approx(expected, rel=1e-12)

    single = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]])
    assert answer_confidence(single, decode(single, "multiclass", False), "multiclass", False).tolist() == [0.7, 0.5]


def test_correctness_needs_the_whole_label_set():
    answers = np.array([[True, False], [True, True]])
    labels = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert answer_correct(answers, labels, "multilabel").tolist() == [True, False]
    assert answer_correct(np.array([0, 2]), np.array([0, 1]), "multiclass").tolist() == [True, False]


def test_requires_a_label_reads_every_split_given():
    assert requires_a_label([np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 1.0]])])
    assert not requires_a_label([np.array([[1.0, 0.0]]), np.array([[0.0, 0.0]])])


def test_p_values_count_wrong_answers_at_least_as_confident():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.55, 0.45]])  # (5, 2)
    labels = np.array([0, 1, 1, 0, 1])  # rows 1, 3, and 4 are answered wrongly
    rule = answer_rule_record(probs, labels, "binary", None)

    assert rule["wrong_confidences"] == [0.55, 0.6, 0.8]
    assert rule["calibration_rows"] == 5
    assert rule["calibration_error"] == pytest.approx(0.6)
    assert rule["default_target"] == pytest.approx(0.3)
    p = conformal_p_values(np.array([0.95, 0.8, 0.6, 0.5]), rule)
    assert p.tolist() == pytest.approx([1 / 6, 2 / 6, 3 / 6, 4 / 6])
    assert answer_rule_record(probs, labels.astype(float), "regression", None) is None


def test_selection_is_benjamini_hochberg():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 0.2, size=40)  # (40,)
    target = 0.1

    ordered = np.sort(p)
    passing = [k for k in range(1, 41) if ordered[k - 1] <= target * k / 40]
    expected = p <= ordered[passing[-1] - 1] if passing else np.zeros(40, dtype=bool)
    assert np.array_equal(select_answers(p, target), expected)
    assert not select_answers(np.array([0.5, 0.9]), 0.1).any()
    assert select_answers(np.zeros(0), 0.1).shape == (0,)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        select_answers(p, 0.0)


def test_selection_keeps_the_error_among_answered_within_target_on_exchangeable_rows():
    rng = np.random.default_rng(3)
    target, errors = 0.1, []
    for _ in range(300):
        # Confidence is informative: a row is right with probability equal to its confidence.
        confidence = rng.uniform(0.5, 1.0, size=400)  # (400,)
        correct = rng.uniform(size=400) < confidence
        cal, test = slice(0, 200), slice(200, 400)
        rule = {"wrong_confidences": np.sort(confidence[cal][~correct[cal]]).tolist(), "calibration_rows": 200}
        answered = select_answers(conformal_p_values(confidence[test], rule), target)
        errors.append(1.0 - correct[test][answered].mean() if answered.any() else 0.0)
    assert np.mean(errors) <= target + 0.01
