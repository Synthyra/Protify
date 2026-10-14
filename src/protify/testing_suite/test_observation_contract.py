"""Missing calibration truth limits package claims and never becomes negative evidence."""

import copy
import numpy as np
import pytest

from dataclasses import replace

from protify.uncertainty.observation_contract import (
    answer_truth_bounds, calibration_observations, partial_answer_unavailable, validate_observation_contract,
)
from protify.uncertainty.predictions import SplitPredictions


def partial_predictions(kind="multilabel"):
    labels = np.array([[0.0, 1.0], [1.0, 99.0], [99.0, 99.0]]) if kind == "multilabel" else np.array([0.0, 1.0, 99.0])
    mask = labels != 99
    width = 1 if kind == "regression" else 2
    split = SplitPredictions(labels, np.zeros((3, width)), label_mask=mask)
    return dict.fromkeys(("v_fit", "v_cal", "test"), split)


@pytest.mark.parametrize("kind", ["binary", "multiclass", "multilabel", "regression"])
def test_contract_uses_only_measured_calibration_truth(kind):
    splits = partial_predictions(kind)
    record = calibration_observations(splits, kind)
    validate_observation_contract(record, kind, 1 if kind == "regression" else 2)
    assert record["support"]["v_fit"]["observed"] == (3 if kind == "multilabel" else 2)
    assert partial_answer_unavailable(record) == (kind != "regression")
    if kind == "multilabel":
        assert record["support"]["v_cal"]["labels"][1] == {
            "index": 1, "positive": 1, "negative": 0, "unknown": 2, "discrimination_status": "insufficient_observed_classes",
        }
    for role in splits:
        labels = splits[role].labels.copy()
        labels[~splits[role].label_mask] = np.nan
        splits[role] = replace(splits[role], labels=labels)
    assert calibration_observations(splits, kind) == record
    splits["test"] = None  # Test outcomes and even their availability cannot choose this contract.
    assert calibration_observations(splits, kind) == record


@pytest.mark.parametrize("role", ["v_fit", "v_cal"])
def test_empty_observed_calibration_is_refused(role):
    splits = partial_predictions()
    splits[role] = replace(splits[role], label_mask=np.zeros_like(splits[role].label_mask))
    with pytest.raises(ValueError, match="requires observed"):
        calibration_observations(splits, "multilabel")


def test_complete_and_legacy_calibration_preserve_their_claims():
    splits = partial_predictions("binary")
    splits = {role: replace(split, labels=np.array([0, 1, 0]), label_mask=None) for role, split in splits.items()}
    assert calibration_observations(splits, "binary") is None
    splits["v_fit"] = replace(splits["v_fit"], label_mask=np.ones(3, dtype=bool))
    contract = calibration_observations(splits, "binary")
    assert contract["whole_answer"] == "complete_calibration_truth"
    validate_observation_contract(contract, "binary", 2)


@pytest.mark.parametrize("mutation", ["claim", "count", "negative", "status", "test", "rows", "boolean", "missing"])
def test_loader_validation_rejects_overstated_support(mutation):
    record = copy.deepcopy(calibration_observations(partial_predictions(), "multilabel"))
    support = record["support"]["v_cal"]
    if mutation == "claim":
        record["whole_answer"] = "complete_calibration_truth"
    elif mutation == "count":
        support["observed"] += 1
    elif mutation == "negative":
        support["labels"][1]["negative"] = 1
    elif mutation == "status":
        support["labels"][1]["discrimination_status"] = "supported"
    elif mutation == "test":
        record["support"]["test"] = support
    elif mutation == "rows":
        support["rows"] = 0
    elif mutation == "boolean":
        support["observed"] = True
    else:
        del record["missing_truth"]
    with pytest.raises(ValueError):
        validate_observation_contract(record, "multilabel", 2)


@pytest.mark.parametrize("column", [False, True])
def test_scalar_correctness_distinguishes_wrong_correct_and_unknown(column):
    labels = np.array([0.0, 0.0, np.nan])
    mask = np.array([True, True, False])
    if column:
        labels, mask = labels[:, None], mask[:, None]
    certain, possible = answer_truth_bounds(np.array([0, 1, 1]), labels, "binary", mask)
    np.testing.assert_array_equal(certain, [True, False, False])
    np.testing.assert_array_equal(possible, [True, False, True])


def test_partial_agreement_is_unknown_but_observed_disagreement_establishes_error():
    labels = np.array([[1.0, 0.0], [1.0, np.nan], [0.0, np.nan], [np.nan, np.nan]])
    mask = np.isfinite(labels)
    certain, possible = answer_truth_bounds(np.ones_like(mask), labels, "multilabel", mask)
    np.testing.assert_array_equal(certain, [False, False, False, False])
    np.testing.assert_array_equal(possible, [False, True, False, True])
    labels[~mask] = -123
    changed = answer_truth_bounds(np.ones_like(mask), labels, "multilabel", mask)
    np.testing.assert_array_equal(changed, (certain, possible))


def test_deleting_an_annotation_widens_truth_bounds_instead_of_creating_a_negative():
    labels = np.ones((1, 2))
    mask = np.ones((1, 2), dtype=bool)
    assert all(values.item() for values in answer_truth_bounds(mask, labels, "multilabel", mask))
    mask[0, 0] = False
    certain, possible = answer_truth_bounds(np.ones((1, 2), dtype=bool), labels, "multilabel", mask)
    assert not certain.item() and possible.item()
