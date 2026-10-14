"""Package claims supported by observed calibration truth, independent of test outcomes.

Missingness is not assumed random. Calibration describes measured targets; it cannot identify
whole-answer correctness or guarantee coverage for unobserved labels without further evidence.
Shapes: n rows, c labels, b rows in one bounded prediction block.
"""

from __future__ import annotations

import numpy as np

from collections.abc import Mapping
from typing import Any

from .metrics import CLASSIFICATION_KINDS, TASK_KINDS
from .prediction_arrays import prediction_row_blocks
from .predictions import SplitPredictions
from .report import label_support
from ..metrics import validate_observed_labels


OBSERVATION_SCHEME = "observed_label_package_v1"


def calibration_observations(splits: Mapping[str, SplitPredictions], kind: str) -> dict[str, Any] | None:
    """Freeze V_fit/V_cal support; test masks never choose inference behavior."""
    if kind not in TASK_KINDS:
        raise ValueError("observation contract has an unsupported task type")
    if all(splits[role].label_mask is None for role in ("v_fit", "v_cal")):
        return None
    support = {}
    for role in ("v_fit", "v_cal"):
        prediction = splits[role]
        # labels, mask: (n, c) for multilabel, otherwise (n,) or (n, 1).
        mask = prediction.label_mask
        if mask is None:
            mask = np.broadcast_to(np.asarray(True), prediction.labels.shape)  # same shape, no dense allocation
        support[role] = {"rows": len(prediction.labels), **label_support(kind, prediction.labels, mask)}
        if support[role]["observed"] == 0:
            raise ValueError(f"package calibration requires observed {role} targets")
    partial = any(record["unknown"] for record in support.values())
    return {
        "scheme": OBSERVATION_SCHEME, "task_type": kind, "missing_truth": "unknown",
        "calibration_scope": "observed_targets", "unobserved_truth_coverage": "not_established",
        "whole_answer": "unavailable_partial_calibration" if partial and kind != "regression" else
                        "not_applicable" if kind == "regression" else "complete_calibration_truth",
        "support": support,
    }


def partial_answer_unavailable(contract: Mapping[str, Any] | None) -> bool:
    return contract is not None and contract.get("whole_answer") == "unavailable_partial_calibration"


def validate_observation_contract(contract: Mapping[str, Any], kind: str, width: int) -> None:
    """Reject a stripped or contradictory capability claim at package reload."""
    if (not isinstance(contract, Mapping) or kind not in TASK_KINDS or type(width) is not int or width < 1
            or set(contract) != {"scheme", "task_type", "missing_truth", "calibration_scope", "unobserved_truth_coverage", "whole_answer", "support"}
            or contract["scheme"] != OBSERVATION_SCHEME or contract["task_type"] != kind
            or contract["missing_truth"] != "unknown" or contract["calibration_scope"] != "observed_targets"
            or contract["unobserved_truth_coverage"] != "not_established"):
        raise ValueError("invalid packaged observation contract")
    support = contract["support"]
    if not isinstance(support, dict) or set(support) != {"v_fit", "v_cal"}:
        raise ValueError("packaged observation support must use V_fit and V_cal only")
    for record in support.values():
        if not isinstance(record, dict) or any(type(record.get(key)) is not int for key in ("rows", "observed", "unknown")):
            raise ValueError("packaged observation counts must be integers")
        if record["rows"] < 1 or record["observed"] < 1 or record["unknown"] < 0 or record.get("scope") != "observed_targets":
            raise ValueError("packaged calibration requires observed truth")
        expected = record["rows"] * (width if kind == "multilabel" else 1)
        if record["observed"] + record["unknown"] != expected:
            raise ValueError("packaged observation counts disagree with rows and vocabulary")
        if kind == "multilabel":
            labels = record.get("labels")
            if not isinstance(labels, list) or len(labels) != width:
                raise ValueError("packaged support must describe every label")
            observed = 0
            for index, label in enumerate(labels):
                if (not isinstance(label, dict) or label.get("index") != index
                        or any(type(label.get(key)) is not int or label[key] < 0 for key in ("positive", "negative", "unknown"))
                        or label["positive"] + label["negative"] + label["unknown"] != record["rows"]):
                    raise ValueError("invalid per-label observation counts")
                status = "supported" if label["positive"] and label["negative"] else "insufficient_observed_classes"
                if label.get("discrimination_status") != status:
                    raise ValueError("packaged label support overstates observed classes")
                observed += label["positive"] + label["negative"]
            if observed != record["observed"]:
                raise ValueError("per-label counts differ from the observed total")
        elif kind in CLASSIFICATION_KINDS:
            counts = record.get("class_counts")
            if (not isinstance(counts, dict) or not counts
                    or any(str(index) not in {str(value) for value in range(width)}
                           or type(count) is not int or count < 1 for index, count in counts.items())
                    or sum(counts.values()) != record["observed"]):
                raise ValueError("invalid observed class counts")
            status = "supported" if len(counts) > 1 else "insufficient_observed_classes"
            if record.get("discrimination_status") != status:
                raise ValueError("packaged class support overstates observed classes")
    partial = any(record["unknown"] for record in support.values())
    expected = "not_applicable" if kind == "regression" else "unavailable_partial_calibration" if partial else "complete_calibration_truth"
    if contract["whole_answer"] != expected:
        raise ValueError("packaged whole-answer claim disagrees with its observation support")


def answer_truth_bounds(
    answers: np.ndarray, labels: np.ndarray, kind: str, mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return certainly-correct and possibly-correct flags; unknown is never false truth."""
    # answers, labels, mask: (n, c) for multilabel, otherwise (n,).
    if kind not in (*CLASSIFICATION_KINDS, "multilabel"):
        raise ValueError("answer correctness requires a classification task")
    if mask is None:
        mask = np.broadcast_to(np.asarray(True), labels.shape)  # (n, c) or (n,)
    if kind == "multilabel" and answers.shape != labels.shape:
        raise ValueError("answers must have exactly the truth shape")
    if kind != "multilabel" and (answers.shape != (len(labels),) or labels.shape not in ((len(labels),), (len(labels), 1))):
        raise ValueError("scalar answers must match the truth rows")
    certain = np.empty(len(labels), dtype=bool)  # (n,)
    possible = np.empty(len(labels), dtype=bool)  # (n,)
    offset = 0
    for predicted, truth, observed in prediction_row_blocks(answers, labels, mask):
        # predicted, truth, observed: (b, c) or (b,).
        validate_observed_labels(truth, observed)
        if kind != "multilabel":
            truth, observed = truth.reshape(-1), observed.reshape(-1)  # (b,), (b,)
        matches = np.ones(truth.shape, dtype=bool)  # (b, c) or (b,)
        matches[observed] = predicted[observed] == (truth[observed] > 0.5 if kind == "multilabel" else truth[observed])  # (observed,)
        if kind == "multilabel":
            consistent = matches.all(axis=1)  # (b,)
            complete = observed.all(axis=1)  # (b,)
        else:
            consistent, complete = matches, observed  # (b,), (b,)
        end = offset + len(truth)
        certain[offset:end], possible[offset:end] = consistent & complete, consistent  # (b,), (b,)
        offset = end
    return certain, possible  # (n,), (n,)
