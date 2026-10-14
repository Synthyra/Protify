"""Fit a package's correctness model from its cell's cross-fit, and check what the package ships.

`crossfit.crossfit_cell` leaves, for each fold head, the correctness model's columns on a sample
of its held-out rows, and for the served head its reference (`reference/`) and its signals on
V_fit, V_cal, and test (`served_correctness.npz`). `package_correctness` fits the model on the
out-of-fold rows at `correctness.LEVEL`, scores V_cal and test with the package's own
confidences, and returns the answer rule ranked by that score, the metrics, and the files
`correctness/` holds. It also fits V_cal's novelty cutoff and support table from the
reference's distances, and the `answer_or_abstain_v1` policy over the rule and that record
(`applicability.py`), and reports what the policy does with test's rows, decided as one
request and one row per request. `check_shipped_correctness` then predicts the cross-fit's
check rows through the written package and refuses one whose reliability or novelty drifts
from the cross-fit's.

Shapes: `n` rows of a split, `c` classes or labels, `f` features.
"""

from __future__ import annotations

import numpy as np
import torch

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foundry.metrics.calibration import expected_calibration_error
from foundry.metrics.selective import area_under_risk_coverage
from .answering import answer_confidence, answer_correct, conformal_p_values, decode, scored_rule_record
from .applicability import (
    NOVELTY_SIGNAL,
    applicability_status,
    decide,
    decide_alone,
    decision_outcome,
    decision_policy,
    evaluate_applicability,
    fit_applicability,
)
from .correctness import (
    LEVEL,
    MODEL_FILE,
    CorrectnessModel,
    LatentReference,
    feature_matrix,
    feature_names,
    model_record,
)
from .crossfit import RECORD, REFERENCE_DIRECTORY, SERVED_CORRECTNESS, fold_name
from .crossfit_artifacts import CALIBRATION_SCOPE, verify_calibration_artifacts
from .report import read_json, write_json


# The reliability a package gives the check rows, against the cross-fit's for the same rows. The
# cross-fit reads the head on the training device, in TF32 on CUDA; a package often runs on CPU.
CHECK_MEDIAN_TOLERANCE = 1e-3
CHECK_MAX_TOLERANCE = 0.1
ECE_BINS = 15
PROBABILITY_FLOOR = 1e-6


@dataclass(frozen=True)
class PackagedCorrectness:
    """What a package takes from its cell's cross-fit."""

    model: CorrectnessModel
    reference: LatentReference
    record: dict[str, Any]  # correctness.json
    rule: dict[str, Any]  # the answer rule, ranked by reliability
    applicability: dict[str, Any]  # V_cal's novelty cutoff and support table, or why there is none
    policy: dict[str, Any]  # answer_or_abstain_v1 over the rule and the applicability record
    reliability: dict[str, np.ndarray]  # split -> (n,), for V_cal and test
    metrics: dict[str, Any]
    evidence: dict[str, Any]  # test rows under the policy: their novelty and their decisions; empty without test
    check_rows: np.ndarray  # (m,) rows of check_split
    check_features: np.ndarray  # (m, W) their raw joined features
    check_novelty: np.ndarray  # (m,) the cross-fit's novelty signal on those rows
    check_split: str = "test"


def _served_arrays(directory: Path) -> dict[str, np.ndarray]:
    with np.load(directory / SERVED_CORRECTNESS, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}  # (...) per name, any shape


def out_of_fold_rows(directory: Path, crossfit: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Every fold's out-of-fold columns at `LEVEL`, concatenated, as `name -> (n,)`."""
    plan = crossfit["plan"]
    if LEVEL not in plan["levels"]:
        raise ValueError(f"{directory} was cross-fit at levels {plan['levels']}; the correctness model is fitted at {LEVEL}.")

    columns: dict[str, list[np.ndarray]] = {}
    for fold in range(plan["folds"]):
        path = directory / f"{fold_name(LEVEL, fold)}.npz"
        if not path.exists():
            raise ValueError(f"{directory} has no {path.name}; finish the cross-fit before packaging.")
        with np.load(path, allow_pickle=False) as archive:
            names = [name for name in archive.files if name.startswith("correctness__") and name != "correctness__reference_rows"]
            if not names:
                raise ValueError(f"{path} was cross-fit without the correctness columns.")
            for name in names:
                columns.setdefault(name.removeprefix("correctness__"), []).append(archive[name])
    return {name: np.concatenate(parts) for name, parts in columns.items()}  # (n,) or (n, ...) per column name, concatenated over folds


def correctness_metrics(reliability: np.ndarray, confidence: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    """Brier, NLL, equal-mass ECE, and AURC of the reliability, each beside the confidence's."""
    # reliability, confidence, correct: (n,)
    right = correct.astype(np.float64)
    metrics = {}
    for name, score in (("reliability", reliability), ("confidence", confidence)):
        clipped = np.clip(np.asarray(score, dtype=np.float64), PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)  # (n,)
        metrics[f"{name}_brier"] = float(np.mean((clipped - right) ** 2))
        metrics[f"{name}_nll"] = float(-np.mean(right * np.log(clipped) + (1.0 - right) * np.log1p(-clipped)))
        metrics[f"{name}_ece"] = float(expected_calibration_error(right, clipped, ECE_BINS))
        metrics[f"{name}_aurc"] = float(area_under_risk_coverage(1.0 - right, -clipped))
    return metrics


def package_correctness(
    crossfit_dir: str,
    *,
    kind: str,
    num_labels: int,
    shipped_seeds: Sequence[int],
    probs: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    at_least_one_label: bool | None,
    calibration_only: bool = False,
) -> PackagedCorrectness:
    """The correctness model, fitted on the cross-fit's out-of-fold rows, and what it does on V_cal and test.

    `probs` and `labels` are the shipped prediction's calibrated probabilities and the labels on
    `v_cal` and `test`. The package's answers must be the ones the cross-fit read its signals
    for, and its one shipped head the cross-fit's served head. `calibration_only` accepts a
    completed calibration-only cross-fit and scores only V_cal. It does not make that
    cross-fit eligible for `build_package`, which still requires completed test evaluation.
    """
    # probs: (n, c) per split name, labels: (n,) or (n, c) per split name
    directory = Path(crossfit_dir)
    crossfit = read_json(str(directory / RECORD))
    scope = CALIBRATION_SCOPE if calibration_only else "full"
    if crossfit.get("evaluation_scope", "full") != scope:
        raise ValueError("correctness fitting and cross-fit evaluation scope disagree")
    splits = ("v_cal",) if calibration_only else ("v_cal", "test")
    if calibration_only:
        if set(probs) != set(splits) or set(labels) != set(splits):
            raise ValueError("calibration-only correctness takes V_cal probabilities and labels alone")
        verify_calibration_artifacts(directory, crossfit, complete=True)
        if crossfit.get("task_type") != kind or crossfit.get("num_labels") != num_labels:
            raise ValueError("correctness task or vocabulary width differs from the calibration cross-fit")
        with np.load(directory / "served.npz", allow_pickle=False) as archive:
            if not np.array_equal(labels["v_cal"], archive["v_cal__labels"]):
                raise ValueError("correctness V_cal labels differ from the calibration cross-fit")
    if list(shipped_seeds) != [crossfit["served_seed"]]:
        raise ValueError(f"the package ships seeds {list(shipped_seeds)}, but {directory} read the head of seed {crossfit['served_seed']}.")
    served_record = crossfit.get("served_correctness")
    if served_record is None:
        raise ValueError(f"{directory} has no served correctness signals; finish the cross-fit before packaging.")
    check_split = served_record.get("check_split", "test")
    if check_split != ("v_cal" if calibration_only else "test"):
        raise ValueError("correctness check rows disagree with the cross-fit evaluation scope")
    rule_needs_label = bool(at_least_one_label) if kind == "multilabel" else False
    if served_record["at_least_one_label"] != rule_needs_label:
        raise ValueError(f"{directory} decoded answers with at_least_one_label {served_record['at_least_one_label']}, the package with {rule_needs_label}.")

    out_of_fold = out_of_fold_rows(directory, crossfit)
    reference_directory = directory / REFERENCE_DIRECTORY
    scales = read_json(str(reference_directory / "reference.json"))["scales"]
    reference = LatentReference.load(reference_directory, kind, num_labels, scales, torch.device("cpu"))
    names = feature_names(kind, reference.has_sae)
    model = CorrectnessModel.fit(feature_matrix(out_of_fold["confidence"], out_of_fold, names), out_of_fold["correct"], names)

    served = _served_arrays(directory)
    reliability, confidence, correct = {}, {}, {}
    for split in splits:
        answers = decode(probs[split], kind, rule_needs_label)
        if not np.array_equal(answers, served[f"{split}__answers"]):
            raise ValueError(f"the package's {split} answers differ from those {directory} read its signals for.")
        confidence[split] = answer_confidence(probs[split], answers, kind, rule_needs_label)  # (n,)
        correct[split] = answer_correct(answers, labels[split], kind)  # (n,)
        signals = {name: served[f"{split}__{name}"] for name in names}
        reliability[split] = model.probability(feature_matrix(confidence[split], signals, names))  # (n,)

    fitted_on = {
        "level": LEVEL,
        "folds": crossfit["plan"]["folds"],
        "out_of_fold_rows": int(out_of_fold["correct"].shape[0]),
        "out_of_fold_error": float(1.0 - out_of_fold["correct"].mean()),
        "reference_rows": served_record["reference_rows"],
    }
    novelty = {split: served[f"{split}__{NOVELTY_SIGNAL}"] for split in splits}  # each (n,)
    rule = scored_rule_record(reliability["v_cal"], correct["v_cal"], kind, at_least_one_label, "correctness")
    applicability = fit_applicability(
        novelty["v_cal"], reliability["v_cal"], correct["v_cal"], score_name="correctness", reference_sha256=reference.content_sha256()
    )
    policy = decision_policy(rule, applicability)
    return PackagedCorrectness(
        model=model,
        reference=reference,
        record=model_record(model, reference, fitted_on),
        rule=rule,
        applicability=applicability,
        policy=policy,
        reliability=reliability,
        metrics={
            split: correctness_metrics(reliability[split], confidence[split], correct[split]) for split in splits
        },
        evidence=held_out_evidence(rule, applicability, policy, novelty["test"], reliability["test"], correct["test"]) if "test" in splits else {},
        check_rows=served["check__rows"],
        check_features=served["check__features"],
        check_novelty=served[f"{check_split}__{NOVELTY_SIGNAL}"][served["check__rows"]],
        check_split=check_split,
    )


def held_out_evidence(
    rule: Mapping[str, Any], applicability: Mapping[str, Any], policy: Mapping[str, Any],
    novelty: np.ndarray, reliability: np.ndarray, correct: np.ndarray,
) -> dict[str, Any]:
    """What the policy does on rows that took no part in calibrating it: their novelty, and their answers and abstentions.

    `decisions` decides the rows as one request, the way selection's error statement is made. `single_row_decisions` decides each
    row as its own request, which answers every row the first answers and more, and whose answered error is the one a caller
    who sends one protein at a time meets.
    """
    # novelty, reliability: (n,); correct: (n,) bool
    status = None
    evidence: dict[str, Any] = {}
    if applicability["status"] == "calibrated":
        status = applicability_status(applicability, novelty)  # (n,)
        evidence["applicability"] = evaluate_applicability(applicability, novelty, reliability, correct)
    p_values = conformal_p_values(reliability, rule)  # (n,)
    evidence["decisions"] = decision_outcome(decide(policy, applicability, status=status, score=reliability, p_values=p_values), correct)
    evidence["single_row_decisions"] = decision_outcome(
        decide_alone(policy, applicability, status=status, score=reliability, p_values=p_values), correct
    )
    return evidence


def write_correctness(directory: Path, correctness: PackagedCorrectness) -> None:
    """`correctness/`: the reference the model reads, `reference.npz`, and `correctness.json`."""
    directory.mkdir()
    correctness.reference.save(directory)
    write_json(str(directory / MODEL_FILE), correctness.record)


def check_shipped_correctness(head: Any, correctness: PackagedCorrectness) -> dict[str, float]:
    """Refuse a package whose reliability, or novelty distance when it ships a cutoff, on the check rows drifts from the cross-fit's; return the drift."""
    prediction = head.predict(correctness.check_features)
    expected = correctness.reliability[correctness.check_split][correctness.check_rows]  # (m,)
    drift = np.abs(prediction.reliability - expected)  # (m,)
    median, worst = float(np.median(drift)), float(drift.max())
    if median > CHECK_MEDIAN_TOLERANCE or worst > CHECK_MAX_TOLERANCE:
        raise ValueError(
            f"the package's reliability on {drift.size} {correctness.check_split} rows differs from the cross-fit's by a median {median:.2e} "
            f"and at most {worst:.2e}, beyond {CHECK_MEDIAN_TOLERANCE} and {CHECK_MAX_TOLERANCE}."
        )
    checked = {"rows": int(drift.size), "median": median, "max": worst}
    if correctness.applicability["status"] == "calibrated":
        # The novelty cutoff was calibrated on the cross-fit's distances, so the package must measure the same ones.
        shift = np.abs(prediction.signals[NOVELTY_SIGNAL] - correctness.check_novelty)  # (m,)
        checked["novelty_median"], checked["novelty_max"] = float(np.median(shift)), float(shift.max())
        if checked["novelty_median"] > CHECK_MEDIAN_TOLERANCE or checked["novelty_max"] > CHECK_MAX_TOLERANCE:
            raise ValueError(
                f"the package's novelty distance on {shift.size} {correctness.check_split} rows differs from the cross-fit's by a median "
                f"{checked['novelty_median']:.2e} and at most {checked['novelty_max']:.2e}, beyond {CHECK_MEDIAN_TOLERANCE} and {CHECK_MAX_TOLERANCE}."
            )
    return checked
