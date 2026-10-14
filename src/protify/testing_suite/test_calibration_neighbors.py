"""Calibration-set neighbor estimates of correctness, and their evaluation end to end.

The toy calibration sets put rows in two directions of the trunk space. The model claims the
same confidence in both, but is right far more often in one; an estimator that reads its
neighbors' outcomes must separate them, and one far from every calibration row must fall back to
the model's claim.
"""

import json
import math
import numpy as np
import pytest

from scipy.special import expit, softmax


try:
    from src.protify.uncertainty import calibration_neighbors as neighbors
    from src.protify.uncertainty import workflow
    from src.protify.uncertainty.neighbor_evaluation import auroc, coverage_at_error, evaluate_neighbors
    from src.protify.uncertainty.predictions import load_seed_predictions
    from src.protify.testing_suite.test_single_probe_workflow import _cell, toy  # noqa: F401
except ImportError:
    from protify.uncertainty import calibration_neighbors as neighbors
    from protify.uncertainty import workflow
    from protify.uncertainty.neighbor_evaluation import auroc, coverage_at_error, evaluate_neighbors
    from protify.uncertainty.predictions import load_seed_predictions
    from protify.testing_suite.test_single_probe_workflow import _cell, toy  # noqa: F401


def _two_directions(rng: np.random.Generator, count: int, width: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Rows near one of two orthogonal directions, and which one each is near."""
    side = rng.integers(0, 2, size=count)  # (n,)
    centers = np.eye(width)[[0, 1]]  # (2, p)
    return centers[side] + 0.05 * rng.normal(size=(count, width)), side  # (n, p), (n,)


def _toy(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, side = _two_directions(rng, count)
    claimed = np.full(count, 0.8)  # (n,)
    correct = (rng.random(count) < np.where(side == 0, 0.95, 0.4)).astype(np.float64)  # (n,)
    return neighbors.unit_rows(features), side, claimed, correct  # (n, p), (n,), (n,), (n,)


def test_the_residual_estimate_reads_where_the_model_fails():
    rng = np.random.default_rng(0)
    unit, _, claimed, correct = _toy(rng, 600)
    query_unit, query_side, query_claimed, query_correct = _toy(rng, 400)
    rows = neighbors.CalibrationRows(unit, correct, claimed)
    calibration_similarities = neighbors.cosine_similarities(unit, unit)
    query_similarities = neighbors.cosine_similarities(query_unit, unit)

    model = neighbors.estimate("model", rows, calibration_similarities, query_similarities, query_claimed)
    residual = neighbors.estimate("residual", rows, calibration_similarities, query_similarities, query_claimed)
    assert residual.loo_log_likelihood > model.loo_log_likelihood
    assert abs(residual.q[query_side == 0].mean() - 0.95) < 0.05
    assert abs(residual.q[query_side == 1].mean() - 0.4) < 0.07
    assert auroc(query_correct, residual.q) > auroc(query_correct, model.q) + 0.2
    for name in ("knn", "joint"):
        other = neighbors.estimate(name, rows, calibration_similarities, query_similarities, query_claimed)
        assert auroc(query_correct, other.q) > 0.7, name


def test_a_query_far_from_every_calibration_row_keeps_the_model_claim():
    rng = np.random.default_rng(1)
    unit, _, claimed, correct = _toy(rng, 300)
    rows = neighbors.CalibrationRows(unit, correct, claimed)
    far = neighbors.unit_rows(np.eye(16)[[5, 9]])  # (2, p): orthogonal to both directions
    similarities = neighbors.cosine_similarities(far, unit)
    for bandwidth in (0.002, 0.05):
        total, weighted = neighbors._residual_sums(similarities, rows, False, bandwidth)
        np.testing.assert_allclose(weighted / (total + 1.0), 0.0, atol=1e-3 if bandwidth == 0.05 else 1e-12)
    total, effective = neighbors.evidence(similarities, 0.002)
    assert np.all(total < 1e-100) and np.all(effective <= 2.0)


def test_leave_one_out_scores_each_calibration_row_without_itself():
    rng = np.random.default_rng(2)
    unit, _, claimed, correct = _toy(rng, 40)
    rows = neighbors.CalibrationRows(unit, correct, claimed)
    similarities = neighbors.cosine_similarities(unit, unit)
    total, weighted = neighbors._residual_sums(similarities, rows, True, 0.05)
    for row in (0, 17, 39):
        others = np.delete(np.arange(40), row)
        weights = neighbors.kernel_weights(similarities[row, others], 0.05)
        assert total[row] == pytest.approx(weights.sum())
        assert weighted[row] == pytest.approx(weights @ (correct[others] - claimed[others]))
    # A lone right answer among wrong ones is scored by its neighbors alone.
    lone = correct.copy()
    lone[:] = 0.0
    lone[3] = 1.0
    knn = neighbors._knn(similarities, neighbors.CalibrationRows(unit, lone, claimed), claimed, True, 5)
    assert knn[3] == 0.0


def test_claimed_correctness_is_the_argmax_or_label_set_probability():
    logits = np.array([[2.0, 0.0, -1.0], [0.0, 0.5, 0.2]])
    np.testing.assert_allclose(neighbors.claimed_correctness("multiclass", logits), softmax(logits, axis=1).max(axis=1))
    multilabel = np.array([[3.0, -2.0], [0.1, 0.4]])
    expected = np.prod(np.where(multilabel > 0, expit(multilabel), 1.0 - expit(multilabel)), axis=1)
    np.testing.assert_allclose(neighbors.claimed_correctness("multilabel", multilabel), expected)
    labels = np.array([[1, 0], [1, 0]])
    np.testing.assert_array_equal(neighbors.correctness("multilabel", labels, multilabel), [1.0, 0.0])
    np.testing.assert_array_equal(neighbors.correctness("regression", np.array([1.0, 2.0]), np.array([1.2, 3.0]), 0.5), [1.0, 0.0])
    with pytest.raises(ValueError, match="no per-row correctness"):
        neighbors.claimed_correctness("regression", logits)


def test_the_answer_rule_holds_its_target_and_answers_nothing_when_it_cannot():
    rng = np.random.default_rng(3)
    q = rng.random(4000)  # (r,)
    correct = (rng.random(4000) < q).astype(np.float64)  # (r,)
    rule = neighbors.answer_threshold(q, correct, 0.2)
    assert rule.error_bound <= 0.2 and 0.0 < rule.calibration_coverage < 1.0
    fresh_q = rng.random(4000)
    fresh_correct = (rng.random(4000) < fresh_q).astype(np.float64)
    answered = fresh_q >= rule.threshold
    assert 1.0 - fresh_correct[answered].mean() <= 0.2 + 0.02
    impossible = neighbors.answer_threshold(q, np.zeros(4000), 0.1)
    assert impossible.threshold == float("inf") and impossible.calibration_coverage == 0.0
    assert neighbors.clopper_pearson_upper(0, 0) == 1.0 and neighbors.clopper_pearson_upper(5, 5) == 1.0
    assert coverage_at_error(np.array([1.0, 1.0, 0.0, 1.0]), np.array([0.9, 0.8, 0.7, 0.1]), 0.0) == 0.5


def test_the_local_scale_widens_where_calibration_residuals_are_large():
    rng = np.random.default_rng(4)
    features, side = _two_directions(rng, 800)
    unit = neighbors.unit_rows(features)
    residuals = rng.normal(size=800) * np.where(side == 0, 0.2, 2.0)  # (r,)
    query_features, query_side = _two_directions(rng, 1000)
    query_unit = neighbors.unit_rows(query_features)
    query_error = np.abs(rng.normal(size=1000) * np.where(query_side == 0, 0.2, 2.0))  # (n,)
    scale = neighbors.local_scale(
        residuals, neighbors.cosine_similarities(unit, unit), neighbors.cosine_similarities(query_unit, unit)
    )
    assert scale.sigma[query_side == 1].mean() > 5.0 * scale.sigma[query_side == 0].mean()
    half_width = neighbors.normalized_interval_half_widths(residuals, scale.calibration_sigma, scale.sigma, 0.9)
    covered = query_error <= half_width
    assert covered.mean() >= 0.87
    assert covered[query_side == 0].mean() >= 0.85 and covered[query_side == 1].mean() >= 0.85


def test_trunk_features_are_saved_and_the_neighbors_are_scored(toy):  # noqa: redefined-unused
    root, directories, frames, _ = toy
    for data_name in ("toybin", "toyreg"):
        workflow.run_uncertainty_cell(_cell(root, directories[data_name], data_name, "projected", save_trunk_features=True))
    seed = load_seed_predictions(str(root / "runs" / "toybin" / "projected" / "final" / "seed_42.npz"))
    # The `projected` head's expanded MLP reduces to the smallest multiple of 256 covering the labels
    # (ProjectedProbe in probes/projected_probe.py), and the output layer reads that width, not projection_size.
    num_labels = frames["toybin"]["train"]["labels"].nunique()
    trunk_width = math.ceil(num_labels / 256) * 256  # f
    assert seed["test"].trunk.shape == (40, trunk_width) and seed["v_cal"].trunk.shape == (20, trunk_width)
    assert seed["test"].trunk.dtype == np.float32
    cell = json.loads((root / "runs" / "toybin" / "projected" / "final" / "cell.json").read_text())
    assert cell["uncertainty"]["save_trunk_features"] is True

    payload = evaluate_neighbors(str(root / "runs"), str(root / "neighbors"), replicates=20, seed=0)
    assert [cell["data_name"] for cell in payload["cells"]] == ["toybin", "toyreg"]
    binary, regression = payload["cells"]
    assert set(binary["estimators"]) == {"model", "knn", "joint", "residual"}
    assert set(regression["estimators"]) == {"model", "knn", "residual"}
    assert set(regression["intervals"]["0.9"]) == {"absolute", "local"}
    assert {"R1_aurc", "R2_log_loss", "R3_error_auroc", "R4_intervals"} <= set(payload["rules"])
    for name in ("neighbors.json", "neighbors.md", "estimators.csv", "answer_rules.csv", "risk_coverage.png", "identity_gap.png"):
        assert (root / "neighbors" / name).is_file(), name


def test_a_cell_without_trunk_features_is_refused(toy):
    root, directories, _, _ = toy
    workflow.run_uncertainty_cell(_cell(root, directories["toybin"], "toybin", "projected"))
    with pytest.raises(ValueError, match="save_trunk_features"):
        evaluate_neighbors(str(root / "runs"), str(root / "neighbors"), replicates=5)
