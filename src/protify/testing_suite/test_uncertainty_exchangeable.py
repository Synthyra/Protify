"""Scheme T against the unchanged methods, and the cases only scheme T meets.

`exchangeable.py` scores every repeat at once. On a few repeats it must equal what
`build_methods` and `score_method` give when each repeat's rows are sliced out and handed to
them unchanged: exactly for a method with nothing fitted, and, for a calibrated method, exactly
when given the map `build_methods` fitted, while its own batched map fits the repeat's T_fit
rows at least as well as that map.

AURC is checked on its own as well, since both paths now give rows tied in uncertainty their
mean loss: the result must be the mean over every order of the tied rows, the same from draw
counts as from the drawn list, the mean loss under a constant uncertainty, and, without ties,
what it was before, bit for bit.
"""

import numpy as np
import pytest

from itertools import permutations

from foundry.metrics import area_under_risk_coverage


try:
    from src.protify.uncertainty import weighted_metrics
    from src.protify.uncertainty.batched_calibration import temperature_objective, variance_objective
    from src.protify.uncertainty.conformal_sets import SPREAD_FLOOR_FRACTION
    from src.protify.uncertainty.exchangeable import (
        MethodScores,
        RunPlan,
        exchangeable_scores,
        fit_maps,
        identity_layout,
        method_scores,
        mondrian_bin_groups,
        plan_runs,
        reference_splits,
        score_plan,
    )
    from src.protify.uncertainty.identity_bins import IDENTITY_BIN_NAMES, bin_rows
    from src.protify.uncertainty.methods import build_methods, probabilities
    from src.protify.uncertainty.metrics import CLASSIFICATION_KINDS, selective_metrics, tie_averaged_losses
    from src.protify.uncertainty.predictions import SplitPredictions
    from src.protify.uncertainty.report import score_method
    from src.protify.uncertainty.resampling import exchangeable_design
    from src.protify.uncertainty.weighted_metrics import weighted_aurc
except ImportError:
    from protify.uncertainty import weighted_metrics
    from protify.uncertainty.batched_calibration import temperature_objective, variance_objective
    from protify.uncertainty.conformal_sets import SPREAD_FLOOR_FRACTION
    from protify.uncertainty.exchangeable import (
        MethodScores,
        RunPlan,
        exchangeable_scores,
        fit_maps,
        identity_layout,
        method_scores,
        mondrian_bin_groups,
        plan_runs,
        reference_splits,
        score_plan,
    )
    from protify.uncertainty.identity_bins import IDENTITY_BIN_NAMES, bin_rows
    from protify.uncertainty.methods import build_methods, probabilities
    from protify.uncertainty.metrics import CLASSIFICATION_KINDS, selective_metrics, tie_averaged_losses
    from protify.uncertainty.predictions import SplitPredictions
    from protify.uncertainty.report import score_method
    from protify.uncertainty.resampling import exchangeable_design
    from protify.uncertainty.weighted_metrics import weighted_aurc


ROWS = 240
REPEATS = 3
SEEDS = (0, 1, 2)
LEVELS = (0.8, 0.9)
LABEL_TYPES = {"binary": "singlelabel", "multiclass": "singlelabel", "multilabel": "multilabel", "regression": "regression"}
CLASSES = {"binary": 2, "multiclass": 4, "multilabel": 5}


def _labels(kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "regression":
        return rng.normal(size=ROWS).astype(np.float32)  # (n,)
    if kind == "multilabel":
        labels = (rng.random((ROWS, CLASSES[kind])) < 0.3).astype(np.float32)  # (n, c)
        labels[labels.sum(axis=1) == 0, 0] = 1.0
        return labels  # (n, c)
    return rng.integers(0, CLASSES[kind], size=ROWS)  # (n,)


def _test_splits(kind: str, head: str, labels: np.ndarray, rng: np.random.Generator, scale: float = 1.0) -> dict[int, SplitPredictions]:
    """Each seed's test predictions: informative, noisy, with every field the head saves."""
    # labels: (n,) or (n, c)
    splits = {}
    for seed in SEEDS:
        distance = rng.gamma(2.0, size=ROWS).astype(np.float32)  # (n,)
        if kind == "regression":
            outputs = (labels + 0.6 * rng.normal(size=ROWS)).astype(np.float32)  # (n,)
            if head == "sngp":
                extra = {"variance": rng.uniform(0.2, 0.6, size=ROWS).astype(np.float32)}
            else:
                extra = {
                    "mc_outputs": (outputs + 0.05 * rng.normal(size=ROWS)).astype(np.float32),
                    "mc_variance": rng.uniform(0.05, 0.3, size=ROWS).astype(np.float32),
                }
        else:
            target = labels if kind == "multilabel" else np.eye(CLASSES[kind])[labels]  # (n, c)
            outputs = (scale * (2.0 * target - 1.0) + rng.normal(size=target.shape)).astype(np.float32)  # (n, c)
            if head == "sngp":
                extra = {"variance": rng.uniform(0.1, 0.9, size=ROWS).astype(np.float32)}
            else:
                noisy = outputs + 0.3 * rng.normal(size=outputs.shape).astype(np.float32)  # (n, c)
                extra = {"mc_outputs": probabilities(noisy, kind).astype(np.float32)}
        splits[seed] = SplitPredictions(labels=labels, outputs=outputs, nn_distance=distance, **extra)
    return splits


def _assert_close(value: float, expected: float | None, name: str) -> None:
    if expected is None or not np.isfinite(expected):
        assert not np.isfinite(value), name
    else:
        assert float(value) == pytest.approx(expected, rel=1e-6, abs=1e-9), name


def _assert_same_scores(mine: MethodScores, repeat: int, reference: dict) -> int:
    """Every score both sides have, overall and per identity bin; returns how many were compared."""
    compared = 0
    for metric, values in mine.overall.items():
        if metric in reference["test"]:
            _assert_close(values[repeat], reference["test"][metric], f"{mine.name} {metric}")
            compared += 1
    for metric, values in mine.by_bin.items():
        for index, bin_name in enumerate(IDENTITY_BIN_NAMES):
            expected = reference.get("identity_bins", {}).get(bin_name, {})
            if metric in expected:
                _assert_close(values[repeat, index], expected[metric], f"{mine.name} {metric} in {bin_name}")
                compared += 1
    return compared


def _assert_fits_at_least_as_well(plan: RunPlan, kind: str, labels: np.ndarray, fit_rows: np.ndarray) -> None:
    """The batched map's objective on the repeat's T_fit rows is at most that of the map build_methods fitted."""
    # fit_rows: (1, m_fit)
    # labels: (n,) or (n, c)
    batched = fit_maps(plan, kind, labels, fit_rows)  # each (1,)
    rows = np.repeat(fit_rows, 2, axis=0)  # (2, m_fit)
    if kind == "regression":
        a = np.array([batched["a"][0], plan.fitted["a"]])  # (2,)
        b = np.array([batched["b"][0], plan.fitted["b"]])  # (2,)
        objectives = variance_objective(plan.base.mean, plan.base.variance, labels, rows, a, b)  # (2,)
    else:
        temperatures = np.array([batched["temperature"][0], plan.fitted["temperature"]])  # (2,)
        objectives = temperature_objective(plan.logits, labels, kind in CLASSIFICATION_KINDS, rows, temperatures)  # (2,)
    assert objectives[0] <= objectives[1] + 1e-6 * max(1.0, abs(objectives[1])), plan.method


@pytest.mark.parametrize(
    ("kind", "head"),
    [("binary", "projected"), ("multiclass", "sngp"), ("multilabel", "projected"), ("regression", "projected"), ("regression", "sngp")],
)
def test_scheme_t_equals_the_unchanged_methods_on_a_few_repeats(kind, head):
    rng = np.random.default_rng(7)
    labels = _labels(kind, rng)
    test_splits = _test_splits(kind, head, labels, rng)
    identity = rng.uniform(0.0, 1.0, size=ROWS)  # (n,)
    identity[rng.random(ROWS) < 0.2] = 0.0
    design = exchangeable_design(labels, LABEL_TYPES[kind], "toy", REPEATS, seed=0)
    train_label_std = 1.0 if kind == "regression" else 0.0
    repeat_scores = exchangeable_scores(test_splits, head, kind, design, LEVELS, train_label_std, identity)
    assert all(check["passed"] for check in repeat_scores.checks), [check for check in repeat_scores.checks if not check["passed"]]

    scored = {method.name: method for method in repeat_scores.methods}
    fit_labels = labels.astype(np.int64) if kind in CLASSIFICATION_KINDS else labels
    layout = identity_layout(identity, design.conformal_rows)
    spread_floor = SPREAD_FLOOR_FRACTION * train_label_std
    compared = 0
    for repeat in range(REPEATS):
        splits = reference_splits(test_splits, design, repeat)
        methods = build_methods(splits, head, kind)
        plans = plan_runs(methods, splits, head, kind)
        roles = splits[SEEDS[0]]
        role_labels = {role: roles[role].labels for role in ("v_fit", "v_cal", "test")}
        drawn = np.repeat(np.arange(ROWS), design.eval_counts[repeat])  # (n - n // 2,)
        test_bins = None if kind == "multilabel" else bin_rows(identity[drawn])
        window = slice(repeat, repeat + 1)
        for method in methods:
            reference = score_method(method, kind, role_labels, LEVELS, train_label_std, test_bins)
            runs = [plan for plan in plans if plan.method == method.name]
            given = [
                score_plan(
                    plan, kind, fit_labels, design, window,
                    {key: np.array([value]) for key, value in plan.fitted.items()} if plan.calibrated else None,
                    LEVELS, spread_floor, layout,
                )
                for plan in runs
            ]
            compared += _assert_same_scores(method_scores(runs, given)[0], 0, reference)
            if not any(plan.calibrated for plan in runs):
                # Nothing is fitted, so what scheme T reported for this repeat is the same.
                compared += _assert_same_scores(scored[method.name], repeat, reference)
            for plan in runs:
                if plan.calibrated:
                    _assert_fits_at_least_as_well(plan, kind, fit_labels, design.fit_rows[window])
    assert compared > 100


def test_a_float32_probability_of_one_leaves_every_score_finite_and_equal_to_score_method():
    rng = np.random.default_rng(8)
    labels = _labels("multilabel", rng)
    # Logits near 20 round to a float32 probability of exactly 1.0, as a saturated head gives.
    test_splits = _test_splits("multilabel", "projected", labels, rng, scale=20.0)
    assert (probabilities(test_splits[SEEDS[0]].outputs, "multilabel") == 1.0).any()
    design = exchangeable_design(labels, "multilabel", "toy", REPEATS, seed=0)
    repeat_scores = exchangeable_scores(test_splits, "projected", "multilabel", design, LEVELS, 0.0, None)

    for method in repeat_scores.methods:
        for metric in ("mean_uncertainty", "aurc", *(() if method.score_only else ("nll", "brier"))):
            assert np.isfinite(method.overall[metric]).all(), f"{method.name} {metric}"
    references = [check for check in repeat_scores.checks if check["check"] == "repeat 0 against score_method"]
    assert len(references) == len(repeat_scores.methods)
    assert all(check["passed"] for check in references), [check for check in references if not check["passed"]]


def test_mondrian_groups_merge_identity_bins_until_each_holds_the_minimum():
    held = [150, 30, 80, 120, 10, 5]  # T_conformal rows in each identity bin
    codes = np.repeat(np.arange(len(held)), held)  # (n,)
    every_row = np.arange(codes.shape[0])[None]  # (1, n)
    # 150 closes a group, 30 + 80 another, 120 a third, and the short 10 + 5 join it.
    assert mondrian_bin_groups(codes, every_row, minimum=100).tolist() == [[0, 1, 1, 3, 3, 3]]
    assert mondrian_bin_groups(codes, every_row, minimum=1000).tolist() == [[0, 0, 0, 0, 0, 0]]


def _every_order_aurc(losses: np.ndarray, uncertainty: np.ndarray) -> float:
    """foundry's AURC averaged over every order of the rows, which puts each tie group in each of its orders equally often."""
    # losses: (m,); uncertainty: (m,)
    orders = [list(order) for order in permutations(range(losses.shape[0]))]
    return float(np.mean([area_under_risk_coverage(losses[order], uncertainty[order]) for order in orders]))


def test_aurc_is_the_mean_over_every_order_of_the_tied_rows():
    losses = np.array([0.0, 1.0, 1.0, 0.0, 0.5, 2.0, 0.25])  # (7,)
    # Rows 1, 3, and 5 tie, and so do rows 0, 2, and 6; row 4 stands alone.
    uncertainty = np.array([0.3, 0.1, 0.3, 0.1, 0.2, 0.1, 0.3])  # (7,)
    expected = _every_order_aurc(losses, uncertainty)
    # foundry keeps tied rows in the order given, so the stored order alone moves its value.
    assert area_under_risk_coverage(losses, uncertainty) != pytest.approx(expected)
    assert selective_metrics("regression", losses, uncertainty)["aurc"] == pytest.approx(expected, rel=1e-12)
    assert weighted_aurc(np.ones((1, 7), dtype=np.int64), losses, uncertainty)[0] == pytest.approx(expected, rel=1e-12)

    # Drawn with repetition, a row drawn twice is two tied rows, and a row left out is none.
    counts = np.array([2, 1, 0, 1, 1, 2, 0])  # (7,)
    drawn = np.repeat(np.arange(7), counts)  # (7,)
    expected_drawn = _every_order_aurc(losses[drawn], uncertainty[drawn])
    assert weighted_aurc(counts[None], losses, uncertainty)[0] == pytest.approx(expected_drawn, rel=1e-12)


@pytest.mark.parametrize("per_draw", [False, True])
def test_weighted_aurc_is_the_tie_averaged_aurc_of_each_drawn_list(per_draw):
    rng = np.random.default_rng(3)
    draws, rows = 40, 30
    # Misclassification losses and continuous ones, ranked by four levels, so every draw has large tie groups.
    losses = np.where(rng.random(rows) < 0.5, rng.integers(0, 2, size=rows), rng.exponential(size=rows))  # (n,)
    uncertainty = rng.integers(0, 4, size=(draws, rows) if per_draw else rows) / 4.0  # (b, n) or (n,)
    counts = rng.poisson(1.0, size=(draws, rows))  # (b, n)
    left_out = np.atleast_2d(uncertainty)[0] == 0.25  # (n,): a tie group draw 0 leaves out entirely
    counts[0, left_out] = 0
    assert left_out.any() and (counts.sum(axis=1) >= 2).all()

    values = weighted_aurc(counts, losses, uncertainty)  # (b,)
    expected = []
    for draw in range(draws):
        drawn = np.repeat(np.arange(rows), counts[draw])  # (m,)
        draw_uncertainty = np.atleast_2d(uncertainty)[draw if per_draw else 0]  # (n,)
        expected.append(selective_metrics("regression", losses[drawn], draw_uncertainty[drawn])["aurc"])
    np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-14)

    # The rows draw 0 leaves out never reach its sums, even with a loss that is not finite.
    unread = np.where(left_out, np.inf, losses)  # (n,)
    assert weighted_aurc(counts[:1], unread, np.atleast_2d(uncertainty)[:1])[0] == values[0]


def test_a_constant_uncertainty_scores_the_mean_loss():
    rng = np.random.default_rng(4)
    losses = rng.exponential(size=50)  # (n,)
    constant = np.full(50, 0.7)  # (n,)
    assert selective_metrics("regression", losses, constant)["aurc"] == pytest.approx(losses.mean(), rel=1e-12)

    counts = rng.poisson(1.0, size=(20, 50))  # (b, n)
    drawn_mean = (counts * losses).sum(axis=1) / counts.sum(axis=1)  # (b,)
    np.testing.assert_allclose(weighted_aurc(counts, losses, constant), drawn_mean, rtol=1e-12)


def test_untied_rows_leave_aurc_as_it_was_bit_for_bit():
    rng = np.random.default_rng(5)
    losses = rng.exponential(size=200)  # (n,)
    uncertainty = rng.normal(size=200)  # (n,)
    assert np.unique(uncertainty).shape[0] == 200
    assert np.array_equal(tie_averaged_losses(losses, uncertainty), losses)
    assert selective_metrics("regression", losses, uncertainty)["aurc"] == area_under_risk_coverage(losses, uncertainty)

    # A row drawn several times ties only with its own copies, so every draw scores as it did
    # with the averaging taken out, for a shared uncertainty and for one per draw.
    counts = rng.poisson(1.0, size=(50, 200))  # (b, n)
    for draw_uncertainty in (uncertainty, rng.normal(size=(50, 200))):
        averaged = weighted_aurc(counts, losses, draw_uncertainty)  # (b,)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(weighted_metrics, "_tie_group_means", lambda weights, ranked_losses, sorted_values: ranked_losses)
            unaveraged = weighted_aurc(counts, losses, draw_uncertainty)  # (b,)
        assert np.array_equal(averaged, unaveraged)
