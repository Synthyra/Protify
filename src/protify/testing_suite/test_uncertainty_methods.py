"""The uncertainty workflow's parts, on synthetic predictions with known answers.

V_fit and V_cal must be disjoint, cover validation, keep its label mix, and not move with a run
seed. Temperature and variance maps must recover the scale they were generated with, for the
head, MC dropout, and the ensembles alike. Conformal sets and intervals must reach their
nominal coverage on exchangeable rows. Identity bins must put each boundary where the report
says it is.
"""

import json
import numpy as np
import pytest
import torch


try:
    from src.protify.uncertainty.calibration_split import (
        check_manifest,
        resolve_validation_roles,
        roles_manifest,
        split_validation,
        stratum_codes,
    )
    from src.protify.uncertainty.conformal_sets import classification_conformal_rows, regression_conformal_rows
    from src.protify.uncertainty.identity_bins import IDENTITY_BIN_NAMES, bin_rows, identity_codes, row_identity, sequence_digest
    from src.protify.uncertainty.methods import build_methods, fit_variance_calibration, mean_binary_entropy, probabilities, probability_logits
    from src.protify.uncertainty.metrics import brier_score, calibration_error, gaussian_interval, negative_log_likelihood, task_kind
    from src.protify.uncertainty.predictions import SplitPredictions, load_seed_predictions, save_seed_predictions
    from src.protify.uncertainty.report import score_method, write_json
except ImportError:
    from protify.uncertainty.calibration_split import (
        check_manifest,
        resolve_validation_roles,
        roles_manifest,
        split_validation,
        stratum_codes,
    )
    from protify.uncertainty.conformal_sets import classification_conformal_rows, regression_conformal_rows
    from protify.uncertainty.identity_bins import IDENTITY_BIN_NAMES, bin_rows, identity_codes, row_identity, sequence_digest
    from protify.uncertainty.methods import build_methods, fit_variance_calibration, mean_binary_entropy, probabilities, probability_logits
    from protify.uncertainty.metrics import brier_score, calibration_error, gaussian_interval, negative_log_likelihood, task_kind
    from protify.uncertainty.predictions import SplitPredictions, load_seed_predictions, save_seed_predictions
    from protify.uncertainty.report import score_method, write_json


def _validation(count: int = 101) -> dict[str, list]:
    rng = np.random.default_rng(3)
    labels = rng.choice(3, size=count, p=[0.6, 0.3, 0.1]).tolist()
    return {"seqs": [f"M{index:04d}" for index in range(count)], "labels": labels}


def test_the_carve_is_disjoint_complete_stratified_and_seed_independent():
    validation = _validation()
    labels = np.asarray(validation["labels"])
    torch.manual_seed(123)
    np.random.seed(123)
    first = resolve_validation_roles(validation["labels"], "singlelabel", fraction=0.5, seed=0)
    torch.manual_seed(999)
    np.random.seed(999)
    second = resolve_validation_roles(validation["labels"], "singlelabel", fraction=0.5, seed=0)

    assert np.array_equal(first.fit, second.fit) and np.array_equal(first.cal, second.cal)
    assert np.intersect1d(first.fit, first.cal).size == 0
    assert np.array_equal(np.union1d(first.fit, first.cal), np.arange(labels.size))
    assert abs(first.cal.size - labels.size * 0.5) <= 1
    for label in range(3):
        assert abs(np.sum(labels[first.cal] == label) - np.sum(labels == label) * 0.5) <= 1

    other = resolve_validation_roles(validation["labels"], "singlelabel", fraction=0.5, seed=1)
    assert not np.array_equal(other.cal, first.cal)


def test_regression_and_multilabel_carves_stratify_too():
    values = np.linspace(0.0, 10.0, 200)
    codes = stratum_codes(values.tolist(), "regression")
    assert np.bincount(codes).tolist() == [20] * 10

    multilabel = [[1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 0, 1]]
    assert stratum_codes(multilabel, "multilabel").tolist() == [0, 1, -1, 2]

    roles = resolve_validation_roles(values.tolist(), "regression", fraction=0.3, seed=0)
    assert roles.cal.size == 60
    assert np.bincount(codes[roles.cal]).tolist() == [6] * 10


def test_explicit_rows_are_taken_as_given_and_checked():
    validation = _validation(20)
    roles = resolve_validation_roles(validation["labels"], "singlelabel", explicit_rows=([4, 0, 2], [7, 9]))
    assert roles.source == "explicit"
    assert roles.fit.tolist() == [0, 2, 4] and roles.cal.tolist() == [7, 9]

    for explicit in (([0, 1], [1, 2]), ([0, 20], [1]), ([], [1]), ([0, 0], [1])):
        with pytest.raises(ValueError):
            resolve_validation_roles(validation["labels"], "singlelabel", explicit_rows=explicit)


def test_a_fit_that_never_loads_calibration_has_no_calibration_rows():
    validation = _validation(20)
    roles = resolve_validation_roles(
        validation["labels"], "singlelabel", explicit_rows=(np.arange(20), np.arange(20, 20)), require_calibration=False,
    )
    assert roles.source == "explicit" and roles.fit.tolist() == list(range(20)) and roles.cal.size == 0

    manifest = roles_manifest(roles, [f"digest_{row}" for row in range(20)], "singlelabel")
    assert manifest["validation_rows"] == len(manifest["v_fit_rows"]) == 20
    assert manifest["v_cal_rows"] == [] and manifest["v_cal_digests"] == []
    check_manifest(manifest, manifest)

    # The default still demands both roles, and a fit without V_cal can name none, carve none, and skip no V_fit.
    with pytest.raises(ValueError, match="V_cal rows are empty"):
        resolve_validation_roles(validation["labels"], "singlelabel", explicit_rows=([0, 1], []))
    with pytest.raises(ValueError, match="cannot name V_cal rows"):
        resolve_validation_roles(validation["labels"], "singlelabel", explicit_rows=([0, 1], [2]), require_calibration=False)
    with pytest.raises(ValueError, match="explicit V_fit rows"):
        resolve_validation_roles(validation["labels"], "singlelabel", require_calibration=False)
    with pytest.raises(ValueError, match="V_fit rows are empty"):
        resolve_validation_roles(validation["labels"], "singlelabel", explicit_rows=([], []), require_calibration=False)


def test_the_manifest_names_rows_by_digest_and_refuses_a_changed_resume():
    validation = _validation(30)
    v_fit, v_cal, manifest = split_validation(validation, "singlelabel", fraction=0.4, seed=0)
    assert len(v_fit["seqs"]) + len(v_cal["seqs"]) == 30
    assert manifest["v_cal_rows"] == sorted(manifest["v_cal_rows"])
    assert len(manifest["v_cal_digests"]) == len(v_cal["seqs"]) == 12

    check_manifest(manifest, manifest)
    changed = dict(manifest, v_cal_digests=list(reversed(manifest["v_cal_digests"])))
    with pytest.raises(ValueError, match="differ"):
        check_manifest(changed, manifest)


def _splits(outputs: dict[str, np.ndarray], labels: dict[str, np.ndarray], **fields) -> dict[str, SplitPredictions]:
    # outputs: (n, c) or (n,), labels: (n,) or (n, c)
    return {name: SplitPredictions(labels=labels[name], outputs=outputs[name], **fields.get(name, {})) for name in outputs}


def test_temperature_recovers_the_scale_the_labels_were_drawn_at():
    rng = np.random.default_rng(0)
    logits = {name: rng.normal(0.0, 3.0, size=(count, 4)) for name, count in (("v_fit", 20000), ("v_cal", 10), ("test", 10))}
    probabilities = {name: np.exp(value / 2.0) / np.exp(value / 2.0).sum(1, keepdims=True) for name, value in logits.items()}
    labels = {name: np.array([rng.choice(4, p=row) for row in value]) for name, value in probabilities.items()}

    methods = {method.name: method for method in build_methods({42: _splits(logits, labels)}, "projected", "multiclass")}
    assert methods["temperature"].runs[0].fitted["temperature"] == pytest.approx(2.0, abs=0.1)
    assert "ensemble" not in methods


def test_variance_calibration_recovers_a_and_b():
    rng = np.random.default_rng(0)
    variance = rng.uniform(0.5, 3.0, size=50000)
    labels = rng.normal(0.0, np.sqrt(2.0 * variance + 0.5))
    a, b = fit_variance_calibration(np.zeros_like(labels), variance, labels)
    assert a == pytest.approx(2.0, rel=0.08)
    assert b == pytest.approx(0.5, abs=0.15)

    constant_a, constant_b = fit_variance_calibration(np.zeros(4), None, np.array([1.0, -1.0, 2.0, -2.0]))
    assert (constant_a, constant_b) == (0.0, pytest.approx(2.5))


@pytest.mark.parametrize("kind", ["multiclass", "multilabel"])
def test_probability_logits_invert_probabilities(kind):
    rng = np.random.default_rng(0)
    probs = probabilities(rng.normal(0.0, 3.0, size=(50, 4)), kind)  # (50, 4)
    assert np.allclose(probabilities(probability_logits(probs, kind), kind), probs)
    # A float32 probability that saturated at 0 or 1 still has a finite logit.
    assert np.isfinite(probability_logits(np.array([[1.0, 0.0]], dtype=np.float32), kind)).all()


@pytest.mark.parametrize("kind", ["multiclass", "multilabel"])
def test_mc_dropout_and_ensemble_temperatures_recover_the_scale_of_their_probabilities(kind):
    """Both keep only mean probabilities, here twice as confident as the labels were drawn."""
    rng = np.random.default_rng(0)
    logits = {name: rng.normal(0.0, 3.0, size=(count, 4)) for name, count in (("v_fit", 20000), ("v_cal", 10), ("test", 10))}
    truth = {name: probabilities(value / 2.0, kind) for name, value in logits.items()}
    if kind == "multilabel":
        labels = {name: (rng.random(value.shape) < value).astype(np.float64) for name, value in truth.items()}
    else:
        labels = {name: np.array([rng.choice(4, p=row) for row in value]) for name, value in truth.items()}
    mc_fields = {name: {"mc_outputs": probabilities(value, kind)} for name, value in logits.items()}
    seeds = {seed: _splits(logits, labels, **mc_fields) for seed in (42, 43)}

    methods = {method.name: method for method in build_methods(seeds, "projected", kind)}
    for name in ("mc_dropout_temperature", "ensemble_temperature"):
        for run in methods[name].runs:
            assert run.fitted["temperature"] == pytest.approx(2.0, abs=0.1)
            calibrated = probabilities(logits["test"] / run.fitted["temperature"], kind)  # (10, 4)
            assert np.allclose(run.splits["test"].probs, calibrated, atol=1e-6)


def test_mc_dropout_and_ensemble_variance_maps_recover_a_and_b():
    """Labels drawn at variance 2 v + 0.5, around a mean both methods predict with spread v."""
    rng = np.random.default_rng(0)
    spread = {name: rng.uniform(0.5, 3.0, size=count) for name, count in (("v_fit", 50000), ("v_cal", 10), ("test", 10))}
    labels = {name: rng.normal(0.0, np.sqrt(2.0 * value + 0.5)) for name, value in spread.items()}
    mc_fields = {name: {"mc_outputs": np.zeros_like(value), "mc_variance": value} for name, value in spread.items()}
    # Two seeds at plus and minus sqrt(v): their Gaussian mixture has mean 0 and variance v.
    seeds = {
        seed: _splits({name: sign * np.sqrt(value) for name, value in spread.items()}, labels, **mc_fields)
        for seed, sign in ((42, 1.0), (43, -1.0))
    }

    methods = {method.name: method for method in build_methods(seeds, "projected", "regression")}
    for name in ("mc_dropout_temperature", "ensemble_temperature"):
        for run in methods[name].runs:
            assert run.fitted["a"] == pytest.approx(2.0, rel=0.08)
            assert run.fitted["b"] == pytest.approx(0.5, abs=0.15)


def _seed_predictions(head: str, kind: str, rng: np.random.Generator) -> dict[str, SplitPredictions]:
    """One seed's predictions on every split, with every field its head and task save."""
    splits = {}
    for name, count in (("v_fit", 40), ("v_cal", 30), ("test", 50)):
        if kind == "regression":
            labels = rng.normal(size=count)  # (n,)
            outputs = labels + rng.normal(scale=0.5, size=count)  # (n,)
        elif kind == "multilabel":
            labels = (rng.random((count, 3)) < 0.4).astype(np.float64)  # (n, 3)
            outputs = 2.0 * labels - 1.0 + rng.normal(size=(count, 3))  # (n, 3)
        else:
            classes = 2 if kind == "binary" else 3
            labels = rng.integers(0, classes, size=count)  # (n,)
            outputs = 2.0 * np.eye(classes)[labels] + rng.normal(size=(count, classes))  # (n, c)

        fields = {"nn_distance": rng.random(count)}
        if head == "sngp":
            fields["variance"] = rng.uniform(0.1, 1.0, size=count)
        elif kind == "regression":
            fields.update(mc_outputs=outputs + rng.normal(scale=0.1, size=count), mc_variance=rng.uniform(0.1, 1.0, size=count))
        else:
            fields["mc_outputs"] = probabilities(outputs + rng.normal(scale=0.3, size=outputs.shape), kind)
        splits[name] = SplitPredictions(labels=labels, outputs=outputs, **fields)
    return splits


@pytest.mark.parametrize(
    "head, kind, expected",
    [
        ("projected", "binary", ["plain", "plain_entropy", "plain_nn_distance", "temperature", "mc_dropout", "mc_dropout_temperature", "ensemble", "ensemble_temperature"]),
        ("projected", "multilabel", ["plain", "plain_nn_distance", "temperature", "mc_dropout", "mc_dropout_temperature", "ensemble", "ensemble_temperature"]),
        ("projected", "regression", ["plain", "plain_nn_distance", "temperature", "mc_dropout", "mc_dropout_temperature", "ensemble", "ensemble_temperature"]),
        ("sngp", "multiclass", ["sngp", "sngp_entropy", "sngp_nn_distance", "sngp_variance", "sngp_temperature", "sngp_ensemble", "sngp_ensemble_temperature"]),
        ("sngp", "regression", ["sngp", "sngp_nn_distance", "sngp_temperature", "sngp_ensemble", "sngp_ensemble_temperature"]),
    ],
)
def test_each_head_and_task_builds_its_methods_and_scores_them(head, kind, expected):
    rng = np.random.default_rng(0)
    seeds = {seed: _seed_predictions(head, kind, rng) for seed in (42, 43)}
    methods = build_methods(seeds, head, kind)
    assert [method.name for method in methods] == expected

    labels = {name: split.labels for name, split in seeds[42].items()}
    train_label_std = 1.0 if kind == "regression" else 0.0
    for method in methods:
        summary = score_method(method, kind, labels, (0.9,), train_label_std, None)
        assert summary["test"]["count"] == 50.0
        # Every method ranks rows except the plain regression head, which has no variance.
        if method.runs[0].splits["test"].uncertainty is not None:
            assert summary["test"]["aurc"] is not None


def test_conformal_sets_and_intervals_reach_their_coverage_on_exchangeable_rows():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(5), size=12000)  # (n, 5)
    labels = np.array([rng.choice(5, p=row) for row in probs])
    rows = classification_conformal_rows(probs[:2000], labels[:2000], probs[2000:], labels[2000:], (0.8, 0.9))
    for score in ("lac", "aps"):
        for level in (0.8, 0.9):
            assert rows[f"{score}_coverage@{level}"].mean() >= level - 0.015
    assert rows["lac_size@0.8"].mean() < rows["lac_size@0.9"].mean()

    sigma = rng.uniform(0.2, 2.0, size=12000)
    truth = rng.normal(0.0, sigma)
    mean = np.zeros_like(truth)
    for spread in (sigma, None):
        intervals = regression_conformal_rows(
            mean[:2000], truth[:2000], None if spread is None else spread[:2000],
            mean[2000:], truth[2000:], None if spread is None else spread[2000:],
            (0.9,), train_label_std=1.0,
        )
        assert intervals["conformal_coverage@0.9"].mean() >= 0.9 - 0.015
    # With sigma the widths follow it; without, every interval is one width.
    assert np.ptp(intervals["conformal_width@0.9"]) == 0


def test_identity_bins_put_every_boundary_where_the_report_says():
    identity = np.array([0.0, 0.1, 0.3, 0.49, 0.5, 0.7, 0.89, 0.9, 1.0])
    names = [IDENTITY_BIN_NAMES[code] for code in identity_codes(identity)]
    assert names == ["no_hit", "(0,0.3)", "[0.3,0.5)", "[0.3,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.7,0.9)", "[0.9,1.0]", "[0.9,1.0]"]
    assert sum(rows.size for rows in bin_rows(identity).values()) == identity.size
    with pytest.raises(ValueError):
        identity_codes(np.array([1.2]))

    lookup = {sequence_digest("MAAA"): 0.2, sequence_digest("MCCC"): 0.95}
    assert row_identity(lookup, [["MAAA", "MCCC"], ["MCCC", "MAAA"]]).tolist() == [0.95, 0.95]


def test_calibration_scores_behave_at_their_extremes():
    labels = np.array([0, 1, 1, 0])
    perfect = np.eye(2)[labels]  # (4, 2)
    assert brier_score("binary", labels, perfect) == 0.0
    assert calibration_error("binary", labels, perfect) == 0.0
    covered, width = gaussian_interval(np.zeros(3), np.zeros(3), np.ones(3), 0.9)
    assert covered.all() and np.allclose(width, 2 * 1.6448536)
    assert task_kind("singlelabel", 2) == "binary" and task_kind("sigmoid_regression", 1) == "regression"


def test_multilabel_scores_stay_finite_where_float32_probabilities_reach_zero_and_one():
    # An MC dropout mean of saturated sigmoids reaches exactly 1.0 and 0.0 in float32, where the
    # clip at 1 - 1e-12 rounds back to 1.0. EC's fit crashed on one such row.
    probs = np.array([[1.0, 0.0, 0.5], [0.9, 1.0, 0.0]], dtype=np.float32)  # (2, 3)
    labels = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])  # (2, 3)
    entropy = mean_binary_entropy(probs)  # (2,)
    assert np.isfinite(entropy).all()
    assert entropy[0] == pytest.approx(np.log(2.0) / 3.0, abs=1e-9)
    # The two wrong labels at 1.0 and 0.0 each cost -log(1e-12), not infinity.
    expected = (np.log(2.0) + np.log(10.0) + 2.0 * np.log(1e12)) / 6.0
    assert negative_log_likelihood("multilabel", labels, probs) == pytest.approx(expected, rel=1e-6)


def test_seed_predictions_round_trip(tmp_path):
    predictions = {
        "v_fit": SplitPredictions(labels=np.array([0, 1]), outputs=np.ones((2, 2)), nn_distance=np.array([0.5, 1.0])),
        "test": SplitPredictions(labels=np.array([1]), outputs=np.zeros((1, 2)), digests=np.array(["abc"])),
    }
    path = str(tmp_path / "seed_42.npz")
    save_seed_predictions(path, predictions)
    loaded = load_seed_predictions(path)
    assert set(loaded) == {"v_fit", "test"}
    assert np.array_equal(loaded["v_fit"].nn_distance, [0.5, 1.0])
    assert loaded["test"].digests.tolist() == ["abc"] and loaded["test"].variance is None


def test_a_method_is_scored_overall_per_seed_and_per_bin():
    rng = np.random.default_rng(0)
    seeds = {}
    labels = {name: rng.integers(0, 2, size=count) for name, count in (("v_fit", 40), ("v_cal", 30), ("test", 50))}
    for seed in (42, 43):
        outputs = {name: rng.normal(size=(value.size, 2)) + 2.0 * np.eye(2)[value] for name, value in labels.items()}
        seeds[seed] = _splits(outputs, labels)
    methods = {method.name: method for method in build_methods(seeds, "projected", "binary")}
    assert list(methods) == ["plain", "plain_entropy", "temperature", "ensemble", "ensemble_temperature"]

    bins = bin_rows(np.linspace(0.0, 1.0, 50))
    summary = score_method(methods["plain"], "binary", labels, (0.9,), 0.0, bins)
    assert summary["seeds"] == [42, 43] and len(summary["per_run"]) == 2
    assert {"mcc", "auroc", "ece", "nll", "brier", "aurc", "error_auroc", "lac_coverage@0.9", "aps_size@0.9"} <= set(summary["test"])
    assert set(summary["identity_bins"]) == set(IDENTITY_BIN_NAMES)
    assert sum(scores["count"] for scores in summary["identity_bins"].values()) == 50

    entropy = score_method(methods["plain_entropy"], "binary", labels, (0.9,), 0.0, None)
    assert "mcc" not in entropy["test"] and "aurc" in entropy["test"]


def test_report_json_refuses_a_nan(tmp_path):
    path = tmp_path / "report.json"
    write_json(str(path), {"value": 1.0})
    assert json.loads(path.read_text()) == {"value": 1.0}
    with pytest.raises(ValueError):
        write_json(str(path), {"value": float("nan")})
