"""Hand-derived canonical metrics, including unsupported targets and ranking definitions."""

import numpy as np
import pytest

from protify.uncertainty.methods import Predictive
from protify.uncertainty.metric_contracts import PRIMARY, canonical_task_metrics, fmax_counts, fmax_from_counts, metric_contract, scored_term_columns
from protify.uncertainty.metrics import task_metrics
from protify.uncertainty.report import score_rows


def binary(labels, positive):
    probability = np.column_stack((1 - np.asarray(positive), positive))
    return canonical_task_metrics("binary", np.asarray(labels), probs=probability)


def test_average_precision_is_not_trapezoidal_pr_area():
    scores = binary([1, 0, 1], [0.9, 0.8, 0.7])
    assert scores["average_precision"] == pytest.approx(5 / 6)
    assert scores["pr_auc_trapezoidal"] == pytest.approx(19 / 24)
    assert scores["auroc"] == pytest.approx(0.5)
    # All predictions are positive; MCC's prediction variance is zero.
    assert scores["mcc"] is None
    assert scores["f1"] == pytest.approx(0.8)


def test_tied_scores_are_rank_groups_not_input_order():
    first = binary([1, 0, 0, 1], [0.8, 0.8, 0.3, 0.3])
    second = binary([0, 1, 1, 0], [0.8, 0.8, 0.3, 0.3])
    assert first == second
    assert first["average_precision"] == 0.5
    assert first["auroc"] == 0.5
    assert first["pr_auc_trapezoidal"] == pytest.approx(0.625)


def test_mcc_undefined_stays_null_only_in_the_canonical_contract():
    probs = np.array([[0.9, 0.1], [0.7, 0.3]])
    assert canonical_task_metrics("binary", np.zeros(2), probs)["mcc"] is None
    assert task_metrics("binary", np.zeros(2), probs)["mcc"] == 0.0
    assert binary([0, 0], [0.1, 0.2])["average_precision"] is None
    assert binary([1, 1], [0.1, 0.2])["auroc"] is None


def test_multilabel_mask_and_macro_micro_denominators():
    labels = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
    mask = np.array([[True, True, False], [True, True, False], [True, False, False]])
    probs = np.array([[0.9, 0.2, np.nan], [0.1, 0.8, np.nan], [0.1, np.nan, np.nan]])
    task_scores = canonical_task_metrics("multilabel", labels, probs, label_mask=mask)
    # Across known entries: tp=1, fp=1, fn=2. Label F1s are 2/3 and 0.
    assert task_scores["f1"] == task_scores["micro_f1"] == pytest.approx(0.4)
    assert task_scores["macro_f1"] == pytest.approx(1 / 3)
    assert task_scores["macro_f1_defined_labels"] == 2
    assert task_scores["macro_average_precision_defined_labels"] == 2
    assert task_scores["macro_auroc_defined_labels"] == 2
    altered = probs.copy()
    altered[~mask] = 0.999
    assert canonical_task_metrics("multilabel", labels, altered, label_mask=mask) == task_scores


def test_f1_threshold_cannot_be_selected_from_scored_labels():
    labels, probs = np.array([[1], [0]]), np.array([[0.4], [0.3]])
    assert canonical_task_metrics("multilabel", labels, probs)["f1"] == 0.0
    assert task_metrics("multilabel", labels, probs)["f1"] > 0.9
    assert canonical_task_metrics("multilabel", np.array([[0], [0]]), probs)["f1"] is None


def test_multiclass_uses_argmax_and_counts_only_defined_rankings():
    labels = np.array([0, 1, 1])
    probs = np.array([[0.6, 0.3, 0.1], [0.4, 0.5, 0.1], [0.6, 0.3, 0.1]])
    scores = canonical_task_metrics("multiclass", labels, probs)
    assert scores["accuracy"] == pytest.approx(2 / 3)
    assert scores["macro_f1"] == pytest.approx(2 / 3)
    assert scores["macro_f1_defined_labels"] == 2
    assert scores["macro_auroc_defined_labels"] == 2
    assert scores["mcc"] == pytest.approx(0.5)


def test_regression_errors_remain_in_declared_target_scale():
    task_scores = canonical_task_metrics("regression", np.array([0., 1., 2., 99.]), mean=np.array([0., 2., 1., np.nan]), label_mask=np.array([True, True, True, False]))
    assert task_scores == pytest.approx({"spearman": 0.5, "r2": 0.0, "mse": 2 / 3, "rmse": np.sqrt(2 / 3), "mae": 2 / 3})
    constant = canonical_task_metrics("regression", np.array([1., 1.]), mean=np.array([2., 3.]))
    assert constant["spearman"] is None and constant["r2"] is None


def test_report_adds_calibration_without_replacing_strict_scores():
    labels, probs = np.array([0, 1]), np.array([[0.8, 0.2], [0.4, 0.6]])
    scores = score_rows("binary", Predictive(probs=probs), labels, (), canonical=True)
    assert scores["brier"] == pytest.approx(0.1)
    assert scores["nll"] == pytest.approx(-np.log(0.48) / 2)
    assert scores["ece"] == pytest.approx(0.3)
    assert scores["average_precision"] == 1.0


@pytest.mark.parametrize("probs", [np.array([[0.9, 0.9]]), np.array([[-0.1, 1.1]]), np.array([[np.nan, np.nan]])])
def test_malformed_observed_probabilities_are_refused(probs):
    with pytest.raises(ValueError, match="probabilities"):
        canonical_task_metrics("binary", np.array([1]), probs)


def primary_cases():
    """Each kind's scoring inputs, with masks, ties, and undefined primaries."""
    generator = np.random.default_rng(7)
    rows, width = 40, 6
    labels = (generator.random((rows, width)) < 0.3).astype(np.int64)
    mask = generator.random((rows, width)) < 0.7
    mask[:, 5] = False  # a label with no observed entry
    scores = generator.random((rows, width))
    scores[:, 0] = np.round(scores[:, 0], 1)  # ties
    classes = generator.integers(0, 3, rows)
    class_probs = generator.dirichlet(np.ones(3), rows)
    return {
        "multilabel": ("multilabel", labels, scores, None, mask),
        "multilabel_unmasked": ("multilabel", labels, scores, None, None),
        "multilabel_undefined": ("multilabel", np.zeros((rows, width), dtype=np.int64), np.full((rows, width), 0.1), None, None),
        "binary": ("binary", classes % 2, np.column_stack((1 - class_probs[:, 0], class_probs[:, 0])), None, None),
        "binary_undefined": ("binary", np.zeros(4, dtype=np.int64), np.array([[0.9, 0.1]] * 4), None, None),
        "multiclass": ("multiclass", classes, class_probs, None, generator.random(rows) < 0.8),
        "regression": ("regression", generator.normal(size=rows), None, generator.normal(size=rows), generator.random(rows) < 0.8),
        "regression_constant": ("regression", np.ones(rows), None, generator.normal(size=rows), None),
    }


@pytest.mark.parametrize("case", sorted(primary_cases()))
def test_the_primary_alone_equals_the_primary_of_every_metric(case):
    kind, labels, probs, mean, mask = primary_cases()[case]
    every = canonical_task_metrics(kind, labels, probs, mean, mask)
    primary = canonical_task_metrics(kind, labels, probs, mean, mask, primary_only=True)
    name = PRIMARY[kind]
    assert primary[name] == every[name]
    assert set(primary) <= set(every)


@pytest.mark.parametrize("kind", ["multilabel", "binary"])
def test_the_primary_alone_refuses_what_every_metric_refuses(kind):
    labels = np.array([[1, 0]] if kind == "multilabel" else [1])
    probs = np.array([[0.9, 1.5]] if kind == "multilabel" else [[0.9, 0.9]])
    with pytest.raises(ValueError, match="probabilities"):
        canonical_task_metrics(kind, labels, probs, primary_only=True)
    with pytest.raises(ValueError, match="observed targets"):
        canonical_task_metrics(kind, labels, probs, label_mask=np.zeros(labels.shape, dtype=bool), primary_only=True)


def test_the_primary_alone_never_ranks_a_label(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a ranking metric was computed")
    monkeypatch.setattr("protify.uncertainty.metric_contracts.binary_ranking_metrics", refuse)
    kind, labels, probs, mean, mask = primary_cases()["multilabel"]
    assert canonical_task_metrics(kind, labels, probs, mean, mask, primary_only=True)["f1"] is not None


def scored_terms_split():
    """Twenty proteins, two terms with ten positives each and a third with three, whose probabilities the third's rank would spoil."""
    labels = np.zeros((20, 3), dtype=np.int64)
    labels[:10, 0] = 1  # term A: proteins 0 to 9
    labels[10:, 1] = 1  # term B: proteins 10 to 19
    labels[:3, 2] = 1  # term C: three positives, below the minimum
    probs = np.zeros((20, 3))
    probs[:10, 0], probs[10:, 0] = 0.85, 0.35
    probs[:10, 1], probs[10:15, 1], probs[15:, 1] = 0.35, 0.55, 0.2
    probs[:, 2] = 1.0 - labels[:, 2] * 0.9  # term C is ranked backwards and called positive for every negative
    return labels, probs


def test_fmax_and_average_precision_read_only_terms_with_enough_positives():
    labels, probs = scored_terms_split()
    task_scores = canonical_task_metrics("multilabel", labels, probs)
    assert task_scores["scored_terms"] == 2
    # Above 0.35 and up to 0.55 fifteen proteins predict one hit each and recall is 15 of 20: F = 2 * 1 * 0.75 / 1.75. No other threshold beats it.
    assert task_scores["fmax"] == pytest.approx(6 / 7)
    # All 40 entries: ten hits at 0.85, five at 0.55, twenty negatives at 0.35, five hits at 0.2 with twenty negatives above them.
    assert task_scores["scored_micro_aupr"] == pytest.approx(0.5 + 0.25 + 0.5 * 0.25)
    assert task_scores["scored_macro_aupr"] == pytest.approx((1.0 + 0.75) / 2)
    # The unscored term still counts in the unrestricted macro, which is why the card reads the scored values.
    assert task_scores["macro_average_precision_defined_labels"] == 3


def test_a_split_with_no_term_of_ten_positives_scores_nothing_and_imputes_nothing():
    labels, probs = scored_terms_split()
    task_scores = canonical_task_metrics("multilabel", labels[:, 2:], probs[:, 2:])
    assert task_scores["scored_terms"] == 0
    assert task_scores["fmax"] is None and task_scores["scored_micro_aupr"] is None and task_scores["scored_macro_aupr"] is None


def test_scored_terms_count_observed_positives_only():
    labels, probs = scored_terms_split()
    mask = np.ones(labels.shape, dtype=bool)
    mask[:2, 0] = False  # two of term A's ten positives are unobserved, so A drops below the minimum
    probs = np.where(mask, probs, np.nan)
    task_scores = canonical_task_metrics("multilabel", labels, probs, label_mask=mask)
    assert task_scores["scored_terms"] == 1
    assert task_scores["scored_macro_aupr"] == pytest.approx(0.75)


def test_fmax_from_draw_counts_equals_fmax_of_the_repeated_proteins():
    labels, probs = scored_terms_split()
    scored = scored_term_columns(labels, np.ones(labels.shape, dtype=bool))
    kept_labels, kept_probs = labels[:, scored], probs[:, scored]
    mask = np.ones(kept_labels.shape, dtype=bool)
    draws = np.arange(20) % 3  # each protein is drawn zero, one or two times
    repeated = np.repeat(np.arange(20), draws)
    weighted = fmax_from_counts(fmax_counts(kept_labels, kept_probs, mask), draws)
    assert weighted == pytest.approx(fmax_from_counts(fmax_counts(kept_labels[repeated], kept_probs[repeated], mask[repeated])))
    assert fmax_from_counts(fmax_counts(kept_labels, kept_probs, mask), np.zeros(20)) is None


def test_only_a_multilabel_contract_names_the_scored_term_rule():
    release = {"manifest_sha256": "a" * 64, "endpoint": "go", "units": "labels", "target_transform": {"name": "identity"}, "label_vocabulary": ["a", "b"]}
    contract = metric_contract({**release, "task_type": "multilabel"})
    assert "at_least_10_observed_positives" in contract["scored_terms"] and {"fmax", "scored_aupr"} <= set(contract)
    assert not {"scored_terms", "fmax", "scored_aupr"} & set(metric_contract({**release, "task_type": "binary"}))


def test_a_multilabel_selection_on_fmax_counts_every_term_with_a_positive_and_the_default_stays_f1():
    labels, probs = scored_terms_split()
    selected = canonical_task_metrics("multilabel", labels, probs, primary_only=True, selection="fmax")
    assert set(selected) == {"fmax"}
    # Term C has three positives, below the reporting minimum of ten, yet it takes part in the selection score of a small split.
    assert selected["fmax"] != pytest.approx(canonical_task_metrics("multilabel", labels, probs)["fmax"])
    assert set(canonical_task_metrics("multilabel", labels, probs, primary_only=True)) == {"f1"}
    assert set(canonical_task_metrics("multilabel", labels, probs, primary_only=True, selection="f1")) == {"f1"}
    assert canonical_task_metrics("multilabel", np.zeros((4, 2), dtype=int), np.full((4, 2), 0.5), primary_only=True, selection="fmax")["fmax"] is None


@pytest.mark.parametrize("kind,selection", [("binary", "fmax"), ("multilabel", "mcc"), ("regression", "f1")])
def test_a_selection_must_be_one_the_task_kind_offers(kind, selection):
    with pytest.raises(ValueError, match="selects on one of"):
        canonical_task_metrics(kind, np.array([[0, 1]]), np.array([[0.5, 0.5]]), primary_only=True, selection=selection)
