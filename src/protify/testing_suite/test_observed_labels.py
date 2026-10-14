"""Observed targets survive batch construction, optimization, and prediction persistence."""

import copy
import numpy as np
import pytest
import torch

from dataclasses import replace
from functools import partial
from datasets import Dataset
from torch import nn
from transformers import EvalPrediction

from protify import metrics as metrics_module
from protify.data.branch_features import JoinedBranchCollator, JoinedBranchDataset, JoinedBranchEmbeddings
from protify.metrics import get_compute_metrics
from protify.probes.losses import compute_probe_loss, get_loss_fct
from protify.probes.projected_probe import ProjectedProbe, ProjectedProbeConfig
from protify.uncertainty import methods as uncertainty_methods
from protify.uncertainty.methods import build_methods, probabilities
from protify.uncertainty.metrics import brier_score, negative_log_likelihood, row_losses
from protify.uncertainty.predictions import SplitPredictions, collect_predictions, load_seed_predictions, save_seed_predictions
from protify.uncertainty.report import conformal_rows, label_support, score_method
from protify.uncertainty.workflow import agreed_labels


def probe(task_type="multilabel", **overrides):
    settings = {"input_size": 6, "projection_size": 8, "dropout": 0.0, "task_type": task_type, "num_labels": 3 if task_type in {"singlelabel", "multilabel"} else 1}
    settings.update(overrides)
    return ProjectedProbe(ProjectedProbeConfig(**settings)).eval()


@pytest.mark.parametrize("multiclass", [False, True])
def test_temperature_does_not_select_all_observed_targets_at_once(multiclass, monkeypatch):
    class BoundedSelection(np.ndarray):
        def __getitem__(self, key):
            if isinstance(key, np.ndarray) and key.dtype == np.bool_ and self.size > 128:
                raise AssertionError("full-array boolean selection")
            return super().__getitem__(key)

    generator = np.random.default_rng(71)
    logits = generator.normal(size=(257, 7))
    labels = generator.integers(0, 7, size=257) if multiclass else generator.integers(0, 2, size=logits.shape)
    mask = generator.random(labels.shape) > 0.3
    expected = uncertainty_methods.fit_temperature(logits[mask], labels[mask], multiclass=multiclass)
    monkeypatch.setattr(uncertainty_methods, "fit_temperature", partial(uncertainty_methods.fit_temperature, batch_elements=128))
    fitted = uncertainty_methods._fit_observed_temperature(
        logits.view(BoundedSelection), labels.view(BoundedSelection),
        "multiclass" if multiclass else "multilabel", mask,
    )
    assert fitted == pytest.approx(expected, rel=2e-7)


@pytest.mark.parametrize("task_type", ["singlelabel", "multilabel", "regression", "sigmoid_regression"])
@pytest.mark.parametrize("all_missing", [False, True])
def test_masked_loss_matches_observed_only_and_has_zero_unknown_gradient(task_type, all_missing):
    torch.manual_seed(8)
    classes = 3 if task_type in {"singlelabel", "multilabel"} else 1
    logits = torch.randn(5, classes)  # (b, c)
    if task_type == "sigmoid_regression":
        logits = logits.sigmoid()
    logits.requires_grad_()
    labels = torch.randint(0, classes, (5,)) if task_type == "singlelabel" else torch.rand(5, classes)
    mask = torch.rand(labels.shape) > 0.4
    if all_missing:
        mask[:] = False
    labels = labels.float()
    labels[~mask] = float("nan")
    criterion = get_loss_fct(task_type)
    actual = compute_probe_loss(criterion, logits, labels, task_type, classes, mask)
    if all_missing:
        assert actual.item() == 0
    elif task_type == "singlelabel":
        torch.testing.assert_close(actual, criterion(logits[mask], labels[mask].long()))
    else:
        torch.testing.assert_close(actual, criterion(logits[mask], labels[mask]))
    actual.backward()
    unknown = ~mask[:, None].expand_as(logits) if task_type == "singlelabel" else ~mask
    assert torch.equal(logits.grad[unknown], torch.zeros_like(logits.grad[unknown]))
    assert torch.isfinite(logits.grad).all()
    assert criterion.reduction == "mean"


@pytest.mark.parametrize("task_type", ["singlelabel", "multilabel", "regression", "sigmoid_regression"])
def test_all_observed_loss_and_gradient_preserve_legacy_behavior(task_type):
    torch.manual_seed(7)
    model = probe(task_type)
    embeddings = torch.randn(6, 6)  # (b, d)
    labels = torch.randint(0, 3, (6,)) if task_type == "singlelabel" else torch.rand(6, model.num_labels)
    first = model(embeddings, labels=labels).loss
    first.backward()
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    model.zero_grad()
    second = model(embeddings, labels=labels, label_mask=torch.ones_like(labels, dtype=torch.bool)).loss
    second.backward()
    torch.testing.assert_close(first, second)
    for expected, parameter in zip(gradients, model.parameters(), strict=True):
        torch.testing.assert_close(expected, parameter.grad)


def test_per_label_weights_keep_their_columns_and_criterion_is_not_mutated():
    logits = torch.tensor([[0.1, -0.2, 0.5], [-0.4, 0.7, 1.0]], requires_grad=True)
    labels = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    mask = torch.tensor([[True, False, True], [False, True, False]])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0, 3.0, 5.0]))
    unreduced = copy.copy(criterion)
    unreduced.reduction = "none"
    expected = unreduced(logits, labels)[mask].mean()
    torch.testing.assert_close(compute_probe_loss(criterion, logits, labels, "multilabel", 3, mask), expected)
    assert criterion.reduction == "mean"


@pytest.mark.parametrize("output", ["faithful", "beta_nll"])
@pytest.mark.parametrize("all_missing", [False, True])
def test_variance_heads_ignore_unobserved_targets(output, all_missing):
    torch.manual_seed(3)
    model = probe("regression", output=output)
    embeddings = torch.randn(4, 6)
    labels = torch.tensor([1.0, float("nan"), 4.0, float("nan")])
    mask = torch.tensor([True, False, True, False])
    if all_missing:
        mask[:] = False
    actual = model(embeddings, labels=labels, label_mask=mask).loss
    expected = model(embeddings[mask], labels=labels[mask]).loss if mask.any() else torch.tensor(0.0)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("task_type", ["singlelabel", "regression"])
def test_sngp_posterior_uses_only_observed_rows(task_type):
    torch.manual_seed(4)
    first = probe(task_type, output="gaussian_process")
    second = copy.deepcopy(first)
    embeddings = torch.randn(5, 6)
    labels = torch.tensor([0.0, 1.0, 2.0, 1.0, 0.0])
    mask = torch.tensor([True, False, True, True, False])
    first.fit_posterior([{"embeddings": embeddings, "labels": labels, "label_mask": mask}])
    second.fit_posterior([{"embeddings": embeddings[mask], "labels": labels[mask]}])
    for key, expected in second.state_dict().items():
        torch.testing.assert_close(first.state_dict()[key], expected)


def joined_fixture(as_hf=False):
    sequences = ["MACD", "MKLT", "MAVV"]
    embeddings = np.stack([np.arange(6.0, dtype=np.float32) + index for index in range(3)])  # (n, d)
    joined = JoinedBranchEmbeddings(sequences, [embeddings])
    columns = {"seqs": sequences, "labels": [[1.0, float("nan"), 0.0], [float("nan")] * 3, [0.0, 1.0, 0.0]], "label_mask": [[True, False, True], [False] * 3, [True] * 3]}
    dataset = JoinedBranchDataset(Dataset.from_dict(columns) if as_hf else columns, joined, "multilabel", False)
    return dataset, JoinedBranchCollator(joined, False, None)


@pytest.mark.parametrize("as_hf", [False, True])
def test_joined_batches_and_saved_predictions_preserve_masks(tmp_path, as_hf):
    dataset, collator = joined_fixture(as_hf)
    batch = collator([dataset[0], dataset[1]])
    assert batch["label_mask"].tolist() == [[True, False, True], [False] * 3]
    assert batch["labels"].tolist() == [[1, 0, 0], [0] * 3]
    predictions, _ = collect_predictions(probe(), dataset, collator, task_kind="multilabel", batch_size=2, device=torch.device("cpu"))
    path = str(tmp_path / "predictions.npz")
    save_seed_predictions(path, {"test": predictions})
    restored = load_seed_predictions(path)["test"]
    np.testing.assert_array_equal(restored.label_mask, np.array(dataset.label_masks))
    np.testing.assert_array_equal(restored.labels, predictions.labels)
    np.testing.assert_array_equal(restored.outputs, predictions.outputs)


def test_joined_class_labels_are_validated_before_integer_cast():
    joined = JoinedBranchEmbeddings(["AA", "CC"], [np.zeros((2, 6), dtype=np.float32)])
    columns = {"seqs": ["AA", "CC"], "labels": [1.5, np.nan], "label_mask": [True, False]}
    dataset = JoinedBranchDataset(columns, joined, "singlelabel", False)
    with pytest.raises(ValueError, match="integers"):
        dataset[0]
    assert dataset[1][1].item() == 0
    assert dataset[1][2].item() is False


@pytest.mark.parametrize("mask", [torch.ones(2, 3), torch.ones(2, 1, dtype=torch.bool)])
def test_wrong_mask_types_and_shapes_fail(mask):
    with pytest.raises(ValueError, match="boolean.*labels shape"):
        probe()(torch.randn(2, 6), labels=torch.zeros(2, 3), label_mask=mask)


def test_observed_nonfinite_values_fail_in_model_and_saved_predictions():
    labels = torch.tensor([[float("nan"), 1.0, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        probe()(torch.randn(1, 6), labels=labels, label_mask=torch.ones_like(labels, dtype=torch.bool))
    with pytest.raises(ValueError, match="finite"):
        SplitPredictions(labels.numpy(), np.zeros((1, 3)), label_mask=np.ones((1, 3), dtype=bool))


def test_missing_and_explicit_masks_cannot_mix_within_one_batch():
    dataset, collator = joined_fixture()
    # Compact rows carry observation status with their sparse target, rather than in a
    # separately allocated dense tensor. Removing that status must still refuse the batch.
    indices, target = dataset[1]
    with pytest.raises(ValueError, match="mix"):
        collator([dataset[0], (indices, replace(target, explicit_mask=False))])
    with pytest.raises(ValueError, match="mix"):
        collator([((0,), torch.zeros(3), torch.ones(3, dtype=torch.bool)), ((1,), torch.zeros(3))])


def test_multilabel_metrics_match_hand_calculation_and_ignore_unknown_fillers():
    logits = np.array([[1.0, 30.0, -1.0], [-1.0, -30.0, 1.0], [0.0, 0.0, 0.0]])
    labels = np.array([[1.0, -5.0, 0.0], [0.0, 6.0, 1.0], [4.0, 3.0, 5.0]])
    mask = np.array([[True, False, True], [True, False, True], [False] * 3])
    probs = probabilities(logits, "multilabel")
    expected_nll = np.log1p(np.exp(-1.0))
    assert negative_log_likelihood("multilabel", labels, probs, mask) == pytest.approx(expected_nll)
    assert brier_score("multilabel", labels, probs, mask) == pytest.approx((1.0 / (1.0 + np.exp(1.0))) ** 2)
    np.testing.assert_allclose(row_losses("multilabel", labels, probs=probs, label_mask=mask), [0, 0, np.nan], equal_nan=True)
    compute = get_compute_metrics("multilabel", observed_labels=True)
    first = compute(EvalPrediction(predictions=logits, label_ids=(labels, mask)))
    labels[~mask] = -9000
    second = compute(EvalPrediction(predictions=logits, label_ids=(labels, mask)))
    assert first == second
    assert first["observed_labels"] == 4 and first["unobserved_labels"] == 5
    assert first["accuracy"] == 1.0 and first["pr_auc"] == 1.0


@pytest.mark.parametrize("kind", ["binary", "multilabel", "regression"])
def test_calibration_and_reports_ignore_unknown_targets(kind):
    generator = np.random.default_rng(9)
    seeds = {}
    for seed in (1, 2):
        splits = {}
        for name in ("v_fit", "v_cal", "test"):
            outputs = generator.normal(size=(24, 3)) if kind == "multilabel" else generator.normal(size=(24, 2)) if kind == "binary" else generator.normal(size=24)
            labels = generator.integers(0, 2, size=outputs.shape if kind == "multilabel" else 24)
            if kind == "regression":
                labels = generator.normal(size=24)
            mask = generator.random(labels.shape) > 0.3
            if seed == 2:
                labels, mask = seeds[1][name].labels.copy(), seeds[1][name].label_mask.copy()
            splits[name] = SplitPredictions(labels, outputs, label_mask=mask)
        seeds[seed] = splits
    changed = copy.deepcopy(seeds)
    for splits in changed.values():
        for split in splits.values():
            split.labels[~split.label_mask] = 200
    methods = build_methods(seeds, "projected", kind)
    changed_methods = build_methods(changed, "projected", kind)
    labels = agreed_labels(seeds, [1, 2])
    other_labels = agreed_labels(changed, [1, 2])
    masks = {name: split.label_mask for name, split in seeds[1].items()}
    for first, second in zip(methods, changed_methods, strict=True):
        assert [run.fitted for run in first.runs] == [run.fitted for run in second.runs]
        assert score_method(first, kind, labels, (0.8,), 1.0, None, masks) == score_method(second, kind, other_labels, (0.8,), 1.0, None, masks)
        if kind != "multilabel":
            rows = conformal_rows(first.runs[0], kind, other_labels, (0.8,), 1.0, masks)
            for name, values in rows.items():
                if "coverage@" in name:
                    assert np.isnan(values[~masks["test"]]).all()
    changed[2]["v_fit"].label_mask[0] = ~changed[2]["v_fit"].label_mask[0]
    with pytest.raises(ValueError, match="observation masks"):
        agreed_labels(changed, [1, 2])


def test_no_observed_calibration_labels_cannot_fit_temperature():
    split = SplitPredictions(np.zeros((4, 3)), np.zeros((4, 3)), label_mask=np.zeros((4, 3), dtype=bool))
    with pytest.raises(ValueError, match="observed validation"):
        build_methods({1: {name: split for name in ("v_fit", "v_cal", "test")}}, "projected", "multilabel")


def test_masked_cell_trains_calibrates_resumes_with_observed_truth_contract(tmp_path):
    from protify.testing_suite import test_uncertainty_workflow as fixture
    from protify.uncertainty.package import read_finished_cell

    frames, data_dir = fixture._toy_dataset(tmp_path)
    fixture._write_caches(frames, tmp_path / "embeddings")
    for name, frame in frames.items():
        frame["label_mask"] = np.arange(len(frame)) % 3 != 0
        frame.to_csv(tmp_path / "data" / "toyset" / f"{name}.csv", index=False)
    cell = fixture._cell(tmp_path, data_dir, "masked", num_runs=1, head="projected", save_heads=True)
    explicit = (list(range(20)), list(range(20, 40)))
    report = fixture.workflow.run_uncertainty_cell(cell, explicit_rows=explicit)
    assert report["methods"]["plain"]["test"]["observed_labels"] == 20
    assert report["methods"]["plain"]["test"]["unobserved_labels"] == 10
    saved = load_seed_predictions(str(tmp_path / "runs" / "masked" / "seed_42.npz"))
    np.testing.assert_array_equal(saved["test"].label_mask, frames["test"]["label_mask"].to_numpy())
    assert fixture.workflow.run_uncertainty_cell(cell, explicit_rows=explicit) == report
    assert read_finished_cell(cell.output_dir).observation_contract["whole_answer"] == "unavailable_partial_calibration"
    # A changed training mask must not silently reuse a head trained on different truth.
    frames["train"].loc[0, "label_mask"] = True
    frames["train"].to_csv(tmp_path / "data" / "toyset" / "train.csv", index=False)
    with pytest.raises(ValueError, match="label observations changed"):
        fixture.workflow.run_uncertainty_cell(cell, explicit_rows=explicit)


def test_observed_selection_loss_weights_targets_instead_of_batch_rows():
    labels = np.array([[1, 99, 99], [1, 0, 1]])
    mask = np.array([[True, False, False], [True, True, True]])
    logits = np.array([[0.0, -100.0, 100.0], [2.0, -2.0, 2.0]])
    compute = get_compute_metrics("multilabel", observed_labels=True)
    scores = compute(EvalPrediction(predictions=logits, label_ids=(labels, mask)))
    expected = (np.log(2.0) + 3.0 * np.log1p(np.exp(-2.0))) / 4.0
    assert scores["observed_loss"] == pytest.approx(expected)
    assert scores["observed_loss"] != pytest.approx((np.log(2.0) + np.log1p(np.exp(-2.0))) / 2.0)
    order = np.array([1, 0])
    assert compute(EvalPrediction(predictions=logits[order], label_ids=(labels[order], mask[order])))["observed_loss"] == pytest.approx(expected)


@pytest.mark.parametrize("block_entries", [3, 11, 64, 2**24])
def test_blocked_selection_metrics_equal_the_complete_masked_selection_loss(monkeypatch, block_entries):
    generator = np.random.default_rng(23)
    logits = generator.normal(scale=3.0, size=(37, 11)).astype(np.float32)  # (n, c)
    labels = generator.integers(0, 2, size=logits.shape).astype(np.float32)  # (n, c)
    mask = generator.random(logits.shape) > 0.4  # (n, c)
    labels[~mask], logits[~mask] = -9000.0, 1e4
    monkeypatch.setattr(metrics_module, "SELECTION_BLOCK_ENTRIES", block_entries)
    prediction = EvalPrediction(predictions=logits, label_ids=(labels, mask))
    complete = get_compute_metrics("multilabel", observed_labels=True)(prediction)
    selected = get_compute_metrics("multilabel", observed_labels=True, selection_only=True)(prediction)
    assert set(selected) == {"observed_loss", "observed_labels", "unobserved_labels"}
    assert selected["observed_loss"] == pytest.approx(complete["observed_loss"], rel=1e-12)
    assert (selected["observed_labels"], selected["unobserved_labels"]) == (complete["observed_labels"], complete["unobserved_labels"])
    logits[~mask] = np.nan
    assert get_compute_metrics("multilabel", observed_labels=True, selection_only=True)(prediction) == selected, "unobserved predictions are never read"


def test_selection_metrics_are_bounded_by_a_block_of_rows_however_wide_the_vocabulary(monkeypatch):
    seen = []
    real = torch.nn.functional.binary_cross_entropy_with_logits
    monkeypatch.setattr(metrics_module, "SELECTION_BLOCK_ENTRIES", 200)
    monkeypatch.setattr(torch.nn.functional, "binary_cross_entropy_with_logits", lambda logits, labels, **options: seen.append(logits.numel()) or real(logits, labels, **options))
    generator = np.random.default_rng(3)
    labels = generator.integers(0, 2, size=(101, 50)).astype(np.float32)  # (n, c)
    mask = np.ones(labels.shape, dtype=bool)  # (n, c)
    get_compute_metrics("multilabel", observed_labels=True, selection_only=True)(EvalPrediction(predictions=generator.normal(size=labels.shape), label_ids=(labels, mask)))
    assert len(seen) == 26 and max(seen) == 200 and sum(seen) == labels.size


@pytest.mark.parametrize("change,message", [
    ("nonfinite_prediction", "predictions must be finite"), ("nonbinary_target", "zero or one"), ("nothing_observed", "at least one observed"),
    ("integer_mask", "boolean"), ("unmasked_labels", "label_names"), ("shape", "same shape"),
])
def test_selection_metrics_refuse_what_the_complete_metrics_refuse(change, message):
    logits = np.zeros((4, 5), dtype=np.float32)  # (n, c)
    labels = np.tile([0.0, 1.0, 0.0, 1.0, 1.0], (4, 1)).astype(np.float32)  # (n, c)
    mask = np.ones(labels.shape, dtype=bool)  # (n, c)
    label_ids = (labels, mask)
    if change == "nonfinite_prediction":
        logits[2, 3] = np.inf
    elif change == "nonbinary_target":
        labels[1, 1] = 0.5
    elif change == "nothing_observed":
        mask[:] = False
    elif change == "integer_mask":
        label_ids = (labels, mask.astype(np.int64))
    elif change == "unmasked_labels":
        label_ids = labels
    else:
        logits = logits[:, :4]
    for compute in (get_compute_metrics("multilabel", observed_labels=True), get_compute_metrics("multilabel", observed_labels=True, selection_only=True)):
        if change == "shape" and compute.__name__ == "masked_metrics":
            continue  # the complete metrics fail on a shape mismatch with an index error rather than a message
        with pytest.raises(ValueError, match=message):
            compute(EvalPrediction(predictions=logits, label_ids=label_ids))


@pytest.mark.parametrize("task_type,observed,tokenwise", [("singlelabel", True, False), ("regression", True, False), ("multilabel", False, False), ("multilabel", True, True)])
def test_selection_only_is_offered_only_for_masked_pooled_multilabel(task_type, observed, tokenwise):
    with pytest.raises(ValueError, match="selection-only"):
        get_compute_metrics(task_type, tokenwise=tokenwise, observed_labels=observed, selection_only=True)


@pytest.mark.parametrize("task_type", ["singlelabel", "regression", "sigmoid_regression"])
def test_observed_selection_loss_reuses_standard_task_criterion(task_type):
    labels = np.array([0, 1, np.nan, 0, 1])
    mask = np.array([True, True, False, True, True])
    outputs = np.array([[2.0, -2.0], [-1.0, 1.0], [np.nan, np.nan], [0.4, -0.4], [-3.0, 3.0]]) if task_type == "singlelabel" else np.array([0.1, 0.9, np.nan, 0.2, 0.8])
    scores = get_compute_metrics(task_type, observed_labels=True)(EvalPrediction(predictions=outputs, label_ids=(labels, mask)))
    dtype = torch.long if task_type == "singlelabel" else torch.float64
    expected = get_loss_fct(task_type)(torch.tensor(outputs[mask]), torch.tensor(labels[mask], dtype=dtype))
    assert scores["observed_loss"] == pytest.approx(float(expected))


def test_masked_resume_requires_original_observation_record(tmp_path):
    from types import SimpleNamespace
    from protify.uncertainty.workflow import _record_observations

    split = {"seqs": ["AA", "CC"], "labels": [0, 1], "label_mask": [True, True]}
    cell_data = SimpleNamespace(train=split, valid=split, test=split)
    (tmp_path / "seed_42.npz").write_bytes(b"finished predictions")
    with pytest.raises(ValueError, match="no label observation record"):
        _record_observations(str(tmp_path), cell_data)


def test_deleting_an_annotation_increases_unknown_without_creating_a_negative():
    labels = np.array([[1, 1, 0], [0, 1, 0]])
    mask = np.array([[True, True, False], [True, True, False]])
    before = label_support("multilabel", labels, mask)
    mask[0, 0] = False
    after = label_support("multilabel", labels, mask)
    assert before["labels"][0]["negative"] == after["labels"][0]["negative"] == 1
    assert after["unknown"] == before["unknown"] + 1
    assert before["labels"][0]["discrimination_status"] == "supported"
    assert all(row["discrimination_status"] == "insufficient_observed_classes" for row in after["labels"])


@pytest.mark.parametrize("kind", ["binary", "regression"])
def test_column_shaped_scalar_targets_have_identical_calibration_and_reports(kind):
    generator = np.random.default_rng(54)
    seeds = {1: {}}
    for name in ("v_fit", "v_cal", "test"):
        labels = generator.integers(0, 2, 16) if kind == "binary" else generator.normal(size=16)
        outputs = generator.normal(size=(16, 2)) if kind == "binary" else generator.normal(size=16)
        seeds[1][name] = SplitPredictions(labels, outputs, label_mask=np.arange(16) % 3 != 0)
    columns = copy.deepcopy(seeds)
    for split in columns[1].values():
        split.labels = split.labels[:, None]
        split.label_mask = split.label_mask[:, None]
    for first, second in zip(build_methods(seeds, "projected", kind), build_methods(columns, "projected", kind), strict=True):
        first_report = score_method(first, kind, {key: split.labels for key, split in seeds[1].items()}, (0.8,), 1.0, None, {key: split.label_mask for key, split in seeds[1].items()})
        second_report = score_method(second, kind, {key: split.labels for key, split in columns[1].items()}, (0.8,), 1.0, None, {key: split.label_mask for key, split in columns[1].items()})
        assert first_report == second_report


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("task_type", ["singlelabel", "multilabel", "regression"])
def test_compiled_cuda_masked_head_matches_eager_loss_and_gradients(task_type):
    torch.manual_seed(17)
    eager = probe(task_type).cuda()
    compiled = torch.compile(copy.deepcopy(eager))
    embeddings = torch.randn(6, 6, device="cuda")
    labels = torch.randint(0, 3, (6,), device="cuda").float() if task_type == "singlelabel" else torch.rand(6, eager.num_labels, device="cuda")
    for density in (1.0, 0.5, 0.0):
        mask = torch.rand(labels.shape, device="cuda") < density
        observed_labels = torch.where(mask, labels, float("nan"))
        eager.zero_grad(set_to_none=True)
        compiled.zero_grad(set_to_none=True)
        expected = eager(embeddings, labels=observed_labels, label_mask=mask).loss
        actual = compiled(embeddings, labels=observed_labels, label_mask=mask).loss
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        expected.backward()
        actual.backward()
        for first, second in zip(eager.parameters(), compiled.parameters(), strict=True):
            torch.testing.assert_close(second.grad, first.grad, rtol=2e-4, atol=2e-5)
