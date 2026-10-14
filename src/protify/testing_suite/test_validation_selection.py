"""Initial-checkpoint selection, native training, and resume integrity."""

from __future__ import annotations

import copy
import json
import pytest
import torch

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from transformers import default_data_collator

from protify.data.cluster_sampling import ClusterSamplingPlan, validate_sampling_receipt
from protify.probes.trainers import TrainerArguments, TrainerMixin
from protify.probes.training_recipe import validate_training_receipt
from protify.probes.validation_selection import POLICY, ValidationSelection, selection_contract
from protify.uncertainty.cell import cell_from_mapping


def reading(selection: ValidationSelection, model: torch.nn.Module, step: int, mcc: float, loss: float) -> SimpleNamespace:
    control = SimpleNamespace(should_training_stop=False)
    selection.on_evaluate(None, SimpleNamespace(global_step=step), control,
                          {"eval_mcc": mcc, "eval_observed_loss": loss}, model)
    return control


@pytest.mark.parametrize("winner", [0, 1, 2])
def test_exact_restoration_and_loss_ties(winner: int) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    selection = ValidationSelection(POLICY)
    readings = {0: [(0.8, 1.0), (0.7, 0.1), (0.8, 1.0)],
                1: [(0.5, 1.0), (0.8, 1.0), (0.7, 0.1)],
                2: [(0.5, 1.0), (0.8, 1.0), (0.8, 0.5)]}[winner]
    for step, (mcc, loss) in enumerate(readings):
        with torch.no_grad():
            model.weight.fill_(step)
        reading(selection, model, step, mcc, loss)
    selection.on_train_end(None, None, None)
    reading(selection, model, 3, 1.0, 0.0)  # post-training scoring cannot change selection
    outcome = selection.restore(model, SimpleNamespace(global_step=2, epoch=1.0, max_steps=2))
    assert model.weight.item() == winner and len(selection.curve) == 3
    recipe = {"checkpoint_selection": selection_contract(POLICY)}
    receipt = {"seed": 42, "recipe": recipe, "outcome": outcome}
    validate_training_receipt(recipe, receipt, 42)
    tampered = copy.deepcopy(receipt)
    tampered["outcome"]["best_metric"] += 0.1
    with pytest.raises(ValueError, match="winner"):
        validate_training_receipt(recipe, tampered, 42)
    with pytest.raises(ValueError, match="differs"):
        validate_training_receipt({}, {**receipt, "recipe": {}}, 42)


def test_six_stale_checks_and_finite_initial_metrics() -> None:
    selection, model = ValidationSelection(POLICY), torch.nn.Linear(1, 1)
    reading(selection, model, 0, 0.8, 1.0)
    for step in range(1, 7):
        assert reading(selection, model, step, 0.7, 0.1).should_training_stop == (step == 6)
    with pytest.raises(ValueError, match="finite"):
        reading(selection, model, 7, float("nan"), 0.1)
    with pytest.raises(ValueError, match="initial"):
        reading(ValidationSelection(POLICY), model, 1, 0.1, 1.0)


@pytest.mark.parametrize("settings", [{"cluster_class_balance": "yes"}, {"checkpoint_selection": "other"},
                                      {"checkpoint_selection": POLICY, "selection_metric": "roc_auc"}])
def test_cell_rejects_invalid_recipe(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        cell_from_mapping({"cell_id": "fixture", "experiment_id": "fixture", "output_dir": "/unused",
                           "uncertainty": {"branch_specs": ["ESM2-8:mean"], "dataset_manifest": "/dataset.json",
                                           "dataset_sha256": "a" * 64, **settings}})


class BinaryRows(torch.utils.data.Dataset):
    frozen_release = True
    label_masks = [True] * 16

    def __init__(self) -> None:
        self.sampling_plan = ClusterSamplingPlan.from_dataset({
            "observation_id": [str(index) for index in range(16)],
            "protein_id": [str(index) for index in range(16)],
            "cluster_id": [str(index) for index in range(16)],
            "labels": [index % 2 for index in range(16)], "label_mask": self.label_masks,
        }, 32, balance_classes=True)

    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"embeddings": torch.tensor([float(index % 2)]),  # (d=1,)
                "labels": torch.tensor(index % 2), "label_mask": torch.tensor(True)}  # scalars


class BinaryHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        # embeddings: (b, 1); labels: (b,); label_mask: (b,)
        logits = self.linear(embeddings)  # (b, 2)
        return {"logits": logits, "loss": torch.nn.functional.cross_entropy(logits[label_mask], labels[label_mask])}  # logits: (b, 2); loss: ()


def test_native_trainer_restores_initial_and_records_balanced_exposure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import protify.probes.trainers as trainers

    monkeypatch.chdir(tmp_path)
    # Deterministic validation readings deliberately make the initial head the winner.
    scores = iter([(0.9, 1.0)] + [(0.8, 0.5)] * 3 + [(0.9, 1.0)])

    def metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
        mcc, loss = next(scores)
        return {"mcc": mcc, "observed_loss": loss}

    monkeypatch.setattr(trainers, "get_compute_metrics", lambda *args, **kwargs: metrics)
    train, valid = BinaryRows(), BinaryRows()
    model = BinaryHead()
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}  # parameter-shaped tensors
    arguments = TrainerArguments(model_save_dir=str(tmp_path), torch_compile=False, probe_max_steps=3,
                                 probe_warmup_steps=0, probe_eval_strategy="steps", probe_eval_steps=1,
                                 probe_batch_size=4, task_type="singlelabel", num_workers=0,
                                 probe_precision="float32", record_training_recipe=True)
    main = SimpleNamespace(trainer_args=arguments, probe_args=SimpleNamespace(tokenwise=False, num_labels=2, task_type="singlelabel"))
    trained, _, _, _, _ = TrainerMixin._train(main, model, train, valid, None, default_data_collator,
                                             None, "fixture", "fixture", "fixture", evaluate_test=False,
                                             checkpoint_selection=POLICY)
    assert all(torch.equal(value, initial[name]) for name, value in trained.state_dict().items())
    assert train.training_outcome["global_step"] == 3 and train.training_outcome["best_checkpoint"] == "initial"
    receipt = {"seed": arguments.seed, "recipe": train.training_recipe, "outcome": train.training_outcome}
    validate_training_receipt(train.training_recipe, receipt, arguments.seed)
    validate_sampling_receipt(train.sampling_plan, train.sampling_receipt, arguments.seed)
    counts = train.sampling_receipt["observation_counts"]
    assert sum(count for key, count in counts.items() if int(key) % 2) == 6


def test_declared_workflow_binds_selection_and_balance_on_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from protify.testing_suite.test_declared_dataset import release_fixture
    from protify.testing_suite import test_uncertainty_workflow as fixture

    monkeypatch.chdir(tmp_path)
    path, digest, _, rows = release_fixture(tmp_path, "binary")
    mapping = fixture._cell_mapping(tmp_path, "unused", "balanced_initial", num_runs=1)
    mapping["protify_args"].pop("data_dirs")
    mapping["protify_args"].update(probe_max_steps=3, probe_eval_strategy="steps", probe_eval_steps=1)
    mapping["uncertainty"].update(head="projected", dataset_manifest=str(path), dataset_sha256=digest,
                                   save_heads=True, require_cached_embeddings=True, evaluation_mode="validation_only",
                                   cluster_class_balance=True, cluster_draws_per_epoch=64, checkpoint_selection=POLICY)
    cell = cell_from_mapping(mapping)
    collect = fixture.workflow.collect_predictions

    def validation_predictions(model: Any, dataset: Any, *args: Any, **kwargs: Any) -> Any:
        assert dataset.observation_ids == [row["observation_id"] for row in rows["validation"]]
        return collect(model, dataset, *args, **kwargs)

    monkeypatch.setattr(fixture.workflow, "collect_predictions", validation_predictions)
    monkeypatch.setattr(fixture.workflow, "build_methods", lambda *args, **kwargs: pytest.fail("fit entered calibration"))
    report = fixture.workflow.run_uncertainty_cell(cell)
    record = json.loads((Path(cell.output_dir) / "seed_42.json").read_text())
    counts = record["sampling"]["observation_counts"]
    positive = sum(counts.get(row["observation_id"], 0) * row["labels"] for row in rows["train"])
    assert 2 * positive == sum(counts.values()) == 48
    assert report["training_recipe"]["checkpoint_selection"] == selection_contract(POLICY)
    assert fixture.workflow.run_uncertainty_cell(cell) == report
    for key, value in (("checkpoint_selection", None), ("cluster_class_balance", False)):
        changed = copy.deepcopy(mapping)
        changed["uncertainty"][key] = value
        with pytest.raises(ValueError, match="different training settings"):
            fixture.workflow.run_uncertainty_cell(cell_from_mapping(changed))
