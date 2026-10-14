"""Family balance, assay balance, and consumed exposures under the real trainer loop."""

from __future__ import annotations

import copy
import hashlib
import pytest
import torch

from collections import Counter
from dataclasses import replace
from torch.utils.data import Dataset
from transformers import TrainingArguments, default_data_collator

from protify.data.cluster_sampling import ClusterAssaySampler, ClusterSamplingPlan, observation_digest_bytes, validate_sampling_receipt
from protify.probes.cluster_trainer import ClusterTrainer
from protify.uncertainty.cell import cell_from_mapping
from protify.uncertainty.workflow import _record_sampling


def metadata():
    # Cluster small has two proteins, one with 40 distinct assays. Cluster large has
    # 100 proteins. The final row is unknown and must receive zero exposure.
    proteins = ["s0"] * 40 + ["s1"] + [f"l{i}" for i in range(100)] + ["unknown"]
    return {
        "observation_id": [f"assay_{i}" for i in range(len(proteins))], "protein_id": proteins,
        "cluster_id": ["small"] * 41 + ["large"] * 100 + ["unknown_cluster"],
        "label_mask": [True] * 141 + [False],
    }


def test_families_and_repeated_assays_do_not_gain_sampling_weight():
    plan = ClusterSamplingPlan.from_dataset(metadata(), 100_000)
    indices = list(ClusterAssaySampler(plan, 13))
    clusters = Counter(plan.cluster_ids[row] for row in indices)
    proteins = Counter(plan.protein_ids[row] for row in indices)
    assert set(clusters) == {"small", "large"}
    assert clusters["small"] / len(indices) == pytest.approx(0.5, abs=0.01)
    assert proteins["s0"] / clusters["small"] == pytest.approx(0.5, abs=0.015)
    assert proteins["s1"] / clusters["small"] == pytest.approx(0.5, abs=0.015)
    assert "unknown" not in proteins
    assays = Counter(plan.observation_ids[row] for row in indices if plan.protein_ids[row] == "s0")
    assert len(assays) == 40
    assert all(count / proteins["s0"] == pytest.approx(1 / 40, abs=0.006) for count in assays.values())


def test_epochs_are_repeatable_and_default_exposure_uses_eligible_clusters():
    plan = ClusterSamplingPlan.from_dataset(metadata())
    assert plan.draws_per_epoch == 2
    assert plan.record()["eligible_observations"] == 141
    sampler = ClusterAssaySampler(replace(plan, draws_per_epoch=50), 17)
    first = list(sampler)
    assert first == list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert second != first and second == list(sampler)
    sampler.set_epoch(0)
    assert list(sampler) == first


@pytest.mark.parametrize("mutation", ["duplicate", "membership", "mask", "no_labels", "length", "draws"])
def test_invalid_sampling_metadata_is_rejected(mutation):
    sampling_metadata = metadata()
    if mutation == "duplicate":
        sampling_metadata["observation_id"][0] = sampling_metadata["observation_id"][1]
    elif mutation == "membership":
        sampling_metadata["cluster_id"][0] = "another"
    elif mutation == "mask":
        sampling_metadata["label_mask"][0] = 1
    elif mutation == "no_labels":
        sampling_metadata["label_mask"] = [False] * len(sampling_metadata["label_mask"])
    elif mutation == "length":
        sampling_metadata["protein_id"].pop()
    with pytest.raises(ValueError):
        ClusterSamplingPlan.from_dataset(sampling_metadata, True if mutation == "draws" else None)


class Rows(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"embeddings": torch.tensor([index / self.size], dtype=torch.float32), "labels": torch.tensor(index % 2, dtype=torch.float32)}


class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, embeddings, labels):
        logits = self.linear(embeddings).squeeze(-1)  # (b,)
        return {"loss": (logits - labels).square().mean(), "logits": logits}


@pytest.mark.parametrize("workers,max_steps,epochs,draws", [(2, 1, 3, 25), (0, -1, 2, 7)])
def test_real_trainer_records_consumed_rows_not_prefetched_rows(tmp_path, workers, max_steps, epochs, draws):
    plan = ClusterSamplingPlan.from_dataset(metadata(), draws)
    arguments = TrainingArguments(
        output_dir=str(tmp_path / "fit"), per_device_train_batch_size=4, num_train_epochs=epochs,
        max_steps=max_steps, report_to="none", use_cpu=True, save_strategy="no", disable_tqdm=True,
        dataloader_num_workers=workers, seed=42,
    )
    trainer = ClusterTrainer(sampling_plan=plan, model=Probe(), args=arguments,
                             train_dataset=Rows(len(plan.observation_ids)), data_collator=default_data_collator)
    trainer.train()
    receipt = trainer.sampling_receipt()
    validate_sampling_receipt(plan, receipt, 42)
    expected_count = 4 if max_steps == 1 else epochs * draws
    assert sum(receipt["observation_counts"].values()) == expected_count
    assert len(receipt["epochs"]) == (1 if max_steps == 1 else epochs)
    assert sum(record["consumed_rows"] for record in receipt["epochs"]) == expected_count
    # Evaluation uses its own fixed order and cannot add to training exposure.
    before = copy.deepcopy(receipt)
    trainer.evaluate(Rows(9))
    assert trainer.sampling_receipt() == before
    changed = copy.deepcopy(receipt)
    changed["cluster_counts"][next(iter(changed["cluster_counts"]))] += 1
    with pytest.raises(ValueError, match="cluster_counts"):
        validate_sampling_receipt(plan, changed, 42)


def test_sampling_plan_is_a_required_resume_binding(tmp_path):
    from protify.uncertainty.report import write_json

    plan = ClusterSamplingPlan.from_dataset(metadata(), 10)
    _record_sampling(str(tmp_path), plan, [42])
    with pytest.raises(ValueError, match="plan changed"):
        _record_sampling(str(tmp_path), replace(plan, draws_per_epoch=11), [42])
    (tmp_path / "seed_42.npz").write_bytes(b"fixture")
    write_json(str(tmp_path / "seed_42.json"), {"seed": 42})
    with pytest.raises(ValueError, match="sampling receipt"):
        _record_sampling(str(tmp_path), plan, [42])


def test_receipt_checks_order_not_only_aggregate_counts():
    plan = ClusterSamplingPlan.from_dataset(metadata(), 1)
    row, = list(ClusterAssaySampler(plan, 42))
    receipt = {
        "plan": plan.record(), "seed": 42, "observation_counts": {plan.observation_ids[row]: 1},
        "protein_counts": {plan.protein_ids[row]: 1}, "cluster_counts": {plan.cluster_ids[row]: 1},
        "epochs": [{"epoch": 0, "consumed_rows": 1, "observation_order_sha256": hashlib.sha256(observation_digest_bytes(plan.observation_ids[row])).hexdigest()}],
    }
    validate_sampling_receipt(plan, receipt, 42)
    receipt["epochs"][0]["observation_order_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="row order"):
        validate_sampling_receipt(plan, receipt, 42)


@pytest.mark.parametrize("draws", [True, 0, -1, 1.5, "10"])
def test_cell_rejects_invalid_epoch_draw_counts(draws):
    with pytest.raises(ValueError, match="positive integer"):
        cell_from_mapping({"cell_id": "fixture", "experiment_id": "fixture", "output_dir": "/unused",
                           "uncertainty": {"branch_specs": ["ESM2-8:mean"], "dataset_manifest": "/dataset.json",
                                           "dataset_sha256": "a" * 64, "cluster_draws_per_epoch": draws}})
