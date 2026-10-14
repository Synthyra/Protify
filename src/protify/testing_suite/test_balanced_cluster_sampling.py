"""Exact batch balance retains cluster/protein/assay weighting and exposure replay."""

from __future__ import annotations

import hashlib
import pytest
import torch

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from transformers import TrainingArguments, default_data_collator

from protify.data.branch_features import JoinedBranchDataset
from protify.data.cluster_sampling import ClusterAssaySampler, ClusterSamplingPlan, observation_digest_bytes, validate_sampling_receipt
from protify.probes.cluster_trainer import ClusterTrainer
from protify.testing_suite.test_cluster_sampling import Probe, Rows


def metadata() -> dict[str, list]:
    proteins = ["negative_repeated"] * 40 + ["negative_single"] + [f"positive_{index}" for index in range(100)]
    return {"observation_id": [f"assay_{index}" for index in range(len(proteins))], "protein_id": proteins,
            "cluster_id": ["negative_small"] * 40 + ["negative_other"] + ["positive_large"] * 100,
            "labels": [0.0] * 41 + [1.0] * 100, "label_mask": [True] * len(proteins)}


def test_balanced_batches_and_conditional_cluster_weighting() -> None:
    cohort = metadata()
    plan = ClusterSamplingPlan.from_dataset(cohort, 64_000, balance_classes=True)
    draws = list(ClusterAssaySampler(plan, 42))
    assert draws == list(ClusterAssaySampler(plan, 42))
    assert all(sum(cohort["labels"][row] for row in draws[start:start + 64]) == 32 for start in range(0, len(draws), 64))
    clusters = Counter(plan.cluster_ids[row] for row in draws)
    assert clusters["negative_small"] / 32_000 == pytest.approx(0.5, abs=0.02)
    assert clusters["positive_large"] == 32_000
    assert len({row for row in draws if row < 40}) == 40
    sampler = ClusterAssaySampler(plan, 42)
    sampler.set_epoch(1)
    assert list(sampler) != draws


def test_balanced_consumption_receipt_reconstructs_partial_prefix() -> None:
    plan = ClusterSamplingPlan.from_dataset(metadata(), 1024, balance_classes=True)
    rows = list(ClusterAssaySampler(plan, 1729))[:192]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(observation_digest_bytes(plan.observation_ids[row]))
    receipt = {"plan": plan.record(), "seed": 1729,
               "observation_counts": dict(Counter(plan.observation_ids[row] for row in rows)),
               "protein_counts": dict(Counter(plan.protein_ids[row] for row in rows)),
               "cluster_counts": dict(Counter(plan.cluster_ids[row] for row in rows)),
               "epochs": [{"epoch": 0, "consumed_rows": len(rows), "observation_order_sha256": digest.hexdigest()}]}
    validate_sampling_receipt(plan, receipt, 1729)
    receipt["epochs"][0]["observation_order_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="row order"):
        validate_sampling_receipt(plan, receipt, 1729)


def test_balanced_plan_rejects_odd_epochs_and_missing_class() -> None:
    cohort = metadata()
    with pytest.raises(ValueError, match="even"):
        ClusterSamplingPlan.from_dataset(cohort, 17, balance_classes=True)
    cohort["labels"] = [0.0] * len(cohort["labels"])
    with pytest.raises(ValueError, match="both classes"):
        ClusterSamplingPlan.from_dataset(cohort, 64, balance_classes=True)


def test_native_trainer_consumes_exactly_balanced_batches(tmp_path: Path) -> None:
    cohort = metadata()
    plan = ClusterSamplingPlan.from_dataset(cohort, 512, balance_classes=True)

    class BinaryRows(Rows):
        def __getitem__(self, index: int) -> dict[str, Any]:
            example = super().__getitem__(index)
            example["labels"] = torch.tensor(cohort["labels"][index], dtype=torch.float32)  # ()
            return example

    arguments = TrainingArguments(output_dir=str(tmp_path / "fit"), per_device_train_batch_size=64, max_steps=3,
                                  report_to="none", use_cpu=True, save_strategy="no", disable_tqdm=True, seed=42)
    trainer = ClusterTrainer(sampling_plan=plan, model=Probe(), args=arguments,
                             train_dataset=BinaryRows(len(plan.observation_ids)), data_collator=default_data_collator)
    trainer.train()
    receipt = trainer.sampling_receipt()
    validate_sampling_receipt(plan, receipt, 42)
    by_label = Counter()
    for index, observation in enumerate(plan.observation_ids):
        by_label[cohort["labels"][index]] += receipt["observation_counts"].get(observation, 0)
    assert by_label == {0.0: 96, 1.0: 96}


def test_native_dataset_constructs_balanced_plan_once() -> None:
    cohort = metadata()
    cohort["seqs"] = ["A" * (index + 1) for index in range(len(cohort["labels"]))]
    joined = SimpleNamespace(rows={sequence: index for index, sequence in enumerate(cohort["seqs"])})
    dataset = JoinedBranchDataset(cohort, joined, "singlelabel", False, train=True, frozen_release=True,
                                  cluster_draws_per_epoch=64, cluster_class_balance=True)
    draws = list(ClusterAssaySampler(dataset.sampling_plan, 42))
    assert sum(cohort["labels"][row] for row in draws) == 32
    assert dataset.frozen_release and dataset.train and dataset.observation_ids == cohort["observation_id"]
