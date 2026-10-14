"""A declared release's sampler and realized, successful training-batch exposures."""

from __future__ import annotations

import hashlib
import torch

from collections import Counter
from dataclasses import dataclass
from typing import Any
from torch.utils.data import Dataset
from transformers import Trainer


try:
    from data.cluster_sampling import ClusterAssaySampler, ClusterSamplingPlan, observation_digest_bytes
except ImportError:
    from ..data.cluster_sampling import ClusterAssaySampler, ClusterSamplingPlan, observation_digest_bytes


ROW_KEY = "cluster_sampling_row"


@dataclass(frozen=True)
class SamplingExample:
    row: int
    example: Any


class IndexedTrainingRows(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> SamplingExample:
        return SamplingExample(index, self.dataset[index])


class SamplingCollator:
    """Preserve evaluation items and attach row indices only to indexed training items."""

    def __init__(self, collator: Any) -> None:
        self.collator = collator

    def __call__(self, examples: list[Any]) -> dict[str, torch.Tensor]:
        sampled = [isinstance(example, SamplingExample) for example in examples]
        if not any(sampled):
            return self.collator(examples)  # input_ids, attention_mask: (b, l_t) and labels, from the collator
        if not all(sampled):
            raise ValueError("a batch cannot mix training sampling indices with evaluation rows")
        batch = self.collator([example.example for example in examples])
        batch[ROW_KEY] = torch.tensor([example.row for example in examples], dtype=torch.long)  # (b,)
        return batch  # the collator fields, plus ROW_KEY: (b,)


class ClusterTrainer(Trainer):
    """Single-process trainer; distributed exposure accounting is deliberately unsupported."""

    def __init__(self, *, sampling_plan: ClusterSamplingPlan, **kwargs: Any) -> None:
        self.sampling_plan = sampling_plan
        self.cluster_sampler = ClusterAssaySampler(sampling_plan, kwargs["args"].seed)
        self.observation_counts: Counter[int] = Counter()
        self.epoch_counts: Counter[int] = Counter()
        self.epoch_digests: dict[int, Any] = {}
        if kwargs["args"].world_size != 1:
            raise ValueError("cluster sampling currently requires one training process for exact exposure accounting")
        if kwargs["args"].dataloader_drop_last:
            raise ValueError("cluster sampling forbids dropping the last training batch")
        kwargs["train_dataset"] = IndexedTrainingRows(kwargs["train_dataset"])
        kwargs["data_collator"] = SamplingCollator(kwargs["data_collator"])
        super().__init__(**kwargs)

    def _get_train_sampler(self, train_dataset: Any = None) -> ClusterAssaySampler:
        return self.cluster_sampler

    def training_step(self, model: torch.nn.Module, inputs: dict[str, Any], num_items_in_batch: Any = None) -> torch.Tensor:
        inputs = dict(inputs)
        rows = inputs.pop(ROW_KEY).detach().cpu().tolist()
        loss = super().training_step(model, inputs, num_items_in_batch)
        self.observation_counts.update(rows)
        epoch = self.cluster_sampler.epoch
        self.epoch_counts[epoch] += len(rows)
        digest = self.epoch_digests.setdefault(epoch, hashlib.sha256())
        for row in rows:
            digest.update(observation_digest_bytes(self.sampling_plan.observation_ids[row]))
        return loss  # ()

    def sampling_receipt(self) -> dict[str, Any]:
        plan = self.sampling_plan
        clusters: Counter[str] = Counter()
        proteins: Counter[str] = Counter()
        for row, count in self.observation_counts.items():
            clusters[plan.cluster_ids[row]] += count
            proteins[plan.protein_ids[row]] += count
        return {
            "plan": plan.record(), "seed": self.cluster_sampler.seed,
            "observation_counts": {plan.observation_ids[row]: count for row, count in sorted(self.observation_counts.items())},
            "protein_counts": dict(sorted(proteins.items())), "cluster_counts": dict(sorted(clusters.items())),
            "epochs": [{"epoch": epoch, "consumed_rows": count, "observation_order_sha256": self.epoch_digests[epoch].hexdigest()}
                       for epoch, count in sorted(self.epoch_counts.items())],
        }
