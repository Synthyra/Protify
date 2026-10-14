"""Cluster, protein, then assay draws for a declared single-protein training split."""

from __future__ import annotations

import hashlib
import json
import numpy as np

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice, repeat
from typing import Any
from torch.utils.data import Sampler

from .label_batches import SparseLabelRow


ALGORITHM = "cluster_protein_assay_uniform_v1"


@dataclass(frozen=True)
class ClusterSamplingPlan:
    """Training-only membership; masks determine eligibility.

    One default epoch draws as many observations as there are eligible clusters, with
    replacement at all three levels. A configured epoch length changes exposure, not
    cluster probabilities. A protein's repeated assays share that protein's probability.
    Explicit binary class balance stratifies the hierarchy by observed target, drawing
    one member of each class in every consecutive pair. Defaults never inspect targets.
    """

    observation_ids: tuple[str, ...]
    protein_ids: tuple[str, ...]
    cluster_ids: tuple[str, ...]
    # Outer groups are clusters; inside are proteins, each containing eligible row indices.
    groups: tuple[tuple[tuple[int, ...], ...], ...]
    draws_per_epoch: int
    inventory_sha256: str
    class_groups: tuple[tuple[tuple[tuple[int, ...], ...], ...], ...] = ()

    @classmethod
    def from_dataset(cls, dataset: Mapping[str, Sequence[Any]], draws_per_epoch: int | None = None,
                     *, balance_classes: bool = False) -> ClusterSamplingPlan:
        columns = {name: tuple(dataset[name]) for name in ("observation_id", "protein_id", "cluster_id")}
        size = len(columns["observation_id"])
        if not size or any(len(values) != size for values in columns.values()):
            raise ValueError("cluster sampling requires aligned nonempty training metadata")
        for name in ("observation_id", "protein_id", "cluster_id"):
            if any(not isinstance(value, str) or not value.strip() for value in columns[name]):
                raise ValueError(f"cluster sampling requires nonempty {name} strings")
        if len(set(columns["observation_id"])) != size:
            raise ValueError("cluster sampling observation IDs must be unique")
        groups: dict[str, dict[str, list[int]]] = {}
        strata: dict[int, dict[str, dict[str, list[int]]]] = {0: {}, 1: {}}
        memberships: dict[str, str] = {}
        digest = hashlib.sha256(b"[")
        labels = dataset["labels"] if "labels" in getattr(dataset, "column_names", dataset) else repeat(None, size)
        for index, (observation, protein, cluster, mask, label) in enumerate(zip(*columns.values(), dataset["label_mask"], labels, strict=True)):
            if isinstance(mask, Mapping):
                eligible = SparseLabelRow.from_records(label, mask).observed_count > 0
            else:
                observed = np.asarray(mask)  # () or (c,)
                if observed.dtype != np.bool_ or observed.ndim > 1 or observed.size == 0:
                    raise ValueError("cluster sampling masks must be boolean scalars or nonempty vectors")
                eligible = bool(observed.any())
            if memberships.setdefault(protein, cluster) != cluster:
                raise ValueError("a protein cannot belong to multiple sampling clusters")
            if index:
                digest.update(b",")
            digest.update(json.dumps((observation, protein, cluster, eligible), separators=(",", ":")).encode())
            if eligible:
                groups.setdefault(cluster, {}).setdefault(protein, []).append(index)
                if balance_classes:
                    if not isinstance(label, (int, float, np.number)) or label not in (0, 1) or np.asarray(mask).ndim != 0:
                        raise ValueError("class-balanced cluster sampling requires observed scalar binary labels")
                    strata[int(label)].setdefault(cluster, {}).setdefault(protein, []).append(index)
        if not groups:
            raise ValueError("cluster sampling has no observed training targets")
        draws = len(groups) if draws_per_epoch is None else draws_per_epoch
        if type(draws) is not int or draws < 1:
            raise ValueError("cluster draws per epoch must be a positive integer")
        if balance_classes and (draws % 2 or not all(strata.values())):
            raise ValueError("class-balanced cluster sampling requires both classes and an even epoch length")
        ordered = tuple(
            tuple(tuple(sorted(rows, key=lambda row: columns["observation_id"][row])) for _, rows in sorted(proteins.items()))
            for _, proteins in sorted(groups.items())
        )
        digest.update(b"]")
        class_groups = tuple(
            tuple(tuple(tuple(sorted(rows, key=lambda row: columns["observation_id"][row]))
                        for _, rows in sorted(proteins.items())) for _, proteins in sorted(strata[label].items()))
            for label in (0, 1)
        ) if balance_classes else ()
        if balance_classes:
            digest.update(json.dumps(class_groups, separators=(",", ":")).encode())
        return cls(columns["observation_id"], columns["protein_id"], columns["cluster_id"], ordered, draws, digest.hexdigest(), class_groups)

    def record(self) -> dict[str, Any]:
        record = {
            "algorithm": ALGORITHM, "inventory_sha256": self.inventory_sha256,
            "draws_per_epoch": self.draws_per_epoch, "replacement": True,
            "training_observations": len(self.observation_ids), "eligible_clusters": len(self.groups),
            "eligible_proteins": sum(len(cluster) for cluster in self.groups),
            "eligible_observations": sum(len(protein) for cluster in self.groups for protein in cluster),
        }
        if self.class_groups:
            record.update(algorithm="class_stratified_cluster_protein_assay_v1", class_balance="one draw per class in each consecutive pair",
                          class_eligible_clusters=[len(groups) for groups in self.class_groups])
        return record


class ClusterAssaySampler(Sampler[int]):
    """Epoch-addressed RNG, independent of global state and loader prefetch."""

    def __init__(self, plan: ClusterSamplingPlan, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("cluster sampling seed must be a nonnegative integer")
        self.plan = plan
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("cluster sampling epoch must be a nonnegative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.plan.draws_per_epoch

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 731]))
        if self.plan.class_groups:
            for _ in range(len(self) // 2):
                first = int(rng.integers(2))
                for label in (first, 1 - first):
                    groups = self.plan.class_groups[label]
                    cluster = groups[int(rng.integers(len(groups)))]
                    protein = cluster[int(rng.integers(len(cluster)))]
                    yield protein[int(rng.integers(len(protein)))]
            return
        for _ in range(len(self)):
            cluster = self.plan.groups[int(rng.integers(len(self.plan.groups)))]
            protein = cluster[int(rng.integers(len(cluster)))]
            yield protein[int(rng.integers(len(protein)))]


def observation_digest_bytes(observation: str) -> bytes:
    """Unambiguous record delimiters even if a source observation ID contains a newline."""
    return (json.dumps(observation, ensure_ascii=True) + "\n").encode()


def validate_sampling_receipt(plan: ClusterSamplingPlan, receipt: Mapping[str, Any], seed: int) -> None:
    """Reconstruct each consumed epoch prefix and compare every reported exposure."""
    if not isinstance(receipt, Mapping) or set(receipt) != {"plan", "seed", "observation_counts", "protein_counts", "cluster_counts", "epochs"}:
        raise ValueError("missing or invalid cluster sampling receipt")
    if receipt["plan"] != plan.record() or type(receipt["seed"]) is not int or receipt["seed"] != seed:
        raise ValueError("sampling receipt belongs to a different plan or seed")
    epochs = receipt["epochs"]
    if not isinstance(epochs, list) or not epochs:
        raise ValueError("sampling receipt must record consumed training epochs")
    counters: dict[str, Counter[str]] = {key: Counter() for key in ("observation_counts", "protein_counts", "cluster_counts")}
    sampler = ClusterAssaySampler(plan, seed)
    for index, record in enumerate(epochs):
        if not isinstance(record, dict) or set(record) != {"epoch", "consumed_rows", "observation_order_sha256"}:
            raise ValueError("invalid sampling epoch record")
        count = record["consumed_rows"]
        if type(record["epoch"]) is not int or record["epoch"] != index or type(count) is not int or not 1 <= count <= len(sampler):
            raise ValueError("invalid sampling epoch or consumed row count")
        if index < len(epochs) - 1 and count != len(sampler):
            raise ValueError("only the final sampling epoch may be partial")
        sampler.set_epoch(index)
        digest = hashlib.sha256()
        for row in islice(sampler, count):
            digest.update(observation_digest_bytes(plan.observation_ids[row]))
            for key, values in (("observation_counts", plan.observation_ids), ("protein_counts", plan.protein_ids), ("cluster_counts", plan.cluster_ids)):
                counters[key][values[row]] += 1
        if digest.hexdigest() != record["observation_order_sha256"]:
            raise ValueError("sampling receipt row order differs from its deterministic epoch prefix")
    for key, expected in counters.items():
        counts = receipt[key]
        if not isinstance(counts, dict) or any(type(value) is not int or value < 1 for value in counts.values()) or counts != dict(expected):
            raise ValueError(f"sampling receipt {key} differs from the consumed epoch draws")
