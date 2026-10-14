"""Homology-disjoint splits: whole clusters assigned toward target row counts.

Every member of a cluster goes to the cluster's split, so no protein is evaluated against a
homolog it was trained on. Targets are counted in proteins, not clusters, because cluster sizes
are skewed and a 5% cluster share can hold far more or far fewer than 5% of the rows. Each
split with a positive target is first seeded with one cluster, the rest are placed largest
first where they most reduce the distance from the targets, and single-cluster moves then
refine the result.

Ties are broken by a SHA-256 rank of the seed and the cluster identifier, never by input order
or by Python's global random state, so the same clusters and seed give the same splits on any
machine. Split names follow Hugging Face: `train`, `validation`, `test`.

Promoted from annotation_vocabulary's `splitting.py`, its only consumer so far, ahead of a
second because Logan's plan names splits as a step of `foundry.datasets`.
"""

from __future__ import annotations

import hashlib
import math

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


# New datasets split by identity strata (`stratified.py`, docs/conventions/splits.md). This
# three-way assignment stays, so annotation_vocabulary's published splits still rebuild.
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Requested fractions of protein rows in each split."""

    train: float = 0.90
    validation: float = 0.05
    test: float = 0.05

    def __post_init__(self) -> None:
        values = self.as_dict()
        for split, ratio in values.items():
            if not 0.0 < ratio < 1.0:
                raise ValueError(f"{split} ratio must be in (0, 1)")
        if not math.isclose(sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Split ratios must sum to 1")

    def as_dict(self) -> dict[str, float]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


DEFAULT_RATIOS = SplitRatios()


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """Each cluster's and each accession's split, with requested and observed row counts."""

    cluster_to_split: dict[str, str]
    accession_to_split: dict[str, str]
    target_rows: dict[str, int]
    actual_rows: dict[str, int]
    ratios: SplitRatios
    seed: int

    def to_dict(self) -> dict[str, object]:
        """A JSON-compatible split manifest."""
        return {
            "ratios": self.ratios.as_dict(),
            "seed": self.seed,
            "target_rows": dict(self.target_rows),
            "actual_rows": dict(self.actual_rows),
            "cluster_to_split": dict(sorted(self.cluster_to_split.items())),
            "accession_to_split": dict(sorted(self.accession_to_split.items())),
        }


def assign_clusters_to_splits(
    cluster_to_accessions: Mapping[str, Sequence[str]],
    ratios: SplitRatios = DEFAULT_RATIOS,
    seed: int = 42,
    *,
    expected_accessions: Sequence[str] | None = None,
    refinement_passes: int = 4,
) -> SplitAssignment:
    """Assign each whole cluster toward the target row counts, then verify the result.

    `cluster_to_accessions` is what `Clustering.clusters` holds. `expected_accessions`, when
    given, must be exactly the clustered accessions, which catches a clustering run that
    silently dropped proteins.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("Split seed must be an integer")
    if refinement_passes < 0:
        raise ValueError("refinement_passes must be nonnegative")

    clusters = _normalize_clusters(cluster_to_accessions, expected_accessions)
    cluster_sizes = {cluster_id: len(accessions) for cluster_id, accessions in clusters.items()}
    target_rows = _target_row_counts(sum(cluster_sizes.values()), ratios)
    positive_target_splits = [split for split in SPLIT_NAMES if target_rows[split] > 0]
    if len(clusters) < len(positive_target_splits):
        raise ValueError("Too few clusters to give every positive-target split a complete cluster")

    cluster_to_split: dict[str, str] = {}
    actual_rows = {split: 0 for split in SPLIT_NAMES}
    cluster_counts = {split: 0 for split in SPLIT_NAMES}

    def assign(cluster_id: str, split: str) -> None:
        cluster_to_split[cluster_id] = split
        actual_rows[split] += cluster_sizes[cluster_id]
        cluster_counts[split] += 1

    # Seed every split that should hold rows: the smaller targets with the smallest clusters,
    # so no split starts over its target, and the largest target with the largest cluster.
    available = set(clusters)
    if positive_target_splits:
        largest_target_split = min(
            positive_target_splits, key=lambda split: (-target_rows[split], SPLIT_NAMES.index(split))
        )
        smaller_target_splits = sorted(
            (split for split in positive_target_splits if split != largest_target_split),
            key=lambda split: (target_rows[split], SPLIT_NAMES.index(split)),
        )
        for split in smaller_target_splits:
            cluster_id = min(
                available,
                key=lambda candidate: (cluster_sizes[candidate], _seed_rank(seed, f"seed:{split}:{candidate}"), candidate),
            )
            assign(cluster_id, split)
            available.remove(cluster_id)

        cluster_id = min(
            available,
            key=lambda candidate: (
                -cluster_sizes[candidate],
                _seed_rank(seed, f"seed:{largest_target_split}:{candidate}"),
                candidate,
            ),
        )
        assign(cluster_id, largest_target_split)
        available.remove(cluster_id)

    ordered_clusters = sorted(
        available,
        key=lambda cluster_id: (-cluster_sizes[cluster_id], _seed_rank(seed, f"order:{cluster_id}"), cluster_id),
    )
    for cluster_id in ordered_clusters:
        size = cluster_sizes[cluster_id]
        split = min(
            SPLIT_NAMES,
            key=lambda candidate: (
                _objective_after_add(actual_rows, target_rows, candidate, size),
                _seed_rank(seed, f"assign:{cluster_id}:{candidate}"),
                SPLIT_NAMES.index(candidate),
            ),
        )
        assign(cluster_id, split)

    _refine_assignment(cluster_to_split, cluster_sizes, actual_rows, cluster_counts, target_rows, seed, refinement_passes)
    accession_to_split = {
        accession: cluster_to_split[cluster_id] for cluster_id in sorted(clusters) for accession in clusters[cluster_id]
    }
    assignment = SplitAssignment(
        cluster_to_split=dict(sorted(cluster_to_split.items())),
        accession_to_split=dict(sorted(accession_to_split.items())),
        target_rows=target_rows,
        actual_rows=dict(actual_rows),
        ratios=ratios,
        seed=seed,
    )
    validate_split_assignment(clusters, assignment, expected_accessions=expected_accessions)
    return assignment


def validate_split_assignment(
    cluster_to_accessions: Mapping[str, Sequence[str]],
    assignment: SplitAssignment,
    *,
    expected_accessions: Sequence[str] | None = None,
) -> None:
    """Raise unless every accession is assigned, no cluster spans splits, and the counts hold.

    Written to run on an assignment read back from disk as well as on a fresh one, so a split
    manifest can be checked independently of the code that produced it.
    """
    clusters = _normalize_clusters(cluster_to_accessions, expected_accessions)
    expected_clusters = set(clusters)
    assigned_clusters = set(assignment.cluster_to_split)
    if assigned_clusters != expected_clusters:
        missing = sorted(expected_clusters - assigned_clusters)
        unexpected = sorted(assigned_clusters - expected_clusters)
        raise ValueError(
            f"Cluster assignment does not match cluster memberships: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    expected_accession_set = {accession for accessions in clusters.values() for accession in accessions}
    assigned_accessions = set(assignment.accession_to_split)
    if assigned_accessions != expected_accession_set:
        missing = sorted(expected_accession_set - assigned_accessions)
        unexpected = sorted(assigned_accessions - expected_accession_set)
        raise ValueError(f"Accession assignment is incomplete: missing={missing[:5]}, unexpected={unexpected[:5]}")

    observed_rows = {split: 0 for split in SPLIT_NAMES}
    for cluster_id, accessions in clusters.items():
        cluster_split = assignment.cluster_to_split[cluster_id]
        if cluster_split not in SPLIT_NAMES:
            raise ValueError(f"Cluster {cluster_id!r} has unknown split {cluster_split!r}")
        observed_rows[cluster_split] += len(accessions)
        for accession in accessions:
            accession_split = assignment.accession_to_split[accession]
            if accession_split not in SPLIT_NAMES:
                raise ValueError(f"Accession {accession!r} has unknown split {accession_split!r}")
            if accession_split != cluster_split:
                raise ValueError(
                    f"Cluster {cluster_id!r} spans splits: cluster={cluster_split!r}, "
                    f"accession {accession!r}={accession_split!r}"
                )

    if assignment.actual_rows != observed_rows:
        raise ValueError(f"Recorded split row counts are incorrect: {assignment.actual_rows} != {observed_rows}")
    target_rows = _target_row_counts(len(expected_accession_set), assignment.ratios)
    if assignment.target_rows != target_rows:
        raise ValueError(f"Recorded target row counts are incorrect: {assignment.target_rows} != {target_rows}")
    for split in SPLIT_NAMES:
        if target_rows[split] > 0 and observed_rows[split] == 0:
            raise ValueError(f"Positive-target split {split!r} is empty")


def _normalize_clusters(
    cluster_to_accessions: Mapping[str, Sequence[str]],
    expected_accessions: Sequence[str] | None,
) -> dict[str, tuple[str, ...]]:
    if not cluster_to_accessions:
        raise ValueError("Cluster memberships are empty")

    normalized: dict[str, tuple[str, ...]] = {}
    accession_to_cluster: dict[str, str] = {}
    for cluster_id in sorted(cluster_to_accessions):
        if not cluster_id or cluster_id.strip() != cluster_id:
            raise ValueError("Cluster identifiers must be non-empty and stripped")
        raw_accessions = cluster_to_accessions[cluster_id]
        if isinstance(raw_accessions, (str, bytes)):
            raise TypeError(f"Cluster {cluster_id!r} members must be a sequence of accessions")
        accessions = tuple(sorted(raw_accessions))
        if not accessions:
            raise ValueError(f"Cluster {cluster_id!r} is empty")
        if len(set(accessions)) != len(accessions):
            raise ValueError(f"Cluster {cluster_id!r} contains duplicate accessions")
        for accession in accessions:
            if not accession or accession.strip() != accession:
                raise ValueError("Accessions must be non-empty and stripped")
            previous_cluster = accession_to_cluster.get(accession)
            if previous_cluster is not None:
                raise ValueError(f"Accession {accession!r} occurs in clusters {previous_cluster!r} and {cluster_id!r}")
            accession_to_cluster[accession] = cluster_id
        normalized[cluster_id] = accessions

    if expected_accessions is not None:
        expected_list = list(expected_accessions)
        expected_set = set(expected_list)
        if len(expected_set) != len(expected_list):
            raise ValueError("Expected accessions contain duplicates")
        observed_set = set(accession_to_cluster)
        missing = sorted(expected_set - observed_set)
        unexpected = sorted(observed_set - expected_set)
        if missing or unexpected:
            raise ValueError(
                f"Cluster memberships do not match expected accessions: missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
    return normalized


def _target_row_counts(total_rows: int, ratios: SplitRatios) -> dict[str, int]:
    """Largest-remainder rounding, so the targets always sum to `total_rows`."""
    raw_targets = {split: total_rows * ratio for split, ratio in ratios.as_dict().items()}
    targets = {split: math.floor(raw_targets[split]) for split in SPLIT_NAMES}
    remaining = total_rows - sum(targets.values())
    remainder_order = sorted(
        SPLIT_NAMES, key=lambda split: (-(raw_targets[split] - targets[split]), SPLIT_NAMES.index(split))
    )
    for split in remainder_order[:remaining]:
        targets[split] += 1
    return targets


def _objective(actual_rows: Mapping[str, int], target_rows: Mapping[str, int]) -> float:
    """Squared shortfall or excess per split, relative to its target, so small splits count."""
    return sum((actual_rows[split] - target_rows[split]) ** 2 / max(1, target_rows[split]) for split in SPLIT_NAMES)


def _objective_after_add(actual_rows: Mapping[str, int], target_rows: Mapping[str, int], split: str, size: int) -> float:
    candidate_rows = dict(actual_rows)
    candidate_rows[split] += size
    return _objective(candidate_rows, target_rows)


def _refine_assignment(
    cluster_to_split: dict[str, str],
    cluster_sizes: Mapping[str, int],
    actual_rows: dict[str, int],
    cluster_counts: dict[str, int],
    target_rows: Mapping[str, int],
    seed: int,
    refinement_passes: int,
) -> None:
    """Move single clusters while a move improves the objective, never emptying a split."""
    ordered_clusters = sorted(
        cluster_to_split,
        key=lambda cluster_id: (cluster_sizes[cluster_id], _seed_rank(seed, f"refine:{cluster_id}"), cluster_id),
    )
    current_score = _objective(actual_rows, target_rows)
    for _ in range(refinement_passes):
        improved = False
        for cluster_id in ordered_clusters:
            source_split = cluster_to_split[cluster_id]
            if target_rows[source_split] > 0 and cluster_counts[source_split] == 1:
                continue
            size = cluster_sizes[cluster_id]
            candidates: list[tuple[float, int, int, str]] = []
            for destination_split in SPLIT_NAMES:
                if destination_split == source_split:
                    continue
                candidate_rows = dict(actual_rows)
                candidate_rows[source_split] -= size
                candidate_rows[destination_split] += size
                candidate_score = _objective(candidate_rows, target_rows)
                if candidate_score + 1.0e-12 < current_score:
                    candidates.append(
                        (
                            candidate_score,
                            _seed_rank(seed, f"move:{cluster_id}:{source_split}:{destination_split}"),
                            SPLIT_NAMES.index(destination_split),
                            destination_split,
                        )
                    )
            if not candidates:
                continue
            best_score, _, _, destination_split = min(candidates)
            actual_rows[source_split] -= size
            actual_rows[destination_split] += size
            cluster_counts[source_split] -= 1
            cluster_counts[destination_split] += 1
            cluster_to_split[cluster_id] = destination_split
            current_score = best_score
            improved = True
        if not improved:
            return


def _seed_rank(seed: int, value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{value}".encode()).digest()[:8], "big")
