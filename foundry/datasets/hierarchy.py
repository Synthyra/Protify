"""Nested homology clusters: MMseqs2 run as a cascade, each level clustering the one before.

Every protein is clustered at 0.9 identity, then the 0.9 representatives at 0.7, their
representatives at 0.5, and theirs at 0.3. Each 0.9 cluster therefore lies inside exactly one
cluster at every lower level, so a split can hold out a whole cluster at any level and know
which clusters contain it. Coverage is 0.8 of both sequences at every level, MMseqs's default.
This is the hierarchy the identity-stratified splits hold out from
(docs/decisions/foundry/datasets/2026-09-24_identity_stratified_splits.md).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .mmseqs import ClusterMap, CommandRunner, MMseqsConfig, cluster_sequences


HIERARCHY_LEVELS = (0.9, 0.7, 0.5, 0.3)
HIERARCHY_COVERAGE = 0.8


@dataclass(frozen=True, slots=True)
class ClusterHierarchy:
    """Clusters at each identity level, finest first, and the cluster containing each one.

    A cluster is named by its representative, so one protein can name a cluster at several
    levels; a cluster is always addressed with its level.
    """

    levels: tuple[float, ...]
    # Level to cluster to every protein under it, sorted.
    members: dict[float, ClusterMap]
    # Level to cluster to the cluster containing it at the next coarser level; the coarsest
    # level has none.
    parent: dict[float, dict[str, str]]
    # One `Clustering.origin` per level, finest first; empty for a hierarchy built from memberships.
    origin: tuple[dict[str, object], ...] = ()

    @property
    def finest(self) -> float:
        return self.levels[0]

    @property
    def proteins(self) -> tuple[str, ...]:
        return tuple(sorted(protein for members in self.members[self.finest].values() for protein in members))

    def lineage(self, cluster: str) -> tuple[str, ...]:
        """A finest-level cluster and the cluster containing it at each coarser level, in level order."""
        chain = [cluster]
        for level in self.levels[:-1]:
            chain.append(self.parent[level][chain[-1]])
        return tuple(chain)


def nest_clusterings(
    levels: Sequence[float],
    clusterings: Sequence[Mapping[str, Sequence[str]]],
    origin: Sequence[dict[str, object]] = (),
) -> ClusterHierarchy:
    """Nest one clustering per level into a hierarchy, checking that each covers the level before.

    `clusterings[0]` maps each finest-level representative to its proteins. Every later one maps
    each of its representatives to the representatives of the level before, so its members must
    be exactly those representatives, each once, and each of its representatives must be a
    member of its own cluster, as `parse_memberships` guarantees for MMseqs output.
    """
    levels = tuple(levels)
    if not levels or len(levels) != len(clusterings):
        raise ValueError("a hierarchy needs one clustering per level")
    # Each level against the next, so the last level has no partner.
    if any(not 0.0 < level <= 1.0 for level in levels) or any(
        finer <= coarser for finer, coarser in zip(levels, levels[1:], strict=False)
    ):
        raise ValueError(f"levels must fall strictly within (0, 1], finest first, got {levels}")
    if origin and len(origin) != len(levels):
        raise ValueError("origin needs one record per level")

    members: dict[float, ClusterMap] = {}
    parent: dict[float, dict[str, str]] = {}
    below: ClusterMap | None = None
    for index, (level, clustering) in enumerate(zip(levels, clusterings, strict=True)):
        if not clustering:
            raise ValueError(f"the clustering at {level} is empty")
        seen: dict[str, str] = {}
        for representative, children in clustering.items():
            if isinstance(children, (str, bytes)):
                raise TypeError(f"members of {representative!r} at {level} must be a sequence of identifiers")
            if representative not in children:
                raise ValueError(f"representative {representative!r} at {level} is not a member of its own cluster")
            for child in children:
                if child in seen:
                    raise ValueError(f"{child!r} is in clusters {seen[child]!r} and {representative!r} at {level}")
                seen[child] = representative
        if below is None:
            members[level] = {representative: tuple(sorted(children)) for representative, children in clustering.items()}
        else:
            finer = levels[index - 1]
            missing, unexpected = sorted(set(below) - set(seen)), sorted(set(seen) - set(below))
            if missing or unexpected:
                raise ValueError(
                    f"the clustering at {level} must cover the {finer} representatives exactly: "
                    f"missing={missing[:5]}, unexpected={unexpected[:5]}"
                )
            parent[finer] = dict(sorted(seen.items()))
            members[level] = {
                representative: tuple(sorted(protein for child in children for protein in below[child]))
                for representative, children in clustering.items()
            }
        members[level] = dict(sorted(members[level].items()))
        below = members[level]
    return ClusterHierarchy(levels, members, parent, tuple(origin))


def cluster_hierarchy(
    sequences: Mapping[str, str],
    directory: Path,
    *,
    levels: Sequence[float] = HIERARCHY_LEVELS,
    min_coverage: float = HIERARCHY_COVERAGE,
    runner: CommandRunner | None = None,
    linear_from: float | None = None,
    **settings: Any,
) -> ClusterHierarchy:
    """Cluster `sequences` at each of `levels`, finest first, each level on the representatives of the one before.

    Every level is a `cluster_sequences` stage under `directory`, so a rerun reuses each level
    that finished. `settings` go to `MMseqsConfig`, such as `threads`, `binary`, or
    `expected_version`. A level of `linear_from` identity or more runs `linclust` (see
    `MMseqsConfig.linear`): use it for the finest level of a set with thousands of close variants of
    one scaffold, where the full search never finishes. Coarser levels then cluster the far fewer
    representatives with the full search.
    """
    clusterings: list[ClusterMap] = []
    origin: list[dict[str, object]] = []
    current = dict(sequences)
    for level in levels:
        linear = linear_from is not None and level >= linear_from
        config = MMseqsConfig(min_sequence_identity=level, min_coverage=min_coverage, linear=linear, **settings)
        clustering = cluster_sequences(current, directory, config, runner=runner)
        clusterings.append(clustering.clusters)
        origin.append(clustering.origin)
        current = {representative: sequences[representative] for representative in clustering.clusters}
    return nest_clusterings(levels, clusterings, origin)
