"""Production Atlas sampling for paired observations."""

from __future__ import annotations

import numpy as np

from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import TypedDict


SamplingId = str | int
SpeciesPair = tuple[SamplingId, SamplingId]


class SamplingPair(TypedDict):
    """Chain clusters and ordered taxonomy stratum for one paired row."""

    species_left: SamplingId | None
    species_right: SamplingId | None
    cluster_left: SamplingId
    cluster_right: SamplingId


class ClusterPatchSampler:
    """Draw a stratum, a chain cluster, then an incident paired row.

    Each patch contains distinct row indices. Repeats remain possible between
    patches and batches. Stratum probability is proportional to its number of
    row-cluster incidences raised to ``stratum_exponent``: production Atlas uses
    0.5, 0 draws strata uniformly, and 1 uses the incidence counts directly.
    A row contributes once per distinct endpoint cluster.

    Taxonomy strata are ordered pairs. Consumers preserve their chosen ordering
    and handle endpoint swaps after sampling. A dataset without taxonomy can
    explicitly assign all rows to one stratum; unknown taxonomy is not inferred.
    """

    def __init__(
        self,
        pairs: Sequence[SamplingPair],
        seed: int,
        patch_size: int = 64,
        stratum_exponent: float = 0.5,
    ) -> None:
        self.pairs = pairs
        self.seed = seed
        self.patch_size = patch_size
        self.stratum_exponent = float(stratum_exponent)

        if not np.isfinite(self.stratum_exponent) or not 0.0 <= self.stratum_exponent <= 1.0:
            raise ValueError("The stratum exponent must be finite and within [0, 1]")

        buckets: dict[SpeciesPair, dict[SamplingId, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, row in enumerate(pairs):
            left_species, right_species = row["species_left"], row["species_right"]
            if left_species is None or right_species is None:
                raise ValueError("Unknown species cannot enter paired sampling")

            species = (left_species, right_species)
            for cluster in set((row["cluster_left"], row["cluster_right"])):
                buckets[species][cluster].append(index)

        self.strata = sorted(buckets)
        self.buckets = {species: dict(buckets[species]) for species in self.strata}
        self.clusters = {species: sorted(self.buckets[species]) for species in self.strata}

        if not self.strata or patch_size <= 0:
            raise ValueError("Cluster sampling requires eligible paired rows and positive patch size")

        # n_strata is the number of ordered taxonomy pairs.
        counts = np.asarray(
            [sum(len(rows) for rows in self.buckets[species].values()) for species in self.strata],
            dtype=np.float64,
        )  # (n_strata,).
        weights = counts ** self.stratum_exponent  # (n_strata,).
        self.stratum_weights = weights / weights.sum()  # (n_strata,), sums to 1.

    def _draw_stratum(self, rng: np.random.Generator) -> SpeciesPair:
        if self.stratum_exponent == 0.0:
            # Keep the historical integer draw so explicit v1 settings reproduce.
            return self.strata[int(rng.integers(len(self.strata)))]

        return self.strata[int(rng.choice(len(self.strata), p=self.stratum_weights))]

    def sample(self, step: int, size: int) -> list[int]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
        selected: list[int] = []
        while len(selected) < size:
            selected.extend(self._draw_patch(rng, min(self.patch_size, size - len(selected))))

        return selected[:size]

    def sample_patches(
        self, step: int, n_patches: int, group_consecutive: int = 1
    ) -> Iterator[list[int]]:
        """Keep patch boundaries for loaders that require one taxonomy stratum.

        A small stratum can yield fewer than ``patch_size`` rows. Consecutive
        groups keep one stratum for consumers that accumulate taxonomy patches.
        Fixed patch counts keep distributed loader step counts equal.
        """
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
        if group_consecutive <= 0:
            raise ValueError("Consecutive patch groups must be positive")
        species: SpeciesPair | None = None
        for index in range(n_patches):
            if index % group_consecutive == 0:
                species = self._draw_stratum(rng)
            yield self._draw_patch(rng, self.patch_size, species)

    def _draw_patch(
        self, rng: np.random.Generator, size: int, species: SpeciesPair | None = None
    ) -> list[int]:
        if species is None:
            species = self._draw_stratum(rng)
        clusters = self.clusters[species]
        patch: list[int] = []
        seen: set[int] = set()
        for _ in range(self.patch_size * 8):
            cluster = clusters[int(rng.integers(len(clusters)))]
            indices = self.buckets[species][cluster]
            index = indices[int(rng.integers(len(indices)))]
            if index not in seen:
                patch.append(index)
                seen.add(index)
            if len(patch) >= size:
                break

        return patch
