"""Seeded parity and production weighting for paired-row sampling."""

from __future__ import annotations

import numpy as np
import pytest

from collections import defaultdict
from collections.abc import Sequence

from foundry.datasets.pair_sampling import ClusterPatchSampler, SamplingPair


class FrozenAtlasSampler:
    """Pre-promotion Atlas body, retained as an independent seeded reference."""

    def __init__(
        self,
        pairs: Sequence[SamplingPair],
        seed: int,
        patch_size: int = 64,
        stratum_exponent: float = 0.0,
    ) -> None:
        self.pairs = pairs
        self.seed = seed
        self.patch_size = patch_size
        self.stratum_exponent = float(stratum_exponent)
        if not np.isfinite(self.stratum_exponent) or not 0.0 <= self.stratum_exponent <= 1.0:
            raise ValueError("The stratum exponent must be finite and within [0, 1]")
        buckets = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(pairs):
            species = (row["species_left"], row["species_right"])
            if None in species:
                raise ValueError("Unknown species cannot enter contrastive sampling")
            for cluster in set((row["cluster_left"], row["cluster_right"])):
                buckets[species][cluster].append(index)
        self.strata = sorted(buckets)
        self.buckets = {species: dict(buckets[species]) for species in self.strata}
        self.clusters = {species: sorted(self.buckets[species]) for species in self.strata}
        if not self.strata or patch_size <= 0:
            raise ValueError("Cluster sampling requires eligible positive rows and positive patch size")
        counts = np.asarray(
            [sum(len(rows) for rows in self.buckets[species].values()) for species in self.strata],
            dtype=np.float64,
        )  # (n_strata,).
        weights = counts ** self.stratum_exponent  # (n_strata,).
        self.stratum_weights = weights / weights.sum()  # (n_strata,).

    def _draw_stratum(self, rng: np.random.Generator) -> tuple[str | int | None, str | int | None]:
        if self.stratum_exponent == 0.0:
            return self.strata[int(rng.integers(len(self.strata)))]
        return self.strata[int(rng.choice(len(self.strata), p=self.stratum_weights))]

    def sample(self, step: int, size: int) -> list[int]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
        selected = []
        while len(selected) < size:
            species = self._draw_stratum(rng)
            clusters = self.clusters[species]
            patch = []
            seen = set()
            for _ in range(self.patch_size * 8):
                cluster = clusters[int(rng.integers(len(clusters)))]
                indices = self.buckets[species][cluster]
                index = indices[int(rng.integers(len(indices)))]
                if index not in seen:
                    patch.append(index)
                    seen.add(index)
                if len(patch) >= min(self.patch_size, size - len(selected)):
                    break
            selected.extend(patch)
        return selected[:size]


def sampling_rows() -> list[SamplingPair]:
    rows: list[SamplingPair] = []
    for species_left, species_right, count in [(9606, 9606, 40), (562, 562, 13), (562, 9606, 7)]:
        for index in range(count):
            rows.append({
                "species_left": species_left,
                "species_right": species_right,
                "cluster_left": f"{species_left}_{index % 9}",
                "cluster_right": f"{species_right}_{(index * 3) % 11}",
            })
    return rows


@pytest.mark.parametrize("exponent", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("patch_size", [1, 8, 64])
def test_explicit_settings_preserve_seeded_atlas_draws(exponent: float, patch_size: int) -> None:
    rows = sampling_rows()
    expected = FrozenAtlasSampler(rows, 41, patch_size, exponent)
    actual = ClusterPatchSampler(rows, 41, patch_size, exponent)
    np.testing.assert_array_equal(actual.stratum_weights, expected.stratum_weights)
    for step in [0, 1, 17, 100]:
        for size in [0, 1, 7, 64, 131]:
            assert actual.sample(step, size) == expected.sample(step, size)


def test_default_uses_square_root_incidence_weighting() -> None:
    rows: list[SamplingPair] = [
        {"species_left": 1, "species_right": 1, "cluster_left": "a", "cluster_right": "a"},
        {"species_left": 2, "species_right": 2, "cluster_left": "b", "cluster_right": "c"},
        {"species_left": 2, "species_right": 2, "cluster_left": "c", "cluster_right": "d"},
    ]
    sampler = ClusterPatchSampler(rows, 0)
    assert sampler.patch_size == 64
    assert sampler.stratum_exponent == 0.5
    np.testing.assert_allclose(sampler.stratum_weights, [1 / 3, 2 / 3])
    assert sampler.buckets[(1, 1)]["a"] == [0]


def test_patch_is_one_stratum_and_has_no_duplicate_row_indices() -> None:
    sampler = ClusterPatchSampler(sampling_rows(), 41, patch_size=4)
    draws = sampler.sample(step=0, size=40)
    rows = sampler.pairs
    for offset in range(0, 40, 4):
        patch = draws[offset:offset + 4]
        assert len(set(patch)) == 4
        strata = {(rows[index]["species_left"], rows[index]["species_right"]) for index in patch}
        assert len(strata) == 1


def test_sparse_stratum_can_repeat_across_partial_patches() -> None:
    row: SamplingPair = {
        "species_left": "unknown_dataset",
        "species_right": "unknown_dataset",
        "cluster_left": "a",
        "cluster_right": "b",
    }
    sampler = ClusterPatchSampler([row], 12)
    assert sampler.sample(0, 70) == [0] * 70


def test_determinism_does_not_depend_on_other_draws() -> None:
    sampler = ClusterPatchSampler(sampling_rows(), 41)
    expected = sampler.sample(17, 192)
    sampler.sample(3, 300)
    np.random.default_rng(12).random(20)  # (20,).
    assert sampler.sample(17, 192) == expected
    assert ClusterPatchSampler(sampling_rows(), 41).sample(17, 192) == expected
    assert sampler.sample(18, 192) != expected


@pytest.mark.parametrize("side", ["species_left", "species_right"])
def test_missing_taxonomy_is_refused(side: str) -> None:
    rows = sampling_rows()
    rows[0][side] = None
    with pytest.raises(ValueError, match="Unknown species"):
        ClusterPatchSampler(rows, 41)


@pytest.mark.parametrize("exponent", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_stratum_exponent_is_refused(exponent: float) -> None:
    with pytest.raises(ValueError, match="stratum exponent"):
        ClusterPatchSampler(sampling_rows(), 41, stratum_exponent=exponent)


def test_empty_rows_or_nonpositive_patch_are_refused() -> None:
    with pytest.raises(ValueError, match="eligible paired rows"):
        ClusterPatchSampler([], 41)
    with pytest.raises(ValueError, match="positive patch size"):
        ClusterPatchSampler(sampling_rows(), 41, patch_size=0)


def test_partial_patch_boundaries_remain_one_stratum() -> None:
    rows = sampling_rows()
    sampler = ClusterPatchSampler(rows, 41, patch_size=64)
    patches = list(sampler.sample_patches(0, 8))
    assert len(patches) == 8
    assert any(len(patch) < 64 for patch in patches)
    for patch in patches:
        assert len(patch) == len(set(patch))
        assert len({(rows[index]["species_left"], rows[index]["species_right"]) for index in patch}) == 1


def test_consecutive_patch_groups_keep_one_stratum() -> None:
    rows = sampling_rows()
    sampler = ClusterPatchSampler(rows, 41, patch_size=8)
    patches = list(sampler.sample_patches(0, 12, group_consecutive=4))
    for offset in range(0, 12, 4):
        assert len({
            (rows[index]["species_left"], rows[index]["species_right"])
            for patch in patches[offset:offset + 4] for index in patch
        }) == 1
    with pytest.raises(ValueError, match="groups must be positive"):
        list(sampler.sample_patches(0, 1, group_consecutive=0))
