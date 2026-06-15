import math
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset as TorchDataset


class ParallelRunDataset(TorchDataset):
    """Wrap a pooled embedding dataset with one deterministic index stream per run."""

    def __init__(
            self,
            base_dataset: TorchDataset,
            run_seeds: Sequence[int],
            independent_shuffles: bool = True,
            index_strategy: str = 'permutation',
        ):
        self.base_dataset = base_dataset
        self.run_seeds = tuple(run_seeds)
        self.independent_shuffles = independent_shuffles
        assert index_strategy in ('permutation', 'affine'), "index_strategy must be 'permutation' or 'affine'."
        self.index_strategy = index_strategy
        self.length = len(base_dataset)
        assert self.length > 0, "ParallelRunDataset requires a non-empty base dataset."
        assert len(self.run_seeds) > 0, "ParallelRunDataset requires at least one run seed."
        if self.index_strategy == 'permutation':
            self._run_indices = self._build_run_indices()
            self._affine_params: Tuple[Tuple[int, int], ...] = ()
        else:
            self._run_indices = ()
            self._affine_params = self._build_affine_params()

    def _build_run_indices(self) -> Tuple[torch.Tensor, ...]:
        run_indices = []
        base_indices = torch.arange(self.length, dtype=torch.long)
        for seed in self.run_seeds:
            if self.independent_shuffles:
                generator = torch.Generator()
                generator.manual_seed(seed)
                indices = torch.randperm(self.length, generator=generator)
            else:
                indices = base_indices
            run_indices.append(indices)
        return tuple(run_indices)

    def _build_affine_params(self) -> Tuple[Tuple[int, int], ...]:
        params = []
        for seed in self.run_seeds:
            if self.independent_shuffles:
                multiplier = (abs(seed) * 2) + 1
                while math.gcd(multiplier, self.length) != 1:
                    multiplier += 2
                offset = abs(seed) % self.length
            else:
                multiplier = 1
                offset = 0
            params.append((multiplier, offset))
        return tuple(params)

    def __len__(self) -> int:
        return self.length

    @property
    def num_runs(self) -> int:
        return len(self.run_seeds)

    @property
    def index_memory_bytes(self) -> int:
        return sum(indices.nelement() * indices.element_size() for indices in self._run_indices)

    def run_indices_for(self, idx: int) -> Tuple[int, ...]:
        assert 0 <= idx < self.length, f"Index {idx} out of bounds for length {self.length}."
        if self.index_strategy == 'permutation':
            return tuple(int(indices[idx].item()) for indices in self._run_indices)
        return tuple(((multiplier * idx) + offset) % self.length for multiplier, offset in self._affine_params)

    def __getitem__(self, idx: int):
        indices = self.run_indices_for(idx)
        items = [self.base_dataset[base_idx] for base_idx in indices]
        return self._stack_items(items)

    def _stack_items(self, items: List[Tuple[torch.Tensor, torch.Tensor]]):
        first_item = items[0]
        assert isinstance(first_item, tuple), "ParallelRunDataset expects tuple dataset items."
        assert len(first_item) == 2, (
            "ParallelRunDataset currently supports pooled embedding items of "
            "(embedding, label) only."
        )
        embeddings = torch.stack([item[0] for item in items])
        label_tensors = [
            item[1] if isinstance(item[1], torch.Tensor) else torch.tensor(item[1])
            for item in items
        ]
        labels = torch.stack(label_tensors)
        return embeddings, labels
