import math
import torch

from typing import Any, Sequence
from torch.utils.data import Dataset as TorchDataset


class ParallelRunDataset(TorchDataset):
    """Wrap a pooled embedding dataset with one deterministic index stream per run."""

    def __init__(
        self,
        base_dataset: TorchDataset,
        run_seeds: Sequence[int],
        independent_shuffles: bool = True,
        index_strategy: str = 'permutation',
    ) -> None:
        self.base_dataset = base_dataset
        self.run_seeds = tuple(run_seeds)
        self.independent_shuffles = independent_shuffles
        assert index_strategy in ('permutation', 'affine'), "index_strategy must be 'permutation' or 'affine'."
        self.index_strategy = index_strategy
        self.length = len(base_dataset)
        assert self.length > 0, "ParallelRunDataset requires a non-empty base dataset."
        assert len(self.run_seeds) > 0, "ParallelRunDataset requires at least one run seed."
        if self.index_strategy == 'permutation':
            self._run_indices = self._build_run_indices()  # r tensors, each (n,)
            self._affine_params: tuple[tuple[int, int], ...] = ()
        else:
            self._run_indices = ()
            self._affine_params = self._build_affine_params()
        # Cache shapes are (n, d) and (n, ...) when the pooled dataset is supported.
        self._tensor_embeddings, self._tensor_labels = self._build_tensor_cache_if_supported()

    def _build_run_indices(self) -> tuple[torch.Tensor, ...]:
        run_indices: list[torch.Tensor] = []
        base_indices = torch.arange(self.length, dtype=torch.long)  # (n,)
        for seed in self.run_seeds:
            if self.independent_shuffles:
                generator = torch.Generator()
                generator.manual_seed(seed)
                indices = torch.randperm(self.length, generator=generator)  # (n,)
            else:
                indices = base_indices  # (n,)
            run_indices.append(indices)
        return tuple(run_indices)  # r tensors, each (n,)

    def _build_affine_params(self) -> tuple[tuple[int, int], ...]:
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

    @property
    def uses_tensor_cache(self) -> bool:
        return self._tensor_embeddings is not None

    def _build_tensor_cache_if_supported(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if '__dict__' not in dir(self.base_dataset):
            return None, None
        dataset_state = self.base_dataset.__dict__
        if 'tensors' in dataset_state:
            tensors = dataset_state['tensors']
            if len(tensors) != 2:
                return None, None
            tensor_embeddings, tensor_labels = tensors  # (n, d), (n, ...)
            return tensor_embeddings, tensor_labels  # (n, d), (n, ...)

        required_keys = (
            'embeddings',
            'labels',
            'task_type',
            'full',
            'embedding_standardizer',
        )
        if any(key not in dataset_state for key in required_keys):
            return None, None
        if dataset_state['full']:
            return None, None

        processed_embeddings: list[torch.Tensor] = []
        embedding_standardizer = dataset_state['embedding_standardizer']
        for embedding in dataset_state['embeddings']:
            embedding_t = embedding.float()  # (1, d) or (d,)
            if embedding_standardizer is not None:
                embedding_t = embedding_standardizer.transform_tensor(embedding_t)  # (1, d) or (d,)
            processed_embeddings.append(embedding_t.squeeze(0))  # (d,)
        tensor_embeddings = torch.stack(processed_embeddings)  # (n, d)

        task_type = dataset_state['task_type']
        if task_type in ('multilabel', 'regression', 'sigmoid_regression'):
            tensor_labels = torch.tensor(dataset_state['labels'], dtype=torch.float)  # (n, ...)
        else:
            tensor_labels = torch.tensor(dataset_state['labels'], dtype=torch.long)  # (n, ...)
        return tensor_embeddings, tensor_labels  # (n, d), (n, ...)

    def run_indices_for(self, idx: int) -> tuple[int, ...]:
        assert 0 <= idx < self.length, f"Index {idx} out of bounds for length {self.length}."
        if self.index_strategy == 'permutation':
            return tuple(int(indices[idx].item()) for indices in self._run_indices)
        return tuple(((multiplier * idx) + offset) % self.length for multiplier, offset in self._affine_params)

    def run_index_tensor_for(self, idx: int) -> torch.Tensor:
        assert 0 <= idx < self.length, f"Index {idx} out of bounds for length {self.length}."
        if self.index_strategy == 'permutation':
            return torch.stack([indices[idx] for indices in self._run_indices])  # (r,)
        return torch.tensor(
            [((multiplier * idx) + offset) % self.length for multiplier, offset in self._affine_params],
            dtype=torch.long,
        )  # (r,)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.uses_tensor_cache:
            indices = self.run_index_tensor_for(idx)  # (r,)
            embedding_indices = indices.to(self._tensor_embeddings.device)  # (r,)
            label_indices = indices.to(self._tensor_labels.device)  # (r,)
            return (
                self._tensor_embeddings.index_select(0, embedding_indices),  # (r, d)
                self._tensor_labels.index_select(0, label_indices),  # (r, ...)
            )
        indices = self.run_indices_for(idx)
        items = [self.base_dataset[base_idx] for base_idx in indices]
        return self._stack_items(items)

    def _stack_items(
        self,
        items: list[tuple[torch.Tensor, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first_item = items[0]
        assert isinstance(first_item, tuple), "ParallelRunDataset expects tuple dataset items."
        assert len(first_item) == 2, (
            "ParallelRunDataset currently supports pooled embedding items of "
            "(embedding, label) only."
        )
        embeddings = torch.stack([item[0] for item in items])  # (r, d)
        label_tensors = [
            item[1] if isinstance(item[1], torch.Tensor) else torch.tensor(item[1])  # (...)
            for item in items
        ]
        labels = torch.stack(label_tensors)  # (r, ...)
        return embeddings, labels  # (r, d), (r, ...)
