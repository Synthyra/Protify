import pytest
import torch
from torch.utils.data import TensorDataset

try:
    from src.protify import probes as probes_package
    from src.protify.data.data_collators import EmbedsLabelsCollator
    from src.protify.probes.parallel_probe_batches import ParallelRunDataset
except ImportError:
    try:
        from protify import probes as probes_package
        from protify.data.data_collators import EmbedsLabelsCollator
        from protify.probes.parallel_probe_batches import ParallelRunDataset
    except ImportError:
        from .. import probes as probes_package
        from ..data.data_collators import EmbedsLabelsCollator
        from ..probes.parallel_probe_batches import ParallelRunDataset


def _base_dataset() -> TensorDataset:
    embeddings = torch.arange(24, dtype=torch.float32).view(6, 4)
    labels = torch.arange(6, dtype=torch.long)
    return TensorDataset(embeddings, labels)


def test_parallel_run_dataset_package_export() -> None:
    assert "ParallelRunDataset" in probes_package.__all__
    assert probes_package.ParallelRunDataset is ParallelRunDataset


def test_parallel_run_dataset_uses_deterministic_independent_indices() -> None:
    dataset = ParallelRunDataset(_base_dataset(), run_seeds=(3, 7, 11), independent_shuffles=True)
    first_indices = dataset.run_indices_for(0)
    rebuilt = ParallelRunDataset(_base_dataset(), run_seeds=(3, 7, 11), independent_shuffles=True)

    assert dataset.num_runs == 3
    assert dataset.index_memory_bytes == 3 * len(_base_dataset()) * 8
    assert dataset.run_indices_for(0) == rebuilt.run_indices_for(0)
    assert len(set(first_indices)) > 1

    embeddings, labels = dataset[0]
    base_embeddings, base_labels = _base_dataset().tensors
    for run_idx, base_idx in enumerate(first_indices):
        assert torch.equal(embeddings[run_idx], base_embeddings[base_idx])
        assert torch.equal(labels[run_idx], base_labels[base_idx])


def test_parallel_run_dataset_can_share_order_across_runs() -> None:
    dataset = ParallelRunDataset(_base_dataset(), run_seeds=(3, 7, 11), independent_shuffles=False)

    assert dataset.run_indices_for(0) == (0, 0, 0)
    assert dataset.run_indices_for(4) == (4, 4, 4)


def test_parallel_run_dataset_affine_strategy_is_bijective_without_index_storage() -> None:
    dataset = ParallelRunDataset(
        _base_dataset(),
        run_seeds=(3, 7),
        independent_shuffles=True,
        index_strategy='affine',
    )

    assert dataset.index_memory_bytes == 0
    for run_idx in range(dataset.num_runs):
        seen = [dataset.run_indices_for(idx)[run_idx] for idx in range(len(dataset))]
        assert sorted(seen) == list(range(len(dataset)))


def test_parallel_run_dataset_rejects_invalid_index_strategy() -> None:
    with pytest.raises(AssertionError, match="index_strategy"):
        ParallelRunDataset(_base_dataset(), run_seeds=(3,), index_strategy='unknown')


def test_parallel_run_dataset_collates_to_run_specific_probe_shape() -> None:
    dataset = ParallelRunDataset(_base_dataset(), run_seeds=(3, 7), independent_shuffles=True)
    collator = EmbedsLabelsCollator(full=False, task_type='singlelabel', tokenwise=False)

    batch = collator([dataset[0], dataset[1], dataset[2]])

    assert batch['embeddings'].shape == (3, 2, 4)
    assert batch['labels'].shape == (3, 2)
    assert batch['labels'].dtype == torch.long


def test_parallel_run_dataset_rejects_empty_base_dataset() -> None:
    empty = TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long))

    with pytest.raises(AssertionError, match="non-empty"):
        ParallelRunDataset(empty, run_seeds=(1,))


def test_parallel_run_dataset_rejects_pair_items_for_now() -> None:
    pair_dataset = TensorDataset(
        torch.randn(4, 2),
        torch.randn(4, 2),
        torch.arange(4, dtype=torch.long),
    )
    dataset = ParallelRunDataset(pair_dataset, run_seeds=(1, 2))

    with pytest.raises(AssertionError, match="pooled embedding items"):
        dataset[0]
