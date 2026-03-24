"""Tests for seed_utils.py reproducibility utilities."""

import random

import numpy as np
import torch

try:
    from src.protify.seed_utils import set_global_seed, get_global_seed, seed_worker, dataloader_generator
except ImportError:
    try:
        from protify.seed_utils import set_global_seed, get_global_seed, seed_worker, dataloader_generator
    except ImportError:
        from ..seed_utils import set_global_seed, get_global_seed, seed_worker, dataloader_generator


def test_set_global_seed_returns_seed() -> None:
    result = set_global_seed(42)
    assert result == 42


def test_get_global_seed_after_set() -> None:
    set_global_seed(123)
    assert get_global_seed() == 123


def test_set_global_seed_none_generates_seed() -> None:
    result = set_global_seed(None)
    assert isinstance(result, int)
    assert result >= 0


def test_reproducibility_torch() -> None:
    set_global_seed(99)
    a = torch.randn(5)
    set_global_seed(99)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_reproducibility_numpy() -> None:
    set_global_seed(99)
    a = np.random.rand(5)
    set_global_seed(99)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_reproducibility_random() -> None:
    set_global_seed(99)
    a = [random.random() for _ in range(5)]
    set_global_seed(99)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_worker_deterministic() -> None:
    torch.manual_seed(42)
    seed_worker(0)
    a = np.random.rand(3)
    val_a = random.random()

    torch.manual_seed(42)
    seed_worker(0)
    b = np.random.rand(3)
    val_b = random.random()

    assert np.array_equal(a, b)
    assert val_a == val_b


def test_dataloader_generator_returns_generator() -> None:
    g = dataloader_generator(42)
    assert isinstance(g, torch.Generator)
