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
    first_draw = torch.randn(5)  # (n=5,)
    set_global_seed(99)
    second_draw = torch.randn(5)  # (n=5,)
    assert torch.equal(first_draw, second_draw)


def test_reproducibility_numpy() -> None:
    set_global_seed(99)
    first_draw = np.random.rand(5)  # (n=5,)
    set_global_seed(99)
    second_draw = np.random.rand(5)  # (n=5,)
    assert np.array_equal(first_draw, second_draw)


def test_reproducibility_random() -> None:
    set_global_seed(99)
    first_draw = [random.random() for _ in range(5)]
    set_global_seed(99)
    second_draw = [random.random() for _ in range(5)]
    assert first_draw == second_draw


def test_seed_worker_deterministic() -> None:
    torch.manual_seed(42)
    seed_worker(0)
    first_worker_draw = np.random.rand(3)  # (n=3,)
    first_random_value = random.random()

    torch.manual_seed(42)
    seed_worker(0)
    second_worker_draw = np.random.rand(3)  # (n=3,)
    second_random_value = random.random()

    assert np.array_equal(first_worker_draw, second_worker_draw)
    assert first_random_value == second_random_value


def test_dataloader_generator_returns_generator() -> None:
    generator = dataloader_generator(42)
    assert isinstance(generator, torch.Generator)
