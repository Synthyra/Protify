import numpy as np
import torch
import pytest


collect_ignore = ["embedding_test.py"]


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
    config.addinivalue_line("markers", "slow: slow tests (>10s)")


@pytest.fixture
def tiny_embeddings():
    """(batch=2, seq_len=4, hidden=16) random embeddings."""
    torch.manual_seed(0)
    return torch.randn(2, 4, 16)


@pytest.fixture
def attention_mask_2d():
    """Bool mask: first sample has 3 real tokens, second has 4."""
    return torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)


@pytest.fixture
def binary_labels_np():
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=50)


@pytest.fixture
def multiclass_labels_np():
    rng = np.random.default_rng(42)
    return rng.integers(0, 3, size=50)


@pytest.fixture
def regression_labels_np():
    rng = np.random.default_rng(42)
    return rng.standard_normal(50)


@pytest.fixture
def multilabel_labels_np():
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=(50, 4))
