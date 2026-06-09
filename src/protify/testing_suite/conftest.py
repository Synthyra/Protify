import os
import sys

import numpy as np
import torch
import pytest


collect_ignore = ["embedding_test.py"]


def _normalize_fastplms_imports() -> None:
    fastplms_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "fastplms")
    )
    fastplms_package = os.path.join(fastplms_root, "fastplms")
    assert os.path.isdir(fastplms_package), f"FastPLMs package not found: {fastplms_package}"
    if fastplms_root in sys.path:
        sys.path.remove(fastplms_root)
    sys.path.insert(0, fastplms_root)
    if "fastplms" in sys.modules:
        loaded_fastplms = sys.modules["fastplms"]
        if "__path__" in loaded_fastplms.__dict__:
            loaded_paths = [
                os.path.abspath(path)
                for path in loaded_fastplms.__path__
            ]
        else:
            loaded_paths = []
        if os.path.abspath(fastplms_package) not in loaded_paths:
            for module_name in list(sys.modules):
                if module_name == "fastplms" or module_name.startswith("fastplms."):
                    del sys.modules[module_name]


def pytest_configure(config):
    _normalize_fastplms_imports()
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
    config.addinivalue_line("markers", "slow: slow tests (>10s)")


def pytest_runtest_setup(item):
    _normalize_fastplms_imports()


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
