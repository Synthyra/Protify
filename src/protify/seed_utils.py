"""
Global seed management utilities for reproducible experiments.

This module provides a centralized way to set random seeds across all
random number generators used in the platform (torch, numpy, scikit-learn, random).
"""

import os
import random
import time

import numpy as np
from typing import Optional

# Global variable to store the current seed
_GLOBAL_SEED: Optional[int] = None


def get_global_seed() -> Optional[int]:
    """Return the current global seed, or None when no seed has been set."""
    return _GLOBAL_SEED


def set_cublas_workspace_config() -> None:
    """Set CUBLAS workspace config to an allowed deterministic value.

    Must be set BEFORE importing torch. Valid values (per NVIDIA docs):
      - ":4096:8" (recommended)
      - ":16:8"   (minimal workspace)
    """
    # An explicit environment value belongs to the caller.
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id: int) -> None:
    """Use with torch.utils.data.DataLoader(worker_init_fn=seed_worker) to sync NumPy/random per-worker."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dataloader_generator(seed: Optional[int]):
    """Build the seeded generator passed to ``DataLoader(generator=...)``."""
    import torch

    if seed is None:
        seed = set_global_seed()

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def set_global_seed(seed: Optional[int] = None) -> int:
    """
    Set the global random seed for all random number generators.
    
    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch
    
    Args:
        seed: The seed value to use. If None, uses current timestamp.
    
    Returns:
        The seed value that was set.
    """
    global _GLOBAL_SEED

    if seed is None:
        seed = int(time.time() * 1000000) % (2**31)

    _GLOBAL_SEED = seed

    random.seed(seed)
    np.random.seed(seed)

    # Import torch lazily to avoid initializing CUDA before env is set elsewhere
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    return seed


def set_determinism() -> None:
    import torch

    # Deterministic kernels can significantly reduce throughput.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
        except Exception as error:
            print(f"torch.use_deterministic_algorithms is not available: {error}")
            print(f"torch version: {torch.__version__}")
            print("Make sure you are using the correct version of torch")
