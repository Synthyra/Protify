"""Checks that loading a checkpoint fills each buffer the checkpoint does not hold.

Transformers 5 builds a model on the meta device, copies in the tensors the checkpoint holds, and
allocates every other parameter and buffer as uninitialized memory for the model's `_init_weights`
to fill. A buffer registered with `persistent=False`, such as a rotary frequency table, is in no
checkpoint, so it keeps whatever that memory held unless `_init_weights` rebuilds it. Unfilled
rotary tables of this kind turned every Atlas protein feature into NaN.
"""

import torch

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager


@contextmanager
def uninitialized_memory_as_nan() -> Iterator[None]:
    """Fill every allocation made inside the block with NaN, or with the maximum for integers.

    A tensor that a load never fills keeps whatever its allocation held. When the allocator
    recycles a block just freed by an identical tensor, that can be the right value by chance, and
    a comparison would pass without the load having filled it. With deterministic algorithms on,
    PyTorch fills uninitialized memory with a known value instead, so an unfilled tensor fails
    every comparison. Operations with no deterministic implementation warn rather than raise, and
    both settings are restored on exit.
    """
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = fill


def assert_unsaved_buffers_match(
    model: torch.nn.Module,
    checkpoint_keys: Iterable[str],
    build_fresh: Callable[[], torch.nn.Module],
) -> int:
    """Assert every buffer of `model` that the checkpoint lacks equals a freshly built model's.

    Each buffer is compared exactly, at the loaded buffer's device and dtype, because a load that
    fills it correctly runs the same arithmetic the constructor does.

    Args:
        model: The loaded model.
        checkpoint_keys: The tensor names the checkpoint holds, such as the keys of a safetensors
            file or of the saved model's `state_dict()`. A key may carry the model's
            `base_model_prefix`, as a checkpoint saved from a task head does.
        build_fresh: Builds the same architecture directly, usually `lambda: type(model)(config)`.
            It is called only when there is a buffer to compare.

    Returns:
        How many buffers the checkpoint lacks, so a caller can assert the ones under test exist.
    """
    keys = set(checkpoint_keys)
    prefix = getattr(model, "base_model_prefix", "")
    unsaved = {
        name: buffer
        for name, buffer in model.named_buffers()
        if name not in keys and (not prefix or f"{prefix}.{name}" not in keys)
    }
    if not unsaved:
        return 0

    fresh = dict(build_fresh().named_buffers())
    for name, buffer in unsaved.items():  # buffer: (...), whatever shape the model registered
        assert name in fresh, f"{name} is not a buffer of a freshly built model"
        expected = fresh[name].to(device=buffer.device, dtype=buffer.dtype)  # (...)
        torch.testing.assert_close(buffer, expected, rtol=0, atol=0, msg=name)
    return len(unsaved)
