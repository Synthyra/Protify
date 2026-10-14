"""Attention backend selection, masking, and unpadding, from `foundry.layers.attention_backends`.

The kernels backend keeps the FlashAttention kernel it loads in that module's `FLASH_KERNEL`,
`FLASH_KERNEL_VARIANT`, and `FLASH_KERNEL_DTYPES`. Read or reset them there, because a name
imported here would keep its value from import time.
"""

from foundry.layers.attention_backends import (
    FLASH_KERNEL_PREFERENCE,
    VALID_ATTENTION_BACKENDS,
    AttentionBackend,
    BlockMask,
    IndexFirstAxis,
    IndexPutFirstAxis,
    build_attention_masks,
    flex_attention_func,
    index_first_axis,
    index_put_first_axis,
    kernels_attention_func,
    pad_input,
    resolve_attention_backend,
    sdpa_attention_func,
)


__all__ = [
    "FLASH_KERNEL_PREFERENCE",
    "VALID_ATTENTION_BACKENDS",
    "AttentionBackend",
    "BlockMask",
    "IndexFirstAxis",
    "IndexPutFirstAxis",
    "build_attention_masks",
    "flex_attention_func",
    "index_first_axis",
    "index_put_first_axis",
    "kernels_attention_func",
    "pad_input",
    "resolve_attention_backend",
    "sdpa_attention_func",
]
