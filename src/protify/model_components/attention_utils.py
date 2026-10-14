"""Attention backend selection, masking, and unpadding, from the InterpNet model family.

The code is `interpnet.attention_utils`'s. The kernels backend keeps the FlashAttention kernel it
loads in that module's `FLASH_KERNEL`, `FLASH_KERNEL_VARIANT`, and `FLASH_KERNEL_DTYPES`. Read or
reset them there, because a name imported here would keep its value from import time.
"""

from interpnet.attention_utils import (
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
