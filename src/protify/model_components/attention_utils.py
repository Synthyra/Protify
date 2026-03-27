from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
except ImportError:
    create_block_mask = None
    flex_attention = None

    if TYPE_CHECKING:
        from torch.nn.attention.flex_attention import BlockMask
    else:
        BlockMask = None

try:
    from kernels import get_kernel
except ImportError:
    get_kernel = None


class AttentionBackend(Enum):
    KERNELS = "kernels"
    FLEX = "flex"
    SDPA = "sdpa"


VALID_ATTENTION_BACKENDS = tuple(backend.value for backend in AttentionBackend)


def _infer_kernels_flash_variant(kernel: Any) -> Optional[str]:
    if hasattr(kernel, "fwd") and hasattr(kernel, "varlen_fwd"):
        return "flash_attn2"
    if hasattr(kernel, "flash_attn_func") and hasattr(kernel, "flash_attn_varlen_func"):
        return "flash_attn3"
    return None


def _load_kernels_flash() -> Tuple[Optional[Any], Optional[str]]:
    if get_kernel is None:
        return None, None

    for kernel_name in ("kernels-community/flash-attn3", "kernels-community/flash-attn2"):
        try:
            flash_kernel = get_kernel(kernel_name)
            flash_variant = _infer_kernels_flash_variant(flash_kernel)
            assert flash_variant is not None, f"Loaded kernel {kernel_name} does not expose a supported API."
            return flash_kernel, flash_variant
        except Exception:
            continue

    return None, None


_FLASH_KERNELS_LOADED = False
FLASH_KERNEL = None
FLASH_KERNEL_VARIANT = None


def _ensure_flash_kernels_loaded():
    global _FLASH_KERNELS_LOADED, FLASH_KERNEL, FLASH_KERNEL_VARIANT
    if _FLASH_KERNELS_LOADED:
        return
    _FLASH_KERNELS_LOADED = True
    FLASH_KERNEL, FLASH_KERNEL_VARIANT = _load_kernels_flash()


def resolve_attention_backend(requested_backend: str) -> AttentionBackend:
    normalized_backend = requested_backend.lower()
    if normalized_backend == "kernels_flash":
        normalized_backend = AttentionBackend.KERNELS.value

    assert normalized_backend in VALID_ATTENTION_BACKENDS, (
        f"Unsupported attention backend: {requested_backend}. "
        f"Expected one of {VALID_ATTENTION_BACKENDS}."
    )

    if normalized_backend == AttentionBackend.KERNELS.value:
        _ensure_flash_kernels_loaded()
        assert FLASH_KERNEL is not None, "The kernels attention backend is unavailable in this environment."
        return AttentionBackend.KERNELS

    if normalized_backend == AttentionBackend.FLEX.value:
        if flex_attention is None or create_block_mask is None:
            warnings.warn(
                "Flex attention is unavailable in this environment; falling back to sdpa.",
                stacklevel=2,
            )
            return AttentionBackend.SDPA
        return AttentionBackend.FLEX

    return AttentionBackend.SDPA


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size,
        num_key_value_heads,
        n_rep,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch_size, num_key_value_heads * n_rep, seq_len, head_dim)


def sdpa_attention_func(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask_4d: Optional[torch.Tensor],
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask_4d,
        is_causal=False,
    )


def flex_attention_func(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    flex_block_mask: Optional[BlockMask],
) -> torch.Tensor:
    assert flex_attention is not None, "Flex attention is unavailable in this environment."
    return flex_attention(
        query_states,
        key_states,
        value_states,
        block_mask=flex_block_mask,
    )


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, indices) -> torch.Tensor:  # type: ignore[no-untyped-def]
        ctx.save_for_backward(indices)
        assert input_tensor.ndim >= 2
        ctx.first_axis_dim = input_tensor.shape[0]
        other_shape = input_tensor.shape[1:]
        flattened_dim = other_shape.numel()
        return torch.gather(
            rearrange(input_tensor, "b ... -> b (...)"),
            0,
            repeat(indices, "z -> z d", d=flattened_dim),
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[torch.Tensor, None]:  # type: ignore[no-untyped-def]
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            (ctx.first_axis_dim, grad_output.shape[1]),
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim) -> torch.Tensor:  # type: ignore[no-untyped-def]
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[torch.Tensor, None, None]:  # type: ignore[no-untyped-def]
        (indices,) = ctx.saved_tensors
        return grad_output[indices], None, None


index_first_axis = IndexFirstAxis.apply
index_put_first_axis = IndexPutFirstAxis.apply


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    output = index_put_first_axis(hidden_states, indices, batch_size * seq_len)
    return rearrange(output, "(b s) ... -> b s ...", b=batch_size)


def _kernels_flash_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    assert FLASH_KERNEL is not None, "The kernels attention backend is unavailable in this environment."

    if FLASH_KERNEL_VARIANT == "flash_attn2":
        return FLASH_KERNEL.fwd(q=query_states, k=key_states, v=value_states, is_causal=False)[0]

    if FLASH_KERNEL_VARIANT == "flash_attn3":
        output = FLASH_KERNEL.flash_attn_func(
            q=query_states,
            k=key_states,
            v=value_states,
            causal=False,
        )
        if isinstance(output, tuple):
            return output[0]
        return output

    raise AssertionError(f"Unsupported kernels flash attention variant: {FLASH_KERNEL_VARIANT}")


def _kernels_flash_varlen_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> torch.Tensor:
    assert FLASH_KERNEL is not None, "The kernels attention backend is unavailable in this environment."

    if FLASH_KERNEL_VARIANT == "flash_attn2":
        return FLASH_KERNEL.varlen_fwd(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            is_causal=False,
        )[0]

    if FLASH_KERNEL_VARIANT == "flash_attn3":
        output = FLASH_KERNEL.flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=False,
        )
        if isinstance(output, tuple):
            return output[0]
        return output

    raise AssertionError(f"Unsupported kernels flash attention variant: {FLASH_KERNEL_VARIANT}")


def _unpad_input(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask_2d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[int, int]]:
    batch_size, seq_len, num_heads, head_dim = query_states.shape
    _, _, num_key_value_heads, _ = key_states.shape
    assert key_states.shape[1] == seq_len, "Query and key sequence lengths must match for kernels attention."
    assert value_states.shape[1] == seq_len, "Query and value sequence lengths must match for kernels attention."

    valid_lengths = attention_mask_2d.sum(dim=1).int()
    cu_seqlens = F.pad(valid_lengths.cumsum(0, dtype=torch.int32), (1, 0))
    max_seq_len = int(valid_lengths.max().item())
    indices = attention_mask_2d.flatten().nonzero(as_tuple=False).flatten()

    query_states = index_first_axis(query_states.reshape(batch_size * seq_len, num_heads, head_dim), indices)
    key_states = index_first_axis(key_states.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices)
    value_states = index_first_axis(value_states.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices)
    return query_states, key_states, value_states, indices, (cu_seqlens, cu_seqlens), (max_seq_len, max_seq_len)


def kernels_attention_func(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask_2d: Optional[torch.Tensor],
) -> torch.Tensor:
    if attention_mask_2d is None:
        return _kernels_flash_forward(query_states, key_states, value_states)

    batch_size, seq_len = query_states.shape[:2]
    (
        query_states,
        key_states,
        value_states,
        indices,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_q, max_seqlen_k),
    ) = _unpad_input(query_states, key_states, value_states, attention_mask_2d)

    attn_output = _kernels_flash_varlen_forward(
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    return pad_input(attn_output, indices, batch_size, seq_len)


@torch.compiler.disable
def build_attention_masks(
    attention_backend: AttentionBackend,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None,
    attention_mask_2d: Optional[torch.Tensor] = None,
    attention_mask_4d: Optional[torch.Tensor] = None,
    flex_block_mask: Optional[BlockMask] = None,
    output_attentions: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[BlockMask]]:
    if attention_mask is not None:
        if attention_mask.ndim == 2:
            attention_mask_2d = attention_mask.bool()
        elif attention_mask.ndim == 4:
            attention_mask_4d = attention_mask.bool()
        else:
            raise AssertionError(
                f"attention_mask must be 2D or 4D, but received shape {tuple(attention_mask.shape)}."
            )

    if attention_mask_2d is None and attention_mask_4d is None and flex_block_mask is None:
        return None, None, None

    if attention_mask_2d is None and attention_mask_4d is not None:
        attention_mask_2d = attention_mask_4d[:, 0, 0, :]

    if attention_backend == AttentionBackend.KERNELS:
        return attention_mask_2d, None, None

    if attention_backend == AttentionBackend.FLEX and flex_block_mask is None and not output_attentions:
        assert attention_mask_2d is not None, (
            "Flex attention requires a 2D padding mask or an explicit block mask. "
            "Arbitrary 4D masks should be passed as flex_block_mask."
        )
        assert create_block_mask is not None, "Flex attention is unavailable in this environment."
        valid_lengths = attention_mask_2d.sum(dim=-1)

        def mask_mod(batch_idx, head_idx, query_idx, key_idx):
            return (query_idx < valid_lengths[batch_idx]) & (key_idx < valid_lengths[batch_idx])

        flex_block_mask = create_block_mask(mask_mod, batch_size, 1, seq_len, seq_len, device=device)
        return attention_mask_2d, None, flex_block_mask

    # SDPA / manual: only mask key dimension so padding query positions attend to
    # real keys and produce valid outputs instead of NaN from softmax(-inf,...,-inf).
    if attention_mask_4d is None and attention_mask_2d is not None:
        attention_mask_4d = attention_mask_2d[:, None, None, :]

    return attention_mask_2d, attention_mask_4d, flex_block_mask
