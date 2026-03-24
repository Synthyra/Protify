"""Tests for model_components: MLP, SwiGLU, RotaryEmbedding, MultiHeadAttention."""

import torch
import pytest

try:
    from src.protify.model_components.mlp import intermediate_correction_fn, SwiGLU, swiglu_ln_ffn
    from src.protify.model_components.attention import RotaryEmbedding, rotate_half, MultiHeadAttention
except ImportError:
    try:
        from protify.model_components.mlp import intermediate_correction_fn, SwiGLU, swiglu_ln_ffn
        from protify.model_components.attention import RotaryEmbedding, rotate_half, MultiHeadAttention
    except ImportError:
        from ..model_components.mlp import intermediate_correction_fn, SwiGLU, swiglu_ln_ffn
        from ..model_components.attention import RotaryEmbedding, rotate_half, MultiHeadAttention


def test_intermediate_correction_fn_multiple_of_256() -> None:
    for hidden_size in [16, 128, 256, 768, 1024]:
        for ratio in [8 / 3, 4.0, 2.0]:
            result = intermediate_correction_fn(ratio, hidden_size)
            assert result % 256 == 0, f"Not multiple of 256: {result} for hidden={hidden_size}, ratio={ratio}"


def test_intermediate_correction_fn_known_value() -> None:
    # 8/3 * 768 = 2048, already multiple of 256
    assert intermediate_correction_fn(8 / 3, 768) == 2048


def test_intermediate_correction_fn_rounds_up() -> None:
    # 8/3 * 16 = 42.67, rounded up to nearest 256 = 256
    result = intermediate_correction_fn(8 / 3, 16)
    assert result == 256


def test_swiglu_output_shape() -> None:
    torch.manual_seed(0)
    swiglu = SwiGLU()
    x = torch.randn(2, 4, 32)
    out = swiglu(x)
    assert out.shape == (2, 4, 16)


def test_swiglu_ln_ffn_output_shape() -> None:
    torch.manual_seed(0)
    ffn = swiglu_ln_ffn(hidden_size=16, expansion_ratio=8 / 3, dropout=0.0)
    x = torch.randn(2, 4, 16)
    out = ffn(x)
    assert out.shape == (2, 4, 16)


def test_rotate_half_preserves_shape() -> None:
    x = torch.randn(2, 4, 2, 8)
    out = rotate_half(x)
    assert out.shape == x.shape


def test_rotate_half_values() -> None:
    # For non-interleaved: cat(-x2, x1) where x1, x2 = chunk(2)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = rotate_half(x)
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    assert torch.allclose(out, expected)


def test_rotary_embedding_output_shapes() -> None:
    torch.manual_seed(0)
    # dim is half of head_dim (operates on pairs)
    rotary = RotaryEmbedding(dim=4)
    # shape: (batch, seq_len, n_heads, head_dim)
    q = torch.randn(2, 4, 2, 8)
    k = torch.randn(2, 4, 2, 8)
    q_rot, k_rot = rotary(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rotary_embedding_cache_grows() -> None:
    rotary = RotaryEmbedding(dim=4)
    q_short = torch.randn(1, 2, 1, 8)
    k_short = torch.randn(1, 2, 1, 8)
    rotary(q_short, k_short)
    cached_len_1 = rotary._seq_len_cached
    q_long = torch.randn(1, 8, 1, 8)
    k_long = torch.randn(1, 8, 1, 8)
    rotary(q_long, k_long)
    cached_len_2 = rotary._seq_len_cached
    assert cached_len_2 >= cached_len_1


def test_multihead_attention_output_shape() -> None:
    torch.manual_seed(0)
    mha = MultiHeadAttention(hidden_size=16, n_heads=2, attention_backend="sdpa")
    x = torch.randn(2, 4, 16)
    out, _, _ = mha(x)
    assert out.shape == (2, 4, 16)


def test_multihead_attention_with_2d_mask() -> None:
    torch.manual_seed(0)
    mha = MultiHeadAttention(hidden_size=16, n_heads=2, attention_backend="sdpa")
    x = torch.randn(2, 4, 16)
    mask_2d = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    # Convert 2D mask to 4D for sdpa: (b, 1, 1, L)
    mask_4d = mask_2d[:, None, None, :].expand(-1, -1, 4, -1)
    out, _, _ = mha(x, attention_mask_4d=mask_4d)
    assert out.shape == (2, 4, 16)
