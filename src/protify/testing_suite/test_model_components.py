"""Tests for model_components: MLP, SwiGLU, RotaryEmbedding, MultiHeadAttention."""

import torch

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
    X = torch.randn(2, 4, 32)  # (2, 4, 32)
    output = swiglu(X)  # (2, 4, 16)
    assert output.shape == (2, 4, 16)


def test_swiglu_ln_ffn_output_shape() -> None:
    torch.manual_seed(0)
    ffn = swiglu_ln_ffn(hidden_size=16, expansion_ratio=8 / 3, dropout=0.0)
    X = torch.randn(2, 4, 16)  # (2, 4, 16)
    output = ffn(X)  # (2, 4, 16)
    assert output.shape == (2, 4, 16)


def test_rotate_half_preserves_shape() -> None:
    X = torch.randn(2, 4, 2, 8)  # (2, 4, 2, 8)
    output = rotate_half(X)  # (2, 4, 2, 8)
    assert output.shape == X.shape


def test_rotate_half_values() -> None:
    # For non-interleaved: cat(-x2, x1) where x1, x2 = chunk(2)
    X = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # (1, 4)
    output = rotate_half(X)  # (1, 4)
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])  # (1, 4)
    assert torch.allclose(output, expected)


def test_rotary_embedding_output_shapes() -> None:
    torch.manual_seed(0)
    # dim is half of head_dim (operates on pairs)
    rotary = RotaryEmbedding(dim=4)
    Q = torch.randn(2, 4, 2, 8)  # (2, 4, 2, 8)
    K = torch.randn(2, 4, 2, 8)  # (2, 4, 2, 8)
    Q_rotated, K_rotated = rotary(Q, K)  # each (2, 4, 2, 8)
    assert Q_rotated.shape == Q.shape
    assert K_rotated.shape == K.shape


def test_rotary_embedding_cache_grows() -> None:
    rotary = RotaryEmbedding(dim=4)
    Q_short = torch.randn(1, 2, 1, 8)  # (1, 2, 1, 8)
    K_short = torch.randn(1, 2, 1, 8)  # (1, 2, 1, 8)
    rotary(Q_short, K_short)  # returned tensors are each (1, 2, 1, 8)
    cached_len_1 = rotary._seq_len_cached
    Q_long = torch.randn(1, 8, 1, 8)  # (1, 8, 1, 8)
    K_long = torch.randn(1, 8, 1, 8)  # (1, 8, 1, 8)
    rotary(Q_long, K_long)  # returned tensors are each (1, 8, 1, 8)
    cached_len_2 = rotary._seq_len_cached
    assert cached_len_2 >= cached_len_1


def test_multihead_attention_output_shape() -> None:
    torch.manual_seed(0)
    mha = MultiHeadAttention(hidden_size=16, n_heads=2, attention_backend="sdpa")
    X = torch.randn(2, 4, 16)  # (2, 4, 16)
    output, _, _ = mha(X)  # (2, 4, 16)
    assert output.shape == (2, 4, 16)


def test_multihead_attention_with_2d_mask() -> None:
    torch.manual_seed(0)
    mha = MultiHeadAttention(hidden_size=16, n_heads=2, attention_backend="sdpa")
    X = torch.randn(2, 4, 16)  # (2, 4, 16)
    attention_mask_2d = torch.tensor(
        [[1, 1, 1, 0], [1, 1, 1, 1]],
        dtype=torch.bool,
    )  # (2, 4)
    attention_mask_4d = attention_mask_2d[:, None, None, :].expand(
        -1,
        -1,
        4,
        -1,
    )  # (2, 1, 4, 4)
    output, _, _ = mha(X, attention_mask_4d=attention_mask_4d)  # (2, 4, 16)
    assert output.shape == (2, 4, 16)
