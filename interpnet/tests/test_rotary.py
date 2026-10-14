"""Interleaved rotary embedding is the half-split rotation with its channels reordered.

Each frequency turns one channel pair: the matching halves by default, or two neighboring channels
with `interleaved`. Reordering the channels between the two layouts must commute with the rotation.
"""

import torch

from interpnet.attention import apply_rotary_emb_torch


def test_interleaved_matches_half_split_under_channel_reorder() -> None:
    batch_size, seq_len, num_heads, head_dim, rotary_half = 2, 7, 3, 10, 4
    x = torch.randn(batch_size, seq_len, num_heads, head_dim)  # (b, l, h, d_h)
    angles = torch.randn(seq_len, rotary_half)  # (l, d_r / 2)
    cos, sin = angles.cos(), angles.sin()  # each: (l, d_r / 2)

    # Channel i of the first half pairs with channel i of the second; interleaved, 2i with 2i + 1.
    to_interleaved = torch.stack(
        (torch.arange(rotary_half), torch.arange(rotary_half) + rotary_half), dim=-1
    ).flatten()  # (d_r,)
    order = torch.cat((to_interleaved, torch.arange(2 * rotary_half, head_dim)))  # (d_h,)

    half_split = apply_rotary_emb_torch(x, cos, sin)  # (b, l, h, d_h)
    interleaved = apply_rotary_emb_torch(x[..., order], cos, sin, interleaved=True)  # (b, l, h, d_h)

    torch.testing.assert_close(interleaved, half_split[..., order])


def test_interleaved_rotation_keeps_each_pair_norm() -> None:
    x = torch.randn(1, 5, 2, 8)  # (b, l, h, d_h)
    angles = torch.randn(5, 4)  # (l, d_r / 2)
    rotated = apply_rotary_emb_torch(x, angles.cos(), angles.sin(), interleaved=True)  # (b, l, h, d_h)

    rotated_norm = rotated.unflatten(-1, (4, 2)).norm(dim=-1)  # (b, l, h, d_r / 2)
    original_norm = x.unflatten(-1, (4, 2)).norm(dim=-1)  # (b, l, h, d_r / 2)
    torch.testing.assert_close(rotated_norm, original_norm)
