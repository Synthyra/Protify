import torch
import torch.nn as nn
import torch.nn.functional as F


def intermediate_correction_fn(expansion_ratio: float, hidden_size: int) -> int:
    return int(((expansion_ratio * hidden_size) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 2 * d_ff)
        x1, x2 = x.chunk(2, dim=-1)  # (..., d_ff), (..., d_ff)
        return F.silu(x1) * x2  # (..., d_ff)


def swiglu_ln_ffn(
    hidden_size: int,
    expansion_ratio: float,
    dropout: float = 0.1,
    use_bias: bool = False,
) -> nn.Sequential:
    """Build a feed-forward network mapping ``(..., d)`` to ``(..., d)``."""
    intermediate_size = intermediate_correction_fn(expansion_ratio, hidden_size)  # d_ff
    return nn.Sequential(
        nn.LayerNorm(hidden_size),  # (..., d)
        nn.Linear(hidden_size, intermediate_size * 2, bias=use_bias),  # (..., 2 * d_ff)
        SwiGLU(),  # (..., d_ff)
        nn.Dropout(dropout),  # (..., d_ff)
        nn.Linear(intermediate_size, hidden_size, bias=use_bias),  # (..., d)
    )
