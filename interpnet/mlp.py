"""The SwiGLU feed-forward block of InterpNet's transformer layers, from `foundry.layers.feedforward`.

`SwiGLU` is foundry's, `intermediate_correction_fn` is foundry's `correction_fn_256`, and
`swiglu_ln_ffn` builds foundry's block with a biased layer normalization and `nn.Linear`'s default
initialization, as InterpNet's checkpoints were trained. This module keeps their names.
"""

from torch import nn

from foundry.layers.feedforward import SwiGLU, swiglu_ffn
from foundry.layers.feedforward import correction_fn_256 as intermediate_correction_fn


__all__ = ["SwiGLU", "intermediate_correction_fn", "swiglu_ln_ffn"]


def swiglu_ln_ffn(
    hidden_size: int,
    expansion_ratio: float,
    dropout: float = 0.1,
    use_bias: bool = False,
) -> nn.Sequential:
    """Build a normalized SwiGLU feed-forward block that maps (..., d) to (..., d).

    `use_bias` gives both projections a bias.
    """
    return swiglu_ffn(hidden_size, expansion_ratio, dropout, bias=use_bias, norm_bias=True, xavier_init=False)
