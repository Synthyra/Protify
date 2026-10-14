"""The SwiGLU feed-forward block and classifier sizing, from `foundry.layers.feedforward`.

`intermediate_correction_fn` is foundry's `correction_fn_256`. Protify's probes import these
under this path.
"""

from foundry.layers.feedforward import SwiGLU, swiglu_ln_ffn
from foundry.layers.feedforward import correction_fn_256 as intermediate_correction_fn


__all__ = ["SwiGLU", "intermediate_correction_fn", "swiglu_ln_ffn"]
