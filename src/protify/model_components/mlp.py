"""The SwiGLU feed-forward block and classifier sizing, from the InterpNet model family.

The code is `interpnet.mlp`'s. Protify's probes import it under this path.
"""

from interpnet.mlp import SwiGLU, intermediate_correction_fn, swiglu_ln_ffn


__all__ = ["SwiGLU", "intermediate_correction_fn", "swiglu_ln_ffn"]
