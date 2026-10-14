"""Rotary embeddings, multi-head attention, and token-parameter readouts, from the InterpNet family.

The code is `interpnet.attention`'s and `interpnet.attention_logits`'s. Protify's probes import it
under this path.
"""

from interpnet.attention import (
    LayerNorm,
    Linear,
    MultiHeadAttention,
    RotaryEmbedding,
    apply_rotary_emb_torch,
    rotate_half,
)
from interpnet.attention_logits import AttentionLogitsSequence, AttentionLogitsToken


__all__ = [
    "AttentionLogitsSequence",
    "AttentionLogitsToken",
    "LayerNorm",
    "Linear",
    "MultiHeadAttention",
    "RotaryEmbedding",
    "apply_rotary_emb_torch",
    "rotate_half",
]
