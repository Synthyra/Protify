"""Reduce sparse autoencoder features to one vector per protein.

Kept apart from `esmc_sae`, which imports the vendored FastPLMs package at module level.
An exported probe repository ships the Protify runtime without that submodule and loads
its backbone from the Hub instead, so the pooling it has to reproduce exactly cannot live
behind a FastPLMs import. The arithmetic is `foundry.embedding`'s, which imports only torch,
and `export_packaged_model` ships that package beside the runtime. Training and an exported
probe therefore pool through the same code.

Shapes throughout: `b` sequences in a batch, `l` padded length, `n` valid tokens across the
whole batch in row-major order, `c` codebook features, `p` pooling operations.
"""

from __future__ import annotations

import torch

from foundry.embedding import POOLING_REDUCTIONS, pool_features, row_layout


# Reductions a sparse codebook tensor supports. Anything else would need the dense
# (b, l, c) activations, which is the tensor this pooling exists to avoid materializing.
SAE_POOLING_TYPES = POOLING_REDUCTIONS


def pool_sae_outputs(
    features: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling_types: tuple[str, ...],
) -> torch.Tensor:
    """Concatenate every requested reduction of one layer's codebook features, per sequence.

    `features` holds one row per valid token of the whole batch, in row-major order, which is
    how FastPLMs flattens them. Every attended token counts, CLS and EOS included, so a protein of
    l residues pools over l + 2 rows (docs/conventions/probes.md). Earlier Protify dropped the two
    special tokens, as the Biohub reference workflow does; the canonical convention does not.
    """
    # features: (n, c), sparse or dense, n = every attended token in the batch; attention_mask: (b, l + 2) padded
    token_mask = attention_mask.bool()  # (b, l + 2)
    layout = row_layout(token_mask, token_mask)  # every attended token kept
    return pool_features(features, layout, int(token_mask.shape[0]), pooling=pooling_types)  # (b, p * c), e.g. max over l + 2 rows


def validate_sae_pooling_types(pooling_types: tuple[str, ...], codebook_dim: int) -> None:
    """Reject a pooling name the sparse tensor cannot express."""
    unsupported = [name for name in pooling_types if name not in SAE_POOLING_TYPES]
    if unsupported:
        raise ValueError(
            f"SAE embeddings support pooling types {SAE_POOLING_TYPES}; received "
            f"{unsupported}. Other pooling would need the dense (b, l, c) activation "
            f"tensor, which is {codebook_dim} wide per residue."
        )
