"""The two control bases, generated where they are read and never stored.

A control carries no learned information, so there is nothing to compute once and keep. Its rows are
a deterministic function of the sequence, and the reader makes them at load. The arithmetic is the
embedding engine's (`foundry.embedding.controls.SyntheticControlView`), so a Protify read, a native
read and a model card all describe one control:

- `OneHot-Protein` reads each token as a one-hot row over the 26 residue symbols plus CLS and EOS,
  so its pooled mean is the token composition and its pooled variance is p(1 - p).
- `Random` reads each token as an independent N(0, 1) row of width 960, seeded by the SHA-256 of the
  sequence text, so equal sequences get equal rows in every split and every run.

Both follow the canonical row convention of docs/conventions/probes.md: a protein of l residues is
l + 2 token rows, CLS first and EOS last, and every pooled statistic covers all l + 2. A sequence over
`max_length - 2` residues is cut to `max_length - 2` first, as the tokenizer cuts it.

This module only names the controls Protify's registry knows, maps `max_length` (tokens) to the
engine's `max_residues`, and orders the pooled statistics as a branch spec asks.

Shapes: `b` sequences, `l` residues (so `l + 2` token rows), `w` columns of one statistic (`v` = 28
one-hot columns, `d` = 960 random-vector columns), `m` pooled statistics concatenated (mean, var).
"""

from __future__ import annotations

import numpy as np
import torch

from collections.abc import Sequence
from torch import Tensor

from foundry.embedding import ONE_HOT as ENGINE_ONE_HOT, RANDOM_VECTOR as ENGINE_RANDOM_VECTOR
from foundry.embedding import SyntheticControlView, sequence_inventory
from foundry.embedding.controls import CONTROL_SCHEMA, ONE_HOT_SYMBOLS, ONE_HOT_WIDTH, POOLED_STREAM, RANDOM_WIDTH, TOKEN_STREAM, control_width as engine_width


ONE_HOT = "OneHot-Protein"
RANDOM = "Random"
CONTROL_MODELS = (ONE_HOT, RANDOM)
# The engine's name for each Protify control.
ENGINE_KIND = {ONE_HOT: ENGINE_ONE_HOT, RANDOM: ENGINE_RANDOM_VECTOR}
# The statistics a control can pool, in the order the engine stores them: mean, then population variance.
CONTROL_POOLING = ("mean", "var")
# The shortest protein the convention can hold: CLS, one residue, EOS.
MINIMUM_ROWS = 3


def is_control_model(model_name: str) -> bool:
    """Whether a Protify model name is one of the two generated controls."""
    return model_name in CONTROL_MODELS


def control_width(model_name: str) -> int:
    """The columns `w` of one pooled statistic of a control."""
    if model_name not in ENGINE_KIND:
        raise ValueError(f"{model_name!r} is not a control base; the controls are {CONTROL_MODELS}.")
    return engine_width(ENGINE_KIND[model_name])  # w


def check_pooling(model_name: str, pooling_types: Sequence[str]) -> tuple[str, ...]:
    """The pooling types of a control read, refusing one the control cannot generate."""
    control_width(model_name)
    pooling = tuple(pooling_types)
    unsupported = [name for name in pooling if name not in CONTROL_POOLING]
    if not pooling or unsupported or len(set(pooling)) != len(pooling):
        raise ValueError(
            f"{model_name} generates the pooled statistics {CONTROL_POOLING}, each at most once; received {list(pooling_types)}."
        )
    return pooling


def control_view(model_name: str, sequences: Sequence[str], max_length: int) -> SyntheticControlView:
    """The engine's generator of this control's rows over `sequences`, cut at `max_length` tokens."""
    control_width(model_name)
    if max_length < MINIMUM_ROWS:
        raise ValueError(f"max_length {max_length} cannot hold one residue and its CLS and EOS tokens.")
    return SyntheticControlView(ENGINE_KIND[model_name], sequence_inventory(list(dict.fromkeys(sequences))), max_residues=max_length - 2)


def _ordered(pooled: Tensor, pooling: tuple[str, ...]) -> Tensor:
    """The engine's `(b, 2w)` mean-then-variance rows, cut to the statistics asked and in the order asked."""
    # pooled: (b, 2w)
    width = pooled.shape[1] // 2  # w
    halves = {"mean": pooled[:, :width], "var": pooled[:, width:]}  # each (b, w)
    return torch.cat([halves[name] for name in pooling], dim=1)  # (b, m * w)


def control_rows(model_name: str, sequences: Sequence[str], pooling_types: Sequence[str], max_length: int) -> Tensor:
    """The pooled control rows of b proteins, one row each, in the order given."""
    pooling = check_pooling(model_name, pooling_types)
    if not len(sequences):
        return torch.empty((0, len(pooling) * control_width(model_name)))  # (0, m * w)
    view = control_view(model_name, sequences, max_length)
    # The view builds every requested protein's token rows at once, so a block holds at most its read budget's rows.
    blocks = [view.pooled(list(sequences[start:start + view.budget.max_rows])) for start in range(0, len(sequences), view.budget.max_rows)]  # each (b_j, 2w)
    return _ordered(torch.cat(blocks), pooling)  # (b, m * w), float32


def control_pooled(model_name: str, sequence: str, pooling_types: Sequence[str], max_length: int) -> np.ndarray:
    """One protein's pooled control row."""
    return control_rows(model_name, [sequence], pooling_types, max_length)[0].numpy()  # (m * w,)


def control_matrix(model_name: str, sequence: str, max_length: int) -> Tensor:
    """One protein's per-token control rows: CLS, each kept residue, EOS."""
    return control_view(model_name, [sequence], max_length).rows(TOKEN_STREAM, [sequence])[0]  # (l + 2, w)


__all__ = [
    "CONTROL_MODELS", "CONTROL_POOLING", "CONTROL_SCHEMA", "ONE_HOT", "ONE_HOT_SYMBOLS", "ONE_HOT_WIDTH", "POOLED_STREAM", "RANDOM",
    "RANDOM_WIDTH", "TOKEN_STREAM", "check_pooling", "control_matrix", "control_pooled", "control_rows", "control_view", "control_width",
    "is_control_model",
]
