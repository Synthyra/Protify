"""Protify's two baseline encodings as tap models whose only state is the token rows themselves.

`OneHot-Protein` reads each token as a one-hot row over 26 residue symbols, CLS and EOS (28 columns), so its pooled
mean is the token composition, length included through the CLS and EOS columns. `Random` reads each token as a
standard normal row of width 960 drawn from the sequence's SHA-256, so its pooled statistics carry the length and
nothing else. The rows come from the embedding engine behind Protify's control bases
(`base_models.control_bases`), so they are those bases' rows. As a module with an `embed` and an empty `transformer`,
a baseline goes through the hooks, stores and probes that ESMC goes through, with `embedding` as its one site. It has
no weights, so it takes no untrained-model control and cannot be randomly initialized.

Symbols: b sequences; l token columns (CLS, residues, EOS, padding); w the baseline's width.
"""

from __future__ import annotations

import torch

from torch import Tensor, nn

from ..base_models.control_bases import CONTROL_MODELS, ENGINE_KIND, control_width
from foundry.embedding import SyntheticControlView, sequence_inventory
from foundry.embedding.controls import TOKEN_STREAM
from foundry.embedding.readers import ReadBudget


BASELINE_SITES = ("embedding",)


def is_baseline(model_name: str) -> bool:
    return model_name in CONTROL_MODELS


class BaselineEmbedding(nn.Embedding):
    """The token rows of one baseline for a batch of token ids, generated from the sequences the ids spell."""

    def __init__(self, model_name: str, vocabulary: dict[str, int], special: set[int]) -> None:
        width = control_width(model_name)
        super().__init__(len(vocabulary), width, _weight=torch.zeros(len(vocabulary), width), _freeze=True)
        self.model_name = model_name
        self.letters = {index: token for token, index in vocabulary.items() if len(token) == 1}
        self.special = set(special)

    def reset_parameters(self) -> None:
        """Nothing to reset: the rows are generated, and the table is never read."""

    def sequences(self, input_ids: Tensor) -> list[str]:
        """The residues each row of token ids spells, with CLS, EOS and padding dropped."""
        # input_ids: (b, l)
        return ["".join(self.letters[index] for index in row if index not in self.special) for row in input_ids.tolist()]

    def forward(self, input_ids: Tensor) -> Tensor:
        # input_ids: (b, l) -> (b, l, w): the CLS row, the residue rows, the EOS row, then zeros at padding
        sequences = self.sequences(input_ids)
        sizes = [len(sequence) + 2 for sequence in sequences]
        # One read for the whole batch, which its caller has bounded in tokens already; the engine's default budget
        # (64 sequences, 32,768 rows) is smaller than a batch.
        budget = ReadBudget(max_rows=len(sequences), max_padded_residues=sum(sizes), max_bytes=sum(sizes) * self.embedding_dim * 4)
        view = SyntheticControlView(ENGINE_KIND[self.model_name], sequence_inventory(list(dict.fromkeys(sequences))), max_residues=max(sizes) - 2, budget=budget)
        rows = view.rows(TOKEN_STREAM, sequences)  # each (l_i + 2, w)
        out = torch.zeros(input_ids.shape + (self.embedding_dim,), dtype=torch.float32)  # (b, l, w)
        for index, block in enumerate(rows):
            out[index, : block.shape[0]] = block
        return out.to(input_ids.device)


class NoBlocks(nn.Module):
    """A transformer with no blocks, so the embedding is the whole model."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.norm = nn.Identity()


class BaselineModel(nn.Module):
    """A baseline encoding with ESMC's `embed` and `transformer` layout, so the tap hooks find its one state."""

    def __init__(self, model_name: str, tokenizer: object) -> None:
        super().__init__()
        if not is_baseline(model_name):
            raise ValueError(f"{model_name!r} is not a baseline; the baselines are {CONTROL_MODELS}.")
        special = {tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
        self.embed = BaselineEmbedding(model_name, tokenizer.tokenizer.get_vocab(), special)
        self.transformer = NoBlocks()

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, **kwargs: object) -> Tensor:
        # input_ids: (b, l) -> (b, l, w)
        return self.transformer.norm(self.embed(input_ids))
