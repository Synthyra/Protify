"""The ligand family: SELFIES-TED-LARGE, the ligand tower of Atlas-PLI, over canonical SMILES.

A row is one molecule. Its key is the SHA-256 of its canonical SMILES (`foundry.chemistry.canonicalize_smiles`, the one
string every model input is written as), so every spelling of a molecule is one row. The model reads that molecule as
SELFIES, in the spaced form the SELFIES-TED tokenizer reads (`foundry.chemistry.space_selfies`), cut after a whole symbol at
`MAX_SYMBOLS` (`foundry.chemistry.truncate_selfies`, what Atlas cuts with in training, `preembed.py`, evaluation and
serving), and the tokenizer adds CLS first and SEP last, so a molecule of l symbols is l + 2 rows, the convention of the
protein stores (row 0 CLS, rows 1..l symbols, row l + 1 SEP; padding is the only masked row).

What Atlas stores is not the encoder. Atlas calls `AutoModel`, which is `BartModel`, with `input_ids` and `attention_mask`
alone, and reads `last_hidden_state`; `BartModel` then builds `decoder_input_ids` by shifting the ids right, so that state
is the decoder's, teacher forced on the molecule it reads and cross-attending to the encoder. The store keeps both:
`decoder` is exactly Atlas's state, `encoder` is the encoder's `last_hidden_state`, the SELFIES-TED embedding the
checkpoint's authors describe and the SAE project reads. The forward runs once and gives both.

Symbols: b sequences of a batch; m padded token columns; d hidden width (1,024); l SELFIES symbols of a molecule after the cut.
"""

from __future__ import annotations

import torch

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from torch import Tensor

from .encoder_rows import CanonicalInput, EncoderFamily, EncoderHooks, EncoderStream, Tokenized


LIGAND_NORMALIZATION = "canonical_smiles_v1"
CONTEXT_ROWS = 1024  # BART's positions: the longest row count the encoder and the decoder read
MAX_SYMBOLS = CONTEXT_ROWS - 2  # SELFIES symbols kept, so that CLS and SEP fit the context
TOKENIZE_CHUNK = 4096  # inputs the tokenizer reads at once


def canonical_ligand(raw: object) -> CanonicalInput:
    """A raw SMILES as its canonical SMILES (the key text), its SELFIES symbol count, and its spaced SELFIES.

    Raises `foundry.chemistry.ChemistryError`, a ``ValueError``, for a string that is not a molecule or that SELFIES cannot
    write as itself; nothing is repaired or replaced.
    """
    from foundry.chemistry import canonicalize_smiles, selfies_symbols, smiles_to_selfies, space_selfies

    if not isinstance(raw, str):
        raise ValueError(f"A ligand is a SMILES string; received {type(raw).__name__}.")
    smiles = canonicalize_smiles(raw)
    selfies = space_selfies(smiles_to_selfies(smiles))
    return CanonicalInput(smiles, len(selfies_symbols(selfies)), selfies)


class SelfiesTedHooks:
    """The tokenizer and forward of SELFIES-TED-LARGE, a `BartModel`, as the executor calls them."""

    def __init__(self, model: Any, tokenizer: Any, max_units: int = MAX_SYMBOLS) -> None:
        config = model.config
        if (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id) != (
            config.bos_token_id, config.eos_token_id, config.pad_token_id,
        ):
            raise ValueError("The tokenizer's CLS, SEP and PAD ids differ from the model's BOS, EOS and PAD ids.")
        if config.encoder_layers != config.decoder_layers or len(tokenizer) > config.vocab_size:
            raise ValueError("The model is not the symmetric BART the ligand family reads.")
        self.model = model
        self.tokenizer = tokenizer
        self.max_units = max_units
        self.pad_id = int(config.pad_token_id)
        self.start_id = int(config.decoder_start_token_id)
        self.unknown_id = int(tokenizer.unk_token_id)

    def tokenize(self, inputs: Sequence[CanonicalInput]) -> list[Tokenized | str]:
        """CLS, the tokens of the SELFIES cut after a whole symbol, SEP; each symbol must be one token.

        A symbol the vocabulary lacks is one [UNK], which the facts count. A fragment separator attached to a symbol the
        vocabulary lacks (`[O-1].[Xx+1]`) is not split off by the tokenizer, so two symbols read as one [UNK]: the row would
        not be l + 2 rows for l symbols, and the reason says so.
        """
        from foundry.chemistry import truncate_selfies

        encoded: list[Tokenized | str] = []
        for start in range(0, len(inputs), TOKENIZE_CHUNK):
            chunk = inputs[start:start + TOKENIZE_CHUNK]
            cut = [truncate_selfies(item.payload, self.max_units) for item in chunk]
            for item, ids, text in zip(chunk, self.tokenizer(cut, add_special_tokens=True)["input_ids"], cut, strict=True):
                symbols = min(item.units, self.max_units)
                if len(ids) != symbols + 2 or ids[0] != self.tokenizer.cls_token_id or ids[-1] != self.tokenizer.sep_token_id:
                    encoded.append(f"{item.text} reads as {len(ids)} tokens, not CLS, its {symbols} symbols and SEP: {text[:80]}")
                    continue
                unknown = sum(token == self.unknown_id for token in ids)
                encoded.append(Tokenized(tuple(ids), {"selfies": item.payload, "unknown_tokens": unknown}))
        return encoded

    def hidden(self, input_ids: Tensor, token_mask: Tensor) -> dict[str, Tensor]:
        """The encoder's last hidden state and the decoder's, from one forward of `BartModel` with its own shifted decoder input."""
        from transformers.models.bart.modeling_bart import shift_tokens_right

        # input_ids: (b, m) CLS, symbols, SEP, then PAD; token_mask: (b, m) true on every row but the padding
        decoder_input_ids = shift_tokens_right(input_ids, self.pad_id, self.start_id)  # (b, m) what BartModel builds when given none
        output = self.model(input_ids=input_ids, attention_mask=token_mask.long(), decoder_input_ids=decoder_input_ids)
        return {"encoder": output.encoder_last_hidden_state, "decoder": output.last_hidden_state}  # each (b, m, d)


def load_selfies_ted(snapshot: Path, device: str) -> tuple[Any, EncoderHooks]:
    """Load the verified checkpoint in float32 as Atlas does, with the weights frozen and in eval mode.

    The checkpoint is a `BartForConditionalGeneration`; its language-model head is not loaded, and a missing weight of
    the model itself stops the load.
    """
    from transformers import AutoTokenizer, BartModel

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model, info = BartModel.from_pretrained(str(snapshot), dtype=torch.float32, local_files_only=True, output_loading_info=True)
    if info["missing_keys"] or info["mismatched_keys"]:
        raise ValueError(f"The checkpoint does not fill BartModel: missing {info['missing_keys']}, mismatched {info['mismatched_keys']}.")
    model = model.to(device).eval().requires_grad_(False)
    return model, SelfiesTedHooks(model, tokenizer)


def final_layer(model: Any) -> int:
    """The depth of both stacks of the model: the index of each one's last hidden state."""
    return int(model.config.encoder_layers)


LIGAND_FAMILY = EncoderFamily(
    id="ligand",
    normalization_version=LIGAND_NORMALIZATION,
    unit="SELFIES symbols",
    special_rows=2,
    max_units=MAX_SYMBOLS,
    canonicalize=canonical_ligand,
    streams=(
        EncoderStream("encoder_hidden", "encoder", None, "bfloat16"),
        EncoderStream("decoder_hidden", "decoder", None, "bfloat16"),
        EncoderStream("encoder_mean_var", "encoder", ("mean", "var"), "float32"),
        EncoderStream("decoder_mean_var", "decoder", ("mean", "var"), "float32"),
    ),
    tokenization={
        "input": "canonical_smiles", "canonicalizer": "foundry.chemistry.canonicalize_smiles",
        "to_selfies": "foundry.chemistry.smiles_to_selfies", "serialization": "foundry.chemistry.space_selfies",
        "cut": "foundry.chemistry.truncate_selfies, whole symbols", "max_symbols": MAX_SYMBOLS,
        "special_tokens": "CLS before the symbols, SEP after; PAD is the only masked row",
    },
    forward={
        "model": "transformers.BartModel", "mode": "eval, float32 weights, no autocast",
        "attention_mask": "true on every row but the padding",
        "decoder_input_ids": "input_ids shifted right with decoder_start_token_id first (the one BartModel builds)",
        "sources": {"encoder": "encoder_last_hidden_state", "decoder": "last_hidden_state, the state Atlas-PLI reads"},
    },
    final_layer=final_layer,
    identity_modules=(
        "foundry.embedding.ligand", "foundry.chemistry.smiles", "foundry.chemistry.conversion", "foundry.chemistry.selfies_text",
    ),
    load=load_selfies_ted,
    has_tokenizer=True,
    packages=("tokenizers", "rdkit", "selfies"),
)


__all__ = [
    "CONTEXT_ROWS", "LIGAND_FAMILY", "LIGAND_NORMALIZATION", "MAX_SYMBOLS", "SelfiesTedHooks", "canonical_ligand",
    "load_selfies_ted",
]
