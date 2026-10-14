"""The annotation family: the Annotation Transformer (AT) over a protein's annotation terms, the CAMP annotation tower.

A row is one annotation entry: the set of term ids a protein carries under one annotation vocabulary. Its canonical text is
the distinct term ids in ascending order joined by commas, and its key is the SHA-256 of that text, so two spellings
of one set (any order, repeats) are one row. It is the `annotation_id` of the annotation vocabulary releases, which hash the
same text, so a row is found by the id a release already carries. The ids mean something only under the vocabulary release that numbered them,
so the release's repository, revision and content digest are the store's tokenizer pin: another vocabulary is another key.

AT has no special tokens: its vocabulary holds the terms, the pad id and a mask id, and Atlas feeds it the ids alone. A
row of l terms is l rows, so this family's descriptors are `feature_spec_v1` (rows are the units, nothing is added) and
a pooled stream is the mean and population variance over the l rows. An entry longer than `MAX_TERMS` is cut after its
`MAX_TERMS`-th smallest id. The terms are presented in ascending order, so a consumer that runs AT live must sort them too.

Symbols: b sequences of a batch; m padded token columns; d hidden width of the model; l terms of an entry after the cut.
"""

from __future__ import annotations

import numbers
import torch

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from torch import Tensor

from .encoder_rows import CanonicalInput, EncoderFamily, EncoderHooks, EncoderStream, Tokenized


ANNOTATION_NORMALIZATION = "annotation_terms_v1"
MAX_TERMS = 1024  # the context AT was configured for (`max_position_embeddings`)


def canonical_annotation(raw: object) -> CanonicalInput:
    """A raw entry (term ids in a list, an array, or one string of ids) as its sorted distinct ids joined by commas."""
    if isinstance(raw, str):
        pieces: Iterable[object] = raw.replace(",", " ").split()
    elif isinstance(raw, Iterable):
        pieces = raw
    else:
        raise ValueError(f"An annotation entry is term ids; received {type(raw).__name__}.")
    identifiers: set[int] = set()
    for piece in pieces:
        if isinstance(piece, bool) or not (isinstance(piece, (str, numbers.Integral))):
            raise ValueError(f"A term id is an integer; received {piece!r}.")
        identifiers.add(int(piece))
    if not identifiers:
        raise ValueError("An annotation entry needs at least one term.")
    text = ",".join(str(term) for term in sorted(identifiers))
    return CanonicalInput(text, len(identifiers), text)


class AnnotationHooks:
    """The ids and forward of an Annotation Transformer, as the executor calls them."""

    def __init__(self, model: Any, max_units: int = MAX_TERMS) -> None:
        config = model.config
        self.model = model
        self.max_units = max_units
        self.pad_id = int(config.pad_token_id)
        self.mask_id = int(config.mask_token_id)
        self.vocab_size = int(config.vocab_size)
        self.terms: range | None = None  # the ids that are terms, once a vocabulary release is bound

    def bind_vocabulary(self, terms: int) -> None:
        """Say that ids 0 to ``terms`` - 1 are the terms and every other id (pad, bos, eos, mask, unk) is not."""
        self.terms = range(terms)

    def is_term(self, identifier: int) -> bool:
        if self.terms is not None:
            return identifier in self.terms
        return 0 < identifier < self.vocab_size and identifier != self.mask_id  # an unpinned model: id 0 is its pad

    def tokenize(self, inputs: Sequence[CanonicalInput]) -> list[Tokenized | str]:
        """The entry's ascending term ids, cut to `MAX_TERMS`; an id outside the vocabulary, or the pad or mask id, has no row."""
        encoded: list[Tokenized | str] = []
        for item in inputs:
            ids = tuple(int(term) for term in item.payload.split(","))[: self.max_units]
            outside = [term for term in ids if not self.is_term(term)]
            if outside:
                encoded.append(
                    f"Entry {item.text[:60]!r} holds ids that are not terms of the vocabulary ({outside[:5]}): "
                    f"{'terms are 0 to ' + str(len(self.terms) - 1) if self.terms is not None else 'ids run 1 to ' + str(self.vocab_size - 1) + ' but the mask id'}"
                    f" (pad {self.pad_id}, mask {self.mask_id})."
                )
                continue
            encoded.append(Tokenized(ids, {}))
        return encoded

    def hidden(self, input_ids: Tensor, token_mask: Tensor) -> dict[str, Tensor]:
        """The model's final normalized hidden state of every term."""
        # input_ids: (b, m) ascending term ids then pad; token_mask: (b, m) true on every term
        output = self.model(input_ids=input_ids, attention_mask=token_mask.long())
        return {"final": output.last_hidden_state}  # (b, m, d)


def load_annotation_transformer(snapshot: Path, device: str) -> tuple[Any, EncoderHooks]:
    """Load a verified AT checkpoint in float32, as Atlas does, frozen and in eval mode."""
    from transformers import AutoModel

    from annotation_transformer import register_with_auto_classes

    register_with_auto_classes()
    model, info = AutoModel.from_pretrained(str(snapshot), dtype=torch.float32, local_files_only=True, output_loading_info=True)
    if info["missing_keys"] or info["mismatched_keys"]:
        raise ValueError(f"The checkpoint does not fill the model: missing {info['missing_keys']}, mismatched {info['mismatched_keys']}.")
    model = model.to(device).eval().requires_grad_(False)
    return model, AnnotationHooks(model)


def final_layer(model: Any) -> int:
    """The depth of the model: the index of its last hidden state."""
    return int(model.config.num_hidden_layers)


ANNOTATION_FAMILY = EncoderFamily(
    id="annotation",
    normalization_version=ANNOTATION_NORMALIZATION,
    unit="annotation terms",
    special_rows=0,
    max_units=MAX_TERMS,
    canonicalize=canonical_annotation,
    streams=(
        EncoderStream("final_hidden", "final", None, "bfloat16"),
        EncoderStream("final_mean_var", "final", ("mean", "var"), "float32"),
    ),
    tokenization={
        "input": "annotation_term_ids", "order": "ascending, each id once", "cut": f"first {MAX_TERMS} ids",
        "max_terms": MAX_TERMS, "special_tokens": "none; PAD is the only masked row",
        "valid_ids": "the vocabulary release's terms, ids 0 to terms - 1; pad, bos, eos, mask and unk have no row",
    },
    forward={
        "model": "annotation_transformer.AnnotationTransformer", "mode": "eval, float32 weights, no autocast",
        "attention_mask": "true on every term", "sources": {"final": "last_hidden_state, after the final layer norm"},
    },
    final_layer=final_layer,
    identity_modules=("foundry.embedding.annotation",),
    load=load_annotation_transformer,
    has_tokenizer=False,
    packages=(),
)


__all__ = [
    "ANNOTATION_FAMILY", "ANNOTATION_NORMALIZATION", "MAX_TERMS", "AnnotationHooks", "canonical_annotation",
    "load_annotation_transformer",
]
