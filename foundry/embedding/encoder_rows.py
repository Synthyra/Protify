"""Rows of a non-protein encoder: what an input becomes, how batches are cut, and how a batch is run and reduced.

This is the code whose bytes decide a stored value for the ligand and annotation bases, so a store's keys hash it: edit it
only when the rows are meant to change. Everything that only moves bytes (the contract, the writer, the job) is in
`encoders`, and changing that never changes a key.

A family turns a raw input into a `CanonicalInput` (the text that is hashed to the row's key, the count of units the
encoder reads, and the payload the tokenizer reads), a model and its hooks turn payloads into token ids and ids into hidden
states, and `EncoderExecutor` runs batches the way FastPLMs' `TokenTapExecutor` does for proteins: lengths come from the
host, so no device value is read while a batch is built or run, the outputs are copied to pinned buffers, and one device
flag checks finiteness after the copy lands. A per-token stream keeps every attended row, CLS and EOS included when the
family has them; a pooled stream is the mean and population variance over those same rows (`pool_token_rows`).

Symbols: b sequences of a batch; m padded token columns of a batch (the longest row count); n = sum of the rows of the
b sequences; d hidden width; p pooling reductions.
"""

from __future__ import annotations

import numpy as np
import torch

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from torch import Tensor


if TYPE_CHECKING:
    from fastplms.features.async_writer import PackedBatch


@dataclass(frozen=True, slots=True)
class CanonicalInput:
    """One input in its canonical form, before any tokenizer or model.

    ``text`` is what the row's key hashes: a canonical SMILES, or an annotation entry's sorted term ids joined by single
    spaces. ``units`` counts what the encoder reads from it before any crop (SELFIES symbols, annotation terms), and
    ``payload`` is what its tokenizer reads (the spaced SELFIES string, or the same term ids).
    """

    text: str
    units: int
    payload: str


@dataclass(frozen=True, slots=True)
class Tokenized:
    """One input's token ids after the crop, special tokens included, and the facts the inventory records about it."""

    ids: tuple[int, ...]
    facts: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PreparedRow:
    """A row ready to embed: its key text and SHA-256, units before the crop, the ids it stores rows for, and its facts."""

    text: str
    sha256: str
    units: int
    ids: tuple[int, ...]
    facts: Mapping[str, object]

    @property
    def rows(self) -> int:
        """Stored rows of every per-token stream: one per id, CLS and EOS included when the family has them."""
        return len(self.ids)


@dataclass(frozen=True, slots=True)
class EncoderStream:
    """One stored stream: the model output it reads, how it is reduced, and its stored dtype."""

    name: str  # the stream's name in the store
    source: str  # the key of `EncoderHooks.hidden` it reads
    pooling: tuple[str, ...] | None  # None: one row per token; else the reductions of one vector per sequence
    dtype: str  # the stored dtype


class EncoderHooks(Protocol):
    """A loaded model's tokenizer and forward, as the executor calls them."""

    pad_id: int

    def tokenize(self, inputs: Sequence[CanonicalInput]) -> list[Tokenized | str]:
        """Token ids (special tokens included, cropped) of each input, or the reason the model cannot read it."""
        ...

    def hidden(self, input_ids: Tensor, token_mask: Tensor) -> dict[str, Tensor]:
        """Each source's hidden states, ``(b, m, d)``, from ``input_ids`` ``(b, m)`` and ``token_mask`` ``(b, m)``, in eval mode."""
        ...


@dataclass(frozen=True, slots=True)
class EncoderFamily:
    """What a kind of input is: how it is identified, cut, tokenized and read, with the streams it stores."""

    id: str  # "ligand" or "annotation"
    normalization_version: str  # the SequenceKey's: which rule made the key text
    unit: str  # what `units` counts, for the card
    special_rows: int  # rows a sequence adds to its units: 2 (CLS, EOS) or 0
    max_units: int  # the crop in units, so that a row fits the model's context
    canonicalize: Callable[[object], CanonicalInput]  # pure and picklable: a worker process calls it
    streams: tuple[EncoderStream, ...]
    tokenization: Mapping[str, object]  # plain data saying how a payload becomes ids, which enters every key
    forward: Mapping[str, object]  # plain data saying what the model computes for each source
    final_layer: Callable[[Any], int]  # the model's last hidden-state index (its depth)
    identity_modules: tuple[str, ...]  # modules whose bytes decide the key text and the ids, hashed into each reducer
    load: Callable[[Path, str], tuple[Any, EncoderHooks]]  # a verified snapshot and a device to the model and its hooks
    has_tokenizer: bool  # whether the checkpoint's repository also holds the vocabulary
    packages: tuple[str, ...]  # libraries beyond torch and transformers whose versions a store records as evidence

    @property
    def keeps_special_tokens(self) -> bool:
        return self.special_rows > 0


def plan_row_batches(
    rows: Sequence[int], *, max_sequences: int, max_tokens: int, window: int,
) -> Iterator[tuple[int, ...]]:
    """Group sequences into batches of similar length under a padded-row budget; yields indices into ``rows``.

    ``rows`` holds the stored rows of each sequence. Sequences are sorted longest first inside each window of ``window``
    consecutive ones, so a batch pads to its first member: at most ``max_sequences`` sequences and
    ``b * rows[first] <= max_tokens`` padded rows. Longest first also makes an out-of-memory failure show on the first batch.
    """
    if min(max_sequences, max_tokens, window) < 1:
        raise ValueError("max_sequences, max_tokens and window must be positive.")
    for start in range(0, len(rows), window):
        order = sorted(range(start, min(start + window, len(rows))), key=lambda index: (-rows[index], index))
        batch: list[int] = []
        for index in order:
            if rows[index] > max_tokens:
                raise ValueError(f"A row of {rows[index]} token rows is more than max_tokens={max_tokens}.")
            if batch and (len(batch) + 1 > max_sequences or (len(batch) + 1) * rows[batch[0]] > max_tokens):
                yield tuple(batch)
                batch = []
            batch.append(index)
        if batch:
            yield tuple(batch)


class EncoderExecutor:
    """Run prepared rows through a model and hand each batch to the writer as pinned host tensors."""

    def __init__(
        self, hooks: EncoderHooks, prepared: Mapping[str, PreparedRow], streams: Sequence[EncoderStream], *, device: torch.device,
    ) -> None:
        if not streams:
            raise ValueError("An executor needs at least one stream.")
        self.hooks = hooks
        self.prepared = prepared
        self.streams = tuple(streams)
        self.device = device

    def rows_for(self, text: str) -> int:
        """Stored rows of one text in a per-token stream."""
        return self.prepared[text].rows

    def run_batch(self, texts: Sequence[str], digests: Sequence[str]) -> PackedBatch:
        """Embed one batch and start the copies of its outputs to the host; returns without waiting for them."""
        rows = [self.prepared[text] for text in texts]
        counts = np.fromiter((row.rows for row in rows), dtype=np.int64, count=len(rows))  # (b,) attended rows
        m = int(counts.max())
        ids = np.full((len(rows), m), self.hooks.pad_id, dtype=np.int64)  # (b, m)
        for index, row in enumerate(rows):
            ids[index, : row.rows] = row.ids
        owner = np.repeat(np.arange(len(rows), dtype=np.int64), counts)  # (n,) the sequence of each attended row
        starts = np.cumsum(counts) - counts  # (b,) first packed row of each sequence
        within = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(starts, counts)  # (n,) position in its sequence
        flat_index = torch.from_numpy(owner * m + within).to(self.device)  # (n,) into the row-major (b * m) grid
        input_ids = torch.from_numpy(ids).to(self.device)  # (b, m)
        lengths = torch.from_numpy(counts).to(self.device)  # (b,)
        token_mask = torch.arange(m, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)  # (b, m) padding is the only masked row
        with torch.inference_mode():
            hidden = self.hooks.hidden(input_ids, token_mask)  # source -> (b, m, d)
            outputs: dict[str, dict[str, Tensor]] = {}
            for stream in self.streams:
                states = hidden[stream.source]  # (b, m, d)
                dtype = getattr(torch, stream.dtype)
                if stream.pooling is None:
                    outputs[stream.name] = {"values": states.reshape(-1, states.shape[-1])[flat_index].to(dtype)}  # (n, d)
                else:
                    outputs[stream.name] = {"values": pooled_rows(states, token_mask, stream.pooling).to(dtype)}  # (b, p * d)
        return deliver(outputs, texts, digests, tuple(int(count) for count in counts), self.device)


def pooled_rows(states: Tensor, token_mask: Tensor, pooling: Sequence[str]) -> Tensor:
    """The reductions of every attended row of each sequence, in float32, from the one pooling FastPLMs' token runs use."""
    from fastplms.embeddings.pooling import pool_token_rows

    # states: (b, m, d) any floating dtype; token_mask: (b, m) true on every attended row; the result is (b, p * d) float32.
    return pool_token_rows(states.float(), token_mask, tuple(pooling))  # (b, p * d)


def deliver(
    outputs: Mapping[str, Mapping[str, Tensor]], sequences: Sequence[str], digests: Sequence[str], rows: tuple[int, ...],
    device: torch.device,
) -> PackedBatch:
    """Start the device-to-host copies, record one event after them, and check finiteness on the device.

    FastPLMs' `TokenTapExecutor._deliver` does the same for proteins, and cannot be called from here: the executor is bound
    to its proteins and its file is in the engine digest of the published stores, so changing it would re-key them.
    """
    from fastplms.features.async_writer import PackedBatch

    # outputs: (b, w) pooled, (n, w) per token row, per stream; n = sum of the rows of the b sequences.
    finite = [torch.isfinite(tensor).all() for group in outputs.values() for tensor in group.values() if tensor.is_floating_point()]
    everything_finite = torch.stack(finite).all() if finite else torch.ones((), dtype=torch.bool, device=device)
    if device.type != "cuda":
        hosts = {name: dict(group) for name, group in outputs.items()}

        def wait() -> None:
            if not bool(everything_finite):
                raise ValueError("A stream produced a non-finite value.")
    else:
        hosts = {name: {key: pinned_copy(tensor) for key, tensor in group.items()} for name, group in outputs.items()}
        flag = pinned_copy(everything_finite)
        event = torch.cuda.Event()
        event.record()

        def wait() -> None:
            event.synchronize()
            if not bool(flag):
                raise ValueError("A stream produced a non-finite value.")

    return PackedBatch(tuple(sequences), tuple(digests), rows, hosts, wait)


def pinned_copy(tensor: Tensor) -> Tensor:
    """A pinned host tensor the copy of ``tensor`` is queued into on the current stream, without waiting."""
    # tensor: (...) any shape; the host copy has the same shape and dtype.
    host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
    host.copy_(tensor.detach(), non_blocking=True)
    return host  # (...) the shape of tensor


__all__ = [
    "CanonicalInput", "EncoderExecutor", "EncoderFamily", "EncoderHooks", "EncoderStream", "PreparedRow", "Tokenized",
    "deliver", "plan_row_batches", "pooled_rows", "pinned_copy",
]
