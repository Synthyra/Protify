"""Random-vector and one-hot control bases, generated where they are read and never stored.

A control carries no learned information, so its rows are a deterministic function of the
sequence and the reader makes them at load. Both follow the canonical row convention: a protein
of l residues is l + 2 token rows, row 0 CLS, rows 1..l residues, row l + 1 EOS, and every pooled
statistic covers all l + 2 rows. A sequence over `max_residues` is cut from the N-terminus first.

Definitions match Protify's canonical controls (`protify.base_models.control_bases`):

- one-hot reads each token as one row over the 26 residue symbols plus CLS and EOS (28 columns);
- random-vector reads each token as an independent N(0, 1) row of width 960.

One difference from Protify's random pooling: Protify draws pooled statistics from their exact
distribution and never builds the token rows, so its rows and pooled values cannot agree. Here the
token rows are drawn from the row's SHA-256 and pooled afterwards, so `rows` and `pooled` always do.
The stream is numpy's raw PCG64 output through Box-Muller in float64, because numpy promises the
bit-generator stream across versions and does not promise `Generator.standard_normal`.

Symbols: `b` sequences, `l` residues after the crop, `v` one-hot columns, `d` random-vector columns,
`w` columns of one stream (`v` or `d`), `n` = l + 2 token rows of one sequence.
"""

from __future__ import annotations

import hashlib
import numpy as np
import torch

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from torch import Tensor

from .contracts import CANONICAL_MAX_RESIDUES, SPECIAL_TOKEN_ROWS, SequenceKey, normalize_sequence
from .readers import ReadBudget, ResidueBatch, _crop_ranges, _place_rows


RANDOM_VECTOR = "random_vector"
ONE_HOT = "one_hot"
CONTROL_KINDS = (RANDOM_VECTOR, ONE_HOT)
CONTROL_SCHEMA = "control_v1"
# Stream names a control answers to, matching the real bases' `final_hidden` and `final_mean_var`.
TOKEN_STREAM = "final_hidden"
POOLED_STREAM = "final_mean_var"
RANDOM_WIDTH = 960  # d, ESMC-300's hidden width, as Protify's random control
# Protify's AA_SET order, then CLS and EOS, so a one-hot column means the same in both packages.
RESIDUE_SYMBOLS = "LAGVSERTIPDKQNFYMHWCXBUOZ*"
ONE_HOT_SYMBOLS = (*RESIDUE_SYMBOLS, "<cls>", "<eos>")
ONE_HOT_WIDTH = len(ONE_HOT_SYMBOLS)  # v
_SYMBOL_COLUMN = {symbol: column for column, symbol in enumerate(ONE_HOT_SYMBOLS)}
_CLS_COLUMN = _SYMBOL_COLUMN["<cls>"]
_EOS_COLUMN = _SYMBOL_COLUMN["<eos>"]
# Separates the random stream from any other use of a sequence's SHA-256 as entropy.
_RANDOM_SPAWN_KEY = (1,)
_FLOAT53_UNIT = 2.0**-53
_BOX_MULLER_BLOCK = 1 << 14  # pairs per block of `random_rows`: 128 KB per float64 buffer, inside a core's L2 cache


def control_width(kind: str) -> int:
    """Columns `w` of one stream of a control."""
    if kind == RANDOM_VECTOR:
        return RANDOM_WIDTH  # d
    if kind == ONE_HOT:
        return ONE_HOT_WIDTH  # v
    raise ValueError(f"{kind!r} is not a control; the controls are {CONTROL_KINDS}.")


def one_hot_rows(residues: str) -> Tensor:
    """One-hot token rows of a cropped sequence: CLS, then each residue, then EOS."""
    try:
        columns = [_CLS_COLUMN, *(_SYMBOL_COLUMN[symbol] for symbol in residues), _EOS_COLUMN]  # n columns
    except KeyError as error:
        raise ValueError(f"{error} is not one of the residue symbols {RESIDUE_SYMBOLS}.") from None
    rows = torch.zeros((len(columns), ONE_HOT_WIDTH), dtype=torch.float32)  # (l+2, v)
    rows[torch.arange(len(columns)), torch.tensor(columns)] = 1.0  # one 1 per row; row 0 CLS, row l+1 EOS
    return rows  # (l+2, v)


def random_rows(sha256: str, n_rows: int, width: int = RANDOM_WIDTH) -> Tensor:
    """`n_rows` independent N(0, 1) rows, seeded by the sequence's SHA-256 and stable under a longer draw.

    Draws run row-major, so the first rows of a longer draw equal a shorter one when `width` is even. Uniform pair
    (u_1, u_2) gives sqrt(-2 log u_1) cos(2 pi u_2), then the same radius times sin(2 pi u_2).

    The work runs in blocks of _BOX_MULLER_BLOCK pairs through reused buffers, element for element the arithmetic of a
    single pass, so the values do not depend on the block size. A protein's draw was otherwise about ten fresh arrays
    of several MB, and threads making controls in parallel then waited on the allocator: blocks made eight threads
    about three times faster.
    """
    count = n_rows * width
    pairs = (count + 1) // 2
    seed = np.random.SeedSequence(int(sha256, 16), spawn_key=_RANDOM_SPAWN_KEY)
    generator = np.random.PCG64(seed)
    normals = np.empty(2 * pairs, dtype=np.float32)  # (2 * pairs,)
    uniform = np.empty(2 * _BOX_MULLER_BLOCK, dtype=np.float64)  # (2 * k,), k pairs per block
    radius = np.empty(_BOX_MULLER_BLOCK, dtype=np.float64)  # (k,)
    angle = np.empty(_BOX_MULLER_BLOCK, dtype=np.float64)  # (k,)
    product = np.empty(_BOX_MULLER_BLOCK, dtype=np.float64)  # (k,)
    for first in range(0, pairs, _BOX_MULLER_BLOCK):
        size = min(_BOX_MULLER_BLOCK, pairs - first)  # k' pairs in this block
        bits = generator.random_raw(2 * size)  # (2 * k',) uint64, continuing the stream
        np.right_shift(bits, np.uint64(11), out=bits)
        block = uniform[: 2 * size]  # (2 * k',)
        block[:] = bits
        block += 0.5
        block *= _FLOAT53_UNIT  # (2 * k',) in (0, 1)
        r, theta, value = radius[:size], angle[:size], product[:size]  # (k',) each
        np.log(block[0::2], out=r)
        r *= -2.0
        np.sqrt(r, out=r)
        np.multiply(2.0 * np.pi, block[1::2], out=theta)
        np.cos(theta, out=value)
        value *= r
        normals[2 * first : 2 * (first + size) : 2] = value  # float64 rounded to float32
        np.sin(theta, out=value)
        value *= r
        normals[2 * first + 1 : 2 * (first + size) : 2] = value
    return torch.from_numpy(normals[:count].reshape(n_rows, width))  # (n, d)


def pool_mean_var(rows: Tensor) -> Tensor:
    """Mean then population variance over every token row, in float32: the `final_mean_var` definition."""
    # rows: (n, w), all l + 2 token rows of one sequence
    mean = rows.mean(dim=0)  # (w,)
    variance = rows.var(dim=0, unbiased=False)  # (w,)
    return torch.cat([mean, variance])  # (2w,)


@dataclass(frozen=True, slots=True)
class ControlDescriptor:
    """What one control stream holds, as `FeatureView.descriptor` describes a stored one."""

    name: str
    layout: str  # "ragged" for the token stream, "dense" for the pooled one
    width: int  # w, or 2w when pooled
    dtype: torch.dtype


class SyntheticControlView:
    """Read a control like a base: `rows`, `pooled`, `residues` and `lengths`, with `FeatureView`'s shapes.

    `inventory` maps each sequence text to its `SequenceKey`; the key's SHA-256 seeds the random
    control, and a sequence keeps its first `max_residues` residues. Nothing is read from or written
    to storage. `width` overrides the random control's default of 960 (it must be even).
    """

    names = (TOKEN_STREAM, POOLED_STREAM)

    def __init__(
        self, kind: str, inventory: Mapping[str, SequenceKey], *, width: int | None = None,
        max_residues: int = CANONICAL_MAX_RESIDUES, budget: ReadBudget | None = None,
    ) -> None:
        self.kind = kind
        self.width = control_width(kind) if width is None else width  # w
        if kind == ONE_HOT and self.width != ONE_HOT_WIDTH:
            raise ValueError(f"The one-hot control has {ONE_HOT_WIDTH} columns; it takes no width.")
        if self.width < 2 or self.width % 2:
            raise ValueError("A control width must be a positive even integer.")
        if max_residues < 1:
            raise ValueError("max_residues must be positive.")
        self.max_residues = max_residues
        self.budget = ReadBudget() if budget is None else budget
        self._inventory = dict(inventory)
        for sequence, key in self._inventory.items():
            if (len(sequence) != key.length or sequence != normalize_sequence(sequence)
                    or hashlib.sha256(sequence.encode("ascii")).hexdigest() != key.sha256):
                # The key seeds the random control, so a key that is not the text's hash would make rows of another protein.
                raise ValueError(
                    "Protein text does not match the supplied sequence inventory, or is not normalized; "
                    "build the inventory with sequence_inventory."
                )

    @property
    def sequences(self) -> tuple[str, ...]:
        return tuple(self._inventory)

    @property
    def identity(self) -> dict[str, object]:
        """The arithmetic that made the rows, for a record that read a control."""
        return {
            "schema": CONTROL_SCHEMA, "kind": self.kind, "width": self.width,
            "max_residues": self.max_residues, "special_tokens": "kept",
            "seed": "row sha256" if self.kind == RANDOM_VECTOR else None,
            "pooling": "mean then population variance over all l+2 rows, float32",
        }

    def descriptor(self, name: str) -> ControlDescriptor:
        if name == TOKEN_STREAM:
            return ControlDescriptor(name, "ragged", self.width, torch.float32)
        if name == POOLED_STREAM:
            return ControlDescriptor(name, "dense", 2 * self.width, torch.float32)
        raise KeyError(f"A control answers to {self.names}; received {name!r}.")

    def close(self) -> None:
        """Nothing is held open; present so a control swaps for a `FeatureView` in a `with` block."""

    def __enter__(self) -> SyntheticControlView:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def lengths(self, name: str, sequences: Sequence[str]) -> tuple[int, ...]:
        """Residues `l` of each sequence after the crop, as `FeatureView.lengths`: CLS and EOS are not counted."""
        self.descriptor(name)
        return tuple(self._residues(sequence) for sequence in sequences)

    def rows(self, name: str, sequences: Sequence[str]) -> list[Tensor]:
        """Float32 rows in request order: `(l+2, w)` per sequence for tokens, `(2w,)` pooled."""
        self.descriptor(name)
        requested = self._requested(sequences)
        counts = [self._residues(sequence) + SPECIAL_TOKEN_ROWS for sequence in requested]  # b token rows n_i = l_i + 2
        if name == TOKEN_STREAM:
            self.budget.check(len(requested), padded_residues=sum(counts), tensor_bytes=sum(counts) * self.width * 4)
            return [self._token_rows(sequence) for sequence in requested]  # b of (l+2, w)
        # Pooled rows are made and reduced one sequence at a time: the output is (b, 2w), the transient (n_i, w).
        self.budget.check(
            len(requested), tensor_bytes=max(len(requested) * 2 * self.width, max(counts, default=0) * self.width) * 4,
        )
        return [pool_mean_var(self._token_rows(sequence)) for sequence in requested]  # b of (2w,)

    def pooled(self, sequences: Sequence[str], names: Sequence[str] | None = None) -> Tensor:
        """Mean then variance over all l + 2 rows, float32 `(b, 2w)`, as the stored `final_mean_var`."""
        if names is not None and tuple(names) != (POOLED_STREAM,):
            raise ValueError(f"A control pools only {POOLED_STREAM!r}; received {tuple(names)}.")
        rows = self.rows(POOLED_STREAM, sequences)  # b of (2w,)
        if not rows:
            return torch.empty((0, 2 * self.width), dtype=torch.float32)  # (0, 2w)
        return torch.stack(rows)  # (b, 2w)

    def residues(
        self, sequences: Sequence[str], names: Sequence[str] | None = None, *,
        crops: Sequence[tuple[int, int]] | None = None, reserve_bytes: int = 0,
    ) -> ResidueBatch:
        """Token rows padded to one length n: values `(b, n, w)`, row 0 CLS, rows 1..l_i residues, row l_i + 1 EOS.

        As `FeatureView.residues` for a v2 branch: a crop is a residue interval `[start, stop)` that keeps the
        CLS and EOS rows around the cropped residues, `lengths` counts rows, and padding is the only masked row.
        """
        names = (TOKEN_STREAM,) if names is None else tuple(names)
        if names != (TOKEN_STREAM,):
            raise ValueError(f"A control has one token stream, {TOKEN_STREAM!r}; received {names}.")
        requested = self._requested(sequences)
        ranges = _crop_ranges(self.lengths(TOKEN_STREAM, requested), crops)  # (b,) residue intervals [start, stop)
        row_counts = [stop - start + SPECIAL_TOKEN_ROWS for start, stop in ranges]  # (b,) n_i = cropped l_i + 2
        padded_length = max(row_counts, default=0)  # n = max n_i
        self.budget.check(
            len(requested), padded_residues=len(requested) * padded_length,
            tensor_bytes=len(requested) * padded_length * (self.width * 4 + 1) + reserve_bytes,
        )
        padded = torch.zeros((len(requested), padded_length, self.width), dtype=torch.float32)  # (b, n, w)
        for index, sequence in enumerate(requested):
            start, stop = ranges[index]
            _place_rows(padded[index], self._token_rows(sequence), start, stop, SPECIAL_TOKEN_ROWS)  # (n_i, w)
        length_tensor = torch.tensor(row_counts, dtype=torch.int64)  # (b,)
        mask = torch.arange(padded_length)[None, :] < length_tensor[:, None]  # (b, n), padding only
        starts = None if crops is None else torch.tensor([start for start, _ in ranges], dtype=torch.int64)  # (b,)
        return ResidueBatch({TOKEN_STREAM: padded}, mask, length_tensor, {}, starts)

    def _known(self, sequence: str) -> str:
        """The inventory's spelling of a requested protein: its exact text, else its `normalize_sequence` form."""
        if sequence in self._inventory:
            return sequence
        try:
            normalized = normalize_sequence(sequence)
        except ValueError:
            normalized = sequence
        if normalized not in self._inventory:
            raise KeyError("A requested sequence is outside this control's inventory.")
        return normalized

    def _requested(self, sequences: Sequence[str]) -> tuple[str, ...]:
        requested = tuple(sequences)
        self.budget.check(len(requested))
        return tuple(self._known(sequence) for sequence in requested)

    def _residues(self, sequence: str) -> int:
        return min(self._inventory[self._known(sequence)].length, self.max_residues)  # l

    def _token_rows(self, sequence: str) -> Tensor:
        """The l + 2 token rows of one sequence: row 0 CLS, rows 1..l residues, row l + 1 EOS."""
        kept = sequence[: self.max_residues]
        if self.kind == ONE_HOT:
            return one_hot_rows(kept)  # (l+2, v)
        return random_rows(self._inventory[sequence].sha256, len(kept) + 2, self.width)  # (l+2, d)


__all__ = [
    "CONTROL_KINDS", "CONTROL_SCHEMA", "ONE_HOT", "ONE_HOT_SYMBOLS", "ONE_HOT_WIDTH",
    "POOLED_STREAM", "RANDOM_VECTOR", "RANDOM_WIDTH", "TOKEN_STREAM", "ControlDescriptor",
    "SyntheticControlView", "control_width", "one_hot_rows", "pool_mean_var", "random_rows",
]
