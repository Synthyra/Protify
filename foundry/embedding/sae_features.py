"""Sparse-autoencoder features from FastPLMs outputs, down to one vector per sequence.

FastPLMs owns the SAE model (`fastplms.models.esm_plusplus.modeling_esm_plusplus_sae`) and
returns SAE features flattened over the batch: one row per attended token, row-major over
`(sequence, position)`, with no index saying which row is which. Seven projects rebuilt that
index and pooled the rows themselves, several by position arithmetic that assumes right padding
and exactly one BOS and one EOS. This module owns those steps once:

- `encode_topk` keeps each residue's `k` strongest codes as `(indices, values)`, without the
  dense `(rows, c)` scatter the layer's own forward performs. A profile names its readout: `HardTopKReadout`
  (`sae_topk_canonical_v1`) keeps the exact top k with ties chosen by code index, and `ContinuousReadout`
  (`sae_topk_v2`, profile v1's) fades a code to zero as it nears the cutoff. With `fixed_chunks` every encoder GEMM has
  the same shape.
- `row_layout` recovers each flattened row's sequence and position from the mask FastPLMs used,
  and `concatenated_rows` does the same for residues concatenated sequence by sequence.
- `interior_tokens` drops each sequence's first and last token, the Biohub reference policy.
- `pool_features` reduces rows to sequences by max, mean, or sum without densifying to
  `(b, l, c)`, which is infeasible at codebook widths of 8k to 131k.
- `pool_all_rows` does the same over every token row of a canonical run, CLS and EOS included, with no
  keep mask and so no device read.

Dense `(b, l, d)` pooling is not here: `fastplms.embeddings.Pooler` owns it.

Symbols: b sequences of a batch; l token columns of the padded batch; d hidden width; c SAE codebook; k codes kept per
row; n flattened attended rows (sum of l_i residues, or of l_i + 2 when CLS and EOS are kept); p reductions.
"""

from __future__ import annotations

import torch

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple
from torch import Tensor, nn


POOLING_REDUCTIONS = ("max", "mean", "sum")
# Added to each residue's sample standard deviation, as in the published Biohub SAEs and
# FastPLMs' `STANDARDIZATION_EPSILON`.
STANDARDIZATION_EPSILON = 1e-5
READOUT_CONTINUOUS = "sae_topk_v2"
READOUT_HARD_CANONICAL = "sae_topk_canonical_v1"
CHUNK_POLICY_FIXED = "fixed_rows_zero_pad_v1"  # every chunk has chunk_rows rows; the last is padded with zero rows
CODE_ORDER_CANONICAL = "value_desc_code_asc_v1"  # strongest first, equal values by code index
ZERO_SLOTS_CANONICAL = "lowest_free_codes_v1"  # a slot of value zero names the lowest code no other slot holds


@dataclass(frozen=True, slots=True)
class ContinuousReadout:
    """The continuous top-k readout, `sae_topk_v2`: no code drops out at the cutoff, it fades to zero there.

    Of a row's rectified pre-activations x, `topk(k + 1)` gives the kept codes and the cutoff tau, the (k + 1)-th
    largest. A kept code is stored as `x_i * clamp((x_i - tau) / delta, 0, 1)` with `delta = rho * max(x)`: a code
    more than `delta` above the cutoff keeps its exact value, and one inside the band shrinks to zero at the cutoff.
    Two codes that trade places at the cutoff both stood at zero there, so the swap changes no stored value. tau and
    max(x) are order statistics, each 1-Lipschitz in the max norm, so the stored codes move by at most
    `(2 + 2 / rho)` times the largest change of any pre-activation. A row with no positive code stores zeros.
    """

    rho: float

    def __post_init__(self) -> None:
        if type(self.rho) is not float or not 0.0 < self.rho <= 1.0:
            raise ValueError(f"ContinuousReadout.rho must be a float in (0, 1]; received {self.rho!r}.")

    @property
    def lipschitz_bound(self) -> float:
        """The readout's Lipschitz constant in the max norm, from pre-activations to stored values."""
        return 2.0 + 2.0 / self.rho

    def describe(self) -> dict[str, object]:
        """The readout as a tap identity records it."""
        return {"readout": READOUT_CONTINUOUS, "rho": self.rho, "code_order": CODE_ORDER_CANONICAL,
                "zero_slots": ZERO_SLOTS_CANONICAL}


@dataclass(frozen=True, slots=True)
class HardTopKReadout:
    """The SAE's own top-k, `sae_topk_canonical_v1`: each row keeps its `k` largest codes at their exact values.

    The selection is made a function of the values alone: of codes with equal values at the cutoff, the lowest code
    indices are kept, slots are ordered strongest first and equal values by code index, and a slot of value zero names
    the lowest code no other slot holds. Within one hardware class the input is bitwise fixed, so this is too; it does
    not bound differences between classes, which is why the profiles that use it compute on one class only.
    """

    def describe(self) -> dict[str, object]:
        """The readout as a tap identity records it."""
        return {"readout": READOUT_HARD_CANONICAL, "code_order": CODE_ORDER_CANONICAL, "zero_slots": ZERO_SLOTS_CANONICAL}


Readout = ContinuousReadout | HardTopKReadout  # the readout of a profile's SAE taps; None is the v2 stores' plain top-k


@dataclass(frozen=True)
class TopKFeatures:
    """The `k` strongest SAE codes of each residue row: codebook index and magnitude.

    A magnitude of zero marks an inactive slot; its index carries no meaning.
    """

    indices: Tensor  # (n, k) int64 codebook positions
    values: Tensor  # (n, k) non-negative magnitudes
    codebook_dim: int  # c

    def to_dense(self) -> Tensor:
        dense = torch.zeros(  # (n, c)
            self.indices.shape[0], self.codebook_dim, dtype=self.values.dtype, device=self.values.device
        )
        return dense.scatter(-1, self.indices, self.values)  # (n, c)


class RowLayout(NamedTuple):
    """Where each row of a flattened SAE output came from."""

    owner: Tensor  # (n,) sequence index of each row
    position: Tensor  # (n,) position of each row within its sequence
    keep: Tensor  # (n,) bool, the row is a residue to pool


def encode_topk(
    residue_states: Tensor, sae_layer: nn.Module, *, chunk_rows: int = 1024, readout: Readout | None = None,
    fixed_chunks: bool = False,
) -> TopKFeatures:
    """Top-k SAE codes of residue hidden states, in the SAE's dtype.

    This is the encoder of `ESMplusplusSAELayer.forward` and of Biohub's layers: standardize each
    residue by its sample standard deviation, subtract the decoder bias, project, rectify, and
    keep the `k` largest. It reads only `W_enc`, `b_dec`, and `params.k`, which both expose.
    The layer's forward would also scatter the top k into a second dense `(rows, c)` tensor,
    which this skips. `tests/tier1_unit` holds the result to the FastPLMs layer's own forward, so
    a change there fails a test instead of drifting.

    The projection is still dense, 512 MB for 1,024 rows at c = 131,072 in float32, so rows are
    encoded `chunk_rows` at a time. Gradients are not kept: this is for extraction.

    ``fixed_chunks`` pads the last chunk with zero rows to ``chunk_rows``, so every projection GEMM has one shape
    whatever the batch holds (a GEMM's reduction order follows its shape); the padding rows are dropped. With a
    ``readout`` (a profile's) the kept values follow it, `HardTopKReadout` choosing ties at the cutoff by code index, and
    the slots are put in canonical order: strongest first, equal values by code index, and a slot of value zero names the
    lowest code index no other slot of its row holds.
    """
    # residue_states: (n, d); W_enc: (d, c); b_dec: (d,)
    encoder, decoder_bias, k = sae_layer.W_enc, sae_layer.b_dec, int(sae_layer.params.k)
    index_chunks, value_chunks = [], []
    with torch.no_grad():
        for start in range(0, residue_states.shape[0], chunk_rows):
            chunk = residue_states[start : start + chunk_rows].to(encoder.dtype)  # (m, d)
            rows = chunk.shape[0]
            if fixed_chunks and rows < chunk_rows:
                chunk = torch.cat([chunk, chunk.new_zeros(chunk_rows - rows, chunk.shape[1])])  # (chunk_rows, d)
            centered = chunk - chunk.mean(dim=-1, keepdim=True)  # (m, d)
            standardized = centered / (centered.std(dim=-1, keepdim=True) + STANDARDIZATION_EPSILON)  # (m, d)
            activations = torch.relu((standardized - decoder_bias) @ encoder)  # (m, c)
            if readout is None:
                retained = activations.topk(k, dim=-1)  # values, indices: (m, k)
                indices, values = retained.indices, retained.values  # (m, k), (m, k)
            elif isinstance(readout, HardTopKReadout):
                values, indices = canonical_topk(activations, k)  # (m, k), (m, k)
            else:
                values, indices = continuous_topk(activations, k, readout)  # (m, k), (m, k)
            index_chunks.append(indices[:rows])  # (rows, k)
            value_chunks.append(values[:rows])  # (rows, k)
    if not index_chunks:
        index_chunks.append(residue_states.new_empty(0, k, dtype=torch.int64))  # (0, k)
        value_chunks.append(residue_states.new_empty(0, k, dtype=encoder.dtype))  # (0, k)
    return TopKFeatures(
        indices=torch.cat(index_chunks),  # (n, k)
        values=torch.cat(value_chunks),  # (n, k)
        codebook_dim=int(encoder.shape[1]),
    )


def canonical_topk(activations: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """The `k` largest codes of each row at their exact values, values then indices, chosen and ordered canonically.

    `torch.topk` may keep any of several codes tied at the cutoff. A row whose k-th and (k + 1)-th values are equal and
    positive (rare in float32) is chosen again exactly: every code above the tied value, then the lowest code indices
    holding it. A tie at zero needs nothing, because a zero slot's code is renamed by `canonical_slots`.
    """
    # activations: (m, c) rectified pre-activations
    if activations.shape[1] <= k:
        top = activations.topk(k, dim=-1, sorted=True)  # values, indices: (m, k), every code
        return canonical_slots(top.values, top.indices)  # (m, k), (m, k)
    top = activations.topk(k + 1, dim=-1, sorted=True)  # values, indices: (m, k + 1), strongest first
    values, indices = top.values[:, :k], top.indices[:, :k]  # (m, k), (m, k)
    tied = (top.values[:, k - 1] == top.values[:, k]) & (top.values[:, k - 1] > 0)  # (m,)
    if tied.any():
        rows = tied.nonzero().flatten()  # (t,)
        level = top.values[rows, k - 1 : k]  # (t, 1) the tied value
        candidates = activations[rows]  # (t, c)
        above, equal = candidates > level, candidates == level  # (t, c) each
        room = k - above.sum(dim=1, keepdim=True)  # (t, 1) slots left for the tied codes
        chosen = above | (equal & (equal.cumsum(dim=1) <= room))  # (t, c) exactly k per row, lowest tied codes
        picked = chosen.nonzero()[:, 1].view(len(rows), k)  # (t, k) ascending code indices
        values, indices = values.clone(), indices.clone()  # (m, k) each
        values[rows], indices[rows] = candidates.gather(1, picked), picked  # (t, k) each
    return canonical_slots(values, indices)  # (m, k), (m, k)


def continuous_topk(activations: Tensor, k: int, readout: ContinuousReadout) -> tuple[Tensor, Tensor]:
    """The `k` codes of each row under the continuous readout, values then indices, in canonical slot order."""
    # activations: (m, c) rectified pre-activations, c > k
    if activations.shape[1] <= k:
        raise ValueError(f"The continuous readout needs more than k = {k} codes to find its cutoff; the codebook has {activations.shape[1]}.")
    top = activations.topk(k + 1, dim=-1, sorted=True)  # values, indices: (m, k + 1), strongest first
    cutoff = top.values[:, k:]  # (m, 1) tau, the (k + 1)-th largest
    band = (readout.rho * top.values[:, :1]).clamp_min(torch.finfo(activations.dtype).tiny)  # (m, 1) delta = rho * max(x)
    kept = top.values[:, :k]  # (m, k)
    # A row with no positive code has kept == cutoff == 0, so its weights and values are 0, never 0 / 0.
    values = kept * ((kept - cutoff) / band).clamp(0.0, 1.0)  # (m, k)
    return canonical_slots(values, top.indices[:, :k])  # (m, k), (m, k)


def canonical_slots(values: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Order a row's slots strongest first, equal values by code index, and name each zero slot canonically.

    `torch.topk` promises no order among equal values, and a slot of value zero names an arbitrary code. Here a zero
    slot takes the lowest code index that no positive slot of its row holds, in ascending order, so a row's
    indices are a function of its values alone and stay distinct. Of the codes 0..k-1 at most k - z are held by the
    k - z positive slots, so the z zero slots always find their codes there.
    """
    # values, indices: (m, k) non-negative values and their code indices
    k = values.shape[1]
    active = values > 0  # (m, k)
    taken = torch.zeros(values.shape[0], k + 1, dtype=torch.int8, device=values.device)  # (m, k + 1); column k collects codes >= k
    taken.scatter_(1, torch.where(active, indices.clamp(max=k), k), 1)  # (m, k + 1)
    free = torch.argsort(taken[:, :k], dim=1, stable=True)  # (m, k) codes below k that no positive slot holds, ascending, first
    by_code = torch.argsort(indices, dim=1, stable=True)  # (m, k)
    values, indices = values.gather(1, by_code), indices.gather(1, by_code)  # (m, k), (m, k) ascending code
    by_value = torch.argsort(values, dim=1, descending=True, stable=True)  # (m, k) strongest first, equal values by code
    values, indices = values.gather(1, by_value), indices.gather(1, by_value)  # (m, k), (m, k)
    positive = active.sum(dim=1, keepdim=True)  # (m, 1) positive slots, which now come first
    slot = torch.arange(k, device=values.device).expand_as(indices)  # (m, k)
    zero_rank = (slot - positive).clamp_min(0)  # (m, k) each zero slot's rank among its row's zero slots
    indices = torch.where(slot >= positive, free.gather(1, zero_rank), indices)  # (m, k)
    return values, indices  # (m, k), (m, k)


def row_layout(token_mask: Tensor, residue_mask: Tensor) -> RowLayout:
    """Recover the sequence and position of every row FastPLMs flattened out of `token_mask`.

    FastPLMs encodes `layer_states[token_mask]`, and boolean indexing and `nonzero` walk a
    tensor in the same row-major order, so this holds for either padding side. `residue_mask`
    narrows the rows to pool, for example to biological residues or to `interior_tokens`.
    """
    # token_mask, residue_mask: (b, l)
    if token_mask.shape != residue_mask.shape:
        raise ValueError(f"token and residue masks differ in shape: {tuple(token_mask.shape)} and {tuple(residue_mask.shape)}.")
    token_mask, residue_mask = token_mask.bool(), residue_mask.bool()
    if bool((residue_mask & ~token_mask).any()):
        raise ValueError("residue_mask must be a subset of token_mask; rows outside token_mask do not exist.")
    owner, position = token_mask.nonzero(as_tuple=True)  # (n,), (n,)
    return RowLayout(owner=owner, position=position, keep=residue_mask[token_mask])


def concatenated_rows(lengths: Tensor) -> RowLayout:
    """Layout of rows concatenated sequence by sequence: `lengths[i]` rows belong to sequence `i`.

    For states already reduced to residues, such as per-sequence tensors joined by `torch.cat`.
    Positions count residues from zero within each sequence, and every row is kept.
    """
    # lengths: (b,)
    lengths = lengths.to(torch.int64)
    owner = torch.repeat_interleave(torch.arange(lengths.shape[0], device=lengths.device), lengths)  # (n,)
    starts = torch.cumsum(lengths, dim=0) - lengths  # (b,)
    position = torch.arange(owner.shape[0], device=lengths.device) - starts[owner]  # (n,)
    return RowLayout(owner=owner, position=position, keep=torch.ones_like(owner, dtype=torch.bool))


def interior_tokens(token_mask: Tensor) -> Tensor:
    """Attended tokens other than each sequence's first and last.

    Under ESM tokenizers those two are BOS and EOS, whose activations describe no residue.
    Dropping them reproduces the Biohub reference workflow whatever the padding side. A
    sequence of two or fewer tokens has no interior.
    """
    # token_mask: (b, l)
    token_mask = token_mask.bool()
    positions = torch.arange(token_mask.shape[1], device=token_mask.device).expand_as(token_mask)  # (b, l)
    first = torch.where(token_mask, positions, token_mask.shape[1]).amin(dim=1, keepdim=True)  # (b, 1)
    last = torch.where(token_mask, positions, -1).amax(dim=1, keepdim=True)  # (b, 1)
    return token_mask & (positions > first) & (positions < last)  # (b, l)


def pool_features(
    features: Tensor | TopKFeatures,
    layout: RowLayout,
    n_sequences: int,
    *,
    pooling: Sequence[str] = ("max",),
) -> Tensor:
    """One float32 vector per sequence from per-row SAE features, reductions concatenated in order.

    `features` has one row per `layout` row: dense `(n, c)`, sparse COO `(n, c)`, or
    `TopKFeatures`. Rows with `layout.keep` false are ignored. `mean` divides by the kept rows of
    each sequence. SAE magnitudes are non-negative, which makes zero the identity for `max` and
    lets a sequence with no kept rows pool to zeros; negative input is rejected rather than
    silently clipped.
    """
    # features: (n, c) dense or sparse, or top-k indices and values (n, k); one row per layout row.
    unknown = [name for name in pooling if name not in POOLING_REDUCTIONS]
    if unknown or not pooling:
        raise ValueError(f"pooling must name some of {POOLING_REDUCTIONS}; received {tuple(pooling)}.")
    rows, columns, values, codebook_dim = _entries(features)  # (nnz,), (nnz,), (nnz,), c
    n_rows = layout.owner.shape[0]
    if _row_count(features) != n_rows:
        raise ValueError(f"features have {_row_count(features)} rows but the layout has {n_rows}.")
    if values.numel() and bool((values < 0).any()):
        raise ValueError("SAE features must be non-negative; zero is the identity this pooling relies on.")

    kept = layout.keep[rows]  # (nnz,)
    flat = layout.owner[rows[kept]] * codebook_dim + columns[kept]  # (m,)
    values = values[kept]  # (m,)
    residues = torch.bincount(layout.owner[layout.keep], minlength=n_sequences).clamp_min(1)  # (b,)

    pooled = []
    for name in pooling:
        reduced = torch.zeros(n_sequences * codebook_dim, dtype=torch.float32, device=values.device)  # (b * c,)
        reduced.scatter_reduce_(0, flat, values, reduce="amax" if name == "max" else "sum", include_self=True)
        reduced = reduced.view(n_sequences, codebook_dim)  # (b, c)
        pooled.append(reduced / residues.unsqueeze(-1) if name == "mean" else reduced)
    return torch.cat(pooled, dim=-1)  # (b, p * c), p = len(pooling)


def pool_all_rows(
    features: TopKFeatures, owner: Tensor, counts: Tensor, *, pooling: Sequence[str] = ("max",),
) -> Tensor:
    """One float32 vector per sequence over every token row, CLS and EOS included, with no device read.

    ``features`` holds the top-k codes of the n = sum(l_i + 2) attended rows of a batch: indices and values
    (n, k). ``owner`` (n,) names each row's sequence and ``counts`` (b,) holds l_i + 2. Unlike
    ``pool_features`` there is no keep mask, so nothing is selected by a data-dependent index and the host
    never waits. SAE magnitudes are non-negative by construction (a rectifier precedes the top-k), which is
    why zero is the identity for max and padding slots of value zero change nothing.
    """
    # owner: (n,) int64, the sequence of each row.
    # counts: (b,) int64, the attended rows l_i + 2 of each sequence.
    unknown = [name for name in pooling if name not in POOLING_REDUCTIONS]
    if unknown or not pooling:
        raise ValueError(f"pooling must name some of {POOLING_REDUCTIONS}; received {tuple(pooling)}.")
    n_sequences, codebook = counts.shape[0], features.codebook_dim
    flat = (owner.unsqueeze(1) * codebook + features.indices).reshape(-1)  # (n * k,)
    values = features.values.reshape(-1).float()  # (n * k,)
    pooled = []
    for name in pooling:
        reduced = torch.zeros(n_sequences * codebook, dtype=torch.float32, device=values.device)  # (b * c,)
        reduced.scatter_reduce_(0, flat, values, reduce="amax" if name == "max" else "sum", include_self=True)
        reduced = reduced.view(n_sequences, codebook)  # (b, c)
        pooled.append(reduced / counts.unsqueeze(-1).float() if name == "mean" else reduced)
    return torch.cat(pooled, dim=-1)  # (b, p * c), p = len(pooling)


def _row_count(features: Tensor | TopKFeatures) -> int:
    # features: (n, c) dense or sparse, or top-k indices and values (n, k); the row count is n.
    return int(features.indices.shape[0] if isinstance(features, TopKFeatures) else features.shape[0])


def _entries(features: Tensor | TopKFeatures) -> tuple[Tensor, Tensor, Tensor, int]:
    """Every stored feature as (row, column, float32 value), plus the codebook width."""
    # features: (n, c) dense or sparse, or top-k indices and values (n, k).
    if isinstance(features, TopKFeatures):
        n, k = features.indices.shape
        rows = torch.arange(n, device=features.indices.device).repeat_interleave(k)  # (n * k,)
        return rows, features.indices.reshape(-1), features.values.reshape(-1).float(), features.codebook_dim  # (n * k,) each
    if features.is_sparse:
        features = features.coalesce()
        rows, columns = features.indices()  # (nnz,), (nnz,)
        return rows, columns, features.values().float(), int(features.shape[1])  # (nnz,) each
    rows, columns = features.nonzero(as_tuple=True)  # (nnz,), (nnz,)
    return rows, columns, features[rows, columns].float(), int(features.shape[1])  # (nnz,) each
