"""SAE taps: sparse-autoencoder features from a FastPLMs one-pass tap plan.

`sae_tap` builds a FastPLMs `ReducedTap` whose reducer encodes the tapped hidden state with an SAE
and pools the codes per sequence, so SAE features come out of the same forward pass as any dense
tap. Two row policies exist, chosen when the tap is built:

- Residue taps (default, legacy): use the biological-residue mask of dense taps. This drops BOS, EOS,
  padding, and internal special tokens, so a protein of l residues contributes l rows.
- Token taps (`keep_special_tokens=True`, canonical): encode every attended token, CLS and EOS included,
  so the same protein contributes l + 2 rows (row 0 CLS, rows 1..l residues, row l + 1 EOS) and padding is
  the only masked position. They run in FastPLMs' token executor, which supplies the row selection and a
  per-batch cache, so the pooled and per-token SAE taps encode each batch once.

Encoding goes through `encode_topk`, `chunk_rows` rows at a time. FastPLMs'
`forward(compute_sae=True)` instead scatters every attended token's codes into one dense
`(n, c)` tensor before `to_sparse()`; this path never builds it.

A token tap built with a readout (`ContinuousReadout` or `HardTopKReadout`) is an embedding profile's tap: fixed-shape
encoder chunks, the profile's readout and canonical slot order (`sae_features`). It carries its own schema, so its keys
never meet those of the v2 stores' plain top-k taps.

FastPLMs is imported only inside `sae_tap`. Importing `foundry.embedding` needs nothing beyond
torch, because Protify ships this package inside exported probes that carry no FastPLMs source.

Symbols: b sequences of a batch; l residues of one sequence after the crop; d hidden width; c SAE codebook;
k top-k codes kept per row; n attended rows of a batch (sum of l_i for residue taps, sum of l_i + 2 for
token taps); p pooling reductions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from torch import Tensor, nn

from .sae_features import (
    CHUNK_POLICY_FIXED, POOLING_REDUCTIONS, STANDARDIZATION_EPSILON, ContinuousReadout, Readout, TopKFeatures, encode_topk,
    pool_all_rows, pool_features, row_layout,
)


if TYPE_CHECKING:
    from fastplms.embeddings import ReducedTap, SparseResidueTap, TapBatch
    from fastplms.features import TopKRow


# Bump when the reducer's arithmetic changes, so run fingerprints change with it. Token taps are a different
# row policy of the same arithmetic, so they carry their own schema and never share a key with residue taps;
# the profile's taps (a readout, fixed chunks) carry theirs.
SAE_TAP_SCHEMA = 2
SAE_TAP_TOKEN_SCHEMA = 3
SAE_TAP_PROFILE_SCHEMA = 4
SAE_RESIDUE_TAP_SCHEMA = 1
SAE_RESIDUE_TAP_TOKEN_SCHEMA = 2
SAE_RESIDUE_TAP_PROFILE_SCHEMA = 3
ROWS_RESIDUES = "biological_residue_mask"
ROWS_TOKENS = "all_attended_tokens"


def sae_tap_schema(*, keep_special_tokens: bool, profile: bool) -> int:
    """The schema of a pooled SAE tap for its row policy and readout."""
    if profile:
        return SAE_TAP_PROFILE_SCHEMA
    return SAE_TAP_TOKEN_SCHEMA if keep_special_tokens else SAE_TAP_SCHEMA


def sae_residue_tap_schema(*, keep_special_tokens: bool, profile: bool) -> int:
    """The schema of a per-token SAE tap for its row policy and readout."""
    if profile:
        return SAE_RESIDUE_TAP_PROFILE_SCHEMA
    return SAE_RESIDUE_TAP_TOKEN_SCHEMA if keep_special_tokens else SAE_RESIDUE_TAP_SCHEMA


def _sae_identity(
    name: str, sae_layer: nn.Module, layer: int, chunk_rows: int, rows: str, readout: Readout | None = None,
) -> dict[str, object]:
    """Validate the common encoder inputs and hash the exact arithmetic dependencies."""
    from fastplms.embeddings import tensor_sha256

    encoder = getattr(sae_layer, "W_enc", None)  # (d, c)
    decoder_bias = getattr(sae_layer, "b_dec", None)  # (d,)
    k = getattr(getattr(sae_layer, "params", None), "k", None)
    if not isinstance(encoder, Tensor) or encoder.ndim != 2:
        raise TypeError("sae_layer must expose W_enc, a (d, c) tensor.")
    if not isinstance(decoder_bias, Tensor) or tuple(decoder_bias.shape) != (encoder.shape[0],):
        raise TypeError("sae_layer must expose b_dec, a (d,) tensor matching W_enc.")
    if not isinstance(k, int) or isinstance(k, bool) or not 0 < k <= encoder.shape[1]:
        raise ValueError(f"sae_layer.params.k must be an integer in 1..{encoder.shape[1]}.")
    declared_layer = getattr(sae_layer, "layer", None)
    if isinstance(declared_layer, int) and not isinstance(declared_layer, bool) and declared_layer != layer:
        raise ValueError(
            f"sae_tap {name!r} taps hidden state {layer}, but the SAE was trained on hidden state "
            f"{declared_layer}; pass the SAE's own hidden-state index."
        )
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer.")
    if isinstance(readout, ContinuousReadout) and k >= encoder.shape[1]:
        raise ValueError("The continuous readout needs a codebook wider than k to find its cutoff.")
    identity: dict[str, object] = {
        "rows": rows, "standardization_correction": 1,
        "standardization_epsilon": STANDARDIZATION_EPSILON,
        "encoder_dtype": str(encoder.dtype).removeprefix("torch."),
        "chunk_rows": chunk_rows, "k": k, "codebook_dim": int(encoder.shape[1]),
        "W_enc_sha256": tensor_sha256(encoder), "b_dec_sha256": tensor_sha256(decoder_bias),
    }
    if readout is not None:  # a v2 store's plain top-k tap keeps exactly its published fields
        identity.update(readout.describe(), chunk_policy=CHUNK_POLICY_FIXED)
    return identity


def _token_codes(
    batch: TapBatch, sae_layer: nn.Module, layer: int, chunk_rows: int, readout: Readout | None = None,
) -> TopKFeatures:
    """Top-k codes of every attended token row of the batch, encoded once and shared by the SAE taps.

    A token tap runs only in the token executor, which supplies `batch.rows` (host-known row selection) and a
    per-batch `batch.cache`. Selecting rows with a prebuilt index keeps the host from waiting on the device. A
    profile tap (a ``readout``) encodes in fixed-shape chunks.
    """
    if batch.rows is None:
        raise ValueError("A token SAE tap runs only in FastPLMs' token executor, which supplies the row selection.")
    key = ("sae_topk", id(sae_layer), layer, chunk_rows, str(batch.X.dtype), readout)
    if key not in batch.cache:
        rows = batch.rows.gather(batch.X)  # (b, m, d) -> (n, d): CLS, residues and EOS of each sequence, no padding
        batch.cache[key] = encode_topk(  # indices, values: (n, k)
            rows, sae_layer, chunk_rows=chunk_rows, readout=readout, fixed_chunks=readout is not None,
        )
    return batch.cache[key]


def sae_tap(
    name: str,
    sae_layer: nn.Module,
    layer: int,
    pooling: str | Sequence[str] = ("max",),
    *,
    chunk_rows: int = 1024,
    keep_special_tokens: bool = False,
    readout: Readout | None = None,
) -> ReducedTap:
    """A tap that reduces hidden state `layer` to pooled SAE codes, one `(p * c,)` vector per sequence.

    `sae_layer` is a FastPLMs `ESMplusplusSAELayer` or an official Biohub layer on the model's
    device; the reducer reads only `W_enc`, `b_dec`, and `params.k`. `layer` is the FastPLMs
    hidden-state index the SAE reads and must equal the layer's own `.layer` when it declares one.
    `pooling` names reductions from `POOLING_REDUCTIONS`, concatenated in order. The tap's identity
    hashes the SAE weights when the tap is built, so build a new tap after changing them.
    `keep_special_tokens=True` pools over all l + 2 token rows (CLS and EOS included) instead of the l residues.
    A ``readout`` (token taps only) makes it an embedding profile's tap: fixed encoder chunks, the profile's readout.
    """
    # Imported here so that importing foundry.embedding needs no FastPLMs source.
    from fastplms.embeddings import ReducedTap

    if readout is not None and not keep_special_tokens:
        raise ValueError("A profile's readout belongs to the token row policy; set keep_special_tokens=True.")
    identity = _sae_identity(
        name, sae_layer, layer, chunk_rows, ROWS_TOKENS if keep_special_tokens else ROWS_RESIDUES, readout,
    )
    pooling_names = (pooling,) if isinstance(pooling, str) else tuple(pooling)
    unknown = [reduction for reduction in pooling_names if reduction not in POOLING_REDUCTIONS]
    if not pooling_names or unknown or len(set(pooling_names)) != len(pooling_names):
        raise ValueError(
            f"pooling must name distinct reductions from {POOLING_REDUCTIONS}; received {pooling_names}."
        )

    def reduce(batch: TapBatch) -> Tensor:
        # batch.X: (b, l, d); masks: (b, l). Validate containment before selecting rows.
        row_layout(batch.token_mask, batch.residue_mask)
        rows = batch.residue_mask.bool()  # (b, l)
        if not bool(rows.any(dim=1).all()):
            raise ValueError("Every SAE tap sample must contain at least one biological residue.")
        codes = encode_topk(batch.X[rows], sae_layer, chunk_rows=chunk_rows)  # indices, values: (n, k)
        return pool_features(codes, row_layout(rows, rows), rows.shape[0], pooling=pooling_names)  # (b, p * c)

    def reduce_tokens(batch: TapBatch) -> Tensor:
        # batch.X: (b, m, d) padded to m = max(l) + 2; the selection keeps CLS, residues and EOS of each sequence.
        codes = _token_codes(batch, sae_layer, layer, chunk_rows, readout)  # indices, values: (n, k), n = sum(l_i + 2)
        selection = batch.selection()
        # The reduction runs over all l + 2 rows of a sequence; padding never entered the rows.
        return pool_all_rows(codes, selection.owner, selection.sizes, pooling=pooling_names)  # (b, p * c)

    identity.update({
        "reducer": "foundry.embedding.sae_tap",
        "schema": sae_tap_schema(keep_special_tokens=keep_special_tokens, profile=readout is not None),
        "pooling_dtype": "float32",
        "pooling": list(pooling_names),
    })
    return ReducedTap(name=name, layer=layer, reduce=reduce_tokens if keep_special_tokens else reduce, identity=identity)


def sae_residue_tap(
    name: str, sae_layer: nn.Module, layer: int, *, chunk_rows: int = 1024, keep_special_tokens: bool = False,
    readout: Readout | None = None,
) -> SparseResidueTap:
    """Keep all k SAE codes per row, in the encoder's dtype and top-k order.

    This shares encoder arithmetic with the pooled tap, but defers pooling until the consumer
    selects its crop. Zero-valued top-k entries are retained. Only row-by-k outputs leave
    the reducer; the temporary SAE projection is bounded by ``chunk_rows``. With
    `keep_special_tokens=True` a protein of l residues stores (l + 2, k) codes: row 0 CLS, rows 1..l
    residues, row l + 1 EOS. A legacy residue tap stores (l, k) and excludes CLS and EOS. A ``readout``
    (token taps only) stores that readout's values in canonical slot order instead of top-k order.
    """
    from fastplms.embeddings import SparseResidueTap
    from fastplms.features import TopKRow

    if readout is not None and not keep_special_tokens:
        raise ValueError("A profile's readout belongs to the token row policy; set keep_special_tokens=True.")
    rows_policy = ROWS_TOKENS if keep_special_tokens else ROWS_RESIDUES
    identity = _sae_identity(name, sae_layer, layer, chunk_rows, rows_policy, readout)
    identity.update({
        "reducer": "foundry.embedding.sae_residue_tap",
        "schema": sae_residue_tap_schema(keep_special_tokens=keep_special_tokens, profile=readout is not None),
        "retain_zero_entries": True,
    })
    if readout is None:  # a profile tap's identity already names its canonical code order
        identity["code_order"] = "encoder_topk"

    def reduce(batch: TapBatch) -> list[TopKRow]:
        if keep_special_tokens:
            codes = _token_codes(batch, sae_layer, layer, chunk_rows, readout)  # indices, values: (n, k)
            counts = list(batch.selection().counts)  # (b,) l_i + 2 rows of each sequence
        else:
            row_layout(batch.token_mask, batch.residue_mask)
            mask = batch.residue_mask.bool()  # (b, l), excludes every special token and padding.
            counts = mask.sum(dim=1).tolist()  # (b,) residues l_i
            if any(count == 0 for count in counts):
                raise ValueError("Every SAE tap sample must contain at least one biological residue.")
            codes = encode_topk(batch.X[mask], sae_layer, chunk_rows=chunk_rows)  # indices, values: (n, k)
        return [TopKRow(indices, values) for indices, values in zip(
            codes.indices.split(counts), codes.values.split(counts), strict=True,
        )]

    def reduce_packed(batch: TapBatch) -> TopKRow:
        codes = _token_codes(batch, sae_layer, layer, chunk_rows, readout)  # indices, values: (n, k)
        return TopKRow(codes.indices, codes.values)  # every sequence's l_i + 2 rows, back to back

    return SparseResidueTap(
        name, layer, reduce, identity, codebook_size=sae_layer.W_enc.shape[1], sparse_count=sae_layer.params.k,
        reduce_packed=reduce_packed if keep_special_tokens else None,
    )
