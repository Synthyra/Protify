"""Verified, bounded training views over the existing FastPLMs feature store.

Descriptors and inventory identities must come from independently qualified extraction and
dataset records. This reader validates their agreement, without loading an encoder. It holds
sequence metadata, not corpus tensors, and imports FastPLMs only when a view is constructed.

Symbols: `b` requested rows, `l` residues after the crop, `r_i` residues of row `i`, `w` a branch's
stored columns (its pooled width, or `d`, 256, or `k` per token), `k` SAE codes kept per token, and
`nnz_i` codes stored in a CSR row.

Rows. A legacy v1 residue branch stores `r_i` rows per sequence. A canonical v2 branch keeps the
special tokens and stores `r_i + 2`: row 0 CLS, rows 1..r_i residues, row `r_i + 1` EOS. `lengths`
always counts residues. `residues` pads to `l + 2` rows for v2, and a crop is a residue interval
`[start, stop)` that keeps the CLS and EOS rows around the cropped residues. Padding is the only
masked position, so a mask counts CLS and EOS as real rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
import torch

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast
from torch import Tensor

from .contracts import (
    PROTEIN_NORMALIZATION, SPECIAL_TOKEN_ROWS, FeatureRowKey, FeatureSpec, SequenceKey, spelling_normalizer, text_is_unit_count,
)


if TYPE_CHECKING:
    from fastplms.features import FeatureReader, FeatureStore, SparseRow, StoredFeature, TopKRow


VERIFY_WORKERS = min(4, os.cpu_count() or 1)  # parts verified at once; each loads its whole tensors, so memory grows with this


def checked_row_identity(
    record: Mapping[str, object], feature: FeatureSpec, sequence: SequenceKey,
) -> FeatureRowKey:
    """Require the original sequence and every retained row of its inference prefix.

    A v1 row retains its `l` residues; a v2 row retains `l + 2` token rows, CLS and EOS included.
    """
    row = FeatureRowKey.from_dict(record, feature=feature)
    if row.sequence != sequence:
        raise ValueError("Stored row sequence identity differs from the original inventory.")
    if not row.complete:
        raise ValueError("Canonical row positions do not preserve every residue of the inference prefix.")
    return row


@dataclass(frozen=True, slots=True)
class ReadBudget:
    """Limits for returned tensors, not Python metadata or file-verification working memory.

    A residue batch charges padded positions, including repeated observations. ``max_bytes``
    includes output masks and lengths. Training loaders must additionally budget queued batches,
    model activations, transient conversions and the immutable parts touched by a request.
    """

    max_rows: int = 64
    max_padded_residues: int = 32768
    max_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in (
            self.max_rows, self.max_padded_residues, self.max_bytes,
        )):
            raise ValueError("Reader limits must be positive integers.")

    def check(self, rows: int, *, padded_residues: int = 0, tensor_bytes: int = 0) -> None:
        if rows > self.max_rows or padded_residues > self.max_padded_residues or tensor_bytes > self.max_bytes:
            raise ValueError("Requested feature batch exceeds its declared read budget.")

    def queued_bytes(self, *, workers: int, prefetch_factor: int) -> int:
        """Conservative output-tensor capacity: worker queues plus producer and consumer batches."""
        if type(workers) is not int or workers < 0 or type(prefetch_factor) is not int or prefetch_factor < 1:
            raise ValueError("workers must be nonnegative and prefetch_factor positive.")
        return (workers * prefetch_factor + 2) * self.max_bytes


@dataclass(frozen=True, slots=True)
class ResidueBatch:
    """Rows padded to one length n: values (b, n, w) per branch, mask (b, n), lengths (b,).

    Legacy v1 branches have n = l residues. Canonical v2 branches keep the special tokens, so n = l + 2:
    row 0 CLS, rows 1..l_i residues, row l_i + 1 EOS, and `lengths` counts those rows. ``indices`` carries
    int32 (b, n, k) codes only for sparse branches. ``starts`` contains original zero-based residue
    crop starts when explicit crops were requested; otherwise all start at zero and it is None.
    Padding is zero in both values and indices and false in the mask; CLS and EOS are true.
    """

    values: Mapping[str, Tensor]  # (b, n, w) per branch; w = k for a sparse branch
    mask: Tensor  # (b, n) bool
    lengths: Tensor  # (b,) int64, rows per sequence (l_i + 2 for v2)
    indices: Mapping[str, Tensor] = field(default_factory=dict)  # (b, n, k) int32 per sparse branch
    starts: Tensor | None = None  # (b,) int64


def _place_rows(destination: Tensor, stored: Tensor, start: int, stop: int, special: int) -> None:
    """Copy one sequence's crop `[start, stop)` of residues into its padded row block.

    A v1 block holds the residues alone. A v2 block holds CLS, the cropped residues, then EOS, so the
    crop keeps both special rows even though they sit outside the residue interval.
    """
    # destination: (n, w) padded block of one sequence; stored: (r_i + special, w) as stored on disk
    if not special:
        destination[: stop - start] = stored[start:stop]  # (stop - start, w)
        return
    kept = stop - start
    destination[0] = stored[0]  # CLS: row 0 of the stored rows
    destination[1 : 1 + kept] = stored[1 + start : 1 + stop]  # residues r_start..r_stop-1, shifted past CLS
    destination[1 + kept] = stored[-1]  # EOS: the last stored row, row r_i + 1


def _row_storage(descriptor: StoredFeature) -> tuple[int, int, bool]:
    """Stored columns, bytes per entry, and whether sequence length multiplies the row."""
    sparse = descriptor.layout in {"csr", "ragged_topk"}
    width = cast(int, descriptor.sparse_count) if descriptor.layout == "ragged_topk" else descriptor.width
    size = torch.empty((), dtype=descriptor.dtype).element_size() + (4 if sparse else 0)
    return width, size, descriptor.layout in {"ragged", "ragged_topk"}


def _crop_ranges(lengths: Sequence[int], crops: Sequence[tuple[int, int]] | None) -> tuple[tuple[int, int], ...]:
    """Explicit nonempty half-open intervals within each independently verified inference prefix."""
    if crops is None:
        return tuple((0, length) for length in lengths)
    if len(crops) != len(lengths):
        raise ValueError("Supply exactly one crop for each requested sequence.")
    for crop, length in zip(crops, lengths, strict=True):
        if (not isinstance(crop, tuple) or len(crop) != 2
                or any(type(value) is not int for value in crop)
                or not 0 <= crop[0] < crop[1] <= length):
            raise ValueError("Crops must be nonempty integer (start, stop) intervals within the inference prefix.")
    return tuple(crops)


class FeatureView:
    """A pickleable, read-only view whose only corpus-sized state is sequence metadata.

    ``features`` contains independently selected physical descriptors, including complete
    FeatureSpecs, widths and dtypes. Construction verifies every inventory row before fitting,
    which reads every row once. ``verify_rows=False`` instead qualifies from the index, the commit
    markers and the row-identity sidecars, verifies each part holding a requested row once (or
    accepts it from the receipt under ``receipt_directory``) and reads no row, so its cost does not
    grow with the rows' bytes; ``trust_receipts=False`` ignores recorded receipts and refreshes them.
    By default every read reopens the descriptor and rechecks committed bytes and row identities.
    ``persistent_reader=True`` uses native mapped readers after initial qualification. Each part
    is verified once per reader lifetime and is assumed immutable while open. Native reader
    pickling drops live handles, so spawned workers reverify the parts they touch. Workers never
    create or repair an index; repair with a writable store before constructing the view.
    """

    def __init__(
        self, root: str | Path, features: Mapping[str, StoredFeature],
        inventory: Mapping[str, SequenceKey], *, budget: ReadBudget | None = None,
        content_pins: Mapping[str, Mapping[str, str]] | None = None,
        persistent_reader: bool = False, verify_rows: bool = True,
        receipt_directory: str | Path | None = None, trust_receipts: bool = True,
    ) -> None:
        if not features or any(not isinstance(name, str) or not name for name in features):
            raise ValueError("A feature view requires named branches.")
        budget = ReadBudget() if budget is None else budget
        if not isinstance(budget, ReadBudget):
            raise TypeError("budget must be a ReadBudget.")
        if type(persistent_reader) is not bool:
            raise TypeError("persistent_reader must be Boolean.")
        self.root = str(Path(root).resolve())
        self.budget = budget
        self.persistent_reader = persistent_reader
        self._receipt_directory = None if receipt_directory is None else Path(receipt_directory)
        self._trust_receipts = trust_receipts
        self._readers: dict[str, FeatureReader] = {}
        self._checked_sequences: dict[str, set[str]] = {}
        self._finite_sequences: dict[str, set[str]] = {}
        self._closed = False
        self._inventory = dict(inventory)
        self._descriptors = {name: json.dumps(feature.payload(), sort_keys=True) for name, feature in features.items()}
        self._content_pins = None if content_pins is None else {name: dict(pins) for name, pins in content_pins.items()}
        if self._content_pins is not None and set(self._content_pins) != set(self._descriptors):
            raise ValueError("Content pins must cover exactly the view's feature branches.")
        versions = {key.normalization_version for key in self._inventory.values() if isinstance(key, SequenceKey)}
        if len(versions) > 1:
            raise ValueError(f"One view reads one kind of row; the inventory mixes normalizations {sorted(versions)}.")
        self._normalize = spelling_normalizer(next(iter(versions), PROTEIN_NORMALIZATION))
        for sequence, key in self._inventory.items():
            if (not isinstance(sequence, str) or not isinstance(key, SequenceKey)
                    or hashlib.sha256(sequence.encode("utf-8")).hexdigest() != key.sha256
                    or (text_is_unit_count(key.normalization_version) and len(sequence) != key.length)):
                raise ValueError("Row text does not match the independently supplied sequence inventory.")
        # verify_rows reads and checks every stored row once per branch, which grows with the corpus' bytes. Without it, the
        # cost is the index lookups, one pass over each part not yet in the receipt, and one row-identity sidecar per part.
        for name in self.names:
            started = time.perf_counter()
            action = "verifying" if verify_rows else "indexing"
            print(f"FeatureView: {action} {len(self._inventory):,} rows of {name!r}", file=sys.stderr, flush=True)
            if verify_rows:
                self._verify_every_row(name)
            else:
                self._qualify_from_index(name)
            print(
                f"FeatureView: {action[:-3]}ed {len(self._inventory):,} rows of {name!r} in {time.perf_counter() - started:.1f}s",
                file=sys.stderr,
                flush=True,
            )

    def _verify_every_row(self, name: str) -> None:
        """Read, identity-check and scan every inventory row of one branch, in budget-sized batches."""
        budget = self.budget
        store, feature = self._source(name)
        # Query even for an empty selection, so a missing or corrupt index cannot pass.
        len(store)
        if self.persistent_reader:
            cast("FeatureReader", store).verify(tuple(self._inventory), workers=VERIFY_WORKERS, progress=f"verify {name}")
        descriptor = store.spec
        width, per_entry, per_residue = _row_storage(descriptor)
        pending: list[str] = []
        pending_bytes, pending_residues = 0, 0
        for sequence, key in self._inventory.items():
            stored = feature.stored_rows(key)  # l, or l + 2 when CLS and EOS are kept
            length = stored if per_residue else 1
            row_residues = stored if per_residue else 0
            row_bytes = length * width * per_entry
            budget.check(1, padded_residues=row_residues, tensor_bytes=row_bytes)
            if pending and (len(pending) == budget.max_rows or pending_bytes + row_bytes > budget.max_bytes
                            or pending_residues + row_residues > budget.max_padded_residues):
                self.rows(name, pending)
                pending, pending_bytes, pending_residues = [], 0, 0
            pending.append(sequence)
            pending_bytes += row_bytes
            pending_residues += row_residues
        if pending:
            self.rows(name, pending)

    def _qualify_from_index(self, name: str) -> None:
        """Qualify one branch without reading a tensor row: keys by index lookup, parts by receipt, identities by sidecar.

        Every inventory row must have an index address that its commit marker agrees with, and a stored row count equal to
        the contract's (`l + 2` for a canonical branch). Each part holding a requested row is verified in full once, or
        accepted from the receipt, and each requested row's identity is checked against its part's sidecar, one sidecar parse
        per part. Values are not read, so non-finite values surface on the first read of a row, which scans it once.
        """
        from fastplms.features import FeatureReader

        budget = self.budget
        if self.persistent_reader:
            reader, feature = self._source(name)
            reader = cast("FeatureReader", reader)
        else:
            store, feature = self._open(name)
            reader = FeatureReader(store, receipt=self._receipt_file(name), trust_receipt=self._trust_receipts)
        try:
            len(reader)  # even for an empty selection, so a missing or corrupt index cannot pass
            width, per_entry, per_residue = _row_storage(reader.spec)
            sequences = tuple(self._inventory)
            addresses = reader.addresses(sequences)  # (n,) index lookups; a missing row raises KeyError
            in_part: dict[tuple[str, int], list[tuple[str, int]]] = {}
            for sequence, address in zip(sequences, addresses, strict=True):
                stored = feature.stored_rows(self._inventory[sequence])  # l, or l + 2 when CLS and EOS are kept
                if address.residues != (stored if per_residue else 0):
                    raise ValueError("Cached tensor residue count differs from its row identity.")
                budget.check(1, padded_residues=stored if per_residue else 0, tensor_bytes=(stored if per_residue else 1) * width * per_entry)
                in_part.setdefault((address.segment, address.part), []).append((sequence, address.residues))
            reader.verify(sequences, workers=VERIFY_WORKERS, progress=f"verify {name}")
            checked = self._checked_sequences.setdefault(name, set()) if self.persistent_reader else set()
            for members in in_part.values():
                texts = [sequence for sequence, _ in members]
                for (sequence, residues), record in zip(members, reader.row_metadata(texts), strict=True):
                    row = checked_row_identity(record, feature, self._inventory[sequence])
                    if per_residue and len(row.retained_positions) != residues:
                        raise ValueError("Cached tensor residue count differs from its row identity.")
                checked.update(texts)
        finally:
            if not self.persistent_reader:
                reader.close()

    def _receipt_file(self, name: str) -> Path | None:
        """Where branch `name` keeps its verification receipt, or None when this view keeps none."""
        if self._receipt_directory is None:
            return None
        return self._receipt_directory / f"{self.descriptor(name).key}.json"

    @property
    def sequences(self) -> tuple[str, ...]:
        return tuple(self._inventory)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._descriptors)

    def close(self) -> None:
        """Release mapped readers and index connections; further physical reads raise."""
        self._closed = True
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> FeatureView:
        if self._closed:
            raise ValueError("This feature view is closed.")
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def descriptor(self, name: str) -> StoredFeature:
        """Return a detached descriptor; caller mutation cannot change this view's contract."""
        from fastplms.features import StoredFeature

        return StoredFeature.from_payload(json.loads(self._descriptors[name]))

    def _open(self, name: str) -> tuple[FeatureStore, FeatureSpec]:
        from fastplms.features import FeatureStore

        if self._closed:
            raise ValueError("This feature view is closed.")
        expected = self.descriptor(name)
        feature = FeatureSpec(expected.descriptor)
        spec = feature.to_dict()
        layouts = {"dense_pooled": "dense", "sae_pooled": "csr", "dense_residue": "ragged",
                   "structural_residue": "ragged", "structural_pooled": "dense", "sae_residue": "ragged_topk"}
        if (expected.key != feature.key.name or expected.layout != layouts.get(spec["kind"])
                or expected.dtype != getattr(torch, spec["output_dtype"]) or expected.positions):
            raise ValueError("Unsupported or incompatible feature key, layout, dtype or positions.")
        parameters = spec["reducer"]["parameters"]
        if spec["kind"] == "sae_pooled":
            width = parameters.get("codebook_dim", 0) * len(spec["pooling"])
        elif spec["kind"] == "sae_residue":
            width = parameters.get("codebook_dim")
        elif spec["kind"] == "structural_pooled":
            width = parameters.get("output_width", 0) * len(spec["pooling"])
        else:
            width = parameters.get("output_width")
        if (spec["kind"] in {"sae_pooled", "sae_residue", "structural_residue", "structural_pooled"}
                and expected.width != width):
            raise ValueError("Stored width disagrees with the full feature contract.")
        if spec["kind"] == "sae_residue" and (
            type(parameters.get("codebook_dim")) is not int or type(parameters.get("k")) is not int
            or expected.sparse_count != parameters["k"]
        ):
            raise ValueError("Stored sparse count disagrees with the full feature contract.")
        store = FeatureStore.read_only(
            Path(self.root) / expected.key,
            content_pins=None if self._content_pins is None else self._content_pins[name],
        )
        if store.spec != expected:
            raise ValueError("Stored feature descriptor differs from the independently supplied contract.")
        return store, feature

    def _source(self, name: str) -> tuple[FeatureStore | FeatureReader, FeatureSpec]:
        """Use the existing native mapped reader only under the declared immutable-file policy."""
        if self._closed:
            raise ValueError("This feature view is closed.")
        if not self.persistent_reader:
            return self._open(name)
        reader = self._readers.get(name)
        if reader is None:
            from fastplms.features import FeatureReader

            store, feature = self._open(name)
            reader = FeatureReader(store, receipt=self._receipt_file(name), trust_receipt=self._trust_receipts)
            self._readers[name] = reader
            return reader, feature
        return reader, FeatureSpec(reader.spec.descriptor)

    def content_pins(self) -> dict[str, dict[str, str]]:
        """Freeze immutable files selected by this inventory, using the native store's verifier.

        Persist these pins with the expected descriptors in a separately pinned fit input.
        This records bytes after source qualification; it does not establish extraction origin.
        """
        pins_by_branch: dict[str, dict[str, str]] = {}
        sequences = self.sequences
        # Native pinning deduplicates parts across the complete selection. Chunking by the
        # output batch budget would rehash the same part for every chunk that touches it.
        for name in self.names:
            started = time.perf_counter()
            print(f"FeatureView: pinning the files of {len(sequences):,} rows of {name!r}", file=sys.stderr, flush=True)
            if self.persistent_reader:
                reader, _ = self._source(name)
                pins_by_branch[name] = cast("FeatureReader", reader).content_pins(sequences, workers=VERIFY_WORKERS)
            else:
                store, _ = self._open(name)
                pins_by_branch[name] = store.content_pins(sequences)
            print(
                f"FeatureView: pinned {len(pins_by_branch[name]):,} files of {name!r} in {time.perf_counter() - started:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        return pins_by_branch

    def _identities(
        self, name: str, store: FeatureStore | FeatureReader, feature: FeatureSpec, sequences: Sequence[str],
    ) -> None:
        if self.persistent_reader:
            checked = self._checked_sequences.setdefault(name, set())
            sequences = tuple(sequence for sequence in dict.fromkeys(sequences) if sequence not in checked)
            if not sequences:
                return
            records = cast("FeatureReader", store).row_metadata(sequences)
        else:
            records = cast("FeatureStore", store).row_metadata(sequences, verify_data=True)
        rows = tuple(checked_row_identity(record, feature, self._inventory[sequence])
                     for sequence, record in zip(sequences, records, strict=True))
        if store.spec.layout in {"ragged", "ragged_topk"} and store.residue_counts(sequences) != [len(row.retained_positions) for row in rows]:
            raise ValueError("Cached tensor residue count differs from its row identity.")
        if self.persistent_reader:
            checked.update(sequences)

    def _known(self, sequence: str) -> str:
        """The inventory's spelling of a requested row: its exact text, else the form its normalization names.

        A protein is keyed by the normalized text, so "acd" and "ACD " ask for the row of "ACD". A ligand or an annotation
        entry has no such forgiveness (`contracts.spelling_normalizer`): only the exact canonical text names its row.
        """
        if sequence in self._inventory:
            return sequence
        try:
            normalized = self._normalize(sequence)
        except ValueError:
            normalized = sequence
        if normalized not in self._inventory:
            raise KeyError("A requested sequence is outside this view's verified inventory.")
        return normalized

    def _requested(self, sequences: Sequence[str]) -> tuple[str, ...]:
        if self._closed:
            raise ValueError("This feature view is closed.")
        sequences = tuple(sequences)
        self.budget.check(len(sequences))
        return tuple(self._known(sequence) for sequence in sequences)

    def lengths(self, name: str, sequences: Sequence[str]) -> tuple[int, ...]:
        """Residues `l` of each inference prefix from the expected contract, never counting CLS or EOS."""
        feature = FeatureSpec(self.descriptor(name).descriptor)
        return tuple(feature.context_length(self._inventory[self._known(sequence)]) for sequence in sequences)

    def _stored_rows(self, name: str, sequences: Sequence[str]) -> tuple[int, ...]:
        """Rows each sequence stores: `l`, or `l + 2` for a branch that keeps CLS and EOS."""
        feature = FeatureSpec(self.descriptor(name).descriptor)
        return tuple(feature.stored_rows(self._inventory[sequence]) for sequence in sequences)

    def rows(self, name: str, sequences: Sequence[str]) -> list[Tensor] | list[SparseRow] | list[TopKRow]:
        """Native rows in request order; both sparse layouts stay sparse, including duplicates."""
        sequences = self._requested(sequences)
        descriptor = self.descriptor(name)
        width, per_entry, per_residue = _row_storage(descriptor)
        lengths = self._stored_rows(name, sequences) if per_residue else (1,) * len(sequences)
        # CSR has at most width distinct codes, each with one int32 index and one value.
        tensor_bytes = sum(lengths) * width * per_entry
        self.budget.check(len(sequences), tensor_bytes=tensor_bytes,
                          padded_residues=sum(lengths) if per_residue else 0)
        store, feature = self._source(name)
        self._identities(name, store, feature, sequences)
        if descriptor.layout == "csr":
            values = store.read_sparse(sequences)  # b SparseRows, indices and values (nnz_i,)
        elif descriptor.layout == "ragged_topk":
            values = store.read_topk(sequences)  # b TopKRows, indices and values (r_i, k)
        else:
            values = store.read(sequences)  # b tensors, (w,) dense or (r_i, w) ragged
        # A persistent view verifies each part once and treats it as immutable, and its opening reads every row, so a row's values are
        # scanned on first read only; the repeated scans were over half of the time to assemble a training batch.
        scanned = self._finite_sequences.setdefault(name, set()) if self.persistent_reader else set()
        for sequence, value in zip(sequences, values, strict=True):
            if sequence in scanned:
                continue
            tensor = value.values if descriptor.layout in {"csr", "ragged_topk"} else value
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("Feature values must be finite.")
            if descriptor.layout == "ragged_topk" and bool((tensor < 0).any()):
                raise ValueError("SAE residue magnitudes must be nonnegative.")
            if self.persistent_reader:
                scanned.add(sequence)
        return values  # b rows: (w,), (r_i, w), SparseRow (nnz_i,), or TopKRow (r_i, k)

    def pooled(self, sequences: Sequence[str], names: Sequence[str] | None = None) -> Tensor:
        """Join selected pooled branches as FP32 (b, w), w the sum of their widths, densifying only this batch."""
        sequences = self._requested(sequences)
        names = self.names if names is None else tuple(names)
        descriptors = [self.descriptor(name) for name in names]
        if not names or len(set(names)) != len(names) or any(feature.layout not in {"dense", "csr"} for feature in descriptors):
            raise ValueError("pooled requires one or more dense/CSR pooled branches.")
        width = sum(feature.width for feature in descriptors)  # w
        self.budget.check(len(sequences), tensor_bytes=len(sequences) * width * 4)
        joined = torch.zeros((len(sequences), width), dtype=torch.float32)  # (b, w)
        offset = 0
        # One descriptor was read for each name.
        for name, descriptor in zip(names, descriptors, strict=True):
            for row_index, value in enumerate(self.rows(name, sequences)):
                if descriptor.layout == "csr":
                    joined[row_index, offset + value.indices.to(torch.int64)] = value.values.float()  # (nnz_i,)
                else:
                    joined[row_index, offset:offset + descriptor.width] = value.float()  # (w_j,), this branch's width
            offset += descriptor.width
        return joined  # (b, w)

    def residues(
        self, sequences: Sequence[str], names: Sequence[str] | None = None, *,
        crops: Sequence[tuple[int, int]] | None = None, reserve_bytes: int = 0,
    ) -> ResidueBatch:
        """Crop each branch's residue rows at the same positions and pad them, keeping sparse codes at width k.

        Crops use original zero-based, half-open residue positions within the stored prefix.
        ``reserve_bytes`` budgets a consumer's additional output before allocation. Source rows
        are read in full under their own budget; a short crop does not bypass source limits.
        """
        sequences = self._requested(sequences)
        names = self.names if names is None else tuple(names)
        descriptors = [self.descriptor(name) for name in names]
        if not names or len(set(names)) != len(names) or any(feature.layout not in {"ragged", "ragged_topk"} for feature in descriptors):
            raise ValueError("residues requires one or more ragged residue branches.")
        if type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError("reserve_bytes must be a nonnegative integer.")
        residue_counts = self.lengths(names[0], sequences)  # (b,) l_i residues of each inference prefix
        if any(self.lengths(name, sequences) != residue_counts for name in names[1:]):
            raise ValueError("Residue branches have different inference prefixes.")
        kept = {FeatureSpec(feature.descriptor).keeps_special_tokens for feature in descriptors}
        if len(kept) != 1:
            raise ValueError("Residue branches must all keep the special tokens, or none.")
        special = SPECIAL_TOKEN_ROWS if kept.pop() else 0  # CLS and EOS rows around each crop
        ranges = _crop_ranges(residue_counts, crops)  # (b,) residue intervals [start, stop)
        row_counts = tuple(stop - start + special for start, stop in ranges)  # (b,) rows per sequence, n_i
        batch_size, padded_length = len(sequences), max(row_counts, default=0)  # n = max n_i
        padded_residues = batch_size * padded_length
        tensor_bytes = padded_residues * (1 + sum(
            _row_storage(feature)[0] * _row_storage(feature)[1] for feature in descriptors
        )) + batch_size * (8 if crops is None else 16) + reserve_bytes
        self.budget.check(batch_size, padded_residues=padded_residues, tensor_bytes=tensor_bytes)
        values: dict[str, Tensor] = {}
        indices: dict[str, Tensor] = {}
        # One descriptor was read for each name.
        for name, descriptor in zip(names, descriptors, strict=True):
            width = _row_storage(descriptor)[0]  # w: d, 256, or k
            padded = torch.zeros((batch_size, padded_length, width), dtype=descriptor.dtype)  # (b, n, w)
            sparse = descriptor.layout == "ragged_topk"
            if sparse:
                indices[name] = torch.zeros((batch_size, padded_length, width), dtype=torch.int32)  # (b, n, k)
            for row_index, value in enumerate(self.rows(name, sequences)):
                start, stop = ranges[row_index]
                if sparse:
                    _place_rows(indices[name][row_index], value.indices, start, stop, special)
                    value = value.values  # (r_i + special, k)
                _place_rows(padded[row_index], value, start, stop, special)
            values[name] = padded  # (b, n, w)
        length_tensor = torch.tensor(row_counts, dtype=torch.int64)  # (b,)
        mask = torch.arange(padded_length)[None, :] < length_tensor[:, None]  # (b, n), padding only
        starts = None if crops is None else torch.tensor([start for start, _ in ranges], dtype=torch.int64)  # (b,) if present
        return ResidueBatch(values, mask, length_tensor, indices, starts)


class LengthBucketBatches:
    """Bounded-window batching of observation indices, with deterministic epoch shuffling.

    Order entries may repeat; none are deduplicated or dropped. Only one window is sorted.
    The caller uses observation indices to restore evaluation order after bucketed execution.

    ``lengths`` are residues l, as `FeatureView.lengths` returns them. A canonical (special tokens kept)
    `ResidueBatch` pads to l + 2 rows, and ``max_padded_residues`` limits padded rows, so pass
    ``special_rows=SPECIAL_TOKEN_ROWS`` for such a view: each observation then costs l + 2. A residue-only (v1)
    view pads to l, so the default 0 is right for it.
    """

    def __init__(
        self, lengths: Sequence[int], *, max_rows: int, max_padded_residues: int,
        window_size: int = 1024, seed: int | None = None, special_rows: int = 0,
    ) -> None:
        if any(type(value) is not int or value < 1 for value in (max_rows, max_padded_residues, window_size)):
            raise ValueError("Batch limits and window_size must be positive integers.")
        if type(special_rows) is not int or special_rows < 0:
            raise ValueError("special_rows must be a nonnegative integer.")
        if window_size < max_rows or any(
            type(length) is not int or length < 1 or length + special_rows > max_padded_residues for length in lengths
        ):
            raise ValueError("Invalid sequence length or bucket window smaller than max_rows.")
        self.special_rows = special_rows
        self.lengths = tuple(lengths)  # (count,) residues l per observation; each pads to l + special_rows rows
        self.max_rows = max_rows
        self.max_padded_residues = max_padded_residues
        self.window_size = window_size
        self.seed = seed

    def batches(self, order: Sequence[int] | None = None, *, epoch: int = 0) -> Iterator[list[int]]:
        order = range(len(self.lengths)) if order is None else order
        if any(type(index) is not int or not 0 <= index < len(self.lengths) for index in order):
            raise ValueError("Observation order contains an invalid index.")
        rng = None if self.seed is None else random.Random(self.seed + epoch)
        for start in range(0, len(order), self.window_size):
            window = list(order[start:start + self.window_size])
            if rng is not None:
                rng.shuffle(window)
            window.sort(key=self.lengths.__getitem__)  # the shift by special_rows is the same for all, so the order holds
            batches: list[list[int]] = []
            batch: list[int] = []
            longest = 0
            for index in window:
                length = self.lengths[index] + self.special_rows  # padded rows of this observation
                if batch and (len(batch) == self.max_rows or (len(batch) + 1) * max(longest, length) > self.max_padded_residues):
                    batches.append(batch)
                    batch, longest = [], 0
                batch.append(index)
                longest = max(longest, length)
            if batch:
                batches.append(batch)
            if rng is not None:
                rng.shuffle(batches)
            yield from batches

    def __iter__(self) -> Iterator[list[int]]:
        return self.batches()
