"""A procedural declared release and its ordinary caches, written block by block for whole-path capacity work.

Every row is a pure function of its global index and the fixture's `Dimensions`, so neither the release nor the
feature stores are ever held whole, and any block can be regenerated. The proteins, targets, and features are
artificial: they establish capacity and numerical agreement only, never a homology-qualified split or a scientific
result.

Shapes: `b` rows in a block, `c` labels, `k` true labels per row, `o` observed labels per row, `w` a feature width.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from foundry.digests import sha256_file
from protify.data.declared_dataset import SCHEMA, SPARSE_ENCODING, SPARSE_SCHEMA, SPLIT_NAMES, declared_target_fields
from protify.feature_cache import embedding_feature, embedding_layout


AMINO_ACIDS = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", dtype=np.uint8)
SEQUENCE_LENGTH = 13
# Rows of one declared cluster. Every split boundary is a multiple, so no cluster crosses a split.
CLUSTER_ROWS = 4
# Rows generated, and written as one Parquet row group or one feature part, at a time.
BLOCK_ROWS = 2048
# The tensor bytes one stored feature part may hold: the suite's feature execution bounds its parts the same way.
PART_BYTES = 256 * 1024**2
DATASET_ID = "capacity_fixture"


@dataclass(frozen=True, slots=True)
class Dimensions:
    """What one fixture holds: its splits, its vocabulary, its truth density, and its feature widths."""

    train_rows: int
    scored_rows: int  # each of validation, test, and calibration
    labels: int
    positives: int  # true labels of every row
    observed_fraction: float  # the share of a training row's labels whose truth is observed; 1.0 observes them all
    dense_width: int  # the final state's mean and variance, twice the hidden size
    sae_width: int  # the sparse autoencoder's codebook width
    sae_nonzeros: int  # codes active in a max-pooled protein, on average
    seed: int = 0
    # The same share for the validation, test, and calibration rows; None gives them the training rows' share.
    scored_observed_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.train_rows < CLUSTER_ROWS or self.train_rows % CLUSTER_ROWS or self.scored_rows < CLUSTER_ROWS or self.scored_rows % CLUSTER_ROWS:
            raise ValueError(f"every split needs a positive multiple of {CLUSTER_ROWS} rows")
        if not (0 < self.observed_fraction <= 1 and 0 < self.observed_share(scored=True) <= 1):
            raise ValueError("every observed fraction must be in (0, 1]")
        if self.labels < 2 or not all(1 <= self.positives <= self.observed_per_row(scored) for scored in (False, True)):
            raise ValueError("every row needs at least one true label among its observed labels, in a vocabulary of at least two")
        if self.dense_width < 2 or self.dense_width % 2 or not 1 <= self.sae_nonzeros <= self.sae_width:
            raise ValueError("the dense width is a positive even number, and a protein activates between one code and every code")
        if self.train_rows + 3 * self.scored_rows > 2 * 10**9:
            raise ValueError("the row space of the fixture's sequences is exhausted")

    @property
    def total_rows(self) -> int:
        return self.train_rows + 3 * self.scored_rows

    def observed_share(self, *, scored: bool) -> float:
        return self.observed_fraction if self.scored_observed_fraction is None or not scored else self.scored_observed_fraction

    def observed_per_row(self, scored: bool) -> int:
        """The labels of a training row, or of a validation, test, or calibration row, whose truth is observed."""
        share = self.observed_share(scored=scored)
        if share == 1:
            return self.labels
        return min(self.labels, max(self.positives, round(share * self.labels)))

    def default_observed(self, scored: bool) -> bool:
        """The polarity `SparseLabelRow` requires: the smaller of the observed and unknown sets is stored, False on a tie."""
        return self.observed_per_row(scored) > self.labels // 2

    def label_entries(self) -> int:
        """The index entries the release stores across all four splits: each row's true labels and its stored mask columns."""
        entries = 0
        for scored, rows in ((False, self.train_rows), (True, 3 * self.scored_rows)):
            observed = self.observed_per_row(scored)
            stored = 0 if observed == self.labels else self.labels - observed if self.default_observed(scored) else observed
            entries += rows * (self.positives + stored)
        return entries

    def scored_observed_targets(self) -> int:
        """The observed (row, label) truths of one scored split, which the report ranks."""
        return self.scored_rows * self.observed_per_row(True)

    def splits(self) -> dict[str, tuple[int, int]]:
        """Each split's half-open range of global row indices, in the release's split order."""
        counts = (self.train_rows, self.scored_rows, self.scored_rows, self.scored_rows)
        ranges, start = {}, 0
        for name, count in zip(SPLIT_NAMES, counts, strict=True):
            ranges[name] = (start, start + count)
            start += count
        return ranges

    def record(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LabelBlock:
    """The truth of `b` rows: sorted true labels, and the sorted observed labels unless every label is observed."""

    positives: np.ndarray  # (b, k) int64
    observed: np.ndarray | None  # (b, o) int64
    unknown: np.ndarray | None  # (b, c - o) int64
    default_observed: bool  # which of the two sets the release stores, by `Dimensions.default_observed`


def blocks(first: int, last: int, size: int = BLOCK_ROWS) -> Iterator[tuple[int, int]]:
    for start in range(first, last, size):
        yield start, min(start + size, last)


def sequences_of(first: int, last: int) -> list[bytes]:
    """One unique 13-residue protein per index: `M`, then the index in base 20 over twelve places."""
    ordinal = np.arange(first, last, dtype=np.int64)  # (b,)
    letters = np.empty((last - first, SEQUENCE_LENGTH), dtype=np.uint8)  # (b, 13)
    letters[:, 0] = ord("M")
    for place in range(1, SEQUENCE_LENGTH):
        letters[:, place] = AMINO_ACIDS[ordinal % len(AMINO_ACIDS)]
        ordinal //= len(AMINO_ACIDS)
    return letters.view(f"S{SEQUENCE_LENGTH}").ravel().tolist()


def label_block(dims: Dimensions, first: int, last: int) -> LabelBlock:
    """The truth of rows `first` to `last`, which lie in one split, drawn from that block's own generator."""
    scored = first >= dims.train_rows
    if not scored and last > dims.train_rows:
        raise ValueError("a block of labels lies within one split")
    observed_per_row = dims.observed_per_row(scored)
    default_observed = dims.default_observed(scored)
    rng = np.random.default_rng([dims.seed, 1, first])
    keys = rng.random((last - first, dims.labels), dtype=np.float32)  # (b, c)
    if observed_per_row == dims.labels:
        chosen = np.argpartition(keys, dims.positives - 1, axis=1)[:, :dims.positives] if dims.positives < dims.labels else np.argsort(keys, axis=1)
        return LabelBlock(np.sort(chosen, axis=1), None, None, default_observed)
    split = np.argpartition(keys, observed_per_row - 1, axis=1)  # (b, c), the o smallest keys first
    observed, unknown = split[:, :observed_per_row], split[:, observed_per_row:]
    if dims.positives < observed_per_row:
        within = np.argpartition(np.take_along_axis(keys, observed, axis=1), dims.positives - 1, axis=1)[:, :dims.positives]
        positives = np.take_along_axis(observed, within, axis=1)
    else:
        positives = observed
    return LabelBlock(np.sort(positives, axis=1), np.sort(observed, axis=1), np.sort(unknown, axis=1), default_observed)


def _rows_as_list(rows: np.ndarray, value_type: pa.DataType) -> pa.ListArray:
    """A list column whose every row is one row of the `(b, k)` matrix; `k` may be zero."""
    # rows: (b, k)
    count, width = rows.shape
    offsets = pa.array(np.arange(count + 1, dtype=np.int64) * width, type=pa.int32())  # (b + 1,)
    return pa.ListArray.from_arrays(offsets, pa.array(rows.reshape(-1), type=value_type))


def compact_targets(dims: Dimensions, block: LabelBlock) -> tuple[pa.StructArray, pa.StructArray]:
    """The block's `sparse_observed_v1` labels and observation masks, without a width-sized row."""
    label_type, mask_type = (field.type for field in declared_target_fields("multilabel", compact=True))
    count = block.positives.shape[0]
    labels = pa.StructArray.from_arrays(
        [pa.array(np.full(count, dims.labels, dtype=np.int32)), _rows_as_list(block.positives, pa.int64()),
         _rows_as_list(np.ones(block.positives.shape, dtype=np.float32), pa.float32())],
        fields=list(label_type),
    )
    if block.observed is None:
        stored = np.empty((count, 0), dtype=np.int64)
    else:
        stored = block.unknown if block.default_observed else block.observed
    mask = pa.StructArray.from_arrays(
        [pa.array(np.full(count, block.default_observed or block.observed is None, dtype=np.bool_)), _rows_as_list(stored, pa.int64())],
        fields=list(mask_type),
    )
    return labels, mask


def dense_targets(dims: Dimensions, block: LabelBlock) -> tuple[pa.ListArray, pa.ListArray]:
    """The same truth as `version 1` dense rows: `(b, c)` float64 targets with zero fillers, and boolean masks."""
    count = block.positives.shape[0]
    targets = np.zeros((count, dims.labels), dtype=np.float64)  # (b, c)
    rows = np.arange(count)[:, None]
    targets[rows, block.positives] = 1.0
    masks = np.ones((count, dims.labels), dtype=np.bool_)  # (b, c)
    if block.observed is not None:
        masks[rows, block.unknown] = False
    return _rows_as_list(targets, pa.float64()), _rows_as_list(masks, pa.bool_())


def release_schema(*, compact: bool) -> pa.Schema:
    return pa.schema([
        *(pa.field(name, pa.string(), nullable=False) for name in ("observation_id", "protein_id", "cluster_id", "seqs")),
        *declared_target_fields("multilabel", compact=compact),
    ])


def write_release(root: Path, dims: Dimensions, *, compact: bool = True) -> tuple[Path, str]:
    """Write the four Parquet splits and their manifest under `root`; returns the manifest and its digest.

    `compact=False` writes the same truth as a `version 1` dense release, which only a small vocabulary can afford.
    """
    if not compact and dims.total_rows * dims.labels > 5 * 10**7:
        raise ValueError("a dense release holds every label of every row; ask for one only at small dimensions")
    root.mkdir(parents=True, exist_ok=True)
    schema = release_schema(compact=compact)
    manifest = {
        "schema": SPARSE_SCHEMA if compact else SCHEMA, "dataset_id": DATASET_ID, "endpoint": "artificial_observed_truth",
        "task_type": "multilabel", "units": "binary", "target_transform": {"name": "identity", "source_units": "binary", "inverse": "identity"},
        "label_vocabulary": [f"label_{index}" for index in range(dims.labels)],
        "evidence_policy_sha256": "e" * 64, "split_audit_sha256": "a" * 64, "splits": {},
    }
    if compact:
        manifest["label_encoding"] = SPARSE_ENCODING
    for name, (first, last) in dims.splits().items():
        path = root / f"{name}.parquet"
        with pq.ParquetWriter(path, schema, compression="zstd") as writer:
            for start, stop in blocks(first, last):
                raw = sequences_of(start, stop)
                block = label_block(dims, start, stop)
                labels, mask = compact_targets(dims, block) if compact else dense_targets(dims, block)
                columns = [
                    pa.array([f"assay_{index}" for index in range(start, stop)], type=pa.string()),
                    pa.array([hashlib.sha256(sequence).hexdigest() for sequence in raw], type=pa.string()),
                    pa.array([f"{name}_cluster_{(index - first) // CLUSTER_ROWS}" for index in range(start, stop)], type=pa.string()),
                    pa.array([sequence.decode("ascii") for sequence in raw], type=pa.string()),
                    labels, mask,
                ]
                writer.write_table(pa.Table.from_arrays(columns, schema=schema))
        manifest["splits"][name] = {"path": path.name, "rows": last - first, "sha256": sha256_file(path)}
    path = root / "dataset.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path, sha256_file(path)


def dense_features(dims: Dimensions, first: int, last: int) -> torch.Tensor:
    """The final state's mean then variance for a block: `(b, w)` bfloat16, mean unit normal and variance positive."""
    rng = np.random.default_rng([dims.seed, 2, first])
    values = rng.standard_normal((last - first, dims.dense_width), dtype=np.float32)  # (b, w)
    half = dims.dense_width // 2
    values[:, half:] = np.abs(values[:, half:]) * 0.5 + 0.1
    return torch.from_numpy(values).to(torch.bfloat16)  # (b, w) bfloat16


def sae_features(dims: Dimensions, first: int, last: int) -> torch.Tensor:
    """A max-pooled codebook row for each protein: `(b, w)` bfloat16, positive on about `sae_nonzeros` random codes."""
    rng = np.random.default_rng([dims.seed, 3, first])
    active = rng.random((last - first, dims.sae_width), dtype=np.float32) < dims.sae_nonzeros / dims.sae_width  # (b, w)
    codes = np.zeros(active.shape, dtype=np.float32)  # (b, w)
    codes[active] = rng.random(int(active.sum()), dtype=np.float32) * 3.0 + 0.05
    return torch.from_numpy(codes).to(torch.bfloat16)  # (b, w) bfloat16


def write_feature_stores(dims: Dimensions, descriptors: Mapping[str, Mapping[str, Any]], root: Path) -> dict[str, dict[str, Any]]:
    """Write the `sae` and `dense` stores of every row of the fixture under `root`.

    `descriptors` holds the two descriptors as the run's own arguments name them, so the stores are the ones the
    run reads. Each store is one segment of parts of `BLOCK_ROWS` rows, committed once, as the suite's own embedding
    writes a store: `CacheWriter` commits a window at a time and every commit verifies every earlier window, which
    costs the square of the row count (0.2 s per earlier window of 2,048 rows, so hours at a million rows). The stores
    are complete only when this returns. Each store's directory, layout, row count, and nonzero count come back, the
    nonzeros being what `sae_nonzeros` averages.
    """
    from fastplms.features import CSR, FeatureStore, SparseRow

    generators = {"sae": sae_features, "dense": dense_features}
    widths = {"sae": dims.sae_width, "dense": dims.dense_width}
    if set(descriptors) != set(generators):
        raise ValueError("descriptors must name exactly the sae and dense stores")
    root.mkdir(parents=True, exist_ok=True)
    layouts = {name: embedding_layout(descriptor) for name, descriptor in descriptors.items()}
    stores = {name: FeatureStore.open(root, embedding_feature(descriptors[name], layout=layouts[name], width=widths[name])) for name in generators}
    nonzeros = dict.fromkeys(generators, 0)
    metadata = {"writer": "protify.testing_suite.procedural_release_fixture", "seed": dims.seed}
    with ExitStack() as segments:
        writers = {name: segments.enter_context(store.segment(f"procedural-{name}", metadata)) for name, store in stores.items()}
        for start, stop in blocks(0, dims.total_rows):
            names = [sequence.decode("ascii") for sequence in sequences_of(start, stop)]
            for name, writer in writers.items():
                block = generators[name](dims, start, stop)  # (b, w)
                nonzeros[name] += int(torch.count_nonzero(block))
                values = block.to(stores[name].spec.dtype)  # (b, w), widened when the store is wider than the generator
                rows = [SparseRow.from_dense(row) for row in values] if layouts[name] == CSR else list(values)
                writer.append_bounded(names, rows, max_tensor_bytes=PART_BYTES)
    return {
        name: {"directory": str(stores[name].directory), "layout": layouts[name], "rows": dims.total_rows,
               "width": widths[name], "nonzeros": nonzeros[name]}
        for name in generators
    }
