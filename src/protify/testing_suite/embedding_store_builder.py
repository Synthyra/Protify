"""Build embedding stores for tests: what an embedding run leaves in `--embedding_save_dir`.

A consumer test used to write a SQLite or `.pth` cache by hand. Now the cache is a feature store, and a
store is only found when its descriptor is the run's own, so these builders take either the descriptor's
facts or the run's `EmbeddingArguments` and write through `feature_cache.CacheWriter`, the writer every
Protify run uses. The old formats, for the tests that pin their conversion, are written by
`legacy_cache_writer`.
"""

from __future__ import annotations

import shutil

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from torch import Tensor

from fastplms.features import FeatureStore


if TYPE_CHECKING:
    from protify.embedder import EmbeddingArguments

try:
    from src.protify import feature_cache
except ImportError:
    from protify import feature_cache


def cache_descriptor(
    model_name: str = "ESM2-8",
    *,
    matrix: bool = False,
    pooling: Sequence[str] = ("mean",),
    hidden_state_index: int = -1,
    max_length: int = 2048,
    dtype: str = "float32",
) -> dict[str, Any]:
    """The descriptor of one cached representation, from the facts that name it."""
    return feature_cache.embedding_descriptor(
        model_name,
        matrix_embed=matrix,
        pooling_types=list(pooling),
        hidden_state_index=hidden_state_index,
        max_length=max_length,
        dtype=dtype,
    )


def write_store(root: str | Path, descriptor: Mapping[str, Any], rows: Mapping[str, Tensor], *, window_rows: int = 2048) -> Path:
    """Commit `rows` (sequence to tensor) as the store of `descriptor` under `root`; returns its directory."""
    # rows: (d,) pooled or (l + 2, d) token rows by sequence
    with feature_cache.CacheWriter(
        root, descriptor, layout=feature_cache.embedding_layout(descriptor), window_rows=window_rows,
    ) as writer:
        writer.add(list(rows), list(rows.values()))
    return writer.directory


def build_store(
    root: str | Path,
    rows: Mapping[str, Tensor],
    model_name: str = "ESM2-8",
    *,
    matrix: bool = False,
    pooling: Sequence[str] = ("mean",),
    hidden_state_index: int = -1,
    max_length: int = 2048,
    dtype: str = "float32",
) -> Path:
    """The store of a representation named by its facts, holding `rows`."""
    # rows: (d,) pooled or (l + 2, d) token rows by sequence
    descriptor = cache_descriptor(
        model_name, matrix=matrix, pooling=pooling, hidden_state_index=hidden_state_index,
        max_length=max_length, dtype=dtype,
    )
    return write_store(root, descriptor, rows)


def build_store_for(arguments: EmbeddingArguments, model_name: str, rows: Mapping[str, Tensor], *, matrix: bool | None = None) -> Path:
    """The store an `EmbeddingArguments` run reads for `model_name`, holding `rows`.

    The descriptor is the run's own, so the code under test finds the store the way production does.
    `arguments.embedding_save_dir` is the root.
    """
    # rows: (d,) pooled or (l + 2, d) token rows by sequence
    descriptor = arguments.cache_descriptor(model_name, matrix)
    return write_store(arguments.embedding_save_dir, descriptor, rows)


def without_rows(directory: str | Path, sequences: Iterable[str], candidates: Iterable[str]) -> Path:
    """Rebuild the store at `directory` without `sequences`; every other row is kept bit for bit.

    A store has no delete, so this reads the rows of `candidates` that the store holds, other than
    `sequences`, and writes them again into a fresh store of the same descriptor, layout, and dtype.
    `candidates` must name every sequence the store holds, since a store cannot list its own.
    """
    directory = Path(directory)
    spec = FeatureStore.read_only(directory).spec
    dropped = set(sequences)
    held = set(candidates) - set(feature_cache.missing_sequences(directory, candidates))
    kept = [sequence for sequence in dict.fromkeys(candidates) if sequence in held and sequence not in dropped]
    rows = feature_cache.read_embeddings(directory, kept)
    shutil.rmtree(directory)
    with feature_cache.CacheWriter(directory.parent, spec.descriptor, layout=spec.layout) as writer:
        writer.add(kept, [rows[sequence] for sequence in kept])
    return writer.directory
