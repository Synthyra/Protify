"""Read-only decoders of Protify's original embedding caches, and their conversion into the feature store.

Before the workspace kept embeddings in one feature store (`fastplms.features`), Protify wrote each
cache in a format of its own, named by `embedder.get_embedding_filename`:

* SQLite, `<name>.db`: a table `embeddings(sequence text primary key, embedding blob)` whose blobs
  use Protify's compact codec. Version 0x01 holds a dense array (header: version, dtype code, ndim,
  shape). Version 0x02 holds a coordinate list (flat indices, then values), which SAE models used to
  save space. Dtype code 0 is float16, 1 is bfloat16 stored as float16 bytes (the first encoding, read
  but never written since code 3), 2 is float32, and 3 is a bfloat16 bit pattern. Older blobs are
  `torch.save` bytes, or raw float32 whose shape the reader supplies.
* PyTorch, `<name>.pth`: a pickled dictionary from sequence to tensor.

Nothing in this module writes either format. `convert_legacy_cache` moves one file into a feature store
with `fastplms.features.convert_rows`, which decodes the file twice and proves that every stored row
equals the decoded one bit for bit. The old file is never modified or deleted. From a shell:

    python -m protify.legacy_cache embeddings/ESMC-600_False_len2048_mean_var.db

Shapes: `d` is an embedding width, `l` a stored token count including the leading and trailing
special tokens, `n` the number of nonzero coordinates of a sparse blob.
"""

from __future__ import annotations

import argparse
import io
import re
import sqlite3
import struct
import sys
import numpy as np
import torch

from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from torch import Tensor


if TYPE_CHECKING:
    from fastplms.features import ConversionReceipt, SparseRow


# Compact blob serialization constants.
_COMPACT_VERSION = 0x01
# Code 1 is the original bfloat16 encoding, which converted to float16 bytes. bfloat16 carries a wider
# exponent than float16, so that conversion silently flushed values below the float16 normal range:
# 1.0e-6 came back as 1.01e-6. Code 3 stores the bfloat16 bit pattern instead and is exact. Code 1 is
# still decoded so that caches written before code 3 existed keep converting.
_LEGACY_BFLOAT16_CODE = 1
_CODE_TO_DTYPE = {0: torch.float16, 1: torch.bfloat16, 2: torch.float32, 3: torch.bfloat16}
_CODE_TO_NP_DTYPE = {0: np.float16, 1: np.float16, 2: np.float32, 3: np.int16}

# Coordinate-list blob format, selected by the model that produced the embedding (SAE models whose
# active fraction k / codebook_dim is small).
_SPARSE_VERSION = 0x02
_INDEX_CODE_TO_NP = {0: np.uint16, 1: np.uint32}

KINDS = {".db": "sqlite", ".pth": "torch", ".pt": "torch"}
FORMATS = {"sqlite": "protify_sqlite_blobs", "torch": "protify_torch_dict"}
DECODER = "protify.legacy_cache"


def torch_load(source: Any) -> Any:
    """`torch.load` onto the CPU, refusing pickled objects other than tensors and plain containers."""
    return torch.load(source, map_location="cpu", weights_only=True)


def _restore_dtype(t: Tensor, dtype_code: int) -> Tensor:
    """Recover the stored dtype from the raw array a blob decodes to."""
    # t: (*shape), the raw array the blob decodes to; every return keeps that shape.
    if dtype_code == _LEGACY_BFLOAT16_CODE:
        return t.to(torch.bfloat16)  # (*shape), float16 values written before code 3 existed
    if _CODE_TO_DTYPE[dtype_code] == torch.bfloat16:
        return t.view(torch.bfloat16)  # (*shape), raw bit pattern

    return t  # (*shape)


def blob_dtype(blob: bytes) -> torch.dtype | None:
    """The dtype a compact blob decodes to, read from its header alone, or None for any other blob."""
    if len(blob) >= 2 and blob[0] in (_COMPACT_VERSION, _SPARSE_VERSION) and blob[1] in _CODE_TO_DTYPE:
        return _CODE_TO_DTYPE[blob[1]]
    return None


def embedding_blob_to_tensor(
    blob: bytes,
    fallback_shape: tuple[int, ...] | None = None,
) -> Tensor:
    """Deserialize an embedding blob from SQLite.

    Tries compact binary format first (version byte 0x01), then the coordinate-list
    format (0x02), then PyTorch torch.save format, then legacy raw float32 with
    fallback_shape. Both compact formats return a dense tensor.
    """
    if len(blob) >= 3 and blob[0] == _SPARSE_VERSION and blob[1] in _CODE_TO_DTYPE:
        dtype_code = blob[1]
        index_code = blob[2]
        ndim = struct.unpack_from('<i', blob, 3)[0]
        shape = struct.unpack_from(f'<{ndim}i', blob, 7)
        nnz = struct.unpack_from('<i', blob, 7 + 4 * ndim)[0]
        np_dtype = _CODE_TO_NP_DTYPE[dtype_code]
        np_index_dtype = _INDEX_CODE_TO_NP[index_code]

        index_offset = 11 + 4 * ndim
        indices = np.frombuffer(blob, dtype=np_index_dtype, count=nnz, offset=index_offset)  # (n,)
        value_offset = index_offset + nnz * np_index_dtype().itemsize
        values = np.frombuffer(blob, dtype=np_dtype, count=nnz, offset=value_offset)  # (n,)

        dense = np.zeros(int(np.prod(shape)) if shape else 1, dtype=np_dtype)  # (numel,)
        dense[indices] = values
        return _restore_dtype(torch.from_numpy(dense.reshape(shape)), dtype_code)  # (*shape) as written

    if len(blob) >= 2 and blob[0] == _COMPACT_VERSION and blob[1] in _CODE_TO_DTYPE:
        dtype_code = blob[1]
        ndim = struct.unpack_from('<i', blob, 2)[0]
        shape = struct.unpack_from(f'<{ndim}i', blob, 6)
        data_offset = 6 + 4 * ndim
        np_dtype = _CODE_TO_NP_DTYPE[dtype_code]
        arr = np.frombuffer(blob, dtype=np_dtype, offset=data_offset).reshape(shape).copy()
        return _restore_dtype(torch.from_numpy(arr), dtype_code)  # (*shape) as written

    try:
        t = torch_load(io.BytesIO(blob))
        if isinstance(t, Tensor):
            return t  # (*shape) as saved
    except Exception:  # noqa: broad-except silent-except arbitrary bytes make torch's unpickler raise UnpicklingError, IndexError, EOFError, UnicodeDecodeError, KeyError or struct.error; any of them means the blob is raw float32
        pass
    if fallback_shape is not None:
        return torch.tensor(  # fallback_shape: (d,) pooled or (n, d)
            np.frombuffer(blob, dtype=np.float32).reshape(fallback_shape)
        )
    raise ValueError(
        "Blob is not in compact/PyTorch format and no fallback_shape provided for legacy float32."
    )


def cache_kind(path: str | Path) -> str:
    """Which old format a file holds, by its extension: `sqlite` or `torch`."""
    suffix = Path(path).suffix
    if suffix not in KINDS:
        raise ValueError(f"{path} is not an old Protify cache: expected one of {sorted(KINDS)}.")
    return KINDS[suffix]


def _read_only_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve(strict=True).as_uri() + "?mode=ro", uri=True)


def _require_embeddings_table(connection: sqlite3.Connection) -> None:
    """Refuse a database that lacks Protify's `embeddings(sequence, embedding)` table."""
    table = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
    ).fetchone()
    if table is None:
        raise ValueError("SQLite input must use Protify's embeddings(sequence, embedding) schema.")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(embeddings)")}
    if not {"sequence", "embedding"}.issubset(columns):
        raise ValueError("SQLite table 'embeddings' must contain sequence and embedding columns.")


def sqlite_rows(
    path: str | Path, *, fallback_shape: Callable[[str], tuple[int, ...]] | None = None,
) -> Iterator[tuple[str, Tensor]]:
    """Every `(sequence, tensor)` of a SQLite cache in insertion order, opened read-only, as it was stored."""
    with closing(_read_only_connection(Path(path))) as connection:
        _require_embeddings_table(connection)
        for sequence, blob in connection.execute("SELECT sequence, embedding FROM embeddings ORDER BY rowid"):
            shape = None if fallback_shape is None else fallback_shape(sequence)
            yield sequence, embedding_blob_to_tensor(blob, fallback_shape=shape)  # (*s)


def torch_rows(path: str | Path) -> Iterator[tuple[str, Tensor]]:
    """Every `(sequence, tensor)` of a pickled cache dictionary, loaded once on CPU."""
    values = torch_load(Path(path).resolve(strict=True))
    if not isinstance(values, dict):
        raise ValueError("Torch embedding cache must be a sequence-to-tensor dictionary")
    for sequence, row in values.items():
        if not isinstance(row, Tensor):
            raise ValueError("Torch embedding cache contains a non-tensor row")
        yield sequence, row  # (*s)


@contextmanager
def cached_rows(
    path: Path, *, kind: str, fallback_shape: Callable[[str], tuple[int, ...]] | None = None,
) -> Iterator[Callable[[str], Tensor]]:
    """Yield a raw lookup. SQLite is lazy; a Torch dictionary loads once on CPU.

    The caller owns input integrity and memory admission. Shapes and dtypes are exactly
    those stored; this layer neither pools residue matrices nor widens BF16. Raw legacy FP32
    blobs need an explicit caller-supplied shape; without one they remain undecodable.
    """
    if kind not in {"sqlite", "torch"}:
        raise ValueError("cache kind must be sqlite or torch")
    path = path.resolve(strict=True)
    if kind == "sqlite":
        with closing(_read_only_connection(path)) as connection:
            def lookup(sequence: str) -> Tensor:
                rows = connection.execute(
                    "SELECT embedding FROM embeddings WHERE sequence = ?", (sequence,),
                ).fetchmany(2)
                if not rows:
                    raise KeyError("embedding cache has no row for the requested sequence")
                if len(rows) != 1:
                    raise ValueError("embedding cache repeats a requested sequence")
                shape = None if fallback_shape is None else fallback_shape(sequence)
                return embedding_blob_to_tensor(rows[0][0], fallback_shape=shape)  # Stored shape (*s).

            yield lookup
        return

    values = torch_load(path)
    if not isinstance(values, dict):
        raise ValueError("Torch embedding cache must be a sequence-to-tensor dictionary")

    def lookup(sequence: str) -> Tensor:
        if values is None:
            raise RuntimeError("embedding cache reader is closed")
        row = values[sequence]  # Stored shape (*s).
        if not isinstance(row, Tensor):
            raise ValueError("Torch embedding cache contains a non-tensor row")
        return row  # (*s)

    try:
        yield lookup
    finally:
        values = None


def cache_sequences(path: str | Path, kind: str) -> list[str]:
    """Every sequence an old cache holds, without decoding any embedding of a SQLite cache."""
    if kind == "sqlite":
        with closing(_read_only_connection(Path(path))) as connection:
            _require_embeddings_table(connection)
            return [sequence for (sequence,) in connection.execute("SELECT sequence FROM embeddings")]
    return [sequence for sequence, _ in torch_rows(path)]


@dataclass(frozen=True)
class CacheScan:
    """What one pass over an old cache found: how many rows, which dtypes, and the first row's shape."""

    rows: int
    dtypes: frozenset[torch.dtype]
    first_shape: tuple[int, ...]


def _fallback_shape(matrix: bool) -> Callable[[str], tuple[int, ...]]:
    """The shape a raw float32 blob without a header takes: a pooled row, or one row per token."""
    return (lambda sequence: (len(sequence), -1)) if matrix else (lambda sequence: (1, -1))


def scan_cache(
    path: str | Path, kind: str, fallback_shape: Callable[[str], tuple[int, ...]] | None = None,
) -> CacheScan:
    """Count an old cache's rows and collect the dtypes they were stored in.

    A cache written across runs can mix dtypes, since a run replaced rows at its own `embed_dtype`, so
    the store's dtype comes from every row, not the first. Compact blobs give their dtype from a
    two-byte header; other blobs are decoded.
    """
    dtypes: set[torch.dtype] = set()
    first_shape: tuple[int, ...] | None = None
    rows = 0
    if kind == "sqlite":
        with closing(_read_only_connection(Path(path))) as connection:
            _require_embeddings_table(connection)
            for sequence, blob in connection.execute("SELECT sequence, embedding FROM embeddings"):
                rows += 1
                dtype = blob_dtype(blob)
                if dtype is None or first_shape is None:
                    shape = None if fallback_shape is None else fallback_shape(sequence)
                    row = embedding_blob_to_tensor(blob, fallback_shape=shape)
                    dtype = row.dtype
                    if first_shape is None:
                        first_shape = tuple(row.shape)
                dtypes.add(dtype)
    else:
        for _, row in torch_rows(path):
            rows += 1
            dtypes.add(row.dtype)
            if first_shape is None:
                first_shape = tuple(row.shape)
    return CacheScan(rows, frozenset(dtypes), () if first_shape is None else first_shape)


def store_row(sequence: str, row: Tensor, *, matrix: bool, sparse: bool) -> Tensor | SparseRow:
    """One decoded row in the form the store's layout takes.

    A pooled vector was stored as `(d,)`, or `(1, d)` by the oldest raw blobs, and a per-token matrix
    as `(l, d)`. A sparse layout keeps the exact nonzero entries, which is all a coordinate-list blob
    ever held.
    """
    # row: (n, d) token rows for a matrix store, else (d,) or (1, d); n is whatever the old embedder kept per sequence.
    if matrix:
        if row.ndim != 2:
            raise ValueError(
                f"A per-token cache row must be (l, d); the row of a sequence of {len(sequence)} "
                f"residues is {tuple(row.shape)}."
            )
        return row  # (l, d)
    if not (row.ndim == 1 or (row.ndim == 2 and row.shape[0] == 1)):
        raise ValueError(
            f"A pooled cache row must be (d,) or (1, d); the row of a sequence of {len(sequence)} "
            f"residues is {tuple(row.shape)}."
        )
    vector = row.reshape(-1)  # (d,)
    if sparse:
        from fastplms.features import SparseRow

        return SparseRow.from_dense(vector)  # (d,) as its nonzeros
    return vector  # (d,)


def store_rows(
    path: str | Path, kind: str, *, matrix: bool, sparse: bool, only: frozenset[str] | None = None,
) -> Callable[[], Iterator[tuple[str, Tensor | SparseRow]]]:
    """The decoder `convert_rows` calls, which takes no arguments because it is called twice.

    `only` keeps just those sequences, for a conversion that adds to a store that already holds others.
    """
    fallback = _fallback_shape(matrix)

    def decode() -> Iterator[tuple[str, Tensor | SparseRow]]:
        source = sqlite_rows(path, fallback_shape=fallback) if kind == "sqlite" else torch_rows(path)
        for sequence, row in source:
            if only is None or sequence in only:
                yield sequence, store_row(sequence, row, matrix=matrix, sparse=sparse)

    return decode  # () -> (sequence, (d,) or (n, d) row) pairs


@dataclass(frozen=True)
class LegacyConversion:
    """One conversion: the store it filled, the dtype that store holds, and `convert_rows`' receipt."""

    directory: Path
    dtype: str
    receipt: ConversionReceipt


def _feature_cache() -> Any:
    try:
        import feature_cache
    except ImportError:
        from . import feature_cache
    return feature_cache


def convert_legacy_cache(
    path: str | Path,
    root: str | Path,
    descriptor: Mapping[str, Any],
    *,
    request_dtype: bool = True,
    skip_existing: bool = False,
    rows_per_segment: int = 100_000,
    window_rows: int = 1_024,
) -> LegacyConversion | None:
    """Move one old cache into the feature store under `root`, and prove the rows survived.

    `descriptor` is the `feature_cache.embedding_descriptor` of the representation the file holds. Its
    `dtype` is the dtype the caller stores; the store gets the narrowest dtype that holds both that and
    every row of the file, because the store refuses a lossy cast. With `request_dtype` False the
    descriptor's dtype is ignored and the file's own rows decide. When a store for the representation
    already exists, its dtype decides, and a file whose rows it cannot hold exactly is skipped.

    By default every row of the file is compared with the store, so a store that already holds
    different numbers for a sequence fails loudly. `skip_existing` converts only the sequences the
    store lacks, for a store that is also being filled by embedding. Returns None when there is nothing
    to convert or the store cannot hold the file's rows.
    """
    from fastplms.features import FeatureStore, convert_rows, describe_file

    cache = _feature_cache()
    refusal = cache.legacy_conversion_refusal(descriptor)
    if refusal is not None:
        raise ValueError(refusal)
    path = Path(path).resolve(strict=True)
    kind = cache_kind(path)
    matrix = descriptor["form"] == "matrix"
    sparse = cache.stores_sparse(descriptor)
    scan = scan_cache(path, kind, _fallback_shape(matrix))
    if scan.rows == 0:
        raise ValueError(f"{path} holds no embeddings.")
    source = cache.narrowest_holding(*(cache.dtype_name(dtype) for dtype in scan.dtypes))
    requested = descriptor["dtype"] if request_dtype else source
    descriptor = {**descriptor, "dtype": requested}
    layout = cache.embedding_layout(descriptor)
    width = scan.first_shape[-1]

    located = cache.locate_feature(root, descriptor)
    if cache.has_feature(located):
        spec = FeatureStore.read_only(located).spec
        if not cache.holds(cache.dtype_name(spec.dtype), source):
            print(
                f"Skipped {path.name}: its {source} rows do not fit exactly in {spec.key}, which stores "
                f"{cache.dtype_name(spec.dtype)}."
            )
            return None
        if spec.layout != layout or spec.width != width:
            raise ValueError(
                f"{located} holds {spec.layout} rows of width {spec.width}, but {path.name} holds "
                f"{layout} rows of width {width}. The descriptor names a different representation."
            )
    else:
        target = cache.narrowest_holding(source, requested)
        spec = cache.embedding_feature({**descriptor, "dtype": target}, layout=layout, width=width)
    store = FeatureStore.open(cache.feature_directory(root, spec.descriptor).parent, spec)

    only = None
    if skip_existing:
        absent = store.missing(cache_sequences(path, kind))
        if not absent:
            print(f"Nothing to convert from {path.name}: every sequence is already in {spec.key}.")
            return None
        only = frozenset(absent)
    origin = {
        "format": FORMATS[kind],
        "decoder": DECODER,
        "files": [describe_file(path)],
        "only_missing": skip_existing,
    }
    receipt = convert_rows(
        store, store_rows(path, kind, matrix=matrix, sparse=sparse, only=only), origin=origin,
        rows_per_segment=rows_per_segment, window_rows=window_rows,
    )
    return LegacyConversion(store.directory, cache.dtype_name(spec.dtype), receipt)


_CARBON_IDENTITY = (
    r"(?:_rev(?P<revision>local_[0-9a-f]{64}|[0-9a-f]{40})"
    r"_prep(?P<preprocessing>carbon_[A-Za-z0-9_]+?_qwen_[0-9a-f]{12}))?"
)
_FILENAME = re.compile(
    r"^(?P<model>.+?)_(?P<matrix>True|False)(?:_hs(?P<hidden>-?\d+))?(?:_len(?P<length>\d+))?"
    + _CARBON_IDENTITY
    + r"(?:_(?P<pooling>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*))?\.(?P<extension>db|pth)$"
)


def parse_cache_filename(name: str) -> dict[str, Any]:
    """Read back what `embedder.get_embedding_filename` put in an old cache's name.

    The pooling in a name is sorted, so it is the order the columns hold only when the run pooled in
    sorted order. Pass the real order to the converter when it differs.
    """
    found = _FILENAME.match(name)
    if found is None:
        raise ValueError(f"{name} does not look like a Protify cache name.")
    pooling = found["pooling"]
    return {
        "model_name": found["model"],
        "matrix_embed": found["matrix"] == "True",
        "hidden_state_index": int(found["hidden"]) if found["hidden"] is not None else -1,
        "max_length": int(found["length"]) if found["length"] is not None else None,
        "carbon": None if found["revision"] is None else (found["revision"], found["preprocessing"]),
        "pooling_types": [] if pooling is None else pooling.split("_"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m protify.legacy_cache",
        description=(
            "Convert one old Protify embedding cache (.db or .pth) into the feature store. The file is "
            "read twice and never changed; every stored row is compared with the decoded row bit for bit."
        ),
    )
    parser.add_argument("path", help="the old cache file")
    parser.add_argument("--root", default=None, help="feature store root, default: the folder of the cache")
    parser.add_argument("--model", default=None, help="resolved model name, default: read from the file name")
    parser.add_argument("--matrix", dest="matrix", action="store_true", default=None, help="the cache holds per-token matrices")
    parser.add_argument("--vector", dest="matrix", action="store_false", help="the cache holds pooled vectors")
    parser.add_argument("--pooling", nargs="+", default=None, help="pooling types in the order the columns hold, default: the file name's, which is sorted")
    parser.add_argument("--hidden_state_index", type=int, default=None, help="default: read from the file name")
    parser.add_argument("--max_length", type=int, default=None, help="default: read from the file name")
    parser.add_argument("--dtype", default=None, choices=["float16", "bfloat16", "float32", "float64"], help="store dtype, default: the narrowest that holds every row")
    parser.add_argument("--skip_existing", action="store_true", help="convert only sequences the store lacks, instead of comparing every row")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cache = _feature_cache()
    path = Path(args.path).resolve(strict=True)
    named = parse_cache_filename(path.name)
    model_name = args.model or named["model_name"]
    matrix = named["matrix_embed"] if args.matrix is None else args.matrix
    max_length = args.max_length if args.max_length is not None else named["max_length"]
    if max_length is None:
        parser.error("the file name has no _len<max_length> part, so pass --max_length")
    pooling = args.pooling if args.pooling is not None else named["pooling_types"]
    if not matrix and not pooling:
        parser.error("a pooled cache needs its pooling types, from the file name or --pooling")
    if "carbon" in model_name.lower() and named["carbon"] is None:
        parser.error("a CARBON cache name carries its _rev<revision>_prep<schema>; this one does not")
    descriptor = cache.embedding_descriptor(
        model_name,
        matrix_embed=matrix,
        pooling_types=pooling,
        hidden_state_index=args.hidden_state_index if args.hidden_state_index is not None else named["hidden_state_index"],
        max_length=max_length,
        dtype=args.dtype or "float32",
        carbon=named["carbon"],
        # An old SAE cache pooled the interior tokens only, and says so in its store.
        special_tokens="excluded" if cache.sae_selection(model_name) is not None else "included",
    )
    # Without --dtype the file's own rows decide the store dtype, so the placeholder above is ignored.
    conversion = convert_legacy_cache(
        path, args.root or path.parent, descriptor,
        request_dtype=args.dtype is not None, skip_existing=args.skip_existing,
    )
    if conversion is None:
        return 1
    receipt = conversion.receipt
    print(
        f"Converted {path.name} into {conversion.directory} ({conversion.dtype}): "
        f"{receipt.source_rows} rows read, {receipt.written} written, {receipt.skipped} already present, "
        f"{receipt.verified} compared bit for bit with the file."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
