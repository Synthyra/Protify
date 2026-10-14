"""Protify's embedding caches, kept in the workspace's one feature store.

Every representation Protify computes for a set of sequences is one feature: one row per sequence,
addressed by the SHA-256 of the sequence. Each lives in its own directory under the embedding folder
(`--embedding_save_dir`, default `embeddings`), named by the digest of a descriptor that states
everything that changes its numbers:

    schema              protify_embedding_v1
    model               the resolved model name, which already carries SAE layer, sparsity and codebook
    revision            CARBON's resolved model revision; None for every other model, whose cache never
                        recorded one
    preprocessing       CARBON's preprocessing schema; None otherwise
    form                vector (pooled) or matrix (one row per token)
    pooling             the pooling types in the order the columns are concatenated; None for a matrix
    hidden_state_index  the hidden state read, -1 for the last
    max_length          the truncation limit, in tokens
    special_tokens      included: pooling and matrices keep the leading and trailing special tokens, so
                        a matrix has len(sequence) + 2 rows and every pooled statistic, an SAE model's
                        max included, covers all of them (docs/conventions/probes.md). Nothing writes
                        `excluded`, the earlier SAE convention of pooling the interior tokens only
    dtype               the stored dtype

The key is `foundry.embedding.plain_feature_key`, not `feature_key`, because that key claims the
biological-residue pooling FastPLMs' taps produce, and Protify's legacy pooling includes the special
tokens. Layout follows the representation: `dense` for a pooled vector, `csr` when an SAE model stores
its pooled codes sparsely (`EsmcSaeSelection.sparse_storage`), `ragged` for a matrix.

The stored dtype is part of the key, and a store serves a request for any dtype it holds exactly: a
float32 store serves a float16 or bfloat16 run, and rows a run writes into it are widened losslessly.
Nothing is ever narrowed, so a store's numbers are the numbers of the run that made them.

The two control bases, `Random` and `OneHot-Protein`, have no store. Their rows are a function of the
sequence (`base_models.control_bases`), so the reader generates them on request: `feature_directory`
names a virtual `generated__...` directory that holds nothing, `has_feature` is true for it,
`missing_sequences` is empty, `open_reader` yields a `GeneratedReader`, and `CacheWriter` refuses it.
Every Protify path that opens a cache therefore reads a control without ever writing one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import torch

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from torch import Tensor

from foundry.digests import sha256_file
from foundry.embedding import plain_feature, plain_feature_key


if TYPE_CHECKING:
    from fastplms.features import FeatureReader, FeatureStore, SparseRow, StoredFeature


OWNER = "protify"
SCHEMA = "protify_embedding_v1"
# A window of rows is committed as one segment, so a run that dies loses at most one window.
WINDOW_ROWS = 2048
WINDOW_BYTES = 256 * 1024**2
# Reading a store into memory takes this many sequences per index query.
READ_CHUNK = 1024
# The environment variable that routes one logical root's stores to other directories (`feature_root`).
FEATURE_ROUTES_VARIABLE = "PROTIFY_FEATURE_ROOTS"

# Stored dtypes, narrowest first, and the dtypes whose values each holds exactly.
DTYPES = ("float16", "bfloat16", "float32", "float64")
_EXACT = {
    "float16": frozenset({"float16"}),
    "bfloat16": frozenset({"bfloat16"}),
    "float32": frozenset({"float16", "bfloat16", "float32"}),
    "float64": frozenset(DTYPES),
}
_SLUG_WORD = re.compile(r"[^A-Za-z0-9._-]+")
# What `--model_dtype` and `--embed_dtype` call the three torch dtypes.
_ABBREVIATIONS = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def dtype_name(dtype: torch.dtype | str) -> str:
    """A dtype's stored name, refusing one no cache stores."""
    name = str(dtype).removeprefix("torch.")
    name = _ABBREVIATIONS.get(name, name)
    if name not in DTYPES:
        raise ValueError(f"A cache stores {DTYPES} values; received {dtype!r}.")
    return name


def resolve_embed_dtype(embed_dtype: torch.dtype | str | None, model_dtype: torch.dtype | str | None = None) -> torch.dtype:
    """The torch dtype a run stores its embeddings in, from the two settings `main.py` reads.

    A parsed namespace leaves `embed_dtype` None, and `MainProcess.apply_current_settings` then stores
    in the model dtype, which is bf16 unless one is named. Resolving it the same way here lets code that
    holds only the parsed arguments name the store the run will write.
    """
    if embed_dtype is None:
        embed_dtype = "bf16" if model_dtype is None else model_dtype
    return getattr(torch, dtype_name(embed_dtype))


def holds(wide: str, narrow: str) -> bool:
    """Whether dtype `wide` represents every value of dtype `narrow` exactly."""
    return narrow in _EXACT[wide]


def narrowest_holding(*names: str) -> str:
    """The narrowest stored dtype that holds every one of these dtypes' values exactly."""
    for candidate in DTYPES:
        if all(holds(candidate, name) for name in names):
            return candidate
    raise ValueError(f"No stored dtype holds {names}.")


def _controls() -> Any:
    """The control-base module, imported as `sae_selection` imports its neighbor."""
    from .base_models import control_bases
    return control_bases


def sae_selection(model_name: str) -> Any:
    """The SAE checkpoint a model name encodes, or None for every other model."""
    from .base_models.esmc_sae import is_sae_model_name, parse_sae_model_name
    return parse_sae_model_name(model_name) if is_sae_model_name(model_name) else None


def carbon_cache_identity(dispatch_name: str, model_path: str | None = None) -> tuple[str, str] | None:
    """The model revision and preprocessing schema a CARBON cache is keyed by, or None for other models.

    Both change stored values. A local model has no Hub commit, so its identity is the digest of its
    resolved path, marked `local_` rather than passed off as an immutable revision.
    """
    if "carbon" not in dispatch_name.lower():
        return None
    from .base_models.carbon import CARBON_PREPROCESSING_SCHEMA, resolve_carbon_source

    source, revision = resolve_carbon_source(dispatch_name, model_path=model_path)
    if revision is None:
        resolved_source = os.path.normcase(os.path.realpath(source))
        revision = f"local_{hashlib.sha256(resolved_source.encode('utf-8')).hexdigest()}"
    return revision, CARBON_PREPROCESSING_SCHEMA


def embedding_descriptor(
    model_name: str,
    *,
    matrix_embed: bool,
    pooling_types: Sequence[str],
    hidden_state_index: int,
    max_length: int,
    dtype: torch.dtype | str,
    model_type: str | None = None,
    model_path: str | None = None,
    carbon: tuple[str, str] | None = None,
    special_tokens: str = "included",
) -> dict[str, Any]:
    """The plain data that identifies one cached representation, and so its store.

    `carbon` is the `(revision, preprocessing)` pair of a CARBON cache when the caller already knows it,
    such as the converter reading it back from an old file name; otherwise it is resolved from
    `model_type` (the dispatch keyword when it differs from the display name) and `model_path`.
    `special_tokens` is `included` for everything a current writer makes; only the converter of an old
    SAE cache, whose rows pooled the interior tokens, passes `excluded`.
    """
    if special_tokens not in ("included", "excluded"):
        raise ValueError(f"special_tokens is included or excluded, not {special_tokens!r}.")
    if carbon is None:
        carbon = carbon_cache_identity(model_type or model_name, model_path)
    revision, preprocessing = (None, None) if carbon is None else carbon
    return {
        "schema": SCHEMA,
        "model": model_name,
        "revision": revision,
        "preprocessing": preprocessing,
        "form": "matrix" if matrix_embed else "vector",
        "pooling": None if matrix_embed else list(pooling_types),
        "hidden_state_index": int(hidden_state_index),
        "max_length": int(max_length),
        "special_tokens": special_tokens,
        "dtype": dtype_name(dtype),
    }


def legacy_conversion_refusal(descriptor: Mapping[str, Any]) -> str | None:
    """Why an old cache cannot become the store this descriptor names, or None when it can.

    An old SAE cache pooled the interior tokens only, and canonical pooling covers all l + 2 rows,
    so converting it under an `included` descriptor would label other numbers as this representation.
    It converts only under `excluded`, which no current reader asks for. A control has no store.
    """
    model = descriptor["model"]
    if is_generated_model(model):
        return f"{model} is generated at load, so there is no store to convert an old cache into."
    if descriptor["form"] == "vector" and sae_selection(model) is not None and descriptor["special_tokens"] == "included":
        return (
            f"An old {model} cache pooled the interior tokens only, but canonical pooling covers all l + 2 rows, "
            "so its rows are not this representation. Embed the sequences again."
        )
    return None


def stores_sparse(descriptor: Mapping[str, Any]) -> bool:
    """Whether this representation is stored as sparse coordinates: an SAE model with few active codes."""
    if descriptor["form"] != "vector":
        return False
    selection = sae_selection(descriptor["model"])
    return selection is not None and selection.sparse_storage


def embedding_layout(descriptor: Mapping[str, Any]) -> str:
    """The store layout of a representation: `ragged` matrices, `csr` sparse SAE codes, else `dense`."""
    from fastplms.features import CSR, DENSE, RAGGED

    if descriptor["form"] == "matrix":
        return RAGGED
    return CSR if stores_sparse(descriptor) else DENSE


def _slug_words(descriptor: Mapping[str, Any]) -> tuple[str, ...]:
    """A few readable words for the directory name. The digest, not the slug, decides identity."""
    words = [descriptor["model"], descriptor["form"]]
    if descriptor["pooling"]:
        words.append("-".join(descriptor["pooling"]))
    if descriptor["hidden_state_index"] != -1:
        words.append(f"hs{descriptor['hidden_state_index']}")
    words += [f"len{descriptor['max_length']}", descriptor["dtype"]]
    return tuple(_SLUG_WORD.sub("-", word).strip("-._") or "x" for word in words)


def embedding_feature(descriptor: Mapping[str, Any], *, layout: str, width: int) -> StoredFeature:
    """The `StoredFeature` of a descriptor, whose dtype field is the dtype of the rows."""
    return plain_feature(
        OWNER, descriptor, layout=layout, width=width, dtype=getattr(torch, descriptor["dtype"]),
        slug=_slug_words(descriptor),
    )


def feature_key(descriptor: Mapping[str, Any]) -> str:
    """The directory name of a descriptor's store."""
    return plain_feature_key(OWNER, descriptor, slug=_slug_words(descriptor)).name


def feature_root(root: str | Path, descriptor: Mapping[str, Any]) -> Path:
    """The directory a descriptor's store lives under: `root`, unless FEATURE_ROUTES_VARIABLE routes its model.

    The variable holds JSON, `{"logical_root": <root>, "routes": {<model prefix>: <directory>}}`,
    which a launcher sets to spread one root over several volumes. A prefix routes the model of
    that name and every name that extends it after a `-`, so `ESMC-600` also routes
    `ESMC-600-SAE-l27-k64-c16384`, and the longest matching prefix wins. Under the logical root, a
    model no route names raises rather than landing somewhere the launcher does not commit.
    Routing changes storage only; descriptor keys and everything else that names a store stay intact.
    """
    routing = os.environ.get(FEATURE_ROUTES_VARIABLE)
    if not routing:
        return Path(root)

    configuration = json.loads(routing)
    if Path(root) != Path(configuration["logical_root"]):
        return Path(root)

    model = descriptor["model"]
    prefixes = [prefix for prefix in configuration["routes"] if model == prefix or model.startswith(f"{prefix}-")]
    if not prefixes:
        raise ValueError(f"{FEATURE_ROUTES_VARIABLE} routes {root} but names no route for {model!r}.")
    return Path(configuration["routes"][max(prefixes, key=len)])


GENERATED_PREFIX = "generated"
# generated__<model>__<form>__<pooling joined by '-', or 'none'>__len<max_length>
_GENERATED_NAME = re.compile(
    rf"{GENERATED_PREFIX}__(?P<model>[A-Za-z0-9-]+)__(?P<form>vector|matrix)__(?P<pooling>[a-z]+(?:-[a-z]+)*)__len(?P<max_length>\d+)"
)
GENERATED_SCHEMA = "protify_generated_feature_v1"


@dataclass(frozen=True)
class GeneratedFeature:
    """What a generated control read is: the control, its form, its pooling, and its token limit."""

    model: str
    form: str
    pooling: tuple[str, ...]
    max_length: int

    @property
    def width(self) -> int:
        """Columns of one row: a pooled row's statistics side by side, or a per-token row."""
        columns = _controls().control_width(self.model)
        return columns if self.form == "matrix" else columns * len(self.pooling)

    @property
    def descriptor(self) -> dict[str, Any]:
        """The identity of these rows as plain data, which a package records beside its other features."""
        return {
            "schema": GENERATED_SCHEMA, "model": self.model, "form": self.form,
            "pooling": None if self.form == "matrix" else list(self.pooling), "max_length": self.max_length,
            "special_tokens": "included", "generator": _controls().CONTROL_SCHEMA,
        }


@dataclass(frozen=True)
class GeneratedSpec:
    """The part of a store's spec a reader's caller reads: the row width and layout, and the identity."""

    width: int
    layout: str
    descriptor: dict[str, Any]


def generated_feature(descriptor: Mapping[str, Any]) -> GeneratedFeature:
    """The generated read a control's descriptor asks for, refusing what a control cannot generate."""
    controls = _controls()
    model, form = descriptor["model"], descriptor["form"]
    if descriptor["hidden_state_index"] != -1:
        raise ValueError(f"{model} has no layers, so hidden_state_index {descriptor['hidden_state_index']} reads nothing.")
    if form == "matrix":
        controls.control_width(model)
        return GeneratedFeature(model, form, ("none",), int(descriptor["max_length"]))
    pooling = controls.check_pooling(model, descriptor["pooling"])
    return GeneratedFeature(model, form, pooling, int(descriptor["max_length"]))


def generated_name(feature: GeneratedFeature) -> str:
    """The virtual directory name of a generated read, which carries everything its rows depend on."""
    return f"{GENERATED_PREFIX}__{feature.model}__{feature.form}__{'-'.join(feature.pooling)}__len{feature.max_length}"


def parse_generated(directory: str | Path) -> GeneratedFeature | None:
    """The generated read a directory names, or None when it is an ordinary store directory."""
    match = _GENERATED_NAME.fullmatch(Path(directory).name)
    if match is None:
        return None
    pooling = tuple(match["pooling"].split("-"))
    return GeneratedFeature(match["model"], match["form"], pooling, int(match["max_length"]))


class GeneratedReader:
    """The reader of a generated control: rows made on request, with nothing opened and nothing stored.

    It answers the calls Protify makes of a store's `FeatureReader`: `read`, `missing`, `spec`, membership,
    and the context manager. Every sequence is present, since a control has a row for any sequence.
    """

    def __init__(self, feature: GeneratedFeature) -> None:
        self.feature = feature

    @property
    def spec(self) -> GeneratedSpec:
        return GeneratedSpec(self.feature.width, "ragged" if self.feature.form == "matrix" else "dense", self.feature.descriptor)

    def __enter__(self) -> GeneratedReader:
        return self

    def __exit__(self, *exception: object) -> None:
        return None

    def close(self) -> None:
        return None

    def __contains__(self, sequence: str) -> bool:
        return True

    def missing(self, sequences: Iterable[str]) -> tuple[str, ...]:
        return ()

    def read(self, sequences: Sequence[str]) -> list[Tensor]:
        """One row per sequence, made now from the sequence alone: nothing is read from disk."""
        controls = _controls()
        feature = self.feature
        if feature.form == "matrix":
            # b of (l + 2, w): token rows, CLS first and EOS last, generated at load.
            return [controls.control_matrix(feature.model, sequence, feature.max_length) for sequence in sequences]  # b of (l + 2, w)
        # (b, m * w) pooled over all l + 2 rows, m statistics side by side; split into b rows of (m * w,).
        return list(controls.control_rows(feature.model, sequences, feature.pooling, feature.max_length))  # b of (m * w,)


def is_generated_model(model_name: str) -> bool:
    """Whether a model name is a control, whose rows the reader generates instead of storing."""
    return _controls().is_control_model(model_name)


def feature_directory(root: str | Path, descriptor: Mapping[str, Any]) -> Path:
    """Where the store of exactly this descriptor lives under `root`, whether or not it exists yet.

    A control has no store: its path is the virtual directory of its generated read, which nothing creates.
    """
    if is_generated_model(descriptor["model"]):
        return Path(root) / generated_name(generated_feature(descriptor))
    return feature_root(root, descriptor) / feature_key(descriptor)


def has_feature(directory: str | Path) -> bool:
    """Whether `directory` holds a feature store, or names a generated read, which is always available."""
    from fastplms.features.store import FEATURE_FILE

    return parse_generated(directory) is not None or (Path(directory) / FEATURE_FILE).is_file()


def locate_feature(root: str | Path, descriptor: Mapping[str, Any]) -> Path:
    """The store that serves this descriptor: one holding its dtype exactly, or a wider one.

    A request for bfloat16 is served by a bfloat16 store, else by a float32 or float64 one, whose values
    include every bfloat16 value. When none exists this is the path where the exact store would be made.
    """
    requested = descriptor["dtype"]
    for held in DTYPES:
        if holds(held, requested):
            candidate = feature_directory(root, {**descriptor, "dtype": held})
            if has_feature(candidate):
                return candidate
    return feature_directory(root, descriptor)


def require_feature(directory: str | Path, legacy: Iterable[str | Path] = ()) -> Path:
    """`directory` as a Path, refusing when it holds no store, and saying how to convert an old cache."""
    directory = Path(directory)
    if has_feature(directory):
        return directory
    old = [str(path) for path in legacy if Path(path).is_file()]
    hint = (
        f" An old cache is there: {', '.join(old)}. Convert it with `python -m protify.legacy_cache <file>`, "
        "or let an embedding run adopt it."
        if old else " Embed the sequences first."
    )
    raise FileNotFoundError(f"No embedding store at {directory}.{hint}")


def missing_sequences(directory: str | Path, sequences: Iterable[str]) -> tuple[str, ...]:
    """The distinct sequences a store lacks, in the order given; all of them when there is no store.

    A generated control lacks none.
    """
    from fastplms.features import FeatureReader

    if not has_feature(directory):
        return tuple(dict.fromkeys(sequences))
    if parse_generated(directory) is not None:
        return ()
    with FeatureReader.open(directory) as reader:
        return reader.missing(sequences)


def open_reader(directory: str | Path, legacy: Iterable[str | Path] = ()) -> FeatureReader | GeneratedReader:
    """A reader over an existing store for random access, which the caller closes.

    Training reads a few rows per batch, so it holds one reader for the run instead of opening the
    store per read. A reader pickles as its store, so a spawned data loader worker reopens it. A
    generated control gets a `GeneratedReader` instead, which pickles as its few fields.
    """
    from fastplms.features import FeatureReader

    feature = parse_generated(directory)
    if feature is not None:
        return GeneratedReader(feature)
    return FeatureReader.open(require_feature(directory, legacy))


@contextmanager
def row_lookup(directory: str | Path, legacy: Iterable[str | Path] = ()) -> Iterator[Callable[[str], Tensor]]:
    """Yield `sequence -> row` over one store, opened once for the whole `with` block.

    A row is `(d,)` for a pooled representation, densified when the store keeps it sparse, and
    `(l + 2, d)` per token for a matrix (CLS, the l residues, EOS). A sequence the store lacks raises `KeyError`.
    """
    with open_reader(directory, legacy) as reader:
        yield lambda sequence: reader.read([sequence])[0]


def read_embeddings(directory: str | Path, sequences: Iterable[str]) -> dict[str, Tensor]:
    """The rows of these sequences in memory, each an independent tensor: `(d,)` pooled or `(l + 2, d)` per token (CLS, residues, EOS).

    A sparse store gives its rows densified. A sequence the store lacks raises `KeyError`.
    """
    from fastplms.features import FeatureReader

    unique = list(dict.fromkeys(sequences))
    rows: dict[str, Tensor] = {}
    if not unique:
        return rows  # sequence -> (d,) or (l + 2, d)
    feature = parse_generated(directory)
    with GeneratedReader(feature) if feature is not None else FeatureReader.open(directory) as reader:
        for start in range(0, len(unique), READ_CHUNK):
            chunk = unique[start:start + READ_CHUNK]
            for sequence, row in zip(chunk, reader.read(chunk), strict=True):
                rows[sequence] = row.clone()
    return rows  # sequence -> (d,) or (l + 2, d)


def store_fingerprint(directory: str | Path) -> str:
    """One digest for a store's committed content, refusing a store whose parts differ from their markers.

    Every part is verified against its commit marker, and each marker holds its parts' SHA-256, so the
    digest of `feature.json` and the markers pins every stored byte. It replaces the whole-file hash a
    frozen SQLite cache used to get.
    """
    from fastplms.features import FeatureReader
    from fastplms.features.store import COMMIT_FILE, FEATURE_FILE, SEGMENTS_DIRECTORY

    directory = Path(directory)
    generated = parse_generated(directory)
    if generated is not None:
        # No bytes to hash: the generator's schema and the read's identity are the whole content.
        return hashlib.sha256(json.dumps(generated.descriptor, sort_keys=True).encode()).hexdigest()
    with FeatureReader.open(directory) as reader:
        reader.verify()
    names = [FEATURE_FILE] + sorted(
        f"{SEGMENTS_DIRECTORY}/{marker.parent.name}/{COMMIT_FILE}"
        for marker in (directory / SEGMENTS_DIRECTORY).glob(f"*/{COMMIT_FILE}")
    )
    digest = hashlib.sha256()
    for name in names:
        digest.update(f"{name}\0{sha256_file(directory / name)}\n".encode())
    return digest.hexdigest()


class CacheWriter:
    """Write embeddings into their store one window at a time.

    A pipeline calls `add` with each batch it embeds. Rows collect until a window is full, then commit as
    one segment (`fastplms.features.write_rows`), so a run that dies keeps every committed window and a
    restart embeds only what `missing_sequences` still reports. The store is created when the first
    window commits, from that window's row width; a store that already serves the descriptor is reused
    with its own dtype, and the layout it holds must be the one this run writes.

    Rows are `(d,)` for `dense` and `csr` (a dense vector, compressed by keeping its exact nonzeros) and
    `(l + 2, d)` for `ragged`. The dtype of every row must fit exactly in the store's, and is widened to it.
    """

    def __init__(
        self,
        root: str | Path,
        descriptor: Mapping[str, Any],
        *,
        layout: str,
        window_rows: int = WINDOW_ROWS,
        window_bytes: int = WINDOW_BYTES,
        label: str | None = None,
    ) -> None:
        from fastplms.features import CSR, DENSE, RAGGED

        if layout not in (DENSE, CSR, RAGGED):
            raise ValueError(f"A Protify cache is dense, csr, or ragged; received {layout!r}.")
        if is_generated_model(descriptor["model"]):
            raise ValueError(f"{descriptor['model']} is generated at load and never stored, so nothing writes its cache.")
        if window_rows < 1 or window_bytes < 1:
            raise ValueError("window_rows and window_bytes must be positive.")
        self.root = Path(root)
        self.descriptor = dict(descriptor)
        self.layout = layout
        self.window_rows = window_rows
        self.window_bytes = window_bytes
        self.label = label or f"{time.strftime('%Y%m%dT%H%M%S')}-pid{os.getpid()}"
        self.rows_written = 0
        self.windows = 0
        self._pending: dict[str, Tensor] = {}
        self._pending_bytes = 0
        self._store: FeatureStore | None = None

    @property
    def directory(self) -> Path | None:
        """The store's directory once it is open, else None."""
        return None if self._store is None else self._store.directory

    def add(self, sequences: Sequence[str], rows: Sequence[Tensor]) -> None:
        """Buffer rows, committing a window each time one fills, so a long list is not held whole."""
        # rows: (d,) pooled or (l + 2, d) token rows, one per sequence, b rows in all.
        for sequence, row in zip(sequences, rows, strict=True):
            if sequence in self._pending:
                raise ValueError("A sequence was embedded twice in one window; each store row is written once.")
            self._pending[sequence] = row.detach().cpu()
            self._pending_bytes += row.numel() * row.element_size()
            if len(self._pending) >= self.window_rows or self._pending_bytes >= self.window_bytes:
                self.flush()

    def flush(self) -> None:
        """Commit every buffered row as one segment."""
        from fastplms.features import write_rows

        if not self._pending:
            return
        store = self._open(next(iter(self._pending.values())))
        # A parallel run may have committed some of these sequences since this one listed what was missing.
        absent = set(store.missing(list(self._pending)))
        rows = {
            sequence: self._encode(row, store.spec)
            for sequence, row in self._pending.items() if sequence in absent
        }
        self._pending = {}
        self._pending_bytes = 0
        metadata = {"writer": "protify.feature_cache.CacheWriter", "run": self.label, "window": self.windows}
        written = write_rows(store, rows, metadata=metadata)
        if written:
            self.rows_written += written
            self.windows += 1

    def close(self) -> None:
        """Commit what is buffered."""
        self.flush()

    def __enter__(self) -> CacheWriter:
        return self

    def __exit__(self, kind: object, error: object, trace: object) -> None:
        # A run that raised keeps its committed windows and drops the open one, as a dead run does.
        if kind is None:
            self.close()

    def _open(self, first_row: Tensor) -> FeatureStore:
        # first_row: (d,) or (l + 2, d); the stored width d is its last axis.
        from fastplms.features import FeatureStore

        if self._store is not None:
            return self._store
        located = locate_feature(self.root, self.descriptor)
        if has_feature(located):
            spec = FeatureStore.read_only(located).spec
            if spec.layout != self.layout:
                raise ValueError(
                    f"{located} holds {spec.layout} rows, but this run writes {self.layout} rows. "
                    "The model's storage policy changed under the same descriptor."
                )
        else:
            spec = embedding_feature(self.descriptor, layout=self.layout, width=int(first_row.shape[-1]))
        # Deep verification re-hashes every committed part on each commit, so a run's writes would grow
        # with the store; without it a reader still verifies each part the first time it touches it.
        self._store = FeatureStore.open(located.parent, spec, deep_verify=False)
        return self._store

    def _encode(self, row: Tensor, spec: StoredFeature) -> Tensor | SparseRow:
        # row: (d,) or (l + 2, d), cast to the store's dtype and, for csr, kept as its nonzeros.
        from fastplms.features import CSR, SparseRow

        if not holds(dtype_name(spec.dtype), dtype_name(row.dtype)):
            raise ValueError(
                f"{spec.key} stores {dtype_name(spec.dtype)}, which would round {dtype_name(row.dtype)} rows."
            )
        values = row.to(spec.dtype)
        return SparseRow.from_dense(values) if spec.layout == CSR else values  # (d,) csr row or the dense row
