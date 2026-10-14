"""Canonical stores of the encoders that are not protein models: contracts, runtime, inputs and the embedding run.

The ESMC stores (`canonical`) hold FastPLMs' taps over proteins. Atlas has two more towers: SELFIES-TED-LARGE reads a
ligand and the Annotation Transformer reads a protein's annotation terms. Logan, 2026-10-02: "just like ESMC we should have an
embedding store for the SELFIES language model and Annotation Transformer we use." They are the same stores: a stream is a
`StoredFeature` of the FastPLMs feature store named by a `feature_spec_v2` (CLS and EOS kept) or `feature_spec_v1` (no
special tokens) contract, a row is keyed by the SHA-256 of its canonical text, per-token streams and pooled streams sit in a
token store and a pooled store, `FeatureView` and `open_canonical_view` read them, and `job` embeds, pilots, verifies, sweeps,
seeds and pushes them. What a family changes is how a raw input becomes a row key and token ids, what the model computes and
what the manifest says, all in `ligand` and `annotation`; `encoder_rows` is the code that decides a stored value.

FastPLMs' `embed_token_features` is bound to its proteins (its taps, residue vocabulary and executor), so `embed_encoder_features`
repeats its few lines of orchestration over the same `AsyncFeatureWriter`, and `EncoderExtractionContract` subclasses
`FeatureExtractionContract`, replacing only the capture (a `FeatureSpec` measured from a Hugging Face model with no taps) and the
inventory check.

Symbols: b sequences of a batch; l units of a row after the crop; d hidden width; n rows of a store.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import torch

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from tqdm.auto import tqdm

from .annotation import ANNOTATION_FAMILY
from .canonical import (
    FAMILY_ANNOTATION, FAMILY_LIGAND, STORE_POOLED, STORE_TOKEN, CanonicalBase, CanonicalRuntime, file_identity,
)
from .contracts import (
    FEATURE_SCHEMA, FEATURE_SCHEMA_V2, MASK_POLICY_V1, MASK_POLICY_V2, SPECIAL_TOKENS_KEPT, FeatureSpec,
    SequenceKey, _digest,
)
from .encoder_rows import (
    CanonicalInput, EncoderExecutor, EncoderFamily, EncoderHooks, EncoderStream, PreparedRow, plan_row_batches,
)
from .ligand import LIGAND_FAMILY
from .persistence import FeatureExtractionContract


if TYPE_CHECKING:
    from fastplms.features import SegmentReceipt, StoredFeature
    from torch import nn


FAMILIES: dict[str, EncoderFamily] = {FAMILY_LIGAND: LIGAND_FAMILY, FAMILY_ANNOTATION: ANNOTATION_FAMILY}
FAMILY_DEFAULT_COLUMNS = {FAMILY_LIGAND: ("smiles_sha256", "smiles"), FAMILY_ANNOTATION: ("entry_sha256", "terms")}
BATCH_ALGORITHM = "row_budget_length_sorted_v1"  # sequences longest first inside windows, packed under a padded-row budget
RUN_SCHEMA = "encoder_features_v1"
INVENTORY_DIRECTORY = "inventory"
PROGRESS_SECONDS = 15
CANONICALIZE_CHUNK = 2048  # inputs one worker canonicalizes at a time
GIB = 1024**3


@dataclass(frozen=True, slots=True)
class EncoderBinding:
    """A loaded non-protein model: its family, its tokenizer and forward, and the pins the manifest records."""

    family: EncoderFamily
    hooks: EncoderHooks
    pins: Mapping[str, Any]  # {"model": {repository, revision, files}, "tokenizer": {...}} as verified before loading


def family_of(base: CanonicalBase) -> EncoderFamily:
    """The family of a non-protein base."""
    if base.family not in FAMILIES:
        raise ValueError(f"{base.id} is a {base.family} base; the encoder families are {sorted(FAMILIES)}.")
    return FAMILIES[base.family]


# --- Inputs ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FamilyInputs:
    """The distinct canonical inputs of a shard in file order, their SHA-256 keys, and the rows set aside."""

    inputs: list[CanonicalInput]
    keys: list[str]
    rejected: list[str]  # keys (or hashes of the raw value) of inputs the family could not canonicalize
    reasons: list[str]  # why, one line each, parallel to ``rejected``


def _canonicalize_chunk(
    function: Callable[[object], CanonicalInput], chunk: Sequence[object],
) -> list[tuple[CanonicalInput | None, str | None]]:
    """Each value of ``chunk`` as a canonical input or the reason it has none; a bug is never swallowed."""
    results: list[tuple[CanonicalInput | None, str | None]] = []
    for raw in chunk:
        try:
            results.append((function(raw), None))
        except ValueError as error:
            results.append((None, f"{type(error).__name__}: {error}"))
    return results


def canonicalize_all(
    family: EncoderFamily, values: Sequence[object], *, workers: int = 1,
) -> list[tuple[CanonicalInput | None, str | None]]:
    """Canonicalize ``values`` in order, over ``workers`` processes when more than one, with progress."""
    chunks = [values[start:start + CANONICALIZE_CHUNK] for start in range(0, len(values), CANONICALIZE_CHUNK)]
    results: list[tuple[CanonicalInput | None, str | None]] = []
    with tqdm(total=len(values), desc=f"canonicalize {family.id}", unit="row", mininterval=PROGRESS_SECONDS, file=sys.stderr) as bar:
        if workers <= 1:
            for chunk in chunks:
                results += _canonicalize_chunk(family.canonicalize, chunk)
                bar.update(len(chunk))
            return results
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for chunk, done in zip(chunks, pool.map(_canonicalize_chunk, [family.canonicalize] * len(chunks), chunks), strict=True):
                results += done
                bar.update(len(chunk))
    return results


def key_of(text: str) -> str:
    """The SHA-256 of a row's key text, which is its row key in the store (`fastplms.features.sequence_digest`)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_family_inputs(
    family: EncoderFamily, path: Path, *, digest_column: str | None, value_column: str, shard: tuple[int, int],
    limit: int | None, workers: int = 1, skip: bool = False,
) -> FamilyInputs:
    """The distinct canonical inputs of the parquet at ``path``, in file order, for shard ``index`` of ``count``.

    Each value is canonicalized (`EncoderFamily.canonicalize`), and the SHA-256 of the canonical text is its key. When the
    file also states a key per row (``digest_column`` present), it must equal that hash, so the file's keys are the store's
    keys; a disagreement is an error, never a second row. Rows whose values canonicalize to one text are one row. A value the
    family cannot read stops the job unless ``skip``, which sets it aside and records it. Shard ``index`` keeps the keys whose
    value is ``index`` modulo ``count``, and ``limit`` the first N of those.
    """
    import pyarrow.parquet as parquet

    names = set(parquet.read_schema(path).names)
    stated = digest_column if digest_column in names else None
    table = parquet.read_table(path, columns=[value_column, *([stated] if stated else [])])
    values = table.column(value_column).to_pylist()
    digests = table.column(stated).to_pylist() if stated else [None] * len(values)
    firsts: dict[object, int] = {}
    for position, raw in enumerate(values):
        firsts.setdefault(raw if isinstance(raw, str) else tuple(raw) if isinstance(raw, (list, tuple)) else repr(raw), position)
    canonical = dict(zip(firsts, canonicalize_all(family, [values[position] for position in firsts.values()], workers=workers), strict=True))
    index, count = shard
    inputs: list[CanonicalInput] = []
    keys: list[str] = []
    rejected: list[str] = []
    reasons: list[str] = []
    seen: set[str] = set()
    wrong = 0
    for raw, digest in zip(values, digests, strict=True):
        item, reason = canonical[raw if isinstance(raw, str) else tuple(raw) if isinstance(raw, (list, tuple)) else repr(raw)]
        if item is None:
            rejected.append(digest or key_of(repr(raw)))
            reasons.append(f"{reason} ({str(raw)[:60]})")
            continue
        key = key_of(item.text)
        if digest is not None and digest != key:
            wrong += 1
            continue
        if key in seen:
            continue
        seen.add(key)
        if int(key, 16) % count == index and (limit is None or len(inputs) < limit):
            inputs.append(item)
            keys.append(key)
    if wrong:
        raise ValueError(
            f"{wrong} rows of {path} state a key that is not the SHA-256 of their canonical {family.id} text "
            f"({family.normalization_version}); fix the file."
        )
    first_rejected = dict(zip(rejected, reasons, strict=True))
    if first_rejected and not skip:
        raise ValueError(
            f"{len(first_rejected)} {family.id} inputs cannot be read, such as {list(first_rejected.values())[:3]}; rerun with "
            "--skip-unrepresentable to set them aside and record them in the receipt."
        )
    if not inputs:
        raise ValueError(f"{path} has no rows in shard {index}/{count}.")
    return FamilyInputs(inputs, keys, list(first_rejected), list(first_rejected.values()))


@dataclass(frozen=True, slots=True)
class PreparedInputs:
    """The rows the model can read, by key text, and the keys and reasons of the inputs it cannot."""

    rows: dict[str, PreparedRow]
    rejected: list[str]
    reasons: list[str]


def prepare_rows(family: EncoderFamily, hooks: EncoderHooks, inputs: Sequence[CanonicalInput], keys: Sequence[str]) -> PreparedInputs:
    """Each input's token ids and facts, keyed by its text; every row's id count must be its units (cropped) plus the special rows.

    An input the model cannot read (`EncoderHooks.tokenize` gives a reason) is not a row: it is returned with its key and reason.
    """
    tokenized = hooks.tokenize(inputs)
    if len(tokenized) != len(inputs):
        raise ValueError("The tokenizer returned a different number of rows than it was given.")
    rows: dict[str, PreparedRow] = {}
    rejected: list[str] = []
    reasons: list[str] = []
    for item, key, tokens in zip(inputs, keys, tokenized, strict=True):
        if isinstance(tokens, str):
            rejected.append(key)
            reasons.append(tokens)
            continue
        expected = min(item.units, family.max_units) + family.special_rows
        if len(tokens.ids) != expected:
            raise ValueError(f"{item.text[:60]!r} has {len(tokens.ids)} token ids, not its {expected} rows.")
        rows[item.text] = PreparedRow(
            item.text, key, item.units, tokens.ids,
            {"units": item.units, "rows": len(tokens.ids), "truncated": item.units > family.max_units, **tokens.facts},
        )
    return PreparedInputs(rows, rejected, reasons)


def inventory_of(family: EncoderFamily, items: Sequence[CanonicalInput], keys: Sequence[str]) -> dict[str, SequenceKey]:
    """Text to `SequenceKey` for canonical inputs: the identity the contract and `FeatureView` hold rows to; no tokenizer needed."""
    return {item.text: SequenceKey(key, item.units, family.normalization_version) for item, key in zip(items, keys, strict=True)}


def canonical_inventory(base: CanonicalBase, values: Iterable[object]) -> dict[str, SequenceKey]:
    """The inventory `open_canonical_view` reads a non-protein store with: each distinct input's canonical text and its key.

    ``values`` are raw SMILES for the ligand base or term-id lists for the annotation base, in any spelling. Ask the view for a
    row by its canonical text, `family_of(base).canonicalize(value).text`; the inventory holds no tokenizer, only the key,
    the units before the crop and the normalization, which is all a reader checks the stored rows against.
    """
    family = family_of(base)
    items = {item.text: item for item in map(family.canonicalize, values)}
    return inventory_of(family, list(items.values()), [key_of(text) for text in items])


def row_inventory(family: EncoderFamily, rows: Mapping[str, PreparedRow]) -> dict[str, SequenceKey]:
    """`inventory_of` for prepared rows."""
    return {text: SequenceKey(row.sha256, row.units, family.normalization_version) for text, row in rows.items()}


def write_row_inventory(root: Path, rows: Mapping[str, PreparedRow]) -> Path:
    """Record each row's key, text, units, stored rows and facts as parquet under the store, once per distinct set of rows.

    The feature store keeps only a row's digest and its stored row count, never its text. This file is what makes the Hub
    repository readable without the dataset that fed it: the canonical SMILES and SELFIES of a ligand, its symbol count and its
    row count, or an annotation entry's terms. A rerun on the same rows finds the file and leaves it.
    """
    import pyarrow as pa
    import pyarrow.parquet as parquet

    name = hashlib.sha256("".join(sorted(row.sha256 for row in rows.values())).encode("ascii")).hexdigest()[:16]
    path = root / INVENTORY_DIRECTORY / f"{name}.parquet"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [{"sha256": row.sha256, "text": row.text, **row.facts} for row in rows.values()]
    temporary = path.with_name(path.name + ".writing")
    parquet.write_table(pa.Table.from_pylist(records), temporary)
    temporary.replace(path)
    return path


# --- Contract -------------------------------------------------------------------------------------------------------


def numerical_policy(device: torch.device, recorded_environment: Mapping[str, Any] | None) -> dict[str, Any]:
    """The arithmetic settings a forward runs under, and the machine, which a store keeps from its first run."""
    hardware: dict[str, Any] = {
        "machine": platform.machine(), "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
    }
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "autocast_enabled": torch.is_autocast_enabled(device.type),
        "autocast_dtype": str(torch.get_autocast_dtype(device.type)).removeprefix("torch."),
        "hardware": dict(recorded_environment["hardware"]) if recorded_environment is not None else hardware,
    }


def encoder_feature_specs(
    model: nn.Module,
    *,
    family: EncoderFamily,
    streams: Sequence[EncoderStream],
    model_source: Mapping[str, str],
    tokenizer_source: Mapping[str, str],
    engine_sha256: str,
    reducer_sha256: Mapping[str, str],
    normalization_version: str,
    max_length: int,
    truncate: bool,
    batch_size: int,
    batch_window_size: int,
    max_tokens_per_batch: int,
    dtype: torch.dtype | None = None,
    keep_special_tokens: bool = False,
    recorded_environment: Mapping[str, Any] | None = None,
    manifest_pins: Mapping[str, Any] | None = None,
) -> dict[str, FeatureSpec]:
    """Measure one `FeatureSpec` per stream from a loaded model and its pins, without running it.

    This is `capture_feature_specs` for a model that has no taps. ``model_source`` and ``tokenizer_source`` are the verified
    pins (the tokenizer's carries the digest of its files, or of the vocabulary release for annotations, because a loaded
    tokenizer's own fingerprint holds the path it was read from). The weights are hashed now, and `before_commit` recaptures
    them. The crop is a prefix at ``max_length`` units. ``manifest_pins`` is the manifest's record and enters no key. A family
    that keeps CLS and EOS is `feature_spec_v2`; one that has none is `feature_spec_v1`.
    """
    from fastplms.embeddings.identity import _model_state_sha256, _software_versions
    from fastplms.embeddings.pooling import POOLING_SEMANTICS, POOLING_SEMANTICS_TOKENS

    del manifest_pins  # recorded by the manifest, never part of a stream's identity
    keeps = family.keeps_special_tokens
    if keep_special_tokens != keeps or normalization_version != family.normalization_version or max_length != family.max_units:
        raise ValueError("The capture options differ from the family's row policy.")
    if dtype is not None or not truncate:
        raise ValueError("An encoder store runs the model's own dtype and crops long inputs from the end of the units.")
    if len(model_source) != 2 or set(tokenizer_source) != {"repository", "revision", "content_sha256"}:
        raise ValueError("model_source is {repository, revision}; tokenizer_source adds content_sha256.")
    parameter_dtypes = {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()}
    if len(parameter_dtypes) != 1:
        raise ValueError("Feature capture requires one explicit backbone parameter dtype.")
    (model_dtype,) = parameter_dtypes
    devices = {parameter.device for parameter in model.parameters()}
    if len(devices) != 1 or next(iter(devices)).type not in {"cpu", "cuda"}:
        raise ValueError("Feature capture qualifies single-device CPU or CUDA execution only.")
    (device,) = devices
    config = json.loads(model.config.to_json_string(use_diff=False))
    for name in ("_name_or_path", "_commit_hash", "transformers_version"):
        config.pop(name, None)  # these locate the checkpoint; they do not change what it computes
    model_pin = {
        **model_source, "content_sha256": _model_state_sha256(model),
        "config_sha256": _digest({"class": f"{type(model).__module__}.{type(model).__qualname__}", "configuration": config}),
    }
    software = {name: str(version) for name, version in _software_versions().items() if version is not None}
    for package in family.packages:
        software[package] = importlib.metadata.version(package)
    if recorded_environment is not None:
        software = dict(recorded_environment["software"])
    layer = family.final_layer(model)
    backend = getattr(model.config, "_attn_implementation", None) or "unspecified"
    policy = numerical_policy(device, recorded_environment)
    specs: dict[str, FeatureSpec] = {}
    for stream in streams:
        pooled = stream.pooling is not None
        semantics = POOLING_SEMANTICS_TOKENS if keeps else POOLING_SEMANTICS
        document: dict[str, Any] = {
            "schema": FEATURE_SCHEMA_V2 if keeps else FEATURE_SCHEMA, "kind": "dense_pooled" if pooled else "dense_residue",
            "model": model_pin, "tokenizer": dict(tokenizer_source), "layers": [layer], "layer_count": layer,
            "pooling": list(stream.pooling) if pooled and stream.pooling else None, "output_dtype": stream.dtype,
            "normalization_version": normalization_version, "mask_policy": MASK_POLICY_V2 if keeps else MASK_POLICY_V1,
            "reducer": {
                "name": "foundry_encoder_rows", "version": "encoder_rows_v1", "implementation_sha256": reducer_sha256[stream.name],
                "parameters": {
                    "family": family.id, "stack": stream.source, "unit": family.unit, "special_rows": family.special_rows,
                    "tokenization": dict(family.tokenization), "forward": dict(family.forward),
                    **({"pooling_semantics": dict(semantics)} if pooled else {}),
                    "input_dtype": str(model_dtype).removeprefix("torch."), "layer_numbering": "stack_final_v1",
                },
            },
            "execution": {
                "engine_sha256": engine_sha256, "compute_dtype": str(model_dtype).removeprefix("torch."),
                "accumulator_dtype": "float32", "attention_backend": backend, "device_type": device.type,
                "software": software, "numerical_policy": policy, "kernel_sha256": None,
                "batch_policy": {
                    "algorithm": BATCH_ALGORITHM, "max_sequences": batch_size, "max_tokens": max_tokens_per_batch,
                    "window_size": batch_window_size,
                },
            },
            "length": {"max_residues": max_length, "overlength": "prefix_crop"},
            "sae": None, "structural": None, "projection_sha256": None,
        }
        if keeps:
            document["special_tokens"] = SPECIAL_TOKENS_KEPT  # CLS and EOS rows are kept: l + 2 rows per sequence
        specs[stream.name] = FeatureSpec(document)
    return specs


class EncoderExtractionContract(FeatureExtractionContract):
    """`FeatureExtractionContract` for a family that is not protein: specs measured from a Hugging Face model, no taps.

    Everything that binds rows is the parent's: the stored features (layout, width and dtype from each spec), the key check
    on every row, the identity written beside each part, the cache validation and the recapture before a commit. Only the
    capture and the inventory check are the family's. A family without special tokens binds `feature_spec_v1` rows, which
    list their positions, where the parent binds ranges.
    """

    def _capture(self) -> dict[str, FeatureSpec]:
        return encoder_feature_specs(self._model, **self._capture_options)

    def _expected_inventory(self, sequences: Sequence[str]) -> None:
        from fastplms.features.store import sequence_digest

        family: EncoderFamily = self._capture_options["family"]
        for sequence in sequences:
            key = self._inventory.get(sequence)
            if key is None or key.sha256 != sequence_digest(sequence):
                raise ValueError("Row text does not match the independently supplied inventory.")
            if key.normalization_version != self._capture_options["normalization_version"]:
                raise ValueError("Inventory and feature normalization versions differ.")
            for spec in tuple(self.specs.values())[:1]:  # every stream shares one length policy
                spec.bind(key, range(spec.context_length(key) + family.special_rows) if family.keeps_special_tokens
                          else tuple(range(spec.context_length(key))))

    def row_identities(self, name: str, sequences: Sequence[str], digests: Sequence[str]) -> list[dict[str, object]]:
        if self.keep_special_tokens:
            return super().row_identities(name, sequences, digests)
        self.check_row_keys(sequences, digests)
        spec = self.specs[name]
        return [spec.bind(key, tuple(range(spec.context_length(key)))).to_dict() for key in (self._inventory[text] for text in sequences)]

    @property
    def manifest_policy(self) -> dict[str, Any]:
        """The fields of `canonical.json` that say what a row of this family is, beside the streams."""
        family: EncoderFamily = self._capture_options["family"]
        return {
            "family": family.id, "special_tokens": "kept" if family.keeps_special_tokens else "none",
            "row_layout": (f"row 0 CLS, rows 1..l {family.unit}, row l+1 SEP; padding is the only masked position"
                           if family.keeps_special_tokens else f"rows 1..l {family.unit}, no special tokens; padding is the only masked position"),
            "max_units": family.max_units, "unit": family.unit, "normalization": family.normalization_version,
            "sae_layer": None, "pins": self._capture_options.get("manifest_pins"),
        }


# --- Runtime --------------------------------------------------------------------------------------------------------


def module_files(names: Sequence[str]) -> dict[str, Path]:
    """Each module's source file under a stable logical name, so a digest is the same on every machine."""
    files: dict[str, Path] = {}
    for name in names:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(f"Cannot find the source of module {name}.")
        files[name.replace(".", "/") + ".py"] = Path(spec.origin)
    return files


def bind_encoder(
    base: CanonicalBase, family: EncoderFamily, model: nn.Module, hooks: EncoderHooks, *, model_source: Mapping[str, str],
    tokenizer_source: Mapping[str, str], checkpoints: Mapping[str, Any], pooled: bool = False,
    max_sequences: int | None = None, max_tokens: int | None = None, window: int = 65536,
) -> CanonicalRuntime:
    """The runtime of a loaded model: the capture options its contract measures with, and the evidence of its bytes.

    The engine digest binds every file the run executes, and each stream's reducer digest the files that decide its values:
    `encoder_rows`, the family's modules (which include the chemistry that makes a ligand's SELFIES) and, for a pooled
    stream, the pooling FastPLMs' token runs use. The engine digest, the library versions and the machine are evidence of
    where the first rows were made, which a store keeps (`canonical.inherit_recorded_evidence`).
    """
    from fastplms.embeddings import pooling

    wanted = tuple(stream for stream in family.streams if stream.name in base.store_streams(STORE_POOLED if pooled else STORE_TOKEN))
    code = module_files(("foundry.embedding.encoder_rows", *family.identity_modules))
    pooling_file = {"models/fastplms/embeddings/pooling.py": Path(pooling.__file__).resolve()}
    engine = file_identity({**code, **pooling_file})
    rows = file_identity(code)
    pooled_rows = file_identity({**code, **pooling_file})
    reducers = {stream.name: rows if stream.pooling is None else pooled_rows for stream in wanted}
    capture = dict(
        family=family, streams=wanted, model_source=dict(model_source), tokenizer_source=dict(tokenizer_source),
        engine_sha256=engine["sha256"], reducer_sha256={name: value["sha256"] for name, value in reducers.items()},
        normalization_version=family.normalization_version, max_length=family.max_units, truncate=True,
        batch_size=max_sequences or base.max_sequences, batch_window_size=window,
        max_tokens_per_batch=max_tokens or base.max_tokens, dtype=None, keep_special_tokens=family.keeps_special_tokens,
        manifest_pins={"model": checkpoints.get("model"), "tokenizer": checkpoints.get("tokenizer")},
    )
    evidence = {"checkpoints": dict(checkpoints), "engine": engine, "reducers": reducers}
    return CanonicalRuntime(base, model, (), capture, evidence, EncoderBinding(family, hooks, capture["manifest_pins"]))


def build_encoder_runtime(
    base: CanonicalBase, cache: Path, *, device: str = "cuda", fetch: bool = False, max_sequences: int | None = None,
    max_tokens: int | None = None, window: int = 65536, pooled: bool = False,
) -> CanonicalRuntime:
    """Verify the pinned checkpoint, load the model and its hooks, and bind the options a contract measures.

    ``fetch`` downloads the pinned files into ``cache`` first. A missing or altered file fails with its path before anything
    loads (`dependencies.verified_snapshot`).
    """
    from .dependencies import fetch_snapshot, verified_snapshot
    from .encoder_pins import TOKENIZER_FILES, fetch_vocabulary, pinned_source, pinned_vocabulary, verified_vocabulary

    family = family_of(base)
    source = pinned_source(base.id)
    if fetch:
        fetch_snapshot(source, cache)
    snapshot, pin = verified_snapshot(source, cache)
    model, hooks = family.load(snapshot, device)
    if int(model.config.hidden_size) != base.hidden_width:
        raise ValueError(f"The loaded model is {model.config.hidden_size} wide, not the {base.hidden_width} of {base.label}.")
    checkpoints: dict[str, Any] = {"model": {"repository": pin["repository"], "revision": pin["revision"], "files": pin["sha256"]}}
    if family.has_tokenizer:
        files = {path: digest for path, digest in pin["sha256"].items() if path in TOKENIZER_FILES}
        checkpoints["tokenizer"] = {"repository": pin["repository"], "revision": pin["revision"], "files": files}
        tokenizer_source = {"repository": pin["repository"], "revision": pin["revision"], "content_sha256": _digest({"files": files})}
    else:
        vocabulary = pinned_vocabulary(base.id)
        if fetch:
            fetch_vocabulary(vocabulary, cache)
        evidence = verified_vocabulary(vocabulary, cache)
        config = model.config
        if (int(config.vocab_size), int(config.pad_token_id), int(config.mask_token_id)) != (
            vocabulary.terms + len(vocabulary.specials), vocabulary.specials["<pad>"], vocabulary.specials["<mask>"],
        ):
            raise ValueError(f"The model's vocabulary (size {config.vocab_size}, pad {config.pad_token_id}, mask {config.mask_token_id}) "
                             f"is not the pinned release's ({vocabulary.terms} terms and {len(vocabulary.specials)} specials).")
        hooks.bind_vocabulary(vocabulary.terms)
        checkpoints["tokenizer"] = {"repository": vocabulary.repository, "revision": vocabulary.revision, **evidence}
        tokenizer_source = {"repository": vocabulary.repository, "revision": vocabulary.revision, "content_sha256": _digest(evidence)}
    return bind_encoder(
        base, family, model, hooks, model_source={"repository": pin["repository"], "revision": pin["revision"]},
        tokenizer_source=tokenizer_source, checkpoints=checkpoints, pooled=pooled, max_sequences=max_sequences,
        max_tokens=max_tokens, window=window,
    )


# --- The run --------------------------------------------------------------------------------------------------------


def embed_encoder_features(
    runtime: CanonicalRuntime,
    prepared: Mapping[str, PreparedRow],
    texts: Sequence[str],
    keys: Sequence[str],
    root: str | Path,
    features: Mapping[str, StoredFeature],
    *,
    contract: EncoderExtractionContract,
    metadata: Mapping[str, Any],
    max_sequences: int,
    max_tokens: int,
    window: int,
    part_bytes: int = GIB,
    segment_bytes: int = 8 * GIB,
    queue_bytes: int = 4 * GIB,
    workers: int = 4,
    verify_cached: bool = True,
    progress: Callable[[int], None] | None = None,
    on_plan: Callable[[int], None] | None = None,
) -> dict[str, tuple[SegmentReceipt, ...]]:
    """Fill each named feature under ``root`` from one pass over the rows it lacks; `embed_token_features` for these families.

    ``texts`` are the rows' key texts and ``keys`` their SHA-256; ``prepared`` holds each text's token ids. Batches hold at
    most ``max_sequences`` sequences and ``max_tokens`` padded rows, sorted by length inside windows of ``window``. Parts
    are about ``part_bytes``, a segment commits every ``segment_bytes`` across the streams, and at most ``queue_bytes`` of
    finished rows wait for the writer. Returns the committed segments of each stream that gained rows; an empty result means
    the model never ran. A killed run commits whole segments only, so a rerun resumes from the last one.
    """
    from fastplms.features import AsyncFeatureWriter, FeatureStore

    binding = runtime.encoder
    if binding is None:
        raise ValueError("This runtime has no encoder binding; it is a protein runtime.")
    family = binding.family
    streams = [stream for stream in family.streams if stream.name in features]
    if {stream.name for stream in streams} != set(features):
        raise ValueError(f"features must name streams of the {family.id} family; received {sorted(features)}.")
    if not texts or len(texts) != len(keys):
        raise ValueError("embed_encoder_features needs one key per text and at least one text.")
    contract.check_row_keys(texts, keys)  # a dict lookup per row: a caller's digest is never trusted
    contract.validate(runtime.model, (), features, (), {
        "max_length": family.max_units, "truncate": True, "batch_size": max_sequences, "batch_window_size": window,
        "max_tokens_per_batch": max_tokens, "dtype": None,
    })
    stores = {name: FeatureStore.open(root, spec, deep_verify=False) for name, spec in features.items()}
    wanted = {name: frozenset(set(keys) - store.present_digests(list(keys))) for name, store in stores.items()}
    if verify_cached:
        for name, store in stores.items():
            cached = [text for text, key in zip(texts, keys, strict=True) if key not in wanted[name]]
            if cached:
                contract.validate_cached(name, store, cached)
    chosen = [position for position, key in enumerate(keys) if any(key in group for group in wanted.values())]
    if not chosen:
        return {}
    order_texts = [texts[position] for position in chosen]
    order_keys = [keys[position] for position in chosen]
    if on_plan is not None:
        on_plan(len(order_texts))
    run = {
        "schema": RUN_SCHEMA, "family": family.id, "special_tokens": "kept" if family.keeps_special_tokens else "none",
        "max_units": family.max_units,
        "batch_policy": {"algorithm": BATCH_ALGORITHM, "max_sequences": max_sequences, "max_tokens": max_tokens, "window": window},
        "model_state_fingerprint": contract.model_state_sha256,
        "storage_policy": {"max_part_bytes": part_bytes, "segment_bytes": segment_bytes},
    }
    # Each stream's own missing rows name the segment, so a rerun after a partial commit never collides with it.
    wanted_digests = {
        features[name].key: hashlib.sha256("".join(sorted(group)).encode("ascii")).hexdigest() for name, group in wanted.items()
    }
    fingerprint = hashlib.sha256(json.dumps({"run": run, "wanted": wanted_digests}, sort_keys=True).encode("utf-8")).hexdigest()[:32]
    executor = EncoderExecutor(binding.hooks, prepared, streams, device=next(runtime.model.parameters()).device)
    writer = AsyncFeatureWriter(
        stores, fingerprint=fingerprint, metadata={**dict(metadata), **run}, wanted=wanted, row_records=contract.row_identities,
        part_bytes=part_bytes, segment_bytes=segment_bytes, queue_bytes=queue_bytes, before_commit=contract.before_commit,
        workers=workers, progress=progress,
    )
    try:
        for members in plan_row_batches([executor.rows_for(text) for text in order_texts], max_sequences=max_sequences,
                                        max_tokens=max_tokens, window=window):
            writer.submit(executor.run_batch([order_texts[i] for i in members], [order_keys[i] for i in members]))
    except BaseException:
        writer.abort()
        raise
    receipts = writer.close()
    return {name: tuple(group) for name, group in receipts.items() if group}


# --- The card -------------------------------------------------------------------------------------------------------


def encoder_hub_card(base: CanonicalBase, manifest: Mapping[str, Any]) -> str:
    """The README of a non-protein base's Hub dataset repo: what a row is, its layout, its pins and how to read it."""
    family = family_of(base)
    pooled = manifest.get("store_kind") == STORE_POOLED
    streams = "\n".join(f"| `{name}` | {record['layout']} | {record['width']} | {record['dtype']} |" for name, record in manifest["streams"].items())
    pins = manifest["pins"]
    model, repository = pins["model"], pins["model"]["repository"]
    kind = "pooled" if pooled else "token"
    if family.id == "ligand":
        identity = ("A row is one molecule, keyed by the SHA-256 of its canonical SMILES "
                    "(`foundry.chemistry.canonicalize_smiles`). The model reads its SELFIES, cut after a whole symbol at "
                    f"{manifest['max_units']} symbols. `inventory/*.parquet` lists each row's SMILES, SELFIES, symbol count and row count.")
        layout = ("A molecule of l symbols is l + 2 token rows: row 0 CLS, rows 1..l the symbols, row l + 1 SEP. Padding is the only "
                  "masked position, and a pooled vector (mean, then population variance) covers all l + 2 rows. ")
        states = ("`decoder_*` is the state Atlas-PLI reads (`BartModel.last_hidden_state`, the decoder teacher forced on the "
                  "molecule); `encoder_*` is the encoder's `last_hidden_state`.")
    else:
        identity = ("A row is one annotation entry, keyed by the SHA-256 of its distinct term ids in ascending order joined by "
                    "commas, which is the `annotation_id` of the annotation vocabulary releases, under the release the descriptors pin. "
                    "`inventory/*.parquet` lists each row's terms.")
        layout = ("An entry of l terms is l token rows; the model has no special tokens. Padding is the only masked position, and a "
                  "pooled vector (mean, then population variance) covers all l rows. ")
        states = "`final_*` is the model's last hidden state, after its final layer norm."
    where = (f"The per-token streams are in `{base.hub_repo}`." if pooled else
             f"This repository holds the pooled streams of its rows too; `{base.pooled_hub_repo}` holds them alone.")
    rows = f"{layout}{states} {where}"
    scope = "One vector per row." if pooled else "Per-token and pooled rows."
    return f"""---
pretty_name: {base.label} {kind} embeddings
tags: [embeddings, {family.id}]
---
# {base.label} {kind} embeddings

{scope} Model: `{repository}` at revision `{model['revision']}`, weights SHA-256 `{manifest['model_state_sha256']}`.

{identity}

{rows}

| stream | layout | width | dtype |
|---|---|---|---|
{streams}

Read it with `foundry.embedding.open_canonical_view(root, inventory)` (a `FeatureView`), or with `fastplms.features.FeatureStore`
directly. `canonical.json` lists the streams and pins each descriptor. The store only appends.
"""


__all__ = [
    "BATCH_ALGORITHM", "FAMILIES", "FAMILY_DEFAULT_COLUMNS", "INVENTORY_DIRECTORY", "EncoderBinding", "EncoderExtractionContract",
    "FamilyInputs", "PreparedInputs", "bind_encoder", "build_encoder_runtime", "canonical_inventory", "canonicalize_all", "embed_encoder_features", "encoder_feature_specs",
    "encoder_hub_card", "family_of", "inventory_of", "key_of", "module_files", "prepare_rows", "read_family_inputs", "row_inventory",
    "write_row_inventory",
]
