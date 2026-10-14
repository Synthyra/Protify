"""Complete representation contracts and sequence-row keys for the shared feature store.

Pins name independently verified artifacts; this module does not fetch or authenticate them.
Representation contracts exclude dataset membership and sequence content, so unrelated datasets
can share a store. Row keys add the canonical sequence identity and exact retained positions.

Three schemas coexist. `feature_spec_v1` rows are residue-only: a sequence of l residues has l rows.
`feature_spec_v2` keeps the special tokens, so it has l + 2 rows, row 0 CLS, rows 1..l residues,
row l + 1 EOS, and declares `special_tokens: kept`. The two never share a key. A v2 row key's
`retained_positions` index those token rows, so `len(retained_positions)` is always the number of
stored rows, v1 or v2. `feature_spec_v3` keeps v2's rows and is computed under an embedding profile
(`numerics`): its execution binds the profile's digest and the hardware class, its batch policy is the
profile's fixed geometry, and the library versions and machine that v2 keyed on are run evidence instead.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from .keys import FEATURE_DTYPES, FeatureKey, _pinned_revision, _repository, _slug_part


JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
FEATURE_SCHEMA = "feature_spec_v1"
FEATURE_SCHEMA_V2 = "feature_spec_v2"
FEATURE_SCHEMA_V3 = "feature_spec_v3"
ROW_SCHEMA = "feature_row_v1"
ROW_SCHEMA_V2 = "feature_row_v2"
FEATURE_SCHEMAS = frozenset({FEATURE_SCHEMA, FEATURE_SCHEMA_V2, FEATURE_SCHEMA_V3})
TOKEN_SCHEMAS = frozenset({FEATURE_SCHEMA_V2, FEATURE_SCHEMA_V3})  # the schemas whose rows keep CLS and EOS
SPECIAL_TOKENS_KEPT = "kept"
MASK_POLICY_V1 = "attention_mask_minus_special_tokens_v1"
MASK_POLICY_V2 = "attention_mask_all_tokens_v1"
# CLS before the residues and EOS after them.
SPECIAL_TOKEN_ROWS = 2
# The N-terminal crop in residues: with CLS and EOS it fills a 2,048-token context. The engine names the same
# number as `CANONICAL_MAX_RESIDUES` in fastplms.embeddings.token_batches, and a test keeps the two equal.
CANONICAL_MAX_RESIDUES = 2046
FEATURE_KINDS = frozenset({
    "dense_pooled", "dense_residue", "sae_pooled", "sae_residue", "structural_pooled", "structural_residue",
})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_VERSIONED_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
_FIELDS = frozenset({
    "schema", "kind", "model", "tokenizer", "layers", "layer_count", "pooling",
    "output_dtype", "normalization_version", "mask_policy", "reducer", "execution",
    "length", "sae", "structural", "projection_sha256",
})
_FIELDS_V2 = _FIELDS | {"special_tokens"}
_EXECUTION_FIELDS = frozenset({
    "engine_sha256", "compute_dtype", "accumulator_dtype", "attention_backend",
    "device_type", "software", "kernel_sha256", "batch_policy", "numerical_policy",
})
# A v3 key binds no source or library digest: the profile pins the library closure, and every run's canary must match
# its class's golden digests bitwise, so a code change that moves a value stops the run instead of moving a key.
_EXECUTION_FIELDS_V3 = (_EXECUTION_FIELDS - {"software", "engine_sha256"}) | {"profile", "hardware_class"}
_REDUCER_FIELDS = frozenset({"name", "version", "implementation_sha256", "parameters"})
_REDUCER_FIELDS_V3 = _REDUCER_FIELDS - {"implementation_sha256"}
_GEOMETRY_POLICY_FIELDS = frozenset({
    "algorithm", "bucket_tokens", "token_budget", "max_rows", "max_columns", "partial_batch",
})
_HARDWARE_CLASS_FIELDS = frozenset({"id", "device_type", "compute_capability", "multiprocessors"})


_ASCII_WHITESPACE = str.maketrans("", "", " \t\n\r\v\f")


def normalize_sequence(text: str) -> str:
    """The text a row is keyed by: ASCII whitespace removed and letters uppercased, no residue rewritten.

    This is the one place the canonical key text is made, and it is the rule of `protein_ascii_v1` and of the
    dataset census keys (`sha256` of this text's ASCII bytes). `sequence_inventory`, the reader and the job all
    call it, so two spellings of one protein ("acd", "ACD ", "A C D") are one row, never two. Whether every
    letter has a token is the tokenizer's question, answered by `ResidueVocabulary`, not this function's.
    """
    if not isinstance(text, str) or not text.isascii():
        raise ValueError("A protein sequence must be ASCII text.")
    normalized = text.translate(_ASCII_WHITESPACE).upper()
    if not normalized:
        raise ValueError("A protein sequence must not be empty.")
    return normalized


PROTEIN_NORMALIZATION = "protein_ascii_v1"


def spelling_normalizer(normalization_version: str) -> Callable[[str], str]:
    """The rule that turns a requested spelling into a row's key text, for the inventory's normalization.

    A protein is keyed by `normalize_sequence`, so a reader forgives its spelling (case and whitespace). Any other
    normalization (`canonical_smiles_v1`, `annotation_terms_v1`) made its key text from a molecule or a term list, and
    only that exact text names the row: uppercasing a SMILES would turn aromatic carbon into aliphatic and reach another
    molecule's key. A reader of such a store asks for the canonical text it was given by the dataset.
    """
    return normalize_sequence if normalization_version == PROTEIN_NORMALIZATION else str


def text_is_unit_count(normalization_version: str) -> bool:
    """Whether the key's `length` is the length of the key text, as a protein's is.

    For a ligand `length` counts the SELFIES symbols of the canonical SMILES and for an annotation entry its terms,
    which the text alone does not state: the row's identity record and `job verify` hold them to what the encoder reads.
    """
    return normalization_version == PROTEIN_NORMALIZATION


def _canonical_json(value: object) -> str:
    """Reject lossy key coercion, nonfinite numbers, and objects with unstable serialization."""
    def check(item: object) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
            return
        raise ValueError("Feature contracts require finite JSON values and string object keys.")

    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _fields(value: object, expected: set[str] | frozenset[str], where: str) -> dict[str, Any]:
    """`value` as a JSON object with exactly the `expected` fields, whose values are not yet checked."""
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{where} must have exactly these fields: {sorted(expected)}.")
    return value


def _sha256(value: object, where: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{where} must be a full lowercase SHA-256 digest.")


def _name(value: object, where: str) -> None:
    if not isinstance(value, str) or not _VERSIONED_NAME.fullmatch(value):
        raise ValueError(f"{where} must be a nonempty versioned identifier.")


def _positive(value: object, where: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{where} must be a positive integer.")


def _pin(value: object, *, role: str, config: bool = False) -> dict[str, Any]:
    fields = {"repository", "revision", "content_sha256"}
    if config:
        fields.add("config_sha256")
    pin = _fields(value, fields, role)
    _repository(pin["repository"], f"{role}.repository")
    pin["revision"] = _pinned_revision(pin["revision"], f"{role}.revision")
    _sha256(pin["content_sha256"], f"{role}.content_sha256")
    if config:
        _sha256(pin["config_sha256"], f"{role}.config_sha256")
    return pin


def _validate_spec(spec: dict[str, Any]) -> None:
    keeps_special_tokens = isinstance(spec, dict) and spec.get("schema") in TOKEN_SCHEMAS
    profiled = isinstance(spec, dict) and spec.get("schema") == FEATURE_SCHEMA_V3
    _fields(spec, _FIELDS_V2 if keeps_special_tokens else _FIELDS, "Feature specification")
    if spec["schema"] not in FEATURE_SCHEMAS or spec["kind"] not in FEATURE_KINDS:
        raise ValueError("Unsupported feature specification schema or kind.")
    if keeps_special_tokens and (
        spec["special_tokens"] != SPECIAL_TOKENS_KEPT or spec["mask_policy"] != MASK_POLICY_V2
    ):
        raise ValueError(
            f"{spec['schema']} must declare special_tokens={SPECIAL_TOKENS_KEPT!r} and "
            f"mask_policy={MASK_POLICY_V2!r}."
        )
    if not keeps_special_tokens and spec["mask_policy"] == MASK_POLICY_V2:
        raise ValueError(f"Only {sorted(TOKEN_SCHEMAS)} may use mask_policy {MASK_POLICY_V2!r}.")
    _pin(spec["model"], role="model", config=True)
    _pin(spec["tokenizer"], role="tokenizer")
    _positive(spec["layer_count"], "layer_count")
    layers = spec["layers"]
    if (
        not isinstance(layers, list) or not layers
        or any(type(layer) is not int or not 0 <= layer <= spec["layer_count"] for layer in layers)
        or layers != sorted(set(layers))
    ):
        raise ValueError("layers must be distinct sorted hidden-state indices in 0..layer_count.")
    _name(spec["normalization_version"], "normalization_version")
    _name(spec["mask_policy"], "mask_policy")
    if spec["output_dtype"] not in FEATURE_DTYPES:
        raise ValueError("output_dtype must name a supported floating dtype.")
    pooled = spec["kind"].endswith("_pooled")
    pooling = spec["pooling"]
    if pooled:
        if not isinstance(pooling, list) or not pooling:
            raise ValueError("A pooled feature requires ordered pooling names.")
        for name in pooling:
            _name(name, "pooling name")
        if len(set(pooling)) != len(pooling):
            raise ValueError("Pooling names must not repeat.")
    elif pooling is not None:
        raise ValueError("A residue feature must use pooling=null.")

    reducer = _fields(spec["reducer"], _REDUCER_FIELDS_V3 if profiled else _REDUCER_FIELDS, "reducer")
    _name(reducer["name"], "reducer.name")
    _name(reducer["version"], "reducer.version")
    if not profiled:
        _sha256(reducer["implementation_sha256"], "reducer.implementation_sha256")
    if not isinstance(reducer["parameters"], dict):
        raise ValueError("reducer.parameters must be an explicit JSON object, empty if unused.")
    execution = _fields(spec["execution"], _EXECUTION_FIELDS_V3 if profiled else _EXECUTION_FIELDS, "execution")
    if not profiled:
        _sha256(execution["engine_sha256"], "execution.engine_sha256")
    if execution["kernel_sha256"] is not None:
        _sha256(execution["kernel_sha256"], "execution.kernel_sha256")
    for dtype_field in ("compute_dtype", "accumulator_dtype"):
        if execution[dtype_field] not in FEATURE_DTYPES:
            raise ValueError(f"execution.{dtype_field} must name a supported floating dtype.")
    _name(execution["attention_backend"], "execution.attention_backend")
    _name(execution["device_type"], "execution.device_type")
    if not isinstance(execution["numerical_policy"], dict) or not execution["numerical_policy"]:
        raise ValueError("execution.numerical_policy must record the effective arithmetic settings.")
    if profiled:
        _validate_profiled_execution(execution)
    else:
        software = execution["software"]
        if not isinstance(software, dict) or not {"torch", "transformers"} <= software.keys():
            raise ValueError("execution.software must include torch and transformers versions.")
        for value in software.values():
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError("execution.software versions must be nonempty strings.")
        batch = _fields(execution["batch_policy"], {
            "algorithm", "max_sequences", "max_tokens", "window_size",
        }, "batch_policy")
        _name(batch["algorithm"], "batch_policy.algorithm")
        _positive(batch["max_sequences"], "batch_policy.max_sequences")
        _positive(batch["window_size"], "batch_policy.window_size")
        if batch["window_size"] < batch["max_sequences"]:
            raise ValueError("batch_policy.window_size must be at least max_sequences.")
        if batch["max_tokens"] is not None:
            _positive(batch["max_tokens"], "batch_policy.max_tokens")
    length = _fields(spec["length"], {"max_residues", "overlength"}, "length")
    if length["max_residues"] is not None:
        _positive(length["max_residues"], "length.max_residues")
    if length["overlength"] not in {"reject", "prefix_crop"}:
        raise ValueError("length.overlength must be reject or prefix_crop.")

    for role, needed in (
        ("sae", spec["kind"].startswith("sae_")),
        ("structural", spec["kind"].startswith("structural_")),
    ):
        if needed:
            _pin(spec[role], role=role)
        elif spec[role] is not None:
            raise ValueError(f"{role} is not a dependency of this feature kind.")
    if spec["kind"].startswith("structural_"):
        _sha256(spec["projection_sha256"], "projection_sha256")
    elif spec["projection_sha256"] is not None:
        raise ValueError("Only structural features carry projection_sha256.")


def _validate_profiled_execution(execution: dict[str, Any]) -> None:
    """The v3 execution fields: the profile's digest, the hardware class, and the fixed geometry as the batch policy."""
    profile = _fields(execution["profile"], {"id", "sha256"}, "execution.profile")
    _name(profile["id"], "execution.profile.id")
    _sha256(profile["sha256"], "execution.profile.sha256")
    hardware = _fields(execution["hardware_class"], _HARDWARE_CLASS_FIELDS, "execution.hardware_class")
    _name(hardware["id"], "execution.hardware_class.id")
    if hardware["device_type"] != execution["device_type"]:
        raise ValueError("execution.hardware_class.device_type must equal execution.device_type.")
    if hardware["device_type"] == "cuda":
        capability = hardware["compute_capability"]
        if (not isinstance(capability, list) or len(capability) != 2 or any(type(part) is not int for part in capability)):
            raise ValueError("A CUDA hardware class states its compute capability as [major, minor].")
        _positive(hardware["multiprocessors"], "execution.hardware_class.multiprocessors")
    elif hardware["compute_capability"] is not None or hardware["multiprocessors"] is not None:
        raise ValueError("A CPU hardware class states no compute capability or multiprocessors.")
    batch = _fields(execution["batch_policy"], _GEOMETRY_POLICY_FIELDS, "batch_policy")
    _name(batch["algorithm"], "batch_policy.algorithm")
    _name(batch["partial_batch"], "batch_policy.partial_batch")
    for name in ("bucket_tokens", "token_budget", "max_rows", "max_columns"):
        _positive(batch[name], f"batch_policy.{name}")


@dataclass(frozen=True, slots=True, init=False)
class FeatureSpec:
    """An immutable, validated representation contract, independent of sequence membership.

    The serialized copy prevents a nested caller mutation from changing an existing key.
    A pin is a required assertion, not proof of artifact acceptance or backbone compatibility.
    The parsed document and the key digest are computed once, because a store binds every
    row of a run to the same contract.
    """

    _json: str
    _parsed: dict[str, Any] = field(compare=False, hash=False, repr=False)
    _key_digest: str = field(compare=False, hash=False, repr=False)

    def __init__(self, specification: Mapping[str, object]) -> None:
        spec = json.loads(_canonical_json(dict(specification)))
        _validate_spec(spec)
        encoded = _canonical_json(spec)
        object.__setattr__(self, "_json", encoded)
        # Read-only inside this class; `to_dict` hands out a fresh copy.
        object.__setattr__(self, "_parsed", spec)
        object.__setattr__(self, "_key_digest", hashlib.sha256(encoded.encode("ascii")).hexdigest())

    @property
    def keeps_special_tokens(self) -> bool:
        """Whether rows hold CLS and EOS besides the residues (schemas v2 and v3)."""
        return self._parsed["schema"] in TOKEN_SCHEMAS

    @property
    def profile(self) -> dict[str, str] | None:
        """The embedding profile a v3 feature was computed under, `{"id", "sha256"}`; None for v1 and v2."""
        if self._parsed["schema"] != FEATURE_SCHEMA_V3:
            return None
        return dict(self._parsed["execution"]["profile"])

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a detached JSON document suitable for feature.json and release records."""
        return json.loads(self._json)

    @property
    def key(self) -> FeatureKey:
        """The full contract digest decides identity; the readable prefix is only a label."""
        spec = self._parsed
        slug = f"{spec['kind']}__{_slug_part(spec['model']['repository'])}__contract1"
        return FeatureKey(slug, self._key_digest)

    def context_length(self, sequence: SequenceKey) -> int:
        """Residues `l` of the inference prefix of `sequence`, after the declared crop, never counting CLS or EOS."""
        spec = self._parsed
        if sequence.normalization_version != spec["normalization_version"]:
            raise ValueError("Sequence and representation normalization versions differ.")
        length = spec["length"]
        limit = length["max_residues"]
        if limit is not None and sequence.length > limit and length["overlength"] == "reject":
            raise ValueError("Sequence exceeds the representation's declared length limit.")
        return sequence.length if limit is None else min(sequence.length, limit)

    def stored_rows(self, sequence: SequenceKey) -> int:
        """Rows a per-token feature stores for `sequence`: `l` in v1, `l + 2` in v2."""
        return self.context_length(sequence) + (SPECIAL_TOKEN_ROWS if self.keeps_special_tokens else 0)

    def bind(self, sequence: SequenceKey, retained_positions: Sequence[int]) -> FeatureRowKey:
        """Bind exact positions retained after the declared inference crop.

        v1 positions are original residue coordinates in `0..l-1`. v2 positions are token rows in
        `0..l+1`, row 0 CLS and row l + 1 EOS; a canonical v2 row retains `range(l + 2)`.
        """
        if not isinstance(sequence, SequenceKey):
            raise TypeError("sequence must be a SequenceKey from the canonical input inventory.")
        context_length = self.context_length(sequence)
        keeps = self.keeps_special_tokens
        row_count = context_length + (SPECIAL_TOKEN_ROWS if keeps else 0)
        positions: tuple[int, ...] | range
        if keeps and isinstance(retained_positions, range):
            positions = retained_positions
            valid = positions.step == 1 and 0 <= positions.start < positions.stop <= row_count
        else:
            positions = tuple(retained_positions)
            valid = (
                bool(positions)
                and all(type(position) is int and 0 <= position < row_count for position in positions)
                and tuple(sorted(set(positions))) == positions
            )
        if not valid:
            raise ValueError("Retained positions must be nonempty, distinct, ordered, and inside the inference prefix.")
        return FeatureRowKey(self._key_digest, sequence, context_length, positions, special_tokens=keeps)


@dataclass(frozen=True, slots=True)
class SequenceKey:
    """The builder's verified normalized-sequence SHA-256, length, and normalization version.

    Construct from the canonical inventory, never from an accession or raw unnormalized input.
    This value performs no second normalization or hashing of biological sequences.
    """

    sha256: str
    length: int
    normalization_version: str

    def __post_init__(self) -> None:
        _sha256(self.sha256, "sequence.sha256")
        _positive(self.length, "sequence.length")
        _name(self.normalization_version, "sequence.normalization_version")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "sha256": self.sha256, "length": self.length,
            "normalization_version": self.normalization_version,
        }


@dataclass(frozen=True, slots=True)
class FeatureRowKey:
    """One sequence-representation slice, with full digests and zero-based retained positions.

    With `special_tokens` false (v1) positions are residue coordinates `0..context_length-1`.
    With it true (v2) they are token rows `0..context_length+1`: row 0 CLS, rows 1..l residues,
    row l + 1 EOS. Either way `len(retained_positions)` is the number of stored rows.
    """

    feature_sha256: str
    sequence: SequenceKey
    context_length: int
    retained_positions: tuple[int, ...] | range
    special_tokens: bool = False

    def __post_init__(self) -> None:
        _sha256(self.feature_sha256, "feature_sha256")
        if not isinstance(self.sequence, SequenceKey):
            raise TypeError("sequence must be a SequenceKey.")
        _positive(self.context_length, "context_length")
        if self.context_length > self.sequence.length:
            raise ValueError("Inference context exceeds the original sequence length.")
        positions = self.retained_positions
        row_count = self.context_length + (SPECIAL_TOKEN_ROWS if self.special_tokens else 0)
        if isinstance(positions, range):
            valid = (positions.step == 1 and 0 <= positions.start < positions.stop <= row_count
                     and self.special_tokens)
        else:
            valid = (
                type(positions) is tuple and bool(positions)
                and all(type(position) is int and 0 <= position < row_count for position in positions)
                and tuple(sorted(set(positions))) == positions
            )
        if not valid:
            raise ValueError("retained_positions must be an ordered, nonempty tuple inside context_length.")

    @property
    def complete(self) -> bool:
        """Whether every row of the inference prefix is retained, CLS and EOS included for v2."""
        rows = self.context_length + (SPECIAL_TOKEN_ROWS if self.special_tokens else 0)
        positions = self.retained_positions
        return positions == range(rows) if isinstance(positions, range) else positions == tuple(range(rows))

    def to_dict(self) -> dict[str, JsonValue]:
        if not self.special_tokens:
            return {
                "schema": ROW_SCHEMA, "feature_sha256": self.feature_sha256,
                "sequence": self.sequence.to_dict(), "context_length": self.context_length,
                "retained_positions": list(self.retained_positions),
            }
        positions = self.retained_positions
        # A complete v2 row is a range, so a million rows cost no position lists.
        encoded: JsonValue = (
            {"start": positions.start, "stop": positions.stop} if isinstance(positions, range)
            else list(positions)
        )
        return {
            "schema": ROW_SCHEMA_V2, "feature_sha256": self.feature_sha256,
            "sequence": self.sequence.to_dict(), "context_length": self.context_length,
            "retained_positions": encoded, "special_tokens": SPECIAL_TOKENS_KEPT,
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    @classmethod
    def from_dict(cls, record: Mapping[str, object], *, feature: FeatureSpec) -> FeatureRowKey:
        """Reconstruct and verify a persisted row against its independently supplied contract."""
        keeps = feature.keeps_special_tokens
        row = _fields(dict(record), {
            "schema", "feature_sha256", "sequence", "context_length", "retained_positions",
            *(("special_tokens",) if keeps else ()),
        }, "Feature row")
        if (row["schema"] != (ROW_SCHEMA_V2 if keeps else ROW_SCHEMA)
                or row["feature_sha256"] != feature.key.digest):
            raise ValueError("Feature row schema or representation digest does not match.")
        if keeps and row["special_tokens"] != SPECIAL_TOKENS_KEPT:
            raise ValueError("Feature row special_tokens does not match its representation.")
        sequence = _fields(row["sequence"], {"sha256", "length", "normalization_version"}, "sequence")
        stored = row["retained_positions"]
        if keeps and isinstance(stored, dict):
            _fields(stored, {"start", "stop"}, "retained_positions")
            if type(stored["start"]) is not int or type(stored["stop"]) is not int:
                raise ValueError("Serialized retained_positions range must hold integers.")
            positions: Sequence[int] = range(stored["start"], stored["stop"])
        elif isinstance(stored, list):
            positions = stored
        else:
            raise ValueError("Serialized retained_positions must be a JSON array.")
        bound = feature.bind(SequenceKey(**sequence), positions)
        if type(row["context_length"]) is not int or row["context_length"] != bound.context_length:
            raise ValueError("Feature row context_length disagrees with the representation crop policy.")
        return bound
