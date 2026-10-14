"""Enforce measured feature contracts at the existing FastPLMs cache/write boundary.

This adapter does not encode proteins or implement storage. It supplies validation callbacks
to ``fastplms.embeddings.embed_into_features`` and binds its measured residue positions to the
canonical inventory. FastPLMs imports stay lazy for exported probes without that package.

Two row policies share it. A legacy contract (`feature_spec_v1`) stores l residue rows per protein. A
canonical token contract (`keep_special_tokens=True`, `feature_spec_v2`) stores l + 2 rows (row 0 CLS, rows
1..l residues, row l + 1 EOS) in every per-token stream and pools over all of them; its rows are written by
FastPLMs' token executor and asynchronous writer, which call ``row_identities`` once per part. A profile
contract (`feature_spec_v3`) keeps those rows and runs every batch at the profile's fixed geometry; its
recapture before each commit measures the numerics flags again, so a flag changed mid-run stops the commit.

Symbols: l residues of a protein after the N-terminal crop; w stored columns of a stream.
"""

from __future__ import annotations

import torch

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from torch import nn

from .capture import capture_feature_specs
from .contracts import SPECIAL_TOKEN_ROWS, FeatureRowKey, FeatureSpec, SequenceKey
from .readers import checked_row_identity


if TYPE_CHECKING:
    from fastplms.embeddings import TapRecord
    from fastplms.embeddings.taps import Tap
    from fastplms.embeddings.token_batches import BatchGeometry
    from fastplms.features import FeatureStore, StoredFeature


_EXTRACTION_OPTIONS = (
    "max_length", "truncate", "batch_size", "batch_window_size", "max_tokens_per_batch", "dtype",
)
# A profile contract's options: the batch shapes come from its geometry, so no budget is stated.
_PROFILE_EXTRACTION_OPTIONS = ("max_length", "truncate", "dtype", "geometry")
_RUN_BATCH_OPTIONS = ("batch_size", "max_tokens_per_batch")


class FeatureExtractionContract:
    """A frozen contract snapshot with live model references for mandatory recapture.

    ``inventory`` maps exact normalized protein text to its independently verified ``SequenceKey``.
    Build it from the dataset's canonical inventory, never from accessions or a second normalization.
    ``capture_options`` are the explicit arguments of ``capture_feature_specs``. Pass this object
    as ``contract=`` to FastPLMs, its ``stored_features`` as the features, and its
    ``embedding_options`` as extraction arguments. Rebuild after changing a dependency. With
    ``keep_special_tokens=True`` among the capture options, the contract is the canonical l + 2 row
    policy, and ``embed_into_features`` reads it from ``keep_special_tokens`` on the contract, so a
    caller passes nothing extra and cannot run a canonical contract down the residue-only path.
    """

    def __init__(
        self, model: nn.Module, taps: Sequence[Tap], inventory: Mapping[str, SequenceKey],
        *, free_run_batch: bool = False, **capture_options: Any,
    ) -> None:
        """``free_run_batch`` lets a run batch other than the budget its keys state (a pooled store, see `key_batch`).

        With ``profile`` among the capture options the contract is a `feature_spec_v3` one: its batch shapes are the
        ``geometry`` capture option, and ``free_run_batch`` is refused, because a profile row is the same in any batch.
        """
        self.profiled = capture_options.get("profile") is not None
        if self.profiled and free_run_batch:
            raise ValueError("A profile contract fixes every batch shape; free_run_batch belongs to v2 pooled stores.")
        self._free_run_batch = free_run_batch
        required = _PROFILE_EXTRACTION_OPTIONS if self.profiled else _EXTRACTION_OPTIONS
        missing = set(required) - capture_options.keys()
        if missing:
            raise ValueError(f"Contract capture requires explicit extraction options: {sorted(missing)}.")
        if not inventory or any(not isinstance(key, SequenceKey) for key in inventory.values()):
            raise ValueError("Supply the verified sequence inventory as text-to-SequenceKey entries.")
        self._model = model
        self._taps = tuple(taps)
        self._inventory = MappingProxyType(dict(inventory))
        self._capture_options = dict(capture_options)
        self.keep_special_tokens = bool(capture_options.get("keep_special_tokens", False))
        self.specs: Mapping[str, FeatureSpec] = MappingProxyType(self._capture())
        self._expected_inventory(tuple(inventory))

    def _capture(self) -> dict[str, FeatureSpec]:
        """Measure the specs now, hashing the backbone's bytes: a weight changed anywhere, however it was written."""
        return capture_feature_specs(self._model, self._taps, **self._capture_options)

    @property
    def embedding_options(self) -> dict[str, Any]:
        """The extraction options a run states to the contract; a profile run states its geometry as recorded."""
        if self.profiled:
            options = {key: self._capture_options[key] for key in ("max_length", "truncate", "dtype")}
            options["geometry"] = self._capture_options["geometry"].describe()
            return options
        options = {key: self._capture_options[key] for key in _EXTRACTION_OPTIONS}
        if self._capture_options.get("fixed_batch_size") is not None:
            options["fixed_batch_size"] = self._capture_options["fixed_batch_size"]
        return options

    @property
    def geometry(self) -> BatchGeometry | None:
        """The fixed batch shapes of a profile contract; None for a v1 or v2 contract."""
        return self._capture_options.get("geometry") if self.profiled else None

    @property
    def model_initialization(self) -> Mapping[str, Any] | None:
        """How a model that is no released checkpoint was made, as its keys record it (the random control's seeded draw
        and, under a profile, its saved file); None for a released checkpoint."""
        return self._capture_options.get("model_initialization")

    @property
    def model_state_sha256(self) -> str:
        """The SHA-256 of the backbone's weights as measured when the contract was captured, for run metadata."""
        return str(next(iter(self.specs.values())).to_dict()["model"]["content_sha256"])


    @property
    def stored_features(self) -> dict[str, StoredFeature]:
        """Fresh physical descriptors derived from immutable measured contracts."""
        from fastplms.features import CSR, DENSE, RAGGED, RAGGED_TOPK, StoredFeature

        features: dict[str, StoredFeature] = {}
        for name, spec in self.specs.items():
            document = spec.to_dict()
            kind, parameters = document["kind"], document["reducer"]["parameters"]
            # width: stored columns per row; p * d, p * c or p * 256 pooled, d, 256, or c per token row.
            if kind.startswith("dense_"):
                width = self._model.config.hidden_size
                if kind == "dense_pooled":
                    width *= len(document["pooling"])
                layout = DENSE if kind == "dense_pooled" else RAGGED
            elif kind == "sae_pooled":
                layout, width = CSR, parameters["codebook_dim"] * len(document["pooling"])
            elif kind == "structural_residue":
                layout, width = RAGGED, parameters["output_width"]
            elif kind == "structural_pooled":
                layout, width = DENSE, parameters["output_width"] * len(document["pooling"])
            elif kind == "sae_residue":
                layout, width = RAGGED_TOPK, parameters["codebook_dim"]
            else:
                raise ValueError(f"No qualified persisted tap adapter for {kind!r}.")
            features[name] = StoredFeature(
                key=spec.key.name, layout=layout, width=width, dtype=getattr(torch, document["output_dtype"]),
                descriptor=document, sparse_count=parameters["k"] if kind == "sae_residue" else None,
            )
        return features

    def _expected_inventory(self, sequences: Sequence[str]) -> None:
        from fastplms.embeddings.batches import canonical_residue_ids
        from fastplms.embeddings.tokens import ResidueVocabulary
        from fastplms.features.store import sequence_digest

        # The token policy checks residues through the lookup table, verified against the tokenizer once.
        vocabulary = ResidueVocabulary(self._model.tokenizer) if self.keep_special_tokens else None
        for sequence in sequences:
            key = self._inventory.get(sequence)
            if key is None or key.sha256 != sequence_digest(sequence) or key.length != len(sequence):
                raise ValueError("Protein text does not match the independently supplied sequence inventory.")
            if key.normalization_version != self._capture_options["normalization_version"]:
                raise ValueError("Inventory and feature normalization versions differ.")
            if vocabulary is None:
                canonical_residue_ids(sequence, self._model.tokenizer)
            else:
                vocabulary.encode(sequence)  # (l + 2,); raises on a residue the tokenizer lacks
            # bind also enforces the original-length rejection policy, even on cache hits. A token contract's
            # specs share one length policy, so one bind per sequence covers them all.
            for spec in tuple(self.specs.values())[: 1 if self.keep_special_tokens else None]:
                spec.bind(key, range(spec.context_length(key) + SPECIAL_TOKEN_ROWS) if self.keep_special_tokens else (0,))

    def validate(
        self, model: nn.Module, sequences: Sequence[str], features: Mapping[str, StoredFeature],
        taps: Sequence[Tap], options: Mapping[str, Any],
    ) -> None:
        if model is not self._model or len(taps) != len(self._taps) or any(
            actual is not expected for actual, expected in zip(taps, self._taps, strict=True)
        ):
            raise ValueError("Feature contract was captured for a different model or tap plan.")
        measured = self.embedding_options
        if self._free_run_batch:  # the keys state the seed's budget; the run's own is recorded in each segment
            options = {key: value for key, value in options.items() if key not in _RUN_BATCH_OPTIONS}
            measured = {key: value for key, value in measured.items() if key not in _RUN_BATCH_OPTIONS}
        if dict(options) != measured:
            raise ValueError("Extraction options differ from the measured feature contract.")
        self.before_commit()
        if dict(features) != self.stored_features:
            raise ValueError("Stored feature key, layout, width, dtype or full descriptor differs from capture.")
        self._expected_inventory(sequences)

    def before_commit(self) -> None:
        """Recapture every live dependency immediately before the writer's visibility boundary."""
        observed = self._capture()
        if dict(self.specs) != observed:
            raise ValueError("Feature computation changed after capture; the segment cannot be committed.")

    def _validate_row(self, name: str, sequence: str, record: Mapping[str, object]) -> FeatureRowKey:
        return checked_row_identity(record, self.specs[name], self._inventory[sequence])

    def validate_cached(self, name: str, store: FeatureStore, sequences: Sequence[str]) -> None:
        """Validate complete contracts and committed row identities before any cache hit is reused."""
        if store.spec != self.stored_features[name]:
            raise ValueError("Cached feature descriptor differs from the measured full contract.")
        identities = store.row_metadata(sequences, verify_data=True)
        # The store answers each query with one entry per requested sequence.
        for sequence, record, residues in zip(sequences, identities, store.residue_counts(sequences), strict=True):
            key = self._validate_row(name, sequence, record)
            if store.spec.layout in {"ragged", "ragged_topk"} and residues != len(key.retained_positions):
                raise ValueError("Cached tensor residue count differs from its row identity.")

    def check_row_keys(self, sequences: Sequence[str], digests: Sequence[str]) -> None:
        """Raise unless each digest is the inventory's key for its sequence: a dict lookup per row, no hashing.

        A caller's digests choose which rows a run skips and name the rows it writes, so none is trusted.
        """
        if len(sequences) != len(digests):
            raise ValueError("A token run needs one row key per sequence.")
        for sequence, digest in zip(sequences, digests, strict=True):
            key = self._inventory.get(sequence)
            if key is None or key.sha256 != digest:
                raise ValueError("A row key differs from the sequence inventory's key for its protein.")

    def row_identities(self, name: str, sequences: Sequence[str], digests: Sequence[str]) -> list[dict[str, object]]:
        """Persisted identity of each token row of a canonical run: all l + 2 rows of the inference prefix.

        ``digests`` are the rows' SHA-256 keys; each must be the inventory's key for its text. An identity binds
        the original text's length, whose crop becomes l.
        """
        if not self.keep_special_tokens:
            raise ValueError("Only a token contract binds rows by key; a residue contract binds engine records.")
        self.check_row_keys(sequences, digests)
        spec = self.specs[name]
        identities: list[dict[str, object]] = []
        for sequence in sequences:
            key = self._inventory[sequence]  # the inventory's key: its sha256 equals the digest, checked above
            identities.append(spec.bind(key, range(spec.context_length(key) + SPECIAL_TOKEN_ROWS)).to_dict())
        return identities

    def bind_rows(self, name: str, records: Sequence[TapRecord]) -> list[dict[str, object]]:
        from fastplms.features import TopKRow
        from fastplms.features.layouts import validate_topk

        if self.keep_special_tokens:
            raise ValueError("A token contract writes through embed_token_features and row_identities.")
        identities: list[dict[str, object]] = []
        stored = self.stored_features[name]
        for record in records:
            if record.retained_positions is None:
                raise ValueError("Canonical feature rows require engine-verified residue positions.")
            key = self.specs[name].bind(self._inventory[record.sequence], record.retained_positions)
            identity = key.to_dict()
            self._validate_row(name, record.sequence, identity)
            # r = len(key.retained_positions), w = stored.width: a TopKRow of (r, k), (r, w) ragged, or (w,) pooled.
            value = record.tensors[name]
            if stored.layout == "ragged_topk":
                if not isinstance(value, TopKRow):
                    raise ValueError("A sparse residue contract requires TopKRow outputs.")
                validate_topk(value.indices, value.values, stored.width, stored.sparse_count)
                if value.values.dtype != stored.dtype or value.values.shape[0] != len(key.retained_positions):
                    raise ValueError("Sparse residue dtype or row count differs from its contract.")
                identities.append(identity)
                continue
            if not isinstance(value, torch.Tensor):
                raise ValueError("A dense or pooled contract requires tensor outputs.")
            expected_shape = ((len(key.retained_positions), stored.width)  # (r, w)
                              if stored.layout == "ragged" else (stored.width,))  # (w,)
            if value.dtype != stored.dtype or tuple(value.shape) != expected_shape or not bool(torch.isfinite(value).all()):
                raise ValueError("Feature tensor dtype, dimensions or finiteness disagree with its contract.")
            identities.append(identity)
        return identities
