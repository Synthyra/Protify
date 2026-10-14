"""Frozen native-store inputs for declared Protify datasets.

The exporter receives an independently qualified FeatureView. Its output binds cache bytes,
branch order and read budgets to one dataset manifest. It does not certify extraction origin
or dataset acceptance. Importing a cell uses validated declared sequence text, never accessions.
"""

from __future__ import annotations

import hashlib
import json

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from foundry.digests import sha256_file
from foundry.embedding import FeatureSpec, FeatureView, ReadBudget, SequenceKey
from .branch_features import IndexedBranchEmbeddings, parse_branch_spec
from .declared_dataset import checked_digest
from ..base_models.esmc_sae import parse_sae_model_name


SCHEMA = "protify_feature_input_v1"
FIELDS = {"schema", "dataset_sha256", "root", "branches", "budget", "max_queued_bytes"}


if TYPE_CHECKING:
    from fastplms.features import StoredFeature


def _regular_path(path: Path, *, validated_parents: set[Path] | None = None) -> Path:
    """Refuse aliases; optionally reuse parents within one immutable-input construction."""
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("native feature inputs require absolute paths without symlinks")
    pending_parents = []
    for parent in path.parents:
        if validated_parents is not None and parent in validated_parents:
            break
        if parent.is_symlink():
            raise ValueError("native feature inputs require absolute paths without symlinks")
        pending_parents.append(parent)
    if not path.is_file():
        raise ValueError(f"native feature input is not a regular file: {path}")
    # Absolute, alias-free paths without '..' are already canonical. This cache belongs
    # to one frozen-input construction; every leaf is still checked independently.
    if validated_parents is not None and ".." not in path.parts:
        validated_parents.update(pending_parents)
        return path
    resolved = path.resolve(strict=True)
    if validated_parents is not None:
        validated_parents.update(pending_parents)
    return resolved


def _branch_contract(token: str, descriptor: StoredFeature) -> None:
    spec = parse_branch_spec(token)
    contract = FeatureSpec(descriptor.descriptor).to_dict()
    kind = "sae_pooled" if spec.sparse else "dense_pooled"
    if (contract["kind"] != kind or tuple(contract["pooling"]) != spec.pooling_types
            or contract["normalization_version"] != "protein_ascii_v1"):
        raise ValueError("native branch kind, pooling or normalization differs from the declared branch")
    # Names are display bindings. The full immutable descriptor, not an alias or its width,
    # establishes the encoder identity, including compatible small ESMFold2 encoders.
    if spec.sparse:
        selection = parse_sae_model_name(spec.model_name)
        parameters = contract["reducer"]["parameters"]
        if (contract["layers"] != [selection.layer] or parameters.get("k") != selection.k
                or parameters.get("codebook_dim") != selection.codebook_dim):
            raise ValueError("native SAE layer, sparsity or codebook width differs from the declared branch")
    else:
        layer = spec.hidden_state_index
        if layer < 0:
            layer += int(contract["layer_count"]) + 1
        if contract["layers"] != [layer]:
            raise ValueError("native branch hidden state differs from its declared branch")


def feature_input_document(
    view: FeatureView, branch_specs: Sequence[str], dataset_sha256: str, *, max_queued_bytes: int,
) -> dict[str, Any]:
    """Freeze selected immutable files after qualification; caller persists and pins this JSON."""
    checked_digest(dataset_sha256)
    if len(branch_specs) != len(view.names) or len(set(branch_specs)) != len(branch_specs):
        raise ValueError("one distinct branch token is required per selected feature")
    if type(max_queued_bytes) is not int or max_queued_bytes < 2 * view.budget.max_bytes:
        raise ValueError("queued output capacity must allow producer and consumer batches")
    pins = view.content_pins()
    branches = []
    for token, name in zip(branch_specs, view.names, strict=True):
        descriptor = view.descriptor(name)
        _branch_contract(token, descriptor)
        branches.append({"spec": token, "feature": descriptor.payload(), "files": pins[name]})
    return {"schema": SCHEMA, "dataset_sha256": dataset_sha256, "root": view.root,
            "branches": branches, "budget": asdict(view.budget), "max_queued_bytes": max_queued_bytes}


class FrozenFeatureInput:
    """A separately pinned input document; its mutable SQLite index is deliberately excluded."""

    def __init__(
        self, path: str | Path, sha256: str, *, dataset_sha256: str, branch_specs: Sequence[str],
    ) -> None:
        from fastplms.features import FeatureStore, StoredFeature

        checked_digest(sha256)
        checked_digest(dataset_sha256)
        self.path = _regular_path(Path(path))
        encoded = self.path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != sha256:
            raise ValueError("native feature input differs from its independent SHA-256 pin")
        document = json.loads(encoded)
        if not isinstance(document, dict) or set(document) != FIELDS or document["schema"] != SCHEMA:
            raise ValueError("invalid native feature input schema")
        if document["dataset_sha256"] != dataset_sha256:
            raise ValueError("native feature input names a different dataset manifest")
        self.root = Path(document["root"])
        if not self.root.is_absolute() or any(part.is_symlink() for part in (self.root, *self.root.parents)):
            raise ValueError("native feature root must be absolute and without symlinks")
        budget = document["budget"]
        if not isinstance(budget, dict) or set(budget) != {"max_rows", "max_padded_residues", "max_bytes"}:
            raise ValueError("native feature input requires explicit reader limits")
        self.budget = ReadBudget(**budget)
        self.max_queued_bytes = document["max_queued_bytes"]
        if type(self.max_queued_bytes) is not int or self.max_queued_bytes < 2 * self.budget.max_bytes:
            raise ValueError("invalid queued feature output capacity")
        branches = document["branches"]
        if (not isinstance(branches, list) or not branches
                or any(not isinstance(row, dict) or set(row) != {"spec", "feature", "files"} for row in branches)
                or [row["spec"] for row in branches] != list(branch_specs)
                or len(set(branch_specs)) != len(branch_specs)):
            raise ValueError("native feature branch order differs from the declared cell")
        self.features, self.pins = {}, {}
        self._files = {str(self.path): sha256}
        validated_parents: set[Path] = set()
        for row in branches:
            token = row["spec"]
            descriptor = StoredFeature.from_payload(row["feature"])
            _branch_contract(token, descriptor)
            pins = row["files"]
            if not isinstance(pins, dict):
                raise ValueError("native feature files must map relative paths to SHA-256 pins")
            directory = self.root / descriptor.key
            # The native store owns the allowed immutable filenames and digest validation.
            FeatureStore(directory, descriptor, read_only=True, content_pins=pins)
            for relative, digest in pins.items():
                target = str(_regular_path(directory / relative, validated_parents=validated_parents))
                if target in self._files and self._files[target] != digest:
                    raise ValueError("native branches disagree on a feature file pin")
                self._files[target] = digest
            self.features[token], self.pins[token] = descriptor, dict(pins)

    @property
    def files(self) -> dict[str, str]:
        """Absolute immutable file names and expected digests, including this input document."""
        return dict(self._files)

    def verify_files(self) -> None:
        for name, digest in self._files.items():
            path = _regular_path(Path(name))
            if sha256_file(path) != digest:
                raise ValueError("native feature bytes differ from their independent content pin")

    def joined(
        self, sequences: Sequence[str], *, batch_size: int, workers: int, prefetch_factor: int = 2,
    ) -> IndexedBranchEmbeddings:
        """Qualify all selected rows, then retain only metadata and bounded native readers."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("native feature batch size must be positive")
        if self.budget.queued_bytes(workers=workers, prefetch_factor=prefetch_factor) > self.max_queued_bytes:
            raise ValueError("loader queues exceed the frozen feature output capacity")
        inventory = {sequence: SequenceKey(hashlib.sha256(sequence.encode("ascii")).hexdigest(), len(sequence), "protein_ascii_v1")
                     for sequence in sequences}
        view = FeatureView(self.root, self.features, inventory, budget=self.budget, content_pins=self.pins,
                           persistent_reader=True)
        joined = IndexedBranchEmbeddings(view)
        if batch_size > joined.block_rows:
            raise ValueError("loader batch exceeds the frozen feature read budget")
        return joined
