"""Read pinned, typed four-way releases without inference, resampling, or truncation.

This reader verifies release bytes and row membership. Experimental provenance and fresh
homology audits belong to the release builder; a digest alone does not establish either.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import pyarrow as pa
import pyarrow.parquet as pq

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from datasets import Dataset, config as dataset_config

from foundry.digests import sha256_file
from .label_batches import SparseLabelRow


SPLIT_NAMES = ("train", "validation", "test", "calibration")
SCHEMA = "protify_declared_dataset_v1"
SPARSE_SCHEMA = "protify_declared_dataset_v2"
SPARSE_ENCODING = "sparse_observed_v1"
REQUIRED_COLUMNS = frozenset({"observation_id", "protein_id", "cluster_id", "seqs", "labels", "label_mask"})
SHA256 = re.compile(r"[0-9a-f]{64}")
PROTEIN = re.compile(r"[ACDEFGHIKLMNPQRSTVWYBXZJUO]+")


def declared_target_fields(kind: str, *, compact: bool = False) -> list[pa.Field]:
    """Physical Arrow target columns for the declared dense or compact representation."""
    if kind not in {"binary", "multiclass", "multilabel", "regression"} or (compact and kind != "multilabel"):
        raise ValueError("compact targets require a supported multilabel task")
    label_type, mask_type = pa.float64(), pa.bool_()
    if compact:
        label_type = pa.struct([
            pa.field("width", pa.int32(), nullable=False),
            pa.field("indices", pa.list_(pa.int64()), nullable=False),
            pa.field("values", pa.list_(pa.float32()), nullable=False),
        ])
        mask_type = pa.struct([
            pa.field("default_observed", pa.bool_(), nullable=False),
            pa.field("exceptions", pa.list_(pa.int64()), nullable=False),
        ])
    elif kind == "multilabel":
        label_type, mask_type = pa.list_(label_type), pa.list_(mask_type)
    return [pa.field("labels", label_type, nullable=False), pa.field("label_mask", mask_type, nullable=False)]


def checked_digest(value: object) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError("an immutable lowercase SHA-256 digest is required")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")
    return value


def _artifact(root: Path, record: dict[str, Any]) -> Path:
    """Resolve a content-pinned artifact inside the release directory."""
    relative = Path(_text(record["path"], "artifact path"))
    if relative.suffix != ".parquet":
        raise ValueError("declared split artifacts must be Parquet files")
    path = (root / relative).resolve(strict=True)
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("release artifacts must stay inside the manifest directory")
    if sha256_file(path) != checked_digest(record["sha256"]):
        raise ValueError(f"artifact digest mismatch: {relative}")
    return path


def _every_label_observed(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """A mask column that marks every label of each row observed, in the row's own dense or compact encoding."""
    observed = [{"default_observed": True, "exceptions": []} if isinstance(label, dict) else [True] * len(label) for label in batch["labels"]]
    return {"label_mask": observed}


@dataclass(frozen=True)
class DeclaredDataset:
    manifest: dict[str, Any]
    manifest_sha256: str
    splits: dict[str, Dataset]

    def read_unlisted_as_negative(self) -> DeclaredDataset:
        """The release read as a closed world: a label a row does not carry is a negative, so every label is observed.

        Meant for a release whose labels are already closed upward, such as Gene Ontology terms with their ancestors, so
        nothing is propagated here. Only the observation masks change; the manifest and its digest stay the release's own.
        """
        if self.manifest["task_type"] != "multilabel":
            raise ValueError("only a multilabel release can be read as a closed world")
        closed = {
            name: split.map(_every_label_observed, batched=True, features=split.features, desc=f"closed world {name}")
            for name, split in self.splits.items()
        }
        return replace(self, splits=closed)

    @property
    def label_type(self) -> str:
        kind = self.manifest["task_type"]
        return "singlelabel" if kind in {"binary", "multiclass"} else kind

    @property
    def num_labels(self) -> int:
        kind = self.manifest["task_type"]
        return 2 if kind == "binary" else 1 if kind == "regression" else len(self.manifest["label_vocabulary"])

    def training_record(self) -> dict[str, Any]:
        """Portable release identity, including label order and transformation meaning."""
        return {"manifest_sha256": self.manifest_sha256, **self.manifest}


def validate_declared_manifest(manifest: dict[str, Any]) -> None:
    required = {"schema", "dataset_id", "endpoint", "task_type", "units", "target_transform", "label_vocabulary", "evidence_policy_sha256", "split_audit_sha256", "splits"}
    compact = manifest.get("schema") == SPARSE_SCHEMA
    if compact:
        required.add("label_encoding")
    if set(manifest) != required or manifest["schema"] not in {SCHEMA, SPARSE_SCHEMA}:
        raise ValueError("unsupported or incomplete declared dataset manifest")
    if compact and (manifest["task_type"] != "multilabel" or manifest["label_encoding"] != SPARSE_ENCODING):
        raise ValueError("version-2 releases require multilabel sparse_observed_v1 targets")
    for name in ("dataset_id", "endpoint", "units"):
        _text(manifest[name], name)
    if re.fullmatch(r"[A-Za-z0-9._-]+", manifest["dataset_id"]) is None or manifest["dataset_id"] in {".", ".."}:
        raise ValueError("dataset_id must be safe as a single path component")
    kind = manifest["task_type"]
    if kind not in {"binary", "multiclass", "multilabel", "regression"}:
        raise ValueError("the release must declare a supported task_type")
    vocabulary = manifest["label_vocabulary"]
    if not isinstance(vocabulary, list) or not vocabulary or any(not isinstance(label, str) for label in vocabulary):
        raise ValueError("label_vocabulary must be a nonempty ordered list of strings")
    for label in vocabulary:
        _text(label, "vocabulary label")
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("label_vocabulary contains duplicates")
    if kind == "binary" and len(vocabulary) != 2:
        raise ValueError("binary releases name classes zero and one in that order")
    if kind == "regression" and len(vocabulary) != 1:
        raise ValueError("regression releases name exactly one target")
    if kind == "multiclass" and len(vocabulary) < 2:
        raise ValueError("multiclass releases need at least two classes")
    transform = manifest["target_transform"]
    if not isinstance(transform, dict) or set(transform) != {"name", "source_units", "inverse"}:
        raise ValueError("target_transform requires name, source_units, and inverse")
    if (transform["name"], transform["inverse"]) not in {("identity", "identity"), ("log10", "pow10")}:
        raise ValueError("unsupported target transformation or inverse")
    if kind != "regression" and transform["name"] != "identity":
        raise ValueError("classification targets must use the identity transformation")
    _text(transform["source_units"], "source_units")
    checked_digest(manifest["evidence_policy_sha256"])
    checked_digest(manifest["split_audit_sha256"])
    if not isinstance(manifest["splits"], dict) or set(manifest["splits"]) != set(SPLIT_NAMES):
        raise ValueError("the release must contain exactly train, validation, test, and calibration")
    for record in manifest["splits"].values():
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "rows"}:
            raise ValueError("every split requires path, sha256, and rows")
        if type(record["rows"]) is not int or record["rows"] < 1:
            raise ValueError("every declared split must contain at least one row")
        _text(record["path"], "artifact path")
        checked_digest(record["sha256"])


def _check_target(row: dict[str, Any], kind: str, width: int, *, compact: bool = False) -> None:
    targets, mask = row["labels"], row["label_mask"]
    if compact:
        sparse = SparseLabelRow.from_records(targets, mask)
        if sparse.width != width:
            raise ValueError("sparse targets must match the frozen vocabulary width")
        if any(value != 1 for value in targets["values"]):
            raise ValueError("observed binary targets must be zero or one")
        return
    if kind == "multilabel":
        if not isinstance(targets, list) or not isinstance(mask, list) or len(targets) != width or len(mask) != width:
            raise ValueError("multilabel targets and masks must match the frozen vocabulary width")
    else:
        if isinstance(targets, list) or isinstance(mask, list):
            raise ValueError("scalar tasks require scalar labels and masks")
        targets, mask = [targets], [mask]
    for value, observed in zip(targets, mask, strict=True):
        if type(observed) is not bool:
            raise ValueError("observation masks must contain booleans")
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError("release targets must be finite numbers, with zero fillers for unknowns")
        if not observed and value != 0:
            raise ValueError("unknown release targets require zero fillers and false masks")
        if observed and kind in {"binary", "multilabel"} and value not in {0, 1}:
            raise ValueError("observed binary targets must be zero or one")
        if observed and kind == "multiclass" and (value != int(value) or not 0 <= value < width):
            raise ValueError("observed classes must index the frozen vocabulary")


def load_declared_dataset(
    path: str | Path,
    expected_sha256: str,
    *,
    max_sequence_length: int | None = None,
    split_names: Sequence[str] = SPLIT_NAMES,
) -> DeclaredDataset:
    """Read verified Parquet rows, preserving their order, counts, labels, and partitions.

    Validation scans Arrow batches rather than constructing dense NumPy label matrices.
    Dataset loading uses Hugging Face's external Arrow cache. Sparse vocabulary projection
    and the release builder's source/homology audits are separate contracts.

    Only `split_names` are opened and byte-verified. Membership checks cover those splits
    together, not excluded payloads. The complete manifest remains the release identity;
    `splits` names precisely the verified payloads. This permits a sealed test file to be
    absent during calibration without inventing an empty test dataset.
    """
    split_names = tuple(split_names)
    if not split_names or len(set(split_names)) != len(split_names) or not set(split_names) <= set(SPLIT_NAMES):
        raise ValueError("split_names must be a nonempty, unique selection of declared splits")
    path = Path(path).expanduser().resolve(strict=True)
    if path.suffix != ".json":
        raise ValueError("declared dataset manifests must be JSON files")
    digest = sha256_file(path)
    if digest != checked_digest(expected_sha256):
        raise ValueError("declared dataset manifest digest mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_declared_manifest(manifest)
    kind, width = manifest["task_type"], len(manifest["label_vocabulary"])
    observations: set[str] = set()
    protein_splits: dict[str, str] = {}
    cluster_splits: dict[str, str] = {}
    protein_clusters: dict[str, str] = {}
    splits = {}
    for name in split_names:
        record = manifest["splits"][name]
        artifact = _artifact(path.parent, record)
        parquet = pq.ParquetFile(artifact)
        if not set(parquet.schema_arrow.names) >= REQUIRED_COLUMNS:
            raise ValueError(f"{name} lacks declared row identity, sequence, target, or mask columns")
        if parquet.metadata.num_rows != record["rows"]:
            raise ValueError(f"{name} row count differs from its manifest")
        for batch in parquet.iter_batches(batch_size=8192, columns=sorted(REQUIRED_COLUMNS)):
            for row in batch.to_pylist():
                observation = _text(row["observation_id"], "observation_id")
                if observation in observations:
                    raise ValueError("observation ids must be globally unique across release rows")
                observations.add(observation)
                sequence = _text(row["seqs"], "seqs")
                if PROTEIN.fullmatch(sequence) is None:
                    raise ValueError("release sequences must already be normalized amino-acid strings")
                if max_sequence_length is not None and len(sequence) > max_sequence_length:
                    raise ValueError("a frozen release sequence exceeds the feature budget; truncation and row dropping are forbidden")
                protein = checked_digest(row["protein_id"])
                if hashlib.sha256(sequence.encode("ascii")).hexdigest() != protein:
                    raise ValueError("protein_id differs from its sequence SHA-256")
                cluster = _text(row["cluster_id"], "cluster_id")
                if protein_splits.setdefault(protein, name) != name or cluster_splits.setdefault(cluster, name) != name:
                    raise ValueError("a protein or declared cluster crosses frozen splits")
                if protein_clusters.setdefault(protein, cluster) != cluster:
                    raise ValueError("a protein has inconsistent declared cluster membership")
                _check_target(row, kind, width, compact=manifest["schema"] == SPARSE_SCHEMA)
        # The builder's config id binds full content rather than file metadata. Keeping
        # the digest out of cache_dir avoids its duplication in Windows lock filenames.
        # https://huggingface.co/docs/datasets/package_reference/builder_classes
        cache = Path(dataset_config.HF_DATASETS_CACHE) / ("pd1" if manifest["schema"] == SCHEMA else "pd2")
        splits[name] = Dataset.from_parquet(str(artifact), cache_dir=str(cache), config_id=record["sha256"])
        if sha256_file(artifact) != record["sha256"]:
            raise ValueError(f"{name} artifact changed during loading")
    if sha256_file(path) != digest:
        raise ValueError("declared dataset manifest changed during loading")
    return DeclaredDataset(manifest, digest, splits)
