"""Resolve, fetch and load the pinned checkpoints behind a canonical embedding store.

A canonical store reads four pinned artifacts: the ESMC base, its tokenizer, the Biohub SAE of the base's depth
layer, and the released ESMFold2 layer mixture and projection. This module selects each from the FastPLMs model
registry (never a Hub lookup or a fallback), downloads missing files at their pinned revision, checks every file
against its registered digest before anything loads, and loads the SAE and the structural tensors. It imports
FastPLMs and torch only inside functions, so importing it, and the `foundry.embedding` package it sits in, needs neither.

This selection takes no suite manifest, so a job with only a base id can run, and `production_suite.feature_dependencies`
wraps it: the suite's planning layer selects and identifies the pins through `resolve_dependencies` and
`checkpoint_identity` without importing torch, and loads the SAE and the structural tensors through `load_sae`,
`structural_weights` and `verified_snapshot` here. A contract failure raises `DependencyContractError`, a `ValueError`, so
a caller with its own error type can catch that one class and re-raise it.

Symbols: d hidden width of the base; n = depth + 1 hidden states of the structural mixture; c SAE codebook;
k SAE codes kept per token.
"""

from __future__ import annotations

import json
import re

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict


if TYPE_CHECKING:
    from fastplms.models.esm_plusplus.modeling_esm_plusplus_sae import ESMplusplusSAELayer
    from fastplms.registry import CheckpointSource, ModelRegistry, SparseAutoencoderSpec
    from torch import Tensor


# The registry model whose structural projection each base reads.
STRUCTURAL_MODELS = {"esmc_small": "esmfold2_300", "esmc_large": "esmfold2_600", "esmc_6b": "esmfold2"}
# The released ESMFold2 tensors the structural stream needs, by the name `structural_tap` uses for each.
STRUCTURAL_KEYS = {
    "layer_logits": "language_model.base_z_combine",
    "projection.0.weight": "language_model.base_z_linear.0.weight",
    "projection.0.bias": "language_model.base_z_linear.0.bias",
    "projection.1.weight": "language_model.base_z_linear.1.weight",
}
SAE_K = 64
SAE_CODEBOOK = 16384  # c
_TOKENIZER_FILES = frozenset({"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"})


class DependencyContractError(ValueError):
    """A pinned dependency is unresolved, mutable, mismatched or inconsistent with what a canonical store requires."""


class CheckpointIdentity(TypedDict):
    repository: str
    revision: str
    # Each file's registry digest, as `algorithm:hex`, by its path in the repository.
    files: dict[str, str]


class SnapshotEvidence(CheckpointIdentity):
    # Each file's SHA-256 as read from the cached snapshot, by its path.
    sha256: dict[str, str]


@dataclass(frozen=True)
class CanonicalDependencies:
    """The pinned artifacts of one base's canonical store."""

    base_id: str
    base: CheckpointSource
    tokenizer: CheckpointSource
    sae: SparseAutoencoderSpec
    structural_model_id: str
    structural: CheckpointSource


def checkpoint_identity(source: CheckpointSource) -> CheckpointIdentity:
    """Serialize one complete immutable file inventory from the model registry."""
    from fastplms.registry import FileDigest

    if source.unresolved_files or not source.files:
        raise DependencyContractError(f"Feature source {source.repo_id} has unresolved files")
    if re.fullmatch(r"[0-9a-f]{40}", source.revision) is None:
        raise DependencyContractError(f"Feature source {source.repo_id} must use an immutable revision")
    if len({item.path for item in source.files}) != len(source.files):
        raise DependencyContractError("Feature source contains duplicate file identities")
    for item in source.files:
        FileDigest.parse(f"{item.path}={item.encoded}")  # a registry-built file always parses; a hand-built one may not
    return {
        "repository": source.repo_id,
        "revision": source.revision,
        "files": {item.path: item.encoded for item in source.files},
    }


def resolve_dependencies(
    registry: ModelRegistry, base_id: str, *, sae_layer: int, sae_k: int = SAE_K, sae_codebook: int = SAE_CODEBOOK,
) -> CanonicalDependencies:
    """Select the exact base, tokenizer, SAE and structural pins of ``base_id`` from the registry.

    The SAE is the one of `sae_layer` with `sae_k` codes kept per token over a codebook of `sae_codebook`, which are the
    canonical ones unless a caller's own manifest states others.
    """
    from fastplms.registry import CheckpointSource

    if base_id not in STRUCTURAL_MODELS or base_id not in registry:
        raise DependencyContractError(f"No canonical feature contract for base {base_id!r}")
    base = registry[base_id]
    if base.family.id != "esm_plusplus":
        raise DependencyContractError("Canonical features require an ESMC base")
    sae = registry.sae_for_base(base_id, layer=sae_layer, k=sae_k, codebook_dim=sae_codebook)
    tokenizer = registry[base.tokenizer_source_id].fast if base.tokenizer_source_id else base.fast
    files = tuple(item for item in tokenizer.files if item.path in _TOKENIZER_FILES)
    if not any(item.path == "tokenizer.json" for item in files):
        raise DependencyContractError("Canonical ESMC tokenizer lacks its pinned tokenizer.json")
    fold = registry[STRUCTURAL_MODELS[base_id]]
    if (fold.backbone_model or fold.family.backbone_model) != base_id:
        raise DependencyContractError("Structural projection is registered against a different base")
    dependencies = CanonicalDependencies(
        base_id, base.fast, CheckpointSource(tokenizer.repo_id, tokenizer.revision, files), sae,
        STRUCTURAL_MODELS[base_id], fold.official,
    )
    for source in (dependencies.base, dependencies.tokenizer, dependencies.sae.checkpoint, dependencies.structural):
        checkpoint_identity(source)  # reject an incomplete source before a run starts
    return dependencies


def snapshot_path(source: CheckpointSource, cache: Path) -> Path:
    """Where the Hub cache layout keeps ``source``'s revision under ``cache``."""
    return cache / ("models--" + source.repo_id.replace("/", "--")) / "snapshots" / source.revision


def fetch_snapshot(source: CheckpointSource, cache: Path) -> Path:
    """Download the pinned files of ``source`` into ``cache`` and return the snapshot directory."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=source.repo_id, revision=source.revision, cache_dir=str(cache),
        allow_patterns=[item.path for item in source.files],
    )
    return snapshot_path(source, cache)


def verified_snapshot(source: CheckpointSource, cache: Path) -> tuple[Path, SnapshotEvidence]:
    """Check all pinned cached bytes, including Git-blob identities for small JSON files."""
    from fastplms.checkpoint_files import verify_checkpoint
    from foundry.digests import sha256_file

    checkpoint_identity(source)
    snapshot = snapshot_path(source, cache)
    verify_checkpoint(snapshot, source)
    return snapshot, {**checkpoint_identity(source),
                      "sha256": {item.path: sha256_file(snapshot / item.path) for item in source.files}}


def structural_weights(snapshot: Path, *, width: int, depth: int) -> dict[str, Tensor]:
    """Read only the released layer mixture and projection, after the checkpoint's bytes were verified."""
    import torch

    from safetensors import safe_open

    config = json.loads((snapshot / "config.json").read_text())
    if (config.get("lm_d_model"), config.get("lm_num_layers"), config.get("d_pair")) != (width, depth, 256):
        raise ValueError("Structural config does not match the selected encoder")
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as source:
        weights = {name: source.get_tensor(key) for name, key in STRUCTURAL_KEYS.items()}
    shapes = {"layer_logits": (depth + 1,), "projection.0.weight": (width,),
              "projection.0.bias": (width,), "projection.1.weight": (256, width)}
    for name, value in weights.items():
        if value.shape != shapes[name] or not torch.isfinite(value).all():
            raise ValueError(f"Invalid released structural tensor {name}")
    return weights  # layer_logits (n,), projection.0.weight and bias (d,), projection.1.weight (256, d)


def load_sae(
    spec: SparseAutoencoderSpec, snapshot: Path, *, device: str = "cpu",
    dtype: Literal["float32", "bfloat16"] = "float32",
) -> ESMplusplusSAELayer:
    """Verify a downloaded SAE's bytes and input contract before the existing loader runs."""
    import torch

    from fastplms.models.esm_plusplus.modeling_esm_plusplus_sae import load_esmc_sae_layers
    from foundry.downloads.hub import HubFile

    if dtype not in {"float32", "bfloat16"}:
        raise DependencyContractError("Canonical SAE precision must be float32 or bfloat16")
    checkpoint_identity(spec.checkpoint)
    if set(spec.checkpoint.file_map) != {"config.json", f"layer_{spec.layer}.safetensors"}:
        raise DependencyContractError("Canonical SAE needs exactly its registered config and layer shard")
    for item in spec.checkpoint.files:
        if item.algorithm != "sha256":
            raise DependencyContractError("Canonical SAE inputs require SHA-256 file identities")
        HubFile(spec.checkpoint.repo_id, item.path, spec.checkpoint.revision, item.digest,
                repo_type="model").verify(snapshot / item.path)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    dimensions = {"d_model": spec.input_width, "codebook_dim": spec.codebook_dim, "k": spec.k}
    if any(type(config.get(key)) is not int or config[key] != value for key, value in dimensions.items()):
        raise DependencyContractError("Pinned SAE config disagrees with its registered dimensions")
    if (config.get("use_residual_update_instead_of_states", False) is not False
            or config.get("available_layers") != [spec.layer]):
        raise DependencyContractError("Pinned SAE must expose the registered hidden-state layer")
    return load_esmc_sae_layers(
        snapshot, [spec.layer], local_files_only=True, device=device, dtype=getattr(torch, dtype)
    )[spec.layer].eval().requires_grad_(False)


__all__ = [
    "SAE_CODEBOOK", "SAE_K", "STRUCTURAL_KEYS", "STRUCTURAL_MODELS", "CanonicalDependencies", "CheckpointIdentity",
    "DependencyContractError", "SnapshotEvidence", "checkpoint_identity", "fetch_snapshot", "load_sae",
    "resolve_dependencies", "snapshot_path", "structural_weights", "verified_snapshot",
]
