"""Measure immutable feature contracts from a loaded FastPLMs backbone and its tap plan."""

from __future__ import annotations

import json
import platform
import torch

from collections.abc import Mapping, Sequence
from importlib.metadata import version
from typing import TYPE_CHECKING
from torch import Tensor, nn

from .contracts import (
    FEATURE_SCHEMA, FEATURE_SCHEMA_V2, FEATURE_SCHEMA_V3, MASK_POLICY_V1, MASK_POLICY_V2, SPECIAL_TOKENS_KEPT, FeatureSpec,
    _digest, _fields, _positive, _sha256,
)
from .keys import _pinned_revision, _repository
from .sae_features import Readout
from .structural import STRUCTURAL_TAP_POOLED_SCHEMA, STRUCTURAL_TAP_SCHEMA, STRUCTURAL_TAP_TOKEN_SCHEMA, structural_tap
from .taps import ROWS_RESIDUES, ROWS_TOKENS, sae_residue_tap, sae_residue_tap_schema, sae_tap, sae_tap_schema


if TYPE_CHECKING:
    from fastplms.embeddings.taps import Tap
    from fastplms.embeddings.token_batches import BatchGeometry
    from .numerics import EmbeddingProfile, HardwareClass


# The batch algorithm a token run records: sequences sorted longest first inside windows, packed under a token budget.
TOKEN_BATCH_ALGORITHM = "token_budget_length_sorted_v1"
LEGACY_BATCH_ALGORITHM = "bounded_length_bucketed_stable_output_v1"


def _source(source: Mapping[str, str], name: str) -> dict[str, str]:
    fields = _fields(dict(source), {"repository", "revision"}, name)
    return {
        "repository": _repository(fields["repository"], f"{name}.repository"),
        "revision": _pinned_revision(fields["revision"], f"{name}.revision"),
    }


def capture_feature_specs(
    model: nn.Module,
    taps: Sequence[Tap],
    *,
    model_source: Mapping[str, str],
    tokenizer_source: Mapping[str, str],
    engine_sha256: str,
    reducer_sha256: Mapping[str, str],
    normalization_version: str,
    max_length: int | None,
    truncate: bool,
    batch_size: int | None = None,
    batch_window_size: int | None = None,
    max_tokens_per_batch: int | None = None,
    dtype: torch.dtype | None = None,
    sae_sources: Mapping[str, Mapping[str, str]] | None = None,
    sae_layers: Mapping[str, nn.Module] | None = None,
    structural_sources: Mapping[str, Mapping[str, str]] | None = None,
    structural_weights: Mapping[str, Mapping[str, Tensor]] | None = None,
    keep_special_tokens: bool = False,
    model_state_sha256: str | None = None,
    model_initialization: Mapping[str, object] | None = None,
    recorded_environment: Mapping[str, object] | None = None,
    fixed_batch_size: int | None = None,
    profile: EmbeddingProfile | None = None,
    hardware_class: HardwareClass | None = None,
    geometry: BatchGeometry | None = None,
) -> dict[str, FeatureSpec]:
    """Snapshot supported dense, SAE and structural taps without running the backbone.

    Source and reducer digests come from the verified code bundle, never a mutable branch name.
    Model/tensor/config/tokenizer bytes are measured now. Keep the modules frozen between this
    capture and extraction, and recapture before committing a store segment. Supply external
    SAE layers by tap name; attaching them to the backbone would couple every dense key to them.
    Structural taps similarly require their verified source pins and four loaded weight tensors.

    `keep_special_tokens=True` measures `feature_spec_v2` contracts: every per-token stream holds l + 2
    rows (row 0 CLS, rows 1..l residues, row l + 1 EOS), padding is the only masked position, pooling
    covers all those rows, and the taps must be the token variants (their identity says
    `rows: all_attended_tokens`). `model_state_sha256` takes a digest the caller measured a moment ago for
    these exact weights, which skips one hash of the backbone. `FeatureExtractionContract` never passes it:
    its recapture before a commit exists to read the bytes again, since a weight written through `.data`
    leaves no version counter behind.
    `model_initialization` records how a model that is not a released checkpoint was made (a random-init
    control's seed), and enters the model pin's config digest.
    `recorded_environment` is `{"software": ..., "hardware": ...}` as an existing store's descriptors state them.
    The library versions and the machine are evidence of where the first rows were made, not of what a value is
    and are retained when deriving an existing store's keys. This does not establish numerical compatibility:
    a later run must pass independent verification, including changed batch shapes, before appending rows.
    The reduction, split-K and BLAS policies are measured live and enter every feature identity.

    A ``profile`` (with its ``hardware_class`` and the base's ``geometry``) measures `feature_spec_v3` contracts: the
    canonical token rows computed under an embedding profile. The key binds the profile's digest, the class and the
    geometry instead of the library versions, machine and token budget of v2, which become run evidence; the numerics
    flags are measured live and must already be the profile's (`numerics.check_numerics`). SAE taps must carry the
    profile's continuous readout. None of the batch-budget options apply.
    """
    if profile is not None:
        if hardware_class is None or geometry is None:
            raise ValueError("A profile capture needs the hardware class and the base's batch geometry.")
        if not keep_special_tokens or not truncate:
            raise ValueError("A profile capture is the canonical token contract: keep_special_tokens and truncate are True.")
        if any(value is not None for value in (batch_size, batch_window_size, max_tokens_per_batch, fixed_batch_size,
                                               recorded_environment)):
            raise ValueError("A profile capture takes its batch shapes from the geometry and keys on no recorded environment.")
    elif hardware_class is not None or geometry is not None:
        raise ValueError("A hardware class and a geometry belong to a profile capture.")
    elif batch_size is None or batch_window_size is None:
        raise ValueError("A capture without a profile states its batch_size and batch_window_size.")
    if fixed_batch_size is not None:
        if not keep_special_tokens or type(fixed_batch_size) is not int or fixed_batch_size != batch_size or fixed_batch_size < 1:
            raise ValueError("Fixed batches require the canonical token path and matching positive batch_size.")
        if max_length is None or max_tokens_per_batch is None or max_tokens_per_batch < fixed_batch_size * (max_length + 2):
            raise ValueError("Fixed batches require a token budget covering the full cropped context.")
    # structural_weights: (n,) layer_logits, (d,) projection.0.weight and .bias, (256, d)
    # projection.1.weight per tap name, where n is the mixture's hidden states and d the base width.
    from fastplms.embeddings.identity import (
        _attention_backend,
        _attention_kernel_metadata,
        _model_device,
        _model_identity_metadata,
        _model_state_sha256,
        _software_versions,
        _tokenizer_content_sha256,
    )
    from fastplms.embeddings.pooling import POOLING_SEMANTICS, POOLING_SEMANTICS_TOKENS
    from fastplms.embeddings.taps import HiddenTap, ReducedTap, SparseResidueTap, StreamingTap, plan_taps

    base = _source(model_source, "model_source")
    tokenizer_pin = _source(tokenizer_source, "tokenizer_source")
    _sha256(engine_sha256, "engine_sha256")
    if profile is None:
        _positive(batch_size, "batch_size")
        _positive(batch_window_size, "batch_window_size")
        if batch_window_size < batch_size:
            raise ValueError("batch_window_size must be at least batch_size.")
    if type(truncate) is not bool or type(keep_special_tokens) is not bool:
        raise ValueError("truncate and keep_special_tokens must be boolean.")
    if keep_special_tokens and not truncate:
        raise ValueError("A canonical token contract crops overlength proteins from the N-terminus; set truncate=True.")
    if dtype is not None and dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("dtype must be a floating torch dtype or None.")
    if getattr(model, "embedding_tap_support", False) is not True:
        raise ValueError("Feature capture requires a backbone with FastPLMs one-pass taps.")
    if getattr(model, "_sae_models", None):
        raise ValueError("Keep SAE layers external to the backbone when capturing feature identities.")
    if getattr(model, "peft_config", None):
        raise ValueError("A trained adapter needs its own accepted base identity before feature capture.")
    identity = _model_identity_metadata(model)
    loaded_pin = (
        (identity["checkpoint_repo_id"], identity["checkpoint_revision"])
        if identity["checkpoint_repo_id"] else (identity["model_id"], identity["model_revision"])
    )
    if loaded_pin != (base["repository"], base["revision"]):
        raise ValueError("Loaded model repository/revision does not match model_source.")
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Feature capture requires an explicit loaded tokenizer.")
    plan = plan_taps(taps, int(model.embedding_tap_state_count))
    if set(reducer_sha256) != {tap.name for tap in plan.taps}:
        raise ValueError("Supply one reducer implementation digest for every tap name.")
    for value in reducer_sha256.values():
        _sha256(value, "reducer implementation digest")
    reduced_names = {tap.name for tap in plan.taps if isinstance(tap, (ReducedTap, SparseResidueTap))}
    if set(sae_sources or {}) != reduced_names or set(sae_layers or {}) != reduced_names:
        raise ValueError("Each SAE tap requires its exact source pin and loaded external layer.")
    structural_names = {tap.name for tap in plan.taps if isinstance(tap, StreamingTap)}
    if set(structural_sources or {}) != structural_names or set(structural_weights or {}) != structural_names:
        raise ValueError("Each structural tap requires its exact source pin and loaded external weights.")
    parameter_dtypes = {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()}
    if len(parameter_dtypes) != 1:
        raise ValueError("Feature capture requires one explicit backbone parameter dtype.")
    (model_dtype,) = parameter_dtypes
    backend = _attention_backend(model)
    resolution = getattr(model, "attention_resolution", None)
    if backend in {None, "auto"} or (resolution is not None and resolution.deferred):
        raise ValueError("Resolve the attention backend before feature capture.")
    # Transformers owns conversion of integer label keys and dtype values into JSON.
    # Parse its full document, then apply the contract's strict finite-value checks.
    config = json.loads(model.config.to_json_string(use_diff=False))
    # These locate the same checkpoint; they do not change its tensor/config computation.
    for name in (
        "_name_or_path", "_commit_hash", "transformers_version", "fastplms_model_id",
        "fastplms_checkpoint_repo_id", "fastplms_checkpoint_revision", "fastplms_checkpoint_hash",
        "fastplms_weights_revision", "fastplms_runtime_revision", "fastplms_source_tree_sha256",
        "fastplms_runtime_bundle_sha256",
    ):
        config.pop(name, None)
    base.update({
        "content_sha256": model_state_sha256 or _model_state_sha256(model),
        "config_sha256": _digest({
            "class": f"{type(model).__module__}.{type(model).__qualname__}", "configuration": config,
            **({"initialization": dict(model_initialization)} if model_initialization else {}),
        }),
    })
    tokenizer_pin["content_sha256"] = _digest({
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "content_sha256": _tokenizer_content_sha256(tokenizer),
    })
    kernel = _attention_kernel_metadata(backend)
    software = {name: str(version) for name, version in _software_versions().items() if version is not None}
    software["tokenizers"] = version("tokenizers")
    if recorded_environment is not None:
        software = dict(recorded_environment["software"])
    device = _model_device(model)
    device_type = device.type
    if device_type not in {"cpu", "cuda"}:
        raise ValueError("Feature capture currently qualifies CPU and CUDA execution only.")
    if any(parameter.device != device for parameter in model.parameters()):
        raise ValueError("Feature capture requires a single-device backbone.")
    if profile is not None:
        if hardware_class is not None and hardware_class.device_type != device_type:
            raise ValueError(f"Hardware class {hardware_class.id} runs on {hardware_class.device_type}, not {device_type}.")
        if hardware_class is not None:
            from .numerics import check_live_class, describe_device

            check_live_class(hardware_class, describe_device(device))  # again before every commit
        numerical_policy = _profile_policy(model, device)
    else:
        numerical_policy = _legacy_policy(model, device, recorded_environment, fixed_batch_size)
    specs: dict[str, FeatureSpec] = {}
    # The plan gives one resolved layer per tap.
    keeps = keep_special_tokens
    expected_rows = ROWS_TOKENS if keeps else ROWS_RESIDUES
    for tap, layer in zip(plan.taps, plan.layers, strict=True):
        input_dtype = dtype or model_dtype
        sae = None
        structural, projection_sha256 = None, None
        feature_layers = [layer]
        if isinstance(tap, HiddenTap):
            input_dtype = tap.dtype or input_dtype
            output_dtype = input_dtype
            kind = "dense_residue" if tap.pooling is None else "dense_pooled"
            pooling = None if tap.pooling is None else list(tap.pooling)
            reducer = {
                "name": "fastplms_hidden_tap", "version": "pooling_v3" if keeps else "pooling_v2",
                "parameters": {"pooling_semantics": dict(POOLING_SEMANTICS_TOKENS if keeps else POOLING_SEMANTICS)},
            }
            accumulator = torch.float64 if input_dtype == torch.float64 else torch.float32
        elif isinstance(tap, StreamingTap):
            recorded = dict(tap.identity)
            pooled = tap.pooling is not None  # structural_mean_var: the token rows pooled per protein
            schema = STRUCTURAL_TAP_POOLED_SCHEMA if pooled else STRUCTURAL_TAP_TOKEN_SCHEMA if keeps else STRUCTURAL_TAP_SCHEMA
            if (recorded.get("reducer") != "foundry.embedding.structural_tap" or recorded.get("schema") != schema
                    or recorded.get("rows") != expected_rows):
                raise ValueError("This capture adapter supports only the current structural streaming reducer for its row policy.")
            compute_dtype = getattr(torch, recorded["compute_dtype"])
            weights = structural_weights[tap.name]
            observed = structural_tap(tap.name, weights, compute_dtype=compute_dtype, keep_special_tokens=keeps,
                                      pooling=tap.pooling)
            if (dict(observed.identity) != recorded or observed.layers != tap.layers
                    or observed.required_state_count != tap.required_state_count
                    or (observed.pooling, observed.dtype) != (tap.pooling, tap.dtype)):
                raise ValueError("Structural tap identity is stale; rebuild the tap from its current weights.")
            if recorded["input_width"] != model.config.hidden_size or any(value.device != device for value in weights.values()):
                raise ValueError("Structural weights must match the backbone width and device.")
            input_dtype = model_dtype  # Streaming reducers receive original, unconverted states.
            if input_dtype not in (torch.bfloat16, torch.float16, torch.float32) or (
                    compute_dtype == torch.float32 and input_dtype != torch.float32):
                raise ValueError("Unsupported structural input and compute dtype combination.")
            output_dtype = torch.float32 if pooled else compute_dtype
            structural = _source(structural_sources[tap.name], f"structural_sources.{tap.name}")
            structural["content_sha256"] = _digest(recorded["weights_sha256"])
            projection_sha256 = _digest({key: value for key, value in recorded["weights_sha256"].items() if key != "layer_logits"})
            kind = "structural_pooled" if pooled else "structural_residue"
            pooling, accumulator = (list(tap.pooling) if pooled else None), torch.float32
            feature_layers = list(tap.layers)
            reducer = {
                "name": recorded["reducer"],
                "version": "structural_tap_pooled_v1" if pooled else "structural_tap_v2" if keeps else "structural_tap_v1",
                "parameters": recorded,
            }
        else:
            recorded = dict(tap.identity)
            readout = _tap_readout(recorded, profile, tap.name)
            profiled = readout is not None
            if isinstance(tap, SparseResidueTap):
                schema = sae_residue_tap_schema(keep_special_tokens=keeps, profile=profiled)
                if (recorded.get("reducer") != "foundry.embedding.sae_residue_tap" or recorded.get("schema") != schema
                        or recorded.get("rows") != expected_rows):
                    raise ValueError("This capture adapter supports only the current sparse residue SAE reducer for its row policy.")
                observed = sae_residue_tap(
                    tap.name, sae_layers[tap.name], layer, chunk_rows=recorded["chunk_rows"], keep_special_tokens=keeps,
                    readout=readout,
                )
                if (observed.codebook_size, observed.sparse_count) != (tap.codebook_size, tap.sparse_count):
                    raise ValueError("Sparse residue tap dimensions differ from its SAE.")
                output_dtype, accumulator = getattr(torch, recorded["encoder_dtype"]), getattr(torch, recorded["encoder_dtype"])
                kind, pooling = "sae_residue", None
                reducer_version = "sae_residue_tap_v3" if profiled else ("sae_residue_tap_v2" if keeps else "sae_residue_tap_v1")
            else:
                schema = sae_tap_schema(keep_special_tokens=keeps, profile=profiled)
                if (recorded.get("reducer") != "foundry.embedding.sae_tap" or recorded.get("schema") != schema
                        or recorded.get("rows") != expected_rows):
                    raise ValueError("This capture adapter supports only the current pooled SAE reducer for its row policy.")
                observed = sae_tap(
                    tap.name, sae_layers[tap.name], layer, recorded["pooling"],
                    chunk_rows=recorded["chunk_rows"], keep_special_tokens=keeps, readout=readout,
                )
                output_dtype, accumulator = torch.float32, torch.float32
                kind, pooling = "sae_pooled", recorded["pooling"]
                reducer_version = "sae_tap_v4" if profiled else ("sae_tap_v3" if keeps else "sae_tap_v2")
            if dict(observed.identity) != recorded:
                raise ValueError("SAE tap identity is stale; rebuild the tap from its current weights.")
            sae = _source(sae_sources[tap.name], f"sae_sources.{tap.name}")
            sae["content_sha256"] = _digest({
                "W_enc_sha256": recorded["W_enc_sha256"], "b_dec_sha256": recorded["b_dec_sha256"],
            })
            reducer = {"name": recorded["reducer"], "version": reducer_version, "parameters": recorded}
        reducer["parameters"]["input_dtype"] = str(input_dtype).removeprefix("torch.")
        reducer["parameters"]["layer_numbering"] = "input_to_block_final_normalized_v1"
        execution: dict[str, object] = {
            "compute_dtype": str(model_dtype).removeprefix("torch."),
            "accumulator_dtype": str(accumulator).removeprefix("torch."), "attention_backend": backend,
            "device_type": device_type, "numerical_policy": numerical_policy,
            "kernel_sha256": None if kernel is None else _digest(kernel),
        }
        if profile is not None:
            # The source digests are run evidence: the canary's golden digests, not a key, hold a v3 store to its code.
            assert hardware_class is not None and geometry is not None  # checked on entry
            execution.update(profile=profile.binding(), hardware_class=hardware_class.record(),
                             batch_policy=dict(geometry.describe()))
        else:
            reducer["implementation_sha256"] = reducer_sha256[tap.name]
            execution.update(engine_sha256=engine_sha256, software=software, batch_policy={
                "algorithm": "fixed_rows_duplicate_pad_v1" if fixed_batch_size is not None else (
                    TOKEN_BATCH_ALGORITHM if keeps else LEGACY_BATCH_ALGORITHM),
                "max_sequences": batch_size, "max_tokens": max_tokens_per_batch,
                "window_size": batch_window_size,
            })
        document = {
            "schema": FEATURE_SCHEMA_V3 if profile is not None else (FEATURE_SCHEMA_V2 if keeps else FEATURE_SCHEMA),
            "kind": kind, "model": base,
            "tokenizer": tokenizer_pin, "layers": feature_layers,
            "layer_count": int(model.embedding_tap_state_count) - 1,
            "pooling": pooling, "output_dtype": str(output_dtype).removeprefix("torch."),
            "normalization_version": normalization_version,
            "mask_policy": MASK_POLICY_V2 if keeps else MASK_POLICY_V1, "reducer": reducer,
            "execution": execution,
            "length": {"max_residues": max_length, "overlength": "prefix_crop" if truncate else "reject"},
            "sae": sae, "structural": structural, "projection_sha256": projection_sha256,
        }
        if keeps:
            # CLS and EOS rows are kept, so a protein of l residues stores l + 2 rows in every per-token stream.
            document["special_tokens"] = SPECIAL_TOKENS_KEPT
        specs[tap.name] = FeatureSpec(document)
    return specs


def _tap_readout(recorded: Mapping[str, object], profile: EmbeddingProfile | None, name: str) -> Readout | None:
    """The readout an SAE tap was built with, held to the profile's: a profile tap under a profile only."""
    if profile is None:
        if "readout" in recorded:
            raise ValueError(f"SAE tap {name!r} carries a profile readout; capture it under its profile.")
        return None
    expected = profile.readout
    described = expected.describe()
    if {field: recorded.get(field) for field in described} != described or (
            "rho" in recorded and "rho" not in described):
        raise ValueError(f"SAE tap {name!r} must carry {profile.id}'s readout {described}.")
    return expected


def _profile_policy(model: nn.Module, device: torch.device) -> dict[str, object]:
    """The numerical policy a v3 key states: every flag measured now, and the precision mode the backbone runs in.

    The flags must already be the profile's (`numerics.check_numerics` runs before any capture); measuring them here
    keeps a draft profile's unpinned values, such as workspace sizes, in the key. The precision mode is read from the
    backbone's dtype, which `canonical.build_runtime` takes from the profile for the base.
    """
    from .numerics import measure_numerics, precision_of

    precision = getattr(model, "esmc_precision_status", None)
    return {
        "measured": measure_numerics(device), "precision_mode": precision_of(next(model.parameters()).dtype),
        "esmc_fp8": bool(getattr(model, "_esmc_fp8", False)),
        "esmc_precision": None if precision is None else precision.as_dict(),
    }


def _legacy_policy(
    model: nn.Module, device: torch.device, recorded_environment: Mapping[str, object] | None, fixed_batch_size: int | None,
) -> dict[str, object]:
    """The numerical policy a v1 or v2 key states, exactly as the published keys were derived."""
    device_type = device.type
    precision = getattr(model, "esmc_precision_status", None)
    numerical_policy: dict[str, object] = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cuda_matmul_allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "cuda_matmul_allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "cuda_matmul_allow_fp16_reduced_precision_reduction_split_k": getattr(
            torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction_split_k", None),
        "cuda_matmul_allow_bf16_reduced_precision_reduction_split_k": getattr(
            torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction_split_k", None),
        "cuda_preferred_blas_library": str(torch.backends.cuda.preferred_blas_library()),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "autocast_enabled": torch.is_autocast_enabled(device_type),
        "autocast_dtype": str(torch.get_autocast_dtype(device_type)).removeprefix("torch."),
        "esmc_fp8": bool(getattr(model, "_esmc_fp8", False)),
        "esmc_precision": None if precision is None else precision.as_dict(),
        "hardware": {
            "machine": platform.machine(), "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "cuda_name": torch.cuda.get_device_name(device) if device_type == "cuda" else None,
            "cuda_capability": list(torch.cuda.get_device_capability(device)) if device_type == "cuda" else None,
        },
    }
    if recorded_environment is not None:
        numerical_policy["hardware"] = dict(recorded_environment["hardware"])
    if fixed_batch_size is not None:
        numerical_policy["fixed_batch_size"] = fixed_batch_size
        numerical_policy["partial_batch_policy"] = "repeat_last_sequence_discard_duplicate_outputs_v1"
    return numerical_policy
