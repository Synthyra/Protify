"""The embedding profile: one enforced definition of how a canonical embedding is computed.

`embedding_profile_v1` fixes everything that decides a stored value besides the model and the protein: the container's
library closure, the numerics flags, the precision mode, the attention kernel, each base's batch geometry and the SAE
readout. That part, the profile's ``function``, has a digest that enters every stream key (`feature_spec_v3`), so equal
keys mean one function. The rest of the file records who may compute it: the hardware classes, the devices each has
enrolled, and each class's canary digests and bounds. Those records grow without changing the function or a key.

A hardware class is a compute capability and a multiprocessor count, inside one library closure: cuBLAS promises bitwise
results only for one toolkit version on GPUs of one architecture and SM count. Within a class (tier 1) rows are bitwise
equal on every member; across classes (tier 2) they agree within the profile's bounds, measured on the canary panel.

- `load_profile` reads and validates a profile from `profiles/`; `EmbeddingProfile.sha256` is the function's digest.
- `apply_profile` sets the numerics flags; `measure_numerics` reads every one back; `check_numerics` refuses a difference.
- `attest_environment` measures the library closure from the installed binaries and checks that the loaded ones are
  those; `check_environment` refuses a closure the profile does not list, or a process outside a container.
- `device_class` names the hardware class of a device and refuses one the profile has not enrolled; `check_live_class`
  measures the device again at every capture and commit; `check_reference_class` keeps embedding, piloting, verifying
  and recording goldens to the reference class; `check_publishable` lets only a qualified profile's reference-class
  stores into the repositories.
- `canary_panel`, `stream_digests` and `check_canary` hold a run to its class's golden digests before a row is written.
- `golden_panel`, `golden_rows`, `write_golden` and `check_golden` are the short check before every embed and pilot:
  one full batch in each of three buckets, held bitwise to digests and reference rows kept in the repository
  (`profiles/golden/<profile>/<base>.safetensors`), recorded once on the reference class.
- `bf16_steps`, `row_scale_steps` and `relative_l2` measure a tier-2 class against the reference class; `panel_tensors`
  reads the rows a qualification keeps for that comparison and `compare_panels` reports it per stream.
- `activation_peaks` and `headroom_report` record every module's largest output during a qualification: the FP16
  headroom a precision mode needs.

Symbols: n rows compared; w columns of a stream.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import sys
import torch

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from torch import Tensor

from .contracts import _canonical_json, _digest
from .sae_features import READOUT_CONTINUOUS, READOUT_HARD_CANONICAL, ContinuousReadout, HardTopKReadout, Readout


if TYPE_CHECKING:
    from fastplms.embeddings.token_batches import BatchGeometry


PROFILE_DIRECTORY = Path(__file__).resolve().parent / "profiles"
PROFILE_SCHEMA = "embedding_profile_schema_v1"
STATUS_DRAFT = "draft"  # pilots and qualification only; never writes a canonical store
STATUS_QUALIFIED = "qualified"
PRECISION_BF16_BACKBONE = "bf16_backbone"  # BF16 backbone, FP32 SAE and pooling, continuous readout
# The qualification's ablation arms: the backbone's weights and activations in FP16 (the tensor cores' BF16 speed, eight
# times finer rounding) or FP32 (a reference without 16-bit rounding, far slower). Stored per-token rows stay BF16.
PRECISION_FP16_BACKBONE = "fp16_backbone"
PRECISION_FP32_BACKBONE = "fp32_backbone"
BACKBONE_DTYPES = {PRECISION_BF16_BACKBONE: torch.bfloat16, PRECISION_FP16_BACKBONE: torch.float16,
                   PRECISION_FP32_BACKBONE: torch.float32}
PRECISION_MODES = frozenset(BACKBONE_DTYPES)


def precision_of(dtype: torch.dtype) -> str:
    """The precision mode whose backbone runs in ``dtype``."""
    modes = [mode for mode, backbone in BACKBONE_DTYPES.items() if backbone == dtype]
    if not modes:
        raise ValueError(f"No precision mode runs its backbone in {dtype}; the modes run {sorted(map(str, BACKBONE_DTYPES.values()))}.")
    return modes[0]
ATTENTION_EFFICIENT = "efficient_attention"
PER_TOKEN_LAYOUTS = frozenset({"ragged", "ragged_topk"})  # stored rows per token; "dense" and "csr" are pooled
PANEL_SCHEMA = "embedding_panel_v1"  # a qualification's comparison rows (`panel_tensors`), beside its receipt
TIER_BITWISE = 1
TIER_BOUNDED = 2
# NVIDIA architectures by compute capability, for the name of a provisional class (`provisional_class`).
ARCHITECTURES = {(7, 0): "volta", (7, 5): "turing", (8, 0): "ampere", (8, 6): "ampere", (8, 7): "ampere", (8, 9): "ada",
                 (9, 0): "hopper", (10, 0): "blackwell", (12, 0): "blackwell"}
CLOSURE_SCHEMA = "embedding_closure_v1"
# The Python distributions whose code decides a value, and the binaries the GPU arithmetic runs in. Every file under a
# CUDA library wheel counts, and of torch's own libraries those holding its kernels and dispatch.
CLOSURE_PACKAGES = ("torch", "transformers", "tokenizers", "numpy", "safetensors")
TORCH_LIBRARIES = ("libtorch_cuda.so", "libc10_cuda.so", "libtorch_cpu.so", "libc10.so")
NUMERICS_LIBRARY = re.compile(r"lib(torch_cuda|c10_cuda|cublas|cublasLt|cudnn[A-Za-z_]*|cudart|nvrtc[A-Za-z-]*|nvJitLink)\.so")
CANARY_SEED = 20261006
CANARY_RESIDUES = "ACDEFGHIKLMNPQRSTVWY"
CANARY_RARE = "XBUZO"  # unknown, ambiguous and rare residues the tokenizer holds; ESMC has no J
CANARY_OVERLENGTH = 2100  # residues, cropped to the canonical 2046
# The golden check before every embed and pilot (`golden_panel`, `check_golden`): one full batch in each of three buckets,
# with reference rows kept in the repository beside the profile.
GOLDEN_SCHEMA = "embedding_golden_v1"
GOLDEN_SEED = 20261007
GOLDEN_DIRECTORY = PROFILE_DIRECTORY / "golden"
NUMERICS_FIELDS = frozenset({
    "preferred_blas_library", "bf16_reduced_precision_reduction", "bf16_reduced_precision_reduction_split_k",
    "fp16_reduced_precision_reduction", "fp16_reduced_precision_reduction_split_k", "fp16_accumulation",
    "fp32_precision", "matmul_fp32_precision", "cudnn_fp32_precision", "cudnn_conv_fp32_precision",
    "cudnn_rnn_fp32_precision", "deterministic_algorithms", "deterministic_warn_only", "fill_uninitialized_memory",
    "cudnn_deterministic", "cudnn_benchmark", "cublas_workspace_bytes", "cublaslt_workspace_bytes",
    "fp16_bf16_reduction_math_sdp", "autocast",
})
CUDA_ONLY_FIELDS = frozenset({"cublas_workspace_bytes", "cublaslt_workspace_bytes"})  # reading one initializes CUDA
FUNCTION_FIELDS = frozenset({
    "precision_mode", "attention_backend", "numerics", "environment_variables", "environments", "geometry", "readout",
})
# Optional function fields: absent from profile v1, whose digest they would otherwise change. `extra_streams` names the
# streams a profile adds to every store of a trained base beyond the base's own.
OPTIONAL_FUNCTION_FIELDS = frozenset({"extra_streams"})
EXTRA_STREAMS = frozenset({"layer_mean_var", "structural_mean_var"})  # `canonical.LAYER_MEAN_VAR`, `STRUCTURAL_MEAN_VAR`
GEOMETRY_FIELDS = frozenset({"bucket_tokens", "token_budget", "max_rows", "sae_chunk_rows"})
CLASS_FIELDS = frozenset({"device_type", "compute_capability", "multiprocessors", "tier", "reference", "bases", "members"})


@dataclass(frozen=True, slots=True)
class HardwareClass:
    """Devices that compute one function bitwise alike: a compute capability and a multiprocessor count, or the CPU."""

    id: str
    device_type: str  # "cuda", or "cpu" for a test profile
    compute_capability: tuple[int, int] | None
    multiprocessors: int | None
    tier: int  # 1 bitwise with the reference class, 2 within the profile's bounds of it
    reference: bool
    bases: tuple[str, ...] | None  # the bases it is qualified for; None for every base
    members: tuple[str, ...]  # device names enrolled by a qualification receipt

    def record(self) -> dict[str, Any]:
        """What a feature key states about the class: its id and what defines it, never its members."""
        return {"id": self.id, "device_type": self.device_type,
                "compute_capability": None if self.compute_capability is None else list(self.compute_capability),
                "multiprocessors": self.multiprocessors}


@dataclass(frozen=True, slots=True)
class EmbeddingProfile:
    """A validated profile: the function that decides every stored value, and who is qualified to compute it."""

    id: str
    status: str
    function: Mapping[str, Any]
    classes: Mapping[str, HardwareClass]
    canary: Mapping[str, Any]
    document: Mapping[str, Any] = field(repr=False)

    @property
    def sha256(self) -> str:
        """The digest of the id and the function: what a feature key binds."""
        return _digest({"id": self.id, "function": dict(self.function)})

    def binding(self) -> dict[str, str]:
        """The profile as a feature key states it."""
        return {"id": self.id, "sha256": self.sha256}

    def precision_mode(self, base: str) -> str:
        """The precision mode of ``base``: each base has its own, chosen by the headroom its activations leave."""
        modes = self.function["precision_mode"]
        if base not in modes:
            raise ValueError(f"{self.id} has no precision mode for {base!r}; it covers {sorted(modes)}.")
        return str(modes[base])

    def backbone_dtype(self, base: str) -> torch.dtype:
        """The dtype of ``base``'s backbone weights and activations under its precision mode."""
        return BACKBONE_DTYPES[self.precision_mode(base)]

    @property
    def readout(self) -> Readout:
        """The readout of the profile's SAE taps: the exact top k (profile v2) or the continuous band (profile v1)."""
        record = self.function["readout"]
        if record["name"] == READOUT_HARD_CANONICAL:
            return HardTopKReadout()
        return ContinuousReadout(float(record["rho"]))

    @property
    def extra_streams(self) -> tuple[str, ...]:
        """The streams the profile adds to every store of a trained base, in its listed order."""
        return tuple(self.function.get("extra_streams", ()))

    def geometry(self, base: str) -> BatchGeometry:
        """The fixed batch shapes of ``base``."""
        from fastplms.embeddings.token_batches import BatchGeometry

        table = self._base_geometry(base)
        return BatchGeometry(bucket_tokens=table["bucket_tokens"], token_budget=table["token_budget"],
                             max_rows=table["max_rows"])

    def sae_chunk_rows(self, base: str) -> int:
        """Rows of every SAE encoder chunk of ``base``."""
        return int(self._base_geometry(base)["sae_chunk_rows"])

    def _base_geometry(self, base: str) -> Mapping[str, int]:
        tables = self.function["geometry"]
        if base not in tables:
            raise ValueError(f"{self.id} has no batch geometry for {base!r}; it covers {sorted(tables)}.")
        return tables[base]

    @property
    def reference_class(self) -> HardwareClass:
        (reference,) = (item for item in self.classes.values() if item.reference)
        return reference


def load_profile(name: str | Path) -> EmbeddingProfile:
    """The profile ``name`` (an id under `profiles/`, or a path), validated."""
    path = Path(name) if Path(name).suffix == ".json" else PROFILE_DIRECTORY / f"{name}.json"
    if not path.is_file():
        known = sorted(item.stem for item in PROFILE_DIRECTORY.glob("*.json"))
        raise FileNotFoundError(f"No embedding profile {name!r}; the profiles are {known}.")
    return parse_profile(json.loads(path.read_text(encoding="utf-8")))


def parse_profile(document: Mapping[str, Any]) -> EmbeddingProfile:
    """Validate a profile document: every function field present and well formed, one reference class."""
    expected = {"schema", "id", "status", "function", "classes", "canary"}
    if set(document) != expected:
        raise ValueError(f"An embedding profile has exactly the fields {sorted(expected)}; received {sorted(document)}.")
    if document["schema"] != PROFILE_SCHEMA:
        raise ValueError(f"Unsupported profile schema {document['schema']!r}; expected {PROFILE_SCHEMA}.")
    status = document["status"]
    if status not in {STATUS_DRAFT, STATUS_QUALIFIED}:
        raise ValueError(f"A profile's status is {STATUS_DRAFT} or {STATUS_QUALIFIED}; received {status!r}.")
    function = document["function"]
    if not isinstance(function, dict) or not FUNCTION_FIELDS <= set(function) <= FUNCTION_FIELDS | OPTIONAL_FUNCTION_FIELDS:
        raise ValueError(f"A profile's function has the fields {sorted(FUNCTION_FIELDS)}, and may have "
                         f"{sorted(OPTIONAL_FUNCTION_FIELDS)}.")
    extra = function.get("extra_streams", [])
    if not isinstance(extra, list) or len(set(extra)) != len(extra) or not set(extra) <= EXTRA_STREAMS:
        raise ValueError(f"extra_streams lists distinct streams of {sorted(EXTRA_STREAMS)}; received {extra!r}.")
    modes = function["precision_mode"]
    if not isinstance(modes, dict) or set(modes) != set(function["geometry"]):
        raise ValueError("A profile's precision_mode maps every base of its geometry to a precision mode.")
    if any(mode not in PRECISION_MODES for mode in modes.values()):
        raise ValueError(f"Unsupported precision mode in {modes}; implemented: {sorted(PRECISION_MODES)}.")
    if function["attention_backend"] != ATTENTION_EFFICIENT:
        raise ValueError(f"The profile's attention backend must be {ATTENTION_EFFICIENT}.")
    numerics = function["numerics"]
    if not isinstance(numerics, dict) or set(numerics) != NUMERICS_FIELDS:
        raise ValueError(f"A profile's numerics name exactly {sorted(NUMERICS_FIELDS)}.")
    unpinned = sorted(name for name, value in numerics.items() if value is None)
    if status == STATUS_QUALIFIED and unpinned:
        raise ValueError(f"A qualified profile pins every numerics value; {unpinned} are unpinned.")
    if not isinstance(function["environment_variables"], dict) or not all(
            isinstance(value, str) for value in function["environment_variables"].values()):
        raise ValueError("environment_variables maps each variable to its exact string value.")
    environments = function["environments"]
    if not isinstance(environments, dict) or not environments:
        raise ValueError("environments maps each platform to the closure digest its container must attest.")
    if status == STATUS_QUALIFIED and any(value is None for value in environments.values()):
        raise ValueError("A qualified profile pins the closure digest of every platform it lists.")
    for base, table in function["geometry"].items():
        if not isinstance(table, dict) or set(table) != GEOMETRY_FIELDS or not all(
                type(value) is int and value > 0 for value in table.values()):
            raise ValueError(f"The geometry of {base} has exactly the positive integers {sorted(GEOMETRY_FIELDS)}.")
    readout = function["readout"]
    if readout == {"name": READOUT_HARD_CANONICAL}:
        pass
    elif isinstance(readout, dict) and set(readout) == {"name", "rho"} and readout["name"] == READOUT_CONTINUOUS:
        ContinuousReadout(float(readout["rho"]))
    else:
        raise ValueError(f"The profile's readout is {{'name': {READOUT_HARD_CANONICAL!r}}} or "
                         f"{{'name': {READOUT_CONTINUOUS!r}, 'rho': ...}}; received {readout!r}.")
    classes = {class_id: _hardware_class(class_id, record) for class_id, record in document["classes"].items()}
    if sum(item.reference for item in classes.values()) != 1:
        raise ValueError("A profile names exactly one reference class.")
    if any(item.reference and item.tier != TIER_BITWISE for item in classes.values()):
        raise ValueError("The reference class is tier 1 by definition.")
    _check_canary_records(document["canary"], classes, tuple(function["geometry"]), qualified=status == STATUS_QUALIFIED)
    profile = EmbeddingProfile(
        id=_identifier(document["id"], "id"), status=status, function=MappingProxyType(json.loads(_canonical_json(function))),
        classes=MappingProxyType(classes), canary=MappingProxyType(dict(document["canary"])), document=dict(document),
    )
    for base in function["geometry"]:
        profile.geometry(base)  # BatchGeometry validates the table
    return profile


def _check_canary_records(canary: object, classes: Mapping[str, HardwareClass], bases: Sequence[str], *,
                          qualified: bool) -> None:
    """The canary's records: golden digests and bounds by class and base, and in a qualified profile, complete.

    Qualified, every enrolled class has golden digests for each base it covers and a tier-2 class has bounds too, so
    whatever an enrolled device may compute has a canary to match before a row is written (`check_canary`).
    """
    if not isinstance(canary, dict) or set(canary) != {"golden", "bounds"}:
        raise ValueError("A profile's canary has exactly the fields ['bounds', 'golden'], each by class and base.")
    if not qualified:
        return
    (reference,) = (item for item in classes.values() if item.reference)
    if not reference.members:
        raise ValueError(f"A qualified profile enrolls a device of its reference class {reference.id}.")
    for item in classes.values():
        if not item.members:
            continue
        covered = item.bases if item.bases is not None else bases
        lacking = [base for base in covered if not canary["golden"].get(item.id, {}).get(base)]
        if item.tier == TIER_BOUNDED:
            lacking += [f"{base} bounds" for base in covered if not canary["bounds"].get(item.id, {}).get(base)]
        if lacking:
            raise ValueError(f"A qualified profile records golden digests, and bounds for tier 2, for every base an "
                             f"enrolled class covers; class {item.id} lacks {lacking}.")


def _identifier(value: object, where: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_]*", value):
        raise ValueError(f"A profile's {where} is a lowercase identifier; received {value!r}.")
    return value


def _hardware_class(class_id: str, record: Mapping[str, Any]) -> HardwareClass:
    if not isinstance(record, dict) or set(record) != CLASS_FIELDS:
        raise ValueError(f"Hardware class {class_id} has exactly the fields {sorted(CLASS_FIELDS)}.")
    device_type = record["device_type"]
    if device_type == "cuda":
        capability = record["compute_capability"]
        if (not isinstance(capability, list) or len(capability) != 2 or not all(type(part) is int for part in capability)
                or type(record["multiprocessors"]) is not int):
            raise ValueError(f"CUDA class {class_id} needs compute_capability [major, minor] and multiprocessors.")
    elif device_type != "cpu":
        raise ValueError(f"Hardware class {class_id} runs on cuda or cpu; received {device_type!r}.")
    if record["tier"] not in {TIER_BITWISE, TIER_BOUNDED} or type(record["reference"]) is not bool:
        raise ValueError(f"Hardware class {class_id} needs tier 1 or 2 and a boolean reference.")
    bases = record["bases"]
    return HardwareClass(
        id=_identifier(class_id, "class id"), device_type=device_type,
        compute_capability=None if device_type == "cpu" else tuple(record["compute_capability"]),
        multiprocessors=None if device_type == "cpu" else record["multiprocessors"], tier=record["tier"],
        reference=record["reference"], bases=None if bases is None else tuple(bases), members=tuple(record["members"]),
    )


def apply_profile(profile: EmbeddingProfile, device: torch.device) -> None:
    """Set every numerics flag the profile pins, process-wide; a value the profile leaves unpinned stays as it is.

    The TF32 controls are set only through the `fp32_precision` API: PyTorch refuses to read a mix of that and the
    legacy `allow_tf32` flags. The variables the image sets (`environment_variables`) must already hold their values,
    because cuBLAS reads them when its first handle is made.
    """
    numerics = profile.function["numerics"]
    for name, value in profile.function["environment_variables"].items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"{name} is {os.environ.get(name)!r}; {profile.id} needs {value!r}, set by its image.")
    backends = torch.backends
    for module, name in ((backends, "fp32_precision"), (backends.cuda.matmul, "matmul_fp32_precision"),
                         (backends.cudnn, "cudnn_fp32_precision"), (backends.cudnn.conv, "cudnn_conv_fp32_precision"),
                         (backends.cudnn.rnn, "cudnn_rnn_fp32_precision")):
        if numerics[name] is not None:
            module.fp32_precision = numerics[name]
    matmul = backends.cuda.matmul
    for prefix in ("bf16", "fp16"):
        allowed, split_k = numerics[f"{prefix}_reduced_precision_reduction"], numerics[f"{prefix}_reduced_precision_reduction_split_k"]
        if allowed is not None and split_k is not None:
            setattr(matmul, f"allow_{prefix}_reduced_precision_reduction", (allowed, split_k))
    if numerics["fp16_accumulation"] is not None:
        matmul.allow_fp16_accumulation = numerics["fp16_accumulation"]
    if numerics["deterministic_algorithms"] is not None:
        torch.use_deterministic_algorithms(numerics["deterministic_algorithms"], warn_only=bool(numerics["deterministic_warn_only"]))
    if numerics["fill_uninitialized_memory"] is not None:
        torch.utils.deterministic.fill_uninitialized_memory = numerics["fill_uninitialized_memory"]  # type: ignore[attr-defined]
    if numerics["cudnn_deterministic"] is not None:
        backends.cudnn.deterministic = numerics["cudnn_deterministic"]
    if numerics["cudnn_benchmark"] is not None:
        backends.cudnn.benchmark = numerics["cudnn_benchmark"]
    if numerics["fp16_bf16_reduction_math_sdp"] is not None:
        backends.cuda.allow_fp16_bf16_reduction_math_sdp(numerics["fp16_bf16_reduction_math_sdp"])
    if device.type == "cuda":
        if numerics["preferred_blas_library"] is not None:
            backends.cuda.preferred_blas_library(numerics["preferred_blas_library"])
        if numerics["cublas_workspace_bytes"] is not None:
            backends.cuda.cublas_workspace_size(numerics["cublas_workspace_bytes"])
        if numerics["cublaslt_workspace_bytes"] is not None:
            backends.cuda.cublaslt_workspace_size(numerics["cublaslt_workspace_bytes"])


@contextmanager
def restored_numerics() -> Iterator[None]:
    """Put every process-wide flag `apply_profile` sets back as it was when the block ends.

    For a process that computes under a profile for a while and then goes on under its own flags, such as a test suite;
    the CUDA-only settings (BLAS library, workspaces) are left alone, since reading them initializes CUDA.
    """
    backends = torch.backends
    modules = (backends, backends.cuda.matmul, backends.cudnn, backends.cudnn.conv, backends.cudnn.rnn)
    precisions = [module.fp32_precision for module in modules]
    matmul = backends.cuda.matmul
    reductions = {prefix: (getattr(matmul, f"allow_{prefix}_reduced_precision_reduction"),
                           getattr(matmul, f"allow_{prefix}_reduced_precision_reduction_split_k")) for prefix in ("bf16", "fp16")}
    accumulation = matmul.allow_fp16_accumulation
    deterministic = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    fill = torch.utils.deterministic.fill_uninitialized_memory  # type: ignore[attr-defined]
    cudnn = (backends.cudnn.deterministic, backends.cudnn.benchmark)
    sdp = backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    try:
        yield
    finally:
        for module, precision in zip(modules, precisions, strict=True):
            module.fp32_precision = precision
        for prefix, state in reductions.items():
            setattr(matmul, f"allow_{prefix}_reduced_precision_reduction", state)
        matmul.allow_fp16_accumulation = accumulation
        torch.use_deterministic_algorithms(deterministic[0], warn_only=deterministic[1])
        torch.utils.deterministic.fill_uninitialized_memory = fill  # type: ignore[attr-defined]
        backends.cudnn.deterministic, backends.cudnn.benchmark = cudnn
        backends.cuda.allow_fp16_bf16_reduction_math_sdp(sdp)


def measure_numerics(device: torch.device) -> dict[str, Any]:
    """Every numerics flag as PyTorch reports it now. On a CPU device the CUDA-only getters read None."""
    backends = torch.backends
    matmul = backends.cuda.matmul
    cuda = device.type == "cuda"
    return {
        "preferred_blas_library": _blas_name(backends.cuda.preferred_blas_library()) if cuda else None,
        "bf16_reduced_precision_reduction": bool(matmul.allow_bf16_reduced_precision_reduction),
        "bf16_reduced_precision_reduction_split_k": bool(matmul.allow_bf16_reduced_precision_reduction_split_k),
        "fp16_reduced_precision_reduction": bool(matmul.allow_fp16_reduced_precision_reduction),
        "fp16_reduced_precision_reduction_split_k": bool(matmul.allow_fp16_reduced_precision_reduction_split_k),
        "fp16_accumulation": bool(matmul.allow_fp16_accumulation),
        "fp32_precision": backends.fp32_precision,
        "matmul_fp32_precision": matmul.fp32_precision,
        "cudnn_fp32_precision": backends.cudnn.fp32_precision,
        "cudnn_conv_fp32_precision": backends.cudnn.conv.fp32_precision,
        "cudnn_rnn_fp32_precision": backends.cudnn.rnn.fp32_precision,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "fill_uninitialized_memory": bool(torch.utils.deterministic.fill_uninitialized_memory),  # type: ignore[attr-defined]
        "cudnn_deterministic": bool(backends.cudnn.deterministic),
        "cudnn_benchmark": bool(backends.cudnn.benchmark),
        "cublas_workspace_bytes": backends.cuda.cublas_workspace_size() if cuda else None,
        "cublaslt_workspace_bytes": backends.cuda.cublaslt_workspace_size() if cuda else None,
        "fp16_bf16_reduction_math_sdp": bool(backends.cuda.fp16_bf16_reduction_math_sdp_allowed()),
        "autocast": torch.is_autocast_enabled(device.type),
    }


def _blas_name(backend: object) -> str:
    """`_BlasBackend.Cublaslt` as the name `preferred_blas_library` accepts, `cublaslt`."""
    return str(getattr(backend, "name", backend)).lower()


def check_numerics(profile: EmbeddingProfile, measured: Mapping[str, Any], device: torch.device) -> None:
    """Refuse when a measured flag differs from the profile's value; an unpinned value is recorded, never checked."""
    pinned = {name: value for name, value in profile.function["numerics"].items() if value is not None}
    if device.type != "cuda":
        pinned = {name: value for name, value in pinned.items() if name not in CUDA_ONLY_FIELDS and name != "preferred_blas_library"}
    wrong = {name: (measured[name], value) for name, value in pinned.items() if measured[name] != value}
    if wrong:
        raise RuntimeError(f"Numerics differ from {profile.id} (measured, required): {wrong}. Nothing is computed.")


@contextmanager
def attention_kernel(profile: EmbeddingProfile) -> Iterator[None]:
    """Run inside the profile's one attention kernel, which raises rather than fall back when it cannot run."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if profile.function["attention_backend"] != ATTENTION_EFFICIENT:
        raise ValueError(f"Unsupported attention backend {profile.function['attention_backend']!r}.")
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        yield


@contextmanager
def activation_peaks(model: torch.nn.Module) -> Iterator[dict[str, Tensor]]:
    """Record the largest magnitude each submodule of ``model`` outputs while the context is open, by module name.

    The FP16 headroom audit: FP16 holds magnitudes up to 65,504, so a precision mode that computes in FP16 needs every
    activation well below it. The hooks only read outputs, so the computation and its digests are unchanged. A tuple
    output counts its first tensor.
    """
    peaks: dict[str, Tensor] = {}

    def recorder(name: str) -> Any:
        def record(module: torch.nn.Module, inputs: Any, output: Any) -> None:
            tensor = next((item for item in output if isinstance(item, Tensor)), None) if isinstance(output, tuple) else output
            if isinstance(tensor, Tensor) and tensor.is_floating_point() and tensor.numel():
                peak = tensor.detach().abs().amax().float()  # () on the module's device, read at the end
                peaks[name] = peak if name not in peaks else torch.maximum(peaks[name], peak)
        return record

    handles = [module.register_forward_hook(recorder(name)) for name, module in model.named_modules() if name]
    try:
        yield peaks
    finally:
        for handle in handles:
            handle.remove()


def headroom_report(peaks: Mapping[str, Tensor], *, top: int = 8) -> dict[str, Any]:
    """The audit's verdict: the largest activation, FP16's maximum over it, and the modules with the largest peaks."""
    # peaks: () one scalar per module name
    values = sorted(((float(peak), name) for name, peak in peaks.items()), reverse=True)
    if not values:
        raise ValueError("The headroom audit recorded no activation.")
    largest = values[0][0]
    return {"largest": largest, "fp16_headroom": float(torch.finfo(torch.float16).max) / max(largest, 1e-30),
            "modules": len(values), "peaks": [{"module": name, "largest": value} for value, name in values[:top]]}


def in_container() -> bool:
    """Whether this process runs in a container: Docker's marker file, or a Modal task."""
    return Path("/.dockerenv").exists() or "MODAL_TASK_ID" in os.environ


def platform_key() -> str:
    """The platform a closure is attested for, such as `linux/aarch64` (a GH200) or `linux/x86_64` (Modal)."""
    return f"{platform.system().lower()}/{platform.machine()}"


def _site_packages() -> Path:
    return Path(torch.__file__).resolve().parent.parent


def closure_files() -> dict[str, Path]:
    """The binaries GPU arithmetic runs in: every library of the CUDA wheels and torch's kernel libraries."""
    site = _site_packages()
    files = {f"torch/lib/{name}": site / "torch" / "lib" / name for name in TORCH_LIBRARIES
             if (site / "torch" / "lib" / name).is_file()}
    nvidia = site / "nvidia"
    if nvidia.is_dir():
        for path in sorted(nvidia.rglob("*.so*")):
            if path.is_file() and not path.is_symlink():
                files[path.relative_to(site).as_posix()] = path
    return files


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def loaded_numerics_libraries() -> list[Path]:
    """The CUDA arithmetic libraries mapped into this process now, read from `/proc/self/maps`."""
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    loaded = set()
    for line in maps.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and NUMERICS_LIBRARY.search(Path(parts[5].strip()).name):
            loaded.add(Path(parts[5].strip()).resolve())
    return sorted(loaded)


def attest_environment() -> dict[str, Any]:
    """The library closure of this process: package versions, CUDA and cuDNN versions, and the binaries' SHA-256.

    Every CUDA arithmetic library mapped into the process must be one of the hashed files, so a second copy found
    first on the library path cannot do the arithmetic unseen. ``closure_sha256`` is what a profile lists per platform.
    """
    packages = {}
    for name in CLOSURE_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    files = closure_files()
    known = {path.resolve() for path in files.values()}
    stray = [str(path) for path in loaded_numerics_libraries() if path not in known]
    if stray:
        raise RuntimeError(f"CUDA arithmetic libraries outside the closure are loaded: {stray}. Run the profile's image.")
    closure = {
        "schema": CLOSURE_SCHEMA, "platform": platform_key(), "python": platform.python_version(), "packages": packages,
        "torch_cuda": torch.version.cuda, "torch_git": torch.version.git_version,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "libraries": {name: _sha256_file(path) for name, path in files.items()},
    }
    return {"closure_sha256": _digest(closure), "closure": closure}


def check_environment(profile: EmbeddingProfile, attestation: Mapping[str, Any]) -> None:
    """Refuse outside a container, on a platform the profile does not list, or with another library closure.

    A draft profile with no closure recorded for the platform accepts any closure: that is how its first attestation
    is taken. A qualified profile pins every platform's.
    """
    if not in_container():
        raise RuntimeError("An embedding profile runs only in its container (Docker or Modal); this process is on a host.")
    expected = profile.function["environments"]
    platform_name = attestation["closure"]["platform"]
    if platform_name not in expected:
        raise RuntimeError(f"{profile.id} lists no closure for {platform_name}; it covers {sorted(expected)}.")
    pinned = expected[platform_name]
    if pinned is not None and pinned != attestation["closure_sha256"]:
        raise RuntimeError(
            f"This container's library closure is {attestation['closure_sha256']}, not the {pinned} that {profile.id} "
            f"pins for {platform_name}. An image upgrade is a new profile and a rebuild."
        )


@dataclass(frozen=True, slots=True)
class DeviceRecord:
    """The evidence of one device: what defines its class, and what only says where a row was made."""

    device_type: str
    name: str
    compute_capability: tuple[int, int] | None
    multiprocessors: int | None
    total_memory_bytes: int | None
    uuid: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"device_type": self.device_type, "name": self.name,
                "compute_capability": None if self.compute_capability is None else list(self.compute_capability),
                "multiprocessors": self.multiprocessors, "total_memory_bytes": self.total_memory_bytes, "uuid": self.uuid}


def describe_device(device: torch.device) -> DeviceRecord:
    """The device's name, capability, multiprocessors and memory."""
    if device.type != "cuda":
        return DeviceRecord("cpu", platform.machine(), None, None, None, None)
    properties = torch.cuda.get_device_properties(device)
    return DeviceRecord("cuda", properties.name, (properties.major, properties.minor), properties.multi_processor_count,
                        properties.total_memory, str(getattr(properties, "uuid", "")) or None)


def device_class(profile: EmbeddingProfile, record: DeviceRecord, *, base: str, enrolling: bool = False) -> HardwareClass:
    """The hardware class ``record`` belongs to, refused unless the device is enrolled and the class covers ``base``.

    ``enrolling`` (qualification) skips the membership check: a device joins a class only by a qualification receipt
    whose canary digests equal the class's golden ones. A CUDA device in no class of the profile is measured under a
    provisional tier-2 class named from its capability and multiprocessors (`provisional_class`), which the profile
    lists only once its receipts are read; only qualification runs under one. The reference class also qualifies a base
    it does not list yet (the random control, 2026-10-07), whose enrollment then lists it; another class's list is the
    bases it may ever compute, such as no ESMC-6B on an A10.
    """
    if record.device_type == "cpu":
        matches = [item for item in profile.classes.values() if item.device_type == "cpu"]
    else:
        matches = [item for item in profile.classes.values() if item.device_type == "cuda"
                   and item.compute_capability == record.compute_capability and item.multiprocessors == record.multiprocessors]
    if not matches and enrolling and record.device_type == "cuda":
        return provisional_class(record)
    if not matches:
        raise RuntimeError(
            f"{record.name} (capability {record.compute_capability}, {record.multiprocessors} multiprocessors) is in no "
            f"hardware class of {profile.id}; its classes are {sorted(profile.classes)}."
        )
    (hardware,) = matches
    if hardware.bases is not None and base not in hardware.bases and not (enrolling and hardware.reference):
        raise RuntimeError(f"Class {hardware.id} is qualified for {list(hardware.bases)}, not {base}.")
    if not enrolling and record.name not in hardware.members:
        raise RuntimeError(
            f"{record.name} is not enrolled in class {hardware.id} of {profile.id} (members: {list(hardware.members)}). "
            "Run `job qualify` on it first."
        )
    return hardware


def provisional_class(record: DeviceRecord) -> HardwareClass:
    """The class a CUDA device in no class of a profile is qualified under: tier 2, every base, no members.

    Named like the profile's classes, `<architecture>_sm<capability>_<multiprocessors>`, so a receipt's id is the one
    the profile takes when it lists the class.
    """
    assert record.compute_capability is not None and record.multiprocessors is not None  # a CUDA device has both
    major, minor = record.compute_capability
    return HardwareClass(id=f"{ARCHITECTURES.get((major, minor), 'cuda')}_sm{major}{minor}_{record.multiprocessors}",
                         device_type="cuda", compute_capability=(major, minor), multiprocessors=record.multiprocessors,
                         tier=TIER_BOUNDED, reference=False, bases=None, members=())


def check_live_class(hardware: HardwareClass, record: DeviceRecord) -> None:
    """Refuse unless the device computing now (``record``) has the capability and multiprocessors of ``hardware``.

    Every capture runs it, and a capture runs again before each commit (`FeatureExtractionContract.before_commit`), so
    no row is committed under a class its device does not have.
    """
    if record.device_type != hardware.device_type or (hardware.device_type == "cuda" and (
            record.compute_capability, record.multiprocessors) != (hardware.compute_capability, hardware.multiprocessors)):
        raise RuntimeError(
            f"{record.name} (capability {record.compute_capability}, {record.multiprocessors} multiprocessors) is not a "
            f"device of class {hardware.id}; nothing is committed."
        )


def check_reference_class(profile: EmbeddingProfile, hardware: HardwareClass, command: str) -> None:
    """Refuse ``command`` unless the device is of the profile's reference class.

    Canonical rows come from the reference class alone (Hopper with 132 multiprocessors: GH200, H100 SXM, H200), whose
    members compute them bitwise alike; every other class only qualifies, so its distance from the reference is measured
    and nothing it computes is kept.
    """
    reference = profile.reference_class
    if hardware.id != reference.id:
        raise RuntimeError(f"`{command}` runs on the reference class {reference.id} of {profile.id} only (devices "
                           f"{list(reference.members)}); this device is class {hardware.id}, which only qualifies.")


def check_publishable(manifest: Mapping[str, Any], profile: EmbeddingProfile) -> None:
    """Refuse to publish a profile store unless its profile is qualified and unchanged and its rows are the reference class's.

    ``profile`` is the profile the manifest names, as loaded now. The two repositories hold the reference class's rows
    only, so a reader's exact `verify` holds there; a store computed on another class names that class in its keys and
    stays where it was made.
    """
    binding, stored_class = manifest["profile"], manifest["hardware_class"]["id"]
    if (profile.id, profile.sha256) != (binding["id"], binding["sha256"]):
        raise RuntimeError(f"The store was computed under {binding['id']} {binding['sha256'][:12]}, and that profile is now "
                           f"{profile.sha256[:12]}: a changed function is a new profile and a rebuild.")
    if profile.status != STATUS_QUALIFIED:
        raise RuntimeError(f"{profile.id} is a draft; its stores are scratch and are never published.")
    if stored_class != profile.reference_class.id:
        raise RuntimeError(f"The store was computed on class {stored_class}; the repositories hold rows of the reference "
                           f"class {profile.reference_class.id} only.")


def canary_panel(geometry: BatchGeometry, *, max_residues: int) -> list[str]:
    """The canary's proteins: every bucket filled to its full batch, with rare residues, a duplicate and a cropped one.

    Deterministic: the same sequences in the same order on every machine. A bucket's batches hold lengths spread over
    the bucket, so the panel runs every shape the geometry has at its full row count.
    """
    rng = random.Random(CANARY_SEED)
    panel: list[str] = []
    for columns, rows in geometry.shapes().items():
        longest = min(columns - 2, max_residues)
        shortest = max(1, columns - geometry.bucket_tokens - 1)
        bucket = []
        for index in range(rows):
            length = longest - index % (longest - shortest + 1)
            bucket.append("".join(rng.choice(CANARY_RESIDUES) for _ in range(length)))
        rare = bucket[0]
        bucket[0] = "".join(CANARY_RARE[i % len(CANARY_RARE)] if i % 7 == 3 else residue for i, residue in enumerate(rare))
        if rows > 1:
            bucket[-1] = bucket[0]  # one protein twice in one batch: its two rows must be equal
        panel += bucket
    panel[-1] = "".join(rng.choice(CANARY_RESIDUES) for _ in range(CANARY_OVERLENGTH))  # cropped to max_residues
    return panel


def stream_digests(batches: Sequence[Mapping[str, Mapping[str, Tensor]]]) -> dict[str, str]:
    """One SHA-256 per stream over the bytes of every batch's output tensors, in order: a bitwise fingerprint."""
    # batches: (b, w) pooled, (n, w) per token, (n, k) top-k indices and values, for each batch, stream and tensor name
    digests: dict[str, Any] = {}
    for batch in batches:
        for stream, tensors in batch.items():
            digest = digests.setdefault(stream, hashlib.sha256())
            for name in sorted(tensors):
                tensor = tensors[name].detach().to("cpu").contiguous()  # same shape, host memory
                digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)};".encode("ascii"))
                digest.update(tensor.view(torch.uint8).numpy().tobytes() if tensor.numel() else b"")
    return {stream: digest.hexdigest() for stream, digest in sorted(digests.items())}


def check_canary(profile: EmbeddingProfile, hardware: HardwareClass, base: str, digests: Mapping[str, str]) -> str:
    """Hold a run's canary digests to the golden ones of its class and base; returns "matched" or "unrecorded".

    A draft profile without golden digests for the pair is "unrecorded" (qualification records them); a qualified one
    refuses, except for a base the class does not list yet, which only qualification reaches (`device_class`). A digest
    that differs stops the run before any row is written. A pooled run computes a subset of the token streams (the same
    taps over the same forward), so each stream it has is held to its golden digest.
    """
    golden = profile.canary.get("golden", {}).get(hardware.id, {}).get(base)
    if golden is None:
        listed = hardware.bases is None or base in hardware.bases
        if profile.status == STATUS_QUALIFIED and listed:
            raise RuntimeError(f"{profile.id} has no golden canary digests for {base} on class {hardware.id}.")
        return "unrecorded"
    wrong = sorted(stream for stream in digests if golden.get(stream) != digests[stream])
    if wrong or not digests:
        raise RuntimeError(
            f"The canary of {base} on class {hardware.id} differs from {profile.id}'s golden digests in {wrong or 'its streams'}. "
            "A driver, library, flag or device changed what this machine computes; nothing is written."
        )
    return "matched"


def golden_buckets(geometry: BatchGeometry) -> tuple[int, ...]:
    """The golden check's buckets: the geometry's shortest, middle and longest, each run as one full batch."""
    columns = sorted(geometry.shapes())
    return tuple(sorted({columns[0], columns[len(columns) // 2], columns[-1]}))


def golden_panel(geometry: BatchGeometry, *, max_residues: int) -> dict[int, list[str]]:
    """The golden check's proteins: for each golden bucket, one full batch with lengths spread over the bucket.

    Deterministic on every machine and apart from the canary panel. Each batch's first protein is its longest; the
    shortest bucket's first protein is the one whose per-token rows the golden file keeps (`golden_rows`).
    """
    panel: dict[int, list[str]] = {}
    for columns in golden_buckets(geometry):
        rng = random.Random(f"{GOLDEN_SEED}:{columns}")
        longest = min(columns - 2, max_residues)
        shortest = max(1, columns - geometry.bucket_tokens - 1)
        panel[columns] = ["".join(rng.choice(CANARY_RESIDUES) for _ in range(longest - index % (longest - shortest + 1)))
                          for index in range(geometry.rows(columns))]
    return panel


def golden_panel_sha256(panel: Mapping[int, Sequence[str]]) -> str:
    """The digest of a golden panel's proteins and buckets, so a golden file is held to the panel it was recorded on."""
    document = {str(columns): list(texts) for columns, texts in sorted(panel.items())}
    return hashlib.sha256(_canonical_json(document).encode("ascii")).hexdigest()


def golden_rows(batches: Mapping[int, tuple[Mapping[str, Mapping[str, Tensor]], Sequence[int]]],
                layouts: Mapping[str, str]) -> dict[str, Tensor]:
    """The reference rows a golden file keeps, named `<stream>/<bucket>/<tensor>`.

    ``batches`` maps each golden bucket to its batch's host streams and the stored rows of each sequence (a
    `PackedBatch`'s ``streams`` and ``rows``); ``layouts`` each stream's layout. Kept: the first protein's row of every
    pooled stream in every bucket, and the first protein's per-token rows in the shortest bucket only, so the file
    stays well under a megabyte per base.
    """
    # batches: (n, w) ragged, (n, k) values and indices ragged top-k or (b, w) pooled stream tensors per bucket,
    # with rows (b,) l_i + 2
    shortest = min(batches)
    tensors: dict[str, Tensor] = {}
    for columns, (streams, rows) in sorted(batches.items()):
        for stream, group in sorted(streams.items()):
            per_token = layouts[stream] in PER_TOKEN_LAYOUTS
            if per_token and columns != shortest:
                continue
            for name, tensor in sorted(group.items()):
                kept = tensor[:rows[0]] if per_token else tensor[:1]  # (l + 2, w) or (l + 2, k), or (1, w)
                tensors[f"{stream}/{columns}/{name}"] = kept.detach().to("cpu").contiguous().clone()
    return tensors  # name -> (l + 2, w) or (l + 2, k) per-token rows, or (1, w) pooled rows


def golden_path(profile: EmbeddingProfile, base: str) -> Path:
    """Where the repository keeps the golden file of ``profile`` and ``base``."""
    return GOLDEN_DIRECTORY / profile.id / f"{base}.safetensors"


def write_golden(path: Path, tensors: Mapping[str, Tensor], metadata: Mapping[str, Any]) -> None:
    """Write a golden file; one never replaces another (a changed golden is a new profile)."""
    from safetensors.torch import save_file

    # tensors: (l + 2, w), (l + 2, k) or (1, w) rows by name (`golden_rows`)
    if path.exists():
        raise FileExistsError(f"{path} exists; a golden file is recorded once per profile and base.")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(dict(tensors), str(path), metadata={"schema": GOLDEN_SCHEMA,
                                                 **{key: json.dumps(value) for key, value in metadata.items()}})


def read_golden(path: Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """A golden file's reference rows and its records."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        stored = handle.metadata() or {}
        names = handle.keys()  # a safetensors handle is not iterable
        tensors = {name: handle.get_tensor(name) for name in names}  # name -> (l + 2, w), (l + 2, k) or (1, w)
    if stored.get("schema") != GOLDEN_SCHEMA:
        raise ValueError(f"{path} is not a golden file of schema {GOLDEN_SCHEMA}.")
    return tensors, {key: json.loads(value) for key, value in stored.items() if key != "schema"}  # rows (l + 2, w), (l + 2, k), (1, w)


def check_golden(profile: EmbeddingProfile, hardware: HardwareClass, base: str, *, panel_sha256: str,
                 digests: Mapping[str, str], rows: Mapping[str, Tensor], path: Path | None = None) -> str:
    """Hold a run's golden batches to the repository's golden file bitwise; returns "matched" or "unrecorded".

    ``digests`` are the run's `stream_digests` over its golden batches and ``rows`` its `golden_rows`. A draft profile
    without a golden file is "unrecorded"; a qualified one refuses. The file must have been recorded under this
    profile's function, class, base and panel; then every stream the run computes (a pooled run computes a subset of
    the token streams) must match its digest. A difference raises with each stream's distance from the kept rows.
    """
    # rows: (l + 2, w), (l + 2, k) or (1, w) by name, as `golden_rows` names them
    path = golden_path(profile, base) if path is None else path
    if not path.is_file():
        if profile.status == STATUS_QUALIFIED:
            raise RuntimeError(f"{profile.id} has no golden file for {base} at {path}; record it on the reference class "
                               "with `job golden`.")
        return "unrecorded"
    reference, records = read_golden(path)
    expected = {"profile": profile.binding(), "hardware_class": hardware.id, "base": base, "panel_sha256": panel_sha256}
    differing = {key: records.get(key) for key, value in expected.items() if records.get(key) != value}
    if differing:
        raise RuntimeError(f"{path} was recorded for {differing}, not {({key: expected[key] for key in differing})}.")
    golden = records["digests"]
    unrecorded = sorted(set(digests) - set(golden))
    if unrecorded or not digests:
        raise RuntimeError(f"{path} records no golden digest for the streams {unrecorded or 'of this run'}.")
    wrong = sorted(stream for stream in digests if golden[stream] != digests[stream])
    if wrong:
        distances = {}
        for name, tensor in sorted(reference.items()):
            if name.split("/")[0] not in wrong or name not in rows:
                continue
            observed = rows[name]
            distances[name] = (f"{int((observed != tensor).sum())} of {tensor.numel()} differ"
                               if not tensor.is_floating_point() else f"relative L2 {relative_l2(tensor, observed):.3g}")
        raise RuntimeError(
            f"The golden batches of {base} on class {hardware.id} differ from {path.name} in {wrong} ({distances}). "
            "A driver, library, flag or device changed what this machine computes; nothing is written."
        )
    return "matched"


def bf16_steps(reference: Tensor, observed: Tensor) -> Tensor:
    """Each element's difference in units of one BF16 step at the reference value's magnitude."""
    # reference, observed: (n, w) the same rows from two classes
    if reference.shape != observed.shape:
        raise ValueError(f"Compared tensors differ in shape: {tuple(reference.shape)} and {tuple(observed.shape)}.")
    reference, observed = reference.double(), observed.double()  # (n, w) each
    magnitude = reference.abs().clamp_min(torch.finfo(torch.bfloat16).tiny)  # (n, w)
    step = torch.exp2(torch.floor(torch.log2(magnitude)) - 7)  # (n, w) the spacing of BF16 values near the reference
    return (observed - reference).abs() / step  # (n, w)


def relative_l2(reference: Tensor, observed: Tensor) -> float:
    """The L2 norm of the difference over the reference's norm, in float64."""
    # reference, observed: (n, w) or (w,), the same shape
    if reference.shape != observed.shape:
        raise ValueError(f"Compared tensors differ in shape: {tuple(reference.shape)} and {tuple(observed.shape)}.")
    difference = observed.double() - reference.double()  # same shape, float64
    return float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference.double()).clamp_min(1e-30))


def panel_selection(panel: Sequence[str], geometry: BatchGeometry) -> list[str]:
    """The proteins whose per-token rows a panel file keeps: the first of each bucket, so every shape is compared."""
    selected, start = [], 0
    for rows in geometry.shapes().values():
        selected.append(panel[start])
        start += rows
    return selected


def panel_tensors(view: Any, layouts: Mapping[str, str], texts: Sequence[str], selected: Sequence[str],
                  keys: Mapping[str, str]) -> dict[str, Tensor]:
    """A qualification's comparison rows, read back from its store: every protein's pooled rows, the selected proteins'
    per-token rows, keyed `<stream>/<protein sha256>`, with `/indices` and `/values` for top-k and sparse rows.

    ``view`` is the store's `FeatureView`, ``layouts`` each stream's layout, ``keys`` each text's row key.
    """
    tensors: dict[str, Tensor] = {}
    for stream, layout in sorted(layouts.items()):
        per_token = layout in PER_TOKEN_LAYOUTS
        chosen = list(selected) if per_token else list(texts)
        step = 1 if per_token else view.budget.max_rows  # reads within the view's budget: a protein, or a chunk of rows
        for start in range(0, len(chosen), step):
            chunk = chosen[start:start + step]
            for text, row in zip(chunk, view.rows(stream, chunk), strict=True):
                name = f"{stream}/{keys[text]}"
                if isinstance(row, Tensor):
                    tensors[name] = row.contiguous()  # (l + 2, w) per token, or (w,) pooled
                else:  # a TopKRow, (l + 2, k), or a SparseRow, (nnz,)
                    tensors[f"{name}/indices"] = row.indices.contiguous()
                    tensors[f"{name}/values"] = row.values.contiguous()
    return tensors  # name -> (l + 2, w) or (w,) rows, or (l + 2, k) or (nnz,) indices and values


def compare_panels(reference: Mapping[str, Tensor], observed: Mapping[str, Tensor]) -> dict[str, dict[str, float]]:
    """How far another device's panel rows lie from the reference's, per stream.

    Dense per-token rows: each element's difference in BF16 steps at the reference value (largest, 99.9th percentile,
    median, share above one step) and each protein's relative L2. Pooled dense and sparse rows: each protein's relative
    L2. Top-k rows: per token, the Jaccard of the codes holding a nonzero value and the relative L2 of the values over
    the union of codes. Both panels must hold the same rows.
    """
    # reference, observed: (l + 2, w) or (w,) rows, or (l + 2, k) or (nnz,) indices and values, by name (panel_tensors)
    if set(reference) != set(observed):
        raise ValueError(f"The panels hold different rows: {sorted(set(reference) ^ set(observed))[:4]} ...")
    rows: dict[str, set[str]] = {}
    for name in reference:
        stream, key, *_ = name.split("/")
        rows.setdefault(stream, set()).add(key)
    report: dict[str, dict[str, float]] = {}
    for stream, keys in sorted(rows.items()):
        ordered = sorted(keys)
        if f"{stream}/{ordered[0]}/indices" in reference:
            report[stream] = _compare_codes(reference, observed, stream, ordered)
        else:
            report[stream] = _compare_dense(reference, observed, stream, ordered)
    return report


def protein_differences(reference: Mapping[str, Tensor], observed: Mapping[str, Tensor], stream: str,
                        lengths: Mapping[str, int], *, worst: int = 5) -> dict[str, Any]:
    """Each protein's relative L2 between two panels' rows of one stream, summarized: quantiles and the ``worst`` proteins
    with their lengths, and for sparse and top-k rows the ratio of observed to reference values on the codes both hold.

    Code rows are compared densified over the codebook, so a code one side lacks counts at its full value. The value ratio
    tells a systematic change (a readout that shrinks values: ratios above one) from noise (ratios around one).
    ``lengths`` maps each row key to its protein's residue count.
    """
    # reference, observed: (l + 2, w) or (w,) dense rows, or (l + 2, k) or (nnz,) indices and values, by name (panel_tensors)
    keys = sorted({name.split("/")[1] for name in reference if name.split("/")[0] == stream})
    differences: dict[str, float] = {}
    ratios: list[Tensor] = []
    for key in keys:
        name = f"{stream}/{key}"
        if f"{name}/indices" not in reference:
            differences[key] = relative_l2(reference[name], observed[name])
            continue
        indices = (reference[f"{name}/indices"], observed[f"{name}/indices"])  # (l + 2, k) or (nnz,) each
        width = max((int(item.max()) + 1 for item in indices if item.numel()), default=0)
        dense_reference = _densified(indices[0], reference[f"{name}/values"], width)  # (l + 2, c) or (c,)
        dense_observed = _densified(indices[1], observed[f"{name}/values"], width)  # (l + 2, c) or (c,)
        differences[key] = relative_l2(dense_reference, dense_observed)
        shared = (dense_reference != 0) & (dense_observed != 0)  # (l + 2, c) or (c,)
        ratios.append(dense_observed[shared] / dense_reference[shared])  # (shared codes,)
    ordered = torch.tensor([differences[key] for key in keys], dtype=torch.float64).sort().values  # (proteins,)
    summary: dict[str, Any] = {
        "relative_l2_p90": float(ordered[int(0.9 * (len(ordered) - 1))]),
        "relative_l2_p99": float(ordered[int(0.99 * (len(ordered) - 1))]),
        "worst": [{"key": key, "residues": lengths[key], "relative_l2": differences[key]}
                  for key in sorted(keys, key=differences.__getitem__, reverse=True)[:worst]],
    }
    if ratios:
        pooled = torch.cat(ratios).sort().values  # (shared codes over all proteins,)
        if pooled.numel():
            summary.update({f"shared_value_ratio_{label}": float(pooled[int(share * (pooled.numel() - 1))])
                            for label, share in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))})
    return summary


def _densified(indices: Tensor, values: Tensor, width: int) -> Tensor:
    """Top-k rows (r, k) as dense (r, c) rows, or one sparse row (nnz,) as a dense (c,) row, in float64."""
    # indices, values: (r, k) top-k rows or (nnz,) one sparse row
    if indices.ndim == 1:
        return torch.zeros(width, dtype=torch.float64).index_put_((indices.long(),), values.double())  # (c,)
    dense = torch.zeros(indices.shape[0], width, dtype=torch.float64)  # (r, c)
    return dense.scatter_add_(1, indices.long(), values.double())  # (r, c); a zero slot adds nothing


def row_scale_steps(reference: Tensor, observed: Tensor) -> Tensor:
    """Each element's difference in BF16 steps at its row's RMS: one step is the BF16 spacing of a typical value.

    `bf16_steps` measures at each element's own magnitude, where a value near zero has a vanishing step; this measures
    every element of a row against one scale, the row's.
    """
    # reference, observed: (r, w) rows or (w,) one row
    if reference.shape != observed.shape:
        raise ValueError(f"Compared tensors differ in shape: {tuple(reference.shape)} and {tuple(observed.shape)}.")
    width = reference.shape[-1]
    rows_reference, rows_observed = reference.double().reshape(-1, width), observed.double().reshape(-1, width)  # (r, w)
    rms = rows_reference.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(torch.finfo(torch.bfloat16).tiny)  # (r, 1)
    step = torch.exp2(torch.floor(torch.log2(rms)) - 7)  # (r, 1) the BF16 spacing at the row's RMS
    return (rows_observed - rows_reference).abs() / step  # (r, w)


def _quantiles(steps: Tensor, prefix: str, share: str) -> dict[str, float]:
    """The largest, 99.9th percentile and median of sorted step counts, and (as ``share``) the share above one step."""
    # steps: (m,) sorted ascending
    return {f"{prefix}_max": float(steps[-1]), f"{prefix}_p999": float(steps[int(0.999 * (steps.numel() - 1))]),
            f"{prefix}_median": float(steps[(steps.numel() - 1) // 2]), share: float((steps > 1).double().mean())}


def _compare_dense(reference: Mapping[str, Tensor], observed: Mapping[str, Tensor], stream: str,
                   keys: Sequence[str]) -> dict[str, float]:
    # reference, observed: (l + 2, w) per-token or (w,) pooled rows of ``stream``, by name
    pairs = [(reference[f"{stream}/{key}"], observed[f"{stream}/{key}"]) for key in keys]
    relative = torch.tensor([relative_l2(ref, obs) for ref, obs in pairs])
    summary = {"proteins": float(len(keys)), "relative_l2_max": float(relative.max()),
               "relative_l2_median": float(relative.median())}
    row_steps = torch.cat([row_scale_steps(ref, obs).flatten().float() for ref, obs in pairs]).sort().values  # (m,)
    summary.update({"elements": float(row_steps.numel()),
                    **_quantiles(row_steps, "row_bf16_steps", "share_above_one_row_step")})
    if pairs[0][0].ndim == 2:  # per token: (l + 2, w) rows, also measured at each element's own magnitude
        steps = torch.cat([bf16_steps(ref, obs).flatten().float() for ref, obs in pairs]).sort().values  # (m,)
        summary.update(_quantiles(steps, "bf16_steps", "share_above_one_step"))
    return summary


def _compare_codes(reference: Mapping[str, Tensor], observed: Mapping[str, Tensor], stream: str,
                   keys: Sequence[str]) -> dict[str, float]:
    # reference, observed: (l + 2, k) top-k or (nnz,) sparse indices and values of ``stream``, by name
    jaccards, relatives = [], []
    for key in keys:
        ref_indices, ref_values = reference[f"{stream}/{key}/indices"], reference[f"{stream}/{key}/values"].double()
        obs_indices, obs_values = observed[f"{stream}/{key}/indices"], observed[f"{stream}/{key}/values"].double()
        if ref_indices.ndim == 1:  # a pooled sparse row (nnz,): compared densified over the codebook
            both = torch.cat([ref_indices, obs_indices])  # (nnz_ref + nnz_obs,)
            width = int(both.max()) + 1 if both.numel() else 0
            dense_ref = torch.zeros(width, dtype=torch.float64).index_put_((ref_indices.long(),), ref_values)  # (c,)
            dense_obs = torch.zeros(width, dtype=torch.float64).index_put_((obs_indices.long(),), obs_values)  # (c,)
            relatives.append(torch.tensor([relative_l2(dense_ref, dense_obs)], dtype=torch.float64))
            jaccards.append(_jaccard(dense_ref != 0, dense_obs != 0))
            continue
        # per token: indices and values (r, k); same[t, i, j] says code i of the reference is code j observed
        same = ref_indices[:, :, None] == obs_indices[:, None, :]  # (r, k, k)
        matched = (same * obs_values[:, None, :]).sum(-1)  # (r, k) the observed value of each reference code
        unmatched = ~same.any(1)  # (r, k) observed codes the reference lacks
        squared = ((ref_values - matched) ** 2).sum(-1) + (obs_values ** 2 * unmatched).sum(-1)  # (r,)
        relatives.append(squared.sqrt() / ref_values.pow(2).sum(-1).sqrt().clamp_min(1e-30))  # (r,)
        live_ref, live_obs = ref_values != 0, obs_values != 0  # (r, k) each
        shared = (same & live_ref[:, :, None] & live_obs[:, None, :]).any(-1).sum(-1)  # (r,)
        union = live_ref.sum(-1) + live_obs.sum(-1) - shared  # (r,)
        jaccards.append(torch.where(union > 0, shared / union.clamp_min(1), torch.ones(len(union), dtype=torch.float64)))
    jaccard, relative = torch.cat(jaccards).double(), torch.cat(relatives)  # (rows compared,) each
    return {"rows": float(jaccard.numel()), "jaccard_min": float(jaccard.min()), "jaccard_mean": float(jaccard.mean()),
            "share_identical_codes": float((jaccard == 1).double().mean()), "relative_l2_max": float(relative.max()),
            "relative_l2_median": float(relative.median())}


def _jaccard(left: Tensor, right: Tensor) -> Tensor:
    """The Jaccard of two masks over one codebook, as a one-element float64 tensor (1 when both are empty)."""
    # left, right: (c,) boolean
    union = int((left | right).sum())
    return torch.tensor([int((left & right).sum()) / union if union else 1.0], dtype=torch.float64)  # (1,)


def profile_evidence(profile: EmbeddingProfile, attestation: Mapping[str, Any], record: DeviceRecord,
                     hardware: HardwareClass, measured: Mapping[str, Any]) -> dict[str, Any]:
    """What a run records beside its rows: the profile, the closure, the device and the measured flags."""
    return {"profile": profile.binding(), "profile_status": profile.status, "closure_sha256": attestation["closure_sha256"],
            "closure": attestation["closure"], "device": record.to_dict(), "hardware_class": hardware.record(),
            "numerics": dict(measured), "python": sys.version.split()[0]}


__all__ = [
    "ATTENTION_EFFICIENT", "BACKBONE_DTYPES", "CANARY_SEED", "PANEL_SCHEMA", "PRECISION_BF16_BACKBONE", "PRECISION_FP16_BACKBONE",
    "PRECISION_FP32_BACKBONE", "PROFILE_DIRECTORY", "STATUS_DRAFT",
    "STATUS_QUALIFIED",
    "TIER_BITWISE", "TIER_BOUNDED", "DeviceRecord", "EmbeddingProfile", "HardwareClass", "activation_peaks", "apply_profile",
    "headroom_report", "precision_of", "row_scale_steps",
    "attention_kernel", "attest_environment", "bf16_steps", "canary_panel", "check_canary", "check_environment",
    "check_live_class", "check_numerics", "check_publishable", "closure_files", "compare_panels", "describe_device",
    "device_class", "in_container", "load_profile",
    "measure_numerics", "panel_selection", "panel_tensors", "parse_profile", "platform_key", "profile_evidence",
    "protein_differences", "provisional_class", "relative_l2", "stream_digests",
]
