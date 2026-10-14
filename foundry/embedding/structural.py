"""Frozen ESMFold2 structural features through FastPLMs' streaming tap engine.

Each layer is normalized and projected on arrival. Only width-256 projections survive to the
reference einsum, preserving its reduction order without retaining full-width hidden states.
No encoder, checkpoint download, or dependency on the Atlas research package lives here.

Symbols: b sequences of a batch; l token columns of the padded batch (residues, or l + 2 with CLS and EOS);
d hidden width; n hidden states of the mixture (depth + 1). The output is (b, l, 256); which rows are kept
(residues only, or CLS, residues and EOS) is the engine's mask, not this projection's.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING
from torch import Tensor


if TYPE_CHECKING:
    from fastplms.embeddings import StreamingTap, TapBatch


STRUCTURAL_TAP_SCHEMA = 1
STRUCTURAL_TAP_TOKEN_SCHEMA = 2  # the same arithmetic over all l + 2 token rows, CLS and EOS included
STRUCTURAL_TAP_POOLED_SCHEMA = 3  # those l + 2 rows pooled per sequence in float32 (`structural_mean_var`)
STRUCTURAL_WEIGHT_NAMES = (
    "layer_logits", "projection.0.weight", "projection.0.bias", "projection.1.weight",
)


class _StructuralAccumulator:
    """One batch's narrow projections, in the released mixture's layer order."""

    def __init__(self, weights: Mapping[str, Tensor], compute_dtype: torch.dtype) -> None:
        # weights: (n,) layer_logits, (d,) norm weight and bias, (256, d) projection; FP32 and frozen.
        self.weights = weights
        self.compute_dtype = compute_dtype
        self.projected: list[Tensor] = []

    def update(self, layer: int, batch: TapBatch) -> None:
        # X: (b, l, d); projection: (256, d); output: (b, l, 256).
        X = batch.X
        projection = self.weights["projection.1.weight"]
        if layer != len(self.projected) or layer >= self.weights["layer_logits"].numel():
            raise ValueError("Structural states must arrive in their complete checkpoint order.")
        if X.shape[-1] != projection.shape[1] or X.device != projection.device:
            raise ValueError("Structural weights and hidden states must share width and device.")
        if X.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("Structural taps currently support BF16, FP16 or FP32 hidden states.")
        if self.compute_dtype == torch.float32 and X.dtype != torch.float32:
            raise ValueError("FP32 structural arithmetic requires FP32 hidden states.")
        half = self.compute_dtype != torch.float32  # BF16 or FP16 arithmetic under autocast
        with torch.autocast(X.device.type, enabled=half, dtype=self.compute_dtype if half else torch.bfloat16):
            normalized = F.layer_norm(
                X, (X.shape[-1],), self.weights["projection.0.weight"], self.weights["projection.0.bias"], 1e-5,
            )  # (b, l, d)
            self.projected.append(F.linear(normalized, projection))  # (b, l, 256)

    def finish(self) -> Tensor:
        if len(self.projected) != self.weights["layer_logits"].numel():
            raise ValueError("Structural mixture is missing checkpoint layers.")
        projected = torch.stack(self.projected)  # (n, b, l, 256)
        self.projected.clear()
        coefficients = self.weights["layer_logits"].softmax(0).to(projected.dtype)  # (n,)
        with torch.autocast(projected.device.type, enabled=False):
            return torch.einsum("n,nbld->bld", coefficients, projected)  # (b, l, 256)


def structural_tap(
    name: str, weights: Mapping[str, Tensor], *, compute_dtype: torch.dtype = torch.bfloat16,
    keep_special_tokens: bool = False, pooling: Sequence[str] | None = None,
) -> StreamingTap:
    """Bind the four verified released tensors to one frozen per-row feature tap.

    The caller verifies the checkpoint bytes and passes FP32 tensors on the backbone device.
    Keys match AtlasV2's ``StructuralMixture.state_dict``. The tensors remain borrowed and must
    stay frozen; rebuild the tap after changing them. ``capture_feature_specs`` recaptures their
    digests separately from the base and the source checkpoint before committing stored rows.
    The projection is row-wise, so one arithmetic serves both row policies: the engine decides which rows of
    the (b, l, 256) output are kept. A residue tap keeps l rows per protein; with `keep_special_tokens=True`
    the engine keeps all l + 2 (CLS, residues, EOS), and the identity records the policy. ``pooling`` (token
    rows only, such as ``("mean", "var")``) pools those l + 2 rows per protein in float32 instead, one
    (p * 256,) row per protein.
    """
    from fastplms.embeddings import StreamingTap, tensor_sha256

    # weights: (n,) layer_logits, (d,) norm weight and bias, (256, d) projection; FP32 and frozen.
    if set(weights) != set(STRUCTURAL_WEIGHT_NAMES):
        raise ValueError(f"Structural weights must contain exactly {STRUCTURAL_WEIGHT_NAMES}.")
    weights = dict(weights)
    if any(not isinstance(value, Tensor) or value.dtype != torch.float32 for value in weights.values()):
        raise ValueError("Released structural weights must be FP32 tensors.")
    logits, norm_weight, norm_bias, projection = (weights[key] for key in STRUCTURAL_WEIGHT_NAMES)
    if logits.ndim != 1 or logits.numel() < 2 or projection.ndim != 2 or projection.shape[0] != 256:
        raise ValueError("Structural weights require a layer mixture and a width-256 projection.")
    width = projection.shape[1]
    if width <= 0 or norm_weight.shape != (width,) or norm_bias.shape != (width,):
        raise ValueError("Structural normalization and projection widths disagree.")
    if len({value.device for value in weights.values()}) != 1 or projection.device.type not in {"cpu", "cuda"}:
        raise ValueError("Structural weights must share one CPU or CUDA device.")
    if any(value.requires_grad or not bool(torch.isfinite(value).all()) for value in weights.values()):
        raise ValueError("Structural weights must be frozen and finite.")
    if compute_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Structural compute_dtype must be BF16, FP16 or FP32.")
    if pooling is not None and not keep_special_tokens:
        raise ValueError("A pooled structural tap pools all l + 2 token rows; set keep_special_tokens=True.")
    schema = STRUCTURAL_TAP_TOKEN_SCHEMA if keep_special_tokens else STRUCTURAL_TAP_SCHEMA
    identity = {
        "reducer": "foundry.embedding.structural_tap",
        "schema": schema if pooling is None else STRUCTURAL_TAP_POOLED_SCHEMA,
        "arithmetic": "layerwise_projection_stacked_einsum_v1", "layer_norm_epsilon": 1e-5,
        "parameter_dtype": "float32", "compute_dtype": str(compute_dtype).removeprefix("torch."),
        "input_width": width, "output_width": 256, "state_count": logits.numel(),
        "weights_sha256": {key: tensor_sha256(value) for key, value in weights.items()},
        "rows": "all_attended_tokens" if keep_special_tokens else "biological_residue_mask",
    }
    if pooling is not None:  # the finished rows in float32, then fastplms' token pooling
        identity.update(pooling=list(pooling), pooled_dtype="float32")
    return StreamingTap(
        name, tuple(range(logits.numel())), lambda: _StructuralAccumulator(weights, compute_dtype), identity,
        required_state_count=logits.numel(), pooling=None if pooling is None else tuple(pooling),
        dtype=None if pooling is None else torch.float32,
    )
