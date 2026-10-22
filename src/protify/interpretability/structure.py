"""ESMFold2's structure stream: the learned (b, l, 256) projection of ESMC's hidden states that its folding trunk reads.

ESMFold2 reads n = n_b + 1 hidden states of its ESMC, ESM++'s `hidden_states`: the embedding output, the outputs of
blocks 0 to n_b - 2, and the final norm's output in place of the last block's. One LayerNorm and one linear map project
each to 256, and the n projections are summed with the softmax of n learned logits. The weights are those of the
released ESMFold2 checkpoint for each base, pinned by revision and the SHA-256 of its file. The arithmetic is the one
the reference embedding stores use: each state normalized and projected on arrival under bfloat16 autocast with
float32 weights, the projections stacked, and the stack mixed by an einsum in its own dtype.

Symbols: b sequences; l token columns (CLS, residues, EOS, padding); d model width; n_b transformer blocks; n = n_b + 1
hidden states of the mixture.
"""

from __future__ import annotations

import hashlib
import torch
import torch.nn.functional as F

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from torch import Tensor, nn

from .sites import STRUCTURE_WIDTH, ModelDims


# The ESMFold2 checkpoint whose language model is each base, by the base's (d, n_b): repository, revision and the
# SHA-256 of its `model.safetensors`.
CHECKPOINTS = {
    (960, 30): ("biohub/ESMFold2-Experimental-Fast-base300M-step1500k", "21531e59002c9205284715e28ee802dafb430637",
                "44d6797c5efebf24753d502b40950e0874871c96ceea14f2d7f7e39cebac67fd"),
    (1152, 36): ("biohub/ESMFold2-Experimental-Fast-base600M-step1500k", "15cf2d6648692f6c17cee1297d8a285476fffa9b",
                 "11a53c1b4700b6c62a5a464fc3ec7076160c19e8584157f88e13275116cfd602"),
    (2560, 80): ("biohub/ESMFold2", "1ebf0e3481a5184eb6171d40615c79e384b48796",
                 "138fd4350d6892b81ce6be7ff9bf5a93ae9d4d3751f46a27438a3f9f0dcefa0e"),
}
KEYS = {
    "layer_logits": "language_model.base_z_combine",
    "norm_weight": "language_model.base_z_linear.0.weight",
    "norm_bias": "language_model.base_z_linear.0.bias",
    "projection": "language_model.base_z_linear.1.weight",
}
LAYER_NORM_EPS = 1e-5
COMPUTE_DTYPE = torch.bfloat16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_weights(weights: dict[str, Tensor], dims: ModelDims) -> dict[str, Tensor]:
    """The four tensors, float32, finite and sized for a base of `dims`."""
    shapes = {"layer_logits": (dims.blocks + 1,), "norm_weight": (dims.d,), "norm_bias": (dims.d,),
              "projection": (STRUCTURE_WIDTH, dims.d)}
    if set(weights) != set(shapes):
        raise ValueError(f"Structure weights must be exactly {sorted(shapes)}, not {sorted(weights)}.")
    for name, value in weights.items():
        if tuple(value.shape) != shapes[name] or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Structure weight {name} is {tuple(value.shape)} {value.dtype}; expected {shapes[name]} float32, finite.")
    return weights


@lru_cache(maxsize=None)
def released_weights(d: int, blocks: int) -> dict[str, Tensor]:
    """The released mixture and projection of the base with width `d` and `blocks` blocks, on the CPU, read from the
    pinned checkpoint after its bytes match the pinned digest."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    if (d, blocks) not in CHECKPOINTS:
        raise ValueError(f"No ESMFold2 reads a base of width {d} with {blocks} blocks; the structure stream exists for "
                         f"ESMC-300, ESMC-600 and ESMC-6B only.")
    repository, revision, digest = CHECKPOINTS[(d, blocks)]
    path = Path(hf_hub_download(repository, "model.safetensors", revision=revision))
    if _sha256(path) != digest:
        raise ValueError(f"{repository}@{revision} model.safetensors does not match its pinned SHA-256.")
    with safe_open(path, framework="pt", device="cpu") as source:
        weights = {name: source.get_tensor(key) for name, key in KEYS.items()}
    return check_weights(weights, ModelDims(d=d, f=0, heads=1, blocks=blocks))


class StructureMixture:
    """One forward pass's structure stream, built from the hidden states as they arrive in ESMFold2's order."""

    def __init__(self, weights: dict[str, Tensor], emit: Callable[[Tensor], None]) -> None:
        # weights: (n,) layer_logits, (d,) norm weight and bias, (256, d) projection, float32 on the model's device
        self.weights = weights
        self.emit = emit
        self.projected: list[Tensor] = []

    def add(self, X: Tensor) -> None:
        # X: (b, l, d) -> (b, l, 256)
        if len(self.projected) >= self.weights["layer_logits"].numel():
            raise ValueError("More hidden states arrived than ESMFold2's mixture reads; a forward pass ended unfinished.")
        with torch.autocast(X.device.type, dtype=COMPUTE_DTYPE):
            normalized = F.layer_norm(X, (X.shape[-1],), self.weights["norm_weight"], self.weights["norm_bias"], LAYER_NORM_EPS)
            self.projected.append(F.linear(normalized, self.weights["projection"]))
        if len(self.projected) == self.weights["layer_logits"].numel():
            self.emit(self.finish())

    def finish(self) -> Tensor:
        projected = torch.stack(self.projected)  # (n, b, l, 256)
        self.projected = []
        coefficients = self.weights["layer_logits"].softmax(0).to(projected.dtype)  # (n,)
        with torch.autocast(projected.device.type, enabled=False):
            return torch.einsum("n,nbld->bld", coefficients, projected)  # (b, l, 256)


def structure_hooks(backbone: nn.Module, dims: ModelDims, emit: Callable[[Tensor], None],
                    weights: dict[str, Tensor] | None = None) -> list[torch.utils.hooks.RemovableHandle]:
    """Hooks on the n states ESMFold2 reads that hand each forward pass's (b, l, 256) structure stream to `emit` once
    the final norm has run; `weights` defaults to the released ones for the backbone's dimensions."""
    if dims.blocks == 0:
        raise ValueError("A baseline encoding has no hidden states for ESMFold2 to mix.")
    device = backbone.embed.weight.device
    weights = {name: value.to(device) for name, value in check_weights(weights or released_weights(dims.d, dims.blocks), dims).items()}
    mixture = StructureMixture(weights, emit)

    def state_hook(module: nn.Module, args: tuple[Tensor, ...], output: Tensor | tuple[Tensor, ...]) -> None:
        mixture.add((output[0] if isinstance(output, tuple) else output).detach())  # (b, l, d)

    blocks = backbone.transformer.blocks
    modules = [backbone.embed, *(blocks[layer] for layer in range(dims.blocks - 1)), backbone.transformer.norm]
    return [module.register_forward_hook(state_hook) for module in modules]
