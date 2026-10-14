"""Forward hooks that hand every requested ESMC state to a consumer during a single forward pass.

A consumer receives `(site, layer, X)` with X of shape (b, l, w) as each state is computed, in forward order, and
must not keep a reference it does not need: pooling or probing inside the call keeps memory to one state at a
time. Rows of X at padded positions hold whatever the model computed there; consumers mask them.

Symbols: b sequences; l token columns (CLS, residues, EOS, padding); d model width; f SwiGLU hidden width;
h heads; d_h head width.
"""

from __future__ import annotations

import torch

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from torch import Tensor, nn

from .sites import ModelDims, TapSpec


TapConsumer = Callable[[str, int, Tensor], None]


def esmc_backbone(model: nn.Module) -> nn.Module:
    """The module holding `embed` and a `transformer` with `blocks` and `norm`, wherever it is nested."""
    for module in model.modules():
        transformer = getattr(module, "transformer", None)
        if (
            isinstance(getattr(module, "embed", None), nn.Embedding)
            and transformer is not None
            and hasattr(transformer, "blocks")
            and hasattr(transformer, "norm")
        ):
            return module
    raise ValueError("No ESMC backbone, a module with `embed` and `transformer.blocks`, was found in the model.")


def model_dims(model: nn.Module) -> ModelDims:
    backbone = esmc_backbone(model)
    first = backbone.transformer.blocks[0]
    return ModelDims(
        d=first.attn.out_proj.in_features,
        f=first.ffn[3].in_features,
        heads=first.attn.n_heads,
        blocks=len(backbone.transformer.blocks),
    )


@contextmanager
def tap_hooks(model: nn.Module, spec: TapSpec, consumer: TapConsumer) -> Iterator[ModelDims]:
    """Register the hooks a spec needs for the duration of the block, and remove them after."""
    backbone = esmc_backbone(model)
    dims = model_dims(model)
    wanted = set(spec.sites)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def output_hook(site: str, layer: int) -> Callable[[nn.Module, tuple[Tensor, ...], Tensor], None]:
        def hook(module: nn.Module, args: tuple[Tensor, ...], output: Tensor) -> None:
            consumer(site, layer, output.detach())  # (b, l, w)
        return hook

    def input_hook(site: str, layer: int) -> Callable[[nn.Module, tuple[Tensor, ...]], None]:
        def hook(module: nn.Module, args: tuple[Tensor, ...]) -> None:
            consumer(site, layer, args[0].detach())  # (b, l, w)
        return hook

    def block_output_hook(layer: int) -> Callable[[nn.Module, tuple[Tensor, ...], tuple[Tensor, ...]], None]:
        def hook(module: nn.Module, args: tuple[Tensor, ...], output: tuple[Tensor, ...]) -> None:
            consumer("resid_post", layer, output[0].detach())  # (b, l, d)
        return hook

    def value_hook(layer: int) -> Callable[[nn.Module, tuple[Tensor, ...], Tensor], None]:
        def hook(module: nn.Module, args: tuple[Tensor, ...], qkv: Tensor) -> None:
            # qkv: (b, l, 3 * d), chunked into queries, keys and values in that order
            consumer("v", layer, qkv[..., 2 * dims.d:].detach())  # (b, l, d)
        return hook

    def rotary_hook(layer: int) -> Callable[[nn.Module, tuple[Tensor, ...], tuple[Tensor, Tensor]], None]:
        def hook(module: nn.Module, args: tuple[Tensor, ...], output: tuple[Tensor, Tensor]) -> None:
            q_rotated, k_rotated = output  # each (b, l, h, d_h)
            if "q_rot" in wanted:
                consumer("q_rot", layer, q_rotated.detach().flatten(-2))  # (b, l, d)
            if "k_rot" in wanted:
                consumer("k_rot", layer, k_rotated.detach().flatten(-2))  # (b, l, d)
        return hook

    def up_projection_hook(layer: int) -> Callable[[nn.Module, tuple[Tensor, ...], Tensor], None]:
        def hook(module: nn.Module, args: tuple[Tensor, ...], projected: Tensor) -> None:
            # projected: (b, l, 2 * f); SwiGLU applies silu to the first half and multiplies by the second
            if "mlp_gate" in wanted:
                consumer("mlp_gate", layer, projected[..., : dims.f].detach())  # (b, l, f)
            if "mlp_up" in wanted:
                consumer("mlp_up", layer, projected[..., dims.f:].detach())  # (b, l, f)
        return hook

    try:
        if "embedding" in wanted:
            handles.append(backbone.embed.register_forward_hook(output_hook("embedding", 0)))
        for layer in spec.block_indices(dims):
            block = backbone.transformer.blocks[layer]
            attention = block.attn
            if "attn_ln" in wanted:
                handles.append(attention.layernorm_qkv[0].register_forward_hook(output_hook("attn_ln", layer)))
            if "q" in wanted:
                handles.append(attention.q_ln.register_forward_hook(output_hook("q", layer)))
            if "k" in wanted:
                handles.append(attention.k_ln.register_forward_hook(output_hook("k", layer)))
            if "q_rot" in wanted or "k_rot" in wanted:
                handles.append(attention.rotary.register_forward_hook(rotary_hook(layer)))
            if "v" in wanted:
                handles.append(attention.layernorm_qkv.register_forward_hook(value_hook(layer)))
            if "attn_context" in wanted:
                handles.append(attention.out_proj.register_forward_pre_hook(input_hook("attn_context", layer)))
            if "attn_out" in wanted:
                handles.append(attention.out_proj.register_forward_hook(output_hook("attn_out", layer)))
            if "resid_mid" in wanted:
                handles.append(block.ffn.register_forward_pre_hook(input_hook("resid_mid", layer)))
            if "mlp_ln" in wanted:
                handles.append(block.ffn[0].register_forward_hook(output_hook("mlp_ln", layer)))
            if "mlp_gate" in wanted or "mlp_up" in wanted:
                handles.append(block.ffn[1].register_forward_hook(up_projection_hook(layer)))
            if "mlp_act" in wanted:
                handles.append(block.ffn[2].register_forward_hook(output_hook("mlp_act", layer)))
            if "mlp_out" in wanted:
                handles.append(block.ffn[3].register_forward_hook(output_hook("mlp_out", layer)))
            if "resid_post" in wanted:
                handles.append(block.register_forward_hook(block_output_hook(layer)))
        if "final_norm" in wanted:
            handles.append(backbone.transformer.norm.register_forward_hook(output_hook("final_norm", 0)))
        yield dims
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def capture_states(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    spec: TapSpec,
) -> dict[tuple[str, int], Tensor]:
    """Every requested state of one batch, keyed by (site, layer). Meant for small batches and tests."""
    # input_ids, attention_mask: (b, l)
    states: dict[tuple[str, int], Tensor] = {}

    def keep(site: str, layer: int, X: Tensor) -> None:
        states[(site, layer)] = X.clone()  # (b, l, w)

    with tap_hooks(model, spec, keep):
        model(input_ids=input_ids, attention_mask=attention_mask)
    return states
