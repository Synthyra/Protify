"""The token-wise states a tap can read inside an ESMC (ESM++) transformer, and what to read.

Every site is an (l, w) matrix per sequence: l token rows (CLS, the residues, EOS) and a width w that is the
model width d, or the SwiGLU width f for the three MLP-internal sites. A per-block site has one state per
transformer block; `embedding` and `final_norm` have one state each.

Symbols: d model width; f SwiGLU hidden width; h attention heads; d_h = d / h; n_b transformer blocks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TapSite:
    """One readable state of a block (or of the whole model) and how wide it is."""

    name: str
    family: str  # residual, attention or mlp
    per_block: bool
    width: str  # "d" or "f"
    splits_by_head: bool
    description: str


ESMC_SITES: tuple[TapSite, ...] = (
    TapSite("embedding", "residual", False, "d", False, "token embeddings, the input to block 0"),
    TapSite("attn_ln", "attention", True, "d", False, "the LayerNorm output that feeds the QKV projection"),
    TapSite("q", "attention", True, "d", True, "queries after q_ln, before the rotary rotation"),
    TapSite("k", "attention", True, "d", True, "keys after k_ln, before the rotary rotation"),
    TapSite("q_rot", "attention", True, "d", True, "queries after the rotary rotation"),
    TapSite("k_rot", "attention", True, "d", True, "keys after the rotary rotation"),
    TapSite("v", "attention", True, "d", True, "values"),
    TapSite("attn_context", "attention", True, "d", True, "softmax(QK^T / sqrt(d_h)) V, heads concatenated, the out_proj input"),
    TapSite("attn_out", "attention", True, "d", False, "out_proj output, added to the residual stream after scaling"),
    TapSite("resid_mid", "residual", True, "d", False, "the residual stream after the attention update"),
    TapSite("mlp_ln", "mlp", True, "d", False, "the LayerNorm output that feeds the MLP"),
    TapSite("mlp_gate", "mlp", True, "f", False, "the gate half of the up projection, the SiLU input"),
    TapSite("mlp_up", "mlp", True, "f", False, "the linear half of the up projection"),
    TapSite("mlp_act", "mlp", True, "f", False, "the SwiGLU output, silu(gate) * up, the down projection input"),
    TapSite("mlp_out", "mlp", True, "d", False, "the down projection output, added to the residual stream after scaling"),
    TapSite("resid_post", "residual", True, "d", False, "the block output, the residual stream after the MLP update"),
    TapSite("final_norm", "residual", False, "d", False, "the final LayerNorm output"),
)
SITES_BY_NAME = {site.name: site for site in ESMC_SITES}
# The residual stream alone: the embedding, every block output and the final norm.
RESIDUAL_SITES = ("embedding", "resid_post", "final_norm")
ALL_SITES = tuple(site.name for site in ESMC_SITES)


@dataclass(frozen=True)
class ModelDims:
    """The widths that size every site of one model."""

    d: int
    f: int
    heads: int
    blocks: int

    @property
    def head_width(self) -> int:
        return self.d // self.heads

    def width(self, site: TapSite) -> int:
        return self.d if site.width == "d" else self.f


@dataclass(frozen=True)
class TapSpec:
    """Which sites and blocks to read, and whether head-split sites also get one probe per head.

    `blocks` names transformer blocks by index (negative counts from the end); None reads all of them.
    """

    sites: tuple[str, ...] = ALL_SITES
    blocks: tuple[int, ...] | None = None
    per_head: bool = True

    def __post_init__(self) -> None:
        unknown = sorted(set(self.sites) - set(SITES_BY_NAME))
        if unknown:
            raise ValueError(f"Unknown tap sites {unknown}; known sites are {list(SITES_BY_NAME)}.")
        if len(set(self.sites)) != len(self.sites):
            raise ValueError(f"Tap sites repeat: {list(self.sites)}.")

    def block_indices(self, dims: ModelDims) -> tuple[int, ...]:
        if self.blocks is None:
            return tuple(range(dims.blocks))
        resolved = sorted({block % dims.blocks for block in self.blocks})
        if any(not -dims.blocks <= block < dims.blocks for block in self.blocks):
            raise ValueError(f"Block indices {list(self.blocks)} fall outside 0..{dims.blocks - 1}.")
        return tuple(resolved)

    def layers(self, site_name: str, dims: ModelDims) -> tuple[int, ...]:
        """The state indices a site holds: block indices for a per-block site, (0,) otherwise."""
        site = SITES_BY_NAME[site_name]
        return self.block_indices(dims) if site.per_block else (0,)


@dataclass(frozen=True)
class SiteLayout:
    """How one site of a spec is stored and probed.

    Every layer gets one probe over the full width. A head-split site read with `per_head` also gets one probe
    per layer and head over that head's `head_width` columns, so `heads` is 0 when there are no head probes.
    """

    name: str
    layers: tuple[int, ...]
    width: int
    heads: int
    head_width: int

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def full_names(self) -> list[str]:
        return [f"{self.name}:{layer}" for layer in self.layers]

    def head_names(self) -> list[str]:
        return [f"{self.name}:{layer}:h{head}" for layer in self.layers for head in range(self.heads)]


def site_layouts(spec: TapSpec, dims: ModelDims) -> dict[str, SiteLayout]:
    """The storage and probe layout of every site a spec reads, in catalog order."""
    layouts = {}
    for site in ESMC_SITES:
        if site.name not in spec.sites:
            continue
        heads = dims.heads if (spec.per_head and site.splits_by_head) else 0
        layouts[site.name] = SiteLayout(
            site.name, spec.layers(site.name, dims), dims.width(site), heads, dims.head_width
        )
    return layouts


def parse_sites(names: Sequence[str] | str | None) -> tuple[str, ...]:
    """Sites from a CLI value: 'all', 'residual', or comma-separated names."""
    if names is None or names == "all":
        return ALL_SITES
    if isinstance(names, str):
        names = [name.strip() for name in names.split(",") if name.strip()]
    expanded: list[str] = []
    for name in names:
        expanded.extend(RESIDUAL_SITES if name == "residual" else (name,))
    return tuple(dict.fromkeys(expanded))
