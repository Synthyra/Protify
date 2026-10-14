"""ESMC sparse autoencoder adapters over the FastPLMs hidden-state SAE contract.

FastPLMs owns the SAE runtime: `ESMplusplusModel.load_sae_models` reads a Biohub
repository and `compute_sae=True` returns one sparse `(n, c)` tensor per attached layer,
covering the valid tokens of the whole batch. `foundry.embedding` owns pooling those rows
back into sequences, through `sae_pooling`, which an exported probe shares. This module owns
what is Protify's instead: which checkpoints the registry offers, how a model name encodes
one, which reductions and which tokens pool into one vector per protein, and whether that
vector is stored sparsely.

Model names carry the full checkpoint identity, for example
`ESMC-300-SAE-l23-k64-c8192`, so embedding cache filenames, result tables, and plots
separate SAE variants without any further plumbing. A bare `ESMC-300-SAE` resolves to the
depth layer each backbone publishes, at the default sparsity and codebook width.
"""

import json
import re
import torch
import torch.nn as nn

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fastplms_loading import ensure_fastplms_submodule_on_path, load_fastplms_model
from .sae_pooling import pool_sae_outputs, validate_sae_pooling_types


ensure_fastplms_submodule_on_path()

# The SAE layer and its loader come through the ESM++ modeling module, which imports both,
# so every FastPLMs name still arrives in the one statement this path barrier allows.
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusModel,
    ESMplusplusSAELayer,
    load_esmc_sae_layers,
)

from .esmc import ESMTokenizerWrapper, get_esmc_tokenizer


@dataclass(frozen=True)
class EsmcSaeCoverage:
    """Which sparse autoencoders Biohub publishes for one ESMC backbone."""

    biohub_name: str
    model_path: str
    hidden_size: int
    num_hidden_layers: int
    # Biohub trains every sparsity and codebook width at this layer only.
    depth_layer: int
    # Codebook widths published at every other layer, always at ALL_LAYER_K.
    all_layer_codebook_dims: tuple[int, ...]


# Protify display prefix to the published coverage. `biohub_name` and `model_path` are the
# SAE repository stem and the ESM++ backbone checkpoint the SAE reads from.
ESMC_SAE_COVERAGE: dict[str, EsmcSaeCoverage] = {
    'ESMC-300': EsmcSaeCoverage('ESMC-300M', 'Synthyra/ESMplusplus_small', 960, 30, 23, (16384,)),
    'ESMC-600': EsmcSaeCoverage('ESMC-600M', 'Synthyra/ESMplusplus_large', 1152, 36, 27, (16384,)),
    'ESMC-6B': EsmcSaeCoverage('ESMC-6B', 'Synthyra/ESMplusplus_6B', 2560, 80, 60, (16384, 131072)),
}

# The commit each Biohub repository is loaded at, so a model name always means the same weights.
# A cache key names the model, not the commit, so an unpinned load let a republished checkpoint
# write different numbers under an existing key. Every repository the registry can resolve is
# pinned; `verified` records the day the commits were read from the Hub.
SAE_REVISIONS: dict[str, str] = json.loads(
    (Path(__file__).with_name('esmc_sae_revisions.json')).read_text(encoding='utf-8')
)['revisions']

SAE_K_VALUES = (16, 32, 64, 128, 256, 512)
SAE_CODEBOOK_DIMS = (8192, 16384, 32768, 65536, 131072)
# Sparsity Biohub trains at every layer other than the depth layer.
ALL_LAYER_K = 64

DEFAULT_SAE_K = 64
DEFAULT_SAE_CODEBOOK_DIM = 8192

# Active feature fraction, k / codebook_dim, below which pooled features stay sparse
# enough across the whole length range to store as coordinates. See
# EsmcSaeSelection.sparse_storage for the measurements this sits between.
SPARSE_ACTIVE_FRACTION = 0.005

_NAME_PATTERN = re.compile(
    r'^(?P<backbone>ESMC-(?:300|600|6B))-SAE'
    r'(?:-l(?P<layer>\d+)-k(?P<k>\d+)-c(?P<codebook>\d+))?$',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EsmcSaeSelection:
    """One published Biohub sparse autoencoder checkpoint."""

    backbone: str
    layer: int
    k: int
    codebook_dim: int

    @property
    def sparse_storage(self) -> bool:
        """Whether these features belong in the store's sparse `csr` layout, which keeps exact nonzeros.

        Per residue exactly `k` of `codebook_dim` features are active, whatever the
        sequence length. Pooling unions those sets across residues, so pooled density
        does grow with length, and how fast it grows tracks the active fraction
        `k / codebook_dim` rather than either number alone.

        Measured max-pooled density at codebook 16384, from short proteins to past 1200
        residues: 1.7% to 5.1% at k=16, 9.7% to 34.2% at k=64, and 19.6% to 56.5% at
        k=256. Only the last crosses the roughly 45% break-even. The threshold below sits
        between the k=64 and k=256 active fractions, and independently reproduces the
        codebook 8192 result at k=64, where measured storage split almost evenly between
        the two formats and dense was the better single choice.
        """
        return self.k / self.codebook_dim <= SPARSE_ACTIVE_FRACTION

    @property
    def repository(self) -> str:
        """The Biohub repository publishing this checkpoint.

        Biohub splits publication two ways. The depth layer has one repository per
        sparsity and width, and names the layer. Every other layer is served by an
        all-layer repository that ships one shard per layer and leaves the layer out of
        its name, so a `layer{n}` name there resolves to no repository at all.
        """
        coverage = ESMC_SAE_COVERAGE[self.backbone]
        stem = f'biohub/{coverage.biohub_name}-sae'
        if self.layer == coverage.depth_layer:
            return f'{stem}-layer{self.layer}-k{self.k}-codebook{self.codebook_dim}'

        return f'{stem}-k{self.k}-codebook{self.codebook_dim}'

    @property
    def revision(self) -> str:
        """The pinned commit of this checkpoint's repository (`SAE_REVISIONS`)."""
        return SAE_REVISIONS[self.repository]


def resolve_sae_selection(backbone: str, layer: int, k: int, codebook_dim: int) -> EsmcSaeSelection:
    """Validate one checkpoint request against what Biohub publishes."""
    coverage = ESMC_SAE_COVERAGE[backbone]
    if not 0 <= layer <= coverage.num_hidden_layers:
        raise ValueError(
            f'{backbone} has hidden states 0 to {coverage.num_hidden_layers}; '
            f'--sae_layer {layer} is outside that range.'
        )
    if k not in SAE_K_VALUES:
        raise ValueError(f'--sae_k {k} is not published; Biohub trains {list(SAE_K_VALUES)}.')
    if codebook_dim not in SAE_CODEBOOK_DIMS:
        raise ValueError(
            f'--sae_codebook_dim {codebook_dim} is not published; '
            f'Biohub trains {list(SAE_CODEBOOK_DIMS)}.'
        )

    if layer != coverage.depth_layer:
        # Away from the depth layer Biohub publishes one sparsity and a short width list.
        if k != ALL_LAYER_K or codebook_dim not in coverage.all_layer_codebook_dims:
            raise ValueError(
                f'{backbone} layer {layer} only publishes k={ALL_LAYER_K} at codebook widths '
                f'{list(coverage.all_layer_codebook_dims)}. Layer {coverage.depth_layer} '
                f'publishes every sparsity and width.'
            )

    return EsmcSaeSelection(backbone, layer, k, codebook_dim)


def is_sae_model_name(name: str) -> bool:
    return _NAME_PATTERN.match(name) is not None


def resolve_sae_model_name(
    name: str,
    sae_layer: int | None = None,
    sae_k: int | None = None,
    sae_codebook_dim: int | None = None,
) -> str:
    """Expand a bare SAE alias into its fully qualified name.

    A fully qualified name is returned unchanged, and conflicting overrides raise rather
    than being dropped, so a name always describes the checkpoint that produced it.
    """
    match = _NAME_PATTERN.match(name)
    if match is None:
        raise ValueError(f"{name!r} is not an ESMC SAE model name.")

    backbone = _canonical_backbone(match.group('backbone'))
    requested = {'sae_layer': sae_layer, 'sae_k': sae_k, 'sae_codebook_dim': sae_codebook_dim}

    if match.group('layer') is not None:
        declared = {
            'sae_layer': int(match.group('layer')),
            'sae_k': int(match.group('k')),
            'sae_codebook_dim': int(match.group('codebook')),
        }
        conflicts = {
            key: (value, declared[key])
            for key, value in requested.items()
            if value is not None and value != declared[key]
        }
        if conflicts:
            details = '; '.join(f"--{k}={v[0]} contradicts {v[1]}" for k, v in conflicts.items())
            raise ValueError(f"{name} already fixes its SAE checkpoint, but {details}.")

        selection = resolve_sae_selection(
            backbone, declared['sae_layer'], declared['sae_k'], declared['sae_codebook_dim']
        )
    else:
        selection = resolve_sae_selection(
            backbone,
            sae_layer if sae_layer is not None else ESMC_SAE_COVERAGE[backbone].depth_layer,
            sae_k if sae_k is not None else DEFAULT_SAE_K,
            sae_codebook_dim if sae_codebook_dim is not None else DEFAULT_SAE_CODEBOOK_DIM,
        )

    return f'{backbone}-SAE-l{selection.layer}-k{selection.k}-c{selection.codebook_dim}'


def parse_sae_model_name(name: str) -> EsmcSaeSelection:
    """Read a fully qualified SAE model name back into its checkpoint."""
    match = _NAME_PATTERN.match(name)
    if match is None or match.group('layer') is None:
        raise ValueError(
            f"{name!r} is not a fully qualified ESMC SAE model name. Expected the form "
            f"ESMC-300-SAE-l23-k64-c8192, which resolve_sae_model_name produces."
        )

    return resolve_sae_selection(
        _canonical_backbone(match.group('backbone')),
        int(match.group('layer')),
        int(match.group('k')),
        int(match.group('codebook')),
    )


def _canonical_backbone(prefix: str) -> str:
    for candidate in ESMC_SAE_COVERAGE:
        if candidate.lower() == prefix.lower():
            return candidate

    raise ValueError(f"Unknown ESMC SAE backbone prefix {prefix!r}.")


class EsmcSaeForEmbedding(nn.Module):
    """ESM++ backbone with a Biohub sparse autoencoder read off one hidden state.

    Returns pooled codebook features rather than residue embeddings, because the dense
    per-residue activation tensor is too large to store at these widths.
    """

    # Reduction happens in forward, over the sparse feature tensor. Embedder reads this to
    # skip building a Pooler, whose names are validated against dense (b, l, d) activations
    # and so overlap SAE_POOLING_TYPES without matching it.
    pools_internally = True

    def __init__(
        self,
        model_path: str,
        selection: EsmcSaeSelection,
        pooling_types: tuple[str, ...] = ('max',),
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        validate_sae_pooling_types(tuple(pooling_types), selection.codebook_dim)

        self.esm = load_fastplms_model(ESMplusplusModel, model_path, dtype=dtype)
        self.esm.load_sae_models(selection.repository, [selection.layer], revision=selection.revision)
        self.selection = selection
        self.pooling_types = tuple(pooling_types)
        self.sae_key = f'layer{selection.layer}'

    @property
    def sparse_storage(self) -> bool:
        """Storage format for these features, decided by the checkpoint that produced them.

        See `EsmcSaeSelection.sparse_storage` for the density measurements behind it.
        """
        return self.selection.sparse_storage

    def sae_and_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Pooled codebook features and every hidden state of the pass that produced them.

        Returning the hidden states needs every block, so this pass also computes the
        final normalized state. The codebook's own layer and that final state are therefore
        both already on hand when the features exist, and one backbone pass can fill the
        SAE cache and a dense cache at any depth, instead of running the backbone again
        over every protein for another view of the same pass.
        """
        # input_ids: (b, l); attention_mask: (b, l) or None
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)  # (b, l)

        model_output = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            compute_sae=True,
            output_hidden_states=True,
        )
        features = model_output.sae_outputs[self.sae_key]  # sparse (n, c)
        pooled = pool_sae_outputs(features, attention_mask, self.pooling_types)  # (b, p * c)
        # The input to each block, then the final normalized state: n_layers + 1 states of
        # (b, l, d). Index `layer` is the residual stream the codebook read, and index -1 is
        # `last_hidden_state`, the state an ordinary run pools by default.
        hidden_states = model_output.hidden_states  # n_layers + 1 of (b, l, d)
        return pooled, hidden_states  # (b, p * c) pooled, then n_layers + 1 states of (b, l, d)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        hidden_state_index: int = -1,
        **kwargs: Any,
    ) -> torch.Tensor:
        # input_ids: (b, l); attention_mask: (b, l) or None
        if output_attentions:
            raise ValueError("SAE embeddings do not expose attentions, so 'parti' pooling is out.")
        if hidden_state_index not in (-1, self.selection.layer):
            raise ValueError(
                f"{self.selection.repository} reads layer {self.selection.layer}; "
                f"--embedding_hidden_state_index {hidden_state_index} contradicts it."
            )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)  # (b, l)

        model_output = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            compute_sae=True,
        )
        features = model_output.sae_outputs[self.sae_key]  # sparse (n, c)
        return pool_sae_outputs(features, attention_mask, self.pooling_types)  # (b, p * c)


class EsmcSaeResidueEncoder(nn.Module):
    """A published sparse autoencoder reading residues some earlier ESM++ pass produced.

    Every sparse autoencoder trained at one layer reads the same residual stream, so a whole
    grid of sparsities and codebook widths follows from a single backbone pass:
    `EsmcSaeForEmbedding.sae_and_hidden_states` hands back that pass's hidden states, the
    layer's residues among them, and one encoder here turns those residues into one more set
    of pooled codebook features. Adding a grid cell therefore costs its codebook shards and
    one more encode, never another trip through the blocks.

    FastPLMs still owns the encoder. This loads it through the published loader and calls it
    through the same `get_sae_output` contract `ESMplusplusModel` uses internally, so the
    features match what an attached run would have produced.
    """

    def __init__(
        self,
        selection: EsmcSaeSelection,
        pooling_types: tuple[str, ...] = ('max',),
        device: torch.device | str = 'cpu',
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        validate_sae_pooling_types(tuple(pooling_types), selection.codebook_dim)

        # Pass the backbone's dtype so the codebook consumes its hidden states directly.
        self.sae: ESMplusplusSAELayer = load_esmc_sae_layers(
            selection.repository, [selection.layer], revision=selection.revision, device=device, dtype=dtype
        )[selection.layer]
        self.selection = selection
        self.pooling_types = tuple(pooling_types)

    @property
    def sparse_storage(self) -> bool:
        """Storage format for these features, decided by the checkpoint that produced them."""
        return self.selection.sparse_storage

    def forward(self, residues: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # residues: (b, l, d) hidden state this checkpoint was trained on; attention_mask: (b, l)
        token_mask = attention_mask.to(torch.bool)  # (b, l)
        # get_sae_output gathers valid tokens by boolean mask, which copies, and every step of
        # the encoder is out of place. The residues tensor therefore survives untouched for the
        # next configuration and for the dense pooling that reads the same layer.
        encoded = self.sae.get_sae_output(residues, token_mask)
        features = encoded.feature_magnitudes.detach()  # (n, c)
        return pool_sae_outputs(features, attention_mask, self.pooling_types)  # (b, p * c)


def build_esmc_sae_model(
    preset: str,
    masked_lm: bool = False,
    dtype: torch.dtype | None = None,
    model_path: str | None = None,
    pooling_types: tuple[str, ...] | None = ('max',),
    **kwargs: Any,
) -> tuple[nn.Module, ESMTokenizerWrapper]:
    if masked_lm:
        raise ValueError("SAE models produce codebook features, not masked-language-model logits.")

    selection = parse_sae_model_name(preset)

    if pooling_types is None:
        raise ValueError(
            f"{preset} cannot produce matrix embeddings. One residue of codebook "
            f"{selection.codebook_dim} costs {4 * selection.codebook_dim // 1024} KiB "
            f"dense, so per-residue SAE features do not fit the embedding cache. Drop "
            f"--matrix_embed and pool with --embedding_pooling_types max."
        )

    backbone_path = model_path or ESMC_SAE_COVERAGE[selection.backbone].model_path
    model = EsmcSaeForEmbedding(
        backbone_path, selection, pooling_types=tuple(pooling_types), dtype=dtype
    ).eval()
    return model, get_esmc_tokenizer(preset)
