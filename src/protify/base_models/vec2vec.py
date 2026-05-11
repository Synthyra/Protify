"""
HuggingFace-compatible vec2vec implementation for embedding translation.
Based on: "Harnessing the Universal Geometry of Embeddings" (arXiv:2505.12540)

Kept in sync with ProteinRepresentationEnhancement/models/vec2vec.py so that
any checkpoint pushed by the training repo loads cleanly here for downstream
supervised probing.
"""

from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModel, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from pooler import Pooler

from .base_tokenizer import BaseSequenceTokenizer
from .supported_models import all_presets_with_paths


# =============================================================================
# Utilities (mirrored from models/utils.py + models/attention.py upstream)
# =============================================================================

def _linear_layer(input_size: int, output_size: int, bias: bool = False) -> nn.Linear:
    layer = nn.Linear(input_size, output_size, bias=bias)
    nn.init.xavier_normal_(layer.weight)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


def _parameter_layer(size):
    param = nn.Parameter(torch.randn(size))
    nn.init.xavier_normal_(param)
    return param


Linear = partial(_linear_layer, bias=False)


def correction_fn_256(expansion_ratio: float, hidden_size: int) -> int:
    return int(((expansion_ratio * hidden_size) + 255) // 256 * 256)


class AttentionPooler(nn.Module):
    """
    Cross-attention pool (b, L, hidden_size) -> (b, n_tokens, hidden_size).
    Used by Vec2VecLearnedPooling to pool matrix embeddings before translation.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, n_tokens: int = 1):
        super().__init__()
        self.n_tokens = n_tokens
        self.n_heads = intermediate_size // 64
        self.d_head = 64
        self.input = Linear(hidden_size, intermediate_size)
        self.Q = _parameter_layer((1, n_tokens, intermediate_size))
        self.Wq = Linear(intermediate_size, intermediate_size)
        self.Wv = Linear(intermediate_size, intermediate_size)
        self.Wk = Linear(intermediate_size, intermediate_size)
        self.Wo = Linear(intermediate_size, hidden_size)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=self.n_heads)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, L, _ = x.size()
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].expand(b, 1, self.n_tokens, L).bool()
        x = self.input(x)
        q = self.Wq(self.Q).expand(b, -1, -1)
        v = self.Wv(x)
        k = self.Wk(x)
        q, k, v = map(self.reshaper, (q, k, v))
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=False)
        attn = rearrange(attn, "b h s d -> b s (h d)")
        return self.Wo(attn)


# =============================================================================
# Configuration
# =============================================================================

class Vec2VecConfig(PretrainedConfig):
    """Configuration for Vec2Vec model."""

    model_type = "vec2vec"

    def __init__(
        self,
        encoder_names: List[str] = None,
        encoder_paths: List[str] = None,
        encoder_dims: List[int] = None,
        d_adapter: int = 1024,
        d_hidden: int = 1024,
        d_transform: int = 1024,
        adapter_depth: int = 3,
        transform_depth: int = 4,
        disc_dim: int = 1024,
        disc_depth: int = 5,
        weight_init: str = "kaiming",
        norm_style: str = "batch",
        normalize_embeddings: bool = False,
        expansion_ratio: float = 2.0,
        learned_pooling: bool = False,
        # Loss coefficients (only read during training; kept here so configs load cleanly)
        loss_coefficient_rec: float = 1.0,
        loss_coefficient_vsp: float = 1.0,
        loss_coefficient_cc_trans: float = 10.0,
        loss_coefficient_cc_vsp: float = 10.0,
        loss_coefficient_cc_rec: float = 0.0,
        loss_coefficient_reverse_rec: float = 0.0,
        loss_coefficient_gen: float = 1.0,
        loss_coefficient_latent_gen: float = 1.0,
        loss_coefficient_similarity_gen: float = 0.0,
        loss_coefficient_disc: float = 1.0,
        loss_coefficient_r1_penalty: float = 0.0,
        loss_coefficient_learned_pooling_vsp: float = 1.0,
        noise_level: float = 0.0,
        max_grad_norm: float = 1000.0,
        rec_sim_type: str = "cosine",
        trans_sim_type: str = "cosine",
        vsp_sim_type: str = "cosine",
        sim_type: Optional[str] = None,
        sigmoid: bool = False,
        gan_style: str = "least_squares",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_names = encoder_names or ["model_a", "model_b"]
        self.encoder_paths = encoder_paths or encoder_names or ["model_a", "model_b"]
        self.encoder_dims = encoder_dims or [768, 768]
        self.d_adapter = d_adapter
        self.d_hidden = d_hidden
        self.d_transform = d_transform
        self.adapter_depth = adapter_depth
        self.transform_depth = transform_depth
        self.disc_dim = disc_dim
        self.disc_depth = disc_depth
        self.weight_init = weight_init
        self.norm_style = norm_style
        self.normalize_embeddings = normalize_embeddings
        self.expansion_ratio = expansion_ratio
        self.learned_pooling = learned_pooling
        self.loss_coefficient_rec = loss_coefficient_rec
        self.loss_coefficient_vsp = loss_coefficient_vsp
        self.loss_coefficient_cc_trans = loss_coefficient_cc_trans
        self.loss_coefficient_cc_vsp = loss_coefficient_cc_vsp
        self.loss_coefficient_cc_rec = loss_coefficient_cc_rec
        self.loss_coefficient_reverse_rec = loss_coefficient_reverse_rec
        self.loss_coefficient_gen = loss_coefficient_gen
        self.loss_coefficient_latent_gen = loss_coefficient_latent_gen
        self.loss_coefficient_similarity_gen = loss_coefficient_similarity_gen
        self.loss_coefficient_disc = loss_coefficient_disc
        self.loss_coefficient_r1_penalty = loss_coefficient_r1_penalty
        self.loss_coefficient_learned_pooling_vsp = loss_coefficient_learned_pooling_vsp
        self.noise_level = noise_level
        self.max_grad_norm = max_grad_norm
        if sim_type is not None:
            rec_sim_type = sim_type
            trans_sim_type = sim_type
            vsp_sim_type = sim_type if sim_type in ("cosine", "dot") else "cosine"
        self.rec_sim_type = rec_sim_type
        self.trans_sim_type = trans_sim_type
        self.vsp_sim_type = vsp_sim_type
        self.sim_type = sim_type if sim_type is not None else rec_sim_type
        self.sigmoid = sigmoid
        self.gan_style = gan_style

    def get_encoder_dims_dict(self) -> Dict[str, int]:
        return dict(zip(self.encoder_names, self.encoder_dims))

    def get_encoder_paths_dict(self) -> Dict[str, str]:
        return dict(zip(self.encoder_names, self.encoder_paths))

    def get_path_for_name(self, name: str) -> str:
        paths_dict = self.get_encoder_paths_dict()
        return paths_dict.get(name, name)


# =============================================================================
# Model Outputs
# =============================================================================

@dataclass
class Vec2VecOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    reconstructions: Optional[Dict[str, torch.Tensor]] = None
    translations: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    latents: Optional[Dict[str, torch.Tensor]] = None
    metrics: Optional[Dict[str, float]] = None


# =============================================================================
# Model Components
# =============================================================================

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def _initialize_weights(self, weight_init: str):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if weight_init == "kaiming":
                    nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in", nonlinearity="relu")
                elif weight_init == "xavier":
                    nn.init.xavier_normal_(module.weight)
                elif weight_init == "orthogonal":
                    nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.normal_(module.weight, mean=1.0, std=0.02)
                nn.init.normal_(module.bias, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def _add_residual(self, input_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if input_x.shape[1] < x.shape[1]:
            padding = torch.zeros(x.shape[0], x.shape[1] - input_x.shape[1], device=x.device)
            input_x = torch.cat([input_x, padding], dim=1)
        elif input_x.shape[1] > x.shape[1]:
            input_x = input_x[:, :x.shape[1]]
        return x + input_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class MLPWithResidual(BaseModule):
    def __init__(
        self,
        depth: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        norm_style: str = "batch",
        weight_init: str = "kaiming",
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        norm_layer = nn.BatchNorm1d if norm_style == "batch" else nn.LayerNorm

        for layer_idx in range(depth):
            if layer_idx == 0:
                h_dim = out_dim if depth == 1 else hidden_dim
                self.layers.append(nn.Sequential(nn.Linear(in_dim, h_dim), nn.SiLU()))
            elif layer_idx < depth - 1:
                self.layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    norm_layer(hidden_dim),
                    nn.Dropout(p=0.1),
                ))
            else:
                self.layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Dropout(p=0.1),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, out_dim),
                ))
        self._initialize_weights(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            input_x = x
            x = layer(x)
            x = self._add_residual(input_x, x)
        return x


class Discriminator(BaseModule):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 1024,
        depth: int = 5,
        weight_init: str = "kaiming",
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        if depth >= 2:
            layers = [nn.Linear(latent_dim, hidden_dim), nn.Dropout(0.0)]
            for _ in range(depth - 2):
                layers.extend([
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.0),
                ])
            layers.extend([nn.SiLU(), nn.Linear(hidden_dim, 1)])
            self.layers.append(nn.Sequential(*layers))
        else:
            self.layers.append(nn.Linear(latent_dim, 1))

        self._initialize_weights(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# =============================================================================
# Main Model
# =============================================================================

class Vec2VecModel(PreTrainedModel):
    """
    Vec2Vec model for embedding translation between different spaces.

    Architecture:
        Input -> In Adapter -> Transform -> Out Adapter -> Output
    """

    config_class = Vec2VecConfig

    def __init__(self, config: Vec2VecConfig):
        super().__init__(config)
        self.config = config
        encoder_dims = config.get_encoder_dims_dict()

        self.transform = MLPWithResidual(
            depth=config.transform_depth,
            in_dim=config.d_adapter,
            hidden_dim=config.d_transform,
            out_dim=config.d_adapter,
            norm_style=config.norm_style,
            weight_init=config.weight_init,
        )

        self.in_adapters = nn.ModuleDict()
        self.out_adapters = nn.ModuleDict()

        for name, dim in encoder_dims.items():
            self.in_adapters[name] = MLPWithResidual(
                config.adapter_depth, dim, config.d_hidden, config.d_adapter,
                config.norm_style, config.weight_init,
            )
            self.out_adapters[name] = MLPWithResidual(
                config.adapter_depth, config.d_adapter, config.d_hidden, dim,
                config.norm_style, config.weight_init,
            )

        self.discriminators = nn.ModuleDict()
        for name, dim in encoder_dims.items():
            self.discriminators[name] = Discriminator(
                dim, config.disc_dim, config.disc_depth, config.weight_init
            )
        self.discriminators["latent"] = Discriminator(
            config.d_adapter, config.disc_dim, config.disc_depth, config.weight_init
        )

        self.post_init()

    def add_encoder(self, name: str, dim: int, overwrite: bool = False):
        if name in self.in_adapters and not overwrite:
            print(f"Encoder {name} already exists, skipping...")
            return

        self.in_adapters[name] = MLPWithResidual(
            self.config.adapter_depth, dim, self.config.d_hidden, self.config.d_adapter,
            self.config.norm_style, self.config.weight_init,
        )
        self.out_adapters[name] = MLPWithResidual(
            self.config.adapter_depth, self.config.d_adapter, self.config.d_hidden, dim,
            self.config.norm_style, self.config.weight_init,
        )
        self.discriminators[name] = Discriminator(
            dim, self.config.disc_dim, self.config.disc_depth, self.config.weight_init
        )

        if name not in self.config.encoder_names:
            self.config.encoder_names.append(name)
            self.config.encoder_dims.append(dim)

    def _get_latent(
        self,
        emb: torch.Tensor,
        encoder_name: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self.in_adapters[encoder_name](emb)
        return self.transform(z)

    def _decode(
        self,
        latent: torch.Tensor,
        encoder_name: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.out_adapters[encoder_name](latent)
        if self.config.normalize_embeddings:
            out = F.normalize(out, p=2, dim=1)
        return out

    def translate(
        self,
        embeddings: torch.Tensor,
        src: str,
        tgt: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent = self._get_latent(embeddings, src, attention_mask)
        return self._decode(latent, tgt, attention_mask)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        attention_masks: Optional[Dict[str, torch.Tensor]] = None,
        noise_level: float = None,
        return_latents: bool = False,
    ) -> Vec2VecOutput:
        noise_level = noise_level if noise_level is not None else self.config.noise_level

        reconstructions: Dict[str, torch.Tensor] = {}
        translations: Dict[str, Dict[str, torch.Tensor]] = {}
        latents: Dict[str, torch.Tensor] = {}

        for src_name, emb in inputs.items():
            if self.training and noise_level > 0.0:
                emb = emb + torch.randn_like(emb) * noise_level
                emb = F.normalize(emb, p=2, dim=1)

            latent = self._get_latent(emb, src_name)
            if return_latents:
                latents[src_name] = latent

            for tgt_name in inputs.keys():
                decoded = self._decode(latent, tgt_name)
                if tgt_name == src_name:
                    reconstructions[src_name] = decoded
                else:
                    if tgt_name not in translations:
                        translations[tgt_name] = {}
                    translations[tgt_name][src_name] = decoded

        return Vec2VecOutput(
            reconstructions=reconstructions,
            translations=translations,
            latents=latents if return_latents else None,
        )


class Vec2VecLearnedPooling(Vec2VecModel):
    """
    Vec2Vec variant that pools matrix embeddings (b, L, d) internally with a
    learned AttentionPooler before the standard translator.
    """

    config_class = Vec2VecConfig

    def __init__(self, config: Vec2VecConfig):
        super().__init__(config)
        encoder_dims = config.get_encoder_dims_dict()

        self.normalize_embeddings = config.normalize_embeddings
        self.poolers = nn.ModuleDict()
        for name, dim in encoder_dims.items():
            intermediate_size = correction_fn_256(config.expansion_ratio, dim)
            self.poolers[name] = AttentionPooler(
                hidden_size=dim,
                intermediate_size=intermediate_size,
                n_tokens=1,
            )

        self.post_init()

    def add_encoder(self, name: str, dim: int, overwrite: bool = False):
        super().add_encoder(name, dim, overwrite)
        if name not in self.poolers or overwrite:
            intermediate_size = correction_fn_256(self.config.expansion_ratio, dim)
            self.poolers[name] = AttentionPooler(
                hidden_size=dim,
                intermediate_size=intermediate_size,
                n_tokens=1,
            )

    def pool(
        self,
        embeddings: torch.Tensor,
        encoder_name: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooled = self.poolers[encoder_name](embeddings, attention_mask).squeeze(1)
        if self.normalize_embeddings:
            pooled = F.normalize(pooled, p=2, dim=1)
        return pooled

    def _get_latent(
        self,
        emb: torch.Tensor,
        encoder_name: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if emb.dim() == 3:
            emb = self.pool(emb, encoder_name, attention_mask)
        z = self.in_adapters[encoder_name](emb)
        return self.transform(z)

    def translate(
        self,
        embeddings: torch.Tensor,
        src: str,
        tgt: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent = self._get_latent(embeddings, src, attention_mask)
        return self._decode(latent, tgt)


# =============================================================================
# Protify integration
# =============================================================================

presets = {
    'vec2vec-ESM2-8-ESM2-35': 'lhallee/ESM2-8-ESM2-35-sequence-sequence',
    'vec2vec-ESM2-8-ESM2-150': 'lhallee/ESM2-8-ESM2-150-sequence-sequence',
    'vec2vec-ESM2-8-ESM2-650': 'lhallee/ESM2-8-ESM2-650-sequence-sequence',
    'vec2vec-ESM2-8-ESM2-3B': 'lhallee/ESM2-8-ESM2-3B-sequence-sequence',
    'vec2vec-ESM2-35-ESM2-150': 'lhallee/ESM2-35-ESM2-150-sequence-sequence',
    'vec2vec-ESM2-35-ESM2-650': 'lhallee/ESM2-35-ESM2-650-sequence-sequence',
    'vec2vec-ESM2-35-ESM2-3B': 'lhallee/ESM2-35-ESM2-3B-sequence-sequence',
    'vec2vec-ESM2-150-ESM2-650': 'lhallee/ESM2-150-ESM2-650-sequence-sequence',
    'vec2vec-ESM2-150-ESM2-3B': 'lhallee/ESM2-150-ESM2-3B-sequence-sequence',
    'vec2vec-ESM2-650-ESM2-3B': 'lhallee/ESM2-650-ESM2-3B-sequence-sequence',
    'vec2vec-ESM2-650-ModernBERT-base-contrastive': 'lhallee/ESM2-650-ModernBERT-base-sequence-sequence-contrastive',
    'vec2vec-ESM2-650-ModernBERT-large-contrastive': 'lhallee/ESM2-650-ModernBERT-large-sequence-sequence-contrastive',
}


class Vec2VecTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)

    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault('return_tensors', 'pt')
        kwargs.setdefault('padding', 'longest')
        kwargs.setdefault('add_special_tokens', True)
        return self.tokenizer(sequences, **kwargs)


class Vec2VecForEmbedding(nn.Module):
    """
    Wraps a frozen base PLM + a Vec2Vec translator so Protify sees a single
    "encoder" whose output is the *translated* embedding.

    Direction convention (matches ProteinRepresentationEnhancement training):
        - model_name_a := encoder_names[0]  (source side at training time)
        - model_name_b := encoder_names[1]  (target side)
        - base_model loads model_name_a and its hidden states are pooled
          (mean+var, unless learned_pooling is set) then translated A -> B.

    For the canonical small-to-big ablation, pairs are trained with the SMALLER
    model as encoder_names[0], so Protify embeds with the cheap model and
    returns an approximation of the larger model's pooled embedding.
    """

    # forward() returns a fully pooled + translated (B, D) vector; callers
    # (e.g. Protify's embedder) must not apply their own pooler on top.
    already_pooled: bool = True

    def __init__(
        self,
        config: Vec2VecConfig,
        base_model: AutoModel,
        vec2vec_model: Vec2VecModel,
        model_name_a: str,
        model_name_b: str,
    ):
        super().__init__()
        self.base_model = base_model
        self.vec2vec_model = vec2vec_model
        self.config = config
        self.learned_pooling = bool(getattr(config, 'learned_pooling', False))
        self.pooler = None if self.learned_pooling else Pooler(['mean', 'var'])
        self.model_name_a = model_name_a
        self.model_name_b = model_name_b
        self.normalize = config.normalize_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        base_state = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state
        # Translator weights are loaded fp32; under autocast, base_state may be
        # bf16 which collides with the Linear weight dtype. Cast base output to
        # the translator's parameter dtype so autocast does not hand a bf16
        # input to an fp32 Linear mid-way through the wrapper.
        translator_dtype = next(self.vec2vec_model.parameters()).dtype
        if self.learned_pooling:
            # Translator has its own AttentionPooler; pass raw hidden states.
            translated = self.vec2vec_model.translate(
                base_state.to(translator_dtype),
                src=self.model_name_a,
                tgt=self.model_name_b,
                attention_mask=attention_mask,
            )
        else:
            base_vec = self.pooler(base_state, attention_mask=attention_mask)
            if self.normalize:
                base_vec = F.normalize(base_vec, p=2, dim=1)
            translated = self.vec2vec_model.translate(
                base_vec.to(translator_dtype),
                src=self.model_name_a,
                tgt=self.model_name_b,
            )
        return translated


def get_vec2vec_tokenizer(preset: str, model_path: str = None):
    path = model_path or all_presets_with_paths[preset]
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception:
        # vec2vec repos often don't ship a tokenizer; AutoModel can't load a
        # `vec2vec` model_type via AutoFactory either. Resolve the source
        # encoder from the Vec2VecConfig and load its tokenizer directly.
        config = Vec2VecConfig.from_pretrained(path)
        encoder_a_preset = config.encoder_names[0]
        encoder_a_path = all_presets_with_paths[encoder_a_preset]
        tokenizer = AutoTokenizer.from_pretrained(encoder_a_path, trust_remote_code=True)
    return Vec2VecTokenizerWrapper(tokenizer)


def build_vec2vec_model(
    preset: str,
    masked_lm: bool = False,
    dtype: torch.dtype = None,
    model_path: str = None,
    **kwargs,
):
    if masked_lm:
        raise ValueError("Masked LM is not supported for Vec2VecForEmbedding")

    model_path = model_path or presets[preset]
    config = Vec2VecConfig.from_pretrained(model_path)

    # Preserve training-time direction: encoder_names[0] is the source side.
    encoder_names = config.encoder_names
    assert len(encoder_names) >= 2, f"Vec2Vec checkpoint needs >=2 encoders, got {encoder_names}"
    model_name_a = encoder_names[0]
    model_name_b = encoder_names[1]

    base_model = AutoModel.from_pretrained(
        all_presets_with_paths[model_name_a], dtype=dtype, trust_remote_code=True,
    )
    base_tokenizer = base_model.tokenizer

    translator_cls = Vec2VecLearnedPooling if getattr(config, 'learned_pooling', False) else Vec2VecModel
    vec2vec_model = translator_cls.from_pretrained(model_path, config=config)

    model = Vec2VecForEmbedding(config, base_model, vec2vec_model, model_name_a, model_name_b)
    tokenizer = Vec2VecTokenizerWrapper(base_tokenizer)
    return model, tokenizer


def get_vec2vec_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False):
    raise ValueError("Vec2VecForTraining is not supported yet")


if __name__ == '__main__':
    # py -m src.protify.base_models.vec2vec
    model, tokenizer = build_vec2vec_model('vec2vec-ESM2-8-ESM2-35')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))
