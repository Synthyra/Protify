# From https://github.com/DLS5-Omics/multimolecule/blob/master/multimolecule/models/calm/modeling_calm.py
# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

from __future__ import annotations

import torch

from itertools import product
from typing import Any
from warnings import warn
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import PretrainedConfig
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward

from .base_tokenizer import BatchSequenceTokenizer
from .calm_utils import RotaryEmbedding, DnaTokenizer
from .fastplms_loading import ensure_fastplms_submodule_on_path
from .hidden_state_selection import select_hidden_state


ensure_fastplms_submodule_on_path()

from fastplms.models.dplm.modeling_dplm import FAST_DPLM_ENCODER


class CaLmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CaLmModel`][multimolecule.models.CaLmModel]. It
    is used to instantiate a CaLM model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the CaLM
    [oxpig/CaLM](https://github.com/oxpig/CaLM) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the CaLM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`CaLmModel`].
            Defaults to 131 if `codon=True` else 26.
        codon:
            Whether to use codon tokenization.
        hidden_size:
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers:
            Number of hidden layers in the Transformer encoder.
        num_attention_heads:
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size:
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act:
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout:
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout:
            The dropout ratio for the attention probabilities.
        max_position_embeddings:
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range:
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps:
            The epsilon used by the layer normalization layers.
        position_embedding_type:
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`,
            `"rotary"`.
            For positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder:
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        add_cross_attention:
            Whether each layer also attends to an encoder's hidden states. Only a decoder can. Transformers 5
            no longer gives every configuration this field, so CaLM declares it.
        use_cache:
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        emb_layer_norm_before:
            Whether to apply layer normalization after embeddings but before the main stem of the network.
        token_dropout:
            When this is enabled, masked tokens are treated as if they had been dropped out by input dropout.
        head:
            The configuration of the head.
        lm_head:
            The configuration of the masked language model head.

    Examples:
        >>> from multimolecule import CaLmConfig, CaLmModel
        >>> # Initializing a CaLM multimolecule/calm style configuration
        >>> configuration = CaLmConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/calm style configuration
        >>> model = CaLmModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "calm"

    def __init__(
        self,
        vocab_size: int | None = None,
        codon: bool = True,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        max_position_embeddings: int = 1026,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        position_embedding_type: str = "rotary",
        is_decoder: bool = False,
        add_cross_attention: bool = False,
        use_cache: bool = True,
        emb_layer_norm_before: bool = False,
        token_dropout: bool = False,
        head: None = None,
        lm_head: None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if vocab_size is None:
            vocab_size = 131 if codon else 26
        self.vocab_size = vocab_size
        self.codon = codon
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        self.use_cache = use_cache
        self.emb_layer_norm_before = emb_layer_norm_before
        self.token_dropout = token_dropout
        self.head = head
        self.lm_head = lm_head



class CaLmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CaLmConfig
    all_tied_weights_keys = {}
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CaLmLayer", "CaLmEmbeddings"]

    # Adapted from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights, and rebuild the buffers no checkpoint holds.

        `CaLmEmbeddings.position_ids` and `RotaryEmbedding.inv_freq` are registered with
        `persistent=False`. Transformers 5 loads a checkpoint into a model built on the meta device
        and passes each module whose tensors the checkpoint lacks here, with those tensors on
        uninitialized memory, so both are recomputed from the formulas their constructors use.
        """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, CaLmEmbeddings):
            positions = torch.arange(module.position_ids.shape[-1], device=module.position_ids.device)  # (l_max,)
            init.copy_(module.position_ids, positions.expand_as(module.position_ids))  # (1, l_max)
        elif isinstance(module, RotaryEmbedding):
            init.copy_(module.inv_freq, module.compute_inv_freq(module.inv_freq.device))  # (d_h / 2,)

    # Transformers 5 removed `get_extended_attention_mask`, `invert_attention_mask`, and
    # `get_head_mask` from `PreTrainedModel`. CaLM's forward pass still calls all three, so they
    # are defined here with the behavior Transformers 4 gave them, except `get_head_mask`, which
    # is FastPLMs' DPLM encoder's (below).
    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: tuple[int, int]) -> Tensor:
        """The additive form of a self-attention mask that holds 1 where a key may be attended.

        A `(b, l_k)` padding mask becomes `(b, 1, 1, l_k)`, broadcast over heads and queries. A
        decoder also hides each position from the ones before it, giving `(b, 1, l, l_k)`, where
        every query sees the `l_k - l` cached keys ahead of the current `l` positions. A
        `(b, l, l_k)` mask becomes `(b, 1, l, l_k)`.
        """
        # attention_mask: (b, l_k) or (b, l, l_k)
        if attention_mask.dim() == 3:
            extended_mask = attention_mask[:, None, :, :]  # (b, 1, l, l_k)
        elif attention_mask.dim() == 2 and self.config.is_decoder:
            seq_length = input_shape[1]  # l
            positions = torch.arange(seq_length, device=attention_mask.device)  # (l,)
            causal_mask = (positions[None, :] <= positions[:, None]).to(attention_mask.dtype)  # (l, l), query by key
            cached_length = attention_mask.shape[1] - seq_length  # l_k - l
            causal_mask = torch.cat(
                [causal_mask.new_ones(seq_length, cached_length), causal_mask], dim=-1
            )  # (l, l_k)
            extended_mask = causal_mask[None, None, :, :] * attention_mask[:, None, None, :]  # (b, 1, l, l_k)
        elif attention_mask.dim() == 2:
            extended_mask = attention_mask[:, None, None, :]  # (b, 1, 1, l_k)
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )
        return self._additive_mask(extended_mask)  # (b, 1, 1, l_k) or (b, 1, l, l_k), additive

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """The additive form of a cross-attention mask that holds 1 where an encoder position may be attended."""
        # encoder_attention_mask: (b, l_e) or (b, l, l_e)
        if encoder_attention_mask.dim() == 3:
            extended_mask = encoder_attention_mask[:, None, :, :]  # (b, 1, l, l_e)
        elif encoder_attention_mask.dim() == 2:
            extended_mask = encoder_attention_mask[:, None, None, :]  # (b, 1, 1, l_e)
        else:
            raise ValueError(
                f"An encoder attention mask has 2 or 3 dimensions, not {encoder_attention_mask.dim()}"
            )
        return self._additive_mask(extended_mask)  # (b, 1, 1, l_e) or (b, 1, l, l_e), additive

    def _additive_mask(self, mask: Tensor) -> Tensor:
        """0 where `mask` is 1 and the dtype's minimum where it is 0, to be added to attention scores."""
        # mask: (b, 1, 1, l_k) or (b, 1, l, l_k)
        mask = mask.to(dtype=self.dtype)
        return (1.0 - mask) * torch.finfo(self.dtype).min  # (b, 1, 1, l_k) or (b, 1, l, l_k), the shape of mask

    # FastPLMs' DPLM encoder carries the same Transformers 4 head mask, a `(h,)` or `(n_layers, h)`
    # mask as `(n_layers, 1, h, 1, 1)` in the model's dtype, so CaLM uses its two methods as they are.
    _convert_head_mask_to_5d = FAST_DPLM_ENCODER._convert_head_mask_to_5d
    get_head_mask = FAST_DPLM_ENCODER.get_head_mask


class CaLmModel(CaLmPreTrainedModel):
    """
    Examples:
        >>> import torch
        >>> from multimolecule import CaLmConfig, CaLmModel, RnaTokenizer
        >>> config = CaLmConfig()
        >>> model = CaLmModel(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input)
        >>> output["last_hidden_state"].shape
        torch.Size([1, 7, 768])
        >>> output["pooler_output"].shape
        torch.Size([1, 768])
    """

    def __init__(self, config: CaLmConfig, add_pooling_layer: bool = True) -> None:
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.embeddings = CaLmEmbeddings(config)
        self.encoder = CaLmEncoder(config)
        self.pooler = CaLmPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        head_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: tuple[tuple[Tensor, Tensor, Tensor, Tensor], ...] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, ...] | BaseModelOutputWithPoolingAndCrossAttentions:
        r"""
        Args:
            encoder_hidden_states:
                Shape: `(batch_size, sequence_length, hidden_size)`

                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
                the model is configured as a decoder.
            encoder_attention_mask:
                Shape: `(batch_size, sequence_length)`

                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
                in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            past_key_values:
                Tuple of length `config.n_layers` with each tuple having 4 tensors of shape
                `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)

                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up
                decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            use_cache:
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        # input_ids: (b, l), attention_mask: (b, l), position_ids: (b, l), head_mask: (h,) or (n_layers, h), inputs_embeds: (b, l, d), encoder_hidden_states: (b, l_e, d), encoder_attention_mask: (b, l_e), past_key_values: (b, h, l_p, d_h) per cached tensor, 4 per layer
        if kwargs:
            warn(
                f"Additional keyword arguments `{', '.join(kwargs)}` are detected in "
                f"`{self.__class__.__name__}.forward`, they will be ignored.\n"
                "This is provided for backward compatibility and may lead to unexpected behavior.",
                stacklevel=2,
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device  # type: ignore[union-attr]

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            if input_ids is not None and self.pad_token_id is not None:
                attention_mask = input_ids.ne(self.pad_token_id)
            else:
                attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
                warn(
                    "attention_mask is not specified, and cannot be inferred from input_ids."
                    "Assuming all tokens are not masked.",
                    stacklevel=2,
                )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]  # (b, l, d), (b, d), then the encoder outputs after the first

        return BaseModelOutputWithPoolingAndCrossAttentions(  # last_hidden_state (b, l, d), pooler_output (b, d)
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class CaLmEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config: CaLmConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        if config.emb_layer_norm_before:
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.layer_norm = None
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        self.padding_idx = config.pad_token_id
        if self.position_embedding_type == "absolute":
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
            )
        else:
            self.position_embeddings = None
        self.token_dropout = config.token_dropout
        self.mask_token_id = config.mask_token_id
        self.pad_token_id = config.pad_token_id

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        past_key_values_length: int = 0,
    ) -> Tensor:
        # input_ids: (b, l), attention_mask: (b, l), position_ids: (b, l), inputs_embeds: (b, l, d)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        if self.token_dropout:
            if input_ids is None:
                raise ValueError("Token dropout is only supported when input_ids are provided")
            embeddings = embeddings.masked_fill((input_ids == self.mask_token_id).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8  # Hardcoded as the ratio used in all CaLM model training runs
            src_lengths = attention_mask.sum(-1)  # type: ignore[union-attr]
            mask_ratio_observed = (input_ids == self.mask_token_id).sum(-1).float() / src_lengths
            embeddings = (embeddings * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]).to(embeddings)

        if self.position_embedding_type == "absolute":
            if position_ids is None:
                if input_ids is not None:
                    position_ids = create_position_ids_from_input_ids(
                        input_ids, self.padding_idx, past_key_values_length
                    )
                else:
                    position_ids = create_position_ids_from_inputs_embeds(inputs_embeds, self.padding_idx)
                # This is a bug in the original implementation
                position_ids = position_ids + 1
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        if attention_mask is not None:
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(embeddings.dtype)
        return embeddings  # (b, l, d)


class CaLmEncoder(nn.Module):
    def __init__(self, config: CaLmConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([CaLmLayer(config) for _ in range(config.num_hidden_layers)])
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: tuple[tuple[torch.FloatTensor, ...], ...] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> tuple[Tensor, ...] | BaseModelOutputWithPastAndCrossAttentions:
        # hidden_states: (b, l, d), attention_mask: (b, 1, l, l_k), head_mask: (n_layers, 1, h, 1, 1), encoder_hidden_states: (b, l_e, d), encoder_attention_mask: (b, 1, 1, l_e), past_key_values: (b, h, l_p, d_h) per cached tensor, 4 per layer
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training and use_cache:
            warn("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...", stacklevel=2)
            use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore[operator]

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = next_decoder_cache + (layer_outputs[-1],)  # type: ignore[operator]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)  # type: ignore[operator]
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)  # type: ignore[operator]

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore[operator]

        if not return_dict:
            return tuple(  # (b, l, d), then the caches and attentions that are not None
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(  # last_hidden_state (b, l, d)
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class CaLmLayer(nn.Module):
    def __init__(self, config: CaLmConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = CaLmAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = CaLmAttention(config, position_embedding_type="absolute")
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = CaLmIntermediate(config)
        self.output = CaLmOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: tuple[torch.FloatTensor, torch.FloatTensor] | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, ...]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # hidden_states: (b, l, d), attention_mask: (b, 1, l, l_k), head_mask: (1, h, 1, 1), encoder_hidden_states: (b, l_e, d), encoder_attention_mask: (b, 1, 1, l_e), past_key_value: (b, h, l_p, d_h) per cached tensor, up to 4
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated"
                    " with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs  # (b, l, d), then attentions (b, h, l, l_k) and, in a decoder, the cached keys and values

    def feed_forward_chunk(self, attention_output: Tensor) -> Tensor:
        # attention_output: (b, l, d)
        attention_output_ln = self.layer_norm(attention_output)
        intermediate_output = self.intermediate(attention_output_ln)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output  # (b, l, d)


class CaLmAttention(nn.Module):
    def __init__(self, config: CaLmConfig, position_embedding_type: str | None = None) -> None:
        super().__init__()
        self.self = CaLmSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = CaLmSelfOutput(config)
        self.pruned_heads: set = set()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: tuple[torch.FloatTensor, torch.FloatTensor] | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, ...]:
        # hidden_states: (b, l, d), attention_mask: (b, 1, l, l_k), head_mask: (1, h, 1, 1), encoder_hidden_states: (b, l_e, d), encoder_attention_mask: (b, 1, 1, l_e), past_key_value: (b, h, l_p, d_h) per cached tensor, 2
        hidden_states_ln = self.layer_norm(hidden_states)
        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs  # (b, l, d), then attentions (b, h, l, l_k) and the cache


class CaLmSelfAttention(nn.Module):
    def __init__(self, config: CaLmConfig, position_embedding_type: str | None = None) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_dropout)
        self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")
        self.rotary_embeddings = None
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        elif self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(embedding_dim=self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        # x: (b, l, d)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.transpose(1, 2)  # (b, h, l, d_h)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: tuple[torch.FloatTensor, torch.FloatTensor] | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, ...]:
        # hidden_states: (b, l, d), attention_mask: (b, 1, l, l_k), head_mask: (1, h, 1, 1), encoder_hidden_states: (b, l_e, d), encoder_attention_mask: (b, 1, 1, l_e), past_key_value: (b, h, l_p, d_h) per cached tensor, 2
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        query_layer = query_layer * self.attention_head_size**-0.5

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(Tensor, Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(Tensor, Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)  # type: ignore[misc]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # type: ignore[attr-defined]

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]  # type: ignore[attr-defined]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in CaLmModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs.to(value_layer.dtype), value_layer)

        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs  # context (b, l, d), then attention (b, h, l, l_k) and the cache


class CaLmSelfOutput(nn.Module):
    def __init__(self, config: CaLmConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        # hidden_states: (b, l, d), input_tensor: (b, l, d)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states  # (b, l, d)


class CaLmIntermediate(nn.Module):
    def __init__(self, config: CaLmConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act

    def forward(self, hidden_states: Tensor) -> Tensor:
        # hidden_states: (b, l, d)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states  # (b, l, d_i)


class CaLmOutput(nn.Module):
    def __init__(self, config: CaLmConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        # hidden_states: (b, l, d_i), input_tensor: (b, l, d)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states  # (b, l, d)


# Copied from transformers.models.bert.modeling_bert.BertPooler
class CaLmPooler(nn.Module):
    def __init__(self, config: CaLmConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # hidden_states: (b, l, d)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output  # (b, d)


def create_position_ids_from_inputs_embeds(inputs_embeds: torch.FloatTensor, padding_idx: int = 0) -> torch.LongTensor:
    # inputs_embeds: (b, l, d)
    input_shape = inputs_embeds.size()[:-1]
    sequence_length = input_shape[1]

    position_ids = torch.arange(
        padding_idx + 1, sequence_length + padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
    )
    return position_ids.unsqueeze(0).expand(input_shape)  # (b, l)


def create_position_ids_from_input_ids(
    input_ids: torch.LongTensor, padding_idx: int = 0, past_key_values_length: int = 0
) -> torch.LongTensor:
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    # input_ids: (b, l)
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (
        (torch.cumsum(mask, dim=1, dtype=mask.dtype) + past_key_values_length) * mask + past_key_values_length
    ) * mask
    return incremental_indices.long() + padding_idx  # (b, l)


presets = {
    'CaLM': 'multimolecule/calm',
}

# The commit of each preset whose weights and tokenizer reproduce the outputs its model card
# publishes (testing_suite/test_calm_checkpoint.py). A custom model path loads as given.
DEFAULT_REVISIONS = {
    'CaLM': '63fb4125a6139353defa40cbdb0c2f5e76a8cc75',
}

# Every codon a coding DNA sequence can hold, which CaLM's vocabulary must name.
DNA_CODONS = tuple(''.join(bases) for bases in product('ACGT', repeat=3))


def _normalize_calm_preset(preset: str) -> str:
    if preset in presets:
        return preset
    if 'calm' in preset.lower():
        return 'CaLM'
    raise ValueError(f"Model {preset} not supported")


def _calm_source(preset: str, model_path: str | None) -> tuple[str, str | None]:
    """The path and revision to load: the preset at its pinned revision, or a custom path as given."""
    normalized_preset = _normalize_calm_preset(preset)
    if model_path is not None:
        return model_path, None
    return presets[normalized_preset], DEFAULT_REVISIONS[normalized_preset]


def _load_calm_backbone(
    model_path: str,
    add_pooling_layer: bool = False,
    dtype: torch.dtype | None = None,
    revision: str | None = None,
) -> CaLmModel:
    model, loading_info = CaLmModel.from_pretrained(
        model_path,
        revision=revision,
        dtype=dtype,
        add_pooling_layer=add_pooling_layer,
        output_loading_info=True,
    )
    missing_keys = loading_info["missing_keys"]
    unexpected_keys = loading_info["unexpected_keys"]
    mismatched_keys = loading_info["mismatched_keys"]
    error_msgs = loading_info["error_msgs"]
    disallowed_unexpected_keys = [key for key in unexpected_keys if not key.startswith("lm_head.")]

    assert len(missing_keys) == 0, (
        f"CaLM load had missing keys: {missing_keys}"
    )
    assert len(mismatched_keys) == 0, (
        f"CaLM load had mismatched keys: {mismatched_keys}"
    )
    assert len(disallowed_unexpected_keys) == 0, (
        "CaLM load had unexpected keys outside lm_head.*: "
        f"{disallowed_unexpected_keys}"
    )
    assert len(error_msgs) == 0, (
        f"CaLM load had loader errors: {error_msgs}"
    )
    return model


class CaLMTokenizerWrapper(BatchSequenceTokenizer):
    def __init__(self, tokenizer: DnaTokenizer) -> None:
        super().__init__(tokenizer)


class CaLmForEmbedding(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype | None = None, revision: str | None = None) -> None:
        super().__init__()
        self.calm = _load_calm_backbone(model_path, add_pooling_layer=False, dtype=dtype, revision=revision)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            output_attentions: bool | None = None,
            output_hidden_states: bool | None = False,
            hidden_state_index: int = -1,
            **kwargs: Any,
    ) -> torch.Tensor:
        # input_ids: (b, l), attention_mask: (b, l)
        output_hidden_states = output_hidden_states or hidden_state_index != -1
        out = self.calm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_state = select_hidden_state(
            out.last_hidden_state,
            out.hidden_states,
            hidden_state_index,
        )
        if output_attentions:
            return hidden_state, out.attentions  # (b, l, d), tuple of (b, h, l, l)
        return hidden_state  # (b, l, d)


def get_calm_tokenizer(preset: str, model_path: str | None = None) -> CaLMTokenizerWrapper:
    """CaLM's codon tokenizer, which reads coding DNA, and U as T.

    A vocabulary that lacks a DNA codon, such as one spelled in RNA, is refused: every codon it
    lacks would read as `<unk>`.
    """
    path, revision = _calm_source(preset, model_path)
    tokenizer = DnaTokenizer.from_pretrained(path, revision=revision)
    vocabulary = tokenizer.get_vocab()
    missing_codons = [codon for codon in DNA_CODONS if codon not in vocabulary]
    if missing_codons:
        raise ValueError(
            f"The CaLM vocabulary at {path} lacks {len(missing_codons)} of the 64 DNA codons, "
            f"such as {missing_codons[:3]}, so they would read as {tokenizer.unk_token}"
        )
    return CaLMTokenizerWrapper(tokenizer)


def build_calm_model(
    preset: str,
    masked_lm: bool = False,
    dtype: torch.dtype | None = None,
    model_path: str | None = None,
    **kwargs: Any,
) -> tuple[nn.Module, CaLMTokenizerWrapper]:
    path, revision = _calm_source(preset, model_path)
    if masked_lm:
        raise ValueError(f"Model {preset} does not support masked language modeling")
    else:
        model = CaLmForEmbedding(path, dtype=dtype, revision=revision).eval()
    tokenizer = get_calm_tokenizer(preset, model_path=model_path)
    return model, tokenizer


def get_calm_for_training(
    preset: str,
    tokenwise: bool = False,
    num_labels: int | None = None,
    hybrid: bool = False,
    dtype: torch.dtype | None = None,
    model_path: str | None = None,
) -> tuple[CaLmModel, CaLMTokenizerWrapper]:
    path, revision = _calm_source(preset, model_path)
    if hybrid:
        model = _load_calm_backbone(path, add_pooling_layer=False, dtype=dtype, revision=revision).eval()
    else:
        raise ValueError(f"Model {preset} does not support training")
    tokenizer = get_calm_tokenizer(preset)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.calm
    model, tokenizer = build_calm_model('CaLM')
    print(model)
    print(tokenizer)
    tokenized = tokenizer('GCCAGTCGCTGACAGCCGCGG')
    print(model(**tokenized).shape)
    print(tokenized)
