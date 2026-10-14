"""The probes' transformer stack, from `foundry.layers.probe_transformer`, and Protify's masked language model on it.

`Transformer`, `TransformerBlock`, `TransformerOutput`, and the head-size resolution are
`foundry.layers.probe_transformer`'s. `TransformerConfig`, `TransformerPreTrainedModel`, and
`TransformerForMaskedLM` are Protify's.
"""

import torch

from typing import Any, ClassVar
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig

from foundry.layers.probe_attention import RotaryEmbedding
from foundry.layers.probe_transformer import (
    HEAD_SIZE_UNSET,
    Transformer,
    TransformerBlock,
    TransformerOutput,
    resolve_head_size,
)


__all__ = [
    "Transformer",
    "TransformerBlock",
    "TransformerConfig",
    "TransformerForMaskedLM",
    "TransformerOutput",
    "TransformerPreTrainedModel",
]


class TransformerConfig(PretrainedConfig):
    model_type = "transformer"

    def __init__(
        self,
        hidden_size: int = 512,
        head_size: Any = HEAD_SIZE_UNSET,
        n_layers: int = 12,
        vocab_size: int = 32000,
        expansion_ratio: float = 8 / 3,
        dropout: float = 0.1,
        rotary: bool = True,
        attention_backend: str = "flex",
        output_s_max: bool = False,
        max_seq_len: int = 2048,
        attn_implementation: str | None = None,
        **kwargs: Any,
    ) -> None:
        legacy_n_heads = kwargs.pop("n_heads", None)
        head_size = resolve_head_size(hidden_size, head_size, legacy_n_heads, default_head_size=64)
        super().__init__(**kwargs)
        self.hidden_size = hidden_size  # d
        self.head_size = head_size  # d_h
        self.n_heads = hidden_size // head_size  # h
        self.n_layers = n_layers  # r
        self.expansion_ratio = expansion_ratio
        self.dropout = dropout
        self.rotary = rotary
        self.vocab_size = vocab_size  # v
        self.output_s_max = output_s_max
        self.max_seq_len = max_seq_len
        self.attention_backend = attn_implementation if attn_implementation is not None else attention_backend


class TransformerPreTrainedModel(PreTrainedModel):
    """Loading behavior every model built on this `Transformer` shares, including the probes."""

    # Transformers 5 reads this `{tied parameter: source parameter}` mapping with `.keys()` while
    # it loads a checkpoint. These models tie no weights.
    all_tied_weights_keys: ClassVar[dict[str, str]] = {}

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Fill a tensor that the loaded checkpoint did not supply.

        Transformers 5 builds the model on the meta device, copies in the tensors the checkpoint
        holds, allocates every other parameter and buffer as uninitialized memory, and passes each
        module that owns one here. No checkpoint holds the rotary tables, so
        `RotaryEmbedding.rebuild_tables` refills them. Left as allocated, they rotated queries and
        keys by whatever the memory held. Every other module keeps the Transformers default.
        """
        if isinstance(module, RotaryEmbedding):
            module.rebuild_tables()
            return

        super()._init_weights(module)


class TransformerForMaskedLM(TransformerPreTrainedModel):
    config_class = TransformerConfig

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.transformer = Transformer(
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            expansion_ratio=config.expansion_ratio,
            dropout=config.dropout,
            rotary=config.rotary,
            attention_backend=config.attention_backend,
            max_seq_len=config.max_seq_len,
        )
        self.lm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.vocab_size),
        )
        self.ce_loss = nn.CrossEntropyLoss()
        self.vocab_size = config.vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_preds: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool | None = None,
    ) -> TransformerOutput:
        # input_ids, attention_mask, labels: (b, l)
        x = self.embeddings(input_ids)  # (b, l, d)
        if output_s_max is None:
            output_s_max = self.config.output_s_max

        transformer_outputs = self.transformer(
            hidden_states=x,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
        )  # last hidden state (b, l, d) plus optional layer histories
        logits = self.lm_head(transformer_outputs.last_hidden_state)  # (b, l, v)
        loss = None
        if labels is not None:
            flat_logits = logits.view(-1, self.vocab_size)  # (b * l, v)
            flat_labels = labels.view(-1)  # (b * l,)
            loss = self.ce_loss(flat_logits, flat_labels)  # ()
        output_logits = logits.argmax(dim=-1) if return_preds else logits  # (b, l) or (b, l, v)
        return TransformerOutput(
            loss=loss,
            logits=output_logits,
            last_hidden_state=transformer_outputs.last_hidden_state,  # (b, l, d)
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            s_max=transformer_outputs.s_max,
        )
