import torch
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput

from .attention import LayerNorm, MultiHeadAttention
from .attention_utils import AttentionBackend, BlockMask, build_attention_masks, resolve_attention_backend
from .mlp import swiglu_ln_ffn


_UNSET = object()


def _resolve_head_size(
    hidden_size: int,
    head_size: Any,
    n_heads_legacy: Optional[int],
    default_head_size: int,
) -> int:
    if head_size is _UNSET and n_heads_legacy is None:
        head_size = default_head_size
    elif head_size is _UNSET and n_heads_legacy is not None:
        assert hidden_size % n_heads_legacy == 0, (
            f"hidden_size {hidden_size} not divisible by legacy n_heads {n_heads_legacy}"
        )
        head_size = hidden_size // n_heads_legacy
    elif head_size is not _UNSET and n_heads_legacy is not None:
        assert hidden_size % n_heads_legacy == 0, (
            f"hidden_size {hidden_size} not divisible by legacy n_heads {n_heads_legacy}"
        )
        derived = hidden_size // n_heads_legacy
        assert derived == head_size, (
            f"Conflicting head_size={head_size} and legacy n_heads={n_heads_legacy} "
            f"(derived head_size={derived})"
        )
    assert hidden_size % head_size == 0, (
        f"hidden_size {hidden_size} not divisible by head_size {head_size}"
    )
    return head_size


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        expansion_ratio: float = 8 / 3,
        dropout: float = 0.1,
        rotary: bool = True,
        use_bias: bool = False,
        attention_backend: str = "flex",
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.attn_norm = LayerNorm(hidden_size, bias=use_bias)
        self.attn = MultiHeadAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            rotary=rotary,
            attention_backend=attention_backend,
            use_bias=use_bias,
            max_seq_len=max_seq_len,
        )
        self.ffn = swiglu_ln_ffn(hidden_size, expansion_ratio, dropout, use_bias)

    def set_attention_backend(self, attention_backend: str) -> None:
        self.attn.set_attention_backend(attention_backend)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], list[torch.Tensor] | None]:
        # hidden_states: (b, l, d); masks: (b, l) and (b, 1, 1, l) or (b, 1, l, l)
        residual = hidden_states  # (b, l, d)
        normalized_states = self.attn_norm(hidden_states)  # (b, l, d)
        attn_output, attention_weights, s_max = self.attn(
            hidden_states=normalized_states,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )  # (b, l, d), optional (b, h, l, l), optional h scalars
        hidden_states = residual + attn_output  # (b, l, d)
        ffn_output = self.ffn(hidden_states)  # (b, l, d)
        hidden_states = hidden_states + ffn_output  # (b, l, d)
        return hidden_states, attention_weights, s_max  # (b, l, d), optional (b, h, l, l), optional h scalars


@dataclass
class TransformerOutput(ModelOutput):
    # r = number of transformer layers
    loss: Optional[torch.Tensor] = None  # ()
    logits: Optional[torch.Tensor] = None  # (b, l) predictions or (b, l, v) raw logits
    last_hidden_state: Optional[torch.Tensor] = None  # (b, l, d)
    hidden_states: Tuple[torch.Tensor, ...] | None = None  # r tensors shaped (b, l, d)
    attentions: Tuple[Optional[torch.Tensor], ...] | None = None  # r optional tensors shaped (b, h, l, l)
    s_max: Tuple[List[torch.Tensor] | None, ...] | None = None  # r optional lists of h scalar tensors


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        n_layers: int,
        expansion_ratio: float = 8 / 3,
        dropout: float = 0.1,
        rotary: bool = True,
        use_bias: bool = False,
        attention_backend: str = "flex",
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    n_heads=n_heads,
                    expansion_ratio=expansion_ratio,
                    dropout=dropout,
                    rotary=rotary,
                    use_bias=use_bias,
                    attention_backend=attention_backend,
                    max_seq_len=max_seq_len,
                )
                for _ in range(n_layers)
            ]
        )
        self.attention_backend = resolve_attention_backend(attention_backend)

    def set_attention_backend(self, attention_backend: str) -> None:
        self.attention_backend = resolve_attention_backend(attention_backend)
        for layer in self.layers:
            layer.set_attention_backend(attention_backend)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool = False,
    ) -> TransformerOutput:
        # hidden_states: (b, l, d); attention_mask/attention_mask_2d: (b, l)
        # q and k are explicit query and key lengths; self-attention ordinarily uses q = k = l.
        # attention_mask_4d: (b, 1, 1, l) or (b, 1, q, k)
        batch_size, seq_len, _ = hidden_states.shape  # b, l, d
        attention_mask_2d, attention_mask_4d, flex_block_mask = build_attention_masks(
            attention_backend=self.attention_backend,
            batch_size=batch_size,
            seq_len=seq_len,
            device=hidden_states.device,
            attention_mask=attention_mask,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
        )  # optional (b, l), optional (b, 1, 1, l) or (b, 1, q, k), optional BlockMask

        hidden_state_history = () if output_hidden_states else None
        attention_history = () if output_attentions else None
        s_max_history = () if output_s_max else None

        for layer in self.layers:
            hidden_states, attention_weights, s_max = layer(
                hidden_states=hidden_states,
                attention_mask_2d=attention_mask_2d,
                attention_mask_4d=attention_mask_4d,
                flex_block_mask=flex_block_mask,
                output_attentions=output_attentions,
                output_s_max=output_s_max,
            )  # (b, l, d), optional (b, h, l, l), optional h scalars
            if output_hidden_states:
                hidden_state_history += (hidden_states,)  # next tuple entry: (b, l, d)
            if output_attentions:
                attention_history += (attention_weights,)  # next tuple entry: (b, h, l, l)
            if output_s_max:
                s_max_history += (s_max,)  # next tuple entry: h scalar tensors

        return TransformerOutput(
            last_hidden_state=hidden_states,  # (b, l, d)
            hidden_states=hidden_state_history,
            attentions=attention_history,
            s_max=s_max_history,
        )


class TransformerConfig(PretrainedConfig):
    model_type = "transformer"

    def __init__(
        self,
        hidden_size: int = 512,
        head_size=_UNSET,
        n_layers: int = 12,
        vocab_size: int = 32000,
        expansion_ratio: float = 8 / 3,
        dropout: float = 0.1,
        rotary: bool = True,
        attention_backend: str = "flex",
        output_s_max: bool = False,
        max_seq_len: int = 2048,
        attn_implementation: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        legacy_n_heads = kwargs.pop("n_heads", None)
        head_size = _resolve_head_size(hidden_size, head_size, legacy_n_heads, default_head_size=64)
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


class TransformerForMaskedLM(PreTrainedModel):
    config_class = TransformerConfig
    all_tied_weights_keys = {}

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
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_preds: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: Optional[bool] = None,
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
