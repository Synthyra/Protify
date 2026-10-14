"""InterpNet's configuration and its sequence and token prediction heads.

Shape symbols beyond the standard ones: d_in is the width of the input residue embeddings,
`input_size`, and d is `hidden_size` with transformer layers and d_in without them.
"""

import torch

from dataclasses import dataclass
from typing import Any, ClassVar
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from .attention import RotaryEmbedding
from .attention_logits import AttentionLogitsSequence, AttentionLogitsToken
from .losses import get_loss_fct
from .transformer import Transformer, _UNSET, _resolve_head_size


@dataclass
class InterpNetOutput(SequenceClassifierOutput):
    """Probe logits, with the per-residue logits and s_max bounds when they are requested."""

    s_max: tuple[list[torch.Tensor] | None, ...] | None = None  # each layer: h scalar tensors
    sims: torch.Tensor | None = None  # (b, l, c)


class InterpNetConfig(PretrainedConfig):
    """Architecture and task settings of an InterpNet probe, saved with its checkpoint."""

    model_type = "interpnet"

    def __init__(
        self,
        input_size: int = 768,
        hidden_size: int = 512,
        dropout: float = 0.2,
        num_labels: int = 2,
        n_layers: int = 1,
        sim_type: str = "dot",
        head_size: int | object = _UNSET,
        task_type: str = "singlelabel",
        expansion_ratio: float = 8 / 3,
        rotary: bool = True,
        attention_backend: str = "flex",
        output_s_max: bool = False,
        use_bias: bool = False,
        max_seq_len: int = 2048,
        **kwargs: Any,
    ) -> None:
        if "id2label" in kwargs and num_labels == 2:
            num_labels = len(kwargs["id2label"])

        legacy_n_heads = kwargs.pop("n_heads", None)
        head_size = _resolve_head_size(hidden_size, head_size, legacy_n_heads, default_head_size=128)
        super().__init__(**kwargs)
        assert task_type != "regression" or num_labels == 1, "Regression task must have exactly one label."
        self.input_size = input_size  # d_in
        self.hidden_size = hidden_size  # d, when n_layers > 0
        self.dropout = dropout
        self.task_type = task_type
        self.num_labels = num_labels  # c
        self.n_layers = n_layers
        self.sim_type = sim_type
        self.expansion_ratio = expansion_ratio
        self.head_size = head_size  # d_h
        self.n_heads = hidden_size // head_size  # h
        self.rotary = rotary
        self.attention_backend = attention_backend
        self.output_s_max = output_s_max
        self.use_bias = use_bias
        self.max_seq_len = max_seq_len  # l_max, the longest sequence the rotary tables cover


class InterpNetPreTrainedModel(PreTrainedModel):
    """Loading behavior the InterpNet heads share, for checkpoints read with `from_pretrained`."""

    config_class = InterpNetConfig
    # Transformers 5 reads this `{tied parameter: source parameter}` mapping with `.keys()` while
    # it loads a checkpoint. `post_init` would derive it, but it also re-initializes every
    # parameter, so the heads do not call it and declare the mapping. InterpNet ties no weights.
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


class InterpNetForSequenceClassification(InterpNetPreTrainedModel):
    """Predict one label, class distribution, or value per sequence from its residue embeddings."""

    def __init__(self, config: InterpNetConfig) -> None:
        super().__init__(config)
        if config.n_layers > 0:
            self.input_proj = nn.Linear(config.input_size, config.hidden_size, bias=config.use_bias)
            self.transformer = Transformer(
                hidden_size=config.hidden_size,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                expansion_ratio=config.expansion_ratio,
                dropout=config.dropout,
                rotary=config.rotary,
                use_bias=config.use_bias,
                attention_backend=config.attention_backend,
                max_seq_len=config.max_seq_len,
            )

        self.get_logits = AttentionLogitsSequence(
            hidden_size=config.hidden_size if config.n_layers > 0 else config.input_size,
            num_labels=config.num_labels,
            sim_type=config.sim_type,
        )
        self.n_layers = config.n_layers
        self.num_labels = config.num_labels
        self.task_type = config.task_type
        self.loss_fct = get_loss_fct(config.task_type)
        self.config = config

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool | None = None,
        output_sims: bool = False,
    ) -> InterpNetOutput:
        """Predict sequence labels, and the loss when labels are given.

        With `output_attentions`, `attentions` holds the per-residue logits, the same tensor as
        `sims`, not the transformer's attention weights.
        """
        # embeddings: (b, l, d_in); attention_mask: (b, l) or None
        # labels: (b,), (b, c) for multilabel tasks, or None
        transformer_hidden_states = None  # each: (b, l, d), once set
        transformer_s_max = None  # each layer: h scalar tensors, once set
        if output_s_max is None:
            output_s_max = self.config.output_s_max

        if self.n_layers > 0:
            embeddings = embeddings.to(next(self.input_proj.parameters()).dtype)  # (b, l, d_in)
            x = self.input_proj(embeddings)  # (b, l, d)
            transformer_outputs = self.transformer(
                hidden_states=x,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_s_max=output_s_max,
            )
            x = transformer_outputs.last_hidden_state  # (b, l, d)
            transformer_hidden_states = transformer_outputs.hidden_states  # each: (b, l, d)
            transformer_s_max = transformer_outputs.s_max  # each layer: h scalar tensors
        else:
            embeddings = embeddings.to(next(self.get_logits.parameters()).dtype)  # (b, l, d_in)
            x = embeddings  # (b, l, d), where d is d_in

        logits, sims, x = self.get_logits(x, attention_mask)  # (b, c), (b, l, c), (b, l, d)
        hidden_state_output = transformer_hidden_states if self.n_layers > 0 else (x,)  # each: (b, l, d)
        if self.task_type == "sigmoid_regression":
            logits = logits.sigmoid()  # (b, c)

        loss = None
        if labels is not None:
            if self.task_type in ("regression", "sigmoid_regression"):
                loss = self.loss_fct(logits.flatten(), labels.view(-1).float())  # ()
            elif self.task_type == "multilabel":
                loss = self.loss_fct(logits, labels.float())  # ()
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())  # ()

        return InterpNetOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_state_output if output_hidden_states else None,
            attentions=sims if output_attentions else None,
            sims=sims if output_sims else None,
            s_max=transformer_s_max if output_s_max else None,
        )


class InterpNetForTokenClassification(InterpNetPreTrainedModel):
    """Predict one label, class distribution, or value per residue from its embedding."""

    def __init__(self, config: InterpNetConfig) -> None:
        super().__init__(config)
        if config.n_layers > 0:
            self.input_proj = nn.Linear(config.input_size, config.hidden_size, bias=config.use_bias)
            self.transformer = Transformer(
                hidden_size=config.hidden_size,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                expansion_ratio=config.expansion_ratio,
                dropout=config.dropout,
                rotary=config.rotary,
                use_bias=config.use_bias,
                attention_backend=config.attention_backend,
                max_seq_len=config.max_seq_len,
            )

        self.get_logits = AttentionLogitsToken(
            hidden_size=config.hidden_size if config.n_layers > 0 else config.input_size,
            num_labels=config.num_labels,
            sim_type=config.sim_type,
        )
        self.n_layers = config.n_layers
        self.num_labels = config.num_labels
        self.task_type = config.task_type
        self.loss_fct = get_loss_fct(config.task_type)
        self.config = config

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool | None = None,
        **kwargs: Any,
    ) -> InterpNetOutput:
        """Predict per-residue labels, and the loss when labels are given."""
        # embeddings: (b, l, d_in); attention_mask: (b, l) or None
        # labels: (b, l), (b, l, c) for multilabel tasks, or None
        del kwargs
        transformer_hidden_states = None  # each: (b, l, d), once set
        transformer_attentions = None  # each: (b, h, l, l), once set
        transformer_s_max = None  # each layer: h scalar tensors, once set
        if output_s_max is None:
            output_s_max = self.config.output_s_max

        if self.n_layers > 0:
            embeddings = embeddings.to(next(self.input_proj.parameters()).dtype)  # (b, l, d_in)
            x = self.input_proj(embeddings)  # (b, l, d)
            transformer_outputs = self.transformer(
                hidden_states=x,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_s_max=output_s_max,
            )
            x = transformer_outputs.last_hidden_state  # (b, l, d)
            transformer_hidden_states = transformer_outputs.hidden_states  # each: (b, l, d)
            transformer_attentions = transformer_outputs.attentions  # each: (b, h, l, l)
            transformer_s_max = transformer_outputs.s_max  # each layer: h scalar tensors
        else:
            embeddings = embeddings.to(next(self.get_logits.parameters()).dtype)  # (b, l, d_in)
            x = embeddings  # (b, l, d), where d is d_in

        logits = self.get_logits(x, attention_mask)  # (b, l, c)
        if self.task_type == "sigmoid_regression":
            logits = logits.sigmoid()  # (b, l, c)

        loss = None
        if labels is not None:
            if self.task_type in ("regression", "sigmoid_regression"):
                loss = self.loss_fct(logits.flatten(), labels.view(-1).float())  # ()
            elif self.task_type == "multilabel":
                loss = self.loss_fct(logits, labels.float())  # ()
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())  # ()

        return InterpNetOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_hidden_states if output_hidden_states else None,
            attentions=transformer_attentions if output_attentions else None,
            s_max=transformer_s_max if output_s_max else None,
        )
