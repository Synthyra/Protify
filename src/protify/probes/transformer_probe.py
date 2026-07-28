import torch

from dataclasses import dataclass
from typing import Any
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput


try:
    from ..pooler import Pooler
except ImportError:
    try:
        from protify.pooler import Pooler
    except ImportError:
        from pooler import Pooler

try:
    from ..model_components.mlp import intermediate_correction_fn
except ImportError:
    try:
        from protify.model_components.mlp import intermediate_correction_fn
    except ImportError:
        from model_components.mlp import intermediate_correction_fn

try:
    from ..model_components.transformer import Transformer, _UNSET, _resolve_head_size
except ImportError:
    try:
        from protify.model_components.transformer import Transformer, _UNSET, _resolve_head_size
    except ImportError:
        from model_components.transformer import Transformer, _UNSET, _resolve_head_size

from .losses import get_loss_fct


class BoMPooling(nn.Module):
    """Bag-of-Mers pooling (Hoang & Singh 2025, eq. 5, Avg-based variant).

    K-mer average representations of the transformer output H are used as queries
    in a cross-attention over H (keys/values); the resulting per-k-mer outputs
    are averaged to a single d-vector. Padding positions are excluded both from
    the attention (key mask) and from the final average (k-mers that touch any
    padded position are dropped).
    """

    def __init__(
        self,
        hidden_size: int,
        k: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, (
            f'hidden_size {hidden_size} not divisible by num_heads {num_heads}'
        )
        assert k >= 1, f'bom_k must be >= 1, got {k}'
        self.hidden_size = hidden_size  # d
        self.k = k
        self.num_heads = num_heads  # h
        self.head_dim = hidden_size // num_heads  # d_h
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = dropout

    def forward(
        self,
        emb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # emb: (b, l, d); attention_mask: (b, l) or None.
        batch_size, sequence_length, hidden_size = emb.shape
        k = min(self.k, sequence_length)
        # m is the number of candidate k-mer windows: l - k + 1.
        kernel = torch.ones(
            hidden_size,
            1,
            k,
            device=emb.device,
            dtype=emb.dtype,
        ) / k  # (d, 1, k)
        H_prime = F.conv1d(
            emb.transpose(1, 2),  # (b, d, l)
            kernel,
            groups=hidden_size,
        ).transpose(1, 2)  # (b, m, d)

        Q = self.q_proj(H_prime)  # (b, m, d)
        K = self.k_proj(emb)  # (b, l, d)
        V = self.v_proj(emb)  # (b, l, d)
        m = Q.shape[1]
        Q = Q.view(batch_size, m, self.num_heads, self.head_dim).transpose(1, 2)  # (b, h, m, d_h)
        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)  # (b, h, l, d_h)
        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)  # (b, h, l, d_h)

        attn_mask = None
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(Q.dtype)  # (b, 1, 1, l)
            attn_mask = (1.0 - key_mask) * torch.finfo(Q.dtype).min  # (b, 1, 1, l)

        out = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (b, h, m, d_h)
        out = out.transpose(1, 2).contiguous().view(
            batch_size,
            m,
            hidden_size,
        )  # (b, m, d)
        out = self.out_proj(out)  # (b, m, d)

        if attention_mask is not None and k > 1:
            kmer_valid = (
                attention_mask.unfold(dimension=1, size=k, step=1)
                .min(dim=-1)
                .values.to(out.dtype)
            )  # (b, m)
            kmer_valid = kmer_valid.unsqueeze(-1)  # (b, m, 1)
            pooled = (
                (out * kmer_valid).sum(dim=1)
                / kmer_valid.sum(dim=1).clamp(min=1.0)
            )  # (b, d)
        elif attention_mask is not None:
            mask = attention_mask.to(out.dtype).unsqueeze(-1)  # (b, l, 1); m == l when k == 1
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (b, d)
        else:
            pooled = out.mean(dim=1)  # (b, d)
        return pooled  # (b, d)


@dataclass
class ProbeSequenceClassifierOutput(SequenceClassifierOutput):
    s_max: tuple[list[torch.Tensor] | None, ...] | None = None


@dataclass
class ProbeTokenClassifierOutput(TokenClassifierOutput):
    s_max: tuple[list[torch.Tensor] | None, ...] | None = None


class TransformerProbeConfig(PretrainedConfig):
    model_type = "probe"

    def __init__(
        self,
        input_size: int = 768,
        hidden_size: int = 512,
        classifier_size: int = 4096,
        transformer_dropout: float = 0.1,
        classifier_dropout: float = 0.2,
        num_labels: int = 2,
        n_layers: int = 1,
        head_size: Any = _UNSET,
        task_type: str = "singlelabel",
        rotary: bool = True,
        pre_ln: bool = True,
        probe_pooling_types: list[str] = ["mean", "cls"],
        use_bias: bool = False,
        add_token_ids: bool = False,
        attention_backend: str = "flex",
        output_s_max: bool = False,
        max_seq_len: int = 2048,
        bom_k: int = 60,
        **kwargs: Any,
    ) -> None:
        legacy_n_heads = kwargs.pop("n_heads", None)
        head_size = _resolve_head_size(hidden_size, head_size, legacy_n_heads, default_head_size=128)
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size  # d
        self.classifier_size = classifier_size
        self.transformer_dropout = transformer_dropout
        self.classifier_dropout = classifier_dropout
        self.task_type = task_type
        self.num_labels = num_labels  # c
        self.head_size = head_size  # d_h
        self.n_heads = hidden_size // head_size  # h
        self.n_layers = n_layers
        self.rotary = rotary
        self.pre_ln = pre_ln
        self.pooling_types = probe_pooling_types
        self.use_bias = use_bias
        self.add_token_ids = add_token_ids
        self.attention_backend = attention_backend
        self.output_s_max = output_s_max
        self.max_seq_len = max_seq_len
        self.bom_k = bom_k


class TransformerForSequenceClassification(PreTrainedModel):
    config_class = TransformerProbeConfig
    all_tied_weights_keys = {}

    def __init__(self, config: TransformerProbeConfig) -> None:
        super().__init__(config)
        self.config = config
        self.task_type = config.task_type
        self.loss_fct = get_loss_fct(config.task_type)
        self.num_labels = config.num_labels  # c
        self.input_size = config.input_size
        self.add_token_ids = config.add_token_ids

        if config.pre_ln:
            self.input_layer = nn.Sequential(
                nn.LayerNorm(config.input_size),
                nn.Linear(config.input_size, config.hidden_size, bias=config.use_bias),
            )
        else:
            self.input_layer = nn.Linear(config.input_size, config.hidden_size, bias=config.use_bias)

        if self.add_token_ids:
            self.token_type_embedding = nn.Embedding(2, config.hidden_size)

        self.transformer = Transformer(
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            expansion_ratio=8 / 3,
            dropout=config.transformer_dropout,
            rotary=config.rotary,
            use_bias=config.use_bias,
            attention_backend=config.attention_backend,
            max_seq_len=config.max_seq_len,
        )

        classifier_input_size = config.hidden_size * len(config.pooling_types)
        proj_dim = intermediate_correction_fn(expansion_ratio=2, hidden_size=config.num_labels)
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_size),
            nn.Linear(classifier_input_size, config.classifier_size, bias=config.use_bias),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_size, proj_dim, bias=config.use_bias),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(proj_dim, config.num_labels, bias=config.use_bias),
        )
        self.use_bom = 'bom' in config.pooling_types
        non_bom_types = [pooling_type for pooling_type in config.pooling_types if pooling_type != 'bom']
        self.pooler = Pooler(non_bom_types) if non_bom_types else None
        if self.use_bom:
            self.bom = BoMPooling(
                hidden_size=config.hidden_size,
                k=config.bom_k,
                num_heads=config.n_heads,
                dropout=config.transformer_dropout,
            )

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool | None = None,
    ) -> ProbeSequenceClassifierOutput:
        # embeddings: (b, l, d_in); attention_mask/token_type_ids: (b, l) or None.
        # labels: (b,), (b, 1), or (b, c), depending on the task.
        embeddings = embeddings.to(next(self.input_layer.parameters()).dtype)  # (b, l, d_in)
        hidden_states = self.input_layer(embeddings)  # (b, l, d)

        if self.add_token_ids and token_type_ids is not None:
            token_type_embeddings = self.token_type_embedding(token_type_ids)  # (b, l, d)
            hidden_states = hidden_states + token_type_embeddings  # (b, l, d)

        if output_s_max is None:
            output_s_max = self.config.output_s_max

        transformer_outputs = self.transformer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
        )
        hidden_states = transformer_outputs.last_hidden_state  # (b, l, d)
        pooled_parts: list[torch.Tensor] = []
        if self.pooler is not None:
            pooled_parts.append(self.pooler(hidden_states, attention_mask))  # (b, p_non_bom * d)
        if self.use_bom:
            pooled_parts.append(self.bom(hidden_states, attention_mask))  # (b, d)
        pooled = torch.cat(pooled_parts, dim=-1)  # (b, p * d)
        logits = self.classifier(pooled)  # (b, c)
        if self.task_type == "sigmoid_regression":
            logits = logits.sigmoid()  # (b, c)

        loss = None
        if labels is not None:
            if self.task_type == "regression":
                loss = self.loss_fct(
                    logits.view(-1),  # (b * c,)
                    labels.view(-1).float(),  # (b * c,)
                )  # ()
            elif self.task_type == "sigmoid_regression":
                loss = self.loss_fct(
                    logits.view(-1),  # (b * c,)
                    labels.view(-1).float(),  # (b * c,)
                )  # ()
            elif self.task_type == "multilabel":
                loss = self.loss_fct(logits, labels.float())  # () from (b, c), (b, c)
            else:
                loss = self.loss_fct(
                    logits.view(-1, self.num_labels),  # (b, c)
                    labels.view(-1).long(),  # (b,)
                )  # ()

        # logits: (b, c); loss: () or None.
        return ProbeSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states if output_hidden_states else None,
            attentions=transformer_outputs.attentions if output_attentions else None,
            s_max=transformer_outputs.s_max if output_s_max else None,
        )


class TransformerForTokenClassification(PreTrainedModel):
    config_class = TransformerProbeConfig
    all_tied_weights_keys = {}

    def __init__(self, config: TransformerProbeConfig) -> None:
        super().__init__(config)
        self.config = config
        self.task_type = config.task_type
        self.loss_fct = get_loss_fct(config.task_type)
        self.num_labels = config.num_labels  # c
        self.input_size = config.input_size
        self.input_layer = nn.Linear(config.input_size, config.hidden_size, bias=config.use_bias)

        self.transformer = Transformer(
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            expansion_ratio=8 / 3,
            dropout=config.transformer_dropout,
            rotary=config.rotary,
            use_bias=config.use_bias,
            attention_backend=config.attention_backend,
            max_seq_len=config.max_seq_len,
        )

        proj_dim = intermediate_correction_fn(expansion_ratio=2, hidden_size=config.num_labels)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.classifier_size, bias=config.use_bias),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_size, proj_dim, bias=config.use_bias),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(proj_dim, proj_dim, bias=config.use_bias),
            nn.ReLU(),
            nn.Linear(proj_dim, config.num_labels, bias=config.use_bias),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: bool | None = None,
    ) -> ProbeTokenClassifierOutput:
        # embeddings: (b, l, d_in); attention_mask: (b, l) or None.
        # labels: (b, l), (b, l, 1), or (b, l, c), depending on the task.
        embeddings = embeddings.to(next(self.input_layer.parameters()).dtype)  # (b, l, d_in)
        hidden_states = self.input_layer(embeddings)  # (b, l, d)

        if output_s_max is None:
            output_s_max = self.config.output_s_max

        transformer_outputs = self.transformer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_s_max=output_s_max,
        )
        hidden_states = transformer_outputs.last_hidden_state  # (b, l, d)
        logits = self.classifier(hidden_states)  # (b, l, c)
        if self.task_type == "sigmoid_regression":
            logits = logits.sigmoid()  # (b, l, c)

        loss = None
        if labels is not None:
            if self.task_type == "regression":
                loss = self.loss_fct(
                    logits.view(-1),  # (b * l * c,)
                    labels.view(-1).float(),  # (b * l * c,)
                )  # ()
            elif self.task_type == "sigmoid_regression":
                loss = self.loss_fct(
                    logits.view(-1),  # (b * l * c,)
                    labels.view(-1).float(),  # (b * l * c,)
                )  # ()
            elif self.task_type == "multilabel":
                loss = self.loss_fct(logits, labels.float())  # () from (b, l, c), (b, l, c)
            else:
                loss = self.loss_fct(
                    logits.view(-1, self.num_labels),  # (b * l, c)
                    labels.view(-1).long(),  # (b * l,)
                )  # ()

        # logits: (b, l, c); loss: () or None.
        return ProbeTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states if output_hidden_states else None,
            attentions=transformer_outputs.attentions if output_attentions else None,
            s_max=transformer_outputs.s_max if output_s_max else None,
        )
