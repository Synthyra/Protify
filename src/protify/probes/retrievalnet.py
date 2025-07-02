import torch
from torch import nn
from typing import Optional
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from model_components.attention import AttentionLogitsSequence, AttentionLogitsToken, Linear
from model_components.transformer import TokenFormer, Transformer
from .losses import get_loss_fct


class RetrievalNetConfig(PretrainedConfig):
    model_type = "retrievalnet"
    def __init__(
            self,
            input_dim: int = 768,
            hidden_size: int = 512,
            dropout: float = 0.2,
            num_labels: int = 2,
            n_layers: int = 1,
            sim_type: str = 'dot',
            token_attention: bool = False,
            n_heads: int = 4,
            task_type: str = 'singlelabel',
            expansion_ratio: float = 8 / 3,
            **kwargs,
    ):
        super().__init__(**kwargs)
        assert task_type != 'regression' or num_labels == 1, "Regression task must have exactly one label"
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.task_type = task_type
        self.num_labels = num_labels
        self.n_layers = n_layers
        self.sim_type = sim_type
        self.token_attention = token_attention
        self.expansion_ratio = expansion_ratio
        self.n_heads = n_heads


class RetrievalNetForSequenceClassification(PreTrainedModel):
    config_class = RetrievalNetConfig
    def __init__(self, config: RetrievalNetConfig):
        super().__init__(config)
        # If n_layers == 0, only learn how to distribute labels over the raw embeddings
        if config.n_layers > 0:
            self.input_proj = nn.Linear(config.input_dim, config.hidden_size)
        
            transformer_class = TokenFormer if config.token_attention else Transformer
            self.transformer = transformer_class(
                hidden_size=config.hidden_size,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                expansion_ratio=config.expansion_ratio,
                dropout=config.dropout,
                rotary=True,
            )
            
        self.get_logits = AttentionLogitsSequence(
            hidden_size=config.hidden_size if config.n_layers > 0 else config.input_dim,
            num_labels=config.num_labels,
            sim_type=config.sim_type,
        )

        self.n_layers = config.n_layers
        self.num_labels = config.num_labels
        self.task_type = config.task_type
        self.loss_fct = get_loss_fct(config.task_type)

    def forward(
            self,
            embeddings,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
    ) -> SequenceClassifierOutput:
        if self.n_layers > 0:
            x = self.input_proj(embeddings) # (bs, seq_len, hidden_size)
            x = self.transformer(x, attention_mask) # (bs, seq_len, hidden_size)
        else:
            x = embeddings

        logits, sims, x = self.get_logits(x, attention_mask) 
        loss = None
        if labels is not None:
            if self.task_type == 'regression':
                loss = self.loss_fct(logits.flatten(), labels.view(-1).float())
            elif self.task_type == 'multilabel':
                loss = self.loss_fct(logits, labels.float())
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())
    
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=x if output_hidden_states else None,
            attentions=sims if output_attentions else None
        )


class RetrievalNetForTokenClassification(PreTrainedModel):
    config_class = RetrievalNetConfig
    def __init__(self, config: RetrievalNetConfig):
        super().__init__(config)
        if config.n_layers > 0:
            self.input_proj = nn.Linear(config.input_dim, config.hidden_size)
            self.transformer = TokenFormer(
                hidden_size=config.hidden_size,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                expansion_ratio=config.expansion_ratio,
                dropout=config.dropout,
                rotary=config.rotary,
            )

        self.get_logits = AttentionLogitsToken(
            hidden_size=config.hidden_size if config.n_layers > 0 else config.input_dim,
            num_labels=config.num_labels,
            sim_type=config.sim_type,
        )

        self.n_layers = config.n_layers
        self.num_labels = config.num_labels
        self.task_type = config.task_type
        self.loss_fct = get_loss_fct(config.task_type)

    def forward(
            self,
            embeddings,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> SequenceClassifierOutput:
        if self.n_layers > 0:
            x = self.input_proj(embeddings) # (bs, seq_len, hidden_size)
            x = self.transformer(x, attention_mask) # (bs, seq_len, hidden_size)
        else:
            x = embeddings

        logits = self.get_logits(x)

        loss = None
        if labels is not None:
            if self.task_type == 'regression':
                loss = self.loss_fct(logits.flatten(), labels.view(-1).float())
            elif self.task_type == 'multilabel':
                loss = self.loss_fct(logits, labels.float())
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )