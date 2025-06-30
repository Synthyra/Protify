import torch
from torch import nn
from typing import Optional
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from model_components.attention import AttentionLogitsSequence, AttentionLogitsToken, Linear
from model_components.transformer import PTransformer, Transformer
from .losses import get_loss_fct


class RetrievalNetConfig(PretrainedConfig):
    model_type = "retrievalnet"
    def __init__(
            self,
            input_dim: int = 768,
            hidden_dim: int = 512,
            dropout: float = 0.2,
            num_labels: int = 2,
            n_layers: int = 1,
            sim_type: str = 'dot',
            n_heads: int = 4,
            task_type: str = 'singlelabel',
            expansion_ratio: float = 8 / 3,
            **kwargs,
    ):
        super().__init__(**kwargs)
        assert task_type != 'regression' or num_labels == 1, "Regression task must have exactly one label"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.task_type = task_type
        self.num_labels = num_labels
        self.n_layers = n_layers
        self.sim_type = sim_type
        self.expansion_ratio = expansion_ratio
        self.n_heads = n_heads


class RetrievalNetForSequenceClassification(PreTrainedModel):
    config_class = RetrievalNetConfig
    def __init__(self, config: RetrievalNetConfig):
        super().__init__(config)
        #self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        #self.transformer = PTransformer(
        #    hidden_size=config.hidden_dim,
        #    n_heads=config.n_heads,
        #    n_layers=config.n_layers,
        #    expansion_ratio=config.expansion_ratio,
        #    dropout=config.dropout,
        #    rotary=True,
        #)

        # If n_layers == 0, only learn how to distribute labels over the raw embeddings
        if config.n_layers == 0:
            self.label_dist_head = nn.Linear(config.input_dim, config.num_labels)
        else:
            self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
            self.transformer = Transformer(
                hidden_size=config.hidden_dim,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                expansion_ratio=config.expansion_ratio,
                dropout=config.dropout,
                rotary=True,
            )
            self.get_logits = AttentionLogitsSequence(
                hidden_size=config.hidden_dim,
                num_labels=config.num_labels,
                sim_type=config.sim_type,
            )

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
        # Zero‐layer
        if self.config.n_layers == 0:
            if embeddings.dim() == 3:
                 pooled = embeddings.mean(dim=1)
            elif embeddings.dim() == 2:
                 pooled = embeddings
            else:
                 raise ValueError(f"Expected embeddings to be 2D or 3D, got {embeddings.shape}")

            logits = self.label_dist_head(pooled)
            loss = None
            if labels is not None:
                if self.task_type == 'regression':
                    loss = self.loss_fct(logits.flatten(), labels.view(-1).float())
                elif self.task_type == 'multilabel':
                    loss = self.loss_fct(logits, labels.float())
                else:
                    loss = self.loss_fct(logits.view(-1, self.num_labels),
                                         labels.view(-1).long())
            return SequenceClassifierOutput(loss=loss, logits=logits)

        # Default multi‐layer path
        x = self.input_proj(embeddings) # (bs, seq_len, hidden_dim)
        x = self.transformer(x, attention_mask) # (bs, seq_len, hidden_dim)
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
        if config.n_layers == 0:
             self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        else:
            self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
            self.transformer = PTransformer(
                hidden_size=config.hidden_dim,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                expansion_ratio=config.expansion_ratio,
                dropout=config.dropout,
                rotary=config.rotary,
            )
        self.get_logits = AttentionLogitsToken(hidden_size=config.hidden_dim, num_labels=config.num_labels, sim_type=config.sim_type)

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
        x = self.input_proj(embeddings) # (bs, seq_len, hidden_dim)
        x = self.transformer(x, attention_mask) # (bs, seq_len, hidden_dim)
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