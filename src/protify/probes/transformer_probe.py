import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
from typing import List, Optional
from pooler import Pooler
from model_components.mlp import intermediate_correction_fn
from model_components.transformer import Transformer
from .losses import get_loss_fct


class TransformerProbeConfig(PretrainedConfig):
    model_type = "probe"
    def __init__(
            self,
            input_dim: int = 768,
            hidden_dim: int = 512,
            classifier_dim: int = 4096,
            transformer_dropout: float = 0.1,
            classifier_dropout: float = 0.2,
            num_labels: int = 2,
            n_layers: int = 1,
            sim_type:str = 'cosine',
            n_heads: int = 4,
            task_type: str = 'singlelabel',
            rotary: bool = True,
            pre_ln: bool = True,
            probe_pooling_types: List[str] = ['mean', 'cls'],
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.classifier_dim = classifier_dim
        self.transformer_dropout = transformer_dropout
        self.classifier_dropout = classifier_dropout
        self.task_type = task_type
        self.num_labels = num_labels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.rotary = rotary
        self.pre_ln = pre_ln
        self.pooling_types = probe_pooling_types
        self.sim_type = sim_type


class TransformerForSequenceClassification(PreTrainedModel):
    config_class = TransformerProbeConfig
    def __init__(self, config: TransformerProbeConfig):
        super().__init__(config)
        self.config = config
        self.task_type = config.task_type
        self.loss_fct = get_loss_fct(config.task_type)
        self.num_labels = config.num_labels
        self.input_dim = config.input_dim

        if config.pre_ln:
            self.input_layer = nn.Sequential(
                nn.LayerNorm(config.input_dim),
                nn.Linear(config.input_dim, config.hidden_dim)
            )
        else:
            self.input_layer = nn.Linear(config.input_dim, config.hidden_dim)

        self.transformer = Transformer(
            hidden_size=config.hidden_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            expansion_ratio=8 / 3,
            dropout=config.transformer_dropout,
            rotary=config.rotary,
        )

        classifier_input_dim = config.hidden_dim * len(config.pooling_types)
        proj_dim = intermediate_correction_fn(expansion_ratio=2, hidden_size=config.num_labels)
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, config.classifier_dim),
            nn.ReLU(),
            nn.Dropout(config.transformer_dropout),
            nn.Linear(config.classifier_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(proj_dim, config.num_labels)
        )
        self.pooler = Pooler(config.pooling_types)

    def forward(
            self,
            embeddings,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
    ) -> SequenceClassifierOutput:
        x = self.input_layer(embeddings)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            if attention_mask is not None and attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1)
        x = self.transformer(x, attention_mask)
        x = self.pooler(x, attention_mask)
        logits = self.classifier(x)
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
            attentions=None
        )
    

class TransformerForTokenClassification(PreTrainedModel):
    config_class = TransformerProbeConfig
    def __init__(self, config: TransformerProbeConfig):
        super().__init__(config)
        self.config = config
        self.task_type = config.task_type
        self.loss_fct = get_loss_fct(config.task_type)
        self.num_labels = config.num_labels
        self.input_dim = config.input_dim
        self.input_layer = nn.Linear(config.input_dim, config.hidden_dim)
        self.transformer = Transformer(
            hidden_size=config.hidden_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            expansion_ratio=8 / 3,
            dropout=config.transformer_dropout,
            rotary=config.rotary,
        )

        proj_dim = intermediate_correction_fn(expansion_ratio=2, hidden_size=config.num_labels)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.classifier_dim),
            nn.ReLU(),
            nn.Dropout(config.transformer_dropout),
            nn.Linear(config.classifier_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, config.num_labels)
        )

    def forward(
            self,
            embeddings,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
    ) -> TokenClassifierOutput:
        x = self.input_layer(embeddings)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            if attention_mask is not None and attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1)
        x = self.transformer(x, attention_mask)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )
