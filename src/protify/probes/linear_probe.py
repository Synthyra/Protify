import torch
from torch import nn
from typing import Optional
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from model_components.mlp import intermediate_correction_fn
from .losses import get_loss_fct


class LinearProbeConfig(PretrainedConfig):
    model_type = "linear_probe"
    def __init__(
            self,
            input_dim: int = 768,
            hidden_size: int = 8192,
            dropout: float = 0.2,
            num_labels: int = 2,
            n_layers: int = 1,
            task_type: str = 'singlelabel',
            pre_ln: bool = True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.task_type = task_type
        self.num_labels = num_labels
        self.n_layers = n_layers
        self.pre_ln = pre_ln


class LinearProbe(PreTrainedModel):
    config_class = LinearProbeConfig
    def __init__(self, config: LinearProbeConfig):
        super().__init__(config)
        self.config = config
        self.task_type = config.task_type
        self.loss_fct = get_loss_fct(config.task_type)
        self.num_labels = config.num_labels
        layers = []
        if config.pre_ln:
            layers.append(nn.LayerNorm(config.input_dim))
        layers.append(nn.Linear(config.input_dim, config.hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.dropout))
        
        for _ in range(config.n_layers):
            layers.append(nn.Linear(config.hidden_size, config.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))

        proj_dim = intermediate_correction_fn(2, config.num_labels) # finds nearest multiple of 256 of 2 * num_labels
        layers.append(nn.LayerNorm(config.hidden_size))
        layers.append(nn.Linear(config.hidden_size, proj_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(proj_dim, config.num_labels))
        self.layers = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None) -> SequenceClassifierOutput:
        logits = self.layers(embeddings)
        loss = None
        if labels is not None:
            if self.task_type == 'regression':
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
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
