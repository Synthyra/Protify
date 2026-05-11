import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from typing import List, Optional
try:
    from ..pooler import Pooler
except ImportError:
    try:
        from protify.pooler import Pooler
    except ImportError:
        from pooler import Pooler


class HybridProbeConfig(PretrainedConfig):
    model_type = "hybrid_probe"
    def __init__(
            self,
            tokenwise: bool = False,
            matrix_embed: bool = False,
            pooling_types: List[str] = ['mean', 'cls'],
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenwise = tokenwise
        self.matrix_embed = matrix_embed
        self.pooling_types = pooling_types


class HybridProbe(PreTrainedModel):
    config_class = HybridProbeConfig
    all_tied_weights_keys = {}
    def __init__(self, config: HybridProbeConfig, model: nn.Module, probe: nn.Module):
        super().__init__(config)
        self.config = config
        self.pool_before_probe = not config.tokenwise and not config.matrix_embed
        self.pooler = Pooler(config.pooling_types)
        self.model = model
        self.probe = probe

    def forward(
            self,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
    ):
        x = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pool_before_probe:
            x = self.pooler(x, attention_mask)
            y = self.probe(x, labels=labels)
        else:
            y = self.probe(x, attention_mask=attention_mask, labels=labels)
        return y
