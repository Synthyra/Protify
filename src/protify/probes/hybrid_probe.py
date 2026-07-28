import torch

from typing import Any
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig

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
        pooling_types: list[str] = ['mean', 'cls'],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tokenwise = tokenwise
        self.matrix_embed = matrix_embed
        self.pooling_types = pooling_types


class HybridProbe(PreTrainedModel):
    config_class = HybridProbeConfig
    all_tied_weights_keys = {}

    def __init__(self, config: HybridProbeConfig, model: nn.Module, probe: nn.Module) -> None:
        super().__init__(config)
        self.config = config
        self.pool_before_probe = not config.tokenwise and not config.matrix_embed
        self.pooler = Pooler(config.pooling_types)
        self.model = model
        self.probe = probe

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> Any:
        # input_ids: (b, l); attention_mask: (b, l) or None. Label shape is task-dependent.
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state  # (b, l, d)

        if self.pool_before_probe:
            # p is the configured number of pooling operations.
            probe_embeddings = self.pooler(hidden_states, attention_mask)  # (b, p * d)
            probe_output = self.probe(probe_embeddings, labels=labels)
        else:
            probe_output = self.probe(
                hidden_states,
                attention_mask=attention_mask,
                labels=labels,
            )

        # probe_output.logits: (b, c) or (b, l, c), depending on the probe contract.
        return probe_output
