import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any
from transformers import EsmConfig, EsmTokenizer
from transformers.utils import ModelOutput

try:
    from ..model_components.transformer import TransformerConfig, TransformerForMaskedLM
except ImportError:
    from protify.model_components.transformer import TransformerConfig, TransformerForMaskedLM


presets = {
    'Random': 'random',
    'Random-Transformer': 'facebook/esm2_t12_35M_UR50D', # default is 35M version
    'Random-ESM2-8': 'facebook/esm2_t6_8M_UR50D',
    'Random-ESM2-35': 'facebook/esm2_t12_35M_UR50D',
    'Random-ESM2-150': 'facebook/esm2_t30_150M_UR50D',
    'Random-ESM2-650': 'facebook/esm2_t36_650M_UR50D',
}


@dataclass
class RandomModelOutput(ModelOutput):
    last_hidden_state: torch.Tensor | None = None
    logits: torch.Tensor | None = None


class RandomModel(nn.Module):
    def __init__(self, config: EsmConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size  # d
        self.holder_param = torch.nn.Parameter(torch.randn(1, 1, self.hidden_size))  # (1, 1, d)
        self.lm_head = nn.Linear(self.hidden_size, config.vocab_size)  # v is config.vocab_size.

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor | RandomModelOutput:
        # input_ids: (b, l); attention_mask: (b, l) or None
        device = self.holder_param.device
        b, l = input_ids.shape
        last_hidden_state = torch.randn(
            b,
            l,
            self.hidden_size,
            device=device,
            dtype=self.holder_param.dtype,
        )  # (b, l, d)
        if return_logits:
            logits = self.lm_head(last_hidden_state)  # (b, l, v)
            return RandomModelOutput(last_hidden_state=last_hidden_state, logits=logits)
        return last_hidden_state  # (b, l, d)


class RandomTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = TransformerForMaskedLM(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        # input_ids: (b, l); attention_mask: (b, l) or None
        if output_attentions:
            model_output = self.transformer(
                input_ids,
                attention_mask,
                output_attentions=output_attentions,
            )  # last_hidden_state: (b, l, d); attentions: each (b, h, l, l)
            return model_output.last_hidden_state, model_output.attentions  # (b, l, d), per-layer attentions

        hidden_state = self.transformer(
            input_ids,
            attention_mask,
        ).last_hidden_state  # (b, l, d)
        return hidden_state  # (b, l, d)


class RandomTransformerForMaskedLM(nn.Module):
    """Random-initialized transformer that returns logits for ProteinGym scoring."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = TransformerForMaskedLM(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> RandomModelOutput:
        # input_ids: (b, l); attention_mask: (b, l) or None
        model_output = self.transformer(
            input_ids,
            attention_mask,
            return_preds=False,
        )  # last_hidden_state: (b, l, d); logits: (b, l, v)
        return RandomModelOutput(
            last_hidden_state=model_output.last_hidden_state,
            logits=model_output.logits,
        )


def _build_random_transformer_config(preset: str) -> TransformerConfig:
    esm_config = EsmConfig.from_pretrained(presets[preset])
    config = TransformerConfig()
    config.hidden_size = esm_config.hidden_size
    config.n_heads = esm_config.num_attention_heads
    config.n_layers = esm_config.num_hidden_layers
    config.vocab_size = esm_config.vocab_size
    config.attn_implementation = 'sdpa'
    return config


def build_random_model(
    preset: str,
    masked_lm: bool = False,
    model_path: str | None = None,
    **kwargs: Any,
) -> tuple[nn.Module, EsmTokenizer]:
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    if preset == 'Random':
        model = RandomModel(EsmConfig.from_pretrained('facebook/esm2_t12_35M_UR50D'))
    else:
        config = _build_random_transformer_config(preset)
        if masked_lm:
            model = RandomTransformerForMaskedLM(config).eval()
        else:
            model = RandomTransformer(config).eval()
    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = build_random_model('Random-Transformer')
    print(model)
    print(tokenizer)
