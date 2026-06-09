import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM

from .utils import select_hidden_state


"""
Custom models are currently supposed to load completely from AutoModel.from_pretrained(path, trust_remote_code=True)
"""


class CustomModelForEmbedding(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype = None):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)
        if hasattr(self.model, 'tokenizer'):
            self.tokenizer = self.model.tokenizer

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = False,
            hidden_state_index: int = -1,
            **kwargs,
    ) -> torch.Tensor:
        output_hidden_states = output_hidden_states or hidden_state_index != -1
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_state = select_hidden_state(
            out.last_hidden_state,
            out.hidden_states,
            hidden_state_index,
        )
        if output_attentions:
            return hidden_state, out.attentions
        return hidden_state


def build_custom_model(model_path: str, masked_lm: bool = False, dtype: torch.dtype = None, **kwargs):
    if masked_lm:
        model = AutoModelForMaskedLM.from_pretrained(model_path, dtype=dtype, trust_remote_code=True).eval()
    else:
        model = CustomModelForEmbedding(model_path, dtype=dtype).eval()
    try:
        tokenizer = model.tokenizer
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def build_custom_tokenizer(model_path: str, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer


if __name__ == "__main__":
    # py -m src.protify.base_models.custom_model
    model, tokenizer = build_custom_model('answerdotai/ModernBERT-base')
    print(model)
    print(tokenizer)
    seq = 'MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'
    encoded = tokenizer.encode(seq)
    decoded = tokenizer.decode(encoded)
    print(encoded)
    print(decoded)
