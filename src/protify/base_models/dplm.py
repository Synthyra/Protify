"""
We use the FastPLM implementation of DPLM.
"""
import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict

from .utils import ensure_fastplms_submodule_on_path, select_hidden_state


ensure_fastplms_submodule_on_path()

from fastplms.dplm.modeling_dplm import (
    DPLMForMaskedLM,
    DPLMForSequenceClassification,
    DPLMForTokenClassification,
)
from transformers import EsmTokenizer
from .base_tokenizer import BaseSequenceTokenizer


presets = {
    'DPLM-150': 'airkingbd/dplm_150m',
    'DPLM-650': 'airkingbd/dplm_650m',
    'DPLM-3B': 'airkingbd/dplm_3b',
}


class DPLMTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: EsmTokenizer):
        super().__init__(tokenizer)

    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault('return_tensors', 'pt')
        kwargs.setdefault('padding', 'max_length')
        kwargs.setdefault('truncation', True)
        kwargs.setdefault('add_special_tokens', True)
        tokenized = self.tokenizer(sequences, **kwargs)
        return tokenized


class DPLMForEmbedding(nn.Module):
    def __init__(self, model_path: str, return_logits: bool = False, dtype: torch.dtype = None):
        super().__init__()
        self.dplm = DPLMForMaskedLM.from_pretrained(model_path, dtype=dtype, attn_backend="flex")
        self.return_logits = return_logits
        self.dplm.attn_backend = "flex"

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
        out = self.dplm(
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
        if self.return_logits:
            return hidden_state, out.logits
        return hidden_state


def get_dplm_tokenizer(preset: str, model_path: str = None):
    return DPLMTokenizerWrapper(EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D'))


def build_dplm_model(preset: str, masked_lm: bool = False, dtype: torch.dtype = None, model_path: str = None, **kwargs):
    model = DPLMForEmbedding(model_path or presets[preset], return_logits=masked_lm, dtype=dtype).eval()
    tokenizer = get_dplm_tokenizer(preset)
    return model, tokenizer


def get_dplm_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False, dtype: torch.dtype = None, model_path: str = None):
    model_path = model_path or presets[preset]
    if hybrid:
        model = DPLMForMaskedLM.from_pretrained(model_path, dtype=dtype, attn_backend="flex").eval()
    else:
        if tokenwise:
            model = DPLMForTokenClassification.from_pretrained(
                model_path,
                num_labels=num_labels,
                dtype=dtype,
                attn_backend="flex",
            ).eval()
        else:
            model = DPLMForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_labels,
                dtype=dtype,
                attn_backend="flex",
            ).eval()
    model.attn_backend = "flex"
    tokenizer = get_dplm_tokenizer(preset)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.dplm
    model, tokenizer = build_dplm_model('DPLM-150')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))
