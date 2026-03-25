"""
We use the FastPLM implementation of ESMC.
"""
import sys
import os
import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict

_FASTPLMS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fastplms')
if _FASTPLMS not in sys.path:
    sys.path.insert(0, _FASTPLMS)

from fastplms.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusModel,
    ESMplusplusForMaskedLM,
    ESMplusplusForSequenceClassification,
    ESMplusplusForTokenClassification,
)
from .base_tokenizer import BaseSequenceTokenizer
from .esmc_utils import EsmSequenceTokenizer


presets = {
    'ESMC-300': 'Synthyra/ESMplusplus_small',
    'ESMC-600': 'Synthyra/ESMplusplus_large',
}


class ESMTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: EsmSequenceTokenizer):
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


class ESMplusplusForEmbedding(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype = None):
        super().__init__()
        self.esm = ESMplusplusModel.from_pretrained(model_path, dtype=dtype, attn_backend="flex")
        self.esm.attn_backend = "flex"

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = False,
            **kwargs,
    ) -> torch.Tensor:
        if output_attentions:
            out = self.esm(input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
            return out.last_hidden_state, out.attentions
        else:
            return self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def get_esmc_tokenizer(preset: str, model_path: str = None):
    tokenizer = EsmSequenceTokenizer()
    return ESMTokenizerWrapper(tokenizer)


def build_esmc_model(preset: str, masked_lm: bool = False, dtype: torch.dtype = None, model_path: str = None, **kwargs):
    path = model_path or presets[preset]
    if masked_lm:
        model = ESMplusplusForMaskedLM.from_pretrained(path, dtype=dtype, attn_backend="flex").eval()
        model.attn_backend = "flex"
    else:
        model = ESMplusplusForEmbedding(path, dtype=dtype).eval()
    tokenizer = get_esmc_tokenizer(preset)
    return model, tokenizer


def get_esmc_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False, dtype: torch.dtype = None, model_path: str = None):
    model_path = model_path or presets[preset]
    if hybrid:
        model = ESMplusplusModel.from_pretrained(model_path, dtype=dtype, attn_backend="flex").eval()
    else:
        if tokenwise:
            model = ESMplusplusForTokenClassification.from_pretrained(
                model_path,
                num_labels=num_labels,
                dtype=dtype,
                attn_backend="flex",
            ).eval()
        else:
            model = ESMplusplusForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_labels,
                dtype=dtype,
                attn_backend="flex",
            ).eval()
    model.attn_backend = "flex"
    tokenizer = get_esmc_tokenizer(preset)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.esmc
    model, tokenizer = build_esmc_model('ESMC-300')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))
