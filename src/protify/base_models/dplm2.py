"""
We use the FastPLM implementation of DPLM2.
"""
import sys
import os
import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict

_FASTPLMS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fastplms')
if _FASTPLMS not in sys.path:
    sys.path.insert(0, _FASTPLMS)

from fastplms.dplm2.modeling_dplm2 import (
    DPLM2ForMaskedLM,
    DPLM2ForSequenceClassification,
    DPLM2ForTokenClassification,
)
from transformers import EsmTokenizer
from .base_tokenizer import BaseSequenceTokenizer


presets = {
    "DPLM2-150": "airkingbd/dplm2_150m",
    "DPLM2-650": "airkingbd/dplm2_650m",
    "DPLM2-3B": "airkingbd/dplm2_3b",
}


class DPLM2TokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: EsmTokenizer):
        super().__init__(tokenizer)

    def __call__(
        self, sequences: Union[str, List[str]], **kwargs
    ) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", "max_length")
        kwargs.setdefault("truncation", True)
        kwargs.setdefault("add_special_tokens", True)
        tokenized = self.tokenizer(sequences, **kwargs)
        return tokenized


class DPLM2ForEmbedding(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype = None):
        super().__init__()
        self.dplm2 = DPLM2ForMaskedLM.from_pretrained(model_path, dtype=dtype, attn_backend="flex")
        self.dplm2.attn_backend = "flex"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        out = self.dplm2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        if output_attentions:
            return out.last_hidden_state, out.attentions
        return out.last_hidden_state


def get_dplm2_tokenizer(preset: str, model_path: str = None):
    return DPLM2TokenizerWrapper(EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D"))


def build_dplm2_model(preset: str, masked_lm: bool = False, dtype: torch.dtype = None, model_path: str = None, **kwargs):
    model_path = model_path or presets[preset]
    if masked_lm:
        model = DPLM2ForMaskedLM.from_pretrained(model_path, dtype=dtype, attn_backend="flex").eval()
        model.attn_backend = "flex"
    else:
        model = DPLM2ForEmbedding(model_path, dtype=dtype).eval()
    tokenizer = get_dplm2_tokenizer(preset)
    return model, tokenizer


def get_dplm2_for_training(
    preset: str,
    tokenwise: bool = False,
    num_labels: int = None,
    hybrid: bool = False,
    dtype: torch.dtype = None,
    model_path: str = None,
):
    model_path = model_path or presets[preset]
    if hybrid:
        model = DPLM2ForMaskedLM.from_pretrained(model_path, dtype=dtype, attn_backend="flex").eval()
    else:
        if tokenwise:
            model = DPLM2ForTokenClassification.from_pretrained(
                model_path,
                num_labels=num_labels,
                dtype=dtype,
                attn_backend="flex",
            ).eval()
        else:
            model = DPLM2ForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_labels,
                dtype=dtype,
                attn_backend="flex",
            ).eval()
    model.attn_backend = "flex"
    tokenizer = get_dplm2_tokenizer(preset)
    return model, tokenizer


if __name__ == "__main__":
    # py -m src.protify.base_models.dplm2
    model, tokenizer = build_dplm2_model("DPLM2-150")
    print(model)
    print(tokenizer)
    print(tokenizer("MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL"))
