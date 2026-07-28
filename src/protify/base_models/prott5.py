import re

import torch
import torch.nn as nn
from typing import Any
from transformers import T5EncoderModel, T5Tokenizer

from .base_tokenizer import BaseSequenceTokenizer
from .t5 import T5ForSequenceClassification, T5ForTokenClassification
from .utils import select_hidden_state


presets = {
    'ProtT5': 'Rostlab/prot_t5_xl_half_uniref50-enc',
    'ProtT5-XL-UniRef50-full-prec': 'Rostlab/prot_t5_xl_uniref50',
    'ProtT5-XXL-UniRef50': 'Rostlab/prot_t5_xxl_uniref50',
    'ProtT5-XL-BFD': 'Rostlab/prot_t5_xl_bfd',
    'ProtT5-XXL-BFD': 'Rostlab/prot_t5_xxl_bfd',
}


class T5TokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: T5Tokenizer) -> None:
        super().__init__(tokenizer)

    def __call__(self, sequences: str | list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", "max_length")
        kwargs.setdefault("truncation", True)
        kwargs.setdefault("add_special_tokens", True)
        sequences = [re.sub(r"[UZOB]", "X", seq) for seq in sequences]
        sequences = [" ".join(seq) for seq in sequences]
        token_batch = self.tokenizer(sequences, **kwargs)
        return token_batch  # default tensor fields: (b, l)


class Prott5ForEmbedding(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.plm = T5EncoderModel.from_pretrained(model_path, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = False,
        hidden_state_index: int = -1,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        # input_ids: (b, l); attention_mask: (b, l) or None
        output_hidden_states = output_hidden_states or hidden_state_index != -1
        model_output = self.plm(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )  # last_hidden_state: (b, l, d); hidden states: each (b, l, d); attentions: each (b, h, l, l)
        hidden_state = select_hidden_state(
            model_output.last_hidden_state,
            model_output.hidden_states,
            hidden_state_index,
        )  # (b, l, d)
        if output_attentions:
            return hidden_state, model_output.attentions  # (b, l, d), each attention (b, h, l, l)
        return hidden_state  # (b, l, d)


def get_prott5_tokenizer(preset: str, model_path: str | None = None) -> T5TokenizerWrapper:
    return T5TokenizerWrapper(T5Tokenizer.from_pretrained(model_path or presets[preset]))


def build_prott5_model(
    preset: str,
    masked_lm: bool = False,
    dtype: torch.dtype | None = None,
    model_path: str | None = None,
    **kwargs: Any,
) -> tuple[nn.Module, T5TokenizerWrapper]:
    model_path = model_path or presets[preset]
    model = Prott5ForEmbedding(model_path, dtype=dtype).eval()
    tokenizer = get_prott5_tokenizer(preset)
    return model, tokenizer


def get_prott5_for_training(
    preset: str,
    tokenwise: bool = False,
    num_labels: int | None = None,
    hybrid: bool = False,
    dtype: torch.dtype | None = None,
    model_path: str | None = None,
) -> tuple[nn.Module, T5TokenizerWrapper]:
    model_path = model_path or presets[preset]
    if hybrid:
        model = T5EncoderModel.from_pretrained(model_path, dtype=dtype).eval()
    else:
        if tokenwise:
            model = T5ForTokenClassification.from_pretrained(model_path, num_labels=num_labels, dtype=dtype).eval()
        else:
            model = T5ForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, dtype=dtype).eval()
    tokenizer = get_prott5_tokenizer(preset)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.prott5
    model, tokenizer = build_prott5_model('ProtT5')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))
