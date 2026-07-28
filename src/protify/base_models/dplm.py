"""DPLM adapters backed by the vendored FastPLMs implementation."""

import torch
import torch.nn as nn
from typing import Any

from .utils import ensure_fastplms_submodule_on_path, load_fastplms_model, select_hidden_state


ensure_fastplms_submodule_on_path()

from fastplms.models.dplm.modeling_dplm import (
    DPLMForMaskedLM,
    DPLMForSequenceClassification,
    DPLMForTokenClassification,
)
from transformers import EsmTokenizer
from .base_tokenizer import BaseSequenceTokenizer


presets = {
    'DPLM-150': 'Synthyra/DPLM-150M',
    'DPLM-650': 'Synthyra/DPLM-650M',
    'DPLM-3B': 'Synthyra/DPLM-3B',
}


class DPLMTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: EsmTokenizer) -> None:
        super().__init__(tokenizer)

    def __call__(self, sequences: str | list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", "max_length")
        kwargs.setdefault("truncation", True)
        kwargs.setdefault("add_special_tokens", True)
        token_batch = self.tokenizer(sequences, **kwargs)
        return token_batch  # default tensor fields: (b, l)


class DPLMForEmbedding(nn.Module):
    def __init__(
        self,
        model_path: str,
        return_logits: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dplm = load_fastplms_model(DPLMForMaskedLM, model_path, dtype=dtype)
        self.return_logits = return_logits

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
        model_output = self.dplm(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )  # last/hidden states: (b, l, d); logits: (b, l, v); attentions: each (b, h, l, l)
        hidden_state = select_hidden_state(
            model_output.last_hidden_state,
            model_output.hidden_states,
            hidden_state_index,
        )  # (b, l, d)
        if output_attentions:
            return hidden_state, model_output.attentions  # (b, l, d), each attention (b, h, l, l)
        if self.return_logits:
            return hidden_state, model_output.logits  # (b, l, d), (b, l, v)
        return hidden_state  # (b, l, d)


def get_dplm_tokenizer(preset: str, model_path: str | None = None) -> DPLMTokenizerWrapper:
    return DPLMTokenizerWrapper(EsmTokenizer.from_pretrained(model_path or presets[preset]))


def build_dplm_model(
    preset: str,
    masked_lm: bool = False,
    dtype: torch.dtype | None = None,
    model_path: str | None = None,
    **kwargs: Any,
) -> tuple[nn.Module, DPLMTokenizerWrapper]:
    model = DPLMForEmbedding(model_path or presets[preset], return_logits=masked_lm, dtype=dtype).eval()
    tokenizer = get_dplm_tokenizer(preset, model_path=model_path or presets[preset])
    return model, tokenizer


def get_dplm_for_training(
    preset: str,
    tokenwise: bool = False,
    num_labels: int | None = None,
    hybrid: bool = False,
    dtype: torch.dtype | None = None,
    model_path: str | None = None,
) -> tuple[nn.Module, DPLMTokenizerWrapper]:
    model_path = model_path or presets[preset]
    if hybrid:
        model = load_fastplms_model(DPLMForMaskedLM, model_path, dtype=dtype).eval()
    else:
        if tokenwise:
            model = load_fastplms_model(
                DPLMForTokenClassification,
                model_path,
                num_labels=num_labels,
                dtype=dtype,
            ).eval()
        else:
            model = load_fastplms_model(
                DPLMForSequenceClassification,
                model_path,
                num_labels=num_labels,
                dtype=dtype,
            ).eval()
    tokenizer = get_dplm_tokenizer(preset, model_path=model_path)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.dplm
    model, tokenizer = build_dplm_model('DPLM-150')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))
