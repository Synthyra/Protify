"""ESM++ adapters backed by the vendored FastPLMs implementation."""

import torch
import torch.nn as nn
from typing import Any

from .utils import ensure_fastplms_submodule_on_path, load_fastplms_model, select_hidden_state


ensure_fastplms_submodule_on_path()

from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusModel,
    ESMplusplusForMaskedLM,
    ESMplusplusForSequenceClassification,
    ESMplusplusForTokenClassification,
    EsmSequenceTokenizer,
)
from .base_tokenizer import BaseSequenceTokenizer


presets = {
    'ESMC-300': 'Synthyra/ESMplusplus_small',
    'ESMC-600': 'Synthyra/ESMplusplus_large',
    'ESMC-6B': 'Synthyra/ESMplusplus_6B',
}


class ESMTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: EsmSequenceTokenizer) -> None:
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


class ESMplusplusForEmbedding(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.esm = load_fastplms_model(ESMplusplusModel, model_path, dtype=dtype)

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
        model_output = self.esm(
            input_ids=input_ids,
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


def get_esmc_tokenizer(preset: str, model_path: str | None = None) -> ESMTokenizerWrapper:
    tokenizer = EsmSequenceTokenizer()
    return ESMTokenizerWrapper(tokenizer)


def build_esmc_model(
    preset: str,
    masked_lm: bool = False,
    dtype: torch.dtype | None = None,
    model_path: str | None = None,
    **kwargs: Any,
) -> tuple[nn.Module, ESMTokenizerWrapper]:
    path = model_path or presets[preset]
    if masked_lm:
        model = load_fastplms_model(ESMplusplusForMaskedLM, path, dtype=dtype).eval()
    else:
        model = ESMplusplusForEmbedding(path, dtype=dtype).eval()
    tokenizer = get_esmc_tokenizer(preset)
    return model, tokenizer


def get_esmc_for_training(
    preset: str,
    tokenwise: bool = False,
    num_labels: int | None = None,
    hybrid: bool = False,
    dtype: torch.dtype | None = None,
    model_path: str | None = None,
) -> tuple[nn.Module, ESMTokenizerWrapper]:
    model_path = model_path or presets[preset]
    if hybrid:
        model = load_fastplms_model(ESMplusplusModel, model_path, dtype=dtype).eval()
    else:
        if tokenwise:
            model = load_fastplms_model(
                ESMplusplusForTokenClassification,
                model_path,
                num_labels=num_labels,
                dtype=dtype,
            ).eval()
        else:
            model = load_fastplms_model(
                ESMplusplusForSequenceClassification,
                model_path,
                num_labels=num_labels,
                dtype=dtype,
            ).eval()
    tokenizer = get_esmc_tokenizer(preset)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.esmc
    model, tokenizer = build_esmc_model('ESMC-300')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))
