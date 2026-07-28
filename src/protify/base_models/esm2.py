"""ESM2 adapters backed by the vendored FastPLMs implementation."""

import torch
import torch.nn as nn
from typing import Any

from .utils import ensure_fastplms_submodule_on_path, load_fastplms_model, select_hidden_state


ensure_fastplms_submodule_on_path()

from fastplms.models.esm2.modeling_fastesm import (
    FastEsmModel,
    FastEsmForMaskedLM,
    FastEsmForSequenceClassification,
    FastEsmForTokenClassification,
)
from transformers import EsmTokenizer
from .base_tokenizer import BaseSequenceTokenizer


presets = {
    'ESM2-8': 'Synthyra/ESM2-8M',
    'ESM2-35': 'Synthyra/ESM2-35M',
    'ESM2-150': 'Synthyra/ESM2-150M',
    'ESM2-650': 'Synthyra/ESM2-650M',
    'ESM2-3B': 'Synthyra/ESM2-3B',
    'DSM-150': 'GleghornLab/DSM_150',
    'DSM-650': 'GleghornLab/DSM_650',
    'DSM-PPI': 'Synthyra/DSM_ppi_full',
}


class ESM2TokenizerWrapper(BaseSequenceTokenizer):
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


class FastEsmForEmbedding(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.esm = load_fastplms_model(FastEsmModel, model_path, dtype=dtype)

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


def get_esm2_tokenizer(preset: str, model_path: str | None = None) -> ESM2TokenizerWrapper:
    return ESM2TokenizerWrapper(EsmTokenizer.from_pretrained(model_path or presets[preset]))


def build_esm2_model(
    preset: str,
    masked_lm: bool = False,
    dtype: torch.dtype | None = None,
    model_path: str | None = None,
    **kwargs: Any,
) -> tuple[nn.Module, ESM2TokenizerWrapper]:
    path = model_path or presets[preset]
    if masked_lm:
        model = load_fastplms_model(FastEsmForMaskedLM, path, dtype=dtype).eval()
    else:
        model = FastEsmForEmbedding(path, dtype=dtype).eval()
    tokenizer = get_esm2_tokenizer(preset, model_path=path)
    return model, tokenizer


def get_esm2_for_training(
    preset: str,
    tokenwise: bool = False,
    num_labels: int | None = None,
    hybrid: bool = False,
    dtype: torch.dtype | None = None,
    model_path: str | None = None,
) -> tuple[nn.Module, ESM2TokenizerWrapper]:
    model_path = model_path or presets[preset]
    if hybrid:
        model = load_fastplms_model(FastEsmModel, model_path, dtype=dtype).eval()
    else:
        if tokenwise:
            model = load_fastplms_model(
                FastEsmForTokenClassification,
                model_path,
                num_labels=num_labels,
                dtype=dtype,
            ).eval()
        else:
            model = load_fastplms_model(
                FastEsmForSequenceClassification,
                model_path,
                num_labels=num_labels,
                dtype=dtype,
            ).eval()
    tokenizer = get_esm2_tokenizer(preset, model_path=model_path)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.esm2
    model, tokenizer = build_esm2_model('ESM2-8')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))
