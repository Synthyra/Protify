"""E1 adapters backed by the vendored FastPLMs implementation."""

import torch
import torch.nn as nn
from typing import Any

from .utils import ensure_fastplms_submodule_on_path, load_fastplms_model, select_hidden_state


ensure_fastplms_submodule_on_path()

from fastplms.models.e1.modeling_e1 import (
    E1Model,
    E1ForMaskedLM,
    E1ForSequenceClassification,
    E1ForTokenClassification,
    E1BatchPreparer,
)
from .base_tokenizer import BaseSequenceTokenizer


presets = {
    'E1-150': 'Synthyra/Profluent-E1-150M',
    'E1-300': 'Synthyra/Profluent-E1-300M',
    'E1-600': 'Synthyra/Profluent-E1-600M',
}


class E1TokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: E1BatchPreparer) -> None:
        super().__init__(tokenizer)

    def __call__(self, sequences: str | list[str], **kwargs: Any) -> dict[str, Any]:
        if isinstance(sequences, str):
            sequences = [sequences]
        token_batch = self.tokenizer.get_batch_kwargs(sequences)
        # input_ids, sequence ids, position ids, and labels: (b, l); context metadata: length b.
        return token_batch


class E1ForEmbedding(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.e1 = load_fastplms_model(E1Model, model_path, dtype=dtype)

    def forward(
        self,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        hidden_state_index: int = -1,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        # Tensor fields in kwargs: input_ids, sequence ids, position ids, and labels are (b, l).
        output_hidden_states = output_hidden_states or hidden_state_index != -1
        # E1BatchPreparer emits labels for masked-LM/scoring consumers, while
        # the strict E1Model encoder contract intentionally does not accept them.
        kwargs.pop("labels", None)
        model_output = self.e1(
            **kwargs,
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


def get_e1_tokenizer(preset: str, model_path: str | None = None) -> E1TokenizerWrapper:
    tokenizer = E1BatchPreparer(tokenizer_source=model_path or presets[preset])
    return E1TokenizerWrapper(tokenizer)


def build_e1_model(
    preset: str,
    masked_lm: bool = False,
    dtype: torch.dtype | None = None,
    model_path: str | None = None,
    **kwargs: Any,
) -> tuple[nn.Module, E1TokenizerWrapper]:
    model_path = model_path or presets[preset]
    if masked_lm:
        model = load_fastplms_model(E1ForMaskedLM, model_path, dtype=dtype).eval()
    else:
        model = E1ForEmbedding(model_path, dtype=dtype).eval()
    tokenizer = get_e1_tokenizer(preset, model_path=model_path)
    return model, tokenizer


def get_e1_for_training(
    preset: str,
    tokenwise: bool = False,
    num_labels: int | None = None,
    hybrid: bool = False,
    dtype: torch.dtype | None = None,
    model_path: str | None = None,
) -> tuple[nn.Module, E1TokenizerWrapper]:
    model_path = model_path or presets[preset]
    if hybrid:
        model = load_fastplms_model(E1Model, model_path, dtype=dtype).eval()
    else:
        if tokenwise:
            model = load_fastplms_model(
                E1ForTokenClassification,
                model_path,
                num_labels=num_labels,
                dtype=dtype,
            ).eval()
        else:
            model = load_fastplms_model(
                E1ForSequenceClassification,
                model_path,
                num_labels=num_labels,
                dtype=dtype,
            ).eval()
    tokenizer = get_e1_tokenizer(preset, model_path=model_path)
    return model, tokenizer


if __name__ == '__main__':
    # py -m base_models.e1
    model, tokenizer = build_e1_model('E1-150')
    print(model)
    print(tokenizer)
    print(tokenizer(['MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL', 'MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL']))
