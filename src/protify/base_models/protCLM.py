import torch
import torch.nn as nn
from typing import Any
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from .base_tokenizer import BaseSequenceTokenizer
from .utils import select_hidden_state


presets = {
    "ProtCLM-1b": "biomap-research/proteinglm-1b-clm",
}


class ProtCLMTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: Any) -> None:
        super().__init__(tokenizer)

    def __call__(self, sequences: str | list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", "longest")
        kwargs.setdefault("add_special_tokens", True)
        token_batch = self.tokenizer(sequences, **kwargs)
        return token_batch  # default tensor fields: (b, l)


class ProtCLMForEmbedding(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.plm = AutoModel.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        hidden_state_index: int = -1,
        **kwargs: Any,
    ) -> torch.Tensor:
        # input_ids: (b, l); attention_mask: (b, l) or None
        output_hidden_states = bool(output_hidden_states) or hidden_state_index != -1
        assert not output_attentions or not output_hidden_states, (
            "output_attentions=True and output_hidden_states=True are not supported by ProtCLMForEmbedding."
        )

        model_output = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )  # last_hidden_state: (b, l, d); hidden states: each (b, l, d)
        hidden_state = select_hidden_state(
            model_output.last_hidden_state,
            model_output.hidden_states,
            hidden_state_index,
        )  # (b, l, d)
        return hidden_state  # (b, l, d)


def get_protCLM_tokenizer(preset: str, model_path: str | None = None) -> ProtCLMTokenizerWrapper:
    return ProtCLMTokenizerWrapper(
        AutoTokenizer.from_pretrained(model_path or presets[preset], trust_remote_code=True)
    )


def build_protCLM(
    preset: str,
    masked_lm: bool = False,
    dtype: torch.dtype | None = None,
    model_path: str | None = None,
    **kwargs: Any,
) -> tuple[nn.Module, ProtCLMTokenizerWrapper]:
    if masked_lm:
        raise ValueError(f"Model {preset} does not support masked language modeling")
    model_path = model_path or presets[preset]
    model = ProtCLMForEmbedding(model_path, dtype=dtype).eval()
    tokenizer = get_protCLM_tokenizer(preset)
    return model, tokenizer


def get_protCLM_for_training(
    preset: str,
    tokenwise: bool = False,
    num_labels: int | None = None,
    hybrid: bool = False,
    dtype: torch.dtype | None = None,
    model_path: str | None = None,
) -> tuple[nn.Module, ProtCLMTokenizerWrapper]:
    model_path = model_path or presets[preset]
    if hybrid:
        model = AutoModel.from_pretrained(model_path, dtype=dtype, trust_remote_code=True).eval()
    else:
        if tokenwise:
            model = AutoModelForTokenClassification.from_pretrained(
                model_path, num_labels=num_labels, dtype=dtype, trust_remote_code=True
            ).eval()
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=num_labels, dtype=dtype, trust_remote_code=True
            ).eval()
    tokenizer = get_protCLM_tokenizer(preset)
    return model, tokenizer


if __name__ == "__main__":
    # py -m src.protify.base_models.protCLM
    model, tokenizer = build_protCLM("ProtCLM-1b")
    print(model)
    print(tokenizer)
    print(tokenizer("MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL"))
