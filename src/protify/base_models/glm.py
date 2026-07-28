import torch
import torch.nn as nn
from typing import Any
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from .base_tokenizer import BaseSequenceTokenizer
from .utils import select_hidden_state


presets = {
    'GLM2-150': 'tattabio/gLM2_150M',
    'GLM2-650': 'tattabio/gLM2_650M',
    'GLM2-GAIA': 'tattabio/gLM2_650M_embed'
}


class GLMTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: Any) -> None:
        super().__init__(tokenizer)
        self.plus_token = "<+>"
        if self.plus_token not in self.tokenizer.vocab:
            print(f"Warning: Token '{self.plus_token}' not found in GLM tokenizer vocabulary.")

    def __call__(self, sequences: str | list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", "max_length")
        kwargs.setdefault("truncation", True)
        kwargs.setdefault("add_special_tokens", True)
        modified_sequences = [self.plus_token + seq for seq in sequences]
        token_batch = self.tokenizer(modified_sequences, **kwargs)
        return token_batch  # default tensor fields: (b, l)


class gLM2ForEmbedding(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.glm2 = AutoModel.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = False,
        hidden_state_index: int = -1,
        token_type_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # input_ids, attention_mask, and token_type_ids: (b, l); masks/type ids may be None.
        output_hidden_states = output_hidden_states or hidden_state_index != -1
        assert not output_attentions or not output_hidden_states, (
            "output_attentions=True and output_hidden_states=True are not supported by gLM2ForEmbedding."
        )

        model_output = self.glm2(
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


class gLM2GAIAForEmbedding(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.glm2_embed = AutoModel.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)
        self.glm2 = self.glm2_embed.glm2

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = False,
        hidden_state_index: int = -1,
        token_type_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # input_ids, attention_mask, and token_type_ids: (b, l); masks/type ids may be None.
        output_hidden_states = output_hidden_states or hidden_state_index != -1
        assert not output_attentions or not output_hidden_states, (
            "output_attentions=True and output_hidden_states=True are not supported by gLM2ForEmbedding."
        )

        model_output = self.glm2(
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


def get_glm2_tokenizer(preset: str, model_path: str | None = None) -> GLMTokenizerWrapper:
    return GLMTokenizerWrapper(AutoTokenizer.from_pretrained(model_path or presets[preset], trust_remote_code=True))


def build_glm2_model(
    preset: str,
    masked_lm: bool = False,
    dtype: torch.dtype | None = None,
    model_path: str | None = None,
    **kwargs: Any,
) -> tuple[nn.Module, GLMTokenizerWrapper]:
    model_path = model_path or presets[preset]
    if masked_lm:
        model = AutoModelForMaskedLM.from_pretrained(model_path, dtype=dtype, trust_remote_code=True).eval()
    else:
        if preset == "GLM2-GAIA":
            model = gLM2GAIAForEmbedding(model_path, dtype=dtype).eval()
        else:
            model = gLM2ForEmbedding(model_path, dtype=dtype).eval()
    tokenizer = get_glm2_tokenizer(preset)
    return model, tokenizer


def get_glm2_for_training(
    preset: str,
    tokenwise: bool = False,
    num_labels: int | None = None,
    hybrid: bool = False,
    dtype: torch.dtype | None = None,
    model_path: str | None = None,
) -> tuple[nn.Module, GLMTokenizerWrapper]:
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
    tokenizer = get_glm2_tokenizer(preset)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.glm
    model, tokenizer = build_glm2_model('GLM2-650')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))
