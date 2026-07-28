import torch
import torch.nn as nn
from typing import Any
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from .utils import select_hidden_state


# Custom checkpoints must be loadable through AutoModel with their remote implementation.


class CustomModelForEmbedding(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)
        if hasattr(self.model, "tokenizer"):
            self.tokenizer = self.model.tokenizer

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
        model_output = self.model(
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


def build_custom_model(
    model_path: str,
    masked_lm: bool = False,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> tuple[nn.Module, Any]:
    if masked_lm:
        model = AutoModelForMaskedLM.from_pretrained(model_path, dtype=dtype, trust_remote_code=True).eval()
    else:
        model = CustomModelForEmbedding(model_path, dtype=dtype).eval()
    try:
        tokenizer = model.tokenizer
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def build_custom_tokenizer(model_path: str, **kwargs: Any) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer


if __name__ == "__main__":
    # py -m src.protify.base_models.custom_model
    model, tokenizer = build_custom_model("answerdotai/ModernBERT-base")
    print(model)
    print(tokenizer)
    seq = "MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL"
    encoded = tokenizer.encode(seq)
    decoded = tokenizer.decode(encoded)
    print(encoded)
    print(decoded)
