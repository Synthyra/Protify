"""
We use the FastPLM implementation of DPLM2.
"""
import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict

from .utils import ensure_fastplms_submodule_on_path, load_fastplms_model, select_hidden_state


ensure_fastplms_submodule_on_path()

from fastplms.models.dplm2.modeling_dplm2 import (
    DPLM2ForMaskedLM,
    DPLM2ForSequenceClassification,
    DPLM2ForTokenClassification,
)
from fastplms.models.dplm2.tokenization_dplm2 import DPLM2Tokenizer
from .base_tokenizer import BaseSequenceTokenizer


presets = {
    "DPLM2-150": "Synthyra/DPLM2-150M",
    "DPLM2-650": "Synthyra/DPLM2-650M",
    "DPLM2-3B": "Synthyra/DPLM2-3B",
}


class DPLM2TokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: DPLM2Tokenizer):
        super().__init__(tokenizer)

    def __call__(
        self, sequences: Union[str, List[str]], **kwargs
    ) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        sequences = [
            f"{self.tokenizer.aa_cls_token}{sequence}{self.tokenizer.aa_eos_token}"
            for sequence in sequences
        ]
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", "max_length")
        kwargs.setdefault("truncation", True)
        kwargs["add_special_tokens"] = False
        tokenized = self.tokenizer(sequences, **kwargs)
        return tokenized


class DPLM2ForEmbedding(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype = None):
        super().__init__()
        self.dplm2 = load_fastplms_model(
            DPLM2ForMaskedLM,
            model_path,
            dtype=dtype,
            attn_implementation="sdpa",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False,
        hidden_state_index: int = -1,
        **kwargs,
    ) -> torch.Tensor:
        output_hidden_states = output_hidden_states or hidden_state_index != -1
        out = self.dplm2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_state = select_hidden_state(
            out.last_hidden_state,
            out.hidden_states,
            hidden_state_index,
        )
        if output_attentions:
            return hidden_state, out.attentions
        return hidden_state


def get_dplm2_tokenizer(preset: str, model_path: str = None):
    return DPLM2TokenizerWrapper(
        DPLM2Tokenizer.from_pretrained(model_path or presets[preset])
    )


def build_dplm2_model(preset: str, masked_lm: bool = False, dtype: torch.dtype = None, model_path: str = None, **kwargs):
    model_path = model_path or presets[preset]
    if masked_lm:
        model = load_fastplms_model(
            DPLM2ForMaskedLM,
            model_path,
            dtype=dtype,
            attn_implementation="sdpa",
        ).eval()
    else:
        model = DPLM2ForEmbedding(model_path, dtype=dtype).eval()
    tokenizer = get_dplm2_tokenizer(preset, model_path=model_path)
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
        model = load_fastplms_model(
            DPLM2ForMaskedLM,
            model_path,
            dtype=dtype,
            attn_implementation="sdpa",
        ).eval()
    else:
        if tokenwise:
            model = load_fastplms_model(
                DPLM2ForTokenClassification,
                model_path,
                num_labels=num_labels,
                dtype=dtype,
                attn_implementation="sdpa",
            ).eval()
        else:
            model = load_fastplms_model(
                DPLM2ForSequenceClassification,
                model_path,
                num_labels=num_labels,
                dtype=dtype,
                attn_implementation="sdpa",
            ).eval()
    tokenizer = get_dplm2_tokenizer(preset, model_path=model_path)
    return model, tokenizer


if __name__ == "__main__":
    # py -m src.protify.base_models.dplm2
    model, tokenizer = build_dplm2_model("DPLM2-150")
    print(model)
    print(tokenizer)
    print(tokenizer("MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL"))
