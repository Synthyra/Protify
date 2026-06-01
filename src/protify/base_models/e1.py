"""
We use the FastPLM implementation of E1.
"""
import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict, Tuple

from .utils import ensure_fastplms_submodule_on_path, select_hidden_state


ensure_fastplms_submodule_on_path()

from fastplms.e1.modeling_e1 import (
    E1Model,
    E1ForMaskedLM,
    E1ForSequenceClassification,
    E1ForTokenClassification,
)
from .base_tokenizer import BaseSequenceTokenizer
from .e1_utils import E1BatchPreparer


presets = {
    'E1-150': 'Synthyra/Profluent-E1-150M',
    'E1-300': 'Synthyra/Profluent-E1-300M',
    'E1-600': 'Synthyra/Profluent-E1-600M',
}


class E1TokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: E1BatchPreparer):
        super().__init__(tokenizer)

    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        tokenized = self.tokenizer.get_batch_kwargs(sequences)
        return tokenized


class E1ForEmbedding(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype = None):
        super().__init__()
        self.e1 = E1Model.from_pretrained(model_path, dtype=dtype, attn_backend="flex")
        self.e1.attn_backend = "flex"

    def forward(
            self,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            hidden_state_index: int = -1,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        output_hidden_states = output_hidden_states or hidden_state_index != -1
        out = self.e1(
            **kwargs,
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


def get_e1_tokenizer(preset: str, model_path: str = None):
    tokenizer = E1BatchPreparer()
    return E1TokenizerWrapper(tokenizer)


def build_e1_model(preset: str, masked_lm: bool = False, dtype: torch.dtype = None, model_path: str = None, **kwargs):
    model_path = model_path or presets[preset]
    if masked_lm:
        model = E1ForMaskedLM.from_pretrained(model_path, dtype=dtype, attn_backend="flex").eval()
        model.attn_backend = "flex"
    else:
        model = E1ForEmbedding(model_path, dtype=dtype).eval()
    tokenizer = get_e1_tokenizer(preset)
    return model, tokenizer


def get_e1_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False, dtype: torch.dtype = None, model_path: str = None):
    model_path = model_path or presets[preset]
    if hybrid:
        model = E1Model.from_pretrained(model_path, dtype=dtype, attn_backend="flex").eval()
    else:
        if tokenwise:
            model = E1ForTokenClassification.from_pretrained(
                model_path,
                num_labels=num_labels,
                dtype=dtype,
                attn_backend="flex",
            ).eval()
        else:
            model = E1ForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_labels,
                dtype=dtype,
                attn_backend="flex",
            ).eval()
    model.attn_backend = "flex"
    tokenizer = get_e1_tokenizer(preset)
    return model, tokenizer


if __name__ == '__main__':
    # py -m base_models.e1
    model, tokenizer = build_e1_model('E1-150')
    print(model)
    print(tokenizer)
    print(tokenizer(['MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL', 'MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL']))
