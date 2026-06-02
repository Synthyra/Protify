"""
We use the FastESM2 implementation of ESM2.
"""
import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict

from .utils import ensure_fastplms_submodule_on_path, select_hidden_state


ensure_fastplms_submodule_on_path()

from fastplms.esm2.modeling_fastesm import (
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
    def __init__(self, tokenizer: EsmTokenizer):
        super().__init__(tokenizer)

    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault('return_tensors', 'pt')
        kwargs.setdefault('padding', 'max_length')
        kwargs.setdefault('truncation', True)
        kwargs.setdefault('add_special_tokens', True)
        tokenized = self.tokenizer(sequences, **kwargs)
        return tokenized


class FastEsmForEmbedding(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype = None):
        super().__init__()
        self.esm = FastEsmModel.from_pretrained(model_path, dtype=dtype, attn_backend="flex")
        self.esm.attn_backend = "flex"

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
        out = self.esm(
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


def get_esm2_tokenizer(preset: str, model_path: str = None):
    return ESM2TokenizerWrapper(EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D'))


def build_esm2_model(preset: str, masked_lm: bool = False, dtype: torch.dtype = None, model_path: str = None, **kwargs):
    path = model_path or presets[preset]
    if masked_lm:
        model = FastEsmForMaskedLM.from_pretrained(path, dtype=dtype, attn_backend="flex").eval()
        model.attn_backend = "flex"
    else:
        model = FastEsmForEmbedding(path, dtype=dtype).eval()
    tokenizer = get_esm2_tokenizer(preset)
    return model, tokenizer


def get_esm2_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False, dtype: torch.dtype = None, model_path: str = None):
    model_path = model_path or presets[preset]
    if hybrid:
        model = FastEsmModel.from_pretrained(model_path, dtype=dtype, attn_backend="flex").eval()
    else:
        if tokenwise:
            model = FastEsmForTokenClassification.from_pretrained(
                model_path,
                num_labels=num_labels,
                dtype=dtype,
                attn_backend="flex",
            ).eval()
        else:
            model = FastEsmForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_labels,
                dtype=dtype,
                attn_backend="flex",
            ).eval()
    model.attn_backend = "flex"
    tokenizer = get_esm2_tokenizer(preset)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.esm2
    model, tokenizer = build_esm2_model('ESM2-8')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))
