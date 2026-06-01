import torch
import torch.nn as nn
import re
from typing import Optional, Union, List, Dict
from transformers import (
    BertModel,
    BertTokenizer,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertForMaskedLM,
)
from .base_tokenizer import BaseSequenceTokenizer
from .utils import select_hidden_state


presets = {
    'ProtBert': 'Rostlab/prot_bert',
    'ProtBert-BFD': 'Rostlab/prot_bert_bfd',
}


class BERTTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: BertTokenizer):
        super().__init__(tokenizer)
        
    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault('return_tensors', 'pt')
        kwargs.setdefault('padding', 'max_length')
        kwargs.setdefault('truncation', True)
        kwargs.setdefault('add_special_tokens', True)
        sequences = [re.sub(r"[UZOB]", "X", seq) for seq in sequences]
        sequences = [' '.join(seq) for seq in sequences]
        tokenized = self.tokenizer(sequences, **kwargs)
        return tokenized


class ProtBertForEmbedding(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype = None):
        super().__init__()
        self.plm = BertModel.from_pretrained(model_path, dtype=dtype, attn_implementation="sdpa")

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
        out = self.plm(
            input_ids,
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


def get_protbert_tokenizer(preset: str, model_path: str = None):
    return BERTTokenizerWrapper(BertTokenizer.from_pretrained('Rostlab/prot_bert'))


def build_protbert_model(preset: str, masked_lm: bool = False, dtype: torch.dtype = None, model_path: str = None, **kwargs):
    model_path = model_path or presets[preset]
    if masked_lm:
        model = BertForMaskedLM.from_pretrained(model_path, dtype=dtype, attn_implementation="sdpa").eval()
    else:
        model = ProtBertForEmbedding(model_path, dtype=dtype).eval()
    tokenizer = get_protbert_tokenizer(preset)
    return model, tokenizer


def get_protbert_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False, dtype: torch.dtype = None, model_path: str = None):
    model_path = model_path or presets[preset]
    if hybrid:
        model = BertModel.from_pretrained(model_path, dtype=dtype).eval()
    else:
        if tokenwise:
            model = BertForTokenClassification.from_pretrained(model_path, num_labels=num_labels, dtype=dtype).eval()
        else:
            model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, dtype=dtype).eval()
    tokenizer = get_protbert_tokenizer(preset)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.protbert
    model, tokenizer = build_protbert_model('ProtBert')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))
