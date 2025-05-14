import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)
from .base_tokenizer import BaseSequenceTokenizer


presets = {
    "ProtCLM-1b": "biomap-research/proteinglm-1b-clm",
    #"ProtCLM-3b": "biomap-research/proteinglm-3b-clm",
    #"ProtCLM-7b": "biomap-research/proteinglm-7b-clm"
}


class ProtCLMTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)
    def __call__(self, sequences: Union[str, List[str]], **kwargs):
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", "longest")
        kwargs.setdefault("add_special_tokens", True)
        return self.tokenizer(sequences, **kwargs)

class ProtCLMForEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.plm = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> torch.Tensor:  
        assert not output_attentions or not output_hidden_states, (
            "output_attentions=True and output_hidden_states=True are not supported by ProtCLMForEmbedding."
        )

        out = self.plm(
            input_ids=input_ids, 
            attention_mask=attention_mask
            )
        return out.last_hidden_state



def get_protCLM_tokenizer(preset: str) -> BaseSequenceTokenizer:
    return ProtCLMTokenizerWrapper(
        AutoTokenizer.from_pretrained(presets[preset], trust_remote_code=True)
    )

def build_protCLM(preset: str) -> Tuple[AutoModel, BaseSequenceTokenizer]:
    model_path = presets[preset]
    model = ProtCLMForEmbedding(model_path, trust_remote_code=True).eval()
    tokenizer = get_protCLM_tokenizer(preset)
    return model, tokenizer

def get_protCLM_for_training(
    preset: str,
    tokenwise: bool = False,
    num_labels: int = None,
    hybrid: bool = False
    ):
    model_path = presets[preset]
    if hybrid:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval()
    else:
        if tokenwise:
            model = AutoModelForTokenClassification.from_pretrained(
                model_path, num_labels=num_labels, trust_remote_code=True
            ).eval()
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=num_labels, trust_remote_code=True
            ).eval()
    tokenizer = get_protCLM_tokenizer(preset)
    return model, tokenizer

if __name__ == "__main__":
    # py -m src.protify.base_models.protCLM
    model, tokenizer = build_protCLM("ProtCLM-1b")
    print(model)
    print(tokenizer)
    print(tokenizer("MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL"))
