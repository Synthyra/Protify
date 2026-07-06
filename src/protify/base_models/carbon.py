import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Dict
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from .base_tokenizer import BaseSequenceTokenizer
from .utils import select_hidden_state


presets = {
    "CARBON-500M": "HuggingFaceBio/Carbon-500M",
    "CARBON-3B": "HuggingFaceBio/Carbon-3B",
    "CARBON-8B": "HuggingFaceBio/Carbon-8B",
}


DNA_OPEN_TOKEN = "<dna>"
DNA_CLOSE_TOKEN = "</dna>"


class CarbonTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)
        # Right padding keeps the '</dna>' separator as the last real token, which
        # is requred for 'eos' pooling
        self.tokenizer.padding_side = "right"
        if DNA_OPEN_TOKEN not in self.tokenizer.get_vocab():
            print(f"Warning: '{DNA_OPEN_TOKEN}' not in CARBON tokenizer vocab; DNA mode may not engage.")

    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", "max_length")
        kwargs.setdefault("truncation", True)
        kwargs["add_special_tokens"] = False
        # CARBON expects 6-mers inside <dna>...</dna>. Trim long sequences at the
        # character level (keeping the tail, ~6 nt/token, reserving 2 tokens for the
        # tags) so token truncation never drops the trailing '</dna>', then wrap.
        max_length = kwargs.get("max_length")
        if max_length is not None and max_length > 2:
            char_budget = (max_length - 2) * 6
            sequences = [seq[-char_budget:] for seq in sequences]
        wrapped = [
            f"{DNA_OPEN_TOKEN}{seq[: (len(seq) // 6) * 6]}{DNA_CLOSE_TOKEN}"
            for seq in sequences
        ]
        return self.tokenizer(wrapped, **kwargs)


class CarbonForEmbedding(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype = None):
        super().__init__()
        self.carbon = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False,
        hidden_state_index: int = -1,
        **kwargs,
    ) -> torch.Tensor:
        assert not output_attentions, (
            "CARBON does not return attentions; 'parti' pooling is unsupported."
        )
        out = self.carbon(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return select_hidden_state(out.hidden_states[-1], out.hidden_states, hidden_state_index)


def get_carbon_tokenizer(preset: str, model_path: str = None) -> BaseSequenceTokenizer:
    return CarbonTokenizerWrapper(
        AutoTokenizer.from_pretrained(model_path or presets[preset], trust_remote_code=True)
    )


def build_carbon_model(
    preset: str,
    masked_lm: bool = False,
    dtype: torch.dtype = None,
    model_path: str = None,
    **kwargs,
) -> Tuple[nn.Module, BaseSequenceTokenizer]:
    if masked_lm:
        raise ValueError(f"Model {preset} does not support masked language modeling")
    model_path = model_path or presets[preset]
    model = CarbonForEmbedding(model_path, dtype=dtype).eval()
    tokenizer = get_carbon_tokenizer(preset, model_path=model_path)
    return model, tokenizer


def get_carbon_for_training(
    preset: str,
    tokenwise: bool = False,
    num_labels: int = None,
    hybrid: bool = False,
    dtype: torch.dtype = None,
    model_path: str = None,
):
    if tokenwise:
        raise NotImplementedError(
            "CARBON uses 6-mer DNA tokens (one token spans 6 nt), so per-token labels do "
            "not align; only sequence-level fine-tuning is supported."
        )
    model_path = model_path or presets[preset]
    tokenizer = get_carbon_tokenizer(preset, model_path=model_path)
    if hybrid:
        # CARBON is a stock Llama (config model_type "llama", no auto_map). AutoModel
        # returns the bare LlamaModel whose last_hidden_state HybridProbe.forward reads.
        model = AutoModel.from_pretrained(model_path, dtype=dtype, trust_remote_code=True).eval()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, dtype=dtype, trust_remote_code=True, num_labels=num_labels
        ).eval()
        # LlamaForSequenceClassification finds the classification token via pad_token_id,
        # which CARBON leaves unset (head fails for batch > 1). Copy it from the tokenizer,
        # so the last non-pad token is the '</dna>' separator.
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


if __name__ == "__main__":
    # py -m src.protify.base_models.carbon
    model, tokenizer = build_carbon_model("CARBON-500M")
    print(model)
    print(tokenizer)
    print(tokenizer("ATCG" * 30))
