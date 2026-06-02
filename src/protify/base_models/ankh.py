import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict, Any
from transformers import T5EncoderModel, AutoTokenizer, T5ForConditionalGeneration

from .base_tokenizer import BaseSequenceTokenizer
from .t5 import T5ForSequenceClassification, T5ForTokenClassification
from .utils import select_hidden_state


presets = {
    'ANKH-Base': 'Synthyra/ANKH_base',
    'ANKH-Large': 'Synthyra/ANKH_large',
    'ANKH2-Large': 'Synthyra/ANKH2_large',
}


class ANKHTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer):
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


class AnkhForEmbedding(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype = None):
        super().__init__()
        self.plm = T5EncoderModel.from_pretrained(model_path, dtype=dtype)

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


class AnkhForProteinGym(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype = None):
        super().__init__()
        self.plm = T5ForConditionalGeneration.from_pretrained(model_path, dtype=dtype)

    @torch.no_grad()
    def position_log_probs(
        self,
        seq: str,
        pos: int,
        tokenizer: Any,
        device: Optional[torch.device] = None,
        sentinel: str = "<extra_id_0>",
    ) -> torch.Tensor:
        """
        Compute log-probs over the vocab for the single position `pos` in `seq`
        using T5-style span corruption:

        - Encoder input: replace seq[pos] with <extra_id_0>
        - Decoder input: start with <extra_id_0>
        - The logits at the last decoder position correspond to the first token of the span,
          i.e., the masked residue distribution.

        Returns: tensor of shape [vocab_size] (log-probs).
        """
        assert 0 <= pos < len(seq), f"pos {pos} out of range for len={len(seq)}"

        # Resolve device
        if device is None:
            device = next(self.parameters()).device

        # Build encoder ids = tokenized left + sentinel + tokenized right (no spaces).
        left, right = seq[:pos], seq[pos+1:]
        if left:
            left_ids = tokenizer(left, add_special_tokens=False)["input_ids"][0].tolist()
        else:
            left_ids = []
        if right:
            right_ids = tokenizer(right, add_special_tokens=False)["input_ids"][0].tolist()
        else:
            right_ids = []

        sent_id = tokenizer.convert_tokens_to_ids(sentinel)
        if sent_id is None:
            raise ValueError(f"Sentinel token {sentinel} not found in tokenizer.")

        enc_ids = torch.tensor([left_ids + [sent_id] + right_ids], dtype=torch.long, device=device)
        enc_mask = torch.ones_like(enc_ids, device=device)

        # Decoder primed with the SAME sentinel; the next token distribution is what we want.
        dec_ids = torch.tensor([[sent_id]], dtype=torch.long, device=device)

        out = self(
            input_ids=enc_ids,
            attention_mask=enc_mask,
            decoder_input_ids=dec_ids,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
        )
        logits = out.logits  # [1, 1, vocab]
        log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
        return log_probs


def get_ankh_tokenizer(preset: str, model_path: str = None):
    return ANKHTokenizerWrapper(AutoTokenizer.from_pretrained('Synthyra/ANKH_base'))


def build_ankh_model(preset: str, masked_lm: bool = False, dtype: torch.dtype = None, model_path: str = None, **kwargs):
    model_path = model_path or presets[preset]
    if masked_lm:
        model = T5ForConditionalGeneration.from_pretrained(model_path, dtype=dtype).eval()
    else:
        model = AnkhForEmbedding(model_path, dtype=dtype).eval()
    tokenizer = get_ankh_tokenizer(preset)
    return model, tokenizer


def get_ankh_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False, dtype: torch.dtype = None, model_path: str = None):
    model_path = model_path or presets[preset]
    if hybrid:
        model = T5EncoderModel.from_pretrained(model_path, dtype=dtype).eval()
    else:
        if tokenwise:
            model = T5ForTokenClassification.from_pretrained(model_path, num_labels=num_labels, dtype=dtype).eval()
        else:
            model = T5ForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, dtype=dtype).eval()
    tokenizer = get_ankh_tokenizer(preset)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.ankh
    model, tokenizer = build_ankh_model('ANKH-Base')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))
