import torch
from typing import List, Dict, Union


class BaseSequenceTokenizer:
    def __init__(self, tokenizer):
        if tokenizer is None:
            raise ValueError("Tokenizer cannot be None.")
        self.tokenizer = tokenizer

    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        # Default tokenizer args if not provided
        kwargs.setdefault('return_tensors', 'pt')
        kwargs.setdefault('padding', 'max_length')
        kwargs.setdefault('truncation', True)
        kwargs.setdefault('add_special_tokens', True)

        return self.tokenizer(sequences, **kwargs)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self):
        return getattr(self.tokenizer, 'pad_token_id')

    @property
    def eos_token_id(self):
        return getattr(self.tokenizer, 'eos_token_id')

    @property
    def cls_token_id(self):
        return getattr(self.tokenizer, 'cls_token_id')

    @property
    def mask_token_id(self):
        return getattr(self.tokenizer, 'mask_token_id')
    
    @property
    def convert_tokens_to_ids(self):
        return getattr(self.tokenizer, 'convert_tokens_to_ids')
    
    def save_pretrained(self, save_dir: str):
        self.tokenizer.save_pretrained(save_dir)