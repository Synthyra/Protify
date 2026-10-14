import re
import torch

from typing import Any


class BaseSequenceTokenizer:
    def __init__(self, tokenizer: Any) -> None:
        if tokenizer is None:
            raise ValueError("Tokenizer cannot be None.")
        self.tokenizer = tokenizer

    def __call__(self, sequences: str | list[str], **kwargs: Any) -> Any:
        # Match Protify's batched embedding defaults unless the caller overrides them.
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", "max_length")
        kwargs.setdefault("truncation", True)
        kwargs.setdefault("add_special_tokens", True)

        token_batch = self.tokenizer(sequences, **kwargs)
        # token_batch tensor fields: (b, ...); trailing dimensions depend on the tokenizer options.
        return token_batch

    @property
    def vocab_size(self) -> Any:
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> Any:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> Any:
        return self.tokenizer.eos_token_id

    @property
    def cls_token_id(self) -> Any:
        return self.tokenizer.cls_token_id

    @property
    def mask_token_id(self) -> Any:
        return self.tokenizer.mask_token_id

    @property
    def convert_tokens_to_ids(self) -> Any:
        return self.tokenizer.convert_tokens_to_ids

    def save_pretrained(self, save_dir: str) -> None:
        self.tokenizer.save_pretrained(save_dir)


class BatchSequenceTokenizer(BaseSequenceTokenizer):
    """Reads a lone sequence as a batch of one, then tokenizes with Protify's batched defaults."""

    def __call__(self, sequences: str | list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        token_batch = super().__call__(sequences, **kwargs)
        return token_batch  # default tensor fields: (b, l)


class ProtTransTokenizerWrapper(BaseSequenceTokenizer):
    """The ProtTrans models' input: residues separated by spaces, with U, Z, O, and B read as X."""

    def __call__(self, sequences: str | list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", "max_length")
        kwargs.setdefault("truncation", True)
        kwargs.setdefault("add_special_tokens", True)
        sequences = [re.sub(r"[UZOB]", "X", seq) for seq in sequences]
        sequences = [" ".join(seq) for seq in sequences]
        token_batch = self.tokenizer(sequences, **kwargs)
        return token_batch  # default tensor fields: (b, l)
