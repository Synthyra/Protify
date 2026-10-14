# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

from __future__ import annotations

import os
import torch
import torch.nn as nn

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from functools import cache
from itertools import product
from pathlib import Path
from typing import Any
from torch import Tensor
from transformers.tokenization_utils import PreTrainedTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}
SPECIAL_TOKENS_MAP = {
    "pad_token": {
        "content": "<pad>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
    "cls_token": {
        "content": "<cls>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
    "eos_token": {
        "content": "<eos>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
    "unk_token": {
        "content": "<unk>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
    "mask_token": {
        "content": "<mask>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
    "null_token": {
        "content": "<null>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
}

# CaLM reads coding DNA: `multimolecule/calm` declares multimolecule's DNA tokenizer, and its
# vocabulary holds the codons over A, C, G, T, and N.
STANDARD_ALPHABET = list("ACGTNRYSWKMBDHV.X*-I")

IUPAC_ALPHABET = list("ACGTNRYSWKMBDHV.")

STREAMLINE_ALPHABET = list("ACGTN")

NUCLEOBASE_ALPHABET = list("ACGT")

ALPHABETS = {
    "standard": STANDARD_ALPHABET,
    "iupac": IUPAC_ALPHABET,
    "streamline": STREAMLINE_ALPHABET,
    "nucleobase": NUCLEOBASE_ALPHABET,
}

VOCAB_MAPPING = {
    "R": "AG",
    "Y": "CT",
    "S": "CG",
    "W": "AT",
    "K": "GT",
    "M": "AC",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
    "X": "ACGT",
}

TOKENIZER_CONFIG = {
    "tokenizer_class": "DnaTokenizer",
    "clean_up_tokenization_spaces": True,
}


def get_alphabet(alphabet: list[str] | str | None = None, nmers: int = 1, **kwargs: Any) -> Alphabet:
    if alphabet is None:
        alphabet = STANDARD_ALPHABET if nmers <= 1 else STREAMLINE_ALPHABET
    elif isinstance(alphabet, str):
        alphabet = ALPHABETS[alphabet]
    return Alphabet(alphabet, nmers=nmers, **kwargs)


def get_vocab_mapping() -> dict[str, str]:
    return VOCAB_MAPPING


def get_special_tokens_map() -> dict[str, dict[str, Any]]:
    return SPECIAL_TOKENS_MAP


def get_tokenizer_config(add_special_tokens: bool = False) -> dict[str, Any]:
    config = TOKENIZER_CONFIG
    if add_special_tokens:
        config.setdefault("added_tokens_decoder", {})
        for i, v in enumerate(SPECIAL_TOKENS_MAP.values()):
            config["added_tokens_decoder"][str(i)] = v  # type: ignore[index]
    return config


class Alphabet:
    prepend_tokens: tuple[str, ...] = ("<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>")
    append_tokens: tuple[str, ...] = ()
    tokens: tuple[str, ...]
    nmers: int

    def __init__(
        self,
        tokens: Sequence[str],
        prepend_tokens: tuple[str, ...] | None = None,
        append_tokens: tuple[str, ...] | None = None,
        nmers: int = 1,
    ) -> None:
        if isinstance(tokens, Alphabet):
            tokens = tokens.tokens
        self.tokens = tuple(tokens)
        if prepend_tokens is not None:
            self.prepend_tokens = tuple(prepend_tokens)
        if append_tokens is not None:
            self.append_tokens = tuple(append_tokens)
        self.nmers = nmers

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return self._vocabulary(self.prepend_tokens, self.tokens, self.nmers, self.append_tokens)

    @staticmethod
    @cache
    def _vocabulary(
        prepend_tokens: tuple[str, ...], tokens: tuple[str, ...], nmers: int, append_tokens: tuple[str, ...]
    ) -> tuple[str, ...]:
        return prepend_tokens + generate_kmer_vocabulary(tokens, nmers) + append_tokens

    def __iter__(self) -> Iterator[str]:
        return iter(self.vocabulary)

    def __len__(self) -> int:
        return len(self.vocabulary)

    def __contains__(self, item: str) -> bool:
        return item in self.vocabulary

    def __repr__(self) -> str:
        repr_parts = [f"Alphabet(tokens={self.tokens}"]
        if self.nmers > 1:
            repr_parts.append(f"nmers={self.nmers}")
        repr_parts.append(f"prepend_tokens={self.prepend_tokens}")
        repr_parts.append(f"append_tokens={self.append_tokens})")
        return ", ".join(repr_parts)


def _merge_extra_special_tokens(
    additional_special_tokens: list[str] | tuple[str, ...] | None,
    kwargs: dict[str, Any],
) -> list[str] | tuple[str, ...] | None:
    if "extra_special_tokens" not in kwargs:
        return additional_special_tokens

    extra_special_tokens = kwargs.pop("extra_special_tokens")
    if additional_special_tokens is None:
        merged_special_tokens = []
    else:
        merged_special_tokens = list(additional_special_tokens)

    if isinstance(extra_special_tokens, dict):
        extra_tokens = list(extra_special_tokens.values())
    elif isinstance(extra_special_tokens, (list, tuple)):
        extra_tokens = list(extra_special_tokens)
    else:
        raise TypeError(
            f"extra_special_tokens must be dict, list, or tuple, got {type(extra_special_tokens).__name__}"
        )

    for token in extra_tokens:
        token_value = token
        if isinstance(token, dict) and "content" in token:
            token_value = token["content"]
        if token_value not in merged_special_tokens:
            merged_special_tokens.append(token_value)
    return merged_special_tokens


def generate_kmer_vocabulary(vocabulary: tuple[str, ...], nmers: int = 1) -> tuple[str, ...]:
    """
    Generates a kmer vocabulary given an original vocabulary and the size of kmer.

    Args:
        vocabulary (List[str]): The original vocabulary.
        nmers (int, defaults to 1): The size of kmer to generate.

    Returns:
        vocabulary (List[str]): The kmer vocabulary.
    """

    if nmers <= 1:
        return vocabulary

    special_tokens, tokens = [], []
    for token in vocabulary:
        if token.startswith("<") or token.startswith("["):
            special_tokens.append(token)
        else:
            tokens.append(token)

    return tuple(special_tokens) + tuple("".join(kmer) for kmer in product(tokens, repeat=nmers))


class Tokenizer(PreTrainedTokenizer):
    """
    Constructs a Base tokenizer.

    Args:
        alphabet: List of tokens or an Alphabet object to use in tokenization.
            Either alphabet or vocab_file must be specified.
        bos_token: A special token representing the beginning of a sequence.
        cls_token: A special token representing the classification token.
        pad_token: A special token representing padding.
        eos_token: A special token representing the end of a sequence.
        sep_token: A special token representing the separator token.
        unk_token: A special token representing unknown tokens.
        mask_token: A special token representing the mask token.
        null_token: A special token representing the null token.
        additional_special_tokens: Additional special tokens to add to the vocabulary.
        do_upper_case: Whether to convert input to uppercase.
        vocab_file: Path to a vocabulary file.
            Either alphabet or vocab_file must be specified.

    Examples:
        >>> from multimolecule.tokenisers import Tokenizer
        >>> tokenizer = Tokenizer(["A", "C", "G", "T", "N"], unk_token="N")
        >>> tokenizer('ACGTN')["input_ids"]
        [0, 1, 2, 3, 4]
        >>> tokenizer('acgtn')["input_ids"]
        [0, 1, 2, 3, 4]
        >>> len(tokenizer)
        5
        >>> tokenizer = Tokenizer(["A", "C", "G", "T", "N"], unk_token="N", do_upper_case=False)
        >>> tokenizer('ACGTN')["input_ids"]
        [0, 1, 2, 3, 4]
        >>> tokenizer('acgtn')["input_ids"]
        [4, 4, 4, 4, 4]
        >>> tokenizer('ACgtN')["input_ids"]
        [0, 1, 4, 4, 4]
        >>> tokenizer = Tokenizer(["<pad>", "<cls>", "A", "C", "G", "T", "N", "<mask>", "<eos>"])
        >>> tokenizer('ACGTN')["input_ids"]
        [1, 2, 3, 4, 5, 6, 8]
        >>> tokenizer('AC<mask>GTN')["input_ids"]
        [1, 2, 3, 7, 4, 5, 6, 8]
        >>> tokenizer(['TATATAT', 'ATCGN'], padding=True)["input_ids"]
        [[1, 5, 2, 5, 2, 5, 2, 5, 8], [1, 2, 5, 3, 4, 6, 8, 0, 0]]
    """

    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = VOCAB_FILES_NAMES
    do_upper_case: bool = True

    def __init__(
        self,
        alphabet: Alphabet | list[str] | None = None,
        bos_token: str | None = ...,  # type: ignore[assignment]
        cls_token: str | None = ...,  # type: ignore[assignment]
        pad_token: str | None = ...,  # type: ignore[assignment]
        eos_token: str | None = ...,  # type: ignore[assignment]
        sep_token: str | None = ...,  # type: ignore[assignment]
        unk_token: str | None = ...,  # type: ignore[assignment]
        mask_token: str | None = ...,  # type: ignore[assignment]
        null_token: str | None = ...,  # type: ignore[assignment]
        additional_special_tokens: list[str] | tuple[str, ...] | None = None,
        do_upper_case: bool = True,
        vocab_file: str | None = None,
        **kwargs: Any,
    ) -> None:
        if alphabet is None and vocab_file is None:
            raise ValueError("You must specify either alphabet or vocab_file")

        if vocab_file is not None:
            alphabet = self.load_vocabulary(vocab_file)

        self._id_to_token: OrderedDict[int, str] = OrderedDict(enumerate(alphabet))
        self._token_to_id: OrderedDict[str, int] = OrderedDict({tok: ind for ind, tok in enumerate(alphabet)})

        if cls_token is ...:
            cls_token = self.identify_special_token(alphabet, "cls")
        if bos_token is ...:
            bos_token = cls_token
        if pad_token is ...:
            pad_token = self.identify_special_token(alphabet, "pad")
        if eos_token is ...:
            eos_token = self.identify_special_token(alphabet, "eos")
        if sep_token is ...:
            sep_token = self.identify_special_token(alphabet, "sep") or self.identify_special_token(alphabet, "eos")
        if unk_token is ...:
            unk_token = self.identify_special_token(alphabet, "unk")
        if mask_token is ...:
            mask_token = self.identify_special_token(alphabet, "mask")
        if null_token is ...:
            null_token = self.identify_special_token(alphabet, "null")
        additional_special_tokens = _merge_extra_special_tokens(additional_special_tokens, kwargs)
        if additional_special_tokens is None:
            additional_special_tokens = []
        if null_token in alphabet and null_token not in additional_special_tokens:  # type: ignore[operator]
            additional_special_tokens = list(additional_special_tokens)
            additional_special_tokens.append(null_token)

        super().__init__(
            bos_token=bos_token,
            cls_token=cls_token,
            pad_token=pad_token,
            eos_token=eos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        self.do_upper_case = do_upper_case
        self._id_to_token.update(self.added_tokens_decoder)
        self._token_to_id.update(self.added_tokens_encoder)

        # CaLM tokenization is character-level, so vocab tokens do not need trie-based splitting.

    def _tokenize(self, text: str, **kwargs: Any) -> list[str]:
        if self.do_upper_case:
            text = text.upper()
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        id = self._token_to_id.get(token, self.unk_token_id)
        if id is None:
            raise ValueError(f"Token {token} is not in the vocabulary, and no UNK token is set!")
        return id

    def _convert_id_to_token(self, index: int) -> str:
        token = self._id_to_token.get(index, self.unk_token)
        if token is None:
            raise ValueError(f"ID {index} is not in the vocabulary, and no UNK token is set!")
        return token

    def token_to_id(self, token: str) -> int:
        return self._convert_token_to_id(token)

    def id_to_token(self, index: int) -> str:
        return self._convert_id_to_token(index)

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        bos = [self.bos_token_id]  # points to cls
        sep = [self.sep_token_id]  # points to eos
        eos = [self.eos_token_id]  # eos is eos
        if token_ids_1 is None:
            if self.bos_token_id is None:
                if self.eos_token_id is None:
                    return token_ids_0
                return token_ids_0 + eos
            if self.eos_token_id is None:
                return bos + token_ids_0
            return bos + token_ids_0 + eos
        if self.eos_token_id is None:
            raise ValueError("Cannot tokenize multiple sequences when EOS token is not set!")
        return bos + token_ids_0 + sep + token_ids_1 + eos

    def get_special_tokens_mask(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None, already_has_special_tokens: bool = False
    ) -> list[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            return [1 if token in self.all_special_ids else 0 for token in token_ids_0]
        mask = [0] * len(token_ids_0)
        if self.bos_token_id is not None:
            mask = [1] + mask
        if self.sep_token_id is not None:
            mask += [1]
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1)
            if self.eos_token_id is not None:
                mask += [1]
        return mask

    @staticmethod
    def load_vocabulary(vocab_file: str | Path) -> list[str]:
        with open(vocab_file, encoding="utf-8") as reader:
            vocabulary = reader.read().splitlines()
        return vocabulary

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt")
        with open(vocab_file, "w") as f:
            f.write("\n".join(self.all_tokens))
        return (vocab_file,)

    @staticmethod
    def identify_special_token(alphabet: Alphabet | list[str], token: str) -> str | None:
        tokens = [i for i in alphabet if token in i.lower()]
        if len(tokens) == 1:
            return tokens[0]
        if len(tokens) == 0:
            return None
        raise ValueError(f"Token {token} is ambiguous, could be {tokens}")

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab, **self.added_tokens_encoder)

    @property
    def vocab(self) -> OrderedDict[str, int]:
        return self._token_to_id.copy()

    @property
    def all_tokens(self) -> list[str]:
        return list(self.get_vocab().keys())

    @property
    def vocab_size(self) -> int:
        return len(self.all_tokens)


class DnaTokenizer(Tokenizer):
    """
    Tokenizer for DNA sequences, multimolecule's `DnaTokenizer`, which `multimolecule/calm` declares.

    Args:
        alphabet: alphabet to use for tokenization.

            - If is `None`, the standard DNA alphabet will be used.
            - If is a `string`, it should correspond to the name of a predefined alphabet. The options include
                + `standard`
                + `iupac`
                + `streamline`
                + `nucleobase`
            - If is an alphabet or a list of characters, that specific alphabet will be used.
        nmers: Size of kmer to tokenize.
        codon: Whether to tokenize into codons.
        replace_U_with_T: Whether to replace U with T, so an mRNA reads as its cDNA.
        do_upper_case: Whether to convert input to uppercase.

    Examples:
        >>> tokenizer = DnaTokenizer()
        >>> tokenizer('atgtggtaa')["input_ids"]
        [1, 23, 93, 81, 2]
        >>> tokenizer('auguggtaa')["input_ids"]
        [1, 23, 93, 81, 2]
        >>> tokenizer('atgtggta')["input_ids"]
        Traceback (most recent call last):
        ValueError: length of input sequence must be a multiple of 3 for codon tokenization, but got 8
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        alphabet: Alphabet | str | list[str] | None = None,
        nmers: int = 1,
        codon: bool = True,
        replace_U_with_T: bool = True,  # noqa: naming serialized tokenizer config key
        do_upper_case: bool = True,
        additional_special_tokens: list[str] | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        if codon and (nmers > 1 and nmers != 3):
            raise ValueError("Codon and nmers cannot be used together.")
        if codon:
            nmers = 3  # set to 3 to get correct vocab
        if not isinstance(alphabet, Alphabet):
            alphabet = get_alphabet(alphabet, nmers=nmers)
        additional_special_tokens = _merge_extra_special_tokens(additional_special_tokens, kwargs)
        super().__init__(
            alphabet=alphabet,
            nmers=nmers,
            codon=codon,
            replace_U_with_T=replace_U_with_T,
            do_upper_case=do_upper_case,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        self.replace_U_with_T = replace_U_with_T
        self.nmers = nmers
        self.codon = codon

    def _tokenize(self, text: str, **kwargs: Any) -> list[str]:
        if self.do_upper_case:
            text = text.upper()
        if self.replace_U_with_T:
            text = text.replace("U", "T")
        if self.codon:
            if len(text) % 3 != 0:
                raise ValueError(
                    f"length of input sequence must be a multiple of 3 for codon tokenization, but got {len(text)}"
                )
            return [text[i : i + 3] for i in range(0, len(text), 3)]
        if self.nmers > 1:
            return [text[i : i + self.nmers] for i in range(len(text) - self.nmers + 1)]  # noqa: E203
        return list(text)


class RotaryEmbedding(nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer).

    Query and keys are transformed by rotation
    matrices which depend on their relative positions.

    Each rotation turns an adjacent channel pair, 2i and 2i + 1, as the original CaLM does and as
    multimolecule's `RotaryEmbedding(interleaved=True)` does for CaLM. Turning channel halves, i
    and i + d_h / 2, is a different position encoding, under which CaLM's weights give other outputs.

    Tip: **Cache**
        The inverse frequency buffer is cached and updated only when the sequence length changes or the device changes.

    Success: **Sequence Length**
        Rotary Embedding is irrespective of the sequence length and can be used for any sequence length.
        Use the `scale` parameter to extend context length beyond training (e.g., scale=2.0 doubles effective context).

    Example:
        >>> embedding = RotaryEmbedding(embedding_dim=64)
        >>> query, key = torch.randn(2, 4, 28, 64), torch.randn(2, 4, 28, 64)
        >>> query, key = embedding(query, key)
        >>> query.shape
        torch.Size([2, 4, 28, 64])
        >>> # For extended context length
        >>> embedding_extended = RotaryEmbedding(embedding_dim=64, scale=2.0)
        >>> embedding.state_dict()  # no weight in state_dict
        OrderedDict()
    """

    _seq_len_cached: int | None = None
    _cos_cached: Tensor | None = None
    _sin_cached: Tensor | None = None

    def __init__(
        self,
        embedding_dim: int,
        base: float = 10000.0,
        scale: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize rotary position embeddings.

        Args:
            embedding_dim: Dimension of the embeddings (must be even)
            base: Base for computing inverse frequencies. Defaults to 10000.0.
            scale: Scaling factor for frequencies. Values > 1.0 extend context length
                   (e.g., scale=2.0 doubles the effective context). Defaults to 1.0.
            dtype: Data type for computations. Defaults to torch.float32.
        """
        super().__init__()
        self.embedding_dim = embedding_dim  # d_h
        self.base = base
        self.frequency_dtype = dtype
        self.register_buffer("inv_freq", self.compute_inv_freq(), persistent=False)  # (d_h / 2,)
        self.scale = scale

    def compute_inv_freq(self, device: torch.device | None = None) -> Tensor:
        """
        Compute the inverse frequency of each rotated channel pair.

        No checkpoint holds `inv_freq`, so `CaLmPreTrainedModel._init_weights` refills it from here
        after `from_pretrained`.
        """
        channels = torch.arange(0, self.embedding_dim, 2, device=device, dtype=self.frequency_dtype)  # (d_h / 2,)
        return 1.0 / (self.base ** (channels / self.embedding_dim))  # (d_h / 2,)

    def forward(self, q: Tensor, k: Tensor, offset: int = 0, seq_length: int | None = None) -> tuple[Tensor, Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.

        Args:
            q: Query tensor of shape `(batch_size, num_heads, seq_length, embedding_dim)`
            k: Key tensor of shape `(batch_size, num_heads, seq_length, embedding_dim)`
            offset: Position offset for the start of the sequence (used with past_key_values).
                    Defaults to 0.
            seq_length: Full sequence length including offset. If None, uses the sequence length
                    from the input tensors. Required when offset > 0.

        Returns:
            Tuple of (rotated_query, rotated_key) tensors with the same shapes as inputs.
        """
        # q: (b, h, l, d_h), k: (b, h, l, d_h)
        if offset > 0 and seq_length is None:
            raise ValueError("seq_length must be provided when offset > 0")

        if seq_length is None:
            seq_length = k.shape[-2]

        self._update_cos_sin_tables(k, seq_len_dim=-2, seq_length=seq_length)
        return self.apply_rotary_pos_emb(q, offset=offset), self.apply_rotary_pos_emb(k, offset=offset)  # (b, h, l, d_h), (b, h, l, d_h)

    def _update_cos_sin_tables(
        self, x: Tensor, seq_len_dim: int = 2, seq_length: int | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        Update cached cos/sin tables for rotary embeddings.

        Args:
            x: Input tensor to determine device and dtype
            seq_len_dim: Dimension containing sequence length (default: -2)
            seq_length: Full sequence length to cache. If None, uses x.shape[seq_len_dim]
        """
        # x: (b, h, l, d_h)
        if seq_length is None:
            seq_length = x.shape[seq_len_dim]

        if seq_length != self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_length
            inv_freq = self.inv_freq
            if not isinstance(inv_freq, Tensor):
                raise RuntimeError("inv_freq buffer is not a Tensor")
            t = torch.arange(seq_length, device=x.device, dtype=inv_freq.dtype)  # (l,)
            # Apply scaling: divide frequencies by scale to extend context length
            freqs = torch.outer(t, inv_freq) / self.scale  # (l, d_h / 2)
            # Both channels of a pair turn by the pair's angle.
            emb = freqs.repeat_interleave(2, dim=-1).to(x.device)  # (l, d_h)
            self._cos_cached = emb.cos()[None, None, :, :]  # (1, 1, l, d_h)
            self._sin_cached = emb.sin()[None, None, :, :]  # (1, 1, l, d_h)
        # At this point, _cos_cached and _sin_cached are guaranteed to be Tensor
        assert self._cos_cached is not None and self._sin_cached is not None
        return self._cos_cached, self._sin_cached  # (1, 1, l, d_h), (1, 1, l, d_h)

    def apply_rotary_pos_emb(self, x: Tensor, offset: int = 0) -> Tensor:
        """
        Apply rotary position embeddings to a tensor.

        Args:
            x: Input tensor of shape `(batch_size, num_heads, seq_length, embedding_dim)`
            offset: Position offset for the start of the sequence (used with past_key_values).
                    Defaults to 0.

        Returns:
            Rotated tensor with the same shape as input.
        """
        if self._cos_cached is None or self._sin_cached is None:
            raise RuntimeError("Cos/sin tables not initialized. Call forward() or _update_cos_sin_tables() first.")

        # x: (b, h, l, d_h)
        cos = self._cos_cached[:, :, offset : offset + x.shape[-2], :]  # (1, 1, l, d_h)
        sin = self._sin_cached[:, :, offset : offset + x.shape[-2], :]  # (1, 1, l, d_h)
        return (x * cos) + (self.rotate_half(x) * sin)  # (b, h, l, d_h)

    @staticmethod
    def rotate_half(x: Tensor) -> Tensor:
        """Turn each channel pair (x_2i, x_2i+1) a quarter turn, to (-x_2i+1, x_2i)."""
        # x: (..., d_h)
        return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)  # (..., d_h)
