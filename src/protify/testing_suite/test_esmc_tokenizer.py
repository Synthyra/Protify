"""The ESMC adapter's tokenizer is FastPLMs' `EsmSequenceTokenizer`, and gives the ids Protify's own copy gave.

Until 2026-10-06 `base_models/esmc_utils.py` held a second copy of the tokenizer that nothing imported. The ids,
masks, and special tokens below were recorded from that copy, so the retired file can stay retired.

Shapes: `b` sequences, `l` tokens in the longest row.
"""

try:
    from src.protify.base_models.esmc import get_esmc_tokenizer
except ImportError:
    from protify.base_models.esmc import get_esmc_tokenizer


CLS = 0
PAD = 1
EOS = 2
UNK = 3
CHAIN_BREAK = 31
MASK = 32
SEQUENCES = ["MKTAYIAKQR", "MK", "ACDEFGHIKLMNPQRSTVWY|MKV", "XBUZO.-"]
WIDTH = 26  # l, the tokens of the chain-break sequence: its 24 characters between <cls> and <eos>
ROWS = [  # the ids of each sequence before padding
    [CLS, 20, 15, 11, 5, 19, 12, 5, 15, 16, 10, EOS],
    [CLS, 20, 15, EOS],
    [CLS, 5, 23, 13, 9, 18, 6, 21, 12, 15, 4, 20, 17, 14, 16, 10, 8, 11, 7, 22, 19, CHAIN_BREAK, 20, 15, 7, EOS],
    [CLS, 24, 25, 26, 27, 28, 29, 30, EOS],
]


def test_the_adapter_builds_fastplms_tokenizer():
    tokenizer = get_esmc_tokenizer("ESMC-300").tokenizer

    assert type(tokenizer).__name__ == "EsmSequenceTokenizer"
    assert type(tokenizer).__module__.startswith("fastplms.")


def test_a_padded_batch_has_the_ids_and_mask_the_old_copy_gave():
    tokenizer = get_esmc_tokenizer("ESMC-300").tokenizer

    batch = tokenizer(SEQUENCES, padding=True, return_tensors="pt")

    assert batch["input_ids"].shape == (4, WIDTH)  # (b, l)
    assert batch["input_ids"].tolist() == [row + [PAD] * (WIDTH - len(row)) for row in ROWS]
    assert batch["attention_mask"].tolist() == [[1] * len(row) + [0] * (WIDTH - len(row)) for row in ROWS]
    assert sorted(batch) == ["attention_mask", "input_ids"]


def test_a_pair_of_sequences_is_closed_with_an_eos_after_each():
    tokenizer = get_esmc_tokenizer("ESMC-300").tokenizer

    batch = tokenizer("MKT", "ACD", return_tensors="pt")

    assert batch["input_ids"].tolist() == [[CLS, 20, 15, 11, EOS, 5, 23, 13, EOS]]  # (1, 9)
    assert batch["attention_mask"].tolist() == [[1] * 9]  # (1, 9)


def test_the_special_tokens_and_vocabulary_are_the_ones_the_old_copy_gave():
    tokenizer = get_esmc_tokenizer("ESMC-300").tokenizer

    assert tokenizer.vocab_size == 33
    assert tokenizer.all_token_ids == list(range(33))
    assert (tokenizer.cls_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id) == (CLS, PAD, EOS)
    assert (tokenizer.unk_token_id, tokenizer.mask_token_id) == (UNK, MASK)
    assert tokenizer.bos_token_id == CLS
    assert (tokenizer.chain_break_token, tokenizer.chain_break_token_id) == ("|", CHAIN_BREAK)
    assert sorted(tokenizer.special_token_ids) == [CLS, PAD, EOS, UNK, CHAIN_BREAK, MASK]
    assert list(tokenizer.model_input_names) == ["input_ids", "attention_mask"]
