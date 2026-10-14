"""CaLM's forward pass under Transformers 5, its rotary embedding, and its codon tokenizer.

Transformers 5 removed `get_extended_attention_mask` and `invert_attention_mask` from
`PreTrainedModel`. CaLM's forward pass called the first on every input, so every forward pass
raised, and CaLM now builds both masks itself. Its head mask is FastPLMs' DPLM encoder's, which gives
what Transformers 4 did, except that a mask of another rank is a `ValueError` where Protify's own copy
asserted.

CaLM was trained with rotary embeddings that turn adjacent channel pairs, and
`multimolecule/calm` holds a vocabulary of DNA codons. Protify's copy turned channel halves
instead, and read every T as U, so each codon holding a T became `<unk>`. Either fault changes
every embedding `multimolecule/calm` gives, and `test_calm_checkpoint.py` holds the checkpoint to
its published outputs.

These tests run a tiny random CaLM and a tokenizer built from CaLM's vocabulary on CPU, so they
need neither the Hub nor a GPU.
"""

import itertools
import json
import pytest
import torch

from protify.base_models.calm import CaLmConfig, CaLmModel, get_calm_tokenizer
from protify.base_models.calm_utils import RotaryEmbedding


VOCAB_SIZE = 131  # v, CaLM's codon vocabulary
PAD_TOKEN_ID = 0
MASK_TOKEN_ID = 4
FIRST_CODON_ID = 5  # ids below this are special tokens
LENGTH = 10  # l, tokens per padded row
REAL_LENGTH = 6  # tokens in the shorter row before its padding
HEAD_WIDTH = 8  # d_h
# The tokenizer files `multimolecule/calm` holds: six special tokens, then the 125 codons over
# A, C, G, T, and N, and a configuration that names multimolecule's DNA tokenizer.
CODON_VOCABULARY = ("<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>") + tuple(
    "".join(bases) for bases in itertools.product("ACGTN", repeat=3)
)
CALM_TOKENIZER_CONFIG = {
    "tokenizer_class": "DnaTokenizer",
    "codon": True,
    "nmers": 3,
    "replace_U_with_T": True,
    "bos_token": "<cls>",
    "cls_token": "<cls>",
    "eos_token": "<eos>",
    "sep_token": "<eos>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
    "extra_special_tokens": ["<null>"],
}


def _calm(**overrides) -> CaLmModel:
    config = CaLmConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=64,
        pad_token_id=PAD_TOKEN_ID,
        mask_token_id=MASK_TOKEN_ID,
        **overrides,
    )
    torch.manual_seed(0)
    return CaLmModel(config, add_pooling_layer=False).eval()


def _codons(batch: int, length: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(FIRST_CODON_ID, VOCAB_SIZE, (batch, length), generator=generator)  # (b, l)


def test_a_padded_sequence_encodes_as_it_does_alone():
    model = _calm()
    input_ids = _codons(2, LENGTH)  # (2, l)
    input_ids[1, REAL_LENGTH:] = PAD_TOKEN_ID
    attention_mask = (input_ids != PAD_TOKEN_ID).long()  # (2, l)

    with torch.inference_mode():
        batched = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (2, l, d)
        alone = model(
            input_ids=input_ids[1:, :REAL_LENGTH], attention_mask=attention_mask[1:, :REAL_LENGTH]
        ).last_hidden_state  # (1, l_r, d)

    assert torch.isfinite(batched).all()
    torch.testing.assert_close(batched[1, :REAL_LENGTH], alone[0], rtol=1e-5, atol=1e-6)


def test_the_extended_mask_is_zero_for_real_tokens_and_the_minimum_for_padding():
    model = _calm()
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])  # (2, 4)
    minimum = torch.finfo(model.dtype).min

    extended = model.get_extended_attention_mask(attention_mask, attention_mask.shape)  # (2, 1, 1, 4)

    expected = torch.tensor([[0.0, 0.0, 0.0, minimum], [0.0, 0.0, minimum, minimum]])[:, None, None, :]
    assert extended.dtype == model.dtype
    torch.testing.assert_close(extended, expected, rtol=0, atol=0)


def test_a_cross_attention_mask_is_zero_for_encoder_tokens_and_the_minimum_for_padding():
    model = _calm()
    encoder_mask = torch.tensor([[1, 1, 0]])  # (1, 3)
    minimum = torch.finfo(model.dtype).min

    extended = model.invert_attention_mask(encoder_mask)  # (1, 1, 1, 3)

    torch.testing.assert_close(extended, torch.tensor([[[[0.0, 0.0, minimum]]]]), rtol=0, atol=0)


def test_a_decoder_mask_hides_each_position_from_the_ones_before_it():
    """Each query sees the cached keys, then the current positions up to its own."""
    model = _calm(is_decoder=True)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])  # (1, l_k): two cached keys, then three positions

    extended = model.get_extended_attention_mask(attention_mask, (1, 3))  # (1, 1, 3, 5)

    visible = extended[0, 0] == 0  # (3, 5), query by key
    expected = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, False],  # the last key is padding
        ]
    )
    assert torch.equal(visible, expected)


def test_a_decoder_position_ignores_the_positions_after_it():
    model = _calm(is_decoder=True)
    prefix = _codons(1, REAL_LENGTH, seed=0)  # (1, l_r)
    first = torch.cat([prefix, _codons(1, LENGTH - REAL_LENGTH, seed=1)], dim=1)  # (1, l)
    second = torch.cat([prefix, _codons(1, LENGTH - REAL_LENGTH, seed=2)], dim=1)  # (1, l)
    attention_mask = torch.ones_like(first)  # (1, l)

    with torch.inference_mode():
        first_states = model(input_ids=first, attention_mask=attention_mask).last_hidden_state  # (1, l, d)
        second_states = model(input_ids=second, attention_mask=attention_mask).last_hidden_state  # (1, l, d)

    assert torch.isfinite(first_states).all()
    torch.testing.assert_close(first_states[:, :REAL_LENGTH], second_states[:, :REAL_LENGTH], rtol=0, atol=0)
    assert not torch.equal(first_states[:, REAL_LENGTH:], second_states[:, REAL_LENGTH:])


def test_a_head_mask_is_one_mask_per_layer_in_the_models_dtype():
    """A `(h,)` mask applies to every layer, and an `(n_layers, h)` mask gives each layer its own."""
    model = _calm()
    for_every_layer = torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.float64)  # (h,)
    per_layer = torch.tensor([[0.0, 0.125, 0.25, 0.375], [0.5, 0.625, 0.75, 0.875]])  # (n_layers, h)

    shared = model.get_head_mask(for_every_layer, 2)  # (n_layers, 1, h, 1, 1)
    layered = model.get_head_mask(per_layer, 2)  # (n_layers, 1, h, 1, 1)
    chunked = model.get_head_mask(for_every_layer, 2, is_attention_chunked=True)  # (n_layers, 1, h, 1, 1, 1)

    expected_shared = torch.tensor([0.0, 0.25, 0.5, 0.75]).reshape(1, 1, 4, 1, 1).expand(2, 1, 4, 1, 1)
    assert shared.dtype == model.dtype
    assert torch.equal(shared, expected_shared)
    assert torch.equal(layered, per_layer.reshape(2, 1, 4, 1, 1))
    assert torch.equal(chunked, expected_shared.unsqueeze(-1))


def test_no_head_mask_is_one_none_per_layer():
    assert _calm().get_head_mask(None, 2) == [None, None]


def test_a_head_mask_of_another_rank_is_a_value_error():
    model = _calm()

    for head_mask in (torch.tensor(1.0), torch.ones(2, 2, 2)):
        with pytest.raises(ValueError, match=r"head_mask\.dim"):
            model.get_head_mask(head_mask, 2)


def test_the_head_mask_methods_are_those_of_the_fastplms_dplm_encoder():
    for name in ("get_head_mask", "_convert_head_mask_to_5d"):
        method = getattr(CaLmModel, name)

        assert method.__module__.endswith("fastplms.models.dplm.modeling_dplm"), name
        assert method.__qualname__ == f"FAST_DPLM_ENCODER.{name}", name


def test_rotary_turns_adjacent_channel_pairs():
    """Position p turns channels 2i and 2i + 1 together, by p times the pair's frequency.

    The first pair's frequency is 1, so at position 1 a query holding only channel 0 turns by one
    radian into channel 1. Turning channel halves instead would move it into channel d_h / 2.
    """
    rotary = RotaryEmbedding(embedding_dim=HEAD_WIDTH)
    query = torch.zeros(1, 1, 2, HEAD_WIDTH)  # (b, h, l, d_h), positions 0 and 1
    query[..., 0] = 1.0
    turned = torch.zeros(HEAD_WIDTH)  # (d_h,)
    turned[0], turned[1] = torch.cos(torch.tensor(1.0)), torch.sin(torch.tensor(1.0))

    rotated, _ = rotary(query, query)  # (1, 1, 2, d_h)

    torch.testing.assert_close(rotated[0, 0, 0], query[0, 0, 0])  # position 0 does not turn
    torch.testing.assert_close(rotated[0, 0, 1], turned)


def test_the_tokenizer_reads_each_codon_of_a_coding_sequence(tmp_path):
    """Each codon of a coding sequence gets its own id, from DNA or from the matching mRNA."""
    (tmp_path / "vocab.txt").write_text("\n".join(CODON_VOCABULARY), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(CALM_TOKENIZER_CONFIG), encoding="utf-8")
    tokenizer = get_calm_tokenizer("CaLM", model_path=str(tmp_path)).tokenizer
    coding_sequence = "ATGTGGTTTTAA"  # Met, Trp, Phe, stop: three of the four codons hold a T
    codons = ("<cls>", "ATG", "TGG", "TTT", "TAA", "<eos>")
    expected = [CODON_VOCABULARY.index(codon) for codon in codons]

    assert tokenizer(coding_sequence)["input_ids"] == expected
    assert tokenizer(coding_sequence.replace("T", "U"))["input_ids"] == expected
