"""Why the Embedder's loop does not call FastPLMs `embed_dataset`: what the two return differs.

Both run on the same seeded stand-in model, which implements the `_embed` call `embed_dataset` makes and the
`forward` call the Embedder makes. The Embedder keeps CLS and EOS in every pooled value and in every matrix row,
so a sequence of l residues gives l + 2 rows. `embed_dataset` masks to the biological residues, so it gives l rows
and pools over them. Dropping the two special rows from the Embedder's matrix rows gives exactly `embed_dataset`'s
rows, so the special tokens are the whole difference, and pooled values only agree where a special row changes
neither the mean nor the maximum.
"""

import pytest
import torch

from fastplms.embeddings.runner import embed_dataset
from protify.embedder import Embedder, EmbeddingArguments


pytestmark = [
    pytest.mark.filterwarnings("ignore:Downloading embeddings is recommended"),
    pytest.mark.filterwarnings("ignore:'pin_memory' argument is set as true"),
]

SEQUENCES = ["MKTAYIAKQR", "MKTAYIAKQRQISFVKSHFSRQ", "GGGGSGGGGS", "ACDEFGHIKL"]
WIDTH = 4


class Tokenizer:
    """Residues to ids between a CLS id 1 and an EOS id 2, padded with 0 as a Hugging Face tokenizer pads."""

    pooling_token_id = None
    all_special_ids = (0, 1, 2)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def __call__(self, sequences, return_tensors="pt", padding="longest", truncation=True, max_length=64,
                 pad_to_multiple_of=None, **_ignored):
        encoded = [[1, *(ord(residue) for residue in sequence[: max_length - 2]), 2] for sequence in sequences]
        width = max_length if padding == "max_length" else max(len(tokens) for tokens in encoded)
        if pad_to_multiple_of:
            width = -(-width // pad_to_multiple_of) * pad_to_multiple_of
        input_ids = torch.zeros(len(encoded), width, dtype=torch.long)  # (b, l)
        attention_mask = torch.zeros(len(encoded), width, dtype=torch.long)  # (b, l)
        for row, tokens in enumerate(encoded):
            input_ids[row, : len(tokens)] = torch.tensor(tokens)
            attention_mask[row, : len(tokens)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class Model(torch.nn.Module):
    """A seeded embedding table, reachable by `_embed` as FastPLMs calls it and by `forward` as Protify calls it."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.table = torch.nn.Parameter(torch.randn(128, WIDTH, generator=generator), requires_grad=False)

    def _embed(self, input_ids, attention_mask=None, **_ignored):
        # input_ids: (b, l); attention_mask: (b, l)
        return self.table[input_ids]  # (b, l, d)

    def forward(self, input_ids, attention_mask=None, **_ignored):
        # input_ids: (b, l); attention_mask: (b, l)
        return self._embed(input_ids, attention_mask)  # (b, l, d)


@pytest.fixture(autouse=True)
def on_the_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def protify_rows(root, pooling, matrix_embed):
    arguments = EmbeddingArguments(
        embedding_save_dir=str(root), embedding_pooling_types=[pooling], matrix_embed=matrix_embed,
        padding="longest", max_length=64, embedding_batch_size=2, save_embeddings=False, embed_dtype=torch.float32,
    )
    return Embedder(arguments, SEQUENCES)._embed_sequences(list(SEQUENCES), str(root), Model().eval(), Tokenizer(), {})


def fastplms_rows(pooling, matrix_embed):
    embedded = embed_dataset(
        Model().eval(), SEQUENCES, tokenizer=Tokenizer(), batch_size=2, max_length=64, dtype=torch.float32,
        pooling=None if matrix_embed else pooling, full_embeddings=matrix_embed,
    )
    return {record.sequence: record.tensor for record in embedded.records}


def test_matrix_rows_differ_by_exactly_the_cls_and_eos_rows(tmp_path):
    ours = protify_rows(tmp_path, "mean", matrix_embed=True)
    theirs = fastplms_rows("mean", matrix_embed=True)
    for sequence in SEQUENCES:
        assert ours[sequence].shape == (len(sequence) + 2, WIDTH)  # (l + 2, d)
        assert theirs[sequence].shape == (len(sequence), WIDTH)  # (l, d)
        torch.testing.assert_close(ours[sequence][1:-1], theirs[sequence], rtol=0, atol=0)


def test_pooled_mean_differs_because_cls_and_eos_are_pooled(tmp_path):
    ours = protify_rows(tmp_path, "mean", matrix_embed=False)
    theirs = fastplms_rows("mean", matrix_embed=False)
    for sequence in SEQUENCES:
        assert not torch.allclose(ours[sequence], theirs[sequence])  # (d,) each
    matrix = protify_rows(tmp_path, "mean", matrix_embed=True)
    for sequence in SEQUENCES:
        torch.testing.assert_close(ours[sequence], matrix[sequence].mean(dim=0))  # (d,)
        torch.testing.assert_close(theirs[sequence], matrix[sequence][1:-1].mean(dim=0))  # (d,)


def test_pooled_max_agrees_only_where_a_special_row_is_not_the_maximum(tmp_path):
    ours = protify_rows(tmp_path, "max", matrix_embed=False)
    theirs = fastplms_rows("max", matrix_embed=False)
    agreeing = [sequence for sequence in SEQUENCES if torch.equal(ours[sequence], theirs[sequence])]
    assert 0 < len(agreeing) < len(SEQUENCES)
