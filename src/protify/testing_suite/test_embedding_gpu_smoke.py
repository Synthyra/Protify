"""Embedding real checkpoints on a GPU writes one feature store per model, with one pooled row per sequence.

These load published weights, so they carry the `gpu` and `slow` markers, which the CPU suite deselects;
`ws test --projects property_lowcode --include gpu --include slow` runs them on a machine with the weights.
"""

import pytest
import torch

from pathlib import Path

from protify.embedder import Embedder, EmbeddingArguments


pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _arguments(save_dir: Path, hidden_state_index: int, batch_size: int) -> EmbeddingArguments:
    return EmbeddingArguments(
        embedding_batch_size=batch_size,
        embedding_num_workers=0,
        matrix_embed=False,
        embedding_pooling_types=["mean"],
        save_embeddings=True,
        embed_dtype=torch.bfloat16,
        model_dtype=torch.bfloat16,
        sql=False,
        embedding_save_dir=str(save_dir),
        padding="longest",
        max_length=64,
        embedding_hidden_state_index=hidden_state_index,
        autocast=True,
    )


def _check_store(arguments: EmbeddingArguments, model_name: str, sequences: list[str], embeddings: dict) -> None:
    assert Path(arguments.feature_directory(model_name)).is_dir()
    assert set(embeddings) == set(sequences)
    for sequence in sequences:
        embedding = embeddings[sequence]  # (d,)
        assert embedding.ndim == 1
        assert embedding.numel() > 0


@pytest.mark.parametrize(
    "model_name,hidden_state_index",
    [
        ("ESM2-8", -1),
        ("ESM2-8", 1),
        ("ESMC-300", -1),
        ("ESMC-300", 1),
    ],
)
def test_gpu_embedding_hidden_state_smoke(tmp_path: Path, model_name: str, hidden_state_index: int) -> None:
    assert torch.cuda.is_available(), "GPU smoke requires CUDA."
    sequences = ["MKTAYIAKQRQISFVKSHFSRQ", "GASGDLV"]
    arguments = _arguments(tmp_path, hidden_state_index, batch_size=2)

    embeddings = Embedder(arguments, sequences)(model_name)  # sequence -> (d,)

    _check_store(arguments, model_name, sequences, embeddings)


def test_gpu_embedding_esmc_6b_smoke(tmp_path: Path) -> None:
    assert torch.cuda.is_available(), "GPU smoke requires CUDA."
    sequences = ["MKTAYIAKQRQISFVKSHFSRQ"]
    arguments = _arguments(tmp_path, -1, batch_size=1)

    embeddings = Embedder(arguments, sequences)("ESMC-6B")  # sequence -> (d,)

    _check_store(arguments, "ESMC-6B", sequences, embeddings)
