import numpy as np
import torch

try:
    from src.protify.data.dataset_classes import (
        EmbeddingStandardizer,
        EmbedsLabelsDataset,
        PairEmbedsLabelsDataset,
    )
except ImportError:
    try:
        from protify.data.dataset_classes import (
            EmbeddingStandardizer,
            EmbedsLabelsDataset,
            PairEmbedsLabelsDataset,
        )
    except ImportError:
        from ..data.dataset_classes import (
            EmbeddingStandardizer,
            EmbedsLabelsDataset,
            PairEmbedsLabelsDataset,
        )


def test_embedding_standardizer_fit_tensors_standardizes_training_features() -> None:
    embeddings = [
        torch.tensor([1.0, 10.0]),
        torch.tensor([3.0, 30.0]),
        torch.tensor([5.0, 50.0]),
    ]
    standardizer = EmbeddingStandardizer.fit_tensors(embeddings)
    transformed = torch.cat([
        standardizer.transform_tensor(embedding).reshape(1, -1)
        for embedding in embeddings
    ])
    assert torch.allclose(transformed.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(transformed.std(dim=0, unbiased=False), torch.ones(2), atol=1e-6)


def test_embedding_standardizer_handles_constant_features() -> None:
    features = np.array(
        [
            [1.0, 2.0],
            [1.0, 4.0],
            [1.0, 6.0],
        ],
        dtype=np.float32,
    )
    standardizer = EmbeddingStandardizer.fit_numpy(features)
    transformed = standardizer.transform_numpy(features)
    assert np.isfinite(transformed).all()
    assert np.allclose(transformed[:, 0], np.zeros(3))


def test_embeds_labels_dataset_applies_standardizer() -> None:
    emb_dict = {
        "a": torch.tensor([1.0, 10.0]),
        "b": torch.tensor([3.0, 30.0]),
        "c": torch.tensor([5.0, 50.0]),
    }
    hf_dataset = {"seqs": ["a", "b", "c"], "labels": [0, 1, 0]}
    standardizer = EmbeddingStandardizer.fit_tensors(emb_dict[seq] for seq in hf_dataset["seqs"])
    dataset = EmbedsLabelsDataset(
        hf_dataset=hf_dataset,
        emb_dict=emb_dict,
        embedding_standardizer=standardizer,
    )
    emb, label = dataset[0]
    expected = standardizer.transform_tensor(emb_dict["a"]).reshape(-1)
    assert torch.allclose(emb, expected)
    assert label.item() == 0


def test_pair_dataset_standardizes_after_pair_concatenation() -> None:
    emb_dict = {
        "a": torch.tensor([1.0, 10.0]),
        "b": torch.tensor([3.0, 30.0]),
        "c": torch.tensor([5.0, 50.0]),
        "d": torch.tensor([7.0, 70.0]),
    }
    train_pairs = [
        torch.tensor([[1.0, 10.0, 3.0, 30.0]]),
        torch.tensor([[5.0, 50.0, 7.0, 70.0]]),
    ]
    standardizer = EmbeddingStandardizer.fit_tensors(train_pairs)
    hf_dataset = {"SeqA": ["a"], "SeqB": ["b"], "labels": [1]}
    dataset = PairEmbedsLabelsDataset(
        hf_dataset=hf_dataset,
        emb_dict=emb_dict,
        input_size=4,
        embedding_standardizer=standardizer,
    )
    emb_a, emb_b, label = dataset[0]
    observed = torch.cat([emb_a.reshape(1, -1), emb_b.reshape(1, -1)], dim=-1)
    expected = standardizer.transform_tensor(train_pairs[0])
    assert torch.allclose(observed, expected)
    assert label.item() == 1
