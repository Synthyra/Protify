"""An InterpNet sequence probe runs forward and backward over padded batches."""

import pytest
import torch

from interpnet import InterpNetConfig, InterpNetForSequenceClassification


def test_sequence_probe_trains_on_padded_cpu_inputs() -> None:
    config = InterpNetConfig(
        input_size=6,
        hidden_size=8,
        head_size=4,
        n_layers=1,
        num_labels=2,
        dropout=0.0,
        attention_backend="sdpa",
        output_s_max=False,
        max_seq_len=8,
    )
    model = InterpNetForSequenceClassification(config)
    embeddings = torch.randn((2, 5, 6), requires_grad=True)  # (b=2, l=5, d_in=6)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])  # (b=2, l=5)
    labels = torch.tensor([0, 1])  # (b=2,)

    output = model(
        embeddings,
        attention_mask,
        labels,
        output_attentions=True,
        output_hidden_states=True,
        output_sims=True,
    )
    assert output.loss is not None
    assert output.logits is not None
    output.loss.backward()

    assert output.logits.shape == (2, 2)
    assert output.sims is not None and output.sims.shape == (2, 5, 2)
    assert output.hidden_states is not None
    assert output.hidden_states[-1].shape == (2, 5, 8)
    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_sequence_probe_trains_in_bfloat16_on_cuda() -> None:
    device = torch.device("cuda")
    config = InterpNetConfig(
        input_size=12,
        hidden_size=16,
        head_size=4,
        n_layers=2,
        num_labels=2,
        dropout=0.0,
        attention_backend="sdpa",
        output_s_max=False,
        max_seq_len=16,
    )
    model = InterpNetForSequenceClassification(config).to(device=device, dtype=torch.bfloat16)
    embeddings = torch.randn(
        (3, 11, 12),
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )  # (b=3, l=11, d_in=12)
    attention_mask = torch.tensor(
        [[1] * 11, [1] * 8 + [0] * 3, [1] * 5 + [0] * 6],
        device=device,
    )  # (b=3, l=11)
    labels = torch.tensor([0, 1, 0], device=device)  # (b=3,)

    output = model(
        embeddings,
        attention_mask,
        labels,
        output_attentions=True,
        output_hidden_states=True,
        output_sims=True,
    )
    assert output.loss is not None
    assert output.logits is not None
    output.loss.backward()

    assert output.logits.is_cuda
    assert bool(torch.isfinite(output.logits).all().item())
    assert embeddings.grad is not None
    assert bool(torch.isfinite(embeddings.grad).all().item())
