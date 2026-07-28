"""Offline contracts for Protify's T5 classification heads."""

import pytest
import torch
from transformers import T5Config

from src.protify.base_models.t5 import (
    T5ForSequenceClassification,
    T5ForTokenClassification,
)


def _tiny_config() -> T5Config:
    return T5Config(
        vocab_size=16,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        classifier_dropout=0.0,
        num_labels=3,
        pad_token_id=0,
        eos_token_id=1,
    )


@pytest.mark.parametrize(
    ("model_class", "labels", "expected_shape"),
    (
        (T5ForSequenceClassification, torch.tensor([1, 2]), (2, 3)),
        (
            T5ForTokenClassification,
            torch.tensor([[1, 2, 0, -100], [2, 0, -100, -100]]),
            (2, 4, 3),
        ),
    ),
)
def test_t5_classification_heads_accept_optional_labels(
    model_class: type[T5ForSequenceClassification] | type[T5ForTokenClassification],
    labels: torch.Tensor,
    expected_shape: tuple[int, ...],
):
    model = model_class(_tiny_config())
    input_ids = torch.tensor([[2, 3, 1, 0], [4, 1, 0, 0]])  # (b=2, l=4)
    attention_mask = input_ids.ne(0)  # (b, l)

    unlabeled_output = model(input_ids=input_ids, attention_mask=attention_mask)
    assert unlabeled_output.loss is None
    assert unlabeled_output.logits.shape == expected_shape

    positional_output = model(
        input_ids,
        attention_mask,
        None,
        None,
        False,
        False,
        True,
    )
    assert positional_output.loss is None
    assert positional_output.logits.shape == expected_shape

    labeled_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    assert labeled_output.logits.shape == expected_shape
    assert labeled_output.loss is not None
    assert labeled_output.loss.ndim == 0
    assert torch.isfinite(labeled_output.loss)

    labeled_output.loss.backward()
    assert model.shared.weight.grad is not None
    assert torch.isfinite(model.shared.weight.grad).all()
