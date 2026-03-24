"""Tests for probes/get_probe.py factory dispatch and probe forward passes."""

import torch
import pytest

try:
    from src.protify.probes.get_probe import ProbeArguments, get_probe, rebuild_probe_from_saved_config
    from src.protify.probes.linear_probe import LinearProbe
    from src.protify.probes.transformer_probe import (
        TransformerForSequenceClassification, TransformerForTokenClassification,
    )
    from src.protify.probes.lyra_probe import LyraForSequenceClassification, LyraForTokenClassification
except ImportError:
    try:
        from protify.probes.get_probe import ProbeArguments, get_probe, rebuild_probe_from_saved_config
        from protify.probes.linear_probe import LinearProbe
        from protify.probes.transformer_probe import (
            TransformerForSequenceClassification, TransformerForTokenClassification,
        )
        from protify.probes.lyra_probe import LyraForSequenceClassification, LyraForTokenClassification
    except ImportError:
        from ..probes.get_probe import ProbeArguments, get_probe, rebuild_probe_from_saved_config
        from ..probes.linear_probe import LinearProbe
        from ..probes.transformer_probe import (
            TransformerForSequenceClassification, TransformerForTokenClassification,
        )
        from ..probes.lyra_probe import LyraForSequenceClassification, LyraForTokenClassification


def _make_args(**kwargs) -> ProbeArguments:
    defaults = dict(
        input_size=16,
        hidden_size=32,
        num_labels=3,
        n_layers=1,
        dropout=0.1,
        task_type="singlelabel",
        n_heads=2,
        classifier_size=24,
        transformer_dropout=0.1,
        classifier_dropout=0.1,
        rotary=False,
        attention_backend="sdpa",
        output_s_max=False,
        pre_ln=True,
        probe_pooling_types=["mean"],
        use_bias=False,
        add_token_ids=False,
        expansion_ratio=8 / 3,
    )
    defaults.update(kwargs)
    return ProbeArguments(**defaults)


def test_get_probe_linear_sequence() -> None:
    args = _make_args(probe_type="linear", tokenwise=False)
    probe = get_probe(args)
    assert isinstance(probe, LinearProbe)


def test_get_probe_transformer_sequence() -> None:
    args = _make_args(probe_type="transformer", tokenwise=False)
    probe = get_probe(args)
    assert isinstance(probe, TransformerForSequenceClassification)


def test_get_probe_transformer_tokenwise() -> None:
    args = _make_args(probe_type="transformer", tokenwise=True)
    probe = get_probe(args)
    assert isinstance(probe, TransformerForTokenClassification)


def test_get_probe_lyra_sequence() -> None:
    args = _make_args(probe_type="lyra", tokenwise=False)
    probe = get_probe(args)
    assert isinstance(probe, LyraForSequenceClassification)


def test_get_probe_lyra_tokenwise() -> None:
    args = _make_args(probe_type="lyra", tokenwise=True)
    probe = get_probe(args)
    assert isinstance(probe, LyraForTokenClassification)


def test_get_probe_invalid_raises() -> None:
    args = _make_args(probe_type="nonexistent", tokenwise=False)
    with pytest.raises(ValueError):
        get_probe(args)


def test_linear_probe_forward_singlelabel() -> None:
    torch.manual_seed(0)
    args = _make_args(probe_type="linear", tokenwise=False, task_type="singlelabel", num_labels=3)
    probe = get_probe(args).eval()
    x = torch.randn(2, 16)
    out = probe(x)
    assert out.logits.shape == (2, 3)


def test_linear_probe_forward_regression() -> None:
    torch.manual_seed(0)
    args = _make_args(probe_type="linear", tokenwise=False, task_type="regression", num_labels=1)
    probe = get_probe(args).eval()
    x = torch.randn(2, 16)
    out = probe(x)
    assert out.logits.shape == (2, 1)


def test_linear_probe_forward_multilabel() -> None:
    torch.manual_seed(0)
    args = _make_args(probe_type="linear", tokenwise=False, task_type="multilabel", num_labels=4)
    probe = get_probe(args).eval()
    x = torch.randn(2, 16)
    out = probe(x)
    assert out.logits.shape == (2, 4)


def test_linear_probe_forward_with_labels_returns_loss() -> None:
    torch.manual_seed(0)
    args = _make_args(probe_type="linear", tokenwise=False, task_type="singlelabel", num_labels=3)
    probe = get_probe(args)
    x = torch.randn(2, 16)
    labels = torch.tensor([0, 2])
    out = probe(x, labels=labels)
    assert out.loss is not None
    assert out.loss.ndim == 0
    assert out.loss.item() > 0


def test_linear_probe_forward_sigmoid_regression() -> None:
    torch.manual_seed(0)
    args = _make_args(
        probe_type="linear", tokenwise=False,
        task_type="sigmoid_regression", num_labels=1,
    )
    probe = get_probe(args).eval()
    x = torch.randn(2, 16)
    out = probe(x)
    assert out.logits.shape == (2, 1)
    # sigmoid_regression should output values in [0, 1]
    assert (out.logits >= 0).all()
    assert (out.logits <= 1).all()


def test_rebuild_probe_from_saved_config_linear() -> None:
    args = _make_args(probe_type="linear", tokenwise=False, num_labels=3)
    original = get_probe(args)
    config_dict = original.config.to_dict()
    rebuilt = rebuild_probe_from_saved_config("linear", False, config_dict)
    assert isinstance(rebuilt, LinearProbe)
    assert rebuilt.config.num_labels == 3
