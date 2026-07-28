import torch
import pytest

try:
    from model_components.attention_utils import AttentionBackend, resolve_attention_backend
    from model_components.transformer import Transformer
except ImportError:
    try:
        from protify.model_components.attention_utils import AttentionBackend, resolve_attention_backend
        from protify.model_components.transformer import Transformer
    except ImportError:
        from ..model_components.attention_utils import AttentionBackend, resolve_attention_backend
        from ..model_components.transformer import Transformer


def test_transformer_returns_attentions_and_s_max_from_2d_mask() -> None:
    torch.manual_seed(0)
    transformer = Transformer(
        hidden_size=16,
        n_heads=4,
        n_layers=2,
        rotary=True,
        attention_backend="flex",
    )
    hidden_states = torch.randn(2, 4, 16)  # (b=2, l=4, d=16)
    attention_mask = torch.tensor(
        [[1, 1, 1, 0], [1, 1, 1, 1]],
        dtype=torch.bool,
    )  # (b=2, l=4)

    outputs = transformer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True,
        output_s_max=True,
    )
    # last_hidden_state: (b, l, d); each hidden state: (b, l, d);
    # each attention map: (b, h=4, l, l); two s_max layer entries,
    # each containing h=4 scalar tensors ()

    assert outputs.last_hidden_state.shape == (2, 4, 16)
    assert outputs.hidden_states is not None
    assert len(outputs.hidden_states) == 2
    assert outputs.attentions is not None
    assert len(outputs.attentions) == 2
    assert outputs.attentions[0] is not None
    assert outputs.attentions[0].shape == (2, 4, 4, 4)
    assert outputs.s_max is not None
    assert len(outputs.s_max) == 2
    assert outputs.s_max[0] is not None
    assert len(outputs.s_max[0]) == 4


def test_transformer_accepts_explicit_4d_attention_mask() -> None:
    torch.manual_seed(0)
    transformer = Transformer(
        hidden_size=16,
        n_heads=4,
        n_layers=1,
        rotary=True,
        attention_backend="sdpa",
    )
    hidden_states = torch.randn(2, 4, 16)  # (b=2, l=4, d=16)
    attention_mask_4d = torch.tensor(
        [
            [[
                [True, True, True, False],
                [True, True, True, False],
                [True, True, True, False],
                [False, False, False, False],
            ]],
            [[
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, True],
            ]],
        ],
        dtype=torch.bool,
    )  # (b=2, 1, l=4, l=4)

    outputs = transformer(
        hidden_states=hidden_states,
        attention_mask_4d=attention_mask_4d,
        output_s_max=True,
    )
    # last_hidden_state: (b, l, d); one layer of h=4 scalar () s_max entries

    assert outputs.last_hidden_state.shape == (2, 4, 16)
    assert outputs.s_max is not None
    assert len(outputs.s_max) == 1
    assert outputs.s_max[0] is not None
    assert len(outputs.s_max[0]) == 4


def test_resolve_attention_backend_rejects_unknown_values() -> None:
    assert resolve_attention_backend("sdpa") == AttentionBackend.SDPA
    with pytest.raises(AssertionError):
        resolve_attention_backend("unknown")
