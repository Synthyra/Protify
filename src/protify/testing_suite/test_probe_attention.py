import logging
import pytest
import torch

import fastplms.attention as fastplms_attention

from types import SimpleNamespace

# The kernels backend keeps its loaded kernel in the module that defines the backends.
from foundry.layers import attention_backends
from protify.model_components.attention_utils import AttentionBackend, resolve_attention_backend
from protify.model_components.transformer import Transformer


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


@pytest.fixture
def unloaded_flash_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention_backends, "FLASH_KERNEL", None)
    monkeypatch.setattr(attention_backends, "FLASH_KERNEL_VARIANT", None)
    monkeypatch.setattr(attention_backends, "FLASH_KERNEL_DTYPES", ())


@pytest.mark.usefixtures("unloaded_flash_kernel")
def test_kernels_backend_raises_when_its_kernel_fails_to_load(monkeypatch: pytest.MonkeyPatch) -> None:
    attempted: list[str] = []

    def failing_load(implementation: str) -> tuple[object, str]:
        attempted.append(implementation)
        raise RuntimeError(f"Unable to load the manifest-pinned kernel for {implementation!r}.")

    monkeypatch.setattr(fastplms_attention, "flash_kernel_unsupported_reason", lambda _implementation: None)
    monkeypatch.setattr(fastplms_attention, "_ensure_flash_kernels_loaded", failing_load)

    with pytest.raises(RuntimeError, match="Unable to load the manifest-pinned kernel"):
        resolve_attention_backend("kernels")

    # Only an unsupported platform or GPU moves on to FlashAttention 2; a failed load does not.
    assert attempted == ["flash_attention_3"]
    assert attention_backends.FLASH_KERNEL is None


@pytest.mark.usefixtures("unloaded_flash_kernel")
def test_kernels_backend_passes_over_only_a_kernel_this_platform_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    flash_attention_2 = object()
    reasons = {
        "flash_attention_3": "kernels.lock pins no build of kernels-community/flash-attn3 for this platform.",
        "flash_attention_2": None,
    }
    monkeypatch.setattr(fastplms_attention, "flash_kernel_unsupported_reason", reasons.__getitem__)
    monkeypatch.setattr(
        fastplms_attention,
        "_ensure_flash_kernels_loaded",
        lambda implementation: (flash_attention_2, "flash_attn2"),
    )

    with caplog.at_level(logging.INFO, logger=attention_backends.__name__):
        assert resolve_attention_backend("kernels_flash") == AttentionBackend.KERNELS

    assert attention_backends.FLASH_KERNEL is flash_attention_2
    assert attention_backends.FLASH_KERNEL_VARIANT == "flash_attn2"
    assert "uses flash_attention_2" in caplog.text
    assert "pins no build of kernels-community/flash-attn3" in caplog.text


@pytest.mark.usefixtures("unloaded_flash_kernel")
def test_kernels_backend_names_every_reason_when_no_kernel_runs_here(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_load(implementation: str) -> tuple[object, str]:
        raise AssertionError(f"{implementation} must not load on a platform that cannot run it")

    monkeypatch.setattr(
        fastplms_attention,
        "flash_kernel_unsupported_reason",
        lambda implementation: f"{implementation} runs on a CUDA device, and none is visible.",
    )
    monkeypatch.setattr(fastplms_attention, "_ensure_flash_kernels_loaded", unexpected_load)

    with pytest.raises(RuntimeError, match=r"flash_attention_3 runs on a CUDA device.*flash_attention_2 runs"):
        resolve_attention_backend("kernels")


@pytest.mark.usefixtures("unloaded_flash_kernel")
def test_kernels_backend_raises_on_float32_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    query_dtypes: list[torch.dtype] = []

    def flash_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
        # q: (b, l, h, d_h); k, v: (b, l, h_kv, d_h)
        query_dtypes.append(q.dtype)
        return torch.zeros_like(q)  # (b, l, h, d_h)

    flash_attention_3 = SimpleNamespace(flash_attn_func=flash_attn_func)
    monkeypatch.setattr(fastplms_attention, "flash_kernel_unsupported_reason", lambda _implementation: None)
    monkeypatch.setattr(
        fastplms_attention,
        "_ensure_flash_kernels_loaded",
        lambda _implementation: (flash_attention_3, "flash_attn3"),
    )
    resolve_attention_backend("kernels")
    query_states = torch.randn(2, 4, 2, 8)  # (b, l, h, d_h)

    # The dtypes come from the FastPLMs manifest, which lists only bfloat16 for FlashAttention 3.
    with pytest.raises(
        RuntimeError,
        match=r"kernels attention cannot run torch\.float32 inputs: flash_attn3 takes torch\.bfloat16",
    ):
        attention_backends.kernels_attention_func(query_states, query_states, query_states, None)
    assert query_dtypes == []

    bfloat16_states = query_states.to(torch.bfloat16)  # (b, l, h, d_h)
    attention_backends.kernels_attention_func(bfloat16_states, bfloat16_states, bfloat16_states, None)
    assert query_dtypes == [torch.bfloat16]


@pytest.mark.usefixtures("unloaded_flash_kernel")
def test_kernels_backend_returned_attention_puts_no_weight_on_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    def kernel_call(**_kwargs: object) -> object:
        raise RuntimeError("Returned attention weights must not come from the FlashAttention kernel.")

    flash_attention_3 = SimpleNamespace(flash_attn_func=kernel_call, flash_attn_varlen_func=kernel_call)
    monkeypatch.setattr(fastplms_attention, "flash_kernel_unsupported_reason", lambda _implementation: None)
    monkeypatch.setattr(
        fastplms_attention,
        "_ensure_flash_kernels_loaded",
        lambda _implementation: (flash_attention_3, "flash_attn3"),
    )
    torch.manual_seed(0)
    transformer = Transformer(hidden_size=16, n_heads=4, n_layers=2, attention_backend="kernels").eval()
    hidden_states = torch.randn(2, 6, 16)  # (b, l, d)
    # The second sequence has four real positions and two padding positions.
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)  # (b, l)

    def assert_padding_is_masked() -> None:
        # The weights come from eager attention, so float32 is fine and the kernel never runs.
        with torch.no_grad():
            padded = transformer(hidden_states, attention_mask=attention_mask, output_attentions=True)
            alone = transformer(hidden_states[1:, :4], output_attentions=True)
        for attention in padded.attentions:  # each: (b, h, l, l)
            assert torch.all(attention[1, :, :, 4:] == 0), "attention weight on padded keys"
        torch.testing.assert_close(
            padded.last_hidden_state[1, :4],  # (4, d)
            alone.last_hidden_state[0],  # (4, d)
            atol=1e-5,
            rtol=1e-5,
        )

    assert_padding_is_masked()

    # Put the fault back: the kernels backend built no 4D mask, even for the weights path.
    build_masks = attention_backends.build_attention_masks
    monkeypatch.setattr(
        f"{Transformer.__module__}.build_attention_masks",
        lambda **kwargs: build_masks(**{**kwargs, "output_attentions": False}),
    )
    with pytest.raises(AssertionError):
        assert_padding_is_masked()
