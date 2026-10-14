"""InterpNet's kernels backend loads FlashAttention through FastPLMs' kernels.lock, or raises.

A kernel is passed over only when FastPLMs reports that this platform or GPU cannot run its
locked build. A download, digest, or import failure raises rather than turning into another
backend. FlashAttention takes only inputs of a dtype that its FastPLMs manifest entry lists, and
any other dtype raises before the kernel runs. Returned attention weights, which only the eager
path computes, put no weight on padded keys. Every case below replaces the FastPLMs loading calls,
so nothing is downloaded; the dtypes come from the real manifest.
"""

import logging
import pytest
import torch

from collections.abc import Callable

from fastplms import attention as fastplms_attention
from interpnet import attention_utils, transformer


NO_FLASH_ATTENTION_3 = "kernels.lock pins no build of kernels-community/flash-attn3 for this platform."
HIDDEN_SIZE = 32  # d
# Two proteins of six positions; the second ends in two padding positions.
ATTENTION_MASK = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])  # (b, l)
REAL_LENGTH = 4  # l_real, real positions of the second protein


@pytest.fixture(autouse=True)
def unloaded_flash_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start with no FlashAttention kernel chosen, as a fresh process would."""
    monkeypatch.setattr(attention_utils, "FLASH_KERNEL", None)
    monkeypatch.setattr(attention_utils, "FLASH_KERNEL_VARIANT", None)
    monkeypatch.setattr(attention_utils, "FLASH_KERNEL_DTYPES", ())


class RecordingFlashAttention3:
    """Stands in for the FlashAttention 3 kernel and records the dtype of every query it gets."""

    def __init__(self) -> None:
        self.query_dtypes: list[torch.dtype] = []

    def flash_attn_func(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        # q: (b, l, h, d_h); k, v: (b, l, h_kv, d_h)
        self.query_dtypes.append(q.dtype)
        return torch.zeros_like(q)  # (b, l, h, d_h)

    def flash_attn_varlen_func(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
    ) -> torch.Tensor:
        # q: (n_valid, h, d_h); k, v: (n_valid, h_kv, d_h), where n_valid counts the unpadded tokens
        # cu_seqlens_q, cu_seqlens_k: (b + 1,)
        self.query_dtypes.append(q.dtype)
        return torch.zeros_like(q)  # (n_valid, h, d_h)


def flash_attention_3_runs_here(monkeypatch: pytest.MonkeyPatch, kernel: RecordingFlashAttention3) -> None:
    """Report a locked FlashAttention 3 build runnable here, and load `kernel` for it."""
    monkeypatch.setattr(fastplms_attention, "flash_kernel_unsupported_reason", lambda _implementation: None)
    monkeypatch.setattr(
        fastplms_attention,
        "_ensure_flash_kernels_loaded",
        lambda _implementation: (kernel, "flash_attn3"),
    )


def failing_load(attempted: list[str]) -> Callable[[str], tuple[object, str]]:
    """A kernel loader that records each implementation it is asked for, then raises."""

    def load(implementation: str) -> tuple[object, str]:
        attempted.append(implementation)
        raise RuntimeError(f"Unable to load the manifest-pinned kernel for {implementation!r}.")

    return load


def unexpected_load(implementation: str) -> tuple[object, str]:
    raise AssertionError(f"{implementation} must not load on a platform that cannot run it")


def assert_padding_is_masked(stack: transformer.Transformer, hidden_states: torch.Tensor) -> None:
    """Assert that returned attention and the second protein's real states ignore its padding."""
    # hidden_states: (b, l, d)
    with torch.no_grad():
        padded = stack(hidden_states, attention_mask=ATTENTION_MASK, output_attentions=True)  # states: (b, l, d)
        alone = stack(hidden_states[1:, :REAL_LENGTH], output_attentions=True)  # states: (1, l_real, d)

    for attention in padded.attentions:  # each: (b, h, l, l)
        assert torch.all(attention[1, :, :, REAL_LENGTH:] == 0), "attention weight on padded keys"
    torch.testing.assert_close(
        padded.last_hidden_state[1, :REAL_LENGTH],  # (l_real, d)
        alone.last_hidden_state[0],  # (l_real, d)
        atol=1e-5,
        rtol=1e-5,
    )


def test_raises_when_a_supported_kernel_fails_to_load(monkeypatch: pytest.MonkeyPatch) -> None:
    attempted: list[str] = []
    monkeypatch.setattr(fastplms_attention, "flash_kernel_unsupported_reason", lambda _implementation: None)
    monkeypatch.setattr(fastplms_attention, "_ensure_flash_kernels_loaded", failing_load(attempted))

    with pytest.raises(RuntimeError, match="Unable to load the manifest-pinned kernel"):
        attention_utils.resolve_attention_backend("kernels")

    # Only an unsupported platform or GPU moves on to FlashAttention 2; a failed load does not.
    assert attempted == ["flash_attention_3"]
    assert attention_utils.FLASH_KERNEL is None


def test_passes_over_only_a_kernel_this_platform_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    flash_attention_2 = object()
    reasons = {"flash_attention_3": NO_FLASH_ATTENTION_3, "flash_attention_2": None}
    monkeypatch.setattr(fastplms_attention, "flash_kernel_unsupported_reason", reasons.__getitem__)
    monkeypatch.setattr(
        fastplms_attention,
        "_ensure_flash_kernels_loaded",
        lambda _implementation: (flash_attention_2, "flash_attn2"),
    )

    with caplog.at_level(logging.INFO, logger=attention_utils.__name__):
        resolved = attention_utils.resolve_attention_backend("kernels_flash")

    assert resolved == attention_utils.AttentionBackend.KERNELS
    assert attention_utils.FLASH_KERNEL is flash_attention_2
    assert attention_utils.FLASH_KERNEL_VARIANT == "flash_attn2"
    assert "uses flash_attention_2" in caplog.text
    assert NO_FLASH_ATTENTION_3 in caplog.text


def test_names_every_reason_when_no_kernel_runs_here(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fastplms_attention,
        "flash_kernel_unsupported_reason",
        lambda implementation: f"{implementation} runs on a CUDA device, and none is visible.",
    )
    monkeypatch.setattr(fastplms_attention, "_ensure_flash_kernels_loaded", unexpected_load)

    with pytest.raises(
        RuntimeError,
        match=r"flash_attention_3 runs on a CUDA device.*flash_attention_2 runs",
    ):
        attention_utils.resolve_attention_backend("kernels")


@pytest.mark.parametrize("padding_mask", [None, ATTENTION_MASK.bool()], ids=["dense", "unpadded"])
def test_raises_on_a_dtype_its_manifest_entry_does_not_list(
    monkeypatch: pytest.MonkeyPatch,
    padding_mask: torch.Tensor | None,
) -> None:
    # padding_mask: (b, l) or None, which select the unpadded and the dense kernel calls
    kernel = RecordingFlashAttention3()
    flash_attention_3_runs_here(monkeypatch, kernel)
    attention_utils.resolve_attention_backend("kernels")
    query_states = torch.randn(2, 6, 2, 16)  # (b, l, h, d_h)

    # The FastPLMs manifest lists only bfloat16 for FlashAttention 3.
    with pytest.raises(
        RuntimeError,
        match=r"kernels attention cannot run torch\.float32 inputs: flash_attn3 takes torch\.bfloat16",
    ):
        attention_utils.kernels_attention_func(query_states, query_states, query_states, padding_mask)
    assert kernel.query_dtypes == []

    bfloat16_states = query_states.to(torch.bfloat16)  # (b, l, h, d_h)
    attention_utils.kernels_attention_func(bfloat16_states, bfloat16_states, bfloat16_states, padding_mask)
    assert kernel.query_dtypes == [torch.bfloat16]


def test_returned_attention_puts_no_weight_on_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    kernel = RecordingFlashAttention3()
    flash_attention_3_runs_here(monkeypatch, kernel)
    torch.manual_seed(0)
    stack = transformer.Transformer(
        hidden_size=HIDDEN_SIZE,
        n_heads=2,
        n_layers=2,
        attention_backend="kernels",
    ).eval()
    hidden_states = torch.randn(2, 6, HIDDEN_SIZE)  # (b, l, d)

    # The weights come from eager attention, so float32 is fine and the kernel never runs.
    assert_padding_is_masked(stack, hidden_states)
    assert kernel.query_dtypes == []

    # Put the fault back: the kernels backend built no 4D mask, even for the weights path.
    build_masks = transformer.build_attention_masks
    monkeypatch.setattr(
        transformer,
        "build_attention_masks",
        lambda **kwargs: build_masks(**{**kwargs, "output_attentions": False}),
    )
    with pytest.raises(AssertionError):
        assert_padding_is_masked(stack, hidden_states)
