"""A saved InterpNet probe loads back whole under Transformers 5, rotary tables included.

Every published InterpNet probe is read with `from_pretrained`: by the property oracles, by synth's
oracle runtime, and by sequence_gradient's design critics. Transformers 5 builds the model on the
meta device, copies in the tensors the checkpoint holds, and leaves every other buffer on
uninitialized memory for the model's `_init_weights`. The rotary tables are registered with
`persistent=False`, so no checkpoint holds them, and until `InterpNetPreTrainedModel._init_weights`
rebuilt them every loaded probe attended through whatever the memory held.

These tests save tiny, randomly initialized probes and load them back, so they need neither the
Hub nor a GPU. Allocations made while loading hold NaN, so a buffer the load leaves unfilled
fails deterministically.
"""

import pytest
import torch

from pathlib import Path
from typing import TypedDict
from transformers import PreTrainedModel

from foundry.checkpoint_loading import assert_unsaved_buffers_match, uninitialized_memory_as_nan
from interpnet import InterpNetConfig, InterpNetForSequenceClassification, InterpNetForTokenClassification
from interpnet.interpnet import InterpNetPreTrainedModel


BATCH = 2  # b
LENGTH = 10  # l, residues per padded row
INPUT_SIZE = 16  # d_in, the protein language model width a probe reads
PROBE_CLASSES = pytest.mark.parametrize(
    "model_class",
    [InterpNetForSequenceClassification, InterpNetForTokenClassification],
    ids=["sequence", "token"],
)


class LoadingInfo(TypedDict):
    """What Transformers 5 reports from `from_pretrained(..., output_loading_info=True)`."""

    missing_keys: set[str]
    unexpected_keys: set[str]
    mismatched_keys: set[tuple[str, tuple[int, ...], tuple[int, ...]]]
    error_msgs: list[str]


class ProbeInputs(TypedDict):
    """The keyword arguments of a probe's forward pass."""

    embeddings: torch.Tensor  # (b, l, d_in)
    attention_mask: torch.Tensor  # (b, l)


def _probe(model_class: type[InterpNetPreTrainedModel]) -> InterpNetPreTrainedModel:
    config = InterpNetConfig(
        input_size=INPUT_SIZE,
        hidden_size=32,
        head_size=8,
        n_layers=2,
        num_labels=3,
        dropout=0.0,
        attention_backend="sdpa",
        max_seq_len=64,
    )
    torch.manual_seed(0)
    return model_class(config).eval()


def _save_and_load(model: InterpNetPreTrainedModel, tmp_path: Path) -> tuple[InterpNetPreTrainedModel, LoadingInfo]:
    """Load `model` back from disk with every allocation made while loading filled with NaN."""
    model.save_pretrained(str(tmp_path))
    with uninitialized_memory_as_nan():
        loaded, loading_info = type(model).from_pretrained(str(tmp_path), output_loading_info=True)
    return loaded.eval(), loading_info


def _inputs() -> ProbeInputs:
    generator = torch.Generator().manual_seed(0)
    mask = torch.ones(BATCH, LENGTH, dtype=torch.long)  # (b, l)
    mask[1, 6:] = 0  # the second row holds six residues, then padding
    embeddings = torch.randn(BATCH, LENGTH, INPUT_SIZE, generator=generator)  # (b, l, d_in)
    return {"embeddings": embeddings, "attention_mask": mask}  # (b, l, d_in) and (b, l)


@PROBE_CLASSES
class TestRoundTrip:
    def test_every_saved_tensor_loads(self, model_class, tmp_path):
        _, loading_info = _save_and_load(_probe(model_class), tmp_path)

        assert not loading_info["missing_keys"]
        assert not loading_info["unexpected_keys"]
        assert not loading_info["mismatched_keys"]

    def test_unsaved_buffers_match_a_freshly_built_probe(self, model_class, tmp_path):
        saved = _probe(model_class)
        loaded, _ = _save_and_load(saved, tmp_path)

        unsaved = assert_unsaved_buffers_match(loaded, saved.state_dict(), lambda: model_class(loaded.config))

        assert unsaved > 0, "the rotary tables are the buffers under test"

    def test_loaded_probe_computes_what_the_saved_one_did(self, model_class, tmp_path):
        saved = _probe(model_class)
        loaded, _ = _save_and_load(saved, tmp_path)

        with torch.inference_mode():
            expected = saved(**_inputs()).logits  # (b, c) or (b, l, c)
            actual = loaded(**_inputs()).logits  # (b, c) or (b, l, c)

        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_the_rotary_tables_come_from_init_weights(self, model_class, tmp_path, monkeypatch):
        """Put the fault back: without InterpNet's `_init_weights`, the loaded tables hold NaN."""
        saved = _probe(model_class)
        monkeypatch.setattr(InterpNetPreTrainedModel, "_init_weights", PreTrainedModel._init_weights)
        loaded, _ = _save_and_load(saved, tmp_path)

        with pytest.raises(AssertionError):
            assert_unsaved_buffers_match(loaded, saved.state_dict(), lambda: model_class(loaded.config))


@PROBE_CLASSES
def test_the_probe_ties_no_weights_and_says_so(model_class, monkeypatch, tmp_path):
    """InterpNet declares an empty tied-weights mapping, and Transformers 5 loads nothing without it."""
    saved = _probe(model_class)
    parameters = [parameter for _, parameter in saved.named_parameters(remove_duplicate=False)]

    assert InterpNetPreTrainedModel.all_tied_weights_keys == {}
    assert len({id(parameter) for parameter in parameters}) == len(parameters), "a parameter is shared"

    # Put the fault back: without the declaration, loading the checkpoint raises.
    saved.save_pretrained(str(tmp_path))
    monkeypatch.delattr(InterpNetPreTrainedModel, "all_tied_weights_keys")
    with pytest.raises(AttributeError, match="all_tied_weights_keys"):
        model_class.from_pretrained(str(tmp_path))
