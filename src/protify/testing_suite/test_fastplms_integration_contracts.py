"""Protify-specific regression tests for the FastPLMs 1.0 integration."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.protify.base_models.dplm import presets as dplm_presets
from src.protify.base_models.dplm2 import DPLM2TokenizerWrapper
from src.protify.base_models.dplm2 import presets as dplm2_presets
from src.protify.base_models.e1 import E1ForEmbedding
from src.protify.base_models.supported_models import all_presets_with_paths
from src.protify.base_models.utils import load_fastplms_model


class _RecordingModelClass:
    calls = []

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        cls.calls.append((model_path, kwargs))
        return SimpleNamespace(model_path=model_path, kwargs=kwargs)


def test_loader_uses_v1_attention_and_fp32_parameter_policy():
    _RecordingModelClass.calls.clear()
    _RecordingModelClass.__module__ = "fastplms.models.esm2.modeling_fastesm"

    model = load_fastplms_model(
        _RecordingModelClass,
        "local-model",
        dtype=torch.bfloat16,
    )

    assert model.kwargs == {
        "dtype": torch.float32,
        "attn_implementation": "flex_attention",
    }


def test_loader_preserves_static_bf16_and_rejects_fp16():
    _RecordingModelClass.calls.clear()
    _RecordingModelClass.__module__ = (
        "fastplms.models.esm_plusplus.modeling_esm_plusplus"
    )

    model = load_fastplms_model(
        _RecordingModelClass,
        "local-model",
        dtype=torch.bfloat16,
    )
    assert model.kwargs["dtype"] is torch.bfloat16

    with pytest.raises(ValueError, match="float32 and bfloat16"):
        load_fastplms_model(
            _RecordingModelClass,
            "local-model",
            dtype=torch.float16,
        )


def test_dplm2_tokenizer_wrapper_adds_amino_acid_boundaries():
    class RecordingTokenizer:
        aa_cls_token = "<cls_aa>"
        aa_eos_token = "<eos_aa>"

        def __call__(self, sequences, **kwargs):
            self.sequences = sequences
            self.kwargs = kwargs
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    tokenizer = RecordingTokenizer()
    wrapped = DPLM2TokenizerWrapper(tokenizer)

    result = wrapped(["ACD"], padding="longest")

    assert tokenizer.sequences == ["<cls_aa>ACD<eos_aa>"]
    assert tokenizer.kwargs["add_special_tokens"] is False
    assert tokenizer.kwargs["padding"] == "longest"
    assert "input_ids" in result


def test_e1_embedding_drops_batch_preparer_labels():
    class RecordingEncoder(nn.Module):
        def forward(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                last_hidden_state=torch.ones(1, 3, 2),
                hidden_states=None,
                attentions=None,
            )

    wrapper = E1ForEmbedding.__new__(E1ForEmbedding)
    nn.Module.__init__(wrapper)
    wrapper.e1 = RecordingEncoder()

    output = wrapper(
        input_ids=torch.ones(1, 3, dtype=torch.long),
        labels=torch.ones(1, 3, dtype=torch.long),
    )

    assert output.shape == (1, 3, 2)
    assert "labels" not in wrapper.e1.kwargs
    assert "input_ids" in wrapper.e1.kwargs


def test_dplm_presets_match_fastplms_registry():
    from fastplms import get_model_registry

    registry = get_model_registry()
    expected = {
        "DPLM-150": registry["dplm_150m"].fast.repo_id,
        "DPLM-650": registry["dplm_650m"].fast.repo_id,
        "DPLM-3B": registry["dplm_3b"].fast.repo_id,
        "DPLM2-150": registry["dplm2_150m"].fast.repo_id,
        "DPLM2-650": registry["dplm2_650m"].fast.repo_id,
        "DPLM2-3B": registry["dplm2_3b"].fast.repo_id,
    }

    assert {**dplm_presets, **dplm2_presets} == expected
    assert {name: all_presets_with_paths[name] for name in expected} == expected


def test_fastplms_validation_dependency_versions():
    import transformers

    assert torch.__version__.split("+", 1)[0] == "2.13.0"
    assert transformers.__version__ == "5.13.0"
