"""Protify's probes and CaLM load a saved checkpoint back unchanged under Transformers 5.

Transformers 5 builds a model on the meta device, copies in the tensors the checkpoint holds, and
leaves every other parameter and buffer to the model's `_init_weights` as uninitialized memory.
The transformer probes register their rotary tables with `persistent=False`, and CaLM its rotary
frequencies and position ids, so no checkpoint holds them. Until `_init_weights` rebuilt them, a
loaded probe or CaLM backbone attended through whatever the memory held.

Transformers 5 also saves a config's label count only as `id2label`, which each probe config then
overwrote with its default of two, so a probe saved with any other count failed to load.

These tests save tiny, randomly initialized models and load them back with `from_pretrained`, CaLM
through the loader its presets use, so they need neither the Hub nor a GPU. Allocations made while
loading hold NaN, so a buffer the load leaves unfilled cannot match by chance. A probe packaged
inside `PackagedProbeModel` and loaded through `AutoModel` is covered in
`test_packaged_probe_export.py`.
"""

import pytest
import torch

from foundry.checkpoint_loading import assert_unsaved_buffers_match, uninitialized_memory_as_nan


try:
    from src.protify.base_models.calm import CaLmConfig, CaLmModel, _load_calm_backbone
    from src.protify.probes.get_probe import rebuild_probe_from_saved_config
    from src.protify.probes.lyra_probe import LyraConfig, LyraForSequenceClassification, LyraForTokenClassification
    from src.protify.probes.mlp_probe import MLPProbe, MLPProbeConfig
    from src.protify.probes.transformer_probe import (
        TransformerForSequenceClassification,
        TransformerForTokenClassification,
        TransformerProbeConfig,
    )
except ImportError:
    try:
        from protify.base_models.calm import CaLmConfig, CaLmModel, _load_calm_backbone
        from protify.probes.get_probe import rebuild_probe_from_saved_config
        from protify.probes.lyra_probe import LyraConfig, LyraForSequenceClassification, LyraForTokenClassification
        from protify.probes.mlp_probe import MLPProbe, MLPProbeConfig
        from protify.probes.transformer_probe import (
            TransformerForSequenceClassification,
            TransformerForTokenClassification,
            TransformerProbeConfig,
        )
    except ImportError:
        from ..base_models.calm import CaLmConfig, CaLmModel, _load_calm_backbone
        from ..probes.get_probe import rebuild_probe_from_saved_config
        from ..probes.lyra_probe import LyraConfig, LyraForSequenceClassification, LyraForTokenClassification
        from ..probes.mlp_probe import MLPProbe, MLPProbeConfig
        from ..probes.transformer_probe import (
            TransformerForSequenceClassification,
            TransformerForTokenClassification,
            TransformerProbeConfig,
        )


BATCH = 2  # b
LENGTH = 10  # l, tokens per padded row
PROBE_INPUT_SIZE = 16  # d_in, the embedding width a probe reads
NUM_LABELS = 5  # c, a label count other than the probe configs' default of two
CALM_VOCAB_SIZE = 131  # v, CaLM's codon vocabulary
CALM_PAD_TOKEN_ID = 0
CALM_MASK_TOKEN_ID = 4


def _padding_mask() -> torch.Tensor:
    mask = torch.ones(BATCH, LENGTH, dtype=torch.long)  # (b, l)
    mask[1, 6:] = 0  # the second row holds six tokens, then padding
    return mask  # (b, l)


def _probe(model_class: type) -> torch.nn.Module:
    config = TransformerProbeConfig(
        input_size=PROBE_INPUT_SIZE,
        hidden_size=32,
        classifier_size=24,
        transformer_dropout=0.0,
        classifier_dropout=0.0,
        num_labels=NUM_LABELS,
        n_layers=2,
        head_size=8,
        rotary=True,
        attention_backend="sdpa",
        max_seq_len=64,
    )
    torch.manual_seed(0)
    return model_class(config).eval()


def _lyra_config() -> LyraConfig:
    # Nonzero dropout, so the probe builds its `DropoutNd`, which once could not be built on the
    # meta device where `from_pretrained` builds. The probes are compared in eval mode.
    return LyraConfig(input_size=PROBE_INPUT_SIZE, hidden_size=32, dropout=0.2, num_labels=NUM_LABELS)


# Each probe family as `rebuild_probe_from_saved_config` names it (type, tokenwise), a builder, and
# the embeddings it reads: pooled rows for the MLP, residue rows for the others.
LABELLED_PROBES = {
    "mlp": (
        "mlp",
        False,
        lambda: MLPProbe(MLPProbeConfig(input_size=PROBE_INPUT_SIZE, hidden_size=32, dropout=0.0, num_labels=NUM_LABELS)),
        (BATCH, PROBE_INPUT_SIZE),
    ),
    "transformer_sequence": (
        "transformer",
        False,
        lambda: _probe(TransformerForSequenceClassification),
        (BATCH, LENGTH, PROBE_INPUT_SIZE),
    ),
    "transformer_token": (
        "transformer",
        True,
        lambda: _probe(TransformerForTokenClassification),
        (BATCH, LENGTH, PROBE_INPUT_SIZE),
    ),
    "lyra_sequence": (
        "lyra",
        False,
        lambda: LyraForSequenceClassification(_lyra_config()),
        (BATCH, LENGTH, PROBE_INPUT_SIZE),
    ),
    "lyra_token": (
        "lyra",
        True,
        lambda: LyraForTokenClassification(_lyra_config()),
        (BATCH, LENGTH, PROBE_INPUT_SIZE),
    ),
}


def _labelled_probe(kind: str) -> torch.nn.Module:
    _, _, build, _ = LABELLED_PROBES[kind]
    torch.manual_seed(0)
    return build().eval()


def _calm() -> CaLmModel:
    config = CaLmConfig(
        vocab_size=CALM_VOCAB_SIZE,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=64,
        pad_token_id=CALM_PAD_TOKEN_ID,
        mask_token_id=CALM_MASK_TOKEN_ID,
    )
    torch.manual_seed(0)
    return CaLmModel(config, add_pooling_layer=False).eval()


def _probe_round_trip(model_class: type, tmp_path) -> tuple[torch.nn.Module, torch.nn.Module, dict]:
    saved = _probe(model_class)
    saved.save_pretrained(str(tmp_path))
    with uninitialized_memory_as_nan():
        loaded, loading_info = model_class.from_pretrained(str(tmp_path), output_loading_info=True)
    return saved, loaded.eval(), loading_info


def _calm_round_trip(tmp_path) -> tuple[CaLmModel, CaLmModel]:
    saved = _calm()
    saved.save_pretrained(str(tmp_path))
    with uninitialized_memory_as_nan():
        # The loader every CaLM preset goes through; it also refuses a partial load.
        loaded = _load_calm_backbone(str(tmp_path), add_pooling_layer=False)
    return saved, loaded.eval()


@pytest.mark.parametrize(
    "model_class",
    [TransformerForSequenceClassification, TransformerForTokenClassification],
    ids=["sequence", "token"],
)
class TestTransformerProbeRoundTrip:
    def test_every_saved_tensor_loads(self, model_class, tmp_path):
        _, _, loading_info = _probe_round_trip(model_class, tmp_path)

        assert not loading_info["missing_keys"]
        assert not loading_info["unexpected_keys"]
        assert not loading_info["mismatched_keys"]

    def test_unsaved_buffers_match_a_freshly_built_probe(self, model_class, tmp_path):
        saved, loaded, _ = _probe_round_trip(model_class, tmp_path)

        unsaved = assert_unsaved_buffers_match(loaded, saved.state_dict(), lambda: model_class(loaded.config))

        assert unsaved > 0, "the rotary tables are the buffers under test"

    def test_loaded_probe_computes_what_the_saved_one_did(self, model_class, tmp_path):
        saved, loaded, _ = _probe_round_trip(model_class, tmp_path)
        generator = torch.Generator().manual_seed(0)
        embeddings = torch.randn(BATCH, LENGTH, PROBE_INPUT_SIZE, generator=generator)  # (b, l, d_in)

        with torch.inference_mode():
            expected = saved(embeddings=embeddings, attention_mask=_padding_mask()).logits  # (b, c) or (b, l, c)
            actual = loaded(embeddings=embeddings, attention_mask=_padding_mask()).logits  # (b, c) or (b, l, c)

        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", list(LABELLED_PROBES))
class TestSavedLabelCount:
    def test_from_pretrained_keeps_the_label_count(self, kind, tmp_path):
        saved = _labelled_probe(kind)
        saved.save_pretrained(str(tmp_path))
        generator = torch.Generator().manual_seed(0)
        embeddings = torch.randn(*LABELLED_PROBES[kind][3], generator=generator)  # (b, d_in) or (b, l, d_in)

        with uninitialized_memory_as_nan():
            loaded = type(saved).from_pretrained(str(tmp_path)).eval()
        with torch.inference_mode():
            expected = saved(embeddings=embeddings).logits  # (b, c) or (b, l, c)
            actual = loaded(embeddings=embeddings).logits  # (b, c) or (b, l, c)

        assert loaded.config.num_labels == NUM_LABELS
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rebuild_from_the_saved_config_keeps_the_label_count(self, kind):
        probe_type, tokenwise, _, _ = LABELLED_PROBES[kind]
        saved = _labelled_probe(kind)

        rebuilt = rebuild_probe_from_saved_config(probe_type, tokenwise, saved.config.to_dict())

        assert rebuilt.config.num_labels == NUM_LABELS
        rebuilt.load_state_dict(saved.state_dict())  # strict: every tensor has its saved shape


class TestCaLmRoundTrip:
    def test_unsaved_buffers_match_a_freshly_built_backbone(self, tmp_path):
        saved, loaded = _calm_round_trip(tmp_path)

        unsaved = assert_unsaved_buffers_match(
            loaded, saved.state_dict(), lambda: CaLmModel(loaded.config, add_pooling_layer=False)
        )

        assert unsaved > 0, "the rotary frequencies and position ids are the buffers under test"

    def test_loaded_backbone_computes_what_the_saved_one_did(self, tmp_path):
        saved, loaded = _calm_round_trip(tmp_path)
        generator = torch.Generator().manual_seed(0)
        input_ids = torch.randint(CALM_MASK_TOKEN_ID + 1, CALM_VOCAB_SIZE, (BATCH, LENGTH), generator=generator)  # (b, l)
        attention_mask = _padding_mask()  # (b, l)
        input_ids[attention_mask == 0] = CALM_PAD_TOKEN_ID

        with torch.inference_mode():
            expected = saved(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (b, l, d)
            actual = loaded(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (b, l, d)

        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
