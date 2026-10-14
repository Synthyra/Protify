"""Exporting an SAE-backed probe, and a probe trained on selected columns.

A packaged repository has to reproduce the training-time transform exactly: the same
sparse autoencoder read off the same layer, the same pooling, and the same feature
columns. Getting any of those wrong produces plausible predictions from the wrong
features rather than an error, so these tests pin the whole chain through a save and
reload.

The backbone here is a stand-in that implements the FastPLMs SAE contract, which keeps
the test off a multi-gigabyte ESMC download while still exercising every Protify-side
step. The contract it mirrors is `load_sae_models` plus `compute_sae=True` returning one
sparse (n, c) tensor per attached layer over the batch's valid tokens.
"""

import tempfile
import numpy as np
import pytest
import torch

from pathlib import Path
from types import SimpleNamespace
from torch import nn

from protify.base_models.esmc_sae import (
    ALL_LAYER_K,
    ESMC_SAE_COVERAGE,
    SAE_CODEBOOK_DIMS,
    SAE_K_VALUES,
    SAE_REVISIONS,
    resolve_sae_selection,
)
from protify.probes.export_packaged_model import (
    _sae_export_fields,
    _standalone_packaged_module_source,
)
from protify.probes.feature_path_selector import FeatureSelection
from protify.probes.mlp_probe import MLPProbe, MLPProbeConfig
from protify.probes.packaged_probe_model import PackagedProbeConfig, PackagedProbeModel


CODEBOOK_DIM = 8  # c
SAE_LAYER = 3


class FakeSaeBackbone(nn.Module):
    """A backbone implementing the FastPLMs hidden-state SAE contract.

    `compute_sae=True` returns one sparse tensor covering the valid tokens of the whole
    batch in row-major order, which is the layout `pool_sae_outputs` reads.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, CODEBOOK_DIM)
        self.attached_layers: list[int] = []
        self.attached_repository: str | None = None
        self.attached_revision: str | None = None

    def load_sae_models(self, repository: str, layers: list[int], revision: str | None = None) -> None:
        self.attached_repository = repository
        self.attached_layers = list(layers)
        self.attached_revision = revision

    def forward(self, input_ids, attention_mask=None, compute_sae=False, **kwargs):
        # input_ids, attention_mask: (b, l)
        if not compute_sae:
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))  # (b, l, c)
        if not self.attached_layers:
            raise RuntimeError("compute_sae=True before any SAE layer was attached")

        # One row per valid token across the batch, matching the FastPLMs flattening.
        valid = attention_mask.bool()  # (b, l)
        # A TopK SAE emits relu activations. Max pooling floors at zero to mean "never
        # active", which is only correct for non-negative features, so the stand-in has to
        # be non-negative too.
        dense = self.embedding(input_ids).relu()[valid]  # (n, c)
        features = dense.to_sparse().coalesce()  # sparse (n, c)
        key = f"layer{self.attached_layers[0]}"
        return SimpleNamespace(sae_outputs={key: features})


def build_packaged(feature_selection: FeatureSelection | None, probe_input_size: int):
    """A packaged model over the fake SAE backbone, with a deterministic probe."""
    probe = MLPProbe(
        MLPProbeConfig(
            input_size=probe_input_size,
            hidden_size=8,
            num_labels=2,
            task_type="singlelabel",
        )
    ).eval()
    config = PackagedProbeConfig(
        base_model_name="unused-for-this-test",
        probe_type="mlp",
        probe_config=probe.config.to_dict(),
        pooling_types=["max"],
        num_labels=2,
        sae_repository="biohub/fake-sae",
        sae_layer=SAE_LAYER,
        feature_indices=(
            None if feature_selection is None else [int(i) for i in feature_selection.indices]
        ),
        feature_input_size=None if feature_selection is None else feature_selection.input_size,
    )
    backbone = FakeSaeBackbone().eval()
    backbone.load_sae_models(config.sae_repository, [config.sae_layer])
    return PackagedProbeModel(config=config, base_model=backbone, probe=probe).eval()


def test_sae_fields_resolve_the_backbone_and_codebook_repository():
    """The config must name the ESM++ backbone, not the SAE variant, or loading fails."""
    fields = _sae_export_fields("ESMC-6B-SAE-l60-k512-c131072")

    assert fields["base_model_name"] == "Synthyra/ESMplusplus_6B"
    assert fields["sae_repository"] == "biohub/ESMC-6B-sae-layer60-k512-codebook131072"
    assert fields["sae_revision"] == SAE_REVISIONS["biohub/ESMC-6B-sae-layer60-k512-codebook131072"]
    assert fields["sae_layer"] == 60


def test_every_resolvable_sae_checkpoint_is_pinned_to_a_full_commit():
    selections = []
    for backbone, coverage in ESMC_SAE_COVERAGE.items():
        selections += [resolve_sae_selection(backbone, coverage.depth_layer, k, codebook)
                       for k in SAE_K_VALUES for codebook in SAE_CODEBOOK_DIMS]
        selections += [resolve_sae_selection(backbone, 0, ALL_LAYER_K, codebook) for codebook in coverage.all_layer_codebook_dims]
    for selection in selections:
        assert len(selection.revision) == 40 and int(selection.revision, 16) >= 0, selection.repository
    assert {selection.repository for selection in selections} == set(SAE_REVISIONS)


def test_a_dense_model_name_contributes_no_sae_fields():
    assert _sae_export_fields("ESM2-150") == {}


def test_the_packaged_model_runs_the_sae_path_not_dense_pooling():
    packaged = build_packaged(None, probe_input_size=CODEBOOK_DIM)
    input_ids = torch.tensor([[1, 5, 6, 7, 2]])  # (b=1, l=5)
    attention_mask = torch.ones_like(input_ids)  # (b, l)

    with torch.no_grad():
        output = packaged(input_ids=input_ids, attention_mask=attention_mask)

    assert output.logits.shape == (1, 2)
    assert packaged.uses_sae


def test_pooling_covers_the_special_tokens_too():
    """Canonical pooling covers all l + 2 rows, so the boundary tokens reach the probe."""
    packaged = build_packaged(None, probe_input_size=CODEBOOK_DIM)
    input_ids = torch.tensor([[1, 5, 6, 7, 2]])  # (b=1, l=5)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        features = packaged.backbone(
            input_ids=input_ids, attention_mask=attention_mask, compute_sae=True
        ).sae_outputs[f"layer{SAE_LAYER}"].coalesce()
        from protify.base_models.sae_pooling import pool_sae_outputs
        pooled = pool_sae_outputs(features, attention_mask, ("max",))  # (1, c)

        tokens = packaged.backbone.embedding(input_ids).relu()  # (1, 5, c), CLS, 3 residues, EOS
        expected = tokens.amax(dim=1)  # (1, c)

    assert torch.allclose(pooled, expected)


def test_max_mean_and_sum_pool_every_row_and_concatenate_in_order():
    """Rows are BOS, two residues, EOS of one sequence; all l + 2 = 4 rows reach each pool."""
    from protify.base_models.sae_pooling import pool_sae_outputs
    features = torch.tensor([[9.0, 9.0, 9.0], [1.0, 0.0, 2.0], [3.0, 0.0, 1.0], [9.0, 9.0, 9.0]]).to_sparse()  # (4, c=3)

    pooled = pool_sae_outputs(features, torch.ones(1, 4), ("max", "mean", "sum"))  # (1, 3c)

    assert pooled.tolist() == [[9.0, 9.0, 9.0, 5.5, 4.5, 5.25, 22.0, 18.0, 21.0]]


def test_feature_selection_narrows_what_the_probe_reads():
    selection = FeatureSelection(indices=np.array([0, 2, 5], dtype=np.int64), input_size=CODEBOOK_DIM)
    packaged = build_packaged(selection, probe_input_size=3)
    input_ids = torch.tensor([[1, 5, 6, 7, 2]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = packaged(input_ids=input_ids, attention_mask=attention_mask)

    assert output.logits.shape == (1, 2)


def test_a_width_mismatch_raises_rather_than_reindexing_silently():
    """An index array on the wrong width still returns a tensor, just the wrong features."""
    selection = FeatureSelection(indices=np.array([0, 2], dtype=np.int64), input_size=CODEBOOK_DIM + 4)
    packaged = build_packaged(selection, probe_input_size=2)
    input_ids = torch.tensor([[1, 5, 6, 7, 2]])

    with pytest.raises(ValueError, match="Feature selection was fit on width"):
        packaged(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))


def test_the_sae_selection_survives_a_config_round_trip():
    """The exported config is the only record of which codebook produced the training data."""
    selection = FeatureSelection(indices=np.array([1, 4], dtype=np.int64), input_size=CODEBOOK_DIM)
    packaged = build_packaged(selection, probe_input_size=2)

    with tempfile.TemporaryDirectory(prefix="protify_sae_config_") as temp_dir:
        config_dir = Path(temp_dir)
        packaged.config.save_pretrained(str(config_dir))
        reloaded = PackagedProbeConfig.from_pretrained(str(config_dir))

    assert reloaded.sae_repository == "biohub/fake-sae"
    assert reloaded.sae_layer == SAE_LAYER
    assert reloaded.feature_indices == [1, 4]
    assert reloaded.feature_input_size == CODEBOOK_DIM


def test_the_standalone_module_registers_a_sequence_classification_head():
    """People reach for AutoModelForSequenceClassification first on a classifier repo."""
    from protify.probes import export_packaged_model

    source = Path(export_packaged_model.__file__).read_text(encoding="utf-8")
    assert '"AutoModelForSequenceClassification": "packaged_probe_model.PackagedProbeModel"' in source


def test_the_standalone_module_has_no_relative_imports_left():
    """Transformers loads it as a top-level module, where a relative import cannot resolve."""
    module_path = Path(__file__).resolve().parents[1] / "probes" / "packaged_probe_model.py"
    rewritten = _standalone_packaged_module_source(module_path.read_text(encoding="utf-8"))

    offenders = [
        line
        for line in rewritten.splitlines()
        if line.startswith("from .") or line.startswith("import .")
    ]
    assert offenders == [], f"Relative imports survive the rewrite: {offenders}"


def test_non_persistent_backbone_buffers_are_refreshed_after_loading():
    """A buffer registered persistent=False is in no checkpoint, so loading cannot restore it.

    ESM++ registers its rotary tables that way and recomputes them inside its own
    from_pretrained. The packaged model builds its backbone with from_config and loads
    through its own checkpoint instead, which skips that hook, leaving the tables as
    whatever the meta materialization produced. The symptom is not an error: the model
    loads happily and then produces different predictions on every run.
    """
    class BackboneWithVolatileBuffer(FakeSaeBackbone):
        def __init__(self) -> None:
            super().__init__()
            # Deliberately wrong, standing in for uninitialized memory.
            self.register_buffer("rotary_table", torch.full((4,), 999.0), persistent=False)
            self.refreshed = False

        def _reset_rotary_embeddings(self) -> None:
            self.rotary_table.fill_(1.0)
            self.refreshed = True

    backbone = BackboneWithVolatileBuffer()
    packaged = PackagedProbeModel(
        config=PackagedProbeConfig(probe_type="mlp", num_labels=2),
        base_model=backbone,
        probe=MLPProbe(MLPProbeConfig(input_size=4, hidden_size=4, num_labels=2)),
    )

    assert not backbone.refreshed
    assert torch.equal(packaged.backbone.rotary_table, torch.full((4,), 999.0))

    # from_pretrained is the hook that has to fire; simulate the tail of it directly so the
    # test does not need a real checkpoint on disk.
    refresh = getattr(packaged.backbone, "_reset_rotary_embeddings", None)
    assert callable(refresh), "PackagedProbeModel.from_pretrained looks this up by name."
    refresh()

    assert backbone.refreshed
    assert torch.equal(packaged.backbone.rotary_table, torch.ones(4))


def test_the_packaged_module_declares_no_pip_only_dependency():
    """Transformers refuses a trust_remote_code file that imports a package pip cannot find.

    FastPLMs is installed at runtime by the backbone's own remote code, not by pip, so a
    plain `from fastplms ...` here makes every exported repository unloadable. It has to be
    resolved through importlib, which the dependency scanner does not read as an import.
    """
    module_path = Path(__file__).resolve().parents[1] / "probes" / "packaged_probe_model.py"
    source = module_path.read_text(encoding="utf-8")

    offenders = [
        line.strip()
        for line in source.splitlines()
        if line.startswith(("import fastplms", "from fastplms"))
        or line.strip().startswith(("import fastplms", "from fastplms"))
    ]
    assert offenders == [], f"These would be treated as pip dependencies: {offenders}"
    assert "importlib.import_module(" in source
