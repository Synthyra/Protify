"""New fused heads expand and saved residual heads retain their topology."""

import pytest
import torch

from protify.probes.get_probe import ProbeArguments, get_probe, rebuild_probe_from_saved_config
from protify.probes.projected_probe import ExpandedProjectedBlock, ProjectedProbe, ProjectedProbeConfig, ResidualBlock


def test_expanded_head_roundtrip(tmp_path):
    config = ProjectedProbeConfig(input_size=12, branch_widths=[8, 4], projection_size=8,
                                  projected_hidden_size=32, dropout=0.2, num_labels=3, trunk="expanded_mlp")
    model = ProjectedProbe(config).eval()
    assert isinstance(model.blocks[0], ExpandedProjectedBlock)
    assert model.blocks[0].expand.out_features == 32
    assert model.output.in_features == 256
    X = torch.randn(4, 12)  # (n, d)
    expected = model(X).logits  # (n, c)
    model.save_pretrained(tmp_path)
    restored = ProjectedProbe.from_pretrained(tmp_path).eval()
    assert restored.config.projected_hidden_size == 32
    torch.testing.assert_close(restored(X).logits, expected)


def test_default_trunk_and_invalid_expansion():
    legacy = ProjectedProbe(ProjectedProbeConfig())
    assert isinstance(legacy.blocks[0], ResidualBlock)
    restored = rebuild_probe_from_saved_config("projected", False, legacy.config.to_dict())
    restored.load_state_dict(legacy.state_dict())
    assert isinstance(restored.blocks[0], ResidualBlock)
    default = get_probe(ProbeArguments(probe_type="projected", input_size=12, num_labels=3))
    assert default.blocks[0].expand.out_features == 8192
    assert default.output.in_features == 256
    with pytest.raises(ValueError, match="positive integer"):
        ProjectedProbe(ProjectedProbeConfig(projected_hidden_size=0, trunk="expanded_mlp"))
    with pytest.raises(ValueError, match="non-GP"):
        ProjectedProbe(ProjectedProbeConfig(projected_hidden_size=32, output="gaussian_process"))


@pytest.mark.parametrize("classes,penultimate", [(3, 256), (256, 256), (257, 512), (513, 768)])
def test_training_default_class_aligned_width(classes, penultimate):
    model = get_probe(ProbeArguments(probe_type="projected", input_size=12, branch_widths=[8, 4],
                                    projection_size=8, num_labels=classes, projected_hidden_size=32))
    assert model.config.trunk == "expanded_mlp"
    assert model.output.in_features == penultimate
    assert model.output.out_features == classes
