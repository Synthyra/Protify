"""The default probe featurization on each ESMC base (docs/conventions/probes.md).

Every probe on an ESMC base reads the Biohub SAE at the base's depth layer, max-pooled, beside the
final hidden state's mean and variance. The tokens must parse as ordinary branch specs and name
the three SAE repositories Biohub publishes at k 64 and codebook 16384.
"""

import pytest

from protify.base_models.esmc_sae import parse_sae_model_name
from protify.data.branch_features import default_branch_specs, parse_branch_spec


EXPECTED = {
    "ESMC-300": ("ESMC-300-SAE-l23-k64-c16384", "biohub/ESMC-300M-sae-layer23-k64-codebook16384"),
    "ESMC-600": ("ESMC-600-SAE-l27-k64-c16384", "biohub/ESMC-600M-sae-layer27-k64-codebook16384"),
    "ESMC-6B": ("ESMC-6B-SAE-l60-k64-c16384", "biohub/ESMC-6B-sae-layer60-k64-codebook16384"),
}


@pytest.mark.parametrize("backbone", sorted(EXPECTED))
def test_default_branches_name_the_depth_layer_sae_and_the_final_state(backbone: str) -> None:
    sae_token, dense_token = default_branch_specs(backbone)
    sae, dense = parse_branch_spec(sae_token), parse_branch_spec(dense_token)
    sae_name, repository = EXPECTED[backbone]

    assert sae.model_name == sae_name
    assert sae.pooling_types == ("max",)
    assert parse_sae_model_name(sae.model_name).repository == repository
    assert dense.model_name == backbone
    assert dense.pooling_types == ("mean", "var")
    assert dense.hidden_state_index == -1


def test_default_branches_refuse_a_non_esmc_backbone() -> None:
    with pytest.raises(ValueError, match="default branches are defined for"):
        default_branch_specs("ESM2-650")
