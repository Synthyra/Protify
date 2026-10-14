"""`register_probe_type`, the one seam through which a package outside Protify adds a probe head.

A registered type builds, saves its own settings and no other type's, rebuilds from a saved config, and cannot
displace a built-in probe or another package's type.
"""

import pytest
import torch

from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from protify.probes import get_probe as registry
from protify.probes.get_probe import (
    ProbeArguments,
    get_probe,
    rebuild_probe_from_saved_config,
    register_external_probe,
    register_probe_type,
    registered_probe_types,
)


class ToyConfig(PretrainedConfig):
    model_type = "toy_probe"

    def __init__(self, input_size: int = 4, num_labels: int = 2, trunk_width: int = 8, **kwargs) -> None:
        super().__init__(num_labels=num_labels, **kwargs)
        self.input_size = input_size
        self.trunk_width = trunk_width


class ToyProbe(PreTrainedModel):
    config_class = ToyConfig

    def __init__(self, config: ToyConfig) -> None:
        super().__init__(config)
        self.layers = nn.Sequential(nn.Linear(config.input_size, config.trunk_width), nn.Linear(config.trunk_width, config.num_labels))

    def forward(self, embeddings: torch.Tensor, attention_mask=None, labels=None) -> torch.Tensor:
        return self.layers(embeddings)  # (b, d) -> (b, c)


TOY_FIELDS = ("input_size", "num_labels", "trunk_width")


def build_toy(probe_type: str, fields: dict, tokenwise: bool) -> ToyProbe:
    return ToyProbe(ToyConfig(**{name: fields[name] for name in TOY_FIELDS}))


def rebuild_toy(probe_type: str, config: dict, tokenwise: bool) -> ToyProbe:
    # A config saved before `trunk_width` existed rebuilds at the old default.
    return ToyProbe(ToyConfig(input_size=config["input_size"], num_labels=config["num_labels"], trunk_width=config.get("trunk_width", 8)))


def toy_settings(probe_type: str, options: dict) -> dict:
    return {"trunk_width": options.get("trunk_width", 8)}


def other_build(probe_type: str, fields: dict, tokenwise: bool) -> ToyProbe:
    return build_toy(probe_type, fields, tokenwise)


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_REGISTERED_PROBE_TYPES", dict(registry._REGISTERED_PROBE_TYPES))


def test_a_registered_type_builds_and_keeps_its_settings_to_itself() -> None:
    register_probe_type("toy", build_toy, rebuild=rebuild_toy, settings=toy_settings)

    toy_args = ProbeArguments(probe_type="toy", input_size=4, num_labels=3, trunk_width=16)
    probe = get_probe(toy_args)
    assert isinstance(probe, ToyProbe) and probe.config.trunk_width == 16 and probe.config.num_labels == 3
    assert probe(torch.zeros(2, 4)).shape == (2, 3)
    assert ProbeArguments(probe_type="toy").trunk_width == 8  # the type's default
    assert not hasattr(ProbeArguments(probe_type="mlp", trunk_width=16), "trunk_width")
    assert "toy" in registered_probe_types()


def test_a_saved_config_rebuilds_through_rebuild_or_else_build() -> None:
    register_probe_type("toy", build_toy, rebuild=rebuild_toy, settings=toy_settings)
    saved = get_probe(ProbeArguments(probe_type="toy", input_size=4, num_labels=2, trunk_width=12)).config.to_dict()

    rebuilt = rebuild_probe_from_saved_config("toy", False, saved)
    assert isinstance(rebuilt, ToyProbe) and rebuilt.config.trunk_width == 12
    older = {name: value for name, value in saved.items() if name != "trunk_width"}
    assert rebuild_probe_from_saved_config("toy", False, older).config.trunk_width == 8

    register_probe_type("toy_without_rebuild", build_toy, settings=toy_settings)
    assert rebuild_probe_from_saved_config("toy_without_rebuild", False, saved).config.trunk_width == 12


@pytest.mark.parametrize("name", ["mlp", "linear", "transformer", "lyra"])
def test_a_built_in_probe_cannot_be_replaced(name: str) -> None:
    with pytest.raises(ValueError, match="Protify probe type"):
        register_probe_type(name, build_toy)
    with pytest.raises(ValueError, match="Protify probe type"):
        register_external_probe(name, __name__, "ToyProbe", "ToyProbe", "ToyConfig")


def test_another_definition_cannot_take_a_registered_name_and_the_same_one_registers_again() -> None:
    register_probe_type("toy", build_toy, settings=toy_settings)
    register_probe_type("toy", build_toy, settings=toy_settings)  # a second import of the same module
    with pytest.raises(ValueError, match="already registered"):
        register_probe_type("toy", other_build)
    with pytest.raises(ValueError, match="already registered"):
        register_external_probe("toy", __name__, "ToyProbe", "ToyProbe", "ToyConfig")


def test_an_external_probe_registers_by_name_under_the_same_guards() -> None:
    register_external_probe("toy_external", __name__, "ToyProbe", "ToyProbe", "ToyConfig")
    register_external_probe("toy_external", __name__, "ToyProbe", "ToyProbe", "ToyConfig")
    probe = get_probe(ProbeArguments(probe_type="toy_external", input_size=4, num_labels=2))
    assert isinstance(probe, ToyProbe)
    with pytest.raises(ValueError, match="already registered"):
        register_external_probe("toy_external", __name__, "ToyProbe", "ToyProbe", "PretrainedConfig")


def test_an_unknown_type_names_the_registered_ones() -> None:
    register_probe_type("toy", build_toy)
    with pytest.raises(ValueError, match="toy"):
        get_probe(ProbeArguments(probe_type="nonexistent"))
    with pytest.raises(ValueError, match="Registered probe types"):
        rebuild_probe_from_saved_config("nonexistent", False, {"input_size": 4, "num_labels": 2})
