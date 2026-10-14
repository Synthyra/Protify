"""`--probe_pooling_types` reaches the transformer and Lyra probes, and a saved probe keeps it.

ProbeArguments stores the flag as `pooling_types`, the name each probe config stores and saves.
Each config once set its own default after `super().__init__` had stored the value passed, so
`get_probe`, a hyperparameter sweep, and `from_pretrained` all built the default pooling whatever
was asked for, and `bom` pooling never ran. Leaving the flag unset keeps each probe's default:
mean and cls for the transformer probe, mean for Lyra, whose classifier reads one pooled vector.
"""

import json
import logging
import sys
import pytest
import torch
import yaml

from pathlib import Path
from types import SimpleNamespace

from foundry.checkpoint_loading import uninitialized_memory_as_nan
from protify.cloud_cli import _build_cloud_config_from_args
from protify.hyperopt_utils import HyperoptModule
from protify.main import parse_arguments
from protify.probes.get_probe import ProbeArguments, get_probe, rebuild_probe_from_saved_config
from protify.probes.transformer_probe import BoMPooling


BATCH = 2  # b
LENGTH = 6  # l, residues per padded row
INPUT_SIZE = 16  # d_in, the embedding width a probe reads
HIDDEN_SIZE = 32  # d
NUM_LABELS = 3  # c

# Each probe family as get_probe names it: (probe_type, tokenwise).
FAMILIES = {
    "transformer_sequence": ("transformer", False),
    "transformer_token": ("transformer", True),
    "lyra_sequence": ("lyra", False),
    "lyra_token": ("lyra", True),
}
# The pooling each probe type builds when the flag is unset, and one it accepts other than that.
DEFAULT_POOLING = {"transformer": ["mean", "cls"], "lyra": ["mean"]}
REQUESTED_POOLING = {"transformer": ["max", "var"], "lyra": ["max"]}

# A probe saved with pooling other than its default. Two types keep the transformer sequence
# probe's classifier width, so a reload with the default pooling loads cleanly and only the logits
# differ. One type changes that width, so a reload with the default cannot load at all.
SAVED_POOLING = [
    ("transformer_sequence", ["max", "var"]),
    ("transformer_sequence", ["max"]),
    ("transformer_token", ["max"]),
    ("lyra_sequence", ["max"]),
    ("lyra_token", ["max"]),
]
SAVED_POOLING_IDS = [f"{family}-{'-'.join(pooling_types)}" for family, pooling_types in SAVED_POOLING]


def probe_arguments(probe_type: str, tokenwise: bool, **settings: object) -> ProbeArguments:
    """Arguments for a small CPU probe. The pooling stays unset unless `settings` names it."""
    return ProbeArguments(
        probe_type=probe_type,
        tokenwise=tokenwise,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_labels=NUM_LABELS,
        head_size=8,
        classifier_size=24,
        dropout=0.0,
        transformer_dropout=0.0,
        classifier_dropout=0.0,
        rotary=False,
        attention_backend="sdpa",
        **settings,
    )


def residue_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    embeddings = torch.randn(BATCH, LENGTH, INPUT_SIZE, generator=generator)  # (b, l, d_in)
    attention_mask = torch.ones(BATCH, LENGTH, dtype=torch.long)  # (b, l)
    attention_mask[1, 4:] = 0  # the second row holds four residues, then padding
    return embeddings, attention_mask  # (b, l, d_in), (b, l)


def saved_probe(family: str, pooling_types: list[str]) -> torch.nn.Module:
    probe_type, tokenwise = FAMILIES[family]
    torch.manual_seed(0)
    return get_probe(probe_arguments(probe_type, tokenwise, probe_pooling_types=pooling_types)).eval()


@pytest.mark.parametrize("family", list(FAMILIES))
def test_the_flag_reaches_the_built_probe(family: str) -> None:
    probe_type, tokenwise = FAMILIES[family]
    requested = REQUESTED_POOLING[probe_type]

    probe = get_probe(probe_arguments(probe_type, tokenwise, probe_pooling_types=requested))

    assert probe.config.pooling_types == requested
    if not tokenwise:
        # A token probe classifies every residue, so only the sequence probes build a pooler.
        assert probe.pooler.pooling_types == requested


@pytest.mark.parametrize("family", list(FAMILIES))
def test_unset_pooling_keeps_the_probe_default(family: str) -> None:
    """A run that leaves the flag unset builds what every earlier run built.

    It also passes an unset value through get_probe's `bom` check.
    """
    probe_type, tokenwise = FAMILIES[family]
    arguments = probe_arguments(probe_type, tokenwise)

    probe = get_probe(arguments)

    assert arguments.pooling_types is None
    assert probe.config.pooling_types == DEFAULT_POOLING[probe_type]


@pytest.mark.parametrize(("family", "pooling_types"), SAVED_POOLING, ids=SAVED_POOLING_IDS)
class TestSavedPooling:
    def test_from_pretrained_keeps_it(self, family: str, pooling_types: list[str], tmp_path) -> None:
        saved = saved_probe(family, pooling_types)
        saved.save_pretrained(str(tmp_path))
        embeddings, attention_mask = residue_embeddings()  # (b, l, d_in), (b, l)

        with uninitialized_memory_as_nan():
            loaded = type(saved).from_pretrained(str(tmp_path)).eval()
        with torch.inference_mode():
            expected = saved(embeddings=embeddings, attention_mask=attention_mask).logits  # (b, c) or (b, l, c)
            actual = loaded(embeddings=embeddings, attention_mask=attention_mask).logits  # (b, c) or (b, l, c)

        assert loaded.config.pooling_types == pooling_types
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rebuild_from_the_saved_config_keeps_it(self, family: str, pooling_types: list[str]) -> None:
        probe_type, tokenwise = FAMILIES[family]
        saved = saved_probe(family, pooling_types)
        embeddings, attention_mask = residue_embeddings()  # (b, l, d_in), (b, l)

        rebuilt = rebuild_probe_from_saved_config(probe_type, tokenwise, saved.config.to_dict())
        rebuilt.load_state_dict(saved.state_dict())  # strict: every tensor has its saved shape
        with torch.inference_mode():
            expected = saved(embeddings=embeddings, attention_mask=attention_mask).logits  # (b, c) or (b, l, c)
            actual = rebuilt.eval()(embeddings=embeddings, attention_mask=attention_mask).logits  # (b, c) or (b, l, c)

        assert rebuilt.config.pooling_types == pooling_types
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_bom_pooling_takes_effect_in_the_transformer_sequence_probe() -> None:
    torch.manual_seed(0)
    probe = get_probe(probe_arguments("transformer", False, probe_pooling_types=["mean", "bom"], bom_k=3)).eval()
    embeddings, attention_mask = residue_embeddings()  # (b, l, d_in), (b, l)

    with torch.no_grad():
        logits = probe(embeddings=embeddings, attention_mask=attention_mask).logits  # (b, c)
        # With BoM's output projection zeroed, its half of the classifier input is zero.
        probe.bom.out_proj.weight.zero_()
        logits_without_bom = probe(embeddings=embeddings, attention_mask=attention_mask).logits  # (b, c)

    assert isinstance(probe.bom, BoMPooling)
    assert probe.pooler.pooling_types == ["mean"]
    assert probe.classifier[0].normalized_shape == (2 * HIDDEN_SIZE,)
    assert logits.shape == (BATCH, NUM_LABELS)
    assert not torch.equal(logits, logits_without_bom)


@pytest.mark.parametrize("tokenwise", [False, True], ids=["sequence", "token"])
def test_lyra_refuses_two_pooling_types_when_it_is_configured(tokenwise: bool) -> None:
    """Lyra's classifier reads one pooled vector, so a second type would fail only at the first batch."""
    with pytest.raises(ValueError, match="Lyra takes exactly one pooling type"):
        get_probe(probe_arguments("lyra", tokenwise, probe_pooling_types=["mean", "max"]))


@pytest.mark.parametrize(
    ("probe_type", "swept", "expected"),
    [("transformer", "max-var", ["max", "var"]), ("lyra", "max", ["max"])],
)
def test_a_swept_pooling_reaches_the_probe_and_the_next_trial_resets_it(
    probe_type: str, swept: str, expected: list[str]
) -> None:
    main_process = SimpleNamespace(
        probe_args=probe_arguments(probe_type, False),
        trainer_args=SimpleNamespace(),
        embedding_args=SimpleNamespace(pooling_types=["mean"], hidden_state_index=-1),
        full_args=SimpleNamespace(random_pair_flipping=False),
    )
    module = HyperoptModule(
        main_process, "test-model", "test-data", dataset=None, emb_dict=None, sweep_config={}, results_list=[]
    )

    module.apply_config({"probe_pooling_types": swept})
    swept_probe = get_probe(main_process.probe_args)
    module.apply_config({})
    next_probe = get_probe(main_process.probe_args)

    assert swept_probe.config.pooling_types == expected
    assert next_probe.config.pooling_types == DEFAULT_POOLING[probe_type]


SHIPPED_SWEEP = Path(__file__).resolve().parents[1] / "yamls" / "sweep.yaml"
# The shipped sweep's pooling values plus two `bom` values, and what each probe family may be offered
# of them: `bom` runs only in the transformer sequence probe, and Lyra takes one type.
SWEPT_POOLING = ["mean", "cls", "mean-cls", "mean-var", "bom", "mean-bom"]
OFFERED_POOLING = {
    "transformer_sequence": SWEPT_POOLING,
    "transformer_token": ["mean", "cls", "mean-cls", "mean-var"],
    "lyra_sequence": ["mean", "cls"],
    "lyra_token": ["mean", "cls"],
}


class SweepStarted(Exception):
    """Raised by the stand-in `sweep_search`, so a test stops before any trial runs."""


def start_sweep(
    probe_args: ProbeArguments,
    parameters: dict[str, dict[str, object]],
    tmp_path,
    monkeypatch,
) -> tuple[dict[str, dict[str, object]], SimpleNamespace]:
    """Run `run_wandb_hyperopt` up to `sweep_search`, and return the parameters it would sweep."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(yaml.safe_dump({"parameters": parameters}), encoding="utf-8")
    main_process = SimpleNamespace(
        logger=logging.getLogger(__name__),
        full_args=SimpleNamespace(
            sweep_config_path=str(sweep_path),
            sweep_method="random",
            sweep_metric_cls="eval_mcc",
            sweep_metric_reg="eval_spearman_rho",
            sweep_goal="maximize",
            sweep_name=None,
            full_finetuning=False,
            wandb_project="test-project",
            wandb_entity=None,
        ),
        probe_args=probe_args,
        trainer_args=SimpleNamespace(seed=42),
        embedding_args=SimpleNamespace(embedding_save_dir=str(tmp_path), pooling_types=["mean"], hidden_state_index=-1),
        model_args=SimpleNamespace(model_names=["test-model"]),
        datasets={"test-data": (None, None, None, NUM_LABELS, "singlelabel", False)},
        all_seqs=["MKTAYIAK"],
        _sql=True,
        _full=False,
        _max_length=LENGTH,
        _embedding_store=lambda model_name: str(tmp_path / model_name),
        get_embedding_dim_sql=lambda save_path, sequence, tokenizer: INPUT_SIZE,
    )
    started: dict[str, dict[str, object]] = {}

    def sweep(sweep: dict[str, object], seed: int) -> None:
        started.update(sweep)
        raise SweepStarted

    hyperopt_module = sys.modules[HyperoptModule.__module__]
    monkeypatch.setattr(hyperopt_module, "get_tokenizer", lambda model_name: None)
    monkeypatch.setattr(hyperopt_module, "sweep_search", sweep)
    with pytest.raises(SweepStarted):
        HyperoptModule.run_wandb_hyperopt(main_process)

    return started["parameters"], main_process


@pytest.mark.parametrize("family", list(FAMILIES))
def test_a_sweep_offers_each_probe_only_pooling_it_takes(family: str, tmp_path, monkeypatch) -> None:
    """Every pooling value a sweep offers builds the probe, so no trial stops at get_probe.

    The sweep once offered Lyra the transformer probe's two-type values, which LyraConfig refuses.
    """
    probe_type, tokenwise = FAMILIES[family]
    parameters = {"lr": {"values": [1e-4]}, "probe_pooling_types": {"values": SWEPT_POOLING}}

    swept, main_process = start_sweep(probe_arguments(probe_type, tokenwise), parameters, tmp_path, monkeypatch)
    module = HyperoptModule(
        main_process, "test-model", "test-data", dataset=None, emb_dict=None, sweep_config={}, results_list=[]
    )
    for pooling in swept["probe_pooling_types"]["values"]:
        module.apply_config({"probe_pooling_types": pooling})
        get_probe(main_process.probe_args)

    assert swept["probe_pooling_types"] == {"values": OFFERED_POOLING[family]}
    assert swept["lr"] == {"values": [1e-4]}


def test_a_sweep_over_probe_types_offers_only_pooling_every_one_takes(tmp_path, monkeypatch) -> None:
    """A trial picks its probe type and its pooling independently, so each value must suit both types."""
    parameters = {
        "probe_type": {"values": ["transformer", "lyra"]},
        "probe_pooling_types": {"values": ["mean", "mean-cls", "max"], "probabilities": [0.2, 0.5, 0.3]},
    }

    swept, _ = start_sweep(probe_arguments("transformer", False), parameters, tmp_path, monkeypatch)

    assert swept["probe_pooling_types"]["values"] == ["mean", "max"]
    assert swept["probe_pooling_types"]["probabilities"] == pytest.approx([0.4, 0.6])


def test_a_lyra_sweep_without_one_type_values_keeps_the_configured_pooling(tmp_path, monkeypatch) -> None:
    parameters = {"lr": {"values": [1e-4]}, "probe_pooling_types": {"values": ["mean-cls", "mean-var"]}}

    swept, _ = start_sweep(probe_arguments("lyra", False), parameters, tmp_path, monkeypatch)

    assert "probe_pooling_types" not in swept
    assert swept["lr"] == {"values": [1e-4]}


def test_the_shipped_transformer_sweep_offers_only_hidden_sizes_its_heads_divide(tmp_path, monkeypatch) -> None:
    """The shipped sweep draws hidden_size in steps of 8, and a transformer probe's heads are 128 wide.

    The sweep once offered every step, so apply_config refused 953 of its 1017 hidden sizes.
    """
    shipped = yaml.safe_load(SHIPPED_SWEEP.read_text(encoding="utf-8"))["parameters"]["hidden_size"]
    arguments = ProbeArguments(probe_type="transformer")

    swept, main_process = start_sweep(arguments, {"hidden_size": shipped}, tmp_path, monkeypatch)
    offered = swept["hidden_size"]
    # q_uniform draws round(X / q) * q for X uniform in [min, max], so min and max bound every draw.
    hidden_sizes = range(offered["min"], offered["max"] + 1, offered["q"])
    module = HyperoptModule(
        main_process, "test-model", "test-data", dataset=None, emb_dict=None, sweep_config={}, results_list=[]
    )
    for hidden_size in hidden_sizes:
        module.apply_config({"hidden_size": hidden_size})

    assert offered["distribution"] == "q_uniform"
    assert offered["min"] % offered["q"] == 0
    assert list(hidden_sizes) == [
        size for size in range(shipped["min"], shipped["max"] + 1, shipped["q"]) if size % arguments.head_size == 0
    ]


@pytest.mark.parametrize(
    ("probe_type", "parameters", "offered"),
    [
        ("transformer", {"hidden_size": {"values": [96, 128, 200, 256]}}, {"values": [128, 256]}),
        (
            "transformer",
            {"hidden_size": {"distribution": "int_uniform", "min": 100, "max": 1000}},
            {"distribution": "q_uniform", "min": 128, "max": 896, "q": 128},
        ),
        ("transformer", {"hidden_size": {"values": [192, 256, 384]}, "head_size": {"values": [64, 96]}}, {"values": [192, 384]}),
        ("mlp", {"probe_type": {"values": ["mlp", "transformer"]}, "hidden_size": {"values": [72, 128]}}, {"values": [128]}),
        ("transformer", {"hidden_size": {"values": [72, 80]}}, None),
        ("mlp", {"hidden_size": {"values": [72, 128]}}, {"values": [72, 128]}),
    ],
    ids=["values", "int_uniform", "swept_head_sizes", "swept_probe_types", "none_left", "mlp"],
)
def test_a_sweep_offers_hidden_sizes_every_head_size_divides(
    probe_type: str,
    parameters: dict[str, dict[str, object]],
    offered: dict[str, object] | None,
    tmp_path,
    monkeypatch,
) -> None:
    """A sweep that can land on the transformer probe offers hidden sizes that every head size it can draw divides."""
    swept, _ = start_sweep(ProbeArguments(probe_type=probe_type), parameters, tmp_path, monkeypatch)

    assert swept.get("hidden_size") == offered


@pytest.mark.parametrize(
    ("flag", "expected"),
    [([], None), (["--probe_pooling_types", "max", "var"], ["max", "var"])],
    ids=["unset", "set"],
)
def test_the_command_line_and_a_cloud_run_pass_the_pooling_only_when_set(
    flag: list[str], expected: list[str] | None, monkeypatch
) -> None:
    monkeypatch.setattr(sys.modules[parse_arguments.__module__], "load_secrets_env", lambda **_: [])
    monkeypatch.setattr(sys, "argv", ["main.py", "--data_names", "DeepLoc-2", *flag])

    arguments = parse_arguments()
    # A cloud run sends these arguments as JSON, and the backend builds the same MainProcess from them.
    cloud_config = json.loads(json.dumps(_build_cloud_config_from_args(arguments)))

    assert ProbeArguments(**vars(arguments)).pooling_types == expected
    assert ProbeArguments(**cloud_config).pooling_types == expected


@pytest.mark.parametrize(
    ("setting", "expected"),
    [("", None), ("probe_pooling_types: null", None), ("probe_pooling_types: [max]", ["max"])],
    ids=["absent", "null", "set"],
)
def test_a_yaml_config_passes_the_pooling_only_when_set(
    setting: str, expected: list[str] | None, tmp_path, monkeypatch
) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(f"data_names:\n  - DeepLoc-2\n{setting}\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[parse_arguments.__module__], "load_secrets_env", lambda **_: [])
    monkeypatch.setattr(sys, "argv", ["main.py", "--yaml_path", str(yaml_path)])

    arguments = parse_arguments()

    assert ProbeArguments(**vars(arguments)).pooling_types == expected
