"""Sweeping an estimator probe against a neural one in a single sweep.

A sweep over `probe_type` is how the two probe families get compared under one search,
which needs three things that used to be missing: the trial has to reach the estimator
path at all, estimator hyperparameters have to reach the learner's constructor rather
than being dropped by the parameter allowlist, and both families have to report their
metrics under the same names so the objective can be read by name.
"""

import pytest

from types import SimpleNamespace


try:
    from src.protify.hyperopt_utils import HyperoptModule
    from src.protify.probes.estimator_probe import with_split_metric_prefix
    from src.protify.probes.get_probe import ProbeArguments, get_probe
except ImportError:
    from protify.hyperopt_utils import HyperoptModule
    from protify.probes.estimator_probe import with_split_metric_prefix
    from protify.probes.get_probe import ProbeArguments, get_probe


def hyperopt_module(probe_type: str = "mlp") -> HyperoptModule:
    """A HyperoptModule whose MainProcess is only what apply_config actually touches."""
    main_process = SimpleNamespace(
        probe_args=ProbeArguments(probe_type=probe_type),
        trainer_args=SimpleNamespace(lr=1e-3, seed=42),
        embedding_args=SimpleNamespace(pooling_types=["mean"], hidden_state_index=-1),
        full_args=SimpleNamespace(random_pair_flipping=False),
    )
    return HyperoptModule(
        main_process=main_process,
        model_name="test-model",
        data_name="test-data",
        dataset=(None, None, None, 2, "singlelabel", False),
        emb_dict=None,
        sweep_config={},
        results_list=[],
    )


def test_estimator_hyperparameters_reach_the_overrides_dict():
    """max_depth is a learner constructor argument, not a ProbeArguments field."""
    module = hyperopt_module("xgboost")

    module.apply_config({"max_depth": 7, "n_estimators": 300, "colsample_bytree": 0.6})

    assert module.mp.probe_args.estimator_overrides == {
        "max_depth": 7,
        "n_estimators": 300,
        "colsample_bytree": 0.6,
    }
    assert not hasattr(module.mp.probe_args, "max_depth")


def test_integer_estimator_hyperparameters_are_not_left_as_floats():
    """A wandb sweep hands back floats, and xgboost rejects a float max_depth."""
    module = hyperopt_module("xgboost")

    module.apply_config({"max_depth": 7.0, "n_estimators": 300.0})

    assert module.mp.probe_args.estimator_overrides["max_depth"] == 7
    assert isinstance(module.mp.probe_args.estimator_overrides["max_depth"], int)
    assert isinstance(module.mp.probe_args.estimator_overrides["n_estimators"], int)


def test_a_later_trial_does_not_inherit_the_previous_overrides():
    """Trials that sweep probe_type see both families' parameters, so this has to reset."""
    module = hyperopt_module("xgboost")

    module.apply_config({"max_depth": 7})
    module.apply_config({"n_estimators": 100})

    assert module.mp.probe_args.estimator_overrides == {"n_estimators": 100}


def test_a_neural_trial_gets_an_empty_override_dict():
    module = hyperopt_module("mlp")

    module.apply_config({"hidden_size": 512, "dropout": 0.1})

    assert module.mp.probe_args.estimator_overrides == {}
    assert module.mp.probe_args.hidden_size == 512


def test_a_swept_probe_type_is_normalized_before_it_reaches_the_probe():
    """apply_config sets the attribute directly, bypassing the constructor's normalization."""
    module = hyperopt_module("mlp")

    module.apply_config({"probe_type": "linear"})

    assert module.mp.probe_args.probe_type == "mlp"


# The default sweep draws hidden_size in steps of 8. 72 is not a multiple of the default head size, 128.
UNEVEN_HIDDEN_SIZE = 72


@pytest.mark.parametrize(
    ("probe_type", "trial"),
    [("mlp", {}), ("lyra", {}), ("transformer", {"probe_type": "lyra"})],
    ids=["mlp", "lyra", "swept_to_lyra"],
)
def test_a_probe_without_attention_heads_takes_any_hidden_size(probe_type, trial):
    """The head-size check once ran for every probe, so MLP and Lyra trials refused most draws."""
    module = hyperopt_module(probe_type)

    module.apply_config({**trial, "hidden_size": UNEVEN_HIDDEN_SIZE})
    probe = get_probe(module.mp.probe_args)

    assert probe.config.hidden_size == UNEVEN_HIDDEN_SIZE


@pytest.mark.parametrize(
    ("probe_type", "trial"),
    [("transformer", {}), ("mlp", {"probe_type": "transformer"})],
    ids=["transformer", "swept_to_transformer"],
)
def test_the_transformer_probe_still_needs_hidden_size_to_divide_into_heads(probe_type, trial):
    module = hyperopt_module(probe_type)

    with pytest.raises(AssertionError, match="not divisible by head_size"):
        module.apply_config({**trial, "hidden_size": UNEVEN_HIDDEN_SIZE})


@pytest.mark.parametrize(
    "raw, split, expected",
    [
        ({"mcc": 0.5}, "eval", {"eval_mcc": 0.5}),
        ({"mcc": 0.5}, "test", {"test_mcc": 0.5}),
        ({"spearman_rho": -0.2, "r_squared": 0.1}, "test", {"test_spearman_rho": -0.2, "test_r_squared": 0.1}),
        ({}, "eval", {}),
    ],
)
def test_estimator_metrics_are_renamed_onto_the_trainer_convention(raw, split, expected):
    """The Trainer names a metric after the split it came from, and so must the estimators.

    evaluate() produces eval_ and predict() produces test_. A sweep reads its objective
    from the validation metrics while the results table is written from the test metrics,
    so a single prefix for both would leave one of them unreadable.
    """
    assert with_split_metric_prefix(raw, split) == expected


def test_the_sweep_metric_defaults_are_emitted_by_both_probe_families():
    """eval_loss cannot be the default, because an estimator probe never reports a loss."""
    try:
        from src.protify.main import DEFAULT_SWEEP_METRIC_CLS, DEFAULT_SWEEP_METRIC_REG
    except ImportError:
        from protify.main import DEFAULT_SWEEP_METRIC_CLS, DEFAULT_SWEEP_METRIC_REG

    assert DEFAULT_SWEEP_METRIC_CLS == "eval_mcc"
    assert DEFAULT_SWEEP_METRIC_REG == "eval_spearman_rho"
    # Both names must survive the prefixing an estimator run applies to its raw metrics.
    prefixed = with_split_metric_prefix({"mcc": 0.0, "spearman_rho": 0.0}, "eval")
    assert DEFAULT_SWEEP_METRIC_CLS in prefixed
    assert DEFAULT_SWEEP_METRIC_REG in prefixed


def test_probe_arguments_default_pooling_is_a_usable_value():
    """The hand-written __init__ never resolves a dataclasses.field default.

    Left as written, ProbeArguments().pooling_types was a Field object rather than a list
    of pooling names, which the transformer probe would have read as a pooling spec. Unset is
    None, which each probe reads as its own default.
    """
    assert ProbeArguments().pooling_types is None
    assert ProbeArguments(probe_pooling_types=["max"]).pooling_types == ["max"]


def test_probe_arguments_stay_json_serializable():
    """get_probe splats this whole object into the probe config, which is saved as JSON.

    Anything unserializable parked here fails at the first Trainer checkpoint save, deep
    inside transformers and far from the assignment that caused it. The fitted feature
    selection is the obvious candidate, which is why it travels to the exporter as an
    argument instead.
    """
    import json

    for probe_type in ('mlp', 'transformer', 'xgboost'):
        arguments = ProbeArguments(probe_type=probe_type)
        arguments.estimator_overrides = {'max_depth': 7}
        json.dumps(arguments.__dict__)
