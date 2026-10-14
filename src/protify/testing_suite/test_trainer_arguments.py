import pytest

from types import SimpleNamespace

from protify.probes.fit_hooks import FitHooks
from protify.probes.trainers import TrainerArguments


def test_trainer_arguments_phase_lrs_scheduler_and_optimizer(tmp_path) -> None:
    args = TrainerArguments(
        model_save_dir=str(tmp_path),
        num_epochs=3,
        base_num_epochs=5,
        lr=1e-4,
        probe_lr=2e-4,
        base_lr=3e-5,
        lr_scheduler="linear",
        optimizer="sgd",
        torch_compile=False,
    )

    probe_args = args(probe=True)
    base_args = args(probe=False)

    assert probe_args.learning_rate == pytest.approx(2e-4)
    assert base_args.learning_rate == pytest.approx(3e-5)
    assert probe_args.num_train_epochs == pytest.approx(3)
    assert base_args.num_train_epochs == pytest.approx(5)
    assert "linear" in str(probe_args.lr_scheduler_type).lower()
    assert "sgd" in str(probe_args.optim).lower()


def test_trainer_arguments_phase_lrs_fall_back_to_shared_lr(tmp_path) -> None:
    args = TrainerArguments(
        model_save_dir=str(tmp_path),
        num_epochs=2,
        lr=7e-5,
        probe_lr=None,
        base_lr=None,
        torch_compile=False,
    )

    assert args(probe=True).learning_rate == pytest.approx(7e-5)
    assert args(probe=False).learning_rate == pytest.approx(7e-5)


def test_probe_controls_are_effective_and_do_not_change_base_phase(tmp_path):
    arguments = TrainerArguments(
        model_save_dir=str(tmp_path), torch_compile=False, probe_max_steps=9,
        probe_warmup_steps=2, probe_max_grad_norm=0.25, probe_eval_strategy="steps", probe_eval_steps=3,
        early_stopping_threshold=0.02, adam_beta2=0.98, eval_accumulation_steps="7",
    )
    probe, base = arguments(), arguments(probe=False)
    assert (probe.max_steps, probe.warmup_steps, probe.max_grad_norm, probe.eval_steps, probe.save_steps) == (9, 2, 0.25, 3, 3)
    assert (base.max_steps, base.warmup_steps, base.max_grad_norm) == (-1, 1000, 1.0)
    assert (probe.adam_beta2, probe.eval_accumulation_steps) == (0.98, 7)
    assert arguments.early_stopping()["early_stopping_threshold"] == 0.02


@pytest.mark.parametrize("setting,value", [
    ("probe_max_steps", 0), ("probe_warmup_steps", -1), ("probe_max_grad_norm", float("nan")),
    ("probe_eval_steps", 0), ("probe_precision", "auto"), ("probe_eval_strategy", "never"),
    ("early_stopping_threshold", -0.1), ("adam_beta2", 1.0), ("adam_epsilon", 0),
    ("eval_accumulation_steps", "0"), ("probe_batch_size", True),
])
def test_invalid_recipe_controls_fail_before_training(tmp_path, setting, value):
    arguments = TrainerArguments(model_save_dir=str(tmp_path), torch_compile=False, **{setting: value})
    with pytest.raises(ValueError, match=setting):
        arguments()


def test_auto_validation_cadence_uses_the_hooks_draws_and_observed_loss(tmp_path):
    class ManyDraws(FitHooks):
        def draws_per_epoch(self, train_dataset):
            return 300000

    class Rows:
        label_masks = [True] * 10
        fit_hooks = ManyDraws()

        def __len__(self):
            return 10

    arguments = TrainerArguments(model_save_dir=str(tmp_path), torch_compile=False, probe_batch_size=64, probe_grad_accum=2)
    effective = arguments.for_datasets(SimpleNamespace(num_labels=1), Rows(), Rows(), Rows())
    assert effective.eval_strategy.value == "steps"
    assert effective.eval_steps == effective.save_steps == 781
    assert effective.metric_for_best_model == "eval_observed_loss"
    assert effective.label_names == ["labels", "label_mask"]
    Rows.fit_hooks = None  # the default hooks read the split once per epoch
    assert arguments.for_datasets(SimpleNamespace(num_labels=1), Rows(), Rows(), Rows()).eval_strategy.value == "epoch"


@pytest.mark.parametrize("precision,fp16,bf16", [("float32", False, False), ("float16", True, False), ("bfloat16", False, True)])
def test_precision_is_forwarded_explicitly(tmp_path, monkeypatch, precision, fp16, bf16):
    import sys

    trainer_module = sys.modules[TrainerArguments.__module__]
    monkeypatch.setattr(trainer_module, "TrainingArguments", lambda **kwargs: kwargs)
    arguments = TrainerArguments(model_save_dir=str(tmp_path), probe_precision=precision)
    effective = arguments()
    assert (effective["fp16"], effective["bf16"]) == (fp16, bf16)
    assert (arguments(probe=False)["fp16"], arguments(probe=False)["bf16"]) == (False, False)


def test_trainer_arguments_rejects_invalid_parallel_probe_batch_mode(tmp_path) -> None:
    with pytest.raises(AssertionError, match="parallel_probe_batch_mode"):
        TrainerArguments(
            model_save_dir=str(tmp_path),
            parallel_probe_batch_mode="different",
            torch_compile=False,
        )


def test_trainer_arguments_rejects_invalid_parallel_probe_index_strategy(tmp_path) -> None:
    with pytest.raises(AssertionError, match="parallel_probe_index_strategy"):
        TrainerArguments(
            model_save_dir=str(tmp_path),
            parallel_probe_index_strategy="different",
            torch_compile=False,
        )


def test_trainer_arguments_rejects_invalid_parallel_probe_max_group_size(tmp_path) -> None:
    with pytest.raises(AssertionError, match="parallel_probe_max_group_size"):
        TrainerArguments(
            model_save_dir=str(tmp_path),
            parallel_probe_max_group_size=0,
            torch_compile=False,
        )


def test_trainer_arguments_rejects_invalid_parallel_probe_training_state_budget(tmp_path) -> None:
    with pytest.raises(AssertionError, match="parallel_probe_training_state_budget_gb"):
        TrainerArguments(
            model_save_dir=str(tmp_path),
            parallel_probe_training_state_budget_gb=0.0,
            torch_compile=False,
        )


def test_trainer_arguments_rejects_invalid_parallel_probe_estimated_peak_budget(tmp_path) -> None:
    with pytest.raises(AssertionError, match="parallel_probe_estimated_peak_budget_gb"):
        TrainerArguments(
            model_save_dir=str(tmp_path),
            parallel_probe_estimated_peak_budget_gb=0.0,
            torch_compile=False,
        )


def test_trainer_arguments_rejects_invalid_parallel_probe_max_grad_norm(tmp_path) -> None:
    with pytest.raises(AssertionError, match="parallel_probe_max_grad_norm"):
        TrainerArguments(
            model_save_dir=str(tmp_path),
            parallel_probe_max_grad_norm=-0.1,
            torch_compile=False,
        )


def test_trainer_arguments_rejects_invalid_parallel_probe_grad_clip_mode(tmp_path) -> None:
    with pytest.raises(AssertionError, match="parallel_probe_grad_clip_mode"):
        TrainerArguments(
            model_save_dir=str(tmp_path),
            parallel_probe_grad_clip_mode="per_model",
            torch_compile=False,
        )


def test_trainer_arguments_rejects_invalid_parallel_probe_ensemble_average_mode(tmp_path) -> None:
    with pytest.raises(AssertionError, match="parallel_probe_ensemble_average_mode"):
        TrainerArguments(
            model_save_dir=str(tmp_path),
            parallel_probe_ensemble_average_mode="votes",
            torch_compile=False,
        )
