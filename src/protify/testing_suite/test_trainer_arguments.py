import pytest

try:
    from src.protify.probes.trainers import TrainerArguments
except ImportError:
    try:
        from protify.probes.trainers import TrainerArguments
    except ImportError:
        from ..probes.trainers import TrainerArguments


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
