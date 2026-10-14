"""W&B runs held to the workspace convention: one run per training job, and a sweep run whose trials are table rows.

A stand-in `wandb` module records what each run was asked to do, so nothing reaches the network.
"""

import sys
import types
import optuna
import pytest

from pathlib import Path
from types import SimpleNamespace


try:
    from src.protify import hyperopt_utils, run_tracking
except ImportError:
    from protify import hyperopt_utils, run_tracking


class FakeRun:
    def __init__(self, **options: object) -> None:
        self.options, self.id, self.url = options, str(options["id"]), None
        self.logged: list[tuple[dict[str, object], int | None]] = []
        self.summary: dict[str, float] = {}
        self.exit_code: int | None = None

    def log(self, values: dict[str, object], step: int | None = None) -> None:
        self.logged.append((values, step))

    def define_metric(self, name: str, step_metric: str | None = None) -> None:
        del name, step_metric

    def finish(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code


@pytest.fixture
def wandb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> types.ModuleType:
    module = types.ModuleType("wandb")
    module.runs = []

    def init(**options: object) -> FakeRun:
        module.runs.append(FakeRun(**options))
        return module.runs[-1]

    module.init = init
    module.Settings = lambda **settings: settings
    module.Table = lambda columns, data: {"columns": columns, "data": data}  # noqa: vague-name  (wandb.Table's keyword)
    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.setenv("WANDB_API_KEY", "placeholder-not-a-credential")
    monkeypatch.setenv("FOUNDRY_SECRETS", str(tmp_path / "absent.secrets.env"))
    # `ws test` keeps W&B offline through these; the stand-in already does, and the tests read the mode a job chose.
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WS_PILOT", raising=False)
    return module


def trainer_args(tmp_path: Path, report_to: str = "wandb") -> SimpleNamespace:
    return SimpleNamespace(report_to=report_to, log_dir=str(tmp_path / "logs"), wandb_project="Protify", wandb_entity=None)


class Job:
    """A `TrainerMixin` stand-in whose trainer method trains two seeds and reports their means."""

    def __init__(self, arguments: SimpleNamespace) -> None:
        self.trainer_args, self.probe_args, self.callbacks = arguments, SimpleNamespace(probe_type="mlp"), []

    @run_tracking.reports_training_job
    def trainer_probe(self, *, model_name: str, data_name: str) -> tuple[None, dict[str, object], dict[str, object], None, None]:
        self.callbacks = [run_tracking.trainer_callbacks() for _seed in range(2)]
        valid = {"eval_mcc": "0.5000±0.1000", "eval_mcc_mean": 0.5, "eval_mcc_std": 0.1}
        return None, valid, {"test_mcc_mean": 0.4, "test_mcc_std": 0.1}, None, None


def test_report_to_wandb_makes_one_run_per_job_charting_the_first_seed_and_the_mean(tmp_path: Path, wandb: types.ModuleType) -> None:
    job = Job(trainer_args(tmp_path))
    job.trainer_probe(model_name="ESMC-300", data_name="DeepLoc-2")

    (run,) = wandb.runs
    assert run.options["mode"] == "online" and run.options["job_type"] == "train"
    assert [len(callbacks) for callbacks in job.callbacks] == [1, 0], "a second seed would restart the curve"
    assert run.summary["val/mcc"] == 0.5 and run.summary["test/mcc"] == 0.4
    assert not any(name.endswith("_std") for name in run.summary)


def test_no_run_unless_asked_and_none_inside_an_open_sweep(tmp_path: Path, wandb: types.ModuleType) -> None:
    Job(trainer_args(tmp_path, report_to="none")).trainer_probe(model_name="ESMC-300", data_name="DeepLoc-2")
    assert wandb.runs == []

    with run_tracking.training_job(trainer_args(tmp_path), "ESMC-300", "DeepLoc-2", {}, force=True) as sweep:
        with run_tracking.curves_closed():
            Job(trainer_args(tmp_path)).trainer_probe(model_name="ESMC-300", data_name="DeepLoc-2")
        assert sweep.curve_open, "the final fit still charts the sweep run's curve"
    assert len(wandb.runs) == 1


def test_a_sweep_entry_draws_within_its_distribution() -> None:
    trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0)).ask()
    entries = {
        "lr": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
        "hidden_size": {"distribution": "q_uniform", "min": 64, "max": 512, "q": 64},
        "n_layers": {"distribution": "int_uniform", "min": 1, "max": 4},
        "pre_ln": {"values": [True, False]},
        "dropout": {"value": 0.1},
    }
    drawn = {name: hyperopt_utils.suggest(trial, name, spec) for name, spec in entries.items()}

    assert 1e-6 <= drawn["lr"] <= 1e-2 and drawn["hidden_size"] % 64 == 0 and drawn["n_layers"] in range(1, 5)
    assert drawn["pre_ln"] in (True, False) and drawn["dropout"] == 0.1


def test_a_grid_sweep_refuses_a_range() -> None:
    sweep = {"method": "grid", "metric": {"goal": "maximize"}, "parameters": {"lr": {"min": 1e-4, "max": 1e-2}}}
    with pytest.raises(ValueError, match="list its values"):
        hyperopt_utils.sweep_search(sweep, seed=42)
