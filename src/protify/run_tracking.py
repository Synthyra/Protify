"""Protify's Weights & Biases runs, held to the workspace's W&B convention by `foundry.logging.Run`.

`--report_to wandb` opens one run per training job: one model's probe, fine-tune or hybrid on one
dataset. The Hugging Face Trainer reports through `Run.trainer_callback()` with `report_to="none"`,
as `train/`, `val/` and `test/` curves, and the job's final validation and test metrics become the
run's summary. Only the job's first Trainer charts its curve: further seeds restart the step count,
and a curve per seed is a key per seed, so their metrics reach W&B as the job's mean.

A sweep (`--use_wandb_hyperopt`) is one run too: its trials are rows of one `tuning/trials` table,
and the final fit with the best configuration charts the curve. A job inside it reports to the
sweep's run instead of opening its own.

A run needs `WANDB_API_KEY` in the environment or in a `.secrets.env` that `foundry.secrets_env`
finds; every record also stays under `--log_dir`.
"""

from __future__ import annotations

import contextlib
import functools
import re
import uuid

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from foundry.logging import Run, trainer_metrics


T = TypeVar("T")
# What a run directory's name may hold; model names carry `/` and dataset names spaces.
UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class Job:
    """An open run, and whether its curve is still free for the next Trainer to chart."""

    run: Run
    curve_open: bool = True


_jobs: list[Job] = []


def open_job() -> Job | None:
    """The run the current training job reports to, if any."""
    return _jobs[-1] if _jobs else None


@contextlib.contextmanager
def training_job(trainer_args: Any, model_name: str, data_name: str, config: Mapping[str, Any], *,
                 name: str | None = None, force: bool = False) -> Iterator[Job | None]:
    """Open the job's run under `--report_to wandb` (or `force`), unless a sweep's run is already open."""
    if open_job() is not None or not (force or trainer_args.report_to == "wandb"):
        yield None
        return
    name = name or f"{data_name}-{model_name}"
    # Protify never resumes a job, and a directory holding `run.json` would resume that run, so each job gets its own.
    directory = Path(trainer_args.log_dir) / "wandb" / f"{UNSAFE.sub('_', name)}-{uuid.uuid4().hex[:8]}"
    with Run(directory, config, project=trainer_args.wandb_project, entity=trainer_args.wandb_entity, job_type="train",
             name=name, tags=(f"model:{model_name}", f"data:{data_name}")) as run:
        _jobs.append(Job(run))
        try:
            yield _jobs[-1]
        finally:
            _jobs.pop()


def trainer_callbacks() -> list[Any]:
    """The open run's Trainer callback, for the job's first Trainer only."""
    job = open_job()
    if job is None or not job.curve_open:
        return []
    job.curve_open = False
    return [job.run.trainer_callback()]


@contextlib.contextmanager
def curves_closed() -> Iterator[None]:
    """Keep Trainers inside the block off the open run, as a sweep's trials are."""
    job = open_job()
    if job is None:
        yield
        return
    curve_open, job.curve_open = job.curve_open, False
    try:
        yield
    finally:
        job.curve_open = curve_open


def job_summary(valid_metrics: Mapping[str, Any] | None, test_metrics: Mapping[str, Any] | None) -> dict[str, float]:
    """A job's final metrics as `val/` and `test/` scalars; a multi-seed job contributes its means."""
    merged = {**(valid_metrics or {}), **(test_metrics or {})}
    means = {name.removesuffix("_mean"): value for name, value in merged.items() if name.endswith("_mean")}
    scalars = means or {name: value for name, value in merged.items() if not name.endswith("_std")}
    return {name: float(value) for name, value in trainer_metrics(scalars).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)}


def reports_training_job(job: Callable[..., T]) -> Callable[..., T]:
    """Run a `TrainerMixin.trainer_*` method inside its job's run; it returns `(model, valid, test, ...)`."""

    @functools.wraps(job)
    def tracked(self: Any, *args: Any, **kwargs: Any) -> T:
        config = {"model_name": kwargs["model_name"], "data_name": kwargs["data_name"],
                  "probe": dict(vars(self.probe_args)), "trainer": dict(vars(self.trainer_args))}
        with training_job(self.trainer_args, kwargs["model_name"], kwargs["data_name"], config) as opened:
            outcome = job(self, *args, **kwargs)
            if opened is not None:
                opened.run.finish(summary=job_summary(outcome[1], outcome[2]))
        return outcome

    return tracked
