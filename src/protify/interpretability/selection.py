"""One global training setting from sweeps: the grid point that costs the probes the least validation score.

A sweep scores every probe under every (learning rate, weight decay) pair of a grid on the validation split, once per
batch configuration (a sweep label); an MLP sweep adds hidden width and dropout to the setting. A probe's shortfall
under a setting is its best validation score over every setting of every label, minus its score under that setting.
Shortfalls are averaged over the probes of each task and family, then over families, then over tasks, so a head
family's many probes and a large task's many proteins count no more than any other. Sweeps of several seed
replicates (one root each) take each probe's best within its own replicate and average the shortfalls over
replicates with the probes. Among the settings within `tolerance` of the smallest mean shortfall, the fastest label
wins (its tasks finished in the fewest seconds, averaged over replicates), then the narrowest hidden layer.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from collections.abc import Sequence
from pathlib import Path


SETTING = ["label", "hidden", "dropout", "learning_rate", "weight_decay"]
PROBE = ["replicate", "task", "family", "probe_name"]


def read_sweeps(roots: Path | Sequence[Path], labels: Sequence[str]) -> pd.DataFrame:
    """Every task's sweep table under `<root>/sweeps/<label>/` of each root, with the root's replicate index, the
    label, its batch, each task's seconds and its kind (regression, multilabel, binary or multiclass)."""
    roots = [roots] if isinstance(roots, Path) else [Path(root) for root in roots]
    frames = []
    for replicate, root in enumerate(roots):
        for label in labels:
            paths = sorted((root / "sweeps" / label).glob("*/sweep.parquet"))
            if not paths:
                raise FileNotFoundError(f"No sweep tables under {root / 'sweeps' / label}.")
            for path in paths:
                record = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
                frame = pd.read_parquet(path)
                frame["replicate"] = replicate
                frame["label"] = label
                frame["batch_size"] = record["settings"]["batch_size"]
                frame["max_tokens"] = record["settings"]["max_tokens"]
                frame["task_seconds"] = record["seconds"]
                frame["kind"] = record["task"]["kind"]
                frames.append(frame)
    sweeps = pd.concat(frames, ignore_index=True)
    for column in ("hidden", "dropout"):  # sweeps written before MLP settings were recorded are linear
        if column not in sweeps:
            sweeps[column] = np.nan
    tasks = sweeps.groupby(["label", "replicate"])["task"].agg(lambda names: tuple(sorted(set(names))))
    if tasks.nunique() != 1:
        raise ValueError(f"Labels or replicates swept different tasks, so their seconds do not compare: {tasks.to_dict()}.")
    return sweeps


def _complete(sweeps: pd.DataFrame) -> pd.DataFrame:
    """A sweep table with the replicate and MLP columns a single linear sweep may lack."""
    defaults = {"replicate": 0, "hidden": np.nan, "dropout": np.nan}
    return sweeps.assign(**{column: value for column, value in defaults.items() if column not in sweeps})


def _per_task_shortfalls(sweeps: pd.DataFrame) -> pd.DataFrame:
    """Each setting's shortfall per task: per probe within its replicate, averaged over probes and replicates of a
    family, then over families."""
    sweeps = _complete(sweeps)
    best = sweeps.groupby(PROBE)["validation_score"].transform("max")
    shortfalls = sweeps.assign(shortfall=best - sweeps["validation_score"])
    per_family = shortfalls.groupby(SETTING + ["task", "family"], as_index=False, dropna=False)["shortfall"].mean()
    return per_family.groupby(SETTING + ["task"], as_index=False, dropna=False)["shortfall"].mean()


def rank_settings(sweeps: pd.DataFrame) -> pd.DataFrame:
    """Each setting's mean and worst per-task shortfall, and the seconds its label took, best first."""
    sweeps = _complete(sweeps)
    per_task = _per_task_shortfalls(sweeps)
    task_seconds = sweeps.groupby(["label", "replicate", "task"])["task_seconds"].first()
    seconds = task_seconds.groupby(["label", "replicate"]).sum().groupby("label").mean().rename("seconds")
    ranked = per_task.groupby(SETTING, dropna=False).agg(
        mean_shortfall=("shortfall", "mean"),
        worst_task_shortfall=("shortfall", "max"),
        tasks=("task", "nunique"),
    ).reset_index()
    ranked = ranked.join(seconds, on="label")
    batches = sweeps.groupby("label")[["batch_size", "max_tokens"]].first()
    return ranked.join(batches, on="label").sort_values("mean_shortfall", ignore_index=True)


def choose_setting(ranked: pd.DataFrame, tolerance: float) -> pd.Series:
    """The fastest setting whose mean shortfall is within `tolerance` of the smallest, then the narrowest hidden layer."""
    eligible = ranked[ranked["mean_shortfall"] <= ranked["mean_shortfall"].min() + tolerance]
    return eligible.sort_values(["seconds", "hidden", "mean_shortfall"]).iloc[0]


def task_optima(sweeps: pd.DataFrame) -> pd.DataFrame:
    """Each task's own best setting by the same averaging, to show how far the global choice is from it."""
    per_task = _per_task_shortfalls(sweeps)
    return per_task.loc[per_task.groupby("task")["shortfall"].idxmin()].sort_values("task", ignore_index=True)
