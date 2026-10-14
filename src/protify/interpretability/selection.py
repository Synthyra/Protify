"""One global training setting from sweeps: the grid point that costs the probes the least validation score.

A sweep scores every probe under every (learning rate, weight decay) pair of a grid on the validation split, once per
batch configuration (a sweep label). A probe's shortfall under a setting is its best validation score over every
setting of every label, minus its score under that setting. Shortfalls are averaged over the probes of each task and
family, then over families, then over tasks, so a head family's many probes and a large task's many proteins count
no more than any other. Among the settings within `tolerance` of the smallest mean shortfall, the fastest label wins:
its tasks finished in the fewest seconds.
"""

from __future__ import annotations

import json
import pandas as pd

from collections.abc import Sequence
from pathlib import Path


SETTING = ["label", "learning_rate", "weight_decay"]


def read_sweeps(root: Path, labels: Sequence[str]) -> pd.DataFrame:
    """Every task's sweep table under `root/sweeps/<label>/`, with the label, its batch and each task's seconds."""
    frames = []
    for label in labels:
        paths = sorted((root / "sweeps" / label).glob("*/sweep.parquet"))
        if not paths:
            raise FileNotFoundError(f"No sweep tables under {root / 'sweeps' / label}.")
        for path in paths:
            record = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
            frame = pd.read_parquet(path)
            frame["label"] = label
            frame["batch_size"] = record["settings"]["batch_size"]
            frame["max_tokens"] = record["settings"]["max_tokens"]
            frame["task_seconds"] = record["seconds"]
            frames.append(frame)
    sweeps = pd.concat(frames, ignore_index=True)
    tasks = sweeps.groupby("label")["task"].agg(lambda names: tuple(sorted(set(names))))
    if tasks.nunique() != 1:
        raise ValueError(f"Labels swept different tasks, so their seconds do not compare: {tasks.to_dict()}.")
    return sweeps


def rank_settings(sweeps: pd.DataFrame) -> pd.DataFrame:
    """Each setting's mean and worst per-task shortfall, and the seconds its label took, best first."""
    best = sweeps.groupby(["task", "family", "probe_name"])["validation_score"].transform("max")
    shortfalls = sweeps.assign(shortfall=best - sweeps["validation_score"])
    per_family = shortfalls.groupby(SETTING + ["task", "family"], as_index=False)["shortfall"].mean()
    per_task = per_family.groupby(SETTING + ["task"], as_index=False)["shortfall"].mean()
    seconds = sweeps.groupby(["label", "task"])["task_seconds"].first().groupby("label").sum().rename("seconds")
    ranked = per_task.groupby(SETTING).agg(
        mean_shortfall=("shortfall", "mean"),
        worst_task_shortfall=("shortfall", "max"),
        tasks=("task", "nunique"),
    ).reset_index()
    ranked = ranked.join(seconds, on="label")
    batches = sweeps.groupby("label")[["batch_size", "max_tokens"]].first()
    return ranked.join(batches, on="label").sort_values("mean_shortfall", ignore_index=True)


def choose_setting(ranked: pd.DataFrame, tolerance: float) -> pd.Series:
    """The fastest setting whose mean shortfall is within `tolerance` of the smallest."""
    eligible = ranked[ranked["mean_shortfall"] <= ranked["mean_shortfall"].min() + tolerance]
    return eligible.sort_values(["seconds", "mean_shortfall"]).iloc[0]


def task_optima(sweeps: pd.DataFrame) -> pd.DataFrame:
    """Each task's own best setting by the same averaging, to show how far the global choice is from it."""
    best = sweeps.groupby(["task", "family", "probe_name"])["validation_score"].transform("max")
    shortfalls = sweeps.assign(shortfall=best - sweeps["validation_score"])
    per_family = shortfalls.groupby(SETTING + ["task", "family"], as_index=False)["shortfall"].mean()
    per_task = per_family.groupby(SETTING + ["task"], as_index=False)["shortfall"].mean()
    return per_task.loc[per_task.groupby("task")["shortfall"].idxmin()].sort_values("task", ignore_index=True)
