"""Converged linear probes: what an AdamW probe leaves on the table, at the states its variant reports.

For each protein task and linear results label (`linear`, `linear_mean`), the states `protein_intervals` reports (the
best residual-stream and best any-site full-width probe, chosen on validation averaged over replicates) are refitted
on the same standardized pooled rows by a converged solver, and scored on the same test draws:

- multilabel, binary and multiclass tasks: L-BFGS on the penalized mean cross entropy (outputs share nothing but the
  penalty, so the joint problem is the per-label logistic regressions), along a decreasing grid of L2 penalties, each
  fit warm-started from the last and run until the largest absolute gradient is under `TOLERANCE` or `ITERATIONS` are
  spent; the walk stops after two fits in a row that do not converge, since small penalties leave these problems
  nearly separable and ill conditioned;
- regression (ProtParam): ridge regression in closed form for every penalty.

The penalty is the one with the best validation score in the task's own metric among the converged fits. A fit
depends on the store and the state alone, not on a seed, so replicate roots that share a store share one fit; each
replicate's own store (a randomly initialized model's) gets its own. Fits run in float64, on the GPU when one is given:
float32 stalls well above the tolerance, because the line search cannot resolve the steps.

Symbols: n proteins of a split; w columns a probe reads; C outputs.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from collections.abc import Callable, Sequence
from pathlib import Path
from torch import Tensor

from .concepts import ConceptData
from .pooled_store import PooledTapStore, store_path
from .protein_intervals import probe_rows, task_intervals
from .protein_probing import feature_moments
from .scores import INTERVAL_DRAWS, INTERVAL_SEED, ResamplePlan, ScoreAccumulator
from .training import recorded_settings


TOLERANCE = 1e-4
ITERATIONS = 3000
CHUNK = 100  # L-BFGS iterations between convergence checks
PENALTIES = tuple(10.0 ** exponent for exponent in np.arange(-6.0, 0.51, 0.5))  # 1e-6 to 3.2
STANDARDIZE_FLOOR = 1e-6  # as the probe bank's standardization


def standardized_rows(store: PooledTapStore, probe_name: str, data: ConceptData, pooling: str, device: torch.device) -> dict[str, Tensor]:
    """One state's rows of every split, standardized on the training proteins as the probe bank standardizes them:
    split -> (n, w) float32."""
    raw = {split: probe_rows(store, probe_name, data.splits[split].sequences, pooling, device) for split in ("train", "validation", "test")}  # (n, 1, w) each
    mean, variance = feature_moments(raw["train"])  # (1, w) each
    inverse = variance.clamp_min(STANDARDIZE_FLOOR).rsqrt()
    return {split: ((values.to(torch.float32) - mean) * inverse)[:, 0] for split, values in raw.items()}


def headline(kind: str, outputs: int, Z: Tensor, targets: Tensor) -> float:
    """The task metric of logits Z (n, C) on a whole split."""
    plan = ResamplePlan.draw(Z.shape[0], 0, device=Z.device)
    accumulator = ScoreAccumulator(kind, 1, outputs, plan, per_label=False)
    accumulator.add(Z.unsqueeze(1), targets, torch.arange(Z.shape[0], device=Z.device))
    return float(accumulator.scores()[0, 0])


def fit_logistic(X: Tensor, y: Tensor, loss: str, penalty: float, weight: Tensor, bias: Tensor) -> tuple[Tensor, Tensor, int, float]:
    """L-BFGS on mean cross entropy plus penalty / 2 times the squared weights, from (weight, bias); returns the fit,
    the iterations spent and the largest absolute gradient."""
    # X: (n, w); y: (n,) classes or (n, C) 0/1; weight: (w, C); bias: (C,)
    X = X.to(torch.float64)
    y = y if loss == "multiclass" else y.to(torch.float64)
    weight, bias = weight.to(torch.float64).clone().requires_grad_(True), bias.to(torch.float64).clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([weight, bias], lr=1.0, max_iter=CHUNK, tolerance_grad=TOLERANCE / 10, tolerance_change=1e-12,
                                  history_size=20, line_search_fn="strong_wolfe")

    def objective() -> Tensor:
        Z = X @ weight + bias  # (n, C)
        data = F.cross_entropy(Z, y) if loss == "multiclass" else F.binary_cross_entropy_with_logits(Z, y, reduction="none").sum(dim=1).mean()
        return data + 0.5 * penalty * weight.square().sum()

    def closure() -> Tensor:  # L-BFGS evaluates the loss and its gradient through the closure
        optimizer.zero_grad()
        value = objective()
        value.backward()
        return value

    def largest_gradient() -> float:
        closure()
        return max(weight.grad.abs().max().item(), bias.grad.abs().max().item())

    spent = 0
    gradient = largest_gradient()
    while spent < ITERATIONS and gradient > TOLERANCE:
        optimizer.step(closure)
        spent += CHUNK
        gradient = largest_gradient()
    return weight.detach().to(torch.float32), bias.detach().to(torch.float32), spent, gradient


def penalty_path(data: ConceptData, rows: dict[str, Tensor], targets: dict[str, Tensor]) -> list[dict[str, object]]:
    """Every fit along the penalty grid, with its validation score: penalty, weight (w, C), bias (C,), iterations,
    largest gradient, converged, validation."""
    task = data.task
    X = rows["train"]  # (n, w)
    fits = []
    if task.loss == "regression":
        Xd, Yd = X.to(torch.float64), targets["train"].to(torch.float64)
        centre = Yd.mean(dim=0)  # (C,)
        values, vectors = torch.linalg.eigh(Xd.T @ Xd)  # (w,), (w, w)
        cross = vectors.T @ (Xd.T @ (Yd - centre))  # (w, C)
        for penalty in PENALTIES:
            weight = (vectors @ (cross / (values.clamp_min(0).unsqueeze(1) + X.shape[0] * penalty))).to(torch.float32)  # (w, C)
            fits.append({"penalty": penalty, "weight": weight, "bias": centre.to(torch.float32), "iterations": 0, "gradient": 0.0, "converged": True})
    else:
        weight = torch.zeros(X.shape[1], task.outputs, device=X.device)
        bias = torch.zeros(task.outputs, device=X.device)
        failures = 0
        for penalty in sorted(PENALTIES, reverse=True):
            weight, bias, spent, gradient = fit_logistic(X, targets["train"], task.loss, penalty, weight, bias)
            converged = gradient <= TOLERANCE
            fits.append({"penalty": penalty, "weight": weight, "bias": bias, "iterations": spent, "gradient": gradient, "converged": converged})
            failures = 0 if converged else failures + 1
            if failures == 2:
                break
    for fit in fits:
        fit["validation"] = headline(task.kind, task.outputs, rows["validation"] @ fit["weight"] + fit["bias"], targets["validation"])
    return fits


def chosen_fit(fits: list[dict[str, object]]) -> dict[str, object] | None:
    """The converged fit with the best validation score (the larger penalty on a tie); None if none converged."""
    converged = [fit for fit in fits if fit["converged"]]
    return max(converged, key=lambda fit: (fit["validation"], fit["penalty"])) if converged else None


def solve_tasks(
    roots: Sequence[Path],
    labels: Sequence[str],
    tasks: Sequence[str],
    out: Path,
    read_task: Callable[[str], ConceptData],
    device: torch.device,
    draws: int = INTERVAL_DRAWS,
    seed: int = INTERVAL_SEED,
    overwrite: bool = False,
) -> list[Path]:
    """Converged fits at each linear label's reported states, scored on the test draws as `protein_intervals` scores
    the AdamW probes: `<out>/<task>.parquet` with labels `<label>_solver`, and the penalty paths in
    `<out>/<task>_fits.parquet`. A state without a converged fit is left out of the scores and marked in the paths."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name in tasks:
        path = out / f"{name}.parquet"
        if path.exists() and not overwrite:
            print(f"have {path}")
            continue
        data = read_task(name)
        targets = {split: data.protein_targets(split).to(device) for split in ("train", "validation", "test")}
        cache: dict[tuple[str, str, str], Tensor | None] = {}
        records: list[dict[str, object]] = []

        def outputs_of(root: Path, label: str, data: ConceptData, probe_name: str) -> Tensor:
            settings = recorded_settings(json.loads((root / "results" / label / name / "run.json").read_text(encoding="utf-8")))
            store = PooledTapStore(store_path(root, "trained", "original"))
            key = (str(store.root.resolve()), probe_name, settings.pooling)
            if key not in cache:
                rows = standardized_rows(store, probe_name, data, settings.pooling, device)
                fits = penalty_path(data, rows, targets)
                chosen = chosen_fit(fits)
                records.extend({"task": name, "label": label, "root": str(root), "probe_name": probe_name, "pooling": settings.pooling,
                                **{field: fit[field] for field in ("penalty", "iterations", "gradient", "converged", "validation")},
                                "chosen": fit is chosen} for fit in fits)
                cache[key] = rows["test"] @ chosen["weight"] + chosen["bias"] if chosen is not None else None  # (n, C)
            if cache[key] is None:
                raise _Unconverged(probe_name)
            return cache[key]

        frames = []
        for label in labels:
            try:
                table = task_intervals(roots, [label], data, device, draws, seed, outputs_of)
            except _Unconverged as missing:
                print(f"{name} {label}: no converged fit at {missing}; left out")
                continue
            frames.append(table.assign(label=f"{label}_solver"))
        (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()).to_parquet(path, index=False)
        pd.DataFrame(records).to_parquet(out / f"{name}_fits.parquet", index=False)
        written.append(path)
        print(f"{name}: {len(records)} fits -> {path}")
    return written


class _Unconverged(Exception):
    """No fit along the penalty grid converged at a state, so it has no converged solution to score."""
