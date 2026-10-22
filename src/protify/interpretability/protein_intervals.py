"""Bootstrap intervals for the protein tasks: the states a probe variant would report, scored on draws of the test
proteins and averaged over seed replicates.

For each task and results label (such as `linear`, `linear_mean`, `mlp`, `mlp_mean`), the full-width probe with the
highest validation score averaged over the replicate roots is chosen twice: among the residual stream's states
(`stream`: the embedding, the block outputs and the final norm) and among every site (`any`). Each replicate's kept
weights of that probe score the test proteins, and every metric is computed on the whole test split and on
`INTERVAL_DRAWS` bootstrap draws of its proteins (`protein_draws`, seed `INTERVAL_SEED`: the same draws for every
label, model and replicate), then averaged over replicates draw by draw. A difference between two labels (the gain
of mean and variance pooling over the mean alone, or that gain's difference between probe kinds) is then a
difference draw by draw, and its interval the 2.5% and 97.5% quantiles of those differences.

Metrics: the task's headline score (MCC, or mean cosine similarity for ProtParam) and, for classification, macro ROC
AUC and average precision under every draw, and macro balanced MCC on the whole split alone (its class-balanced views
are drawn from the split itself).

Symbols: S replicates; n test proteins; C outputs; B draws.
"""

from __future__ import annotations

import json
import pandas as pd
import torch

from collections.abc import Callable, Sequence
from pathlib import Path
from safetensors.torch import load_file
from torch import Tensor

from .concepts import ConceptData
from .pooled_store import PooledTapStore, store_path
from .protein_probing import bank_outputs, full_view
from .scores import INTERVAL_DRAWS, INTERVAL_SEED, ScoreAccumulator, draw_plan, drawn_ranking, protein_draws, ranking_summary
from .sites import RESIDUAL_SITES
from .training import recorded_settings, restore_bank


CHOICES = ("stream", "any")


def chosen_states(roots: Sequence[Path], label: str, task: str) -> dict[str, str]:
    """The full-width probe of the residual stream and of any site with the highest validation score averaged over the
    replicate roots: choice -> probe name."""
    frames = [pd.read_parquet(Path(root) / "results" / label / task / "validation.parquet") for root in roots]
    validation = pd.concat(frames, ignore_index=True)
    full = validation[validation["family"] == "full"]
    mean = full.groupby("probe_name")["validation_score"].mean()  # (probes,)
    chosen = {"any": str(mean.idxmax())}
    stream = mean[mean.index.str.split(":").str[0].isin(RESIDUAL_SITES)]
    if len(stream):
        chosen["stream"] = str(stream.idxmax())
    return chosen


def probe_rows(store: PooledTapStore, probe_name: str, sequences: Sequence[str], pooling: str, device: torch.device) -> Tensor:
    """The pooled rows one full-width probe reads: (n, 1, w'), mean and variance or the mean alone."""
    site, layer = probe_name.split(":")[:2]
    slot = store.layers(site).index(int(layer))
    loaded = store.load(site, store.rows_for(sequences), device, slice(slot, slot + 1))  # (n, 1, 2 w)
    return full_view(loaded, pooling)


def probe_outputs(root: Path, label: str, data: ConceptData, probe_name: str, split: str, device: torch.device) -> Tensor:
    """One replicate's outputs of one kept full-width probe on a split: (n, C)."""
    folder = Path(root) / "results" / label / data.task.name
    record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    settings = recorded_settings(record)
    store = PooledTapStore(store_path(Path(root), "trained", "original"))
    if store.manifest.normalized != settings.normalize:
        raise SystemExit(f"{store.root} does not pool the token states {folder} was trained on (normalize={settings.normalize}).")
    site, layer = probe_name.split(":")[:2]
    position = store.layers(site).index(int(layer))
    prefix = f"{site}.full."
    saved = {key.removeprefix(prefix): value[position: position + 1] for key, value in load_file(str(folder / "probes.safetensors")).items()
             if key.startswith(prefix)}
    bank = restore_bank([probe_name], saved, data.task.outputs, data.task.loss, settings, device)
    rows = probe_rows(store, probe_name, data.splits[split].sequences, settings.pooling, device)  # (n, 1, w')
    return bank_outputs(bank, rows)[:, 0]  # (n, C)


def draw_scores(outputs: Tensor, targets: Tensor, kind: str, counts: Tensor) -> dict[str, Tensor]:
    """Each metric of S replicates' outputs on the whole split (row 0) and every draw, averaged over replicates:
    (B + 1,) each."""
    # outputs: (S, n, C); targets: (n,) or (n, C); counts: (B, n)
    replicates, proteins, width = outputs.shape
    Z = outputs.transpose(0, 1)  # (n, S, C): the replicates as groups
    plan = draw_plan(counts.to(outputs.device))
    accumulator = ScoreAccumulator(kind, replicates, width, plan, per_label=False)
    accumulator.add(Z, targets, torch.arange(proteins, device=outputs.device))
    scores = {"score": accumulator.scores().mean(dim=1)}  # (B + 1,)
    if kind == "regression":
        return scores
    for name, values in drawn_ranking(Z, targets, kind, plan.membership).items():  # (S, B + 1) each
        scores[name] = values.mean(dim=0)
    whole = torch.full((counts.shape[0] + 1,), torch.nan, dtype=torch.float64, device=outputs.device)  # (B + 1,)
    whole[0] = ranking_summary(Z, targets, kind)["balanced_mcc"].mean()
    scores["balanced_mcc"] = whole
    return scores


def task_intervals(
    roots: Sequence[Path],
    labels: Sequence[str],
    data: ConceptData,
    device: torch.device,
    draws: int = INTERVAL_DRAWS,
    seed: int = INTERVAL_SEED,
    outputs_of: Callable[[Path, str, ConceptData, str], Tensor] | None = None,
) -> pd.DataFrame:
    """One task's long table: label, choice, probe name, metric, draw (-1 for the whole test split) and value, every
    label's chosen states scored on the same draws. `outputs_of(root, label, data, probe_name)` gives a replicate's
    test outputs (n, C), by default its kept AdamW weights; the converged solver passes its own fits."""
    task = data.task
    outputs_of = outputs_of or (lambda root, label, data, name: probe_outputs(root, label, data, name, "test", device))
    targets = data.protein_targets("test").to(device)
    counts = protein_draws(len(data.splits["test"].sequences), draws, seed)  # (B, n)
    frames = []
    for label in labels:
        for choice, probe_name in chosen_states(roots, label, task.name).items():
            outputs = torch.stack([outputs_of(Path(root), label, data, probe_name) for root in roots])  # (S, n, C)
            for metric, values in draw_scores(outputs, targets, task.kind, counts).items():
                frames.append(pd.DataFrame({
                    "task": task.name, "label": label, "choice": choice, "probe_name": probe_name, "metric": metric,
                    "draw": range(-1, draws), "value": values.cpu().numpy(),
                }))
    return pd.concat(frames, ignore_index=True)


def protein_intervals(
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
    """`task_intervals` of every task, one parquet table each under `out`; finished tasks are skipped."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name in tasks:
        path = out / f"{name}.parquet"
        if path.exists() and not overwrite:
            print(f"have {path}")
            continue
        table = task_intervals(roots, labels, read_task(name), device, draws, seed)
        table.to_parquet(path, index=False)
        written.append(path)
        print(f"{name}: {table['label'].nunique()} labels, {draws} draws of the test proteins -> {path}")
    return written
