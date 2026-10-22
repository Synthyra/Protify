"""The geometry of pooled states, to read what a gap between linear and MLP probes means.

A linear probe reads a concept when its classes differ in their means along some direction; an MLP also reads them
when they differ in shape (spread, orientation, several clusters) or sit in a curved arrangement. For each pooled
state of a store (the mean over a protein's tokens), this module measures, without training a probe:

- separability by neighbors: a k-nearest-neighbor readout (cosine on standardized rows, k chosen on validation among
  `NEIGHBORS`) scored on the test proteins in the task's metric, a probe-free reference beside the linear and MLP
  scores of the same state;
- separability by means against by shape: for each label (one class against the rest for a multiclass task), the
  Bhattacharyya distance between its positives and negatives in the first `COMPONENTS` principal components of the
  standardized training rows, split into its mean term, 1/8 dmu' S^-1 dmu, which a linear readout can use, and its
  covariance term, 1/2 ln(det S / sqrt(det S1 det S2)), which only a nonlinear one can (S the average of the two
  classes' covariances), averaged over the labels with enough positives;
- the state's own geometry over the training proteins of every task: effective rank (the exponential of the entropy
  of the normalized covariance eigenvalues; Roy and Vetterli 2007), participation ratio, the first component's share
  of variance, anisotropy (mean cosine between proteins; Ethayarajh 2019), the largest coordinate's share of the mean
  squared norm (a massive dimension; Sun et al. 2024) and the TwoNN intrinsic dimension (Facco et al. 2017).

Symbols: n proteins; w state width; k components; L labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from collections.abc import Sequence
from pathlib import Path

from .concepts import ConceptData
from .pooled_store import PooledTapStore
from .protein_intervals import probe_rows
from .solver import STANDARDIZE_FLOOR, headline


NEIGHBORS = (5, 10, 20, 50)
COMPONENTS = 32
MIN_POSITIVES = 20  # a label's positives (and negatives) in the training split for its Bhattacharyya terms
MAX_LABELS = 100  # the most frequent labels of a many-label task
GEOMETRY_PROTEINS = 8000  # training proteins of every task, drawn once, for the state's own geometry
TWONN_PROTEINS = 4000


def standardized(rows: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Every split's rows standardized with the training split's moments, float32."""
    train = rows["train"].to(torch.float32)
    mean = train.mean(dim=0, keepdim=True)
    scale = train.var(dim=0, unbiased=False, keepdim=True).clamp_min(STANDARDIZE_FLOOR).rsqrt()
    return {split: (values.to(torch.float32) - mean) * scale for split, values in rows.items()}


def neighbor_outputs(query: torch.Tensor, reference: torch.Tensor, targets: torch.Tensor, kind: str, outputs: int, k: int,
                     chunk: int = 2048) -> torch.Tensor:
    """A k-nearest-neighbor readout by cosine similarity: the neighbors' class counts (multiclass), the share of
    neighbors with each label less one half (multilabel and binary), or their mean values (regression): (n, C)."""
    reference = F.normalize(reference, dim=-1)
    if kind == "multiclass":
        values = F.one_hot(targets, outputs).to(torch.float32)
    else:
        values = targets.to(torch.float32).reshape(len(targets), -1)
    found = []
    for start in range(0, len(query), chunk):
        similarity = F.normalize(query[start: start + chunk], dim=-1) @ reference.T  # (c, n_ref)
        nearest = similarity.topk(k, dim=-1).indices  # (c, k)
        found.append(values[nearest].mean(dim=1))
    result = torch.cat(found)
    return result - 0.5 if kind in ("multilabel", "binary") else result


def neighbor_score(rows: dict[str, torch.Tensor], data: ConceptData, targets: dict[str, torch.Tensor]) -> dict[str, float]:
    """The k-nearest-neighbor readout's test score, k chosen on validation."""
    task = data.task
    best = None
    for k in NEIGHBORS:
        if k > len(rows["train"]):
            break
        validation = headline(task.kind, task.outputs, neighbor_outputs(rows["validation"], rows["train"], targets["train"], task.kind, task.outputs, k),
                              targets["validation"])
        if best is None or validation > best[0]:
            best = (validation, k)
    validation, k = best
    test = headline(task.kind, task.outputs, neighbor_outputs(rows["test"], rows["train"], targets["train"], task.kind, task.outputs, k), targets["test"])
    return {"knn": test, "knn_validation": validation, "knn_k": k}


def label_columns(data: ConceptData, targets: torch.Tensor) -> torch.Tensor:
    """0/1 label columns of the training targets, the most frequent `MAX_LABELS` with `MIN_POSITIVES` positives and
    negatives each: (n, L) bool."""
    if data.task.kind == "regression":
        return torch.zeros(len(targets), 0, dtype=torch.bool, device=targets.device)
    labels = F.one_hot(targets, data.task.num_labels).bool() if data.task.kind == "multiclass" else targets.reshape(len(targets), -1).bool()
    positives = labels.sum(dim=0)
    usable = (positives >= MIN_POSITIVES) & (len(labels) - positives >= MIN_POSITIVES)
    order = torch.argsort(positives * usable, descending=True)[: min(MAX_LABELS, int(usable.sum()))]
    return labels[:, order]


def bhattacharyya_terms(X: torch.Tensor, labels: torch.Tensor, components: int = COMPONENTS) -> dict[str, float]:
    """Mean and covariance terms of the Bhattacharyya distance between each label's positives and negatives in the
    first principal components of standardized rows X (n, w), averaged over the labels (float64)."""
    if labels.shape[1] == 0:
        return {"bhattacharyya_mean": np.nan, "bhattacharyya_cov": np.nan, "labels": 0}
    centred = X.to(torch.float32) - X.to(torch.float32).mean(dim=0)
    _, _, V = torch.pca_lowrank(centred, q=min(components + 8, *centred.shape), center=False, niter=4)  # randomized PCA
    Y = (centred @ V[:, :components]).to(torch.float64)  # (n, k)
    k = Y.shape[1]
    floor = 1e-3 * Y.var(dim=0).mean() * torch.eye(k, dtype=torch.float64, device=Y.device)
    means, covariances = [], []
    for column in labels.T:
        a, b = Y[column], Y[~column]
        S1, S2 = torch.cov(a.T) + floor, torch.cov(b.T) + floor
        S = (S1 + S2) / 2
        delta = a.mean(dim=0) - b.mean(dim=0)
        means.append(float(delta @ torch.linalg.solve(S, delta)) / 8)
        covariances.append(float(torch.logdet(S) - (torch.logdet(S1) + torch.logdet(S2)) / 2) / 2)
    return {"bhattacharyya_mean": float(np.mean(means)), "bhattacharyya_cov": float(np.mean(covariances)), "labels": labels.shape[1]}


def state_geometry(X: torch.Tensor, twonn_rows: int = TWONN_PROTEINS) -> dict[str, float]:
    """A state's own geometry over proteins X (n, w): see the module's docstring."""
    X = X.to(torch.float32)
    centred = X - X.mean(dim=0)
    gram = centred.T @ centred if X.shape[1] <= X.shape[0] else centred @ centred.T  # the smaller of (w, w) and (n, n)
    values = torch.linalg.eigvalsh(gram / len(X)).clamp_min(0).flip(0).to(torch.float64)  # largest first
    share = values / values.sum()
    nonzero = share[share > 0]
    unit = F.normalize(X, dim=-1)
    total = unit.sum(dim=0)
    anisotropy = float((total @ total - len(X)) / (len(X) * (len(X) - 1)))  # the mean cosine over distinct pairs
    energy = X.to(torch.float64).square().mean(dim=0)
    sample = X[:twonn_rows]
    distances = torch.cdist(sample, sample)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.topk(2, dim=-1, largest=False).values  # (m, 2)
    ratio = nearest[:, 1] / nearest[:, 0].clamp_min(1e-12)
    ratio = ratio[torch.isfinite(ratio) & (ratio > 1)]
    return {
        "effective_rank": float(torch.exp(-(nonzero * nonzero.log()).sum())),
        "participation_ratio": float(values.sum().square() / values.square().sum()),
        "first_component_share": float(share[0]),
        "anisotropy": anisotropy,
        "top_dimension_share": float(energy.max() / energy.sum()),
        "twonn_dimension": float(len(ratio) / ratio.log().sum()) if len(ratio) else np.nan,
    }


def probe_names(store: PooledTapStore, sites: Sequence[str]) -> list[str]:
    return [f"{site}:{layer}" for site in sites if site in store.site_names for layer in store.layers(site)]


@torch.no_grad()
def store_geometry(store: PooledTapStore, tasks: Sequence[ConceptData], sites: Sequence[str], device: torch.device, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For every full-width state of `sites` in the store, on the pooled mean: each task's neighbor score and
    Bhattacharyya terms (one row per task and state), and the state's own geometry over a draw of every task's
    training proteins (one row per state)."""
    everything = sorted({sequence for data in tasks for part in data.splits.values() for sequence in part.sequences})
    position = {sequence: index for index, sequence in enumerate(everything)}
    pool = sorted({position[sequence] for data in tasks for sequence in data.splits["train"].sequences})
    drawn = torch.tensor([pool[index] for index in sorted(np.random.default_rng(seed).permutation(len(pool))[: GEOMETRY_PROTEINS].tolist())], device=device)
    splits = {data.task.name: {split: torch.tensor([position[sequence] for sequence in data.splits[split].sequences], device=device)
                               for split in ("train", "validation", "test")} for data in tasks}
    targets = {data.task.name: {split: data.protein_targets(split).to(device) for split in ("train", "validation", "test")} for data in tasks}
    labels = {data.task.name: label_columns(data, targets[data.task.name]["train"]) for data in tasks}
    separability, geometry = [], []
    for name in probe_names(store, sites):
        site, layer = name.split(":")
        loaded = probe_rows(store, name, everything, "mean", device)[:, 0]  # (n, w), every task's proteins, read once
        geometry.append({"site": site, "layer": int(layer), **state_geometry(loaded[drawn])})
        for data in tasks:
            rows = standardized({split: loaded[chosen] for split, chosen in splits[data.task.name].items()})
            separability.append({"task": data.task.name, "site": site, "layer": int(layer), **neighbor_score(rows, data, targets[data.task.name]),
                                 **bhattacharyya_terms(rows["train"], labels[data.task.name])})
        print(f"{name}: done", flush=True)
    return pd.DataFrame(separability), pd.DataFrame(geometry)
