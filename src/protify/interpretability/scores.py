"""Test scores of many probes at once, on the whole test split and on random halves of its proteins.

Scores are accumulated protein by protein, so a half is just a weighted sum of per-protein counts: row 0 of the
resample axis is the full split and rows 1..R are fixed random halves (R = 7 by default). Classification keeps
confusion counts and reports the Matthews correlation coefficient; regression keeps cosine terms and reports the
mean per-protein cosine similarity between the predicted and true standardized vectors, plus each target's cosine
across proteins. The resample weights may be any nonnegative counts, so bootstrap draws of the test proteins (counts
from `protein_draws`) score the same way.

Ranking metrics sit beside the counts: ROC AUC, average precision and Atlas's balanced MCC of binary scores, and their
macro means over the labels of a protein task (`ranking_summary`, `drawn_ranking`), a multiclass task's classes taken
one against the rest.

Symbols: n rows (proteins, or residues); p proteins of a batch; g groups; C outputs; R resamples; P test proteins;
B bootstrap draws; L labels ranked.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from torch import Tensor


BALANCED_VIEWS = 10  # class-balanced views of an unbalanced split, for balanced MCC
METRIC_CHUNK = 32  # probes whose rankings are computed at once
INTERVAL_DRAWS = 1000  # bootstrap draws of the test proteins, for intervals
INTERVAL_SEED = 0  # one seed for every run, so that every probe and model shares the draws
RANKING_METRICS = ("roc_auc", "average_precision", "balanced_mcc")


@dataclass(frozen=True)
class ResamplePlan:
    """Which test proteins each resample holds: row 0 all of them, then `count` random fractions."""

    membership: Tensor  # (R + 1, P) float, 1 where the protein is in the resample

    @classmethod
    def draw(cls, proteins: int, count: int = 7, fraction: float = 0.5, seed: int = 24895, device: torch.device | str = "cpu") -> "ResamplePlan":
        rng = np.random.default_rng(seed)
        membership = np.zeros((count + 1, proteins), dtype=np.float32)
        membership[0] = 1.0
        size = int(proteins * fraction)
        for row in range(1, count + 1):
            membership[row, rng.choice(proteins, size=size, replace=False)] = 1.0
        return cls(torch.from_numpy(membership).to(device))

    @property
    def rows(self) -> int:
        return self.membership.shape[0]


def multiclass_mcc(confusion: Tensor) -> Tensor:
    """Gorodkin's multiclass MCC of each confusion matrix (true classes on rows), 0 where it is undefined."""
    # confusion: (..., C, C)
    confusion = confusion.to(torch.float64)
    total = confusion.sum(dim=(-2, -1))  # (...)
    correct = confusion.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (...)
    predicted = confusion.sum(dim=-2)  # (..., C)
    true = confusion.sum(dim=-1)  # (..., C)
    numerator = correct * total - (predicted * true).sum(dim=-1)
    denominator = torch.sqrt((total.square() - predicted.square().sum(dim=-1)) * (total.square() - true.square().sum(dim=-1)))
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-300), torch.zeros_like(numerator))


def binary_mcc(counts: Tensor) -> Tensor:
    """MCC of (tp, fp, fn, tn) counts on the last axis, 0 where it is undefined."""
    # counts: (..., 4)
    counts = counts.to(torch.float64)
    tp, fp, fn, tn = counts.unbind(dim=-1)
    numerator = tp * tn - fp * fn
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-300), torch.zeros_like(numerator))


class ScoreAccumulator:
    """Counts for G probes of one task, folded per resample as batches arrive.

    `per_label` keeps each label's counts for multilabel and binary tasks; a sweep, which needs only the headline
    score of many probes, turns it off.
    """

    def __init__(self, kind: str, groups: int, outputs: int, plan: ResamplePlan, per_label: bool = True) -> None:
        self.kind = kind
        self.groups = groups
        self.outputs = outputs
        self.plan = plan
        rows = plan.rows
        device = plan.membership.device
        if kind == "multiclass":
            self.confusion = torch.zeros(rows, groups, outputs, outputs, dtype=torch.float64, device=device)
        elif kind in ("multilabel", "binary"):
            self.micro = torch.zeros(rows, groups, 4, dtype=torch.float64, device=device)
            self.per_label = torch.zeros(rows, groups, outputs, 4, dtype=torch.float64, device=device) if per_label else None
        elif kind == "regression":
            self.cosine_sum = torch.zeros(rows, groups, dtype=torch.float64, device=device)
            self.proteins = torch.zeros(rows, dtype=torch.float64, device=device)
            self.terms = torch.zeros(rows, groups, outputs, 3, dtype=torch.float64, device=device)
        else:
            raise ValueError(f"Unknown task kind {kind!r}.")

    def add(
        self,
        Z: Tensor,
        targets: Tensor,
        proteins: Tensor,
        start: int = 0,
        index: tuple[Tensor, Tensor] | None = None,
    ) -> None:
        """Add the outputs of groups start.. for rows that belong to test proteins `proteins`.

        Z holds logits; targets are class indices (multiclass) or values; `proteins` gives each row's test index.
        `index` is `torch.unique(proteins, return_inverse=True)`, which a caller adding many groups of the same rows
        computes once: `unique` on a device waits for the device, and many small waits are slow on a shared GPU.
        """
        # Z: (n, g, C); targets: (n,) or (n, C); proteins: (n,) long
        g = Z.shape[1]
        local, inverse = index if index is not None else torch.unique(proteins, return_inverse=True)  # (p,), (n,)
        weights = self.plan.membership[:, local].to(torch.float64)  # (R + 1, p)
        stop = start + g
        if self.kind == "multiclass":
            C = self.outputs
            predicted = Z.argmax(dim=-1)  # (n, g)
            cells = (inverse.unsqueeze(1) * g + torch.arange(g, device=Z.device)) * (C * C) + targets.unsqueeze(1) * C + predicted  # (n, g)
            # index_add_ rather than bincount, which waits for the device to size its output
            per_protein = torch.zeros(len(local) * g * C * C, dtype=torch.float64, device=Z.device).index_add_(
                0, cells.reshape(-1), torch.ones(cells.numel(), dtype=torch.float64, device=Z.device)
            )  # (p * g * C * C,)
            folded = weights @ per_protein.view(len(local), g * C * C)  # (R + 1, g * C * C)
            self.confusion[:, start:stop] += folded.view(-1, g, C, C)
            return
        if self.kind in ("multilabel", "binary") and self.per_label is None:
            predicted = Z > 0  # (n, g, C)
            truth = targets.to(torch.bool).unsqueeze(1)  # (n, 1, C)
            tp = (predicted & truth).sum(dim=-1)  # (n, g)
            fp = predicted.sum(dim=-1) - tp  # (n, g)
            fn = truth.sum(dim=-1) - tp  # (n, g)
            tn = self.outputs - tp - fp - fn  # (n, g)
            counts = torch.stack([tp, fp, fn, tn], dim=-1).to(torch.float64)  # (n, g, 4)
            per_protein = torch.zeros(len(local), g, 4, dtype=torch.float64, device=Z.device).index_add_(0, inverse, counts)
            self.micro[:, start:stop] += torch.einsum("rp,pgk->rgk", weights, per_protein)
            return
        if self.kind in ("multilabel", "binary"):
            predicted = Z > 0  # (n, g, C)
            truth = targets.to(torch.bool).unsqueeze(1)  # (n, 1, C)
            counts = torch.stack(
                [predicted & truth, predicted & ~truth, ~predicted & truth, ~predicted & ~truth], dim=-1
            ).to(torch.float64)  # (n, g, C, 4)
            per_protein = torch.zeros(len(local), g, self.outputs, 4, dtype=torch.float64, device=Z.device)
            per_protein.index_add_(0, inverse, counts)  # (p, g, C, 4)
            self.per_label[:, start:stop] += torch.einsum("rp,pgck->rgck", weights, per_protein)
            self.micro[:, start:stop] += torch.einsum("rp,pgk->rgk", weights, per_protein.sum(dim=2))
            return
        values = Z.to(torch.float64)  # (n, g, C)
        truth = targets.to(torch.float64).unsqueeze(1)  # (n, 1, C)
        cosine = torch.nn.functional.cosine_similarity(values, truth.expand_as(values), dim=-1, eps=1e-8)  # (n, g)
        per_protein = torch.zeros(len(local), g, dtype=torch.float64, device=Z.device).index_add_(0, inverse, cosine)  # (p, g)
        self.cosine_sum[:, start:stop] += weights @ per_protein
        if start == 0:
            self.proteins += weights @ torch.ones(len(local), dtype=torch.float64, device=Z.device)
        terms = torch.stack([values * truth, values.square(), truth.square().expand_as(values)], dim=-1)  # (n, g, C, 3)
        per_terms = torch.zeros(len(local), g, self.outputs, 3, dtype=torch.float64, device=Z.device).index_add_(0, inverse, terms)
        self.terms[:, start:stop] += torch.einsum("rp,pgck->rgck", weights, per_terms)

    def scores(self) -> Tensor:
        """The headline score of each resample and group: MCC, or mean cosine similarity."""
        # returns (R + 1, G)
        if self.kind == "multiclass":
            return multiclass_mcc(self.confusion)
        if self.kind in ("multilabel", "binary"):
            return binary_mcc(self.micro)
        return self.cosine_sum / self.proteins.clamp_min(1.0).unsqueeze(1)

    def elementwise(self) -> Tensor | None:
        """Per-label MCC, or per-target cosine across proteins: (R + 1, G, C); None for multiclass or without counts."""
        if self.kind in ("multilabel", "binary"):
            return binary_mcc(self.per_label) if self.per_label is not None else None
        if self.kind == "regression":
            products, predicted, truth = self.terms.unbind(dim=-1)
            return products / torch.sqrt(predicted * truth).clamp_min(1e-300)
        return None


def draw_plan(counts: Tensor) -> ResamplePlan:
    """A plan whose row 0 is the whole split and rows 1..B are bootstrap draws of its proteins (`protein_draws`)."""
    # counts: (B, P)
    return ResamplePlan(torch.cat([torch.ones(1, counts.shape[1], dtype=counts.dtype, device=counts.device), counts]))  # (B + 1, P)


# -- ranking metrics ---------------------------------------------------------------------------------------------


def ranking_metrics(scores: Tensor, labels: Tensor, chunk: int = METRIC_CHUNK) -> tuple[Tensor, Tensor]:
    """ROC AUC and average precision of each column of `scores` against binary `labels`, as scikit-learn computes them
    for untied scores, `chunk` columns at a time."""
    # scores: (n, G); labels: (n,) 0 or 1
    roc_aucs, precisions = [], []
    ranks = torch.arange(1, scores.shape[0] + 1, dtype=torch.float64, device=scores.device).unsqueeze(1)  # (n, 1)
    for start in range(0, scores.shape[1], chunk):
        order = scores[:, start: start + chunk].argsort(dim=0, descending=True)  # (n, g)
        hits = labels.to(torch.float64)[order]  # (n, g), 1 where the k-th highest score belongs to a positive
        positives = hits.sum(dim=0)  # (g,)
        negatives = scores.shape[0] - positives
        true_positives = hits.cumsum(dim=0)  # (n, g)
        precisions.append((true_positives / ranks * hits).sum(dim=0) / positives.clamp_min(1))
        negatives_above = (1.0 - hits).cumsum(dim=0)  # (n, g), negatives ranked at or above each row
        roc_aucs.append(((negatives - negatives_above) * hits).sum(dim=0) / (positives * negatives).clamp_min(1))
    return torch.cat(roc_aucs), torch.cat(precisions)


def threshold_mcc(scores: Tensor, labels: Tensor) -> Tensor:
    """MCC of each column of logits `scores` at probability 0.5 (a logit of 0 or more predicts a positive)."""
    # scores: (n, G); labels: (n,) 0 or 1
    predicted, positive = scores >= 0, labels.to(torch.bool).unsqueeze(1)  # (n, G), (n, 1)
    masks = (predicted & positive, predicted & ~positive, ~predicted & positive, ~predicted & ~positive)
    return binary_mcc(torch.stack([mask.sum(dim=0, dtype=torch.float64) for mask in masks], dim=-1))  # (G,)


def balanced_views(labels: np.ndarray, seed: int = 0, subsamples: int = BALANCED_VIEWS) -> list[np.ndarray]:
    """Index sets of class-balanced views, drawn as Atlas draws them: the majority class downsampled without
    replacement to the minority's size, view i from seed `seed + 11 * (i + 1)`; a balanced split is its own one view."""
    # labels: (n,) 0 or 1
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if len(positive) == len(negative):
        return [np.arange(len(labels))]
    majority, minority = (negative, positive) if len(negative) > len(positive) else (positive, negative)
    return [np.concatenate([minority, np.random.default_rng(seed + 11 * (view + 1)).choice(majority, size=len(minority), replace=False)])
            for view in range(subsamples)]


def balanced_mcc(scores: Tensor, labels: Tensor, seed: int = 0, subsamples: int = BALANCED_VIEWS, chunk: int = METRIC_CHUNK) -> Tensor:
    """Balanced MCC of each column of `scores`, as Atlas defines it: the MCC at the one threshold that maximizes its mean
    over the class-balanced views (`balanced_views`), a score at or above the threshold predicting a positive. Every
    score is a candidate threshold, which reaches every count any view can show."""
    # scores: (n, G); labels: (n,) 0 or 1
    views = balanced_views(labels.cpu().numpy(), seed, subsamples)
    member = torch.zeros((len(views), scores.shape[0]), dtype=torch.bool, device=scores.device)  # (V, n)
    for row, view in enumerate(views):
        member[row, torch.as_tensor(view, device=scores.device)] = True
    positive = labels.to(torch.bool)  # (n,)
    out = []
    for start in range(0, scores.shape[1], chunk):
        ordered, order = scores[:, start: start + chunk].sort(dim=0, descending=True)  # (n, g)
        cut = torch.ones_like(ordered, dtype=torch.bool)  # (n, g), the last row of each run of equal scores
        cut[:-1] = ordered[:-1] != ordered[1:]
        hit = positive[order]  # (n, g)
        mean = torch.zeros(ordered.shape, dtype=torch.float64, device=scores.device)  # (n, g)
        for inside in member:
            kept = inside[order]  # (n, g)
            tp = (hit & kept).cumsum(dim=0, dtype=torch.float64)  # (n, g), the view's positives at or above each cut
            fp = (~hit & kept).cumsum(dim=0, dtype=torch.float64)
            counts = torch.stack([tp, fp, float((positive & inside).sum()) - tp, float((~positive & inside).sum()) - fp], dim=-1)  # (n, g, 4)
            mean += binary_mcc(counts) / len(member)
        out.append(mean.masked_fill(~cut, -torch.inf).max(dim=0).values)
    return torch.cat(out)  # (G,)


def protein_draws(proteins: int, draws: int = INTERVAL_DRAWS, seed: int = INTERVAL_SEED) -> Tensor:
    """Each protein's count in each of `draws` bootstrap draws of a split's `proteins` proteins, taken with
    replacement on the CPU from `seed`, so that every probe and model shares them: (draws, proteins)."""
    picks = torch.randint(0, proteins, (draws, proteins), generator=torch.Generator().manual_seed(seed))  # (B, u)
    return torch.zeros(draws, proteins).scatter_add_(1, picks, torch.ones(draws, proteins))  # (B, u)


def weighted_ranking_metrics(scores: Tensor, labels: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    """ROC AUC and average precision of each column of `scores` under each row of `weights`, as scikit-learn computes
    them with sample weights (a run of tied scores is one threshold): (G, B) each, NaN where a row weighs one class
    alone."""
    # scores: (n, G); labels: (n,) 0 or 1; weights: (B, n), nonnegative
    hit = labels.to(device=scores.device, dtype=torch.float64)  # (n,)
    weights = weights.to(device=scores.device, dtype=torch.float64)
    roc_aucs, precisions = [], []
    for column in scores.T:
        ordered, order = column.sort(descending=True)  # (n,)
        ends = torch.ones_like(ordered, dtype=torch.bool)  # (n,), the last row of each run of equal scores
        ends[:-1] = ordered[:-1] != ordered[1:]
        weight = weights[:, order]  # (B, n)
        tp = (weight * hit[order]).cumsum(dim=1)[:, ends]  # (B, r), positive weight at or above each threshold
        fp = (weight * (1.0 - hit[order])).cumsum(dim=1)[:, ends]
        tp_above, fp_above = F.pad(tp[:, :-1], (1, 0)), F.pad(fp[:, :-1], (1, 0))  # (B, r), at the threshold before
        positives, negatives = tp[:, -1], fp[:, -1]  # (B,)
        both = (positives > 0) & (negatives > 0)
        area = ((fp - fp_above) * (tp + tp_above)).sum(dim=1) / 2  # trapezoids between thresholds
        precision = torch.where(tp + fp > 0, tp / (tp + fp).clamp_min(1e-300), 0.0)  # (B, r)
        roc_aucs.append(torch.where(both, area / (positives * negatives).clamp_min(1e-300), torch.nan))
        precisions.append(torch.where(both, ((tp - tp_above) * precision).sum(dim=1) / positives.clamp_min(1e-300), torch.nan))
    return torch.stack(roc_aucs), torch.stack(precisions)  # (G, B), (G, B)


def label_scores(Z: Tensor, targets: Tensor, kind: str) -> tuple[Tensor, Tensor]:
    """One binary ranking problem per label of a protein task: its scores and its 0/1 labels. A multiclass task's
    labels are its classes, one against the rest, ranked by log-probability; a binary or multilabel task's are its
    outputs, ranked by logit."""
    # Z: (n, G, C) logits; targets: (n,) classes, or (n, C) 0/1 -> scores (n, G, L), labels (n, L)
    if kind == "multiclass":
        return torch.log_softmax(Z.to(torch.float32), dim=-1), F.one_hot(targets.long(), Z.shape[-1])
    return Z.to(torch.float32), targets.reshape(Z.shape[0], -1)


def ranked_labels(labels: Tensor) -> list[int]:
    """The labels a split holds both classes of, the only ones a ranking metric is defined for."""
    # labels: (n, L) 0 or 1
    positives = labels.sum(dim=0)  # (L,)
    return torch.nonzero((positives > 0) & (positives < labels.shape[0])).flatten().tolist()


def ranking_summary(Z: Tensor, targets: Tensor, kind: str) -> dict[str, Tensor]:
    """Macro ROC AUC, average precision and balanced MCC of each group of a classification task, over the labels the
    split holds both classes of: (G,) each, NaN when it holds none."""
    # Z: (n, G, C) logits; targets: (n,) or (n, C)
    scores, labels = label_scores(Z, targets, kind)  # (n, G, L), (n, L)
    kept = ranked_labels(labels)
    totals = {name: torch.zeros(Z.shape[1], dtype=torch.float64, device=Z.device) for name in RANKING_METRICS}  # (G,) each
    for label in kept:
        column, truth = scores[..., label], labels[:, label]  # (n, G), (n,)
        roc_auc, average_precision = ranking_metrics(column, truth)
        totals["roc_auc"] += roc_auc
        totals["average_precision"] += average_precision
        totals["balanced_mcc"] += balanced_mcc(column, truth)
    return {name: total / len(kept) if kept else torch.full_like(total, torch.nan) for name, total in totals.items()}


def drawn_ranking(Z: Tensor, targets: Tensor, kind: str, weights: Tensor) -> dict[str, Tensor]:
    """Macro ROC AUC and average precision of each group under each row of `weights` (whole split and bootstrap
    draws, `draw_plan`): (G, B) each. A label that a draw holds one class of is left out of that draw's mean."""
    # Z: (n, G, C) logits; targets: (n,) or (n, C); weights: (B, n)
    scores, labels = label_scores(Z, targets, kind)  # (n, G, L), (n, L)
    shape = (Z.shape[1], weights.shape[0])
    sums = {name: torch.zeros(shape, dtype=torch.float64, device=Z.device) for name in ("roc_auc", "average_precision")}  # (G, B) each
    counts = {name: torch.zeros(shape, dtype=torch.float64, device=Z.device) for name in sums}
    for label in ranked_labels(labels):
        for name, values in zip(sums, weighted_ranking_metrics(scores[..., label], labels[:, label], weights)):  # (G, B) each
            defined = ~values.isnan()
            sums[name] += torch.where(defined, values, 0.0)
            counts[name] += defined
    return {name: torch.where(counts[name] > 0, sums[name] / counts[name].clamp_min(1), torch.nan) for name in sums}
