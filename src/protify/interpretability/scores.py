"""Test scores of many probes at once, on the whole test split and on random halves of its proteins.

Scores are accumulated protein by protein, so a half is just a weighted sum of per-protein counts: row 0 of the
resample axis is the full split and rows 1..R are fixed random halves (R = 7 by default). Classification keeps
confusion counts and reports the Matthews correlation coefficient; regression keeps cosine terms and reports the
mean per-protein cosine similarity between the predicted and true standardized vectors, plus each target's cosine
across proteins.

Symbols: n rows (proteins, or residues); p proteins of a batch; g groups; C outputs; R resamples; P test proteins.
"""

from __future__ import annotations

import numpy as np
import torch

from dataclasses import dataclass
from torch import Tensor


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
    """Counts for G probes of one task, folded per resample as batches arrive."""

    def __init__(self, kind: str, groups: int, outputs: int, plan: ResamplePlan) -> None:
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
            self.per_label = torch.zeros(rows, groups, outputs, 4, dtype=torch.float64, device=device)
        elif kind == "regression":
            self.cosine_sum = torch.zeros(rows, groups, dtype=torch.float64, device=device)
            self.proteins = torch.zeros(rows, dtype=torch.float64, device=device)
            self.terms = torch.zeros(rows, groups, outputs, 3, dtype=torch.float64, device=device)
        else:
            raise ValueError(f"Unknown task kind {kind!r}.")

    def add(self, Z: Tensor, targets: Tensor, proteins: Tensor, start: int = 0) -> None:
        """Add the outputs of groups start.. for rows that belong to test proteins `proteins`.

        Z holds logits; targets are class indices (multiclass) or values; `proteins` gives each row's test index.
        """
        # Z: (n, g, C); targets: (n,) or (n, C); proteins: (n,) long
        g = Z.shape[1]
        local, inverse = torch.unique(proteins, return_inverse=True)  # (p,), (n,)
        weights = self.plan.membership[:, local].to(torch.float64)  # (R + 1, p)
        stop = start + g
        if self.kind == "multiclass":
            C = self.outputs
            predicted = Z.argmax(dim=-1)  # (n, g)
            cells = (inverse.unsqueeze(1) * g + torch.arange(g, device=Z.device)) * (C * C) + targets.unsqueeze(1) * C + predicted  # (n, g)
            per_protein = torch.bincount(cells.reshape(-1), minlength=len(local) * g * C * C).to(torch.float64)  # (p * g * C * C,)
            folded = weights @ per_protein.view(len(local), g * C * C)  # (R + 1, g * C * C)
            self.confusion[:, start:stop] += folded.view(-1, g, C, C)
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
        """Per-label MCC, or per-target cosine across proteins: (R + 1, G, C); None for multiclass."""
        if self.kind in ("multilabel", "binary"):
            return binary_mcc(self.per_label)
        if self.kind == "regression":
            products, predicted, truth = self.terms.unbind(dim=-1)
            return products / torch.sqrt(predicted * truth).clamp_min(1e-300)
        return None
