"""Training losses for InterpNet sequence and token tasks."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torch import nn


def get_loss_fct(task_type: str, tokenwise: bool = False) -> nn.Module:
    """Return the training loss for a task type."""
    if task_type == "singlelabel":
        return nn.CrossEntropyLoss()

    if task_type == "multilabel":
        return nn.BCEWithLogitsLoss()

    if tokenwise and task_type != "regression":
        return nn.CrossEntropyLoss()

    if task_type == "regression" and not tokenwise:
        return nn.MSELoss()

    if task_type == "sigmoid_regression":
        return SoftBCELoss()

    raise ValueError(f"Unsupported task type: {task_type}")


class SoftBCEWithLogitsLoss(nn.Module):
    """Binary cross entropy on logits, with optional label smoothing and an ignored target value."""

    __constants__ = [
        "weight",
        "pos_weight",
        "reduction",
        "ignore_index",
        "smooth_factor",
    ]

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        ignore_index: float | None = -100.0,
        reduction: str = "mean",
        smooth_factor: float | None = None,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        # weight, pos_weight: (...) shaped like the targets, or None. Without an ignore_index
        # they need only broadcast to the targets.
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Return the loss over the targets that do not equal `ignore_index`."""
        # y_pred, y_true: (...), the same shape; n_kept counts the targets not ignored
        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (1 - self.smooth_factor)  # (...)
        else:
            soft_targets = y_true  # (...)

        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index  # (...)
            if not torch.any(not_ignored_mask):
                return torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)  # ()

            y_pred = y_pred[not_ignored_mask]  # (n_kept,)
            soft_targets = soft_targets[not_ignored_mask]  # (n_kept,)
            weight = self.weight[not_ignored_mask] if self.weight is not None else None  # (n_kept,) or None
            pos_weight = (
                self.pos_weight[not_ignored_mask] if self.pos_weight is not None else None
            )  # (n_kept,) or None
            loss = F.binary_cross_entropy_with_logits(
                y_pred,
                soft_targets,
                weight,
                pos_weight=pos_weight,
                reduction="none",
            )  # (n_kept,)
        else:
            loss = F.binary_cross_entropy_with_logits(
                y_pred,
                soft_targets,
                self.weight,
                pos_weight=self.pos_weight,
                reduction="none",
            )  # (...)

        if self.reduction == "mean":
            loss = loss.mean()  # ()

        if self.reduction == "sum":
            loss = loss.sum()  # ()

        return loss  # () when reduced, else (n_kept,) or (...)


class SoftBCELoss(nn.Module):
    """Binary cross entropy on probabilities, with optional label smoothing and an ignored target value."""

    __constants__ = [
        "weight",
        "pos_weight",
        "reduction",
        "ignore_index",
        "smooth_factor",
    ]

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        ignore_index: float | None = -100.0,
        reduction: str = "mean",
        smooth_factor: float | None = None,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        # TODO(interpnet_soft_bce_pos_weight): SoftBCELoss accepts pos_weight and never applies it
        # weight, pos_weight: (...) shaped like the targets, or None. Without an ignore_index
        # they need only broadcast to the targets. `F.binary_cross_entropy` takes no positive
        # weight, so `pos_weight` is stored with the module and never read.
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Return the loss over the targets that do not equal `ignore_index`."""
        # y_pred, y_true: (...), the same shape; n_kept counts the targets not ignored
        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (1 - self.smooth_factor)  # (...)
        else:
            soft_targets = y_true  # (...)

        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index  # (...)
            if not torch.any(not_ignored_mask):
                return torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)  # ()

            y_pred = y_pred[not_ignored_mask]  # (n_kept,)
            soft_targets = soft_targets[not_ignored_mask]  # (n_kept,)
            weight = self.weight[not_ignored_mask] if self.weight is not None else None  # (n_kept,) or None

            loss = F.binary_cross_entropy(
                y_pred,
                soft_targets,
                weight=weight,
                reduction="none",
            )  # (n_kept,)
        else:
            loss = F.binary_cross_entropy(
                y_pred,
                soft_targets,
                weight=self.weight,
                reduction="none",
            )  # (...)

        if self.reduction == "mean":
            loss = loss.mean()  # ()

        if self.reduction == "sum":
            loss = loss.sum()  # ()

        return loss  # () when reduced, else (n_kept,) or (...)
