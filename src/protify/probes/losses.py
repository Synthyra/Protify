import torch
import torch.nn.functional as F

from torch import nn

try:
    from ..utils import print_message
except ImportError:
    try:
        from protify.utils import print_message
    except ImportError:
        from utils import print_message


def get_loss_fct(task_type: str, tokenwise: bool = False) -> nn.Module:
    """Return the configured loss module for a probe task."""
    if task_type == 'singlelabel':
        loss_fct = nn.CrossEntropyLoss()
    elif task_type == 'multilabel':
        loss_fct = nn.BCEWithLogitsLoss()
    elif tokenwise and task_type != 'regression':
        loss_fct = nn.CrossEntropyLoss()
    elif task_type == 'regression' and not tokenwise:
        loss_fct = nn.MSELoss()
    elif task_type == 'sigmoid_regression':
        loss_fct = SoftBCELoss()
    else:
        print_message(f'Specified wrong classification type {task_type}')
    return loss_fct


# Adapted from segmentation-models-pytorch's soft BCE implementation:
# https://smp.readthedocs.io/en/latest/_modules/segmentation_models_pytorch/losses/soft_bce.html
class SoftBCEWithLogitsLoss(nn.Module):
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
        """Add ignore-index and optional smoothing to BCE-with-logits."""
        super().__init__()
        # weight and pos_weight follow torch BCE broadcasting rules when no
        # ignore index is used. Masking requires tensors shaped like y_true.
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred and y_true: (...) with matching, loss-compatible shapes.
        if self.smooth_factor is not None:
            soft_targets = (
                (1 - y_true) * self.smooth_factor
                + y_true * (1 - self.smooth_factor)
            )  # (...)
        else:
            soft_targets = y_true  # (...)

        # Exclude ignored targets before BCE so they cannot contribute gradients.
        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index  # (...)
            if not torch.any(not_ignored_mask):
                return torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)  # ()

            y_pred = y_pred[not_ignored_mask]  # (n_valid,)
            soft_targets = soft_targets[not_ignored_mask]  # (n_valid,)
            weight = (
                self.weight[not_ignored_mask]
                if self.weight is not None
                else None
            )  # (n_valid,) or None
            pos_weight = (
                self.pos_weight[not_ignored_mask]
                if self.pos_weight is not None
                else None
            )  # (n_valid,) or None
            loss = F.binary_cross_entropy_with_logits(
                y_pred,
                soft_targets,
                weight,
                pos_weight=pos_weight,
                reduction="none",
            )  # (n_valid,)
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

        # () for mean/sum (and the all-ignored early return); reduction="none"
        # is (n_valid,) with masking and otherwise preserves (...).
        return loss


class SoftBCELoss(nn.Module):
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
        """Add ignore-index and optional smoothing to probability-space BCE."""
        super().__init__()
        # weight follows torch BCE broadcasting rules when no ignore index is
        # used. Masking requires a tensor shaped like y_true. pos_weight is
        # retained for API compatibility.
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred and y_true: (...) with matching, loss-compatible shapes.
        if self.smooth_factor is not None:
            soft_targets = (
                (1 - y_true) * self.smooth_factor
                + y_true * (1 - self.smooth_factor)
            )  # (...)
        else:
            soft_targets = y_true  # (...)

        # Exclude ignored targets before BCE so they cannot contribute gradients.
        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index  # (...)
            if not torch.any(not_ignored_mask):
                return torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)  # ()

            y_pred = y_pred[not_ignored_mask]  # (n_valid,)
            soft_targets = soft_targets[not_ignored_mask]  # (n_valid,)
            weight = (
                self.weight[not_ignored_mask]
                if self.weight is not None
                else None
            )  # (n_valid,) or None

            # PyTorch BCE expects probabilities (after sigmoid) and does not
            # support pos_weight. We ignore pos_weight here on purpose.
            loss = F.binary_cross_entropy(
                y_pred,
                soft_targets,
                weight=weight,
                reduction="none",
            )  # (n_valid,)
        else:
            # PyTorch BCE expects probabilities (after sigmoid) and does not
            # support pos_weight. We ignore pos_weight here on purpose.
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

        # () for mean/sum (and the all-ignored early return); reduction="none"
        # is (n_valid,) with masking and otherwise preserves (...).
        return loss
