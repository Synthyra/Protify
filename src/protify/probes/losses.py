import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional

try:
    from ..utils import print_message
except ImportError:
    try:
        from protify.utils import print_message
    except ImportError:
        from utils import print_message


def get_loss_fct(task_type, tokenwise: bool = False):
    """
    Returns loss function based on task type
    """
    if task_type == 'singlelabel':
        loss_fct = nn.CrossEntropyLoss()
    elif task_type == 'multilabel':
        loss_fct = nn.BCEWithLogitsLoss()
    elif tokenwise and not task_type == 'regression':
        loss_fct = nn.CrossEntropyLoss()
    elif task_type == 'regression' and not tokenwise:
        loss_fct = nn.MSELoss()
    elif task_type == 'sigmoid_regression':
        loss_fct = SoftBCELoss()
    else:
        print_message(f'Specified wrong classification type {task_type}')
    return loss_fct


### https://smp.readthedocs.io/en/latest/_modules/segmentation_models_pytorch/losses/soft_bce.html#SoftBCEWithLogitsLoss
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
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[float] = -100.0,
        reduction: str = "mean",
        smooth_factor: Optional[float] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """Drop-in replacement for torch.nn.BCEWithLogitsLoss with few additions:
        ignore_index and label_smoothing

        Args:
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 1] -> [0.9, 0.1, 0.9])

        Shape
             - **y_pred** - torch.Tensor of shape NxCxHxW
             - **y_true** - torch.Tensor of shape NxHxW or Nx1xHxW

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: torch.Tensor of shape (N, C, H, W)
            y_true: torch.Tensor of shape (N, H, W)  or (N, 1, H, W)

        Returns:
            loss: torch.Tensor
        """

        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (1 - self.smooth_factor)
        else:
            soft_targets = y_true

        # If we have an ignore_index, exclude ignored targets BEFORE computing BCE
        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index
            if not torch.any(not_ignored_mask):
                return torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)

            y_pred = y_pred[not_ignored_mask]
            soft_targets = soft_targets[not_ignored_mask]
            weight = self.weight[not_ignored_mask] if self.weight is not None else None
            pos_weight = self.pos_weight[not_ignored_mask] if self.pos_weight is not None else None
            loss = F.binary_cross_entropy_with_logits(
                y_pred,
                soft_targets,
                weight,
                pos_weight=pos_weight,
                reduction="none",
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                y_pred,
                soft_targets,
                self.weight,
                pos_weight=self.pos_weight,
                reduction="none",
            )

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

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
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[float] = -100.0,
        reduction: str = "mean",
        smooth_factor: Optional[float] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """Drop-in replacement for torch.nn.BCELoss with few additions:
        ignore_index and label_smoothing

        Args:
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 1] -> [0.9, 0.1, 0.9])

        Shape
             - **y_pred** - torch.Tensor of shape NxHxW
             - **y_true** - torch.Tensor of shape NxHxW or Nx1xHxW

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: torch.Tensor of shape (N, H, W)
            y_true: torch.Tensor of shape (N, H, W)  or (N, 1, H, W)

        Returns:
            loss: torch.Tensor
        """

        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (1 - self.smooth_factor)
        else:
            soft_targets = y_true

        # If we have an ignore_index, exclude ignored targets BEFORE computing BCE
        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index
            if not torch.any(not_ignored_mask):
                return torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)

            y_pred = y_pred[not_ignored_mask]
            soft_targets = soft_targets[not_ignored_mask]
            weight = self.weight[not_ignored_mask] if self.weight is not None else None

            # PyTorch BCE expects probabilities (after sigmoid) and does not
            # support pos_weight. We ignore pos_weight here on purpose.
            loss = F.binary_cross_entropy(
                y_pred,
                soft_targets,
                weight=weight,
                reduction="none",
            )
        else:
            # PyTorch BCE expects probabilities (after sigmoid) and does not
            # support pos_weight. We ignore pos_weight here on purpose.
            loss = F.binary_cross_entropy(
                y_pred,
                soft_targets,
                weight=self.weight,
                reduction="none",
            )

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss


### tests
if __name__ == '__main__':
    pass ### TODO
