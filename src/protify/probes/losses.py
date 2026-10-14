"""Probe training losses.

`get_loss_fct`, `SoftBCEWithLogitsLoss`, and `SoftBCELoss` are `interpnet.losses`'s, so Protify's
probes and InterpNet train on one implementation. `compute_probe_loss` is Protify's own.
"""

import copy
import torch

from torch import nn

from interpnet.losses import SoftBCELoss, SoftBCEWithLogitsLoss, get_loss_fct


__all__ = ["SoftBCELoss", "SoftBCEWithLogitsLoss", "compute_probe_loss", "get_loss_fct"]


def compute_probe_loss(
    loss_fct: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_type: str,
    num_labels: int,
    label_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reshape logits and labels the way a task's loss expects, then reduce.

    Each loss module carries its own view convention: the regression losses read one
    flat vector, multilabel reads (b, c) against float targets, and cross-entropy reads
    (b, c) against class indices. `sigmoid_regression` arrives already squashed, since
    SoftBCELoss reads probabilities rather than logits.
    """
    # logits: (b, c); labels, label_mask: (b,), (b, 1), or (b, c)
    if label_mask is not None:
        return _masked_probe_loss(loss_fct, logits, labels, task_type, num_labels, label_mask)  # ()
    if task_type in ('regression', 'sigmoid_regression'):
        return loss_fct(logits.view(-1), labels.view(-1).float())  # ()

    batch_size = logits.size(0)
    if task_type == 'multilabel':
        return loss_fct(
            logits.view(batch_size, -1),  # (b, c)
            labels.view(batch_size, -1).float(),  # (b, c)
        )  # ()

    return loss_fct(
        logits.view(-1, num_labels),  # (b, c)
        labels.view(-1).long(),  # (b,)
    )  # ()


def validate_label_mask(labels: torch.Tensor, label_mask: torch.Tensor) -> None:
    """An explicit mask is boolean, shape-exact, and observes finite targets."""
    # labels, label_mask: (...), identical shapes
    if label_mask.dtype != torch.bool or label_mask.shape != labels.shape:
        raise ValueError("label_mask must be boolean with exactly the labels shape")
    if label_mask.device != labels.device:
        raise ValueError("label_mask and labels must be on the same device")
    if not torch.isfinite(labels[label_mask]).all():
        raise ValueError("observed labels must be finite")


def _masked_probe_loss(
    loss_fct: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_type: str,
    num_labels: int,
    label_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the existing criterion only to observed targets, including an empty batch.

    Elementwise criteria keep their full shape so per-label weights still broadcast to the
    intended columns. A shallow criterion copy changes only reduction and ignored sentinels;
    the caller's module and weight buffers are not mutated.
    """
    # logits: (b, c); labels, label_mask: (b,), (b, 1), or (b, c); m = observed targets
    validate_label_mask(labels, label_mask)
    if task_type == "singlelabel":
        kept = label_mask.reshape(-1)  # (b,)
        selected_logits = logits.reshape(-1, num_labels)[kept]  # (m, c)
        selected_labels = labels.reshape(-1)[kept]  # (m,)
        if not kept.any():
            return selected_logits.sum()  # (), differentiable zero
        if ((selected_labels != selected_labels.long()) | (selected_labels < 0) | (selected_labels >= num_labels)).any():
            raise ValueError("observed class labels must be integer indices within the output vocabulary")
        return loss_fct(selected_logits, selected_labels.long())  # ()
    if task_type not in {"multilabel", "regression", "sigmoid_regression"}:
        raise ValueError(f"label masks are unsupported for task type {task_type!r}")
    if logits.numel() != labels.numel():
        raise ValueError("elementwise labels and logits must contain the same number of targets")
    reduction = getattr(loss_fct, "reduction", None)
    if reduction not in {"mean", "sum"}:
        raise ValueError("masked probe loss requires a mean or sum criterion")
    criterion = copy.copy(loss_fct)
    criterion.reduction = "none"
    if isinstance(criterion, (SoftBCELoss, SoftBCEWithLogitsLoss)):
        criterion.ignore_index = None
    safe_labels = torch.where(label_mask, labels, 0)  # labels shape
    safe_logits = torch.where(label_mask.reshape_as(logits), logits, 0)  # (b, c)
    losses = compute_probe_loss(criterion, safe_logits, safe_labels, task_type, num_labels)  # labels shape or (b,)
    if losses.numel() != label_mask.numel():
        raise ValueError("criterion must return one loss per target before masked reduction")
    observed_losses = losses.reshape(-1)[label_mask.reshape(-1)]  # (m,)
    total = observed_losses.sum()  # (), differentiable even when m = 0
    return total if reduction == "sum" else total / max(observed_losses.numel(), 1)  # ()
