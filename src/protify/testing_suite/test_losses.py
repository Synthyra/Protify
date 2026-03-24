"""Tests for probes/losses.py loss function dispatch and custom losses."""

import torch
import pytest
from torch import nn

try:
    from src.protify.probes.losses import get_loss_fct, SoftBCELoss, SoftBCEWithLogitsLoss
except ImportError:
    try:
        from protify.probes.losses import get_loss_fct, SoftBCELoss, SoftBCEWithLogitsLoss
    except ImportError:
        from ..probes.losses import get_loss_fct, SoftBCELoss, SoftBCEWithLogitsLoss


def test_get_loss_fct_singlelabel() -> None:
    loss = get_loss_fct("singlelabel")
    assert isinstance(loss, nn.CrossEntropyLoss)


def test_get_loss_fct_multilabel() -> None:
    loss = get_loss_fct("multilabel")
    assert isinstance(loss, nn.BCEWithLogitsLoss)


def test_get_loss_fct_regression() -> None:
    loss = get_loss_fct("regression", tokenwise=False)
    assert isinstance(loss, nn.MSELoss)


def test_get_loss_fct_sigmoid_regression() -> None:
    loss = get_loss_fct("sigmoid_regression")
    assert isinstance(loss, SoftBCELoss)


def test_get_loss_fct_tokenwise_classification() -> None:
    loss = get_loss_fct("singlelabel", tokenwise=True)
    # tokenwise singlelabel should still be CrossEntropyLoss (singlelabel branch first)
    assert isinstance(loss, nn.CrossEntropyLoss)


def test_get_loss_fct_tokenwise_nonregression() -> None:
    loss = get_loss_fct("string", tokenwise=True)
    assert isinstance(loss, nn.CrossEntropyLoss)


def test_soft_bce_loss_forward() -> None:
    torch.manual_seed(0)
    loss_fn = SoftBCELoss()
    y_pred = torch.sigmoid(torch.randn(4, 3))
    y_true = torch.randint(0, 2, (4, 3)).float()
    loss = loss_fn(y_pred, y_true)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_soft_bce_loss_ignore_index() -> None:
    torch.manual_seed(0)
    loss_fn = SoftBCELoss(ignore_index=-100.0)
    y_pred = torch.sigmoid(torch.randn(4))
    y_true = torch.tensor([1.0, 0.0, -100.0, 1.0])
    loss = loss_fn(y_pred, y_true)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_soft_bce_loss_all_ignored() -> None:
    loss_fn = SoftBCELoss(ignore_index=-100.0)
    y_pred = torch.sigmoid(torch.randn(3))
    y_true = torch.tensor([-100.0, -100.0, -100.0])
    loss = loss_fn(y_pred, y_true)
    assert loss.item() == 0.0


def test_soft_bce_with_logits_loss_forward() -> None:
    torch.manual_seed(0)
    loss_fn = SoftBCEWithLogitsLoss()
    y_pred = torch.randn(4, 3)
    y_true = torch.randint(0, 2, (4, 3)).float()
    loss = loss_fn(y_pred, y_true)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_soft_bce_with_logits_smooth_factor_changes_loss() -> None:
    torch.manual_seed(0)
    y_pred = torch.randn(4, 3)
    y_true = torch.randint(0, 2, (4, 3)).float()

    loss_no_smooth = SoftBCEWithLogitsLoss(smooth_factor=None)(y_pred, y_true)
    loss_smooth = SoftBCEWithLogitsLoss(smooth_factor=0.1)(y_pred, y_true)
    assert loss_no_smooth.item() != loss_smooth.item()


def test_soft_bce_with_logits_all_ignored() -> None:
    loss_fn = SoftBCEWithLogitsLoss(ignore_index=-100.0)
    y_pred = torch.randn(3)
    y_true = torch.tensor([-100.0, -100.0, -100.0])
    loss = loss_fn(y_pred, y_true)
    assert loss.item() == 0.0
