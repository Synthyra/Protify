"""Temperature scaling: one positive scalar that divides logits to minimize held-out loss.

Dividing every logit by the same temperature changes no ranking and no argmax, so accuracy and
AUROC stay as they are while probabilities move toward or away from certainty. Fit it on
validation logits, never on the test set.

The objective and optimizer match sae_xgboost_ppi's historical `gp6b.fit_temperature`.
Bounded blocks accumulate that same mean loss and gradient, without retaining a full graph.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from collections.abc import Callable, Iterator


TEMPERATURE_BATCH_ELEMENTS = 1_048_576


def _temperature_batches(
    logits: np.ndarray, labels: np.ndarray, label_mask: np.ndarray | None, batch_rows: int,
    release_batch: Callable[[], None] | None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Select observed entries within one block; let its owner release mapped input pages.

    Each block holds `b` rows. Unmasked it yields logits and labels `(b, ...)` binary, or `(b, c)`
    and `(b,)` multiclass; masked, the `m` observed entries `(m,)` and `(m,)` binary, or rows
    `(m, c)` and `(m,)` multiclass.
    """
    # logits: (n, ...) binary, or (n, c) multiclass; labels, label_mask: (n, ...) binary, or (n,) multiclass
    for start in range(0, len(logits), batch_rows):
        try:
            block_logits = logits[start:start + batch_rows]  # (b, ...) or (b, c)
            block_labels = labels[start:start + batch_rows]  # (b, ...) or (b,)
            if label_mask is not None:
                observed = label_mask[start:start + batch_rows]  # (b, ...) or (b,), as block_labels
                block_logits = block_logits[observed]  # (m,) binary, or (m, c) multiclass
                block_labels = block_labels[observed]  # (m,)
            if block_labels.size:
                yield block_logits, block_labels
        finally:
            if release_batch is not None:
                release_batch()


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, *, multiclass: bool = False,
    label_mask: np.ndarray | None = None, batch_elements: int = TEMPERATURE_BATCH_ELEMENTS,
    release_batch: Callable[[], None] | None = None,
) -> float:
    """The temperature `T` whose `logits / T` minimize the mean negative log-likelihood of `labels`.

    Binary or multilabel: `logits` and `labels` share a shape, labels in {0, 1}, scored by
    binary cross-entropy averaged over every entry. Multiclass: `logits` (n, c) and integer
    `labels` (n,) in [0, c), scored by cross-entropy. L-BFGS optimizes `log T` in float64 from
    `T = 1`, which keeps `T` positive. A boolean `label_mask` matching `labels` selects
    observed entries (binary) or rows (multiclass); unknown fillers are never scored.

    Each block holds at most `batch_elements` logits, or one complete row when it is wider.
    Blocks contribute their observed count divided by the total, not equal block weights.
    `release_batch`, when supplied by a mapped-input owner, runs after each block is consumed,
    including validation and empty blocks. It may release clean pages, but must not mutate
    the inputs, which remain immutable throughout fitting. Small inputs retain one-block
    arithmetic; multiple blocks may differ by floating-point reduction roundoff.
    """
    # logits: (...) binary or multilabel, or (n, c) multiclass; labels, label_mask: (...) as logits, or (n,) multiclass
    if type(batch_elements) is not int or batch_elements < 1:
        raise ValueError("temperature batch_elements must be a positive integer")
    logits, labels = np.asanyarray(logits), np.asanyarray(labels)  # (...) and (...) binary; (n, c) and (n,) multiclass
    if multiclass:
        objective = F.cross_entropy
        if logits.ndim != 2 or labels.shape != logits.shape[:1]:
            raise ValueError(f"multiclass needs logits (n, c) and labels (n,), got {logits.shape} and {labels.shape}")
    else:
        objective = F.binary_cross_entropy_with_logits
        if logits.shape != labels.shape:
            raise ValueError(f"binary logits and labels must share a shape, got {logits.shape} and {labels.shape}")
    if not logits.size:
        raise ValueError("fitting a temperature needs at least one logit")
    if label_mask is not None:
        label_mask = np.asanyarray(label_mask)  # labels' shape
        if label_mask.dtype != np.bool_ or label_mask.shape != labels.shape:
            raise ValueError("label_mask must be boolean with exactly the labels shape")
    if logits.ndim == 0:
        logits, labels = logits.reshape(1), labels.reshape(1)  # (1,), (1,)
        label_mask = None if label_mask is None else label_mask.reshape(1)  # (1,)
    batch_rows = max(1, batch_elements // math.prod(logits.shape[1:]))  # b

    count = 0
    for block_logits, block_labels in _temperature_batches(logits, labels, label_mask, batch_rows, release_batch):
        if not np.isfinite(block_logits).all():
            raise ValueError("observed calibration predictions must be finite")
        if not np.isfinite(block_labels).all():
            raise ValueError("observed labels must be finite")
        count += block_labels.size
    if not count:
        raise ValueError("temperature calibration requires observed validation labels")

    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)  # ()
    optimizer = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        total_loss = torch.zeros((), dtype=torch.float64)  # ()
        for block_logits, block_labels in _temperature_batches(logits, labels, label_mask, batch_rows, release_batch):
            # Copy only the current block, including read-only or noncontiguous source views.
            # Binary: (b, ...) or (m,) each. Multiclass: (b, c) or (m, c), and (b,) or (m,).
            logit_tensor = torch.from_numpy(np.array(block_logits, dtype=np.float64, copy=True, order="C"))
            label_tensor = torch.from_numpy(
                np.array(block_labels, dtype=np.int64 if multiclass else np.float64, copy=True, order="C")
            )
            loss = objective(logit_tensor / torch.exp(log_t), label_tensor) * (label_tensor.numel() / count)  # ()
            loss.backward()
            total_loss += loss.detach()
        return total_loss  # ()

    optimizer.step(closure)

    return float(torch.exp(log_t).detach())
