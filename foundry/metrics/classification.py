"""The single-label classification metrics DSM reports, as a Hugging Face `compute_metrics`.

DSM's `metrics.classification` held them, and egfr_binding_proxies copied that module whole.
"""

from __future__ import annotations

import numpy as np
import torch

from typing import TYPE_CHECKING
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


if TYPE_CHECKING:
    from transformers import EvalPrediction


__all__ = ["compute_metrics_single_label_classification"]


# The label that marks a position to leave out, as Hugging Face models mark it.
IGNORE_LABEL = -100
# What `auc` reports when there is no AUC to give.
NO_AUC = -100


def compute_metrics_single_label_classification(p: EvalPrediction) -> dict[str, float]:
    """F1, precision, recall, accuracy, MCC, and AUC of single-label predictions, to 5 places.

    The predictions are `p.predictions`, or its first element when it is a tuple, and the labels
    are `p.label_ids`, or its second element. Predictions shaped like the labels are one score
    each, called positive above 0.5; any other shape holds a score per class on its last axis,
    and its arg max is the prediction. Labels of -100 are left out, and labels are cast to int32.
    The confusion matrix over the classes is printed. F1, precision, and recall are averaged over
    the classes weighted by their support, with 0 for a class never predicted.

    `auc` does not rank the scores. It is -100 for one score per label. For per-class scores it is
    the ROC AUC of the arg-max classes: with two classes, that of the hard predictions, which
    scikit-learn makes NaN, with a warning, when the labels left hold one class. A single class of
    scores gives NaN the same way. With three or more classes, and wherever scikit-learn raises,
    it is -100.
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids

    # Scores are (...) with the labels' shape, one per label, or (..., c), one per class.
    preds = torch.tensor(np.array(preds))  # (...) or (..., c)
    y_true = torch.tensor(np.array(labels), dtype=torch.int)  # (...), n elements

    if y_true.size() == preds.size():
        y_pred = (preds > 0.5).int().flatten()  # (n,)
        n_classes = 2  # c
        auc_classes = None
    else:
        n_classes = preds.shape[-1]  # c
        y_pred = preds.argmax(dim=-1).flatten()  # (n_p,), one class per row of scores
        # The multiclass AUC's input: classes, not scores, and taken before any are left out.
        auc_classes = y_pred.clone()  # (n_p,)

    all_classes = np.arange(n_classes)  # (c,)

    y_true = y_true.flatten()  # (n,)
    valid_indices = y_true != IGNORE_LABEL  # (n,)
    y_pred = y_pred[valid_indices]  # (n_v,)
    y_true = y_true[valid_indices]  # (n_v,)

    y_true_np = y_true.numpy()  # (n_v,)
    y_pred_np = y_pred.numpy()  # (n_v,)

    confusion = confusion_matrix(y_true_np, y_pred_np, labels=all_classes)  # (c, c)
    print("\nConfusion Matrix:")
    print(confusion)

    f1 = f1_score(y_true_np, y_pred_np, average="weighted", labels=all_classes, zero_division=0)
    precision = precision_score(
        y_true_np, y_pred_np, average="weighted", labels=all_classes, zero_division=0
    )
    recall = recall_score(y_true_np, y_pred_np, average="weighted", labels=all_classes, zero_division=0)
    accuracy = accuracy_score(y_true_np, y_pred_np)
    mcc = matthews_corrcoef(y_true_np, y_pred_np)

    # Any failure of scikit-learn's AUC reports NO_AUC, as DSM's always has.
    if auc_classes is None:
        auc = NO_AUC
    elif n_classes == 2:
        try:
            auc = roc_auc_score(y_true_np, y_pred_np)
        except Exception:
            auc = NO_AUC
    else:
        y_true_bin = label_binarize(y_true_np, classes=all_classes)  # (n_v, c)
        try:
            auc = roc_auc_score(y_true_bin, auc_classes.numpy(), average="weighted", multi_class="ovr")
        except Exception:
            auc = NO_AUC

    return {
        "f1": round(f1, 5),
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "accuracy": round(accuracy, 5),
        "mcc": round(mcc, 5),
        "auc": round(auc, 5),
    }
