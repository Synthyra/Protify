import numpy as np
import torch

from collections.abc import Callable
from typing import Any
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    hamming_loss,
    make_scorer,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    precision_recall_curve,
    r2_score,
    recall_score,
    roc_auc_score,
)
from transformers import EvalPrediction

from interpnet.losses import get_loss_fct


def softmax(x: np.ndarray) -> np.ndarray:
    # x: (..., c)
    x = x - x.max(axis=-1, keepdims=True)  # (..., c)
    exponentials = np.exp(x)  # (..., c)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)  # (..., c)


def regression_scorer() -> Callable[[np.ndarray, np.ndarray], float]:
    def dual_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # y_true: (n,); y_pred: (n,)
        return spearmanr(y_true, y_pred).correlation * r2_score(y_true, y_pred)

    return dual_score  # a scorer of y_true: (n,), y_pred: (n,)


def classification_scorer() -> Callable[[np.ndarray, np.ndarray], float]:
    def mcc_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # y_true: (n,); y_pred: (n,)
        return matthews_corrcoef(y_true, y_pred)

    return mcc_scorer  # a scorer of y_true: (n,), y_pred: (n,)


def get_classification_scorer() -> Any:
    return make_scorer(classification_scorer(), greater_is_better=True)


def get_regression_scorer() -> Any:
    return make_scorer(regression_scorer(), greater_is_better=True)


def calculate_max_metrics(
    ss: torch.Tensor,
    labels: torch.Tensor,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate precision, recall and F1 metrics for binary classification at a specific cutoff threshold.

    Args:
        ss: Prediction scores tensor, typically between -1 and 1
        labels: Ground truth binary labels tensor (0 or 1)
        cutoff: Classification threshold value

    Returns:
        Tuple containing:
            - F1 score (torch.Tensor)
            - Precision score (torch.Tensor) 
            - Recall score (torch.Tensor)

    Note:
        - Input tensors are converted to float type
        - Handles division by zero cases by returning 0
        - Uses standard binary classification metrics formulas:
            - Precision = TP / (TP + FP)
            - Recall = TP / (TP + FN)
            - F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    # ss: (...); labels: (...)
    ss, labels = ss.float(), labels.float()  # each (...)
    tp = torch.sum((ss >= cutoff) & (labels == 1.0))  # ()
    fp = torch.sum((ss >= cutoff) & (labels == 0.0))  # ()
    fn = torch.sum((ss < cutoff) & (labels == 1.0))  # ()
    precision_denominator = tp + fp  # ()
    precision = torch.where(  # ()
        precision_denominator != 0,
        tp / precision_denominator,
        torch.tensor(0.0),
    )
    recall_denominator = tp + fn  # ()
    recall = torch.where(  # ()
        recall_denominator != 0,
        tp / recall_denominator,
        torch.tensor(0.0),
    )
    f1 = torch.where(  # ()
        (precision + recall) != 0,
        (2 * precision * recall) / (precision + recall),
        torch.tensor(0.0),
    )
    return f1, precision, recall  # each ()


def max_metrics(
    ss: torch.Tensor,
    labels: torch.Tensor,
    increment: float = 0.01,
) -> tuple[float, float, float, float]:
    """
    Find optimal classification metrics by scanning different cutoff thresholds.
    Optimized version that vectorizes calculations across all cutoffs.

    Args:
        ss: Prediction scores tensor, typically between -1 and 1
        labels: Ground truth binary labels tensor (0 or 1)
        increment: Step size for scanning cutoff values, defaults to 0.01

    Returns:
        Tuple containing:
            - Maximum F1 score (float)
            - Maximum precision score (float)
            - Maximum recall score (float) 
            - Optimal cutoff threshold (float)

    Note:
        - Input scores are clamped to [-1, 1] range
        - Handles edge case where all scores are >= 1
        - Scans cutoff values from min score to 1 in increments
        - Handles NaN F1 scores by replacing with -1 before finding max
        - Returns metrics at the threshold that maximizes F1 score
        - Optimized to compute metrics for all cutoffs in parallel using vectorization
    """
    # ss: (n,); labels: (n,)
    ss = torch.nan_to_num(ss, nan=0.0)  # (n,)
    ss = torch.clamp(ss, -1.0, 1.0)  # (n,)
    min_val = ss.min().item()
    max_val = 1
    if min_val >= max_val:
        min_val = 0
    
    ss = ss.float()  # (n,)
    labels = labels.float()  # (n,)

    cutoffs = torch.arange(  # (k,); k = number of candidate thresholds
        min_val,
        max_val,
        increment,
        device=ss.device,
        dtype=ss.dtype,
    )
    n_cutoffs = len(cutoffs)

    if n_cutoffs == 0:
        return 0.0, 0.0, 0.0, min_val

    ss_expanded = ss.unsqueeze(0)  # (1, n)
    cutoffs_expanded = cutoffs.unsqueeze(1)  # (k, 1)
    labels_expanded = labels.unsqueeze(0)  # (1, n)
    predictions = (ss_expanded >= cutoffs_expanded).float()  # (k, n)

    tp = torch.sum(predictions * labels_expanded, dim=1)  # (k,)
    fp = torch.sum(predictions * (1.0 - labels_expanded), dim=1)  # (k,)
    fn = torch.sum((1.0 - predictions) * labels_expanded, dim=1)  # (k,)

    precision_denominator = tp + fp  # (k,)
    precision = torch.where(  # (k,)
        precision_denominator != 0,
        tp / precision_denominator,
        torch.tensor(0.0, device=ss.device),
    )
    recall_denominator = tp + fn  # (k,)
    recall = torch.where(  # (k,)
        recall_denominator != 0,
        tp / recall_denominator,
        torch.tensor(0.0, device=ss.device),
    )
    f1_denominator = precision + recall  # (k,)
    f1s = torch.where(  # (k,)
        f1_denominator != 0,
        (2 * precision * recall) / f1_denominator,
        torch.tensor(0.0, device=ss.device),
    )
    valid_f1s = torch.where(  # (k,)
        torch.isnan(f1s),
        torch.tensor(-1.0, device=ss.device),
        f1s,
    )
    max_index = torch.argmax(valid_f1s)  # ()

    return f1s[max_index].item(), precision[max_index].item(), recall[max_index].item(), cutoffs[max_index].item()



def calculate_robust_roc_auc_multiclass(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Robust ROC AUC for multi-class (single-label) tasks.
    Handles missing classes in y_true by ignoring them in the weighted average.
    """
    # y_true: (n,); probs: (n, c)
    if np.isnan(probs).any():
        probs = np.nan_to_num(probs, nan=0.0)  # (n, c)

    n_classes = probs.shape[1]  # c
    try:
        if n_classes == 2:
            if len(np.unique(y_true)) == 2:
                # probs[:, 1]: (n,)
                return roc_auc_score(y_true, probs[:, 1])
            return -100.0

        y_true_onehot = np.eye(n_classes)[y_true]  # (n, c)
        scores: list[float] = []
        class_weights: list[float] = []
        for i in range(n_classes):
            if len(np.unique(y_true_onehot[:, i])) == 2:
                # y_true_onehot[:, i]: (n,); probs[:, i]: (n,)
                scores.append(roc_auc_score(y_true_onehot[:, i], probs[:, i]))
                class_weights.append(np.sum(y_true_onehot[:, i]))

        if not scores:
            return -100.0

        return float(np.average(scores, weights=class_weights))
    except Exception:  # noqa: broad-except -100.0 marks a metric that cannot be computed; sklearn raises several types for it
        return -100.0


def calculate_robust_pr_auc_multiclass(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Robust PR AUC for multi-class (single-label) tasks.
    """
    # y_true: (n,); probs: (n, c)
    if np.isnan(probs).any():
        probs = np.nan_to_num(probs, nan=0.0)  # (n, c)

    n_classes = probs.shape[1]  # c
    try:
        if n_classes == 2:
            if len(np.unique(y_true)) == 2:
                # probs[:, 1]: (n,)
                precision, recall, _ = precision_recall_curve(y_true, probs[:, 1])
                # precision: (q,); recall: (q,); _: (q - 1,); q = curve points
                return auc(recall, precision)
            return -100.0

        y_true_onehot = np.eye(n_classes)[y_true]  # (n, c)
        scores: list[float] = []
        class_weights: list[float] = []
        for i in range(n_classes):
            if len(np.unique(y_true_onehot[:, i])) == 2:
                # y_true_onehot[:, i]: (n,); probs[:, i]: (n,)
                precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], probs[:, i])
                # precision: (q,); recall: (q,); _: (q - 1,); q = curve points
                scores.append(auc(recall, precision))
                class_weights.append(np.sum(y_true_onehot[:, i]))

        if not scores:
            return -100.0

        return float(np.average(scores, weights=class_weights))
    except Exception:  # noqa: broad-except -100.0 marks a metric that cannot be computed; sklearn raises several types for it
        return -100.0


def calculate_robust_roc_auc_multilabel(y_true: np.ndarray, probs: np.ndarray, label_mask: np.ndarray | None = None) -> float:
    """
    Robust ROC AUC for multi-label tasks (macro average).
    """
    # y_true, probs, label_mask: (n, c)
    if label_mask is not None:
        validate_observed_labels(y_true, label_mask)
        if not np.isfinite(probs[label_mask]).all():
            raise ValueError("observed predictions must be finite")
    if np.isnan(probs).any():
        probs = np.nan_to_num(probs, nan=0.0)  # (n, c)

    scores: list[float] = []
    try:
        for i in range(y_true.shape[1]):
            observed = slice(None) if label_mask is None else label_mask[:, i]
            if len(np.unique(y_true[observed, i])) == 2:
                # y_true[:, i]: (n,); probs[:, i]: (n,)
                scores.append(roc_auc_score(y_true[observed, i], probs[observed, i]))

        if not scores:
            return -100.0
        return float(np.mean(scores))
    except Exception:  # noqa: broad-except -100.0 marks a metric that cannot be computed; sklearn raises several types for it
        return -100.0


def calculate_robust_pr_auc_multilabel(y_true: np.ndarray, probs: np.ndarray, label_mask: np.ndarray | None = None) -> float:
    """
    Robust PR AUC for multi-label tasks (macro average).
    """
    # y_true, probs, label_mask: (n, c)
    if label_mask is not None:
        validate_observed_labels(y_true, label_mask)
        if not np.isfinite(probs[label_mask]).all():
            raise ValueError("observed predictions must be finite")
    if np.isnan(probs).any():
        probs = np.nan_to_num(probs, nan=0.0)  # (n, c)

    scores: list[float] = []
    try:
        for i in range(y_true.shape[1]):
            observed = slice(None) if label_mask is None else label_mask[:, i]
            if len(np.unique(y_true[observed, i])) == 2:
                # y_true[:, i]: (n,); probs[:, i]: (n,)
                precision, recall, _ = precision_recall_curve(y_true[observed, i], probs[observed, i])
                # precision: (q,); recall: (q,); _: (q - 1,); q = curve points
                scores.append(auc(recall, precision))

        if not scores:
            return -100.0
        return float(np.mean(scores))
    except Exception:  # noqa: broad-except -100.0 marks a metric that cannot be computed; sklearn raises several types for it
        return -100.0


def compute_single_label_classification_metrics(p: EvalPrediction) -> dict[str, float]:
    """
    Compute comprehensive metrics for single-label classification tasks.

    Args:
        p: EvalPrediction object containing model predictions and ground truth labels

    Returns:
        Dictionary with the following metrics (all rounded to 5 decimal places):
            - f1: F1 score (weighted average)
            - precision: Precision score (weighted average)
            - recall: Recall score (weighted average)
            - accuracy: Overall accuracy
            - mcc: Matthews Correlation Coefficient
            - roc_auc: Area Under ROC Curve (weighted average)
            - pr_auc: Area Under Precision-Recall Curve (weighted average)

    Note:
        - Handles both binary and multi-class cases
        - For binary case: uses 0.5 threshold on probabilities
        - For multi-class: uses argmax for class prediction
        - Prints confusion matrix for detailed error analysis
        - Uses weighted averaging for multi-class metrics
        - Handles AUC calculation for both binary and multi-class cases
    """
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids
    # logits: (n, c); labels: (n,)

    y_pred = logits.argmax(axis=-1).flatten()  # (n,)
    y_true = labels.flatten().astype(int)  # (n,)
    probs = softmax(logits)  # (n, c)

    roc_auc = calculate_robust_roc_auc_multiclass(y_true, probs)
    pr_auc = calculate_robust_pr_auc_multiclass(y_true, probs)

    cm = confusion_matrix(y_true, y_pred)  # (u, u); u = observed classes
    print("\nConfusion Matrix:")
    print(cm)

    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        'f1': round(f1, 5),
        'precision': round(precision, 5),
        'recall': round(recall, 5),
        'accuracy': round(accuracy, 5),
        'mcc': round(mcc, 5),
        'roc_auc': round(roc_auc, 5),
        'pr_auc': round(pr_auc, 5)
    }


def compute_tokenwise_classification_metrics(p: EvalPrediction) -> dict[str, float]:
    """
    Compute metrics for token-level classification tasks.

    Args:
        p: EvalPrediction object containing model predictions and ground truth labels

    Returns:
        Dictionary containing the following metrics (all rounded to 5 decimal places):
            - accuracy: Overall accuracy
            - f1: F1 score (macro average)
            - precision: Precision score (macro average)
            - recall: Recall score (macro average)
            - mcc: Matthews Correlation Coefficient
            - roc_auc: Area Under ROC Curve (weighted average)
            - pr_auc: Area Under Precision-Recall Curve (weighted average)

    Note:
        - Handles special token padding (-100) by filtering before metric calculation
        - Uses macro averaging for multi-class metrics
        - Converts predictions to class labels using argmax
        - Handles AUC calculation for both binary and multi-class cases
    """
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    # logits: (b, l, c); labels: (b, l)

    y_pred = logits.argmax(axis=-1).flatten()  # (b * l,)
    y_true = labels.flatten()  # (b * l,)
    valid_indices = y_true != -100  # (b * l,)
    y_pred = y_pred[valid_indices]  # (n,); n = valid tokens
    y_true = y_true[valid_indices]  # (n,)

    cm = confusion_matrix(y_true, y_pred)  # (u, u); u = observed classes
    print("\nConfusion Matrix:")
    print(cm)

    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    probs = softmax(logits)  # (b, l, c)
    probs = probs.reshape(-1, probs.shape[-1])  # (b * l, c)
    probs = probs[valid_indices]  # (n, c)

    roc_auc = calculate_robust_roc_auc_multiclass(y_true, probs)
    pr_auc = calculate_robust_pr_auc_multiclass(y_true, probs)

    return {
        'accuracy': round(accuracy, 5),
        'f1': round(f1, 5),
        'precision': round(precision, 5),
        'recall': round(recall, 5),
        'mcc': round(mcc, 5),
        'roc_auc': round(roc_auc, 5),
        'pr_auc': round(pr_auc, 5)
    }


def compute_multi_label_classification_metrics(p: EvalPrediction, label_mask: np.ndarray | None = None) -> dict[str, float]:
    """
    Compute comprehensive metrics for multi-label classification tasks.

    Args:
        p: EvalPrediction object containing model predictions and ground truth labels

    Returns:
        Dictionary containing the following metrics (all rounded to 5 decimal places):
            - accuracy: Overall accuracy
            - f1: F1 score (optimized across thresholds)
            - precision: Precision score (at optimal threshold)
            - recall: Recall score (at optimal threshold)
            - hamming_loss: Proportion of wrong labels
            - threshold: Optimal classification threshold
            - mcc: Matthews Correlation Coefficient
            - roc_auc: Area Under ROC Curve (macro average)
            - pr_auc: Area Under Precision-Recall Curve (macro average)

    Note:
        - Converts inputs to PyTorch tensors
        - Applies softmax to raw predictions
        - Uses threshold optimization for best F1 score
        - Handles multi-class ROC AUC using one-vs-rest
        - All metrics are computed on flattened predictions
    """
    # label_mask: (n, c), true where the label is observed
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids
    # preds: (n, c); labels: (n, c)
    if label_mask is not None:
        validate_observed_labels(labels, label_mask)
        if not label_mask.any():
            raise ValueError("evaluating a masked split requires at least one observed label")
        if not np.isfinite(preds[label_mask]).all():
            raise ValueError("observed predictions must be finite")
        if not np.isin(labels[label_mask], [0, 1]).all():
            raise ValueError("observed multilabel targets must be zero or one")

    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)  # (n, c)
    if not isinstance(labels, torch.Tensor):
        y_true = torch.tensor(labels, dtype=torch.int)  # (n, c)
    else:
        y_true = labels.int()  # (n, c)

    probs = preds.sigmoid()  # (n, c)
    y_pred = (probs > 0.5).int()  # (n, c)

    observed = None if label_mask is None else torch.from_numpy(label_mask)  # (n, c) or None
    probs_flat = probs.reshape(-1) if observed is None else probs[observed]  # (m,), m = observed entries
    y_true_flat = y_true.reshape(-1) if observed is None else y_true[observed]  # (m,)
    f1, prec, recall, thres = max_metrics(probs_flat, y_true_flat)

    y_pred_flat = (y_pred.reshape(-1) if observed is None else y_pred[observed]).numpy()  # (m,)
    y_true_flat = y_true_flat.numpy()  # (m,)

    accuracy = accuracy_score(y_pred_flat, y_true_flat)
    hamming = hamming_loss(y_pred_flat, y_true_flat)
    mcc = matthews_corrcoef(y_true_flat, y_pred_flat)

    y_true_array = y_true.numpy()  # (n, c)
    probabilities_array = probs.numpy()  # (n, c)
    roc_auc = calculate_robust_roc_auc_multilabel(y_true_array, probabilities_array, label_mask)
    pr_auc = calculate_robust_pr_auc_multilabel(y_true_array, probabilities_array, label_mask)

    return {
        'accuracy': round(accuracy, 5),
        'f1': round(f1, 5),
        'precision': round(prec, 5),
        'recall': round(recall, 5),
        'hamming_loss': round(hamming, 5),
        'threshold': round(thres, 5),
        'mcc': round(mcc, 5),
        'roc_auc': round(roc_auc, 5),
        'pr_auc': round(pr_auc, 5)
    }


def compute_regression_metrics(p: EvalPrediction) -> dict[str, float]:
    """
    Compute comprehensive metrics for regression tasks.

    Args:
        p: EvalPrediction object containing model predictions and ground truth values

    Returns:
        Dictionary containing the following metrics (all rounded to 5 decimal places):
            - r_squared: Coefficient of determination (R²)
            - spearman_rho: Spearman rank correlation coefficient
            - spear_pval: P-value for Spearman correlation
            - pearson_rho: Pearson correlation coefficient
            - pear_pval: P-value for Pearson correlation
            - mse: Mean Squared Error
            - mae: Mean Absolute Error
            - rmse: Root Mean Squared Error

    Note:
        - Handles both raw predictions and tuple predictions
        - Flattens inputs to 1D arrays
        - Includes both correlation and error metrics
        - P-values indicate statistical significance of correlations
        - RMSE is calculated as square root of MSE
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids
    # preds: (...); labels: (...)

    y_pred = np.array(preds).flatten()  # (n,)
    y_true = np.array(labels).flatten()  # (n,)

    if np.isnan(y_true).any():
        print("y_true Nans were cast to 0")
        y_true = np.where(np.isnan(y_true), 0, y_true)  # (n,)
    if np.isnan(y_pred).any():
        print("y_pred Nans were cast to 0")
        y_pred = np.where(np.isnan(y_pred), 0, y_pred)  # (n,)

    try:
        spearman_rho, spear_pval = spearmanr(y_pred, y_true)
        pearson_rho, pear_pval = pearsonr(y_pred, y_true)
    except Exception:  # noqa: broad-except -100.0 marks a correlation that cannot be computed; scipy raises several types for it
        spearman_rho = -100.0
        spear_pval = -100.0
        pearson_rho = -100.0
        pear_pval = -100.0

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # ()

    return {
        'r_squared': round(r2, 5),
        'spearman_rho': round(spearman_rho, 5),
        'spear_pval': round(spear_pval, 5),
        'pearson_rho': round(pearson_rho, 5),
        'pear_pval': round(pear_pval, 5),
        'mse': round(mse, 5),
        'mae': round(mae, 5),
        'rmse': round(rmse, 5),
    }


def compute_tokenwise_regression_metrics(p: EvalPrediction) -> dict[str, float]:
    """
    Compute regression metrics tokenwise, ignoring label positions equal to -100.

    Compatible with HF Trainer `compute_metrics` API.
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids
    # preds: (...) or (..., 1); labels: (...)

    y_pred = np.array(preds)  # (...) or (..., 1)
    y_true = np.array(labels)  # (...)

    if y_pred.ndim == y_true.ndim + 1 and y_pred.shape[-1] == 1:
        y_pred = np.squeeze(y_pred, axis=-1)  # (...)

    valid_mask = y_true != -100  # (...)
    y_true = y_true[valid_mask].astype(float)  # (n,); n = valid positions
    y_pred = y_pred[valid_mask].astype(float)  # (n,)

    if y_true.size == 0:
        return {
            'r_squared': -100.0,
            'spearman_rho': -100.0,
            'spear_pval': -100.0,
            'pearson_rho': -100.0,
            'pear_pval': -100.0,
            'mse': -100.0,
            'mae': -100.0,
            'rmse': -100.0,
        }

    if np.isnan(y_true).any():
        print("y_true Nans were cast to 0")
        y_true = np.where(np.isnan(y_true), 0, y_true)  # (n,)
    if np.isnan(y_pred).any():
        print("y_pred Nans were cast to 0")
        y_pred = np.where(np.isnan(y_pred), 0, y_pred)  # (n,)

    try:
        spearman_rho, spear_pval = spearmanr(y_pred, y_true)
        pearson_rho, pear_pval = pearsonr(y_pred, y_true)
    except Exception:  # noqa: broad-except -100.0 marks a correlation that cannot be computed; scipy raises several types for it
        spearman_rho = -100.0
        spear_pval = -100.0
        pearson_rho = -100.0
        pear_pval = -100.0

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # ()

    return {
        'r_squared': round(float(r2), 5),
        'spearman_rho': round(float(spearman_rho), 5),
        'spear_pval': round(float(spear_pval), 5),
        'pearson_rho': round(float(pearson_rho), 5),
        'pear_pval': round(float(pear_pval), 5),
        'mse': round(float(mse), 5),
        'mae': round(float(mae), 5),
        'rmse': round(float(rmse), 5),
    }


# Entries of a multilabel split read at once while selecting a checkpoint: blocks of whole rows, so a wide vocabulary
# never needs more than a few hundred megabytes of float64 beside the split's own predictions.
SELECTION_BLOCK_ENTRIES = 2**24


def masked_selection_metrics(prediction: EvalPrediction) -> dict[str, float]:
    """The selection criterion of a masked multilabel split, and its observation counts, without the metric suite.

    `observed_loss` is the mean binary cross entropy over every observed entry of the split, exactly what
    `get_compute_metrics(..., observed_labels=True)` selects on, summed over blocks of whole rows so the working
    set stays independent of the vocabulary. The complete metrics (thresholded scores, ROC and PR areas) cost
    memory and time in proportion to every observed entry, which a vocabulary of thousands over tens of
    thousands of rows makes infeasible inside a training loop, so a caller asks for them from saved predictions.
    """
    if not isinstance(prediction.label_ids, tuple) or len(prediction.label_ids) != 2:
        raise ValueError("masked evaluation requires Trainer label_names=['labels', 'label_mask']")
    labels, mask = prediction.label_ids  # each (n, c)
    logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions  # (n, c)
    if logits.shape != labels.shape:
        raise ValueError("multilabel predictions and labels must have the same shape")
    block_rows = max(1, SELECTION_BLOCK_ENTRIES // max(int(np.prod(labels.shape[1:])), 1))
    total, observed = 0.0, 0
    for start in range(0, labels.shape[0], block_rows):
        block_labels, block_mask = labels[start:start + block_rows], mask[start:start + block_rows]  # each (b, c)
        validate_observed_labels(block_labels, block_mask)
        kept_logits, kept_labels = logits[start:start + block_rows][block_mask], block_labels[block_mask]  # each (m_b,)
        if not np.isfinite(kept_logits).all():
            raise ValueError("observed predictions must be finite")
        if not np.isin(kept_labels, [0, 1]).all():
            raise ValueError("observed multilabel targets must be zero or one")
        total += float(torch.nn.functional.binary_cross_entropy_with_logits(
            torch.as_tensor(kept_logits, dtype=torch.float64), torch.as_tensor(kept_labels, dtype=torch.float64), reduction="sum",
        ))
        observed += kept_logits.size
    if not observed:
        raise ValueError("evaluating a masked split requires at least one observed label")
    return {"observed_loss": total / observed, "observed_labels": observed, "unobserved_labels": int(mask.size - observed)}


def get_compute_metrics(
    task_type: str,
    tokenwise: bool = False,
    observed_labels: bool = False,
    selection_only: bool = False,
) -> Callable[[EvalPrediction], dict[str, float]]:
    if selection_only and not (observed_labels and task_type == 'multilabel' and not tokenwise):
        raise ValueError("selection-only evaluation is defined for pooled multilabel tasks with explicit observation masks")
    if task_type == 'singlelabel':
        compute_metrics = compute_single_label_classification_metrics
    elif task_type == 'multilabel':
        compute_metrics = compute_multi_label_classification_metrics
    elif task_type == 'sigmoid_regression':
        # Treat sigmoid_regression like regression for metrics
        compute_metrics = compute_tokenwise_regression_metrics if tokenwise else compute_regression_metrics
    elif task_type != 'regression' and tokenwise:
        compute_metrics = compute_tokenwise_classification_metrics
    elif task_type == 'regression' and not tokenwise:
        compute_metrics = compute_regression_metrics
    elif task_type == 'regression' and tokenwise:
        compute_metrics = compute_tokenwise_regression_metrics
    else:
        raise ValueError(f'Task type {task_type} not supported')
    if not observed_labels:
        return compute_metrics
    if tokenwise:
        raise ValueError("explicit observation masks currently support pooled tasks only")
    if selection_only:
        return masked_selection_metrics

    def masked_metrics(prediction: EvalPrediction) -> dict[str, float]:
        if not isinstance(prediction.label_ids, tuple) or len(prediction.label_ids) != 2:
            raise ValueError("masked evaluation requires Trainer label_names=['labels', 'label_mask']")
        labels, mask = prediction.label_ids  # each (n,), (n, 1), or (n, c)
        validate_observed_labels(labels, mask)
        if not mask.any():
            raise ValueError("evaluating a masked split requires at least one observed label")
        logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        if task_type == "multilabel":
            kept_logits, kept_labels = logits[mask], labels[mask]  # (m,), observed targets
            scores = compute_multi_label_classification_metrics(EvalPrediction(predictions=logits, label_ids=labels), mask)
        else:
            observed = mask.reshape(-1)  # (n,)
            kept_logits = logits.reshape(-1, logits.shape[-1])[observed] if task_type == "singlelabel" else logits.reshape(-1)[observed]  # (m, c) or (m,)
            kept_labels = labels.reshape(-1)[observed]  # (m,)
            if not np.isfinite(kept_logits).all():
                raise ValueError("observed predictions must be finite")
            if task_type == "singlelabel" and (
                (kept_labels != np.floor(kept_labels)) | (kept_labels < 0) | (kept_labels >= logits.shape[-1])
            ).any():
                raise ValueError("observed class labels must be integer indices within the output vocabulary")
            scores = compute_metrics(EvalPrediction(predictions=kept_logits, label_ids=kept_labels))
        # Selection uses the standard task criterion over the complete observed split,
        # independent of batch density. Variance heads use their mean's MSE here.
        target_dtype = torch.long if task_type == "singlelabel" else torch.float64
        observed_loss = get_loss_fct(task_type)(torch.as_tensor(kept_logits, dtype=torch.float64), torch.as_tensor(kept_labels, dtype=target_dtype))
        return {**scores, "observed_loss": float(observed_loss), "observed_labels": int(mask.sum()), "unobserved_labels": int(mask.size - mask.sum())}

    return masked_metrics


def validate_observed_labels(labels: np.ndarray, label_mask: np.ndarray) -> None:
    """Keep observation status explicit rather than infer it from a numeric filler."""
    # labels, label_mask: (...), identical shapes
    if label_mask.dtype != np.bool_ or label_mask.shape != labels.shape:
        raise ValueError("label_mask must be boolean with exactly the labels shape")
    if not np.isfinite(labels[label_mask]).all():
        raise ValueError("observed labels must be finite")


def get_compute_metrics_with_balanced(
    base_compute: Callable[[EvalPrediction], dict[str, float]],
    weights: np.ndarray,
    bin_borders: list[float],
    n_resamples: int = 100,
    seed: int = 42,
) -> Callable[[EvalPrediction], dict[str, Any]]:
    """
    Wrap a base compute_metrics callable to also emit balanced regression metrics
    (EpHod-style). Appends `balanced_*` keys. Assumes flattened predictions (after
    dropping -100 positions) align in length with `weights`.
    """
    try:
        from metrics_balanced import compute_balanced_regression_metrics
    except ImportError:
        from .metrics_balanced import compute_balanced_regression_metrics

    # weights: (...)
    weights_arr = np.asarray(weights, dtype=np.float64).flatten()  # (n,)

    def wrapper(p: EvalPrediction) -> dict[str, Any]:
        base_metrics = base_compute(p)
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids
        # preds: (...) or (..., 1); labels: (...)
        y_pred = np.asarray(preds, dtype=np.float64)  # (...) or (..., 1)
        y_true = np.asarray(labels, dtype=np.float64)  # (...)

        if y_pred.ndim == y_true.ndim + 1 and y_pred.shape[-1] == 1:
            y_pred = np.squeeze(y_pred, axis=-1)  # (...)

        y_pred = y_pred.flatten()  # (n_all,)
        y_true = y_true.flatten()  # (n_all,)

        valid_mask = y_true != -100.0  # (n_all,)
        if valid_mask.sum() != y_true.size:
            y_true = y_true[valid_mask]  # (n,); n = valid positions
            y_pred = y_pred[valid_mask]  # (n,)
        # y_true: (n,); y_pred: (n,)

        if np.isnan(y_true).any():
            y_true = np.where(np.isnan(y_true), 0.0, y_true)  # (n,)
        if np.isnan(y_pred).any():
            y_pred = np.where(np.isnan(y_pred), 0.0, y_pred)  # (n,)

        assert y_true.shape == weights_arr.shape, (
            f'balanced metrics shape mismatch: preds={y_true.shape}, weights={weights_arr.shape}'
        )
        bal = compute_balanced_regression_metrics(
            y_true, y_pred, weights_arr,
            bin_borders=bin_borders,
            n_resamples=n_resamples,
            seed=seed,
        )
        for k, v in bal.items():
            base_metrics[f'balanced_{k}'] = v
        return base_metrics

    return wrapper


if __name__ == "__main__":
    # py -m metrics

    print("Running tests for metrics functions...")
    
    # Test compute_single_label_classification_metrics
    print("\n--- compute_single_label_classification_metrics (Binary) ---")
    # 2 samples, 2 classes.
    # Logits: Sample 0 -> class 0 (high, low), Sample 1 -> class 1 (low, high)
    predictions = np.array([[2.0, -1.0], [-1.0, 2.0]])  # (2, 2)
    label_ids = np.array([0, 1])  # (2,)
    p = EvalPrediction(predictions=predictions, label_ids=label_ids)
    metrics = compute_single_label_classification_metrics(p)
    print(metrics)

    print("\n--- compute_single_label_classification_metrics (Multi-class) ---")
    # 3 samples, 3 classes.
    predictions = np.array(  # (3, 3)
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    )
    label_ids = np.array([0, 1, 2])  # (3,)
    p = EvalPrediction(predictions=predictions, label_ids=label_ids)
    metrics = compute_single_label_classification_metrics(p)
    print(metrics)

    # Test compute_tokenwise_classification_metrics
    print("\n--- compute_tokenwise_classification_metrics ---")
    # 1 sample, 3 tokens, 2 classes.
    # Token 0: pred 0, label 0
    # Token 1: pred 1, label 1
    # Token 2: pred 0, label -100 (ignored)
    predictions = np.array(  # (1, 3, 2)
        [[[2.0, -1.0], [-1.0, 2.0], [2.0, -1.0]]]
    )
    label_ids = np.array([[0, 1, -100]])  # (1, 3)
    p = EvalPrediction(predictions=predictions, label_ids=label_ids)
    metrics = compute_tokenwise_classification_metrics(p)
    print(metrics)

    # Test compute_multi_label_classification_metrics
    print("\n--- compute_multi_label_classification_metrics ---")
    # 2 samples, 3 classes
    # Sample 0: pred [1, 0, 1], label [1, 0, 1]
    # Sample 1: pred [0, 1, 0], label [0, 1, 0]
    # Logits need to be high for 1, low for 0.
    predictions = np.array([[5.0, -5.0, 5.0], [-5.0, 5.0, -5.0]])  # (2, 3)
    label_ids = np.array([[1, 0, 1], [0, 1, 0]])  # (2, 3)
    p = EvalPrediction(predictions=predictions, label_ids=label_ids)
    metrics = compute_multi_label_classification_metrics(p)
    print(metrics)

    # Test compute_regression_metrics
    print("\n--- compute_regression_metrics ---")
    predictions = np.array([1.0, 2.0, 3.0])  # (3,)
    label_ids = np.array([1.1, 1.9, 3.2])  # (3,)
    p = EvalPrediction(predictions=predictions, label_ids=label_ids)
    metrics = compute_regression_metrics(p)
    print(metrics)

    # Test compute_tokenwise_regression_metrics
    print("\n--- compute_tokenwise_regression_metrics ---")
    # 1 sample, 3 tokens
    # Token 2 is ignored (-100)
    predictions = np.array([[1.0, 2.0, 5.0]])  # (1, 3)
    label_ids = np.array([[1.1, 1.9, -100.0]])  # (1, 3)
    p = EvalPrediction(predictions=predictions, label_ids=label_ids)
    metrics = compute_tokenwise_regression_metrics(p)
    print(metrics)
