"""The regression metrics DSM reports, as a Hugging Face `compute_metrics`, with two helpers.

DSM's `metrics.regression` held them, and egfr_binding_proxies copied that module whole.
"""

from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


if TYPE_CHECKING:
    from transformers import EvalPrediction


__all__ = ["compute_metrics_regression", "get_residuals", "spearman"]


def spearman(probs: np.ndarray, correct_preds: np.ndarray) -> None:
    """Print Spearman's rho between `probs` and `correct_preds`, each flattened, and its p-value."""
    # probs, correct_preds: (...), n elements each
    correlation, pval = spearmanr(probs.flatten(), correct_preds.flatten())
    print(f"Spearman: {correlation:.3f}, {pval:.3E}")


def get_residuals(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The absolute difference between each label and its prediction, both flattened."""
    # preds: (...); labels: (...), each of n elements
    return np.abs(labels.flatten() - preds.flatten())  # (n,)


def compute_metrics_regression(p: EvalPrediction) -> dict[str, float]:
    """R^2, Spearman's and Pearson's rho with their p-values, MSE, MAE, and RMSE, to 5 places.

    The predictions are `p.predictions`, or its first element when it is a tuple, and the labels
    are `p.label_ids`, or its second element. Both are flattened, and R^2, MSE, and MAE take the
    labels as the truth.
    """
    predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids

    predictions = np.array(predictions).flatten()  # (n,)
    labels = np.array(labels).flatten()  # (n,)

    r2 = r2_score(labels, predictions)
    spearman_rho, spear_pval = spearmanr(predictions, labels)
    pearson_rho, pear_pval = pearsonr(predictions, labels)
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mse)

    return {
        "r_squared": round(r2, 5),
        "spearman_rho": round(spearman_rho, 5),
        "spear_pval": round(spear_pval, 5),
        "pearson_rho": round(pearson_rho, 5),
        "pear_pval": round(pear_pval, 5),
        "mse": round(mse, 5),
        "mae": round(mae, 5),
        "rmse": round(rmse, 5),
    }
