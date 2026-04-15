"""
Balanced regression metrics (EpHod-style).

Port of the evaluation suite from:
    Gado et al., 2025, "Machine learning prediction of enzyme optimum pH".
    https://github.com/jafetgado/EpHod  (ephod/training/trainutils.py)

Exposes:
    - label_distribution_smoothing
    - compute_sample_weights
    - apply_weights_from_reference
    - compute_balanced_regression_metrics
"""

import warnings
import numpy as np
from typing import Dict, List, Optional
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import convolve1d, gaussian_filter1d
from sklearn import metrics


SUPPORTED_METHODS = (
    'none',
    'bin_inv',
    'bin_inv_sqrt',
    'LDS_inv',
    'LDS_inv_sqrt',
    'LDS_extreme',
)


def _default_bin_borders(y: np.ndarray) -> List[float]:
    """Data-driven bin borders at 1/3 and 2/3 quantiles of the training labels."""
    lo = float(np.quantile(y, 1.0 / 3.0))
    hi = float(np.quantile(y, 2.0 / 3.0))
    assert lo < hi, f'Degenerate bin borders: {lo} >= {hi} (label distribution has <3 distinct values)'
    return [lo, hi]


def label_distribution_smoothing(
    y: np.ndarray,
    bins: Optional[int] = None,
    ks: int = 5,
    sigma: float = 2.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Gaussian-kernel smoothed empirical label density (per-sample).

    Faithful port of EpHod's label_distribution_smoothing, which itself follows
    Yang et al. 2021 ("Delving into deep imbalanced regression").

    If `bins` is None, use one bin per unit of y (np.ceil(max - min)).
    """
    y = np.asarray(y, dtype=np.float64).flatten()
    y_min, y_max = float(np.min(y)), float(np.max(y))
    assert y_max > y_min, 'Cannot smooth a constant-label distribution'

    if bins is None:
        bins = int(np.ceil(y_max - y_min))
    assert bins >= 1, f'bins must be >= 1, got {bins}'

    bin_freqs, bin_borders = np.histogram(y, range=(y_min, y_max), bins=bins)

    y_binned = np.zeros(len(y), dtype=np.int64)
    for i in range(bins):
        low, high = bin_borders[i], bin_borders[i + 1]
        locs = np.logical_and(y >= low, y <= high)
        y_binned[locs] = i

    half_ks = (ks - 1) // 2
    base_kernel = [0.0] * half_ks + [1.0] + [0.0] * half_ks
    kernel_window = gaussian_filter1d(base_kernel, sigma=sigma)
    kernel_window = kernel_window / np.max(gaussian_filter1d(base_kernel, sigma=sigma))

    bin_kde = convolve1d(np.array(bin_freqs, dtype=np.float64), weights=kernel_window, mode='constant')
    y_kde = np.array([bin_kde[int(b)] for b in y_binned])

    if normalize:
        min_kde = float(np.min(y_kde))
        assert min_kde > 0.0, 'Smoothed density has zero entries; increase kernel size/sigma or reduce bin count'
        y_kde = y_kde / min_kde

    return y_kde


def compute_sample_weights(
    y: np.ndarray,
    method: str = 'bin_inv',
    bin_borders: Optional[List[float]] = None,
    lds_bins: int = 100,
    lds_ks: int = 5,
    lds_sigma: float = 2.0,
) -> np.ndarray:
    """
    Return per-sample weights for a regression label array.

    Methods (all normalized to mean == 1):
        'none'          ones
        'bin_inv'       digitize by bin_borders, weight = 1 / bin_count
        'bin_inv_sqrt'  sqrt(bin_inv)
        'LDS_inv'       1 / smoothed Gaussian KDE over `lds_bins` bins
        'LDS_inv_sqrt'  sqrt(LDS_inv)
        'LDS_extreme'   LDS_inv with rare values (y<=low or y>=high) doubled
    """
    assert method in SUPPORTED_METHODS, f'method {method!r} not in {SUPPORTED_METHODS}'
    y = np.asarray(y, dtype=np.float64).flatten()
    assert y.ndim == 1 and len(y) > 0

    if method == 'none':
        weights = np.ones(len(y), dtype=np.float64)
    elif method in ('bin_inv', 'bin_inv_sqrt'):
        if bin_borders is None:
            bin_borders = _default_bin_borders(y)
        y_binned = np.digitize(y, bin_borders)
        bin_class, bin_freqs = np.unique(y_binned, return_counts=True)
        inv_freq = dict(zip(bin_class.tolist(), (1.0 / bin_freqs).tolist()))
        weights = np.array([inv_freq[int(v)] for v in y_binned], dtype=np.float64)
    else:
        effdist = label_distribution_smoothing(y, bins=lds_bins, ks=lds_ks, sigma=lds_sigma)
        weights = 1.0 / effdist
        if method == 'LDS_extreme':
            if bin_borders is None:
                bin_borders = _default_bin_borders(y)
            low, high = bin_borders[0], bin_borders[-1]
            relevance = np.logical_or(y <= low, y >= high).astype(np.float64)
            relevance = relevance * (1.0 - 0.5) + 0.5
            weights = weights * relevance

    if method in ('bin_inv_sqrt', 'LDS_inv_sqrt'):
        weights = np.sqrt(weights)

    weights = weights / np.mean(weights)
    return weights


def apply_weights_from_reference(
    y_new: np.ndarray,
    y_ref: np.ndarray,
    method: str,
    bin_borders: Optional[List[float]] = None,
    lds_bins: int = 100,
    lds_ks: int = 5,
    lds_sigma: float = 2.0,
) -> np.ndarray:
    """
    Weight `y_new` using the same scheme derived from `y_ref` (training labels).

    For bin_inv/bin_inv_sqrt: reuse reference bin borders and reference bin frequencies.
    For LDS_*: interpolate the reference KDE at y_new values via nearest bin assignment.
    For none: ones.

    Weights are renormalized to mean 1 on the new split (EpHod's per-split convention).
    """
    assert method in SUPPORTED_METHODS, f'method {method!r} not in {SUPPORTED_METHODS}'
    y_new = np.asarray(y_new, dtype=np.float64).flatten()
    y_ref = np.asarray(y_ref, dtype=np.float64).flatten()
    assert len(y_new) > 0 and len(y_ref) > 0

    if method == 'none':
        return np.ones(len(y_new), dtype=np.float64)

    if method in ('bin_inv', 'bin_inv_sqrt'):
        if bin_borders is None:
            bin_borders = _default_bin_borders(y_ref)
        ref_binned = np.digitize(y_ref, bin_borders)
        new_binned = np.digitize(y_new, bin_borders)
        bin_class, bin_freqs = np.unique(ref_binned, return_counts=True)
        inv_freq = dict(zip(bin_class.tolist(), (1.0 / bin_freqs).tolist()))
        # Fallback: bins unseen in ref get the rarest (largest) weight from ref
        fallback = float(np.max(list(inv_freq.values())))
        weights = np.array(
            [inv_freq[int(v)] if int(v) in inv_freq else fallback for v in new_binned],
            dtype=np.float64,
        )
        if method == 'bin_inv_sqrt':
            weights = np.sqrt(weights)
        weights = weights / np.mean(weights)
        return weights

    # LDS_inv / LDS_inv_sqrt / LDS_extreme: build KDE on ref, query at y_new
    y_ref_min, y_ref_max = float(np.min(y_ref)), float(np.max(y_ref))
    bins = lds_bins
    bin_freqs_ref, bin_edges = np.histogram(y_ref, range=(y_ref_min, y_ref_max), bins=bins)
    half_ks = (lds_ks - 1) // 2
    base_kernel = [0.0] * half_ks + [1.0] + [0.0] * half_ks
    kernel_window = gaussian_filter1d(base_kernel, sigma=lds_sigma)
    kernel_window = kernel_window / np.max(gaussian_filter1d(base_kernel, sigma=lds_sigma))
    bin_kde = convolve1d(np.array(bin_freqs_ref, dtype=np.float64), weights=kernel_window, mode='constant')
    min_kde = float(np.min(bin_kde[bin_kde > 0])) if np.any(bin_kde > 0) else 1.0
    bin_kde_norm = np.where(bin_kde > 0, bin_kde / min_kde, 1.0)

    y_new_clipped = np.clip(y_new, y_ref_min, y_ref_max)
    new_binned = np.clip(
        np.digitize(y_new_clipped, bin_edges[1:-1]),
        0,
        bins - 1,
    )
    effdist = bin_kde_norm[new_binned]
    weights = 1.0 / effdist
    if method == 'LDS_extreme':
        if bin_borders is None:
            bin_borders = _default_bin_borders(y_ref)
        low, high = bin_borders[0], bin_borders[-1]
        relevance = np.logical_or(y_new <= low, y_new >= high).astype(np.float64)
        relevance = relevance * (1.0 - 0.5) + 0.5
        weights = weights * relevance
    if method == 'LDS_inv_sqrt':
        weights = np.sqrt(weights)
    weights = weights / np.mean(weights)
    return weights


def compute_balanced_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    bin_borders: List[float],
    n_resamples: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Full EpHod balanced-metric suite.

    Returns resampled Pearson/Spearman, sample-weighted RMSE/R^2,
    and bin-classification MCC / per-bin F1 / per-bin ROC-AUC (one-vs-rest)
    with per-bin means.
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    weights = np.asarray(weights, dtype=np.float64).flatten()
    assert y_true.shape == y_pred.shape == weights.shape, (
        f'shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}, weights={weights.shape}'
    )
    assert len(bin_borders) >= 1

    rng = np.random.default_rng(seed)
    p = weights / float(np.sum(weights))
    n = len(y_true)

    rhos, rs = [], []
    for _ in range(n_resamples):
        locs = rng.choice(n, size=n, replace=True, p=p)
        yt, yp = y_true[locs], y_pred[locs]
        if np.std(yt) == 0.0 or np.std(yp) == 0.0:
            rhos.append(-100.0)
            rs.append(-100.0)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rhos.append(float(spearmanr(yt, yp)[0]))
            rs.append(float(pearsonr(yt, yp)[0]))

    rhos_arr = np.array(rhos)
    rs_arr = np.array(rs)

    rmse = float(metrics.mean_squared_error(y_true, y_pred, sample_weight=weights)) ** 0.5
    r2 = float(metrics.r2_score(y_true, y_pred, sample_weight=weights))

    y_true_b = np.digitize(y_true, bin_borders)
    y_pred_b = np.digitize(y_pred, bin_borders)

    try:
        mcc = float(metrics.matthews_corrcoef(y_true_b, y_pred_b, sample_weight=weights))
    except Exception:
        mcc = -100.0

    f1_per_bin, auc_per_bin = [], []
    for val in sorted(set(y_true_b.tolist())):
        yt_sel = (y_true_b == val).astype(int)
        yp_sel = (y_pred_b == val).astype(int)
        try:
            f1_per_bin.append(float(metrics.f1_score(yt_sel, yp_sel, sample_weight=weights)))
        except Exception:
            f1_per_bin.append(-100.0)
        if len(np.unique(yt_sel)) < 2:
            auc_per_bin.append(-100.0)
        else:
            try:
                auc_per_bin.append(float(metrics.roc_auc_score(yt_sel, yp_sel, sample_weight=weights)))
            except Exception:
                auc_per_bin.append(-100.0)

    valid_aucs = [a for a in auc_per_bin if a != -100.0]
    valid_f1s = [f for f in f1_per_bin if f != -100.0]

    return {
        'weighted_pearson_rho': round(float(np.mean(rs_arr)), 5),
        'weighted_pearson_rho_std': round(float(np.std(rs_arr)), 5),
        'weighted_spearman_rho': round(float(np.mean(rhos_arr)), 5),
        'weighted_spearman_rho_std': round(float(np.std(rhos_arr)), 5),
        'weighted_rmse': round(rmse, 5),
        'weighted_r_squared': round(r2, 5),
        'binned_mcc': round(mcc, 5),
        'binned_f1_per_bin': [round(f, 5) for f in f1_per_bin],
        'binned_f1_mean': round(float(np.mean(valid_f1s)), 5) if valid_f1s else -100.0,
        'binned_roc_auc_per_bin': [round(a, 5) for a in auc_per_bin],
        'binned_roc_auc_mean': round(float(np.mean(valid_aucs)), 5) if valid_aucs else -100.0,
        'bin_borders': list(bin_borders),
        'n_bins': len(bin_borders) + 1,
        'n_resamples': int(n_resamples),
    }
