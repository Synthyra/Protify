import numpy as np
import pytest

try:
    from src.protify.metrics_balanced import (
        compute_sample_weights,
        apply_weights_from_reference,
        compute_balanced_regression_metrics,
        label_distribution_smoothing,
        SUPPORTED_METHODS,
    )
except ImportError:
    try:
        from protify.metrics_balanced import (
            compute_sample_weights,
            apply_weights_from_reference,
            compute_balanced_regression_metrics,
            label_distribution_smoothing,
            SUPPORTED_METHODS,
        )
    except ImportError:
        from ..metrics_balanced import (
            compute_sample_weights,
            apply_weights_from_reference,
            compute_balanced_regression_metrics,
            label_distribution_smoothing,
            SUPPORTED_METHODS,
        )


def _skewed_labels(rng):
    """90% cluster at pH 7, 10% split between 3 and 11 (EpHod-like)."""
    return np.concatenate([
        rng.normal(7.0, 0.5, size=900),
        rng.uniform(3.0, 4.0, size=50),
        rng.uniform(10.0, 11.0, size=50),
    ])


def test_sample_weights_shape_and_mean_one():
    rng = np.random.default_rng(0)
    y = _skewed_labels(rng)
    for method in SUPPORTED_METHODS:
        w = compute_sample_weights(y, method=method, bin_borders=[5, 9])
        assert w.shape == y.shape
        assert abs(float(np.mean(w)) - 1.0) < 1e-9, f'{method}: mean={w.mean()}'
        assert np.all(w >= 0)


def test_sample_weights_rare_values_heavier():
    rng = np.random.default_rng(1)
    y = _skewed_labels(rng)
    w = compute_sample_weights(y, method='bin_inv', bin_borders=[5, 9])
    rare_mask = (y <= 5) | (y >= 9)
    assert float(np.mean(w[rare_mask])) > float(np.mean(w[~rare_mask]))


def test_sample_weights_none_is_ones():
    y = np.linspace(0, 10, 100)
    w = compute_sample_weights(y, method='none')
    assert np.allclose(w, 1.0)


def test_uniform_labels_bin_inv_near_unweighted():
    rng = np.random.default_rng(2)
    y = rng.uniform(0, 10, size=1000)
    y_pred = y + rng.normal(0, 0.1, size=1000)
    w = compute_sample_weights(y, method='bin_inv')
    bin_borders = [float(np.quantile(y, 1/3)), float(np.quantile(y, 2/3))]
    m = compute_balanced_regression_metrics(y, y_pred, w, bin_borders, n_resamples=50, seed=0)
    unweighted_rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    assert abs(m['weighted_rmse'] - unweighted_rmse) < 0.05


def test_perfect_predictions():
    rng = np.random.default_rng(3)
    y = _skewed_labels(rng)
    w = compute_sample_weights(y, method='bin_inv', bin_borders=[5, 9])
    m = compute_balanced_regression_metrics(y, y.copy(), w, [5, 9], n_resamples=20, seed=0)
    assert m['weighted_r_squared'] == pytest.approx(1.0, abs=1e-6)
    assert m['weighted_rmse'] == pytest.approx(0.0, abs=1e-6)
    assert m['weighted_pearson_rho'] == pytest.approx(1.0, abs=1e-6)
    assert m['weighted_spearman_rho'] == pytest.approx(1.0, abs=1e-6)
    assert m['binned_mcc'] == pytest.approx(1.0, abs=1e-6)


def test_skewed_weighted_rmse_larger_than_unweighted_when_tails_bad():
    rng = np.random.default_rng(4)
    y = _skewed_labels(rng)
    y_pred = y.copy()
    tails = (y <= 5) | (y >= 9)
    y_pred[tails] = 7.0  # model collapses tails to the center
    w = compute_sample_weights(y, method='bin_inv', bin_borders=[5, 9])
    m = compute_balanced_regression_metrics(y, y_pred, w, [5, 9], n_resamples=20, seed=0)
    unweighted_rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    assert m['weighted_rmse'] > unweighted_rmse


def test_resampling_reproducibility():
    rng = np.random.default_rng(5)
    y = _skewed_labels(rng)
    y_pred = y + rng.normal(0, 0.5, size=len(y))
    w = compute_sample_weights(y, method='bin_inv', bin_borders=[5, 9])
    m1 = compute_balanced_regression_metrics(y, y_pred, w, [5, 9], n_resamples=30, seed=123)
    m2 = compute_balanced_regression_metrics(y, y_pred, w, [5, 9], n_resamples=30, seed=123)
    assert m1['weighted_pearson_rho'] == m2['weighted_pearson_rho']
    assert m1['weighted_spearman_rho'] == m2['weighted_spearman_rho']


def test_apply_weights_from_reference_matches_on_self():
    rng = np.random.default_rng(6)
    y = _skewed_labels(rng)
    w_ref = compute_sample_weights(y, method='bin_inv', bin_borders=[5, 9])
    w_applied = apply_weights_from_reference(y, y, method='bin_inv', bin_borders=[5, 9])
    assert np.allclose(w_ref, w_applied, atol=1e-9)


def test_apply_weights_from_reference_none():
    y_new = np.array([1.0, 2.0, 3.0])
    y_ref = np.array([0.0, 1.0, 2.0, 3.0])
    w = apply_weights_from_reference(y_new, y_ref, method='none')
    assert np.allclose(w, 1.0)


def test_lds_weights_finite_and_positive():
    rng = np.random.default_rng(7)
    y = _skewed_labels(rng)
    for method in ('LDS_inv', 'LDS_inv_sqrt', 'LDS_extreme'):
        w = compute_sample_weights(y, method=method, bin_borders=[5, 9])
        assert np.all(np.isfinite(w))
        assert np.all(w > 0)


def test_label_distribution_smoothing_shape():
    rng = np.random.default_rng(8)
    y = _skewed_labels(rng)
    kde = label_distribution_smoothing(y, bins=50, ks=5, sigma=2.0)
    assert kde.shape == y.shape
    assert np.all(kde >= 1.0)  # normalized so min == 1


def test_degenerate_single_class_auc_is_minus_100():
    """All labels in one bin -> exactly one bin present -> all one-vs-rest AUCs are undefined."""
    y = np.full(100, 7.0) + np.random.default_rng(9).normal(0, 0.01, size=100)
    y_pred = y.copy()
    w = np.ones(100)
    m = compute_balanced_regression_metrics(y, y_pred, w, [5, 9], n_resamples=10, seed=0)
    assert len(m['binned_roc_auc_per_bin']) == 1
    assert m['binned_roc_auc_per_bin'][0] == -100.0
    assert m['binned_roc_auc_mean'] == -100.0


def test_shape_handling_2d_input_matches_flattened():
    rng = np.random.default_rng(10)
    y_flat = _skewed_labels(rng)
    y_2d = y_flat.reshape(-1, 1)
    noise = rng.normal(0, 0.3, size=len(y_flat))
    w = compute_sample_weights(y_flat, method='bin_inv', bin_borders=[5, 9])
    m_flat = compute_balanced_regression_metrics(y_flat, y_flat + noise, w, [5, 9], n_resamples=20, seed=7)
    m_2d = compute_balanced_regression_metrics(y_2d, (y_flat + noise).reshape(-1, 1), w, [5, 9], n_resamples=20, seed=7)
    assert m_flat['weighted_rmse'] == m_2d['weighted_rmse']
    assert m_flat['weighted_r_squared'] == m_2d['weighted_r_squared']
    assert m_flat['weighted_pearson_rho'] == m_2d['weighted_pearson_rho']


def test_bin_inv_sqrt_is_sqrt_of_bin_inv():
    rng = np.random.default_rng(11)
    y = _skewed_labels(rng)
    w1 = compute_sample_weights(y, method='bin_inv', bin_borders=[5, 9])
    w2 = compute_sample_weights(y, method='bin_inv_sqrt', bin_borders=[5, 9])
    # Both are normalized to mean 1 independently, so compare ratios within a bin
    # (monotonic-equivalent, but sqrt should compress the weight spread).
    spread1 = float(np.max(w1) / np.min(w1))
    spread2 = float(np.max(w2) / np.min(w2))
    assert spread2 < spread1
    assert spread2 == pytest.approx(np.sqrt(spread1), rel=1e-6)


def test_lds_rare_values_heavier_than_common():
    rng = np.random.default_rng(12)
    y = _skewed_labels(rng)
    for method in ('LDS_inv', 'LDS_extreme'):
        w = compute_sample_weights(y, method=method, bin_borders=[5, 9], lds_bins=50)
        rare = (y <= 5) | (y >= 9)
        assert float(np.mean(w[rare])) > float(np.mean(w[~rare])), method


def test_lds_extreme_doubles_rare_vs_lds_inv():
    rng = np.random.default_rng(13)
    y = _skewed_labels(rng)
    w_inv = compute_sample_weights(y, method='LDS_inv', bin_borders=[5, 9], lds_bins=50)
    w_ext = compute_sample_weights(y, method='LDS_extreme', bin_borders=[5, 9], lds_bins=50)
    rare = (y <= 5) | (y >= 9)
    ratio_inv = float(np.mean(w_inv[rare]) / np.mean(w_inv[~rare]))
    ratio_ext = float(np.mean(w_ext[rare]) / np.mean(w_ext[~rare]))
    assert ratio_ext > ratio_inv


def test_apply_weights_reference_different_distribution():
    """Valid set has a different composition than train; weights should reflect train frequencies."""
    rng = np.random.default_rng(14)
    y_train = _skewed_labels(rng)  # skewed toward 7
    y_valid = rng.uniform(3, 11, size=200)  # uniform across range
    w_valid = apply_weights_from_reference(y_valid, y_train, method='bin_inv', bin_borders=[5, 9])
    assert abs(float(np.mean(w_valid)) - 1.0) < 1e-9
    # Rare-in-training values (tails) should still get larger weights in valid.
    rare = (y_valid <= 5) | (y_valid >= 9)
    assert float(np.mean(w_valid[rare])) > float(np.mean(w_valid[~rare]))


def test_resampled_correlation_more_balanced_than_raw():
    """On highly skewed data, resampled Pearson should differ from raw Pearson — EpHod's key motivation."""
    rng = np.random.default_rng(15)
    y = _skewed_labels(rng)
    y_pred = y + rng.normal(0, 0.5, size=len(y))
    from scipy.stats import pearsonr
    raw_r = float(pearsonr(y, y_pred)[0])
    w = compute_sample_weights(y, method='bin_inv', bin_borders=[5, 9])
    m = compute_balanced_regression_metrics(y, y_pred, w, [5, 9], n_resamples=200, seed=0)
    # Resampled should be higher — balanced data has wider spread, boosting correlation
    assert m['weighted_pearson_rho'] > raw_r


def _ephod_reference_performance(ytrue, ypred, weights, bins, n_resamples, seed):
    """Direct port of EpHod's performance() from trainutils.py — ground truth for parity."""
    from scipy.stats import spearmanr, pearsonr
    from sklearn import metrics as skm
    ytrue = np.asarray(ytrue, dtype=np.float64)
    ypred = np.asarray(ypred, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    perf = {}
    rng = np.random.default_rng(seed)
    p = weights / np.sum(weights)
    rho, r = 0.0, 0.0
    for _ in range(n_resamples):
        locs = rng.choice(len(ytrue), size=len(weights), p=p, replace=True)
        rho += float(spearmanr(ytrue[locs], ypred[locs])[0])
        r += float(pearsonr(ytrue[locs], ypred[locs])[0])
    perf['rho'] = rho / n_resamples
    perf['r'] = r / n_resamples
    perf['rmse'] = float(skm.mean_squared_error(ytrue, ypred, sample_weight=weights)) ** 0.5
    perf['r2'] = float(skm.r2_score(ytrue, ypred, sample_weight=weights))
    ytrue_b = np.digitize(ytrue, bins)
    ypred_b = np.digitize(ypred, bins)
    perf['mcc'] = float(skm.matthews_corrcoef(ytrue_b, ypred_b, sample_weight=weights))
    f1, auc = [], []
    for val in sorted(set(ytrue_b.tolist())):
        yt = (ytrue_b == val).astype(int)
        yp = (ypred_b == val).astype(int)
        f1.append(float(skm.f1_score(yt, yp, sample_weight=weights)))
        if len(np.unique(yt)) >= 2:
            auc.append(float(skm.roc_auc_score(yt, yp, sample_weight=weights)))
        else:
            auc.append(-100.0)
    perf['f1_mean'] = float(np.mean(f1))
    perf['auc_mean'] = float(np.mean([a for a in auc if a != -100.0])) if any(a != -100.0 for a in auc) else -100.0
    return perf


def test_parity_with_ephod_reference():
    """Our output must match EpHod's reference implementation numerically."""
    rng = np.random.default_rng(42)
    y = _skewed_labels(rng)
    y_pred = y + rng.normal(0, 0.4, size=len(y))
    w = compute_sample_weights(y, method='bin_inv', bin_borders=[5, 9])

    # Use a fresh seed-matched draw
    ours = compute_balanced_regression_metrics(y, y_pred, w, [5, 9], n_resamples=100, seed=777)
    ref = _ephod_reference_performance(y, y_pred, w, [5, 9], n_resamples=100, seed=777)

    assert ours['weighted_rmse'] == pytest.approx(ref['rmse'], abs=1e-4)
    assert ours['weighted_r_squared'] == pytest.approx(ref['r2'], abs=1e-4)
    assert ours['binned_mcc'] == pytest.approx(ref['mcc'], abs=1e-4)
    assert ours['binned_f1_mean'] == pytest.approx(ref['f1_mean'], abs=1e-4)
    assert ours['binned_roc_auc_mean'] == pytest.approx(ref['auc_mean'], abs=1e-4)
    # Resampling uses same rng + same seed -> exact match within rounding
    assert ours['weighted_pearson_rho'] == pytest.approx(ref['r'], abs=1e-4)
    assert ours['weighted_spearman_rho'] == pytest.approx(ref['rho'], abs=1e-4)


def test_default_bin_borders_are_tertiles():
    from protify.metrics_balanced import _default_bin_borders  # type: ignore
    y = np.linspace(0, 9, 91)
    borders = _default_bin_borders(y)
    assert borders[0] == pytest.approx(float(np.quantile(y, 1 / 3)), abs=1e-9)
    assert borders[1] == pytest.approx(float(np.quantile(y, 2 / 3)), abs=1e-9)


def test_invalid_method_raises():
    y = np.linspace(0, 10, 100)
    with pytest.raises(AssertionError):
        compute_sample_weights(y, method='not_a_real_method')


def test_constant_labels_raise_in_lds():
    y = np.full(100, 5.0)
    with pytest.raises(AssertionError):
        label_distribution_smoothing(y)


# ---------------------------------------------------------------------------
# Explicit bin_borders override (user-facing CLI --balanced_bin_borders)
# ---------------------------------------------------------------------------

def test_explicit_bin_borders_override_tertile_default():
    """Passing explicit bin_borders must change bin assignments vs tertile default."""
    rng = np.random.default_rng(100)
    y = _skewed_labels(rng)  # cluster near 7, tails at 3-4 and 10-11

    # Tertile default would place borders near ~6.8 and ~7.3 (skewed toward center).
    # Explicit [5, 9] (EpHod-style) produces a very different partition.
    w_default = compute_sample_weights(y, method='bin_inv')
    w_explicit = compute_sample_weights(y, method='bin_inv', bin_borders=[5, 9])

    assert not np.allclose(w_default, w_explicit)
    # Under [5, 9], acidic (<=5) and alkaline (>=9) tails are both rare -> heavier weights.
    rare = (y <= 5) | (y >= 9)
    assert float(np.mean(w_explicit[rare])) > float(np.mean(w_explicit[~rare]))


def test_explicit_bin_borders_three_bins_create_four_classes():
    rng = np.random.default_rng(101)
    y = rng.uniform(0, 10, size=500)
    y_pred = y.copy()
    borders = [2.5, 5.0, 7.5]
    w = compute_sample_weights(y, method='bin_inv', bin_borders=borders)
    m = compute_balanced_regression_metrics(y, y_pred, w, borders, n_resamples=10, seed=0)
    assert m['n_bins'] == 4
    assert m['bin_borders'] == borders
    assert len(m['binned_f1_per_bin']) == 4
    assert len(m['binned_roc_auc_per_bin']) == 4


def test_explicit_bin_borders_propagated_to_digitize():
    """Bin assignments in balanced metrics must honor the passed bin_borders (not recompute)."""
    rng = np.random.default_rng(102)
    y = _skewed_labels(rng)
    y_pred = y + rng.normal(0, 0.8, size=len(y))  # noisy preds so F1 varies with bin choice
    w = compute_sample_weights(y, method='bin_inv', bin_borders=[5, 9])

    m_59 = compute_balanced_regression_metrics(y, y_pred, w, [5, 9], n_resamples=10, seed=0)
    m_46 = compute_balanced_regression_metrics(y, y_pred, w, [4, 6], n_resamples=10, seed=0)

    assert m_59['n_bins'] == 3
    assert m_46['n_bins'] == 3
    assert m_59['bin_borders'] == [5, 9]
    assert m_46['bin_borders'] == [4, 6]
    # Per-bin breakdown must differ because bins are different
    assert m_59['binned_f1_per_bin'] != m_46['binned_f1_per_bin']
    assert m_59['binned_mcc'] != m_46['binned_mcc']


def test_explicit_bin_borders_single_threshold():
    """A single border (e.g., [7]) should give 2 bins (binary classification)."""
    rng = np.random.default_rng(103)
    y = _skewed_labels(rng)
    y_pred = y + rng.normal(0, 0.2, size=len(y))
    w = compute_sample_weights(y, method='bin_inv', bin_borders=[7.0])
    m = compute_balanced_regression_metrics(y, y_pred, w, [7.0], n_resamples=10, seed=0)
    assert m['n_bins'] == 2
    assert len(m['binned_f1_per_bin']) == 2


def test_apply_weights_from_reference_honors_explicit_borders():
    rng = np.random.default_rng(104)
    y_train = _skewed_labels(rng)
    y_valid = rng.uniform(3, 11, size=200)

    w_default = apply_weights_from_reference(y_valid, y_train, method='bin_inv')
    w_explicit = apply_weights_from_reference(y_valid, y_train, method='bin_inv', bin_borders=[5, 9])

    assert not np.allclose(w_default, w_explicit)


def test_data_mixin_honors_explicit_bin_borders(monkeypatch):
    """Simulate DataMixin._compute_balanced_weights_for with explicit vs default borders."""
    try:
        from src.protify.metrics_balanced import _default_bin_borders
        from src.protify.metrics_balanced import compute_sample_weights as csw
    except ImportError:
        from protify.metrics_balanced import _default_bin_borders
        from protify.metrics_balanced import compute_sample_weights as csw

    rng = np.random.default_rng(105)
    train_labels = _skewed_labels(rng)

    # Mimic the line in data_mixin.py:
    #   bin_borders = ta.balanced_bin_borders if ta.balanced_bin_borders else _default_bin_borders(train_labels)
    explicit = [5.0, 9.0]
    resolved_explicit = explicit if explicit else _default_bin_borders(train_labels)
    resolved_default = None if None else _default_bin_borders(train_labels)

    assert resolved_explicit == [5.0, 9.0]
    assert resolved_default != [5.0, 9.0]

    w_explicit = csw(train_labels, method='bin_inv', bin_borders=resolved_explicit)
    w_default = csw(train_labels, method='bin_inv', bin_borders=resolved_default)
    assert not np.allclose(w_explicit, w_default)


def test_data_mixin_compute_balanced_weights_without_init():
    """Regression guard: MainProcess skips DataMixin.__init__ due to super() chain,
    so _compute_balanced_weights_for must self-initialize self.balanced_weights."""
    try:
        from protify.data.data_mixin import DataMixin
    except ImportError:
        try:
            from ..data.data_mixin import DataMixin
        except ImportError:
            from src.protify.data.data_mixin import DataMixin

    class FakeTrainerArgs:
        def __init__(self):
            self.balanced_regression_metrics = True
            self.balanced_weight_method = 'bin_inv'
            self.balanced_bin_borders = [5.0, 9.0]
            self.balanced_n_resamples = 10
            self.balanced_lds_bins = 100
            self.balanced_lds_ks = 5
            self.balanced_lds_sigma = 2.0

    class FakeSplit:
        def __init__(self, labels):
            self._labels = labels
        def __getitem__(self, key):
            assert key == 'labels'
            return self._labels

    rng = np.random.default_rng(200)
    labels = _skewed_labels(rng).tolist()
    split = FakeSplit(labels)

    # Bypass DataMixin.__init__ the same way MainProcess does.
    mixin = DataMixin.__new__(DataMixin)
    mixin.trainer_args = FakeTrainerArgs()
    assert 'balanced_weights' not in mixin.__dict__

    # Must not raise AttributeError
    mixin._compute_balanced_weights_for('fake_dataset', split, split, split)

    assert 'balanced_weights' in mixin.__dict__
    assert 'fake_dataset' in mixin.balanced_weights
    bw = mixin.balanced_weights['fake_dataset']
    assert set(bw.keys()) >= {'train', 'valid', 'test', 'bin_borders', 'method'}
    assert bw['bin_borders'] == [5.0, 9.0]


def test_bin_borders_digitize_semantics():
    """Document np.digitize semantics that bin_borders relies on."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    # Default right=False: bins[i-1] <= x < bins[i]
    assert np.digitize(y, [3.0, 7.0]).tolist() == [0, 0, 1, 1, 1, 1, 2, 2, 2]
    # A value exactly on the border lands in the upper bin.
    assert int(np.digitize(np.array([3.0]), [3.0, 7.0])[0]) == 1
