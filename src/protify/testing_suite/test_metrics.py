import numpy as np
import torch
import pytest
from transformers import EvalPrediction

try:
    from src.protify.metrics import (
        softmax,
        regression_scorer,
        classification_scorer,
        calculate_max_metrics,
        max_metrics,
        compute_single_label_classification_metrics,
        compute_tokenwise_classification_metrics,
        compute_multi_label_classification_metrics,
        compute_regression_metrics,
        compute_tokenwise_regression_metrics,
        get_compute_metrics,
        calculate_robust_roc_auc_multiclass,
        calculate_robust_pr_auc_multiclass,
        calculate_robust_roc_auc_multilabel,
        calculate_robust_pr_auc_multilabel,
    )
except ImportError:
    try:
        from protify.metrics import (
            softmax,
            regression_scorer,
            classification_scorer,
            calculate_max_metrics,
            max_metrics,
            compute_single_label_classification_metrics,
            compute_tokenwise_classification_metrics,
            compute_multi_label_classification_metrics,
            compute_regression_metrics,
            compute_tokenwise_regression_metrics,
            get_compute_metrics,
            calculate_robust_roc_auc_multiclass,
            calculate_robust_pr_auc_multiclass,
            calculate_robust_roc_auc_multilabel,
            calculate_robust_pr_auc_multilabel,
        )
    except ImportError:
        from ..metrics import (
            softmax,
            regression_scorer,
            classification_scorer,
            calculate_max_metrics,
            max_metrics,
            compute_single_label_classification_metrics,
            compute_tokenwise_classification_metrics,
            compute_multi_label_classification_metrics,
            compute_regression_metrics,
            compute_tokenwise_regression_metrics,
            get_compute_metrics,
            calculate_robust_roc_auc_multiclass,
            calculate_robust_pr_auc_multiclass,
            calculate_robust_roc_auc_multilabel,
            calculate_robust_pr_auc_multilabel,
        )


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------

class TestSoftmax:
    def test_sums_to_one(self):
        x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        result = softmax(x)
        np.testing.assert_allclose(result.sum(axis=-1), [1.0, 1.0], atol=1e-7)

    def test_uniform_input(self):
        x = np.array([[0.0, 0.0, 0.0]])
        result = softmax(x)
        np.testing.assert_allclose(result, [[1 / 3, 1 / 3, 1 / 3]], atol=1e-7)

    def test_1d_input(self):
        x = np.array([1.0, 2.0])
        result = softmax(x)
        assert result.shape == (2,)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-7)

    def test_large_logits_no_overflow(self):
        x = np.array([[1000.0, 1001.0, 1002.0]])
        result = softmax(x)
        assert not np.any(np.isnan(result)), "Softmax overflowed with large logits"
        assert not np.any(np.isinf(result)), "Softmax produced inf with large logits"
        np.testing.assert_allclose(result.sum(axis=-1), [1.0], atol=1e-7)


# ---------------------------------------------------------------------------
# regression_scorer / classification_scorer
# ---------------------------------------------------------------------------

class TestScorers:
    def test_regression_scorer_perfect(self):
        scorer = regression_scorer()
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        score = scorer(y, y)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_classification_scorer_perfect(self):
        scorer = classification_scorer()
        y = np.array([0, 1, 0, 1, 1])
        score = scorer(y, y)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_classification_scorer_random(self):
        scorer = classification_scorer()
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1])
        score = scorer(y_true, y_pred)
        assert score == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# calculate_max_metrics
# ---------------------------------------------------------------------------

class TestCalculateMaxMetrics:
    def test_perfect_separation(self):
        ss = torch.tensor([0.9, 0.8, 0.1, 0.2])
        labels = torch.tensor([1, 1, 0, 0])
        f1, prec, rec = calculate_max_metrics(ss, labels, cutoff=0.5)
        assert f1.item() == pytest.approx(1.0)
        assert prec.item() == pytest.approx(1.0)
        assert rec.item() == pytest.approx(1.0)

    def test_all_below_cutoff(self):
        ss = torch.tensor([0.1, 0.2, 0.3])
        labels = torch.tensor([1, 1, 0])
        f1, prec, rec = calculate_max_metrics(ss, labels, cutoff=0.9)
        assert rec.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# max_metrics
# ---------------------------------------------------------------------------

class TestMaxMetrics:
    def test_perfect_binary(self):
        ss = torch.tensor([0.9, 0.8, 0.1, 0.05])
        labels = torch.tensor([1, 1, 0, 0])
        f1, prec, rec, cutoff = max_metrics(ss, labels)
        assert f1 == pytest.approx(1.0, abs=0.02)
        assert prec == pytest.approx(1.0, abs=0.02)
        assert rec == pytest.approx(1.0, abs=0.02)

    def test_nan_scores_handled(self):
        ss = torch.tensor([float('nan'), 0.5, 0.9])
        labels = torch.tensor([0, 0, 1])
        f1, prec, rec, cutoff = max_metrics(ss, labels)
        assert not np.isnan(f1)

    def test_all_same_score(self):
        ss = torch.tensor([0.5, 0.5, 0.5, 0.5])
        labels = torch.tensor([1, 0, 1, 0])
        f1, prec, rec, cutoff = max_metrics(ss, labels)
        assert isinstance(f1, float)


# ---------------------------------------------------------------------------
# compute_single_label_classification_metrics
# ---------------------------------------------------------------------------

class TestSingleLabelClassification:
    def test_binary_perfect(self):
        logits = np.array([[5.0, -5.0], [-5.0, 5.0]])
        labels = np.array([0, 1])
        p = EvalPrediction(predictions=logits, label_ids=labels)
        m = compute_single_label_classification_metrics(p)
        assert m['accuracy'] == pytest.approx(1.0)
        assert m['f1'] == pytest.approx(1.0)
        assert m['mcc'] == pytest.approx(1.0)

    def test_multiclass_perfect(self):
        logits = np.array([[5.0, -5.0, -5.0], [-5.0, 5.0, -5.0], [-5.0, -5.0, 5.0]])
        labels = np.array([0, 1, 2])
        p = EvalPrediction(predictions=logits, label_ids=labels)
        m = compute_single_label_classification_metrics(p)
        assert m['accuracy'] == pytest.approx(1.0)
        assert m['f1'] == pytest.approx(1.0)

    def test_expected_keys(self):
        logits = np.array([[2.0, -1.0], [-1.0, 2.0]])
        labels = np.array([0, 1])
        p = EvalPrediction(predictions=logits, label_ids=labels)
        m = compute_single_label_classification_metrics(p)
        expected_keys = {'f1', 'precision', 'recall', 'accuracy', 'mcc', 'roc_auc', 'pr_auc'}
        assert set(m.keys()) == expected_keys

    def test_tuple_predictions(self):
        logits = np.array([[5.0, -5.0], [-5.0, 5.0]])
        labels = np.array([0, 1])
        p = EvalPrediction(predictions=(logits, np.zeros(2)), label_ids=labels)
        m = compute_single_label_classification_metrics(p)
        assert m['accuracy'] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_tokenwise_classification_metrics
# ---------------------------------------------------------------------------

class TestTokenwiseClassification:
    def test_ignores_padding(self):
        # 1 sample, 4 tokens, 2 classes; last token is padding
        logits = np.array([[[5.0, -5.0], [-5.0, 5.0], [5.0, -5.0], [0.0, 0.0]]])
        labels = np.array([[0, 1, 0, -100]])
        p = EvalPrediction(predictions=logits, label_ids=labels)
        m = compute_tokenwise_classification_metrics(p)
        assert m['accuracy'] == pytest.approx(1.0)
        assert m['f1'] == pytest.approx(1.0)

    def test_expected_keys(self):
        logits = np.array([[[2.0, -1.0], [-1.0, 2.0]]])
        labels = np.array([[0, 1]])
        p = EvalPrediction(predictions=logits, label_ids=labels)
        m = compute_tokenwise_classification_metrics(p)
        expected_keys = {'accuracy', 'f1', 'precision', 'recall', 'mcc', 'roc_auc', 'pr_auc'}
        assert set(m.keys()) == expected_keys


# ---------------------------------------------------------------------------
# compute_multi_label_classification_metrics
# ---------------------------------------------------------------------------

class TestMultiLabelClassification:
    def test_perfect(self):
        logits = np.array([[5.0, -5.0, 5.0], [-5.0, 5.0, -5.0]])
        labels = np.array([[1, 0, 1], [0, 1, 0]])
        p = EvalPrediction(predictions=logits, label_ids=labels)
        m = compute_multi_label_classification_metrics(p)
        assert m['accuracy'] == pytest.approx(1.0)
        assert m['hamming_loss'] == pytest.approx(0.0)

    def test_expected_keys(self):
        logits = np.array([[5.0, -5.0], [-5.0, 5.0]])
        labels = np.array([[1, 0], [0, 1]])
        p = EvalPrediction(predictions=logits, label_ids=labels)
        m = compute_multi_label_classification_metrics(p)
        expected_keys = {'accuracy', 'f1', 'precision', 'recall', 'hamming_loss', 'threshold', 'mcc', 'roc_auc', 'pr_auc'}
        assert set(m.keys()) == expected_keys


# ---------------------------------------------------------------------------
# compute_regression_metrics
# ---------------------------------------------------------------------------

class TestRegressionMetrics:
    def test_perfect(self):
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p = EvalPrediction(predictions=preds, label_ids=labels)
        m = compute_regression_metrics(p)
        assert m['r_squared'] == pytest.approx(1.0)
        assert m['mse'] == pytest.approx(0.0, abs=1e-7)
        assert m['mae'] == pytest.approx(0.0, abs=1e-7)
        assert m['spearman_rho'] == pytest.approx(1.0, abs=1e-5)

    def test_expected_keys(self):
        preds = np.array([1.0, 2.0, 3.0])
        labels = np.array([1.1, 1.9, 3.2])
        p = EvalPrediction(predictions=preds, label_ids=labels)
        m = compute_regression_metrics(p)
        expected_keys = {'r_squared', 'spearman_rho', 'spear_pval', 'pearson_rho', 'pear_pval', 'mse', 'mae', 'rmse'}
        assert set(m.keys()) == expected_keys

    def test_tuple_predictions(self):
        preds = np.array([1.0, 2.0, 3.0])
        labels = np.array([1.0, 2.0, 3.0])
        p = EvalPrediction(predictions=(preds, np.zeros(3)), label_ids=(np.zeros(3), labels))
        m = compute_regression_metrics(p)
        assert m['r_squared'] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_tokenwise_regression_metrics
# ---------------------------------------------------------------------------

class TestTokenwiseRegression:
    def test_ignores_padding(self):
        preds = np.array([[1.0, 2.0, 999.0]])
        labels = np.array([[1.0, 2.0, -100.0]])
        p = EvalPrediction(predictions=preds, label_ids=labels)
        m = compute_tokenwise_regression_metrics(p)
        assert m['mse'] == pytest.approx(0.0, abs=1e-7)
        assert m['r_squared'] == pytest.approx(1.0)

    def test_all_padding_returns_sentinel(self):
        preds = np.array([[1.0, 2.0]])
        labels = np.array([[-100.0, -100.0]])
        p = EvalPrediction(predictions=preds, label_ids=labels)
        m = compute_tokenwise_regression_metrics(p)
        assert m['r_squared'] == -100.0
        assert m['spearman_rho'] == -100.0

    def test_squeeze_trailing_dim(self):
        preds = np.array([[[1.0], [2.0], [3.0]]])
        labels = np.array([[1.0, 2.0, 3.0]])
        p = EvalPrediction(predictions=preds, label_ids=labels)
        m = compute_tokenwise_regression_metrics(p)
        assert m['mse'] == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# get_compute_metrics
# ---------------------------------------------------------------------------

class TestGetComputeMetrics:
    def test_singlelabel(self):
        fn = get_compute_metrics('singlelabel')
        assert fn is compute_single_label_classification_metrics

    def test_multilabel(self):
        fn = get_compute_metrics('multilabel')
        assert fn is compute_multi_label_classification_metrics

    def test_regression(self):
        fn = get_compute_metrics('regression')
        assert fn is compute_regression_metrics

    def test_regression_tokenwise(self):
        fn = get_compute_metrics('regression', tokenwise=True)
        assert fn is compute_tokenwise_regression_metrics

    def test_sigmoid_regression(self):
        fn = get_compute_metrics('sigmoid_regression')
        assert fn is compute_regression_metrics

    def test_sigmoid_regression_tokenwise(self):
        fn = get_compute_metrics('sigmoid_regression', tokenwise=True)
        assert fn is compute_tokenwise_regression_metrics

    def test_tokenwise_classification(self):
        # singlelabel is matched first regardless of tokenwise flag;
        # tokenwise classification dispatch requires a non-standard task_type
        fn = get_compute_metrics('singlelabel', tokenwise=True)
        assert fn is compute_single_label_classification_metrics
        # The tokenwise classification branch is reached with non-regression, non-standard types
        fn2 = get_compute_metrics('string', tokenwise=True)
        assert fn2 is compute_tokenwise_classification_metrics


# ---------------------------------------------------------------------------
# Robust AUC helpers
# ---------------------------------------------------------------------------

class TestRobustAUC:
    def test_roc_auc_multiclass_binary(self):
        y_true = np.array([0, 0, 1, 1])
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        score = calculate_robust_roc_auc_multiclass(y_true, probs)
        assert score == pytest.approx(1.0)

    def test_roc_auc_multiclass_single_class_returns_sentinel(self):
        y_true = np.array([0, 0, 0])
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
        score = calculate_robust_roc_auc_multiclass(y_true, probs)
        assert score == -100.0

    def test_pr_auc_multiclass_binary(self):
        y_true = np.array([0, 0, 1, 1])
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        score = calculate_robust_pr_auc_multiclass(y_true, probs)
        assert score > 0.9

    def test_roc_auc_multilabel_perfect(self):
        y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.9], [0.1, 0.1]])
        score = calculate_robust_roc_auc_multilabel(y_true, probs)
        assert score == pytest.approx(1.0)

    def test_pr_auc_multilabel(self):
        y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.9], [0.1, 0.1]])
        score = calculate_robust_pr_auc_multilabel(y_true, probs)
        assert score > 0.9

    def test_roc_auc_multilabel_single_class_column(self):
        y_true = np.array([[1, 0], [1, 0], [1, 0]])
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
        score = calculate_robust_roc_auc_multilabel(y_true, probs)
        # Column 0 is all 1s, column 1 is all 0s: no valid per-label AUC
        assert score == -100.0

    def test_nan_probs_handled(self):
        y_true = np.array([0, 1, 0, 1])
        probs = np.array([[0.9, 0.1], [float('nan'), 0.8], [0.7, 0.3], [0.2, 0.8]])
        score = calculate_robust_roc_auc_multiclass(y_true, probs)
        assert not np.isnan(score)
