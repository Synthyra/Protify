import numpy as np
import pytest
import torch
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


class TestSoftmax:
    def test_sums_to_one(self):
        logits = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])  # (b=2, c=3)
        probabilities = softmax(logits)  # (b, c)
        np.testing.assert_allclose(
            probabilities.sum(axis=-1),
            [1.0, 1.0],
            atol=1e-7,
        )

    def test_uniform_input(self):
        logits = np.array([[0.0, 0.0, 0.0]])  # (b=1, c=3)
        probabilities = softmax(logits)  # (b, c)
        np.testing.assert_allclose(
            probabilities,
            [[1 / 3, 1 / 3, 1 / 3]],
            atol=1e-7,
        )

    def test_1d_input(self):
        logits = np.array([1.0, 2.0])  # (c=2,)
        probabilities = softmax(logits)  # (c,)
        assert probabilities.shape == (2,)
        np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-7)

    def test_large_logits_no_overflow(self):
        logits = np.array([[1000.0, 1001.0, 1002.0]])  # (b=1, c=3)
        probabilities = softmax(logits)  # (b, c)
        assert not np.any(np.isnan(probabilities)), "Softmax overflowed with large logits"
        assert not np.any(np.isinf(probabilities)), "Softmax produced inf with large logits"
        np.testing.assert_allclose(probabilities.sum(axis=-1), [1.0], atol=1e-7)


class TestScorers:
    def test_regression_scorer_perfect(self):
        scorer = regression_scorer()
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # (n=5,)
        score = scorer(targets, targets)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_classification_scorer_perfect(self):
        scorer = classification_scorer()
        targets = np.array([0, 1, 0, 1, 1])  # (n=5,)
        score = scorer(targets, targets)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_classification_scorer_random(self):
        scorer = classification_scorer()
        y_true = np.array([0, 0, 1, 1])  # (n=4,)
        y_pred = np.array([1, 0, 0, 1])  # (n,)
        score = scorer(y_true, y_pred)
        assert score == pytest.approx(0.0, abs=1e-5)


class TestCalculateMaxMetrics:
    def test_perfect_separation(self):
        scores = torch.tensor([0.9, 0.8, 0.1, 0.2])  # (n=4,)
        labels = torch.tensor([1, 1, 0, 0])  # (n,)
        f1, precision, recall = calculate_max_metrics(  # each ()
            scores,
            labels,
            cutoff=0.5,
        )
        assert f1.item() == pytest.approx(1.0)
        assert precision.item() == pytest.approx(1.0)
        assert recall.item() == pytest.approx(1.0)

    def test_all_below_cutoff(self):
        scores = torch.tensor([0.1, 0.2, 0.3])  # (n=3,)
        labels = torch.tensor([1, 1, 0])  # (n,)
        _f1, _precision, recall = calculate_max_metrics(  # each ()
            scores,
            labels,
            cutoff=0.9,
        )
        assert recall.item() == pytest.approx(0.0)


class TestMaxMetrics:
    def test_perfect_binary(self):
        scores = torch.tensor([0.9, 0.8, 0.1, 0.05])  # (n=4,)
        labels = torch.tensor([1, 1, 0, 0])  # (n,)
        f1, precision, recall, _cutoff = max_metrics(scores, labels)
        assert f1 == pytest.approx(1.0, abs=0.02)
        assert precision == pytest.approx(1.0, abs=0.02)
        assert recall == pytest.approx(1.0, abs=0.02)

    def test_nan_scores_handled(self):
        scores = torch.tensor([float('nan'), 0.5, 0.9])  # (n=3,)
        labels = torch.tensor([0, 0, 1])  # (n,)
        f1, _precision, _recall, _cutoff = max_metrics(scores, labels)
        assert not np.isnan(f1)

    def test_all_same_score(self):
        scores = torch.tensor([0.5, 0.5, 0.5, 0.5])  # (n=4,)
        labels = torch.tensor([1, 0, 1, 0])  # (n,)
        f1, _precision, _recall, _cutoff = max_metrics(scores, labels)
        assert isinstance(f1, float)


class TestSingleLabelClassification:
    def test_binary_perfect(self):
        logits = np.array([[5.0, -5.0], [-5.0, 5.0]])  # (b=2, c=2)
        labels = np.array([0, 1])  # (b,)
        evaluation = EvalPrediction(predictions=logits, label_ids=labels)
        metrics = compute_single_label_classification_metrics(evaluation)
        assert metrics['accuracy'] == pytest.approx(1.0)
        assert metrics['f1'] == pytest.approx(1.0)
        assert metrics['mcc'] == pytest.approx(1.0)

    def test_multiclass_perfect(self):
        logits = np.array(  # (b=3, c=3)
            [[5.0, -5.0, -5.0], [-5.0, 5.0, -5.0], [-5.0, -5.0, 5.0]]
        )
        labels = np.array([0, 1, 2])  # (b,)
        evaluation = EvalPrediction(predictions=logits, label_ids=labels)
        metrics = compute_single_label_classification_metrics(evaluation)
        assert metrics['accuracy'] == pytest.approx(1.0)
        assert metrics['f1'] == pytest.approx(1.0)

    def test_expected_keys(self):
        logits = np.array([[2.0, -1.0], [-1.0, 2.0]])  # (b=2, c=2)
        labels = np.array([0, 1])  # (b,)
        evaluation = EvalPrediction(predictions=logits, label_ids=labels)
        metrics = compute_single_label_classification_metrics(evaluation)
        expected_keys = {'f1', 'precision', 'recall', 'accuracy', 'mcc', 'roc_auc', 'pr_auc'}
        assert set(metrics.keys()) == expected_keys

    def test_tuple_predictions(self):
        logits = np.array([[5.0, -5.0], [-5.0, 5.0]])  # (b=2, c=2)
        auxiliary_output = np.zeros(2)  # (b,)
        labels = np.array([0, 1])  # (b,)
        evaluation = EvalPrediction(
            predictions=(logits, auxiliary_output),
            label_ids=labels,
        )
        metrics = compute_single_label_classification_metrics(evaluation)
        assert metrics['accuracy'] == pytest.approx(1.0)


class TestTokenwiseClassification:
    def test_ignores_padding(self):
        # 1 sample, 4 tokens, 2 classes; last token is padding
        logits = np.array(  # (b=1, l=4, c=2)
            [[[5.0, -5.0], [-5.0, 5.0], [5.0, -5.0], [0.0, 0.0]]]
        )
        labels = np.array([[0, 1, 0, -100]])  # (b, l)
        evaluation = EvalPrediction(predictions=logits, label_ids=labels)
        metrics = compute_tokenwise_classification_metrics(evaluation)
        assert metrics['accuracy'] == pytest.approx(1.0)
        assert metrics['f1'] == pytest.approx(1.0)

    def test_expected_keys(self):
        logits = np.array([[[2.0, -1.0], [-1.0, 2.0]]])  # (b=1, l=2, c=2)
        labels = np.array([[0, 1]])  # (b, l)
        evaluation = EvalPrediction(predictions=logits, label_ids=labels)
        metrics = compute_tokenwise_classification_metrics(evaluation)
        expected_keys = {'accuracy', 'f1', 'precision', 'recall', 'mcc', 'roc_auc', 'pr_auc'}
        assert set(metrics.keys()) == expected_keys


class TestMultiLabelClassification:
    def test_perfect(self):
        logits = np.array(  # (b=2, c=3)
            [[5.0, -5.0, 5.0], [-5.0, 5.0, -5.0]]
        )
        labels = np.array([[1, 0, 1], [0, 1, 0]])  # (b, c)
        evaluation = EvalPrediction(predictions=logits, label_ids=labels)
        metrics = compute_multi_label_classification_metrics(evaluation)
        assert metrics['accuracy'] == pytest.approx(1.0)
        assert metrics['hamming_loss'] == pytest.approx(0.0)

    def test_expected_keys(self):
        logits = np.array([[5.0, -5.0], [-5.0, 5.0]])  # (b=2, c=2)
        labels = np.array([[1, 0], [0, 1]])  # (b, c)
        evaluation = EvalPrediction(predictions=logits, label_ids=labels)
        metrics = compute_multi_label_classification_metrics(evaluation)
        expected_keys = {
            'accuracy', 'f1', 'precision', 'recall', 'hamming_loss',
            'threshold', 'mcc', 'roc_auc', 'pr_auc',
        }
        assert set(metrics.keys()) == expected_keys


class TestRegressionMetrics:
    def test_perfect(self):
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # (n=5,)
        labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # (n,)
        evaluation = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_regression_metrics(evaluation)
        assert metrics['r_squared'] == pytest.approx(1.0)
        assert metrics['mse'] == pytest.approx(0.0, abs=1e-7)
        assert metrics['mae'] == pytest.approx(0.0, abs=1e-7)
        assert metrics['spearman_rho'] == pytest.approx(1.0, abs=1e-5)

    def test_expected_keys(self):
        predictions = np.array([1.0, 2.0, 3.0])  # (n=3,)
        labels = np.array([1.1, 1.9, 3.2])  # (n,)
        evaluation = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_regression_metrics(evaluation)
        expected_keys = {
            'r_squared', 'spearman_rho', 'spear_pval', 'pearson_rho',
            'pear_pval', 'mse', 'mae', 'rmse',
        }
        assert set(metrics.keys()) == expected_keys

    def test_tuple_predictions(self):
        predictions = np.array([1.0, 2.0, 3.0])  # (n=3,)
        labels = np.array([1.0, 2.0, 3.0])  # (n,)
        auxiliary_predictions = np.zeros(3)  # (n,)
        auxiliary_labels = np.zeros(3)  # (n,)
        evaluation = EvalPrediction(
            predictions=(predictions, auxiliary_predictions),
            label_ids=(auxiliary_labels, labels),
        )
        metrics = compute_regression_metrics(evaluation)
        assert metrics['r_squared'] == pytest.approx(1.0)


class TestTokenwiseRegression:
    def test_ignores_padding(self):
        predictions = np.array([[1.0, 2.0, 999.0]])  # (b=1, l=3)
        labels = np.array([[1.0, 2.0, -100.0]])  # (b, l)
        evaluation = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_tokenwise_regression_metrics(evaluation)
        assert metrics['mse'] == pytest.approx(0.0, abs=1e-7)
        assert metrics['r_squared'] == pytest.approx(1.0)

    def test_all_padding_returns_sentinel(self):
        predictions = np.array([[1.0, 2.0]])  # (b=1, l=2)
        labels = np.array([[-100.0, -100.0]])  # (b, l)
        evaluation = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_tokenwise_regression_metrics(evaluation)
        assert metrics['r_squared'] == -100.0
        assert metrics['spearman_rho'] == -100.0

    def test_squeeze_trailing_dim(self):
        predictions = np.array([[[1.0], [2.0], [3.0]]])  # (b=1, l=3, 1)
        labels = np.array([[1.0, 2.0, 3.0]])  # (b, l)
        evaluation = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_tokenwise_regression_metrics(evaluation)
        assert metrics['mse'] == pytest.approx(0.0, abs=1e-7)


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


class TestRobustAUC:
    def test_roc_auc_multiclass_binary(self):
        y_true = np.array([0, 0, 1, 1])  # (n=4,)
        probabilities = np.array(  # (n, c=2)
            [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
        )
        score = calculate_robust_roc_auc_multiclass(y_true, probabilities)
        assert score == pytest.approx(1.0)

    def test_roc_auc_multiclass_single_class_returns_sentinel(self):
        y_true = np.array([0, 0, 0])  # (n=3,)
        probabilities = np.array(  # (n, c=2)
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]
        )
        score = calculate_robust_roc_auc_multiclass(y_true, probabilities)
        assert score == -100.0

    def test_pr_auc_multiclass_binary(self):
        y_true = np.array([0, 0, 1, 1])  # (n=4,)
        probabilities = np.array(  # (n, c=2)
            [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
        )
        score = calculate_robust_pr_auc_multiclass(y_true, probabilities)
        assert score > 0.9

    def test_roc_auc_multilabel_perfect(self):
        y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])  # (n=4, c=2)
        probabilities = np.array(  # (n, c)
            [[0.9, 0.1], [0.1, 0.9], [0.9, 0.9], [0.1, 0.1]]
        )
        score = calculate_robust_roc_auc_multilabel(y_true, probabilities)
        assert score == pytest.approx(1.0)

    def test_pr_auc_multilabel(self):
        y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])  # (n=4, c=2)
        probabilities = np.array(  # (n, c)
            [[0.9, 0.1], [0.1, 0.9], [0.9, 0.9], [0.1, 0.1]]
        )
        score = calculate_robust_pr_auc_multilabel(y_true, probabilities)
        assert score > 0.9

    def test_roc_auc_multilabel_single_class_column(self):
        y_true = np.array([[1, 0], [1, 0], [1, 0]])  # (n=3, c=2)
        probabilities = np.array(  # (n, c)
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]
        )
        score = calculate_robust_roc_auc_multilabel(y_true, probabilities)
        # Column 0 is all 1s, column 1 is all 0s: no valid per-label AUC
        assert score == -100.0

    def test_nan_probs_handled(self):
        y_true = np.array([0, 1, 0, 1])  # (n=4,)
        probabilities = np.array(  # (n, c=2)
            [[0.9, 0.1], [float('nan'), 0.8], [0.7, 0.3], [0.2, 0.8]]
        )
        score = calculate_robust_roc_auc_multiclass(y_true, probabilities)
        assert not np.isnan(score)
