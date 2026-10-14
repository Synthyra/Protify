---
id: foundry.metrics
kind: library_module
status: active
aliases: [foundry.metrics]
---
# foundry.metrics

How well a model's predictions agree with what happened: calibration and selective prediction in
NumPy, and the classification and regression summaries DSM reports.

| Name | Does |
|---|---|
| `expected_calibration_error(labels, probs, bins)` | Equal-mass ECE: rows sorted by probability, ties in input order, cut into `bins` groups within one row of each other in size, each weighted by its share of the rows. Binary labels take the probability of 1; top-label multiclass ECE is the same call on `argmax == label` and the maximum probability |
| `risk_coverage_curve(losses, uncertainty, coverage=None)` | `(coverage, risk)`: the mean loss over the least uncertain rows at each coverage, keeping `max(1, round(c * n))` of them. Without a grid, every count `k = 1..n` is a level. Any per-row loss works: misclassification, absolute or squared error |
| `area_under_risk_coverage(losses, uncertainty, coverage=None)` | AURC: the trapezoidal area under that curve divided by the coverage it spans, so a constant risk `r` scores `r`. Lower means the uncertainty puts the costly rows last |

Promoted from sae_xgboost_ppi's `gp_analysis.py`. `expected_calibration_error` is unchanged. The
curve and the area generalize `_risk_coverage` and `_aurc_only`, which fixed 19 coverages from
0.1 to 1.0 and scored misclassification; on that grid and loss they reproduce its numbers
exactly. Protify's uncertainty workflow is the second consumer.

sae_xgboost_ppi still keeps its private copies, the ones its reported numbers came from. Its
Modal image carries foundry now, so nothing blocks the switch.
`tests/test_foundry_equivalence.py` in its suite holds them equal to these until it switches. A
consumer lists `foundry.metrics` under `github.includes`, so `ws publish` vendors it into the
projection.

## Binary summaries from counts and rankings

In `foundry.metrics.binary`, exported by the package, NumPy only.

| Name | Does |
|---|---|
| `compute_metrics_from_counts(tp, fp, fn, tn)` | Precision, recall, accuracy, F1, and MCC of the positive class from a confusion matrix's counts, each 0.0 where its denominator is 0 |
| `calculate_topk_metrics(labels, probs, k_values=(10, 100, 1000))` | For each `k`, the positive share of the `k` highest scores (`topk_pred_pos_acc_{k}`) and the negative share of the `k` lowest (`topk_pred_neg_acc_{k}`), rows with a non-finite label or score left out and `k` capped at the rows left |
| `balanced_subsamples(labels, *, seed=0, subsamples=10)` | Index sets of class-balanced views, the majority class downsampled to the minority's size with Atlas's seeds `seed + 11 * (i + 1)`; a balanced input is its own single view |
| `balanced_mcc(labels, scores, *, seed=0, subsamples=10)` | Balanced MCC: the MCC at the threshold that maximizes its mean over those views (score at or above predicts positive, ties to the lowest threshold), its standard deviation across views, and the threshold; counted from sorted scores, so millions of pairs are cheap. Atlas's threshold calibration and Atlas v3's pair metrics use it |

Promoted from Atlas's `utils/metrics.py`, whose trainer reduces MCC from counts gathered across
processes, and synth's `synth/utilities/metrics.py`, which held both token for token. Both import
them from here under the same names. `nan_balanced_classification_metrics` stays in each project,
because synth's adds a `balanced_threshold` key Atlas's lacks. Tests:
`tests/tier1_unit/test_metrics_binary.py`, and Atlas's `tests/test_utils_metrics.py` from the
consumer's side.

## Canonical binary ranking metrics

`foundry.metrics.ranking.binary_ranking_metrics(labels, scores)` accepts aligned one-dimensional
arrays of observed binary targets and finite real scores. The caller applies the observation
mask first. Scores need not be probabilities. It returns three separately named values:

| Key | Definition and support |
|---|---|
| `average_precision` | Sum of precision weighted by recall increments at distinct score thresholds; requires a positive |
| `pr_auc_trapezoidal` | Trapezoidal area of the threshold precision-recall curve, including recall zero / precision one; requires a positive |
| `auroc` | Probability that a positive outranks a negative, with half credit for ties; requires both classes |

An unsupported denominator returns `None`, including all three values for empty arrays.
Malformed shapes, unobserved/sentinel targets, and non-finite scores raise `ValueError`.
This module needs scikit-learn; the package root remains NumPy only.

Promoted from Protify's declared Oracle scoring when the suite's declared retrieval evaluator
became its second consumer. Protify retains its endpoint, threshold, aggregation, and unit
contract; the suite owns gallery construction, per-query ranking, and unknown-label policies.
See [declared Oracle metrics](../../projects/tooling/property_lowcode/README.md#declared-metric-definitions)
and [declared retrieval evaluation](../../projects/serving/production_suite/evaluation.md#declared-retrieval-evaluation).
Tests live with both consumers: Protify's `test_metric_contracts.py` and the suite's
`tests/test_retrieval_metrics.py`. Their GH200 checks include hand-calculated AP versus PR
area, tied scores, unsupported targets, and a refactor parity run of the existing Oracle cases.

## Multiple testing

`foundry.metrics.multiple_testing`, NumPy only, imported by its module path
(`from foundry.metrics.multiple_testing import benjamini_hochberg`) and not from the package root.
The root's bytes are recorded in the suite capacity receipts, which `production_suite.capacity`
re-checks against the live file, so exporting the name from `foundry.metrics` waits for a re-measured
receipt.

| Name | Does |
|---|---|
| `benjamini_hochberg(p_values, *, nan="raise")` | Benjamini-Hochberg adjusted p-values (q-values) as a float64 array in the input's order and shape: the running minimum, from the largest rank down, of `min(1, p * m / rank)`. Tied p-values share one value. An empty input returns an empty array. `nan="raise"` refuses an entry that is NaN or infinite; `nan="omit"` leaves it out of the tested count `m` and returns NaN there. A finite entry outside `[0, 1]` raises `ValueError` either way |

Any shape is one family of `m = size` tests. A discovery at level `alpha` is an adjusted value at most `alpha`.

Eleven copies of the adjustment lived in phenomics (four), serving_api (two), annotation_vocabulary
(two), atlas (two) and embedding_translation, and they differed in exactly the choices above: whether
they validated, how they treated NaN, what shape they took (a vector, a mapping, a list of
dictionaries), whether an empty family raised, and the evaluation order `(p * m) / rank` against
`p * (m / rank)` (serving_api's abca4 screen, which can differ in the last bit). The copies'
algorithms are held in the tests as reference functions and compared exactly. The step-up
selection that Protify's `select_answers` and an experiment script write (which rows pass at a
target) is a different function and stays with its consumer until a third needs it.

Tests: `tests/tier1_unit/test_metrics_multiple_testing.py`, NumPy only, and one comparison with
embedding_translation's torch copy that the tooling environment skips.

## DSM's classification and regression metrics

`foundry.metrics.classification` and `foundry.metrics.regression` hold the Hugging Face
`compute_metrics` functions DSM wrote, with which its `synthyra_api/affinity_pred.py` scores the
Synthyra API's predictions. egfr_binding_proxies copied that script and both modules byte for
byte. The package does not import these two, so `import foundry.metrics` still needs NumPy only.
They need scikit-learn, and the classification module torch and the regression module SciPy.

| Name | Does |
|---|---|
| `compute_metrics_single_label_classification(p)` | Weighted F1, precision, and recall, accuracy, MCC, and `auc`, each to 5 places, from `p.predictions` (its first element when it is a tuple) and `p.label_ids` (its second). Labels of -100 are left out, labels are cast to int32, and the confusion matrix is printed. One score per label is positive above 0.5; scores per class on the last axis give their arg max. A class never predicted scores 0 |
| `compute_metrics_regression(p)` | R², Spearman's and Pearson's rho with their p-values, MSE, MAE, and RMSE, to 5 places, from the flattened predictions and labels, with the labels as the truth |
| `get_residuals(preds, labels)` | The absolute differences of the flattened arrays |
| `spearman(probs, correct_preds)` | Prints Spearman's rho and its p-value |

`auc` does not rank the scores. It is -100 for one score per label. For scores per class it is the
ROC AUC of the arg-max classes: with two classes, that of the hard predictions, which scikit-learn
1.9 makes NaN, with a warning, when the labels left hold one class, as it does for a single class
of scores. With three or more classes, and wherever scikit-learn raises, it is -100. A consumer
that wants the AUC of the scores needs another function.

DSM's `metrics/classification.py` and `metrics/regression.py`, and egfr_binding_proxies' copies,
import these names from here. Each regression module still prints the metrics of random
predictions when run as a script. What changed besides the location:

- Importing either module no longer imports transformers, which they name only as the type of
  their argument, and so the regression module no longer brings in torch.
- Each project's module holds its own names alone. The originals also held what they imported,
  from `np` to `roc_auc_score`, and a star import took those too.
- A `KeyboardInterrupt` or other `BaseException` raised inside scikit-learn's AUC now propagates.
  The originals' bare `except` turned it into -100.

Wave 12 of [the unification plan](../../docs/decisions/workspace/2026-09-24_foundry_unification.md) compared
the functions with the originals.

Tests: `tests/tier1_unit/test_metrics_calibration.py` and `test_metrics_selective.py`, which need
NumPy only, and `test_metrics_classification.py` and `test_metrics_regression.py`, which need
scikit-learn and SciPy, and torch for classification, so the tooling environment skips them and
tier 1 on the research interpreter runs them.
