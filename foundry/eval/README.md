---
id: foundry.eval
kind: library_module
status: active
aliases: [foundry.eval]
---
# foundry.eval

Procedures that turn held-out predictions into calibrated probabilities, intervals with a
coverage guarantee, and bootstrap replicates, and that score labeled pairs by one all-vs-all pass.

| Name | Does |
|---|---|
| `fit_temperature(logits, labels, *, multiclass=False, label_mask=None, batch_elements=1048576, release_batch=None)` | The temperature `T` minimizing observed-label NLL of `logits / T`: binary cross-entropy for binary or multilabel logits, cross-entropy for multiclass `(n, c)` logits with integer labels. L-BFGS on `log T` in float64, from `T = 1`, with bounded loss/gradient blocks |
| `split_conformal(calibration_mean, calibration_truth, calibration_spread, test_mean, test_spread, alpha=0.1, ...)` | Locally weighted split conformal `Interval`s. Calibration residuals divided by the floored spread are the scores, and each test half-width is their conformal quantile times that row's floored spread. On exchangeable rows each interval holds the truth with probability at least `1 - alpha`, marginally |
| `conformal_quantile(scores, alpha, weights=None)` | The `ceil((n + 1)(1 - alpha))`-th order statistic, infinite when that rank exceeds `n`. Weighted, the smallest score whose normalized cumulative weight reaches `1 - alpha` |
| `calibration_size_for_alpha(alpha)` | `ceil(1 / alpha) - 1`, the fewest calibration rows whose quantile is finite |
| `Interval`, `empirical_coverage(interval, truth)` | Bounds, width, and which rows they cover; the fraction covered |
| `gather_pair_scores(block_scores, left, right, *, n_left, n_right, block_elements=1 << 27)` | One float32 array per score, `(n_pairs,)` in pair order, read out of the all-vs-all score matrices. `block_scores(start, stop)` returns the `(stop - start, n_right)` rows of each matrix for left proteins `start` to `stop`; the left proteins go through in blocks of at most `block_elements` cells, and each block gives up only the pairs whose left protein it holds, so the matrix is never held whole. Blocks no pair touches are skipped |
| `dual_tower_blocks(query, key, right_query=None, right_key=None)` | `block_scores` for a two-tower retrieval model: each block returns the forward score `query_i . key_j` and the symmetric score, the mean of it and `key_i . query_j`. Float32 without TF32, inside autocast or not |
| `stratified_bootstrap_indices(labels, rng)` | One replicate's row indices: each label's rows resampled with replacement to its own count, in `np.unique` order. Scoring several models on the same indices gives a paired bootstrap |

`fit_temperature`, `gather_pair_scores` and `dual_tower_blocks` need torch and load it on first use. Everything else needs NumPy alone.

`fit_temperature` preserves the objective and optimizer of sae_xgboost_ppi's historical
`gp6b.fit_temperature`. It now accumulates loss and gradient in blocks, with one block's
observed count divided by the complete observed count. Unequal blocks therefore retain
the same entry-weighted objective. The default limit is 1,048,576 logits per block, with
one complete row as the minimum for wide inputs. The loss graph is released after each
block; float64 conversion and mask selection never cover the full matrix unless it fits
in one block. One-block inputs retain historical arithmetic, while multiple blocks may
differ by floating-point reduction roundoff.

`label_mask` must be boolean and match the labels exactly. It selects entries for binary
or multilabel inputs and rows for multiclass inputs. Observed labels and logits must be
finite. Unknown fillers may be nonfinite and never enter the objective; no observed targets
is an error. The fit still weights entries equally, without introducing cluster weights or
changing which held-out partition a caller selects.

An owner of read-only mapped inputs can pass `release_batch` to release clean pages after
each block, including validation and empty blocks. The callback must leave every input
unchanged for the repeated optimizer passes. Without such a callback, working arrays are
bounded but mapped pages may remain resident. Protify supplies its owned-mapping release
function, with its platform limitations documented under
[bounded temperature fitting](../../projects/tooling/property_lowcode/README.md#bounded-temperature-fitting).

`stratified_bootstrap_indices` generalizes its `gp_analysis._replicate` from binary to any discrete
labels and draws the same replicate from the same seed. The conformal names come from
enzyme_loop's `uncertainty/conformal.py`, which now imports them and keeps its feedback-shift
weights and simultaneous level. Protify's uncertainty workflow is the second consumer of each.

sae_xgboost_ppi still keeps its private copies, the ones its reported numbers came from. Its
Modal image carries foundry now, so nothing blocks the switch. Its suite holds the copies equal
to these until it switches. A consumer lists `foundry.eval` under
`github.includes`, so `ws publish` vendors it into the projection.

One docstring changed in the move. enzyme_loop's said the weighted quantile includes the test
row's own weight, which the code never did: the weights are normalized over the calibration rows
alone, so equal weights give the empirical quantile, which can sit one order statistic below the
unweighted one. The docstring here describes the code, and the code is unchanged.

Every Atlas training and evaluation scores pairs the way `gather_pair_scores` does: encode each
distinct protein once, score every left protein against every right protein as a matrix product, and
read the labeled pairs out of it (Logan, 2026-09-30). The canonical evaluation suite, `fused_ppi` and
the exploratory AtlasV2 evaluators are its consumers, and none loops over pairs. The cost is
`U_left * U_right` scores however many pairs name the same proteins, and the canonical trainer's own
fast evaluation builds the same matrix whole for its small validation and test splits.

Tests: `tests/tier1_unit/test_eval_pair_scores.py` and `tests/tier1_unit/test_eval_temperature.py`, which need torch, so the tooling environment
skips them; `test_eval_conformal.py` and `test_eval_bootstrap.py`. The conformal code is also
guarded by enzyme_loop's `tests/test_uncertainty.py`, which reaches it through that project's own
module.
