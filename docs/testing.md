# Testing

This page describes the Protify testing suite: where tests live, what they cover, and how to run them.

---

## Overview

Tests are under `src/protify/testing_suite/`. They cover metrics computation, pooling strategies, probe construction and forward passes, model components (MLP, attention, rotary embeddings), loss functions, the feature-store embedding caches and the conversion of old ones, data utilities, seed reproducibility, lazy-predict integration, packaged probe export, and probe attention behavior. Run them with pytest from the repository root.

---

## Test layout

| File | Description |
|------|-------------|
| **conftest.py** | Shared fixtures (tiny embeddings, masks, label arrays) and marker registration (`gpu`, `slow`). |
| **test_metrics.py** | Metric computation: softmax, scorers, threshold optimization, single/multi-label classification, regression, tokenwise, ROC/PR AUC, `get_compute_metrics` dispatch. |
| **test_pooler.py** | All pooling strategies (mean, max, norm, median, std, var, cls, eos, parti/PageRank), mask layouts, explicit boundary validation, and concatenation behavior. |
| **test_carbon.py** | CARBON and transitive Qwen source pinning, audited tokenizer parity, partial 6-mers, exact and 8-aligned padding, cache identity, checkpoint-vocabulary agreement, raw-sequence budgets across mixed model selections, LoRA target scoping, translation-aware length enforcement, lowercase DNA, GUI exposure, hidden-state selection, and embedding/training dispatch. |
| **test_probe_construction.py** | `get_probe` factory dispatch for all probe_type x tokenwise combinations, forward passes (singlelabel, regression, multilabel, sigmoid_regression), loss computation, config roundtrip. |
| **test_probe_pooling.py** | `--probe_pooling_types` reaching the transformer and Lyra probes, sequence and token; each probe's default when unset; pooling kept through `from_pretrained` and `rebuild_probe_from_saved_config`; `bom` in the transformer sequence probe; Lyra's one-type check; sweeps, including that a sweep offers each probe type only pooling it takes, and the transformer probe only hidden sizes its heads divide; the command line, YAML, and cloud configs. |
| **test_model_components.py** | `intermediate_correction_fn`, SwiGLU, `swiglu_ln_ffn`, RotaryEmbedding, MultiHeadAttention shapes and values. |
| **test_losses.py** | `get_loss_fct` dispatch, SoftBCELoss, SoftBCEWithLogitsLoss (forward, ignore_index, smooth_factor). |
| **test_blob_serialization.py** | The dense blob of the old SQLite caches, decoded by `legacy_cache.embedding_blob_to_tensor`: float32/float16/bfloat16, rank, the first bfloat16 encoding (code 1), `blob_dtype`, and the legacy raw and `torch.save` fallbacks. Every blob is written by the test-local writer in `legacy_cache_writer.py`, and a test checks that `legacy_cache` has no blob writers. |
| **test_sparse_blob_serialization.py** | The coordinate-list blob of the old SQLite caches: uint16 and uint32 index widths at the 65,536 boundary, exact nonzeros, negative zero, and both bfloat16 codes. |
| **test_feature_cache.py** | The feature store as Protify keys it: pinned directory names, the descriptor's fields and their effect on the key, layouts, `CacheWriter` windows and restarts, widening into a wider store and never narrowing, sparse and per-token round trips, reader errors, and the store fingerprint. |
| **test_legacy_cache.py** | Conversion of each old format into a store: a cache built inside the test with the test-local writer, converted with `convert_rows`, and read back bit for bit for dense, sparse, matrix, float16, bfloat16, float32, `.db` and `.pth`; row order, refusals, `--skip_existing`, and the command line. |
| **test_cache_reader.py** | `cached_rows` over a store and over both old formats: rank and dtype kept, missing and duplicate rows, read-only access, and closed-reader behavior. |
| **test_embedder_store.py** | The `Embedder` against a store, with stand-in models and no weights: reading, adopting an old `.db` or `.pth` cache, converting a download, committing one window at a time, resuming an interrupted run, matrix rows, and widening. |
| **test_data_mixin_store.py** | `DataMixin`'s store lookup, in-memory read, missing-store errors that name an old cache to convert, and the streamed width. |
| **test_reduce_dim_store.py** | Dimensionality-reduction plots reading a complete store, a per-token store averaged over tokens, and an old cache converted first. |
| **test_run_tracking.py** | W&B runs through `foundry.logging`: one run per `--report_to wandb` training job, the curve charted by its first Trainer only, multi-seed means as the summary, no run inside an open sweep run, and the sweep's Optuna draws and grid refusal. |
| **test_data_utils.py** | Amino acid, DNA, RNA, codon constant validation (`data/sequence_alphabets.py`), translation mapping coverage, `pad_and_concatenate_dimer` (`data/data_collators.py`). |
| **test_seed_utils.py** | `set_global_seed` / `get_global_seed`, reproducibility (torch, numpy, random), `seed_worker`, `dataloader_generator`. |
| **test_lazy_predict.py** | LazyClassifier and LazyRegressor fit, models, predictions, provide_models. |
| **test_packaged_probe_export.py** | Linear and transformer probe save/load roundtrip via PackagedProbeModel, transformer and Lyra probe pooling kept through the AutoModel reload, PPI inference with/without token_type_ids, and an exported runtime that builds the transformer stack from the `interpnet` it carries. |
| **test_parallel_linear_probe.py** | Vectorized multi-seed MLP probe behavior, exported single-run parity, shared and run-specific batches, per-run metrics, ensemble metrics, gradient clipping, chunking, and budget-derived group sizing. |
| **test_parallel_probe_batches.py** | `ParallelRunDataset` deterministic per-run indexing, affine index strategy, tensor-cache fast paths, and collator shapes. |
| **test_parallel_probe_plan.py** | Static grouping and resource estimates for model/dataset/probe/seed universes. |
| **test_parallel_probe_preflight.py** | No-training launch-manifest generation, embedding prerequisites, execution waves, GPU placement metadata, validation templates, and helper-only `--num_labels` behavior. |
| **test_parallel_probe_compare.py** | Sequential-vs-parallel result comparison, metric parity, manifest coverage, telemetry coverage, runner-report gates, and validation verdicts. |
| **test_parallel_probe_benchmark.py** | Synthetic benchmark argument validation, plan summaries, run-specific loaders, and index-memory estimates. |
| **test_parallel_probe_hardware_monitor.py** | `nvidia-smi` monitor command construction, JSONL summary parsing, and dry-run wrapper behavior. |
| **test_parallel_probe_launch_manifest_runner.py** | Launch-manifest dry runs, execution planning, resume skips, wave concurrency, and over-budget execution gates. |
| **test_parallel_probe_logger.py** | Result TSV preservation for parallel-probe timing and identity fields. |
| **test_control_bases.py** | The controls generated at load: a row is a deterministic function of the sequence, token rows are `(l + 2, w)` with CLS and EOS, the pooled mean and variance equal those of the rows, a descriptor or cache of a control is refused, and nothing is written to disk. |
| **test_probe_features.py** | A cell's branches read from the engine's canonical store (a tiny one the engine's taps fill on CPU) or a generated control: each branch spec's stream and columns, the SAE max joined beside the final mean and variance, `mean_final` as the first half of the mean and variance, row order with duplicates, a missing stream or codebook refused, and the provenance a package records. |
| **test_probe_cell.py** | A whole cell trained and packaged from a canonical store or a generated control, both arms: package `features.json` source and backbones, a random-weight store recording its seed and no checkpoint, and an arm asking a store for a stream it lacks refused before training. |
| **test_probe_attention.py** | Transformer attention with 2D/4D masks, s_max output, `resolve_attention_backend`. |
| **test_pair_features.py** | The pair-row rule: `minmax` symmetry under a swap, `concat` asymmetry, width preservation, mode validation, and rejection of per-residue matrices. |
| **test_feature_matrix.py** | Dense against compressed stacking equivalence, layout pinning across splits, chunked compression, live-column detection including stored zeros, and column narrowing in both layouts. |
| **test_estimator_tuning.py** | Validation-only search discipline, the defaults enqueued as the first trial, metric direction, seed averaging, per-task search spaces, and the timeout guard. |
| **test_estimator_artifact.py** | Saved-probe round trip, plain-JSON recipe, pair joining at inference, and composition of two narrowing steps into one index over the original width. |
| **test_precompute_sae_and_dense.py** | One-pass SAE and dense caches: store keys identical to the separate arms', rows the ordinary reader accepts, per-cache resumption, `--data_revisions` reaching the loader one for one, and `--dna_to_aa` reaching it as `main.py` sets it. |
| **test_precompute_sae_multi_config.py** | A grid of SAE sparsities and codebook widths at one layer, with dense caches at that layer and at the final state, filled by exactly one backbone forward per batch; every store keyed as its single-configuration run keys it, shards merged by copying segments, each dense cache pooled at its own depth, resumption per cache, and each SAE cache's own storage decision. |
| **test_split_row_caps.py** | Split row caps as a function of a split's rows and `--split_row_cap_seed` alone: exact counts, class shares and regression deciles kept, a uniform draw where a label gives no stratum, row order kept, a split at or under its cap untouched, filtering before the cap, and dropped rows absent from `all_seqs`. |
| **test_projected_probe.py** | `ProjectedProbe` shapes for one and two branches, single rows and pairs; a branch's projection untouched by another branch's columns; identical trunk and output parameters across arms; config rebuild and `from_pretrained` with the posterior; `main.py` taking `--probe_type projected` and `sngp` with `--projection_size` and `--spectral_bound`, on pooled vectors. |
| **test_sngp.py** | The spectral bound holding within power-iteration error, variance growing away from the training clusters, MAP logits before `fit_posterior` and mean-field after, the posterior surviving a state-dict reload, the regression noise and prior, and MC dropout leaving the power iteration alone. |
| **test_default_branches.py** | `default_branch_specs` names each ESMC base's depth-layer SAE (k 64, codebook 16384), max-pooled, and its final hidden state's mean and variance, both parsing as ordinary branch specs and naming Biohub's repositories; any other backbone is refused. |
| **test_default_probe.py** | `main.py`'s default probe: allowed only when the run names no probe type, pooling, hidden state, or other workflow on the command line or in YAML and has not passed `--no_default_probe`; used only when every model is an ESMC base; and `--model_names ESMC-300` alone, over stores written under the two default branches' keys, training `projected` heads over the joined branches for each of `num_runs` seeds, at the requested projection size, reported under the joined name. |
| **test_branch_features.py** | Branch spec parsing and refusals, store keys equal to single-model runs', the join in spec order with SAE branches kept sparse, per-batch densification of single rows and pairs, training-only pair flips, and the blockwise standardizer. |
| **test_uncertainty_methods.py** | V_fit / V_cal carves (disjoint, complete, stratified, run-seed independent) and explicit rows, temperature and variance calibration recovering known values, conformal coverage on exchangeable rows, identity bin boundaries, and per-seed and per-bin scoring. |
| **test_uncertainty_workflow.py** | A cell end to end on pre-written caches with no model loaded: report fields, seed files, split digests, temperature fitted on V_fit rows and conformal quantiles on V_cal rows only for both role sources, the identity file, and a rerun that trains nothing. |
| **test_uncertainty_compare.py** | The fast MCC equal to scikit-learn's, the replicate F1 equal to Protify's, mirrored paired intervals and the full set of output files, refusal of cells scored on different rows, tuning cells kept out by `*/*/final/report.json` and refused without it, an arm and an SNGP head better by construction found better, recommended, and adopted end to end, scheme T covering where scheme V calibrated on easier rows does not, output that depends on the seed alone, the same bytes from one and two worker processes (`slow`), and the `compare` command line with its pattern, repeats, workers, margin, and cost order. |
| **test_uncertainty_verdicts.py** | The decision rules on synthetic draws: the verdict's precedence, an arm better by construction recommended, near-identical arms equivalent with the cheaper one recommended and the cost order a parameter, `both` paired with the single arm better on V_fit rather than on test, controls that reach a real arm listed and failing, unmeasured controls failing, the cheapest improving uncertainty candidate adopted with the arms that differ named, and split conformal's finite-sample band. |
| **test_uncertainty_exchangeable.py** | Scheme T's batched scoring equal, on a few repeats, to `build_methods` and `score_method` on the same rows for binary, multiclass, multilabel, and regression cells and both heads; its batched calibration maps fitting at least as well as the unchanged ones; a float32 probability of 1.0 leaving every score finite and equal to `score_method`'s; and Mondrian identity bins merged until each holds the minimum. |
| **test_uncertainty_package.py** | `save_heads`: the same seed files with it off, every head and the standardizer with it on, and a resume refused while a finished seed lacks its head or the refit standardizer differs; every shippable method of binary and regression cells with both heads packaged, loaded, and predicting its `metrics.json` and `report.json` scores from the test rows' features; the shipped seed chosen on V_fit, never test; quantiles reproducing `conformal_sets`' coverage; refusal of score-only methods, headless cells, and levels V_cal cannot calibrate; the `package` command's files; and pair rows joined by the cell's rule. |
| **test_single_probe_heads.py** | The single-probe heads and methods: the linear head as one Linear layer on the clamped input; `faithful`'s mean gradient equal to squared error's; `beta_nll` at beta 0 equal to the Gaussian NLL; quantile-clip bounds from training rows alone; the regression Laplace variance equal to the closed-form Bayesian linear regression one, and each task's growing away from the training data; stratified, seeded, covering folds, and no fold head training on a row it scores; per-row losses equal to the training losses; the new seed-file fields round-tripping while older files load; the p-norm order and every new method fitted on V_fit alone; scheme T refitting each new method's choice and matching `score_method` on binary, multiclass, multilabel, and regression cells; and the cell settings a head cannot honor refused. |
| **test_single_probe_verdicts.py** | The single-probe decision rules on constructed scheme T series: an improving candidate in the cheapest tier adopted over a better one in a dearer tier, the p-norm score improving on the primary and AURC conditions alone with its share of the ensemble's gain, the fallback's head following the head verdict, the quantile clip replacing the clamp only when better and worse on nothing, and the JSON and Markdown outputs. |
| **test_single_probe_workflow.py** | Every single-probe head end to end through `run_uncertainty_cell` on pre-written caches: `projected` and `linear` with the Laplace variance, a two-fold loss predictor, the p-norm score, and weight decay; `faithful` and `beta_nll` on regression; `projected_qclip` with bounds equal to the standardized training rows' quantiles; no fold head trained on a row it scored; the packaging refusals; `compare` writing `single_probe.json` with every check passing; and a regression-only head refused on a classification dataset. |
| **embedding_test.py** | Standalone embedding diagnostic script (excluded from pytest collection). |

---

## Constraining the container's CPUs

The canonical command passes no `--cpus`, and it should stay that way unless you also cap
torch. Torch sizes its intra-op thread pool from the host's core count and does not read a
container CPU limit, so `--cpus=12` on a 64-core host puts 64 threads on 12 cores. The
feature-path selector trains through torch autograd and is where that thrashing shows: the
same suite that finishes in about five minutes unconstrained reached 223 of roughly 690
tests in 41 minutes under `--cpus=12`, with no failures, simply not progressing.

If you do need to share the machine, cap torch to match:

```powershell
docker run --rm --ipc=host --cpus=12 -e OMP_NUM_THREADS=12 `
  -v ${PWD}:/workspace -w /workspace protify-env:latest `
  python -m pytest src/protify/testing_suite -v -m "not gpu and not slow"
```

---

## Markers

Tests use pytest markers registered in `conftest.py`:

- `@pytest.mark.gpu` -- requires a CUDA GPU. Skipped by default in CPU-only environments.
- `@pytest.mark.slow` -- tests that take >10 seconds (e.g. loading multiple models).

Filter tests by marker:

```bash
# CPU-only tests (skip gpu and slow)
py -m pytest src/protify/testing_suite -v -m "not gpu and not slow"

# GPU tests only
py -m pytest src/protify/testing_suite -v -m "gpu"

# All tests
py -m pytest src/protify/testing_suite -v
```

---

## How to run

From the **repository root** (so that `src` is on the path):

```bash
py -m pytest src/protify/testing_suite -v
```

To run a single file:

```bash
py -m pytest src/protify/testing_suite/test_metrics.py -v
```

To run with coverage (if you have pytest-cov):

```bash
py -m pytest src/protify/testing_suite -v --cov=src.protify --cov-report=term-missing
```

On Windows use `py`; on Linux/mac you can use `python` if preferred. Ensure the project dependencies are installed (`pip install -r requirements.txt`).

---

## Docker

Build the image first, then run pytest with the workspace mounted:

**Linux / macOS:**
```bash
docker build -t protify-env:latest .
docker run --rm -v "${PWD}":/workspace -w /workspace protify-env:latest python -m pytest src/protify/testing_suite -v
```

**Windows:**
```bash
docker build -t protify-env:latest .
docker run --rm -v "%CD%":/workspace -w /workspace protify-env:latest py -m pytest src/protify/testing_suite -v
```

For GPU-dependent tests, add `--gpus all` to the run command:

```bash
docker run --rm --gpus all -v "${PWD}":/workspace -w /workspace protify-env:latest python -m pytest src/protify/testing_suite -v
```

To run only CPU tests in Docker:

```bash
docker run --rm -v "${PWD}":/workspace -w /workspace protify-env:latest python -m pytest src/protify/testing_suite -v -m "not gpu and not slow"
```

### Parallel-probe focused suite

For reviews that touch vectorized seed-bank training, run the focused suite from `src/protify` inside Docker:

```bash
docker run --rm --gpus all -v "${PWD}":/workspace -e PYTHONPATH=/workspace -w /workspace/src/protify protify-env:latest python -m pytest testing_suite/test_parallel_probe_benchmark.py testing_suite/test_parallel_probe_batches.py testing_suite/test_parallel_probe_compare.py testing_suite/test_parallel_probe_hardware_monitor.py testing_suite/test_parallel_probe_launch_manifest_runner.py testing_suite/test_parallel_probe_plan.py testing_suite/test_parallel_linear_probe.py testing_suite/test_parallel_probe_logger.py testing_suite/test_parallel_probe_preflight.py testing_suite/test_trainer_arguments.py -v
```

The current GH200 workstation validation for this focused suite passed with `180 passed`.

---

## See also

- [Getting started](getting_started.md) for installation and entry points
- [Probes and training](probes_and_training.md) for packaged probe export
- [Models and embeddings](models_and_embeddings.md) for embedder behavior
