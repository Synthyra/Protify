# Testing

This page describes the Protify testing suite: where tests live, what they cover, and how to run them.

---

## Overview

Tests are under `src/protify/testing_suite/`. They cover metrics computation, pooling strategies, probe construction and forward passes, model components (MLP, attention, rotary embeddings), loss functions, the feature-store embedding caches and the conversion of old ones, data utilities, seed reproducibility, lazy-predict integration, packaged probe export, and probe attention behavior. Run them with pytest from the repository root.

---

## Test layout

| File | Description |
|------|-------------|
| **conftest.py** | Shared fixtures (tiny embeddings, masks, label arrays) and marker registration (`gpu`, `slow`, `checkpoint`). |
| **test_package_imports.py** | Every module of the package imports in a fresh interpreter (the check that catches a vendored module a clone lacks), `import protify` loading no heavy library and leaving `sys.path` alone, and `pyproject.toml` declaring the package version. |
| **test_probe_registry.py** | `register_probe_type`: a registered head built, its settings stored for it alone, rebuilt from a saved config through `rebuild` or `build`; built-in names and a second definition of a registered name refused; `register_external_probe` under the same guards. |
| **test_fit_hooks.py** | `FitHooks` through a real CPU fit of a tiny probe: the hooks' order, the posterior fitted on the rows they give, the first evaluation reading validation when they reserve test, the defaults for a dataset without hooks, and the epoch length they set. |
| **test_trainer_arguments.py** | Phase learning rates, schedulers, and optimizers; probe controls effective without touching the base phase; invalid controls refused before training; the evaluation cadence; precision; and the parallel-probe settings. |
| **test_split_row_caps.py** | Split row caps as a function of a split's rows and `--split_row_cap_seed` alone: exact counts, class shares and regression deciles kept, a uniform draw where a label gives no stratum, row order kept, a split at or under its cap untouched, filtering before the cap, and dropped rows absent from `all_seqs`. |
| **test_embedding_gpu_smoke.py** | `gpu` and `slow`: real ESM2-8, ESMC-300, and ESMC-6B checkpoints embedded on a GPU into one feature store each. |
| **test_calm_checkpoint.py** | `checkpoint`: the pinned CaLM checkpoint reproducing its model card's published masked-codon probabilities, from the local Hugging Face cache. |
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
| **test_packaged_probe_export.py** | Linear and transformer probe save/load roundtrip via PackagedProbeModel, transformer and Lyra probe pooling kept through the AutoModel reload, PPI inference with/without token_type_ids, an export holding one module whose imports are absolute on the installed `protify`, and a fresh process reproducing a standardized probe's logits from the exported repository. |
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
| **test_probe_attention.py** | Transformer attention with 2D/4D masks, s_max output, `resolve_attention_backend`. |
| **test_pair_features.py** | The pair-row rule: `minmax` symmetry under a swap, `concat` asymmetry, width preservation, mode validation, and rejection of per-residue matrices. |
| **test_feature_matrix.py** | Dense against compressed stacking equivalence, layout pinning across splits, chunked compression, live-column detection including stored zeros, and column narrowing in both layouts. |
| **test_estimator_tuning.py** | Validation-only search discipline, the defaults enqueued as the first trial, metric direction, seed averaging, per-task search spaces, and the timeout guard. |
| **test_estimator_artifact.py** | Saved-probe round trip, plain-JSON recipe, pair joining at inference, and composition of two narrowing steps into one index over the original width. |
| **embedding_test.py** | Standalone embedding diagnostic script (excluded from pytest collection). |

---

## Constraining the container's CPUs

The canonical command passes no `--cpus`, and it should stay that way unless you also cap
torch. Torch sizes its intra-op thread pool from the host's core count and does not read a
container CPU limit, so `--cpus=12` on a 64-core host puts 64 threads on 12 cores. The
feature-path selector trains through torch autograd and is where that thrashing shows: the
same suite that finishes in about five minutes unconstrained got through about a third of its
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

- `@pytest.mark.gpu` -- requires a CUDA GPU; deselect it on a CPU-only machine.
- `@pytest.mark.slow` -- tests that take >10 seconds (e.g. loading multiple models).
- `@pytest.mark.checkpoint` -- loads a published checkpoint from the local Hugging Face cache, offline, and fails when it is not cached.

Nothing in the suite skips itself: a test that cannot run here is deselected by its marker, and one that is
selected and cannot run fails. There is no CI workflow: run the suite yourself before a pull request, with
`-m "not gpu and not slow and not checkpoint"` on a machine without a GPU.

Filter tests by marker:

```bash
# CPU-only tests, offline, with no GPU
python -m pytest src/protify/testing_suite -v -m "not gpu and not slow and not checkpoint"

# GPU tests only
python -m pytest src/protify/testing_suite -v -m "gpu"

# All tests
python -m pytest src/protify/testing_suite -v
```

---

## How to run

From the **repository root**, after `pip install -e .` (or with `src` and the root on `PYTHONPATH`):

```bash
python -m pytest src/protify/testing_suite -v
```

To run a single file:

```bash
python -m pytest src/protify/testing_suite/test_metrics.py -v
```

To run with coverage (if you have pytest-cov):

```bash
python -m pytest src/protify/testing_suite -v --cov=protify --cov-report=term-missing
```

Use `py` on Windows if `python` is not the interpreter you installed into. Install the dependencies first (`pip install -r requirements.txt`).

---

## Docker

Build the image first, then run pytest with the repository mounted:

**Linux / macOS:**
```bash
docker build -t protify-env:latest .
docker run --rm -v "${PWD}":/workspace -w /workspace protify-env:latest python -m pytest src/protify/testing_suite -v
```

**Windows:**
```bash
docker build -t protify-env:latest .
docker run --rm -v "%CD%":/workspace -w /workspace protify-env:latest python -m pytest src/protify/testing_suite -v
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

For reviews that touch vectorized seed-bank training, run the focused suite inside Docker:

```bash
docker run --rm --gpus all -v "${PWD}":/workspace -w /workspace protify-env:latest python -m pytest src/protify/testing_suite/test_parallel_probe_benchmark.py src/protify/testing_suite/test_parallel_probe_batches.py src/protify/testing_suite/test_parallel_probe_compare.py src/protify/testing_suite/test_parallel_probe_hardware_monitor.py src/protify/testing_suite/test_parallel_probe_launch_manifest_runner.py src/protify/testing_suite/test_parallel_probe_plan.py src/protify/testing_suite/test_parallel_linear_probe.py src/protify/testing_suite/test_parallel_probe_logger.py src/protify/testing_suite/test_parallel_probe_preflight.py src/protify/testing_suite/test_trainer_arguments.py -v
```

The current GH200 workstation validation for this focused suite passed with `180 passed`.

---

## See also

- [Getting started](getting_started.md) for installation and entry points
- [Probes and training](probes_and_training.md) for packaged probe export
- [Models and embeddings](models_and_embeddings.md) for embedder behavior
