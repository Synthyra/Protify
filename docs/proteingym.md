# ProteinGym

This page documents the ProteinGym zero-shot benchmarking flow: `ProteinGymRunner`, scoring methods, `run_proteingym_zero_shot()`, CLI arguments, `expand_dms_ids_all`, and the benchmark performance subprocess. For dataset-based probing, see [Probes and training](probes_and_training.md).

---

## Overview

ProteinGym provides zero-shot variant effect prediction: score substitution or indel assays with a base model (e.g. ESM2) and compare to experimental data via Spearman correlation. Protify runs scoring for selected DMS assay IDs and optionally runs the official benchmark performance script. You can compare multiple scoring methods in one run with `--compare_scoring_methods`.

---

## How it works

1. When `--proteingym` is set and no datasets are passed, `main()` calls `MainProcess.run_proteingym_zero_shot()`.
2. **run_proteingym_zero_shot()** reads `dms_ids`, `mode`, `model_names`, `scoring_method`, `scoring_window`, and `pg_batch_size` from `full_args`. It expands `dms_ids` (e.g. "all") via `expand_dms_ids_all()`, then instantiates `ProteinGymRunner(results_dir, repo_id="GleghornLab/ProteinGym_DMS")` and calls `runner.run(...)`.
3. For each model, the runner loads the base model (masked LM), builds `ProteinGymScorer`, loads each DMS assay with `load_proteingym_dms(dms_id, mode, repo_id)`, runs scoring (substitutions or indels), and saves/merges per-assay CSVs. Optionally it runs `run_benchmark()` to compute Spearman vs reference.
4. If `--compare_scoring_methods` is set (and `--proteingym`), the main entry runs `compare_scoring_methods(...)` instead of the normal zero-shot path; no `run_proteingym_zero_shot()` is called in that branch.

---

## ProteinGymRunner

Defined in [scorer.py](../src/protify/benchmarks/proteingym/scorer.py).

- **Constructor:** `ProteinGymRunner(results_dir, repo_id="GleghornLab/ProteinGym_DMS", device=None)`. Creates `results_dir`; uses CUDA if available.
- **run(dms_ids, model_names, mode="benchmark", scoring_method="masked_marginal", scoring_window="optimal", batch_size=32):** For each model, loads via `get_base_model(..., masked_lm=True)`, builds `ProteinGymScorer`, then for each DMS ID loads data with `load_proteingym_dms(dms_id, mode, repo_id)`. For substitutions: `scorer.score_substitutions(..., scoring_method, scoring_window)`; for indels: `scorer.score_indels(..., scoring_window="sliding")`. Saves/merges CSVs via `_save_results()`. Returns a dict `model_name -> elapsed_time`.
- **run_benchmark(model_names, dms_ids, mode, scoring_method):** Runs the `DMS_benchmark_performance.py` subprocess; outputs go to `results_dir/benchmark_performance`.
- **collect_spearman(results_dir, model_names):** Static method; reads `benchmark_performance/Spearman/Summary_performance_DMS_*_Spearman.csv` and returns `{model_name: spearman}`.

---

## Scoring methods

For **substitutions** (`ProteinGymScorer.score_substitutions`):

| Method | Description |
|--------|-------------|
| masked_marginal | Marginal log prob of the mutated residue in masked context (E1 uses a dedicated path). |
| mutant_marginal | Marginal at the mutated position for the mutant sequence. |
| wildtype_marginal | Marginal at the position for the wildtype sequence. |
| pll | Pseudo-log-likelihood. |
| global_log_prob | Full sequence log probability. |

For **indels**, only PLL over sliding windows is supported; `scoring_window` is typically `"sliding"`.

**scoring_window:** `"optimal"` (single window per variant when possible) or `"sliding"` (multiple windows, then aggregate). For indels, `scoring_window` is forced to `"sliding"` in the run.

---

## Data loading

- **load_proteingym_dms(dms_id, mode, repo_id="GleghornLab/ProteinGym_DMS")** ([data_loader.py](../src/protify/benchmarks/proteingym/data_loader.py)) downloads `by_dms_id/{dms_id}.parquet` from HuggingFace.
- **Modes:** `"benchmark"` (substitutions, no indels), `"indels"`, `"singles"`, `"multiples"`.
- **expand_dms_ids_all(dms_ids, mode)** ([dms_ids.py](../src/protify/benchmarks/proteingym/dms_ids.py)): If any element is `"all"`, replaces with `ALL_INDEL_DMS_IDS` or `ALL_SUBSTITUTION_DMS_IDS` from [dms_ids.py](../src/protify/benchmarks/proteingym/dms_ids.py).

---

## CLI arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--proteingym` | flag | False | Enable ProteinGym zero-shot run. |
| `--dms_ids` | list | [all] | DMS assay IDs or "all". |
| `--mode` | choice | benchmark | benchmark, indels, multiples, singles. |
| `--scoring_method` | choice | masked_marginal | masked_marginal, mutant_marginal, wildtype_marginal, pll, global_log_prob. |
| `--scoring_window` | choice | optimal | optimal, sliding. |
| `--pg_batch_size` | int | 32 | Batch size for scoring. |
| `--compare_scoring_methods` | flag | False | Run scoring method comparison instead of single method. |
| `--score_only` | flag | False | Skip scoring; run benchmark on existing CSVs only. |

When `--proteingym` is True and `--compare_scoring_methods` is True, the main entry runs only the comparison and exits; otherwise it runs `run_proteingym_zero_shot()` (and for indels mode, `scoring_method` is forced to `pll`).

---

## Examples

### Zero-shot on all substitution assays with one model

```bash
py -m src.protify.main --proteingym --model_names ESM2-150 --dms_ids all --mode benchmark
```

### Indels with sliding window

```bash
py -m src.protify.main --proteingym --model_names ESM2-150 --dms_ids all --mode indels
```

### Compare scoring methods

```bash
py -m src.protify.main --proteingym --compare_scoring_methods --model_names ESM2-150 ESM2-650 --dms_ids all
```

### Score only (no benchmark script)

Use `--score_only` after you have already produced the per-assay CSVs and only want to run the benchmark performance step.

---

## See also

- [Configuration](cli_and_config.md) for ProteinGym CLI flags
- [Models and embeddings](models_and_embeddings.md) for base model loading
- [Getting started](getting_started.md) for entry points
