# Configuration

This page documents how configuration works: CLI argument groups, YAML config (base.yaml), merge behavior, and the distinction between `model_names` and `model_paths`/`model_types`. For a quick reference of where each option is used, see the argument tables below and [Getting started](getting_started.md).

---

## Overview

Configuration is a single namespace built by:

1. **Parsing CLI** with `parse_arguments()` in [main.py](../src/protify/main.py).
2. **Optionally loading a YAML file** when `--yaml_path` is set; the YAML is converted to a namespace and merged with the CLI result. **CLI overrides YAML** for any option that was explicitly set on the command line.
3. **Defaults** for ProteinGym, W&B sweep, and a few other options are filled when missing.

The same namespace (`full_args`) is used to build `DataArguments`, `BaseModelArguments`, `ProbeArguments`, `EmbeddingArguments`, and `TrainerArguments` inside `MainProcess.apply_current_settings()`.

---

## How it works

- **CLI-only:** Omit `--yaml_path`; every value comes from argparse defaults or from flags you pass.
- **YAML + CLI:** Pass `--yaml_path path/to/config.yaml`. The file is `yaml.safe_load`'ed; keys are merged into a namespace. Then CLI parsing runs; any CLI option overrides the YAML value. So you can override a few keys without editing the file (e.g. `--num_epochs 10`).
- **Store-true flags:** Merge logic treats store_true/store_false specially so that omitting a flag in YAML does not overwrite a CLI `--flag` or `--no-flag`.

The schema is defined by the union of [base.yaml](../src/protify/yamls/base.yaml) and all options in `parse_arguments()`. YAML can use type tags (e.g. `!!int`, `!!bool`) for clarity.

---

## CLI argument groups

### ID and API keys

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hf_username` | str | Synthyra | Hugging Face username. |
| `--hf_token` | str | None | Hugging Face token. |
| `--synthyra_api_key` | str | None | Synthyra API key for cloud GPU compute. Get yours at [synthyra.com](https://synthyra.com). |
| `--wandb_api_key` | str | None | Weights and Biases API key. |
| `--modal_token_id` | str | None | Modal token ID. |
| `--modal_token_secret` | str | None | Modal token secret. |
| `--modal_api_key` | str | None | Modal key as `token_id:token_secret`. |
| `--rebuild_modal` | flag | False | Force rebuild and deploy Modal backend before run. |
| `--delete_modal_embeddings` | flag | False | Delete embedding cache on Modal volume before submission. |

### Paths

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hf_home` | str | None | Custom Hugging Face cache directory. |
| `--yaml_path` | str | None | Path to YAML config file. |
| `--log_dir` | str | logs | Log directory. |
| `--results_dir` | str | results | Results directory. |
| `--model_save_dir` | str | weights | Directory to save models. |
| `--embedding_save_dir` | str | embeddings | Directory for embeddings. |
| `--download_dir` | str | Synthyra/vector_embeddings | Directory for downloaded embeddings. |
| `--plots_dir` | str | plots | Directory for plots. |
| `--replay_path` | str | None | Path to replay log file. |
| `--pretrained_probe_path` | str | None | Path to pretrained probe (reserved). |

### Data

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--delimiter` | str | , | Delimiter for CSV/TSV from data_dirs. |
| `--col_names` | list | [seqs, labels] | Column names (legacy; often inferred). |
| `--max_length` | int | 2048 | Maximum sequence length. |
| `--padding` | choice | max_length | Padding strategy: `max_length` pads all sequences to `--max_length` (recommended for torch.compile + flex attention); `longest` pads to the longest sequence in each batch. |
| `--trim` | flag | False | If set, drop sequences longer than max_length; else truncate. |
| `--data_names` | list | [] | Dataset names (HuggingFace or preset e.g. standard_benchmark). |
| `--data_dirs` | list | [] | Local directories with train.*/valid.*/test.*. |
| `--aa_to_dna`, `--aa_to_rna`, `--dna_to_aa`, `--rna_to_aa`, `--codon_to_aa`, `--aa_to_codon` | flag | False | Sequence translation (only one may be True). |
| `--random_pair_flipping` | flag | False | Random swap of paired inputs (e.g. PPI). |
| `--multi_column` | list | None | Sequence column names for multi-input tasks. |

### Base model

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_names` | list | None | Preset model names (e.g. ESM2-8). Mutually exclusive with model_paths/model_types. |
| `--model_paths` | list | None | Model paths (HF or local). Must pair with --model_types. |
| `--model_types` | list | None | Type keywords for each path (esm2, custom, etc.). |
| `--model_dtype` | choice | bf16 | fp32, fp16, bf16, float32, float16, bfloat16. |
| `--use_xformers` | flag | False | Use xformers attention for AMPLIFY. |

### Probe

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--probe_type` | choice | linear | linear, transformer, lyra. |
| `--tokenwise` | flag | False | Token-wise prediction. |
| `--hidden_size` | int | 8192 | Hidden size for linear probe MLP. |
| `--transformer_hidden_size` | int | 512 | Hidden size for transformer probe. |
| `--dropout` | float | 0.2 | Dropout rate. |
| `--n_layers` | int | 1 | Number of layers. |
| `--pre_ln` | flag | True | Pre-LayerNorm (store_false to disable). |
| `--classifier_size` | int | 4096 | Classifier feed-forward dimension. |
| `--transformer_dropout` | float | 0.1 | Transformer layer dropout. |
| `--classifier_dropout` | float | 0.2 | Classifier dropout. |
| `--n_heads` | int | 4 | Number of attention heads. |
| `--rotary` | flag | True | Use rotary embeddings (store_false to disable). |
| `--attention_backend` | choice | flex | kernels, flex, sdpa. |
| `--output_s_max` | flag | False | Return s_max from attention layers. |
| `--probe_pooling_types` | list | [mean, var] | Pooling types for probe. |
| `--use_bias` | flag | False | Use bias in Linear layers. |
| `--save_model` | flag | False | Save trained model. |
| `--push_raw_probe` | flag | False | With --save_model, push raw probe class to Hub (load with e.g. Class.from_pretrained(repo_id)) instead of packaged AutoModel. |
| `--production_model` | flag | False | Production model flag. |
| `--lora` | flag | False | Use LoRA. |
| `--lora_r` | int | 8 | LoRA rank. |
| `--lora_alpha` | float | 32.0 | LoRA alpha. |
| `--lora_dropout` | float | 0.01 | LoRA dropout. |
| `--sim_type` | choice | dot | dot, euclidean, cosine. |
| `--add_token_ids` | flag | False | Add token type embeddings for PPI. |

### Scikit

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--scikit_n_iter` | int | 10 | Iterations for scikit tuning. |
| `--scikit_cv` | int | 3 | Cross-validation folds. |
| `--scikit_random_state` | int | None | Random state (None uses global seed). |
| `--scikit_model_name` | str | None | Scikit model name. |
| `--scikit_model_args` | str | None | JSON hyperparameters (skips tuning). |
| `--use_scikit` | flag | False | Use scikit-learn path. |
| `--n_jobs` | int | 1 | Processes for scikit. |

### Embedding

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--embedding_batch_size` | int | 16 | Batch size for embedding. |
| `--embedding_num_workers` | int | 0 | DataLoader workers for embedding. |
| `--num_workers` | int | 0 | DataLoader workers for training. |
| `--download_embeddings` | flag | False | Download precomputed embeddings. |
| `--matrix_embed` | flag | False | Keep per-residue matrices (no pooling). |
| `--embedding_pooling_types` | list | [mean, var] | Pooling for vector embeddings. |
| `--save_embeddings` | flag | False | Save computed embeddings. |
| `--embed_dtype` | choice | None | fp32/fp16/bf16 for embeddings (default: model_dtype). |
| `--sql` | flag | False | Store embeddings in SQLite. |
| `--read_scaler` | int | 100 | Read scaler for SQL. |

### Trainer

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_epochs` | int | 200 | Training epochs. |
| `--probe_batch_size` | int | 64 | Probe batch size. |
| `--base_batch_size` | int | 4 | Base model batch size. |
| `--probe_grad_accum` | int | 1 | Gradient accumulation steps (probe). |
| `--base_grad_accum` | int | 8 | Gradient accumulation steps (base). |
| `--lr` | float | 1e-4 | Learning rate. |
| `--weight_decay` | float | 0.00 | Weight decay. |
| `--patience` | int | 1 | Early-stopping patience. |
| `--seed` | int | None | Random seed (None: time-based). |
| `--deterministic` | flag | False | Deterministic mode. |
| `--full_finetuning` | flag | False | Full model finetuning. |
| `--hybrid_probe` | flag | False | Hybrid probe then finetune. |
| `--num_runs` | int | 1 | Number of seeds; report mean and std. |

### Balanced regression metrics (EpHod-style)

Applied only for regression tasks. See [Probes and Training > Balanced regression metrics](probes_and_training.md#balanced-regression-metrics) for details.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--balanced_regression_metrics` / `--no_balanced_regression_metrics` | flag | True | Toggle computation. |
| `--balanced_weight_method` | choice | `bin_inv` | `none`, `bin_inv`, `bin_inv_sqrt`, `LDS_inv`, `LDS_inv_sqrt`, `LDS_extreme`. |
| `--balanced_bin_borders` | list[float] | None | Explicit bin borders (e.g. `5 9` for pH). None uses training-label tertiles. |
| `--balanced_n_resamples` | int | 100 | Weight-bootstrap draws for resampled Pearson/Spearman. |
| `--balanced_lds_bins` | int | 100 | LDS histogram bin count. |
| `--balanced_lds_ks` | int | 5 | LDS Gaussian kernel size. |
| `--balanced_lds_sigma` | float | 2.0 | LDS Gaussian sigma. |

### ProteinGym

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--proteingym` | flag | False | Run ProteinGym zero-shot. |
| `--dms_ids` | list | [all] | DMS assay IDs or "all". |
| `--mode` | choice | benchmark | benchmark, indels, multiples, singles. |
| `--scoring_method` | choice | masked_marginal | masked_marginal, mutant_marginal, wildtype_marginal, pll, global_log_prob. |
| `--scoring_window` | choice | optimal | optimal, sliding. |
| `--pg_batch_size` | int | 32 | Batch size for ProteinGym. |
| `--compare_scoring_methods` | flag | False | Compare scoring methods. |
| `--score_only` | flag | False | Skip scoring; run benchmark on existing CSVs. |

### W&B sweep

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_wandb_hyperopt` | flag | False | Run W&B hyperparameter sweep. |
| `--wandb_project` | str | (from env/args) | W&B project name. |
| `--wandb_entity` | str | (from env/args) | W&B entity. |
| `--sweep_config_path` | str | yamls/sweep.yaml | Sweep YAML path. |
| `--sweep_count` | int | 10 | Number of trials. |
| `--sweep_method` | choice | bayes | bayes, grid, random. |
| `--sweep_metric_cls` | str | eval_loss | Classification metric to optimize. |
| `--sweep_metric_reg` | str | eval_loss | Regression metric to optimize. |
| `--sweep_goal` | choice | minimize | maximize, minimize. |

---

## YAML config (base.yaml)

The file [src/protify/yamls/base.yaml](../src/protify/yamls/base.yaml) is organized by section. Key names match CLI long options (without the leading dashes). Types can be explicit with YAML tags (e.g. `!!int`, `!!bool`). Example structure:

```yaml
# ID
hf_username: Synthyra
hf_token: null
# ...

# Paths
log_dir: logs
results_dir: results
# ...

# DataArguments
delimiter: ','
max_length: 1024
data_names: [DeepLoc-2]
data_dirs: []
# ...

# BaseModelArguments
model_names: [ESM2-8]
# model_paths: [...]
# model_types: [...]

# ProbeArguments
probe_type: linear
tokenwise: false
# ...

# EmbeddingArguments, TrainerArguments, ScikitArguments, etc.
```

Anything you can set via CLI can be set in YAML; CLI overrides when both are present.

---

## model_names vs model_paths and model_types

- **model_names:** A list of preset names (e.g. `ESM2-8`, `ProtT5`, or `standard` to expand to a standard set). Resolved via [supported models](models_and_embeddings.md) and `get_base_model`/`get_tokenizer`. Use this for built-in HuggingFace models.
- **model_paths + model_types:** For custom or local models. `model_paths` is a list of paths (HuggingFace IDs or local dirs); `model_types` must be the same length and each element is a dispatch keyword (e.g. `esm2`, `custom`). You cannot mix: either set `model_names` or set both `model_paths` and `model_types`.

---

## Examples

### Probe-only with two models and one dataset

```bash
py -m src.protify.main --model_names ESM2-8 ESM2-35 --data_names DeepLoc-2 --num_epochs 5 --results_dir my_results
```

### ProteinGym zero-shot on all substitution assays

```bash
py -m src.protify.main --proteingym --model_names ESM2-150 --dms_ids all --mode benchmark
```

### W&B sweep (after setting W&B credentials)

```bash
py -m src.protify.main --yaml_path src/protify/yamls/base.yaml --use_wandb_hyperopt --sweep_count 5 --wandb_project my_project --wandb_entity my_entity
```

### YAML with CLI overrides

```bash
py -m src.protify.main --yaml_path my_config.yaml --num_epochs 20 --lr 5e-5 --save_embeddings
```

### Docker

Run from the repository root on your host with the workspace mounted and working directory set to `/workspace/src/protify`. Linux/mac example:

```bash
docker run --rm -it --gpus all -v "${PWD}":/workspace -w /workspace/src/protify protify-env:latest python -m main --model_names ESM2-8 ESM2-35 --data_names DeepLoc-2 --num_epochs 5 --results_dir my_results
```

ProteinGym zero-shot:

```bash
docker run --rm -it --gpus all -v "${PWD}":/workspace -w /workspace/src/protify protify-env:latest python -m main --proteingym --model_names ESM2-150 --dms_ids all --mode benchmark
```

YAML with overrides:

```bash
docker run --rm -it --gpus all -v "${PWD}":/workspace -w /workspace/src/protify protify-env:latest python -m main --yaml_path yamls/base.yaml --num_epochs 20 --lr 5e-5 --save_embeddings
```

On Windows use `-v "%CD%":/workspace` and `py -m main` instead of `python -m main`. Ensure the image is built first: `docker build -t protify-env:latest .`

---

## See also

- [Getting started](getting_started.md) for first runs
- [Data](data.md) for data options and supported datasets
- [Models and embeddings](models_and_embeddings.md) for base model and embedding options
- [Probes and training](probes_and_training.md) for probe and trainer options
- [Hyperparameter optimization](hyperparameter_optimization.md) for W&B sweep details
