# Hyperparameter Optimization

This page documents W&B (Weights and Biases) hyperparameter sweeps in Protify: sweep config (sweep.yaml), `HyperoptModule`, `apply_config` and `train_model`, the `run_wandb_hyperopt` flow, and the CLI arguments. For probe and trainer options that can be swept, see [Probes and training](probes_and_training.md) and [Configuration](cli_and_config.md).

---

## Overview

When `--use_wandb_hyperopt` is set, the pipeline searches the parameters defined in the sweep YAML. For each (model, dataset) combination (skipping random/onehot), it runs multiple trials, selects the best config by the chosen metric, writes a CSV of results, and then runs one final training with the best config.

The search runs in process with Optuna, and each (model, dataset) sweep is one W&B run: its trials are rows of the `tuning/trials` table, updated after every trial, and the final fit charts its `train/` and `val/` curves and its `val/` and `test/` metrics. A W&B sweep made one run per trial, which the workspace's W&B convention retired because the run list buried the runs worth reading.

---

## How it works

1. **Sweep config** is loaded from `full_args.sweep_config_path` (default `yamls/sweep.yaml`). Required: `parameters` (W&B format). Optional: `early_terminate` (e.g. hyperband).
2. Parameters are filtered by probe type and LoRA: only keys in `mlp_probe_params` or `transformer_probe_params` (and optionally `lora_params`) are passed to the sweep; the rest are ignored for that run. The `probe_pooling_types` values are then narrowed to those the probe type takes: Lyra takes one pooling type, and only the transformer sequence probe takes `bom`. A sweep that can land on the transformer probe offers only hidden sizes that every head size it can draw divides: a `values` list keeps its multiples, and a `min`/`max` range becomes a `q_uniform` or `q_log_uniform_values` range over them. A sweep over `probe_type` keeps only values that every listed probe type takes. A parameter left with no value is not swept, so its configured value stays.
3. For each (model_name, data_name), embeddings are loaded, `num_labels` and `task_type` are set, and a **HyperoptModule** is created with the sweep config and a shared `results_list`.
4. **The search** is an Optuna study from `sweep_search(sweep, seed)`: TPE for `bayes`, a random sampler for `random`, and a grid for `grid`, which needs every parameter to list its `values`. `suggest` draws each W&B-format entry (`value`, `values`, `uniform`, `log_uniform_values`, `int_uniform`, `q_uniform`, `q_log_uniform_values`). `early_terminate` is not applied: a trial reports its metrics once, when it ends, so the W&B sweep never applied it either. One W&B run opens for the sweep, through `run_tracking.training_job`, before `sweep_count` trials run.
5. **HyperoptModule.objective(trial)** draws the trial's configuration, `apply_config`s it, optionally reloads embeddings if `embedding_pooling_types` or `embedding_hidden_state_index` changed, runs `train_model(sweep_mode=True)` with the run's curve closed, selects the metric via `sweep_metric_cls` or `sweep_metric_reg` by task_type, appends to results_list, resends the trial table, and returns the metric value.
6. After the search, results_list is sorted by the chosen metric (max or min per `sweep_goal`), and a CSV is written to log_dir: `{random_id}_sweep_{data_name}_{model_name}.csv` with columns rank, trial, metric, config, valid_metrics, test_metrics.
7. Best config is applied via `apply_config(best_config)`; `make_plots=True`; one **final** training run with the best config is executed and charted in the sweep's run, whose summary takes its final metrics.

---

## HyperoptModule

Defined in [hyperopt_utils.py](../src/protify/hyperopt_utils.py).

- **Constructor:** `HyperoptModule(main_process, model_name, data_name, dataset, emb_dict, sweep_config, results_list, swept_param_keys=None, parameters=None)`; `parameters` are the narrowed W&B-format entries each trial draws from. Keeps deep copies of base `probe_args` and `trainer_args` and defines which keys are probe/trainer/embedding and which are int-cast.
- **apply_config(cfg):** Restores base probe and trainer args, then applies `cfg`: legacy `n_heads` (deprecated) is migrated to `head_size = hidden_size // n_heads` with a `DeprecationWarning`; asserts `hidden_size % head_size == 0` when the trial's probe type is `transformer`, the one probe with attention heads; ints for int_keys; `transformer_dropout` from `dropout`; `pooling_types` from `probe_pooling_types`; sets attributes on `mp.probe_args`, `mp.trainer_args`, and embedding options such as `mp.embedding_args.pooling_types` and `mp.embedding_args.hidden_state_index` where applicable.
- **train_model(sweep_mode=True):** Dispatches to `_run_full_finetuning`, `_run_hybrid_probe`, or `_run_nn_probe` depending on `full_args`; returns (model/probe, valid_metrics, test_metrics).
- **select_metric(valid_metrics, test_metrics, sweep_metric):** Returns the float value for `sweep_metric` from valid or test; raises with available keys if missing.
- **objective(trial):** one Optuna trial: draw, apply_config, train_model, append to results_list and the trial table, return metric for the goal.

---

## Sweep config (sweep.yaml)

File: [src/protify/yamls/sweep.yaml](../src/protify/yamls/sweep.yaml).

- **Top-level:** `parameters` (required); optional `early_terminate` (e.g. `{type: hyperband, min_iter: 10}`).
- **parameters:** W&B sweep format. Examples:
  - `lr`, `weight_decay`: `distribution: log_uniform_values`, `min`, `max`.
  - `hidden_size`, `n_layers`, `dropout`, `pre_ln`: uniform or int_uniform.
  - Transformer-specific: `transformer_hidden_size`, `classifier_dropout`, `classifier_size`, `probe_pooling_types`, `embedding_pooling_types`, `embedding_hidden_state_index`.
  - LoRA: `lora_r`, `lora_alpha`, `lora_dropout`.
  - Trainer: `num_epochs`, `probe_batch_size`, `base_batch_size`, `probe_grad_accum`, `base_grad_accum`, `patience`, `seed`.

Only parameters that are in the allowed sets (mlp_probe_params, transformer_probe_params, lora_params, trainer_keys, embedding_keys) are applied; the rest are ignored for that probe type.

Example hidden-state sweep entry:

```yaml
parameters:
  embedding_hidden_state_index:
    values: [-1, 3, 6]
```

---

## CLI arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_wandb_hyperopt` | flag | False | Run W&B hyperparameter sweep. |
| `--wandb_project` | str | (env/args) | W&B project name. |
| `--wandb_entity` | str | (env/args) | W&B entity. |
| `--sweep_config_path` | str | yamls/sweep.yaml | Path to sweep YAML. |
| `--sweep_count` | int | 10 | Number of trials per (model, dataset). |
| `--sweep_method` | choice | bayes | bayes, grid, random. |
| `--sweep_metric_cls` | str | eval_loss | Classification metric to optimize. |
| `--sweep_metric_reg` | str | eval_loss | Regression metric to optimize. |
| `--sweep_goal` | choice | minimize | maximize, minimize. |

Set `WANDB_API_KEY`, or put it in the `.secrets.env` that `foundry.secrets_env` loads, for authentication. `foundry.logging.Run` reads the key from the environment, so a key that only `wandb login` stored is not enough.

---

## Examples

### Run a sweep (after setting W&B credentials)

```bash
py -m src.protify.main --yaml_path src/protify/yamls/base.yaml --use_wandb_hyperopt --sweep_count 5 --wandb_project my_project --wandb_entity my_entity
```

### Optimize Spearman for regression

```bash
py -m src.protify.main --data_names DeepLoc-2 --model_names ESM2-8 --use_wandb_hyperopt --sweep_metric_reg eval_spearman_rho --sweep_goal maximize
```

### Custom sweep config

```bash
py -m src.protify.main --yaml_path src/protify/yamls/base.yaml --use_wandb_hyperopt --sweep_config_path my_sweep.yaml --sweep_count 20
```

---

## See also

- [Configuration](cli_and_config.md) for all W&B and sweep flags
- [Probes and training](probes_and_training.md) for probe and trainer options that are swept
- [Models and embeddings](models_and_embeddings.md) for embedding options (e.g. embedding_pooling_types, embedding_hidden_state_index)
- [Logging and replay](logging_and_replay.md) for where sweep CSV and results are written
