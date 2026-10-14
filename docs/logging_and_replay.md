# Logging and Replay

This page documents session logging and replay: `MetricsLogger` (log file and results TSV), `log_metrics()` and `write_results()`, `LogReplayer`, and how replay is triggered from the CLI. For the TSV format and how it is used for plots, see [Visualization](visualization.md).

---

## Overview

Every run uses a **MetricsLogger** that writes:

1. A **text log file** (one per session) with a header, full config (excluding tokens/API keys), and INFO lines that record which methods were called (e.g. "Called method: get_datasets").
2. A **results TSV** (one per session) with rows = datasets, columns = dataset + model names, and cells = JSON-encoded metric dicts. This file is updated after each (dataset, model) result via `log_metrics()` and is the input to `create_plots()`.

**LogReplayer** parses a prior session log, reconstructs the config namespace, and re-runs the same sequence of method names on a new `MainProcess` instance so you can reproduce a workflow without re-entering options.

---

## How it works

**MetricsLogger:**

1. **__init__(args):** Stores `logger_args`; no I/O yet.
2. **start_log_main()** or **start_log_gui():** Calls `_start_file()` (creates `log_dir`, `results_dir`, sets `random_id` from `PROTIFY_JOB_ID`, replay path stem, or `{date}-{time}_{4 letters}`), then overwrites or appends to the log file and calls `_minimial_logger()` to set up the Python logger and `logger_data_tracking = { dataset_name: { model_name: metrics_dict } }`.
3. **log_metrics(dataset, model, metrics_dict, split_name=None):** Filters out non-time keys (keeps `training_time_seconds*`), updates `logger_data_tracking[dataset][model]`, and calls `write_results()`.
4. **write_results():** Sorts datasets and models (e.g. by mean eval_loss), writes the TSV with header `dataset` + model names, each cell `json.dumps(metrics)`.
5. **end_log():** Appends system info (platform, pip list, nvidia-smi, Python version) to the log file.

**Replay:**

1. User passes `--replay_path path/to/session.log`.
2. `main()` creates `LogReplayer(replay_path)`, calls `replayer.parse_log()` to get a namespace of arguments (from lines like `key:\tvalue`; values are `ast.literal_eval`'d when possible). INFO lines in the log contribute to a list of method name strings.
3. `MainProcess(replay_args, GUI=False)` is created; then `replayer.run_replay(main)` runs `getattr(main, method)()` for each method name in order. Missing methods produce a warning.

So the "replayed script" is the sequence of method names that were logged as INFO (from the `@log_method_calls` decorator on MainProcess methods).

---

## MetricsLogger API

| Method | Description |
|--------|-------------|
| `_start_file()` | Creates log_dir, results_dir, sets random_id, log_file, results_file. |
| `start_log_main()` | _start_file, overwrite log with header and args, _minimial_logger. |
| `start_log_gui()` | Same but different header handling. |
| `load_tsv()` | Reads results_file into logger_data_tracking (e.g. for resuming). |
| `write_results()` | Writes logger_data_tracking to results TSV (sorted). |
| `log_metrics(dataset, model, metrics_dict, split_name=None)` | Updates tracking and writes TSV. |
| `end_log()` | Appends system info to log file. |

**Results TSV shape:** Rows = dataset names, Columns = dataset + model names, Cells = JSON metrics (e.g. test_spearman, eval_loss, training_time_seconds).

---

## LogReplayer API

| Method | Description |
|--------|-------------|
| `__init__(log_file_path)` | Stores path; arguments = {}, method_calls = []. |
| `parse_log()` | Reads log; key:value lines -> arguments (literal_eval when possible); INFO lines -> method name strings. Returns SimpleNamespace(**arguments). |
| `run_replay(target_obj)` | For each method name in method_calls, calls getattr(target_obj, method)(). Warns if method missing. |

---

## Replay flow in main.py

When `args.replay_path` is set:

1. `LogReplayer(args.replay_path)` is created.
2. `replay_args = replayer.parse_log()`; `replay_args.replay_path = args.replay_path`; seed and deterministic are applied from replayed args.
3. `main = MainProcess(replay_args, GUI=False)`.
4. Full args are printed; then `replayer.run_replay(main)` runs the replayed method sequence.
5. Normal write_results/generate_plots/end_log are not called automatically inside replay; they run only if the replayed methods include them (e.g. if the original run called `write_results`, that call is replayed).

---

## Examples

### Replay a prior session

```bash
python -m protify.main --replay_path logs/2025-01-15-12-00_ABCD.txt
```

### Where files go

- Log: `log_dir/{random_id}.txt` (e.g. `logs/2025-01-15-12-00_ABCD.txt`).
- Results: `results_dir/{random_id}.tsv` (e.g. `results/2025-01-15-12-00_ABCD.tsv`).

---

## See also

- [Visualization](visualization.md) for how the results TSV is used by create_plots
- [Getting started](getting_started.md) for log_dir and results_dir
- [Configuration](cli_and_config.md) for --replay_path and paths
