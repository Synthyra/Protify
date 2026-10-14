# Visualization

This page documents how Protify produces plots: the main `create_plots()` entry (TSV to radar, bar, and heatmap PNGs), the expected TSV format, regression vs classification metric choice, and the per-run CI plots (regression_ci_plot, classification_ci_plot) generated during training. For logging and the results TSV, see [Logging and replay](logging_and_replay.md).

---

## Overview

Two kinds of visualization are used:

1. **Publication-style comparison plots:** A metrics TSV (datasets x models, cells = JSON metrics) is turned into six PNGs: radar (raw and normalized), bar (raw and normalized), heatmap (raw and normalized). This is done by `create_plots()` in [plot_result.py](../src/protify/visualization/plot_result.py), typically after a run via `MainProcess.generate_plots()`.
2. **Per-run CI plots:** During training, when `make_plots` and `plots_dir` are set, the trainer calls `regression_ci_plot` or `classification_ci_plot` from [ci_plots.py](../src/protify/visualization/ci_plots.py) to save scatter/ROC plots for the best run.

---

## How it works

**create_plots(tsv, outdir, no_std=False):**

1. **load_tsv(tsv):** Reads the TSV; first column is `dataset`; remaining columns are model names. Each cell (except dataset) is parsed as JSON (metrics dict) or as a string like `"0.85±0.02"`.
2. For each row (dataset), regression vs classification is decided via `is_regression(metrics)` (reg: spearman, pearson, r_squared, rmse, mse; cls: accuracy, f1, mcc, auc, etc.).
3. **pick_metric(metrics, REG_PREFS or CLS_PREFS):** Chooses one metric per task type (e.g. regression: spearman then r_squared then pearson; classification: f1 then mcc then accuracy).
4. **get_metric_value_with_std** per model: parses value (float or "mean±std") and builds mean, std, and display string.
5. Datasets are ordered by `DATASET_NAMES` (from [plot_labels.py](../src/protify/visualization/plot_labels.py)), then any others appended. Models are sorted by average score (ascending). Normalized plots use per-dataset min-max normalization and optional reordering.
6. Output directory is `outdir / tsv.stem`. Six PNGs are written (e.g. 450 dpi in the implementation).

---

## TSV format

The results TSV written by [MetricsLogger](logging_and_replay.md) has:

- **Header:** `dataset` followed by one column per model name.
- **Rows:** One per dataset.
- **Cells:** JSON objects with metric keys (e.g. `test_spearman`, `test_spearman_mean`, `test_spearman_std`, `eval_loss`, `training_time_seconds`) or a string like `"0.85±0.02"`.

Example (conceptually):

| dataset  | ESM2-8   | ESM2-35   |
|----------|----------|-----------|
| DeepLoc-2 | {"test_spearman": 0.82, ...} | {"test_spearman": 0.85, "test_spearman_std": 0.02, ...} |

---

## Plots produced

All under `outdir / tsv.stem`:

| File | Description |
|------|-------------|
| `{stem}_radar_all.png` | Radar: datasets = axes, one curve per model; raw scores; "Avg" axis = mean. |
| `{stem}_radar_all_normalized.png` | Same with scores normalized (min-max) per category. |
| `{stem}_bar_all.png` | Bar: datasets on x, score on y, hue = model. |
| `{stem}_bar_all_normalized.png` | Bar with normalized scores. |
| `{stem}_heatmap_all.png` | Heatmap: rows = datasets + "Average", cols = models; color row-normalized; annotations = raw or mean±std (unless no_std); best per row outlined. |
| `{stem}_heatmap_all_normalized.png` | Heatmap with normalized values and sorting; no std in annotations when no_std. |

---

## Metric choice

- **Regression:** Preferences (REG_PREFS) typically put spearman first, then r_squared, pearson, etc. The first matching key in the metrics dict is used.
- **Classification:** Preferences (CLS_PREFS) typically put f1 first, then mcc, accuracy, etc. Time-related keys and `*_mean`/`*_std` are skipped when picking the display metric.

---

## CI plots (training)

- **regression_ci_plot(y_true, y_pred, save_path, title):** Scatter true vs pred, regression line with 95% CI, annotations for R², Spearman ρ, Pearson ρ and p-values; saves PNG (e.g. 300 dpi).
- **classification_ci_plot(y_true, y_pred, save_path, title):** Reshapes/flattens as needed, caps at 10k points, calls **plot_roc_with_ci** from [pauc_plot.py](../src/protify/visualization/pauc_plot.py) for pAUC/ROC. Used for per-run and "best run" plots when `make_plots` and `plots_dir` are set in TrainerArguments.

---

## CLI

To generate the six comparison plots from an existing TSV without running a full pipeline:

```bash
python -m protify.visualization.plot_result --input path/to/results.tsv --output_dir plots --no_std
```

`--no_std` omits standard deviation from heatmap annotations.

---

## Examples

### After a run

Plots are generated automatically when `main()` finishes (if results TSV exists). Output goes to `plots_dir / {tsv_stem}/` (e.g. `plots/2025-01-15-12-00_ABCD/`).

### Standalone from TSV

```bash
python -m protify.visualization.plot_result --input results/my_id.tsv --output_dir plots
```

---

## See also

- [Logging and replay](logging_and_replay.md) for how the results TSV is written
- [Configuration](cli_and_config.md) for `--plots_dir`
- [Probes and training](probes_and_training.md) for `make_plots` and `plots_dir` in TrainerArguments
