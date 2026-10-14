# Getting Started

This guide covers installation, entry points, and your first CLI and YAML runs. For full argument reference, see [Configuration](cli_and_config.md).

---

## Overview

Protify can be run from the command line (with or without a YAML config file) or from the Tk GUI. All paths use the same `MainProcess` pipeline: load config, load data, compute or load embeddings, train probes (or run full finetuning, hybrid, or scikit), then write results and generate plots.

---

## Installation

### From pip

```bash
pip install Protify
```

### Local development (clone and install)

From the repository root:

```bash
git clone https://github.com/Synthyra/Protify.git
cd Protify
git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

The editable install makes `protify`, and the `foundry` code vendored at the repository root, importable
from anywhere:

```bash
python -m protify.main --help
```

### Docker

Run all commands from the **repository root** on your host. Mount the project at `/workspace` and run from
`/workspace`; the image puts `src/` and the repository root on `PYTHONPATH`, so `python -m protify.main` and
`python -m protify.gui` work there.

**Linux / macOS:**
```bash
docker build -t protify-env:latest .
docker run --rm -it --gpus all -v "${PWD}":/workspace -w /workspace protify-env:latest python -m protify.main --model_names ESM2-8 --data_names DeepLoc-2 --num_epochs 2
```

**Windows:**
```bash
docker build -t protify-env:latest .
docker run --rm -it --gpus all -v "%CD%":/workspace -w /workspace protify-env:latest python -m protify.main --model_names ESM2-8 --data_names DeepLoc-2 --num_epochs 2
```

Paths like `--log_dir` and `--results_dir` are relative to `/workspace`, the repository root, so outputs land beside the mounted project.

### Optional: xformers for AMPLIFY

If you use the AMPLIFY model with `--use_xformers`:

```bash
pip install xformers
```

---

## Entry points

| Entry point | Purpose |
|------------|--------|
| `python -m protify.main` | CLI (and optional YAML). |
| `python -m protify.gui` | Tk GUI. |

Use `py` on Windows if `python` is not the interpreter you installed into. Both modules are part of the
package, so they run from any directory once Protify is installed, and from `/workspace` in Docker.

---

## How a run flows

When you pass datasets (`--data_names` or `--data_dirs`), the default flow is:

1. **Parse arguments** (CLI and optionally merge YAML).
2. **Apply settings:** Build `DataArguments`, `BaseModelArguments`, `ProbeArguments`, `EmbeddingArguments`, `TrainerArguments` from the combined config.
3. **Load data:** `get_datasets()` loads from HuggingFace and/or local dirs and normalizes columns.
4. **Embeddings:** `save_embeddings_to_disk()` computes (or downloads/reads) embeddings per model and saves them to the feature store under `embedding_save_dir`, converting an old `.pth` or SQLite cache it finds.
5. **Train:** One of: `run_nn_probes()` (default), `run_full_finetuning()`, `run_hybrid_probes()`, or `run_scikit_scheme()` (or W&B hyperopt).
6. **Write results:** Metrics are written to a TSV in `results_dir`.
7. **Plots:** `create_plots()` generates radar, bar, and heatmap PNGs in `plots_dir`.
8. **End log:** Session log is finalized (e.g. system info appended).

If you pass `--proteingym` and no datasets, only the ProteinGym zero-shot scoring path runs (see [ProteinGym](proteingym.md)).

---

## First CLI run

Minimal probe-only run (one model, one dataset, two epochs):

```bash
python -m protify.main --model_names ESM2-8 --data_names DeepLoc-2 --num_epochs 2
```

The reference config in `src/protify/yamls/base.yaml` runs the recommended stack instead: ESMC 300M read through its sparse autoencoder, max-pooled, into a gradient-boosted probe.

```bash
python -m protify.main --yaml_path src/protify/yamls/base.yaml --data_names DeepLoc-2
```

Command-line defaults:

- `--log_dir logs`, `--results_dir results`, `--plots_dir plots`
- `--embedding_save_dir embeddings`, `--model_save_dir weights`
- Probe: mlp; embeddings: mean and variance pooling; no saving of embeddings unless you set `--save_embeddings`

Results appear in `results/` (TSV of metrics) and `logs/` (session log). Plots are written to `plots/` after the run.

---

## First YAML run

1. Copy or edit the bundled config:

   ```bash
   # From repo root; edit paths if needed
   notepad src\protify\yamls\base.yaml
   ```

2. Set at least:
   - `data_names` (e.g. `['DeepLoc-2']`)
   - `model_names` (e.g. `['ESM2-8']`)
   - Other sections (paths, probe, trainer) have defaults in the file.

3. Run:

   ```bash
   python -m protify.main --yaml_path src/protify/yamls/base.yaml
   ```

CLI flags override YAML. For example:

```bash
python -m protify.main --yaml_path src/protify/yamls/base.yaml --num_epochs 5
```

---

## The sparse autoencoder recipe

[`yamls/production_sae.yaml`](../src/protify/yamls/production_sae.yaml) is the whole
sparse-autoencoder pipeline as one config: ESMC-6B features through a Biohub codebook, max
pooled to one vector per protein, joined symmetrically for pair datasets, dead columns
dropped, and an XGBoost probe whose hyperparameters are searched against the validation
split. Point it at a dataset:

```bash
python -m protify.main --yaml_path src/protify/yamls/production_sae.yaml \
  --data_names gold-ppi
```

The file comments say why each value is what it is. Two are worth knowing before changing
them:

- `sae_k: 16` with `sae_codebook_dim: 131072` is the sparse corner of the published grid.
  Max pooling unions the top-k features of every residue, so pooled sparsity follows `k`
  rather than width. At `sae_k: 512` the same codebook pools to about a third non-zero,
  its cache costs about twenty-five times more per protein, and the support prune has
  almost nothing left to remove.
- `pair_feature_mode: minmax` makes a pair row independent of which protein the dataset
  listed first, so `random_pair_flipping` is off: under a symmetric row it would emit the
  row that is already there.

---

## Where results go

| Output | Default path | Controlled by |
|--------|--------------|---------------|
| Session log | `logs/{random_id}.log` | `--log_dir` |
| Metrics TSV | `results/{random_id}.tsv` | `--results_dir` |
| Plots | `plots/{tsv_stem}/` | `--plots_dir` |
| Saved models | `weights/` | `--model_save_dir` |
| Embeddings | `embeddings/` | `--embedding_save_dir` |

The `random_id` is generated at session start (or from `PROTIFY_JOB_ID` / replay path). See [Logging and replay](logging_and_replay.md).

---

## See also

- [Configuration](cli_and_config.md) for all CLI and YAML options
- [Data](data.md) for datasets and `data_dirs`
- [Models and embeddings](models_and_embeddings.md) for base models and embedding options
- [Probes and training](probes_and_training.md) for probe types and training flows
- [GUI](gui.md) for the Tk interface
