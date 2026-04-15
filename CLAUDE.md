# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from `src/protify/` unless stated otherwise.

```bash
# CLI run
py -m main --model_names ESM2-8 --data_names DeepLoc-2 --num_epochs 100

# GUI
py -m gui

# YAML-driven run
py -m main --yaml_path yamls/base.yaml

# Replay a prior session
py -m main --replay_path logs/<log_id>.txt

# List supported models and datasets
py -m resource_info

# Tests (must run in Docker, not natively on Windows)
docker build -t protify-env:latest .
docker run --rm -v "${PWD}":/workspace -w /workspace/src/protify protify-env:latest python -m pytest testing_suite/ -v

# Run a single test file
docker run --rm -v "${PWD}":/workspace -w /workspace/src/protify protify-env:latest python -m pytest testing_suite/test_metrics.py -v

# CPU-only tests (skip GPU-dependent tests)
docker run --rm -v "${PWD}":/workspace -w /workspace/src/protify protify-env:latest python -m pytest testing_suite/ -v -m "not gpu and not slow"

# GPU tests (add --gpus all)
docker run --rm --gpus all -v "${PWD}":/workspace -w /workspace/src/protify protify-env:latest python -m pytest testing_suite/ -v
```

**Windows note:** Use `MSYS_NO_PATHCONV=1` prefix if running from Git Bash to prevent path mangling.

## Architecture

**Entry points:**
- `main.py` — CLI/YAML orchestrator (~1100 lines); all args parsed here
- `gui.py` — Tkinter GUI with 11 tabs; runs the same pipeline as CLI in background threads
- `cloud_backend.py` — CloudBackend ABC + HTTPCloudBackend; auto-dispatches when `--cloud_api_key` is passed
- `cloud_cli.py` — CLI dispatch for cloud execution (submit, poll, fetch results)

**Main pipeline flow:**

```
args (CLI / YAML / GUI)
    → MainProcess (inherits DataMixin + TrainerMixin)
        ├─ DataMixin (data/)          load HF hub or local CSV datasets
        ├─ Embedder (embedder.py)     generate/pool PLM embeddings
        ├─ get_base_model()           load PLM from base_models/
        ├─ get_probe()                build probe from probes/
        └─ TrainerMixin               train/eval loop + metrics + plots
```

**Key directories:**
- `base_models/` — one file per PLM family (ESM2, ESMC, ProtBert, ANKH, GLM, DPLM, etc.); all share a `get_base_model()` factory in `get_base_models.py`
- `probes/` — LinearProbe, TransformerProbe, Lyra; `lazy_predict.py` for scikit-learn auto-selection
- `data/` — dataset loading, collators, AA↔DNA/RNA translation; supported datasets listed in `supported_datasets.py`
- `model_components/` — attention backends (sdpa/flex/custom kernels), transformer blocks, MLP
- `visualization/` — PAUC curves, dimensionality reduction (t-SNE/UMAP/PCA), radar/bar comparisons, confidence intervals
- `benchmarks/proteingym/` — zero-shot DMS scoring pipeline against ProteinGym assays
- `yamls/` — `base.yaml` (full config template), `sweep.yaml` (W&B Bayesian hyperopt)

**FastPLMs submodule** lives at `src/protify/fastplms`. Base model loaders reference it via:
```python
_FASTPLMS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fastplms')
```

**Outputs per run:**
- `logs/<timestamp>.txt` — full reproducible CLI args
- `results/*.tsv` — per model/dataset metrics
- `plots/<timestamp>/*.png` — all visualizations
- `weights/` — saved probe/model if `--save_model`
- `embeddings/` — cached embeddings if `--save_embeddings`

**Training modes:** probe-only (frozen PLM), full fine-tune, hybrid, scikit (embeddings → sklearn), W&B hyperparameter sweep.

**Balanced regression metrics (EpHod-style):** For `task_type in ('regression', 'sigmoid_regression')`, Protify reports a second suite of weighted and resampled metrics (weighted RMSE / R^2, resampled Pearson / Spearman, binned MCC / F1 / ROC-AUC) alongside standard ones. Sample weights come from the training label distribution (schemes: `bin_inv`, `bin_inv_sqrt`, `LDS_inv`, `LDS_inv_sqrt`, `LDS_extreme`, `none`) and are pre-computed once in `DataMixin._compute_balanced_weights_for`. Bin borders default to tertiles; override with `--balanced_bin_borders 5 9` (pH). Implementation in `src/protify/metrics_balanced.py`.

**Registries (authoritative source of truth):**
- Supported models: `base_models/supported_models.py` (`currently_supported_models` list, 45 entries)
- Supported datasets: `data/supported_datasets.py` (`supported_datasets` dict, 67 entries)

## Testing

Tests live in `testing_suite/`. Working directory must be `src/protify/` (bare imports like `import entrypoint_setup` require it). All tests use synthetic data; no network calls or real dataset downloads.

**Import pattern:** Every test file uses a try/except chain: `from src.protify.X`, then `from protify.X`, then relative `from ..X`.

**Markers:** `@pytest.mark.gpu` (requires CUDA), `@pytest.mark.slow` (>10s). Registered in `testing_suite/conftest.py`.

**`embedding_test.py`** is a standalone diagnostic script, not a pytest test. It is excluded from collection via `collect_ignore` in conftest.

**`test_packaged_probe_export.py`** has a `_copy_runtime_code` helper that copies the full `protify/` source tree into a temp dir and rewrites relative imports in `packaged_probe_model.py` to absolute imports so `transformers` dynamic module loader can resolve them. If probe exports break, check this helper first.

**Embedding pipeline performance:**
- SQL storage uses compact binary blobs (not torch.save), async writer thread, and batch serialization
- `padding='max_length'` enables torch.compile; `padding='longest'` skips compile (flex attention incompatibility) but sorts by length to minimize padding
- `multi_gpu=True` splits sequences across GPUs via `mp.Process`
- `autocast=True` enables mixed-precision inference for float32 models
