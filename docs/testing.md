# Testing

This page describes the Protify testing suite: where tests live, what they cover, and how to run them.

---

## Overview

Tests are under [src/protify/testing_suite/](../src/protify/testing_suite/). They cover metrics computation, pooling strategies, probe construction and forward passes, model components (MLP, attention, rotary embeddings), loss functions, blob serialization, data utilities, seed reproducibility, lazy-predict integration, packaged probe export, and probe attention behavior. Run them with pytest from the repository root.

---

## Test layout

| File | Description |
|------|-------------|
| **conftest.py** | Shared fixtures (tiny embeddings, masks, label arrays) and marker registration (`gpu`, `slow`). |
| **test_metrics.py** | Metric computation: softmax, scorers, threshold optimization, single/multi-label classification, regression, tokenwise, ROC/PR AUC, `get_compute_metrics` dispatch. |
| **test_pooler.py** | All pooling strategies (mean, max, norm, median, std, var, cls, parti/PageRank) with and without attention masks, concatenation behavior. |
| **test_probe_construction.py** | `get_probe` factory dispatch for all probe_type x tokenwise combinations, forward passes (singlelabel, regression, multilabel, sigmoid_regression), loss computation, config roundtrip. |
| **test_model_components.py** | `intermediate_correction_fn`, SwiGLU, `swiglu_ln_ffn`, RotaryEmbedding, MultiHeadAttention shapes and values. |
| **test_losses.py** | `get_loss_fct` dispatch, SoftBCELoss, SoftBCEWithLogitsLoss (forward, ignore_index, smooth_factor). |
| **test_blob_serialization.py** | `tensor_to_embedding_blob` / `embedding_blob_to_tensor` roundtrip for float32/float16/bfloat16, `batch_tensor_to_blobs`, legacy fallback. |
| **test_data_utils.py** | Amino acid, DNA, RNA, codon constant validation, translation mapping coverage, `pad_and_concatenate_dimer`. |
| **test_seed_utils.py** | `set_global_seed` / `get_global_seed`, reproducibility (torch, numpy, random), `seed_worker`, `dataloader_generator`. |
| **test_lazy_predict.py** | LazyClassifier and LazyRegressor fit, models, predictions, provide_models. |
| **test_packaged_probe_export.py** | Linear and transformer probe save/load roundtrip via PackagedProbeModel, PPI inference with/without token_type_ids. |
| **test_probe_attention.py** | Transformer attention with 2D/4D masks, s_max output, `resolve_attention_backend`. |
| **embedding_test.py** | Standalone embedding diagnostic script (excluded from pytest collection). |

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

---

## See also

- [Getting started](getting_started.md) for installation and entry points
- [Probes and training](probes_and_training.md) for packaged probe export
- [Models and embeddings](models_and_embeddings.md) for embedder behavior
