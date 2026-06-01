# Models and Embeddings

This page documents base (protein/chemical) models and embedding generation: `BaseModelArguments` (model_names vs model_paths/model_types), `get_base_model` and `get_tokenizer`, `EmbeddingArguments`, the `Embedder` flow, `get_embedding_filename`, storage (SQL vs PTH), and pooling. To list supported models from the CLI or Python, see [Resource listing](resource_listing.md).

---

## Overview

Protify supports two ways to specify models: **preset names** (`model_names`, e.g. ESM2-8, ProtT5) resolved from a built-in map, or **explicit paths and types** (`model_paths` plus `model_types`) for custom or local models. Embeddings are produced by running the base model (or downloading/reading from disk) and optionally pooling per-residue outputs to vectors. They can be saved as `.pth` (dict of sequence -> tensor) or in SQLite (`.db`).

---

## How it works

1. **BaseModelArguments** is built from config. Either `model_names` is set (preset mode) or `model_paths` and `model_types` are set (path mode). They are mutually exclusive.
2. **model_entries()** yields `(display_name, dispatch_type, model_path)` for each model. Preset mode: `dispatch_type` is the preset name, `model_path` is None. Path mode: `dispatch_type` is the type keyword (e.g. esm2, custom), `model_path` is the path.
3. **Embedder** is called per model. For each model it may: download precomputed embeddings (if `download_embeddings`), read existing embeddings from disk (SQL or PTH), or compute new embeddings via `get_base_model()` and forward passes, then optionally save them.
4. **get_embedding_filename()** defines the filename (and thus cache key) from model name, `matrix_embed`, pooling types, and non-default hidden-state index.

---

## BaseModelArguments

Defined in [get_base_models.py](../src/protify/base_models/get_base_models.py).

| Argument | Type | Description |
|----------|------|-------------|
| `model_names` | List[str] | Preset names (e.g. ESM2-8, ProtT5). Use `['standard']` to expand to standard set, or names containing `'exp'` for experimental. Mutually exclusive with model_paths/model_types. |
| `model_paths` | List[str] | Paths (HuggingFace IDs or local). Must pair with `model_types` (same length). |
| `model_types` | List[str] | Type keyword per path: esm2, esmc, protbert, prott5, ankh, glm, dplm, dplm2, protclm, onehot, amplify, e1, vec2vec, calm, custom, random, etc. |
| `model_dtype` | str | Data type for loading (e.g. bf16, fp32). |

**model_entries()** yields `(display_name, dispatch_type, model_path)` so that the pipeline can call `get_base_model(dispatch_type, ..., model_path=model_path)` and `get_tokenizer(dispatch_type, model_path=model_path)`.

---

## get_base_model and get_tokenizer

- **get_base_model(model_name, masked_lm=False, dtype=None, model_path=None)**  
  Returns `(model, tokenizer)` (or equivalent). Dispatch is by substring in `model_name.lower()`: e.g. random, esm2/dsm, esmc, protbert, prott5, ankh, glm, dplm2, dplm, protclm, onehot, amplify, e1, vec2vec, calm, custom. Custom requires `model_path`.

- **get_base_model_for_training(model_name, tokenwise=False, num_labels=None, hybrid=False, dtype=None, model_path=None)**  
  Same family names; used when training (probe or full/hybrid). Does not support random, onehot, vec2vec, custom in some code paths.

- **get_tokenizer(model_name, model_path=None)**  
  Returns the tokenizer for the given model name (and path for custom/path-based loading).

Supported model type keywords (for `model_types` or in preset names) include: random, esm2, dsm, esmc, protbert, prott5, ankh, glm, dplm, dplm2, protclm, onehot, amplify, e1, vec2vec, calm, custom. See [supported_models.py](../src/protify/base_models/supported_models.py) for `all_presets_with_paths`, `currently_supported_models`, `standard_models`, `experimental_models`.

---

## EmbeddingArguments

Defined in [embedder.py](../src/protify/embedder.py). Constructor maps long names to internal attributes (e.g. `embedding_pooling_types` -> `pooling_types`).

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `embedding_batch_size` | int | 4 | Batch size for embedding forward passes. |
| `embedding_num_workers` | int | 0 | DataLoader workers for embedding. |
| `download_embeddings` | bool | False | If True, download from HuggingFace (e.g. Synthyra precomputed). |
| `download_dir` | str | Synthyra/vector_embeddings | HuggingFace dataset/repo for precomputed embeddings. |
| `matrix_embed` | bool | False | If True, keep per-residue matrices; if False, pool to vectors. |
| `embedding_pooling_types` | List[str] | ['mean'] | Pooling for vectors (e.g. mean, var, parti). |
| `embedding_hidden_state_index` | int | -1 | Hidden-state tuple index to pool from. `-1` preserves the final hidden-state behavior and old cache names. |
| `save_embeddings` | bool | False | Whether to write computed embeddings to disk. |
| `embed_dtype` | dtype | torch.float32 | Dtype for stored embeddings. |
| `model_dtype` | dtype | None | Dtype for base model (None uses default). |
| `sql` | bool | False | Store in SQLite (`.db`) instead of `.pth`. |
| `embedding_save_dir` | str | embeddings | Directory for save/load paths. |
| `padding` | str | max_length | Padding strategy for the embedding collator. `max_length` pads all batches to `max_length` tokens (optimal for torch.compile + flex attention). `longest` pads to the longest sequence in each batch (skips compile, but avoids wasted padding compute). |
| `max_length` | int | 2048 | Maximum sequence length. Always passed to the tokenizer for truncation, regardless of padding mode. |
| `multi_gpu` | bool | False | Split sequences across all available GPUs via `mp.Process`. Each GPU loads its own model copy and embeds a shard. |
| `autocast` | bool | False | Wrap forward pass in `torch.autocast` for mixed-precision inference. Useful when `model_dtype` is float32 but you want float16 compute speed (~1.5x). |

`read_scaler` (CLI/embedding) is used for SQL read batching in dataset building.

---

## Embedder flow

1. **__call__(model_name, model_type=None, model_path=None)**  
   - If `download_embeddings`: `_download_embeddings(model_name)` (download, unzip, optional merge, save under `embedding_save_dir`).
2. **_read_embeddings_from_disk(model_name)**  
   - Builds path via `get_embedding_filename(model_name, matrix_embed, pooling_types, extension='pth' or 'db', hidden_state_index=...)`.
   - **SQL:** Opens/creates `.db`, table `embeddings (sequence, embedding)`; returns `(to_embed, save_path, {})`.  
   - **PTH:** If file exists, loads dict `{seq -> tensor}`; returns `(to_embed, save_path, embeddings_dict)`.
3. If there are sequences left to embed (`len(to_embed) > 0`): get base model and tokenizer via `get_base_model(dispatch_name, ...)`, then **_embed_sequences(...)**.
4. **_embed_sequences** builds a DataLoader over sequences, runs forward passes, applies pooling (or keeps matrix), and either inserts into SQLite or updates the embeddings dict; if `save_embeddings` and not SQL, saves the dict to PTH at the end.

---

## get_embedding_filename

```text
get_embedding_filename(model_name, matrix_embed, pooling_types, extension='pth', hidden_state_index=-1)
```

Returns a filename: `{model_name}_{matrix_embed}[_hs{hidden_state_index}][_{pooling_types}].{extension}`. For vector embeddings, pooling types are sorted and joined with underscore (e.g. `mean_var`). Extension is `pth` or `db` for SQL.

The default `hidden_state_index=-1` omits the hidden-state suffix, so existing caches keep their old names. Non-default indexes use distinct caches, e.g. `ESM2-8_False_hs6_mean_var.pth`.

---

## SQL vs PTH

- **PTH:** One file per (model_name, matrix_embed, pooling_types). Dict mapping sequence string to tensor. Load/save with `torch.load` / `torch.save`.
- **SQL:** One `.db` file per same key. Table `embeddings(sequence, embedding)`. New sequences use INSERT OR REPLACE. Better for very large sequence sets, incremental updates, and Modal volume storage.

### SQL performance optimizations

The SQL path uses several optimizations to minimize the gap with in-memory dict storage:

- **Compact blob format:** Embeddings are serialized as a small binary header + raw numpy bytes instead of `torch.save`. For float16 vectors this is 3.4x smaller and ~18x faster to serialize than pickle-based `torch.save`.
- **Batch serialization:** `batch_tensor_to_blobs()` converts an entire batch of identically-shaped embeddings in one numpy call, avoiding per-embedding overhead.
- **Async writer thread:** SQLite INSERTs run in a background thread via `queue.Queue`, so the GPU never blocks on I/O.
- **Aggressive pragmas:** `PRAGMA synchronous=OFF` and `PRAGMA cache_size=-64000` (64MB) during embedding. Data is reproducible, so fsync is unnecessary.
- **Batch reads:** Training dataset classes use batch `SELECT ... WHERE sequence IN (?)` queries with persistent connections instead of per-sequence lookups.

---

## Pooling types

Common values for `embedding_pooling_types` (and probe-side `probe_pooling_types` where applicable):

- **mean:** Mean over sequence length (masked).
- **var:** Variance (or other stats) over sequence.
- **parti:** Requires attention outputs; see pooler implementation.
- **cls:** Use first token (CLS) representation when available.

Pooling is applied in [pooler.py](../src/protify/pooler.py) via the `Pooler` class when not using `matrix_embed`.

### Hidden-state selection

By default Protify pools the final hidden state. Set `embedding_hidden_state_index` to any valid model hidden-state tuple index to pool an intermediate layer instead. This option participates in cache naming and W&B sweeps, so changing the layer regenerates or reloads the matching embeddings.

---

## Examples

### Preset models

```bash
py -m src.protify.main --model_names ESM2-8 ESM2-35 --data_names DeepLoc-2
```

### Custom model with path

```bash
py -m src.protify.main --model_paths "org/my-model" --model_types custom --data_names DeepLoc-2
```

### Save and reuse embeddings

```bash
py -m src.protify.main --model_names ESM2-8 --data_names DeepLoc-2 --save_embeddings --embedding_pooling_types mean var
```

### Save an intermediate hidden state

```bash
py -m src.protify.main --model_names ESMC-300 --data_names DeepLoc-2 --save_embeddings --embedding_hidden_state_index 12
```

### Download precomputed embeddings

```bash
py -m src.protify.main --model_names ESM2-150 --data_names DeepLoc-2 --download_embeddings
```

### SQLite storage

```bash
py -m src.protify.main --model_names ESM2-8 --data_names DeepLoc-2 --sql --save_embeddings
```

---

## See also

- [Configuration](cli_and_config.md) for model and embedding CLI flags
- [Resource listing](resource_listing.md) for listing and downloading models
- [Data](data.md) for how datasets are loaded before embedding
- [Probes and training](probes_and_training.md) for how embeddings are consumed by probes
