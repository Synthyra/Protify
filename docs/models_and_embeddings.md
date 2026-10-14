# Models and Embeddings

This page documents base (protein/chemical) models and embedding generation: `BaseModelArguments` (model_names vs model_paths/model_types), `get_base_model` and `get_tokenizer`, `EmbeddingArguments`, the `Embedder` flow, the feature store and `--sql`, converting old caches, and pooling. To list supported models from the CLI or Python, see [Resource listing](resource_listing.md).

---

## Overview

Protify supports two ways to specify models: **preset names** (`model_names`, e.g. ESM2-8, ProtT5) resolved from a built-in map, or **explicit paths and types** (`model_paths` plus `model_types`) for custom or local models. Embeddings are produced by running the base model (or downloading/reading from disk) and optionally pooling per-residue outputs to vectors. They are saved in FastPLMs' feature store, one directory per representation under `--embedding_save_dir`. Caches written before that change (`.pth` dictionaries and SQLite `.db` files) are converted into the store, never recomputed; see [Old caches](#old-caches).

---

## How it works

1. **BaseModelArguments** is built from config. Either `model_names` is set (preset mode) or `model_paths` and `model_types` are set (path mode). They are mutually exclusive.
2. **model_entries()** yields `(display_name, dispatch_type, model_path)` for each model. Preset mode: `dispatch_type` is the preset name, `model_path` is None. Path mode: `dispatch_type` is the type keyword (e.g. esm2, custom), `model_path` is the path.
3. **Embedder** is called per model. For each model it may: download precomputed embeddings (if `download_embeddings`), read the rows the feature store already holds, or compute the rest via `get_base_model()` and forward passes, then optionally save them.
4. **`EmbeddingArguments.cache_descriptor()`** defines the store key from the model name, `matrix_embed`, pooling in column order, hidden-state index, tokenizer `max_length`, the special-token policy, and the stored dtype. CARBON keys also include the resolved model commit and preprocessing schema. `get_embedding_filename()` names old cache files only.

---

## BaseModelArguments

Defined in [get_base_models.py](../src/protify/base_models/get_base_models.py).

| Argument | Type | Description |
|----------|------|-------------|
| `model_names` | List[str] | Preset names (e.g. ESM2-8, ProtT5). Use `['standard']` to expand to standard set, or names containing `'exp'` for experimental. Mutually exclusive with model_paths/model_types. |
| `model_paths` | List[str] | Paths (HuggingFace IDs or local). Must pair with `model_types` (same length). Remote CARBON IDs must be pinned as `org/repo@<full-commit-sha>`. |
| `model_types` | List[str] | Type keyword per path: esm2, esmc, carbon, protbert, prott5, ankh, glm, dplm, dplm2, protclm, onehot, amplify, e1, calm, custom, random, etc. |
| `model_dtype` | str | Data type for loading (e.g. bf16, fp32). |

**model_entries()** yields `(display_name, dispatch_type, model_path)` so that the pipeline can call `get_base_model(dispatch_type, ..., model_path=model_path)` and `get_tokenizer(dispatch_type, model_path=model_path)`.

---

## get_base_model and get_tokenizer

- **get_base_model(model_name, masked_lm=False, dtype=None, model_path=None)**  
  Returns `(model, tokenizer)` (or equivalent). Dispatch is by substring in `model_name.lower()`: e.g. random, esm2/dsm, esmc, protbert, prott5, ankh, glm, dplm2, dplm, protclm, onehot, amplify, e1, calm, custom. Custom requires `model_path`.

- **get_base_model_for_training(model_name, tokenwise=False, num_labels=None, hybrid=False, dtype=None, model_path=None)**  
  Same family names; used when training (probe or full/hybrid). Does not support random, onehot, or custom in some code paths.

- **get_tokenizer(model_name, model_path=None)**  
  Returns the tokenizer for the given model name (and path for custom/path-based loading).

Supported model type keywords (for `model_types` or in preset names) include: random, esm2, dsm, esmc, carbon, protbert, prott5, ankh, glm, dplm, dplm2, protclm, onehot, amplify, e1, calm, custom. See [supported_models.py](../src/protify/base_models/supported_models.py) for `all_presets_with_paths`, `currently_supported_models`, `standard_models`, `experimental_models`.

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
| `embedding_hidden_state_index` | int | -1 | Hidden-state tuple index to pool from. `-1` uses `last_hidden_state` without requesting all intermediate states. |
| `save_embeddings` | bool | False | Whether to write computed embeddings to disk. |
| `embed_dtype` | dtype | torch.float32 | Dtype for stored embeddings, part of the store's key: float64, float32, float16, or bfloat16. |
| `model_dtype` | dtype | None | Dtype for base model (None uses default). |
| `sql` | bool | False | Stream rows from the store at train time instead of holding them in memory, and commit embedded rows as they are produced. The name is kept from when this chose a SQLite file; the store is the same either way. |
| `embedding_save_dir` | str | embeddings | The feature store root: each representation is a directory under it. |
| `padding` | str | max_length | Padding strategy for the embedding collator. `max_length` pads all batches to `max_length` tokens (optimal for torch.compile + flex attention). `longest` pads to the longest sequence in each batch (skips compile, but avoids wasted padding compute). |
| `max_length` | int | 2048 | Maximum tokenizer length, including special and model boundary tokens. Always passed to the tokenizer for truncation. CARBON uses two boundary tokens and one token per DNA 6-mer, so its raw DNA budget is `6 * (max_length - 2)` base pairs. |
| `multi_gpu` | bool | False | Split sequences across all available GPUs via `mp.Process`. Each GPU loads its own model copy and embeds a shard. |
| `autocast` | bool | False | Wrap forward pass in `torch.autocast` for mixed-precision inference. Useful when `model_dtype` is float32 but you want float16 compute speed (~1.5x). |

`read_scaler` (CLI/embedding) is used for streamed (`--sql`) read batching in dataset building.

---

## Embedder flow

1. **__call__(model_name, model_type=None, model_path=None)**  
   - If `download_embeddings`: `_download_embeddings(model_name)` (download, unzip, and convert the published file into the store; see [Old caches](#old-caches)).
2. **_read_embeddings_from_disk(model_name)**  
   - Builds the run's descriptor with `EmbeddingArguments.cache_descriptor(...)` and finds its store under `embedding_save_dir` (`feature_directory`). A store that holds the run's dtype exactly, or a wider one, serves it.
   - If the store does not exist and an old `.db` or `.pth` cache of the same representation does, that file is converted first (the file of the run's own mode first).
   - **Store missing:** every sequence is still to embed; returns `(to_embed, save_path, {})`, where `save_path` is the directory the store will be made in.
   - **Store present:** `to_embed` is what `FeatureReader.missing` reports. Under `--sql` the rows stay on disk and the dictionary is empty; otherwise it holds the already embedded rows of this run's sequences, read once.
3. If there are sequences left to embed (`len(to_embed) > 0`): get base model and tokenizer via `get_base_model(dispatch_name, ...)`, then **_embed_sequences(...)**.
4. **_embed_sequences** builds a DataLoader over sequences, runs forward passes, applies pooling (or keeps matrix), and hands each batch of rows to a `CacheWriter` (when `--save_embeddings` or `--sql` is set) and to the in-memory dictionary (unless `--sql`). The writer commits one segment per window of 2,048 rows or 256 MiB, so a run that dies keeps every committed window and the next run embeds only what is missing. With neither flag nothing is written.

---

## The feature store

Every cache is one directory of FastPLMs' feature store (`fastplms.features`), named by a key that `foundry.embedding.plain_feature_key` computes from a descriptor. The descriptor states every fact that changes the stored numbers, and the store keeps it unchanged in its `feature.json`:

| Field | Meaning |
|-------|---------|
| `schema` | `protify_embedding_v1` |
| `model` | the resolved model name, which already carries an SAE model's layer, sparsity, and codebook width |
| `revision`, `preprocessing` | a CARBON model's immutable revision and preprocessing schema; `null` for every other model |
| `form` | `vector` (pooled) or `matrix` (per token) |
| `pooling` | pooling types in the order the columns are concatenated; `null` for a matrix |
| `hidden_state_index` | the hidden state read, `-1` for the last |
| `max_length` | the tokenizer truncation limit |
| `special_tokens` | `included`: every pooled statistic, SAE max included, covers all `len(sequence) + 2` rows, and a matrix has that many rows. `excluded` names only a cache an older Protify wrote for an SAE model, which pooled the interior tokens; no current reader asks for it and the converter alone can label a cache so |
| `dtype` | the stored dtype |

A directory reads like `protify__ESM2-8__vector__mean-var__len1024__float16__fa78626840a79fd2`; the digest, not the words, is the identity. `max_length` is in the key so embeddings produced under different truncation limits are never mixed. A pooled vector is stored `dense`, an SAE vector `csr` when the model's `sparse_storage` says it pays, and a matrix `ragged`.

A store serves a request for any dtype it holds exactly: a float32 store serves float16 and bfloat16 runs, and rows written into a wider store widen and are never narrowed. `--sql` keeps its name and now means streaming rows from the store during training instead of holding them in memory: the run holds one `FeatureReader` and reads a few rows per batch. Without it the rows a run needs are read once into memory. A missing store raises `FileNotFoundError` that says how to make one.

### Old caches

Caches written before the feature store were SQLite `.db` files (table `embeddings(sequence, embedding)`, compact blobs) and pickled `.pth` dictionaries, named by `get_embedding_filename(model_name, matrix_embed, pooling_types, extension, hidden_state_index, max_length, ...)`. That name lists `{model_name}`, matrix mode, an optional hidden-state suffix, `max_length`, and the pooling types sorted and joined with underscores (for example `ESM2-8_False_len2048_mean_var.db`); CARBON names add the model revision and preprocessing schema. Nothing writes these formats now. [legacy_cache.py](../src/protify/legacy_cache.py) reads them, and converts them with `fastplms.features.convert_rows`, which compares every stored row with the decoded one bit for bit. The old file is never changed.

- **By hand:** `python -m protify.legacy_cache embeddings/ESM2-8_False_len2048_mean.db` writes the store next to the file. `--root`, `--model`, `--matrix` or `--vector`, `--pooling`, `--hidden_state_index`, `--max_length`, `--dtype`, and `--skip_existing` override what the file name says. The store dtype is the narrowest that holds every row, and a cast that would change a value raises.
- **On first use:** a run that finds an old file of its own representation and no store converts it before embedding, the file of its own mode first.
- **On download:** `--download_embeddings` converts the published file into the store instead of copying it.

The file name loses the pooling order, so the first use and the download convert only when the run's pooling types are sorted. A cache pooled in another order converts by hand with `--pooling` naming the order it was written in. `--download_embeddings` finds the published files, which were uploaded under the older naming (no `max_length`) at the default `max_length`, so `_download_embeddings` requests the current name first and then falls back to `legacy_embedding_filename`, and only when this run uses that same default. Below it, the published file would carry more of each sequence than the run asked for, so the fallback is skipped and the embeddings are computed locally. Rerunning an interrupted conversion resumes it; one that ends in a mismatch leaves its `segments/convert-*` directories committed, and removing them undoes it.

---

## ESMC sparse autoencoders

A sparse autoencoder (SAE) reads one ESMC hidden state and re-expresses it over a wide codebook in which only `k` features are active per residue. Pooling those activations over the sequence gives one sparse, high-dimensional vector per protein, which is what makes SAE features easy to pool: max pooling asks "did this protein ever express feature j, and how strongly", and needs no learned pooler.

Use them through the model names `ESMC-300-SAE`, `ESMC-600-SAE`, and `ESMC-6B-SAE`:

```bash
python -m protify.main --model_names ESMC-300-SAE --data_names solubility \
  --probe_type xgboost --embedding_pooling_types max --save_embeddings --sql
```

### Choosing a checkpoint

A bare alias resolves to the layer at 75% depth, the only layer Biohub publishes across the whole sparsity and width grid, at `k=64` and codebook width 8192. Override with `--sae_layer`, `--sae_k`, and `--sae_codebook_dim`. The resolved name carries the full identity, for example `ESMC-300-SAE-l23-k64-c8192`, and that name is what appears in embedding store keys, the results table, and plots. Two variants therefore never share a cache.

Published coverage, from `ESMC_SAE_COVERAGE` in [esmc_sae.py](../src/protify/base_models/esmc_sae.py):

| Backbone | Layers | Depth layer | Grid at the depth layer | Every other layer |
|----------|--------|-------------|-------------------------|-------------------|
| ESMC-300 | 0 to 30 | 23 | k in {16, 32, 64, 128, 256, 512} times codebook in {8192, 16384, 32768, 65536, 131072} | k=64, codebook 16384 only |
| ESMC-600 | 0 to 36 | 27 | same grid | k=64, codebook 16384 only |
| ESMC-6B | 0 to 80 | 60 | same grid | k=64, codebook 16384 or 131072 |

Asking for a combination Biohub does not publish raises and names the valid options rather than silently substituting one. Layer indices match `--embedding_hidden_state_index`: index `i` is the input to block `i`, and index `n_layers` is the final normalized state.

### Pooling and storage

FastPLMs owns the SAE runtime. `ESMplusplusModel.load_sae_models` attaches the Biohub checkpoint, and a forward pass with `compute_sae=True` returns one sparse COO tensor of shape `(valid_tokens, codebook_dim)` covering the whole batch. Protify reduces that tensor to one vector per protein with a single scatter, dropping the leading and trailing special tokens the way the Biohub reference workflow does.

SAE models accept `max`, `mean`, and `sum` in `--embedding_pooling_types`. Several pooling types concatenate as usual, so `max mean` yields `2 * w` features. Other pooling types, and `--matrix_embed`, raise: a single residue at codebook width 131072 costs 256 KiB dense.

Embeddings go to the store's `csr` layout, which keeps each row's exact nonzero coordinates, when the SAE they came from is sparse enough to pay for it (`k / codebook_dim` of at most 0.005), decided once from the checkpoint by `EsmcSaeForEmbedding.sparse_storage`. Every other model stores dense and pays nothing for the feature.

The deciding quantity is the active fraction `k / codebook_dim`, not either number alone. Per residue exactly `k` of `codebook_dim` features are active whatever the sequence length. Pooling unions those sets across residues, so pooled density does grow with length, and how fast it grows tracks the active fraction. Measured max-pooled density at codebook 16384, from short proteins to past 1200 residues:

| k | active fraction | pooled density | stored |
|---|---|---|---|
| 16 | 0.10% | 1.7% to 5.1% | coordinates |
| 64 | 0.39% | 9.7% to 34.2% | coordinates |
| 256 | 1.56% | 19.6% to 56.5% | dense |

Only k=256 crosses the roughly 45% break-even. The threshold sits between the k=64 and k=256 active fractions, and independently reproduces the codebook 8192 result at k=64, where the 16208 `gold-ppi` proteins split almost evenly between the formats and dense was the better single choice.

At the widths where coordinates win, they win by a lot: 200 proteins at codebook 65536 stored 3.2 MB against 26.2 MB dense, an eight-fold reduction, with every protein in coordinate form.

### Choosing k when the pooled vector has to be sparse

Widening the codebook at fixed `k` makes the pooled vector sparser; raising `k` at fixed width makes it denser, faster than width can compensate. Measured over 384 cached proteins per cell at ESMC-6B layer 60, max pooled:

| cell | entries stored | columns non-zero for at least one protein |
|---|---|---|
| k=16, c=131072 | 1.07% | 16.9% |
| k=64, c=131072 | 4.47% | 71.6% |
| k=256, c=131072 | 24.94% | 99.9% |
| k=512, c=131072 | 35.21% | 99.9% |
| k=512, c=8192 | 74.28% | 100% |

So "sparse autoencoder" describes the per-residue code, not the pooled vector. At `k=512` with codebook 131072 the pooled vector is a third full and essentially every column is live, which is why that cell stores about 237 KB per protein where `k=16` at the same width stores about 9.5 KB, a twenty-five fold difference. Pick low `k` with a wide codebook whenever the pooled vector's sparsity matters, which is whenever it reaches a tree learner or a wide feature matrix. Accuracy does not follow the same ordering, so sweep the cell per task rather than assuming the sparsest is best.

### Comparing cells in one run

Because the resolved name is the model's whole identity, a fully qualified name is a valid `--model_names` entry and several of them sweep the grid in one command, one results column each:

```bash
python -m protify.main --data_names designed-binder --probe_type xgboost \
  --embedding_pooling_types max --pair_feature_mode minmax --tune_estimator \
  --model_names ESMC-6B-SAE-l60-k16-c65536 ESMC-6B-SAE-l60-k16-c131072 \
                ESMC-6B-SAE-l60-k64-c65536 ESMC-6B-SAE-l60-k64-c131072
```

Omit `--sae_layer`, `--sae_k`, and `--sae_codebook_dim` here: they would contradict the names, and `resolve_sae_model_name` raises rather than silently picking one. Which cell suits a dataset is a per-task question with no general answer, so measure it; every cell at one layer reads the same residual stream, so one backbone pass can fill the whole grid.

### Normalization statistics

Biohub normalizes activations as `(features / max) * idf`. FastPLMs exposes this as `normalize_sae=True` and reads the `max` and `idf` buffers from the checkpoint, defaulting both to ones for shards that never trained them. Protify embeds unnormalized, matching the Biohub reference PPI workflow. Normalization is a per-feature positive rescaling, so it cannot change which features max pooling selects, only their magnitudes; it is not currently exposed as a Protify flag because it would have to reach the embedding descriptor, and so the store key, to keep the two variants separable.

---

## Pooling types

Common values for `embedding_pooling_types` (and probe-side `probe_pooling_types` where applicable):

- **mean:** Mean over sequence length (masked).
- **var:** Variance (or other stats) over sequence.
- **parti:** Requires attention outputs; see pooler implementation.
- **cls:** Use first token (CLS) representation when available.
- **eos:** Use the model-specific closing boundary token. Protify validates that each active sequence contains exactly one such token, so this works with left, right, or non-contiguous padding masks.

Pooling is applied in [pooler.py](../src/protify/pooler.py) via the `Pooler` class when not using `matrix_embed`.

### Hidden-state selection

By default Protify pools the final hidden state. Set `embedding_hidden_state_index` to any valid model hidden-state tuple index to pool an intermediate layer instead. This option participates in cache naming and W&B sweeps, so changing the layer regenerates or reloads the matching embeddings.

---

## Examples

### Preset models

```bash
python -m protify.main --model_names ESM2-8 ESM2-35 --data_names DeepLoc-2
```

### Custom model with path

```bash
python -m protify.main --model_paths "org/my-model" --model_types custom --data_names DeepLoc-2
```

Protify implements and audits CARBON's raw-DNA 6-mer path locally. It loads the stock Qwen base vocabulary from a pinned commit and does not execute CARBON or Qwen repository code. Remote CARBON model weights must still use a full immutable commit revision:

```bash
python -m protify.main --model_paths "org/carbon-model@0123456789abcdef0123456789abcdef01234567" --model_types carbon --data_names my-dna-dataset
```

CARBON uppercases DNA, preserves a final 1-5 bp remainder, and follows the published tokenizer by right-padding that final 6-mer token with `A`. With `padding=max_length`, Protify explicitly produces exact `max_length` tensors for stable compilation.

### Save and reuse embeddings

```bash
python -m protify.main --model_names ESM2-8 --data_names DeepLoc-2 --save_embeddings --embedding_pooling_types mean var
```

### Save an intermediate hidden state

```bash
python -m protify.main --model_names ESMC-300 --data_names DeepLoc-2 --save_embeddings --embedding_hidden_state_index 12
```

### Download precomputed embeddings

```bash
python -m protify.main --model_names ESM2-150 --data_names DeepLoc-2 --download_embeddings
```

### Stream rows from the store

```bash
python -m protify.main --model_names ESM2-8 --data_names DeepLoc-2 --sql --save_embeddings
```

### Convert an old cache

```bash
python -m protify.legacy_cache embeddings/ESM2-8_False_len2048_mean.db
```

---

## See also

- [Configuration](cli_and_config.md) for model and embedding CLI flags
- [Resource listing](resource_listing.md) for listing and downloading models
- [Data](data.md) for how datasets are loaded before embedding
- [Probes and training](probes_and_training.md) for how embeddings are consumed by probes
