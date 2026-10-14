# Data

This page documents how Protify loads and prepares data: `DataArguments`, supported datasets, local directories (`data_dirs`), the `get_data()` flow, column normalization, translation flags, and dataset classes. To list supported datasets from the CLI or Python, see [Resource listing](resource_listing.md).

---

## Overview

Data is specified either by **dataset names** (HuggingFace IDs or special presets like `standard_benchmark`) or by **local directories** containing split files. After loading, columns are normalized (e.g. to `seqs`/`labels` or `SeqA`/`SeqB`/`labels` for PPI), optional sequence translation is applied, and the translated output is trimmed or truncated by the target model's raw-sequence budget. Valid lowercase biological sequences are preserved and normalized to uppercase. The result is a dictionary of datasets keyed by name, each value being `(train_set, valid_set, test_set, num_labels, label_type, ppi)`.

---

## How it works

1. **DataArguments** is built from config (`data_names`, `data_dirs`, `delimiter`, `max_length`, etc.). From `data_names`, the code resolves `data_paths` (HuggingFace dataset IDs) and sets `protein_gym` when the name is `protein_gym`.
2. **DataMixin.get_data()** loads each path: for HuggingFace it uses `load_dataset(path)`; for `data_dirs` it globs split files and reads tabular files with pandas or labeled FASTA directly into HuggingFace `Dataset`.
3. **Splits:** Train is required; at least one of valid or test is required. Valid aliases are `valid`, `validation`, `val`, and `dev`; test aliases are `test` and `testing`. If valid is missing, 10% of train is used; if test is missing, 10% of train is used.
4. **process_datasets()** normalizes column names, drops missing sequence/label, optionally translates sequences, removes zero-length translated outputs, applies trim or truncation to the tokenizer-facing sequence, caps each split's rows when a [row cap](#split-row-caps) is set, and infers `label_type` (e.g. singlelabel, multilabel, regression, tokenwise).
5. For **embedding-based training**, datasets are later built from precomputed embeddings in the feature store (see [Models and embeddings](models_and_embeddings.md#the-feature-store)) via `build_vector_numpy_dataset_from_embeddings` or the PPI/multi-column variants.

---

## DataArguments

Defined in [data_mixin.py](../src/protify/data/data_mixin.py). All arguments that affect data loading and preprocessing:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `data_names` | List[str] | (required for HF) | Dataset names. Can be keys from `supported_datasets`, `standard_benchmark`, `vector_benchmark`, or literal HuggingFace IDs. |
| `data_dirs` | Optional[List[str]] | [] | Local directories; each must contain `train.*` and at least one valid/test split (CSV, TSV, Excel, or labeled FASTA). |
| `delimiter` | str | ',' | Delimiter when loading from `data_dirs`. |
| `col_names` | List[str] | ['seqs', 'labels'] | Column names (legacy; columns are often inferred from the data). |
| `max_length` | int | 1024 | Maximum sequence length for trim/truncation. |
| `trim` | bool | False | If True, drop rows exceeding `max_length`; if False, truncate to `max_length`. |
| `multi_column` | Optional[List[str]] | None | Names of sequence columns for multi-input tasks (e.g. PPI mutation effect). |
| `aa_to_dna` | bool | False | Translate amino acid sequences to DNA codons. |
| `aa_to_rna` | bool | False | Translate amino acid sequences to RNA. |
| `dna_to_aa` | bool | False | Translate DNA to amino acid. |
| `rna_to_aa` | bool | False | Translate RNA to amino acid. |
| `codon_to_aa` | bool | False | Map codon tokens to amino acid. |
| `aa_to_codon` | bool | False | Map amino acid to codon tokens. |
| `max_train_rows`, `max_valid_rows`, `max_test_rows` | Optional[int] | None | Most rows each split of each dataset keeps. A larger split keeps a deterministic subsample of exactly that many rows; see [Split row caps](#split-row-caps). |
| `split_row_cap_seed` | int | 0 | Seed of those subsamples, independent of the global and training seeds. |

Only one of the translation flags may be True. Setting `data_names[0]` to `standard_benchmark` or `vector_benchmark` expands to the corresponding list in [supported_datasets.py](../src/protify/data/supported_datasets.py).

---

## Supported datasets and benchmark lists

- **supported_datasets:** A dict mapping dataset name to HuggingFace dataset ID or path (e.g. `'EC': 'GleghornLab/EC_reg'`).
- **standard_data_benchmark:** List of dataset names used when `data_names=['standard_benchmark']`.
- **vector_benchmark:** List used when `data_names=['vector_benchmark']`.
- **possible_with_vector_reps:** Datasets that can be used with precomputed vector representations.
- **testing:** Small set used in tests.

To list datasets and get info from the CLI:

```bash
py -m src.protify.data.dataset_utils --list
py -m src.protify.data.dataset_utils --info EC
```

See [Resource listing](resource_listing.md) for programmatic access and combined model/dataset listing.

---

## data_dirs and file layout

For each directory in `data_dirs`, `get_data()` looks for files whose stem matches:

- `train.*`
- `valid.*`, `validation.*`, `val.*`, or `dev.*`
- `test.*` or `testing.*`

Tabular files are read with pandas (`read_csv` or `read_excel` by extension) and converted to HuggingFace `Dataset`. FASTA files with extensions `.fa`, `.fasta`, `.faa`, `.fna`, `.ffn`, or `.frn` are supported for supervised single-sequence datasets when every header has explicit `label=` metadata, for example `>seq_001 label=1`. Column names are inferred (e.g. sequence column as `seqs` or `Seq`/`sequence`, label as `labels` or `label`). PPI data is normalized to `SeqA`, `SeqB`, `labels`.

---

## Column normalization and label_type

- **Single-sequence:** Columns become `seqs` and `labels` (or `multi_column` names plus `labels`).
- **PPI:** Columns become `SeqA`, `SeqB`, `labels`.
- **label_type** is inferred: e.g. `singlelabel`, `multilabel`, `regression`, `sigmoid_regression`, `tokenwise`, `string`. This drives the loss and metrics during training.

---

## Split row caps

`max_train_rows`, `max_valid_rows`, and `max_test_rows` (`--max_train_rows` and so on) bound how many rows each split of each dataset keeps, which bounds embedding and training cost on a large dataset. A split with more rows than its cap keeps exactly the cap:

- Integer-valued scalar labels are classes, as label typing reads them, and each class keeps its share, rounded by largest remainder so every class count is within one row of its exact share. Other numeric scalars are regression targets and keep each decile of the split's own labels in proportion. List labels (multilabel) and strings (tokenwise) are drawn uniformly.
- Kept rows stay in their original order, and a split at or under its cap is untouched, so an uncapped dataset keeps identical rows.
- The draw uses `numpy.random.default_rng`, seeded by `split_row_cap_seed` and a CRC-32 of the split name, never by the global seed or Python's per-process `hash`. The same rows therefore survive every `--seed`, a rerun in another process, and loading the dataset alone or beside other datasets.
- The cap applies after the missing-value, sanitization, translation, empty-sequence, and length steps, so it counts rows that survive them, and before `all_seqs` is collected and `label_type` is inferred, so no dropped row is ever embedded.

A dataset without a validation or test split first has one carved from train with the global seed (42 when none is set), so for that dataset the kept rows also follow the global seed.

Each capped split prints one line, `<dataset> <split>: capped <before> rows to <after> (split_row_cap_seed <seed>)`. `main.py`, [precompute_sae_and_dense.py](models_and_embeddings.md#filling-many-caches-from-one-pass), and `python -m protify.uncertainty` (through `protify_args`) all read data through `DataMixin`, so the same caps give each of them the same rows, and a precompute run with the training runs' caps fills exactly the sequences they read.

---

## Dataset classes

Used when building PyTorch datasets from embeddings or raw sequences ([dataset_classes.py](../src/protify/data/dataset_classes.py)):

| Class | Use case |
|-------|----------|
| **EmbedsLabelsDataset** | Single-sequence, embeddings from in-memory dict. |
| **EmbedsLabelsDatasetFromDisk** | Single-sequence, embeddings streamed from the feature store (batched reads through one `FeatureReader`). |
| **PairEmbedsLabelsDataset** | PPI, embeddings from in-memory dict. |
| **PairEmbedsLabelsDatasetFromDisk** | PPI, embeddings streamed from the feature store; optional pair flipping. |
| **MultiEmbedsLabelsDataset** / **MultiEmbedsLabelsDatasetFromDisk** | Multi-column sequence inputs, embeddings from dict or streamed from the feature store. |
| **StringLabelDataset** | Single-sequence, raw sequences and labels. |
| **PairStringLabelDataset** | PPI, raw pairs and labels; optional pair flipping. |
| **SimpleProteinDataset** | Wrapper over a list of sequence strings (e.g. for embedding). |

Collators in [data_collators.py](../src/protify/data/data_collators.py) pair with these for batching (e.g. `EmbedsLabelsCollator`, `PairEmbedsLabelsCollator`, `StringLabelsCollator`).

---

## prepare_scikit_dataset

`DataMixin.prepare_scikit_dataset(model_name, dataset)` builds `X_train`, `y_train`, `X_valid`, `y_valid`, `X_test`, `y_test` and `label_type` from embedding-backed datasets, for use with the scikit-learn path (`run_scikit_scheme`).

---

## Examples

### Single dataset by name

```yaml
data_names: [DeepLoc-2]
max_length: 1024
trim: false
```

### Standard benchmark (many datasets)

```yaml
data_names: [standard_benchmark]
```

### Local directory

```bash
py -m src.protify.main --data_dirs path/to/my_data --delimiter "," --model_names ESM2-8 --data_names []
```

Ensure `path/to/my_data` contains `train.csv` and at least one valid/test split such as `validation.csv` or `testing.tsv`.

### Local FASTA directory

```text
train.fasta
validation.fasta
test.fasta
```

Each record must include `label=` in the header:

```text
>seq_001 label=1
MKTAYIAKQRQISFVKSHFSRQ
```

### Translation (DNA to amino acid)

```yaml
data_names: [my_dna_dataset]
dna_to_aa: true
```

---

## See also

- [Configuration](cli_and_config.md) for data-related CLI flags
- [Resource listing](resource_listing.md) for listing datasets and programmatic access
- [Models and embeddings](models_and_embeddings.md) for how embeddings are built from sequences
- [Probes and training](probes_and_training.md) for how datasets are used in training
