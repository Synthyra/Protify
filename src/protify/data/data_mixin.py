import ast
import os
import random
import re
import zlib
import numpy as np
import torch

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from glob import glob
from typing import TYPE_CHECKING, Any
from datasets import Dataset, DatasetDict, load_dataset
from pandas import read_csv, read_excel

from .cache_reader import cached_rows
from .dataset_classes import EmbeddingStandardizer, read_cached_row
from .feature_matrix import FeatureMatrix, matrix_density, stack_pooled_rows
from .pair_features import DEFAULT_PAIR_FEATURE_MODE, pair_order_matters, pair_vector_halves
from .sequence_alphabets import (
    AA_SET,
    CODON_SET,
    DNA_SET,
    RNA_SET,
    NONCANONICAL_AMINO_ACIDS,
    AMINO_ACID_TO_HUMAN_CODON,
    NONCANONICAL_ALANINE_CODON,
    AA_TO_CODON_TOKEN,
    CODON_TO_AA,
    DNA_CODON_TO_AA,
    RNA_CODON_TO_AA,
)
from .supported_datasets import (
    get_dataset_source,
    resolve_dataset_name,
    rigorous_benchmark,
    standard_data_benchmark,
    supported_datasets,
    vector_benchmark,
)
from ..console_output import print_message
from ..feature_cache import open_reader, read_embeddings, require_feature
from ..metrics_balanced import compute_sample_weights, apply_weights_from_reference, _default_bin_borders
from ..seed_utils import get_global_seed


FASTA_EXTENSIONS = {'.fa', '.fasta', '.faa', '.fna', '.ffn', '.frn'}
SPLIT_ALIASES = {
    'train': ('train',),
    'valid': ('valid', 'validation', 'val', 'dev'),
    'test': ('test', 'testing'),
    # Optional: a dataset that has one gets a calibration report (`calibration_report.py`).
    'calibration': ('calibration', 'calib', 'cal'),
}
# The names a dataset's label and sequence columns go by, the first present winning.
LABEL_COLUMNS = ('labels', 'label', 'Labels', 'Label')
SEQUENCE_COLUMNS = ('seqs', 'Seqs', 'seq', 'Seq', 'sequence', 'Sequence', 'sequences', 'Sequences')
# Dataset names that do not resolve to a single Hub dataset `get_data` loads.
UNPINNABLE_DATA_NAMES = frozenset({'standard_benchmark', 'vector_benchmark', 'rigorous_benchmark', 'protein_gym'})
# A capped regression split keeps each of this many label quantiles in its share, deciles.
REGRESSION_CAP_STRATA = 10


def _to_numpy(embedding: torch.Tensor) -> np.ndarray:
    """Convert an embedding to numpy, widening bfloat16 which numpy cannot represent."""
    # embedding: (d,) or (l, d) per sequence
    if embedding.dtype == torch.bfloat16:
        embedding = embedding.float()
    return embedding.numpy()  # (d,) or (l, d), as numpy


def _as_row(embedding: np.ndarray, full: bool) -> np.ndarray:
    """Reduce one cached embedding to a single row, so concatenation yields (n, d).

    Both raw readers preserve stored rank, so pooled inputs may be (d,) or (1, d).
    Matrix caches instead hold (l, d) per residue, where l counts
    the stored tokens including the leading and trailing special tokens, so l is len(seq)+2
    rather than len(seq). These learners take one vector per sequence, so a matrix averages
    over its own rows rather than being reshaped against the sequence length.
    """
    # embedding: (d,) or (1, d) pooled, or (l, d) per-residue
    return embedding.mean(axis=0, keepdims=True) if full else embedding.reshape(1, -1)  # (1, d)


if TYPE_CHECKING:
    from fastplms.features import FeatureReader
    from ..probes.feature_path_selector import FeatureSelection

# Each dataset's splits, label width, label type, and PPI flag, then the data names in the same order.
ProcessedDatasets = tuple[dict[str, tuple[Dataset, Dataset, Dataset, int, str, bool]], list[str]]


def _narrow(row: np.ndarray, feature_selection: "FeatureSelection | None") -> np.ndarray:
    """Keep only the selected columns of one materialized row.

    Applied per row rather than to the stacked matrix on purpose. At the sparse
    autoencoder widths this exists for, the full-width matrix these builders would
    otherwise return does not fit in host memory, so narrowing has to happen before the
    rows are stacked rather than after.
    """
    # row: (1, d) -> (1, s)
    return row if feature_selection is None else feature_selection.transform_numpy(row)  # (1, s) when columns are selected, else (1, d)


def _print_split_shapes(splits: list[FeatureMatrix]) -> None:
    """Report the stacked train, validation, and test widths, and how full they are."""
    for name, features in zip(('Train', 'Valid', 'Test'), splits, strict=True):
        print_message(
            f'{name}: {features.shape}, {100 * matrix_density(features):.1f}% non-zero'
        )


def split_row_strata(labels: Sequence[object]) -> np.ndarray | None:
    """The stratum each row of a split is drawn from, or None when rows are drawn uniformly.

    Integer-valued scalar labels are classes, as `_not_regression` reads them, so each class
    keeps its share of a capped split. Other numeric scalars are regression targets, binned
    by decile of the split's own labels so both tails survive the cap. Lists (multilabel),
    strings (tokenwise), and anything else give a row no single stratum.
    """
    if not all(isinstance(label, (int, float, np.integer, np.floating)) for label in labels):
        return None

    if all(float(label).is_integer() for label in labels):
        return np.array([int(label) for label in labels], dtype=np.int64)  # (n,) class of each row

    values = np.asarray(labels, dtype=np.float64)  # (n,)
    inner_edges = np.quantile(
        values, np.arange(1, REGRESSION_CAP_STRATA) / REGRESSION_CAP_STRATA
    )  # (9,) boundaries between deciles
    return np.searchsorted(inner_edges, values, side='right')  # (n,) decile of each row, 0 to 9


def stratified_quotas(stratum_sizes: np.ndarray, cap: int) -> np.ndarray:
    """Rows each stratum keeps: its exact share of `cap`, rounded by largest remainder.

    Integer arithmetic throughout, so the quotas sum to exactly `cap` and each lies within
    one row of its exact share `cap * size / total`. Equal remainders go to the stratum
    listed first, which keeps the draw deterministic.
    """
    # stratum_sizes: (s,) rows per stratum
    total = int(stratum_sizes.sum())
    shares = stratum_sizes.astype(np.int64) * cap  # (s,) exact shares, scaled by total
    quotas = shares // total  # (s,)
    remainders = shares % total  # (s,)
    shortfall = cap - int(quotas.sum())
    largest_remainder_first = np.argsort(-remainders, kind='stable')  # (s,)
    quotas[largest_remainder_first[:shortfall]] += 1
    return quotas  # (s,)


def capped_row_indices(labels: Sequence[object], cap: int, seed: int, split_name: str) -> np.ndarray:
    """Indices of the `cap` rows a split keeps, ascending so kept rows keep their order.

    The generator is seeded by `seed` and the split's name alone, never by the global or a
    training seed, so every run and every consumer that caps these rows keeps the same
    ones. zlib.crc32 is stable across processes, where Python's `hash` is salted per process.
    """
    generator = np.random.default_rng([seed, zlib.crc32(split_name.encode('utf-8'))])
    strata = split_row_strata(labels)  # (n,) or None
    if strata is None:
        kept = generator.choice(len(labels), size=cap, replace=False)  # (cap,)
        return np.sort(kept)  # (cap,)

    _, stratum_of_row, stratum_sizes = np.unique(
        strata, return_inverse=True, return_counts=True
    )  # (n,) and (s,), strata in sorted order
    # A stable sort groups each stratum's rows while keeping them in row order.
    rows_by_stratum = np.split(
        np.argsort(stratum_of_row, kind='stable'), np.cumsum(stratum_sizes)[:-1]
    )  # s arrays of row indices
    quotas = stratified_quotas(stratum_sizes, cap)  # (s,)
    kept = np.concatenate([
        generator.choice(rows, size=quota, replace=False)
        for rows, quota in zip(rows_by_stratum, quotas, strict=True)
    ])  # (cap,)
    return np.sort(kept)  # (cap,)


def _checked_row_cap(name: str, cap: int | None) -> int | None:
    """A split row cap as given, or a ValueError for one that is not a positive row count."""
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
        raise ValueError(f'{name} must be a positive number of rows; received {cap!r}.')
    return cap


@dataclass
class DataArguments:
    """
    Args:
    data_paths: List[str]
        paths to the datasets
    data_revisions: list[str] | None
        one immutable Hugging Face revision per entry of data_names, so a run reads
        exactly the dataset commit it names instead of whatever the Hub serves today
    max_length: int
        maximum tokenizer length. DataMixin may convert this to a model-specific
        raw sequence limit before filtering or truncating.
    trim: bool
        whether to drop sequences that exceed max_length instead of truncating
    max_train_rows, max_valid_rows, max_test_rows: int | None
        most rows each split of each dataset keeps. A larger split keeps a deterministic
        subsample of exactly this many rows, stratified by class or regression decile; a
        split at or under its cap is untouched. None keeps every row.
    split_row_cap_seed: int
        seed of those subsamples, separate from the global and training seeds so every
        training seed and every consumer of DataMixin reads the same rows
    """
    def __init__(
            self,
            data_names: list[str],
            data_revisions: list[str] | None = None,
            delimiter: str = ',',
            col_names: Sequence[str] = ('seqs', 'labels'),
            max_length: int = 1024,
            trim: bool = False,
            data_dirs: Sequence[str] | None = (),
            multi_column: list[str] | None = None,
            aa_to_dna: bool = False,
            aa_to_rna: bool = False,
            dna_to_aa: bool = False,
            rna_to_aa: bool = False,
            codon_to_aa: bool = False,
            aa_to_codon: bool = False,
            max_train_rows: int | None = None,
            max_valid_rows: int | None = None,
            max_test_rows: int | None = None,
            split_row_cap_seed: int = 0,
            **kwargs: Any
        ) -> None:
        self.data_names = data_names
        if data_revisions is not None:
            if len(data_revisions) != len(data_names):
                raise ValueError(
                    f'data_revisions must align one for one with data_names; received '
                    f'{len(data_revisions)} revisions for {len(data_names)} datasets.'
                )
            # A benchmark alias expands to several datasets and ProteinGym loads through its
            # own path, so a revision given for either would pin nothing.
            unpinnable = [name for name in data_names if name in UNPINNABLE_DATA_NAMES]
            if unpinnable:
                raise ValueError(f'data_revisions pin one Hub dataset each; {unpinnable} name none.')
        self.data_revisions = data_revisions
        self.data_dirs = data_dirs
        self.delimiter = delimiter
        self.col_names = col_names
        self.max_length = max_length
        self.trim = trim
        self.protein_gym = False
        self.multi_column = multi_column
        self.aa_to_dna = aa_to_dna
        self.aa_to_rna = aa_to_rna
        self.dna_to_aa = dna_to_aa
        self.rna_to_aa = rna_to_aa
        self.codon_to_aa = codon_to_aa
        self.aa_to_codon = aa_to_codon
        self.max_train_rows = _checked_row_cap('max_train_rows', max_train_rows)
        self.max_valid_rows = _checked_row_cap('max_valid_rows', max_valid_rows)
        self.max_test_rows = _checked_row_cap('max_test_rows', max_test_rows)
        # numpy's SeedSequence takes non-negative integers only.
        if isinstance(split_row_cap_seed, bool) or not isinstance(split_row_cap_seed, int) or split_row_cap_seed < 0:
            raise ValueError(f'split_row_cap_seed must be a non-negative integer; received {split_row_cap_seed!r}.')
        self.split_row_cap_seed = split_row_cap_seed

        if len(data_names) > 0:
            if data_names[0] == 'standard_benchmark':
                self.data_paths = [get_dataset_source(data_name) for data_name in standard_data_benchmark]
            elif data_names[0] == 'vector_benchmark':
                self.data_paths = [get_dataset_source(data_name) for data_name in vector_benchmark]
            elif data_names[0] == 'rigorous_benchmark':
                self.data_paths = [get_dataset_source(data_name) for data_name in rigorous_benchmark]
            else:
                self.data_paths = []
                for data_name in data_names:
                    if data_name == 'protein_gym':
                        # For special handling in the main workflow
                        self.protein_gym = True
                        continue
                    resolved_name = resolve_dataset_name(data_name)
                    if resolved_name in supported_datasets:
                        self.data_paths.append(get_dataset_source(resolved_name))
                    else:
                        print(f'{data_name} not found in supported datasets')
                        print('We will attempt to load it from huggingface anyways, but this may not work')
                        self.data_paths.append(data_name)
        else:
            self.data_paths = []
        # With revisions given, every name resolved to exactly one path, so they align here too.
        self.revision_by_path = {}
        if data_revisions is not None:
            self.revision_by_path = dict(zip(self.data_paths, data_revisions, strict=True))

        if data_dirs is not None:
            for dir in data_dirs:
                if not os.path.exists(dir):
                    raise FileNotFoundError(f'{dir} does not exist')

    @property
    def split_row_caps(self) -> dict[str, int | None]:
        """Each split's row cap, under the split name its subsample is seeded by."""
        return {'train': self.max_train_rows, 'valid': self.max_valid_rows, 'test': self.max_test_rows}


class DataMixin:
    def __init__(self, data_args: DataArguments | None = None) -> None:
        # intialize defaults
        self.balanced_weights: dict[str, dict] = {}
        self._sql = False
        self._full = False
        self._pair_feature_mode = DEFAULT_PAIR_FEATURE_MODE
        self._max_length = 1024
        self._data_max_length = 1024
        self._trim = False
        self._delimiter = ','
        self._col_names = ['seqs', 'labels']
        self._aa_to_dna = False
        self._aa_to_rna = False
        self._dna_to_aa = False
        self._rna_to_aa = False
        self._codon_to_aa = False
        self._aa_to_codon = False
        self.data_args = data_args
        self._multi_column = None if data_args is None else getattr(data_args, 'multi_column', None)
        if data_args is not None:
            self._max_length = data_args.max_length
            self._data_max_length = data_args.max_length
            self._trim = data_args.trim
            self._aa_to_dna = data_args.aa_to_dna
            self._aa_to_rna = data_args.aa_to_rna
            self._dna_to_aa = data_args.dna_to_aa
            self._rna_to_aa = data_args.rna_to_aa
            self._codon_to_aa = data_args.codon_to_aa
            self._aa_to_codon = data_args.aa_to_codon

    def _not_regression(self, labels: list[Any]) -> bool: # not a great assumption but works most of the time
        if isinstance(labels, list):
            # Check if first element is itself a list (multilabel case)
            if isinstance(labels[0], list):
                # For multilabel: check all elements in all sublists
                return all(isinstance(element, (int, float)) and element == int(element) 
                          for label in labels for element in label)
            else:
                # For single label: check all elements in the list
                return all(isinstance(label, (int, float)) and label == int(label) 
                          for label in labels)
        else:
            # Fallback for non-list input
            return all(isinstance(label, (int, float)) and label == int(label) for label in labels)

    def _embedding_identity(self, model_name: str) -> tuple[str | None, str | None]:
        """The dispatch keyword and model path `model_args` gives a display name, else `(None, None)`.

        Callers pass the display name. Recovering the dispatch keyword and path from
        `model_args` is what lets a path-mode CARBON entry resolve to the same revision the
        Embedder resolved. Without `model_args`, which is the standalone DataMixin case,
        the display name is the dispatch keyword.
        """
        model_args = getattr(self, 'model_args', None)
        if model_args is not None:
            for display_name, dispatch_name, candidate_path in model_args.model_entries():
                if model_name in (display_name, dispatch_name):
                    return dispatch_name, candidate_path
        return None, None

    def _embedding_cache_filename(self, model_name: str, extension: str) -> str:
        """Name the file an earlier Protify wrote this cache to: what the converter and the Hub download use.

        The cache itself is a feature store now, at `_embedding_store`. The name is built by the same
        arguments the Embedder uses, so the two cannot drift.
        """
        model_type, model_path = self._embedding_identity(model_name)
        return self.embedding_args.cache_filename(
            model_name,
            matrix_embed=self._full,
            extension=extension,
            model_type=model_type,
            model_path=model_path,
        )

    def _embedding_store(self, model_name: str) -> str:
        """The feature store directory that serves this model's cached embeddings, present or not.

        The Embedder writes it and every reader opens it, from the same arguments, so they cannot
        drift: a descriptor of the model, pooling, hidden state, max length, and dtype names it.
        """
        model_type, model_path = self._embedding_identity(model_name)
        return self.embedding_args.feature_directory(
            model_name, matrix_embed=self._full, model_type=model_type, model_path=model_path,
        )

    def _legacy_embedding_files(self, model_name: str) -> list[str]:
        """The old-format cache files of this model that still sit in the embedding folder.

        Nothing reads them. They are named so a missing store can say which file to convert.
        """
        model_type, model_path = self._embedding_identity(model_name)
        return self.embedding_args.legacy_cache_files(
            model_name, matrix_embed=self._full, model_type=model_type, model_path=model_path,
        )

    def _embedding_dict(self, model_name: str, sequences: Iterable[str]) -> dict[str, torch.Tensor]:
        """This model's rows for `sequences`, read once into memory: `(d,)` pooled or `(l, d)` per token.

        The in-memory counterpart of streaming with `--sql`, and what `emb_dict` holds. A sequence the
        store lacks raises `KeyError`; a missing store raises `FileNotFoundError` saying how to make it.
        """
        directory = require_feature(self._embedding_store(model_name), self._legacy_embedding_files(model_name))
        return read_embeddings(directory, sequences)  # (d,) pooled or (l + 2, d) token rows per sequence

    def _encode_labels(self, labels: list[Any], tag2id: dict[str, int]) -> list[torch.Tensor]:
        return [torch.tensor([tag2id[tag] for tag in doc], dtype=torch.long) for doc in labels]  # (l_i,) label ids per sequence

    def _label_type_checker(self, labels: list[Any]) -> str:
        ex = labels[0]
        assert len(labels) > 0, f'Labels is empty: {labels}'
        if self._not_regression(labels):
            if isinstance(ex, list):
                label_type = 'multilabel'
            else:
                label_type = 'singlelabel' # binary or multiclass
        elif isinstance(ex, str):
            label_type = 'string'
        else:
            label_type = 'regression'
        return label_type

    def _is_sigmoid_regression(self, labels: list[Any]) -> bool:
        """Heuristic: labels within [0, 1] and cover the range approximately.
        Uses 10-bin histogram coverage and span threshold.
        """
        arr = []
        for label in labels:
            try:
                arr.extend(label)
            except TypeError:
                arr.append(label)
        arr = np.array(arr, dtype=float).flatten()        

        min_val, max_val = float(arr.min()), float(arr.max())
        cond1 = min_val > 0.0 - 1e-6 and max_val < 1.0 + 1e-6

        # Require substantial span across [0,1]
        cond2 = (max_val - min_val) > 0.75

        # Histogram coverage: at least 7 of 10 bins non-empty
        hist, _ = np.histogram(arr, bins=10, range=(0.0, 1.0))
        cond3 = int((hist > 0).sum()) >= 7

        sigmoid_regression_status = cond1 and cond2 and cond3
        return sigmoid_regression_status

    def _select_from_store(self, reader: "FeatureReader", seq: str, cast_to_torch: bool = True) -> torch.Tensor:
        """One row from an open store reader: `(d,)` pooled or `(l, d)` per token, as stored."""
        embedding = read_cached_row(reader, seq)
        if not cast_to_torch:
            embedding = _to_numpy(embedding)
        return embedding  # (d,) pooled or (l + 2, d) token rows

    def _select_from_pth(self, emb_dict: dict[str, torch.Tensor], seq: str, cast_to_np: bool = False) -> torch.Tensor:
        # Matrix caches store (l, d) with the special tokens kept, so l is not len(seq);
        # only the pooled case has a shape worth asserting here.
        # emb_dict: (d,) pooled or (l + 2, d) token rows per sequence
        embedding = emb_dict[seq]
        if not self._full:
            embedding = embedding.reshape(1, -1)
        if cast_to_np:
            embedding = _to_numpy(embedding)
        return embedding  # (1, d) pooled or (l + 2, d) token rows

    def _labels_to_numpy(self, labels: list[Any]) -> np.ndarray:
        if isinstance(labels[0], list):
            return np.array(labels).flatten()  # (n * k,) from n rows of k labels
        else:
            return np.array([labels]).flatten()  # (n,)

    def _random_order(self, seq_a: str, seq_b: str) -> tuple[str, str]:
        if random.random() < 0.5:
            return seq_a, seq_b
        else:
            return seq_b, seq_a

    def _truncate_pairs(self, ex: dict[str, str]) -> dict[str, str]:
        # Truncate longest first, but if that makes it shorter than the other, truncate that one
        seq_a, seq_b = ex['SeqA'], ex['SeqB']
        trunc_a, trunc_b = seq_a, seq_b
        while len(trunc_a) + len(trunc_b) > self._data_max_length:
            if len(trunc_a) > len(trunc_b):
                trunc_a = trunc_a[:-1]
            else:
                trunc_b = trunc_b[:-1]
        ex['SeqA'] = trunc_a
        ex['SeqB'] = trunc_b
        return ex

    def _canonical_split_for_name(self, split_name: str) -> str | None:
        lowered = split_name.lower()
        for canonical_split, aliases in SPLIT_ALIASES.items():
            if lowered in aliases:
                return canonical_split
        return None

    def _resolve_split_name(self, available_splits: list[str], canonical_split: str) -> str | None:
        lowercase_to_actual = {split.lower(): split for split in available_splits}
        for alias in SPLIT_ALIASES[canonical_split]:
            if alias in lowercase_to_actual:
                return lowercase_to_actual[alias]
        return None

    def _select_train_valid_test_splits(self, dataset: DatasetDict, data_name: str) -> tuple[Dataset, Dataset, Dataset]:
        available_splits = list(dataset.keys())
        train_name = self._resolve_split_name(available_splits, 'train')
        valid_name = self._resolve_split_name(available_splits, 'valid')
        test_name = self._resolve_split_name(available_splits, 'test')

        assert train_name is not None, f'{data_name} does not have a train set'
        assert valid_name is not None or test_name is not None, (
            f'{data_name} does not have a valid or test set, needs at least one'
        )

        seed = get_global_seed() if get_global_seed() is not None else 42
        if valid_name is None:
            train_set = dataset[train_name]
            train_valid_set = train_set.train_test_split(test_size=0.1, seed=seed + 1)
            print_message(f'{data_name} does not have a valid set, created a 10% validation set')
            return train_valid_set['train'], train_valid_set['test'], dataset[test_name]
        if test_name is None:
            train_set = dataset[train_name]
            train_test_set = train_set.train_test_split(test_size=0.1, seed=seed + 2)
            print_message(f'{data_name} does not have a test set, created a 10% test set')
            return train_test_set['train'], dataset[valid_name], train_test_set['test']

        print_message(f'{data_name} has a valid and test set')
        return dataset[train_name], dataset[valid_name], dataset[test_name]

    def _is_fasta_path(self, file_path: str) -> bool:
        suffix = os.path.splitext(file_path)[1].lower()
        return suffix in FASTA_EXTENSIONS

    def _coerce_fasta_label(self, label: str) -> str | int | float | list[Any]:
        if re.fullmatch(r'-?\d+', label) is not None:
            return int(label)
        if re.fullmatch(r'-?((\d+\.\d*)|(\.\d+)|(\d+))([eE][+-]?\d+)?', label) is not None:
            return float(label)
        if label.startswith('[') and label.endswith(']'):
            return ast.literal_eval(label)
        return label

    def _label_from_fasta_header(self, header: str) -> str | int | float | list[Any]:
        match = re.search(r'(?:^|[\s|;,])label=([^\s|;,]+)', header)
        assert match is not None, f'FASTA header is missing explicit label= metadata: {header}'
        return self._coerce_fasta_label(match.group(1))

    def _read_fasta_file(self, file_path: str) -> Dataset:
        sequences, labels = [], []
        header = None
        seq_chunks = []

        def _flush_record() -> None:
            if header is None:
                return
            sequence = ''.join(seq_chunks)
            assert len(sequence) > 0, f'FASTA record has no sequence: {header}'
            sequences.append(sequence)
            labels.append(self._label_from_fasta_header(header))

        with open(file_path, encoding='utf-8') as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if len(line) == 0:
                    continue
                if line.startswith('>'):
                    _flush_record()
                    header = line[1:].strip()
                    seq_chunks = []
                else:
                    assert header is not None, f'FASTA sequence line found before header in {file_path}'
                    seq_chunks.append(line)
            _flush_record()

        assert len(sequences) > 0, f'No FASTA records found in {file_path}'
        return Dataset.from_dict({'seqs': sequences, 'labels': labels})

    def _read_local_split_file(self, file_path: str) -> Dataset:
        suffix = os.path.splitext(file_path)[1].lower()
        if self._is_fasta_path(file_path):
            return self._read_fasta_file(file_path)
        if suffix in {'.xlsx', '.xls'}:
            frame = read_excel(file_path)
        elif suffix == '.tsv':
            frame = read_csv(file_path, delimiter='\t')
        else:
            frame = read_csv(file_path, delimiter=self._delimiter)
        return Dataset.from_pandas(frame)

    def _load_local_dataset_dir(self, data_dir: str) -> dict[str, Dataset]:
        split_files = {}
        for file_path in glob(os.path.join(data_dir, '*')):
            if not os.path.isfile(file_path):
                continue
            stem = os.path.splitext(os.path.basename(file_path))[0]
            canonical_split = self._canonical_split_for_name(stem)
            if canonical_split is None:
                continue
            assert canonical_split not in split_files, (
                f'Multiple files matched {canonical_split} split in {data_dir}: '
                f'{split_files[canonical_split]} and {file_path}'
            )
            split_files[canonical_split] = file_path

        assert 'train' in split_files, f'{data_dir} does not contain a train split file'
        assert 'valid' in split_files or 'test' in split_files, (
            f'{data_dir} does not contain a valid/validation/val/dev or test/testing split file'
        )
        dataset = {
            split_name: self._read_local_split_file(file_path)
            for split_name, file_path in split_files.items()
        }
        train_set, valid_set, test_set = self._select_train_valid_test_splits(dataset, os.path.basename(os.path.normpath(data_dir)))
        return self._with_calibration_split({'train': train_set, 'valid': valid_set, 'test': test_set}, dataset)

    def _active_translation_mode(self) -> str | None:
        mode_to_flag = {
            'aa_to_dna': self._aa_to_dna,
            'aa_to_rna': self._aa_to_rna,
            'dna_to_aa': self._dna_to_aa,
            'rna_to_aa': self._rna_to_aa,
            'codon_to_aa': self._codon_to_aa,
            'aa_to_codon': self._aa_to_codon,
        }
        active_modes = [mode for mode, enabled in mode_to_flag.items() if enabled]
        assert len(active_modes) <= 1, f'Only one translation mode can be enabled at a time, found: {active_modes}'
        return active_modes[0] if len(active_modes) == 1 else None

    def _assert_characters_in_set(self, seq: str, allowed_chars: set[str], mode: str) -> None:
        bad_chars = sorted({char for char in seq if char.upper() not in allowed_chars})
        assert len(bad_chars) == 0, f'Invalid characters for {mode}: {bad_chars}.'

    def _validate_translated_output(self, translated_seq: str, allowed_chars: set[str], mode: str) -> None:
        bad_chars = sorted({char for char in translated_seq if char not in allowed_chars})
        assert len(bad_chars) == 0, f'Translation output for {mode} contains unexpected characters: {bad_chars}.'

    def _normalize_aa_for_nucleotide_translation(self, seq: str) -> str:
        canonical_aas = set(AMINO_ACID_TO_HUMAN_CODON.keys())
        normalized = []
        for residue in seq:
            residue = residue.upper()
            if residue in canonical_aas:
                normalized.append(residue)
            else:
                normalized.append('X')
        return ''.join(normalized)

    def _translate_aa_to_dna(self, seq: str) -> str:
        seq = self._normalize_aa_for_nucleotide_translation(seq)
        dna_codons: list[str] = []
        for residue in seq:
            residue = residue.upper()
            if residue in AMINO_ACID_TO_HUMAN_CODON:
                dna_codons.append(AMINO_ACID_TO_HUMAN_CODON[residue])
            elif residue in NONCANONICAL_AMINO_ACIDS:
                dna_codons.append(NONCANONICAL_ALANINE_CODON)
            else:
                raise AssertionError(f'Unexpected amino acid token "{residue}" while converting aa_to_dna.')
        translated = ''.join(dna_codons)
        self._validate_translated_output(translated, DNA_SET, 'aa_to_dna')
        return translated

    def _translate_aa_to_rna(self, seq: str) -> str:
        dna_translated = self._translate_aa_to_dna(seq)
        translated = dna_translated.replace('T', 'U')
        self._validate_translated_output(translated, RNA_SET, 'aa_to_rna')
        return translated

    def _translate_dna_to_aa(self, seq: str) -> str:
        dna_seq = seq.upper()
        self._assert_characters_in_set(dna_seq, DNA_SET, 'dna_to_aa')
        assert len(dna_seq) % 3 == 0, f'dna_to_aa requires sequence length multiple of 3, got {len(dna_seq)}.'
        aa_seq = []
        for idx in range(0, len(dna_seq), 3):
            codon = dna_seq[idx:idx + 3]
            assert codon in DNA_CODON_TO_AA, f'Unknown DNA codon for dna_to_aa: {codon}'
            translated_char = DNA_CODON_TO_AA[codon]
            if translated_char != '*':
                aa_seq.append(translated_char)
        translated = ''.join(aa_seq)
        self._validate_translated_output(translated, AA_SET - {'*'}, 'dna_to_aa')
        return translated

    def _translate_rna_to_aa(self, seq: str) -> str:
        rna_seq = seq.upper()
        self._assert_characters_in_set(rna_seq, RNA_SET, 'rna_to_aa')
        assert len(rna_seq) % 3 == 0, f'rna_to_aa requires sequence length multiple of 3, got {len(rna_seq)}.'
        aa_seq = []
        for idx in range(0, len(rna_seq), 3):
            codon = rna_seq[idx:idx + 3]
            assert codon in RNA_CODON_TO_AA, f'Unknown RNA codon for rna_to_aa: {codon}'
            translated_char = RNA_CODON_TO_AA[codon]
            if translated_char != '*':
                aa_seq.append(translated_char)
        translated = ''.join(aa_seq)
        self._validate_translated_output(translated, AA_SET - {'*'}, 'rna_to_aa')
        return translated

    def _translate_codon_to_aa(self, seq: str) -> str:
        aa_seq = []
        for token in seq:
            assert token in CODON_TO_AA, f'Unknown codon token for codon_to_aa: {token}'
            translated_char = CODON_TO_AA[token]
            if translated_char != '*':
                aa_seq.append(translated_char)
        translated = ''.join(aa_seq)
        self._validate_translated_output(translated, AA_SET - {'*'}, 'codon_to_aa')
        return translated

    def _translate_aa_to_codon(self, seq: str) -> str:
        codon_tokens = []
        for residue in seq:
            residue = residue.upper()
            if residue in AA_TO_CODON_TOKEN:
                codon_tokens.append(AA_TO_CODON_TOKEN[residue])
            elif residue in NONCANONICAL_AMINO_ACIDS:
                codon_tokens.append(AA_TO_CODON_TOKEN['A'])
            else:
                raise AssertionError(f'Unexpected amino acid token "{residue}" while converting aa_to_codon.')
        translated = ''.join(codon_tokens)
        self._validate_translated_output(translated, CODON_SET, 'aa_to_codon')
        return translated

    def _translate_sequence_for_mode(self, seq: str, mode: str) -> str:
        if mode == 'aa_to_dna':
            return self._translate_aa_to_dna(seq)
        if mode == 'aa_to_rna':
            return self._translate_aa_to_rna(seq)
        if mode == 'dna_to_aa':
            return self._translate_dna_to_aa(seq)
        if mode == 'rna_to_aa':
            return self._translate_rna_to_aa(seq)
        if mode == 'codon_to_aa':
            return self._translate_codon_to_aa(seq)
        if mode == 'aa_to_codon':
            return self._translate_aa_to_codon(seq)
        raise AssertionError(f'Unsupported translation mode: {mode}')

    @staticmethod
    def _sanitize_sequence(seq: str) -> str:
        """Drop characters outside the alphabet, after normalizing case.

        AA_SET is uppercase, so filtering first would delete every lowercase base and leave
        a DNA sequence in the wrong reading frame. Uppercasing first keeps lowercase input.
        For protein data this also means lowercase soft-masked residues are now embedded
        rather than silently removed.
        """
        return ''.join(char for char in seq.upper() if char in AA_SET)

    def _find_first_present_column(self, available_columns: list[str], candidates_ordered: Sequence[str]) -> str:
        """Return the first column from candidates_ordered that exists in available_columns (case-insensitive)."""
        lowercase_to_actual = {col.lower(): col for col in available_columns}
        for candidate in candidates_ordered:
            actual = lowercase_to_actual.get(candidate.lower())
            if actual is not None:
                return actual
        raise KeyError(f"None of the candidate columns were found. Candidates: {candidates_ordered}. Available: {available_columns}")

    def _is_ppi_from_columns(self, available_columns: list[str]) -> bool:
        """Detect if dataset contains paired sequence inputs (SeqA/SeqB variants)."""
        lowercase_columns = set(col.lower() for col in available_columns)
        base_candidates = ['seqs', 'seq', 'sequence', 'sequences']
        for base in base_candidates:
            if (base + 'a') in lowercase_columns and (base + 'b') in lowercase_columns:
                return True
        return False

    def _find_ppi_sequence_columns(self, available_columns: list[str]) -> tuple[str, str]:
        """Return the actual column names for A and B sequences in PPI datasets based on priority."""
        lowercase_to_actual = {col.lower(): col for col in available_columns}
        # Try specific common pairs first (in order)
        specific_pairs = [
            ('SeqA', 'SeqB'),
            ('seqa', 'seqb'),
            ('SeqsA', 'SeqsB'),
        ]
        for cand_a, cand_b in specific_pairs:
            a_actual = lowercase_to_actual.get(cand_a.lower())
            b_actual = lowercase_to_actual.get(cand_b.lower())
            if a_actual is not None and b_actual is not None:
                return a_actual, b_actual

        # Generalized search using base tokens
        base_candidates = ['seqs', 'seq', 'sequence', 'sequences']
        for base in base_candidates:
            a_key = (base + 'a').lower()
            b_key = (base + 'b').lower()
            a_actual = lowercase_to_actual.get(a_key)
            b_actual = lowercase_to_actual.get(b_key)
            if a_actual is not None and b_actual is not None:
                return a_actual, b_actual

        raise KeyError(f"Could not find paired sequence columns for PPI. Available: {available_columns}")

    def _is_missing_value(self, v: object) -> bool:
        if v is None:
            return True
        # float/np.nan handling
        if isinstance(v, float) and np.isnan(v):
            return True
        # list/array handling (check any element)
        if isinstance(v, (list, tuple, np.ndarray)):
            for el in v:
                if el is None:
                    return True
                if isinstance(el, float) and np.isnan(el):
                    return True
        return False

    def _cap_split_rows(self, data_name: str, split_name: str, split: Dataset) -> Dataset:
        """Keep a deterministic, label-stratified subsample of a split larger than its cap.

        The draw depends only on the split's rows, its name, and `split_row_cap_seed`, so a
        dataset keeps the same rows under every training seed, loaded alone or beside other
        datasets, in main.py, the SAE precompute, and the uncertainty workflow alike. A
        split at or under its cap comes back as it was, rows and order unchanged, and a
        calibration split has no cap.
        """
        cap = self.data_args.split_row_caps.get(split_name)
        if cap is None or len(split) <= cap:
            return split

        seed = self.data_args.split_row_cap_seed
        kept = capped_row_indices(list(split['labels']), cap, seed, split_name)  # (cap,)
        print(f'{data_name} {split_name}: capped {len(split)} rows to {cap} (split_row_cap_seed {seed})')
        return split.select(kept)

    def _compute_balanced_weights_for(self, data_name: str, train_set: Dataset, valid_set: Dataset, test_set: Dataset) -> None:
        """Pre-compute EpHod-style sample weights for train/valid/test once up front."""
        if 'balanced_weights' not in self.__dict__:
            self.balanced_weights = {}
        if 'trainer_args' not in self.__dict__ or self.trainer_args is None:
            return
        ta = self.trainer_args
        if 'balanced_regression_metrics' not in ta.__dict__ or not ta.balanced_regression_metrics:
            return

        def _extract(split: Dataset) -> np.ndarray:
            labels = np.asarray(list(split['labels']), dtype=object)
            flat = []
            for item in labels:
                if isinstance(item, (list, tuple, np.ndarray)):
                    arr = np.asarray(item).flatten()
                    flat.extend([float(v) for v in arr if v != -100])
                else:
                    flat.append(float(item))
            return np.asarray(flat, dtype=np.float64)  # (m,) observed values, with -100 dropped

        train_labels = _extract(train_set)
        valid_labels = _extract(valid_set)
        test_labels = _extract(test_set)

        if len(train_labels) < 2 or float(np.max(train_labels)) == float(np.min(train_labels)):
            print_message(f'Skipping balanced weights for {data_name} (degenerate label distribution)')
            return

        bin_borders = ta.balanced_bin_borders if ta.balanced_bin_borders else _default_bin_borders(train_labels)
        method = ta.balanced_weight_method

        train_w = compute_sample_weights(
            train_labels,
            method=method,
            bin_borders=bin_borders,
            lds_bins=ta.balanced_lds_bins,
            lds_ks=ta.balanced_lds_ks,
            lds_sigma=ta.balanced_lds_sigma,
        )
        valid_w = apply_weights_from_reference(
            valid_labels, train_labels,
            method=method, bin_borders=bin_borders,
            lds_bins=ta.balanced_lds_bins, lds_ks=ta.balanced_lds_ks, lds_sigma=ta.balanced_lds_sigma,
        )
        test_w = apply_weights_from_reference(
            test_labels, train_labels,
            method=method, bin_borders=bin_borders,
            lds_bins=ta.balanced_lds_bins, lds_ks=ta.balanced_lds_ks, lds_sigma=ta.balanced_lds_sigma,
        )

        self.balanced_weights[data_name] = {
            'train': train_w,
            'valid': valid_w,
            'test': test_w,
            'bin_borders': list(bin_borders),
            'method': method,
        }
        print_message(
            f'Balanced weights for {data_name}: method={method} bin_borders={bin_borders} '
            f'sizes train/valid/test={len(train_w)}/{len(valid_w)}/{len(test_w)}'
        )

    def process_datasets(
            self,
            hf_datasets: list[tuple[Dataset, Dataset, Dataset, bool] | tuple[Dataset, Dataset, Dataset, bool, Dataset | None]],
            data_names: list[str],
        ) -> ProcessedDatasets:
        """Clean, trim, cap, and label every split of each dataset, and gather the sequences to embed.

        An optional fifth element is the dataset's calibration split. It goes through every step
        the other splits do, apart from the row caps, and lands in `calibration_sets` rather than
        in the returned splits, which stay train, valid, and test.
        """
        max_length = self._data_max_length
        datasets, all_seqs = {}, set()
        self.calibration_sets: dict[str, Dataset] = {}
        translation_mode = self._active_translation_mode()
        for dataset, data_name in zip(hf_datasets, data_names, strict=True):
            print_message(f'Processing {data_name}')
            train_set, valid_set, test_set, ppi = dataset[:4]
            calibration_set = dataset[4] if len(dataset) > 4 else None
            splits = {'train': train_set, 'valid': valid_set, 'test': test_set}
            if calibration_set is not None:
                splits['calibration'] = calibration_set
            for split in splits.values():
                print(split)
            cols = self.data_args.multi_column

            ### sanitize
            # 1) Drop rows with None or NaN in any sequence column(s) or labels
            before = {name: len(split) for name, split in splits.items()}
            if ppi:
                def complete(x: dict[str, Any]) -> bool:
                    return not (self._is_missing_value(x['SeqA']) or self._is_missing_value(x['SeqB']) or self._is_missing_value(x['labels']))
            elif cols:
                for col in cols:
                    assert any(col in split.column_names for split in splits.values()), f"Column {col} not found in dataset {data_name}"

                def complete(x: dict[str, Any]) -> bool:
                    return (not self._is_missing_value(x['labels'])) and all(not self._is_missing_value(x[col]) for col in cols)
            else:
                def complete(x: dict[str, Any]) -> bool:
                    return not (self._is_missing_value(x['seqs']) or self._is_missing_value(x['labels']))
            splits = {name: split.filter(complete) for name, split in splits.items()}
            self._report_removed_rows('Removed None / NaN rows', before, splits)

            # 2) Legacy sanitization for non-translation workflows
            if translation_mode is None:
                splits = {name: self._map_sequences(split, ppi, self._sanitize_sequence) for name, split in splits.items()}

            # 3) Translate before enforcing the target model's raw-sequence
            # budget. Length limits apply to the sequence the tokenizer sees,
            # not to the source alphabet before a 3x expansion or contraction.
            if translation_mode is not None:
                def translate(sequence: str) -> str:
                    return self._translate_sequence_for_mode(sequence, translation_mode)

                splits = {name: self._map_sequences(split, ppi, translate) for name, split in splits.items()}
                print_message(f"Translated sequences with mode {translation_mode} before length enforcement.")

            # 4) Remove any length 0 sequences
            before = {name: len(split) for name, split in splits.items()}
            if ppi:
                def non_empty(x: dict[str, Any]) -> bool:
                    return len(x['SeqA']) > 0 and len(x['SeqB']) > 0
            elif cols:
                def non_empty(x: dict[str, Any]) -> bool:
                    return all(len(x[col]) > 0 for col in cols)
            else:
                def non_empty(x: dict[str, Any]) -> bool:
                    return len(x['seqs']) > 0
            splits = {name: split.filter(non_empty) for name, split in splits.items()}
            self._report_removed_rows('Removed length 0 rows', before, splits)

            # 5) Trim or truncate by length if necessary
            before = {name: len(split) for name, split in splits.items()}
            if self._trim: # trim by length
                if ppi:
                    def fits(x: dict[str, Any]) -> bool:
                        return len(x['SeqA']) + len(x['SeqB']) <= max_length
                elif cols:
                    def fits(x: dict[str, Any]) -> bool:
                        return all(len(x[col]) <= max_length for col in cols)
                else:
                    def fits(x: dict[str, Any]) -> bool:
                        return len(x['seqs']) <= max_length
                splits = {name: split.filter(fits) for name, split in splits.items()}
            elif ppi: # truncate to max_length
                splits = {name: split.map(self._truncate_pairs) for name, split in splits.items()}
            else:
                splits = {name: self._map_sequences(split, ppi, lambda sequence: sequence[:max_length]) for name, split in splits.items()}

            if any(len(split) != before[name] for name, split in splits.items()):
                trimmed = ', '.join(
                    f'{name}: {(before[name] - len(split)) / before[name] * 100 if before[name] else 0.0:.2f}%'
                    for name, split in splits.items()
                )
                print_message(f"Trimmed rows - {trimmed}")

            # 6) Cap each split's rows. After every filter, so a cap counts rows that survive
            # them, and before all_seqs, so the embedder and every consumer see kept rows only.
            splits = {name: self._cap_split_rows(data_name, name, split) for name, split in splits.items()}

            # 7) Record all_seqs
            for split in splits.values():
                if ppi:
                    all_seqs.update(list(split['SeqA']) + list(split['SeqB']))
                elif cols:
                    for col in cols:
                        all_seqs.update(list(split[col]))
                else:
                    all_seqs.update(list(split['seqs']))

            # confirm the type of labels
            check_labels = list(splits['train']['labels'])
            label_type = self._label_type_checker(check_labels)

            if label_type == 'string': # might be string or multilabel
                example = list(splits['valid']['labels'])[0]
                try:
                    new_ex = ast.literal_eval(example)
                    if isinstance(new_ex, list): # if ast runs correctly and is now a list it is multilabel labels
                        label_type = 'multilabel'
                        splits = {name: split.map(lambda ex: {'labels': ast.literal_eval(ex['labels'])}) for name, split in splits.items()}
                except (ValueError, SyntaxError):
                    label_type = 'string' # if ast throws error it is actually string

            if label_type == 'string': # if still string, it's for tokenwise classification
                train_labels = list(splits['train']['labels'])
                unique_tags = set(tag for doc in train_labels for tag in doc)
                tag2id = {tag: id for id, tag in enumerate(sorted(unique_tags))}
                # add cls token to labels
                splits = {name: split.map(lambda ex: {'labels': self._encode_labels(ex['labels'], tag2id=tag2id)}) for name, split in splits.items()}
                label_type = 'tokenwise'
                num_labels = len(unique_tags)
            else:
                if label_type == 'regression':
                    # Detect sigmoid_regression (values in [0,1] covering the range)
                    if self._is_sigmoid_regression(list(splits['train']['labels'])):
                        label_type = 'sigmoid_regression'
                    num_labels = 1
                elif label_type == 'multilabel':
                    train_labels_list = list(splits['train']['labels'])
                    label_lengths = set(len(lab) for lab in train_labels_list)
                    if len(label_lengths) > 1:
                        # Variable-length sparse index lists: convert to multi-hot
                        all_labels = [lab for split in splits.values() for lab in split['labels']]
                        num_labels = max(idx for lab in all_labels for idx in lab) + 1
                        def _to_multihot(ex: dict[str, Any], n: int=num_labels) -> dict[str, list[int]]:
                            vec = [0] * n
                            for idx in ex['labels']:
                                vec[idx] = 1
                            return {'labels': vec}
                        splits = {name: split.map(_to_multihot) for name, split in splits.items()}
                    else:
                        num_labels = len(train_labels_list[0])
                else: # singlelabel classification
                    try:
                        train_labels_list = list(splits['train']['labels'])
                        num_labels = len(train_labels_list[0])
                    except (TypeError, IndexError):
                        unique = np.unique(list(splits['train']['labels']))
                        max_label = max(unique) # sometimes there are missing labels
                        full_list = np.arange(0, max_label+1)
                        num_labels = len(full_list)
            train_set, valid_set, test_set = splits['train'], splits['valid'], splits['test']
            datasets[data_name] = (train_set, valid_set, test_set, num_labels, label_type, ppi)
            if 'calibration' in splits:
                self.calibration_sets[data_name] = splits['calibration']

            if label_type in ('regression', 'sigmoid_regression'):
                self._compute_balanced_weights_for(data_name, train_set, valid_set, test_set)

        print(f'Label type: {label_type}')
        print(f'Number of labels: {num_labels}')

        all_seqs = list(all_seqs)
        all_seqs = sorted(all_seqs, key=len, reverse=True) # longest first
        return datasets, all_seqs

    def _map_sequences(self, split: Dataset, ppi: bool, transform: Callable[[str], str]) -> Dataset:
        """`transform` applied to every sequence column of a split: SeqA and SeqB, the multi columns, or seqs."""
        if ppi:
            return split.map(lambda x: {'SeqA': transform(x['SeqA']), 'SeqB': transform(x['SeqB'])})
        if self.data_args.multi_column:
            for col in self.data_args.multi_column:
                split = split.map(lambda x, _col=col: {_col: transform(x[_col])})
            return split
        return split.map(lambda x: {'seqs': transform(x['seqs'])})

    @staticmethod
    def _report_removed_rows(action: str, before: dict[str, int], splits: dict[str, Dataset]) -> None:
        if any(len(split) != before[name] for name, split in splits.items()):
            removed = ', '.join(f'{name}: {before[name] - len(split)}' for name, split in splits.items())
            print_message(f"{action} - {removed}")

    def get_data(self) -> ProcessedDatasets:
        """
        Supports tabular local splits and labeled FASTA files. A split named calibration, calib,
        or cal is carried beside the others when a dataset has one, and is never required.
        """
        datasets, data_names = [], []

        for data_path in self.data_args.data_paths:
            data_name = data_path.split('/')[-1]
            print_message(f'Loading {data_name}')
            dataset = load_dataset(data_path, revision=self.data_args.revision_by_path.get(data_path))
            if 'inverse' in data_name.lower():
                dataset = dataset.rename_columns({'seqs': 'labels', 'labels': 'seqs'})
            train_name = self._resolve_split_name(list(dataset.keys()), 'train')
            assert train_name is not None, f'{data_name} does not have a train set'
            ppi = 'SeqA' in dataset[train_name].column_names
            # Fallback PPI detection based on available columns
            if not ppi:
                ppi = self._is_ppi_from_columns(dataset[train_name].column_names)
            print_message(f'PPI (or dual sequence input dataset): {ppi}')

            train_set, valid_set, test_set = self._select_train_valid_test_splits(dataset, data_name)
            splits = self._with_calibration_split({'train': train_set, 'valid': valid_set, 'test': test_set}, dataset)

            print_message(', '.join(f'{name.capitalize()} set: {len(split)}' for name, split in splits.items()))
            splits = self._standardize_columns(splits, ppi, data_name)
            datasets.append((splits['train'], splits['valid'], splits['test'], ppi, splits.get('calibration')))
            data_names.append(data_name)

        for data_dir in self.data_args.data_dirs:
            # local_data/taxon
            data_name = os.path.basename(os.path.normpath(data_dir))
            # Determine PPI by directory hint or columns
            ppi = 'ppi' in data_dir.lower()
            splits = self._load_local_dataset_dir(data_dir)

            # If not indicated by directory, infer from columns
            if not ppi:
                ppi = self._is_ppi_from_columns(splits['train'].column_names)

            splits = self._standardize_columns(splits, ppi, data_name)
            datasets.append((splits['train'], splits['valid'], splits['test'], ppi, splits.get('calibration')))
            data_names.append(data_name)

        return self.process_datasets(hf_datasets=datasets, data_names=data_names)

    def _with_calibration_split(self, splits: dict[str, Dataset], dataset: DatasetDict | dict[str, Dataset]) -> dict[str, Dataset]:
        """The splits, with the dataset's calibration split added when it has one."""
        calibration_name = self._resolve_split_name(list(dataset.keys()), 'calibration')
        if calibration_name is None:
            return splits
        return {**splits, 'calibration': dataset[calibration_name]}

    def _standardize_columns(self, splits: dict[str, Dataset], ppi: bool, data_name: str) -> dict[str, Dataset]:
        """Every split's columns renamed to SeqA, SeqB, or seqs, and labels, with the rest dropped.

        A column is looked up in the splits in order, train first, so a split that lacks it, such
        as an empty one, takes the name another split gives it.
        """
        def first_present(find: Callable[[list[str]], Any]) -> Any:
            for position, split in enumerate(splits.values()):
                try:
                    return find(split.column_names)
                except KeyError:
                    if position == len(splits) - 1:
                        raise
            raise KeyError('no splits to search')

        def find_labels(columns: list[str]) -> str:
            return self._find_first_present_column(columns, LABEL_COLUMNS)

        if ppi:
            # Standardize PPI columns to 'SeqA', 'SeqB', and 'labels'
            print('Standardizing PPI column names')
            a_col, b_col = first_present(self._find_ppi_sequence_columns)
            lbl_col = first_present(find_labels)
            splits = {name: split.rename_columns({a_col: 'SeqA', b_col: 'SeqB', lbl_col: 'labels'}) for name, split in splits.items()}
            print('Removing extras')
            keep_cols = {'SeqA', 'SeqB', 'labels', 'label_mask'}
        else:
            print('Standardizing column names')
            use_multi = self.data_args.multi_column is not None
            if not use_multi:
                seq_col = first_present(lambda columns: self._find_first_present_column(columns, SEQUENCE_COLUMNS))
            label_col = first_present(find_labels)

            # Always standardize label column to 'labels'
            splits = {name: split.rename_columns({label_col: 'labels'}) for name, split in splits.items()}
            if not use_multi:
                splits = {name: split.rename_columns({seq_col: 'seqs'}) for name, split in splits.items()}
                # drop everything else
                print('Removing extras')
                keep_cols = {'seqs', 'labels', 'label_mask'}
            else:
                # Validate requested multi columns exist (assert exact match)
                for col in self.data_args.multi_column:
                    assert any(col in split.column_names for split in splits.values()), f"Column {col} not found in dataset {data_name}"
                # Keep only requested columns and labels
                keep_cols = set(self.data_args.multi_column + ['labels', 'label_mask'])

        return {name: split.remove_columns([col for col in split.column_names if col not in keep_cols]) for name, split in splits.items()}

    def get_embedding_dim_sql(self, save_path: str, test_seq: str, tokenizer: object) -> int:
        """The embedding width of a streamed cache, which `--sql` keeps on disk.

        `save_path` is the cache's store directory (`_embedding_store`), and the width is the store's
        own, checked against a sequence it must hold.
        """
        with open_reader(save_path) as reader:
            assert not reader.missing([test_seq]), f"Test sequence not found in the embedding store: {test_seq}"
            return reader.spec.width

    def get_embedding_dim_pth(self, emb_dict: dict[str, torch.Tensor], test_seq: str, tokenizer: object) -> int:
        # emb_dict: (d,) pooled or (l + 2, d) token rows per sequence
        test_embedding = emb_dict[test_seq]
        embedding_dim = test_embedding.shape[-1]
        return embedding_dim

    @contextmanager
    def _pooled_row_lookup(self, model_name: str) -> Iterator[Callable[[str], np.ndarray]]:
        """Open this model's embedding store and yield `sequence -> (1, d)` row lookup.

        The store stays open for the whole build, so a second split reads through the same open
        index and parts instead of reopening the store per split.
        """
        directory = require_feature(self._embedding_store(model_name), self._legacy_embedding_files(model_name))
        with cached_rows(directory, kind="store") as lookup:
            yield lambda seq: _as_row(_to_numpy(lookup(seq)), self._full)  # (1, d)

    def build_vector_numpy_dataset_from_embeddings(
            self,
            model_name: str,
            train_seqs: Sequence[str],
            valid_seqs: Sequence[str],
            test_seqs: Sequence[str],
            feature_selection: "FeatureSelection | None"=None,
            sparse_layout: bool | None = None,
        ) -> tuple[FeatureMatrix, FeatureMatrix, FeatureMatrix]:
        with self._pooled_row_lookup(model_name) as pooled_row:
            # _as_row already averaged any per-residue matrix, so every row is (1, d).
            splits = []
            for seqs in (train_seqs, valid_seqs, test_seqs):
                rows = (_narrow(pooled_row(seq), feature_selection) for seq in seqs)  # (1, d) each
                features, sparse_layout = stack_pooled_rows(rows, sparse_layout)  # (n_split, d)
                splits.append(features)

        _print_split_shapes(splits)
        return tuple(splits)

    def build_pair_vector_numpy_dataset_from_embeddings(
            self,
            model_name: str,
            train_seqs_a: Sequence[str],
            train_seqs_b: Sequence[str],
            valid_seqs_a: Sequence[str],
            valid_seqs_b: Sequence[str],
            test_seqs_a: Sequence[str],
            test_seqs_b: Sequence[str],
            flip_train_pairs: bool = False,
            feature_selection: "FeatureSelection | None"=None,
            sparse_layout: bool | None = None,
        ) -> tuple[FeatureMatrix, FeatureMatrix, FeatureMatrix]:
        """Stack one row per interaction, joining the two proteins by the pair mode.

        Under `concat` the row is A then B, which a swap changes, so `flip_train_pairs`
        augments the training split with the other ordering. Validation and test keep the
        dataset order, because swapping there would make a score depend on the random
        state rather than on the model. Under `minmax` the row is already symmetric and
        the caller has switched the augmentation off.
        """
        pair_mode = self._pair_feature_mode

        def pair_row(pooled_row: Callable[[str], np.ndarray], seq_a: str, seq_b: str) -> np.ndarray:
            # pooled_row: (1, d) per sequence, as a callable of the sequence
            half_a, half_b = pair_vector_halves(
                pooled_row(seq_a), pooled_row(seq_b), pair_mode
            )  # two of (1, d)
            return _narrow(np.concatenate([half_a, half_b], axis=-1), feature_selection)  # (1, 2 * d)

        with self._pooled_row_lookup(model_name) as pooled_row:
            train_pairs = (
                pair_row(pooled_row, *self._random_order(seq_a, seq_b)) if flip_train_pairs
                else pair_row(pooled_row, seq_a, seq_b)
                for seq_a, seq_b in zip(train_seqs_a, train_seqs_b, strict=True)
            )
            splits = []
            for rows in (
                train_pairs,
                (pair_row(pooled_row, seq_a, seq_b) for seq_a, seq_b in zip(valid_seqs_a, valid_seqs_b, strict=True)),
                (pair_row(pooled_row, seq_a, seq_b) for seq_a, seq_b in zip(test_seqs_a, test_seqs_b, strict=True)),
            ):
                features, sparse_layout = stack_pooled_rows(rows, sparse_layout)  # (n_split, 2 * d)
                splits.append(features)

        _print_split_shapes(splits)
        return tuple(splits)

    def prepare_scikit_dataset(
        self,
        model_name: str,
        dataset: tuple[Dataset, Dataset, Dataset, int, str, bool],
        standardize: bool = True,
        flip_train_pairs: bool = False,
        feature_selection: "FeatureSelection | None"=None,
    ) -> tuple[FeatureMatrix, np.ndarray, FeatureMatrix, np.ndarray, FeatureMatrix, np.ndarray, str]:
        """Materialize pooled embeddings and labels as numpy arrays.

        `standardize` exists for axis-aligned learners: a per-feature shift and scale
        cannot change which splits a decision tree can express, and centering would turn
        a sparse SAE feature matrix dense.

        `flip_train_pairs` carries --random_pair_flipping into the paired builder and is
        ignored for single-sequence datasets.

        `feature_selection` narrows every row as it is read, so the returned arrays are
        (n, s) rather than (n, d). Reading full width and narrowing afterwards would
        defeat the purpose: the full-width matrix is what does not fit.

        The matrices come back compressed when the embeddings are sparse enough to
        warrant it, so a caller has to accept either layout. Standardizing forces the
        dense one, because centering fills in every zero.
        """
        train_set, valid_set, test_set, _, label_type, ppi = dataset
        embedding_scaler = self.embedding_args.embedding_scaler and standardize
        assert isinstance(embedding_scaler, bool), f"Invalid embedding_scaler: {embedding_scaler}"
        sparse_layout = False if (embedding_scaler and not self._full) else None

        if ppi:
            X_train, X_valid, X_test = self.build_pair_vector_numpy_dataset_from_embeddings(
                model_name,
                list(train_set['SeqA']),
                list(train_set['SeqB']),
                list(valid_set['SeqA']),
                list(valid_set['SeqB']),
                list(test_set['SeqA']),
                list(test_set['SeqB']),
                # A symmetric pair mode already emits the swapped row, so the augmentation
                # would only duplicate what is there.
                flip_train_pairs=flip_train_pairs and pair_order_matters(self._pair_feature_mode),
                feature_selection=feature_selection,
                sparse_layout=sparse_layout,
            )
        else:
            X_train, X_valid, X_test = self.build_vector_numpy_dataset_from_embeddings(
                model_name,
                list(train_set['seqs']),
                list(valid_set['seqs']),
                list(test_set['seqs']),
                feature_selection=feature_selection,
                sparse_layout=sparse_layout,
            )

        if not self._full and embedding_scaler:
            print_message('Fitting StandardScaler on scikit training embeddings')
            embedding_standardizer = EmbeddingStandardizer.fit_numpy(X_train)
            X_train = embedding_standardizer.transform_numpy(X_train)
            X_valid = embedding_standardizer.transform_numpy(X_valid)
            X_test = embedding_standardizer.transform_numpy(X_test)

        y_train = self._labels_to_numpy(list(train_set['labels']))
        y_valid = self._labels_to_numpy(list(valid_set['labels']))
        y_test = self._labels_to_numpy(list(test_set['labels']))

        print_message('Numpy dataset shapes with labels')
        print_message(f'Train: {X_train.shape}, {y_train.shape}')
        print_message(f'Valid: {X_valid.shape}, {y_valid.shape}')
        print_message(f'Test: {X_test.shape}, {y_test.shape}')
        return X_train, y_train, X_valid, y_valid, X_test, y_test, label_type  # (n_train, f), (n_train,), (n_valid, f), (n_valid,), (n_test, f), (n_test,), then the label type
