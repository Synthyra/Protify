### imports
import random
import numpy as np
import torch
import torch.nn.functional as F

from collections import OrderedDict
from collections.abc import Collection, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Optional
from datasets import Dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset as TorchDataset
from tqdm.auto import tqdm

from .pair_features import DEFAULT_PAIR_FEATURE_MODE, pair_vector_halves
from ..console_output import print_message
from ..feature_cache import open_reader


if TYPE_CHECKING:
    # Type-only. Importing the probes package here at runtime would pull in its whole
    # __init__, and these classes need nothing from it but one transform method.
    from fastplms.features import FeatureReader
    from ..probes.feature_path_selector import FeatureSelection


class EmbeddingStandardizer:
    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        # mean: (d,), scale: (d,)
        self.mean = torch.from_numpy(mean.astype(np.float32))
        self.scale = torch.from_numpy(scale.astype(np.float32))

    @classmethod
    def fit_tensors(cls, embeddings: Iterable[torch.Tensor]) -> "EmbeddingStandardizer":
        # embeddings: (d,) or (1, d) per sequence
        scaler = StandardScaler()
        n_seen = 0
        for embedding in embeddings:
            features = embedding.reshape(1, -1).float().numpy()
            scaler.partial_fit(features)
            n_seen += features.shape[0]
        assert n_seen > 0, "Cannot fit embedding standardizer without training embeddings."
        return cls(scaler.mean_, scaler.scale_)

    @classmethod
    def fit_numpy(cls, features: np.ndarray) -> "EmbeddingStandardizer":
        # features: (n, d)
        assert features.ndim == 2, f"Expected 2D features, got shape {features.shape}"
        scaler = StandardScaler()
        scaler.fit(features.astype(np.float32))
        return cls(scaler.mean_, scaler.scale_)

    def transform_tensor(self, embedding: torch.Tensor) -> torch.Tensor:
        # embedding: (d,) or (1, d)
        shape = embedding.shape
        features = embedding.reshape(1, -1).float()
        assert features.shape[1] == self.mean.shape[0], (
            f"Embedding dim {features.shape[1]} does not match scaler dim {self.mean.shape[0]}"
        )
        return ((features - self.mean) / self.scale).reshape(shape)  # (d,) or (1, d), the input shape

    def transform_numpy(self, features: np.ndarray) -> np.ndarray:
        # features: (n, d)
        assert features.ndim == 2, f"Expected 2D features, got shape {features.shape}"
        assert features.shape[1] == self.mean.shape[0], (
            f"Feature dim {features.shape[1]} does not match scaler dim {self.mean.shape[0]}"
        )
        mean = self.mean.numpy()
        scale = self.scale.numpy()
        return ((features.astype(np.float32) - mean) / scale).astype(np.float32)  # (n, d)

    def select_features(self, indices: np.ndarray) -> "EmbeddingStandardizer":
        """Keep only the given columns' statistics.

        StandardScaler treats every column independently, so subsetting a scaler fit on
        the full width gives exactly the scaler a refit on those columns would give, and
        saves a second pass over the training embeddings.
        """
        # indices: (s,)
        return EmbeddingStandardizer(
            self.mean.numpy()[indices],  # (s,)
            self.scale.numpy()[indices],  # (s,)
        )


def apply_embedding_transforms(
    embedding: torch.Tensor,
    feature_selection: Optional["FeatureSelection"],
    embedding_standardizer: EmbeddingStandardizer | None,
) -> torch.Tensor:
    """Put a cached vector embedding through the fixed transforms a probe expects.

    Selection runs first, so the scaler statistics describe the columns the probe will
    actually see. `EmbeddingStandardizer.select_features` is what keeps the two in step.
    Order matters and it is the same at training and inference, which is why both
    transforms travel together rather than being applied at their own call sites.
    """
    # embedding: (..., f) -> (..., s) when a selection is present, otherwise unchanged
    if feature_selection is not None:
        embedding = feature_selection.transform_tensor(embedding)
    if embedding_standardizer is not None:
        embedding = embedding_standardizer.transform_tensor(embedding)
    return embedding  # (..., d), or (..., s) when a selection narrows it


def read_cached_row(reader: "FeatureReader", seq: str) -> torch.Tensor:
    """One sequence's row from an embedding store: `(d,)` pooled or `(l, d)` per token.

    A sparse store gives its row densified. A sequence the store lacks raises `AssertionError`, as the
    cache this replaced did, without echoing the sequence.
    """
    try:
        return reader.read([seq])[0]  # (d,) pooled or (l + 2, d) token rows
    except KeyError:
        raise AssertionError(f"Embedding not found for sequence of length {len(seq)}") from None


# The `FromDisk` datasets below stream rows from the feature store, one lookup at a time, with a bounded
# recent-rows cache in front. `db_path` keeps the name it had when the cache was a SQLite file, and now
# names the store's directory (`feature_cache.feature_directory`).


class PairEmbedsLabelsDatasetFromDisk(TorchDataset):
    def __init__(
            self,
            hf_dataset: Dataset,
            col_a: str='SeqA',
            col_b: str='SeqB',
            label_col: str='labels',
            full: bool=False,
            db_path: str='embeddings.db',
            batch_size: int=64,
            read_scaler: int=100,
            input_size: int=768,
            task_type: str='regression',
            train: bool=True,
            random_pair_flipping: bool=False,
            pair_feature_mode: str = DEFAULT_PAIR_FEATURE_MODE,
            embedding_standardizer: EmbeddingStandardizer | None = None,
            feature_selection: Optional["FeatureSelection"] = None,
            **kwargs: Any
        ) -> None:
        self.seqs_a, self.seqs_b, self.labels = list(hf_dataset[col_a]), list(hf_dataset[col_b]), list(hf_dataset[label_col])
        self.db_file = db_path
        self.input_size = input_size
        self.full = full
        self.length = len(self.labels)
        self.task_type = task_type
        self.train = train
        self.random_pair_flipping = random_pair_flipping
        self.pair_feature_mode = pair_feature_mode
        self.embedding_standardizer = embedding_standardizer
        self.feature_selection = feature_selection
        self._cache_max = max(read_scaler * batch_size, 1)
        self._emb_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._reader = open_reader(self.db_file)

    def __len__(self) -> int:
        return self.length

    def check_seqs(self, all_seqs: Collection[str]) -> None:
        missing_seqs = [seq for seq in self.seqs_a + self.seqs_b if seq not in all_seqs]
        if missing_seqs:
            print_message(f'Sequences not found in embeddings: {missing_seqs}')
        else:
            print_message('All sequences in embeddings')

    def _get_emb(self, seq: str) -> torch.Tensor:
        cached = self._emb_cache.get(seq)
        if cached is not None:
            self._emb_cache.move_to_end(seq)
            return cached  # (d,) pooled or (l + 2, d) token rows
        emb = read_cached_row(self._reader, seq)
        self._emb_cache[seq] = emb
        while len(self._emb_cache) > self._cache_max:
            self._emb_cache.popitem(last=False)
        return emb  # (d,) pooled or (l + 2, d) token rows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_a = self._get_emb(self.seqs_a[idx])
        emb_b = self._get_emb(self.seqs_b[idx])
        label = self.labels[idx]

        # Optional random pair order augmentation during training only.
        if self.train and self.random_pair_flipping and random.random() < 0.5:
            emb_a, emb_b = emb_b, emb_a

        if not self.full:
            # The two blocks the collator will join. Under minmax they are the pair
            # sorted feature by feature rather than the proteins themselves, so the swap
            # above becomes a no-op and one interaction has exactly one row.
            emb_a, emb_b = pair_vector_halves(emb_a, emb_b, self.pair_feature_mode)

        if not self.full and (
            self.feature_selection is not None or self.embedding_standardizer is not None
        ):
            # Both transforms were fit on the joined pair row, whichever two blocks the
            # mode above produced, so they are applied joined and split afterwards.
            pair_emb = torch.cat([emb_a.reshape(1, -1), emb_b.reshape(1, -1)], dim=-1)  # (1, 2 * d)
            pair_emb = apply_embedding_transforms(
                pair_emb, self.feature_selection, self.embedding_standardizer
            )  # (1, s)
            split_idx = pair_emb.shape[-1] // 2
            emb_a = pair_emb[:, :split_idx]  # (1, s // 2)
            emb_b = pair_emb[:, split_idx:]  # (1, s - s // 2)

        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return emb_a, emb_b, label  # emb_a: (1, d_a) pooled or (l_a, d); emb_b: (1, d_b) pooled or (l_b, d); label: () or (c,)


class PairEmbedsLabelsDataset(TorchDataset):
    def __init__(
            self,
            hf_dataset: Dataset,
            emb_dict: dict[str, torch.Tensor],
            col_a: str='SeqA',
            col_b: str='SeqB',
            full: bool=False,
            label_col: str='labels',
            input_size: int=768,
            task_type: str='regression',
            train: bool=True,
            random_pair_flipping: bool=False,
            pair_feature_mode: str = DEFAULT_PAIR_FEATURE_MODE,
            embedding_standardizer: EmbeddingStandardizer | None = None,
            feature_selection: Optional["FeatureSelection"] = None,
            **kwargs: Any
        ) -> None:
        # emb_dict: (d,) pooled or (l + 2, d) token rows per sequence
        self.seqs_a = list(hf_dataset[col_a])
        self.seqs_b = list(hf_dataset[col_b])
        self.labels = list(hf_dataset[label_col])
        self.input_size = input_size // 2 if not full else input_size # already scaled if ppi
        self.task_type = task_type
        self.full = full
        self.train = train
        self.random_pair_flipping = random_pair_flipping
        self.pair_feature_mode = pair_feature_mode
        self.embedding_standardizer = embedding_standardizer
        self.feature_selection = feature_selection

        # Combine seqs_a and seqs_b to find all unique sequences needed
        needed_seqs = set(list(hf_dataset[col_a]) + list(hf_dataset[col_b]))
        # Filter emb_dict to keep only the necessary embeddings
        self.emb_dict = {seq: emb_dict[seq] for seq in needed_seqs if seq in emb_dict}
        # Check for any missing embeddings
        missing_seqs = needed_seqs - self.emb_dict.keys()
        if missing_seqs:
            raise ValueError(f"Embeddings not found for sequences: {missing_seqs}")

    def __len__(self) -> int:
        return len(self.labels)
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_a, seq_b = self.seqs_a[idx], self.seqs_b[idx]
        emb_a = self.emb_dict[seq_a].reshape(-1, self.input_size)
        emb_b = self.emb_dict[seq_b].reshape(-1, self.input_size)

        # Optional random pair order augmentation during training only.
        if self.train and self.random_pair_flipping and random.random() < 0.5:
            emb_a, emb_b = emb_b, emb_a

        if not self.full:
            # The two blocks the collator will join. Under minmax they are the pair
            # sorted feature by feature rather than the proteins themselves, so the swap
            # above becomes a no-op and one interaction has exactly one row.
            emb_a, emb_b = pair_vector_halves(emb_a, emb_b, self.pair_feature_mode)

        if not self.full and (
            self.feature_selection is not None or self.embedding_standardizer is not None
        ):
            # Both transforms were fit on the joined pair row, whichever two blocks the
            # mode above produced, so they are applied joined and split afterwards.
            pair_emb = torch.cat([emb_a.reshape(1, -1), emb_b.reshape(1, -1)], dim=-1)  # (1, 2 * d)
            pair_emb = apply_embedding_transforms(
                pair_emb, self.feature_selection, self.embedding_standardizer
            )  # (1, s)
            split_idx = pair_emb.shape[-1] // 2
            emb_a = pair_emb[:, :split_idx]  # (1, s // 2)
            emb_b = pair_emb[:, split_idx:]  # (1, s - s // 2)

        # Prepare the label
        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)

        return emb_a, emb_b, label  # emb_a: (1, d_a) pooled or (l_a, d); emb_b: (1, d_b) pooled or (l_b, d); label: () or (c,)


class EmbedsLabelsDatasetFromDisk(TorchDataset):
    def __init__(
            self,
            hf_dataset: Dataset,
            col_name: str='seqs',
            label_col: str='labels',
            full: bool=False,
            db_path: str='embeddings.db',
            batch_size: int=64,
            read_scaler: int=100,
            input_size: int=768,
            task_type: str='singlelabel',
            embedding_standardizer: EmbeddingStandardizer | None = None,
            feature_selection: Optional["FeatureSelection"] = None,
            **kwargs: Any
        ) -> None:
        self.seqs, self.labels = list(hf_dataset[col_name]), list(hf_dataset[label_col])
        self.length = len(self.labels)
        self.max_length = len(max(self.seqs, key=len))
        print_message(f'Max length: {self.max_length}')

        self.db_file = db_path
        self.input_size = input_size
        self.full = full
        self.task_type = task_type
        self.embedding_standardizer = embedding_standardizer
        self.feature_selection = feature_selection
        self._cache_max = max(read_scaler * batch_size, 1)
        self._emb_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._reader = open_reader(self.db_file)

    def __len__(self) -> int:
        return self.length

    def check_seqs(self, all_seqs: Collection[str]) -> None:
        cond = False
        for seq in self.seqs:
            if seq not in all_seqs:
                cond = True
            if cond:
                break
        if cond:
            print_message('Sequences not found in embeddings')
        else:
            print_message('All sequences in embeddings')

    def _get_emb(self, seq: str) -> torch.Tensor:
        cached = self._emb_cache.get(seq)
        if cached is not None:
            self._emb_cache.move_to_end(seq)
            return cached  # (d,) pooled or (l + 2, d) token rows
        emb = read_cached_row(self._reader, seq)
        self._emb_cache[seq] = emb
        while len(self._emb_cache) > self._cache_max:
            self._emb_cache.popitem(last=False)
        return emb  # (d,) pooled or (l + 2, d) token rows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self._get_emb(self.seqs[idx])
        if self.full:
            padding_needed = self.max_length - emb.size(0)
            emb = F.pad(emb, (0, 0, 0, padding_needed), value=0)
        else:
            emb = apply_embedding_transforms(
                emb, self.feature_selection, self.embedding_standardizer
            )
        label = self.labels[idx]
        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return emb.squeeze(0), label  # (d,) pooled or (l_max, d) padded token rows; label: () or (c,)


class EmbedsLabelsDataset(TorchDataset):
    def __init__(
            self,
            hf_dataset: Dataset,
            emb_dict: dict[str, torch.Tensor],
            col_name: str='seqs',
            label_col: str='labels',
            task_type: str='singlelabel',
            full: bool=False,
            embedding_standardizer: EmbeddingStandardizer | None = None,
            feature_selection: Optional["FeatureSelection"] = None,
            **kwargs: Any
        ) -> None:
        # emb_dict: (d,) pooled or (l + 2, d) token rows per sequence
        self.embeddings = self.get_embs(emb_dict, list(hf_dataset[col_name]))
        self.full = full
        self.labels = list(hf_dataset[label_col])
        self.task_type = task_type
        self.embedding_standardizer = embedding_standardizer
        self.feature_selection = feature_selection
        self.max_length = len(max(list(hf_dataset[col_name]), key=len))
        print_message(f'Max length: {self.max_length}')

    def __len__(self) -> int:
        return len(self.labels)
    
    def get_embs(self, emb_dict: dict[str, torch.Tensor], seqs: Sequence[str]) -> list[torch.Tensor]:
        # emb_dict: (d,) pooled or (l + 2, d) token rows per sequence
        embeddings = []
        for seq in tqdm(seqs, desc='Loading Embeddings'):
            emb = emb_dict[seq]
            embeddings.append(emb)
        return embeddings  # (d,) pooled or (l + 2, d) token rows per sequence

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        emb = self.embeddings[idx].float()
        if self.full:
            padding_needed = self.max_length - emb.size(0)
            emb = F.pad(emb, (0, 0, 0, padding_needed), value=0)
        else:
            emb = apply_embedding_transforms(
                emb, self.feature_selection, self.embedding_standardizer
            )
        return emb.squeeze(0), label  # (d,) pooled or (l_max, d) padded token rows; label: () or (c,)
    

class StringLabelDataset(TorchDataset):    
    def __init__(self, hf_dataset: Dataset, col_name: str='seqs', label_col: str='labels', **kwargs: Any) -> None:
        self.seqs = list(hf_dataset[col_name])
        self.labels = list(hf_dataset[label_col])
        self.lengths = [len(seq) for seq in self.seqs]

    def avg(self) -> float:
        return sum(self.lengths) / len(self.lengths)

    def __len__(self) -> int:
        return len(self.seqs)
    
    def __getitem__(self, idx: int) -> tuple[str, Any]:
        seq = self.seqs[idx]
        label = self.labels[idx]
        return seq, label
    

class PairStringLabelDataset(TorchDataset):
    def __init__(self, hf_dataset: Dataset, col_a: str='SeqA', col_b: str='SeqB', label_col: str='labels', train: bool=True, random_pair_flipping: bool=False, **kwargs: Any) -> None:
        self.seqs_a, self.seqs_b = list(hf_dataset[col_a]), list(hf_dataset[col_b])
        self.labels = list(hf_dataset[label_col])
        self.train = train
        self.random_pair_flipping = random_pair_flipping

    def avg(self) -> float:
        return sum(len(seqa) + len(seqb) for seqa, seqb in zip(self.seqs_a, self.seqs_b, strict=True)) / len(self.seqs_a)

    def __len__(self) -> int:
        return len(self.seqs_a)

    def __getitem__(self, idx: int) -> tuple[str, str, Any]:
        seq_a, seq_b = self.seqs_a[idx], self.seqs_b[idx]
        if self.train and self.random_pair_flipping and random.random() < 0.5:
            seq_a, seq_b = seq_b, seq_a
        return seq_a, seq_b, self.labels[idx]


class SimpleProteinDataset(TorchDataset):
    """Simple dataset for protein sequences."""
    def __init__(self, sequences: list[str]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


class MultiEmbedsLabelsDatasetFromDisk(TorchDataset):
    def __init__(
            self,
            hf_dataset: Dataset,
            seq_cols: list[str],
            label_col: str = 'labels',
            full: bool = False,
            db_path: str = 'embeddings.db',
            batch_size: int = 64,
            read_scaler: int = 100,
            input_size: int = 768,
            task_type: str = 'singlelabel',
            train: bool = True,
            embedding_standardizer: EmbeddingStandardizer | None = None,
            feature_selection: Optional["FeatureSelection"] = None,
            **kwargs: Any,
        ) -> None:
        self.seq_cols = seq_cols
        self.labels = list(hf_dataset[label_col])
        self.length = len(self.labels)
        self.full = full
        self.db_file = db_path
        self.input_size = input_size // len(seq_cols) if not full else input_size # already scaled if multi-column
        self.task_type = task_type
        self.train = train
        self.embedding_standardizer = embedding_standardizer
        self.feature_selection = feature_selection
        self._cache_max = max(read_scaler * batch_size, 1)
        self._emb_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        # Store sequences per column
        self.col_to_seqs = {col: list(hf_dataset[col]) for col in seq_cols}

        # Precompute max combined length for matrix embeddings from raw strings
        if self.full:
            def combined_len_at(i: int) -> int:
                return sum(len(self.col_to_seqs[c][i]) for c in self.seq_cols) + (len(self.seq_cols) - 1)
            self.max_length = max(combined_len_at(i) for i in range(self.length)) if self.length > 0 else 0

        self._reader = open_reader(self.db_file)

    def __len__(self) -> int:
        return self.length

    def _get_emb(self, seq: str) -> torch.Tensor:
        cached = self._emb_cache.get(seq)
        if cached is not None:
            self._emb_cache.move_to_end(seq)
            return cached  # (d,) pooled or (l + 2, d) token rows
        emb = read_cached_row(self._reader, seq)
        self._emb_cache[seq] = emb
        while len(self._emb_cache) > self._cache_max:
            self._emb_cache.popitem(last=False)
        return emb  # (d,) pooled or (l + 2, d) token rows

    def _combine_matrix(self, parts: list[torch.Tensor]) -> torch.Tensor:
        # parts: (l_i, d) per part
        if len(parts) == 0:
            return torch.zeros(0, self.input_size)  # (0, d)
        sep = torch.zeros(1, self.input_size, dtype=parts[0].dtype)
        out = []
        for i, p in enumerate(parts):
            out.append(p)
            if i < len(parts) - 1:
                out.append(sep)
        return torch.cat(out, dim=0)  # (l_total, d), the part rows plus one separator row between parts

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        parts = [self._get_emb(self.col_to_seqs[col][idx]) for col in self.seq_cols]
        if self.full:
            emb = self._combine_matrix(parts)
            if self.max_length:
                pad_needed = self.max_length - emb.size(0)
                if pad_needed > 0:
                    emb = F.pad(emb, (0, 0, 0, pad_needed), value=0)
        else:
            emb = torch.cat([p.reshape(1, -1) for p in parts], dim=-1)
            emb = apply_embedding_transforms(
                emb, self.feature_selection, self.embedding_standardizer
            )

        label = self.labels[idx]
        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return emb.squeeze(0), label  # (d,) pooled or (l_max, d) padded token rows; label: () or (c,)


class MultiEmbedsLabelsDataset(TorchDataset):
    def __init__(
            self,
            hf_dataset: Dataset,
            seq_cols: list[str],
            label_col: str = 'labels',
            full: bool = False,
            emb_dict: dict[str, torch.Tensor] | None = None,
            input_size: int = 768,
            task_type: str = 'singlelabel',
            train: bool = True,
            embedding_standardizer: EmbeddingStandardizer | None = None,
            feature_selection: Optional["FeatureSelection"] = None,
            **kwargs: Any,
        ) -> None:
        # emb_dict: (d,) pooled or (l + 2, d) token rows per sequence
        self.seq_cols = seq_cols
        self.labels = list(hf_dataset[label_col])
        self.full = full
        self.input_size = input_size // len(seq_cols) if not full else input_size
        self.task_type = task_type
        self.train = train
        self.embedding_standardizer = embedding_standardizer
        self.feature_selection = feature_selection

        self.col_to_seqs = {col: list(hf_dataset[col]) for col in seq_cols}

        # Precompute combined embeddings
        self.embeddings = []
        if self.full:
            # compute max_length from strings
            def combined_len_at(i: int) -> int:
                return sum(len(self.col_to_seqs[c][i]) for c in self.seq_cols) + (len(self.seq_cols) - 1)
            self.max_length = max(combined_len_at(i) for i in range(len(self.labels))) if len(self.labels) > 0 else 0

        for i in tqdm(range(len(self.labels)), desc='Loading Multi-Embeddings'):
            parts = []
            for col in self.seq_cols:
                seq = self.col_to_seqs[col][i]
                emb = emb_dict[seq]
                emb = emb.reshape(-1, self.input_size)
                parts.append(emb)
            if self.full:
                emb = self._combine_matrix(parts)
                # pad to max_length
                if self.max_length:
                    pad_needed = self.max_length - emb.size(0)
                    if pad_needed > 0:
                        emb = F.pad(emb, (0, 0, 0, pad_needed), value=0)
            else:
                emb = torch.cat([p.reshape(1, -1) for p in parts], dim=-1)
                emb = apply_embedding_transforms(
                    emb, self.feature_selection, self.embedding_standardizer
                )
            self.embeddings.append(emb)

    def _combine_matrix(self, parts: list[torch.Tensor]) -> torch.Tensor:
        # parts: (l_i, d) per part
        if len(parts) == 0:
            return torch.zeros(0, self.input_size)  # (0, d)
        sep = torch.zeros(1, self.input_size, dtype=parts[0].dtype)
        out = []
        for i, p in enumerate(parts):
            out.append(p)
            if i < len(parts) - 1:
                out.append(sep)
        return torch.cat(out, dim=0)  # (l_total, d), the part rows plus one separator row between parts

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        emb = self.embeddings[idx].float()
        return emb.squeeze(0), label  # (d,) pooled or (l_max, d) padded token rows; label: () or (c,)
