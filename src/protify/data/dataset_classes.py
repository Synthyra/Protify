### imports
import random
import sqlite3
import torch
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple, Union
from torch.utils.data import Dataset as TorchDataset
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler

try:
    from utils import print_message, embedding_blob_to_tensor
except ImportError:
    from ..utils import print_message, embedding_blob_to_tensor


class EmbeddingStandardizer:
    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean = torch.from_numpy(mean.astype(np.float32))
        self.scale = torch.from_numpy(scale.astype(np.float32))

    @classmethod
    def fit_tensors(cls, embeddings: Iterable[torch.Tensor]) -> "EmbeddingStandardizer":
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
        assert features.ndim == 2, f"Expected 2D features, got shape {features.shape}"
        scaler = StandardScaler()
        scaler.fit(features.astype(np.float32))
        return cls(scaler.mean_, scaler.scale_)

    def transform_tensor(self, embedding: torch.Tensor) -> torch.Tensor:
        shape = embedding.shape
        features = embedding.reshape(1, -1).float()
        assert features.shape[1] == self.mean.shape[0], (
            f"Embedding dim {features.shape[1]} does not match scaler dim {self.mean.shape[0]}"
        )
        return ((features - self.mean) / self.scale).reshape(shape)

    def transform_numpy(self, features: np.ndarray) -> np.ndarray:
        assert features.ndim == 2, f"Expected 2D features, got shape {features.shape}"
        assert features.shape[1] == self.mean.shape[0], (
            f"Feature dim {features.shape[1]} does not match scaler dim {self.mean.shape[0]}"
        )
        mean = self.mean.numpy()
        scale = self.scale.numpy()
        return ((features.astype(np.float32) - mean) / scale).astype(np.float32)


class PairEmbedsLabelsDatasetFromDisk(TorchDataset):
    def __init__(
            self,
            hf_dataset,
            col_a='SeqA',
            col_b='SeqB',
            label_col='labels',
            full=False,
            db_path='embeddings.db',
            batch_size=64,
            read_scaler=100,
            input_size=768,
            task_type='regression',
            train=True,
            random_pair_flipping=False,
            embedding_standardizer: Optional[EmbeddingStandardizer] = None,
            **kwargs
        ):
        self.seqs_a, self.seqs_b, self.labels = list(hf_dataset[col_a]), list(hf_dataset[col_b]), list(hf_dataset[label_col])
        self.db_file = db_path
        self.input_size = input_size
        self.full = full
        self.length = len(self.labels)
        self.task_type = task_type
        self.train = train
        self.random_pair_flipping = random_pair_flipping
        self.embedding_standardizer = embedding_standardizer
        self._cache_max = max(read_scaler * batch_size, 1)
        self._emb_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._conn = sqlite3.connect(self.db_file, timeout=30)
        self._cursor = self._conn.cursor()

    def __len__(self):
        return self.length

    def check_seqs(self, all_seqs):
        missing_seqs = [seq for seq in self.seqs_a + self.seqs_b if seq not in all_seqs]
        if missing_seqs:
            print_message(f'Sequences not found in embeddings: {missing_seqs}')
        else:
            print_message('All sequences in embeddings')

    def _get_emb(self, seq: str) -> torch.Tensor:
        cached = self._emb_cache.get(seq)
        if cached is not None:
            self._emb_cache.move_to_end(seq)
            return cached
        self._cursor.execute(
            "SELECT embedding FROM embeddings WHERE sequence = ?",
            (seq,),
        )
        row = self._cursor.fetchone()
        assert row is not None, f"Embedding not found for sequence of length {len(seq)}"
        emb = embedding_blob_to_tensor(row[0], fallback_shape=(-1, self.input_size))
        self._emb_cache[seq] = emb
        while len(self._emb_cache) > self._cache_max:
            self._emb_cache.popitem(last=False)
        return emb

    def __getitem__(self, idx):
        emb_a = self._get_emb(self.seqs_a[idx])
        emb_b = self._get_emb(self.seqs_b[idx])
        label = self.labels[idx]

        # Optional random pair order augmentation during training only.
        if self.train and self.random_pair_flipping and random.random() < 0.5:
            emb_a, emb_b = emb_b, emb_a

        if not self.full and self.embedding_standardizer is not None:
            pair_emb = torch.cat([emb_a.reshape(1, -1), emb_b.reshape(1, -1)], dim=-1)
            pair_emb = self.embedding_standardizer.transform_tensor(pair_emb)
            split_idx = pair_emb.shape[-1] // 2
            emb_a = pair_emb[:, :split_idx]
            emb_b = pair_emb[:, split_idx:]

        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return emb_a, emb_b, label


class PairEmbedsLabelsDataset(TorchDataset):
    def __init__(
            self,
            hf_dataset,
            emb_dict,
            col_a='SeqA',
            col_b='SeqB',
            full=False,
            label_col='labels',
            input_size=768,
            task_type='regression',
            train=True,
            random_pair_flipping=False,
            embedding_standardizer: Optional[EmbeddingStandardizer] = None,
            **kwargs
        ):
        self.seqs_a = list(hf_dataset[col_a])
        self.seqs_b = list(hf_dataset[col_b])
        self.labels = list(hf_dataset[label_col])
        self.input_size = input_size // 2 if not full else input_size # already scaled if ppi
        self.task_type = task_type
        self.full = full
        self.train = train
        self.random_pair_flipping = random_pair_flipping
        self.embedding_standardizer = embedding_standardizer

        # Combine seqs_a and seqs_b to find all unique sequences needed
        needed_seqs = set(list(hf_dataset[col_a]) + list(hf_dataset[col_b]))
        # Filter emb_dict to keep only the necessary embeddings
        self.emb_dict = {seq: emb_dict[seq] for seq in needed_seqs if seq in emb_dict}
        # Check for any missing embeddings
        missing_seqs = needed_seqs - self.emb_dict.keys()
        if missing_seqs:
            raise ValueError(f"Embeddings not found for sequences: {missing_seqs}")

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        seq_a, seq_b = self.seqs_a[idx], self.seqs_b[idx]
        emb_a = self.emb_dict[seq_a].reshape(-1, self.input_size)
        emb_b = self.emb_dict[seq_b].reshape(-1, self.input_size)
        
        # Optional random pair order augmentation during training only.
        if self.train and self.random_pair_flipping and random.random() < 0.5:
            emb_a, emb_b = emb_b, emb_a

        if not self.full and self.embedding_standardizer is not None:
            pair_emb = torch.cat([emb_a.reshape(1, -1), emb_b.reshape(1, -1)], dim=-1)
            pair_emb = self.embedding_standardizer.transform_tensor(pair_emb)
            split_idx = pair_emb.shape[-1] // 2
            emb_a = pair_emb[:, :split_idx]
            emb_b = pair_emb[:, split_idx:]

        # Prepare the label
        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)

        return emb_a, emb_b, label


class EmbedsLabelsDatasetFromDisk(TorchDataset):
    def __init__(
            self,
            hf_dataset,
            col_name='seqs',
            label_col='labels',
            full=False,
            db_path='embeddings.db',
            batch_size=64,
            read_scaler=100,
            input_size=768,
            task_type='singlelabel',
            embedding_standardizer: Optional[EmbeddingStandardizer] = None,
            **kwargs
        ):
        self.seqs, self.labels = list(hf_dataset[col_name]), list(hf_dataset[label_col])
        self.length = len(self.labels)
        self.max_length = len(max(self.seqs, key=len))
        print_message(f'Max length: {self.max_length}')

        self.db_file = db_path
        self.input_size = input_size
        self.full = full
        self.task_type = task_type
        self.embedding_standardizer = embedding_standardizer
        self._cache_max = max(read_scaler * batch_size, 1)
        self._emb_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._conn = sqlite3.connect(self.db_file, timeout=30)
        self._cursor = self._conn.cursor()

    def __len__(self):
        return self.length

    def check_seqs(self, all_seqs):
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
            return cached
        self._cursor.execute(
            "SELECT embedding FROM embeddings WHERE sequence = ?",
            (seq,),
        )
        row = self._cursor.fetchone()
        assert row is not None, f"Embedding not found for sequence of length {len(seq)}"
        emb = embedding_blob_to_tensor(row[0], fallback_shape=(-1, self.input_size))
        self._emb_cache[seq] = emb
        while len(self._emb_cache) > self._cache_max:
            self._emb_cache.popitem(last=False)
        return emb

    def __getitem__(self, idx):
        emb = self._get_emb(self.seqs[idx])
        if self.full:
            padding_needed = self.max_length - emb.size(0)
            emb = F.pad(emb, (0, 0, 0, padding_needed), value=0)
        elif self.embedding_standardizer is not None:
            emb = self.embedding_standardizer.transform_tensor(emb)
        label = self.labels[idx]
        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return emb.squeeze(0), label


class EmbedsLabelsDataset(TorchDataset):
    def __init__(
            self,
            hf_dataset,
            emb_dict,
            col_name='seqs',
            label_col='labels',
            task_type='singlelabel',
            full=False,
            embedding_standardizer: Optional[EmbeddingStandardizer] = None,
            **kwargs
        ):
        self.embeddings = self.get_embs(emb_dict, list(hf_dataset[col_name]))
        self.full = full
        self.labels = list(hf_dataset[label_col])
        self.task_type = task_type
        self.embedding_standardizer = embedding_standardizer
        self.max_length = len(max(list(hf_dataset[col_name]), key=len))
        print_message(f'Max length: {self.max_length}')

    def __len__(self):
        return len(self.labels)
    
    def get_embs(self, emb_dict, seqs):
        embeddings = []
        for seq in tqdm(seqs, desc='Loading Embeddings'):
            emb = emb_dict[seq]
            embeddings.append(emb)
        return embeddings

    def __getitem__(self, idx):
        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        emb = self.embeddings[idx].float()
        if self.full:
            padding_needed = self.max_length - emb.size(0)
            emb = F.pad(emb, (0, 0, 0, padding_needed), value=0)
        elif self.embedding_standardizer is not None:
            emb = self.embedding_standardizer.transform_tensor(emb)
        return emb.squeeze(0), label
    

class StringLabelDataset(TorchDataset):    
    def __init__(self, hf_dataset, col_name='seqs', label_col='labels', **kwargs):
        self.seqs = list(hf_dataset[col_name])
        self.labels = list(hf_dataset[label_col])
        self.lengths = [len(seq) for seq in self.seqs]

    def avg(self):
        return sum(self.lengths) / len(self.lengths)

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]
        return seq, label
    

class PairStringLabelDataset(TorchDataset):
    def __init__(self, hf_dataset, col_a='SeqA', col_b='SeqB', label_col='labels', train=True, random_pair_flipping=False, **kwargs):
        self.seqs_a, self.seqs_b = list(hf_dataset[col_a]), list(hf_dataset[col_b])
        self.labels = list(hf_dataset[label_col])
        self.train = train
        self.random_pair_flipping = random_pair_flipping

    def avg(self):
        return sum(len(seqa) + len(seqb) for seqa, seqb in zip(self.seqs_a, self.seqs_b)) / len(self.seqs_a)

    def __len__(self):
        return len(self.seqs_a)

    def __getitem__(self, idx):
        seq_a, seq_b = self.seqs_a[idx], self.seqs_b[idx]
        if self.train and self.random_pair_flipping and random.random() < 0.5:
            seq_a, seq_b = seq_b, seq_a
        return seq_a, seq_b, self.labels[idx]


class SimpleProteinDataset(TorchDataset):
    """Simple dataset for protein sequences."""
    def __init__(self, sequences: List[str]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


class MultiEmbedsLabelsDatasetFromDisk(TorchDataset):
    def __init__(
            self,
            hf_dataset,
            seq_cols: List[str],
            label_col: str = 'labels',
            full: bool = False,
            db_path: str = 'embeddings.db',
            batch_size: int = 64,
            read_scaler: int = 100,
            input_size: int = 768,
            task_type: str = 'singlelabel',
            train: bool = True,
            embedding_standardizer: Optional[EmbeddingStandardizer] = None,
            **kwargs,
        ):
        self.seq_cols = seq_cols
        self.labels = list(hf_dataset[label_col])
        self.length = len(self.labels)
        self.full = full
        self.db_file = db_path
        self.input_size = input_size // len(seq_cols) if not full else input_size # already scaled if multi-column
        self.task_type = task_type
        self.train = train
        self.embedding_standardizer = embedding_standardizer
        self._cache_max = max(read_scaler * batch_size, 1)
        self._emb_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

        # Store sequences per column
        self.col_to_seqs = {col: list(hf_dataset[col]) for col in seq_cols}

        # Precompute max combined length for matrix embeddings from raw strings
        if self.full:
            def combined_len_at(i: int) -> int:
                return sum(len(self.col_to_seqs[c][i]) for c in self.seq_cols) + (len(self.seq_cols) - 1)
            self.max_length = max(combined_len_at(i) for i in range(self.length)) if self.length > 0 else 0

        self._conn = sqlite3.connect(self.db_file, timeout=30)
        self._cursor = self._conn.cursor()

    def __len__(self):
        return self.length

    def _get_emb(self, seq: str) -> torch.Tensor:
        cached = self._emb_cache.get(seq)
        if cached is not None:
            self._emb_cache.move_to_end(seq)
            return cached
        self._cursor.execute(
            "SELECT embedding FROM embeddings WHERE sequence = ?",
            (seq,),
        )
        row = self._cursor.fetchone()
        assert row is not None, f"Embedding not found for sequence of length {len(seq)}"
        emb = embedding_blob_to_tensor(row[0], fallback_shape=(-1, self.input_size))
        self._emb_cache[seq] = emb
        while len(self._emb_cache) > self._cache_max:
            self._emb_cache.popitem(last=False)
        return emb

    def _combine_matrix(self, parts: List[torch.Tensor]) -> torch.Tensor:
        if len(parts) == 0:
            return torch.zeros(0, self.input_size)
        sep = torch.zeros(1, self.input_size, dtype=parts[0].dtype)
        out = []
        for i, p in enumerate(parts):
            out.append(p)
            if i < len(parts) - 1:
                out.append(sep)
        return torch.cat(out, dim=0)

    def __getitem__(self, idx):
        parts = [self._get_emb(self.col_to_seqs[col][idx]) for col in self.seq_cols]
        if self.full:
            emb = self._combine_matrix(parts)
            if self.max_length:
                pad_needed = self.max_length - emb.size(0)
                if pad_needed > 0:
                    emb = F.pad(emb, (0, 0, 0, pad_needed), value=0)
        else:
            emb = torch.cat([p.reshape(1, -1) for p in parts], dim=-1)
            if self.embedding_standardizer is not None:
                emb = self.embedding_standardizer.transform_tensor(emb)

        label = self.labels[idx]
        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return emb.squeeze(0), label


class MultiEmbedsLabelsDataset(TorchDataset):
    def __init__(
            self,
            hf_dataset,
            seq_cols: List[str],
            label_col: str = 'labels',
            full: bool = False,
            emb_dict: Optional[Dict[str, torch.Tensor]] = None,
            input_size: int = 768,
            task_type: str = 'singlelabel',
            train: bool = True,
            embedding_standardizer: Optional[EmbeddingStandardizer] = None,
            **kwargs,
        ):
        self.seq_cols = seq_cols
        self.labels = list(hf_dataset[label_col])
        self.full = full
        self.input_size = input_size // len(seq_cols) if not full else input_size
        self.task_type = task_type
        self.train = train
        self.embedding_standardizer = embedding_standardizer

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
                if self.embedding_standardizer is not None:
                    emb = self.embedding_standardizer.transform_tensor(emb)
            self.embeddings.append(emb)

    def _combine_matrix(self, parts: List[torch.Tensor]) -> torch.Tensor:
        if len(parts) == 0:
            return torch.zeros(0, self.input_size)
        sep = torch.zeros(1, self.input_size, dtype=parts[0].dtype)
        out = []
        for i, p in enumerate(parts):
            out.append(p)
            if i < len(parts) - 1:
                out.append(sep)
        return torch.cat(out, dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        emb = self.embeddings[idx].float()
        return emb.squeeze(0), label
    
