### imports
import random
import torch
import numpy as np
import sqlite3
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from utils import print_message
from tqdm.auto import tqdm
from typing import List


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
            input_dim=768,
            task_type='regression',
            **kwargs
        ):
        self.seqs_a, self.seqs_b, self.labels = hf_dataset[col_a], hf_dataset[col_b], hf_dataset[label_col]
        self.db_file = db_path
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.full = full
        self.length = len(self.labels)
        self.read_amt = read_scaler * self.batch_size
        self.embeddings_a, self.embeddings_b, self.current_labels = [], [], []
        self.count, self.index = 0, 0
        self.task_type = task_type

    def __len__(self):
        return self.length

    def check_seqs(self, all_seqs):
        missing_seqs = [seq for seq in self.seqs_a + self.seqs_b if seq not in all_seqs]
        if missing_seqs:
            print_message(f'Sequences not found in embeddings: {missing_seqs}')
        else:
            print_message('All sequences in embeddings')

    def reset_epoch(self):
        data = list(zip(self.seqs_a, self.seqs_b, self.labels))
        random.shuffle(data)
        self.seqs_a, self.seqs_b, self.labels = zip(*data)
        self.seqs_a, self.seqs_b, self.labels = list(self.seqs_a), list(self.seqs_b), list(self.labels)
        self.embeddings_a, self.embeddings_b, self.current_labels = [], [], []
        self.count, self.index = 0, 0

    def get_embedding(self, c, seq):
        result = c.execute("SELECT embedding FROM embeddings WHERE sequence=?", (seq,))
        row = result.fetchone()
        if row is None:
            raise ValueError(f"Embedding not found for sequence: {seq}")
        emb_data = row[0]
        emb = torch.tensor(np.frombuffer(emb_data, dtype=np.float32).reshape(-1, self.input_dim))
        return emb

    def read_embeddings(self):
        embeddings_a, embeddings_b, labels = [], [], []
        self.count += self.read_amt
        if self.count >= self.length:
            self.reset_epoch()
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        for i in range(self.count, self.count + self.read_amt):
            if i >= self.length:
                break
            emb_a = self.get_embedding(c, self.seqs_a[i])
            emb_b = self.get_embedding(c, self.seqs_b[i])
            embeddings_a.append(emb_a)
            embeddings_b.append(emb_b)
            labels.append(self.labels[i])
        conn.close()
        self.index = 0
        self.embeddings_a = embeddings_a
        self.embeddings_b = embeddings_b
        self.current_labels = labels

    def __getitem__(self, idx):
        if self.index >= len(self.current_labels) or len(self.current_labels) == 0:
            self.read_embeddings()

        emb_a = self.embeddings_a[self.index]
        emb_b = self.embeddings_b[self.index]
        label = self.current_labels[self.index]

        self.index += 1

        # 50% chance to switch the order of a and b
        if random.random() < 0.5:
            emb_a, emb_b = emb_b, emb_a

        if self.task_type == 'multilabel' or self.task_type == 'regression':
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
            input_dim=768,
            task_type='regression',
            **kwargs
        ):
        self.seqs_a = hf_dataset[col_a]
        self.seqs_b = hf_dataset[col_b]
        self.labels = hf_dataset[label_col]
        self.input_dim = input_dim // 2 if not full else input_dim # already scaled if ppi
        self.task_type = task_type
        self.full = full

        # Combine seqs_a and seqs_b to find all unique sequences needed
        needed_seqs = set(hf_dataset[col_a] + hf_dataset[col_b])
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
        emb_a = self.emb_dict.get(seq_a).reshape(-1, self.input_dim)
        emb_b = self.emb_dict.get(seq_b).reshape(-1, self.input_dim)
        
        # 50% chance to switch the order of a and b
        if random.random() < 0.5:
            emb_a, emb_b = emb_b, emb_a

        # Prepare the label
        if self.task_type in ['multilabel', 'regression']:
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
            input_dim=768,
            task_type='singlelabel',
            **kwargs
        ): 
        self.seqs, self.labels = hf_dataset[col_name], hf_dataset[label_col]
        self.length = len(self.labels)
        self.max_length = len(max(self.seqs, key=len))
        print_message(f'Max length: {self.max_length}')

        self.db_file = db_path
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.full = full

        self.task_type = task_type
        self.read_amt = read_scaler * self.batch_size
        self.embeddings, self.current_labels = [], []
        self.count, self.index = 0, 0

        self.reset_epoch()

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

    def reset_epoch(self):
        data = list(zip(self.seqs, self.labels))
        random.shuffle(data)
        self.seqs, self.labels = zip(*data)
        self.seqs, self.labels = list(self.seqs), list(self.labels)
        self.embeddings, self.current_labels = [], []
        self.count, self.index = 0, 0

    def read_embeddings(self):
        embeddings, labels = [], []
        self.count += self.read_amt
        if self.count >= self.length:
            self.reset_epoch()
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        for i in range(self.count, self.count + self.read_amt):
            if i >= self.length:
                break
            result = c.execute("SELECT embedding FROM embeddings WHERE sequence=?", (self.seqs[i],))
            row = result.fetchone()
            emb_data = row[0]
            emb = torch.tensor(np.frombuffer(emb_data, dtype=np.float32).reshape(-1, self.input_dim))
            if self.full:
                padding_needed = self.max_length - emb.size(0)
                emb = F.pad(emb, (0, 0, 0, padding_needed), value=0)
            embeddings.append(emb)
            labels.append(self.labels[i])
        conn.close()
        self.index = 0
        self.embeddings = embeddings
        self.current_labels = labels

    def __getitem__(self, idx):
        if self.index >= len(self.current_labels) or len(self.current_labels) == 0:
            self.read_embeddings()

        emb = self.embeddings[self.index]
        label = self.current_labels[self.index]

        self.index += 1

        if self.task_type == 'multilabel' or self.task_type == 'regression':
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return emb.squeeze(0), label


class EmbedsLabelsDataset(TorchDataset):
    def __init__(self, hf_dataset, emb_dict, col_name='seqs', label_col='labels', task_type='singlelabel', full=False, **kwargs):
        self.embeddings = self.get_embs(emb_dict, hf_dataset[col_name])
        self.full = full
        self.labels = hf_dataset[label_col]
        self.task_type = task_type
        self.max_length = len(max(hf_dataset[col_name], key=len))
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
        if self.task_type == 'multilabel' or self.task_type == 'regression':
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        emb = self.embeddings[idx].float()
        if self.full:
            padding_needed = self.max_length - emb.size(0)
            emb = F.pad(emb, (0, 0, 0, padding_needed), value=0)
        return emb.squeeze(0), label
    

class StringLabelDataset(TorchDataset):    
    def __init__(self, hf_dataset, col_name='seqs', label_col='labels', **kwargs):
        self.seqs = hf_dataset[col_name]
        self.labels = hf_dataset[label_col]
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
    def __init__(self, hf_dataset, col_a='SeqA', col_b='SeqB', label_col='labels', train=True, **kwargs):
        self.seqs_a, self.seqs_b = hf_dataset[col_a], hf_dataset[col_b]
        self.labels = hf_dataset[label_col]
        self.train = train

    def avg(self):
        return sum(len(seqa) + len(seqb) for seqa, seqb in zip(self.seqs_a, self.seqs_b)) / len(self.seqs_a)

    def __len__(self):
        return len(self.seqs_a)

    def __getitem__(self, idx):
        seq_a, seq_b = self.seqs_a[idx], self.seqs_b[idx]
        if self.train and random.random() < 0.5:
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
    