import torch
import numpy as np
import random
import os
import sqlite3
from typing import List, Tuple, Dict, Optional
from glob import glob
from pandas import read_csv, read_excel
from datasets import load_dataset, Dataset
from dataclasses import dataclass
from utils import print_message
from .supported_datasets import supported_datasets, standard_data_benchmark


AMINO_ACIDS = set('LAGVSERTIPDKQNFYMHWCXBUOZ*')
CODONS = set('aA@bB#$%rRnNdDcCeEqQ^G&ghHiIj+MmlJLkK(fFpPoO=szZwSXTtxWyYuvUV]})')
DNA = set('ATCG')
RNA = set('AUCG')


@dataclass
class DataArguments:
    """
    Args:
    data_paths: List[str]
        paths to the datasets
    max_length: int
        max length of sequences
    trim: bool
        whether to trim sequences to max_length
    """
    def __init__(
            self,
            data_names: List[str],
            delimiter: str = ',',
            col_names: List[str] = ['seqs', 'labels'],
            max_length: int = 1024,
            trim: bool = False,
            data_dirs: Optional[List[str]] = [],
            **kwargs
        ):
        self.data_names = data_names
        self.data_dirs = data_dirs
        self.delimiter = delimiter
        self.col_names = col_names
        self.max_length = max_length
        self.trim = trim

        if len(data_names) > 0:
            if data_names[0] == 'standard_benchmark':
                self.data_paths = [supported_datasets[data_name] for data_name in standard_data_benchmark]
            else:
                self.data_paths = []
                for data_name in data_names:
                    if data_name in supported_datasets:
                        self.data_paths.append(supported_datasets[data_name])
                    else:
                        print(f'{data_name} not found in supported datasets')
                        print('We will attempt to load it from huggingface anyways, but this may not work')
                        self.data_paths.append(data_name)
        else:
            self.data_paths = []
        
        if data_dirs is not None:
            for dir in data_dirs:
                if not os.path.exists(dir):
                    raise FileNotFoundError(f'{dir} does not exist')


class DataMixin:
    def __init__(self, data_args: Optional[DataArguments] = None):
        # intialize defaults
        self._sql = False
        self._full = False
        self._max_length = 1024
        self._trim = False
        self._delimiter = ','
        self._col_names = ['seqs', 'labels']
        self.data_args = data_args

    def _not_regression(self, labels): # not a great assumption but works most of the time
        return all(isinstance(label, (int, float)) and label == int(label) for label in labels)

    def _encode_labels(self, labels, tag2id):
        return [torch.tensor([tag2id[tag] for tag in doc], dtype=torch.long) for doc in labels]

    def _label_type_checker(self, labels):
        ex = labels[0]
        if self._not_regression(labels):
            if isinstance(ex, list):
                label_type = 'multilabel'
            elif isinstance(ex, int) or isinstance(ex, float):
                label_type = 'singlelabel' # binary or multiclass
        elif isinstance(ex, str):
            label_type = 'string'
        else:
            label_type = 'regression'
        return label_type

    def _select_from_sql(self, c, seq, cast_to_torch=True):
        c.execute("SELECT embedding FROM embeddings WHERE sequence = ?", (seq,))
        embedding = np.frombuffer(c.fetchone()[0], dtype=np.float32).reshape(1, -1)
        if self._full:
            embedding = embedding.reshape(len(seq), -1)
        if cast_to_torch:
            embedding = torch.tensor(embedding)
        return embedding

    def _select_from_pth(self, emb_dict, seq, cast_to_np=False):
        embedding = emb_dict[seq].reshape(1, -1)
        if self._full:
            embedding = embedding.reshape(len(seq), -1)
        if cast_to_np:
            embedding = embedding.numpy()
        return embedding

    def _labels_to_numpy(self, labels):
        if isinstance(labels[0], list):
            return np.array(labels).flatten()
        else:
            return np.array([labels]).flatten()

    def _random_order(self, seq_a, seq_b):
        if random.random() < 0.5:
            return seq_a, seq_b
        else:
            return seq_b, seq_a

    def _truncate_pairs(self, ex):
        # Truncate longest first, but if that makes it shorter than the other, truncate that one
        seq_a, seq_b = ex['SeqA'], ex['SeqB']
        trunc_a, trunc_b = seq_a, seq_b
        while len(trunc_a) + len(trunc_b) > self._max_length:
            if len(trunc_a) > len(trunc_b):
                trunc_a = trunc_a[:-1]
            else:
                trunc_b = trunc_b[:-1]
        ex['SeqA'] = trunc_a
        ex['SeqB'] = trunc_b
        return ex

    def process_datasets(
            self,
            hf_datasets: List[Tuple[Dataset, Dataset, Dataset, bool]],
            data_names: List[str],
        )-> Tuple[Dict[str, Tuple[Dataset, Dataset, Dataset, int, str, bool]], List[str]]:
        max_length = self._max_length
        datasets, all_seqs = {}, set()
        for dataset, data_name in zip(hf_datasets, data_names):
            print_message(f'Processing {data_name}')
            train_set, valid_set, test_set, ppi = dataset
            print(train_set)
            print(valid_set)
            print(test_set)
            if self._trim: # trim by length if necessary
                original_train_size, original_valid_size, original_test_size = len(train_set), len(valid_set), len(test_set)
                if ppi:
                    train_set = train_set.filter(lambda x: len(x['SeqA']) + len(x['SeqB']) <= max_length)
                    valid_set = valid_set.filter(lambda x: len(x['SeqA']) + len(x['SeqB']) <= max_length)
                    test_set = test_set.filter(lambda x: len(x['SeqA']) + len(x['SeqB']) <= max_length)
                else:
                    train_set = train_set.filter(lambda x: len(x['seqs']) <= max_length)
                    valid_set = valid_set.filter(lambda x: len(x['seqs']) <= max_length)
                    test_set = test_set.filter(lambda x: len(x['seqs']) <= max_length)
            
                print_message(f'Trimmed {100 * round((original_train_size-len(train_set)) / original_train_size, 2)}% from train')
                print_message(f'Trimmed {100 * round((original_valid_size-len(valid_set)) / original_valid_size, 2)}% from valid')
                print_message(f'Trimmed {100 * round((original_test_size-len(test_set)) / original_test_size, 2)}% from test')

            else: # truncate to max_length
                if ppi:
                    train_set = train_set.map(self._truncate_pairs)
                    valid_set = valid_set.map(self._truncate_pairs)
                    test_set = test_set.map(self._truncate_pairs)
                else:
                    train_set = train_set.map(lambda x: {'seqs': x['seqs'][:max_length]})
                    valid_set = valid_set.map(lambda x: {'seqs': x['seqs'][:max_length]})
                    test_set = test_set.map(lambda x: {'seqs': x['seqs'][:max_length]})

            # sanitize
            # disabling sanitization for now
            if ppi:
                #train_set = train_set.map(lambda x: {'SeqA': ''.join(aa for aa in x['SeqA'] if aa in AMINO_ACIDS),
                #                                     'SeqB': ''.join(aa for aa in x['SeqB'] if aa in AMINO_ACIDS)})
                #valid_set = valid_set.map(lambda x: {'SeqA': ''.join(aa for aa in x['SeqA'] if aa in AMINO_ACIDS),
                #                                     'SeqB': ''.join(aa for aa in x['SeqB'] if aa in AMINO_ACIDS)})
                #test_set = test_set.map(lambda x: {'SeqA': ''.join(aa for aa in x['SeqA'] if aa in AMINO_ACIDS),
                #                                    'SeqB': ''.join(aa for aa in x['SeqB'] if aa in AMINO_ACIDS)})
                all_seqs.update(train_set['SeqA'] + train_set['SeqB'])
                all_seqs.update(valid_set['SeqA'] + valid_set['SeqB'])
                all_seqs.update(test_set['SeqA'] + test_set['SeqB'])
            else:
                #train_set = train_set.map(lambda x: {'seqs': ''.join(aa for aa in x['seqs'] if aa in AMINO_ACIDS)})
                #valid_set = valid_set.map(lambda x: {'seqs': ''.join(aa for aa in x['seqs'] if aa in AMINO_ACIDS)})
                #test_set = test_set.map(lambda x: {'seqs': ''.join(aa for aa in x['seqs'] if aa in AMINO_ACIDS)})
                all_seqs.update(train_set['seqs'])
                all_seqs.update(valid_set['seqs'])
                all_seqs.update(test_set['seqs'])
                
            # confirm the type of labels
            check_labels = valid_set['labels']
            label_type = self._label_type_checker(check_labels)

            if label_type == 'string': # might be string or multilabel
                example = valid_set['labels'][0]
                try:
                    import ast
                    new_ex = ast.literal_eval(example)
                    if isinstance(new_ex, list): # if ast runs correctly and is now a list it is multilabel labels
                        label_type = 'multilabel'
                        train_set = train_set.map(lambda ex: {'labels': ast.literal_eval(ex['labels'])})
                        valid_set = valid_set.map(lambda ex: {'labels': ast.literal_eval(ex['labels'])})
                        test_set = test_set.map(lambda ex: {'labels': ast.literal_eval(ex['labels'])})
                except:
                    label_type = 'string' # if ast throws error it is actually string

            if label_type == 'string': # if still string, it's for tokenwise classification
                train_labels = train_set['labels']
                unique_tags = set(tag for doc in train_labels for tag in doc)
                tag2id = {tag: id for id, tag in enumerate(sorted(unique_tags))}
                # add cls token to labels
                train_set = train_set.map(lambda ex: {'labels': self._encode_labels(ex['labels'], tag2id=tag2id)})
                valid_set = valid_set.map(lambda ex: {'labels': self._encode_labels(ex['labels'], tag2id=tag2id)})
                test_set = test_set.map(lambda ex: {'labels': self._encode_labels(ex['labels'], tag2id=tag2id)})
                label_type = 'tokenwise'
                num_labels = len(unique_tags)
            else:
                if label_type == 'regression':
                    num_labels = 1
                else: # if classification, get the total number of leabels
                    try:
                        num_labels = len(train_set['labels'][0])
                    except:
                        unique = np.unique(train_set['labels'])
                        max_label = max(unique) # sometimes there are missing labels
                        full_list = np.arange(0, max_label+1)
                        num_labels = len(full_list)
            datasets[data_name] = (train_set, valid_set, test_set, num_labels, label_type, ppi)

        all_seqs = list(all_seqs)
        all_seqs = sorted(all_seqs, key=len, reverse=True) # longest first
        return datasets, all_seqs

    def get_data(self):
        """
        Supports .csv, .tsv, .txt
        TODO fasta, fa, fna, etc.
        """
        datasets, data_names = [], []

        for data_path in self.data_args.data_paths:
            data_name = data_path.split('/')[-1]
            print_message(f'Loading {data_name}')
            dataset = load_dataset(data_path)
            if 'inverse' in data_name.lower():
                dataset = dataset.rename_columns({'seqs': 'labels', 'labels': 'seqs'})
            ppi = 'SeqA' in dataset['train'].column_names
            print_message(f'PPI: {ppi}')
            try:
                train_set, valid_set, test_set = dataset['train'], dataset['valid'], dataset['test']
            except:
                # No valid or test set, make 10% splits randomly
                train_set = dataset['train'].train_test_split(test_size=0.2, seed=42)
                valid_set = train_set['test']
                train_set = train_set['train']
                test_set = train_set.train_test_split(test_size=0.5, seed=42)
                test_set = test_set['test']

            if not ppi:
                print('Standardizing column names')
                seq_col = self.data_args.col_names[0]
                label_col = self.data_args.col_names[1]
                train_set = train_set.rename_columns({seq_col: 'seqs', label_col: 'labels'})
                valid_set = valid_set.rename_columns({seq_col: 'seqs', label_col: 'labels'})
                test_set = test_set.rename_columns({seq_col: 'seqs', label_col: 'labels'})
                # drop everything else
                print('Removing extras')
                train_set = train_set.remove_columns([col for col in train_set.column_names if col not in ['seqs', 'labels']])
                valid_set = valid_set.remove_columns([col for col in valid_set.column_names if col not in ['seqs', 'labels']])
                test_set = test_set.remove_columns([col for col in test_set.column_names if col not in ['seqs', 'labels']])

            datasets.append((train_set, valid_set, test_set, ppi))
            data_names.append(data_name)

        for data_dir in self.data_args.data_dirs:
            # local_data/taxon
            data_name = data_dir.split ('/')[-1]
            ppi = 'ppi' in data_dir.lower()
            train_path = glob(os.path.join(data_dir, 'train.*'))[0]
            valid_path = glob(os.path.join(data_dir, 'valid.*'))[0]
            test_path = glob(os.path.join(data_dir, 'test.*'))[0]
            if '.xlsx' in train_path:
                train_set = read_excel(train_path)
                valid_set = read_excel(valid_path)
                test_set = read_excel(test_path)
            else:
                train_set = read_csv(train_path, delimiter=self._delimiter)
                valid_set = read_csv(valid_path, delimiter=self._delimiter)
                test_set = read_csv(test_path, delimiter=self._delimiter)

            train_set = Dataset.from_pandas(train_set)
            valid_set = Dataset.from_pandas(valid_set)
            test_set = Dataset.from_pandas(test_set)

            if not ppi:
                print('Standardizing column names')
                seq_col = self.data_args.col_names[0]
                label_col = self.data_args.col_names[1]
                train_set = train_set.rename_columns({seq_col: 'seqs', label_col: 'labels'})
                valid_set = valid_set.rename_columns({seq_col: 'seqs', label_col: 'labels'})
                test_set = test_set.rename_columns({seq_col: 'seqs', label_col: 'labels'})
                # drop everything else
                print('Removing extras')
                train_set = train_set.remove_columns([col for col in train_set.column_names if col not in ['seqs', 'labels']])
                valid_set = valid_set.remove_columns([col for col in valid_set.column_names if col not in ['seqs', 'labels']])
                test_set = test_set.remove_columns([col for col in test_set.column_names if col not in ['seqs', 'labels']])

            datasets.append((train_set, valid_set, test_set, ppi))
            data_names.append(data_name)

        return self.process_datasets(hf_datasets=datasets, data_names=data_names)

    def get_embedding_dim_sql(self, save_path, test_seq):
        import sqlite3
        test_seq_len = len(test_seq) + 2
        
        with sqlite3.connect(save_path) as conn:
            c = conn.cursor()
            c.execute("SELECT embedding FROM embeddings WHERE sequence = ?", (test_seq,))
            test_embedding = c.fetchone()[0]
            test_embedding = torch.tensor(np.frombuffer(test_embedding, dtype=np.float32).reshape(1, -1))
        if self._full:
            try:
                test_embedding = test_embedding.reshape(test_seq_len, -1)
            except:
                test_embedding = test_embedding.reshape(test_seq_len - 1, -1) # some pLMs have only one special token added
        embedding_dim = test_embedding.shape[-1]
        return embedding_dim

    def get_embedding_dim_pth(self, emb_dict, test_seq):
        test_seq_len = len(test_seq) + 2

        test_embedding = emb_dict[test_seq]
        print(test_embedding.shape)
        if self._full:
            try:
                test_embedding = test_embedding.reshape(test_seq_len, -1)
            except:
                test_embedding = test_embedding.reshape(test_seq_len - 1, -1) # some pLMs have only one special token added
        else:
            test_embedding = test_embedding.reshape(1, -1)
        embedding_dim = test_embedding.shape[-1]
        return embedding_dim

    def build_vector_numpy_dataset_from_embeddings(
            self,
            model_name,
            train_seqs,
            valid_seqs,
            test_seqs,
        ):
        save_dir = self.embedding_args.embedding_save_dir
        train_array, valid_array, test_array = [], [], []
        if self._sql:
            import sqlite3
            save_path = os.path.join(save_dir, f'{model_name}_{self._full}.db')
            with sqlite3.connect(save_path) as conn:
                c = conn.cursor()
                for seq in train_seqs:
                    embedding = self._select_from_sql(c, seq, cast_to_torch=False)
                    train_array.append(embedding)

                for seq in valid_seqs:
                    embedding = self._select_from_sql(c, seq, cast_to_torch=False)
                    valid_array.append(embedding)

                for seq in test_seqs:
                    embedding = self._select_from_sql(c, seq, cast_to_torch=False)
                    test_array.append(embedding)
        else:
            save_path = os.path.join(save_dir, f'{model_name}_{self._full}.pth')
            emb_dict = torch.load(save_path)
            for seq in train_seqs:
                embedding = self._select_from_pth(emb_dict, seq, cast_to_np=True)
                train_array.append(embedding)
                
            for seq in valid_seqs:
                embedding = self._select_from_pth(emb_dict, seq, cast_to_np=True)
                valid_array.append(embedding)

            for seq in test_seqs:
                embedding = self._select_from_pth(emb_dict, seq, cast_to_np=True)
                test_array.append(embedding)
            del emb_dict

        train_array = np.concatenate(train_array, axis=0)
        valid_array = np.concatenate(valid_array, axis=0)
        test_array = np.concatenate(test_array, axis=0)
        
        if self._full: # average over the length of the sequence
            train_array = np.mean(train_array, axis=1)
            valid_array = np.mean(valid_array, axis=1)
            test_array = np.mean(test_array, axis=1)

        print_message('Numpy dataset shapes')
        print_message(f'Train: {train_array.shape}')
        print_message(f'Valid: {valid_array.shape}')
        print_message(f'Test: {test_array.shape}')
        return train_array, valid_array, test_array

    def build_pair_vector_numpy_dataset_from_embeddings(
            self,
            model_name,
            train_seqs_a,
            train_seqs_b,
            valid_seqs_a,
            valid_seqs_b,
            test_seqs_a,
            test_seqs_b,
        ):
        save_dir = self.embedding_args.embedding_save_dir
        train_array, valid_array, test_array = [], [], []
        if self._sql:
            save_path = os.path.join(save_dir, f'{model_name}_{self._full}.db')
            with sqlite3.connect(save_path) as conn:
                c = conn.cursor()
                for seq_a, seq_b in zip(train_seqs_a, train_seqs_b):
                    seq_a, seq_b = self._random_order(seq_a, seq_b)
                    embedding_a = self._select_from_sql(c, seq_a, cast_to_torch=False)
                    embedding_b = self._select_from_sql(c, seq_b, cast_to_torch=False)
                    train_array.append(np.concatenate([embedding_a, embedding_b], axis=-1))

                for seq_a, seq_b in zip(valid_seqs_a, valid_seqs_b):
                    seq_a, seq_b = self._random_order(seq_a, seq_b)
                    embedding_a = self._select_from_sql(c, seq_a, cast_to_torch=False)
                    embedding_b = self._select_from_sql(c, seq_b, cast_to_torch=False)
                    valid_array.append(np.concatenate([embedding_a, embedding_b], axis=-1))

                for seq_a, seq_b in zip(test_seqs_a, test_seqs_b):
                    seq_a, seq_b = self._random_order(seq_a, seq_b)
                    embedding_a = self._select_from_sql(c, seq_a, cast_to_torch=False)
                    embedding_b = self._select_from_sql(c, seq_b, cast_to_torch=False)
                    test_array.append(np.concatenate([embedding_a, embedding_b], axis=-1))
        else:
            save_path = os.path.join(save_dir, f'{model_name}_{self._full}.pth')
            emb_dict = torch.load(save_path)
            for seq_a, seq_b in zip(train_seqs_a, train_seqs_b):
                seq_a, seq_b = self._random_order(seq_a, seq_b)
                embedding_a = self._select_from_pth(emb_dict, seq_a, cast_to_np=True)
                embedding_b = self._select_from_pth(emb_dict, seq_b, cast_to_np=True)
                train_array.append(np.concatenate([embedding_a, embedding_b], axis=-1))

            for seq_a, seq_b in zip(valid_seqs_a, valid_seqs_b):
                seq_a, seq_b = self._random_order(seq_a, seq_b)
                embedding_a = self._select_from_pth(emb_dict, seq_a, cast_to_np=True)
                embedding_b = self._select_from_pth(emb_dict, seq_b, cast_to_np=True)
                valid_array.append(np.concatenate([embedding_a, embedding_b], axis=-1))

            for seq_a, seq_b in zip(test_seqs_a, test_seqs_b):
                seq_a, seq_b = self._random_order(seq_a, seq_b)
                embedding_a = self._select_from_pth(emb_dict, seq_a, cast_to_np=True)
                embedding_b = self._select_from_pth(emb_dict, seq_b, cast_to_np=True)
                test_array.append(np.concatenate([embedding_a, embedding_b], axis=-1))
            del emb_dict

        train_array = np.concatenate(train_array, axis=0)
        valid_array = np.concatenate(valid_array, axis=0)
        test_array = np.concatenate(test_array, axis=0)
        
        if self._full: # average over the length of the sequence
            train_array = np.mean(train_array, axis=1)
            valid_array = np.mean(valid_array, axis=1)
            test_array = np.mean(test_array, axis=1)

        print_message('Numpy dataset shapes')
        print_message(f'Train: {train_array.shape}')
        print_message(f'Valid: {valid_array.shape}')
        print_message(f'Test: {test_array.shape}')
        return train_array, valid_array, test_array

    def prepare_scikit_dataset(self, model_name, dataset):
        train_set, valid_set, test_set, _, label_type, ppi = dataset

        if ppi:
            X_train, X_valid, X_test = self.build_pair_vector_numpy_dataset_from_embeddings(
                model_name,
                train_set['SeqA'],
                train_set['SeqB'],
                valid_set['SeqA'],
                valid_set['SeqB'],
                test_set['SeqA'],
                test_set['SeqB'],
            )
        else:
            X_train, X_valid, X_test = self.build_vector_numpy_dataset_from_embeddings(
                model_name,
                train_set['seqs'],
                valid_set['seqs'],
                test_set['seqs'],
            )

        y_train = self._labels_to_numpy(train_set['labels'])
        y_valid = self._labels_to_numpy(valid_set['labels'])
        y_test = self._labels_to_numpy(test_set['labels'])

        print_message('Numpy dataset shapes with labels')
        print_message(f'Train: {X_train.shape}, {y_train.shape}')
        print_message(f'Valid: {X_valid.shape}, {y_valid.shape}')
        print_message(f'Test: {X_test.shape}, {y_test.shape}')
        return X_train, y_train, X_valid, y_valid, X_test, y_test, label_type
