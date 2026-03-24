import torch
from typing import Any, Dict, List, Tuple, Union
from .utils import pad_and_concatenate_dimer


def _tokenize_kwargs(padding: str, max_length: int) -> Dict[str, Any]:
    """Build tokenizer kwargs for the given padding strategy."""
    kwargs: Dict[str, Any] = dict(padding=padding, return_tensors='pt', add_special_tokens=True, truncation=True)
    if padding == 'max_length':
        kwargs['max_length'] = max_length
    return kwargs


def _pad_matrix_embeds(embeds: List[torch.Tensor], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # pad and concatenate, return padded embeds and mask
    padded_embeds, attention_masks = [], []
    for embed in embeds:
        seq_len = embed.size(0)
        padding_size = max_len - seq_len
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones(max_len, dtype=torch.long)
        if padding_size > 0:
            attention_mask[seq_len:] = 0
            
            # Pad along the sequence dimension (dim=0)
            padding = torch.zeros((padding_size, embed.size(1)), dtype=embed.dtype)
            padded_embed = torch.cat((embed, padding), dim=0)
        else:
            padded_embed = embed
            
        padded_embeds.append(padded_embed)
        attention_masks.append(attention_mask)
        
    return torch.stack(padded_embeds), torch.stack(attention_masks)


class StringCollator:
    def __init__(self, tokenizer: object, padding: str = 'max_length', max_length: int = 2048, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch: Tuple[List[str], List[str]]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(batch, **_tokenize_kwargs(self.padding, self.max_length))


class StringLabelsCollator:
    def __init__(self, tokenizer: object, task_type: str = 'regression', tokenwise: bool = False, padding: str = 'max_length', max_length: int = 2048, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.tokenwise = tokenwise
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch: List[Tuple[str, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        seqs = [ex[0] for ex in batch]
        labels = [ex[1] for ex in batch]

        # Tokenize the sequences
        batch_encoding = self.tokenizer(
            seqs,
            **_tokenize_kwargs(self.padding, self.max_length),
        )
        
        # Handle labels based on tokenwise flag
        if self.tokenwise:
            # For token-wise labels, we need to pad to match the tokenized sequence length
            attention_mask = batch_encoding['attention_mask']
            lengths = [torch.sum(attention_mask[i]).item() for i in range(len(batch))]
            max_length = max(lengths)

            padded_labels = []
            for label in labels:
                if not isinstance(label, torch.Tensor):
                    label = torch.tensor(label)

                label = label.flatten()
                padding_size = max_length - len(label)
                # Pad or truncate labels to match tokenized sequence length
                if padding_size > 0:
                    # Pad with -100 (ignored by loss functions)
                    padding = torch.full((padding_size,), -100, dtype=label.dtype)
                    padded_label = torch.cat((label, padding))
                else:
                    padded_label = label[:max_length]
                padded_labels.append(padded_label)
            
            # Stack all padded labels
            batch_encoding['labels'] = torch.stack(padded_labels)
        else:
            # For sequence-level labels, just stack them
            batch_encoding['labels'] = torch.stack([torch.tensor(ex[1]) for ex in batch])

        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            batch_encoding['labels'] = batch_encoding['labels'].float()
        else:
            batch_encoding['labels'] = batch_encoding['labels'].long()
        
        return batch_encoding


class EmbedsLabelsCollator:
    def __init__(self, full: bool = False, task_type: str = 'regression', tokenwise: bool = False, **kwargs) -> None:
        self.full = full
        self.task_type = task_type
        self.tokenwise = tokenwise
        
    def __call__(self, batch: List[Tuple[torch.Tensor, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        if self.full:
            embeds = [ex[0] for ex in batch]
            labels = [ex[1] for ex in batch]
            
            # Find max sequence length for padding
            max_length = max(embed.size(0) for embed in embeds)
            
            embeds, attention_mask = _pad_matrix_embeds(embeds, max_length)
            
            # Pad labels
            if self.tokenwise:
                padded_labels = []
                for label in labels:
                    if not isinstance(label, torch.Tensor):
                        label = torch.tensor(label)

                    label = label.flatten()
                    padding_size = max_length - len(label)
                    if padding_size > 0:
                        # Use -100 as padding value for labels (ignored by loss functions)
                        padding = torch.full((padding_size,), -100, dtype=label.dtype)
                        padded_label = torch.cat((label, padding))
                    else:
                        padded_label = label[:max_length]
                    padded_labels.append(padded_label)
            else:
                padded_labels = labels

            labels = torch.stack(padded_labels)

            if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
                labels = labels.float()
            else:
                labels = labels.long()

            return {
                'embeddings': embeds,
                'attention_mask': attention_mask,
                'labels': labels,
            }
        else:
            embeds = torch.stack([ex[0] for ex in batch])
            labels = torch.stack([ex[1] for ex in batch])

            if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
                labels = labels.float()
            else:
                labels = labels.long()

            return {
                'embeddings': embeds,
                'labels': labels
            }


class PairCollator_input_ids:
    def __init__(self, tokenizer: object, padding: str = 'max_length', max_length: int = 2048, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch: List[Tuple[str, str, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        seqs_a, seqs_b, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.float)
        tok_kwargs = _tokenize_kwargs(self.padding, self.max_length)
        tok_kwargs.pop('add_special_tokens', None)
        tokenized = self.tokenizer(
            seqs_a, seqs_b,
            **tok_kwargs,
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }


class PairCollator_ab:
    def __init__(self, tokenizer: object, padding: str = 'max_length', max_length: int = 2048, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch: List[Tuple[str, str, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        seqs_a, seqs_b, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.float)
        tok_kwargs = _tokenize_kwargs(self.padding, self.max_length)
        tokenized_a = self.tokenizer(
            seqs_a,
            **tok_kwargs,
        )
        tokenized_b = self.tokenizer(
            seqs_b,
            **tok_kwargs,
        )
        return {
            'input_ids_a': tokenized_a['input_ids'],
            'input_ids_b': tokenized_b['input_ids'],
            'attention_mask_a': tokenized_a['attention_mask'],
            'attention_mask_b': tokenized_b['attention_mask'],
            'labels': labels
        }


class PairEmbedsLabelsCollator:
    def __init__(self, full: bool = False, add_token_ids: bool = False, **kwargs) -> None:
        self.full = full
        self.add_token_ids = add_token_ids
        
    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        if self.full:
            embeds_a = [ex[0] for ex in batch]
            embeds_b = [ex[1] for ex in batch]
            max_len_a = max(embed.size(0) for embed in embeds_a)
            max_len_b = max(embed.size(0) for embed in embeds_b)
            embeds_a, attention_mask_a = _pad_matrix_embeds(embeds_a, max_len_a)
            embeds_b, attention_mask_b = _pad_matrix_embeds(embeds_b, max_len_b)
            embeds, attention_mask = pad_and_concatenate_dimer(embeds_a, embeds_b, attention_mask_a, attention_mask_b)

            labels = torch.stack([ex[2] for ex in batch])

            # For tasks requiring token type IDs, provide them so the model knows
            # which tokens belong to protein A vs protein B
            if self.add_token_ids:
                batch_size = embeds.size(0)
                max_len = embeds.size(1)
                token_type_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
                for i in range(batch_size):
                    a_len = int(attention_mask_a[i].sum().item())
                    b_len = int(attention_mask_b[i].sum().item())
                    # type 0 for protein A, type 1 for protein B
                    token_type_ids[i, a_len:a_len + b_len] = 1
                return {
                    'embeddings': embeds,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'labels': labels
                }

            return {
                'embeddings': embeds,
                'attention_mask': attention_mask,
                'labels': labels
            }
        else:
            embeds_a = torch.stack([ex[0] for ex in batch])
            embeds_b = torch.stack([ex[1] for ex in batch]) 
            labels = torch.stack([ex[2] for ex in batch])
            embeds = torch.cat([embeds_a, embeds_b], dim=-1)
            return {
                'embeddings': embeds,
                'labels': labels
            }


class OneHotCollator:
    def __init__(self, alphabet: str = "ACDEFGHIKLMNPQRSTVWY") -> None:
        # Add X for unknown amino acids, and special CLS and EOS tokens
        alphabet = alphabet + "X"
        self.alphabet = list(alphabet)
        self.mapping = {token: idx for idx, token in enumerate(self.alphabet)}
        
    def __call__(self, batch: List[Tuple]) -> Dict[str, torch.Tensor]:
        seqs = [ex[0] for ex in batch]
        labels = torch.stack([torch.tensor(ex[1]) for ex in batch])
        
        # Find the longest sequence in the batch
        max_len = max(len(seq) for seq in seqs)
        
        # One-hot encode and pad each sequence
        one_hot_tensors, attention_masks = [], []
        
        for seq in seqs:
            seq = list(seq)
            # Create one-hot encoding for each sequence (including CLS and EOS)
            seq_len = len(seq)
            one_hot = torch.zeros(seq_len, len(self.alphabet))
            
            # Add sequence tokens in the middle
            for pos, token in enumerate(seq):
                if token in self.mapping:
                    one_hot[pos, self.mapping[token]] = 1.0
                else:
                    # For non-canonical amino acids, use the X token
                    one_hot[pos, self.mapping["X"]] = 1.0
            
            # Create attention mask (1 for actual tokens, 0 for padding)
            attention_mask = torch.ones(seq_len)
            
            # Pad to the max length in this batch
            padding_size = max_len - seq_len
            if padding_size > 0:
                padding = torch.zeros(padding_size, len(self.alphabet))
                one_hot = torch.cat([one_hot, padding], dim=0)
                # Add zeros to attention mask for padding
                mask_padding = torch.zeros(padding_size)
                attention_mask = torch.cat([attention_mask, mask_padding], dim=0)
            
            one_hot_tensors.append(one_hot)
            attention_masks.append(attention_mask)
        
        # Stack all tensors in the batch
        embeddings = torch.stack(one_hot_tensors)
        attention_masks = torch.stack(attention_masks)
        
        return {
            'embeddings': embeddings,
            'attention_mask': attention_masks,
            'labels': labels,
        }
