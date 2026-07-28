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
    """Pad ``(l_i, d)`` embeddings to a shared sequence length."""
    # embeds: b tensors shaped (l_i, d); max_len: l_max >= max_i(l_i)
    padded_embeds: List[torch.Tensor] = []
    attention_masks: List[torch.Tensor] = []
    for embed in embeds:
        seq_len = embed.size(0)  # l_i
        padding_size = max_len - seq_len  # l_max - l_i

        attention_mask = torch.ones(max_len, dtype=torch.long)  # (l_max,)
        if padding_size > 0:
            attention_mask[seq_len:] = 0  # (l_max - l_i,)
            padding = torch.zeros((padding_size, embed.size(1)), dtype=embed.dtype)  # (l_max - l_i, d)
            padded_embed = torch.cat((embed, padding), dim=0)  # (l_max, d)
        else:
            padded_embed = embed  # (l_max, d) for the supported l_i <= l_max contract

        padded_embeds.append(padded_embed)
        attention_masks.append(attention_mask)

    return (  # (b, l_max, d), (b, l_max)
        torch.stack(padded_embeds),
        torch.stack(attention_masks),
    )


class StringCollator:
    def __init__(
        self,
        tokenizer: object,
        padding: str = 'max_length',
        max_length: int = 2048,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch: Tuple[List[str], List[str]]) -> Dict[str, torch.Tensor]:
        # Tokenizer tensor fields are normally (b, l_t); arbitrary extra fields are preserved.
        return self.tokenizer(batch, **_tokenize_kwargs(self.padding, self.max_length))


class StringLabelsCollator:
    def __init__(
        self,
        tokenizer: object,
        task_type: str = 'regression',
        tokenwise: bool = False,
        padding: str = 'max_length',
        max_length: int = 2048,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.tokenwise = tokenwise
        self.padding = padding
        self.max_length = max_length

    def __call__(
        self,
        batch: List[Tuple[str, Union[float, int, List[float], List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        seqs = [ex[0] for ex in batch]
        labels = [ex[1] for ex in batch]

        batch_encoding = self.tokenizer(
            seqs,
            **_tokenize_kwargs(self.padding, self.max_length),
        )  # tensor fields normally (b, l_t)

        if self.tokenwise:
            attention_mask = batch_encoding['attention_mask']  # (b, l_t)
            lengths = [torch.sum(attention_mask[i]).item() for i in range(len(batch))]
            max_length = max(lengths)  # l_y, the largest valid-token count

            padded_labels: List[torch.Tensor] = []
            for label in labels:
                if not isinstance(label, torch.Tensor):
                    label = torch.tensor(label)  # () or (r_i,)

                label = label.flatten()  # (r_i,)
                padding_size = max_length - len(label)  # l_y - r_i
                if padding_size > 0:
                    padding = torch.full((padding_size,), -100, dtype=label.dtype)  # (l_y - r_i,)
                    padded_label = torch.cat((label, padding))  # (l_y,)
                else:
                    padded_label = label[:max_length]  # (l_y,)
                padded_labels.append(padded_label)

            batch_encoding['labels'] = torch.stack(padded_labels)  # (b, l_y)
        else:
            # Each label is scalar or has a shared task shape s_y.
            batch_encoding['labels'] = torch.stack(  # (b,) or (b, *s_y)
                [torch.tensor(ex[1]) for ex in batch]
            )

        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            batch_encoding['labels'] = batch_encoding['labels'].float()  # unchanged shape
        else:
            batch_encoding['labels'] = batch_encoding['labels'].long()  # unchanged shape

        return batch_encoding


class EmbedsLabelsCollator:
    def __init__(
        self,
        full: bool = False,
        task_type: str = 'regression',
        tokenwise: bool = False,
        **kwargs: Any,
    ) -> None:
        self.full = full
        self.task_type = task_type
        self.tokenwise = tokenwise

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if self.full:
            embeds = [ex[0] for ex in batch]  # b tensors shaped (l_i, d)
            labels = [ex[1] for ex in batch]  # b tensors shaped () or s_y
            max_length = max(embed.size(0) for embed in embeds)  # l_max

            embeds, attention_mask = _pad_matrix_embeds(embeds, max_length)  # (b, l_max, d), (b, l_max)

            if self.tokenwise:
                padded_labels: List[torch.Tensor] = []
                for label in labels:
                    if not isinstance(label, torch.Tensor):
                        label = torch.tensor(label)  # () or (r_i,)

                    label = label.flatten()  # (r_i,)
                    padding_size = max_length - len(label)  # l_max - r_i
                    if padding_size > 0:
                        padding = torch.full((padding_size,), -100, dtype=label.dtype)  # (l_max - r_i,)
                        padded_label = torch.cat((label, padding))  # (l_max,)
                    else:
                        padded_label = label[:max_length]  # (l_max,)
                    padded_labels.append(padded_label)
            else:
                padded_labels = labels  # b tensors shaped () or the shared s_y

            labels = torch.stack(padded_labels)  # (b, l_max), (b,), or (b, *s_y)

            if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
                labels = labels.float()  # unchanged shape
            else:
                labels = labels.long()  # unchanged shape

            return {
                'embeddings': embeds,
                'attention_mask': attention_mask,
                'labels': labels,
            }
        else:
            # Each pooled sample may have an arbitrary shared shape s_emb.
            embeds = torch.stack([ex[0] for ex in batch])  # (b, *s_emb)
            labels = torch.stack([ex[1] for ex in batch])  # (b,) or (b, *s_y)

            if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
                labels = labels.float()  # unchanged shape
            else:
                labels = labels.long()  # unchanged shape

            return {
                'embeddings': embeds,
                'labels': labels
            }


class PairCollator_input_ids:
    def __init__(
        self,
        tokenizer: object,
        padding: str = 'max_length',
        max_length: int = 2048,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch: List[Tuple[str, str, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        seqs_a, seqs_b, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.float)  # (b,)
        tok_kwargs = _tokenize_kwargs(self.padding, self.max_length)
        tok_kwargs.pop('add_special_tokens', None)
        tokenized = self.tokenizer(
            seqs_a, seqs_b,
            **tok_kwargs,
        )  # tensor fields normally (b, l_t)
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }


class PairCollator_ab:
    def __init__(
        self,
        tokenizer: object,
        padding: str = 'max_length',
        max_length: int = 2048,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch: List[Tuple[str, str, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        seqs_a, seqs_b, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.float)  # (b,)
        tok_kwargs = _tokenize_kwargs(self.padding, self.max_length)
        tokenized_a = self.tokenizer(
            seqs_a,
            **tok_kwargs,
        )  # tensor fields normally (b, l_a)
        tokenized_b = self.tokenizer(
            seqs_b,
            **tok_kwargs,
        )  # tensor fields normally (b, l_b)
        return {
            'input_ids_a': tokenized_a['input_ids'],
            'input_ids_b': tokenized_b['input_ids'],
            'attention_mask_a': tokenized_a['attention_mask'],
            'attention_mask_b': tokenized_b['attention_mask'],
            'labels': labels
        }


class PairEmbedsLabelsCollator:
    def __init__(
        self,
        full: bool = False,
        add_token_ids: bool = False,
        **kwargs: Any,
    ) -> None:
        self.full = full
        self.add_token_ids = add_token_ids

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if self.full:
            embeds_a = [ex[0] for ex in batch]  # b tensors shaped (l_a_i, d)
            embeds_b = [ex[1] for ex in batch]  # b tensors shaped (l_b_i, d)
            max_len_a = max(embed.size(0) for embed in embeds_a)  # l_a
            max_len_b = max(embed.size(0) for embed in embeds_b)  # l_b
            embeds_a, attention_mask_a = _pad_matrix_embeds(embeds_a, max_len_a)  # (b, l_a, d), (b, l_a)
            embeds_b, attention_mask_b = _pad_matrix_embeds(embeds_b, max_len_b)  # (b, l_b, d), (b, l_b)
            embeds, attention_mask = pad_and_concatenate_dimer(
                embeds_a,
                embeds_b,
                attention_mask_a,
                attention_mask_b,
            )  # (b, l_pair, d), (b, l_pair)

            labels = torch.stack([ex[2] for ex in batch])  # (b,) or (b, *s_y)

            if self.add_token_ids:
                batch_size = embeds.size(0)  # b
                max_len = embeds.size(1)  # l_pair
                token_type_ids = torch.zeros(batch_size, max_len, dtype=torch.long)  # (b, l_pair)
                for i in range(batch_size):
                    a_len = int(attention_mask_a[i].sum().item())  # l_a_i
                    b_len = int(attention_mask_b[i].sum().item())  # l_b_i
                    token_type_ids[i, a_len:a_len + b_len] = 1  # (b_len,)
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
            embeds_a = torch.stack([ex[0] for ex in batch])  # (b, *s, d_a)
            embeds_b = torch.stack([ex[1] for ex in batch])  # (b, *s, d_b)
            labels = torch.stack([ex[2] for ex in batch])  # (b,) or (b, *s_y)
            embeds = torch.cat([embeds_a, embeds_b], dim=-1)  # (b, *s, d_a + d_b)
            return {
                'embeddings': embeds,
                'labels': labels
            }


class OneHotCollator:
    def __init__(self, alphabet: str = "ACDEFGHIKLMNPQRSTVWY") -> None:
        alphabet = alphabet + "X"
        self.alphabet = list(alphabet)
        self.mapping = {token: idx for idx, token in enumerate(self.alphabet)}

    def __call__(self, batch: List[Tuple[str, Any]]) -> Dict[str, torch.Tensor]:
        seqs = [ex[0] for ex in batch]
        labels = torch.stack([torch.tensor(ex[1]) for ex in batch])  # (b,) or (b, *s_y)
        max_len = max(len(seq) for seq in seqs)  # l_max

        one_hot_tensors: List[torch.Tensor] = []
        attention_masks: List[torch.Tensor] = []

        for seq in seqs:
            seq = list(seq)
            seq_len = len(seq)  # l_i
            one_hot = torch.zeros(seq_len, len(self.alphabet))  # (l_i, c)

            for pos, token in enumerate(seq):
                if token in self.mapping:
                    one_hot[pos, self.mapping[token]] = 1.0  # ()
                else:
                    one_hot[pos, self.mapping["X"]] = 1.0  # ()

            attention_mask = torch.ones(seq_len)  # (l_i,)

            padding_size = max_len - seq_len  # l_max - l_i
            if padding_size > 0:
                padding = torch.zeros(padding_size, len(self.alphabet))  # (l_max - l_i, c)
                one_hot = torch.cat([one_hot, padding], dim=0)  # (l_max, c)
                mask_padding = torch.zeros(padding_size)  # (l_max - l_i,)
                attention_mask = torch.cat([attention_mask, mask_padding], dim=0)  # (l_max,)

            one_hot_tensors.append(one_hot)
            attention_masks.append(attention_mask)

        embeddings = torch.stack(one_hot_tensors)  # (b, l_max, c)
        attention_masks = torch.stack(attention_masks)  # (b, l_max)

        return {
            'embeddings': embeddings,
            'attention_mask': attention_masks,
            'labels': labels,
        }
