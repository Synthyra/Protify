import itertools
import logging
import os
import sys
import torch
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import TypedDict
from tqdm import tqdm

_FASTPLMS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'fastplms',
    'src',
)
if _FASTPLMS not in sys.path:
    sys.path.insert(0, _FASTPLMS)

from fastplms.models.e1.modeling_e1 import (
    DataPrepConfig,
    E1BatchPreparer,
    E1ForMaskedLM,
    E1MaskedLMOutputWithPast,
    KVCache,
    get_context,
)

IndexedSequence = tuple[int, str]
logger = logging.getLogger(__name__)


class E1Prediction(TypedDict, total=False):
    id: str | int
    context_id: str | int | None
    logits: torch.Tensor
    token_embeddings: torch.Tensor
    mean_token_embeddings: torch.Tensor


class E1Predictor:
    def __init__(
        self,
        model: E1ForMaskedLM,
        data_prep_config: DataPrepConfig | None = None,
        max_batch_tokens: int = 65536,
        use_cache: bool = True,
        cache_size: int = 4,
        save_masked_positions_only: bool = False,
        fields_to_save: list[str] = ["logits", "token_embeddings", "mean_token_embeddings"],
        keep_predictions_in_gpu: bool = False,
    ) -> None:
        self.model = model
        self.max_batch_tokens = max_batch_tokens
        self.batch_preparer = E1BatchPreparer(data_prep_config=data_prep_config)
        self.model.eval()
        self.kv_cache = KVCache(cache_size=cache_size) if use_cache else None

        self.fields_to_save = fields_to_save
        self.save_masked_positions_only = save_masked_positions_only
        self.keep_predictions_in_gpu = keep_predictions_in_gpu

    def group_by_length(self, indexed_sequences: list[IndexedSequence]) -> list[list[IndexedSequence]]:
        batches: list[list[IndexedSequence]] = [[]]
        for idx, seq in sorted(indexed_sequences, key=lambda idx_seq: (len(idx_seq[1]), idx_seq[0])):
            if len(batches[-1]) > 0 and len(seq) * (len(batches[-1]) + 1) > self.max_batch_tokens:
                batches.append([])
            batches[-1].append((idx, seq))

        return batches

    def group_by_context(self, indexed_sequences: list[IndexedSequence]) -> list[list[IndexedSequence]]:
        batches: dict[str | None, list[IndexedSequence]] = defaultdict(list)
        for idx, seq in indexed_sequences:
            batches[get_context(seq)].append((idx, seq))
        return list(batches.values())

    def batch_sequences(self, sequences: Sequence[str]) -> list[tuple[list[int], bool]]:
        """
        Batches the sequences and returns indices for the current rank
        We want to keep sequences of similar length together.
        Ensures that no batch exceeds max_batch_tokens
        [For E1, also ensures if context is present, preserve locality of context]
        """
        indexed_sequences: list[IndexedSequence] = list(enumerate(sequences))
        indexed_batches = self.group_by_context(indexed_sequences)
        # Preserve context ordering
        indexed_batches = list(
            itertools.chain.from_iterable([self.group_by_length(batch) for batch in indexed_batches])
        )
        batches = [[item[0] for item in batch] for batch in indexed_batches]

        assert sorted(sum(batches, [])) == list(range(len(sequences))), (
            "Batches must contain all indices with no repetition"
        )

        batches_with_validity = [(b, True) for b in batches]

        return batches_with_validity

    @torch.no_grad()
    def predict_batch(
        self,
        sequences: list[str],
        sequence_metadata: list[dict[str, str | int]],
    ) -> list[E1Prediction]:
        """
        Returns the logits/embeddings for the last sequence for multi-sequence inputs.
        """
        # b is batch size, l is padded token length, c is vocabulary size,
        # d is hidden width, and r_i is the retained token count for item i.
        outputs = self.predict_batch_padded(sequences)
        outputs["logits"] = outputs["logits"].float()  # (b, l, c)
        outputs["embeddings"] = outputs["embeddings"].float()  # (b, l, d)

        token_mask = outputs["non_boundary_token_mask"] & outputs["last_sequence_mask"]  # (b, l)
        if self.save_masked_positions_only:
            token_mask = token_mask & outputs["mask_positions_mask"]  # (b, l)
        predictions = []
        for i in range(len(sequences)):
            pred: E1Prediction = {
                "id": sequence_metadata[i]["id"],
                "context_id": sequence_metadata[i].get("context_id", None),
            }
            if "logits" in self.fields_to_save:
                pred["logits"] = outputs["logits"][i, token_mask[i]]  # (r_i, c)
                if not self.keep_predictions_in_gpu:
                    pred["logits"] = pred["logits"].to("cpu")  # (r_i, c)
            if "token_embeddings" in self.fields_to_save:
                pred["token_embeddings"] = outputs["embeddings"][i, token_mask[i]]  # (r_i, d)
                if not self.keep_predictions_in_gpu:
                    pred["token_embeddings"] = pred["token_embeddings"].to("cpu")  # (r_i, d)
            if "mean_token_embeddings" in self.fields_to_save:
                pred["mean_token_embeddings"] = outputs["embeddings"][i, token_mask[i]].mean(dim=0)  # (d,)
                if not self.keep_predictions_in_gpu:
                    pred["mean_token_embeddings"] = pred["mean_token_embeddings"].to("cpu")  # (d,)
            predictions.append(pred)
        return predictions

    @torch.no_grad()
    def predict_batch_padded(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        """
        If use_cache is True, this function will return the logits/embeddings for the last sequence for multi-sequence inputs.
        If use_cache is False, this function will return the logits/embeddings for every sequence for multi-sequence inputs.

        Returns three additional masks:
        - non_boundary_token_mask: True for tokens that are part of the input sequence i.e not boundary tokens like 1, 2, <bos>, <eos>, <pad>, etc.
        - last_sequence_mask: True for tokens that are part of the last sequence (including boundary tokens) in case of multi-sequence input.
        - mask_positions_mask: True for masked positions.
        - valid_token_mask: True for valid tokens.
        """
        # b is batch size, l is padded token length, c is vocabulary size,
        # and d is hidden width. Batch token and position fields are (b, l).
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type, torch.bfloat16):
            batch = self.batch_preparer.get_batch_kwargs(sequences, device=torch.device(device_type))

            if self.kv_cache is not None:
                self.kv_cache.before_forward(batch)

            output: E1MaskedLMOutputWithPast = self.model(
                input_ids=batch["input_ids"],
                within_seq_position_ids=batch["within_seq_position_ids"],
                global_position_ids=batch["global_position_ids"],
                sequence_ids=batch["sequence_ids"],
                past_key_values=batch.get("past_key_values", None),
                use_cache=batch.get("use_cache", False),
                output_attentions=False,
                output_hidden_states=False,
            )
            if self.kv_cache is not None:
                self.kv_cache.after_forward(batch, output)

            logits = output.logits  # (b, l, c)
            embeddings = output.last_hidden_state  # (b, l, d)

            padding_mask = batch["input_ids"] == self.batch_preparer.pad_token_id  # (b, l)
            last_sequence_mask = (  # (b, l)
                batch["sequence_ids"] == batch["sequence_ids"].max(dim=1)[0][:, None]
            )
            boundary_token_mask = self.batch_preparer.get_boundary_token_mask(batch["input_ids"])  # (b, l)
            mask_positions_mask = self.batch_preparer.get_mask_positions_mask(batch["input_ids"])  # (b, l)

            return {
                "logits": logits,  # (b, l, c)
                "embeddings": embeddings,  # (b, l, d)
                "last_sequence_mask": last_sequence_mask,  # (b, l)
                "non_boundary_token_mask": ~boundary_token_mask,  # (b, l)
                "mask_positions_mask": mask_positions_mask,  # (b, l)
                "valid_token_mask": ~padding_mask,  # (b, l)
            }

    @torch.no_grad()
    def predict(
        self,
        sequences: Sequence[str],
        sequence_ids: Sequence[int | str] | None = None,
        context_seqs: dict[str, str] | None = None,
    ) -> Iterator[E1Prediction]:
        if sequence_ids is None:
            sequence_ids = list(range(len(sequences)))
        if context_seqs:
            sequences_with_context = [
                (ctx + "," + seq, {"context_id": ctx_id, "id": sequence_id})
                for ctx_id, ctx in context_seqs.items()
                for seq, sequence_id in zip(sequences, sequence_ids)
            ]
        else:
            sequences_with_context = [(seq, {"id": sequence_id}) for seq, sequence_id in zip(sequences, sequence_ids)]
        sequences, sequence_metadata = tuple(zip(*sequences_with_context))
        sequence_batch_indices: list[tuple[list[int], bool]] = self.batch_sequences(sequences)
        logger.info(f"Predicting for {len(sequence_batch_indices)} batches")

        for indices, is_valid_batch in tqdm(
            sequence_batch_indices, desc="Predicting batches"
        ):
            sequence_batch = [sequences[i] for i in indices]
            sequence_batch_metadata = [sequence_metadata[i] for i in indices]
            batch_predictions = self.predict_batch(sequence_batch, sequence_batch_metadata)

            if not is_valid_batch:
                continue

            for prediction in batch_predictions:
                yield prediction
