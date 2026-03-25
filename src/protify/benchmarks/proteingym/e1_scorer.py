import sys
import os
from collections import defaultdict
from collections.abc import Sequence
from enum import Enum
from typing import Any, Dict, List, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers.utils import logging

_FASTPLMS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'fastplms')
if _FASTPLMS not in sys.path:
    sys.path.insert(0, _FASTPLMS)

from fastplms.e1.modeling_e1 import E1ForMaskedLM, DataPrepConfig
from .e1_predictor import E1Predictor

logger = logging.get_logger(__name__)


class EncoderScoreMethod(str, Enum):
    WILDTYPE_MARGINAL = "wildtype_marginal"
    MASKED_MARGINAL = "masked_marginal"


def find_mismatches(s1: Union[str, np.ndarray], s2: str) -> List[int]:
    assert isinstance(s1, (str, np.ndarray)), f"s1 must be a string or numpy array, got {type(s1)}"
    assert isinstance(s2, str), f"s2 must be a string, got {type(s2)}"
    assert len(s1) == len(s2), f"s1 and s2 must have the same length, got {len(s1)} and {len(s2)}"
    s1_arr = np.frombuffer(s1.encode(), dtype=np.uint8) if isinstance(s1, str) else s1
    s2_arr = np.frombuffer(s2.encode(), dtype=np.uint8)
    return np.where(s1_arr != s2_arr)[0]


class E1Scorer:
    """
    This scorer is used to score sequences against a parent sequence for E1 models.

    Args:
        model: The E1 model to use for scoring.
        method: The scoring method to use. Either "wildtype_marginal" or "masked_marginal".
        data_prep_config: The data preparation config to use.
        max_batch_tokens: The maximum number of tokens to batch in a single forward pass.
        context_seqs: The context sequences to use for scoring. A dictionary with context ids as keys and
            context sequences as values.
        context_reduction: The context reduction to use. Either "mean" or "none". If "none" is used,
            return the scores for each sequence against each context separately, otherwise return the mean
            over all contexts.
    """

    def __init__(
        self,
        model: E1ForMaskedLM,
        method: EncoderScoreMethod,
        data_prep_config: DataPrepConfig | None = None,
        max_batch_tokens: int = 65536,
    ):
        self.predictor = E1Predictor(
            model,
            data_prep_config=DataPrepConfig(remove_X_tokens=True),
            max_batch_tokens=max_batch_tokens,
            fields_to_save=["logits"],
            save_masked_positions_only=(method == EncoderScoreMethod.MASKED_MARGINAL),
            keep_predictions_in_gpu=True,
            use_cache=True,
        )

        self.method = method
        self.vocab = self.predictor.batch_preparer.tokenizer.get_vocab()

        self.vocab_size = len(self.vocab)

    def mask_sequence(self, sequence: str, mask_position: int) -> str:
        """
        Mask a given position in a sequence.

        Args:
            sequence: The sequence to mask.
            mask_position: The position to mask.

        Returns:
            str: The masked sequence.
        """
        return sequence[:mask_position] + self.predictor.batch_preparer.mask_token + sequence[mask_position + 1 :]

    def find_all_mutated_positions(self, parent_sequence: str, sequences: Sequence[str]) -> List[int]:
        """
        Find all positions in the parent that are mutated in at least one of the sequences.

        Args:
            parent_sequence: The parent sequence.
            sequences: The sequences to check for mutations.

        Returns:
            list[int]:A list of positions that are mutated in at least one of the sequences.
        """
        encoded_parent = np.frombuffer(parent_sequence.encode(), dtype=np.uint8)
        mismatches = [pos for seq in sequences for pos in find_mismatches(encoded_parent, seq)]
        return sorted(set(mismatches))

    def score(
        self,
        parent_sequence: str,
        sequences: Sequence[str],
        sequence_ids: Sequence[int | str] | None = None,
        context_seqs: dict[str, str] | None = None,
        context_reduction: str = "mean",
    ) -> List[Dict[str, Any]]:
        """
        Score a given parent sequence against a list of sequences.

        Args:
            parent_sequence: The parent sequence.
            sequences: The sequences to score.
            sequence_ids: The ids of the sequences.
            context_seqs: The context sequences to use for scoring. A dictionary with context ids as keys and
                context sequences as values.

        Returns:
            list[dict[str, Any]]: A list of dictionaries with the score for each sequence against the parent.
            Dictionary format:
            {
                "id": The id of the sequence.
                "context_id": The id of the context.
                "score": The score for the sequence against the parent.
            }
            If context_reduction is "mean", the context_id will be "mean", otherwise
            it will be the id of the context against which the score is computed.

        Raises:
            ValueError: Invalid scoring method.

        Asserts:
            - Parent sequence must be uppercase and contain only A-Z except for X.
            - Evaluated sequences must be uppercase and contain only A-Z except for X.
            - All sequences must have the same length.
            - Parent sequence must have the same length as the mutants to score.
        """
        assert parent_sequence.isalpha() and parent_sequence.isupper() and "X" not in parent_sequence, (
            "Parent sequence must be uppercase and contain only A-Z except for X"
        )
        assert all(seq.isalpha() and seq.isupper() and "X" not in seq for seq in sequences), (
            "Evaluated sequences must be uppercase and contain only A-Z except for X"
        )
        assert len({len(seq) for seq in sequences}) == 1, "All sequences must have the same length"
        assert len(parent_sequence) == len(sequences[0]), (
            "Parent sequence must have the same length as the mutants to score."
        )

        if self.method == EncoderScoreMethod.MASKED_MARGINAL:
            return self._score_masked_marginal(
                parent_sequence, sequences, sequence_ids, context_seqs, context_reduction
            )

        mutation_positions = self.find_all_mutated_positions(parent_sequence, sequences)
        aggregated_position_scores, context_id_to_index = self.get_position_scores(
            parent_sequence, mutation_positions, context_seqs, context_reduction
        )

        encoded_parent = np.frombuffer(parent_sequence.encode(), dtype=np.uint8)
        scores = []
        for i, seq in tqdm(enumerate(sequences), total=len(sequences), desc="Scoring sequences against parent"):
            seq_id = sequence_ids[i] if sequence_ids is not None else i
            mismatch_positions = find_mismatches(encoded_parent, seq)
            seq_aa = [self.vocab[seq[pos]] for pos in mismatch_positions]
            score = aggregated_position_scores[:, mismatch_positions, seq_aa]
            # parent_aa = [self.vocab[parent_sequence[pos]] for pos in mismatch_positions]
            # score = (
            #     aggregated_log_probs[:, mismatch_positions, seq_aa]
            #     - aggregated_log_probs[:, mismatch_positions, parent_aa]
            # )
            match context_reduction:
                case "mean":
                    scores.append({"id": seq_id, "context_id": "mean", "score": score.sum().item()})
                case "none":
                    score = score.sum(dim=-1)
                    scores.extend(
                        [
                            {"id": seq_id, "context_id": context_id, "score": score[i].item()}
                            for context_id, i in context_id_to_index.items()
                        ]
                    )
                case _:
                    raise ValueError(f"Invalid context reduction: {context_reduction}")

        return scores

    def _score_masked_marginal(
        self,
        parent_sequence: str,
        sequences: Sequence[str],
        sequence_ids: Sequence[int | str] | None = None,
        context_seqs: dict[str, str] | None = None,
        context_reduction: str = "mean",
    ) -> List[Dict[str, Any]]:
        """
        Masked marginal scoring that masks all mutation sites simultaneously
        and groups variants by mutation positions, so that they can share a single forward pass.
        """

        encoded_parent = np.frombuffer(parent_sequence.encode(), dtype=np.uint8)
        mask_token = self.predictor.batch_preparer.mask_token

        # Group variants by their mutation positions tuple
        # pos_tuple -> list of (variant, parent, variant_aa_ids, parent_aa_ids)
        position_groups: dict[tuple[int, ...], list[tuple[int, Any, list[int], list[int]]]] = defaultdict(list)

        for variant, (seq, parent) in enumerate(zip(sequences, sequence_ids)):
            mismatches = tuple(sorted(find_mismatches(encoded_parent, seq)))
            variant_aa_ids = [self.vocab[seq[p]] for p in mismatches]
            parent_aa_ids = [self.vocab[parent_sequence[p]] for p in mismatches]
            position_groups[mismatches].append((variant, parent, variant_aa_ids, parent_aa_ids))

        # Create one masked sequence per unique position set
        masked_seqs_to_score: List[str] = []
        pos_tuple_to_group_idx: dict[tuple[int, ...], int] = {}

        for pos_tuple in position_groups.keys():
            # Create masked sequence with all positions masked simultaneously
            s_list = list(parent_sequence)
            for p in pos_tuple:
                s_list[p] = mask_token
            masked_seq = "".join(s_list)

            pos_tuple_to_group_idx[pos_tuple] = len(masked_seqs_to_score)
            masked_seqs_to_score.append(masked_seq)

        # Run prediction on unique masked sequences (one forward pass per unique position set)
        group_indices = list(range(len(masked_seqs_to_score)))
        predictions = list(self.predictor.predict(masked_seqs_to_score, group_indices, context_seqs=context_seqs))

        # Build log_probs lookup: 
        log_probs_by_group: dict[int, dict[Any, torch.Tensor]] = defaultdict(dict) # group_idx -> context_id -> log_probs tensor
        for p in predictions:
            idx = int(p["id"])
            context_id = p["context_id"]
            logits = p["logits"]  # [num_masked, vocab]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            log_probs_by_group[idx][context_id] = log_probs

        # Calculate scores for each variant using
        scores_by_variant_id: dict[int, list[dict]] = defaultdict(list)

        for pos_tuple, variants in position_groups.items():
            idx = pos_tuple_to_group_idx[pos_tuple]
            context_log_probs = log_probs_by_group[idx]

            for variant, parent, variant_aa_ids, parent_aa_ids in variants:
                for context_id, log_probs in context_log_probs.items():
                    # log_probs shape: [num_positions, vocab]
                    # Positions are in sorted order matching pos_tuple
                    mut_log_probs = log_probs[range(len(pos_tuple)), variant_aa_ids]
                    wt_log_probs = log_probs[range(len(pos_tuple)), parent_aa_ids]
                    score = (mut_log_probs - wt_log_probs).sum().item()
                    scores_by_variant_id[variant].append({"context_id": context_id, "score": score})

        # Aggregate and return final scores in original order
        final_scores = []
        for seq_idx in range(len(sequences)):
            seq_id = sequence_ids[seq_idx]
            seq_scores = scores_by_variant_id[seq_idx]

            if context_reduction == "mean":
                if not seq_scores:
                    avg_score = 0.0
                else:
                    avg_score = np.mean([s["score"] for s in seq_scores])
                final_scores.append({"id": seq_id, "context_id": "mean", "score": avg_score})
            elif context_reduction == "none":
                for s in seq_scores:
                    final_scores.append({"id": seq_id, "context_id": s["context_id"], "score": s["score"]})
            else:
                raise ValueError(f"Invalid context reduction: {context_reduction}")

        return final_scores

    def get_position_scores(
        self,
        parent_sequence: str,
        mutation_positions: Sequence[int],
        context_seqs: dict[str, str] | None = None,
        context_reduction: str = "mean",
    ) -> tuple[torch.Tensor, dict[str | None, int]]:
        context_id_to_index: dict[str | None, int] = (
            {context_id: i for i, context_id in enumerate(context_seqs.keys())}
            if context_seqs is not None
            else {None: 0}  # type: ignore[dict-item]
        )

        match self.method:
            case EncoderScoreMethod.WILDTYPE_MARGINAL:
                records_for_prediction = [(parent_sequence, None)]
            case EncoderScoreMethod.MASKED_MARGINAL:
                records_for_prediction = [
                    (self.mask_sequence(parent_sequence, mask_pos), mask_pos)  # type: ignore[misc]
                    for mask_pos in mutation_positions
                ]
            case _:
                raise ValueError(f"Invalid scoring method: {self.method}")

        # The logic below assumes that the predictor predicts the logits for a give multi-sequence instance only once across ranks.
        # In case of masked marginal scoring, it also assumes that a given masked version of the parent is also only predicted once across ranks.
        # This is assured right now by the validity filter in the predictor.predict function.
        # This assumption is why we can all reduce on aggregated logits using SUM below.
        num_contexts = len(context_id_to_index)
        logger.info(f"Predicting for {len(records_for_prediction)} records with {num_contexts} contexts")
        records, record_ids = zip(*records_for_prediction)
        predictions = list(self.predictor.predict(records, record_ids, context_seqs=context_seqs))
        parent_length = len(parent_sequence)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        aggregated_logits = torch.zeros(num_contexts, parent_length, self.vocab_size, device=device)
        for p in predictions:
            context_index = context_id_to_index[p["context_id"]]
            match self.method:
                case EncoderScoreMethod.WILDTYPE_MARGINAL:
                    # p["logits"].shape = (parent_length, vocab_size)
                    aggregated_logits[context_index] = p["logits"]
                case EncoderScoreMethod.MASKED_MARGINAL:
                    # p["id"] was set to the masked position when constructing the records for prediction.
                    # Since only once position is masked and the `save_masked_positions_only` flag is set to True for the predictor,
                    # we get the logits for the masked position only (p["logits"].shape = (1, vocab_size)).
                    aggregated_logits[context_index, p["id"]] = p["logits"][0]
                case _:
                    raise ValueError(f"Invalid scoring method: {self.method}")

        aggregated_log_probs = torch.nn.functional.log_softmax(aggregated_logits, dim=-1).cpu()

        if context_reduction == "mean":
            aggregated_log_probs = aggregated_log_probs.mean(dim=0, keepdim=True)

        parent_sequence_to_ids = [self.vocab[aa] for aa in parent_sequence]
        parent_log_probs = aggregated_log_probs[:, np.arange(parent_length), parent_sequence_to_ids]
        aggregated_position_scores = aggregated_log_probs - parent_log_probs.unsqueeze(-1)

        return aggregated_position_scores, context_id_to_index