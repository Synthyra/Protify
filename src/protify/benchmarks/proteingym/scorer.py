import re
import os
import sys
import subprocess
import time
from typing import List
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
from tqdm.auto import tqdm


class SequenceProcessor:
    """Handles sequence slicing and mutation parsing for ProteinGym."""
    
    @staticmethod
    def get_optimal_window(mutation_position_relative: int, seq_len_wo_special: int, model_window: int) -> List[int]:
        """
        Select an optimal sequence window that fits the maximum model context size.
        If the sequence length is less than the maximum context size, the full sequence is returned.
        """
        half_model_window = model_window // 2
        if seq_len_wo_special <= model_window:
            return [0, seq_len_wo_special]
        elif mutation_position_relative < half_model_window:
            return [0, model_window]
        elif mutation_position_relative >= seq_len_wo_special - half_model_window:
            return [seq_len_wo_special - model_window, seq_len_wo_special]
        else:
            return [max(0, mutation_position_relative - half_model_window), 
                    min(seq_len_wo_special, mutation_position_relative + half_model_window)]

    @staticmethod
    def get_sequence_slices(df, target_seq, model_context_len, start_idx=1, 
                            scoring_window="optimal", indel_mode=False):
        """
        Process a dataframe containing mutant triplets (substitutions) or full mutated sequences (indels).
        Returns a processed DMS in which sequences have been sliced to satisfy the maximum context window.
        
        Modified from https://github.com/OATML-Markslab/Tranception
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe to be processed
        target_seq : str
            Full reference sequence (wild type)
        model_context_len : int
            Maximum context size for the model
        start_idx : int
            Integer to move to 0-indexing of positions
        scoring_window : str
            Method to slice sequences: "optimal" or "sliding"
        indel_mode : bool
            Flag for scoring insertions and deletions
        """
        len_target_seq = len(target_seq)
        num_mutants = len(df['mutated_seq'])
        df = df.reset_index(drop=True)
        
        if scoring_window == "optimal":
            if not indel_mode:
                df['mutation_barycenter'] = df['mutant'].apply(
                    lambda x: int(np.array([int(mutation[1:-1]) - start_idx for mutation in x.split(':')]).mean())
                )
                df['scoring_optimal_window'] = df['mutation_barycenter'].apply(
                    lambda x: SequenceProcessor.get_optimal_window(x, len_target_seq, model_context_len)
                )
            else:
                df['mutation_barycenter'] = df['mutated_seq'].apply(lambda x: len(x) // 2)
                df['scoring_optimal_window'] = df['mutated_seq'].apply(lambda x: (0, len(x)))
            
            df['sliced_mutated_seq'] = [
                df['mutated_seq'][index][df['scoring_optimal_window'][index][0]:df['scoring_optimal_window'][index][1]] 
                for index in range(num_mutants)
            ]
            df['window_start'] = df['scoring_optimal_window'].map(lambda x: x[0])
            df['window_end'] = df['scoring_optimal_window'].map(lambda x: x[1])
            del df['scoring_optimal_window'], df['mutation_barycenter']
            
            df_wt = df.copy()
            df_wt['mutated_seq'] = [target_seq] * num_mutants
            if indel_mode:
                df_wt['window_end'] = df_wt['mutated_seq'].map(lambda x: len(x))
            df_wt['sliced_mutated_seq'] = [
                target_seq[df_wt['window_start'][index]:df_wt['window_end'][index]] 
                for index in range(num_mutants)
            ]
            df = pd.concat([df, df_wt], axis=0)
            df = df.drop_duplicates()
            keep_cols = [c for c in ['mutant', 'target_seq', 'mutated_seq', 'window_start', 
                                     'window_end', 'sliced_mutated_seq'] if c in df.columns]
            df = df[keep_cols]
            
        elif scoring_window == "sliding":
            if model_context_len is None:
                model_context_len = len_target_seq
            df_list = []
            start = 0
            while start < len_target_seq:
                end = min(start + model_context_len, len_target_seq)
                df_sliced = df.copy()
                df_sliced['sliced_mutated_seq'] = df_sliced['mutated_seq'].map(lambda x: x[start:end])
                df_sliced['window_start'] = [start] * num_mutants
                df_sliced['window_end'] = df_sliced['mutated_seq'].map(lambda x: min(len(x), end))
                df_sliced_wt = df_sliced.copy()
                df_sliced_wt['mutated_seq'] = [target_seq] * num_mutants
                df_sliced_wt['sliced_mutated_seq'] = df_sliced_wt['mutated_seq'].map(lambda x: x[start:end])
                df_sliced_wt['window_end'] = df_sliced_wt['mutated_seq'].map(lambda x: min(len(x), end))
                df_list.append(df_sliced)
                df_list.append(df_sliced_wt)
                start = end
            df_final = pd.concat(df_list, axis=0)
            df = df_final.drop_duplicates()
            keep_cols = [c for c in ['mutant', 'target_seq', 'mutated_seq', 'window_start', 
                                     'window_end', 'sliced_mutated_seq'] if c in df.columns]
            df = df[keep_cols]
            
        return df.reset_index(drop=True)
    
    @staticmethod
    def parse_mutant_string(mutant: str) -> List[Tuple[str, int, str]]:
        """
        Parse a ProteinGym mutant string where each mutation is separated by ':'.
        Example: "I66N:H67T:S73C" -> [("I", 65, "N"), ("H", 66, "T"), ("S", 72, "C")]
        """
        if mutant is None or (isinstance(mutant, float) and np.isnan(mutant)):
            return []
        parts = str(mutant).split(':')
        parsed: List[Tuple[str, int, str]] = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            m = re.match(r"([A-Za-z*])([0-9]+)([A-Za-z*])", p)
            if not m:
                continue
            wt, pos, mt = m.groups()
            # -1 for 0-based indexing
            parsed.append((wt, int(pos) - 1, mt))
        return parsed
    
    @staticmethod
    def aa_to_token_ids(tokenizer) -> Dict[str, int]:
        """Precompute amino acid to token ID mapping."""
        amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        aa_to_id = {}
        for aa in amino_acids:
            token_id = tokenizer.convert_tokens_to_ids(aa)
            if token_id is not None:
                aa_to_id[aa] = token_id
        return aa_to_id


class ProteinGymScorer:
    """Scores protein variants using various scoring methods.
    
    Parameters
    ----------
    model_name : str
    model : Any
    tokenizer : Any
    device : torch.device
    batch_size : int
    """
    
    # Model context lengths (minus 2 for special tokens)
    MODEL_CONTEXT_LENGTH = {
        'ESM2-8': 1022, 
        'ESM2-35': 1022,
        'ESM2-150': 1022,
        'ESM2-650': 1022,
        'ESM2-3B': 1022,
        'ESMC-300': 2046,
        'ESMC-600': 2046,
        'ESMC-6B': 2046,
        'ProtBert': 1022,
        'ProtBert-BFD': 1022,
        'GLM2-150': 4095,
        'GLM2-650': 4095,
        'DSM-150': 1022,
        'DSM-650': 1022,
        'DPLM-150': 1022,
        'DPLM-650': 1022,
        'DPLM-3B': 1022,
        'DPLM2-150': 1022,
        'DPLM2-650': 1022,
        'DPLM2-3B': 1022,
        'Random-Transformer': 1022,
        'AMPLIFY-120': 2046,
        'AMPLIFY-350': 2046,
        'E1-150': 2046,
        'E1-300': 2046,
        'E1-600': 2046,
    }
    
    # Models that don't append EOS token
    GLM2_MODELS = ["GLM2-150", "GLM2-650"]
    
    def __init__(
        self,
        model_name: str,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        # Skip aa_to_id for E1, incompatible with E1Scorer
        if not model_name.lower().startswith("e1"):
            self.aa_to_id = SequenceProcessor.aa_to_token_ids(tokenizer)
        else:
            self.aa_to_id = None
        self.unk_id = getattr(tokenizer, "unk_token_id", None)
        self.context_length = self.MODEL_CONTEXT_LENGTH.get(model_name, 1024)
        
    def score_substitutions(
        self,
        df: pd.DataFrame,
        scoring_method: str = "masked_marginal",
        scoring_window: str = "optimal",
    ) -> pd.DataFrame:
        """Score substitution variants.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with variants to score
        scoring_method : str
            One of: "masked_marginal", "mutant_marginal", "wildtype_marginal", "pll", "global_log_prob"
        scoring_window : str
            "optimal" or "sliding"
        Returns
        -------
        pd.DataFrame
            Output dataframe with 'delta_log_prob' column added
        """
        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame is empty")
        
        # Use E1Scorer for E1 models
        if self.model_name.lower().startswith("e1") and scoring_method == "masked_marginal":
            return self._score_with_e1(df, scoring_method)
        
        # Get sliced sequences
        target_seq = df['target_seq'].iloc[0]
        sliced_df = SequenceProcessor.get_sequence_slices(
            df,
            target_seq=target_seq,
            model_context_len=self.context_length,
            start_idx=1,
            scoring_window=scoring_window,
            indel_mode=False
        )
        
        # Group sliced_df by mutant
        grouped = sliced_df.groupby('mutant')
        mutant_groups = {mutant: group for mutant, group in grouped}
        
        if scoring_method in ["masked_marginal", "mutant_marginal", "wildtype_marginal"]:
            scores = self._score_marginal(df, target_seq, mutant_groups, scoring_method)
        elif scoring_method == "pll":
            scores = self._score_pll_substitutions(sliced_df, target_seq)
        else:  # global_log_prob
            scores = self._score_global_log_prob(sliced_df, target_seq)
        
        out = df.copy()
        if scoring_method in ["masked_marginal", "mutant_marginal", "wildtype_marginal"]:
            out['delta_log_prob'] = scores
        else:
            out['delta_log_prob'] = out['mutant'].map(scores)
        return out
    
    def score_indels(
        self,
        df: pd.DataFrame,
        scoring_window: str = "sliding",
    ) -> pd.DataFrame:
        """Score indel variants using PLL.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with indel variants to score
        scoring_window : str
            Sliding window mode should be used for indels scoring
        Returns
        -------
        pd.DataFrame
            Output dataframe with 'delta_log_prob' column added
        """
        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame is empty")
        
        target_seq = df['target_seq'].iloc[0]
        model_context_len = self.context_length
        if model_context_len is None:
            model_context_len = len(target_seq)
        
        sliced_df = SequenceProcessor.get_sequence_slices(
            df,
            target_seq=target_seq,
            model_context_len=model_context_len,
            start_idx=1,
            scoring_window='sliding',
            indel_mode=True
        )
        
        seqs_to_score = sliced_df['sliced_mutated_seq'].to_list()
        
        print(f"Computing PLL for {len(seqs_to_score)} unique sequences (indels)...")
        
        iterator = tqdm(
            total=len(seqs_to_score),
            desc="PLL computation (indels)",
            unit="seq",
            position=1,
            leave=False,
        )
        
        pll_results = self._calculate_pll_batched(seqs_to_score, iterator)
        iterator.close()
        
        # Grab normalized PLL for indels
        pll_cache = {seq: result[0] for seq, result in zip(seqs_to_score, pll_results)}        
        # Add a mapped column of per-window scores, then average by mutated_seq
        sliced_df['window_score'] = sliced_df['sliced_mutated_seq'].map(pll_cache)
        scores_by_variant = (
            sliced_df.groupby('mutated_seq')['window_score']
            .mean()
            .to_dict()
        )
        
        out = df.copy()
        out['delta_log_prob'] = out['mutated_seq'].map(scores_by_variant)
        return out
    
    def _score_marginal(
        self,
        df: pd.DataFrame,
        target_seq: str,
        mutant_groups: Dict,
        scoring_method: str,
    ) -> List[float]:
        """Score using marginal methods (masked/wildtype/mutant)."""
        
        if scoring_method == "masked_marginal":
            return self._score_masked_marginal(df, target_seq, mutant_groups)
        elif scoring_method == "wildtype_marginal":
            return self._score_wildtype_marginal(df, target_seq, mutant_groups)
        else:  # mutant_marginal
            return self._score_mutant_marginal(df, target_seq, mutant_groups)
    
    def _score_masked_marginal(
        self,
        df: pd.DataFrame,
        target_seq: str,
        mutant_groups: Dict,
    ) -> List[float]:
        """Score using masked marginal method."""
        # Group by (window_start, window_end, pos_tuple) -> List[(row_idx, sorted_muts)]
        position_groups: Dict[Tuple[int, int, Tuple[int, ...]], List[Tuple[int, Tuple[Tuple[str, int, str], ...]]]] = {}
        
        for row_idx, row in enumerate(df.itertuples(index=False)):
            mutant = row.mutant
            muts = SequenceProcessor.parse_mutant_string(mutant)
            
            mutant_slices = mutant_groups.get(mutant)
            wt_slice = mutant_slices[mutant_slices['mutated_seq'] == target_seq]
            if len(wt_slice) == 0:
                raise ValueError(f"No available slice for mutant {mutant} and method masked_marginal")
            slice_row = wt_slice.iloc[0]
            
            window_start = int(slice_row['window_start'])
            window_end = int(slice_row['window_end'])
            
            # Sanity check
            min_pos = min(p for _, p, _ in muts)
            max_pos = max(p for _, p, _ in muts)
            if not (window_start <= min_pos and max_pos < window_end):
                raise ValueError(f"Window {window_start}-{window_end} does not contain all positions for variant {mutant}")
            
            sorted_muts = tuple(sorted(muts, key=lambda x: x[1]))
            pos_tuple = tuple(pos - window_start for _, pos, _ in sorted_muts)
            
            key = (window_start, window_end, pos_tuple)
            position_groups.setdefault(key, []).append((row_idx, sorted_muts))
        
        sequences: List[str] = []
        positions_list: List[List[int]] = []
        variant_info: List[List[Tuple[int, List[Tuple[str, int, str]]]]] = []
        
        for (window_start, window_end, pos_tuple), variants in position_groups.items():
            window_seq = target_seq[window_start:window_end]
            sequences.append(window_seq)
            positions_list.append(list(pos_tuple))
            variant_info.append([(row_idx, list(sorted_muts)) for row_idx, sorted_muts in variants])
        
        total_variants = len(df)
        print(f"Computing scores for {len(sequences)} inputs, covering {total_variants} variants ...")
        
        iterator = tqdm(
            range(0, len(sequences), self.batch_size),
            total=(len(sequences) + self.batch_size - 1) // self.batch_size,
            desc="Assay batches (masked_marginal)",
            unit="batch",
            position=1,
            leave=False,
        )
        
        per_variant_log_probs = self._position_log_probs(
            "masked_marginal", sequences, positions_list, iterator
        )
        
        scores = [0.0] * len(df)
        for variants_in_group, score in zip(variant_info, per_variant_log_probs):
            for row_idx, muts in variants_in_group:
                assert score.size(0) == len(muts), "Mismatch between mutations and gathered logits"
                wt_ids, mt_ids = [], []
                for wt, _pos, mt in muts:
                    wt_id = self.aa_to_id.get(wt)
                    mt_id = self.aa_to_id.get(mt)
                    if wt_id is None or mt_id is None or (self.unk_id is not None and (wt_id == self.unk_id or mt_id == self.unk_id)):
                        raise ValueError(f"WT or MT is not in vocab: {wt} or {mt}")
                    wt_ids.append(wt_id)
                    mt_ids.append(mt_id)
                
                wt_tensor = torch.as_tensor(wt_ids, device=score.device)
                mt_tensor = torch.as_tensor(mt_ids, device=score.device)
                indices = torch.arange(len(wt_ids), device=score.device)
                deltas = score[indices, mt_tensor] - score[indices, wt_tensor]
                scores[row_idx] = deltas.sum().item()
        
        return scores
    
    def _score_wildtype_marginal(
        self,
        df: pd.DataFrame,
        target_seq: str,
        mutant_groups: Dict,
    ) -> List[float]:
        """Score using wildtype marginal method."""
        # Group by (window_start, window_end) -> List[(row_idx, sorted_muts, pos_rels)]
        window_groups: Dict[Tuple[int, int], List[Tuple[int, Tuple[Tuple[str, int, str], ...], Tuple[int, ...]]]] = {}
        
        for row_idx, row in enumerate(df.itertuples(index=False)):
            mutant = row.mutant
            muts = SequenceProcessor.parse_mutant_string(mutant)
            
            mutant_slices = mutant_groups.get(mutant)
            wt_slice = mutant_slices[mutant_slices['mutated_seq'] == target_seq]
            if len(wt_slice) == 0:
                raise ValueError(f"No available slice for mutant {mutant} and method wildtype_marginal")
            slice_row = wt_slice.iloc[0]
            
            window_start = int(slice_row['window_start'])
            window_end = int(slice_row['window_end'])
            
            min_pos = min(p for _, p, _ in muts)
            max_pos = max(p for _, p, _ in muts)
            if not (window_start <= min_pos and max_pos < window_end):
                raise ValueError(f"Window {window_start}-{window_end} does not contain all positions for variant {mutant}")
            
            sorted_muts = tuple(sorted(muts, key=lambda x: x[1]))
            pos_rels = tuple(pos - window_start for _, pos, _ in sorted_muts)
            
            key = (window_start, window_end)
            window_groups.setdefault(key, []).append((row_idx, sorted_muts, pos_rels))
        
        sequences: List[str] = []
        positions_list: List[List[int]] = []
        window_to_variants: List[List[Tuple[int, List[Tuple[str, int, str]], List[int]]]] = []
        
        for (window_start, window_end), variants in window_groups.items():
            window_seq = target_seq[window_start:window_end]
            sequences.append(window_seq)
            
            all_positions = set()
            for _, _, pos_rels in variants:
                all_positions.update(pos_rels)
            positions_list.append(sorted(all_positions))
            window_to_variants.append([(row_idx, list(sorted_muts), list(pos_rels)) for row_idx, sorted_muts, pos_rels in variants])
        
        total_variants = len(df)
        print(f"Computing scores for {len(sequences)} windows, covering {total_variants} variants ...")
        
        iterator = tqdm(
            range(0, len(sequences), self.batch_size),
            total=(len(sequences) + self.batch_size - 1) // self.batch_size,
            desc="Assay batches (wildtype_marginal)",
            unit="batch",
            position=1,
            leave=False,
        )
        
        per_variant_log_probs = self._position_log_probs(
            "wildtype_marginal", sequences, positions_list, iterator
        )
        
        scores = [0.0] * len(df)
        for window_idx, (window_log_probs, variants) in enumerate(zip(per_variant_log_probs, window_to_variants)):
            window_positions = positions_list[window_idx]
            pos_to_idx = {pos: idx for idx, pos in enumerate(window_positions)}
            
            for row_idx, muts, pos_rels in variants:
                pos_indices = torch.tensor([pos_to_idx[pos] for pos in pos_rels], device=window_log_probs.device)
                variant_log_probs = window_log_probs[pos_indices]
                
                assert variant_log_probs.size(0) == len(muts), "Mismatch between mutations and gathered logits"
                wt_ids, mt_ids = [], []
                for wt, _pos, mt in muts:
                    wt_id = self.aa_to_id.get(wt)
                    mt_id = self.aa_to_id.get(mt)
                    if wt_id is None or mt_id is None or (self.unk_id is not None and (wt_id == self.unk_id or mt_id == self.unk_id)):
                        raise ValueError(f"WT or MT is not in vocab: {wt} or {mt}")
                    wt_ids.append(wt_id)
                    mt_ids.append(mt_id)
                
                wt_tensor = torch.as_tensor(wt_ids, device=variant_log_probs.device)
                mt_tensor = torch.as_tensor(mt_ids, device=variant_log_probs.device)
                indices = torch.arange(len(wt_ids), device=variant_log_probs.device)
                deltas = variant_log_probs[indices, mt_tensor] - variant_log_probs[indices, wt_tensor]
                scores[row_idx] = deltas.sum().item()
        
        return scores
    
    def _score_mutant_marginal(
        self,
        df: pd.DataFrame,
        target_seq: str,
        mutant_groups: Dict,
    ) -> List[float]:
        """Score using mutant marginal method."""
        sequences: List[str] = []
        positions_list: List[List[int]] = []
        variant_info: List[Tuple[int, List[Tuple[str, int, str]]]] = []
        
        for row_idx, row in enumerate(df.itertuples(index=False)):
            mutant = row.mutant
            mutated_seq = row.mutated_seq
            muts = SequenceProcessor.parse_mutant_string(mutant)
            
            for wt, pos, mt in muts:
                assert 0 <= pos < len(target_seq), f"Mutation pos {pos} out of range for target_seq length {len(target_seq)}"
            
            mutant_slices = mutant_groups.get(mutant)
            mut_slice = mutant_slices[mutant_slices['mutated_seq'] == mutated_seq]
            if len(mut_slice) == 0:
                raise ValueError(f"No available slice for mutant {mutant} and method mutant_marginal")
            slice_row = mut_slice.iloc[0]
            
            window_start = int(slice_row['window_start'])
            window_end = int(slice_row['window_end'])
            window_seq = slice_row['sliced_mutated_seq']
            
            min_pos = min(p for _, p, _ in muts)
            max_pos = max(p for _, p, _ in muts)
            if not (window_start <= min_pos and max_pos < window_end):
                raise ValueError(f"Window {window_start}-{window_end} does not contain all positions for variant {mutant}")
            
            pos_rels: List[int] = []
            for wt, pos, mt in muts:
                rel = pos - window_start
                assert window_seq[rel] == mutated_seq[pos], f"mutant_marginal: residue mismatch at abs {pos} (rel {rel})"
                pos_rels.append(rel)
            
            sequences.append(window_seq)
            positions_list.append(pos_rels)
            variant_info.append((row_idx, muts))
        
        print(f"Computing scores for {len(sequences)} variants ...")
        
        iterator = tqdm(
            range(0, len(sequences), self.batch_size),
            total=(len(sequences) + self.batch_size - 1) // self.batch_size,
            desc="Assay batches (mutant_marginal)",
            unit="batch",
            position=1,
            leave=False,
        )
        
        per_variant_log_probs = self._position_log_probs(
            "mutant_marginal", sequences, positions_list, iterator
        )
        
        scores = [0.0] * len(df)
        for (row_idx, muts), score in zip(variant_info, per_variant_log_probs):
            assert score.size(0) == len(muts), "Mismatch between mutations and gathered logits"
            muts = sorted(muts, key=lambda x: x[1])
            wt_ids, mt_ids = [], []
            for wt, _pos, mt in muts:
                wt_id = self.aa_to_id.get(wt)
                mt_id = self.aa_to_id.get(mt)
                if wt_id is None or mt_id is None or (self.unk_id is not None and (wt_id == self.unk_id or mt_id == self.unk_id)):
                    raise ValueError(f"WT or MT is not in vocab: {wt} or {mt}")
                wt_ids.append(wt_id)
                mt_ids.append(mt_id)
            
            wt_tensor = torch.as_tensor(wt_ids, device=score.device)
            mt_tensor = torch.as_tensor(mt_ids, device=score.device)
            indices = torch.arange(len(wt_ids), device=score.device)
            deltas = score[indices, mt_tensor] - score[indices, wt_tensor]
            scores[row_idx] = deltas.sum().item()
        
        return scores
    
    def _score_pll_substitutions(
        self,
        sliced_df: pd.DataFrame,
        target_seq: str,
    ) -> Dict[str, float]:
        """Score substitutions using pseudo-log-likelihood."""
        mutated_slices = sliced_df[sliced_df['mutated_seq'] != target_seq].copy()
        
        seqs_to_score = mutated_slices['sliced_mutated_seq'].drop_duplicates().tolist()
        
        print(f"Computing PLL for {len(seqs_to_score)} unique sequences...")
        
        pll_progress = tqdm(
            total=len(seqs_to_score),
            desc="PLL computation",
            unit="seq",
            position=1,
            leave=False,
        )
        
        pll_results = self._calculate_pll_batched(seqs_to_score, pll_progress)
        pll_progress.close()
        
        seq_to_pll = {seq: res[0] for seq, res in zip(seqs_to_score, pll_results)}
        mutated_slices['sequence_pll'] = mutated_slices['sliced_mutated_seq'].map(seq_to_pll)
        
        scores_by_variant = (
            mutated_slices.groupby('mutant')['sequence_pll']
            .first()
            .to_dict()
        )
        
        return scores_by_variant
    
    def _score_global_log_prob(
        self,
        sliced_df: pd.DataFrame,
        target_seq: str,
    ) -> Dict[str, float]:
        """Score using global log probability."""
        mutated_slices = sliced_df[sliced_df['mutated_seq'] != target_seq].copy()
        seqs_to_score = mutated_slices['sliced_mutated_seq'].tolist()
        
        print(f"Computing global log prob for {len(seqs_to_score)} unique sequences...")
        
        iterator = tqdm(
            range(0, len(seqs_to_score), self.batch_size),
            total=(len(seqs_to_score) + self.batch_size - 1) // self.batch_size,
            desc="Global log prob batches",
            unit="batch",
            position=1,
            leave=False,
        )
        
        log_prob_results = self._get_sequence_log_probability_batched(seqs_to_score, iterator)
        
        mutated_slices['sequence_log_prob'] = log_prob_results
        
        scores_by_variant = (
            mutated_slices.groupby('mutant')['sequence_log_prob']
            .first()
            .to_dict()
        )
        
        return scores_by_variant
    
    def _score_with_e1(self, df: pd.DataFrame, scoring_method: str) -> pd.DataFrame:
        """Score variants using E1Scorer."""
        from .e1_scorer import E1Scorer, EncoderScoreMethod
        
        scorer = E1Scorer(model=self.model, method=EncoderScoreMethod.MASKED_MARGINAL)
        # E1 has a context length of 8192, so we don't need to slice the sequences for scoring with these models
        target_seq = df['target_seq'].iloc[0]
        sequences = df['mutated_seq'].tolist()
        sequence_ids = df['mutant'].tolist()
        
        print(f"Scoring {len(sequences)} variants with E1 ({scoring_method})...")
        
        results = scorer.score(
            parent_sequence=target_seq,
            sequences=sequences,
            sequence_ids=sequence_ids,
            context_seqs=None,
            context_reduction="none",
        )
        
        scores_dict = {r["id"]: r["score"] for r in results}
        scores = [scores_dict[mutant] for mutant in sequence_ids]
        
        out = df.copy()
        out['delta_log_prob'] = scores
        return out
    
    @torch.inference_mode()
    def _position_log_probs(
        self,
        scoring_method: str,
        sequences: List[str],
        positions_list: List[List[int]],
        progress_bar,
    ) -> List[torch.Tensor]:
        """Return batched log probabilities for multiple positions per sequence."""
        assert len(sequences) == len(positions_list), "Must have one position list per sequence"
        
        all_log_probs = []
        
        for batch_start in progress_bar:
            batch_end = min(batch_start + self.batch_size, len(sequences))
            batch_sequences = sequences[batch_start:batch_end]
            batch_positions_list = positions_list[batch_start:batch_end]
            
            tokens = self.tokenizer(
                batch_sequences,
                return_tensors='pt',
                add_special_tokens=True,
                padding=False,
            )
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            seq_lengths = attention_mask.sum(dim=1)
            
            # GLM2 does not append EOS
            if self.model_name in self.GLM2_MODELS:
                expected_lengths = torch.tensor([len(seq) + 1 for seq in batch_sequences], device=seq_lengths.device)
                if not torch.equal(seq_lengths, expected_lengths):
                    raise AssertionError("Tokenized length must equal len(sequence)+1 for GLM2 models in the batch")
            else:
                expected_lengths = torch.tensor([len(seq) + 2 for seq in batch_sequences], device=seq_lengths.device)
                if not torch.equal(seq_lengths, expected_lengths):
                    raise AssertionError("Tokenized length must equal len(sequence)+2 for all sequences in the batch")
            
            if scoring_method == "masked_marginal":
                mask_id = self.tokenizer.mask_token_id
                if mask_id is None:
                    mask_id = self.tokenizer.convert_tokens_to_ids(getattr(self.tokenizer, 'mask_token', '<mask>'))
                if mask_id is None:
                    raise ValueError("Tokenizer has no mask token.")
                
                masked_input_ids = input_ids.clone()
                for batch_idx, positions in enumerate(batch_positions_list):
                    token_indices = [pos + 1 for pos in positions]
                    masked_input_ids[batch_idx, token_indices] = mask_id
                
                outputs = self.model(masked_input_ids, attention_mask=attention_mask)
            else:
                outputs = self.model(input_ids, attention_mask=attention_mask)
            
            logits = outputs.logits
            
            for batch_idx, positions in enumerate(batch_positions_list):
                token_indices = torch.tensor([pos + 1 for pos in positions], device=self.device, dtype=torch.long)
                selected_logits = logits[batch_idx, token_indices]
                log_probs = torch.log_softmax(selected_logits, dim=-1)
                all_log_probs.append(log_probs)
        
        return all_log_probs
    
    @torch.inference_mode()
    def _calculate_pll_batched(
        self,
        sequences: List[str],
        progress_bar,
    ) -> List[Tuple[float, float]]:
        """Calculate pseudo-log-likelihood for multiple sequences with batched processing."""
        mask_id = self.tokenizer.mask_token_id
        if mask_id is None:
            mask_id = self.tokenizer.convert_tokens_to_ids(getattr(self.tokenizer, 'mask_token', '<mask>'))
        if mask_id is None:
            raise ValueError("Tokenizer must provide a valid mask token id")
        
        # Group sequences by length for efficient batching
        length_groups = defaultdict(list)
        for idx, seq in enumerate(sequences):
            length_groups[len(seq)].append((idx, seq))
        
        results = [None] * len(sequences)
        
        for seq_len, indexed_seqs in length_groups.items():
            indices = [idx for idx, _ in indexed_seqs]
            seqs = [seq for _, seq in indexed_seqs]
            
            tokens = self.tokenizer(seqs, return_tensors="pt", add_special_tokens=True, padding=False)
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            
            num_seqs = input_ids.size(0)
            
            if self.model_name in self.GLM2_MODELS:
                expected_len = seq_len + 1
                assert input_ids.shape[1] == expected_len, (
                    f"Tokenized length {input_ids.shape[1]} must equal len(sequence)+1 ({expected_len}) for GLM2"
                )
            else:
                expected_len = seq_len + 2
                assert input_ids.shape[1] == expected_len, (
                    f"Tokenized length {input_ids.shape[1]} must equal len(sequence)+2 ({expected_len})"
                )
            
            seq_start = 1
            if self.model_name in self.GLM2_MODELS:
                seq_end = input_ids.size(1)
            else:
                seq_end = input_ids.size(1) - 1
            positions = list(range(seq_start, seq_end))
            L = len(positions)
            
            total_lls = torch.zeros(num_seqs, device=self.device)
            
            for batch_start_idx in range(0, len(positions), self.batch_size):
                batch_end_idx = min(batch_start_idx + self.batch_size, len(positions))
                batch_positions = positions[batch_start_idx:batch_end_idx]
                num_positions = len(batch_positions)
                
                masked_batch = input_ids.unsqueeze(1).expand(-1, num_positions, -1).reshape(num_seqs * num_positions, -1).clone()
                attention_mask_batch = attention_mask.unsqueeze(1).expand(-1, num_positions, -1).reshape(num_seqs * num_positions, -1)
                
                position_tensor = torch.tensor(batch_positions, device=self.device)
                row_indices = torch.arange(num_seqs * num_positions, device=self.device)
                pos_indices = position_tensor.repeat(num_seqs)
                masked_batch[row_indices, pos_indices] = mask_id
                
                outputs = self.model(masked_batch, attention_mask=attention_mask_batch)
                logits = outputs.logits.float()
                
                log_probs = torch.log_softmax(logits, dim=-1)
                
                true_ids = input_ids[:, batch_positions]
                true_ids_flat = true_ids.reshape(-1)
                
                batch_indices = torch.arange(num_seqs * num_positions, device=self.device)
                selected_log_probs = log_probs[batch_indices, pos_indices, true_ids_flat]
                
                selected_log_probs = selected_log_probs.reshape(num_seqs, num_positions)
                total_lls += selected_log_probs.sum(dim=1)
            
            progress_bar.update(num_seqs)
            
            for i, orig_idx in enumerate(indices):
                total_ll = total_lls[i].item()
                results[orig_idx] = (total_ll, total_ll / L)
        
        return results
    
    @torch.inference_mode()
    def _get_sequence_log_probability_batched(
        self,
        sequences: List[str],
        progress_bar,
    ) -> List[float]:
        """Compute log probability for multiple sequences with batched processing."""
        results = []
        
        for batch_start in progress_bar:
            batch_end = min(batch_start + self.batch_size, len(sequences))
            batch_sequences = sequences[batch_start:batch_end]
            
            tokens = self.tokenizer(
                batch_sequences,
                return_tensors='pt',
                add_special_tokens=False,
                padding=True,
            )
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            
            output = self.model(input_ids, attention_mask=attention_mask)
            logits = output.logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)
            
            selected_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
            masked_log_probs = selected_log_probs * attention_mask
            seq_log_probs = masked_log_probs.sum(dim=1)
            
            results.extend(seq_log_probs.tolist())
        
        return results


class ProteinGymRunner:
    """Orchestrates ProteinGym zero-shot scoring across models and assays.
    
    Parameters
    ----------
    results_dir : str
        Directory to save results
    repo_id : str
        HuggingFace repo ID for ProteinGym data
    device : str, optional
        Device to run on (defaults to CUDA if available)
    """
    
    def __init__(
        self,
        results_dir: str,
        repo_id: str = "GleghornLab/ProteinGym_DMS",
        device: Optional[str] = None,
    ):
        self.results_dir = results_dir
        self.repo_id = repo_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        os.makedirs(results_dir, exist_ok=True)
    
    def run(
        self,
        dms_ids: List[str],
        model_names: List[str],
        mode: str = "benchmark",
        scoring_method: str = "masked_marginal",
        scoring_window: str = "optimal",
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Run zero-shot scoring for all specified models and assays.
        
        Parameters
        ----------
        dms_ids : List[str]
            List of DMS assay IDs to score
        model_names : List[str]
            List of model names to use for scoring
        mode : str
            One of: "benchmark", "indels", "singles", "multiples"
        scoring_method : str
            Scoring method to use
        scoring_window : str
            "optimal" or "sliding"
        batch_size : int
            Batch size for inference
            
        Returns
        -------
        Dict[str, float]
            Mapping of model_name -> elapsed_time
        """
        from base_models.get_base_models import get_base_model
        from .data_loader import load_proteingym_dms
        
        timing = {}
        
        for model_name in model_names:
            start_time = time.time()
            
            model, tokenizer = get_base_model(model_name, masked_lm=True)
            model = model.to(self.device)
            
            scorer = ProteinGymScorer(
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                device=self.device,
                batch_size=batch_size,
            )
            
            assay_iterator = tqdm(dms_ids, desc="All assays", unit="assay", position=0)
            
            for dms_id in assay_iterator:
                df = load_proteingym_dms(dms_id, mode=mode, repo_id=self.repo_id)
                if df is None or len(df) == 0:
                    raise ValueError(f"No data found for DMS ID: {dms_id}")
                
                assay_iterator.set_description_str(f"Assay {dms_id}")
                
                if mode == 'indels':
                    results_df = scorer.score_indels(
                        df,
                        scoring_window='sliding',
                    )
                    suffix = 'pll'
                else:
                    results_df = scorer.score_substitutions(
                        df,
                        scoring_method=scoring_method,
                        scoring_window=scoring_window,
                    )
                    suffix = scoring_method
                
                self._save_results(dms_id, results_df, model_name, suffix, mode)
                tqdm.write(f"[Assay {dms_id}] saved/updated")
            
            timing[model_name] = time.time() - start_time
        
        return timing
    
    def _save_results(
        self,
        dms_id: str,
        results_df: pd.DataFrame,
        model_name: str,
        suffix: str,
        mode: str,
    ):
        """Save/merge results for a DMS assay."""
        per_dms_path = os.path.join(self.results_dir, f"{dms_id}_zs_{suffix}.csv")
        
        results_to_save = results_df.copy()
        if 'delta_log_prob' in results_to_save.columns:
            results_to_save = results_to_save.rename(columns={'delta_log_prob': model_name})
        
        # Only keep one row of 'target_seq' (present in first row), blank the rest
        first_target_seq = None
        if 'target_seq' in results_to_save.columns and len(results_to_save) > 0:
            first_target_seq = str(results_to_save['target_seq'].iloc[0])
            results_to_save['target_seq'] = ''
            results_to_save.iloc[0, results_to_save.columns.get_loc('target_seq')] = first_target_seq
        
        # If an aggregated file exists, merge the new model column
        if os.path.exists(per_dms_path):
            try:
                existing = pd.read_csv(per_dms_path)
                if 'mutant' in existing.columns:
                    merged = existing.merge(
                        results_to_save[['mutant', model_name]],
                        on='mutant',
                        how='outer',
                    )
                elif 'mutated_seq' in existing.columns:  # for indels
                    merged = existing.merge(
                        results_to_save[['mutated_seq', model_name]],
                        on='mutated_seq',
                        how='outer',
                    )
                # Re-apply target_seq after merge
                if 'target_seq' in merged.columns and first_target_seq is not None:
                    merged['target_seq'] = ''
                    merged.iloc[0, merged.columns.get_loc('target_seq')] = first_target_seq
                merged.to_csv(per_dms_path, index=False)
            except Exception as e:
                print(f"Error merging results for {dms_id}: {e}")
                results_to_save.to_csv(per_dms_path, index=False)
        else:
            results_to_save.to_csv(per_dms_path, index=False)
    
    def run_benchmark(
        self,
        model_names: List[str],
        dms_ids: List[str],
        mode: str,
        scoring_method: str,
    ):
        """Run the ProteinGym benchmarking script on scored CSV files.
        
        Parameters
        ----------
        model_names : List[str]
            List of model names to evaluate
        dms_ids : List[str]
            List of DMS assay IDs to evaluate
        mode : str
            Mode: 'benchmark', 'indels', 'singles', 'multiples'
        scoring_method : str
            Scoring method used (e.g., 'masked_marginal', 'pll')
        """
        try:
            pg_dir = os.path.join(os.path.dirname(__file__))
            reference_mapping = os.path.join(pg_dir, 'DMS_substitutions.csv')
            config_path = os.path.join(pg_dir, 'config.json')
            perf_out_dir = os.path.join(self.results_dir, 'benchmark_performance')
            os.makedirs(perf_out_dir, exist_ok=True)

            script_path = os.path.join(pg_dir, 'DMS_benchmark_performance.py')
            script_cmd = [
                sys.executable, script_path,
                '--input_scoring_files_folder', self.results_dir,
                '--output_performance_file_folder', perf_out_dir,
                '--DMS_reference_file_path', reference_mapping,
                '--config_file', config_path,
                '--performance_by_depth',
            ]
            script_cmd += ['--scoring_method', scoring_method]
            if isinstance(model_names, (list, tuple)) and len(model_names) > 0:
                script_cmd += ['--selected_model_names', *model_names]
            if isinstance(dms_ids, (list, tuple)) and len(dms_ids) > 0:
                script_cmd += ['--dms_ids', *[str(x) for x in dms_ids]]
            if isinstance(mode, str) and mode.lower() == 'indels':
                script_cmd.append('--indel_mode')
            subprocess.run(script_cmd, check=True)
            
            print(f"Benchmark performance computed. Outputs in {perf_out_dir}")
        except Exception as e:
            print(f"Failed to compute benchmark performance: {e}")
    
    @staticmethod
    def collect_spearman(results_dir: str, model_names: List[str]) -> Dict[str, float]:
        """Parse ProteinGym benchmark Summary CSV and return {model_name: spearman}.
        
        Looks for Summary_performance_DMS_[substitutions|indels]_Spearman.csv and
        creates a dictionary of {model_name: spearman} for the given model names.
        This is used to pass Spearman scores to the visualization module for plotting.
        """
        perf_out_dir = os.path.join(results_dir, 'benchmark_performance')
        spearman_dir = os.path.join(perf_out_dir, 'Spearman')
        sub_csv = os.path.join(spearman_dir, 'Summary_performance_DMS_substitutions_Spearman.csv')
        ind_csv = os.path.join(spearman_dir, 'Summary_performance_DMS_indels_Spearman.csv')
        csv_path = sub_csv if os.path.exists(sub_csv) else ind_csv if os.path.exists(ind_csv) else None
        
        if csv_path is None:
            print(f"ProteinGym Spearman summary not found in {spearman_dir}")
            return {}
        
        df = pd.read_csv(csv_path)
        if 'Model_name' not in df.columns or 'Average_Spearman' not in df.columns:
            print("ProteinGym summary CSV missing required columns: 'Model_name' and 'Average_Spearman'")
            return {}
        
        model_scores = {}
        for _, row in df.iterrows():
            try:
                name = str(row['Model_name'])
                score = float(row['Average_Spearman'])
            except Exception:
                continue
            model_scores[name] = score
        
        out = {}
        for model_name in (model_names or []):
            if model_name in model_scores:
                out[model_name] = float(model_scores[model_name])
        return out
