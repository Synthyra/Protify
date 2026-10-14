"""Batches of sequences bounded by padded tokens, so every forward pass is about the same size.

A sequence of r residues takes r + 2 token columns (CLS and EOS). A batch's cost is its count times its longest
member, so shuffled batches are drawn from length-sorted pools of shuffled sequences, which keeps padding low while
the order still changes every epoch.
"""

from __future__ import annotations

import numpy as np

from collections.abc import Sequence
from dataclasses import dataclass


SPECIAL_TOKENS = 2  # CLS and EOS


@dataclass(frozen=True)
class TokenBatch:
    """The positions of a batch's sequences in the caller's list, and the sequences themselves."""

    indices: tuple[int, ...]
    sequences: tuple[str, ...]

    @property
    def padded_length(self) -> int:
        return max(len(sequence) for sequence in self.sequences) + SPECIAL_TOKENS


def _greedy(order: Sequence[int], sequences: Sequence[str], max_tokens: int, max_sequences: int | None) -> list[list[int]]:
    """Consecutive runs of `order` whose count times longest member stays within `max_tokens`."""
    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for index in order:
        length = len(sequences[index]) + SPECIAL_TOKENS
        grown = max(longest, length)
        full = max_sequences is not None and len(current) >= max_sequences
        if current and (grown * (len(current) + 1) > max_tokens or full):
            batches.append(current)
            current, grown = [], length
        current.append(index)
        longest = grown
    if current:
        batches.append(current)
    return batches


def group_batches(sequences: Sequence[str], group: int, max_tokens: int) -> list[TokenBatch]:
    """Batches of whole groups of `group` consecutive sequences (the last group may be smaller), within `max_tokens`
    padded tokens unless one group alone exceeds them. Groups are packed longest first and never split."""
    groups = [list(range(start, min(start + group, len(sequences)))) for start in range(0, len(sequences), group)]
    longest = [max(len(sequences[index]) for index in members) + SPECIAL_TOKENS for members in groups]
    batches: list[list[int]] = []
    current: list[int] = []
    width = 0
    for position in sorted(range(len(groups)), key=lambda position: -longest[position]):
        members = groups[position]
        grown = max(width, longest[position])
        if current and grown * (len(current) + len(members)) > max_tokens:
            batches.append(current)
            current, grown = [], longest[position]
        current = current + members
        width = grown
    if current:
        batches.append(current)
    return [TokenBatch(tuple(batch), tuple(sequences[index] for index in batch)) for batch in batches]


def token_batches(
    sequences: Sequence[str],
    max_tokens: int,
    shuffle: bool = False,
    seed: int = 0,
    max_sequences: int | None = None,
    pool_batches: int = 64,
) -> list[TokenBatch]:
    """Batches of at most `max_tokens` padded tokens (a single longer sequence makes its own batch).

    Unshuffled batches keep the given order. Shuffled ones sort pools of about `pool_batches` batches by length,
    cut them greedily, and shuffle the batch order.
    """
    if not sequences:
        return []
    if not shuffle:
        groups = _greedy(range(len(sequences)), sequences, max_tokens, max_sequences)
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(sequences))
        mean_length = float(np.mean([len(sequence) for sequence in sequences])) + SPECIAL_TOKENS
        pool_size = max(1, int(pool_batches * max_tokens / mean_length))
        groups = []
        for start in range(0, len(order), pool_size):
            pool = sorted(order[start: start + pool_size].tolist(), key=lambda index: len(sequences[index]))
            groups.extend(_greedy(pool, sequences, max_tokens, max_sequences))
        groups = [groups[position] for position in rng.permutation(len(groups))]
    return [TokenBatch(tuple(group), tuple(sequences[index] for index in group)) for group in groups]
