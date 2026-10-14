"""Score labeled pairs by one all-vs-all pass over the proteins the pairs share.

A benchmark or split names each protein many times and each pair once. The rule for every Atlas
training and evaluation is to encode each distinct protein once, score every left protein against
every right protein as a matrix product, and read the labeled pairs out of that matrix. This module
is the read-out. It never holds the whole matrix: the left proteins go through in row blocks sized to
a fixed element budget, and each block gives up only the pairs whose left protein it holds.
"""

from __future__ import annotations

import numpy as np
import torch

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from torch import Tensor


# One block of the matrix is at most this many elements, per score (512 MiB in float32).
BLOCK_ELEMENTS = 1 << 27

BlockScores = Callable[[int, int], Sequence[Tensor]]


def block_row_count(n_left: int, n_right: int, block_elements: int = BLOCK_ELEMENTS) -> int:
    """Left proteins per block: as many as fit in `block_elements` against every right protein, at least one."""
    return max(1, min(n_left, block_elements // max(1, n_right)))


def gather_pair_scores(block_scores: BlockScores, left: Sequence[int], right: Sequence[int], *, n_left: int, n_right: int,
                       block_elements: int = BLOCK_ELEMENTS) -> tuple[np.ndarray, ...]:
    """One float32 array per score, each `(n_pairs,)` in pair order, read from the all-vs-all matrices.

    `block_scores(start, stop)` returns one `(stop - start, n_right)` tensor per score, the rows of the
    complete matrix for left proteins `start` to `stop`, so a caller computes its scores however it
    likes (a similarity, a temperature scale, a sigmoid) and this module only reads pairs out.
    `left[i]` and `right[i]` are the row and column of pair `i`. A position outside its side's
    range, or a block of the wrong shape or count, raises.
    """
    left_index = np.asarray(left, dtype=np.int64)  # (n_pairs,)
    right_index = np.asarray(right, dtype=np.int64)  # (n_pairs,)
    if left_index.ndim != 1 or left_index.shape != right_index.shape:
        raise ValueError("Pair positions must be two equal-length one-dimensional sequences")
    if left_index.size and (left_index.min() < 0 or left_index.max() >= n_left or right_index.min() < 0 or right_index.max() >= n_right):
        raise ValueError("A pair names a protein outside the proteins that were encoded")

    order = np.argsort(left_index, kind="stable")  # (n_pairs,), pairs grouped by left protein
    sorted_left = left_index[order]  # (n_pairs,)
    step = block_row_count(n_left, n_right, block_elements)
    outputs: list[np.ndarray] | None = None  # one (n_pairs,) array per score
    for start in range(0, n_left, step):
        stop = min(start + step, n_left)
        # The pairs whose left protein is in this block are sorted positions first to last.
        first, last = np.searchsorted(sorted_left, [start, stop])
        if first == last and outputs is not None:
            continue

        blocks = block_scores(start, stop)  # one (stop - start, n_right) tensor per score
        if any(tuple(block.shape) != (stop - start, n_right) for block in blocks):
            raise ValueError("A score block must have one row per left protein of the block and one column per right protein")
        if outputs is None:
            outputs = [np.empty(left_index.size, dtype=np.float32) for _ in blocks]
        elif len(blocks) != len(outputs):
            raise ValueError("Every block must return the same number of scores")
        if first == last:
            continue

        chosen = order[first:last]  # (n_block_pairs,), pair positions
        rows = torch.from_numpy(sorted_left[first:last] - start).to(blocks[0].device)  # (n_block_pairs,)
        columns = torch.from_numpy(right_index[chosen]).to(blocks[0].device)  # (n_block_pairs,)
        # One output per block by the check above.
        for output, block in zip(outputs, blocks, strict=True):
            output[chosen] = block[rows, columns].float().cpu().numpy()  # (n_block_pairs,)

    return tuple(outputs) if outputs is not None else ()  # one (n_pairs,) array per score


def dual_tower_blocks(query: Tensor, key: Tensor, right_query: Tensor | None = None, right_key: Tensor | None = None) -> BlockScores:
    """Blocks of the forward and the symmetric unit-vector similarity of two towers.

    The forward score of left protein `i` and right protein `j` is `query_i . key_j`, the symmetric score
    is the mean of that and `key_i . query_j`. The right side defaults to the left, as in a benchmark whose
    pairs are drawn from one protein list. Each block returns `(forward, symmetric)`.
    """
    # query, key: (q, t); right_query, right_key: (p, t), where q and p count the left and right proteins
    right_query = query if right_query is None else right_query  # (p, t)
    right_key = key if right_key is None else right_key  # (p, t)

    def blocks(start: int, stop: int) -> tuple[Tensor, Tensor]:
        # Unit vectors multiply in full float32, inside an autocast region or under a process-wide TF32 setting alike.
        with torch.autocast(device_type=query.device.type, enabled=False), exact_float32_matmul():
            forward = query[start:stop].float() @ right_key.float().T  # (stop - start, p)
            reverse = key[start:stop].float() @ right_query.float().T  # (stop - start, p)
        return forward, 0.5 * (forward + reverse)  # (stop - start, p) each

    return blocks


@contextmanager
def exact_float32_matmul() -> Iterator[None]:
    """Float32 matrix products without TF32, restoring the process-wide setting on exit."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
