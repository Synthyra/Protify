"""Allocate whole components of related sequences to splits so every split keeps the dataset's label distribution.

A split that holds out families cannot give every term the prevalence it has in train, but how close it comes depends on how the
families are dealt out. `StratifiedSettings(label_allocation="components")` deals them by the method here, in four steps:

1. **Components.** Rows joined by anything a split must keep whole (the same 0.3 cluster, a linked source group, a hit at the
   enforced identity that counts against the biggest reference set) form connected components, and a component goes to one split
   whole. What the table cannot count against the held-out sets, a hit near the e-value limit, is left to repair, which returns the
   few clusters it flags to train. A component over `max_unit` clusters is held in train.
2. **Iterative stratification** (Sechidis, Tsoumakas and Vlahavas 2011, ECML PKDD). Terms are taken rarest first, and each
   component that carries the term goes to the split that still wants most of it, within the split's cluster quota. Rare terms
   come first because the common ones are the ones a later choice can still balance. A component carries a count of rows for each
   term, not a 0 or a 1, so the components that carry the most rows of the term go first, while every split still wants plenty of
   it. Placed late, one component can overshoot a rare term's count by an order of magnitude.
3. **Top-up** to the exact cluster quotas, with single-cluster components, when the placement left a split short.
4. **Swaps.** A component of train and a held-out component of the same weight and row count change places when that lowers the
   held-out split's chi-square distance, the sum over terms of (count - expected)^2 / max(expected, 1) for the count a split holds
   against the dataset's prevalence times its size. Its mean per term is about 1 for a random sample of rows, so it reads as a
   multiple of chance.

Second-order stratification (Szymanski and Kajdanowicz 2017, LIDTA) balances pairs of labels as well. A row of an ontology carries
every ancestor of its terms, so most pairs of terms are ancestor and descendant and their balance follows the single terms', which is
why the first-order method is used.

Shapes: K components, L terms, S splits (the first is train), n rows.
"""

from __future__ import annotations

import sys
import numpy as np
import scipy.sparse as sp

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from scipy.sparse.csgraph import connected_components
from tqdm.auto import tqdm


SWAP_PROPOSALS = 3_000_000
PROGRESS_EVERY = 500_000
# The search has converged when this many proposals per held-out component, and at least `MIN_PATIENCE`, pass without an accepted swap.
PATIENCE_PER_COMPONENT = 50
MIN_PATIENCE = 10_000
# Singles a top-up weighs at once, in component order: it is a fill of what the stratification left, not a search.
TOP_UP_CANDIDATES = 2000


def connected_groups(count: int, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The connected component of each of `count` nodes, as an integer label, over the edges `left[i]` to `right[i]`."""
    # left, right: (m,) node indices of each edge
    edges = sp.coo_matrix((np.ones(len(left), dtype=np.int32), (left, right)), shape=(count, count))  # (count, count)
    return connected_components(edges, directed=False)[1]  # (count,)


def chi_square_per_term(counts: np.ndarray, expected: np.ndarray) -> float:
    """The mean over terms of (count - expected)^2 / max(expected, 1), about 1 for a random sample."""
    # counts: (L,); expected: (L,)
    return float(np.mean((counts - expected) ** 2 / np.maximum(expected, 1.0)))


@dataclass(slots=True)
class Allocation:
    """The split of each component and the running per-term counts of every split."""

    weights: np.ndarray  # (K,) 0.9 clusters of each component, which the quotas count
    rows: np.ndarray  # (K,) rows of each component
    counts: sp.csr_matrix  # (K, L) rows of each component that carry each term
    quota: np.ndarray  # (S,) clusters each split takes
    split: np.ndarray  # (K,) the split of each component, -1 before it is placed
    capacity: np.ndarray  # (S,) clusters each split can still take
    totals: np.ndarray  # (L,) rows that carry each term
    held: np.ndarray  # (S, L) rows of each split that carry each term
    size: np.ndarray  # (S,) rows of each split

    @classmethod
    def start(cls, weights: np.ndarray, rows: np.ndarray, counts: sp.csr_matrix, quotas: Sequence[int]) -> Allocation:
        # weights, rows: (K,); counts: (K, L)
        quota = np.array(quotas, dtype=np.float64)  # (S,)
        return cls(
            weights, rows, counts, quota, np.full(len(weights), -1, dtype=np.int64), quota.copy(),
            np.asarray(counts.sum(axis=0)).ravel(), np.zeros((len(quota), counts.shape[1])), np.zeros(len(quota)),
        )

    def place(self, component: int, split: int) -> None:
        """Put one unplaced component in a split."""
        low, high = self.counts.indptr[component], self.counts.indptr[component + 1]
        self.split[component] = split
        self.capacity[split] -= self.weights[component]
        self.held[split, self.counts.indices[low:high]] += self.counts.data[low:high]
        self.size[split] += self.rows[component]

    def move(self, component: int, split: int) -> None:
        """Take a placed component out of its split and put it in another."""
        low, high = self.counts.indptr[component], self.counts.indptr[component + 1]
        old = self.split[component]
        self.capacity[old] += self.weights[component]
        self.held[old, self.counts.indices[low:high]] -= self.counts.data[low:high]
        self.size[old] -= self.rows[component]
        self.place(component, split)

    def expected(self) -> np.ndarray:
        """The rows each split should hold of each term: the dataset's prevalence times the split's size."""
        return (self.totals / self.size.sum())[None, :] * self.size[:, None]  # (S, L)


def iterative_stratification(allocation: Allocation, forced_train: np.ndarray, seed: int) -> None:
    """Place every component, rarest term first, each in the split that still wants most of that term.

    `desired[s, l]` starts at the term's total times the split's share of the clusters and falls as components that carry the
    term are placed in it. Ties go to the split with the most of its quota left.
    """
    # forced_train: (K,)
    rng = np.random.default_rng(seed)
    desired = allocation.totals[None, :] * (allocation.quota / allocation.quota.sum())[:, None]  # (S, L)
    remaining = allocation.totals.copy()  # (L,) rows that carry each term, in components not yet placed
    by_term = allocation.counts.tocsc()  # (K, L)
    by_component = allocation.counts

    def place(component: int, split: int) -> None:
        low, high = by_component.indptr[component], by_component.indptr[component + 1]
        desired[split, by_component.indices[low:high]] -= by_component.data[low:high]
        remaining[by_component.indices[low:high]] -= by_component.data[low:high]
        allocation.place(component, split)

    for component in np.flatnonzero(forced_train):
        place(int(component), 0)
    with tqdm(total=int((allocation.split < 0).sum()), desc="iterative stratification", unit="component", mininterval=30) as bar:
        while True:
            live = np.flatnonzero(remaining > 0.5)  # (l,) terms with an unplaced carrier
            if len(live) == 0:
                break
            term = int(live[np.argmin(remaining[live] + rng.random(len(live)) * 0.5)])
            holders = by_term.indices[by_term.indptr[term] : by_term.indptr[term + 1]]  # (h,)
            carried = by_term.data[by_term.indptr[term] : by_term.indptr[term + 1]]  # (h,) rows of each holder that carry the term
            unplaced = allocation.split[holders] < 0  # (h,)
            holders, carried = holders[unplaced], carried[unplaced]  # (h',), (h',)
            shuffled = rng.permutation(len(holders))  # (h',)
            holders = holders[shuffled][np.argsort(-carried[shuffled], kind="stable")]  # (h',) most rows first, ties in random order
            for component in holders:
                fits = np.flatnonzero(allocation.capacity >= allocation.weights[component])  # (s,)
                options = fits if len(fits) else np.arange(len(allocation.quota))  # (s,)
                wanted = desired[options, term]  # (s,)
                best = options[wanted == wanted.max()]  # (s',)
                if len(best) > 1:
                    room = allocation.capacity[best] / allocation.quota[best]  # (s',)
                    best = best[room == room.max()]
                place(int(component), int(best[rng.integers(len(best))]))
            bar.update(len(holders))
    # A component that carries no term has nothing to stratify: it fills the room the splits have left, the emptiest first.
    for component in np.flatnonzero(allocation.split < 0):
        fits = np.flatnonzero(allocation.capacity >= allocation.weights[component])  # (s,)
        options = fits if len(fits) else np.arange(len(allocation.quota))  # (s,)
        allocation.place(int(component), int(options[np.argmax(allocation.capacity[options] / allocation.quota[options])]))


def top_up(allocation: Allocation) -> None:
    """Fill each held-out split to its cluster quota with single-cluster components from train, the best for its terms first."""
    target = allocation.totals[None, :] * (allocation.quota / allocation.quota.sum())[:, None]  # (S, L)
    singles = np.flatnonzero((allocation.weights == 1) & (allocation.split == 0))  # (u,)
    for split in range(1, len(allocation.quota)):
        while allocation.capacity[split] > 0:
            if len(singles) == 0:
                raise ValueError("no single-cluster component is left in train to fill a split")
            pick = singles[:TOP_UP_CANDIDATES]  # (v,)
            gain = np.empty(len(pick))  # (v,)
            for place, component in enumerate(pick):
                low, high = allocation.counts.indptr[component], allocation.counts.indptr[component + 1]
                columns, values = allocation.counts.indices[low:high], allocation.counts.data[low:high]
                before = (allocation.held[split, columns] - target[split, columns]) ** 2
                after = (allocation.held[split, columns] + values - target[split, columns]) ** 2
                gain[place] = np.sum((after - before) / np.maximum(target[split, columns], 1.0))
            chosen = int(pick[int(np.argmin(gain))])
            allocation.move(chosen, split)
            singles = singles[singles != chosen]


def swap_search(allocation: Allocation, proposals: int, seed: int) -> dict[str, float]:
    """Exchange equal-class components between train and a held-out split while that lowers the split's chi-square distance.

    Components of equal weight and row count are one class, so a swap leaves every split's size alone and the expected counts
    fixed. A proposal draws a held-out component at random and a train component of its class at random, and is accepted when the
    split's distance falls. The search is greedy and its result depends on the seed only through the proposals it draws. It stops
    at `proposals`, or sooner when `PATIENCE_PER_COMPONENT` proposals for each held-out component pass without an accepted swap.
    """
    rng = np.random.default_rng(seed)
    expected = allocation.expected()  # (S, L)
    scale = 1.0 / np.maximum(expected, 1.0)  # (S, L)
    classes = np.unique(np.stack([allocation.weights, allocation.rows], axis=1), axis=0, return_inverse=True)[1].ravel()  # (K,)
    train_pools: dict[int, list[int]] = {}
    for component in np.flatnonzero(allocation.split == 0):
        train_pools.setdefault(int(classes[component]), []).append(int(component))
    where = {component: place for members in train_pools.values() for place, component in enumerate(members)}
    held = np.flatnonzero(allocation.split > 0)  # (H,) components in an evaluation split
    held_splits = range(1, len(allocation.quota))
    before = [chi_square_per_term(allocation.held[split], expected[split]) for split in held_splits]
    indptr, indices, data = allocation.counts.indptr, allocation.counts.indices, allocation.counts.data
    accepted = 0
    patience = max(MIN_PATIENCE, PATIENCE_PER_COMPONENT * len(held))
    last_accepted = -1
    made = 0
    for step in tqdm(range(proposals), desc="swap search", unit="proposal", mininterval=30, smoothing=0.01):
        if step - last_accepted > patience:
            break
        made = step + 1
        slot = int(rng.integers(len(held)))
        outgoing = int(held[slot])
        pool = train_pools.get(int(classes[outgoing]), [])
        if not pool:
            continue
        split = int(allocation.split[outgoing])
        incoming = pool[int(rng.integers(len(pool)))]
        low_out, high_out, low_in, high_in = indptr[outgoing], indptr[outgoing + 1], indptr[incoming], indptr[incoming + 1]
        columns = np.concatenate([indices[low_out:high_out], indices[low_in:high_in]])  # (a + b,)
        change = np.concatenate([-data[low_out:high_out], data[low_in:high_in]])  # (a + b,)
        unique, inverse = np.unique(columns, return_inverse=True)  # (u,), (a + b,)
        net = np.bincount(inverse, weights=change, minlength=len(unique))  # (u,)
        gap = allocation.held[split, unique] - expected[split, unique]  # (u,)
        if np.sum(((gap + net) ** 2 - gap**2) * scale[split, unique]) < -1e-12:
            allocation.move(outgoing, 0)
            allocation.move(incoming, split)
            held[slot] = incoming
            pool[where[incoming]] = outgoing
            where[outgoing] = where.pop(incoming)
            accepted += 1
            last_accepted = step
        if (step + 1) % PROGRESS_EVERY == 0:
            print(f"swap search: {step + 1:,} proposals, {accepted:,} accepted", file=sys.stderr, flush=True)
    after = [chi_square_per_term(allocation.held[split], expected[split]) for split in held_splits]
    return {
        "proposals": made, "accepted": accepted,
        "chi_square_before": round(float(np.mean(before)), 4), "chi_square_after": round(float(np.mean(after)), 4),
    }


def allocate_components(
    counts: sp.csr_matrix, weights: np.ndarray, rows: np.ndarray, quotas: Sequence[int], forced_train: np.ndarray, *,
    seed: int, proposals: int = SWAP_PROPOSALS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The split of each component (an index into `quotas`, train first) and a record of how it was reached.

    `counts` (K, L) holds the rows of each component that carry each term, `weights` (K,) the 0.9 clusters each holds, which
    `quotas` count exactly, and `rows` (K,) its rows. A component in `forced_train` (K,) goes to train.
    """
    # counts: (K, L); weights, rows, forced_train: (K,)
    allocation = Allocation.start(weights, rows, counts, quotas)
    iterative_stratification(allocation, forced_train, seed)
    stratified = {"clusters": [int(allocation.weights[allocation.split == place].sum()) for place in range(len(quotas))]}
    top_up(allocation)
    record: dict[str, Any] = {
        "components": int(len(weights)), "largest_component_clusters": int(weights.max()), "forced_to_train": int(forced_train.sum()),
        "after_stratification": stratified, "swaps": swap_search(allocation, proposals, seed),
    }
    return allocation.split, record  # (K,)
