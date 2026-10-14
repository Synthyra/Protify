"""Metrics of test rows drawn with repetition, from how many times each row was drawn.

A bootstrap replicate, or the evaluation half of an exchangeable repeat, draws each test row
zero or more times. `metrics.py`, `foundry.metrics`, scikit-learn, and SciPy score such a draw
as a list in which every duplicate is a row of its own. These functions return the same numbers
from the draw counts alone, for many draws at once, so a thousand replicates cost a few array
passes rather than a thousand metric calls. Where a reference breaks ties by input order, the
drawn list is read in row order, which puts each row's duplicates next to one another. AURC is
the exception: `metrics.selective_metrics` gives tied rows their mean loss, and so does
`weighted_aurc`.

Shapes: `b` draws, `n` rows, `c` classes or labels, `e` entries (rows times labels), `k` F1
thresholds, `g` draws that share their F1 thresholds, `bins` calibration-error runs, `u` tie
groups and `a` drawn positions, both counted over every draw together. A per-row input of shape
`(n,)` or `(1, n)` is shared by every draw, so anything sorted by it is sorted once.
"""

from __future__ import annotations

import numpy as np
import torch

from dataclasses import dataclass
from scipy.special import digamma


# Protify's multilabel F1 scans thresholds from the smallest probability to 1 in this step.
F1_THRESHOLD_STEP = 0.01


def weighted_mean(counts: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The mean of `values` over each draw's rows, NaN for a draw with no rows.

    A row that was not drawn contributes nothing, even where its value is infinite.
    """
    # counts: (b, n); values: (n,), (1, n), or (b, n)
    weights = counts.astype(np.float64)  # (b, n)
    total = weights.sum(axis=1)  # (b,)
    drawn_values = np.where(counts > 0, values, 0.0)  # (b, n)
    weighted = np.sum(weights * drawn_values, axis=1)  # (b,)
    return np.divide(weighted, total, out=np.full(total.shape, np.nan), where=total > 0)  # (b,)


def _sorted_draws(counts: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each draw's rows in ascending order of `values`, ties in row order.

    Returns the order, the sorted values, and the sorted counts. A shared `values` is sorted once.
    """
    # counts: (b, n); values: (n,), (1, n), or (b, n)
    values = np.atleast_2d(values)  # (1, n) or (b, n)
    order = np.argsort(values, axis=1, kind="stable")  # (1, n) or (b, n)
    sorted_values = np.take_along_axis(values, order, axis=1)  # (1, n) or (b, n)
    order = np.broadcast_to(order, counts.shape)  # (b, n)
    sorted_counts = np.take_along_axis(counts, order, axis=1)  # (b, n)
    return order, sorted_values, sorted_counts  # (b, n), (1, n) or (b, n), (b, n)


def midranks(counts: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each row's average rank in its draw, and whether the draw holds two distinct values.

    A row drawn `w` times fills `w` tied positions, and rows with equal values share theirs, so
    a row's rank is the mean position of its tie group: SciPy's `rankdata(method="average")` on
    the drawn list. A row that was not drawn gets a finite rank that no weight reads.
    """
    # counts: (b, n); values: (n,), (1, n), or (b, n)
    order, sorted_values, sorted_counts = _sorted_draws(counts, values)  # (b, n), (1 or b, n), (b, n)
    draws = counts.shape[0]
    through = np.cumsum(sorted_counts, axis=1, dtype=np.float64)  # (b, n): draws up to and including each position

    # A tie group ends where the next sorted value differs.
    group_end = np.ones(sorted_values.shape, dtype=bool)  # (1 or b, n)
    group_end[:, :-1] = sorted_values[:, 1:] != sorted_values[:, :-1]
    group_end = np.broadcast_to(group_end, counts.shape)  # (b, n)
    # Draws through the end of each position's group, then through the end of the group before it.
    ends = np.where(group_end, through, np.inf)  # (b, n)
    group_through = np.minimum.accumulate(ends[:, ::-1], axis=1)[:, ::-1]  # (b, n)
    previous_ends = np.where(group_end, through, 0.0)  # (b, n)
    shifted = np.concatenate([np.zeros((draws, 1)), previous_ends[:, :-1]], axis=1)  # (b, n)
    group_before = np.maximum.accumulate(shifted, axis=1)  # (b, n)
    sorted_ranks = group_before + (group_through - group_before + 1.0) / 2.0  # (b, n)

    ranks = np.empty_like(sorted_ranks)  # (b, n)
    np.put_along_axis(ranks, order, sorted_ranks, axis=1)
    drawn_groups = np.sum(group_end & (group_through > group_before), axis=1)  # (b,)
    return ranks, drawn_groups >= 2  # (b, n), (b,)


def weighted_spearman(counts: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Spearman's rho of each draw.

    NaN where either side is constant or holds a NaN, or fewer than two rows were drawn: where
    `metrics.rank_correlation` returns None.
    """
    # counts: (b, n); first, second: (n,), (1, n), or (b, n)
    first_ranks, first_varies = midranks(counts, first)  # (b, n), (b,)
    second_ranks, second_varies = midranks(counts, second)  # (b, n), (b,)
    weights = counts.astype(np.float64)  # (b, n)
    total = weights.sum(axis=1, keepdims=True)  # (b, 1)
    safe_total = np.where(total > 0, total, 1.0)  # (b, 1)
    first_centered = first_ranks - np.sum(weights * first_ranks, axis=1, keepdims=True) / safe_total  # (b, n)
    second_centered = second_ranks - np.sum(weights * second_ranks, axis=1, keepdims=True) / safe_total  # (b, n)
    covariance = np.sum(weights * first_centered * second_centered, axis=1)  # (b,)
    first_variance = np.sum(weights * first_centered ** 2, axis=1)  # (b,)
    second_variance = np.sum(weights * second_centered ** 2, axis=1)  # (b,)

    drawn = counts > 0  # (b, n)
    unscored = np.any((np.isnan(np.atleast_2d(first)) | np.isnan(np.atleast_2d(second))) & drawn, axis=1)  # (b,)
    defined = first_varies & second_varies & (total[:, 0] >= 2) & ~unscored  # (b,)
    scale = np.sqrt(np.where(defined, first_variance * second_variance, 1.0))  # (b,)
    return np.where(defined, covariance / scale, np.nan)  # (b,)


def weighted_auroc(counts: np.ndarray, positive: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """The AUROC of `scores` for `positive` in each draw: scikit-learn's value, ties counted half.

    NaN where a draw holds only one class, or a NaN score, which scikit-learn refuses.
    """
    # counts: (b, n); positive: (n,), (1, n), or (b, n) bool; scores: (n,), (1, n), or (b, n)
    ranks, _ = midranks(counts, scores)  # (b, n)
    weights = counts.astype(np.float64)  # (b, n)
    positive_weights = weights * positive  # (b, n)
    positives = positive_weights.sum(axis=1)  # (b,)
    negatives = weights.sum(axis=1) - positives  # (b,)
    # Mann-Whitney U: the positives' rank sum above its minimum.
    u_statistic = np.sum(positive_weights * ranks, axis=1) - positives * (positives + 1.0) / 2.0  # (b,)
    unscored = np.any(np.isnan(np.atleast_2d(scores)) & (counts > 0), axis=1)  # (b,)
    defined = (positives > 0) & (negatives > 0) & ~unscored  # (b,)
    pairs = np.where(defined, positives * negatives, 1.0)  # (b,)
    return np.where(defined, u_statistic / pairs, np.nan)  # (b,)


def weighted_median(counts: np.ndarray, values: np.ndarray) -> np.ndarray:
    """`np.median` of each draw's values, the mean of the two middle values for an even count."""
    # counts: (b, n); values: (n,), (1, n), or (b, n)
    _, sorted_values, sorted_counts = _sorted_draws(counts, values)  # (b, n), (1 or b, n), (b, n)
    sorted_values = np.broadcast_to(sorted_values, counts.shape)  # (b, n)
    through = np.cumsum(sorted_counts, axis=1)  # (b, n)
    total = through[:, -1]  # (b,)
    # The drawn list's middle positions, counted from 0.
    lower = np.argmax(through > ((total - 1) // 2)[:, None], axis=1)  # (b,)
    upper = np.argmax(through > (total // 2)[:, None], axis=1)  # (b,)
    rows = np.arange(counts.shape[0])  # (b,)
    return (sorted_values[rows, lower] + sorted_values[rows, upper]) / 2.0  # (b,)


def _tie_group_means(weights: np.ndarray, ranked_losses: np.ndarray, sorted_values: np.ndarray) -> np.ndarray:
    """Each ranked position's loss replaced by the draw-weighted mean loss of its tie group.

    A tie group is a run of equal values among one draw's ranked rows, as
    `metrics.tie_averaged_losses` groups a drawn list, where a row drawn `w` times is `w` tied
    copies. Only drawn rows enter the mean, taken as an offset from the loss of the group's first
    drawn row, so a group whose drawn rows share one loss, as a row tied only with rows the draw
    left out does, keeps that loss exactly. A group the draw left out entirely gets 0, which no
    weight reads: no division by zero, and no non-finite loss of a row left out, reaches a sum.
    """
    # weights, ranked_losses: (b, n); sorted_values: (1 or b, n)
    starts = np.ones(sorted_values.shape, dtype=bool)  # (1 or b, n): where each tie group begins
    starts[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    flat_starts = np.broadcast_to(starts, weights.shape).ravel()  # (b * n,)
    # Every draw's first position begins a group, so the draws' groups are numbered in turn.
    group = np.cumsum(flat_starts) - 1  # (b * n,)
    groups = int(group[-1]) + 1 if group.size else 0
    flat_losses = ranked_losses.ravel()  # (b * n,)
    flat_weights = weights.ravel()  # (b * n,)

    drawn = np.flatnonzero(flat_weights > 0)  # (a,), ascending, so a group's drawn positions are consecutive
    drawn_group = group[drawn]  # (a,)
    first_drawn = np.ones(drawn.shape[0], dtype=bool)  # (a,)
    first_drawn[1:] = drawn_group[1:] != drawn_group[:-1]
    reference = np.zeros(groups)  # (u,): each drawn group's first drawn loss, 0 for a group left out
    reference[drawn_group[first_drawn]] = flat_losses[drawn[first_drawn]]

    drawn_weights = flat_weights[drawn]  # (a,)
    offsets = drawn_weights * (flat_losses[drawn] - reference[drawn_group])  # (a,)
    total_offset = np.bincount(drawn_group, weights=offsets, minlength=groups)  # (u,)
    total_weight = np.bincount(drawn_group, weights=drawn_weights, minlength=groups)  # (u,)
    mean_offset = np.divide(total_offset, total_weight, out=np.zeros(groups), where=total_weight > 0)  # (u,)
    return (reference + mean_offset)[group].reshape(weights.shape)  # (b, n)


def weighted_aurc(counts: np.ndarray, losses: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    """`metrics.selective_metrics`' AURC of each draw, NaN below two drawn rows.

    The draw is ranked by ascending uncertainty, and every row takes the draw-weighted mean loss
    of the rows tied with it, which makes the result the expectation over every order of the
    tied rows, as `metrics.tie_averaged_losses` makes it for a drawn list. With `m` drawn rows
    and `r_j` the mean loss of the first `j`, the reference's trapezoid over coverages `j / m`,
    divided by the coverage it spans, is `(sum_j r_j - (r_1 + r_m) / 2) / (m - 1)`. A row drawn
    `w` times after `K` drawn rows of total loss `L` adds
    `sum_{i=1..w} (L + i l) / (K + i) = w l + (L - K l) (H_{K+w} - H_K)` to that sum, with `H`
    the harmonic numbers, so no drawn list is built.
    """
    # counts: (b, n); losses: (n,), (1, n), or (b, n); uncertainty: (n,), (1, n), or (b, n)
    order, sorted_uncertainty, sorted_counts = _sorted_draws(counts, uncertainty)  # (b, n), (1 or b, n), (b, n)
    losses = np.broadcast_to(np.atleast_2d(losses), counts.shape)  # (b, n)
    ranked_losses = np.take_along_axis(losses, order, axis=1).astype(np.float64)  # (b, n)
    weights = sorted_counts.astype(np.float64)  # (b, n)
    sorted_losses = _tie_group_means(weights, ranked_losses, sorted_uncertainty)  # (b, n)

    draws_through = np.cumsum(weights, axis=1)  # (b, n)
    draws_before = draws_through - weights  # (b, n)
    loss_through = np.cumsum(weights * sorted_losses, axis=1)  # (b, n)
    loss_before = loss_through - weights * sorted_losses  # (b, n)
    total = draws_through[:, -1]  # (b,)
    # H_{K+w} - H_K as a digamma difference, exact to rounding at any K.
    harmonic_step = digamma(draws_through + 1.0) - digamma(draws_before + 1.0)  # (b, n)
    blocks = weights * sorted_losses + (loss_before - draws_before * sorted_losses) * harmonic_step  # (b, n)
    risk_sum = blocks.sum(axis=1)  # (b,)

    rows = np.arange(counts.shape[0])  # (b,)
    first_loss = sorted_losses[rows, np.argmax(sorted_counts > 0, axis=1)]  # (b,)
    last_risk = loss_through[:, -1] / np.maximum(total, 1)  # (b,)
    defined = total >= 2  # (b,)
    spanned = np.where(defined, total - 1.0, 1.0)  # (b,)
    return np.where(defined, (risk_sum - (first_loss + last_risk) / 2.0) / spanned, np.nan)  # (b,)


def weighted_mcc(counts: np.ndarray, labels: np.ndarray, predicted: np.ndarray, classes: int) -> np.ndarray:
    """`metrics.matthews_correlation` of each draw: multiclass MCC, 0 where undefined."""
    # counts: (b, n); labels: (n,); predicted: (n,), (1, n), or (b, n)
    draws = counts.shape[0]
    codes = np.broadcast_to(labels * classes + predicted, counts.shape)  # (b, n)
    offsets = np.arange(draws)[:, None] * classes * classes  # (b, 1)
    confusion = np.bincount(
        (offsets + codes).ravel(), weights=counts.ravel().astype(np.float64), minlength=draws * classes * classes
    ).reshape(draws, classes, classes)  # (b, c, c)
    true_counts = confusion.sum(axis=2)  # (b, c)
    predicted_counts = confusion.sum(axis=1)  # (b, c)
    total = confusion.sum(axis=(1, 2))  # (b,)
    correct = np.trace(confusion, axis1=1, axis2=2)  # (b,)
    numerator = correct * total - np.sum(true_counts * predicted_counts, axis=1)  # (b,)
    denominator = np.sqrt(
        (total ** 2 - np.sum(predicted_counts ** 2, axis=1)) * (total ** 2 - np.sum(true_counts ** 2, axis=1))
    )  # (b,)
    return np.divide(numerator, denominator, out=np.zeros(draws), where=denominator > 0)  # (b,)


def weighted_ece(
    counts: np.ndarray, outcomes: np.ndarray, probs: np.ndarray, bins: int, order: np.ndarray | None = None
) -> np.ndarray:
    """`foundry.metrics.expected_calibration_error` of each draw, NaN for a draw with no rows.

    The reference sorts the drawn list by probability, ties in input order, and cuts it into
    `bins` runs whose lengths differ by at most one; each run adds the gap between its outcome
    total and its probability total. A run can end inside one row's duplicates, which then count
    toward both runs, so every run total is a difference of running sums at the run's ends.

    `order`, when given, is a row order that sorts the probabilities of every draw, such as the
    order of probabilities a monotone calibration map was applied to. A draw it does not sort is
    sorted on its own. Rows it orders differently from input order among tied probabilities are
    the one departure from the reference, and move the result only when such a tie straddles
    the end of a run.
    """
    # counts: (b, n); outcomes: (n,), (1, n), or (b, n) in {0, 1}; probs: (n,), (1, n), or (b, n); order: (n,) or None
    probs = np.atleast_2d(probs)  # (1 or b, n)
    if order is None and probs.shape[0] == 1:
        order = np.argsort(probs[0], kind="stable")  # (n,)
    if order is None:
        orders = np.argsort(probs, axis=1, kind="stable")  # (b, n)
    else:
        orders = np.array(np.broadcast_to(order, probs.shape))  # (1 or b, n)
        unsorted = np.any(np.diff(np.take_along_axis(probs, orders, axis=1), axis=1) < 0, axis=1)  # (1 or b,)
        if unsorted.any():
            orders[unsorted] = np.argsort(probs[unsorted], axis=1, kind="stable")

    sorted_probs = np.take_along_axis(probs, orders, axis=1).astype(np.float64)  # (1 or b, n)
    sorted_outcomes = np.take_along_axis(np.atleast_2d(outcomes), orders, axis=1).astype(np.float64)  # (1 or b, n)
    weights = np.take_along_axis(counts, np.broadcast_to(orders, counts.shape), axis=1).astype(np.float64)  # (b, n)
    through = np.cumsum(weights, axis=1)  # (b, n): drawn positions up to and including each row
    outcome_through = np.cumsum(weights * sorted_outcomes, axis=1)  # (b, n)
    prob_through = np.cumsum(weights * sorted_probs, axis=1)  # (b, n)
    return _ece_of_sorted_draws(through, outcome_through, prob_through, sorted_outcomes, sorted_probs, bins)  # (b,)


def _ece_of_sorted_draws(
    through: np.ndarray,
    outcome_through: np.ndarray,
    prob_through: np.ndarray,
    sorted_outcomes: np.ndarray,
    sorted_probs: np.ndarray,
    bins: int,
) -> np.ndarray:
    """Equal-mass calibration error of draws whose rows or entries are in ascending probability.

    `through`, `outcome_through`, and `prob_through` are running totals of the draw counts, and of
    the counts times each outcome and probability, in that order. The reference cuts the drawn
    list into `bins` runs whose lengths differ by at most one, the first ones longer, as
    `np.array_split` does, and adds each run's gap between its outcome total and its probability
    total. A run can end inside one row's duplicates, which then count toward both runs.
    """
    # through, outcome_through, prob_through: (b, m); sorted_outcomes, sorted_probs: (1 or b, m); m rows or entries
    draws, length = through.shape
    total = through[:, -1]  # (b,)
    run_length, extra = np.divmod(total, bins)  # (b,), (b,)
    run_index = np.arange(bins + 1)  # (bins + 1,)
    edges = run_index * run_length[:, None] + np.minimum(run_index, extra[:, None])  # (b, bins + 1)

    # Rows wholly before each edge, for every draw in one search: each draw's positions are offset
    # past the previous draw's.
    span = float(total.max()) + 1.0
    offsets = np.arange(draws)[:, None] * span  # (b, 1)
    before = np.searchsorted((through + offsets).ravel(), (edges + offsets).ravel(), side="right")  # (b * (bins + 1),)
    before = before.reshape(draws, bins + 1) - np.arange(draws)[:, None] * length  # (b, bins + 1)
    last_whole = np.maximum(before - 1, 0)  # (b, bins + 1)
    straddling = np.minimum(before, length - 1)  # (b, bins + 1)
    has_whole = before > 0  # (b, bins + 1)
    through_whole = np.where(has_whole, np.take_along_axis(through, last_whole, axis=1), 0.0)  # (b, bins + 1)
    # Positions of the row straddling each edge that lie before the edge; none past the last row.
    partial = np.where(before < length, edges - through_whole, 0.0)  # (b, bins + 1)

    def mass_before_edges(running: np.ndarray, values: np.ndarray) -> np.ndarray:
        # running: (b, m); values: (1 or b, m)
        whole = np.where(has_whole, np.take_along_axis(running, last_whole, axis=1), 0.0)  # (b, bins + 1)
        return whole + partial * np.take_along_axis(values, straddling, axis=1)  # (b, bins + 1)

    outcome_runs = np.diff(mass_before_edges(outcome_through, sorted_outcomes), axis=1)  # (b, bins)
    prob_runs = np.diff(mass_before_edges(prob_through, sorted_probs), axis=1)  # (b, bins)
    gaps = np.abs(outcome_runs - prob_runs).sum(axis=1)  # (b,)
    return np.divide(gaps, total, out=np.full(draws, np.nan), where=total > 0)  # (b,)


def _protify_cutoffs(start: float) -> np.ndarray:
    """The float32 cutoffs `metrics.max_metrics` scans when the smallest drawn score is `start`."""
    if start >= 1.0:
        start = 0.0
    return torch.arange(start, 1, F1_THRESHOLD_STEP, dtype=torch.float32).numpy()  # (k,)


def _best_protify_f1(true_positive: np.ndarray, called: np.ndarray, positives: np.ndarray) -> np.ndarray:
    """The best F1 over cutoffs, in the reference's float32 arithmetic, operation for operation."""
    # true_positive, called: (b, k) exact counts; positives: (b,) exact counts
    true_positive_32 = true_positive.astype(np.float32)  # (b, k)
    false_positive = (called - true_positive).astype(np.float32)  # (b, k)
    false_negative = (positives[:, None] - true_positive).astype(np.float32)  # (b, k)
    zeros = np.zeros_like(true_positive_32)  # (b, k)

    precision_denominator = true_positive_32 + false_positive  # (b, k)
    precision = np.divide(true_positive_32, precision_denominator, out=zeros.copy(), where=precision_denominator != 0)  # (b, k)
    recall_denominator = true_positive_32 + false_negative  # (b, k)
    recall = np.divide(true_positive_32, recall_denominator, out=zeros.copy(), where=recall_denominator != 0)  # (b, k)
    f1_denominator = precision + recall  # (b, k)
    f1 = np.divide(np.float32(2) * precision * recall, f1_denominator, out=zeros.copy(), where=f1_denominator != 0)  # (b, k)
    return f1.max(axis=1).astype(np.float64)  # (b,)


def _protify_scores(probs: np.ndarray) -> np.ndarray:
    """Probabilities as `max_metrics` reads them before its float32 cast: NaN to 0, clamped to [-1, 1]."""
    # probs: (n, c)
    return np.clip(np.nan_to_num(probs, nan=0.0), -1.0, 1.0)  # (n, c), the dtype of probs


def weighted_protify_f1(counts: np.ndarray, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Protify's multilabel F1 of each draw, every draw reading the same probabilities.

    `metrics.max_metrics` flattens the drawn probabilities, scans float32 cutoffs from their
    minimum to 1 in steps of 0.01, calls an entry positive at or above a cutoff, and keeps the
    best float32 F1. Draws that share a smallest drawn probability share their cutoffs, so for
    each such group a table of every row's called and true-positive entries at each cutoff gives
    the group's counts in one matrix product. NaN for a draw with no rows.
    """
    # counts: (b, n); probs: (n, c); labels: (n, c) in {0, 1}
    rows = probs.shape[0]
    scores = _protify_scores(probs)  # (n, c)
    row_min = scores.min(axis=1)  # (n,), in the probabilities' dtype, as the reference takes its start
    comparable = scores.astype(np.float32)  # (n, c)
    positive = labels > 0.5  # (n, c)
    weights = counts.astype(np.float64)  # (b, n)
    positives = weights @ positive.sum(axis=1)  # (b,)
    starts = np.where(counts > 0, row_min, np.inf).min(axis=1)  # (b,)
    row_index = np.arange(rows)[:, None]  # (n, 1)

    f1 = np.full(counts.shape[0], np.nan)  # (b,)
    for start in np.unique(starts[np.isfinite(starts)]):
        members = np.flatnonzero(starts == start)  # (g,)
        cutoffs = _protify_cutoffs(float(start))  # (k,)
        if cutoffs.size == 0:
            f1[members] = 0.0
            continue
        slots = cutoffs.shape[0] + 1
        # An entry is called at cutoff j when more than j cutoffs lie at or below it.
        at_or_below = np.searchsorted(cutoffs, comparable, side="right")  # (n, c)
        codes = (row_index * slots + at_or_below).ravel()  # (n * c,)
        entries = np.bincount(codes, minlength=rows * slots).reshape(rows, slots)  # (n, k + 1)
        positive_entries = np.bincount(codes, weights=positive.ravel(), minlength=rows * slots).reshape(rows, slots)  # (n, k + 1)
        called = np.cumsum(entries[:, ::-1], axis=1)[:, ::-1][:, 1:].astype(np.float64)  # (n, k)
        true_positive = np.cumsum(positive_entries[:, ::-1], axis=1)[:, ::-1][:, 1:]  # (n, k)
        member_weights = weights[members]  # (g, n)
        f1[members] = _best_protify_f1(member_weights @ true_positive, member_weights @ called, positives[members])
    return f1  # (b,)


@dataclass(frozen=True)
class _SortedEntries:
    """One draw's flat entries in ascending probability, with each entry's row and outcome."""

    probs: np.ndarray  # (e,) float64
    comparable: np.ndarray  # (e,) float32, as max_metrics compares
    rows: np.ndarray  # (e,)
    outcomes: np.ndarray  # (e,) float64


def multilabel_draw_scores(
    counts: np.ndarray, probs: np.ndarray, labels: np.ndarray, bins: int, entry_order: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Protify's F1 and the per-entry calibration error of each draw of a multilabel task.

    Both read the drawn entries in ascending probability. `metrics.max_metrics` counts the entries
    at or above each float32 cutoff, from the smallest drawn probability to 1 in steps of 0.01,
    which is one search per cutoff; `metrics.calibration_error` cuts the sorted entries into
    equal-mass runs. `probs` is shared by every draw, `(n, c)`, or each draw's own, `(b, n, c)`.

    `entry_order`, when given, is a flat entry order expected to sort every draw, such as the
    order of the probabilities a monotone calibration map was applied to; a draw it does not sort
    is sorted on its own. The F1 does not depend on how tied entries are ordered, so it is exact
    either way. The calibration error departs from the reference only where tied entries of one
    row straddle the end of a run, since the reference interleaves a duplicated row's copies.
    NaN for a draw with no rows.
    """
    # counts: (b, n); probs: (n, c) or (b, n, c); labels: (n, c) in {0, 1}; entry_order: (e,) or None
    if np.isnan(probs).any():
        raise ValueError("multilabel probabilities hold NaN; max_metrics would score them as 0.")
    draws, rows = counts.shape
    classes = labels.shape[1]
    entry_rows = np.repeat(np.arange(rows), classes)  # (e,)
    entry_outcomes = labels.reshape(-1).astype(np.float64)  # (e,)
    weights = counts.astype(np.float64)  # (b, n)
    positives = weights @ labels.sum(axis=1, dtype=np.float64)  # (b,)
    fixed = None if entry_order is None else (entry_rows[entry_order], entry_outcomes[entry_order])

    def sorted_entries(draw_probs: np.ndarray) -> _SortedEntries:
        # draw_probs: (n, c)
        flat = draw_probs.reshape(-1)  # (e,)
        values = None if entry_order is None else flat[entry_order]  # (e,)
        if values is not None and not np.any(values[1:] < values[:-1]):
            value_rows, value_outcomes = fixed
        else:
            order = np.argsort(flat, kind="stable")  # (e,)
            values, value_rows, value_outcomes = flat[order], entry_rows[order], entry_outcomes[order]
        # max_metrics clamps to [-1, 1] in the input dtype, then compares in float32.
        comparable = np.clip(values, -1.0, 1.0).astype(np.float32, copy=False)  # (e,)
        return _SortedEntries(values.astype(np.float64), comparable, value_rows, value_outcomes)

    shared = None if probs.ndim == 3 else (sorted_entries(probs), probs.min(axis=1))  # entries, (n,)
    f1 = np.full(draws, np.nan)  # (b,)
    ece = np.full(draws, np.nan)  # (b,)
    for draw in range(draws):
        drawn = counts[draw] > 0  # (n,)
        if not drawn.any():
            continue
        entries, row_min = shared or (sorted_entries(probs[draw]), probs[draw].min(axis=1))  # entries, (n,)
        entry_weights = weights[draw, entries.rows]  # (e,)
        through = np.cumsum(entry_weights)  # (e,)
        outcome_through = np.cumsum(entry_weights * entries.outcomes)  # (e,)
        prob_through = np.cumsum(entry_weights * entries.probs)  # (e,)
        ece[draw] = _ece_of_sorted_draws(
            through[None], outcome_through[None], prob_through[None], entries.outcomes[None], entries.probs[None], bins
        )[0]

        # The start is the smallest drawn probability, clamped, in the probabilities' dtype.
        cutoffs = _protify_cutoffs(float(np.clip(row_min[drawn].min(), -1.0, 1.0)))  # (k,)
        if cutoffs.size == 0:
            f1[draw] = 0.0
            continue
        cuts = np.searchsorted(entries.comparable, cutoffs, side="left")  # (k,): entries from each cut on are called
        called_before = np.where(cuts > 0, through[np.maximum(cuts - 1, 0)], 0.0)  # (k,)
        positive_before = np.where(cuts > 0, outcome_through[np.maximum(cuts - 1, 0)], 0.0)  # (k,)
        true_positive = outcome_through[-1] - positive_before  # (k,)
        called = through[-1] - called_before  # (k,)
        f1[draw] = _best_protify_f1(true_positive[None], called[None], positives[draw:draw + 1])[0]
    return f1, ece  # (b,), (b,)
