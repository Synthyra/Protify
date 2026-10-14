"""How alike the label distributions of a split's evaluation parts and its train part are.

Logan, 2026-09-29: labels should be in each split with distributions as similar as the dataset allows. A split that holds out
whole families cannot match train exactly, so the check compares each evaluation split with what a random sample of proteins
of its size would show: `total_variation` is half the summed absolute difference between the split's label shares and train's,
and `expected_variation` is the same figure for such a sample. Their ratio is `excess`. On subcellular localization at an
enforced 0.3 (2026-09-29, 59 labels, three seeds) the excess was 3.0 to 4.7 when the largest families went to the evaluation
splits first (total variation 0.096 to 0.147), 1.1 to 2.0 with a seeded unit order (0.038 to 0.070), and 1.03 to 1.21 with
`balance_labels` (0.035 to 0.042), against a random sample's 0.034. `measure_label_balance` passes a split whose excess is at most
`max_excess`, 2 by default.

Also reported, per split, are the 5th, 50th and 95th percentile over labels of the split's prevalence relative to train's,
and the labels train carries that the split has none of. The labels are those of `label_sets`, so a caller passes the labels it
means to balance.
"""

from __future__ import annotations

import math

from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass

from .stratified import EVALUATION_SPLITS


# A seeded draw of families shows up to about twice the difference a random sample of proteins does, because a family is drawn whole
# and a protein's labels come together; a draw that hands the big families to the evaluation splits shows three to five times.
DEFAULT_MAX_EXCESS = 2.0
PERCENTILES = (0.05, 0.5, 0.95)


@dataclass(frozen=True, slots=True)
class SplitLabelBalance:
    """One evaluation split's label distribution against train's."""

    split: str
    proteins: int
    total_variation: float
    expected_variation: float
    # The 5th, 50th and 95th percentile over labels of the split's prevalence divided by train's.
    prevalence_ratio: tuple[float, float, float]
    absent_labels: int

    @property
    def excess(self) -> float:
        return self.total_variation / self.expected_variation if self.expected_variation else 0.0


@dataclass(frozen=True, slots=True)
class LabelBalance:
    """The three evaluation splits' balance, and whether each is within `max_excess` of a random sample's."""

    splits: tuple[SplitLabelBalance, ...]
    labels: int
    max_excess: float

    @property
    def worst_excess(self) -> float:
        return max(item.excess for item in self.splits)

    @property
    def passed(self) -> bool:
        return self.worst_excess <= self.max_excess

    def to_dict(self) -> dict[str, object]:
        """A JSON-compatible record: each split with its excess, and whether the check passed."""
        return {
            "labels": self.labels, "max_excess": self.max_excess, "worst_excess": self.worst_excess, "passed": self.passed,
            "splits": [{**asdict(item), "excess": item.excess, "prevalence_ratio": list(item.prevalence_ratio)} for item in self.splits],
        }

    def table(self) -> str:
        """The balance as a Markdown table for a decision record."""
        lines = ["| Split | Proteins | Total variation | Random sample | Excess | Prevalence ratio 5th / 50th / 95th | Labels absent |",
                 "|---|---:|---:|---:|---:|---|---:|"]
        for item in self.splits:
            low, middle, high = item.prevalence_ratio
            lines.append(f"| {item.split} | {item.proteins:,} | {item.total_variation:.3f} | {item.expected_variation:.3f} | {item.excess:.2f} "
                         f"| {low:.2f} / {middle:.2f} / {high:.2f} | {item.absent_labels} |")
        return "\n".join(lines)


def expected_share_deviation(draws: int, share: float) -> float:
    """The mean absolute difference between a label's share of `draws` random draws and its true `share` (de Moivre's formula)."""
    if draws <= 0 or share <= 0.0 or share >= 1.0:
        return 0.0
    top = math.floor(draws * share) + 1
    log_term = (
        math.lgamma(draws + 1) - math.lgamma(top + 1) - math.lgamma(draws - top + 1)
        + top * math.log(share) + (draws - top + 1) * math.log1p(-share)
    )
    return 2.0 * top * math.exp(log_term) / draws


def _percentile(ordered: list[float], fraction: float) -> float:
    """The value at `fraction` of the way through `ordered`, the nearer of two neighbors, the upper on a tie."""
    return ordered[int(fraction * (len(ordered) - 1) + 0.5)]


def measure_label_balance(
    protein_to_split: Mapping[str, str], label_sets: Mapping[str, Collection[str]], *, max_excess: float = DEFAULT_MAX_EXCESS,
) -> LabelBalance:
    """Compare each evaluation split's labels with train's.

    `label_sets` maps every protein to the labels it carries, a protein with none to an empty collection.
    """
    if set(protein_to_split) != set(label_sets):
        raise ValueError("label_sets must cover the split's proteins exactly")
    if not max_excess > 0.0:
        raise ValueError(f"max_excess must be positive, got {max_excess!r}")
    members: dict[str, list[str]] = {}
    for protein, name in protein_to_split.items():
        members.setdefault(name, []).append(protein)
    if "train" not in members or any(name not in members for name in EVALUATION_SPLITS):
        raise ValueError("a label balance needs proteins in train, validation, calibration and test")
    counts = {name: Counter(label for protein in proteins for label in set(label_sets[protein])) for name, proteins in members.items()}
    train = counts["train"]
    if not train:
        raise ValueError("no protein of train carries a label")
    train_proteins = len(members["train"])
    train_occurrences = sum(train.values())
    balances: list[SplitLabelBalance] = []
    for name in EVALUATION_SPLITS:
        held, proteins = counts[name], len(members[name])
        occurrences = sum(held.values())
        if not occurrences:
            raise ValueError(f"no protein of {name} carries a label")
        variation = 0.5 * sum(abs(train[label] / train_occurrences - held[label] / occurrences) for label in set(train) | set(held))
        expected = 0.5 * sum(expected_share_deviation(occurrences, train[label] / train_occurrences) for label in train)
        ratios = sorted((held[label] / proteins) / (train[label] / train_proteins) for label in train)
        low, middle, high = (_percentile(ratios, fraction) for fraction in PERCENTILES)
        balances.append(SplitLabelBalance(
            name, proteins, variation, expected, (low, middle, high), sum(1 for label in train if not held[label]),
        ))
    return LabelBalance(tuple(balances), len(train), max_excess)
