"""What the splits of a dataset share, checked on the rows a builder is about to publish.

The split code promises disjoint splits. This checks its output independently of that code and
returns what it found, so a card or manifest can record the result and a builder can refuse.
"""

from __future__ import annotations

import math

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass


EXAMPLES_PER_FIELD = 5


class LeakageError(ValueError):
    """Two splits share a value, or a value that must be unique repeats inside one split."""


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """Per checked field, the values that break disjointness.

    `shared[field]` maps each value that rows of two or more splits hold to those splits, in the
    order the splits were given. `repeated[field]` maps each value that more than one row of a
    single split holds to that split's row count. It has a key only for the fields checked for
    uniqueness, because most rows of a cluster share its identifier by design.
    """

    split_rows: dict[str, int]
    shared: dict[str, dict[str, tuple[str, ...]]]
    repeated: dict[str, dict[str, dict[str, int]]]

    @property
    def clean(self) -> bool:
        return not any(self.shared.values()) and not any(self.repeated.values())

    def summary(self) -> str:
        """One line per field with a problem, naming the first few values in sorted order."""
        lines: list[str] = []
        for field, holders in self.shared.items():
            examples = [f"{value} ({', '.join(holders[value])})" for value in sorted(holders)[:EXAMPLES_PER_FIELD]]
            if examples:
                lines.append(f"{len(holders)} {field} value(s) in more than one split: {'; '.join(examples)}")
        for field, counts in self.repeated.items():
            examples = [
                f"{value} ({', '.join(f'{split} x{count}' for split, count in counts[value].items())})"
                for value in sorted(counts)[:EXAMPLES_PER_FIELD]
            ]
            if examples:
                lines.append(f"{len(counts)} {field} value(s) repeated within a split: {'; '.join(examples)}")
        return "\n".join(lines)

    def raise_if_leaking(self) -> None:
        if not self.clean:
            raise LeakageError(self.summary())

    def to_dict(self) -> dict[str, object]:
        """The whole report as JSON-ready data, values sorted, for a card or manifest."""
        return {
            "split_rows": dict(self.split_rows),
            "shared": {
                field: {value: list(values[value]) for value in sorted(values)}
                for field, values in self.shared.items()
            },
            "repeated": {
                field: {value: dict(values[value]) for value in sorted(values)}
                for field, values in self.repeated.items()
            },
        }


@dataclass(frozen=True, slots=True)
class HomologyComparison:
    """One directed search, including every query at or above the forbidden identity."""

    query_split: str
    reference_split: str
    query_count: int
    reference_count: int
    maximum_identity: float
    near_duplicates: dict[str, float]


@dataclass(frozen=True, slots=True)
class HomologyLeakageReport:
    """Evidence for every ordered split pair; query coverage can make searches asymmetric."""

    identity_threshold: float
    search_parameters: dict[str, object]
    comparisons: tuple[HomologyComparison, ...]

    @property
    def clean(self) -> bool:
        return not any(comparison.near_duplicates for comparison in self.comparisons)

    def raise_if_leaking(self) -> None:
        leaking = [
            f"{item.query_split}->{item.reference_split}: {len(item.near_duplicates)} queries"
            for item in self.comparisons if item.near_duplicates
        ]
        if leaking:
            raise LeakageError("cross-split homology leakage: " + "; ".join(leaking))

    def to_dict(self) -> dict[str, object]:
        return {
            "identity_threshold": self.identity_threshold,
            "search_parameters": self.search_parameters,
            "comparisons": [asdict(comparison) for comparison in self.comparisons],
        }


def check_cross_split_homology(
    sequences_by_split: Mapping[str, Mapping[str, str]],
    search: Callable[[Mapping[str, str], Mapping[str, str]], Mapping[str, float]],
    *,
    search_parameters: Mapping[str, object],
    identity_threshold: float = 0.9,
) -> HomologyLeakageReport:
    """Search every ordered split pair independently of the allocator's cluster memberships.

    `search` uses the existing homology engine and must return a finite maximum identity for
    every query, including zero for no hit. Search failures propagate; missing queries never
    become evidence of no match. Empty splits are refused for a release audit.
    """
    if isinstance(identity_threshold, bool) or not isinstance(identity_threshold, (int, float)) or not 0 < identity_threshold <= 1:
        raise ValueError("identity_threshold must be in (0, 1]")
    if len(sequences_by_split) < 2:
        raise ValueError("a cross-split homology audit needs at least two nonempty splits")
    owner: dict[str, str] = {}
    for split, sequences in sequences_by_split.items():
        if not isinstance(split, str) or not split or not sequences:
            raise ValueError("every audited split needs a nonempty name and sequence mapping")
        for identifier, sequence in sequences.items():
            if not isinstance(identifier, str) or not identifier or not isinstance(sequence, str) or not sequence:
                raise ValueError(f"split {split!r} has an empty or non-string sequence identifier or sequence")
            if identifier in owner:
                raise LeakageError(f"protein {identifier!r} appears in both {owner[identifier]!r} and {split!r}")
            owner[identifier] = split
    comparisons: list[HomologyComparison] = []
    for query_split in sorted(sequences_by_split):
        queries = sequences_by_split[query_split]
        for reference_split in sorted(sequences_by_split):
            if query_split == reference_split:
                continue
            references = sequences_by_split[reference_split]
            measured = search(queries, references)
            if set(measured) != set(queries):
                raise ValueError(f"homology search {query_split}->{reference_split} did not measure exactly its queries")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1
                for value in measured.values()
            ):
                raise ValueError(f"homology search {query_split}->{reference_split} returned an invalid identity")
            comparisons.append(HomologyComparison(
                query_split=query_split,
                reference_split=reference_split,
                query_count=len(queries),
                reference_count=len(references),
                maximum_identity=float(max(measured.values())),
                near_duplicates={protein: float(measured[protein]) for protein in sorted(measured) if measured[protein] >= identity_threshold},
            ))
    return HomologyLeakageReport(float(identity_threshold), dict(search_parameters), tuple(comparisons))


def check_split_disjointness(
    rows_by_split: Mapping[str, Iterable[Mapping[str, object]]],
    fields: Sequence[str],
    *,
    unique_within_split: Sequence[str] = (),
) -> LeakageReport:
    """Find every value of `fields` that two splits share, and every repeat of `unique_within_split`.

    Each split's rows are read once, so they may be a generator. Every checked value must be a
    string, so that `1` and `"1"` cannot pass as different proteins.
    """
    if not fields or len(set(fields)) != len(fields):
        raise ValueError(f"Fields to check must be distinct and non-empty, got {list(fields)}")
    unchecked = sorted(set(unique_within_split) - set(fields))
    if unchecked:
        raise ValueError(f"unique_within_split names fields that are not checked: {unchecked}")

    first_split: dict[str, dict[str, str]] = {field: {} for field in fields}
    shared: dict[str, dict[str, list[str]]] = {field: {} for field in fields}
    repeated: dict[str, dict[str, dict[str, int]]] = {field: {} for field in unique_within_split}
    split_rows: dict[str, int] = {}
    for split, rows in rows_by_split.items():
        counts: dict[str, Counter[str]] = {field: Counter() for field in unique_within_split}
        row_count = 0
        for index, row in enumerate(rows):
            for field in fields:
                if field not in row:
                    raise ValueError(f"Row {index} of split {split!r} has no {field!r}")
                value = row[field]
                if not isinstance(value, str):
                    raise TypeError(f"Row {index} of split {split!r} has a non-string {field}: {value!r}")
                if field in counts:
                    counts[field][value] += 1
                owner = first_split[field].setdefault(value, split)
                if owner != split:
                    # Splits are read in order, so the current split is appended at most once.
                    holders = shared[field].setdefault(value, [owner])
                    if holders[-1] != split:
                        holders.append(split)
            row_count += 1
        split_rows[split] = row_count
        for field, field_counts in counts.items():
            for value, count in field_counts.items():
                if count > 1:
                    repeated[field].setdefault(value, {})[split] = count

    frozen_shared = {
        field: {value: tuple(holders) for value, holders in values.items()}
        for field, values in shared.items()
    }
    return LeakageReport(split_rows=split_rows, shared=frozen_shared, repeated=repeated)
