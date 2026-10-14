"""The hits of one all-against-all search, answering the searches a split's repair asks without running MMseqs2 again.

Repair returns held-out clusters to train while any of their proteins is a near-duplicate of a protein elsewhere, and each round
asks for new searches: the clusters it moved against the splits they left and joined, the flagged proteins against a whole split
to find which cluster they match. Every one of them is a pair of subsets of one set of sequences, so one search of the set
against itself holds all their answers, and a round costs a lookup where it cost a database build, an index and an alignment
(about 2.5 minutes each on the GO release, 58 rounds in 2.5 hours, 2026-09-30).

MMseqs2 scales an alignment's e-value by the size of the target database it searched, in residues. A search of a
subset therefore reports a smaller e-value for the same pair than the search of the whole set, and counts a hit the whole-set search
left out. `PairTable` applies the same scaling to the stored e-value, so a query of a subset counts the pairs a search of that
subset would count. The all-against-all search keeps hits up to `reach` times the e-value limit, which covers every reference set
of at least `1 / reach` of the residues: the evaluation splits of a release (5 percent each, so the default reach of 25 covers
them). A smaller reference set, such as the few clusters one repair round moved, is answered as if it held `1 / reach` of the
residues, which leaves out the marginal hits only it would have counted; the measurement of the whole assignment, which
ends every repair, has none.

Shapes: n sequences in the table, m stored hits, q queries and p references of one lookup, m_q stored hits of the queries,
and m_c of those the lookup counts.
"""

from __future__ import annotations

import sys
import time
import numpy as np

from array import array
from collections.abc import Collection, Mapping
from dataclasses import replace
from pathlib import Path
from typing import TypedDict

from .homology import DEFAULT_REACH, HIT_COLUMNS, SearchConfig, search_file
from .mmseqs import CommandRunner


class PairTableSummary(TypedDict):
    """What a pair table holds and how it answers, as a split's record keeps it."""

    source: str
    sequences: int
    residues: int
    hits: int
    evalue_max: float
    reach: float


class PairTable:
    """Every hit of the search of a set of sequences against itself, as arrays sorted by query."""

    def __init__(self, sequences: Mapping[str, str], hits: Path, config: SearchConfig, *, reach: float = DEFAULT_REACH) -> None:
        if not sequences:
            raise ValueError("a pair table needs sequences")
        if isinstance(reach, bool) or not isinstance(reach, (int, float)) or reach < 1.0:
            raise ValueError(f"reach must be a number of at least 1, got {reach!r}")
        self.config, self.reach = config, float(reach)
        self.names = tuple(sorted(sequences))  # n
        self.index = {name: position for position, name in enumerate(self.names)}
        self.residues = np.array([len(sequences[name]) for name in self.names], dtype=np.int64)  # (n,)
        self.total_residues = int(self.residues.sum())
        limit = config.evalue_max * self.reach

        started = time.monotonic()
        print(f"Pair table: reading the hits of {len(self.names):,} sequences", file=sys.stderr, flush=True)
        # Each grows to (m,), in file order.
        query, target, identity, evalue = array("i"), array("i"), array("d"), array("d")
        with hits.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != len(HIT_COLUMNS):
                    raise ValueError(f"MMseqs hit line {number} has {len(fields)} fields, expected {len(HIT_COLUMNS)}")
                try:
                    query.append(self.index[fields[0]])
                    target.append(self.index[fields[1]])
                    identity.append(float(fields[2]))
                    evalue.append(float(fields[4]))
                except KeyError as error:
                    raise ValueError(f"MMseqs hit line {number} names a sequence outside this search: {error.args[0]!r}") from error
                except ValueError as error:
                    raise ValueError(f"MMseqs hit line {number} has a malformed number: {line!r}") from error
                if not 0.0 <= identity[-1] <= 1.0 or not 0.0 <= evalue[-1] <= limit:
                    raise ValueError(f"MMseqs hit line {number} is out of range for an e-value limit of {limit:g}: {line!r}")

        rows = np.frombuffer(query, dtype=np.int32)  # (m,), each hit's query row
        order = np.argsort(rows, kind="stable")  # (m,)
        self.target = np.frombuffer(target, dtype=np.int32)[order]  # (m,), sorted by query
        self.identity = np.frombuffer(identity, dtype=np.float64)[order]  # (m,)
        self.evalue = np.frombuffer(evalue, dtype=np.float64)[order]  # (m,)
        # The stored hits of the query in row i are offsets[i]:offsets[i + 1].
        self.offsets = np.searchsorted(rows[order], np.arange(len(self.names) + 1))  # (n + 1,)
        self.hits = len(rows)  # m
        print(
            f"Pair table: {self.hits:,} hits of {len(self.names):,} sequences read in {time.monotonic() - started:.1f} s",
            file=sys.stderr, flush=True,
        )

    @classmethod
    def from_search(
        cls, sequences: Mapping[str, str], directory: Path, config: SearchConfig, *, reach: float = DEFAULT_REACH,
        runner: CommandRunner | None = None,
    ) -> PairTable:
        """Search `sequences` against themselves once, in a stage under `directory` a rerun reuses, at `reach` times the e-value limit."""
        relaxed = replace(config, evalue_max=config.evalue_max * reach)
        return cls(sequences, search_file(sequences, sequences, directory, relaxed, runner=runner), config, reach=reach)

    def describe(self) -> PairTableSummary:
        """What the table holds and how it answers, for a record that says a split was measured from it."""
        return {
            "source": "all_pairs", "sequences": len(self.names), "residues": self.total_residues, "hits": self.hits,
            "evalue_max": self.config.evalue_max, "reach": self.reach,
        }

    def _rows(self, names: Collection[str], role: str) -> np.ndarray:
        try:
            return np.fromiter((self.index[name] for name in names), dtype=np.int64, count=len(names))  # (len(names),)
        except KeyError as error:
            raise ValueError(f"the pair table does not hold the {role} {error.args[0]!r}") from error

    def _counted(self, queries: Mapping[str, str], references: Mapping[str, str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """The hits a search of `queries` against `references` counts: their query's place in `queries`, their row, and `queries`' order."""
        if not queries or not references:
            raise ValueError("a search needs at least one query and one reference")
        query_order = list(queries)  # q
        rows = self._rows(query_order, "query")  # (q,)
        reference_rows = self._rows(references, "reference")  # (p,)
        member = np.zeros(len(self.names), dtype=bool)  # (n,)
        member[reference_rows] = True  # (n,), True at the references
        scale = max(int(self.residues[reference_rows].sum()) / self.total_residues, 1.0 / self.reach)
        starts = self.offsets[rows]  # (q,)
        lengths = self.offsets[rows + 1] - starts  # (q,)
        total = int(lengths.sum())  # m_q
        owner = np.repeat(np.arange(len(rows)), lengths)  # (m_q,), each hit's query place in `query_order`
        # Each hit's index into the stored arrays: its offset within its query's run, plus that run's start.
        position = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths) + np.repeat(starts, lengths)  # (m_q,)
        counted = member[self.target[position]] & (self.evalue[position] * scale <= self.config.evalue_max)  # (m_q,)
        return owner[counted], position[counted], query_order  # (m_c,), (m_c,)

    def counted_hits(self, minimum_identity: float, reference_share: float) -> tuple[np.ndarray, np.ndarray]:
        """The table rows of the query and the target of each stored hit at `minimum_identity` or more that a search against a reference counts.

        The reference holds `reference_share` of the residues, so a hit counts when its e-value times the share (at least `1 / reach`)
        is at most the limit, the rule `_counted` applies to a subset. The rows index `names`.
        """
        if isinstance(minimum_identity, bool) or not isinstance(minimum_identity, (int, float)) or not 0.0 < minimum_identity <= 1.0:
            raise ValueError(f"minimum_identity must be in (0, 1], got {minimum_identity!r}")
        if isinstance(reference_share, bool) or not isinstance(reference_share, (int, float)) or not 0.0 < reference_share <= 1.0:
            raise ValueError(f"reference_share must be in (0, 1], got {reference_share!r}")
        scale = max(float(reference_share), 1.0 / self.reach)
        rows = np.repeat(np.arange(len(self.names), dtype=np.int64), np.diff(self.offsets))  # (m,) each stored hit's query row
        counted = (self.identity >= minimum_identity) & (self.evalue * scale <= self.config.evalue_max)  # (m,)
        return rows[counted], self.target[counted].astype(np.int64)  # (m_c,), (m_c,)

    def max_identity(self, queries: Mapping[str, str], references: Mapping[str, str]) -> dict[str, float]:
        """Each query's highest identity to any reference among the hits a search of these references counts, 0.0 with none.

        The answer of `max_identity_to_reference` for the same sets.
        """
        owner, position, query_order = self._counted(queries, references)  # owner, position: (m_c,)
        best = np.zeros(len(query_order), dtype=np.float64)  # (q,)
        np.maximum.at(best, owner, self.identity[position])  # best: (q,), in place
        return dict(zip(query_order, best.tolist(), strict=True))

    def matching(self, queries: Mapping[str, str], references: Mapping[str, str], *, minimum_identity: float) -> dict[str, tuple[str, ...]]:
        """The references each query is at `minimum_identity` or more to, sorted, and none for a query without one.

        The answer of `matching_references` for the same sets.
        """
        if isinstance(minimum_identity, bool) or not isinstance(minimum_identity, (int, float)) or not 0.0 < minimum_identity <= 1.0:
            raise ValueError(f"minimum_identity must be in (0, 1], got {minimum_identity!r}")
        owner, position, query_order = self._counted(queries, references)  # owner, position: (m_c,)
        close = self.identity[position] >= minimum_identity  # (m_c,)
        found: dict[str, set[str]] = {name: set() for name in query_order}
        for place, row in zip(owner[close].tolist(), self.target[position[close]].tolist(), strict=True):
            found[query_order[place]].add(self.names[row])
        return {name: tuple(sorted(targets)) for name, targets in found.items()}
