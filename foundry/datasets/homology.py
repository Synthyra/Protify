"""Homology search with MMseqs2: how close each query protein comes to a reference set.

A model can score well on test proteins close to its training set and poorly on novel ones, so
a test set is binned by each protein's highest identity to training. `max_identity_to_reference`
searches every query against the references with `mmseqs easy-search` and keeps, per query, the
highest identity (`fident`) among hits whose e-value is at most `evalue_max`. A query without
such a hit scores 0.0, the convention of sae_xgboost_ppi's identity bins.

The run reuses `foundry.datasets.mmseqs`: its FASTA writer, binary lookup, version query, and
command runner, and a stage named by a hash of both sequence sets, the MMseqs version, and every
setting that changes the hits.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sys
import time

from collections.abc import Collection, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

from .mmseqs import Command, CommandRunner, fasta_bytes, mmseqs_version, resolve_binary, run_command
from .stage import run_stage


QUERY_FASTA = "queries.fasta"
REFERENCE_FASTA = "references.fasta"
HITS_NAME = "hits.tsv"
# The columns sae_xgboost_ppi's search wrote, so its hit files parse here too.
HIT_COLUMNS = ("query", "target", "fident", "alnlen", "evalue", "bits")


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Everything that decides how one `mmseqs easy-search` runs and which of its hits count.

    Sensitivity, the cap on targets per query, and the e-value follow sae_xgboost_ppi's search.
    Coverage is optional because consumers disagree: sae_xgboost_ppi requires none, while
    contact_esmc6b's split audits require half of both sequences to align (`min_coverage=0.5`,
    `coverage_mode=0`), so a short local match cannot pass for homology there.
    """

    evalue_max: float = 1e-3
    sensitivity: float = 7.5
    min_coverage: float = 0.0
    coverage_mode: int = 0
    # Mode 3 asks MMseqs for each alignment's sequence identity along with its score and
    # coverage, as foundry's clustering and contact_esmc6b's searches do. None leaves the flag
    # out, as sae_xgboost_ppi's search always did, so MMseqs chooses.
    alignment_mode: int | None = 3
    max_sequences: int = 2_000
    threads: int = 8
    binary: str = "mmseqs"
    # None records whatever version runs; a value refuses any other.
    expected_version: str | None = None
    split_memory_limit: str | None = None

    def __post_init__(self) -> None:
        for name in ("evalue_max", "sensitivity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number, got {value!r}")
        coverage = self.min_coverage
        if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not 0.0 <= coverage <= 1.0:
            raise ValueError(f"min_coverage must be in [0, 1], got {coverage!r}")
        # The command carries two decimals; refuse a value that would be silently rounded.
        if round(coverage, 2) != coverage:
            raise ValueError(f"min_coverage must have at most two decimals, got {coverage!r}")
        for name in ("coverage_mode", "alignment_mode", "max_sequences", "threads"):
            value = getattr(self, name)
            if name == "alignment_mode" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.max_sequences < 1 or self.threads < 1:
            raise ValueError("max_sequences and threads must be positive")
        if not str(self.binary).strip():
            raise ValueError("binary must name an MMseqs executable")
        if self.split_memory_limit is not None and (
            not isinstance(self.split_memory_limit, str)
            or re.fullmatch(r"[1-9][0-9]*[KMG]", self.split_memory_limit) is None
        ):
            raise ValueError("split_memory_limit must be a positive integer with K, M, or G units")

    def result_parameters(self) -> dict[str, object]:
        """The settings that change which hits count; threads, path, and version check do not."""
        excluded = {"threads", "binary", "expected_version"}
        return {
            name: value for name, value in asdict(self).items()
            if name not in excluded and not (name == "split_memory_limit" and value is None)
        }


DEFAULT_SEARCH = SearchConfig()
# Where a split's searches are answered from: one MMseqs2 search each, or the one all-against-all search of `PairTable`, which keeps
# the hits up to `DEFAULT_REACH` times the e-value limit (foundry.datasets.pair_table).
SEARCH_EACH, ALL_PAIRS = "search", "all_pairs"
ANSWER_SOURCES = (SEARCH_EACH, ALL_PAIRS)
DEFAULT_REACH = 25.0


class AnswerSource(TypedDict):
    """Where a split's searches were answered from, at what e-value limit, and how far the one search reached."""

    source: str
    evalue_max: float
    reach: float


def answered_by(config: SearchConfig, *, reach: float = DEFAULT_REACH) -> AnswerSource:
    """The record a split keeps of having been measured from the table: where, at what e-value limit, and how far the search reached."""
    return {"source": ALL_PAIRS, "evalue_max": config.evalue_max, "reach": float(reach)}


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One alignment of a query to a reference, one row of `HIT_COLUMNS`."""

    query: str
    target: str
    # MMseqs `fident`: the fraction of identical residues in the alignment.
    identity: float
    alignment_length: int
    evalue: float
    bit_score: float


def search_command(
    queries: str | Path, references: str | Path, hits: str | Path, temporary: str | Path, config: SearchConfig
) -> Command:
    """`easy-search` of the query FASTA against the reference FASTA, writing `HIT_COLUMNS` as TSV."""
    alignment = () if config.alignment_mode is None else ("--alignment-mode", str(config.alignment_mode))
    memory = () if config.split_memory_limit is None else ("--split-memory-limit", config.split_memory_limit)
    return (
        str(config.binary),
        "easy-search",
        str(queries),
        str(references),
        str(hits),
        str(temporary),
        "--format-output",
        ",".join(HIT_COLUMNS),
        "-e",
        str(config.evalue_max),
        "-s",
        str(config.sensitivity),
        "-c",
        f"{config.min_coverage:.2f}",
        "--cov-mode",
        str(config.coverage_mode),
        *alignment,
        "--max-seqs",
        str(config.max_sequences),
        "--threads",
        str(config.threads),
        *memory,
    )


def parse_search_hits(lines: Iterable[str]) -> list[SearchHit]:
    """Tab-separated `HIT_COLUMNS` rows as hits, refused at the first malformed row."""
    hits: list[SearchHit] = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != len(HIT_COLUMNS):
            raise ValueError(f"MMseqs hit line {line_number} has {len(fields)} fields, expected {len(HIT_COLUMNS)}")
        query, target, identity, alignment_length, evalue, bit_score = fields
        if not query or not target:
            raise ValueError(f"MMseqs hit line {line_number} contains an empty identifier")
        try:
            hit = SearchHit(query, target, float(identity), int(alignment_length), float(evalue), float(bit_score))
        except ValueError as error:
            raise ValueError(f"MMseqs hit line {line_number} has a malformed number: {line!r}") from error
        if not 0.0 <= hit.identity <= 1.0 or hit.alignment_length < 1 or not hit.evalue >= 0.0:
            raise ValueError(f"MMseqs hit line {line_number} is out of range: {line!r}")
        hits.append(hit)
    return hits


def max_identity_to_reference(
    queries: Mapping[str, str],
    references: Mapping[str, str],
    directory: Path,
    config: SearchConfig = DEFAULT_SEARCH,
    *,
    runner: CommandRunner | None = None,
) -> dict[str, float]:
    """Each query's highest identity to any reference, or 0.0 when no hit reaches `evalue_max`.

    `queries` and `references` map identifiers to sequences. Identifiers become FASTA headers, so
    they carry no whitespace. An identifier naming both a query and a reference is searched like
    any other pair, so a protein in both sets reports its identity to itself. The search runs in
    a stage under `directory`, and a rerun on the same inputs reads the finished hits.
    """
    hits = search_hits(queries, references, directory, config, runner=runner)
    return _best_identity(hits, queries.keys(), config.evalue_max)


def matching_references(
    queries: Mapping[str, str],
    references: Mapping[str, str],
    directory: Path,
    config: SearchConfig = DEFAULT_SEARCH,
    *,
    minimum_identity: float,
    runner: CommandRunner | None = None,
) -> dict[str, tuple[str, ...]]:
    """The references each query is at `minimum_identity` or more to, sorted, and none for a query without one.

    `max_identity_to_reference` says how close each query comes, and this says to whom: a hit counts under the same rule,
    an e-value at most `evalue_max`, so a query names a reference exactly when its highest identity reaches
    `minimum_identity` through that reference's hit. Both run one staged search, so the second on the same inputs reads the
    first's hits.
    """
    if isinstance(minimum_identity, bool) or not isinstance(minimum_identity, (int, float)) or not 0.0 < minimum_identity <= 1.0:
        raise ValueError(f"minimum_identity must be in (0, 1], got {minimum_identity!r}")
    found: dict[str, set[str]] = {query: set() for query in queries}
    for hit in search_hits(queries, references, directory, config, runner=runner):
        if hit.evalue <= config.evalue_max and hit.identity >= minimum_identity:
            found[hit.query].add(hit.target)
    return {query: tuple(sorted(targets)) for query, targets in found.items()}


def search_hits(
    queries: Mapping[str, str],
    references: Mapping[str, str],
    directory: Path,
    config: SearchConfig = DEFAULT_SEARCH,
    *,
    runner: CommandRunner | None = None,
) -> list[SearchHit]:
    """Every hit MMseqs reports for the queries against the references, in its output order.

    The staged search behind `max_identity_to_reference`, for a consumer that needs the hits
    themselves, such as sae_xgboost_ppi's comparison of identity with embedding similarity. MMseqs
    already drops hits above `evalue_max`; nothing else is filtered, self-hits included. A hit
    naming a sequence outside the search means the hits do not belong to these inputs, so it is
    refused rather than dropped.
    """
    with search_file(queries, references, directory, config, runner=runner).open(encoding="utf-8") as handle:
        hits = parse_search_hits(handle)
    for hit in hits:
        if hit.query not in queries or hit.target not in references:
            raise ValueError(f"MMseqs hit {hit.query!r} to {hit.target!r} names a sequence outside this search")
    return hits


def search_file(
    queries: Mapping[str, str],
    references: Mapping[str, str],
    directory: Path,
    config: SearchConfig = DEFAULT_SEARCH,
    *,
    runner: CommandRunner | None = None,
) -> Path:
    """The TSV of hits (`HIT_COLUMNS`) of the staged search, run once per input, for a consumer that reads a large one as a stream.

    `search_hits` parses it all into objects, which is fine for thousands of hits and costly for tens of millions.
    """
    if not queries or not references:
        raise ValueError("a search needs at least one query and one reference")
    execute = runner or run_command
    binary = resolve_binary(config.binary)
    version = mmseqs_version(binary, runner=execute)
    if config.expected_version is not None and config.expected_version not in version.splitlines():
        raise RuntimeError(f"Expected MMseqs {config.expected_version}, received {version.splitlines()[0]!r}")

    query_fasta, reference_fasta = fasta_bytes(queries), fasta_bytes(references)
    key = hashlib.sha256(
        json.dumps(
            {
                "queries_sha256": hashlib.sha256(query_fasta).hexdigest(),
                "references_sha256": hashlib.sha256(reference_fasta).hexdigest(),
                "mmseqs_version": version,
                "parameters": config.result_parameters(),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]

    def build(stage: Path) -> None:
        started = time.monotonic()
        label = f"MMseqs2 search of {len(queries):,} queries against {len(references):,} references"
        print(f"{label}: started", file=sys.stderr, flush=True)
        (stage / QUERY_FASTA).write_bytes(query_fasta)
        (stage / REFERENCE_FASTA).write_bytes(reference_fasta)
        execute(search_command(stage / QUERY_FASTA, stage / REFERENCE_FASTA, stage / HITS_NAME, stage / "tmp", config))
        shutil.rmtree(stage / "tmp", ignore_errors=True)
        print(f"{label}: finished in {time.monotonic() - started:.1f} s", file=sys.stderr, flush=True)

    return run_stage(directory / key, (QUERY_FASTA, REFERENCE_FASTA, HITS_NAME), build) / HITS_NAME


def _best_identity(hits: Iterable[SearchHit], queries: Collection[str], evalue_max: float) -> dict[str, float]:
    """The highest identity per query over hits at or below `evalue_max`, in `queries` order."""
    best: dict[str, float] = {}
    for hit in hits:
        if hit.evalue <= evalue_max:
            best[hit.query] = max(best.get(hit.query, 0.0), hit.identity)
    return {query: best.get(query, 0.0) for query in queries}
