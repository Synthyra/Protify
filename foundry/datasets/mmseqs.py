"""Homology clusters from MMseqs2: the commands, a run of them, and their checked output.

A cluster is the unit a homology-disjoint split assigns, so every member of a cluster lands in
the same split. Identity and coverage are required because consumers disagree, from 0.30 to
0.95 identity and 0.5 or 0.8 coverage, and each value changes which proteins count as
homologs. Everything else defaults to the settings annotation_vocabulary and latent_sep share:
local alignment scored by identity (`--alignment-mode 3`), coverage of both sequences
(`--cov-mode 0`), greedy set cover in a single step, then reassignment of each member to the
representative it matches best.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
import time

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from .stage import run_stage


Command = tuple[str, ...]
CommandRunner = Callable[[Command], subprocess.CompletedProcess[str]]
# Representative identifier to its members, the representative included, both sorted.
ClusterMap = dict[str, tuple[str, ...]]

FASTA_NAME = "sequences.fasta"
SEQUENCE_DATABASE = "sequence_db"
CLUSTER_DATABASE = "cluster_db"
MEMBERSHIPS_NAME = "clusters.tsv"
DIAGNOSTIC_CHARACTERS = 8_000
LINEAR_MIN_IDENTITY = 0.8


@dataclass(frozen=True, slots=True)
class MMseqsConfig:
    """Everything that decides how one `mmseqs cluster` run is invoked."""

    min_sequence_identity: float
    min_coverage: float
    coverage_mode: int = 0
    alignment_mode: int = 3
    cluster_mode: int = 0
    single_step_clustering: bool = True
    cluster_reassign: bool = True
    sensitivity: float = 7.5
    max_sequences: int = 20_000
    threads: int = 8
    split_memory_limit: str | None = None
    binary: str = "mmseqs"
    # Run `linclust`, MMseqs2's linear-time clustering, instead of `cluster`. `cluster` compares
    # each sequence with up to `max_sequences` others, which never finishes on a set where
    # thousands of sequences are close variants of one another (a display library's scaffold); `linclust`
    # groups by shared k-mers and misses distant pairs, so it is for identity of LINEAR_MIN_IDENTITY or
    # more. Sensitivity, `max_sequences`, `single_step_clustering`, and `cluster_reassign` do not apply to it.
    linear: bool = False
    # None records whatever version runs; a value refuses any other.
    expected_version: str | None = None

    def __post_init__(self) -> None:
        for name in ("min_sequence_identity", "min_coverage"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value!r}")
            # The command carries two decimals; refuse a value that would be silently rounded.
            if round(value, 2) != value:
                raise ValueError(f"{name} must have at most two decimals, got {value!r}")
        for name in ("coverage_mode", "alignment_mode", "cluster_mode", "max_sequences", "threads"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int):
                raise TypeError(f"{name} must be an integer")
        for name in ("single_step_clustering", "cluster_reassign", "linear"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be True or False")
        if isinstance(self.sensitivity, bool) or not math.isfinite(self.sensitivity) or self.sensitivity <= 0:
            raise ValueError("sensitivity must be positive")
        if self.linear and self.min_sequence_identity < LINEAR_MIN_IDENTITY:
            raise ValueError(f"linclust misses distant pairs; use identity of {LINEAR_MIN_IDENTITY} or more, got {self.min_sequence_identity}")
        if self.max_sequences < 1 or self.threads < 1:
            raise ValueError("max_sequences and threads must be positive")
        if self.split_memory_limit is not None and not self.split_memory_limit.strip():
            raise ValueError("split_memory_limit must be None or a size such as '450G'")
        if not str(self.binary).strip():
            raise ValueError("binary must name an MMseqs executable")

    def result_parameters(self) -> dict[str, object]:
        """The settings that change which clusters come out; threads, memory, and path do not."""
        excluded = {"threads", "split_memory_limit", "binary", "expected_version"}
        # A stage made before `linear` existed keeps its name, so it is recorded only when set.
        if not self.linear:
            excluded.add("linear")
        return {name: value for name, value in asdict(self).items() if name not in excluded}


@dataclass(frozen=True, slots=True)
class Clustering:
    """Clusters, each member's representative, and how the clusters were produced."""

    clusters: ClusterMap
    representative_of: dict[str, str]
    origin: dict[str, object]


def createdb_command(fasta: str | Path, database: str | Path, config: MMseqsConfig) -> Command:
    return (str(config.binary), "createdb", str(fasta), str(database))


def cluster_command(
    database: str | Path, clusters: str | Path, temporary: str | Path, config: MMseqsConfig
) -> Command:
    head = (
        str(config.binary),
        "linclust" if config.linear else "cluster",
        str(database),
        str(clusters),
        str(temporary),
        "--min-seq-id",
        f"{config.min_sequence_identity:.2f}",
        "-c",
        f"{config.min_coverage:.2f}",
        "--cov-mode",
        str(config.coverage_mode),
        "--alignment-mode",
        str(config.alignment_mode),
        "--cluster-mode",
        str(config.cluster_mode),
    )
    reassign = ("--cluster-reassign", str(int(config.cluster_reassign)))
    threads = ("--threads", str(config.threads))
    if config.linear:
        command = (*head, *threads)
    else:
        command = (
            *head,
            "--single-step-clustering",
            str(int(config.single_step_clustering)),
            *reassign,
            "-s",
            str(config.sensitivity),
            "--max-seqs",
            str(config.max_sequences),
            *threads,
        )
    if config.split_memory_limit is None:
        return command
    return (*command, "--split-memory-limit", config.split_memory_limit)


def createtsv_command(
    database: str | Path, clusters: str | Path, memberships: str | Path, config: MMseqsConfig
) -> Command:
    """Representative and member, one row per member, for a database clustered against itself."""
    return (str(config.binary), "createtsv", str(database), str(database), str(clusters), str(memberships))


def parse_memberships(lines: Iterable[str], expected: Collection[str]) -> ClusterMap:
    """`createtsv` rows as clusters, checked to hold each of `expected` exactly once.

    Every representative must be one of `expected` and a member of its own cluster, so a
    truncated or foreign TSV fails here rather than yielding a split with proteins missing.
    """
    expected_set = set(expected)
    members_by_cluster: dict[str, list[str]] = {}
    cluster_of: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"MMseqs membership line {line_number} has {len(fields)} fields, expected 2")
        representative, member = fields
        if not representative or not member:
            raise ValueError(f"MMseqs membership line {line_number} contains an empty identifier")
        if representative not in expected_set:
            raise ValueError(f"MMseqs representative {representative!r} is not a source FASTA accession")
        if member in cluster_of:
            raise ValueError(f"Accession {member!r} occurs more than once in MMseqs memberships")
        cluster_of[member] = representative
        members_by_cluster.setdefault(representative, []).append(member)

    if not members_by_cluster:
        raise ValueError("MMseqs memberships are empty")
    missing = sorted(expected_set - set(cluster_of))
    unexpected = sorted(set(cluster_of) - expected_set)
    if missing or unexpected:
        raise ValueError(
            "MMseqs memberships do not match expected accessions: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    for representative, members in members_by_cluster.items():
        if representative not in members:
            raise ValueError(f"MMseqs representative {representative!r} lacks a self-membership row")
    return {representative: tuple(sorted(members_by_cluster[representative])) for representative in sorted(members_by_cluster)}


def cluster_sequences(
    sequences: Mapping[str, str],
    directory: Path,
    config: MMseqsConfig,
    *,
    runner: CommandRunner | None = None,
) -> Clustering:
    """Cluster `sequences`, keyed by identifier, in a stage under `directory`.

    The stage is named by a hash of the sequences, the MMseqs version, and every parameter
    that changes the result, so a rerun on the same inputs reuses the finished stage and a
    run on changed inputs never does.
    """
    execute = runner or run_command
    binary = resolve_binary(config.binary)
    version = mmseqs_version(binary, runner=execute)
    if config.expected_version is not None and config.expected_version not in version.splitlines():
        raise RuntimeError(f"Expected MMseqs {config.expected_version}, received {version.splitlines()[0]!r}")

    fasta = fasta_bytes(sequences)
    sequences_sha256 = hashlib.sha256(fasta).hexdigest()
    key = hashlib.sha256(
        json.dumps(
            {"sequences_sha256": sequences_sha256, "mmseqs_version": version, "parameters": config.result_parameters()},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    commands = _commands(Path("."), config)
    label = f"MMseqs2 {'linclust' if config.linear else 'cluster'} at {config.min_sequence_identity:.2f} identity"
    # Set when this call runs MMseqs, so a reused stage reports nothing.
    started: float | None = None

    def build(stage: Path) -> None:
        nonlocal started
        started = time.monotonic()
        print(f"{label}: clustering {len(sequences):,} sequences", file=sys.stderr, flush=True)
        (stage / FASTA_NAME).write_bytes(fasta)
        for command in _commands(stage, config):
            execute(command)
        shutil.rmtree(stage / "tmp", ignore_errors=True)

    stage = run_stage(directory / key, (FASTA_NAME, SEQUENCE_DATABASE, CLUSTER_DATABASE, MEMBERSHIPS_NAME), build)
    memberships = stage / MEMBERSHIPS_NAME
    with memberships.open(encoding="utf-8") as handle:
        clusters = parse_memberships(handle, sequences.keys())
    if started is not None:
        print(
            f"{label}: {len(clusters):,} clusters of {len(sequences):,} sequences in {time.monotonic() - started:.1f} s",
            file=sys.stderr, flush=True,
        )
    origin: dict[str, object] = {
        "tool": "MMseqs2",
        "version": version,
        "binary_sha256": _sha256_file(binary),
        "parameters": config.result_parameters(),
        "commands": [list(command) for command in commands],
        "sequence_count": len(sequences),
        "sequences_sha256": sequences_sha256,
        "memberships_sha256": _sha256_file(memberships),
        "stage": key,
    }
    representative_of = {member: representative for representative, members in clusters.items() for member in members}
    return Clustering(clusters, representative_of, origin)


def fasta_bytes(sequences: Mapping[str, str]) -> bytes:
    """One record per sequence, sorted by identifier, so equal inputs give equal bytes."""
    if not sequences:
        raise ValueError("no sequences to cluster")
    lines: list[str] = []
    for identifier in sorted(sequences):
        sequence = sequences[identifier]
        if not identifier or any(character.isspace() for character in identifier) or identifier.startswith(">"):
            raise ValueError(f"sequence identifier {identifier!r} must be non-empty, without whitespace or '>'")
        if not sequence or any(character.isspace() for character in sequence):
            raise ValueError(f"sequence for {identifier!r} must be non-empty and without whitespace")
        lines += [f">{identifier}", sequence]
    return ("\n".join(lines) + "\n").encode("ascii")


def mmseqs_version(binary: str | Path, *, runner: CommandRunner | None = None) -> str:
    """The version MMseqs reports: a release such as `18-8cc5c`, or a commit for a source build."""
    completed = (runner or run_command)((str(binary), "version"))
    lines = [line.strip() for text in (completed.stdout or "", completed.stderr or "") for line in text.splitlines()]
    version = "\n".join(line for line in lines if line)
    if not version:
        raise RuntimeError("MMseqs version command returned no output")
    return version


def resolve_binary(binary: str | Path) -> Path:
    """An explicit path to the executable, or a name found on PATH."""
    candidate = Path(binary).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"MMseqs binary not found: {resolved}")
        return resolved
    located = shutil.which(str(binary))
    if located is None:
        raise FileNotFoundError(f"MMseqs binary not found on PATH: {binary}")
    return Path(located).resolve()


def run_command(command: Command) -> subprocess.CompletedProcess[str]:
    """Run one command without a shell; on failure, raise with the end of its output."""
    completed = subprocess.run(command, check=False, capture_output=True, text=True)  # noqa: S603
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"MMseqs command failed with exit code {completed.returncode}: {diagnostic[-DIAGNOSTIC_CHARACTERS:]}"
        )
    return completed


def _commands(stage: Path, config: MMseqsConfig) -> tuple[Command, Command, Command]:
    database, clusters = stage / SEQUENCE_DATABASE, stage / CLUSTER_DATABASE
    return (
        createdb_command(stage / FASTA_NAME, database, config),
        cluster_command(database, clusters, stage / "tmp", config),
        createtsv_command(database, clusters, stage / MEMBERSHIPS_NAME, config),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
