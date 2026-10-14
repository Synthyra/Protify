"""MMseqs2 `easy-cluster` and CD-HIT, run from the command line or in Docker, for the builders whose splits came from them.

This is not a second clustering API. `foundry.datasets.mmseqs` is the one to build a split on:
staged, version-pinned, and checked. This module keeps the clusterer that DatasetDev's builders,
confounders, Atlas, and synth each held a copy of, because the splits they published came from
these exact commands and a cluster membership must not change behind a published split. It
writes a FASTA beside its results, runs the tool on the machine when MMseqs2 (or, if asked,
CD-HIT) is on the PATH and in Docker otherwise, and reads the report.

The copies differed in a few places, and each difference is a field of `ClusterRunOptions`. Its
defaults reproduce DatasetDev's, confounders', and synth's clusterer; Atlas's sets every field
(see the README for the call). `psutil` sizes the run, and torch is optional: it is asked only
whether CUDA is visible, and only while `gpu` is left unset.

Every path must lie under the working directory, which Docker mounts. Results are cached by
name: a finished report is read again, never recomputed, whatever the sequences now are.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import shutil
import subprocess
import psutil

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Literal
from tqdm.auto import tqdm


__all__ = [
    "ClusterRunOptions",
    "ClusteringConfig",
    "ProteinSequenceClusterer",
    "StaleFastaError",
    "cluster_sequences",
    "cluster_sequences_cdhit",
    "cluster_sequences_mmseqs",
    "cuda_is_visible",
]

# What stands in for `subprocess.run` and `shutil.which`, so a test records commands and writes
# the reports itself.
ProcessRunner = Callable[..., subprocess.CompletedProcess[Any]]
ExecutableFinder = Callable[[str], str | None]
# A cluster's key to its members, in report order. `defaultdict(list)`, as every copy returned.
ClusterMembers = dict[str, list[str]]

MMSEQS_IMAGE = "ghcr.io/soedinglab/mmseqs2"
CDHIT_IMAGE = "cd-hit"
# MMseqs2's `-s`, and `--cov-mode 0`: coverage of both sequences.
MMSEQS_SENSITIVITY = "7.5"
# CD-HIT 4.8.1 rejects `-l` below its own minimum, so its silent `-l 10` filter cannot be turned off.
# A sequence shorter than this is dropped by that filter; `cluster_sequences_cdhit` keeps those out of CD-HIT.
CDHIT_KEEP_LENGTH = 11


class StaleFastaError(ValueError, AssertionError):
    """A cached FASTA no longer matches the sequences to cluster.

    An `AssertionError` as well because the Atlas copy raised one through `assert`, and callers catch that.
    """


@dataclass(frozen=True)
class ClusterRunOptions:
    """Where the copies of this clusterer differ, each a named choice.

    The defaults are DatasetDev's, confounders', and synth's. The Atlas copy differs in every field.

    gpu: `None` adds `--gpu 1` (and `--gpus all` to Docker) when torch reports a CUDA device; `False`
        never does; `True` always does.
    check_cached_fasta: `True` refuses a cached FASTA whose record count differs from the
        sequences, with `StaleFastaError`; `False` reuses it whatever it holds.
    stale_fasta_remedy: how the error says to recover, after "Delete the FASTA or ".
    cdhit_on_path: `True` runs a `cd-hit` found on the PATH and falls back to Docker; `False` always uses Docker.
    cdhit_cluster_key: `"cluster_number"` keys a CD-HIT cluster by its number in the report (the `>Cluster 0`
        line) and `"representative"` by its representative's identifier (the member marked `*`).
    disable_mmseqs_shuffle: `True` adds `--shuffle 0`, which easy-cluster's default cascade needs to run with the
        default createdb mode and which makes the input order deterministic.
    """

    gpu: bool | None = None
    check_cached_fasta: bool = False
    stale_fasta_remedy: str = ""
    cdhit_on_path: bool = False
    cdhit_cluster_key: Literal["cluster_number", "representative"] = "cluster_number"
    disable_mmseqs_shuffle: bool = False

    def __post_init__(self) -> None:
        if self.cdhit_cluster_key not in ("cluster_number", "representative"):
            raise ValueError(f"cdhit_cluster_key must be 'cluster_number' or 'representative', got {self.cdhit_cluster_key!r}")


@dataclass
class ClusteringConfig:
    """One clustering run: the method, its thresholds, and where its files go.

    `method` is `mmseqs2`, `cd-hit`, or `none` (every sequence its own cluster). `cluster_percentage` is
    the identity as a fraction, `coverage` MMseqs2's `-c` and CD-HIT's `-aL`, `n` CD-HIT's word size, and
    `memory_percentage` the share of this machine's memory the tool may use. `base_path` holds the FASTA
    and the reports, and `identifier` and `cache_suffix` name them.
    """

    method: str = "mmseqs2"
    cluster_percentage: float = 0.5
    coverage: float = 0.8
    n: int = 3
    memory_percentage: float = 0.5
    cache_suffix: str = ""
    base_path: str = "data"
    identifier: str = "unknown"
    options: ClusterRunOptions = field(default_factory=ClusterRunOptions)


def cuda_is_visible() -> bool:
    """Whether torch is installed and sees a CUDA device. Torch is optional here, so its absence is False."""
    if importlib.util.find_spec("torch") is None:
        return False
    return bool(import_module("torch").cuda.is_available())


class ProteinSequenceClusterer:
    """Cluster a mapping of identifier to sequence, returning each cluster's key to its member identifiers."""

    def __init__(
        self,
        config: ClusteringConfig,
        runner: ProcessRunner | None = None,
        which: ExecutableFinder | None = None,
    ) -> None:
        self.config = config
        self.runner = runner
        self.which = which

    def _run(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        runner = subprocess.run if self.runner is None else self.runner
        return runner(command, **kwargs)

    def _find(self, executable: str) -> str | None:
        finder = shutil.which if self.which is None else self.which
        return finder(executable)

    def _run_docker_command(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        """Run `cmd`, and when it fails run it again: as is on Windows, under `sudo` elsewhere."""
        with contextlib.suppress(subprocess.CalledProcessError, OSError):
            completed = self._run(cmd, **kwargs)
            if completed.returncode == 0:
                return completed

        if os.name == "nt":
            return self._run(cmd, **kwargs)

        return self._run(["sudo"] + cmd, **kwargs)

    def _validate_paths_under_cwd(self, *paths: str) -> None:
        cwd = os.path.abspath(os.getcwd())
        for p in paths:
            ap = os.path.abspath(p)
            if not (ap == cwd or ap.startswith(cwd + os.sep)):
                raise ValueError(
                    "Path must be under current working directory for docker volume mount. "
                    f"cwd={cwd!r}, path={ap!r}"
                )

    def _path_in_container(self, local_path: str) -> str:
        self._validate_paths_under_cwd(local_path)
        rel = os.path.relpath(
            os.path.abspath(local_path),
            start=os.path.abspath(os.getcwd()),
        )
        return rel.replace(os.sep, "/")

    def _use_gpu(self) -> bool:
        gpu = self.config.options.gpu
        return cuda_is_visible() if gpu is None else gpu

    def _write_fasta_if_missing(self, seq_dict: dict[str, str]) -> str:
        """The FASTA path, written on first use and, with `check_cached_fasta`, checked on every later one."""
        os.makedirs(self.config.base_path, exist_ok=True)
        fasta_path = os.path.join(
            self.config.base_path,
            f"{self.config.identifier}{self.config.cache_suffix}.fasta",
        )
        if os.path.exists(fasta_path):
            if self.config.options.check_cached_fasta:
                self._refuse_stale_fasta(fasta_path, len(seq_dict))
            return fasta_path
        # Line feeds only: the Linux container reads a Windows carriage return as part of every identifier and sequence.
        with open(fasta_path, "w", newline="\n") as f:
            for seq_id, seq in seq_dict.items():
                f.write(f">{seq_id}\n{seq}\n")
        return fasta_path

    def _refuse_stale_fasta(self, fasta_path: str, record_count: int) -> None:
        """Raise when the cached FASTA's header count is not `record_count`.

        The FASTA is reused when present, so a caller whose sequences grew or shrank since the last run
        would otherwise cluster the wrong input for hours.
        """
        with open(fasta_path) as f:
            cached_count = sum(1 for line in f if line.startswith(">"))
        if cached_count == record_count:
            return
        remedy = self.config.options.stale_fasta_remedy
        recovery = f"Delete the FASTA or {remedy}." if remedy else "Delete the FASTA."
        raise StaleFastaError(
            f"Cached FASTA at {fasta_path!r} has {cached_count} records but "
            f"the current seq_dict has {record_count}. The input pool "
            "changed since the FASTA was written (e.g. you added a length "
            "filter, canonicalization shrank the pool, or a source seq "
            f"dict drifted). {recovery}"
        )

    def _compute_resources(self) -> tuple[int, int]:
        """Threads (all but four cores, or one) and megabytes of memory (`memory_percentage` of the machine's)."""
        num_cpu = os.cpu_count() - 4 if os.cpu_count() and os.cpu_count() > 4 else 1
        memory_max = int(
            self.config.memory_percentage * psutil.virtual_memory().total / 1024 / 1024
        )
        return num_cpu, memory_max

    def cdhit(self, seq_dict: dict[str, str]) -> ClusterMembers:
        fasta_path = self._write_fasta_if_missing(seq_dict)
        num_cpu, memory_max = self._compute_resources()
        output_path = os.path.join(
            self.config.base_path,
            f"output_{self.config.identifier}_{self.config.cluster_percentage}{self.config.cache_suffix}",
        )
        cluster_file = f"{output_path}.clstr"
        self._validate_paths_under_cwd(fasta_path, output_path, cluster_file)

        if not os.path.exists(cluster_file):
            native = self.config.options.cdhit_on_path and self._find("cd-hit") is not None
            if native:
                in_fasta = fasta_path
                out_path = output_path
            else:
                in_fasta = self._path_in_container(fasta_path)
                out_path = self._path_in_container(output_path)

            cdhit_args = [
                "cd-hit",
                "-i", in_fasta,
                "-o", out_path,
                "-d", "0",
                "-c", str(self.config.cluster_percentage),
                "-n", str(self.config.n),
                "-T", str(num_cpu),
                "-M", str(memory_max),
                "-aL", str(self.config.coverage),
            ]

            if native:
                print("Running CD-HIT natively...")
                self._run(cdhit_args, check=True)
            else:
                if self.config.options.cdhit_on_path:
                    print("Running CD-HIT via Docker...")
                docker_cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{os.getcwd()}:/data",
                    "-w", "/data",
                    CDHIT_IMAGE,
                ]
                docker_cmd.extend(cdhit_args)
                self._run_docker_command(docker_cmd, check=True)

        return self._read_cdhit_clusters(cluster_file)

    def _read_cdhit_clusters(self, cluster_file: str) -> ClusterMembers:
        """A CD-HIT `.clstr` report as clusters.

        The report lists `>Cluster 0`, then one member per line, the representative marked with a
        trailing `*`:

            >Cluster 0
            0       300aa, >P00930... *
            1       298aa, >P00931... at 95.00%

        Keys are cluster numbers or representatives, as `cdhit_cluster_key` says.
        """
        members_of: ClusterMembers = defaultdict(list)
        representative_of: dict[str, str] = {}
        with open(cluster_file) as f:
            current_cluster = None
            for line in tqdm(f, desc="Reading CD-HIT cluster file"):
                stripped = line.rstrip()
                if stripped.startswith(">"):
                    current_cluster = stripped.split("Cluster")[1].strip()
                else:
                    seq_id = stripped.split(">")[1].split("...")[0].strip()
                    members_of[current_cluster].append(seq_id)
                    if stripped.endswith("*"):
                        representative_of[current_cluster] = seq_id

        if self.config.options.cdhit_cluster_key == "cluster_number":
            return members_of

        by_representative: ClusterMembers = defaultdict(list)
        for cluster_number, members in members_of.items():
            if cluster_number not in representative_of:
                raise ValueError(
                    f"CD-HIT cluster {cluster_number!r} has no rep marker (*) in "
                    f"{cluster_file}; clstr file is malformed."
                )
            by_representative[representative_of[cluster_number]] = members
        return by_representative

    def mmseqs2(self, seq_dict: dict[str, str]) -> ClusterMembers:
        fasta_path = self._write_fasta_if_missing(seq_dict)
        num_cpu, memory_max = self._compute_resources()
        output_prefix = os.path.join(
            self.config.base_path,
            f"mmseqs_{self.config.identifier}_{self.config.cluster_percentage}{self.config.cache_suffix}",
        )
        cluster_tsv = f"{output_prefix}_cluster.tsv"
        tmp_dir = os.path.join(
            self.config.base_path,
            f"tmp_{self.config.identifier}{self.config.cache_suffix}",
        )
        os.makedirs(tmp_dir, exist_ok=True)
        self._validate_paths_under_cwd(fasta_path, cluster_tsv, tmp_dir, self.config.base_path)

        if not os.path.exists(cluster_tsv):
            use_gpu = self._use_gpu()
            native = self._find("mmseqs") is not None
            if native:
                in_fasta = fasta_path
                in_prefix = output_prefix
                in_tmp = tmp_dir
            else:
                in_fasta = self._path_in_container(fasta_path)
                in_prefix = self._path_in_container(output_prefix)
                in_tmp = self._path_in_container(tmp_dir)

            mmseqs_args = [
                "easy-cluster",
                in_fasta,
                in_prefix,
                in_tmp,
                "--min-seq-id",
                str(self.config.cluster_percentage),
                "-c",
                str(self.config.coverage),
                "--threads",
                str(num_cpu),
                "--cov-mode",
                "0",
                "-s",
                MMSEQS_SENSITIVITY,
                "--split-memory-limit",
                f"{memory_max}M",
            ]
            if self.config.options.disable_mmseqs_shuffle:
                mmseqs_args.extend(["--shuffle", "0"])
            if use_gpu:
                mmseqs_args.extend(["--gpu", "1"])

            if native:
                print("Running MMseqs2 natively...")
                self._run(["mmseqs"] + mmseqs_args, check=True)
            else:
                print("Running MMseqs2 via Docker...")
                docker_cmd = ["docker", "run", "--rm", "-v", f"{os.getcwd()}:/app", "-w", "/app"]
                if use_gpu:
                    docker_cmd.extend(["--gpus", "all"])
                docker_cmd.append(MMSEQS_IMAGE)
                docker_cmd.extend(mmseqs_args)
                self._run_docker_command(docker_cmd, check=True)

        cluster_dict: ClusterMembers = defaultdict(list)
        with open(cluster_tsv) as f:
            for line in tqdm(f, desc="Reading MMseqs2 cluster file"):
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    rep_id, member_id = parts
                    cluster_dict[rep_id].append(member_id)
        return cluster_dict

    def none(self, seq_dict: dict[str, str]) -> ClusterMembers:
        cluster_dict: ClusterMembers = defaultdict(list)
        for seq_id in seq_dict:
            cluster_dict[seq_id].append(seq_id)
        return cluster_dict

    def __call__(self, seq_dict: dict[str, str]) -> ClusterMembers:
        method = self.config.method.lower()
        if method == "cd-hit":
            return self.cdhit(seq_dict)
        if method == "mmseqs2":
            return self.mmseqs2(seq_dict)
        if method == "none":
            return self.none(seq_dict)
        raise ValueError(f"Unknown clustering method: {self.config.method}")


def cluster_sequences(
    seq_dict: dict[str, str],
    method: str = "mmseqs2",
    cluster_percentage: float = 0.5,
    coverage: float = 0.8,
    n: int = 3,
    memory_percentage: float = 0.5,
    identifier: str = "unknown",
    cache_suffix: str = "",
    base_path: str = "data",
    options: ClusterRunOptions | None = None,
) -> ClusterMembers:
    """Cluster `seq_dict` as `ProteinSequenceClusterer` does, from the settings of one `ClusteringConfig`.

    `options` left as `None` is `ClusterRunOptions()`.
    """
    config = ClusteringConfig(
        method=method,
        cluster_percentage=cluster_percentage,
        coverage=coverage,
        n=n,
        memory_percentage=memory_percentage,
        cache_suffix=cache_suffix,
        base_path=base_path,
        identifier=identifier,
        options=ClusterRunOptions() if options is None else options,
    )
    return ProteinSequenceClusterer(config)(seq_dict)


def _flatten(cluster_dict: ClusterMembers) -> dict[str, str]:
    """Each member's cluster key."""
    return {member: key for key, members in cluster_dict.items() for member in members}


def _split_work_dir(work_dir: str) -> tuple[str, str]:
    """`work_dir` as the base path and identifier of a run: its parent (`.` for none) and its last component."""
    base_path, identifier = os.path.split(work_dir.rstrip(os.sep))
    return base_path or ".", identifier or "cluster"


def cluster_sequences_mmseqs(
    id_to_seq: dict[str, str],
    work_dir: str,
    identity: float,
    coverage: float,
    options: ClusterRunOptions | None = None,
) -> dict[str, str]:
    """MMseqs2 cluster `id_to_seq` and return a flat `member_id -> cluster_rep` map."""
    base_path, identifier = _split_work_dir(work_dir)
    cluster_dict = cluster_sequences(
        seq_dict=id_to_seq,
        method="mmseqs2",
        cluster_percentage=float(identity),
        coverage=float(coverage),
        identifier=identifier,
        base_path=base_path,
        options=options,
    )
    return _flatten(cluster_dict)


def cluster_sequences_cdhit(
    id_to_seq: dict[str, str],
    work_dir: str,
    identity: float,
    n: int,
    coverage: float = 0.0,
    min_length: int = CDHIT_KEEP_LENGTH,
    options: ClusterRunOptions | None = None,
) -> dict[str, str]:
    """CD-HIT cluster `id_to_seq` and return a flat `member_id -> cluster key` map.

    `identity` is `-c` (0.4 for 40% identity), `n` is `-n`, the word size (CD-HIT's guide gives 2 at 0.4 to
    0.5 and 5 at 0.7 to 1.0), and `coverage` is `-aL`, the alignment coverage of the longer sequence;
    0.0 with CD-HIT's default `-G 1` global alignment is the Bernett-style redundancy reduction.

    A sequence shorter than `min_length` never reaches CD-HIT, whose `-l 10` filter would drop it silently:
    it becomes a singleton cluster, its own key, which cannot share a cluster with anyone and so is safe for
    a split. The default of 11 is CD-HIT's own keep threshold, so no real protein is lost. The key of the
    other sequences is a representative's identifier when `options.cdhit_cluster_key` is `"representative"`
    and a cluster number otherwise.
    """
    long_seqs = {p: s for p, s in id_to_seq.items() if len(s) >= min_length}
    short_seqs = {p: s for p, s in id_to_seq.items() if len(s) < min_length}

    flat: dict[str, str] = {}
    if long_seqs:
        base_path, identifier = _split_work_dir(work_dir)
        cluster_dict = cluster_sequences(
            seq_dict=long_seqs,
            method="cd-hit",
            cluster_percentage=float(identity),
            coverage=float(coverage),
            n=int(n),
            identifier=identifier,
            base_path=base_path,
            options=options,
        )
        flat.update(_flatten(cluster_dict))

    for pid in short_seqs:
        flat[pid] = pid

    return flat
