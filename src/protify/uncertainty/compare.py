"""The ablation's answers and tables across finished cells.

`compare_runs` reads every `report.json` a glob pattern matches under a runs directory, one per
finished cell, answers both questions of the experiment README's pre-registered decision rules
(`verdicts.py`), and writes every file `analysis_tables.write_outputs` names. Question 1 reads
each cell's seed-mean test primary metric on its test rows and on stratified bootstrap
replicates of them (`featurization.py`), and each seed's own, for the seed spread; question 2
reads scheme T (`exchangeable.py`), with scheme V, the report's own numbers, beside it.
Question 2 reads the ablation's heads, `projected` and `sngp`, alone. A comparison holding a
head only the single-probe experiment trains (`single_probe.SINGLE_PROBE_HEADS`) also applies
that experiment's rules to every head and writes `single_probe.json` and `single_probe.md`.

A cell is named by its tags' arm (its cell id when it has none), its dataset, and its head. Two
cells with one name are refused, since a pattern that also matched tuning cells would mix them
in. Every cell of a dataset must have scored the same test rows, by digest and label.

Every draw depends only on the seed, the dataset's name, and its test labels
(`resampling.py`), so cells are analyzed in separate processes, heaviest first, each with one
BLAS thread, and the output is the same for any number of worker processes. `workers=0` runs in
this process instead, for tests and debugging, where BLAS threads can move the last digits.

Shapes: `B` bootstrap replicates, `R` scheme T repeats, `n` test rows.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import re
import time
import numpy as np

from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .analysis_tables import investigation_clause, write_outputs
from .exchangeable import FAMILIAR_FROM, MONDRIAN_MINIMUM_ROWS, NOVEL_BELOW, ExchangeableDesign, ExchangeableResult, exchangeable_scores
from .featurization import PrimaryScorer
from .identity_bins import IDENTITY_BIN_NAMES, bin_counts
from .methods import HEAD_NAMES, PNORM_METHOD
from .predictions import load_seed_predictions
from .report import read_json
from .resampling import CALIBRATION_HALF, bootstrap_counts, exchangeable_design
from .single_probe import SINGLE_PROBE_HEADS, single_probe_verdicts, write_single_probe
from .verdicts import (
    INTERVAL_PERCENTILES,
    ArmScores,
    DecisionRules,
    RepeatSeries,
    pooled_shift,
    question_one,
    question_two,
    seed_spread,
    summary,
)
from .workflow import STANDARDIZER_RECORD, load_standardizer_record


REPORT_NAME = "report.json"
DEFAULT_PATTERN = f"**/{REPORT_NAME}"
DEFAULT_REPLICATES = 1000
DEFAULT_REPEATS = 200
DEFAULT_LEVELS = (0.8, 0.9, 0.95)
# The analysis container has eight CPUs; more processes than that only contend.
MAX_DEFAULT_WORKERS = 8
# How far a cell's test primary here may sit from its report's, relative to the larger of 1 and the report's.
POINT_TOLERANCE = 1e-6
# A coverage metric in report.json names its level after the "@".
COVERAGE_KEY = re.compile(r"^(?:lac|aps|conformal|interval)_coverage@([0-9.eE+-]+)$")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
# The scheme T metrics question 2 pools, besides each dataset's primary metric.
POOLED_METRICS = frozenset({"nll", "gaussian_nll", "aurc", "shift_auroc"})
# The heads the ablation's question 2 reads.
ABLATION_HEADS = frozenset({"projected", "sngp"})


@dataclass(frozen=True)
class CellRun:
    directory: Path
    report: dict[str, Any]

    @property
    def cell_id(self) -> str:
        return self.report["cell_id"]

    @property
    def arm(self) -> str:
        return (self.report.get("tags") or {}).get("arm", self.cell_id)

    @property
    def key(self) -> tuple[str, str, str]:
        """(dataset, arm, head), which names a cell in every table."""
        return self.report["data_name"], self.arm, self.report["head"]

    def seed_path(self, seed: int) -> Path:
        return self.directory / f"seed_{seed}.npz"


@dataclass(frozen=True)
class CellJob:
    index: int
    directory: str
    runs_dir: str
    replicates: int
    repeats: int
    seed: int


@dataclass
class CellResult:
    """What one worker learned about one cell."""

    index: int
    point: float  # the seed-mean test primary on every test row
    draws: np.ndarray  # (B,) the same on each bootstrap replicate
    scheme_t: list[dict[str, Any]]  # per method, metric, and identity bin: the mean and interval over repeats
    series: dict[tuple[str, str], np.ndarray]  # (method, metric) -> (R,), what question 2 pools
    calibration_rows: int
    mondrian_groupings: list[dict[str, Any]] | None
    checks: list[dict[str, Any]]
    seed_points: dict[int, float] = field(default_factory=dict)  # each seed's test primary on every test row
    notes: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0


def find_cells(runs_dir: str, pattern: str = DEFAULT_PATTERN) -> list[CellRun]:
    """Every finished cell whose report.json `pattern` matches under `runs_dir`, in path order."""
    paths = sorted(path for path in Path(runs_dir).glob(pattern) if path.name == REPORT_NAME and path.is_file())
    return [CellRun(path.parent, read_json(str(path))) for path in paths]


def check_unique(cells: Sequence[CellRun]) -> None:
    """Refuse two cells with one (dataset, arm, head)."""
    seen: dict[tuple[str, str, str], CellRun] = {}
    for cell in cells:
        if cell.key in seen:
            data_name, arm, head = cell.key
            raise ValueError(
                f"{seen[cell.key].directory} and {cell.directory} are both arm {arm!r} of {data_name} with head {head}; "
                f"narrow --pattern, such as '*/*/final/{REPORT_NAME}', so it matches one cell of each."
            )
        seen[cell.key] = cell


def _test_rows(path: Path) -> tuple[np.ndarray | None, np.ndarray]:
    """One seed's test digests and labels, read without loading its other arrays."""
    with np.load(path, allow_pickle=False) as archive:
        digests = archive["test__digests"] if "test__digests" in archive.files else None
        return digests, archive["test__labels"]  # (n,) digests or None, (n,) labels


def check_test_rows(cells: Sequence[CellRun]) -> None:
    """Refuse cells of one dataset, or seeds of one cell, that scored different test rows."""
    first: dict[str, tuple[CellRun, np.ndarray | None, np.ndarray]] = {}
    for cell in cells:
        for seed in cell.report["seeds"]:
            digests, labels = _test_rows(cell.seed_path(seed))
            data_name = cell.report["data_name"]
            if data_name not in first:
                first[data_name] = (cell, digests, labels)
                continue
            reference, reference_digests, reference_labels = first[data_name]
            same_digests = (digests is None and reference_digests is None) or (
                digests is not None and reference_digests is not None and np.array_equal(digests, reference_digests)
            )
            if not same_digests or not np.array_equal(labels, reference_labels):
                raise ValueError(f"{cell.cell_id} scored different test rows from {reference.cell_id}.")


def conformal_levels(report: Mapping[str, Any]) -> tuple[float, ...]:
    """The coverage levels a report scored, read from its metric names; the cell default if none."""
    levels = {
        float(match.group(1))
        for record in report.get("methods", {}).values()
        for key in record.get("test", {})
        if (match := COVERAGE_KEY.match(key))
    }
    return tuple(sorted(levels)) or DEFAULT_LEVELS


def label_spread(
    report: Mapping[str, Any], seeds: Sequence[Mapping[str, Any]], kind: str, directory: Path
) -> tuple[float, str]:
    """The regression conformal floor's label standard deviation, and where it came from.

    The fit reads the training labels' standard deviation, which report.json does not record. A
    cell run with `save_heads` records it in `standardizer.npz` in the cell's directory; without
    that file the validation labels, V_fit and V_cal together, stand in for it.
    """
    if kind != "regression":
        return 0.0, "not a regression task"
    if report.get("train_label_std") is not None:
        return float(report["train_label_std"]), "report.json train_label_std"
    record = directory / STANDARDIZER_RECORD
    if record.is_file():
        return load_standardizer_record(str(record))[1], f"{STANDARDIZER_RECORD} train_label_std"
    validation = [seeds[0][name].labels for name in ("v_fit", "v_cal") if name in seeds[0]]
    if validation:
        return float(np.std(np.concatenate(validation).astype(np.float64))), "validation labels, standing in for training labels"
    return float(np.std(seeds[0]["test"].labels.astype(np.float64))), "test labels, standing in for training labels"


def read_identity(report: Mapping[str, Any], rows: int, runs_dir: str) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Each test row's identity to training, from the file the report names, and a check of it.

    The file is looked for where the report says, then at `<runs dir>/../identity/<file name>`,
    the layout the grid writes. A named file found in neither place fails the check, since the
    fit wrote it; a report that names none passes. Either way the scores that need it are skipped.
    """
    recorded = (report.get("identity_bins") or {}).get("path")
    check: dict[str, Any] = {"check": "identity file", "recorded": recorded}
    if not recorded:
        return None, {**check, "passed": True, "read": None, "note": "report.json names no identity file"}  # no identity: None, not an (n,) array
    candidates = [Path(recorded), Path(runs_dir).parent / "identity" / Path(recorded).name]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return None, {  # no identity: None, not an (n,) array
            **check, "passed": False, "read": None, "searched": [str(candidate) for candidate in candidates],
            "note": "report.json names an identity file found in neither place; shift and identity-bin scores are skipped",
        }

    identity = np.asarray(read_json(str(path))["rows"]["test"], dtype=np.float64)  # (n,)
    if identity.shape[0] != rows:
        return None, {**check, "passed": False, "read": str(path), "note": f"holds {identity.shape[0]} test rows, not {rows}"}  # no identity: None, not an (n,) array
    expected = report["identity_bins"].get("counts")
    if expected is not None and bin_counts(identity) != expected:
        return None, {**check, "passed": False, "read": str(path), "note": "its bin counts differ from report.json's, so it changed after the fit"}  # no identity: None, not an (n,) array
    return identity, {**check, "passed": True, "read": str(path)}  # (n,) identity


# One dataset's draws, kept while a process's consecutive cells share the dataset.
_DRAW_CACHE: dict[str, tuple[tuple, Any]] = {}


def _cached_draws(use: str, labels: np.ndarray, label_type: str, data_name: str, count: int, seed: int) -> Any:
    # labels: (n,)
    digest = hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()
    key = (data_name, label_type, count, seed, labels.shape, str(labels.dtype), digest)
    held = _DRAW_CACHE.get(use)
    if held is None or held[0] != key:
        build = bootstrap_counts if use == "bootstrap" else exchangeable_design
        _DRAW_CACHE[use] = (key, build(labels, label_type, data_name, count, seed))
    return _DRAW_CACHE[use][1]


def _scheme_t_rows(exchangeable_result: ExchangeableResult) -> list[dict[str, Any]]:
    """Each method's metric, overall and per identity bin, as its mean and interval over repeats."""
    rows = []
    for method in exchangeable_result.methods:
        for metric, values in method.overall.items():
            rows.append({"method": method.name, "score_only": method.score_only, "metric": metric, "bin": "", **summary(values)})
        for metric, values in method.by_bin.items():
            for index, bin_name in enumerate(IDENTITY_BIN_NAMES):
                rows.append({"method": method.name, "score_only": method.score_only, "metric": metric, "bin": bin_name, **summary(values[:, index])})
    return rows


def _spread_check(report: Mapping[str, Any], method: str, seed_points: Mapping[int, float]) -> dict[str, Any]:
    """The seed spread of the test primary here against the report's `test_std`, taken as report.py takes it.

    report.py writes the standard deviation over the seeds that define the score, with ddof = 1,
    0 for one such seed, and None for none.
    """
    defined = [value for value in seed_points.values() if math.isfinite(value)]
    here = seed_spread(defined) if len(defined) > 1 else (0.0 if defined else None)
    reported = report["methods"].get(method, {}).get("test_std", {}).get(report["primary_metric"])
    check = {"check": "seed spread against report.json", "method": method, "here": here, "report": reported}
    if here is None or reported is None:
        check["passed"] = here is None and reported is None
    else:
        check["passed"] = abs(here - reported) <= POINT_TOLERANCE * max(1.0, abs(reported))
    return check


def analyze_cell(job: CellJob) -> CellResult:
    """Question 1's draws and scheme T for one cell, with the checks of both."""
    started = time.perf_counter()
    cell = CellRun(Path(job.directory), read_json(str(Path(job.directory) / REPORT_NAME)))
    report = cell.report
    kind, label_type, head, data_name = report["task_type"], report["label_type"], report["head"], report["data_name"]
    seeds = [load_seed_predictions(str(cell.seed_path(seed))) for seed in report["seeds"]]
    test = {seed: splits["test"] for seed, splits in zip(report["seeds"], seeds, strict=True)}
    labels = seeds[0]["test"].labels  # (n,) or (n, c)
    rows = labels.shape[0]

    scorer = PrimaryScorer(kind, list(test.values()))
    draws = scorer.score_draws(_cached_draws("bootstrap", labels, label_type, data_name, job.replicates, job.seed))  # (B,)
    every_row = np.arange(rows)  # (n,)
    point = scorer.score(every_row)
    # Each seed's run alone, so two arms' runs of one seed can be paired for the seed spread.
    seed_points = {seed: PrimaryScorer(kind, [split]).score(every_row) for seed, split in test.items()}
    reported = report["methods"].get(HEAD_NAMES[head], {}).get("test", {}).get(report["primary_metric"])
    point_check = {"check": "test primary against report.json", "method": HEAD_NAMES[head], "here": point, "report": reported}
    if reported is None or not math.isfinite(point):
        point_check["passed"] = reported is None and not math.isfinite(point)
    else:
        point_check["passed"] = abs(point - reported) <= POINT_TOLERANCE * max(1.0, abs(reported))

    levels = conformal_levels(report)
    train_label_std, spread_source = label_spread(report, seeds, kind, cell.directory)
    identity, identity_check = read_identity(report, rows, job.runs_dir)
    design: ExchangeableDesign = _cached_draws("exchangeable", labels, label_type, data_name, job.repeats, job.seed)
    exchangeable_result = exchangeable_scores(
        test, head, kind, design, levels, train_label_std, identity, pnorm=PNORM_METHOD in report["methods"]
    )

    return CellResult(
        index=job.index,
        point=point,
        draws=draws,
        scheme_t=_scheme_t_rows(exchangeable_result),
        series={
            (method.name, metric): values
            for method in exchangeable_result.methods
            for metric, values in method.overall.items()
            if metric in POOLED_METRICS or metric == report["primary_metric"]
        },
        calibration_rows=exchangeable_result.calibration_rows,
        mondrian_groupings=exchangeable_result.mondrian_groupings,
        checks=[point_check, _spread_check(report, HEAD_NAMES[head], seed_points), identity_check, *exchangeable_result.checks],
        seed_points=seed_points,
        notes={
            "test_rows": rows,
            "report_test_primary": reported,
            "levels": list(levels),
            "train_label_std": train_label_std,
            "train_label_std_source": spread_source,
            "identity_file": identity_check.get("read"),
        },
        seconds=time.perf_counter() - started,
    )


def available_cpus() -> int:
    """CPUs this process may use: its affinity, capped by a cgroup CPU quota where one is set."""
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    # An unreadable quota leaves the affinity count, which is never more than the machine has.
    with suppress(OSError, ValueError):
        version_two = Path("/sys/fs/cgroup/cpu.max")
        version_one = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        if version_two.is_file():
            quota, period = version_two.read_text().split()[:2]
            if quota != "max":
                count = min(count, max(1, int(quota) // int(period)))
        elif version_one.is_file():
            quota = int(version_one.read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                count = min(count, max(1, quota // period))
    return max(1, count)


def default_workers() -> int:
    return min(MAX_DEFAULT_WORKERS, available_cpus())


@contextmanager
def single_threaded_children() -> Iterator[None]:
    """Worker processes started inside this block run one BLAS and OpenMP thread each."""
    saved = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    os.environ.update({name: "1" for name in THREAD_VARIABLES})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def cell_cost(cell: CellRun) -> float:
    """A rough cost, to start the heaviest cells first: test rows, weighted by classes or labels."""
    rows = (cell.report.get("split") or {}).get("test") or 1
    labels = cell.report.get("num_labels") or 1
    return float(rows) * (1.0 + labels / 8.0) * (1.5 if cell.report["head"] == "projected" else 1.0)


def _job_name(job: CellJob) -> str:
    return os.path.relpath(job.directory, job.runs_dir)


def run_jobs(jobs: Sequence[CellJob], costs: Sequence[float], workers: int) -> list[CellResult]:
    """Every job's result in job order, from `workers` processes, or from this one when `workers` is 0."""
    order = sorted(range(len(jobs)), key=lambda index: (-costs[index], index))
    results: dict[int, CellResult] = {}
    started = time.perf_counter()
    if workers <= 0:
        for done, index in enumerate(order, start=1):
            results[index] = analyze_cell(jobs[index])
            print(f"[{done}/{len(jobs)}] {_job_name(jobs[index])} in {results[index].seconds:.1f}s")
        return [results[index] for index in range(len(jobs))]

    context = multiprocessing.get_context("spawn")
    with single_threaded_children(), ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=context) as pool:
        futures = [pool.submit(analyze_cell, jobs[index]) for index in order]
        try:
            for done, future in enumerate(as_completed(futures), start=1):
                cell_result = future.result()
                results[cell_result.index] = cell_result
                elapsed = time.perf_counter() - started
                print(f"[{done}/{len(jobs)}] {_job_name(jobs[cell_result.index])} in {cell_result.seconds:.1f}s, {elapsed:.0f}s elapsed", flush=True)
        except BaseException:
            # One failed cell stops the analysis without waiting for the queue behind it.
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return [results[index] for index in range(len(jobs))]


def arm_scores(cells: Sequence[CellRun], results: Sequence[CellResult]) -> ArmScores:
    """Question 1's inputs: the projected head's cells."""
    point, draws, v_fit, per_seed = {}, {}, {}, {}
    for cell, cell_result in zip(cells, results, strict=True):
        data_name, arm, head = cell.key
        if head != "projected":
            continue
        point[(data_name, arm)] = cell_result.point
        draws[(data_name, arm)] = cell_result.draws
        v_fit[(data_name, arm)] = cell.report.get("v_fit_primary")
        per_seed[(data_name, arm)] = cell_result.seed_points
    return ArmScores(point, draws, v_fit, per_seed)


def _kinds(cells: Sequence[CellRun]) -> dict[str, str]:
    return {cell.report["data_name"]: cell.report["task_type"] for cell in cells}


def _primaries(cells: Sequence[CellRun]) -> dict[str, str]:
    return {cell.report["data_name"]: cell.report["primary_metric"] for cell in cells}


def repeat_series(cells: Sequence[CellRun], results: Sequence[CellResult]) -> RepeatSeries:
    """Question 2's inputs under scheme T: every pooled metric of every method, per repeat."""
    values = {
        (*cell.key, method, metric): series
        for cell, cell_result in zip(cells, results, strict=True)
        for (method, metric), series in cell_result.series.items()
    }
    return RepeatSeries(values, _kinds(cells), _primaries(cells))


def ablation_heads(series: RepeatSeries) -> RepeatSeries:
    """The series of the ablation's heads, `projected` and `sngp`, whose method names never collide."""
    values = {key: values for key, values in series.values.items() if key[2] in ABLATION_HEADS}
    return RepeatSeries(values, series.kinds, series.primary)


def report_series(cells: Sequence[CellRun]) -> RepeatSeries:
    """Question 2's inputs under scheme V: each report's seed-mean test scores, as a single draw."""
    values = {}
    for cell in cells:
        wanted = POOLED_METRICS | {cell.report["primary_metric"]}
        for method, record in cell.report["methods"].items():
            for metric, value in record["test"].items():
                if metric in wanted:
                    values[(*cell.key, method, metric)] = np.array([math.nan if value is None else float(value)])  # (1,)
    return RepeatSeries(values, _kinds(cells), _primaries(cells))


def compare_runs(
    runs_dir: str,
    output_dir: str,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = 0,
    *,
    pattern: str = DEFAULT_PATTERN,
    repeats: int = DEFAULT_REPEATS,
    workers: int | None = None,
    rules: DecisionRules | None = None,
) -> dict[str, Any]:
    """Answer both questions over every matched cell, write every table, and return verdicts.json's content.

    `workers` defaults to the CPUs available, at most eight; 0 analyzes every cell in this process.
    """
    rules = rules or DecisionRules()
    cells = find_cells(runs_dir, pattern)
    if not cells:
        raise ValueError(f"no {REPORT_NAME} matches {pattern!r} under {runs_dir}.")
    check_unique(cells)
    check_test_rows(cells)
    workers = default_workers() if workers is None else workers
    started = time.perf_counter()

    jobs = [CellJob(index, str(cell.directory), runs_dir, replicates, repeats, seed) for index, cell in enumerate(cells)]
    results = run_jobs(jobs, [cell_cost(cell) for cell in cells], workers)

    q1 = question_one(arm_scores(cells, results), rules)
    present = {cell.arm for cell in cells}
    arms = [arm for arm in rules.real_arms if arm in present]
    series = repeat_series(cells, results)
    # Question 2 names its methods without their head, so it reads the ablation's heads alone;
    # a comparison holding only those heads, as the ablation's does, reads every series.
    q2 = question_two(ablation_heads(series), rules, q1["recommendation"], arms)
    q2["scheme_v"] = question_two(ablation_heads(report_series(cells)), rules, q1["recommendation"], arms)
    q2["shift"] = pooled_shift(series)
    parameters = {
        "runs": runs_dir,
        "pattern": pattern,
        "seed": seed,
        "replicates": replicates,
        "repeats": repeats,
        "calibration_half": CALIBRATION_HALF,
        "interval_percentiles": list(INTERVAL_PERCENTILES),
        "mondrian_minimum_rows": MONDRIAN_MINIMUM_ROWS,
        "shift": {"novel_below": NOVEL_BELOW, "familiar_from": FAMILIAR_FROM},
        "rules": rules.as_dict(),
    }
    verdicts = write_outputs(Path(output_dir), cells=cells, results=results, q1=q1, q2=q2, parameters=parameters, rules=rules)
    if any(cell.key[2] in SINGLE_PROBE_HEADS for cell in cells):
        single_probe = single_probe_verdicts(series, sorted(present))
        write_single_probe(Path(output_dir), single_probe)
        adopted = {
            family: f"{record['adopted']['method']} on {record['adopted']['head']}"
            for families in single_probe["per_arm"].values()
            for family, record in families.items()
        }
        print(f"Single-probe rules adopt {adopted}; see single_probe.md.")
    print(
        f"Compared {len(cells)} cells into {os.fspath(output_dir)} in {time.perf_counter() - started:.0f}s with {workers} workers; "
        f"question 1 recommends {q1['recommendation']}{investigation_clause(q1)}, question 2 adopts {q2['adopted']}; "
        f"{verdicts['checks']['failed']} of {verdicts['checks']['total']} checks failed."
    )
    return verdicts
