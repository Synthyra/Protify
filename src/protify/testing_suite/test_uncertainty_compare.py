"""Comparing finished cells: the pre-registered answers, the tables, and what the pairing needs.

The decision rules read intervals of one arm minus another, so the pairing has to be real:
every replicate scores both cells on the same resampled rows, the cells must have scored the
same rows to begin with, and the fast replicate scores must equal the metrics the report used.
End to end, an arm and a head better by construction must be found better, recommended, and
adopted; scheme T must cover where scheme V, calibrated on easier rows, does not; tuning cells
must stay out; and the output must depend on the seed alone, never on the worker count. Each
cell's seed spread must be the ddof = 1 deviation of its seeds and agree with its report, and an
identity file a report names must be found or fail its check.
"""

import csv
import json
import zlib
import numpy as np
import pytest
import torch

from sklearn.metrics import matthews_corrcoef


try:
    from src.protify.metrics import max_metrics
    from src.protify.uncertainty.__main__ import main, parse_arguments
    from src.protify.uncertainty.compare import DEFAULT_PATTERN, compare_runs, find_cells, label_spread, read_identity
    from src.protify.uncertainty.featurization import PrimaryScorer
    from src.protify.uncertainty.methods import build_methods, probabilities
    from src.protify.uncertainty.metrics import PRIMARY_METRICS, matthews_correlation, task_metrics
    from src.protify.uncertainty.predictions import SplitPredictions, load_seed_predictions, save_seed_predictions
    from src.protify.uncertainty.report import score_method, write_json
    from src.protify.uncertainty.verdicts import json_ready
    from src.protify.uncertainty.workflow import save_standardizer_record
except ImportError:
    from protify.metrics import max_metrics
    from protify.uncertainty.__main__ import main, parse_arguments
    from protify.uncertainty.compare import DEFAULT_PATTERN, compare_runs, find_cells, label_spread, read_identity
    from protify.uncertainty.featurization import PrimaryScorer
    from protify.uncertainty.methods import build_methods, probabilities
    from protify.uncertainty.metrics import PRIMARY_METRICS, matthews_correlation, task_metrics
    from protify.uncertainty.predictions import SplitPredictions, load_seed_predictions, save_seed_predictions
    from protify.uncertainty.report import score_method, write_json
    from protify.uncertainty.verdicts import json_ready
    from protify.uncertainty.workflow import save_standardizer_record


ROWS = 200
GRID_SEEDS = (0, 1, 2)
GRID_LEVELS = (0.8, 0.9)
OUTPUT_FILES = {
    "verdicts.json", "q1_featurization.md", "q2_uncertainty.md", "summary.md", "q1_scores.csv", "q1_pairs.csv",
    "q1_sensitivity.csv", "q2_scheme_t.csv", "q2_pooled.csv", "q2_shift.csv", "q2_identity_coverage.csv",
    "conformal_bands.csv", "checks.csv", "arms.csv", "methods.csv", "identity_bins.csv", "paired_bootstrap.csv",
}
SPREAD_CHECK = "seed spread against report.json"
FINAL_PATTERN = "*/*/final/report.json"


def _write_cell(root, cell_id: str, outputs: dict[int, np.ndarray], labels: np.ndarray, digests: np.ndarray) -> None:
    # outputs: (n, 2) per seed, labels: (n,), digests: (n,)
    directory = root / cell_id
    directory.mkdir(parents=True)
    for seed, seed_outputs in outputs.items():
        predictions = {"test": SplitPredictions(labels=labels, outputs=seed_outputs, digests=digests)}
        save_seed_predictions(str(directory / f"seed_{seed}.npz"), predictions)
    write_json(str(directory / "report.json"), {
        "cell_id": cell_id,
        "data_name": "toyset",
        "head": "projected",
        "display_name": cell_id,
        "task_type": "binary",
        "label_type": "singlelabel",
        "primary_metric": "mcc",
        "seeds": list(outputs),
        "v_fit_primary": 0.5,
        "methods": {"plain": {"test": {"mcc": 0.4, "ece": 0.1}, "test_std": {"mcc": 0.01, "ece": 0.02}}},
        "identity_bins": {"methods": {"plain": {"no_hit": {"count": 3.0, "mcc": 0.2}}}},
    })


def _two_cells(root, digests_b: np.ndarray | None = None) -> None:
    # digests_b: (n,)
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=ROWS)
    digests = np.array([f"row{index:04d}" for index in range(ROWS)])
    informed = {seed: rng.normal(size=(ROWS, 2)) + 3.0 * np.eye(2)[labels] for seed in (42, 43)}
    guessing = {seed: rng.normal(size=(ROWS, 2)) for seed in (42, 43)}
    _write_cell(root, "informed", informed, labels, digests)
    _write_cell(root, "guessing", guessing, labels, digests if digests_b is None else digests_b)


def _grid_labels(kind: str, sizes: dict[str, int], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {  # (n,) per split
        split: rng.normal(size=rows).astype(np.float32) if kind == "regression" else rng.integers(0, 2, size=rows)
        for split, rows in sizes.items()
    }


def _grid_predictions(
    kind: str, head: str, labels: np.ndarray, skill: float, noise: float, rng: np.random.Generator, digests: np.ndarray | None
) -> SplitPredictions:
    """One seed's head on one split: the label's signal at `skill` under Gaussian noise at `noise`."""
    # labels: (n,), digests: (n,)
    rows = labels.shape[0]
    distance = rng.gamma(2.0, size=rows).astype(np.float32)  # (n,)
    if kind == "regression":
        spread = noise / skill
        outputs = (labels + spread * rng.normal(size=rows)).astype(np.float32)  # (n,)
        if head == "sngp":
            extra = {"variance": np.full(rows, spread ** 2, dtype=np.float32)}
        else:
            extra = {
                "mc_outputs": (outputs + 0.05 * rng.normal(size=rows)).astype(np.float32),
                "mc_variance": np.full(rows, 0.1, dtype=np.float32),
            }
    else:
        outputs = (skill * np.eye(2)[labels] + noise * rng.normal(size=(rows, 2))).astype(np.float32)  # (n, 2)
        if head == "sngp":
            extra = {"variance": rng.uniform(0.1, 0.9, size=rows).astype(np.float32)}
        else:
            noisy = outputs + 0.2 * rng.normal(size=(rows, 2)).astype(np.float32)  # (n, 2)
            extra = {"mc_outputs": probabilities(noisy, kind).astype(np.float32)}
    return SplitPredictions(labels=labels, outputs=outputs, nn_distance=distance, digests=digests, **extra)


def _primary(split: SplitPredictions, kind: str) -> float:
    if kind == "regression":
        return task_metrics(kind, split.labels, mean=split.outputs)[PRIMARY_METRICS[kind]]
    return task_metrics(kind, split.labels, probs=probabilities(split.outputs, kind))[PRIMARY_METRICS[kind]]


def _write_grid_cell(
    runs, data_name: str, kind: str, arm: str, head: str, labels: dict[str, np.ndarray], skill: float,
    noise: float = 1.0, validation_noise: float | None = None, stage: str = "final",
) -> None:
    """A finished cell laid out as the grid lays it out, its report built as the workflow builds it."""
    # labels: (n,) per split
    group = f"{data_name}.{arm}"
    directory = runs / group / head / stage
    directory.mkdir(parents=True)
    digests = np.array([f"{data_name}{row:04d}" for row in range(labels["test"].shape[0])])
    seed_predictions = {}
    for seed in GRID_SEEDS:
        rng = np.random.default_rng([zlib.crc32(group.encode()), int(head == "sngp"), seed])
        seed_predictions[seed] = {
            split: _grid_predictions(
                kind, head, labels[split], skill, noise if split == "test" or validation_noise is None else validation_noise,
                rng, digests if split == "test" else None,
            )
            for split in ("v_fit", "v_cal", "test")
        }
        save_seed_predictions(str(directory / f"seed_{seed}.npz"), seed_predictions[seed])

    # The validation labels' spread, which compare uses in place of the training labels'.
    spread = float(np.std(np.concatenate([labels["v_fit"], labels["v_cal"]]).astype(np.float64))) if kind == "regression" else 0.0
    methods = {method.name: score_method(method, kind, labels, GRID_LEVELS, spread, None) for method in build_methods(seed_predictions, head, kind)}
    write_json(str(directory / "report.json"), json_ready({
        "cell_id": f"{group}.{head}.{stage.replace('/', '.')}",
        "tags": {"arm": arm, "group": group, "head": head, "role": stage.split("/")[0]},
        "data_name": data_name,
        "task_type": kind,
        "label_type": "regression" if kind == "regression" else "singlelabel",
        "num_labels": 1 if kind == "regression" else 2,
        "head": head,
        "display_name": arm,
        "primary_metric": PRIMARY_METRICS[kind],
        "seeds": list(GRID_SEEDS),
        "split": {split: int(values.shape[0]) for split, values in labels.items()},
        "v_fit_primary": float(np.mean([_primary(seed_predictions[seed]["v_fit"], kind) for seed in GRID_SEEDS])),
        "methods": methods,
    }))


def test_the_fast_mcc_is_scikit_learns():
    rng = np.random.default_rng(0)
    for classes in (2, 5):
        labels = rng.integers(0, classes, size=300)
        predicted = np.where(rng.random(300) < 0.6, labels, rng.integers(0, classes, size=300))
        assert matthews_correlation(labels, predicted, classes) == pytest.approx(matthews_corrcoef(labels, predicted))
    assert matthews_correlation(np.zeros(5, dtype=int), np.zeros(5, dtype=int), 2) == 0.0


def test_the_replicate_f1_on_every_row_is_protifys():
    rng = np.random.default_rng(0)
    labels = (rng.random((60, 7)) < 0.3).astype(np.float32)  # (60, 7)
    logits = rng.normal(size=(60, 7)) + 2.0 * labels  # (60, 7)
    scorer = PrimaryScorer("multilabel", [SplitPredictions(labels=labels, outputs=logits)])
    probs = 1.0 / (1.0 + np.exp(-logits))
    expected, _, _, _ = max_metrics(torch.from_numpy(probs.reshape(-1)), torch.from_numpy(labels.reshape(-1)))
    assert scorer.score(np.arange(60)) == pytest.approx(expected, abs=1e-6)


def test_paired_intervals_are_mirrored_and_exclude_zero_for_a_real_gap(tmp_path):
    _two_cells(tmp_path / "runs")
    compare_runs(str(tmp_path / "runs"), str(tmp_path / "tables"), replicates=200, seed=0, repeats=4, workers=0)

    with open(tmp_path / "tables" / "paired_bootstrap.csv", encoding="utf-8") as handle:
        rows = {(row["cell_a"], row["cell_b"]): row for row in csv.DictReader(handle)}
    forward, backward = rows[("informed", "guessing")], rows[("guessing", "informed")]
    assert float(forward["difference"]) == pytest.approx(-float(backward["difference"]))
    assert float(forward["lower_95"]) == pytest.approx(-float(backward["upper_95"]))
    assert float(forward["lower_95"]) > 0.0

    assert {path.name for path in (tmp_path / "tables").iterdir()} == OUTPUT_FILES
    for name in ("arms.csv", "methods.csv", "identity_bins.csv", "summary.md"):
        assert (tmp_path / "tables" / name).read_text(encoding="utf-8")


def test_cells_that_scored_different_rows_are_not_compared(tmp_path):
    _two_cells(tmp_path / "runs", digests_b=np.array([f"other{index:04d}" for index in range(ROWS)]))
    with pytest.raises(ValueError, match="different test rows"):
        compare_runs(str(tmp_path / "runs"), str(tmp_path / "tables"), replicates=10, workers=0)


def test_tuning_cells_stay_out_under_the_final_pattern_and_are_refused_without_it(tmp_path):
    runs = tmp_path / "runs"
    labels = _grid_labels("binary", {"v_fit": 60, "v_cal": 60, "test": 160}, 0)
    _write_grid_cell(runs, "toy", "binary", "sae", "projected", labels, 2.0)
    _write_grid_cell(runs, "toy", "binary", "sae", "projected", labels, 2.0, stage="tune/lr_0.001")
    _write_grid_cell(runs, "toy", "binary", "last", "projected", labels, 1.0)

    assert [cell.cell_id for cell in find_cells(str(runs), FINAL_PATTERN)] == ["toy.last.projected.final", "toy.sae.projected.final"]
    assert len(find_cells(str(runs))) == 3
    with pytest.raises(ValueError, match="narrow --pattern"):
        compare_runs(str(runs), str(tmp_path / "tables"), replicates=10, pattern=DEFAULT_PATTERN, workers=0)

    verdicts = compare_runs(str(runs), str(tmp_path / "tables"), replicates=50, pattern=FINAL_PATTERN, repeats=4, workers=0)
    assert [cell["cell_id"] for cell in verdicts["cells"]] == ["toy.last.projected.final", "toy.sae.projected.final"]


def test_an_arm_and_a_head_better_by_construction_are_recommended_and_adopted(tmp_path):
    runs = tmp_path / "runs"
    datasets = {"toy-binary": ("binary", {"v_fit": 120, "v_cal": 120, "test": 400}), "toy-regression": ("regression", {"v_fit": 100, "v_cal": 100, "test": 300})}
    for index, (data_name, (kind, sizes)) in enumerate(datasets.items()):
        labels = _grid_labels(kind, sizes, index)
        for arm, skill in {"sae": 3.0, "last": 1.2, "l27": 1.2, "both": 1.3, "onehot": 0.3, "random": 0.05, "random_init": 0.4}.items():
            _write_grid_cell(runs, data_name, kind, arm, "projected", labels, skill)
        # sae's SNGP head is sharper and less noisy than its plain head, temperature-scaled or not.
        _write_grid_cell(runs, data_name, kind, "sae", "sngp", labels, 5.0, noise=0.7)

    verdicts = compare_runs(str(runs), str(tmp_path / "tables"), replicates=200, pattern=FINAL_PATTERN, repeats=10, workers=0)
    q1, q2 = verdicts["question_1"], verdicts["question_2"]
    pairs = {(record["first"], record["second"]): record for record in q1["pairs"]}
    assert pairs[("sae", "last")]["verdict"] == "sae better"
    assert pairs[("sae", "l27")]["verdict"] == "sae better"
    assert (q1["best_arm"], q1["recommendation"]) == ("sae", "sae")
    assert q1["controls"]["passed"]

    candidates = {record["method"]: record for record in q2["per_arm"]["sae"]["candidates"]}
    assert candidates["sngp_temperature"]["improves"]
    assert candidates["sngp_temperature"]["pooled"]["relative_nll"]["upper"] < 0.0
    assert q2["adopted"] == "sngp_temperature"
    # Only sae has an SNGP head, so every other arm adopts something else, and is named.
    assert q2["arms_differing"] == ["last", "l27", "both"]
    assert verdicts["checks"]["failed"] == 0
    # Every cell's seed spread agrees with its report: 14 projected cells and 2 SNGP.
    assert verdicts["checks"]["by_check"][SPREAD_CHECK] == {"total": 16, "failed": 0}
    assert "**Recommendation: sae.**" in (tmp_path / "tables" / "q1_featurization.md").read_text(encoding="utf-8")
    # The arms without an SNGP cell pool sngp_temperature over no dataset, and the note says so.
    assert "- last: sngp_temperature lacks toy-binary (binary; no sngp cell of last)." in (tmp_path / "tables" / "q2_uncertainty.md").read_text(encoding="utf-8")


def test_each_cells_seed_spread_is_the_ddof1_deviation_of_its_seeds_and_pairs_by_seed(tmp_path):
    runs = tmp_path / "runs"
    labels = _grid_labels("binary", {"v_fit": 60, "v_cal": 60, "test": 160}, 0)
    for arm, skill in (("sae", 2.0), ("last", 1.0)):
        _write_grid_cell(runs, "toy", "binary", arm, "projected", labels, skill)
    verdicts = compare_runs(str(runs), str(tmp_path / "tables"), replicates=20, pattern=FINAL_PATTERN, repeats=4, workers=0)

    per_seed = {
        arm: np.array([_primary(load_seed_predictions(str(runs / f"toy.{arm}" / "projected" / "final" / f"seed_{seed}.npz"))["test"], "binary") for seed in GRID_SEEDS])
        for arm in ("sae", "last")
    }  # each (3,)
    with open(tmp_path / "tables" / "q1_scores.csv", encoding="utf-8") as handle:
        scores = {row["arm"]: row for row in csv.DictReader(handle)}
    for arm, values in per_seed.items():
        assert float(scores[arm]["test_primary_seed_std_ddof1"]) == pytest.approx(np.std(values, ddof=1))
    with open(tmp_path / "tables" / "q1_pairs.csv", encoding="utf-8") as handle:
        pairs = {(row["comparison"], row["data_name"]): row for row in csv.DictReader(handle)}
    # The pair's spread is over seed-matched differences.
    assert float(pairs[("sae - last", "toy")]["seed_std_ddof1"]) == pytest.approx(np.std(per_seed["sae"] - per_seed["last"], ddof=1))
    assert verdicts["checks"]["by_check"][SPREAD_CHECK] == {"total": 2, "failed": 0}


def test_an_identity_file_the_report_names_must_be_found_or_its_check_fails(tmp_path):
    runs = str(tmp_path / "runs")
    # A report that names none passes, with a note.
    identity, check = read_identity({}, 4, runs)
    assert identity is None and check["passed"] and check["note"] == "report.json names no identity file"

    # A named file found in neither place fails.
    named = {"identity_bins": {"path": str(tmp_path / "remote" / "toy.json")}}
    identity, check = read_identity(named, 4, runs)
    assert identity is None and not check["passed"]
    assert check["searched"] == [str(tmp_path / "remote" / "toy.json"), str(tmp_path / "identity" / "toy.json")]

    # The same file beside the runs directory, where the grid writes it, is found and read.
    (tmp_path / "identity").mkdir()
    write_json(str(tmp_path / "identity" / "toy.json"), {"rows": {"test": [0.0, 0.2, 0.6, 0.9]}})
    identity, check = read_identity(named, 4, runs)
    assert check["passed"] and check["read"] == str(tmp_path / "identity" / "toy.json")
    assert identity.tolist() == [0.0, 0.2, 0.6, 0.9]


def test_scheme_t_covers_where_scheme_v_calibrated_on_easier_rows_does_not(tmp_path):
    labels = _grid_labels("regression", {"v_fit": 200, "v_cal": 200, "test": 400}, 0)
    # Validation residuals are a quarter of test's, as when validation proteins sit closer to training.
    _write_grid_cell(tmp_path / "runs", "toy", "regression", "sae", "projected", labels, 1.0, noise=1.0, validation_noise=0.25)
    verdicts = compare_runs(str(tmp_path / "runs"), str(tmp_path / "tables"), replicates=50, repeats=40, workers=0)

    rows = {(row["scheme"], row["method"], row["set"], row["level"]): row for row in verdicts["conformal"]}
    scheme_t, scheme_v = rows[("T", "plain", "conformal", 0.9)], rows[("V", "plain", "conformal", 0.9)]
    assert scheme_t["marginal_lower"] - 0.03 < scheme_t["coverage"] < scheme_t["marginal_upper"] + 0.03
    assert scheme_v["coverage"] < scheme_v["beta_lower"] - 0.2


def test_the_label_spread_reads_the_saved_training_spread_before_validation(tmp_path):
    validation = np.array([0.0, 2.0, 4.0])  # (n,)
    seeds = [{"v_fit": SplitPredictions(labels=validation, outputs=validation)}]

    spread, source = label_spread({}, seeds, "regression", tmp_path)
    assert (spread, source) == (pytest.approx(np.std(validation)), "validation labels, standing in for training labels")

    save_standardizer_record(str(tmp_path / "standardizer.npz"), None, 7.5)
    assert label_spread({}, seeds, "regression", tmp_path) == (7.5, "standardizer.npz train_label_std")
    assert label_spread({"train_label_std": 3.0}, seeds, "regression", tmp_path) == (3.0, "report.json train_label_std")
    assert label_spread({}, seeds, "binary", tmp_path) == (0.0, "not a regression task")


def test_the_output_depends_on_the_seed_alone(tmp_path):
    _two_cells(tmp_path / "runs")
    outputs = {}
    for name, seed in (("first", 0), ("again", 0), ("other", 1)):
        compare_runs(str(tmp_path / "runs"), str(tmp_path / name), 50, seed, repeats=6, workers=0)
        outputs[name] = {path.name: path.read_bytes() for path in (tmp_path / name).iterdir()}
    assert outputs["first"] == outputs["again"]
    assert outputs["first"]["verdicts.json"] != outputs["other"]["verdicts.json"]


@pytest.mark.slow
def test_worker_processes_write_the_same_bytes(tmp_path):
    # Each spawned worker imports protify afresh, so this takes a while.
    _two_cells(tmp_path / "runs")
    outputs = {}
    for workers in (1, 2):
        compare_runs(str(tmp_path / "runs"), str(tmp_path / f"workers_{workers}"), 50, repeats=6, workers=workers)
        outputs[workers] = {path.name: path.read_bytes() for path in (tmp_path / f"workers_{workers}").iterdir()}
    assert outputs[1] == outputs[2]


def test_the_command_line_reaches_compare(tmp_path):
    arguments = parse_arguments(["compare", "--runs", "a", "--output", "b", "--replicates", "7"])
    assert (arguments.command, arguments.runs, arguments.output, arguments.replicates) == ("compare", "a", "b", 7)
    assert (arguments.pattern, arguments.repeats, arguments.workers, arguments.margin) == (DEFAULT_PATTERN, 200, None, 0.01)

    _two_cells(tmp_path / "runs")
    main([
        "compare", "--runs", str(tmp_path / "runs"), "--pattern", "*/report.json", "--output", str(tmp_path / "tables"),
        "--replicates", "5", "--repeats", "4", "--workers", "0", "--margin", "0.02", "--cost-order", "sae", "l27", "last", "both",
    ])
    verdicts = json.loads((tmp_path / "tables" / "verdicts.json").read_text(encoding="utf-8"))
    assert verdicts["parameters"]["pattern"] == "*/report.json"
    assert (verdicts["parameters"]["rules"]["margin"], verdicts["parameters"]["rules"]["cost_order"]) == (0.02, ["sae", "l27", "last", "both"])
    assert (tmp_path / "tables" / "paired_bootstrap.csv").exists()
