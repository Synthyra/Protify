"""Packaging a finished cell, and a package predicting what its cell measured.

Four tiny cells run through the real workflow with `save_heads` on, binary and regression, each
with the projected and the SNGP head and two seeds, on the toy dataset and caches that
`test_uncertainty_workflow.py` builds. Every shippable method of each is packaged, loaded, and
asked to predict the cell's test rows from their joined features, and must give back its
package's metrics.json and its method's report.json numbers. The binary projected cell trains
with dropout, so its MC dropout methods draw passes the cell never drew and are checked for
shape and repeatability; the regression projected cell trains without, so its are exact.
"""

import hashlib
import json
import os
import shutil
import sys
import numpy as np
import pandas as pd
import pytest
import torch

from dataclasses import dataclass, replace
from pathlib import Path


try:
    from src.protify.data.dataset_classes import EmbeddingStandardizer
    from src.protify.uncertainty import workflow
    from src.protify.uncertainty.__main__ import main as uncertainty_main
    from src.protify.uncertainty.cell import UncertaintyCell, cell_from_mapping
    from src.protify.uncertainty.conformal_sets import (
        SET_SCORES,
        class_scores,
        classification_conformal_rows,
        prediction_sets,
        regression_conformal_rows,
        regression_interval,
    )
    from src.protify.uncertainty.methods import MethodRun, build_methods
    from src.protify.uncertainty.metrics import CLASSIFICATION_KINDS
    from src.protify.uncertainty.package import best_v_fit_seed, build_package
    from src.protify.uncertainty.packaged import PackagedHead, calibrated_predictive, load_package
    from src.protify.uncertainty.report import read_json, score_rows, write_json
except ImportError:
    from protify.data.dataset_classes import EmbeddingStandardizer
    from protify.uncertainty import workflow
    from protify.uncertainty.__main__ import main as uncertainty_main
    from protify.uncertainty.cell import UncertaintyCell, cell_from_mapping
    from protify.uncertainty.conformal_sets import (
        SET_SCORES,
        class_scores,
        classification_conformal_rows,
        prediction_sets,
        regression_conformal_rows,
        regression_interval,
    )
    from protify.uncertainty.methods import MethodRun, build_methods
    from protify.uncertainty.metrics import CLASSIFICATION_KINDS
    from protify.uncertainty.package import best_v_fit_seed, build_package
    from protify.uncertainty.packaged import PackagedHead, calibrated_predictive, load_package
    from protify.uncertainty.report import read_json, score_rows, write_json

from .test_uncertainty_workflow import DENSE_WIDTH, SAE_WIDTH, _cell_mapping, _no_training, _toy_dataset, _write_caches


# Cell name -> (dataset, head, dropout); a dropout of None keeps Protify's.
CELLS = {
    "binary_projected": ("binary", "projected", 0.1),
    "binary_sngp": ("binary", "sngp", None),
    "regression_projected": ("regression", "projected", 0.0),
    "regression_sngp": ("regression", "sngp", None),
}
SEEDS = [42, 43]
# The cells' probe_batch_size, so a package batches the test rows exactly as its cell did.
BATCH_SIZE = 16
# The device `_train_seed` predicts on. A package reproduces its cell's outputs to rounding only
# on the same device: Protify turns on TF32 matmuls for CUDA, which move these logits by up to 1e-3.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REVISIONS = {
    "Synthyra/ESMplusplus_small": "a" * 40,
    "biohub/ESMC-300M-sae-layer23-k64-codebook8192": "b" * 40,
    "Synthyra/ESM2-8M": "c" * 40,
}


@dataclass(frozen=True)
class FinishedToy:
    cell: UncertaintyCell
    report: dict
    features: np.ndarray  # (30, W), the test rows' joined branches before standardization
    train_labels: np.ndarray  # (48,)


def _regression_dataset(root, frames: dict[str, pd.DataFrame]) -> tuple[str, dict[str, np.ndarray]]:
    """The toy sequences with float labels that follow the cached signal, and those labels."""
    rng = np.random.default_rng(2)
    data_dir = root / "data" / "toyreg"
    data_dir.mkdir(parents=True)
    labels = {}
    for split, frame in frames.items():
        labels[split] = 10.0 + 2.0 * frame["labels"].to_numpy() + 0.3 * rng.normal(size=len(frame))  # (n,)
        pd.DataFrame({"seqs": frame["seqs"], "labels": labels[split]}).to_csv(data_dir / f"{split}.csv", index=False)
    return str(data_dir), labels  # a path, then (n,) per split


def _recorded_joins(patch: pytest.MonkeyPatch) -> list:
    """Every joined branch set the workflow builds, in order."""
    joins = []
    join_branch_embeddings = workflow.join_branch_embeddings

    def recorded(*arguments, **options):
        joined, widths = join_branch_embeddings(*arguments, **options)
        joins.append(joined)
        return joined, widths

    patch.setattr(workflow, "join_branch_embeddings", recorded)
    return joins


def _packaged_cell(root, data_dir: str, name: str, head: str, dropout: float | None, save_heads: bool = True, num_runs: int = len(SEEDS)):
    mapping = _cell_mapping(root, data_dir, name, num_runs)
    mapping["uncertainty"].update(head=head, mc_dropout_samples=3, save_heads=save_heads)
    if dropout is not None:
        mapping["uncertainty"]["dropout"] = dropout
    return cell_from_mapping(mapping)


@pytest.fixture(scope="module")
def cells(tmp_path_factory):
    root = tmp_path_factory.mktemp("packaging")
    with pytest.MonkeyPatch.context() as patch:
        # Hugging Face writes checkpoints under the working directory; nothing may land in the tree.
        patch.chdir(root)
        patch.setattr(sys.modules[workflow.parse_arguments.__module__], "load_secrets_env", lambda: [])
        frames, binary_dir = _toy_dataset(root)
        _write_caches(frames, root / "embeddings")
        regression_dir, regression_labels = _regression_dataset(root, frames)
        datasets = {
            "binary": (binary_dir, frames["train"]["labels"].to_numpy(), frames["test"]["labels"].to_numpy()),
            "regression": (regression_dir, regression_labels["train"], regression_labels["test"]),
        }
        joins = _recorded_joins(patch)
        test_sequences = list(frames["test"]["seqs"])

        finished = {}
        for name, (dataset, head, dropout) in CELLS.items():
            data_dir, train_labels, test_labels = datasets[dataset]
            cell = _packaged_cell(root, data_dir, name, head, dropout)
            report = workflow.run_uncertainty_cell(cell)
            joined = joins[-1]
            features = joined.block([joined.rows[sequence] for sequence in test_sequences])  # (30, W)
            # The features follow the test split's row order, which the saved labels confirm.
            saved_labels = workflow.load_seeds(cell.output_dir, [42])[42]["test"].labels  # (30,)
            np.testing.assert_allclose(saved_labels, test_labels, rtol=0, atol=1e-5)
            finished[name] = FinishedToy(cell, report, features, train_labels)
        yield root, finished


def _shipped_run(toy: FinishedToy, method_name: str, seeds: list[int]) -> tuple[MethodRun, dict[str, np.ndarray]]:
    """The run `build_methods` gives for the shipped seeds, and every split's labels."""
    seed_predictions = workflow.load_seeds(toy.cell.output_dir, SEEDS)
    labels = workflow.agreed_labels(seed_predictions, SEEDS)
    method = next(method for method in build_methods(seed_predictions, toy.report["head"], toy.report["task_type"]) if method.name == method_name)
    if method.runs[0].seed is None:
        return method.runs[0], labels  # the run, then (n,) per split
    return next(run for run in method.runs if run.seed == seeds[0]), labels  # the run, then (n,) per split


def _conformal_rows(prediction, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Per test row coverage and set size or interval width, from a prediction's sets or intervals."""
    # labels: (n,)
    rows = {}
    for score, by_level in prediction.sets.items():
        for level, members in by_level.items():
            rows[f"{score}_coverage@{level}"] = members[np.arange(labels.shape[0]), labels].astype(np.float64)  # (n,)
            rows[f"{score}_size@{level}"] = members.sum(axis=1).astype(np.float64)  # (n,)
    for level, interval in prediction.intervals.items():
        rows[f"conformal_coverage@{level}"] = interval.covers(labels).astype(np.float64)  # (n,)
        rows[f"conformal_width@{level}"] = interval.width  # (n,)
    return rows  # (n,) per measure


def _assert_scores_match(first: dict, second: dict) -> None:
    """Equal on every score both hold, to float rounding, and None only where the other is None."""
    shared = sorted(set(first) & set(second))
    assert shared
    for name in shared:
        if first[name] is None or second[name] is None:
            assert first[name] is second[name], name
        else:
            assert first[name] == pytest.approx(second[name], rel=1e-5, abs=1e-8), name


def _assert_arrays_match(first: np.ndarray | None, second: np.ndarray | None) -> None:
    # first: (...), second: (...), any shape
    assert (first is None) == (second is None)
    if first is not None:
        np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-7)


def test_with_save_heads_off_a_cell_writes_what_it_always_wrote(cells):
    root, finished = cells
    kept = finished["binary_sngp"]
    cell = _packaged_cell(root, str(root / "data" / "toyset"), "binary_sngp_off", "sngp", None, save_heads=False, num_runs=1)
    assert cell.uncertainty.save_heads is False

    workflow.run_uncertainty_cell(cell)

    output = Path(cell.output_dir)
    assert set(os.listdir(output)) == {"cell.json", "split.json", "seed_42.npz", "seed_42.json", "report.json"}
    assert read_json(str(output / "cell.json"))["uncertainty"]["save_heads"] is False
    # Keeping a head draws no random number and moves no prediction.
    with np.load(output / "seed_42.npz") as off, np.load(Path(kept.cell.output_dir) / "seed_42.npz") as on:
        assert sorted(off.files) == sorted(on.files)
        for name in off.files:
            assert np.array_equal(off[name], on[name]), name


def test_save_heads_keeps_every_head_and_the_standardizer(cells):
    _, finished = cells
    for name, toy in finished.items():
        output = Path(toy.cell.output_dir)
        assert {"heads", "standardizer.npz"} <= set(os.listdir(output)), name
        assert sorted(os.listdir(output / "heads")) == [f"seed_{seed}" for seed in SEEDS]
        assert all(workflow.head_is_saved(str(output), seed) for seed in SEEDS)
        assert read_json(str(output / "cell.json"))["uncertainty"]["save_heads"] is True

        standardizer, train_label_std = workflow.load_standardizer_record(str(output / "standardizer.npz"))
        assert standardizer.mean.shape == standardizer.scale.shape == (SAE_WIDTH + DENSE_WIDTH,)
        expected_std = float(np.std(toy.train_labels.astype(np.float64))) if name.startswith("regression") else 0.0
        assert train_label_std == pytest.approx(expected_std, rel=1e-9)


def _no_loading(*arguments, **options):
    raise AssertionError("a refused resume loaded the dataset")


def test_a_resume_is_refused_while_a_finished_seed_lacks_its_head(cells, tmp_path, monkeypatch):
    _, finished = cells
    toy = finished["binary_sngp"]
    copy = tmp_path / "binary_sngp"
    shutil.copytree(toy.cell.output_dir, copy)
    cell = replace(toy.cell, output_dir=str(copy))

    shutil.move(copy / "heads" / "seed_43", tmp_path / "seed_43_head")
    with monkeypatch.context() as patch:
        patch.setattr(workflow, "load_cell_data", _no_loading)
        with pytest.raises(ValueError, match=r"finished seeds \[43\] without a saved head"):
            workflow.run_uncertainty_cell(cell)
    shutil.move(tmp_path / "seed_43_head", copy / "heads" / "seed_43")

    # A resume that fits another standardizer than the heads trained with is refused too.
    record = copy / "standardizer.npz"
    standardizer, train_label_std = workflow.load_standardizer_record(str(record))
    os.replace(record, tmp_path / "standardizer.npz")
    shifted = EmbeddingStandardizer(standardizer.mean.numpy() + 1.0, standardizer.scale.numpy())
    workflow.save_standardizer_record(str(record), shifted, train_label_std)
    monkeypatch.setattr(workflow, "_train_seed", _no_training)
    with pytest.raises(ValueError, match="records another standardizer"):
        workflow.run_uncertainty_cell(cell)

    os.replace(tmp_path / "standardizer.npz", record)
    assert workflow.run_uncertainty_cell(cell) == toy.report


@pytest.mark.parametrize("name", list(CELLS))
def test_a_package_predicts_what_its_cell_measured(cells, name, tmp_path):
    _, finished = cells
    toy = finished[name]
    kind = toy.report["task_type"]
    methods = build_methods(workflow.load_seeds(toy.cell.output_dir, SEEDS), toy.report["head"], kind)
    shippable = [method.name for method in methods if not method.score_only]
    expected_methods = {
        "projected": {"plain", "temperature", "mc_dropout", "mc_dropout_temperature", "ensemble", "ensemble_temperature"},
        "sngp": {"sngp", "sngp_temperature", "sngp_ensemble", "sngp_ensemble_temperature"},
    }
    assert set(shippable) == expected_methods[toy.report["head"]]

    for method_name in shippable:
        package_dir = tmp_path / method_name
        build_package(toy.cell.output_dir, method_name, str(package_dir))
        package = load_package(package_dir)
        calibration = read_json(str(package_dir / "calibration.json"))
        metrics = read_json(str(package_dir / "metrics.json"))
        levels = tuple(metrics["levels"])
        assert levels == (0.8, 0.9)
        prediction = package.predict(toy.features, batch_size=BATCH_SIZE, device=DEVICE)

        if package.predictive == "mc_dropout" and toy.cell.uncertainty.dropout > 0:
            # Passes the cell never drew, from a generator the package seeds, so only repeatable.
            again = package.predict(toy.features, batch_size=BATCH_SIZE, device=DEVICE)
            assert prediction.probs.shape == (30, 2) and np.allclose(prediction.probs.sum(axis=1), 1.0)
            assert np.array_equal(prediction.probs, again.probs)
            assert np.array_equal(prediction.sets["lac"][0.9], again.sets["lac"][0.9])
            continue

        run, labels = _shipped_run(toy, method_name, calibration["seeds"])
        expected = run.splits["test"]
        _assert_arrays_match(prediction.probs, expected.probs)
        _assert_arrays_match(prediction.mean, expected.mean)
        _assert_arrays_match(prediction.sigma, None if expected.variance is None else np.sqrt(expected.variance))
        _assert_arrays_match(prediction.uncertainty, expected.uncertainty)

        # The answer rule: V_cal's wrong answers give each test answer its p-value, and the
        # rows selected at the default target are the ones metrics.json scored.
        rule = calibration["answer_rule"]
        if kind == "regression":
            assert rule is None and metrics["answer_rule"] is None
            assert prediction.answers is None and prediction.confidence is None and prediction.p_values is None
        else:
            v_cal_probs = run.splits["v_cal"].probs  # (n_cal, c)
            wrong = v_cal_probs.argmax(axis=1) != labels["v_cal"]  # (n_cal,)
            assert rule["calibration_rows"] == labels["v_cal"].shape[0]
            assert rule["wrong_confidences"] == sorted(v_cal_probs.max(axis=1)[wrong].tolist())
            assert np.array_equal(prediction.answers, expected.probs.argmax(axis=1))
            _assert_arrays_match(prediction.confidence, expected.probs.max(axis=1).astype(np.float64))
            wrong_confidence = np.asarray(rule["wrong_confidences"])
            expected_p = [(1 + np.sum(wrong_confidence >= value)) / (rule["calibration_rows"] + 1) for value in prediction.confidence]
            assert np.allclose(prediction.p_values, expected_p, rtol=0, atol=1e-15)
            answered = package.select(prediction)  # (n,)
            correct = prediction.answers == labels["test"]
            assert metrics["answer_rule"]["answered"] == pytest.approx(answered.mean())
            assert metrics["answer_rule"]["accuracy"] == pytest.approx(correct.mean())
            if answered.any():
                assert metrics["answer_rule"]["answered_error"] == pytest.approx(1 - correct[answered].mean())

        # The prediction scored by the report's scorers gives the package's metrics.json...
        per_seed = package.seed_predictions(toy.features, batch_size=BATCH_SIZE, device=DEVICE)
        predictive = calibrated_predictive(per_seed, package.predictive, calibration["calibration"], calibration["fitted"], kind)
        rescored = score_rows(kind, predictive, labels["test"], levels, None, _conformal_rows(prediction, labels["test"]))
        _assert_scores_match(rescored, metrics["test"])
        assert set(rescored) == set(metrics["test"])
        assert metrics["primary"] == metrics["test"][metrics["primary_metric"]]

        # ...and metrics.json and calibration.json give the method's numbers in report.json, all
        # but the APS sets the report measures and no package ships.
        reported = next(entry for entry in toy.report["methods"][method_name]["per_run"] if entry["seed"] == run.seed)
        _assert_scores_match(metrics["test"], reported["test"])
        assert set(metrics["test"]) == {name for name in reported["test"] if not name.startswith("aps_")}
        assert calibration["fitted"] == pytest.approx(reported["fitted"])


def test_the_shipped_seed_is_chosen_on_v_fit_never_test(cells, tmp_path):
    assert best_v_fit_seed([{"seed": 42, "v_fit_primary": 0.5}, {"seed": 43, "v_fit_primary": 0.7}, {"seed": 44, "v_fit_primary": 0.7}]) == 43
    assert best_v_fit_seed([{"seed": 42, "v_fit_primary": None}, {"seed": 43, "v_fit_primary": -0.2}]) == 43
    with pytest.raises(ValueError, match="no seed recorded a V_fit primary metric"):
        best_v_fit_seed([{"seed": 42, "v_fit_primary": None}])

    _, finished = cells
    copy = tmp_path / "binary_sngp"
    shutil.copytree(finished["binary_sngp"].cell.output_dir, copy)
    report = read_json(str(copy / "report.json"))
    # Seed 42 looks best on test and seed 43 on V_fit; the package must ship seed 43.
    for record in report["per_seed"]:
        record["v_fit_primary"] = {42: 0.1, 43: 0.9}[record["seed"]]
    for method in report["methods"].values():
        for entry in method["per_run"]:
            if entry["seed"] == 42:
                entry["test"]["mcc"] = 1.0
    write_json(str(copy / "report.json"), report)

    build_package(str(copy), "sngp_temperature", str(tmp_path / "package"))

    calibration = read_json(str(tmp_path / "package" / "calibration.json"))
    assert calibration["seeds"] == [43]
    assert calibration["seed_choice"]["rule"] == "best_v_fit_primary"
    assert {record["seed"]: record["v_fit_primary"] for record in calibration["seed_choice"]["per_seed"]} == {42: 0.1, 43: 0.9}
    assert sorted(os.listdir(tmp_path / "package" / "heads")) == ["seed_43"]


@pytest.mark.parametrize(
    "name, method_name",
    [("binary_sngp", "sngp_temperature"), ("binary_projected", "ensemble_temperature"), ("regression_sngp", "sngp"), ("regression_projected", "plain")],
)
def test_the_package_quantiles_reproduce_conformal_sets(cells, name, method_name, tmp_path):
    _, finished = cells
    toy = finished[name]
    build_package(toy.cell.output_dir, method_name, str(tmp_path / "package"))
    calibration = read_json(str(tmp_path / "package" / "calibration.json"))
    metrics = read_json(str(tmp_path / "package" / "metrics.json"))
    conformal = calibration["conformal"]
    levels = tuple(conformal["levels"])
    run, labels = _shipped_run(toy, method_name, calibration["seeds"])
    calibration_split, test = run.splits["v_cal"], run.splits["test"]
    test_rows = np.arange(labels["test"].shape[0])

    if toy.report["task_type"] in CLASSIFICATION_KINDS:
        # The report measures every set score; a package ships LAC alone.
        assert "aps" in SET_SCORES and conformal["set_scores"] == ["lac"]
        assert all(set(entry) == {"level", "lac"} for entry in conformal["quantiles"])
        expected = classification_conformal_rows(calibration_split.probs, labels["v_cal"], test.probs, labels["test"], levels)
        for entry in conformal["quantiles"]:
            level = entry["level"]
            for score in conformal["set_scores"]:
                members = class_scores(test.probs, score) <= entry[score]  # (30, c)
                assert np.array_equal(members, prediction_sets(calibration_split.probs, labels["v_cal"], test.probs, level, score))
                covered = members[test_rows, labels["test"]].astype(np.float64)  # (30,)
                assert np.array_equal(covered, expected[f"{score}_coverage@{level}"])
                assert metrics["test"][f"{score}_coverage@{level}"] == pytest.approx(covered.mean())
        return

    # The floor rule of regression_conformal_rows: 1e-3 of the training labels' spread with a
    # sigma, a unit floor on the absolute residual without one.
    has_sigma = test.variance is not None
    assert conformal["spread"] == ("sigma" if has_sigma else "none")
    expected_floor = 1e-3 * float(np.std(toy.train_labels.astype(np.float64))) if has_sigma else 1.0
    assert conformal["spread_floor"] == pytest.approx(expected_floor, rel=1e-9)
    calibration_sigma = np.sqrt(calibration_split.variance) if has_sigma else None
    test_sigma = np.sqrt(test.variance) if has_sigma else None
    expected = regression_conformal_rows(
        calibration_split.mean, labels["v_cal"], calibration_sigma, test.mean, labels["test"], test_sigma, levels, conformal["train_label_std"]
    )
    for entry in conformal["quantiles"]:
        level = entry["level"]
        interval = regression_interval(test.mean, test_sigma, entry["quantile"], conformal["spread_floor"], level)
        assert np.array_equal(interval.covers(labels["test"]).astype(np.float64), expected[f"conformal_coverage@{level}"])
        assert np.array_equal(interval.width, expected[f"conformal_width@{level}"])
        assert metrics["test"][f"conformal_coverage@{level}"] == pytest.approx(expected[f"conformal_coverage@{level}"].mean())

    # An interval has one width for every row exactly when the method has no spread of its own.
    head = load_package(str(tmp_path / "package"))
    assert head.fixed_width_interval == (float(np.ptp(interval.width)) <= 1e-9 * float(interval.width.max()))


def test_a_score_only_method_and_a_cell_without_heads_are_refused(cells, tmp_path):
    _, finished = cells
    source = finished["binary_sngp"].cell.output_dir
    for method_name in ("sngp_entropy", "sngp_nn_distance", "sngp_variance"):
        with pytest.raises(ValueError, match="only ranks rows by another score"):
            build_package(source, method_name, str(tmp_path / method_name))
    with pytest.raises(ValueError, match="no shippable method 'mc_dropout'"):
        build_package(source, "mc_dropout", str(tmp_path / "mc_dropout"))
    # V_cal's 16 rows cannot express 95% coverage, whose quantile is infinite.
    with pytest.raises(ValueError, match=r"cannot calibrate levels \[0\.95\]"):
        build_package(source, "sngp", str(tmp_path / "too_high"), levels=[0.8, 0.95])

    headless = tmp_path / "headless"
    shutil.copytree(source, headless)
    record = read_json(str(headless / "cell.json"))
    record["uncertainty"]["save_heads"] = False
    write_json(str(headless / "cell.json"), record)
    with pytest.raises(ValueError, match="ran without uncertainty.save_heads"):
        build_package(str(headless), "sngp", str(tmp_path / "from_headless"))

    missing = tmp_path / "missing_head"
    shutil.copytree(source, missing)
    shutil.move(missing / "heads" / "seed_43", tmp_path / "seed_43_head")
    with pytest.raises(ValueError, match=r"seeds without a saved head \[43\]"):
        build_package(str(missing), "sngp", str(tmp_path / "from_missing"))

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        build_package(source, "sngp", str(existing))

    # Nothing refused left a package or a staging directory behind.
    assert sorted(os.listdir(tmp_path)) == ["existing", "headless", "missing_head", "seed_43_head"]
    assert os.listdir(existing) == []


def test_the_package_command_writes_a_described_package(cells, tmp_path):
    _, finished = cells
    toy = finished["binary_sngp"]
    revisions = tmp_path / "revisions.json"
    revisions.write_text(json.dumps(REVISIONS), encoding="utf-8")
    output = tmp_path / "package"

    uncertainty_main([
        "package", toy.cell.output_dir, "--method", "sngp_ensemble_temperature", "--output", str(output),
        "--levels", "0.8", "--revisions", str(revisions),
    ])

    assert sorted(os.listdir(output)) == [
        "README.md", "calibration.json", "features.json", "heads", "metrics.json", "origin.json", "standardizer.npz",
    ]
    assert sorted(os.listdir(output / "heads")) == [f"seed_{seed}" for seed in SEEDS]
    calibration = read_json(str(output / "calibration.json"))
    assert calibration["scheme"] == "validation_carve" and calibration["method"] == "sngp_ensemble_temperature"
    assert calibration["seeds"] == SEEDS and calibration["seed_choice"] == {"rule": "every_seed"}
    assert (calibration["predictive"], calibration["calibration"]) == ("ensemble", "probability_temperature")
    assert set(calibration["fitted"]) == {"temperature"} and calibration["mc_dropout_samples"] is None
    assert calibration["rows"] == {"v_fit": 24, "v_cal": 16}
    split = Path(toy.cell.output_dir) / "split.json"
    assert calibration["split_sha256"] == hashlib.sha256(split.read_bytes()).hexdigest()
    assert [entry["level"] for entry in calibration["conformal"]["quantiles"]] == [0.8]

    features = read_json(str(output / "features.json"))
    assert features["width"] == features["input_size"] == SAE_WIDTH + DENSE_WIDTH and features["pair"] is False
    assert [branch["spec"] for branch in features["branches"]] == list(toy.cell.uncertainty.branch_specs)
    assert [branch["width"] for branch in features["branches"]] == [SAE_WIDTH, DENSE_WIDTH]
    assert (features["max_length"], features["embedding_dtype"], features["standardizer"]) == (64, "bfloat16", "standardizer.npz")
    assert (features["input_clip"], features["projection_size"], features["head"]) == (10.0, 32, "sngp")
    assert {(entry["repository"], entry["revision"]) for entry in features["backbones"]} == set(REVISIONS.items())

    origin = read_json(str(output / "origin.json"))
    assert (origin["cell_id"], origin["experiment_id"], origin["shipped_seeds"]) == ("binary_sngp", "toy_experiment", SEEDS)
    assert origin["dataset"] == {"name": toy.report["data_name"], "revision": None}
    assert origin["split"] == toy.report["split"] and origin["seeds"] == SEEDS
    assert origin["protify_revision"]["kind"] in ("git_commit", "source_tree_sha256")

    metrics = read_json(str(output / "metrics.json"))
    assert metrics["levels"] == [0.8] and "lac_coverage@0.8" in metrics["test"] and "lac_coverage@0.9" not in metrics["test"]
    card = (output / "README.md").read_text(encoding="utf-8")
    assert "# binary_sngp: sngp_ensemble_temperature" in card and "Warning" not in card
    # Read as bytes, since text mode would turn CRLF into LF: no em dash and LF endings everywhere.
    card_bytes = (output / "README.md").read_bytes()
    em_dash = chr(0x2014).encode()
    assert em_dash not in card_bytes and b"\r" not in card_bytes

    # Without a revisions file every revision is unknown, and the card says so.
    build_package(toy.cell.output_dir, "sngp", str(tmp_path / "unpinned"))
    unpinned = read_json(str(tmp_path / "unpinned" / "features.json"))
    assert all(entry["revision"] is None for entry in unpinned["backbones"])
    assert "Warning: the revision of" in (tmp_path / "unpinned" / "README.md").read_text(encoding="utf-8")
    # A revisions file that misses a repository the features were embedded with is refused.
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"Synthyra/ESM2-8M": "c" * 40}), encoding="utf-8")
    with pytest.raises(ValueError, match="names no revision for"):
        build_package(toy.cell.output_dir, "sngp", str(tmp_path / "partial"), revisions_path=str(partial))


def test_a_pair_package_joins_two_proteins_by_the_cells_rule():
    mean = np.arange(6, dtype=np.float32)  # (6,)
    scale = np.full(6, 2.0, dtype=np.float32)  # (6,)
    pair = PackagedHead("unused", {}, {"width": 3, "pair": True, "pair_feature_mode": "minmax"}, {}, EmbeddingStandardizer(mean, scale))
    first = np.array([[1.0, 5.0, 3.0]], dtype=np.float32)  # (1, 3)
    second = np.array([[4.0, 2.0, 3.0]], dtype=np.float32)  # (1, 3)

    joined = pair.model_input(first, second)  # (1, 6)

    expected = (np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 3.0]], dtype=np.float32) - mean) / scale  # (1, 6)
    assert np.array_equal(joined, expected)
    # minmax sorts each feature across the pair, so the order the proteins arrive in is moot.
    assert np.array_equal(pair.model_input(second, first), joined)
    with pytest.raises(ValueError, match="second protein"):
        pair.model_input(first)

    single = PackagedHead("unused", {}, {"width": 3, "pair": False}, {}, None)
    assert np.array_equal(single.model_input(first), first)
    with pytest.raises(ValueError, match="no partner"):
        single.model_input(first, second)
    with pytest.raises(ValueError, match=r"\(rows, 3\)"):
        single.model_input(np.zeros((2, 4)))
