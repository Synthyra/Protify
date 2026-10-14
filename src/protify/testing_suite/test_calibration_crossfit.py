"""Compact cross-fitting and correctness calibration with the test payload sealed.

Small real fold heads exercise the statistical path. These fixtures qualify split access
and saved-file integrity, not homology, production accuracy, or large-corpus memory use.
"""

import json
import shutil
import sys
import numpy as np
import pytest
import torch

from pathlib import Path

from fastplms.features import FeatureStore
from foundry.digests import sha256_file
from foundry.eval.temperature import fit_temperature
from protify.data import declared_dataset
from protify.probes.projected_probe import ProjectedProbe
from protify.testing_suite.embedding_store_builder import write_store, without_rows
from protify.testing_suite.test_declared_dataset import compact_release, release_fixture, rewrite_rows
from protify.testing_suite.test_uncertainty_workflow import _cell_mapping
from protify.uncertainty import crossfit, package_correctness, workflow
from protify.uncertainty.__main__ import parse_arguments
from protify.uncertainty.applicability import decision_policy
from protify.uncertainty.cell import cell_from_mapping
from protify.uncertainty.correctness import ShippedCorrectness
from protify.uncertainty.methods import probabilities
from protify.uncertainty.package import build_package
from protify.uncertainty.packaged import PackagedHead


def _stores(root: Path) -> list[Path]:
    """The feature stores under a cache root, in a fixed order."""
    return sorted(path for path in Path(root).iterdir() if (path / "feature.json").is_file())


@pytest.mark.parametrize("splits", [(), ("train", "train"), ("train", "typo"), "train"])
def test_declared_loader_refuses_ambiguous_split_scope(tmp_path, splits):
    with pytest.raises(ValueError, match="split_names"):
        declared_dataset.load_declared_dataset(tmp_path / "absent.json", "a" * 64, split_names=splits)


def test_selected_splits_do_not_stat_or_read_excluded_test(tmp_path, monkeypatch):
    path, digest, manifest, _ = release_fixture(tmp_path)
    (path.parent / manifest["splits"]["test"]["path"]).unlink()
    artifact = declared_dataset._artifact
    opened = []

    def checked(root, record):
        opened.append(record["path"])
        assert record != manifest["splits"]["test"]
        return artifact(root, record)

    monkeypatch.setattr(declared_dataset, "_artifact", checked)
    release = declared_dataset.load_declared_dataset(path, digest, split_names=("train", "validation", "calibration"))
    assert set(release.splits) == {"train", "validation", "calibration"}
    assert opened == ["train.parquet", "validation.parquet", "calibration.parquet"]
    assert release.training_record()["splits"] == manifest["splits"]  # Identity is not an access claim.


@pytest.fixture(scope="module", params=["binary", "binary_second_seed", "multiclass", "multilabel", "multilabel_compact"])
def calibrated(tmp_path_factory, request):
    root = tmp_path_factory.mktemp(f"calibration_{request.param}")
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        patch.setattr(sys.modules[workflow.parse_arguments.__module__], "load_secrets_env", lambda: [])
        kind = request.param.removesuffix("_compact").removesuffix("_second_seed")
        second_seed = request.param.endswith("_second_seed")
        path, digest, manifest, rows = release_fixture(root, kind)
        for index, row in enumerate(rows["train"]):
            row["cluster_id"] = f"family_{index // 4}"
        digest = rewrite_rows(path, manifest, "train", rows["train"])
        if request.param.endswith("_compact"):
            digest = compact_release(path, manifest, rows)
        mapping = _cell_mapping(root, "unused", "validation", num_runs=2 if second_seed else 1)
        mapping["protify_args"].pop("data_dirs")
        mapping["protify_args"]["balanced_regression_metrics"] = False
        mapping["uncertainty"].update(
            head="projected", dataset_manifest=str(path), dataset_sha256=digest, save_heads=True,
            require_cached_embeddings=True, evaluation_mode="validation_only", dropout=0.0,
        )
        cell = cell_from_mapping(mapping)
        workflow.run_uncertainty_cell(cell)
        before = {p.relative_to(cell.output_dir).as_posix(): sha256_file(p) for p in Path(cell.output_dir).rglob("*") if p.is_file()}

        # The fit is finished; calibration has neither the test payload nor its embeddings.
        (path.parent / manifest["splits"]["test"]["path"]).unlink()
        every_sequence = [row["seqs"] for split_rows in rows.values() for row in split_rows]
        for store in _stores(root / "embeddings"):
            without_rows(store, [row["seqs"] for row in rows["test"]], every_sequence)
        train = rows["train"]
        proteins = [row["protein_id"] for row in train]
        clusters = {row["protein_id"]: row["cluster_id"] for row in train}
        assignments = crossfit.assign_folds(clusters, proteins, 5, seed=7)
        plan = root / "plan.json"
        crossfit.write_plan(str(plan), crossfit.plan_document(["0.3"], 5, 7, {"0.3": assignments}))
        output = root / "crossfit"
        served_seed = 43 if second_seed else 42
        record = crossfit.crossfit_cell(cell.output_dir, str(plan), str(output), m8_signals=False,
                                        calibration_only=True, served_seed=served_seed)
        after = {p.relative_to(cell.output_dir).as_posix(): sha256_file(p) for p in Path(cell.output_dir).rglob("*") if p.is_file()}
        assert before == after
        with np.load(output / "served.npz", allow_pickle=False) as archive:
            served = {name: archive[name] for name in archive.files}
        temperature = fit_temperature(served["v_fit__outputs"], served["v_fit__labels"], multiclass=kind != "multilabel")
        probs = {"v_cal": probabilities(served["v_cal__outputs"], kind, temperature=temperature)}
        labels = {"v_cal": served["v_cal__labels"]}
        fitted = package_correctness.package_correctness(
            str(output), kind=kind, num_labels=len(manifest["label_vocabulary"]), shipped_seeds=[served_seed],
            probs=probs, labels=labels, at_least_one_label=kind == "multilabel", calibration_only=True,
        )
        yield cell, output, record, plan, fitted, temperature, kind, probs, labels


def test_real_five_fold_calibration_has_no_test_predictions(calibrated):
    cell, output, record, _, fitted, _, _, _, _ = calibrated
    assert record["status"] == "complete"
    assert record["evaluation_scope"] == "calibration_only"
    assert record["evaluation_splits"] == ["v_fit", "v_cal"]
    assert record["agreement_splits"] == ["v_fit"]
    assert record["split_access"] == {"loaded": ["train", "validation", "calibration"], "excluded": ["test"], "membership_checks": "loaded_splits_only"}
    assert record["served_correctness"]["check_split"] == fitted.check_split == "v_cal"
    assert set(fitted.metrics) == set(fitted.reliability) == {"v_cal"}
    assert fitted.rule["score"] == "correctness"
    # V_cal alone calibrates the novelty cutoff, and no test row is scored under the policy.
    rows = fitted.reliability["v_cal"].shape[0]
    assert fitted.applicability["status"] == ("calibrated" if rows * 0.05 >= 5 else "unavailable")
    assert fitted.policy == decision_policy(fitted.rule, fitted.applicability)
    assert fitted.evidence == {} and fitted.check_novelty.shape == fitted.check_rows.shape
    if fitted.applicability["status"] == "calibrated":
        assert fitted.applicability["reference_sha256"] == fitted.reference.content_sha256()
    held = []
    for fold in range(5):
        assert record["folds"][f"l03_f{fold}"]["training"]["seed"] == record["served_seed"]
        assert record["folds"][f"l03_f{fold}"]["sampling"]
        with np.load(output / f"l03_f{fold}.npz", allow_pickle=False) as archive:
            held.extend(archive["held__rows"].tolist())
            np.testing.assert_array_equal(archive["held__rows"], archive["correctness__rows"])
            assert not any(name.startswith("test__") for name in archive.files)
    assert sorted(held) == list(range(record["train_rows"]))
    for name in ("served.npz", "served_correctness.npz"):
        with np.load(output / name, allow_pickle=False) as archive:
            assert not any(key.startswith("test__") for key in archive.files)
    with pytest.raises(ValueError, match="separate frozen-checkpoint calibration and test stage"):
        build_package(cell.output_dir, "temperature", str(output / "forbidden_package"), crossfit_dir=str(output))


def test_calibration_correctness_survives_real_reload(calibrated, tmp_path):
    cell, _, record, _, fitted, temperature, kind, _, _ = calibrated
    directory = tmp_path / "correctness"
    package_correctness.write_correctness(directory, fitted)
    # A reference read back from disk is the one the package pins its novelty claims to.
    assert ShippedCorrectness.load(directory).reference.content_sha256() == fitted.reference.content_sha256()
    model = ProjectedProbe.from_pretrained(workflow.head_directory(cell.output_dir, record["served_seed"]))
    standardizer, _ = workflow.load_standardizer_record(str(Path(cell.output_dir) / workflow.STANDARDIZER_RECORD))
    # Exercise actual serving arithmetic, without creating a release package or test report.
    head = PackagedHead(
        path=str(tmp_path), calibration={
            "method": "temperature", "task_type": kind, "predictive": "head", "calibration": "logit_temperature",
            "fitted": {"temperature": temperature}, "conformal": None, "answer_rule": fitted.rule,
        }, features={"width": 40, "pair": False}, heads={record["served_seed"]: model}, standardizer=standardizer,
        correctness=ShippedCorrectness.load(directory),
    )
    checked = package_correctness.check_shipped_correctness(head, fitted)
    assert checked["rows"] == len(fitted.check_rows)
    assert checked["median"] <= 1e-3 and checked["max"] <= 0.1


def test_completed_calibration_resumes_without_training(calibrated, monkeypatch):
    cell, output, record, plan, *_ = calibrated
    monkeypatch.setattr(crossfit, "_train_head", lambda *args, **kwargs: pytest.fail("retrained a committed fold"))
    assert crossfit.crossfit_cell(cell.output_dir, str(plan), str(output), m8_signals=False,
                                  calibration_only=True, served_seed=record["served_seed"]) == record


@pytest.mark.parametrize(
    "target", ["head", "standardizer", "predictions", "feature", "feature_rows", "fold", "served", "reference"],
)
def test_changed_sources_and_outputs_cannot_resume(calibrated, tmp_path, target):
    cell, output, record, plan, *_ = calibrated
    store = _stores(cell.protify_args["embedding_save_dir"])[0]
    path = {
        "head": Path(workflow.head_directory(cell.output_dir, record["served_seed"])) / "config.json",
        "standardizer": Path(cell.output_dir) / "standardizer.npz",
        "predictions": Path(cell.output_dir) / f"seed_{record['served_seed']}.npz",
        # A part of a store whose bytes no longer match its commit marker.
        "feature": next((store / "segments").rglob("*.safetensors")),
        "fold": output / "l03_f0.npz", "served": output / "served.npz", "reference": output / "reference" / "reference.npz",
    }.get(target)
    message = "source files or feature caches changed|artifact digest mismatch"
    if target == "feature":
        message = "does not match its commit marker"
    backup = tmp_path / "store"
    if target == "feature_rows":
        # Another run committed a row to a cache the cross-fit was bound to.
        shutil.copytree(store, backup)
        spec = FeatureStore.read_only(store).spec
        write_store(store.parent, spec.descriptor, {"MKTAYIAKQR": torch.zeros(spec.width)})
    else:
        original = path.read_bytes()
        path.write_bytes(original + b" ")
    try:
        with pytest.raises(ValueError, match=message):
            crossfit.crossfit_cell(cell.output_dir, str(plan), str(output), m8_signals=False,
                                   calibration_only=True, served_seed=record["served_seed"])
    finally:
        if target == "feature_rows":
            shutil.rmtree(store)
            shutil.copytree(backup, store)
        else:
            path.write_bytes(original)


def test_calibration_cannot_be_reinterpreted_as_full_evaluation(calibrated):
    cell, output, _, plan, fitted, _, kind, probs, labels = calibrated
    with pytest.raises(ValueError, match="validation-only cell"):
        crossfit.crossfit_cell(cell.output_dir, str(plan), str(output), m8_signals=False)
    with pytest.raises(ValueError, match="evaluation scope"):
        package_correctness.package_correctness(str(output), kind=kind, num_labels=2, shipped_seeds=[42],
                                              probs=probs, labels=labels, at_least_one_label=kind == "multilabel")
    with pytest.raises(ValueError, match="V_cal probabilities and labels alone"):
        package_correctness.package_correctness(str(output), kind=kind, num_labels=2, shipped_seeds=[42],
                                              probs={**probs, "test": probs["v_cal"]}, labels=labels,
                                              at_least_one_label=kind == "multilabel", calibration_only=True)
    assert fitted.check_split == "v_cal"


def test_calibration_cli_scope_is_explicit():
    args = parse_arguments(["crossfit", "cell", "--plan", "plan.json", "--output", "output", "--correctness-only", "--calibration-only"])
    assert args.calibration_only and args.correctness_only


def test_correctness_cannot_replace_calibration_truth(calibrated):
    _, output, record, _, _, _, kind, probs, labels = calibrated
    with pytest.raises(ValueError, match="V_cal labels differ"):
        package_correctness.package_correctness(str(output), kind=kind, num_labels=record["num_labels"], shipped_seeds=[42],
                                              probs=probs, labels={"v_cal": labels["v_cal"] + 1},
                                              at_least_one_label=kind == "multilabel", calibration_only=True)


@pytest.mark.parametrize("change", ["wrong_folds", "wrong_level", "missing_protein", "empty_fold", "split_family"])
def test_calibration_rejects_incompatible_family_plans_before_training(calibrated, tmp_path, monkeypatch, change):
    cell, _, record, plan, *_ = calibrated
    document = json.loads(plan.read_text())
    assignments = document["assignments"]["0.3"]
    if change == "wrong_folds":
        document["folds"] = 6
    elif change == "wrong_level":
        document["levels"] = ["0.7"]
        document["assignments"] = {"0.7": assignments}
    elif change == "missing_protein":
        assignments.pop(next(iter(assignments)))
    elif change == "empty_fold":
        document["assignments"]["0.3"] = {protein: fold % 4 for protein, fold in assignments.items()}
    else:
        protein = next(iter(assignments))
        assignments[protein] = (assignments[protein] + 1) % 5
    changed = tmp_path / "changed_plan.json"
    crossfit.write_plan(str(changed), document)
    monkeypatch.setattr(crossfit, "_train_head", lambda *args, **kwargs: pytest.fail("trained before rejecting plan"))
    with pytest.raises(ValueError, match="five folds|exactly the declared|empty fold|splits a declared protein family"):
        crossfit.crossfit_cell(cell.output_dir, str(changed), str(tmp_path / "output"), m8_signals=False,
                               calibration_only=True, served_seed=record["served_seed"])


def test_interrupted_crossfit_reuses_only_committed_folds(calibrated, tmp_path, monkeypatch):
    cell, _, record, plan, *_ = calibrated
    output = tmp_path / "interrupted"
    train = crossfit._train_head
    calls = []

    def interrupt(*args, **kwargs):
        calls.append(args[6])
        if len(calls) == 2:
            raise RuntimeError("fixture interruption")
        return train(*args, **kwargs)

    monkeypatch.setattr(crossfit, "_train_head", interrupt)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        crossfit.crossfit_cell(cell.output_dir, str(plan), str(output), m8_signals=False,
                               calibration_only=True, served_seed=record["served_seed"])
    partial = json.loads((output / "crossfit.json").read_text())
    assert partial["status"] == "running" and set(partial["folds"]) == {"l03_f0"}
    first = sha256_file(output / "l03_f0.npz")

    def remaining(*args, **kwargs):
        calls.append(args[6])
        return train(*args, **kwargs)

    monkeypatch.setattr(crossfit, "_train_head", remaining)
    finished = crossfit.crossfit_cell(cell.output_dir, str(plan), str(output), m8_signals=False,
                                     calibration_only=True, served_seed=record["served_seed"])
    assert finished["status"] == "complete" and len(calls) == 6
    assert first == sha256_file(output / "l03_f0.npz")


@pytest.mark.parametrize("field", ["outputs", "labels", "label_mask", "observation_ids"])
def test_wrong_saved_validation_rows_fail_before_fold_training(calibrated, tmp_path, monkeypatch, field):
    cell, _, record, plan, *_ = calibrated
    path = Path(cell.output_dir) / f"seed_{record['served_seed']}.npz"
    original = path.read_bytes()
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    key = f"v_fit__{field}"
    if field == "outputs":
        arrays[key] = arrays[key] + 100.0
    elif field == "observation_ids":
        arrays[key] = np.roll(arrays[key], 1)
    elif field == "label_mask":
        arrays[key] = ~arrays[key]
    else:
        arrays[key] = arrays[key] + 1
    try:
        np.savez_compressed(path, **arrays)
        monkeypatch.setattr(crossfit, "_train_head", lambda *args, **kwargs: pytest.fail("trained before head parity"))
        with pytest.raises(ValueError, match="V_fit"):
            crossfit.crossfit_cell(cell.output_dir, str(plan), str(tmp_path / "wrong_rows"), m8_signals=False,
                                   calibration_only=True, served_seed=record["served_seed"])
    finally:
        path.write_bytes(original)
