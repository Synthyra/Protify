"""Cross-fitting a finished cell by protein cluster, and the signals recorded beside each head.

The pure pieces are checked against direct computations: the blocked neighbor search, the
Gaussian scores, fold assignment by cluster, and the identity a hit table gives each fold. Then a
tiny binary cell runs through the real workflow with `save_heads`, and `crossfit_cell` trains its
fold heads from a two-level plan on the toy caches `test_uncertainty_workflow.py` writes, with the
correctness model's columns, and a package ships the model fitted on them.
"""

import json
import sys
import numpy as np
import pytest
import torch

from types import SimpleNamespace


try:
    from src.protify.uncertainty import crossfit, workflow
    from src.protify.uncertainty.cell import cell_from_mapping
    from src.protify.uncertainty.identity_bins import sequence_digest
    from src.protify.uncertainty.report import write_json
except ImportError:
    from protify.uncertainty import crossfit, workflow
    from protify.uncertainty.cell import cell_from_mapping
    from protify.uncertainty.identity_bins import sequence_digest
    from protify.uncertainty.report import write_json

from .test_uncertainty_workflow import _cell_mapping, _toy_dataset, _write_caches


def test_folds_keep_each_cluster_whole_and_use_every_fold():
    proteins = [f"p{index}" for index in range(60)]
    cluster_of = {protein: f"c{index // 3}" for index, protein in enumerate(proteins)}
    folds = crossfit.assign_folds(cluster_of, proteins, 4, seed=0)
    for cluster in set(cluster_of.values()):
        assert len({folds[protein] for protein in proteins if cluster_of[protein] == cluster}) == 1
    assert set(folds.values()) == {0, 1, 2, 3}
    assert folds == crossfit.assign_folds(cluster_of, proteins, 4, seed=0)


def test_a_plan_refuses_unassigned_proteins_and_foreign_folds(tmp_path):
    path = tmp_path / "plan.json"
    crossfit.write_plan(str(path), crossfit.plan_document(["0.3"], 2, 0, {"0.3": {"a": 0, "b": 1}}))
    plan = crossfit.read_plan(str(path))
    np.testing.assert_array_equal(crossfit.held_out_rows(plan, "0.3", 1, ["a", "b", "a"]), [1])
    with pytest.raises(ValueError, match="assigns no fold"):
        crossfit.held_out_rows(plan, "0.3", 0, ["a", "c"])

    crossfit.write_plan(str(path), crossfit.plan_document(["0.3"], 2, 0, {"0.3": {"a": 0, "b": 2}}))
    with pytest.raises(ValueError, match="outside"):
        crossfit.read_plan(str(path))


def test_best_identity_reads_only_the_allowed_targets():
    queries, targets = np.array([0, 0, 1, 2]), np.array([1, 2, 2, 0])
    identities = np.array([0.9, 0.4, 0.7, 0.4], dtype=np.float32)
    allowed = np.array([True, False, True])
    np.testing.assert_allclose(crossfit.best_identity(queries, targets, identities, allowed, 3), [0.4, 0.7, 0.4])
    np.testing.assert_allclose(crossfit.best_identity(queries, targets, identities, np.zeros(3, bool), 3), [0, 0, 0])


@pytest.mark.parametrize("role", ["train", "valid", "test"])
@pytest.mark.parametrize("correctness_only", [False, True])
def test_partial_labels_stop_crossfit_before_features_training_or_resume(tmp_path, monkeypatch, role, correctness_only):
    rows = {"labels": [[1.0, 0.0]], "label_mask": [[True, True]]}
    bad = {**rows, "label_mask": [[True, False]]}
    cell_data = SimpleNamespace(train=rows, valid=rows, test=rows, ppi=False)
    setattr(cell_data, role, bad)
    cell = SimpleNamespace(uncertainty=SimpleNamespace(branch_specs=[]))
    monkeypatch.setattr(crossfit, "_finished_cell", lambda _: (cell, 0))
    monkeypatch.setattr(crossfit, "read_plan", lambda _: None)
    monkeypatch.setattr(crossfit, "cell_arguments", lambda *args: None)
    monkeypatch.setattr(crossfit, "load_split_cell_data", lambda *args: (cell_data, None))
    monkeypatch.setattr(crossfit, "load_cell_branches", lambda *args, **kwargs: pytest.fail("read features before rejecting partial truth"))
    output = tmp_path / "crossfit"
    with pytest.raises(ValueError, match="partial labels"):
        crossfit.crossfit_cell("cell", "plan", str(output), m8_signals=not correctness_only)
    assert not output.exists()


@pytest.mark.parametrize("role", ["reference", "query"])
def test_direct_signal_builder_refuses_partial_truth_before_model_execution(monkeypatch, role):
    complete = SimpleNamespace(labels=[[1.0, 0.0]], label_masks=[[True, True]])
    partial = SimpleNamespace(labels=[[1.0, 0.0]], label_masks=[[True, False]])
    monkeypatch.setattr(crossfit, "trunk_features", lambda *args, **kwargs: pytest.fail("model ran before the guard"))
    with pytest.raises(ValueError, match="partial labels"):
        crossfit.signal_arrays(None, partial if role == "reference" else complete, np.array([[1.0, 0.0]]),
                               {"query": partial if role == "query" else complete}, None, "multilabel", 2, 1, torch.device("cpu"))


def test_complete_explicit_masks_are_accepted_and_malformed_masks_fail():
    crossfit._require_complete_targets([[1.0, 0.0]], [[True, True]], "train")
    crossfit._require_complete_targets([1], [True], "train")
    crossfit._require_complete_targets([1], None, "train")
    for masks in ([[1, 0]], [[True]], []):
        with pytest.raises(ValueError):
            crossfit._require_complete_targets([[1.0, 0.0]], masks, "train")


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    root = tmp_path_factory.mktemp("crossfit")
    with pytest.MonkeyPatch.context() as patch:
        # Hugging Face writes checkpoints under the working directory; nothing may land in the tree.
        patch.chdir(root)
        patch.setattr(sys.modules[workflow.parse_arguments.__module__], "load_secrets_env", lambda: [])
        frames, data_dir = _toy_dataset(root)
        _write_caches(frames, root / "embeddings")
        mapping = _cell_mapping(root, data_dir, "binary_projected", 1)
        mapping["uncertainty"].update(head="projected", save_heads=True, mc_dropout_samples=0)
        cell = cell_from_mapping(mapping)
        workflow.run_uncertainty_cell(cell)

        train_proteins = [sequence_digest(sequence) for sequence in frames["train"]["seqs"]]
        clusters = {"0.3": {protein: f"c{index // 4}" for index, protein in enumerate(train_proteins)},
                    "0.7": {protein: f"c{index // 2}" for index, protein in enumerate(train_proteins)}}
        levels = list(clusters)
        assignments = {level: crossfit.assign_folds(clusters[level], train_proteins, 2, seed=0) for level in levels}
        plan_path = root / "plan.json"
        crossfit.write_plan(str(plan_path), crossfit.plan_document(levels, 2, 0, assignments))
        output = root / "crossfit"
        record = crossfit.crossfit_cell(cell.output_dir, str(plan_path), str(output))
        yield cell, output, record, len(train_proteins), plan_path


def test_every_training_row_is_held_out_once_per_level(finished):
    _, output, record, train_rows, _ = finished
    assert sorted(record["folds"]) == ["l03_f0", "l03_f1", "l07_f0", "l07_f1"]
    for level in ("03", "07"):
        held = np.concatenate([np.load(output / f"l{level}_f{fold}.npz")["held__rows"] for fold in (0, 1)])
        np.testing.assert_array_equal(np.sort(held), np.arange(train_rows))


def test_a_fold_head_never_finds_its_own_held_out_rows_as_neighbors(finished):
    _, output, _, _, _ = finished
    for fold in ("l03_f0", "l07_f1"):
        arrays = np.load(output / f"{fold}.npz")
        held = set(arrays["held__rows"].tolist())
        for split in ("held", "v_fit", "v_cal", "test"):
            for space in ("trunk", "input"):
                assert not held & set(arrays[f"{split}__{space}_neighbors"].ravel().tolist())
                assert np.all(np.diff(arrays[f"{split}__{space}_distances"], axis=1) >= -1e-5)
        assert arrays["held__outputs"].shape == (len(held), 2)
        assert arrays["held__maha_class"].shape == (len(held), 2)


def test_the_served_head_reproduces_its_cell_and_a_rerun_resumes(finished, monkeypatch):
    cell, output, record, _, plan_path = finished
    assert record["served_output_agreement"] <= 1e-3
    served = np.load(output / "served.npz")
    saved = np.load(f"{cell.output_dir}/seed_42.npz")
    np.testing.assert_allclose(served["test__outputs"], saved["test__outputs"], atol=1e-3)
    assert served["test__trunk_neighbors"].shape[1] == min(crossfit.NEIGHBORS, served["train__labels"].shape[0])

    monkeypatch.setattr(crossfit, "_train_head", lambda *arguments, **options: (_ for _ in ()).throw(AssertionError("retrained")))
    assert crossfit.crossfit_cell(cell.output_dir, str(plan_path), str(output))["folds"] == record["folds"]


def test_each_fold_records_the_correctness_columns_of_its_held_out_rows(finished):
    _, output, _, _, _ = finished
    for fold in ("l03_f0", "l03_f1"):
        arrays = np.load(output / f"{fold}.npz")
        rows = arrays["correctness__rows"]
        np.testing.assert_array_equal(rows, arrays["held__rows"])  # every held-out row, under the cap
        for name in ("confidence", "correct", "maha_global", "maha_answer", "rmd", "ddu", "trunk_d1", "saebin_trust", "inpca_agree"):
            assert arrays[f"correctness__{name}"].shape == rows.shape
        assert np.all((arrays["correctness__confidence"] >= 0.5) & (arrays["correctness__confidence"] <= 1.0))


def test_the_served_head_writes_its_reference_and_signals(finished):
    cell, output, record, train_rows, _ = finished
    assert record["served_seed"] == 42 and record["plan"] == {"levels": ["0.3", "0.7"], "folds": 2}
    assert record["served_correctness"]["reference_rows"] == train_rows
    served = np.load(output / crossfit.SERVED_CORRECTNESS)
    saved = np.load(f"{cell.output_dir}/seed_42.npz")
    for split in crossfit.EVALUATION_SPLITS:
        np.testing.assert_array_equal(served[f"{split}__answers"], saved[f"{split}__outputs"].argmax(axis=1))
        assert served[f"{split}__trunk_d1"].shape == served[f"{split}__answers"].shape
    assert served["check__features"].shape == (served["check__rows"].size, 40)
    assert (output / crossfit.REFERENCE_DIRECTORY / "reference.npz").exists()


def test_a_package_ships_the_correctness_model_and_reproduces_it(finished, tmp_path):
    cell, output, _, _, _ = finished
    try:
        from src.protify.uncertainty.package import build_package
        from src.protify.uncertainty.packaged import load_package
    except ImportError:
        from protify.uncertainty.package import build_package
        from protify.uncertainty.packaged import load_package

    target = build_package(cell.output_dir, "temperature", str(tmp_path / "package"), crossfit_dir=str(output))
    head = load_package(target)
    assert head.correctness is not None and head.answer_rule["score"] == "correctness"
    check = json.loads((tmp_path / "package" / "metrics.json").read_text(encoding="utf-8"))["correctness"]["check"]
    assert check["median"] <= 1e-3
    served = np.load(output / crossfit.SERVED_CORRECTNESS)
    prediction = head.predict(served["check__features"])
    assert prediction.reliability.shape == prediction.confidence.shape
    # The toy's out-of-fold rows separate perfectly, so the model saturates; real ones do not.
    assert np.all((prediction.reliability >= 0.0) & (prediction.reliability <= 1.0))
    assert set(prediction.signals) >= set(head.correctness.model.names)
    np.testing.assert_array_equal(prediction.answers, served["test__answers"][served["check__rows"]])

    # The package pins its novelty claims and its decision policy, and decides a request from them.
    calibration = json.loads((tmp_path / "package" / "calibration.json").read_text(encoding="utf-8"))
    metrics = json.loads((tmp_path / "package" / "metrics.json").read_text(encoding="utf-8"))
    assert head.applicability == calibration["applicability"] and head.decision_policy == calibration["decision_policy"]
    assert head.decision_policy["selection"]["score"] == "correctness"
    decisions = head.decide(prediction)
    assert len(decisions.actions) == prediction.answers.shape[0]
    assert set(decisions.actions) <= {"answer", "abstain", "not_assessed"} and all(decisions.reasons)
    pooled, alone = metrics["decision_policy"]["decisions"], metrics["decision_policy"]["single_row_decisions"]
    assert pooled["rows"] == alone["rows"] == int(served["test__answers"].shape[0])
    assert alone["answered"] >= pooled["answered"], "a row answered in a request is answered alone"
    card = (tmp_path / "package" / "README.md").read_text(encoding="utf-8")
    assert "decided as one request" in card and "Decided one row per request" in card
    if head.applicability["status"] == "calibrated":
        assert head.applicability["reference_sha256"] == head.correctness.reference.content_sha256()
        assert "novelty_median" in check

    without = build_package(cell.output_dir, "temperature", str(tmp_path / "plain"))
    plain = load_package(without)
    assert plain.correctness is None
    assert plain.applicability == {"scheme": "latent_distance_applicability_v1", "status": "unavailable", "reason_code": "no_latent_reference"}
    assert plain.novelty(plain.predict(served["check__features"])) is None


def test_another_plan_is_refused_in_the_same_directory(finished, tmp_path):
    cell, output, _, _, _ = finished
    other = tmp_path / "other.json"
    write_json(str(other), crossfit.plan_document(["0.3"], 3, 1, {"0.3": {}}))
    with pytest.raises(ValueError, match="another plan"):
        crossfit.crossfit_cell(cell.output_dir, str(other), str(output))
