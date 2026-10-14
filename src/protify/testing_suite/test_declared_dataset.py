"""Frozen release inputs retain their task meaning and four partition assignments."""

import copy
import hashlib
import json
import os
import sys
import tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from argparse import Namespace
from collections.abc import Iterator
from pathlib import Path
from datasets import config as dataset_config

from foundry.digests import sha256_file
from protify.data.declared_dataset import SCHEMA, SPARSE_ENCODING, SPARSE_SCHEMA, SPLIT_NAMES, load_declared_dataset
from protify.data.label_batches import SparseLabelRow
from protify.testing_suite import test_uncertainty_workflow as fixture
from protify.uncertainty.cell import cell_from_mapping
from protify.uncertainty.methods import probabilities
from protify.uncertainty.metric_contracts import canonical_task_metrics
from protify.uncertainty.package import build_package
from protify.uncertainty.packaged import load_package
from protify.uncertainty.predictions import load_seed_predictions, save_seed_predictions


@pytest.fixture(autouse=True)
def isolated_arrow_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with tempfile.TemporaryDirectory(prefix="pd") as directory:
        monkeypatch.setattr(dataset_config, "HF_DATASETS_CACHE", Path(directory))
        yield


def write_manifest(path: Path, manifest: dict) -> str:
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return sha256_file(path)


def release_fixture(root: Path, kind: str = "binary", classes: int = 3):
    frames, _ = fixture._toy_dataset(root)
    fixture._write_caches(frames, root / "embeddings")
    splits = {"train": frames["train"], "validation": frames["valid"].iloc[:20], "calibration": frames["valid"].iloc[20:], "test": frames["test"]}
    directory = root / "release"
    directory.mkdir()
    manifest = {
        "schema": SCHEMA, "dataset_id": "typed_fixture", "endpoint": "fixture", "task_type": kind,
        "units": "fixture_units", "target_transform": {"name": "identity", "source_units": "fixture_units", "inverse": "identity"},
        "label_vocabulary": ["negative", "positive"] if kind == "binary" else ["target"] if kind == "regression" else ["first", "second", "third"][:classes],
        "evidence_policy_sha256": "e" * 64, "split_audit_sha256": "a" * 64, "splits": {},
    }
    rows = {}
    for name, frame in splits.items():
        rows[name] = []
        for index, record in enumerate(frame.to_dict("records")):
            value = int(record["labels"])
            if kind == "multilabel":
                value = [value, 1 - value, int(index % 2)]
            elif kind == "multiclass":
                value = index % classes
            elif kind == "regression":
                value = float(value) + index * 0.01
            sequence = record["seqs"]
            rows[name].append({
                "observation_id": f"{name}_{index}", "protein_id": hashlib.sha256(sequence.encode()).hexdigest(),
                "cluster_id": f"{name}_cluster_{index}", "seqs": sequence,
                "labels": value, "label_mask": [True] * 3 if kind == "multilabel" else True,
            })
        path = directory / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(rows[name]), path)
        manifest["splits"][name] = {"path": path.name, "sha256": sha256_file(path), "rows": len(rows[name])}
    path = directory / "dataset.json"
    digest = write_manifest(path, manifest)
    return path, digest, manifest, rows


def rewrite_rows(path: Path, manifest: dict, name: str, rows: list[dict]) -> str:
    artifact = path.parent / manifest["splits"][name]["path"]
    pq.write_table(pa.Table.from_pylist(rows), artifact)
    manifest["splits"][name].update(sha256=sha256_file(artifact), rows=len(rows))
    return write_manifest(path, manifest)


def compact_release(path, manifest, rows):
    manifest.update(schema=SPARSE_SCHEMA, label_encoding=SPARSE_ENCODING)
    for name, records in rows.items():
        compact = copy.deepcopy(records)
        for row in compact:
            observed = {index: value for index, (value, measured) in enumerate(zip(row["labels"], row["label_mask"], strict=True)) if measured}
            row["labels"], row["label_mask"] = SparseLabelRow.from_observed(len(row["labels"]), observed).to_records()
        digest = rewrite_rows(path, manifest, name, compact)
    return digest


@pytest.mark.parametrize("change", ["none", "width", "dense_v2", "sparse_v1", "encoding", "scalar"])
def test_versioned_sparse_release_needs_explicit_encoding_and_frozen_width(tmp_path, change):
    path, _, manifest, rows = release_fixture(tmp_path, "multilabel")
    digest = compact_release(path, manifest, rows)
    release = load_declared_dataset(path, digest)
    assert release.training_record()["label_encoding"] == SPARSE_ENCODING
    assert release.num_labels == 3
    if change == "none":
        assert release.splits["train"][0]["labels"]["width"] == 3
        return
    if change == "width":
        changed = list(release.splits["train"])
        changed[0]["labels"]["width"] = 4
        digest = rewrite_rows(path, manifest, "train", changed)
    elif change == "dense_v2":
        digest = rewrite_rows(path, manifest, "train", rows["train"])
    elif change == "sparse_v1":
        manifest["schema"] = SCHEMA
        manifest.pop("label_encoding")
        digest = write_manifest(path, manifest)
    elif change == "encoding":
        manifest["label_encoding"] = "dense"
        digest = write_manifest(path, manifest)
    else:
        manifest["task_type"] = "binary"
        digest = write_manifest(path, manifest)
    with pytest.raises(ValueError):
        load_declared_dataset(path, digest)


@pytest.mark.parametrize("value", [0.5, 1 + 1e-9, -1.0])
def test_sparse_binary_truth_is_validated_before_float32_conversion(tmp_path, value):
    path, _, manifest, rows = release_fixture(tmp_path, "multilabel")
    digest = compact_release(path, manifest, rows)
    release = load_declared_dataset(path, digest)
    changed = list(release.splits["train"])
    row = next(row for row in changed if row["labels"]["values"])
    row["labels"]["values"][0] = value
    digest = rewrite_rows(path, manifest, "train", changed)
    with pytest.raises(ValueError, match="zero or one"):
        load_declared_dataset(path, digest)


def test_sparse_release_trains_calibrates_and_resumes_like_dense_observed_truth(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path, _, manifest, rows = release_fixture(tmp_path, "multilabel")
    for name, records in rows.items():
        for index, row in enumerate(records):
            row["label_mask"][2] = bool(row["labels"][2])  # Positive-only third label.
            if index == 0:
                row["labels"], row["label_mask"] = [0, 0, 0], [False, False, False]
            if index == 1:
                row["labels"][0], row["label_mask"][0] = 0, True  # Measured zero.
        digest = rewrite_rows(path, manifest, name, records)
    outputs = []
    for storage in ("dense", "compact"):
        if storage == "compact":
            digest = compact_release(path, manifest, rows)
        mapping = fixture._cell_mapping(tmp_path, "unused", f"observed_{storage}", num_runs=1)
        mapping["protify_args"].pop("data_dirs")
        mapping["protify_args"]["balanced_regression_metrics"] = False
        mapping["uncertainty"].update(head="projected", dataset_manifest=str(path), dataset_sha256=digest,
                                       save_heads=True, require_cached_embeddings=True, dropout=0.0)
        cell = cell_from_mapping(mapping)
        report = fixture.workflow.run_uncertainty_cell(cell)
        saved = load_seed_predictions(str(Path(cell.output_dir) / "seed_42.npz"))
        for role, split in (("v_fit", "validation"), ("v_cal", "calibration"), ("test", "test")):
            np.testing.assert_array_equal(saved[role].labels, [row["labels"] for row in rows[split]])
            np.testing.assert_array_equal(saved[role].label_mask, [row["label_mask"] for row in rows[split]])
            np.testing.assert_array_equal(saved[role].observation_ids, [row["observation_id"] for row in rows[split]])
        assert fixture.workflow.run_uncertainty_cell(cell) == report
        directory = build_package(cell.output_dir, "temperature", str(tmp_path / f"package_{storage}"))
        package = load_package(directory)
        assert package.answer_rule is None
        assert package.observation_contract["whole_answer"] == "unavailable_partial_calibration"
        support = package.observation_contract["support"]["v_cal"]["labels"][2]
        assert support["negative"] == 0 and support["discrimination_status"] == "insufficient_observed_classes"
        outputs.append((report, saved))
    assert outputs[0][0]["methods"] == outputs[1][0]["methods"]
    for role in ("v_fit", "v_cal", "test"):
        np.testing.assert_array_equal(outputs[0][1][role].outputs, outputs[1][1][role].outputs)


@pytest.mark.parametrize("storage", ["dense", "compact"])
def test_a_positive_only_release_read_as_a_closed_world_trains_on_every_label(tmp_path, monkeypatch, storage):
    """Unlisted labels are negatives under `unlisted_labels: negative`, so the cell scores and calibrates on complete truth."""
    monkeypatch.chdir(tmp_path)
    path, digest, manifest, rows = release_fixture(tmp_path, "multilabel")
    for name, records in rows.items():
        for row in records:
            row["label_mask"] = [bool(value) for value in row["labels"]]  # positive-only: only a listed label is observed
        digest = rewrite_rows(path, manifest, name, records)
    if storage == "compact":
        digest = compact_release(path, manifest, rows)

    def run(reading):
        mapping = fixture._cell_mapping(tmp_path, "unused", f"closed_{storage}_{reading}", num_runs=1)
        mapping["protify_args"].pop("data_dirs")
        mapping["protify_args"]["balanced_regression_metrics"] = False
        mapping["uncertainty"].update(head="projected", dataset_manifest=str(path), dataset_sha256=digest, save_heads=True,
                                      require_cached_embeddings=True, dropout=0.0, unlisted_labels=reading)
        cell = cell_from_mapping(mapping)
        return cell, fixture.workflow.run_uncertainty_cell(cell)

    cell, report = run("negative")
    saved = load_seed_predictions(str(Path(cell.output_dir) / "seed_42.npz"))
    for role, split in (("v_fit", "validation"), ("v_cal", "calibration"), ("test", "test")):
        np.testing.assert_array_equal(saved[role].labels, [row["labels"] for row in rows[split]])  # the listed labels stay positive
        assert saved[role].label_mask.all() and saved[role].label_mask.shape == saved[role].labels.shape
    unknown_cell, unknown = run("unknown")
    unknown_saved = load_seed_predictions(str(Path(unknown_cell.output_dir) / "seed_42.npz"))
    assert not unknown_saved["test"].label_mask.all(), "the default reading still leaves unlisted labels unobserved"
    assert report["dataset_release"] == unknown["dataset_release"], "the release identity is the same under both readings"
    observations = json.loads((Path(cell.output_dir) / "label_observations.json").read_text(encoding="utf-8"))
    assert all(record["unobserved_labels"] == 0 for record in observations["splits"].values())
    directory = build_package(cell.output_dir, "temperature", str(tmp_path / f"package_{storage}"))
    package = load_package(directory)
    assert package.observation_contract["whole_answer"] == "complete_calibration_truth"
    assert package.observation_contract["support"]["v_cal"]["unknown"] == 0
    # All three terms hold ten or more of the 30 test rows' positives, so Fmax and the scored average precisions are the metric contract's own.
    scores = report["methods"]["plain"]["test"]
    expected = canonical_task_metrics("multilabel", saved["test"].labels, probabilities(saved["test"].outputs, "multilabel"))
    assert scores["scored_terms"] == 3 and 0 < scores["fmax"] <= 1
    for name in ("fmax", "scored_micro_aupr", "scored_macro_aupr"):
        assert scores[name] == pytest.approx(expected[name])


@pytest.mark.parametrize("selection,key", [(None, "f1"), ("f1", "f1"), ("fmax", "fmax")])
def test_a_multilabel_cell_selects_its_seeds_on_the_metric_its_task_names(tmp_path, monkeypatch, selection, key):
    monkeypatch.chdir(tmp_path)
    path, digest, _, _ = release_fixture(tmp_path, "multilabel")
    mapping = fixture._cell_mapping(tmp_path, "unused", f"selection_{selection}", num_runs=1)
    mapping["protify_args"].pop("data_dirs")
    mapping["uncertainty"].update(head="projected", dataset_manifest=str(path), dataset_sha256=digest, save_heads=True,
                                  require_cached_embeddings=True, dropout=0.0, **({"selection_metric": selection} if selection else {}))
    cell = cell_from_mapping(mapping)
    report = fixture.workflow.run_uncertainty_cell(cell)
    saved = load_seed_predictions(str(Path(cell.output_dir) / "seed_42.npz"))["v_fit"]
    expected = canonical_task_metrics("multilabel", saved.labels, probabilities(saved.outputs, "multilabel"), label_mask=saved.label_mask,
                                      primary_only=True, selection=selection)[key]
    assert report["per_seed"][0]["v_fit_primary"] == expected and expected is not None
    assert report["primary_metric"] == "f1", "the reported primary is unchanged"
    assert report.get("selection_metric") == selection
    directory = build_package(cell.output_dir, "temperature", str(tmp_path / "package"))
    choice = json.loads((Path(directory) / "calibration.json").read_text(encoding="utf-8"))["seed_choice"]
    assert choice["primary_metric"] == (selection or "f1")


def test_a_selection_metric_needs_a_declared_release_and_a_metric_of_its_kind(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mapping = fixture._cell_mapping(tmp_path, "unused", "selection_refused", num_runs=1)
    mapping["uncertainty"].update(selection_metric="fmax")
    with pytest.raises(ValueError, match="needs dataset_manifest"):
        cell_from_mapping(mapping)
    path, digest, _, _ = release_fixture(tmp_path, "binary")
    mapping["protify_args"].pop("data_dirs")
    mapping["uncertainty"].update(head="projected", dataset_manifest=str(path), dataset_sha256=digest, save_heads=True, require_cached_embeddings=True)
    with pytest.raises(ValueError, match="selects on one of"):
        fixture.workflow.run_uncertainty_cell(cell_from_mapping(mapping))


def test_a_closed_world_needs_a_declared_release_and_a_multilabel_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mapping = fixture._cell_mapping(tmp_path, "unused", "closed_refused", num_runs=1)
    mapping["uncertainty"].update(unlisted_labels="negative")
    with pytest.raises(ValueError, match="dataset_manifest"):
        cell_from_mapping(mapping)
    mapping["uncertainty"].update(unlisted_labels="open")
    with pytest.raises(ValueError, match="unlisted_labels must be one of"):
        cell_from_mapping(mapping)
    path, digest, _, _ = release_fixture(tmp_path, "binary")
    mapping = fixture._cell_mapping(tmp_path, "unused", "closed_binary", num_runs=1)
    mapping["protify_args"].pop("data_dirs")
    mapping["uncertainty"].update(head="projected", dataset_manifest=str(path), dataset_sha256=digest, save_heads=True,
                                  require_cached_embeddings=True, unlisted_labels="negative")
    with pytest.raises(ValueError, match="only a multilabel release"):
        fixture.workflow.run_uncertainty_cell(cell_from_mapping(mapping))


@pytest.mark.parametrize("kind", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("partial_calibration", [False, True])
def test_scalar_package_preserves_unknown_truth_and_never_selects_on_test_masks(tmp_path, monkeypatch, kind, partial_calibration):
    from dataclasses import replace
    from protify.testing_suite.test_uncertainty_package import DEVICE

    monkeypatch.chdir(tmp_path)
    path, _, manifest, rows = release_fixture(tmp_path, kind)
    for name, records in rows.items():
        if partial_calibration or name == "test":
            records[0].update(labels=0, label_mask=False)
        records[1].update(labels=0, label_mask=True)
        digest = rewrite_rows(path, manifest, name, records)
    mapping = fixture._cell_mapping(tmp_path, "unused", "partial_scalar", num_runs=1)
    mapping["protify_args"].pop("data_dirs")
    mapping["protify_args"]["balanced_regression_metrics"] = False
    mapping["uncertainty"].update(head="projected", dataset_manifest=str(path), dataset_sha256=digest,
                                   save_heads=True, require_cached_embeddings=True, dropout=0.0, conformal_levels=[0.5, 0.8])
    joins = []
    join = fixture.workflow.join_branch_embeddings

    def capture(*args, **kwargs):
        joined, widths = join(*args, **kwargs)
        joins.append(joined)
        return joined, widths

    monkeypatch.setattr(fixture.workflow, "join_branch_embeddings", capture)
    cell = cell_from_mapping(mapping)
    report = fixture.workflow.run_uncertainty_cell(cell)
    first_path = Path(build_package(cell.output_dir, "temperature", str(tmp_path / "package")))
    package = load_package(first_path)
    contract = package.observation_contract
    assert contract["support"]["v_cal"]["unknown"] == int(partial_calibration)
    joined = joins[-1]
    features = joined.block([joined.rows[row["seqs"]] for row in rows["test"]])
    prediction = package.predict(features, batch_size=16, device=DEVICE)
    assert prediction.observation_contract == contract
    if kind != "regression":
        assert (prediction.confidence is None) == partial_calibration
        if partial_calibration:
            assert not package.select(prediction).any()
            with pytest.raises(ValueError, match="strictly between"):
                package.select(prediction, 0)
        else:
            assert package.answer_rule is not None
    saved_path = Path(cell.output_dir) / "seed_42.npz"
    saved = load_seed_predictions(str(saved_path))
    changed = {}
    for role, split in saved.items():
        truth = split.labels.copy()
        truth[~split.label_mask] = -777  # Invalid class indices cannot reach any observed-label computation.
        changed[role] = replace(split, labels=truth)
    save_seed_predictions(str(saved_path), changed)
    second_path = Path(build_package(cell.output_dir, "temperature", str(tmp_path / "changed_fillers")))
    for name in ("calibration.json", "metrics.json"):
        assert json.loads((first_path / name).read_text()) == json.loads((second_path / name).read_text())
    metrics = json.loads((first_path / "metrics.json").read_text())
    assert metrics["test_label_support"]["unknown"] == 1
    if kind != "regression":
        assert metrics["answer_rule"]["unknown_correctness"] == 1
        assert metrics["answer_rule"]["accuracy"] is None
        with pytest.raises(ValueError, match="partial truth"):
            build_package(cell.output_dir, "temperature", str(tmp_path / "unqualified_m8"), crossfit_dir="unused")
    reported = report["methods"]["temperature"]["per_run"][0]["test"]
    assert all(value == reported[name] for name, value in metrics["test"].items())
    calibration_path = first_path / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    calibration["observation_contract"]["support"]["v_cal"]["unknown"] += 1
    calibration_path.write_text(json.dumps(calibration))
    with pytest.raises(ValueError, match="observation contract differs"):
        load_package(first_path)


@pytest.mark.parametrize("kind,width,label_type", [("binary", 2, "singlelabel"), ("multiclass", 3, "singlelabel"), ("multilabel", 3, "multilabel"), ("regression", 1, "regression")])
def test_declared_task_and_label_order_override_inference(tmp_path, kind, width, label_type):
    path, digest, manifest, rows = release_fixture(tmp_path, kind)
    release = load_declared_dataset(path, digest)
    assert release.num_labels == width and release.label_type == label_type
    assert release.training_record()["label_vocabulary"] == manifest["label_vocabulary"]
    for name in SPLIT_NAMES:
        assert list(release.splits[name]["observation_id"]) == [row["observation_id"] for row in rows[name]]
        assert list(release.splits[name]["labels"]) == [row["labels"] for row in rows[name]]


def test_embedding_inputs_union_all_four_roles_without_legacy_inference(tmp_path, monkeypatch):
    from protify import precompute_sae_and_dense as embedding

    path, digest, _, rows = release_fixture(tmp_path)
    arguments = Namespace(
        dataset_manifest=[str(path), str(path)], dataset_sha256=[digest, digest],
        data_names=None, data_revisions=None, dna_to_aa=False,
        max_train_rows=None, max_valid_rows=None, max_test_rows=None, max_length=2048,
    )
    monkeypatch.setattr(embedding, "DataMixin", lambda **_: pytest.fail("declared input reached legacy loader"))
    expected = {row["seqs"] for split in rows.values() for row in split}
    actual = embedding.load_input_sequences(arguments, ["ESMC-600-SAE"])
    assert actual == sorted(expected, key=lambda sequence: (len(sequence), sequence))
    arguments.dataset_sha256 = [digest]
    with pytest.raises(ValueError, match="aligned|one.*manifest|one.*digest"):
        embedding.load_input_sequences(arguments, ["ESMC-600-SAE"])
    arguments.dataset_sha256 = [digest, digest]
    arguments.max_train_rows = 1
    with pytest.raises(ValueError, match="row caps"):
        embedding.load_input_sequences(arguments, ["ESMC-600-SAE"])
    arguments.max_train_rows = None
    arguments.max_length = 2
    with pytest.raises(ValueError, match="feature budget"):
        embedding.load_input_sequences(arguments, ["ESMC-600-SAE"])


def test_release_and_artifact_bytes_are_pinned(tmp_path):
    path, digest, manifest, _ = release_fixture(tmp_path)
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        load_declared_dataset(path, "0" * 64)
    artifact = path.parent / manifest["splits"]["test"]["path"]
    artifact.write_bytes(artifact.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        load_declared_dataset(path, digest)


def test_new_content_cannot_reuse_cached_labels_with_unchanged_file_metadata(tmp_path):
    path, _, manifest, rows = release_fixture(tmp_path)
    artifact = path.parent / manifest["splits"]["train"]["path"]
    pq.write_table(pa.Table.from_pylist(rows["train"]), artifact, compression=None, use_dictionary=False)
    manifest["splits"]["train"]["sha256"] = sha256_file(artifact)
    first_digest = write_manifest(path, manifest)
    first = load_declared_dataset(path, first_digest).splits["train"][0]["labels"]
    original_stat = artifact.stat()
    rows["train"][0]["labels"] = 1 - first
    pq.write_table(pa.Table.from_pylist(rows["train"]), artifact, compression=None, use_dictionary=False)
    assert artifact.stat().st_size == original_stat.st_size
    os.utime(artifact, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    manifest["splits"]["train"]["sha256"] = sha256_file(artifact)
    second_digest = write_manifest(path, manifest)
    assert load_declared_dataset(path, second_digest).splits["train"][0]["labels"] == 1 - first


@pytest.mark.parametrize("field,value,message", [
    ("labels", 3, "zero or one"), ("labels", float("nan"), "finite numbers"),
    ("label_mask", 1, "booleans"), ("protein_id", "0" * 64, "sequence SHA"),
    ("seqs", "MA*", "normalized"),
])
def test_invalid_rows_fail_before_training(tmp_path, field, value, message):
    path, _, manifest, rows = release_fixture(tmp_path)
    for row in rows["test"]:
        row[field] = value
    digest = rewrite_rows(path, manifest, "test", rows["test"])
    with pytest.raises(ValueError, match=message):
        load_declared_dataset(path, digest)


@pytest.mark.parametrize("violation", ["protein", "cluster", "observation"])
def test_cross_split_leakage_or_reused_measurement_is_rejected(tmp_path, violation):
    path, _, manifest, rows = release_fixture(tmp_path)
    original, changed = rows["train"][0], rows["test"][0]
    fields = {"protein": ["protein_id", "seqs"], "cluster": ["cluster_id"], "observation": ["observation_id"]}[violation]
    for field in fields:
        changed[field] = original[field]
    digest = rewrite_rows(path, manifest, "test", rows["test"])
    with pytest.raises(ValueError, match="crosses frozen|globally unique"):
        load_declared_dataset(path, digest)


def test_unknown_targets_keep_zero_fillers_and_false_masks(tmp_path):
    path, _, manifest, rows = release_fixture(tmp_path, "multilabel")
    rows["train"][0]["labels"] = [1, 0, 0]
    rows["train"][0]["label_mask"] = [True, True, False]
    digest = rewrite_rows(path, manifest, "train", rows["train"])
    row = load_declared_dataset(path, digest).splits["train"][0]
    assert row["labels"] == [1, 0, 0] and row["label_mask"] == [True, True, False]
    rows["train"][0]["labels"][2] = 1
    digest = rewrite_rows(path, manifest, "train", rows["train"])
    with pytest.raises(ValueError, match="zero fillers"):
        load_declared_dataset(path, digest)


def test_release_cannot_silently_truncate_or_grow_its_vocabulary(tmp_path):
    path, digest, manifest, rows = release_fixture(tmp_path, "multilabel")
    with pytest.raises(ValueError, match="truncation"):
        load_declared_dataset(path, digest, max_sequence_length=8)
    rows["test"][0]["labels"].append(1)
    rows["test"][0]["label_mask"].append(True)
    digest = rewrite_rows(path, manifest, "test", rows["test"])
    with pytest.raises(ValueError, match="vocabulary width"):
        load_declared_dataset(path, digest)


@pytest.mark.parametrize("kind", ["binary", "multiclass", "multilabel", "regression"])
def test_validation_only_fitting_never_predicts_calibration_or_test(tmp_path, monkeypatch, kind):
    monkeypatch.chdir(tmp_path)
    path, digest, _, rows = release_fixture(tmp_path, kind)
    mapping = fixture._cell_mapping(tmp_path, "unused", f"validation_{kind}", num_runs=1)
    mapping["protify_args"].pop("data_dirs")
    mapping["protify_args"]["balanced_regression_metrics"] = False
    mapping["uncertainty"].update(head="projected", dataset_manifest=str(path), dataset_sha256=digest, save_heads=True,
                                   require_cached_embeddings=True, evaluation_mode="validation_only")
    cell = cell_from_mapping(mapping)
    validation_ids = [row["observation_id"] for row in rows["validation"]]
    trainer_module = sys.modules[fixture.workflow.MainProcess._train.__module__]
    original_trainer = trainer_module.ClusterTrainer
    calls = []

    class ValidationTrainer(original_trainer):
        def evaluate(self, eval_dataset=None, *args, **kwargs):
            selected = self.eval_dataset if eval_dataset is None else eval_dataset
            assert list(selected.observation_ids) == validation_ids
            calls.append("trainer_evaluate")
            return super().evaluate(eval_dataset, *args, **kwargs)

        def predict(self, test_dataset, *args, **kwargs):
            assert list(test_dataset.observation_ids) == validation_ids
            calls.append("trainer_predict")
            return super().predict(test_dataset, *args, **kwargs)

    monkeypatch.setattr(trainer_module, "ClusterTrainer", ValidationTrainer)
    collect = fixture.workflow.collect_predictions
    trunk = fixture.workflow.trunk_features

    def validation_predictions(model, dataset, *args, **kwargs):
        assert list(dataset.observation_ids) == validation_ids
        calls.append("validation_predictions")
        return collect(model, dataset, *args, **kwargs)

    def allowed_trunk(model, dataset, *args, **kwargs):
        assert set(dataset.observation_ids) <= {row["observation_id"] for split in ("train", "validation") for row in rows[split]}
        return trunk(model, dataset, *args, **kwargs)

    monkeypatch.setattr(fixture.workflow, "collect_predictions", validation_predictions)
    monkeypatch.setattr(fixture.workflow, "trunk_features", allowed_trunk)
    monkeypatch.setattr(fixture.workflow, "build_methods", lambda *_args, **_kwargs: pytest.fail("fitting entered calibration"))
    report = fixture.workflow.run_uncertainty_cell(cell)
    assert report["evaluation_mode"] == "validation_only" and report["scored_splits"] == ["v_fit"]
    assert report["methods"] == {} and report["per_seed"][0]["inference_split"] == "v_fit"
    assert set(load_seed_predictions(str(Path(cell.output_dir) / "seed_42.npz"))) == {"v_fit"}
    assert calls and "validation_predictions" in calls
    assert fixture.workflow.run_uncertainty_cell(cell) == report
    with pytest.raises(ValueError, match="separate frozen-checkpoint calibration and test stage"):
        build_package(cell.output_dir, "temperature", str(tmp_path / "package"))
    seed_path = Path(cell.output_dir) / "seed_42.json"
    original = seed_path.read_bytes()
    altered = json.loads(original)
    altered["scored_splits"] = ["test"]
    seed_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="evaluation scope differs"):
        fixture.workflow.run_uncertainty_cell(cell)
    seed_path.write_bytes(original)
    mapping["uncertainty"]["evaluation_mode"] = "full"
    with pytest.raises(ValueError, match="different training settings"):
        fixture.workflow.run_uncertainty_cell(cell_from_mapping(mapping))


@pytest.mark.parametrize("change", [{"evaluation_mode": "typo"}, {"save_heads": False}, {"laplace": True}, {"pnorm_uncertainty": True}])
def test_validation_only_rejects_unsupported_cell_settings(tmp_path, change):
    mapping = fixture._cell_mapping(tmp_path, "unused", "invalid", num_runs=1)
    mapping["uncertainty"].update({"dataset_manifest": "dataset.json", "dataset_sha256": "a" * 64,
                                   "save_heads": True, "evaluation_mode": "validation_only", **change})
    with pytest.raises(ValueError, match="evaluation_mode|validation_only"):
        cell_from_mapping(mapping)


@pytest.mark.parametrize("kind,classes", [("binary", 3), ("multiclass", 3), ("multiclass", 2), ("multilabel", 3), ("regression", 3)])
def test_four_way_release_trains_without_legacy_loading_or_split_carving(tmp_path, monkeypatch, kind, classes):
    monkeypatch.chdir(tmp_path)
    path, digest, manifest, rows = release_fixture(tmp_path, kind, classes)
    if kind == "regression":
        manifest["units"] = "log10(s^-1)"
        manifest["target_transform"] = {"name": "log10", "source_units": "s^-1", "inverse": "pow10"}
        digest = write_manifest(path, manifest)
    mapping = fixture._cell_mapping(tmp_path, "unused", f"declared_{kind}", num_runs=1)
    mapping["protify_args"].pop("data_dirs")
    mapping["protify_args"]["balanced_regression_metrics"] = False
    mapping["uncertainty"].update(head="projected", dataset_manifest=str(path), dataset_sha256=digest, save_heads=True, require_cached_embeddings=True)
    cell = cell_from_mapping(mapping)
    def forbid_legacy(*args, **kwargs):
        raise AssertionError("declared release reached legacy dataset inference")
    monkeypatch.setattr(fixture.workflow, "load_cell_data", forbid_legacy)
    trainer_module = sys.modules[fixture.workflow.MainProcess._train.__module__]
    original_trainer = trainer_module.ClusterTrainer
    preselection_rows = []
    class AuditedTrainer(original_trainer):
        selection_complete = False

        def train(self, *args, **kwargs):
            output = super().train(*args, **kwargs)
            self.selection_complete = True
            return output

        def evaluate(self, eval_dataset=None, *args, **kwargs):
            if not self.selection_complete:
                selected = self.eval_dataset if eval_dataset is None else eval_dataset
                identities = list(selected.observation_ids)
                assert identities == [row["observation_id"] for row in rows["validation"]]
                preselection_rows.append(identities)
            return super().evaluate(eval_dataset, *args, **kwargs)
    monkeypatch.setattr(trainer_module, "ClusterTrainer", AuditedTrainer)
    joins = []
    original_join = fixture.workflow.join_branch_embeddings
    def record_join(*args, **kwargs):
        joined, widths = original_join(*args, **kwargs)
        joins.append(joined)
        return joined, widths
    monkeypatch.setattr(fixture.workflow, "join_branch_embeddings", record_join)
    report = fixture.workflow.run_uncertainty_cell(cell)
    assert report["task_type"] == kind
    assert report["dataset_release"]["manifest_sha256"] == digest
    assert report["split"]["v_fit"] == len(rows["validation"])
    assert report["split"]["v_cal"] == len(rows["calibration"])
    assert report["sampling"]["algorithm"] == "cluster_protein_assay_uniform_v1"
    metrics_contract = report["metric_contract"]
    assert metrics_contract["scheme"] == "declared_oracle_metrics_v1"
    assert metrics_contract["dataset_sha256"] == digest
    assert report["per_seed"][0]["metric_contract"] == metrics_contract
    if kind == "regression":
        assert metrics_contract["metric_units"]["mae"] == "log10(s^-1)"
    if kind == "multilabel":
        assert "micro_f1" in report["methods"]["plain"]["test"]
        assert "auprc" not in report["methods"]["plain"]["test"]
    exposure = report["per_seed"][0]["sampling"]
    training = report["per_seed"][0]["training"]
    assert training["recipe"] == report["training_recipe"]
    assert training["seed"] == 42
    effective = training["recipe"]["training_arguments"]
    assert effective["metric_for_best_model"] == "eval_observed_loss"
    assert effective["optim"] == "adamw_torch"
    assert training["outcome"]["restored_best_weights"] is True
    assert training["recipe"]["sampling"] == report["sampling"]
    assert set(exposure["observation_counts"]) <= {row["observation_id"] for row in rows["train"]}
    assert sum(exposure["cluster_counts"].values()) == sum(epoch["consumed_rows"] for epoch in exposure["epochs"])
    saved = load_seed_predictions(str(Path(cell.output_dir) / "seed_42.npz"))
    assert preselection_rows
    for saved_name, source_name in (("v_fit", "validation"), ("v_cal", "calibration"), ("test", "test")):
        np.testing.assert_array_equal(saved[saved_name].labels, np.asarray([row["labels"] for row in rows[source_name]], dtype=saved[saved_name].labels.dtype))
        np.testing.assert_array_equal(saved[saved_name].observation_ids, [row["observation_id"] for row in rows[source_name]])
    assert fixture.workflow.run_uncertainty_cell(cell) == report
    seed_path = Path(cell.output_dir) / "seed_42.json"
    original_seed = seed_path.read_bytes()
    changed_seed = json.loads(original_seed)
    sampled_cluster = next(iter(changed_seed["sampling"]["cluster_counts"]))
    changed_seed["sampling"]["cluster_counts"][sampled_cluster] += 1
    seed_path.write_text(json.dumps(changed_seed))
    with pytest.raises(ValueError, match="cluster_counts"):
        fixture.workflow.run_uncertainty_cell(cell)
    seed_path.write_bytes(original_seed)
    changed_seed = json.loads(original_seed)
    changed_seed["v_fit_primary"] = 12345.0
    seed_path.write_text(json.dumps(changed_seed))
    with pytest.raises(ValueError, match="validation primary differs"):
        fixture.workflow.run_uncertainty_cell(cell)
    seed_path.write_bytes(original_seed)
    metric_path = Path(cell.output_dir) / "metric_contract.json"
    original_metric = metric_path.read_bytes()
    metric_path.unlink()
    with pytest.raises(ValueError, match="lack their declared metric contract"):
        fixture.workflow.run_uncertainty_cell(cell)
    metric_path.write_bytes(original_metric)
    changed_seed = json.loads(original_seed)
    changed_seed["metric_contract"]["scheme"] = "old_metrics"
    seed_path.write_text(json.dumps(changed_seed))
    with pytest.raises(ValueError, match="seed metric contract differs"):
        fixture.workflow.run_uncertainty_cell(cell)
    seed_path.write_bytes(original_seed)
    changed_seed = json.loads(original_seed)
    changed_seed["training"]["recipe"]["training_arguments"]["max_grad_norm"] = 77.0
    seed_path.write_text(json.dumps(changed_seed))
    with pytest.raises(ValueError, match="training recipe changed"):
        fixture.workflow.run_uncertainty_cell(cell)
    seed_path.write_bytes(original_seed)
    recipe_path = Path(cell.output_dir) / "training_recipe.json"
    original_recipe = recipe_path.read_bytes()
    recipe_path.unlink()
    with pytest.raises(ValueError, match="lack their training recipe"):
        fixture.workflow.run_uncertainty_cell(cell)
    recipe_path.write_bytes(original_recipe)
    resolve = trainer_module.TrainerArguments.__call__
    def changed_default(self, *args, **kwargs):
        arguments = resolve(self, *args, **kwargs)
        arguments.max_grad_norm = 0.75
        return arguments
    with monkeypatch.context() as altered:
        altered.setattr(trainer_module.TrainerArguments, "__call__", changed_default)
        with pytest.raises(ValueError, match="resolved training recipe changed"):
            fixture.workflow.run_uncertainty_cell(cell)
    from protify.testing_suite.test_uncertainty_package import DEVICE, REVISIONS

    revisions = tmp_path / "revisions.json"
    revisions.write_text(json.dumps(REVISIONS), encoding="utf-8")
    packaged = build_package(cell.output_dir, "temperature", str(tmp_path / "package"), levels=(0.8,), revisions_path=str(revisions))
    head = load_package(packaged)
    origin = json.loads((Path(packaged) / "origin.json").read_text())
    assert origin["sampling"] == {"plan": report["sampling"], "seeds": {"42": exposure}}
    assert origin["training"] == {"42": training}
    assert origin["metric_contract"] == metrics_contract
    assert json.loads((Path(packaged) / "metrics.json").read_text())["metric_contract"] == metrics_contract
    changed_seed = json.loads(original_seed)
    changed_seed["v_fit_primary"] = 12345.0
    seed_path.write_text(json.dumps(changed_seed))
    with pytest.raises(ValueError, match="package validation primary differs"):
        build_package(cell.output_dir, "temperature", str(tmp_path / "invalid_primary_package"), levels=(0.8,), revisions_path=str(revisions))
    seed_path.write_bytes(original_seed)
    metric_path.write_text("{}")
    with pytest.raises(ValueError, match="matching metric contract"):
        build_package(cell.output_dir, "temperature", str(tmp_path / "invalid_metrics_package"), levels=(0.8,), revisions_path=str(revisions))
    metric_path.write_bytes(original_metric)
    recipe_path.write_text("{}")
    with pytest.raises(ValueError, match="matching training recipe"):
        build_package(cell.output_dir, "temperature", str(tmp_path / "invalid_recipe_package"), levels=(0.8,), revisions_path=str(revisions))
    recipe_path.write_bytes(original_recipe)
    sampling_path = Path(cell.output_dir) / "sampling.json"
    sampling_bytes = sampling_path.read_bytes()
    sampling_path.write_text("{}")
    with pytest.raises(ValueError, match="sampling plan"):
        build_package(cell.output_dir, "temperature", str(tmp_path / "invalid_sampling_package"), levels=(0.8,), revisions_path=str(revisions))
    sampling_path.write_bytes(sampling_bytes)
    assert head.target_contract["manifest_sha256"] == digest
    assert head.target_contract["label_vocabulary"] == report["dataset_release"]["label_vocabulary"]
    assert head.target_contract["units"] == manifest["units"]
    assert head.target_contract["target_transform"] == manifest["target_transform"]
    joined = joins[-1]
    features = joined.block([joined.rows[row["seqs"]] for row in rows["test"]])
    prediction = head.predict(features, batch_size=16, device=DEVICE)
    method = next(method for method in fixture.workflow.build_methods({42: saved}, "projected", kind) if method.name == "temperature")
    expected = method.runs[0].splits["test"]
    if kind == "regression":
        np.testing.assert_allclose(prediction.mean, expected.mean, rtol=1e-5, atol=1e-5)
    else:
        np.testing.assert_allclose(prediction.probs, expected.probs, rtol=1e-5, atol=1e-5)
    calibration_path = Path(packaged) / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    calibration["target_contract"]["units"] = "wrong units"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(ValueError, match="target contract"):
        load_package(packaged)
    calibration.pop("target_contract")
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(ValueError, match="must both be present"):
        load_package(packaged)
    for name in ("max_train_rows", "max_valid_rows", "max_test_rows", "trim"):
        changed = copy.deepcopy(mapping)
        changed["protify_args"][name] = True if name == "trim" else 1
        changed_cell = cell_from_mapping(changed)
        arguments = fixture.workflow.cell_arguments(changed_cell, [fixture.workflow.parse_branch_spec(token) for token in cell.uncertainty.branch_specs])
        with pytest.raises(ValueError, match="forbid"):
            fixture.workflow.load_split_cell_data(changed_cell, arguments, None)
    with pytest.raises(ValueError, match="cannot be re-split"):
        fixture.workflow.split_cell(cell)


@pytest.mark.parametrize("mode", ["full", "validation_only"])
def test_declared_multilabel_fit_selects_on_the_observed_loss_and_never_predicts_through_the_trainer(tmp_path, monkeypatch, mode):
    """The Trainer scores the validation split by the selection criterion alone, and the saved predictions carry every held-out score."""
    monkeypatch.chdir(tmp_path)
    path, digest, _, _ = release_fixture(tmp_path, "multilabel")
    mapping = fixture._cell_mapping(tmp_path, "unused", f"selection_{mode}", num_runs=1)
    mapping["protify_args"].pop("data_dirs")
    mapping["uncertainty"].update(head="projected", dataset_manifest=str(path), dataset_sha256=digest, save_heads=True, require_cached_embeddings=True, evaluation_mode=mode)
    cell = cell_from_mapping(mapping)
    trainer_module = sys.modules[fixture.workflow.MainProcess._train.__module__]
    evaluated, predicted = [], []

    class SpiedTrainer(trainer_module.ClusterTrainer):
        def evaluate(self, *args, **kwargs):
            metrics = super().evaluate(*args, **kwargs)
            evaluated.append(set(metrics))
            return metrics

        def predict(self, *args, **kwargs):
            predicted.append(True)
            return super().predict(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "ClusterTrainer", SpiedTrainer)
    report = fixture.workflow.run_uncertainty_cell(cell)
    assert evaluated and not predicted
    for names in evaluated:
        assert {"eval_observed_loss", "eval_observed_labels", "eval_unobserved_labels"} <= names
        assert not names & {"eval_f1", "eval_mcc", "eval_roc_auc", "eval_pr_auc", "eval_accuracy", "eval_threshold"}
    recorded = report["per_seed"][0]["protify_v_fit_metrics"]
    assert "eval_observed_loss" in recorded and "eval_f1" not in recorded
    if mode == "full":
        assert "micro_f1" in report["methods"]["plain"]["test"]
