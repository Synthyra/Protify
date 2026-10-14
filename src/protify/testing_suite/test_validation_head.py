"""Frozen-head preflight needs no calibration/test payloads or cache rows."""

import json
import sys
import numpy as np
import pytest

from pathlib import Path

from foundry.digests import sha256_file
from protify.data import declared_dataset
from protify.testing_suite.embedding_store_builder import without_rows
from protify.testing_suite.test_calibration_crossfit import _stores
from protify.testing_suite.test_declared_dataset import compact_release, release_fixture, rewrite_rows
from protify.testing_suite.test_uncertainty_workflow import _cell_mapping
from protify.uncertainty import workflow
from protify.uncertainty.cell import cell_from_mapping
from protify.uncertainty.predictions import load_seed_predictions
from protify.uncertainty.validation_head import verify_validation_cell


@pytest.fixture(scope="module", params=["binary", "multiclass", "multilabel", "multilabel_compact", "regression"])
def validation_cell(tmp_path_factory, request):
    root = tmp_path_factory.mktemp(f"validation_preflight_{request.param}")
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        patch.setattr(sys.modules[workflow.parse_arguments.__module__], "load_secrets_env", lambda: [])
        path, digest, manifest, rows = release_fixture(root, request.param.removesuffix("_compact"))
        if request.param.startswith("multilabel"):
            for split in rows:
                rows[split][0]["label_mask"][2] = False
                digest = rewrite_rows(path, manifest, split, rows[split])
        if request.param.endswith("_compact"):
            digest = compact_release(path, manifest, rows)
        mapping = _cell_mapping(root, "unused", "validation", num_runs=2)
        mapping["protify_args"].pop("data_dirs")
        mapping["protify_args"]["balanced_regression_metrics"] = False
        mapping["uncertainty"].update(head="projected", dataset_manifest=str(path), dataset_sha256=digest,
                                       save_heads=True, require_cached_embeddings=True, evaluation_mode="validation_only", dropout=0.0)
        cell = cell_from_mapping(mapping)
        workflow.run_uncertainty_cell(cell)
        for split in ("calibration", "test"):
            (path.parent / manifest["splits"][split]["path"]).unlink()
        held_out = [row["seqs"] for split in ("calibration", "test") for row in rows[split]]
        every_sequence = [row["seqs"] for split_rows in rows.values() for row in split_rows]
        stores = _stores(root / "embeddings")
        assert stores, "the cell wrote no embedding store, so no cache row could be removed"
        for store in stores:
            without_rows(store, held_out, every_sequence)
        yield cell


def test_all_selected_heads_reload_with_held_out_files_and_cache_rows_absent(validation_cell, monkeypatch):
    cell = validation_cell
    root = Path(cell.output_dir)
    before = {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()}
    original = declared_dataset._artifact
    opened = []

    def allowed(directory, record):
        opened.append(record["path"])
        assert record["path"] in {"train.parquet", "validation.parquet"}
        return original(directory, record)

    monkeypatch.setattr(declared_dataset, "_artifact", allowed)
    verification = verify_validation_cell(cell)
    assert opened == ["train.parquet", "validation.parquet"]
    assert verification["seeds"] == [42, 43]
    assert verification["loaded_splits"] == ["train", "validation"]
    assert verification["excluded_splits"] == ["calibration", "test"]
    assert verification["validation_rows"] == verification["validation_families"] == 20
    assert set(verification["maximum_absolute_output_difference"]) == {"42", "43"}
    assert max(verification["maximum_absolute_output_difference"].values()) <= 1e-3
    assert before == {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("kind", ["binary", "multiclass", "multilabel", "regression"])
def test_a_validation_only_fit_needs_no_calibration_or_test_payload_or_cache_row(tmp_path, monkeypatch, kind):
    """Training cannot read held-out labels: it runs with those payloads and cached rows deleted."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys.modules[workflow.parse_arguments.__module__], "load_secrets_env", lambda: [])
    path, digest, manifest, rows = release_fixture(tmp_path, kind)
    if kind == "multilabel":
        for split in rows:
            rows[split][0]["label_mask"][2] = False
            digest = rewrite_rows(path, manifest, split, rows[split])
    held_out = [row["seqs"] for split in ("calibration", "test") for row in rows[split]]
    every_sequence = [row["seqs"] for split_rows in rows.values() for row in split_rows]
    for split in ("calibration", "test"):
        (path.parent / manifest["splits"][split]["path"]).unlink()
    stores = _stores(tmp_path / "embeddings")
    assert stores, "the fixture wrote no embedding store, so no cache row could be removed"
    for store in stores:
        without_rows(store, held_out, every_sequence)

    def mapped(name: str, mode: str) -> dict:
        mapping = _cell_mapping(tmp_path, "unused", name, num_runs=1)
        mapping["protify_args"].pop("data_dirs")
        mapping["protify_args"]["balanced_regression_metrics"] = False
        mapping["uncertainty"].update(head="projected", dataset_manifest=str(path), dataset_sha256=digest, save_heads=True,
                                       require_cached_embeddings=True, evaluation_mode=mode, dropout=0.0)
        return mapping

    original = declared_dataset._artifact
    opened = []

    def recorded(directory, record):
        opened.append(record["path"])
        return original(directory, record)

    monkeypatch.setattr(declared_dataset, "_artifact", recorded)
    cell = cell_from_mapping(mapped(f"unopened_{kind}", "validation_only"))
    report = workflow.run_uncertainty_cell(cell)
    assert opened == ["train.parquet", "validation.parquet"]
    assert report["split_access"] == {"loaded": ["train", "validation"], "excluded": ["test", "calibration"]}
    assert report["split"]["v_fit"] == len(rows["validation"])
    assert report["split"]["v_cal"] is None and report["split"]["test"] is None

    root = Path(cell.output_dir)
    split = json.loads((root / "split.json").read_text())
    assert split["validation_rows"] == len(split["v_fit_rows"]) == len(rows["validation"])
    assert split["v_cal_rows"] == [] and split["v_cal_digests"] == []
    assert set(json.loads((root / "label_observations.json").read_text())["splits"]) == {"train", "valid"}
    assert set(load_seed_predictions(str(root / "seed_42.npz"))) == {"v_fit"}
    assert workflow.run_uncertainty_cell(cell) == report
    assert verify_validation_cell(cell)["loaded_splits"] == ["train", "validation"]

    # Control: the same release cannot start a full evaluation without the payloads it would read.
    with pytest.raises(FileNotFoundError):
        workflow.run_uncertainty_cell(cell_from_mapping(mapped(f"full_{kind}", "full")))


@pytest.mark.parametrize("field", ["outputs", "labels", "label_mask", "observation_ids"])
def test_preflight_refuses_changed_validation_arrays(validation_cell, field):
    path = Path(validation_cell.output_dir) / "seed_43.npz"
    original = path.read_bytes()
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    values = arrays[f"v_fit__{field}"]
    if field == "observation_ids":
        values[0] = "changed_observation"
    elif field == "label_mask":
        values.flat[0] = not values.flat[0]
    else:
        values.flat[0] += 1
    try:
        np.savez(path, **arrays)
        with pytest.raises(ValueError, match="V_fit|reloaded V_fit"):
            verify_validation_cell(validation_cell)
    finally:
        path.write_bytes(original)


def test_preflight_rejects_incomplete_cell_without_creating_records(validation_cell):
    path = Path(validation_cell.output_dir) / "cell.json"
    original = path.read_bytes()
    try:
        path.unlink()
        with pytest.raises(FileNotFoundError):
            verify_validation_cell(validation_cell)
        assert not path.exists()
    finally:
        path.write_bytes(original)
