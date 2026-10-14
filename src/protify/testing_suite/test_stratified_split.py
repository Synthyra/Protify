"""A cell's rows split again by identity strata: the manifest, and the rows each split receives."""

from __future__ import annotations

import json
import numpy as np
import pytest

from types import SimpleNamespace

from foundry.datasets import IdentityStratifiedSplit
from protify.uncertainty import stratified_split, workflow
from protify.uncertainty.identity_bins import sequence_digest
from protify.uncertainty.report import read_json
from protify.uncertainty.stratified_split import (
    SPLIT_SCHEME,
    apply_split_manifest,
    build_split_manifest,
    protein_labels,
    split_row_indices,
)
from protify.uncertainty.workflow import CellData


SEQUENCES = [f"M{'ACDEFGHIKL'[index % 10] * (index + 3)}" for index in range(12)]
# Each sequence's split; "MAAA" (index 0) appears twice, once in train and once in test.
ASSIGNED = ["train"] * 6 + ["validation"] * 2 + ["calibration"] * 2 + ["test"] * 2


def _published_splits() -> tuple[dict, dict, dict]:
    """Train, validation, and test as published: rows 0-5, 6-9 and a repeat of 0, 10-11."""
    train = {"seqs": SEQUENCES[:6], "labels": [0, 1, 0, 1, 0, 1]}
    valid = {"seqs": SEQUENCES[6:10] + [SEQUENCES[0]], "labels": [0, 1, 0, 1, 0]}
    test = {"seqs": SEQUENCES[10:], "labels": [1, 0]}
    return train, valid, test


def _manifest(data_name: str = "toy") -> dict:
    assignment = {sequence_digest(sequence): name for sequence, name in zip(SEQUENCES, ASSIGNED, strict=True)}
    rows = {"train": 7, "validation": 2, "calibration": 2, "test": 2}
    return {"scheme": SPLIT_SCHEME, "data_name": data_name, "rows": rows, "split": {"protein_to_split": assignment}}


def _cell_data(data_name: str = "toy") -> CellData:
    train, valid, test = _published_splits()
    return CellData(SimpleNamespace(datasets={}), data_name, train, valid, test, 2, "singlelabel", False)


def test_the_manifest_moves_every_row_with_its_sequence():
    cell_data, (fit_rows, cal_rows) = apply_split_manifest(_cell_data(), _manifest())

    assert cell_data.train["seqs"] == SEQUENCES[:6] + [SEQUENCES[0]]
    assert cell_data.valid["seqs"] == SEQUENCES[6:10]
    assert cell_data.test["seqs"] == SEQUENCES[10:]
    assert cell_data.valid["labels"] == [0, 1, 0, 1]
    assert fit_rows.tolist() == [0, 1] and cal_rows.tolist() == [2, 3]
    assert cell_data.main.datasets["toy"][:3] == (cell_data.train, cell_data.valid, cell_data.test)


def test_a_manifest_that_does_not_match_the_rows_is_refused():
    with pytest.raises(ValueError, match="is for 'other'"):
        apply_split_manifest(_cell_data(), _manifest("other"))

    manifest = _manifest()
    del manifest["split"]["protein_to_split"][sequence_digest(SEQUENCES[11])]
    with pytest.raises(ValueError, match="names no split for 1"):
        apply_split_manifest(_cell_data(), manifest)

    manifest = _manifest()
    manifest["rows"]["train"] = 6
    with pytest.raises(ValueError, match="but the manifest recorded"):
        apply_split_manifest(_cell_data(), manifest)

    with pytest.raises(ValueError, match="scheme"):
        split_row_indices([], {"scheme": "carve"})


def test_pair_rows_are_refused():
    pairs = {"SeqA": ["MA"], "SeqB": ["MC"], "labels": [1]}
    cell_data = CellData(SimpleNamespace(datasets={}), "toy", pairs, pairs, pairs, 2, "singlelabel", True)
    with pytest.raises(ValueError, match="pair rows"):
        apply_split_manifest(cell_data, _manifest())


def test_each_protein_takes_its_first_rows_label():
    ids = ["a", "b", "a", "c"]
    assert protein_labels(ids, [1, 0, 0, 2], "singlelabel") == {"a": "1", "b": "0", "c": "2"}

    multilabel = [[0, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0]]  # label 0 is rarest, then label 2
    assert protein_labels(ids, multilabel, "multilabel") == {"a": "2", "b": "0", "c": "1"}

    quintiles = protein_labels([str(index) for index in range(10)], [float(index) for index in range(10)], "regression")
    assert len(set(quintiles.values())) == 5


def test_the_manifest_pools_every_split_and_records_each_splits_rows(tmp_path, monkeypatch):
    seen = {}

    def split_by_strata(sequences, directory, *, labels, settings, **options):
        seen.update(sequences=sequences, labels=labels, seed=settings.seed, directory=directory, options=options)
        assignment = {protein: name for protein, name in zip(sequences, ASSIGNED, strict=True)}
        return None, IdentityStratifiedSplit(assignment, {}, {}, {"scheme": "identity_strata"})

    monkeypatch.setattr(stratified_split, "build_identity_stratified_split", split_by_strata)
    monkeypatch.setattr(stratified_split, "validate_identity_stratified_split", lambda hierarchy, split: None)
    path = tmp_path / "splits" / "toy.json"
    manifest = build_split_manifest("toy", _published_splits(), "singlelabel", str(path), seed=7, threads=2)

    assert list(seen["sequences"].values()) == SEQUENCES
    assert seen["labels"][sequence_digest(SEQUENCES[0])] == "0"
    assert seen["seed"] == 7 and seen["options"] == {"threads": 2}
    assert seen["directory"] == tmp_path / "splits" / "toy_mmseqs"
    assert manifest["pooled_rows"] == 13 and manifest["proteins"] == 12
    assert manifest["rows"] == _manifest()["rows"]
    assert read_json(str(path)) == manifest

    # An existing manifest is kept, never split again.
    monkeypatch.setattr(stratified_split, "build_identity_stratified_split", None)
    assert build_split_manifest("toy", _published_splits(), "singlelabel", str(path), seed=8) == manifest
    cell_data, _ = apply_split_manifest(_cell_data(), manifest)
    assert np.array_equal(cell_data.test["labels"], [1, 0])


def test_a_cell_with_a_manifest_takes_its_roles_from_it(monkeypatch, tmp_path):
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(_manifest()))
    monkeypatch.setattr(workflow, "load_cell_data", lambda arguments: _cell_data())
    cell = SimpleNamespace(uncertainty=SimpleNamespace(split_manifest=str(path)))

    cell_data, (fit_rows, cal_rows) = workflow.load_split_cell_data(cell, None, None)
    assert cell_data.valid["seqs"] == SEQUENCES[6:10]
    assert cal_rows.tolist() == [2, 3]
    with pytest.raises(ValueError, match="explicit rows cannot be given too"):
        workflow.load_split_cell_data(cell, None, ([0], [1]))

    plain = SimpleNamespace(uncertainty=SimpleNamespace(split_manifest=None))
    cell_data, rows = workflow.load_split_cell_data(plain, None, ([0], [1]))
    assert cell_data.valid["seqs"][-1] == SEQUENCES[0] and rows == ([0], [1])
