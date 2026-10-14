"""Compact labels preserve measured zeros and unknowns through real joined-head training."""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import numpy as np
import pytest
import torch

from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace
from datasets import Dataset
from torch.utils.data import DataLoader

from protify.data.branch_features import JoinedBranchCollator, JoinedBranchDataset, JoinedBranchEmbeddings
from protify.data.cluster_sampling import ClusterAssaySampler, ClusterSamplingPlan
from protify.data.label_batches import SparseLabelRow, SparseObservedTargets, _columns, collate_sparse_labels, observed_row_counts
from protify.probes.projected_probe import ProjectedProbe, ProjectedProbeConfig
from protify.uncertainty.calibration_split import row_digests
from protify.uncertainty.workflow import _record_observations


@pytest.mark.parametrize("measured", [{}, {0: 0}, {1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1, 2: 0}, {0: 1, 1: 0, 2: 1, 3: 0}])
def test_sparse_source_records_preserve_measured_zero_unknown_and_mask_polarity(measured):
    row = SparseLabelRow.from_observed(4, measured)
    labels, mask = row.to_records()
    restored = SparseLabelRow.from_records(labels, mask)
    assert observed_row_counts(labels, mask) == ((4,), len(measured), 4)
    batch = collate_sparse_labels([restored])  # (b=1, c=4)
    assert batch["labels"].tolist() == [[measured.get(index, 0) for index in range(4)]]
    assert batch["label_mask"].tolist() == [[index in measured for index in range(4)]]
    assert row.default_observed == (len(measured) > 2)
    assert len(row.mask_columns) == min(len(measured), 4 - len(measured))


@pytest.mark.parametrize("change", ["width", "width_bool", "extra_label", "extra_mask", "duplicate", "order", "outside", "index_bool", "zero", "nan", "overflow", "underflow", "values_length", "unknown_nonzero", "mask_type", "mask_polarity", "missing_mask"])
def test_sparse_source_records_fail_before_collation(change):
    labels, mask = SparseLabelRow.from_observed(4, {0: 1, 2: 1}).to_records()
    if change == "width":
        labels["width"] = 0
    elif change == "width_bool":
        labels["width"] = True
    elif change == "extra_label":
        labels["unused"] = 1
    elif change == "extra_mask":
        mask["unused"] = 1
    elif change == "duplicate":
        labels["indices"] = [0, 0]
    elif change == "order":
        labels["indices"] = [2, 0]
    elif change == "outside":
        labels["indices"] = [0, 4]
    elif change == "index_bool":
        labels["indices"] = [False, 2]
    elif change in {"zero", "nan", "overflow", "underflow"}:
        labels["values"][0] = {"zero": 0, "nan": float("nan"), "overflow": 1e100, "underflow": 1e-100}[change]
    elif change == "values_length":
        labels["values"].pop()
    elif change == "unknown_nonzero":
        mask["exceptions"] = [0]
    elif change == "mask_type":
        mask["default_observed"] = 0
    elif change == "mask_polarity":
        mask["default_observed"] = True
        mask["exceptions"] = [1, 3]
    else:
        mask = None
    with pytest.raises(ValueError):
        SparseLabelRow.from_records(labels, mask)


def test_sparse_source_storage_never_expands_the_vocabulary(monkeypatch):
    row = SparseLabelRow.from_observed(1_000_000, {0: 1, 2: 0, 999_999: 1})
    labels, mask = row.to_records()
    original = np.asarray
    def small_arrays(value, *args, **kwargs):
        assert not hasattr(value, "__len__") or len(value) < 100
        return original(value, *args, **kwargs)
    monkeypatch.setattr(np, "asarray", small_arrays)
    storage = SparseObservedTargets.from_rows([labels] * 3, [mask] * 3)
    assert storage.values.shape == (3, 1_000_000)
    assert storage.values.nnz == 6 and storage.mask_entries.nnz == 9
    assert storage.storage_nbytes < 1024
    assert observed_row_counts(labels, mask) == ((1_000_000,), 3, 1_000_000)


def per_element_columns(values, width):
    """The check `_columns` runs in bulk, written one element at a time."""
    if not isinstance(values, list) or any(type(value) is not int or not 0 <= value < width for value in values):
        raise ValueError("sparse columns must be integer indices inside the frozen vocabulary")
    if any(left >= right for left, right in zip(values, values[1:], strict=False)):
        raise ValueError("sparse columns must be strictly increasing and unique")
    return np.asarray(values, dtype=np.int64)


COLUMN_CASES = [
    [], [0], [0, 3], [3, 0], [1, 1], [-1, 2], [0, 4], [4], [True, 2], [False], [np.int64(1)], [np.int64(1), 2], [1.0], [2**70],
    [-(2**70)], [0, 2**63], (0, 1), None, "01", [0, None], [[0]], [0, 1, 2, 3], [0, 1, 1, 3], [1, 0],
]


@pytest.mark.parametrize("values", COLUMN_CASES, ids=repr)
def test_bulk_column_checks_give_the_per_element_answer(values):
    try:
        expected = per_element_columns(values, 4)
    except ValueError as error:
        with pytest.raises(ValueError) as raised:
            _columns(values, 4)
        assert str(raised.value) == str(error)
        return
    actual = _columns(values, 4)
    assert actual.dtype == expected.dtype and actual.tolist() == expected.tolist()


def test_bulk_column_checks_agree_on_random_lists():
    generator = np.random.default_rng(3)
    for _ in range(400):
        values = sorted(generator.choice(40, size=int(generator.integers(0, 12)), replace=False).tolist())
        if values and generator.random() < 0.5:
            values[int(generator.integers(0, len(values)))] += int(generator.integers(-3, 45))  # out of order, duplicated or outside
        if values and generator.random() < 0.1:
            values[int(generator.integers(0, len(values)))] = float(values[0])
        outcomes = []
        for check in (per_element_columns, _columns):
            try:
                outcomes.append(check(values, 40).tolist())
            except ValueError as error:
                outcomes.append(str(error))
        assert outcomes[0] == outcomes[1]


def test_random_sparse_rows_round_trip_and_compress_like_dense_rows():
    generator = np.random.default_rng(5)
    width = 30
    dense_labels, dense_masks, records, measured_rows = [], [], [], []
    for _ in range(60):
        measured = {int(column): float(generator.choice([0.0, 1.0, 0.25, -2.0])) for column in generator.choice(width, size=int(generator.integers(0, width + 1)), replace=False)}
        row = SparseLabelRow.from_observed(width, measured)
        labels, mask = row.to_records()
        restored = SparseLabelRow.from_records(labels, mask)
        for field in ("indices", "values", "mask_columns"):
            assert getattr(restored, field).tolist() == getattr(row, field).tolist()
        assert restored.default_observed == row.default_observed and restored.observed_count == len(measured)
        dense_labels.append([measured.get(column, 0.0) for column in range(width)])
        dense_masks.append([column in measured for column in range(width)])
        records.append((labels, mask))
        measured_rows.append(measured)
    compact = SparseObservedTargets.from_rows([labels for labels, _ in records], [mask for _, mask in records])
    dense = SparseObservedTargets.from_rows(np.array(dense_labels, dtype=np.float32), np.array(dense_masks))
    assert (compact.values != dense.values).nnz == 0 and (compact.mask_entries != dense.mask_entries).nnz == 0
    assert compact.values.data.dtype == dense.values.data.dtype == np.float32
    assert compact.default_observed.tolist() == dense.default_observed.tolist()
    batch = collate_sparse_labels([compact.row(index) for index in range(60)])
    assert batch["labels"].tolist() == np.array(dense_labels, dtype=np.float32).tolist()
    assert batch["label_mask"].tolist() == dense_masks


def test_sparse_resume_counts_and_crossfit_guards_preserve_unknowns(tmp_path):
    from protify.uncertainty.crossfit import _require_complete_targets

    records = [SparseLabelRow.from_observed(4, measured).to_records() for measured in ({0: 0, 1: 1}, {}, {2: 1})]
    split = {"seqs": ["AC", "DE", "FG"], "labels": [row[0] for row in records], "label_mask": [row[1] for row in records]}
    cell_data = SimpleNamespace(train=split, valid=split, test=split)
    assert _record_observations(str(tmp_path), cell_data) is True
    assert _record_observations(str(tmp_path), cell_data) is True
    saved = json.loads((tmp_path / "label_observations.json").read_text())
    assert saved["splits"]["train"]["observed_labels"] == 3
    assert saved["splits"]["train"]["unobserved_labels"] == 9
    with pytest.raises(ValueError, match="partial labels"):
        _require_complete_targets(split["labels"], split["label_mask"], "fixture")
    labels, mask = SparseLabelRow.from_observed(4, {0: 1, 1: 0, 2: 0, 3: 0}).to_records()
    _require_complete_targets([labels], [mask], "complete fixture")


def test_sampling_streams_masks_and_has_identical_draws_for_both_encodings():
    dense = {"observation_id": ["a", "b", "c"], "protein_id": ["p", "p", "q"], "cluster_id": ["c", "c", "d"],
             "labels": [[0, 1], [0, 0], [1, 0]], "label_mask": [[True, True], [False, False], [True, False]]}
    expected = ClusterSamplingPlan.from_dataset(dense, 30)
    records = [SparseLabelRow.from_observed(2, observed).to_records() for observed in ({0: 0, 1: 1}, {}, {0: 1})]
    compact = {**dense, "labels": StreamingColumn([row[0] for row in records]),
               "label_mask": StreamingColumn([row[1] for row in records])}
    actual = ClusterSamplingPlan.from_dataset(compact, 30)
    assert actual == expected
    assert actual.record()["eligible_observations"] == 2
    assert list(ClusterAssaySampler(actual, 42)) == list(ClusterAssaySampler(expected, 42))


@pytest.mark.parametrize("explicit", [False, True])
def test_sparse_batch_matches_dense_values_and_observation_meaning(explicit):
    labels = np.array([[1, 0, 0.25, 0], [0, 0, 0, 0], [0, 1, 0, 1]], dtype=np.float32)
    mask = np.array([[True, True, False, False], [False] * 4, [True] * 4]) if explicit else None
    if explicit:
        labels[~mask] = np.nan
    storage = SparseObservedTargets.from_rows(labels, mask)
    batch = collate_sparse_labels([storage.row(index) for index in (2, 0, 1, 0)])
    selected = [2, 0, 1, 0]
    expected = labels if mask is None else np.where(mask, labels, 0)
    torch.testing.assert_close(batch["labels"], torch.from_numpy(expected[selected]))
    assert batch["labels"].dtype == torch.float32
    if explicit:
        torch.testing.assert_close(batch["label_mask"], torch.from_numpy(mask[selected]))
        assert storage.values.nnz == 3 and storage.mask_entries.nnz == 2
        assert storage.row(0).mask_columns.tolist() == [0, 1]  # Includes the measured zero.
    else:
        assert "label_mask" not in batch and storage.mask_entries.nnz == 0
    assert storage.row(-1).indices.tolist() == storage.row(2).indices.tolist()
    with pytest.raises(IndexError):
        storage.row(3)


@pytest.mark.parametrize("case", ["scalar", "empty", "ragged", "text", "complex", "mask_count", "mask_shape", "mask_numeric", "observed_nan", "observed_inf"])
def test_invalid_targets_fail_before_training(case):
    labels, masks = [[1.0, 0.0], [0.0, 1.0]], [[True, True], [True, False]]
    if case == "scalar":
        labels[0] = 1.0
    elif case == "empty":
        labels[0] = []
    elif case == "ragged":
        labels[1], masks[1] = [1.0], [True]
    elif case == "text":
        labels[0] = ["1", "0"]
    elif case == "complex":
        labels[0] = [1j, 0j]
    elif case == "mask_count":
        masks.pop()
    elif case == "mask_shape":
        masks[0] = [True]
    elif case == "mask_numeric":
        masks[0] = [1, 0]
    else:
        labels[0][0] = float("nan" if case == "observed_nan" else "inf")
    with pytest.raises(ValueError):
        SparseObservedTargets.from_rows(labels, masks)


def test_empty_dataset_and_mixed_batch_contracts_are_explicit():
    empty = SparseObservedTargets.from_rows([])
    assert empty.values.shape == (0, 0)
    with pytest.raises(ValueError, match="empty"):
        collate_sparse_labels([])
    row = SparseObservedTargets.from_rows([[1, 0]]).row(0)
    for changed in (replace(row, width=3), replace(row, explicit_mask=True)):
        with pytest.raises(ValueError, match="mix"):
            collate_sparse_labels([row, changed])
    with pytest.raises(ValueError, match="explicit label mask"):
        collate_sparse_labels([replace(row, mask_columns=np.array([0]))])


def test_nearly_complete_masks_store_unknown_exceptions_without_expanding_negatives():
    labels = np.zeros((3, 1000), dtype=np.float32)
    labels[:, 0] = 1
    masks = np.ones_like(labels, dtype=np.bool_)
    masks[0, 8] = False
    masks[1, [7, 9]] = False
    storage = SparseObservedTargets.from_rows(labels, masks)
    assert storage.mask_entries.nnz == 3 and storage.default_observed.all()
    assert storage.complete_rows.tolist() == [False, False, True]
    batch = collate_sparse_labels([storage.row(index) for index in range(3)])
    torch.testing.assert_close(batch["label_mask"], torch.from_numpy(masks))
    assert storage.storage_nbytes < 128


class RepeatedRows(Sequence):
    """A streamed source that refuses corpus-level array conversion or slicing."""

    def __init__(self, count, row):
        self.count, self.template, self.reads = count, row, 0

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise AssertionError("source must be read one row at a time")
        if index >= self.count:
            raise IndexError
        self.reads += 1
        return self.template

    def __array__(self, *args, **kwargs):
        raise AssertionError("corpus-wide dense label conversion is forbidden")


def test_storage_scales_with_observed_entries_and_selected_batch_not_vocabulary_product(monkeypatch):
    width, count = 100_000, 100
    target = np.zeros(width, dtype=np.float32)
    target[[2, 91]] = 1
    measured = np.zeros(width, dtype=np.bool_)
    measured[[2, 91, 999]] = True
    labels, masks = RepeatedRows(count, target), RepeatedRows(count, measured)
    storage = SparseObservedTargets.from_rows(labels, masks)
    assert labels.reads == masks.reads == count
    assert storage.values.nnz == 2 * count and storage.mask_entries.nnz == 3 * count
    assert storage.storage_nbytes < 100 * count
    # No corpus matrix is densified by either row access or batch assembly.
    monkeypatch.setattr(type(storage.values), "toarray", lambda *args, **kwargs: pytest.fail("whole CSR matrix densified"))
    batch = collate_sparse_labels([storage.row(0), storage.row(count - 1)])
    assert batch["labels"].shape == batch["label_mask"].shape == (2, width)
    assert batch["label_mask"].sum() == 6
    assert len(pickle.dumps(storage.row(0))) < 2048  # Worker transport carries one sparse row.


@pytest.mark.parametrize("as_hf", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_joined_training_and_prediction_inputs_equal_dense_reference(as_hf, explicit):
    torch.manual_seed(17)
    sequences = ["MACD", "MEFG", "MHIK", "MLNP"]
    features = np.arange(24, dtype=np.float32).reshape(4, 6) / 24
    labels = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0], [1, 1, 0]], dtype=np.float32)
    masks = np.array([[True, True, False], [False] * 3, [True] * 3, [False, True, True]])
    columns = {"seqs": sequences, "labels": labels.tolist()}
    if explicit:
        columns["label_mask"] = masks.tolist()
        columns["labels"] = np.where(masks, labels, np.nan).tolist()
    source = Dataset.from_dict(columns) if as_hf else columns
    joined = JoinedBranchEmbeddings(sequences, [features])
    dataset = JoinedBranchDataset(source, joined, "multilabel", False)
    assert isinstance(dataset[0][1], SparseLabelRow)
    if not as_hf:
        assert dataset.labels is columns["labels"]  # No second corpus-sized Python list.
    collator = JoinedBranchCollator(joined, False, None)
    compact = ProjectedProbe(ProjectedProbeConfig(input_size=6, projection_size=8, dropout=0.0, task_type="multilabel", num_labels=3))
    dense = copy.deepcopy(compact)
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.02) for model in (compact, dense)]
    for start in (0, 2, 0):
        indices = [start, start + 1]
        batch = collator([dataset[index] for index in indices])
        reference = {"embeddings": torch.from_numpy(features[indices]), "labels": torch.from_numpy(np.where(masks[indices], labels[indices], 0) if explicit else labels[indices])}
        if explicit:
            reference["label_mask"] = torch.from_numpy(masks[indices])
        for key in reference:
            torch.testing.assert_close(batch[key], reference[key])
        losses = []
        for model, optimizer, inputs in zip((compact, dense), optimizers, (batch, reference), strict=True):
            optimizer.zero_grad()
            loss = model(**inputs).loss
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        torch.testing.assert_close(*losses)
    for key, value in compact.state_dict().items():
        torch.testing.assert_close(value, dense.state_dict()[key])
    compact.eval()
    dense.eval()
    torch.testing.assert_close(compact(torch.from_numpy(features)).logits, dense(torch.from_numpy(features)).logits)


def test_collator_keeps_pair_features_and_rejects_mixed_target_representations():
    joined = JoinedBranchEmbeddings(["MACD", "MEFG"], [np.array([[1, 2], [3, 4]], dtype=np.float32)])
    columns = {"SeqA": ["MACD"], "SeqB": ["MEFG"], "labels": [[1, 0]], "label_mask": [[True, False]]}
    dataset = JoinedBranchDataset(columns, joined, "multilabel", True)
    collator = JoinedBranchCollator(joined, True, None)
    batch = collator([dataset[0]])
    assert batch["embeddings"].tolist() == [[1, 2, 3, 4]]
    with pytest.raises(ValueError, match="mix sparse and dense"):
        collator([dataset[0], ((0, 1), torch.tensor([1.0, 0.0]))])


def test_worker_round_trip_preserves_compact_rows_and_masks():
    sequences = ["MACD", "MEFG"]
    joined = JoinedBranchEmbeddings(sequences, [np.zeros((2, 3), dtype=np.float32)])
    source = {"seqs": sequences, "labels": [[1, 0], [0, 1]], "label_mask": [[True, False], [True, True]]}
    dataset = JoinedBranchDataset(source, joined, "multilabel", False)
    batches = list(DataLoader(dataset, batch_size=1, num_workers=1, multiprocessing_context="spawn", collate_fn=JoinedBranchCollator(joined, False, None)))
    assert [batch["labels"].tolist() for batch in batches] == [[[1, 0]], [[0, 1]]]
    assert [batch["label_mask"].tolist() for batch in batches] == [[[True, False]], [[True, True]]]


class StreamingColumn:
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        raise AssertionError("whole-column list materialization is forbidden")

    def __array__(self, *args, **kwargs):
        raise AssertionError("whole-column array materialization is forbidden")


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_streamed_row_digests_keep_existing_bytes_without_materializing_columns(paired, masked):
    columns = {"labels": [[1.0, 0.0], [0.0, 1.0]]}
    sequences = {"SeqA": ["MACD", "MEFG"], "SeqB": ["MHIK", "MLNP"]} if paired else {"seqs": ["MACD", "MEFG"]}
    columns.update(sequences)
    if masked:
        columns["label_mask"] = [[True, False], [True, True]]
    expected = []
    for index in range(2):
        record = {"sequences": [values[index] for values in sequences.values()], "label": columns["labels"][index]}
        if masked:
            record["label_mask"] = columns["label_mask"][index]
        expected.append(hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:16])
    assert row_digests({key: StreamingColumn(values) for key, values in columns.items()}) == expected
    columns["labels"].pop()
    with pytest.raises(ValueError):
        row_digests({key: StreamingColumn(values) for key, values in columns.items()})


def test_streamed_resume_counts_and_pins_equal_the_existing_record(tmp_path):
    split = {"seqs": ["MACD", "MEFG"], "labels": [[1, 0, 0], [0, 1, 0]], "label_mask": [[True, True, False], [False, True, False]]}
    _record_observations(str(tmp_path), SimpleNamespace(train=split, valid=split, test=split))
    path = tmp_path / "label_observations.json"
    expected = path.read_bytes()
    metadata = json.loads(expected)
    assert metadata["splits"]["train"]["observed_labels"] == 3
    assert metadata["splits"]["train"]["unobserved_labels"] == 3
    stream = {key: StreamingColumn(values) for key, values in split.items()}
    _record_observations(str(tmp_path), SimpleNamespace(train=stream, valid=stream, test=stream))
    assert path.read_bytes() == expected
    split["label_mask"][0][2] = True
    with pytest.raises(ValueError, match="observations changed"):
        _record_observations(str(tmp_path), SimpleNamespace(train=stream, valid=stream, test=stream))


def test_streamed_resume_rejects_ragged_targets_before_recording(tmp_path):
    split = {"seqs": ["MACD", "MEFG"], "labels": [[1, 0], [0]], "label_mask": [[True, True], [True]]}
    with pytest.raises(ValueError, match="one target shape"):
        _record_observations(str(tmp_path), SimpleNamespace(train=split, valid=split, test=split))
    assert not (tmp_path / "label_observations.json").exists()
