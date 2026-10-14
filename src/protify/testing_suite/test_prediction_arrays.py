"""Bounded prediction collection, NumPy interoperability, and corruption refusal."""

import io
import struct
import zipfile
import numpy as np
import pytest
import torch

from dataclasses import fields
from types import SimpleNamespace
from torch import nn
from torch.utils.data import Dataset

from protify.uncertainty import prediction_arrays, predictions
from protify.uncertainty.prediction_arrays import DiskPredictionArray, PredictionAccumulator, mapped_prediction_archive, write_prediction_archive
from protify.uncertainty.predictions import SplitPredictions, collect_predictions, load_seed_predictions, save_seed_predictions


@pytest.mark.parametrize("disk_backed", [False, True])
def test_accumulator_preserves_rows_after_its_temporary_handles_close(disk_backed):
    expected = np.arange(21, dtype=np.float32).reshape(7, 3)
    with PredictionAccumulator(7, disk_backed=disk_backed) as accumulated:
        accumulated.append({"outputs": expected[:2], "labels": expected[:2] % 2})
        accumulated.append({"outputs": expected[2:], "labels": expected[2:] % 2})
        arrays = accumulated.finish()
    assert all(stream.closed for stream in accumulated.streams.values())
    np.testing.assert_array_equal(arrays["outputs"], expected)
    np.testing.assert_array_equal(arrays["labels"], expected % 2)
    assert isinstance(arrays["outputs"], DiskPredictionArray) == disk_backed
    assert arrays["outputs"].flags.writeable != disk_backed
    if disk_backed:
        with pytest.raises(ValueError, match="read-only"):
            arrays["outputs"][0, 0] = 100


@pytest.mark.parametrize("disk_backed", [False, True])
@pytest.mark.parametrize("change", ["missing_field", "shape", "dtype", "overflow", "incomplete", "unaligned", "object"])
def test_accumulator_refuses_partial_or_changing_batches(disk_backed, change):
    with PredictionAccumulator(3, disk_backed=disk_backed) as accumulated:
        accumulated.append({"labels": np.zeros((1, 2)), "outputs": np.ones((1, 2))})
        batch = {"labels": np.zeros((2, 2)), "outputs": np.ones((2, 2))}
        if change == "missing_field":
            batch.pop("labels")
        elif change == "shape":
            batch["labels"] = np.zeros((2, 3))
        elif change == "dtype":
            batch["labels"] = batch["labels"].astype(np.float32)
        elif change == "overflow":
            batch = {name: np.zeros((3, 2)) for name in batch}
        elif change == "unaligned":
            batch["labels"] = np.zeros((1, 2))
        elif change == "object":
            batch["labels"] = np.empty((2, 2), dtype=object)
        with pytest.raises(ValueError):
            accumulated.finish() if change == "incomplete" else accumulated.append(batch)
    assert all(stream.closed for stream in accumulated.streams.values())


class PredictionRows(Dataset):
    def __init__(self, kind, masked):
        self.embeddings = torch.arange(21, dtype=torch.float32).reshape(7, 3) / 10
        self.labels = torch.arange(7, dtype=torch.int64) % 3
        if kind == "multilabel":
            self.labels = torch.eye(3)[self.labels]
        elif kind == "regression":
            self.labels = self.labels.float()
        self.mask = torch.ones_like(self.labels, dtype=torch.bool) if masked else None
        if self.mask is not None:
            self.mask[0] = False
        self.observation_ids = [f"row_{i}" for i in range(7)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"embeddings": self.embeddings[index], "labels": self.labels[index],
                **({"label_mask": self.mask[index]} if self.mask is not None else {})}


class PredictionHead(nn.Module):
    def __init__(self, width, variance):
        super().__init__()
        self.predicts_variance = variance
        self.output = nn.Linear(3, width)
        self.dropout = nn.Dropout(0.25)

    def forward(self, embeddings):
        return SimpleNamespace(logits=self.output(self.dropout(embeddings)))

    def predict_with_variance(self, embeddings):
        return self(embeddings).logits, embeddings.square().mean(dim=-1)


@pytest.mark.parametrize("kind", ["multiclass", "multilabel", "regression"])
@pytest.mark.parametrize("variance", [False, True])
@pytest.mark.parametrize("mc_samples", [0, 3])
@pytest.mark.parametrize("masked", [False, True])
def test_disk_and_ram_collection_are_exact_for_every_output_field(kind, variance, mc_samples, masked, monkeypatch):
    dataset = PredictionRows(kind, masked)
    model = PredictionHead(1 if kind == "regression" else 3, variance)
    results = []
    monkeypatch.setattr(predictions.np, "concatenate", lambda *_args, **_kwargs: pytest.fail("collector concatenated complete batch lists"))
    for disk_backed in (False, True):
        torch.manual_seed(42)
        split_predictions, seconds = collect_predictions(model, dataset, None, task_kind=kind, batch_size=3,
                                             device=torch.device("cpu"), mc_samples=mc_samples, disk_backed=disk_backed)
        assert seconds >= 0
        results.append(split_predictions)
    for field in fields(SplitPredictions):
        expected, actual = getattr(results[0], field.name), getattr(results[1], field.name)
        if expected is None:
            assert actual is None
        else:
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("numpy_writer", [False, True])
def test_archives_interoperate_with_numpy_and_preserve_layouts(tmp_path, monkeypatch, numpy_writer):
    arrays = {
        "c": np.arange(21, dtype=np.float32).reshape(7, 3),
        "f": np.asfortranarray(np.arange(21, dtype=np.float64).reshape(7, 3)),
        "sliced": np.arange(42, dtype=np.int32).reshape(7, 6)[:, ::2],
        "strings": np.array(["assay_α", "assay_β"]),
        "empty": np.zeros((0, 3), dtype=bool), "scalar": np.array(2.5),
        "big_endian": np.arange(9, dtype=">i4"),
    }
    path = tmp_path / "arrays.npz"
    monkeypatch.setattr(prediction_arrays, "BUFFER_BYTES", 16)
    with path.open("wb") as stream:
        np.savez(stream, **arrays) if numpy_writer else write_prediction_archive(stream, arrays)
    mapped = mapped_prediction_archive(path)
    with np.load(path, allow_pickle=False) as original:
        for name, values in arrays.items():
            np.testing.assert_array_equal(original[name], values)
            np.testing.assert_array_equal(mapped[name], values)
            assert mapped[name].dtype == values.dtype
            assert not mapped[name].flags.writeable
    assert mapped["f"].flags.f_contiguous
    assert mapped["c"].flags.c_contiguous


def test_saved_prediction_fields_and_legacy_mutability_survive_mapping(tmp_path):
    labels = np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.float32)
    expected = SplitPredictions(labels, np.arange(4, dtype=np.float32).reshape(2, 2),
                                label_mask=np.array([[True, False], [True, True]]), observation_ids=np.array(["first", "second"]))
    path = str(tmp_path / "seed.npz")
    save_seed_predictions(path, {"test": expected})
    before = (tmp_path / "seed.npz").read_bytes()
    mapped = load_seed_predictions(path, mmap_mode="r")["test"]
    for field in fields(SplitPredictions):
        value = getattr(expected, field.name)
        if value is not None:
            np.testing.assert_array_equal(getattr(mapped, field.name), value)
    assert isinstance(mapped.labels, DiskPredictionArray)
    legacy = load_seed_predictions(path)["test"]
    legacy.outputs[0, 0] = -99
    assert mapped.outputs[0, 0] == 0
    assert (tmp_path / "seed.npz").read_bytes() == before
    with pytest.raises(ValueError, match="mmap_mode"):
        load_seed_predictions(path, mmap_mode="r+")


def test_mapping_rejects_compression_but_legacy_loading_remains_supported(tmp_path):
    path = tmp_path / "compressed.npz"
    np.savez_compressed(path, test__labels=np.array([1]), test__outputs=np.ones((1, 2)))
    assert load_seed_predictions(str(path))["test"].labels.tolist() == [1]
    with pytest.raises(ValueError, match="uncompressed"):
        load_seed_predictions(str(path), mmap_mode="r")


def test_mapping_checks_crc_before_exposing_corrupted_array(tmp_path):
    path = tmp_path / "corrupt.npz"
    np.savez(path, values=np.arange(10, dtype=np.float32))
    with zipfile.ZipFile(path) as archive:
        record = archive.infolist()[0]
    with path.open("r+b") as stream:
        stream.seek(record.header_offset)
        local = stream.read(30)
        lengths = struct.unpack("<HH", local[26:30])
        position = record.header_offset + 30 + sum(lengths) + record.file_size - 1
        stream.seek(position)
        value = stream.read(1)
        stream.seek(position)
        stream.write(bytes([value[0] ^ 1]))
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        mapped_prediction_archive(path)


@pytest.mark.parametrize("change", ["object", "duplicate", "truncated_array", "unsupported_version"])
def test_mapping_refuses_ambiguous_or_invalid_members(tmp_path, change):
    path = tmp_path / "invalid.npz"
    member = io.BytesIO()
    if change == "object":
        np.save(member, np.array([None], dtype=object))
    elif change == "truncated_array":
        np.lib.format.write_array_header_2_0(member, {"descr": "<f4", "fortran_order": False, "shape": (2**40, 3)})
    else:
        np.save(member, np.arange(3))
    content = member.getvalue()
    if change == "unsupported_version":
        content = content[:6] + b"\x03\x00" + content[8:]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("values.npy", content)
        if change == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("values.npy", content)
    with pytest.raises(ValueError):
        mapped_prediction_archive(path)


def test_failed_archive_write_preserves_the_previous_completed_file(tmp_path, monkeypatch):
    path = tmp_path / "seed.npz"
    expected = SplitPredictions(np.array([1]), np.ones((1, 2)))
    save_seed_predictions(str(path), {"test": expected})
    before = path.read_bytes()
    def interrupted(stream, arrays):
        stream.write(b"interrupted")
        raise OSError("fixture disk failure")
    monkeypatch.setattr(predictions, "write_prediction_archive", interrupted)
    with pytest.raises(OSError, match="disk failure"):
        save_seed_predictions(str(path), {"test": expected})
    assert path.read_bytes() == before
    np.testing.assert_array_equal(load_seed_predictions(str(path), mmap_mode="r")["test"].outputs, expected.outputs)


def test_observed_validation_does_not_make_a_whole_matrix_boolean_selection():
    class GuardedLabels(np.ndarray):
        def __getitem__(self, key):
            if isinstance(key, np.ndarray) and key.dtype == bool and key.size > 1_000_000:
                raise AssertionError("whole-matrix observed selection")
            return super().__getitem__(key)
    labels = np.zeros((5, 300_000), dtype=np.float32).view(GuardedLabels)
    mask = np.ones_like(labels, dtype=bool)
    SplitPredictions(labels, labels, label_mask=mask)
    labels[-1, -1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        SplitPredictions(labels, labels, label_mask=mask)
