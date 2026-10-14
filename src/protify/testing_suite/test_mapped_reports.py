"""Mapped report selection and reductions match dense scores, masks, and row order."""

import numpy as np
import pytest

from scipy.special import softmax

from protify.uncertainty import prediction_arrays
from protify.uncertainty.methods import Predictive
from protify.uncertainty.metrics import brier_score, gaussian_nll, negative_log_likelihood, row_losses
from protify.uncertainty.prediction_arrays import DiskPredictionArray, PredictionAccumulator, prediction_row_blocks, select_prediction_rows
from protify.uncertainty.report import label_support, score_rows


def mapped(values):
    with PredictionAccumulator(len(values), disk_backed=True) as accumulator:
        accumulator.append({"values": values})
        return accumulator.finish()["values"]


@pytest.mark.parametrize("rows", [slice(None), slice(1, 14, 2), slice(None, None, -1), np.array([8, 1, -1, 1]), np.arange(17) % 3 == 0, np.array([], dtype=int)])
def test_selection_preserves_order_duplicates_dtype_and_mapping(monkeypatch, rows):
    monkeypatch.setattr(prediction_arrays, "BUFFER_BYTES", 32)
    values = np.arange(68, dtype=np.float64).reshape(17, 4)
    actual = select_prediction_rows(mapped(values), rows)
    np.testing.assert_array_equal(actual, values[rows])
    assert isinstance(actual, DiskPredictionArray) and not actual.flags.writeable
    assert actual.dtype == values.dtype


def test_selection_failure_closes_files_and_releases_source(monkeypatch):
    values = mapped(np.ones((7, 4)))
    monkeypatch.setattr(prediction_arrays, "BUFFER_BYTES", 64)
    original = prediction_arrays.tempfile.TemporaryFile
    files, releases = [], []
    def track(**kwargs):
        stream = original(**kwargs)
        files.append(stream)
        return stream
    monkeypatch.setattr(prediction_arrays.tempfile, "TemporaryFile", track)
    monkeypatch.setattr(prediction_arrays, "release_prediction_pages", lambda value: releases.append(value))
    original_append = PredictionAccumulator.append
    def fail_later(self, batch):
        if self.cursor:
            raise RuntimeError("interrupted selection")
        original_append(self, batch)
    monkeypatch.setattr(PredictionAccumulator, "append", fail_later)
    with pytest.raises(RuntimeError, match="interrupted selection"):
        select_prediction_rows(values, np.array([6, 4, 2, 0]))
    assert files and all(stream.closed for stream in files)
    assert len(releases) == 2 and all(value is values for value in releases)


def test_blocks_are_bounded_and_release_on_close(monkeypatch):
    first, second = mapped(np.ones((9, 4))), np.ones(9)
    monkeypatch.setattr(prediction_arrays, "BUFFER_BYTES", 81)
    released = []
    monkeypatch.setattr(prediction_arrays, "release_prediction_pages", lambda values: released.append(values))
    blocks = prediction_row_blocks(first, second)
    block = next(blocks)
    assert all(type(values) is np.ndarray for values in block)
    assert sum(values.nbytes for values in block) == 80
    blocks.close()
    assert len(released) == 2 and released[0] is first and released[1] is second


@pytest.mark.parametrize("kind", ["binary", "multiclass", "multilabel", "regression"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("masked", [False, True])
def test_full_canonical_report_matches_dense_across_identity_selections(monkeypatch, kind, dtype, masked):
    monkeypatch.setattr(prediction_arrays, "BUFFER_BYTES", 80)
    generator = np.random.default_rng(331)
    width = 2 if kind == "binary" else 5
    labels = generator.integers(0, width, size=17)
    probs = softmax(generator.normal(size=(17, width)), axis=1).astype(dtype)
    mean = variance = None
    if kind == "multilabel":
        labels = generator.integers(0, 2, size=(17, width)).astype(dtype)
        probs = generator.random(labels.shape).astype(dtype)
        probs[0, :2] = [0, 1]
    elif kind == "regression":
        labels = generator.normal(size=17).astype(dtype)
        probs = None
        mean, variance = generator.normal(size=17).astype(dtype), generator.random(17).astype(dtype)
    mask = generator.random(labels.shape) > 0.35 if masked else None
    if masked:
        mask[2:5] = False
        if kind in ("regression", "multilabel"):
            labels[~mask] = np.nan
    fields = {name: values for name, values in {"probs": probs, "mean": mean, "variance": variance, "uncertainty": generator.random(17)}.items() if values is not None}
    dense = Predictive(**fields)
    disk = Predictive(**{name: mapped(values) for name, values in fields.items()})
    disk_labels, disk_mask = mapped(labels), None if mask is None else mapped(mask)
    for rows in (None, np.array([16, 1, 6, 0, 6]), np.array([], dtype=int), np.array([2, 3, 4])):
        expected = score_rows(kind, dense, labels, (0.8, 0.95), rows=rows, label_mask=mask, canonical=True)
        actual = score_rows(kind, disk, disk_labels, (0.8, 0.95), rows=rows, label_mask=disk_mask, canonical=True)
        assert actual == pytest.approx(expected, rel=2e-6, abs=1e-12)
    if masked:
        assert label_support(kind, disk_labels, disk_mask) == label_support(kind, labels, mask)


def test_observed_means_weight_entries_and_skip_wholly_unknown_blocks(monkeypatch):
    monkeypatch.setattr(prediction_arrays, "BUFFER_BYTES", 1)
    labels = np.array([[1., np.nan, np.nan], [np.nan, np.nan, np.nan], [1., 0., 1.]])
    mask = np.isfinite(labels)
    probs = np.array([[0.2, np.nan, np.inf], [np.nan, np.nan, np.nan], [0.9, 0.1, 0.9]])
    for function in (brier_score, negative_log_likelihood):
        expected = function("multilabel", labels, probs, mask)
        actual = function("multilabel", mapped(labels), mapped(probs), mapped(mask))
        assert actual == pytest.approx(expected, rel=1e-12)
    losses = row_losses("multilabel", mapped(labels), mapped(probs), label_mask=mapped(mask))
    assert isinstance(losses, DiskPredictionArray)
    np.testing.assert_allclose(losses, [1., np.nan, 0.], equal_nan=True)


@pytest.mark.parametrize("failure", ["empty", "label", "prediction", "dtype", "shape"])
def test_invalid_observations_are_not_hidden_in_later_blocks(monkeypatch, failure):
    monkeypatch.setattr(prediction_arrays, "BUFFER_BYTES", 1)
    labels, probs, mask = np.zeros((5, 3)), np.full((5, 3), 0.4), np.ones((5, 3), dtype=bool)
    if failure == "empty":
        mask[:] = False
    elif failure == "label":
        labels[-1, -1] = np.nan
    elif failure == "prediction":
        probs[-1, -1] = np.nan
    elif failure == "dtype":
        mask = mask.astype(int)
    else:
        mask = mask[:, :2]
    for function in (brier_score, negative_log_likelihood):
        with pytest.raises(ValueError):
            function("multilabel", mapped(labels), mapped(probs), mapped(mask))


def test_gaussian_nll_uses_observed_rows_and_original_variance_floor(monkeypatch):
    monkeypatch.setattr(prediction_arrays, "BUFFER_BYTES", 1)
    labels = np.array([0., np.nan, 2., -1.])
    mean = np.array([0., np.inf, 1., -2.])
    variance = np.array([0., np.nan, 1., 2.])
    mask = np.array([True, False, True, True])
    assert gaussian_nll(mapped(labels), mapped(mean), mapped(variance), mapped(mask)) == pytest.approx(gaussian_nll(labels, mean, variance, mask))
