"""Mapped probability methods preserve their whole-array arithmetic and prediction contracts."""

import numpy as np
import pytest

from scipy.special import expit, softmax

from protify.uncertainty import methods, prediction_arrays
from protify.uncertainty.packaged import calibrated_predictive
from protify.uncertainty.prediction_arrays import DiskPredictionArray, PredictionAccumulator, prediction_mask_is_complete, transform_prediction_rows
from protify.uncertainty.predictions import SplitPredictions


def mapped(values):
    with PredictionAccumulator(len(values), disk_backed=True) as accumulator:
        accumulator.append({"values": values})
        return accumulator.finish()["values"]


def assert_same(actual, expected):
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-14)


def test_transform_bounds_combined_inputs_preserves_outputs_and_closes_files(monkeypatch):
    first = mapped(np.arange(85, dtype=np.float32).reshape(17, 5))
    second = np.arange(17, dtype=np.int64)
    monkeypatch.setattr(prediction_arrays, "BUFFER_BYTES", 56)
    temporary_file = prediction_arrays.tempfile.TemporaryFile
    files = []

    def tracked_file(*args, **kwargs):
        stream = temporary_file(*args, **kwargs)
        files.append(stream)
        return stream

    monkeypatch.setattr(prediction_arrays.tempfile, "TemporaryFile", tracked_file)
    sizes = []
    def transform(left, right):
        assert type(left) is np.ndarray and type(right) is np.ndarray
        sizes.append(len(left))
        assert left.nbytes + right.nbytes <= 56
        return {"matrix": left + right[:, None], "row": left.sum(axis=1)}
    output = transform_prediction_rows(transform, first, second)
    assert sizes == [2] * 8 + [1]
    assert files and all(stream.closed for stream in files)
    for values in output.values():
        assert isinstance(values, DiskPredictionArray) and not values.flags.writeable
    np.testing.assert_array_equal(output["matrix"], first + second[:, None])
    np.testing.assert_array_equal(output["row"], first.sum(axis=1))


@pytest.mark.parametrize("failure", ["exception", "row_count", "dtype", "fields", "unaligned", "scalar"])
def test_transform_failures_close_every_temporary_file(monkeypatch, failure):
    values = mapped(np.arange(15, dtype=np.float32).reshape(5, 3))
    monkeypatch.setattr(prediction_arrays, "BUFFER_BYTES", 24)
    temporary_file = prediction_arrays.tempfile.TemporaryFile
    files = []
    def tracked_file(*args, **kwargs):
        stream = temporary_file(*args, **kwargs)
        files.append(stream)
        return stream
    monkeypatch.setattr(prediction_arrays.tempfile, "TemporaryFile", tracked_file)
    calls = 0
    def transform(block, *others):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"values": block}
        if failure == "exception":
            raise RuntimeError("interrupted transform")
        if failure == "row_count":
            return {"values": block[:1]}
        if failure == "dtype":
            return {"values": block.astype(np.float64)}
        if failure == "fields":
            return {"changed": block}
        return {"values": block}
    inputs = (values,)
    if failure == "unaligned":
        inputs += (np.zeros(4),)
    elif failure == "scalar":
        inputs += (np.array(3),)
    with pytest.raises((ValueError, RuntimeError)):
        transform_prediction_rows(transform, *inputs)
    assert all(stream.closed for stream in files)


def test_ordinary_arrays_keep_direct_arithmetic_and_writable_outputs(monkeypatch):
    monkeypatch.setattr(prediction_arrays.tempfile, "TemporaryFile", lambda **kwargs: pytest.fail("RAM caller created a file"))
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    calls = []
    def transform(block):
        calls.append(block)
        return {"values": block + 1}
    output = transform_prediction_rows(transform, values)["values"]
    assert len(calls) == 1 and calls[0] is values and output.flags.writeable
    assert type(output) is np.ndarray
    np.testing.assert_array_equal(output, values + 1)


@pytest.mark.parametrize("stored", [False, True])
@pytest.mark.parametrize("kind", ["multilabel", "multiclass"])
def test_large_temperature_does_not_create_float32_answer_ties(stored, kind):
    logits = np.array([[-0.01, 0.02], [0.03, -0.02], [-0.01, -0.02]], dtype=np.float32)
    values = mapped(logits) if stored else logits
    probs = methods.probabilities(values, kind, temperature=1e6)
    assert probs.dtype == np.float64
    if kind == "multilabel":
        np.testing.assert_array_equal(probs >= 0.5, logits >= 0)
    else:
        np.testing.assert_array_equal(probs.argmax(axis=1), logits.argmax(axis=1))


@pytest.mark.parametrize("kind", ["multilabel", "multiclass"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("temperature", [None, 0.5, 2.25])
def test_probability_transforms_and_uncertainties_match_dense_formulas(monkeypatch, kind, dtype, temperature):
    monkeypatch.setattr(prediction_arrays, "BUFFER_BYTES", 70)
    logits = np.random.default_rng(72).normal(size=(17, 7)).astype(dtype)
    logits[0, :2] = [-1000, 1000]
    stored = mapped(logits)
    scaled = logits if temperature is None else logits.astype(np.float64) / temperature
    expected = expit(scaled) if kind == "multilabel" else softmax(scaled, axis=1)
    probs = methods.probabilities(stored, kind, temperature=temperature)
    assert isinstance(probs, DiskPredictionArray) and not probs.flags.writeable
    assert_same(probs, expected)
    clipped = np.clip(expected.astype(np.float64), methods.PROBABILITY_FLOOR, 1 - methods.PROBABILITY_FLOOR)
    expected_logits = np.log(clipped) - np.log1p(-clipped) if kind == "multilabel" else np.log(clipped)
    assert_same(methods.probability_logits(probs, kind), expected_logits)
    expected_uncertainty = -np.mean(clipped * np.log(clipped) + (1-clipped) * np.log1p(-clipped), axis=1)
    assert_same(methods.mean_binary_entropy(probs), expected_uncertainty)
    assert_same(methods.top_label_uncertainty(probs), 1-expected.max(axis=1))
    entropy_clipped = np.clip(expected, methods.PROBABILITY_FLOOR, 1)
    assert_same(methods.predictive_entropy(probs), -np.sum(entropy_clipped*np.log(entropy_clipped), axis=1))
    probability_temperature = 1.7
    expected_calibrated = expit(expected_logits/probability_temperature) if kind == "multilabel" else softmax(expected_logits/probability_temperature, axis=1)
    assert_same(methods.calibrated_probabilities(probs, kind, probability_temperature), expected_calibrated)
    np.testing.assert_array_equal(stored, logits)


@pytest.mark.parametrize("kind", ["multilabel", "multiclass", "regression"])
@pytest.mark.parametrize("member_count", [2, 3, 7])
def test_ensembles_preserve_order_dtype_and_mixed_variances(monkeypatch, kind, member_count):
    monkeypatch.setattr(prediction_arrays, "BUFFER_BYTES", 100)
    generator = np.random.default_rng(81)
    members = []
    for index in range(member_count):
        dtype = np.float32 if index % 2 else np.float64
        if kind == "regression":
            mean = generator.normal(size=17).astype(dtype)
            variance = None if index % 2 else generator.random(17).astype(dtype)
            members.append(methods.Predictive(mean=mean, variance=variance))
        else:
            probs = methods.probabilities(generator.normal(size=(17, 7)).astype(dtype), kind)
            members.append(methods.classification_predictive(probs, kind))
    dense_runs = tuple(methods.MethodRun(index, {"test": member}) for index, member in enumerate(members))
    disk_runs = tuple(methods.MethodRun(index, {"test": methods.Predictive(**{
        name: None if values is None else mapped(values) for name, values in vars(member).items()
    })}) for index, member in enumerate(members))
    expected = methods.ensemble_run(dense_runs, kind).splits["test"]
    actual = methods.ensemble_run(disk_runs, kind).splits["test"]
    for name, values in vars(expected).items():
        if values is not None:
            assert_same(getattr(actual, name), values)
            assert isinstance(getattr(actual, name), DiskPredictionArray)


@pytest.mark.parametrize("kind", ["binary", "multiclass", "multilabel", "regression"])
def test_method_build_and_packaged_calibration_agree_for_mapped_inputs(monkeypatch, kind):
    monkeypatch.setattr(prediction_arrays, "BUFFER_BYTES", 96)
    generator = np.random.default_rng(14)
    dense = {}
    for seed in (42, 43):
        dense[seed] = {}
        for name in ("v_fit", "v_cal", "test"):
            outputs = generator.normal(size=19 if kind == "regression" else (19, 3)).astype(np.float32)
            labels = (generator.normal(size=19) if kind == "regression" else
                      generator.integers(0, 2, size=outputs.shape if kind == "multilabel" else 19))
            if seed == 43:
                labels = dense[42][name].labels.copy()
            extras = {"variance": generator.random(19).astype(np.float32)} if kind == "regression" else {}
            dense[seed][name] = SplitPredictions(labels, outputs, **extras)
    disk = {seed: {name: SplitPredictions(**{key: None if values is None else mapped(values)
                                           for key, values in vars(split).items()})
                   for name, split in splits.items()} for seed, splits in dense.items()}
    expected = methods.build_methods(dense, "projected", kind)
    actual = methods.build_methods(disk, "projected", kind)
    assert [method.name for method in actual] == [method.name for method in expected]
    for left, right in zip(actual, expected, strict=True):
        for observed, target in zip(left.runs, right.runs, strict=True):
            assert observed.fitted == pytest.approx(target.fitted, rel=2e-7)
            for name in target.splits:
                for field, values in vars(target.splits[name]).items():
                    if values is not None:
                        assert_same(getattr(observed.splits[name], field), values)
    for predictive in ("head", "ensemble"):
        seeds = (42,) if predictive == "head" else (42, 43)
        calibration = "variance_map" if kind == "regression" else "logit_temperature" if predictive == "head" else "probability_temperature"
        fitted = {"a": 1.2, "b": 0.1} if kind == "regression" else {"temperature": 1.7}
        left = calibrated_predictive({seed: disk[seed]["test"] for seed in seeds}, predictive, calibration, fitted, kind)
        right = calibrated_predictive({seed: dense[seed]["test"] for seed in seeds}, predictive, calibration, fitted, kind)
        for field, values in vars(right).items():
            if values is not None:
                assert_same(getattr(left, field), values)


@pytest.mark.parametrize("unknown_row", [None, 0, 16])
def test_completeness_check_does_not_read_the_full_mapped_mask(monkeypatch, unknown_row):
    mask = np.ones((17, 7), dtype=bool)
    if unknown_row is not None:
        mask[unknown_row, -1] = False
    stored = mapped(mask)
    monkeypatch.setattr(prediction_arrays, "BUFFER_BYTES", 14)
    releases = []
    monkeypatch.setattr(prediction_arrays, "release_prediction_pages", lambda values: releases.append(values))
    assert prediction_mask_is_complete(stored) == (unknown_row is None)
    assert len(releases) == (1 if unknown_row == 0 else 9)
