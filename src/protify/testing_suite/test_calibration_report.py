"""The calibration report a dataset with a calibration split gets, and the runs that get none.

The report fits on calibration and reads test, so these tests pin both halves on synthetic
predictions whose calibration is known: logits three times too confident must come back
with a temperature near three and a smaller test ECE, and conformal sets and intervals at
90% must cover about 90% of exchangeable test rows.
"""

import numpy as np
import pytest
import torch

from types import SimpleNamespace
from datasets import Dataset

from protify.calibration_report import calibration_report, supports_calibration_report
from protify.data.data_mixin import DataArguments, DataMixin
from protify.probes.trainers import TrainerMixin


def _overconfident_logits(rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Three-class logits whose softmax is calibrated, multiplied by 3 so that it is not."""
    rng = np.random.default_rng(seed)
    true_logits = rng.normal(size=(rows, 3)) * 1.5  # (n, 3)
    probabilities = np.exp(true_logits) / np.exp(true_logits).sum(axis=1, keepdims=True)  # (n, 3)
    labels = np.array([rng.choice(3, p=row) for row in probabilities])  # (n,)
    return true_logits * 3.0, labels  # (n, 3), (n,)


def test_an_overconfident_classifier_is_cooled_and_its_test_calibration_improves() -> None:
    calibration_logits, calibration_labels = _overconfident_logits(4000, seed=0)
    test_logits, test_labels = _overconfident_logits(4000, seed=1)

    report = calibration_report("singlelabel", calibration_logits, calibration_labels, test_logits, test_labels)

    assert report["calibration_rows"] == 4000
    assert report["calibration_temperature"] == pytest.approx(3.0, rel=0.1)
    assert report["calibration_ece_scaled"] < report["calibration_ece"]
    assert report["calibration_nll_scaled"] < report["calibration_nll"]
    assert report["calibration_set_coverage_90"] == pytest.approx(0.9, abs=0.02)
    assert 1.0 <= report["calibration_set_size_90"] <= 3.0


def test_regression_intervals_cover_about_ninety_percent_of_exchangeable_rows() -> None:
    rng = np.random.default_rng(2)
    truth = rng.normal(size=6000)  # (n,)
    predictions = truth + rng.normal(scale=0.5, size=6000)  # (n,)

    report = calibration_report("regression", predictions[:3000, None], truth[:3000], predictions[3000:], truth[3000:])

    assert report["calibration_interval_coverage_90"] == pytest.approx(0.9, abs=0.02)
    # The 90% quantile of |N(0, 0.5)| is about 0.82, so the interval spans about 1.64.
    assert report["calibration_interval_width_90"] == pytest.approx(1.645, rel=0.05)


def test_too_few_calibration_rows_report_no_infinite_sets_or_intervals() -> None:
    """At 90%, the conformal quantile of fewer than 9 rows is infinite."""
    logits, labels = _overconfident_logits(8, seed=3)
    classification = calibration_report("singlelabel", logits, labels, logits, labels)
    regression = calibration_report("regression", np.arange(8.0), np.arange(8.0) + 0.1, np.arange(4.0), np.arange(4.0))

    assert "calibration_temperature" in classification
    assert "calibration_set_coverage_90" not in classification
    assert regression == {"calibration_rows": 8.0}


@pytest.mark.parametrize(("task_type", "supported"), [
    ("singlelabel", True), ("regression", True), ("sigmoid_regression", True), ("multilabel", False), ("tokenwise", False),
])
def test_only_single_label_classification_and_regression_get_the_report(task_type, supported) -> None:
    assert supports_calibration_report(task_type) is supported
    if not supported:
        with pytest.raises(ValueError, match="calibration report covers"):
            calibration_report(task_type, np.zeros((2, 2)), np.zeros(2), np.zeros((2, 2)), np.zeros(2))


def _write_split(directory, name: str, rows: list[tuple[str, int]]) -> None:
    lines = ["seqs,labels", *(f"{sequence},{label}" for sequence, label in rows)]
    (directory / f"{name}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _local_dataset(tmp_path, *, calibration: bool):
    directory = tmp_path / "toy"
    directory.mkdir()
    _write_split(directory, "train", [("MKTAYIAKQR", 0), ("GGGGSGGGGS", 1), ("ACDEFGHIKL", 0)])
    _write_split(directory, "valid", [("WWWWWWWWWW", 1), ("MKTAYIAKQRQ", 0)])
    _write_split(directory, "test", [("PPPPPPPPPP", 1), ("MKTAYIA", 0)])
    if calibration:
        _write_split(directory, "calibration", [("mkvlaagg", 1), ("", 0), ("QQQQQQQQQQ", 0)])
    return DataMixin(DataArguments(data_names=[], data_dirs=[str(directory)])).get_data, directory


def test_a_calibration_split_is_cleaned_like_the_others_and_kept_beside_them(tmp_path) -> None:
    get_data, _ = _local_dataset(tmp_path, calibration=True)
    mixin = get_data.__self__

    datasets, all_seqs = get_data()

    train, valid, test, num_labels, label_type, ppi = datasets["toy"]
    calibration = mixin.calibration_sets["toy"]
    # The empty sequence is dropped and the rest sanitized, exactly as in the other splits.
    assert list(calibration["seqs"]) == ["MKVLAAGG", "QQQQQQQQQQ"]
    assert list(calibration["labels"]) == [1, 0]
    assert {"MKVLAAGG", "QQQQQQQQQQ"} <= set(all_seqs)
    assert (len(train), len(valid), len(test), num_labels, label_type, ppi) == (3, 2, 2, 2, "singlelabel", False)


def test_a_dataset_without_a_calibration_split_loads_as_it_always_did(tmp_path) -> None:
    get_data, _ = _local_dataset(tmp_path, calibration=False)
    mixin = get_data.__self__

    datasets, _ = get_data()

    assert mixin.calibration_sets == {}
    assert len(datasets["toy"]) == 6


class _ToyProbe(torch.nn.Module):
    """Logits that are the input's first two columns, three times too confident."""

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> SimpleNamespace:
        # embeddings: (b, 2)
        return SimpleNamespace(logits=embeddings * 3.0)  # (b, 2)

    def parameters(self, recurse: bool = True):
        yield torch.zeros(1)


def _collate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {"embeddings": torch.stack([row["embeddings"] for row in rows]), "labels": torch.stack([row["labels"] for row in rows])}


def _rows(logits: np.ndarray, labels: np.ndarray) -> list[dict[str, torch.Tensor]]:
    return [{"embeddings": torch.tensor(row, dtype=torch.float32), "labels": torch.tensor(label)} for row, label in zip(logits, labels, strict=True)]


def test_the_trainer_adds_the_report_to_test_metrics_from_the_model_it_keeps() -> None:
    rng = np.random.default_rng(4)
    true_logits = rng.normal(size=(600, 2))  # (n, 2)
    labels = (rng.uniform(size=600) < 1.0 / (1.0 + np.exp(true_logits[:, 0] - true_logits[:, 1]))).astype(np.int64)  # (n,)
    trainer = SimpleNamespace(
        trainer_args=SimpleNamespace(probe_batch_size=64), probe_args=SimpleNamespace(task_type="singlelabel"),
    )
    trainer._predict_split = TrainerMixin._predict_split.__get__(trainer)
    test_metrics = {"test_mcc": 0.1}
    outcome = (_ToyProbe(), {}, test_metrics, true_logits[300:] * 3.0, labels[300:])

    TrainerMixin._add_calibration_report(trainer, outcome, _rows(true_logits[:300], labels[:300]), _collate)

    assert test_metrics["test_mcc"] == 0.1
    assert test_metrics["test_calibration_rows"] == 300
    assert test_metrics["test_calibration_temperature"] > 1.5
    assert test_metrics["test_calibration_ece_scaled"] < test_metrics["test_calibration_ece"]


def test_masked_test_labels_get_no_report() -> None:
    test_metrics = {}
    outcome = (_ToyProbe(), {}, test_metrics, np.zeros((2, 2)), (np.zeros(2), np.ones(2, dtype=bool)))
    TrainerMixin._add_calibration_report(SimpleNamespace(), outcome, [], _collate)
    assert test_metrics == {}


def test_the_calibration_dataset_is_built_as_test_is(monkeypatch) -> None:
    """Built from the same kwargs, so the standardizer and selection test reads apply to it too."""
    built = []

    class _Recorder:
        def __init__(self, **kwargs) -> None:
            built.append(kwargs)

    trainer = SimpleNamespace(
        trainer_args=SimpleNamespace(probe_batch_size=4, read_scaler=100),
        probe_args=SimpleNamespace(input_size=2, task_type="singlelabel", tokenwise=False),
        embedding_args=SimpleNamespace(sql=False), full_args=SimpleNamespace(), _full=False,
        _pair_feature_mode="concat", _augment_pair_order=lambda: False, _fit_embedding_standardizer=lambda *arguments: None,
    )
    monkeypatch.setattr("protify.probes.trainers.EmbedsLabelsDataset", _Recorder)
    monkeypatch.setattr("protify.probes.trainers.EmbedsLabelsCollator", lambda **kwargs: "collator")
    splits = {name: Dataset.from_dict({"seqs": [name.upper()], "labels": [0]}) for name in ("train", "valid", "test", "calibration")}

    datasets = TrainerMixin._build_probe_datasets(
        trainer, splits["train"], splits["valid"], splits["test"], None, {}, None, False, calibration_dataset=splits["calibration"],
    )

    assert datasets[3] == "collator" and isinstance(datasets[4], _Recorder)
    test_kwargs, calibration_kwargs = built[2], built[3]
    assert list(calibration_kwargs["hf_dataset"]["seqs"]) == ["CALIBRATION"]
    assert {key: value for key, value in calibration_kwargs.items() if key != "hf_dataset"} == {
        key: value for key, value in test_kwargs.items() if key != "hf_dataset"
    }
    assert TrainerMixin._build_probe_datasets(trainer, splits["train"], splits["valid"], splits["test"], None, {}, None, False)[4] is None
