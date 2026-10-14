"""The correctness model's pieces, each against a direct computation or a round trip."""

import numpy as np
import pytest
import torch


try:
    from src.protify.uncertainty import correctness
except ImportError:
    from protify.uncertainty import correctness


def test_a_reference_sample_keeps_every_class_and_repeats_itself():
    labels = np.concatenate([np.zeros(500, dtype=int), np.ones(3, dtype=int), np.full(2, 2)])  # (505,)
    chosen = correctness.reference_sample(labels, "multiclass", 20, seed=0)
    assert chosen.size == 20 and np.all(np.diff(chosen) > 0)
    assert set(labels[chosen]) == {0, 1, 2}
    np.testing.assert_array_equal(chosen, correctness.reference_sample(labels, "multiclass", 20, seed=0))
    np.testing.assert_array_equal(correctness.reference_sample(labels, "multiclass", 1000, seed=0), np.arange(505))


def test_neighbor_signals_match_their_definitions():
    match = correctness._AnswerMatch("multiclass", np.array([0, 1, 1, 2]))
    index = np.array([[1, 0, 2, 3]])  # (1, 4), nearest first
    distance = np.array([[1.0, 2.0, 3.0, 4.0]])
    signals = correctness.neighbor_signals(index, distance, match, np.array([0]), scale=2.0)
    assert signals["d1"][0] == pytest.approx(0.5)
    assert signals["dk"][0] == pytest.approx(10.0 / 4 / 2.0)
    assert signals["agree"][0] == pytest.approx(0.25)
    # Nearest labeled otherwise at 1.0, nearest labeled as answered at 2.0.
    assert signals["trust"][0] == pytest.approx(np.log(1.0 + 1e-6) - np.log(2.0 + 1e-6))


def test_multilabel_neighbors_match_whole_label_sets():
    sets = np.array([[1, 0, 1], [1, 0, 0], [1, 0, 1]], dtype=float)  # (3, 3)
    match = correctness._AnswerMatch("multilabel", sets)
    answers = np.array([[True, False, True]])
    signals = correctness.neighbor_signals(np.array([[1, 0, 2]]), np.array([[1.0, 2.0, 3.0]]), match, answers, 1.0)
    assert signals["agree"][0] == pytest.approx(2 / 3)
    assert signals["jacc"][0] == pytest.approx((0.5 + 1.0 + 1.0) / 3)


def test_the_feature_matrix_logs_distances_and_survives_infinities():
    names = ["maha_global", "trunk_agree"]
    features = correctness.feature_matrix(np.array([0.5, 1.0]), {"maha_global": np.array([np.e, 0.0]), "trunk_agree": np.array([np.inf, 0.2])}, names)
    np.testing.assert_allclose(features[0], [0.0, 1.0, 50.0])
    assert features[1, 0] == pytest.approx(np.log((1 - 1e-6) / 1e-6))
    assert features[1, 1] == pytest.approx(np.log(1e-6))


def test_the_model_round_trips_through_its_record_and_refuses_one_class():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 3))  # (200, 3)
    correct = features[:, 0] + 0.3 * rng.normal(size=200) > 0
    model = correctness.CorrectnessModel.fit(features, correct, ["a", "b"])
    again = correctness.CorrectnessModel.from_record(model.to_record())
    np.testing.assert_allclose(again.probability(features), model.probability(features))
    assert np.corrcoef(model.probability(features), features[:, 0])[0, 1] > 0.8
    with pytest.raises(ValueError, match="all right"):
        correctness.CorrectnessModel.fit(features, np.ones(200, dtype=bool), ["a", "b"])


@pytest.mark.parametrize("kind", ["multiclass", "multilabel"])
def test_a_saved_reference_gives_the_signals_it_was_fitted_with(tmp_path, kind):
    rng = np.random.default_rng(1)
    rows = 60
    trunk = torch.tensor(rng.normal(size=(rows, 6)), dtype=torch.float32)  # (60, 6)
    inputs = torch.tensor(rng.normal(size=(rows, 10)), dtype=torch.float32)  # (60, 10)
    if kind == "multiclass":
        labels = rng.integers(0, 3, size=rows)
        answers = rng.integers(0, 3, size=8)
    else:
        labels = (rng.random((rows, 3)) < 0.4).astype(float)
        answers = rng.random((8, 3)) < 0.4
    reference = correctness.LatentReference.fit(kind, 3, trunk, inputs, labels, [6, 7, 8, 9], trunk[:20], inputs[:20])
    assert set(reference.scales) == {"trunk", "inpca", "saebin", "maha"}
    queries = reference.space.rows(trunk[:8] + 0.1, inputs[:8] + 0.1)
    expected = reference.signals(queries, answers)

    reference.save(tmp_path)
    loaded = correctness.LatentReference.load(tmp_path, kind, 3, reference.scales, torch.device("cpu"))
    signals = loaded.signals(loaded.space.rows(trunk[:8] + 0.1, inputs[:8] + 0.1), answers)
    assert set(signals) >= set(correctness.feature_names(kind, True))
    for name, value in expected.items():
        np.testing.assert_allclose(signals[name], value, rtol=1e-5, atol=1e-6, err_msg=name)
