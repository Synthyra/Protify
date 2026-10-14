"""Nearest rows and trunk Gaussians, each checked against a direct computation."""

import numpy as np
import pytest
import torch


try:
    from src.protify.uncertainty import latent_geometry
except ImportError:
    from protify.uncertainty import latent_geometry


def test_nearest_rows_match_a_direct_search_across_blocks(monkeypatch):
    monkeypatch.setattr(latent_geometry, "QUERY_BLOCK", 7)
    monkeypatch.setattr(latent_geometry, "REFERENCE_BLOCK", 11)
    rng = np.random.default_rng(0)
    queries, reference = rng.normal(size=(20, 5)), rng.normal(size=(40, 5))  # (20, 5), (40, 5)
    indices, distances = latent_geometry.nearest_rows(torch.tensor(queries), torch.tensor(reference), 6)

    direct = np.linalg.norm(queries[:, None, :] - reference[None, :, :], axis=2)  # (20, 40)
    expected = np.argsort(direct, axis=1)[:, :6]  # (20, 6)
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_allclose(distances, np.take_along_axis(direct, expected, axis=1), rtol=1e-5)


def test_nearest_rows_measure_near_duplicates_exactly(monkeypatch):
    # cdist takes its matrix-product path only above 25 rows, where float32 |q|^2 + |r|^2 - 2 q.r
    # on rows of norm about 400 is off by about 0.3, far more than these neighbors' distances.
    monkeypatch.setattr(latent_geometry, "QUERY_BLOCK", 32)
    monkeypatch.setattr(latent_geometry, "REFERENCE_BLOCK", 40)
    rng = np.random.default_rng(2)
    bases = 50.0 + rng.normal(size=(10, 64))  # (10, 64)
    scales = 1e-3 * np.arange(1, 7)  # (6,)
    reference = (bases[:, None, :] + scales[None, :, None] * rng.normal(size=(10, 6, 64))).reshape(60, 64)  # (60, 64)
    queries = np.repeat(bases, 4, axis=0) + 1e-4 * rng.normal(size=(40, 64))  # (40, 64)
    queries, reference = queries.astype(np.float32), reference.astype(np.float32)
    indices, distances = latent_geometry.nearest_rows(torch.tensor(queries), torch.tensor(reference), 6)

    direct = np.linalg.norm(queries[:, None, :].astype(np.float64) - reference[None, :, :], axis=2)  # (40, 60)
    expected = np.argsort(direct, axis=1)[:, :6]  # (40, 6)
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_allclose(distances, np.take_along_axis(direct, expected, axis=1), rtol=1e-5, atol=1e-6)


def test_nearest_rows_stop_at_the_reference_size():
    indices, distances = latent_geometry.nearest_rows(torch.zeros(3, 2), torch.ones(4, 2), 32)
    assert indices.shape == distances.shape == (3, 4)


def test_nearest_tanimoto_matches_a_direct_search_across_blocks(monkeypatch):
    monkeypatch.setattr(latent_geometry, "QUERY_BLOCK", 5)
    monkeypatch.setattr(latent_geometry, "REFERENCE_BLOCK", 9)
    rng = np.random.default_rng(3)
    queries = rng.random((17, 40)) < 0.2  # (17, 40)
    reference = rng.random((31, 40)) < 0.2  # (31, 40)
    queries[0] = False  # an empty row is as far as possible from every row
    indices, distances = latent_geometry.nearest_tanimoto(torch.tensor(queries), torch.tensor(reference), 4)

    shared = queries.astype(float) @ reference.T.astype(float)  # (17, 31)
    either = queries.sum(1)[:, None] + reference.sum(1)[None, :] - shared  # (17, 31)
    direct = 1.0 - shared / np.maximum(either, 1e-6)  # (17, 31)
    expected = np.argsort(direct, axis=1, kind="stable")[:, :4]  # (17, 4)
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_allclose(distances, np.take_along_axis(direct, expected, axis=1), rtol=1e-6)
    np.testing.assert_allclose(distances[0], 1.0)


def test_gaussian_scores_match_the_closed_form():
    rng = np.random.default_rng(1)
    labels = np.repeat([0, 1], 200)  # (400,)
    reference = rng.normal(size=(400, 3)) + 4.0 * labels[:, None]  # (400, 3)
    queries = np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0], [20.0, -20.0, 0.0]])  # (3, 3)
    scores = latent_geometry.gaussian_scores(torch.tensor(reference), labels, torch.tensor(queries), 2)

    covariance = np.cov(reference.T)
    ridge = latent_geometry.RIDGE * np.diag(covariance).mean() + 1e-8
    centered = queries - reference.mean(axis=0)
    expected = np.einsum("np,pq,nq->n", centered, np.linalg.inv(covariance + ridge * np.eye(3)), centered)
    np.testing.assert_allclose(scores["maha_global"], expected, rtol=1e-4)

    assert scores["maha_class"].shape == scores["log_density_class"].shape == (3, 2)
    assert scores["maha_class"][0].argmin() == 0 and scores["maha_class"][1].argmin() == 1
    # Far from both classes, both densities fall far below the density at a class mean.
    assert scores["log_density_class"][2].max() < scores["log_density_class"][0, 0] - 50


def test_gaussian_scores_leave_an_absent_class_undefined():
    reference = torch.randn(50, 2, dtype=torch.float64)
    scores = latent_geometry.gaussian_scores(reference, np.zeros(50, dtype=int), reference[:4], 3)
    assert np.isnan(scores["maha_class"][:, 1:]).all() and np.isfinite(scores["maha_class"][:, 0]).all()


def test_regression_scores_carry_only_the_global_distance():
    reference = torch.randn(50, 2, dtype=torch.float64)
    assert set(latent_geometry.gaussian_scores(reference, None, reference[:4], None)) == {"maha_global"}


@pytest.mark.parametrize("class_sizes", [(3, 5, 40, 2), (60, 70)])
def test_trunk_gaussians_score_as_gaussian_scores_does(class_sizes):
    # Classes smaller than the width go through the Woodbury identity, larger ones through their
    # whole covariance; a missing class stays undefined in both.
    rng = np.random.default_rng(4)
    width = 12
    labels = np.concatenate([np.full(size, label) for label, size in enumerate(class_sizes)])  # (r,)
    reference = rng.normal(size=(labels.size, width)) * rng.uniform(0.5, 3.0, size=width) + labels[:, None]  # (r, 12)
    queries = rng.normal(size=(9, width)) * 2.0  # (9, 12)
    classes = len(class_sizes) + 1
    direct = latent_geometry.gaussian_scores(torch.tensor(reference), labels, torch.tensor(queries), classes)
    fitted = latent_geometry.TrunkGaussians(torch.tensor(reference), labels, classes).scores(torch.tensor(queries))

    assert set(fitted) == set(direct)
    for name in direct:
        np.testing.assert_allclose(fitted[name], direct[name], rtol=1e-5, atol=1e-4)
    assert np.isnan(fitted["log_density_class"][:, -1]).all()


def test_trunk_gaussians_of_a_multilabel_head_score_only_the_global_distance():
    reference = torch.randn(30, 4, dtype=torch.float64)
    scores = latent_geometry.TrunkGaussians(reference, None, None).scores(reference[:3])
    np.testing.assert_allclose(scores["maha_global"], latent_geometry.gaussian_scores(reference, None, reference[:3], None)["maha_global"], rtol=1e-6)
    assert set(scores) == {"maha_global"}
