"""SNGP: a spectral bound on the trunk and a Gaussian process output with a Laplace posterior.

What has to hold for its variance to mean anything: every bounded Linear stays within its
bound, the random features approximate the RBF kernel of the stated squared lengthscale, the
posterior variance grows away from the training data, predictions switch from MAP to
mean-field logits exactly when the posterior is fitted, the posterior travels with the
weights, a regression head that fits its training rows exactly keeps a positive noise, and
MC dropout passes, which switch only dropout on, leave the bound's power iteration where
training left it.
"""

import math
import pytest
import torch


try:
    from src.protify.probes.projected_probe import NOISE_VARIANCE_FLOOR_FRACTION, ProjectedProbe, ProjectedProbeConfig
    from src.protify.probes.sngp import MEAN_FIELD_FACTOR, RandomFeatureGaussianProcess, bound_spectral_norm
    from src.protify.uncertainty.predictions import mc_dropout
except ImportError:
    from protify.probes.projected_probe import NOISE_VARIANCE_FLOOR_FRACTION, ProjectedProbe, ProjectedProbeConfig
    from protify.probes.sngp import MEAN_FIELD_FACTOR, RandomFeatureGaussianProcess, bound_spectral_norm
    from protify.uncertainty.predictions import mc_dropout


def _sngp(input_size: int = 8, num_labels: int = 2, task_type: str = "singlelabel", dropout: float = 0.0) -> ProjectedProbe:
    config = ProjectedProbeConfig(
        input_size=input_size,
        projection_size=16,
        dropout=dropout,
        output="gaussian_process",
        gp_num_features=256,
        # The production ratio: squared lengthscale equal to the trunk width.
        gp_lengthscale_squared=16.0,
        num_labels=num_labels,
        task_type=task_type,
    )
    return ProjectedProbe(config)


def _clusters(count: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Two classes, one at +2 and one at -2 on the first four of eight features."""
    labels = torch.arange(count) % 2  # (n,)
    rows = 0.3 * torch.randn(count, 8)  # (n, 8)
    rows[:, :4] += torch.where(labels.unsqueeze(-1) == 1, 2.0, -2.0)
    return rows, labels  # (n, 8), (n,)


def _train(probe: ProjectedProbe, rows: torch.Tensor, labels: torch.Tensor, steps: int = 150) -> None:
    # rows: (n, 8), labels: (n,)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
    probe.train()
    for _ in range(steps):
        optimizer.zero_grad()
        probe(rows, labels=labels).loss.backward()
        optimizer.step()
    probe.eval()


def test_a_bounded_linear_stays_within_its_bound():
    """Within the bound up to the running power iteration's error, which training steps shrink.

    One iteration per training forward underestimates sigma until it converges, which lets the
    norm sit about 0.1% over the bound after five steps; after fifty it is within 1e-4.
    """
    torch.manual_seed(0)
    linear = torch.nn.Linear(64, 32)
    with torch.no_grad():
        linear.weight.mul_(10.0)
    bound_spectral_norm(linear, 0.95)
    linear.train()
    for _ in range(50):
        linear(torch.randn(4, 64))

    assert torch.linalg.matrix_norm(linear.weight.detach(), ord=2) <= 0.95 + 1e-4


def test_a_weight_inside_the_bound_passes_through_unchanged():
    torch.manual_seed(0)
    linear = torch.nn.Linear(16, 16)
    with torch.no_grad():
        linear.weight.mul_(0.01)
    original = linear.weight.detach().clone()
    bound_spectral_norm(linear, 0.95)
    assert torch.allclose(linear.weight, original)


def test_the_random_features_approximate_an_rbf_kernel_of_the_squared_lengthscale():
    """phi(h)^T phi(h') approximates exp(-|h - h'|^2 / (2 l^2)), within random-feature error."""
    torch.manual_seed(0)
    lengthscale_squared = 16.0
    gaussian_process = RandomFeatureGaussianProcess(16, 1, num_features=8192, lengthscale_squared=lengthscale_squared)
    base, direction = torch.randn(16), torch.nn.functional.normalize(torch.randn(16), dim=0)  # (16,), (16,)
    h = torch.stack([base + step * direction for step in (0.0, 1.0, 2.0, 4.0, 8.0)])  # (5, 16)

    features = gaussian_process.features(h)  # (5, D)
    exact = torch.exp(-torch.cdist(h, h) ** 2 / (2.0 * lengthscale_squared))  # (5, 5), from 1 down to e^-2
    assert torch.allclose(features @ features.T, exact, atol=0.05)


def test_variance_grows_away_from_the_training_clusters():
    """Far rows alternate sign, a pattern the branch LayerNorm cannot map onto a cluster.

    LayerNorm removes each row's mean and scale, so a row shifted only along the features the
    clusters leave at zero would normalize onto the opposite cluster and look familiar.
    """
    torch.manual_seed(0)
    probe = _sngp()
    rows, labels = _clusters(512)
    _train(probe, rows, labels, steps=50)
    probe.fit_posterior([{"embeddings": rows, "labels": labels}])

    near, _ = _clusters(64)
    far = 0.3 * torch.randn(64, 8) + 2.0 * torch.tensor([1.0, -1.0] * 4)  # (64, 8)
    _, near_variance = probe.predict_with_variance(near)
    _, far_variance = probe.predict_with_variance(far)
    assert far_variance.mean() > 1.5 * near_variance.mean()


def test_logits_are_map_before_the_posterior_and_mean_field_after():
    torch.manual_seed(0)
    probe = _sngp().eval()
    rows, labels = _clusters(32)
    gaussian_process = probe.gaussian_process

    with torch.no_grad():
        features = gaussian_process.features(probe.trunk_features(rows))  # (n, D)
        map_logits = gaussian_process.beta(features)  # (n, c)
    assert torch.allclose(probe(rows).logits, map_logits)

    probe.fit_posterior([{"embeddings": rows, "labels": labels}])
    with torch.no_grad():
        variance = gaussian_process.variance(features)  # (n,)
    expected = map_logits / torch.sqrt(1.0 + MEAN_FIELD_FACTOR * variance).unsqueeze(-1)  # (n, c)
    assert torch.allclose(probe(rows).logits, expected, atol=1e-6)
    assert not torch.allclose(expected, map_logits)
    assert math.pi / 8 == pytest.approx(MEAN_FIELD_FACTOR)


def test_the_posterior_survives_a_state_dict_reload():
    torch.manual_seed(0)
    probe = _sngp()
    rows, labels = _clusters(32)
    probe.fit_posterior([{"embeddings": rows, "labels": labels}])
    # Evaluation mode, as predict_with_variance requires: a training-mode forward would
    # advance the spectral bound's power iteration between the two predictions.
    probe.eval()

    torch.manual_seed(1)
    reloaded = _sngp()
    assert not bool(reloaded.gaussian_process.posterior_ready)
    reloaded.load_state_dict(probe.state_dict())
    reloaded.eval()

    assert bool(reloaded.gaussian_process.posterior_ready)
    for expected, actual in zip(probe.predict_with_variance(rows), reloaded.predict_with_variance(rows), strict=True):
        assert torch.allclose(expected, actual, atol=1e-6)


def test_regression_posterior_reads_training_error_and_label_variance():
    torch.manual_seed(0)
    probe = _sngp(num_labels=1, task_type="regression")
    rows = torch.randn(64, 8)
    targets = rows[:, 0] * 3.0 + 1.0  # (n,)
    _train(probe, rows, targets, steps=50)
    probe.fit_posterior([{"embeddings": rows, "labels": targets}])

    with torch.no_grad():
        predictions = probe(rows).logits.reshape(-1)  # (n,)
    training_error = torch.mean((predictions - targets) ** 2)
    assert float(probe.gaussian_process.noise_variance) == pytest.approx(float(training_error), rel=1e-4)
    _, variance = probe.predict_with_variance(rows)
    assert bool(torch.all(variance >= probe.gaussian_process.noise_variance))


def test_an_exact_regression_fit_keeps_a_positive_noise_and_finite_variances():
    """Targets equal to the head's own MAP predictions leave a training error of exactly zero.

    Unfloored, the noise variance would be zero and the posterior precision would divide by it.
    """
    torch.manual_seed(0)
    probe = _sngp(num_labels=1, task_type="regression").eval()
    rows = torch.randn(64, 8)  # (n, 8)
    with torch.no_grad():
        targets = probe(rows).logits.reshape(-1)  # (n,), MAP until the posterior is fitted
    probe.fit_posterior([{"embeddings": rows, "labels": targets}])

    noise = float(probe.gaussian_process.noise_variance)
    label_variance = float(targets.double().var(unbiased=False))
    assert math.isfinite(noise) and noise > 0.0
    assert noise == pytest.approx(NOISE_VARIANCE_FLOOR_FRACTION * label_variance, rel=1e-4)
    for queries in (rows, 3.0 * torch.randn(64, 8)):
        _, variance = probe.predict_with_variance(queries)  # (n,)
        assert bool(torch.isfinite(variance).all()) and bool((variance >= 0.999 * noise).all())


def test_mc_dropout_passes_leave_the_power_iteration_alone():
    torch.manual_seed(0)
    probe = _sngp(dropout=0.5).eval()
    parametrization = probe.blocks[0].linear.parametrizations.weight[0]
    before = parametrization.u.clone()

    rows = torch.randn(8, 8)
    with mc_dropout(probe):
        assert probe.blocks[0].dropout.training and not parametrization.training
        first = probe(rows).logits
        second = probe(rows).logits
    assert torch.equal(parametrization.u, before)
    assert not torch.allclose(first, second)
    assert not probe.blocks[0].dropout.training

    # A training pass after the weight moves does advance it.
    with torch.no_grad():
        probe.blocks[0].linear.parametrizations.weight.original.add_(torch.randn(16, 16))
    probe.train()
    probe(rows)
    assert not torch.equal(parametrization.u, before)
