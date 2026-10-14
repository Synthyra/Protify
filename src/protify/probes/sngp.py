"""Spectral-normalized Gaussian process output (SNGP) for the projected probe.

SNGP (Liu et al. 2020) makes a network distance aware in two steps. A spectral bound on every
hidden Linear keeps the trunk from collapsing distant inputs together, and a Gaussian process
output layer, approximated with random Fourier features, turns distance from the training data
into predictive variance. TUnA (Ko et al. 2024) is this head on frozen ESM-2 embeddings.

The Laplace posterior over the output weights is fitted in one pass over the training data
after training, with the weights early stopping restored. Updating it every epoch, as the
original recipe does, leaves "the last epoch's" posterior ill-defined once early stopping
rolls the weights back, and this is that update in exact form.

The kernel is `exp(-|h - h'|^2 / (2 l^2))`, with `l^2` the squared lengthscale. `ProjectedProbe`
feeds it the trunk output after a LayerNorm, which at initialization gives each row zero mean
and unit variance over its `d` features, so `|h|^2 = d` and rows at correlation `rho` sit
`2 d (1 - rho)` apart: their kernel is `exp(-d (1 - rho) / l^2)`. At the default `l^2 = 1024`
with `d = 1024`, uncorrelated rows meet at `e^-1 = 0.37`, anticorrelated ones at
`e^-2 = 0.14`, and each random feature's phase `h W^T` has unit variance. The LayerNorm's gain
is trained and unbounded, so training rescales `|h|^2`. In a CPU check on synthetic clusters
at `l^2 = 1024`, the mean kernel between two random training rows was 0.39 to 0.40 in a fresh
head and 0.24 to 0.38 in a trained one, and across lengthscales `|h|^2` ended between 0.8 d
and 1.7 d, larger under longer ones. A short lengthscale can stop the head from training at all: when
`d (1 - rho) / l^2` is large, rows that share a label share no features at initialization. In
that check `l^2 = 16` failed on most runs and 64 on the hardest data; 128 and above always
trained.

Shapes: `n` rows, `d` trunk width, `D` random features, `c` outputs.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from torch import Tensor, nn
from torch.nn.utils import parametrize


# Mean-field approximation of the softmax under a Gaussian logit, logits / sqrt(1 + lambda * var).
MEAN_FIELD_FACTOR = math.pi / 8
# Power iterations run once when the bound is attached, so the first training step starts from
# a converged singular vector rather than a random one.
INITIAL_POWER_ITERATIONS = 15


class SpectralNormBound(nn.Module):
    """Parametrization `W -> W * min(1, bound / sigma_hat)`.

    `sigma_hat = u^T W v` estimates the largest singular value with one power iteration per
    forward, and the vectors advance only while the parametrized module is in training mode.
    Evaluation, including MC-dropout passes that switch only Dropout modules to training
    mode, therefore reads a fixed estimate. Unlike `torch.nn.utils.parametrizations
    .spectral_norm`, which divides by sigma and so pins every norm to one, a weight already
    inside the bound passes through unchanged.
    """

    def __init__(self, weight: Tensor, bound: float, eps: float = 1e-12) -> None:
        super().__init__()
        # weight: (d_out, d_in)
        if bound <= 0:
            raise ValueError(f"spectral bound must be positive; got {bound}.")

        self.bound = bound
        self.eps = eps
        d_out, d_in = weight.shape
        u = F.normalize(torch.randn(d_out, dtype=weight.dtype, device=weight.device), dim=0, eps=eps)  # (d_out,)
        v = F.normalize(torch.randn(d_in, dtype=weight.dtype, device=weight.device), dim=0, eps=eps)  # (d_in,)
        self.register_buffer("u", u)
        self.register_buffer("v", v)
        self._power_iteration(weight.detach(), INITIAL_POWER_ITERATIONS)

    @torch.no_grad()
    def _power_iteration(self, weight: Tensor, iterations: int) -> None:
        # weight: (d_out, d_in)
        for _ in range(iterations):
            self.v.copy_(F.normalize(weight.T @ self.u, dim=0, eps=self.eps))  # (d_in,)
            self.u.copy_(F.normalize(weight @ self.v, dim=0, eps=self.eps))  # (d_out,)

    def forward(self, weight: Tensor) -> Tensor:
        # weight: (d_out, d_in)
        if self.training:
            self._power_iteration(weight.detach(), 1)

        # The vectors are cloned so autograd keeps the values this forward used, as
        # torch's own spectral_norm does, even after the next step updates them in place.
        u = self.u.clone(memory_format=torch.contiguous_format)  # (d_out,)
        v = self.v.clone(memory_format=torch.contiguous_format)  # (d_in,)
        sigma = torch.dot(u, weight @ v)  # ()
        scale = torch.clamp(self.bound / (sigma + self.eps), max=1.0)  # ()
        return weight * scale  # (d_out, d_in)


def bound_spectral_norm(linear: nn.Linear, bound: float) -> None:
    """Attach `SpectralNormBound` to a Linear's weight."""
    parametrize.register_parametrization(linear, "weight", SpectralNormBound(linear.weight.detach(), bound))


class RandomFeatureGaussianProcess(nn.Module):
    """Gaussian process output layer over random Fourier features, with a Laplace posterior.

    `phi(h) = sqrt(2 / D) cos(h W^T + b)`, with `W` drawn from `N(0, I / lengthscale_squared)`,
    approximates the RBF kernel `exp(-|h - h'|^2 / (2 lengthscale_squared))` on the input, whose
    scale the module docstring relates to the trunk's. The output weights `beta` are trained as
    an ordinary Linear. `fit_posterior` in the owning probe then accumulates the precision of
    `beta` in one pass: `reset_precision`, `accumulate_precision` per batch, and
    `finalize_posterior`. Every piece of state that pass produces is a persistent buffer, so a
    saved state_dict carries the posterior with the weights.
    """

    def __init__(
        self,
        in_features: int,
        num_outputs: int,
        num_features: int = 1024,
        lengthscale_squared: float = 1024.0,
    ) -> None:
        super().__init__()
        self.num_features = num_features  # D
        self.register_buffer(
            "feature_weight", torch.randn(num_features, in_features) / math.sqrt(lengthscale_squared)
        )  # (D, d)
        self.register_buffer("feature_phase", torch.rand(num_features) * 2 * math.pi)  # (D,)
        self.beta = nn.Linear(num_features, num_outputs)
        self.register_buffer("precision", torch.eye(num_features))  # (D, D)
        self.register_buffer("covariance", torch.eye(num_features))  # (D, D)
        self.register_buffer("noise_variance", torch.ones(()))  # ()
        self.register_buffer("posterior_ready", torch.zeros((), dtype=torch.bool))  # ()
        # Data term of the precision while one fitting pass runs, kept in float64 because the
        # Gram matrix of a large training set loses precision in float32 before inversion.
        self._data_precision: Tensor | None = None

    def features(self, h: Tensor) -> Tensor:
        # h: (n, d), already layer-normalized by the owning probe
        projected = h @ self.feature_weight.T + self.feature_phase  # (n, D)
        return math.sqrt(2.0 / self.num_features) * torch.cos(projected)  # (n, D)

    def forward(self, h: Tensor) -> Tensor:
        # h: (n, d)
        return self.beta(self.features(h))  # (n, c), the MAP logits

    def reset_precision(self) -> None:
        """Start a fitting pass. Predictions stay MAP until `finalize_posterior`."""
        self._data_precision = torch.zeros(
            self.num_features, self.num_features, dtype=torch.float64, device=self.precision.device
        )  # (D, D)
        self.posterior_ready.fill_(False)

    def accumulate_precision(self, phi: Tensor, weight: Tensor) -> None:
        """Add `phi^T diag(weight) phi` for one batch to the data term."""
        # phi: (n, D); weight: (n,)
        assert self._data_precision is not None, "reset_precision must start a fitting pass."
        phi = phi.to(torch.float64)  # (n, D)
        weighted = phi * weight.to(torch.float64).unsqueeze(-1)  # (n, D)
        self._data_precision += weighted.T @ phi  # (D, D)

    def finalize_posterior(self, prior_precision: float = 1.0, noise_variance: float = 1.0) -> None:
        """`precision = prior_precision * I + data / noise_variance`, then invert it.

        Classification passes the defaults, since its weights already carry the Laplace
        curvature. Regression passes `1 / Var(y)` and the training mean squared error.
        """
        assert self._data_precision is not None, "reset_precision must start a fitting pass."
        identity = torch.eye(self.num_features, dtype=torch.float64, device=self.precision.device)  # (D, D)
        precision = prior_precision * identity + self._data_precision / noise_variance  # (D, D)
        covariance = torch.cholesky_inverse(torch.linalg.cholesky(precision))  # (D, D)

        self.precision.copy_(precision.to(self.precision.dtype))  # (D, D)
        self.covariance.copy_(covariance.to(self.covariance.dtype))  # (D, D)
        self.noise_variance.fill_(noise_variance)
        self.posterior_ready.fill_(True)
        self._data_precision = None

    def variance(self, phi: Tensor) -> Tensor:
        """Posterior variance of the latent function, `phi^T Sigma phi` per row."""
        # phi: (n, D)
        return ((phi @ self.covariance) * phi).sum(dim=-1)  # (n,)

    def mean_field_logits(self, logits: Tensor, variance: Tensor) -> Tensor:
        # logits: (n, c); variance: (n,)
        return logits / torch.sqrt(1.0 + MEAN_FIELD_FACTOR * variance).unsqueeze(-1)  # (n, c)
