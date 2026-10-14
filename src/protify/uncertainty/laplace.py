"""A last-layer Laplace posterior, Kronecker-factored, for a head that ends in one Linear layer.

After training, with the weights early stopping restored, one pass over the training rows
fits a Gaussian posterior over the output layer's weights and bias, the rest of the head held
at its trained values (Daxberger et al. 2021, "Laplace Redux"). Its precision is the
generalized Gauss-Newton (GGN) of the training negative log-likelihood, approximated by one
Kronecker product (KFAC), plus an isotropic prior:

    P = N (A kron G) + lambda I,   A = mean_i h_i h_i^T,   G = mean_i Lambda_i,

with `h_i = [features_i, 1]` the row the output layer reads plus the bias input, `N` the
training rows, and `Lambda_i` the Hessian of a row's loss in its logits: `diag(p) - p p^T` for
a softmax, `p (1 - p)` per label for sigmoids, and `1 / sigma^2` for a Gaussian. Multilabel
shares one scalar curvature across labels, the mean of `p (1 - p)` over rows and labels, so
its variance is one per row whatever the label. Regression's `sigma^2` is the training mean
squared error, floored as the SNGP head floors it (`NOISE_VARIANCE_FLOOR_FRACTION` of
`Var(y_train)`, and the smallest normal float32), and its `G` is exact.

Scaling: `N A kron G` is the GGN of the negative log-likelihood summed over the training rows,
the likelihood a posterior conditions on, not the mean the trainer minimizes, and `lambda` is
the precision of a zero-mean Gaussian prior on every output weight and the bias. It is left
free: `methods.LAPLACE_PRIOR_PRECISIONS` holds 13 values from 1e-4 to 1e8, and
`laplace_temperature` chooses one on V_fit.

With `A = U_A diag(a) U_A^T` and `G = U_G diag(g) U_G^T`, the posterior covariance is diagonal
in the Kronecker eigenbasis, so the latent variance of logit `j` of a row `h` is

    var_j = sum_m U_G[j, m]^2 sum_i (h U_A)_i^2 / (N a_i g_m + lambda),

one matrix product per batch for every prior precision at once. Eigenvalues are clamped at
zero, since a rounding error can leave a positive semidefinite factor's smallest slightly
negative.

Width: the `linear` head reads the joined input, 18,688 features for the `both` arm, so `A` is
18,689 square. Below `FLOAT64_LIMIT` the factor is accumulated and decomposed in float64;
above it in float32 on the model's device, 1.4 GB for `A` and about three times that for the
eigendecomposition's eigenvectors and workspace, which fits a 16 GB T4. TF32 is switched off
for the whole fit, because Protify turns it on for CUDA and its 10-bit mantissa would round
the Gram matrix far more coarsely than float32 does.

Shapes: `b` rows in a batch, `n` rows, `f` the width the output layer reads, `e = f + 1` with
the bias input, `c` outputs, `k` the outputs of `G` (`c` single-label, 1 otherwise), `L` prior
precisions.
"""

from __future__ import annotations

import numpy as np
import torch

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .metrics import CLASSIFICATION_KINDS
from ..probes.projected_probe import NOISE_VARIANCE_FLOOR_FRACTION, NOISE_VARIANCE_MINIMUM


# Largest `e` whose factor is accumulated and decomposed in float64.
FLOAT64_LIMIT = 4096


@dataclass(frozen=True)
class KroneckerFactors:
    """The eigendecomposed factors of the output layer's GGN, per training row."""

    activation_eigenvalues: Tensor  # (e,)
    activation_eigenvectors: Tensor  # (e, e)
    output_eigenvalues: Tensor  # (k,)
    output_eigenvectors: Tensor  # (k, k)
    rows: int
    # Regression's sigma^2, which sets G = 1 / sigma^2; None for classification.
    noise_variance: float | None = None


@contextmanager
def exact_float32() -> Iterator[None]:
    """Float32 matrix products at full precision, TF32 off, restoring the caller's settings."""
    matmul_precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _output_layer(model: nn.Module) -> nn.Linear:
    output = getattr(model, "output", None)
    if not isinstance(output, nn.Linear) or getattr(model, "variance_output", None) is not None:
        raise ValueError("a last-layer Laplace posterior needs a head whose output is one Linear layer.")
    return output


def _with_bias_input(features: Tensor, dtype: torch.dtype) -> Tensor:
    # features: (b, f)
    ones = torch.ones(features.shape[0], 1, dtype=dtype, device=features.device)  # (b, 1)
    return torch.cat([features.to(dtype), ones], dim=1)  # (b, e)


def _eigh(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Eigenvalues clamped at zero, and eigenvectors, of a symmetric positive semidefinite matrix."""
    # matrix: (e, e)
    symmetric = 0.5 * (matrix + matrix.T)  # (e, e)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)  # (e,), (e, e)
    return eigenvalues.clamp_min(0.0), eigenvectors  # (e,), (e, e)


def factor_dtype(width: int) -> torch.dtype:
    """float64 for a factor of at most `FLOAT64_LIMIT` rows, float32 above it."""
    return torch.float64 if width <= FLOAT64_LIMIT else torch.float32


@torch.no_grad()
def fit_kfac(
    model: nn.Module, dataset: Dataset, collator: object, *, kind: str, batch_size: int, device: torch.device
) -> KroneckerFactors:
    """The Kronecker factors of the output layer's GGN, from one pass over `dataset`, the training rows."""
    output = _output_layer(model)
    model.to(device).eval()
    width = output.in_features + 1  # e
    dtype = factor_dtype(width)
    classes = output.out_features  # c
    with exact_float32():
        activations = torch.zeros(width, width, dtype=dtype, device=device)  # (e, e)
        curvature = torch.zeros(classes, classes, dtype=torch.float64, device=device)  # (c, c)
        rows = 0
        entry_curvature = 0.0
        squared_error = 0.0
        label_sum = 0.0
        label_square_sum = 0.0
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator):
            embeddings = batch["embeddings"].to(device)  # (b, d_in)
            features = model.trunk_features(embeddings)  # (b, f)
            logits = output(features).to(torch.float64)  # (b, c)
            h = _with_bias_input(features, dtype)  # (b, e)
            activations += h.T @ h  # (e, e)
            rows += h.shape[0]
            if kind in CLASSIFICATION_KINDS:
                p = logits.softmax(dim=-1)  # (b, c)
                curvature += torch.diag(p.sum(dim=0)) - p.T @ p  # (c, c)
            elif kind == "multilabel":
                p = logits.sigmoid()  # (b, c)
                entry_curvature += float((p * (1.0 - p)).sum())
            else:
                labels = batch["labels"].to(device).reshape(-1).to(torch.float64)  # (b,)
                squared_error += float(((logits.reshape(-1) - labels) ** 2).sum())
                label_sum += float(labels.sum())
                label_square_sum += float((labels ** 2).sum())

        if rows == 0:
            raise ValueError("a Laplace posterior needs at least one training row.")

        activation_eigenvalues, activation_eigenvectors = _eigh(activations / rows)  # (e,), (e, e)

    noise_variance = None
    if kind in CLASSIFICATION_KINDS:
        output_factor = (curvature / rows).cpu()  # (c, c)
    elif kind == "multilabel":
        output_factor = torch.full((1, 1), entry_curvature / (rows * classes), dtype=torch.float64)  # (1, 1)
    else:
        label_mean = label_sum / rows
        label_variance = label_square_sum / rows - label_mean ** 2
        noise_variance = max(
            squared_error / rows, NOISE_VARIANCE_FLOOR_FRACTION * max(label_variance, 0.0), NOISE_VARIANCE_MINIMUM
        )
        output_factor = torch.full((1, 1), 1.0 / noise_variance, dtype=torch.float64)  # (1, 1)
    output_eigenvalues, output_eigenvectors = _eigh(output_factor)  # (k,), (k, k)
    return KroneckerFactors(
        activation_eigenvalues, activation_eigenvectors, output_eigenvalues, output_eigenvectors, rows, noise_variance
    )


@torch.no_grad()
def latent_variance(
    model: nn.Module,
    factors: KroneckerFactors,
    dataset: Dataset,
    collator: object,
    prior_precisions: Sequence[float],
    *,
    kind: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Each row's latent logit variance at every prior precision.

    `(n, L, c)` per class for single-label tasks, `(n, L)` for multilabel and regression, whose
    output factor is one scalar.
    """
    _output_layer(model)
    model.to(device).eval()
    dtype = factors.activation_eigenvectors.dtype
    eigenvectors = factors.activation_eigenvectors.to(device)  # (e, e)
    precisions = torch.as_tensor(prior_precisions, dtype=torch.float64, device=device)  # (L,)
    posterior = (
        factors.rows
        * factors.activation_eigenvalues.to(device, torch.float64)[:, None, None]
        * factors.output_eigenvalues.to(device, torch.float64)[None, None, :]
        + precisions[None, :, None]
    )  # (e, L, k)
    inverse = (1.0 / posterior).to(dtype).reshape(posterior.shape[0], -1)  # (e, L * k)
    output_weights = (factors.output_eigenvectors.to(device, dtype) ** 2)  # (k, k): [j, m] = U_G[j, m]^2
    variances = []
    with exact_float32():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator):
            features = model.trunk_features(batch["embeddings"].to(device))  # (b, f)
            rotated = _with_bias_input(features, dtype) @ eigenvectors  # (b, e)
            per_output = (rotated ** 2) @ inverse  # (b, L * k)
            per_output = per_output.reshape(rotated.shape[0], precisions.shape[0], -1)  # (b, L, k)
            if kind in CLASSIFICATION_KINDS:
                variance = torch.einsum("blm,jm->blj", per_output, output_weights)  # (b, L, c)
            else:
                variance = per_output[..., 0]  # (b, L)
            variances.append(variance.float().cpu().numpy())
    return np.concatenate(variances)  # (n, L, c) or (n, L)
