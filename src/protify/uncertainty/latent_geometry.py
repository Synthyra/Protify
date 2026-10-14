"""Nearest rows and Gaussian densities in a head's latent spaces, which cross-fitting and the correctness model read.

**Nearest rows.** `nearest_rows` searches by Euclidean distance and `nearest_tanimoto` by Tanimoto
distance on 0/1 rows, one minus shared over either, as chemical fingerprints are compared. Both
search in blocks, so neither the query set nor the reference needs its whole distance matrix at once.

**Gaussians** of the trunk (Lee et al. 2018; Ren et al. 2021; Mukhoti et al. 2023):
`gaussian_scores` fits them to reference rows and scores queries at once. `TrunkGaussians` fits the
same Gaussians once and scores later queries with them, holding each class's covariance through the
Woodbury identity rather than as a `(p, p)` matrix, so a head with thousands of classes fits in memory.
Both give the squared Mahalanobis distance to the reference mean and, for single-label tasks, to each
class mean under the shared covariance, and each class's log density under its own covariance.

Shapes: `n` queries, `r` reference rows, `p` columns, `k` neighbors, `c` classes, `m` one class's rows.
"""

from __future__ import annotations

import math
import numpy as np
import torch

from dataclasses import dataclass
from torch import Tensor


# Query and reference rows per block of a neighbor search.
QUERY_BLOCK = 2048
REFERENCE_BLOCK = 8192
# Elements of the difference tensor measured at once when re-measuring chosen neighbors.
DIFFERENCE_ELEMENTS = 2 ** 28
# Ridge added to every covariance, relative to its mean variance, so a class with fewer rows than
# trunk dimensions still inverts.
RIDGE = 1e-3


def exact_distances(queries: Tensor, reference: Tensor, index: Tensor) -> Tensor:
    """Each query's Euclidean distance to its chosen reference rows, measured from differences, `(q, k)`."""
    # queries: (q, p); reference: (r, p); index: (q, k)
    rows = max(1, DIFFERENCE_ELEMENTS // (index.shape[1] * queries.shape[1]))
    return torch.cat([
        (queries[start:start + rows, None, :] - reference[index[start:start + rows]]).norm(dim=2)  # (rows, k)
        for start in range(0, queries.shape[0], rows)
    ])  # (q, k)


@torch.no_grad()
def nearest_rows(queries: Tensor, reference: Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Each query's `k` nearest reference rows by Euclidean distance, nearest first, in blocks.

    `torch.cdist` computes |q|^2 + |r|^2 - 2 q.r through a matrix product, which Protify runs in
    TF32 on CUDA, so a distance near 0 can be off by 0.3. It only picks the candidates: each chosen
    neighbor is then measured from differences and the neighbors re-sorted.

    Returns reference indices `(n, k)` as int32 and distances `(n, k)` as float32.
    """
    # queries: (n, p); reference: (r, p)
    k = min(k, reference.shape[0])
    indices, distances = [], []
    for start in range(0, queries.shape[0], QUERY_BLOCK):
        block = queries[start:start + QUERY_BLOCK]  # (q, p)
        best_distance = torch.empty((block.shape[0], 0), device=block.device)  # (q, 0)
        best_index = torch.empty((block.shape[0], 0), dtype=torch.long, device=block.device)  # (q, 0)
        for reference_start in range(0, reference.shape[0], REFERENCE_BLOCK):
            candidates = reference[reference_start:reference_start + REFERENCE_BLOCK]  # (r_block, p)
            distance = torch.cdist(block, candidates)  # (q, r_block)
            index = torch.arange(reference_start, reference_start + candidates.shape[0], device=block.device)
            merged_distance = torch.cat([best_distance, distance], dim=1)  # (q, k + r_block)
            merged_index = torch.cat([best_index, index.expand(block.shape[0], -1)], dim=1)  # (q, k + r_block)
            best_distance, order = merged_distance.topk(min(k, merged_distance.shape[1]), dim=1, largest=False)  # (q, k)
            best_index = merged_index.gather(1, order)  # (q, k)
        best_distance, order = exact_distances(block, reference, best_index).sort(dim=1)  # (q, k), (q, k)
        best_index = best_index.gather(1, order)  # (q, k)
        indices.append(best_index.cpu().numpy().astype(np.int32))
        distances.append(best_distance.cpu().numpy().astype(np.float32))
    return np.concatenate(indices), np.concatenate(distances)  # (n, k), (n, k)


@torch.no_grad()
def nearest_tanimoto(queries: Tensor, reference: Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Each query's `k` nearest reference rows by Tanimoto distance on 0/1 rows, nearest first, in blocks.

    The shared count is a matrix product of 0/1 values, exact in float32 and in TF32, whose inputs
    keep 0 and 1 and whose sums stay in float32. Ties keep the lower reference index first.

    Returns reference indices `(n, k)` as int32 and distances `(n, k)` as float32.
    """
    # queries: (n, s) of 0/1; reference: (r, s) of 0/1
    k = min(k, reference.shape[0])
    queries, reference = queries.float(), reference.float()
    reference_counts = reference.sum(dim=1)  # (r,)
    indices, distances = [], []
    for start in range(0, queries.shape[0], QUERY_BLOCK):
        block = queries[start:start + QUERY_BLOCK]  # (q, s)
        block_counts = block.sum(dim=1, keepdim=True)  # (q, 1)
        best_distance = torch.empty((block.shape[0], 0), device=block.device)  # (q, 0)
        best_index = torch.empty((block.shape[0], 0), dtype=torch.long, device=block.device)  # (q, 0)
        for reference_start in range(0, reference.shape[0], REFERENCE_BLOCK):
            candidates = reference[reference_start:reference_start + REFERENCE_BLOCK]  # (r_block, s)
            shared = block @ candidates.T  # (q, r_block)
            either = block_counts + reference_counts[reference_start:reference_start + candidates.shape[0]][None, :] - shared  # (q, r_block)
            distance = 1.0 - shared / either.clamp_min(1e-6)  # (q, r_block)
            index = torch.arange(reference_start, reference_start + candidates.shape[0], device=block.device)
            merged_distance = torch.cat([best_distance, distance], dim=1)  # (q, k + r_block)
            merged_index = torch.cat([best_index, index.expand(block.shape[0], -1)], dim=1)  # (q, k + r_block)
            # A stable sort keeps tied rows in reference order, which topk does not promise.
            best_distance, order = merged_distance.sort(dim=1, stable=True)  # (q, k + r_block)
            best_distance, best_index = best_distance[:, :k], merged_index.gather(1, order[:, :k])  # (q, k)
        indices.append(best_index.cpu().numpy().astype(np.int32))
        distances.append(best_distance.cpu().numpy().astype(np.float32))
    return np.concatenate(indices), np.concatenate(distances)  # (n, k), (n, k)


def regularized_precision(covariance: Tensor) -> tuple[Tensor, float]:
    """The inverse and log determinant of a covariance with the ridge added."""
    # covariance: (p, p)
    p = covariance.shape[0]
    ridge = RIDGE * float(torch.diagonal(covariance).mean()) + 1e-8
    regularized = covariance + ridge * torch.eye(p, dtype=covariance.dtype, device=covariance.device)  # (p, p)
    cholesky = torch.linalg.cholesky(regularized)  # (p, p)
    log_determinant = 2.0 * float(torch.log(torch.diagonal(cholesky)).sum())
    return torch.cholesky_inverse(cholesky), log_determinant  # (p, p), ()


def squared_mahalanobis(queries: Tensor, mean: Tensor, precision: Tensor) -> Tensor:
    # queries: (n, p), mean: (p,), precision: (p, p)
    centered = queries - mean  # (n, p)
    return ((centered @ precision) * centered).sum(dim=1)  # (n,)


@torch.no_grad()
def gaussian_scores(
    reference: Tensor, reference_labels: np.ndarray | None, queries: Tensor, num_classes: int | None
) -> dict[str, np.ndarray]:
    """Squared Mahalanobis distances and class log densities of `queries` under Gaussians fit to `reference`.

    `reference_labels` and `num_classes` are given for single-label tasks only. A class absent from
    the reference scores NaN.
    """
    # reference: (r, p); queries: (n, p)
    # reference_labels: (r,) or None
    reference, queries = reference.double(), queries.double()
    p = reference.shape[1]
    mean = reference.mean(dim=0)  # (p,)
    global_precision, _ = regularized_precision(torch.cov(reference.T))  # (p, p)
    scores = {"maha_global": squared_mahalanobis(queries, mean, global_precision).float().cpu().numpy()}  # (n,)
    if reference_labels is None or num_classes is None:
        return scores  # maha_global (n,)

    labels = torch.as_tensor(reference_labels, device=reference.device).long()  # (r,)
    class_means, centered_rows = [], []
    for label in range(num_classes):
        members = reference[labels == label]  # (r_c, p)
        class_means.append(members.mean(dim=0) if members.shape[0] else torch.full((p,), float("nan"), dtype=reference.dtype, device=reference.device))
        centered_rows.append(members - class_means[-1])
    pooled = torch.cat(centered_rows).T.cov(correction=0)  # (p, p)
    shared_precision, _ = regularized_precision(pooled)  # (p, p)

    class_distance, class_log_density = [], []
    for label in range(num_classes):
        members = centered_rows[label]  # (r_c, p)
        if members.shape[0] == 0:
            class_distance.append(torch.full((queries.shape[0],), float("nan"), dtype=reference.dtype, device=reference.device))
            class_log_density.append(class_distance[-1])
            continue
        class_distance.append(squared_mahalanobis(queries, class_means[label], shared_precision))  # (n,)
        # A class with fewer rows than dimensions leans on the pooled covariance.
        weight = members.shape[0] / (members.shape[0] + p)
        covariance = weight * (members.T @ members / members.shape[0]) + (1.0 - weight) * pooled  # (p, p)
        precision, log_determinant = regularized_precision(covariance)
        squared = squared_mahalanobis(queries, class_means[label], precision)  # (n,)
        class_log_density.append(-0.5 * (squared + log_determinant + p * math.log(2.0 * math.pi)))  # (n,)
    scores["maha_class"] = torch.stack(class_distance, dim=1).float().cpu().numpy()  # (n, c)
    scores["log_density_class"] = torch.stack(class_log_density, dim=1).float().cpu().numpy()  # (n, c)
    return scores  # maha_global (n,), maha_class (n, c), log_density_class (n, c)


@dataclass(frozen=True)
class _ClassDensity:
    """One class's Gaussian under its own shrunk covariance, `a V diag(lam) V^T + r I + b U^T U`.

    `scale` is each eigen-direction of the pooled covariance's variance in this class,
    `(1 - w) lam + r`. With `m < p` rows the class keeps `G = U V`, its centered rows in that
    eigenbasis, and the Cholesky factor of `K = I / b + G diag(1 / scale) G^T`; with `m >= p` it keeps
    the Cholesky factor of its whole covariance, `full`, and neither of the others.
    """

    mean: Tensor  # (p,)
    scale: Tensor  # (p,)
    rows: Tensor | None  # (m, p), G
    inner: Tensor | None  # (m, m), Cholesky factor of K
    full: Tensor | None  # (p, p), Cholesky factor of the regularized covariance
    log_determinant: float


class TrunkGaussians:
    """The trunk Gaussians of `gaussian_scores`, fitted once to reference rows and scored on later queries.

    The global and shared precisions are held as `(p, p)` matrices. Each class's covariance is the
    shared covariance shrunk toward the class's own rows, with the ridge `gaussian_scores` adds, and is
    held through the Woodbury identity and the matrix determinant lemma in the pooled covariance's
    eigenbasis: `(m, p)` rows and an `(m, m)` factor per class, which is what a head with thousands of
    small classes can hold. Scores agree with `gaussian_scores` on the same rows to float64 rounding.
    """

    def __init__(self, reference: Tensor, reference_labels: np.ndarray | None, num_classes: int | None) -> None:
        # reference: (r, p)
        # reference_labels: (r,) or None
        reference = reference.double()
        p = reference.shape[1]
        self.width = p
        self.mean = reference.mean(dim=0)  # (p,)
        self.global_precision, _ = regularized_precision(torch.cov(reference.T))  # (p, p)
        self.single_label = reference_labels is not None and num_classes is not None
        if not self.single_label:
            return

        labels = torch.as_tensor(reference_labels, device=reference.device).long()  # (r,)
        members = [reference[labels == label] for label in range(num_classes)]  # c of (m, p)
        self.present = [block.shape[0] > 0 for block in members]
        nan = torch.full((p,), float("nan"), dtype=reference.dtype, device=reference.device)
        self.class_means = torch.stack([block.mean(dim=0) if block.shape[0] else nan for block in members])  # (c, p)
        centered = [block - self.class_means[label] for label, block in enumerate(members)]  # c of (m, p)
        pooled = torch.cat(centered).T.cov(correction=0)  # (p, p)
        self.shared_precision, _ = regularized_precision(pooled)  # (p, p)
        eigenvalues, eigenvectors = torch.linalg.eigh(pooled)  # (p,), (p, p)
        self.eigenvectors = eigenvectors
        pooled_trace = float(torch.diagonal(pooled).sum())
        self.classes = [
            self._class_density(block, self.class_means[label], pooled, eigenvalues, pooled_trace) if block.shape[0] else None
            for label, block in enumerate(centered)
        ]

    def _class_density(self, centered: Tensor, mean: Tensor, pooled: Tensor, eigenvalues: Tensor, pooled_trace: float) -> _ClassDensity:
        # centered: (m, p)
        # mean: (p,), pooled: (p, p), eigenvalues: (p,)
        m, p = centered.shape
        weight = m / (m + p)
        # gaussian_scores' ridge is RIDGE times the regularized covariance's mean variance, plus 1e-8.
        mean_variance = (weight * float(centered.square().sum()) / m + (1.0 - weight) * pooled_trace) / p
        ridge = RIDGE * mean_variance + 1e-8
        if m >= p:
            covariance = weight * (centered.T @ centered / m) + (1.0 - weight) * pooled  # (p, p)
            regularized = covariance + ridge * torch.eye(p, dtype=covariance.dtype, device=covariance.device)
            cholesky = torch.linalg.cholesky(regularized)  # (p, p)
            log_determinant = 2.0 * float(torch.log(torch.diagonal(cholesky)).sum())
            return _ClassDensity(mean, eigenvalues, None, None, cholesky, log_determinant)

        scale = (1.0 - weight) * eigenvalues + ridge  # (p,)
        shrink = weight / m  # b
        rows = centered @ self.eigenvectors  # (m, p), G
        inner = torch.eye(m, dtype=rows.dtype, device=rows.device) / shrink + (rows / scale) @ rows.T  # (m, m), K
        cholesky = torch.linalg.cholesky(inner)  # (m, m)
        # det(A + b U^T U) = det(A) det(b K) by the matrix determinant lemma.
        log_determinant = float(torch.log(scale).sum()) + m * math.log(shrink) + 2.0 * float(torch.log(torch.diagonal(cholesky)).sum())
        return _ClassDensity(mean, scale, rows, cholesky, None, log_determinant)

    def _class_squared(self, density: _ClassDensity, queries: Tensor) -> Tensor:
        """Each query's squared Mahalanobis distance under one class's own covariance, `(n,)`."""
        # queries: (n, p)
        centered = queries - density.mean  # (n, p)
        if density.full is not None:
            solved = torch.linalg.solve_triangular(density.full, centered.T, upper=False)  # (p, n)
            return solved.square().sum(dim=0)  # (n,)
        rotated = centered @ self.eigenvectors  # (n, p), V^T d
        scaled = rotated / density.scale  # (n, p), A^-1 d in the eigenbasis
        projected = scaled @ density.rows.T  # (n, m), G A^-1 d
        solved = torch.linalg.solve_triangular(density.inner, projected.T, upper=False)  # (m, n)
        return (rotated * scaled).sum(dim=1) - solved.square().sum(dim=0)  # (n,)

    @torch.no_grad()
    def scores(self, queries: Tensor) -> dict[str, np.ndarray]:
        """The queries' scores, as `gaussian_scores` names and shapes them."""
        # queries: (n, p)
        queries = queries.double()
        scores = {"maha_global": squared_mahalanobis(queries, self.mean, self.global_precision).float().cpu().numpy()}  # (n,)
        if not self.single_label:
            return scores  # maha_global (n,)

        n = queries.shape[0]
        nan = torch.full((n,), float("nan"), dtype=queries.dtype, device=queries.device)
        class_distance, class_log_density = [], []
        for label, density in enumerate(self.classes):
            if density is None:
                class_distance.append(nan)
                class_log_density.append(nan)
                continue
            class_distance.append(squared_mahalanobis(queries, self.class_means[label], self.shared_precision))  # (n,)
            squared = self._class_squared(density, queries)  # (n,)
            class_log_density.append(-0.5 * (squared + density.log_determinant + self.width * math.log(2.0 * math.pi)))  # (n,)
        scores["maha_class"] = torch.stack(class_distance, dim=1).float().cpu().numpy()  # (n, c)
        scores["log_density_class"] = torch.stack(class_log_density, dim=1).float().cpu().numpy()  # (n, c)
        return scores  # maha_global (n,), maha_class (n, c), log_density_class (n, c)
