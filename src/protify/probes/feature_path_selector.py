"""Rank embedding features by when they enter an L1 regularization path.

Wide sparse embeddings, such as max-pooled ESMC sparse-autoencoder features, carry
their signal in a subset of their columns, and a probe over all of them wastes capacity
and, for the estimator probes, more host memory than a large dataset can afford.

Ranking columns by univariate variance would throw away exactly the features this is
meant to keep: a low-variance feature is often informative only next to a high-variance
one, and no univariate statistic can see that. This module instead fits a small MLP under
a group-lasso penalty on its first layer, sweeps the penalty from strong to weak, and
records every feature's weight-column norm at each step. Ranking is read off those
recorded trajectories, so it comes from a fit that sees interactions and nonlinearity.

Reading the ranking from the trajectory rather than from exact zeros is what keeps the
optimizer a free choice. A proximal operator would produce exact zeros, but only under
plain gradient descent and only at a step size tied to the learning rate, which converges
far too slowly to sweep 131072 features.

The path fit is a ranking device, not the final model. Stage 3 refits at each candidate
width with no penalty at all, and the caller then trains the real probe on the winner.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from scipy import sparse


try:
    from ..console_output import print_message
except ImportError:
    try:
        from protify.console_output import print_message
    except ImportError:
        from console_output import print_message
from .losses import compute_probe_loss, get_loss_fct


# A feature matrix arrives either dense or as scipy CSR. Pooled SAE features measure
# roughly 10% to 34% dense, which is far too dense for a sparse matrix multiply to pay
# off on a GPU but still beats a dense array on host memory, so storage stays sparse and
# each minibatch is densified on the device.
FeatureMatrix = np.ndarray | sparse.csr_matrix

# Smooths the group norm so its gradient stays defined at zero, and guards the
# per-alpha normalization in `path_importance` when a whole path step collapses.
NORM_EPSILON = 1e-12


@dataclass(frozen=True)
class FeaturePathConfig:
    """Hyperparameters for the path fit.

    `max_fit_rows` subsamples the training split for the path fit alone. Ranking
    features is a far lower-capacity problem than fitting the probe, and the full
    163k-row pair datasets would otherwise let selection dominate both wall-clock and
    host memory. The refit sees every row.
    """

    hidden_size: int = 1024
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs_per_alpha: int = 3
    n_alphas: int = 20
    alpha_min_ratio: float = 1e-3
    l1_ratio: float = 0.9
    max_fit_rows: int = 20000
    min_output_size: int = 256
    refit_epochs: int = 20
    refit_learning_rate: float = 1e-3
    refit_tolerance: float = 0.01
    # Refits per candidate width, averaged. The width search picks the best of about ten
    # candidates on one validation split, which is ten chances for a lucky draw to win, and
    # a width chosen that way can lose more on test than the selection gained. Averaging
    # several refits per width is what stops one noisy fit from setting the probe's input.
    refit_seeds: int = 3
    seed: int = 42
    device: str | None = None


@dataclass(frozen=True)
class FeatureSelection:
    """The embedding columns a probe keeps, and the width they were chosen from.

    This is the whole fitted artifact. It has to travel with the probe and be applied
    identically at inference, because a probe trained on selected columns cannot read a
    full-width embedding.
    """

    indices: np.ndarray  # (s,) int64, ascending
    input_size: int  # f, the width the selection was made from

    def __post_init__(self) -> None:
        if self.indices.ndim != 1:
            raise ValueError(f"Feature indices must be 1D; got shape {self.indices.shape}.")
        if self.indices.size == 0:
            raise ValueError("A feature selection must keep at least one column.")
        if self.indices.dtype != np.int64:
            object.__setattr__(self, "indices", self.indices.astype(np.int64))
        if int(self.indices[-1]) >= self.input_size:
            raise ValueError(
                f"Feature index {int(self.indices[-1])} is outside an embedding of width "
                f"{self.input_size}."
            )

    @property
    def output_size(self) -> int:
        return int(self.indices.shape[0])

    def transform_numpy(self, features: np.ndarray) -> np.ndarray:
        # features: (n, f) -> (n, s)
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features; got shape {features.shape}.")
        if features.shape[1] != self.input_size:
            raise ValueError(
                f"Feature width {features.shape[1]} does not match the selected width "
                f"{self.input_size}."
            )
        return features[:, self.indices]  # (n, s)

    def transform_tensor(self, embedding: torch.Tensor) -> torch.Tensor:
        """Select from one pooled embedding, whatever leading shape it carries.

        For a pair dataset the incoming vector is protein A concatenated with protein B,
        so the selection spans the joined width and may keep different columns on each
        side. That is intended: the path fit sees the concatenation the probe will see.
        """
        # embedding: (..., f) -> (..., s)
        if embedding.shape[-1] != self.input_size:
            raise ValueError(
                f"Embedding width {embedding.shape[-1]} does not match the selected width "
                f"{self.input_size}."
            )
        indices = torch.from_numpy(self.indices).to(embedding.device)  # (s,)
        return embedding.index_select(-1, indices)  # (..., s)

    def narrowed_by(self, kept: np.ndarray) -> FeatureSelection:
        """This selection followed by a second one over the columns it already keeps.

        Two narrowing steps in a row would otherwise have to be replayed in order at
        inference. Composing them into one index array over the original width means the
        probe still carries a single artifact, and it stays exact because the second step
        indexes positions in the first step's output.
        """
        # kept: (s2,) positions within this selection's (s,) output
        return FeatureSelection(indices=self.indices[kept], input_size=self.input_size)

    def save(self, path: str | Path) -> None:
        np.savez(Path(path), indices=self.indices, input_size=np.int64(self.input_size))

    @classmethod
    def load(cls, path: str | Path) -> FeatureSelection:
        with np.load(Path(path)) as stored:
            return cls(indices=stored["indices"], input_size=int(stored["input_size"]))


class _PathMLP(nn.Module):
    """The probe-shaped fit whose first layer carries the group penalty.

    Deliberately not `MLPProbe`. That probe opens with a LayerNorm over the feature axis,
    and those statistics mix every feature together, so shrinking one feature's weights
    would not actually reduce its influence on the forward pass. Here the first operation
    is the penalized Linear, which makes a small column mean exactly one thing.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_labels: int,
        dropout: float,
        task_type: str,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    @property
    def feature_weights(self) -> torch.Tensor:
        # nn.Linear stores its weight as (h, f), so one feature owns one column.
        return self.input_projection.weight  # (h, f)

    def group_norms(self) -> torch.Tensor:
        """Each input feature's outgoing weight column, measured as one group.

        Smoothed by NORM_EPSILON because the exact 2-norm has no gradient at zero, and a
        column the penalty has driven very small would otherwise poison the backward pass
        with NaN.
        """
        # (h, f) -> (f,)
        return torch.sqrt(self.feature_weights.pow(2).sum(dim=0) + NORM_EPSILON ** 2)  # (f,)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (b, f)
        logits = self.head(self.input_projection(X))  # (b, c)
        if self.task_type == 'sigmoid_regression':
            # SoftBCELoss reads probabilities, matching how MLPProbe squashes first.
            return logits.sigmoid()  # (b, c)
        return logits  # (b, c)


def path_importance(path_norms: np.ndarray) -> np.ndarray:
    """Rank features by how much of the regularization path they survive.

    A true lasso path reads off exact zeros: a feature is dead until some alpha and alive
    below it, and the ordering is the order features enter. Gradient training does not
    produce exact zeros, so this reads the same signal off the trajectory instead. At each
    alpha a feature's group norm is expressed as a fraction of the largest group norm at
    that alpha, and its importance is the mean of that fraction across the path.

    That reduces to the classic entry ordering when the norms really are zero before entry,
    since the fraction is then zero at every alpha above entry and positive below, and it
    folds magnitude in as a tie-break where two features enter together. The alpha grid is
    geometric, so a plain mean already weights the path uniformly in log alpha.

    No threshold decides who is dead, which matters because any such cutoff would be an
    arbitrary choice sitting directly under the feature ranking.
    """
    # path_norms: (a, f) -> (f,)
    largest = path_norms.max(axis=1, keepdims=True)  # (a, 1)
    return (path_norms / np.maximum(largest, NORM_EPSILON)).mean(axis=0)  # (f,)


def _feature_support(features: FeatureMatrix) -> np.ndarray:
    """Columns holding at least one nonzero entry."""
    # features: (n, f) -> (n_support,)
    if sparse.issparse(features):
        counts = features.getnnz(axis=0)  # (f,)
    else:
        counts = np.count_nonzero(features, axis=0)  # (f,)
    return np.flatnonzero(counts).astype(np.int64)  # (n_support,)


def _column_statistics(features: FeatureMatrix) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and standard deviation over the fit rows.

    Centering is applied on the device once a minibatch is dense, so the host matrix
    keeps its zeros and its memory advantage.
    """
    # features: (n, f)
    if sparse.issparse(features):
        mean = np.asarray(features.mean(axis=0)).ravel().astype(np.float64)  # (f,)
        second_moment = np.asarray(features.multiply(features).mean(axis=0)).ravel()  # (f,)
    else:
        dense = features.astype(np.float64)  # (n, f)
        mean = dense.mean(axis=0)  # (f,)
        second_moment = (dense ** 2).mean(axis=0)  # (f,)
    scale = np.sqrt(np.maximum(second_moment - mean ** 2, 0.0))  # (f,)
    # A column that is constant across the fit rows carries no signal for the path, and
    # dividing by a zero spread would produce inf. Leave it unscaled; it receives no
    # gradient and dies at the first alpha.
    scale[scale == 0.0] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)  # (f,), (f,)


def _densify_rows(features: FeatureMatrix, rows: np.ndarray) -> np.ndarray:
    # features: (n, f); rows: (b,) -> (b, f)
    selected = features[rows]
    if sparse.issparse(selected):
        return selected.toarray().astype(np.float32)  # (b, f)
    return np.ascontiguousarray(selected, dtype=np.float32)  # (b, f)


def subsample_rows(n_rows: int, max_rows: int, generator: np.random.Generator) -> np.ndarray:
    """Pick fit rows in ascending order, which keeps a CSR row slice cheap.

    Public because a caller that streams embeddings off disk has to apply the same rule
    before it materializes anything. Applying it twice is harmless: the second call sees
    a matrix already at or under the cap and keeps every row.
    """
    # -> (min(n_rows, max_rows),)
    if n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int64)  # (n_rows,)
    return np.sort(generator.choice(n_rows, size=max_rows, replace=False)).astype(np.int64)  # (max_rows,)


def _nearest_power_of_two(count: int) -> int:
    """Round to the nearest power of two in log space.

    Rounding down instead would discard almost half the surviving features whenever the
    survivor count lands just under a power of two.
    """
    if count < 1:
        return 1
    return int(2 ** max(int(round(float(np.log2(count)))), 0))


def _label_tensor(labels: np.ndarray, task_type: str, device: torch.device) -> torch.Tensor:
    # labels: (n,) or (n, c)
    if task_type == "singlelabel":
        return torch.from_numpy(np.asarray(labels)).long().to(device)  # (n,) or (n, c)
    return torch.from_numpy(np.asarray(labels)).float().to(device)  # (n,) or (n, c)


def _num_labels(labels: np.ndarray, task_type: str) -> int:
    """Output width for the path fit, matching how the probe reads this task."""
    # labels: (n,) or (n, c)
    array = np.asarray(labels)
    if task_type == "singlelabel":
        return int(array.max()) + 1
    if array.ndim == 1:
        return 1
    return int(array.shape[1])


class _PathBatches:
    """Minibatches of one split, densified and normalized on the device."""

    def __init__(
        self,
        features: FeatureMatrix,
        labels: torch.Tensor,
        mean: np.ndarray,
        scale: np.ndarray,
        batch_size: int,
        device: torch.device,
    ) -> None:
        # labels: (n,) or (n, c), mean: (f,), scale: (f,)
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.device = device
        self.mean = torch.from_numpy(mean).to(device)  # (f,)
        self.scale = torch.from_numpy(scale).to(device)  # (f,)

    def batches(self, order: np.ndarray | None = None) -> Iterator[tuple[torch.Tensor, np.ndarray]]:
        # order: (n,)
        rows = np.arange(self.features.shape[0], dtype=np.int64) if order is None else order
        for start in range(0, rows.shape[0], self.batch_size):
            # Sorting keeps a CSR row slice sequential; the shuffle across batches is what
            # decorrelates the gradient, not the order within one batch.
            batch_rows = np.sort(rows[start:start + self.batch_size])  # (b,)
            X = torch.from_numpy(_densify_rows(self.features, batch_rows)).to(self.device)  # (b, f)
            X = (X - self.mean) / self.scale  # (b, f)
            yield X, self.labels[batch_rows]  # (b, f), (b,) or (b, c)


def _penalty_scale(
    model: _PathMLP,
    batches: _PathBatches,
    loss_fct: nn.Module,
    task_type: str,
    num_labels: int,
    l1_ratio: float,
) -> float:
    """The alpha at which the group penalty first outweighs the loss for every feature.

    The group-lasso subgradient condition says feature j is pulled toward zero whenever
    `alpha * l1_ratio` exceeds the norm of its loss gradient column. Taking the largest
    such norm sets the top of the path from the data, rather than from a guessed starting
    value that would be wrong by orders of magnitude on a different task or embedding.
    """
    model.eval()
    model.zero_grad(set_to_none=True)
    n_batches = 0
    for X, labels in batches.batches():
        logits = model(X)  # (b, c)
        loss = compute_probe_loss(loss_fct, logits, labels, task_type, num_labels)  # ()
        loss.backward()
        n_batches += 1

    gradient = model.feature_weights.grad  # (h, f)
    if gradient is None:
        raise RuntimeError("The path fit produced no gradient for the penalized layer.")
    column_norms = (gradient / n_batches).norm(dim=0)  # (f,)
    model.zero_grad(set_to_none=True)
    return float(column_norms.max().item()) / max(l1_ratio, NORM_EPSILON)


def _validation_loss(
    model: _PathMLP,
    batches: _PathBatches,
    loss_fct: nn.Module,
    task_type: str,
    num_labels: int,
) -> float:
    model.eval()
    total = 0.0
    n_batches = 0
    with torch.no_grad():
        for X, labels in batches.batches():
            logits = model(X)  # (b, c)
            total += float(
                compute_probe_loss(loss_fct, logits, labels, task_type, num_labels).item()
            )
            n_batches += 1
    return total / max(n_batches, 1)


def _require_finite(name: str, values: FeatureMatrix) -> None:
    """Reject a non-finite input before it becomes a NaN twenty alpha steps later.

    A corrupt embedding cache is the realistic source. Catching it here names the array
    that is wrong, where the downstream guard could only report that the fit diverged.
    """
    stored_values = values.data if sparse.issparse(values) else np.asarray(values)
    if not np.isfinite(stored_values).all():
        raise ValueError(f"{name} contains NaN or infinite values, so no path can be fit.")


def _candidate_widths(n_ranked: int, min_output_size: int) -> list[int]:
    """Powers of two from `min_output_size` up to the ranked set, widest last."""
    widths = []
    width = _nearest_power_of_two(min_output_size)
    while width <= n_ranked:
        widths.append(width)
        width *= 2
    return widths if widths else [n_ranked]


def _select_columns(features: FeatureMatrix, columns: np.ndarray) -> FeatureMatrix:
    """Keep the given columns, returning a layout that still slices rows cheaply."""
    # features: (n, f); columns: (s,) -> (n, s)
    selected = features[:, columns]
    if sparse.issparse(selected):
        return selected.tocsr()  # (n, s)
    return selected  # (n, s)


def _refit_validation_loss(
    train_features: FeatureMatrix,
    train_labels: torch.Tensor,
    valid_features: FeatureMatrix,
    valid_labels: torch.Tensor,
    columns: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    task_type: str,
    num_labels: int,
    config: FeaturePathConfig,
    device: torch.device,
    seed: int,
) -> float:
    """Score one candidate width by training an MLP on it with no penalty at all."""
    # columns: (s,)
    # train_labels: (n_train,) or (n_train, c), valid_labels: (n_valid,) or (n_valid, c), mean: (f,), scale: (f,)
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    model = _PathMLP(
        int(columns.size), config.hidden_size, num_labels, config.dropout, task_type
    ).to(device)
    # The path fit starts every weight at zero so its entry threshold has a closed form.
    # This refit carries no penalty and no threshold, so it takes the ordinary Linear
    # initialization, and Adam, which converges in far fewer epochs.
    model.input_projection.reset_parameters()
    loss_fct = get_loss_fct(task_type)
    train_batches = _PathBatches(
        _select_columns(train_features, columns),
        train_labels,
        mean[columns],  # (s,)
        scale[columns],  # (s,)
        config.batch_size,
        device,
    )
    valid_batches = _PathBatches(
        _select_columns(valid_features, columns),
        valid_labels,
        mean[columns],  # (s,)
        scale[columns],  # (s,)
        config.batch_size,
        device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.refit_learning_rate)
    for _ in range(config.refit_epochs):
        model.train()
        order = generator.permutation(train_features.shape[0])
        for X, labels in train_batches.batches(order):
            loss = compute_probe_loss(
                loss_fct, model(X), labels, task_type, num_labels
            )  # ()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return _validation_loss(model, valid_batches, loss_fct, task_type, num_labels)


def fit_feature_path_selection(
    train_features: FeatureMatrix,
    train_labels: np.ndarray,
    valid_features: FeatureMatrix,
    valid_labels: np.ndarray,
    task_type: str,
    config: FeaturePathConfig | None = None,
) -> FeatureSelection:
    """Choose the embedding columns a probe should keep.

    Fits on train and reads the stopping point off validation. Test is never touched,
    so the selection cannot leak into the reported score.
    """
    # train_features: (n_train, f); valid_features: (n_valid, f)
    # train_labels: (n_train,) or (n_train, c), valid_labels: (n_valid,) or (n_valid, c)
    if train_features.shape[1] != valid_features.shape[1]:
        raise ValueError(
            f"Train width {train_features.shape[1]} does not match valid width "
            f"{valid_features.shape[1]}."
        )
    input_size = int(train_features.shape[1])  # f
    _require_finite('train_features', train_features)
    _require_finite('train_labels', train_labels)
    _require_finite('valid_features', valid_features)
    _require_finite('valid_labels', valid_labels)

    if config is None:
        config = FeaturePathConfig()
    generator = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(
        config.device
        if config.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_rows = subsample_rows(train_features.shape[0], config.max_fit_rows, generator)
    valid_rows = subsample_rows(valid_features.shape[0], config.max_fit_rows, generator)
    train_features = train_features[train_rows]  # (n_fit, f)
    valid_features = valid_features[valid_rows]  # (n_valid_fit, f)
    train_labels = np.asarray(train_labels)[train_rows]
    valid_labels = np.asarray(valid_labels)[valid_rows]

    # Stage 0. A column that is zero for every training protein cannot inform any probe,
    # and carrying it through the path fit costs a weight column and a gradient column.
    support = _feature_support(train_features)  # (f_support,)
    if support.size == 0:
        raise ValueError("Every training embedding column is zero; nothing can be selected.")
    print_message(
        f"Feature path: {support.size} of {input_size} columns have training support"
    )
    train_features = train_features[:, support]  # (n_fit, f_support)
    valid_features = valid_features[:, support]  # (n_valid_fit, f_support)
    n_support = int(support.size)

    mean, scale = _column_statistics(train_features)  # (f_support,), (f_support,)
    num_labels = _num_labels(train_labels, task_type)  # c
    loss_fct = get_loss_fct(task_type)
    model = _PathMLP(
        n_support, config.hidden_size, num_labels, config.dropout, task_type
    ).to(device)

    train_batches = _PathBatches(
        train_features,
        _label_tensor(train_labels, task_type, device),
        mean,
        scale,
        config.batch_size,
        device,
    )
    valid_batches = _PathBatches(
        valid_features,
        _label_tensor(valid_labels, task_type, device),
        mean,
        scale,
        config.batch_size,
        device,
    )

    alpha_max = _penalty_scale(
        model, train_batches, loss_fct, task_type, num_labels, config.l1_ratio
    )
    if not np.isfinite(alpha_max) or alpha_max <= 0.0:
        raise RuntimeError(
            f"The path fit produced a non-positive penalty scale ({alpha_max}); the "
            "training labels or embeddings carry no gradient signal."
        )
    alphas = np.geomspace(alpha_max, alpha_max * config.alpha_min_ratio, config.n_alphas)  # (a,)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    ridge_weight = (1.0 - config.l1_ratio) / 2.0

    # Stage 1. Walk alpha from strong to weak with warm starts, the glmnet traversal, and
    # record every feature's group norm at each step. Reading the ranking off that
    # recorded trajectory, rather than off exact zeros, is what lets this use whatever
    # optimizer converges fastest. A proximal step would give exact zeros but only under
    # plain gradient descent, and only at a step size tied to the learning rate.
    path_norms = np.zeros((config.n_alphas, n_support), dtype=np.float64)  # (a, f_support)

    for alpha_index, alpha in enumerate(alphas):
        group_penalty = float(alpha) * config.l1_ratio
        ridge_penalty = float(alpha) * ridge_weight
        for _ in range(config.epochs_per_alpha):
            model.train()
            order = generator.permutation(train_features.shape[0])
            for X, labels in train_batches.batches(order):
                logits = model(X)  # (b, c)
                loss = compute_probe_loss(loss_fct, logits, labels, task_type, num_labels)  # ()
                # Elastic net over input features: the group-lasso half sums each
                # feature's whole outgoing column as one norm, so shrinkage acts on the
                # feature rather than on individual weights, and the ridge half keeps the
                # objective strictly convex in each group.
                loss = (
                    loss
                    + group_penalty * model.group_norms().sum()
                    + ridge_penalty * model.feature_weights.pow(2).sum()
                )  # ()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            norms = model.group_norms().detach().cpu().numpy()  # (f_support,)
        # Backstop. Inputs are checked for finiteness up front and Adam bounds its own
        # step, so reaching this means the loss itself overflowed.
        if not np.isfinite(norms).all():
            raise RuntimeError(
                f"The path fit diverged at alpha {float(alpha):.4g}. Lower "
                f"learning_rate below {config.learning_rate}."
            )
        path_norms[alpha_index] = norms
        print_message(
            f"Feature path: alpha {float(alpha):.4g}, "
            f"largest group norm {float(norms.max()):.4g}, "
            f"median {float(np.median(norms)):.4g}"
        )

    # Stage 2. Rank by how much of the path each feature survives.
    ranking = np.argsort(-path_importance(path_norms))  # (f_support,)

    # Stage 3. Choose the width by unpenalized refit, not by reading validation loss off
    # the path. On the path a strong penalty shrinks every surviving weight, so its
    # validation loss measures that shrinkage rather than the quality of the subset it
    # kept, and the comparison would run downhill to the dense end of the path every
    # time. Refitting each candidate width with no penalty removes the confound.
    widths = _candidate_widths(n_support, config.min_output_size)  # (n_widths,)
    refit_losses = np.array(
        [
            float(
                np.mean([
                    _refit_validation_loss(
                        train_features,
                        train_batches.labels,
                        valid_features,
                        valid_batches.labels,
                        np.sort(ranking[:width]),
                        mean,
                        scale,
                        task_type,
                        num_labels,
                        config,
                        device,
                        config.seed + offset,
                    )
                    for offset in range(max(config.refit_seeds, 1))
                ])
            )
            for width in widths
        ],
        dtype=np.float64,
    )  # (n_widths,)
    for width, refit_loss in zip(widths, refit_losses, strict=True):
        print_message(f"Feature path: width {width}, unpenalized valid loss {refit_loss:.4f}")
    if not np.isfinite(refit_losses).any():
        raise RuntimeError(
            f"Every unpenalized refit diverged. Lower learning_rate below "
            f"{config.learning_rate}."
        )

    # Prefer the narrowest width that does no measurable harm. Taking the outright
    # minimum would keep a wider set for a difference the validation split cannot
    # resolve, which is the same reason glmnet reports lambda.1se next to lambda.min.
    best = float(np.nanmin(refit_losses))
    tolerance = best + config.refit_tolerance * abs(best)
    output_size = int(widths[int(np.argmax(refit_losses <= tolerance))])

    selected = np.sort(support[ranking[:output_size]])  # (s,)
    print_message(f"Feature path: keeping {selected.size} of {input_size} columns")
    return FeatureSelection(indices=selected, input_size=input_size)
