"""What one trained head says about each split, saved per seed.

Everything the uncertainty methods need is collected once per seed and written to one `.npz`:
the head's outputs, the Gaussian process variance for SNGP, the MC dropout summary for the
plain head, and each row's distance to its nearest training row in the trunk's feature space.
Methods, calibration, and conformal sets are then computed from these files, so a resumed run
retrains nothing and a new method needs no retraining.

Shapes: `n` rows, `c` classes or labels, `p` trunk width, `T` MC dropout passes, `r` training
rows, `q` query rows in a block.
"""

from __future__ import annotations

import os
import time
import numpy as np
import torch

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .prediction_arrays import PredictionAccumulator, mapped_prediction_archive, release_prediction_pages, write_prediction_archive


SPLITS = ("v_fit", "v_cal", "test")
DROPOUT_MODULES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)
QUERY_BLOCK = 1024
REFERENCE_BLOCK = 16384


@dataclass
class SplitPredictions:
    """One seed's head on one split.

    `outputs` holds logits `(n, c)` for classification, mean-field logits for SNGP, and the
    prediction `(n,)` for regression. `variance` is SNGP's: the latent logit variance for
    classification, the predictive variance for regression. `mc_outputs` is the plain head's
    MC dropout mean, probabilities `(n, c)` or predictions `(n,)`, and `mc_variance` the
    spread of its regression passes. A head with a variance output (`faithful`, `beta_nll`)
    saves its predicted variance in `variance` too.

    `laplace_variance` is the last-layer Laplace posterior's latent variance at each prior
    precision of `methods.LAPLACE_PRIOR_PRECISIONS`: `(n, L, c)` per class for single-label,
    `(n, L)` for multilabel, whose labels share one curvature, and for regression.
    `loss_prediction` is the cross-fitted loss predictor's `log(loss + 1e-6)` `(n,)`. `trunk`
    is each row's trunk features `(n, p)`, saved under `save_trunk_features`. Every field
    is per row, so a split's rows slice every field alike, and a file saved before a field
    existed loads it as None.
    """

    labels: np.ndarray
    outputs: np.ndarray
    variance: np.ndarray | None = None
    mc_outputs: np.ndarray | None = None
    mc_variance: np.ndarray | None = None
    nn_distance: np.ndarray | None = None
    digests: np.ndarray | None = None
    laplace_variance: np.ndarray | None = None
    loss_prediction: np.ndarray | None = None
    trunk: np.ndarray | None = None
    label_mask: np.ndarray | None = None
    observation_ids: np.ndarray | None = None
    protein_ids: np.ndarray | None = None
    cluster_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("observation_ids", "protein_ids", "cluster_ids"):
            values = getattr(self, name)
            if values is not None and (values.shape != (len(self.labels),) or values.dtype.kind not in {"U", "S"}):
                raise ValueError(f"{name} must contain one string identity per prediction row")
        # labels, label_mask: (n,), (n, 1), or (n, c)
        if self.label_mask is not None:
            if self.label_mask.dtype != np.bool_ or self.label_mask.shape != self.labels.shape:
                raise ValueError("label_mask must be boolean with exactly the labels shape")
            rows_per_block = max(1, 1_000_000 // max(1, int(np.prod(self.labels.shape[1:]))))
            for start in range(0, len(self.labels), rows_per_block):
                labels, observed = self.labels[start:start + rows_per_block], self.label_mask[start:start + rows_per_block]
                if not np.isfinite(labels[observed]).all():
                    raise ValueError("observed prediction labels must be finite")
                release_prediction_pages(self.labels)
                release_prediction_pages(self.label_mask)


@contextmanager
def mc_dropout(model: nn.Module) -> Iterator[nn.Module]:
    """Evaluation mode everywhere except the Dropout modules.

    Only dropout is switched on. The spectral bound's power iteration advances only in
    training mode, so MC passes leave its singular vectors as training left them.
    """
    was_training = model.training
    model.eval()
    dropouts = [module for module in model.modules() if isinstance(module, DROPOUT_MODULES)]
    for module in dropouts:
        module.train()
    try:
        yield model
    finally:
        model.train(was_training)


def _loader(dataset: Dataset, collator: object, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)


def _probabilities(outputs: Tensor, task_kind: str) -> Tensor:
    # outputs: (b, c)
    if task_kind == "multilabel":
        return outputs.sigmoid()  # (b, c)
    return outputs.softmax(dim=-1)  # (b, c)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    dataset: Dataset,
    collator: object,
    *,
    task_kind: str,
    batch_size: int,
    device: torch.device,
    mc_samples: int = 0,
    disk_backed: bool = False,
) -> tuple[SplitPredictions, float]:
    """The head's predictions on one split, and the seconds its plain forward pass took.

    A head with a Gaussian process output reports mean-field logits and its variance through
    `predict_with_variance`, as a head with a variance output reports its mean and variance.
    With `mc_samples`, the plain head also runs that many dropout passes per batch.
    `disk_backed` streams complete arrays to anonymous temporary files and returns read-only
    mappings. Otherwise arrays are allocated once in RAM. Neither path retains batch lists.
    """
    model.to(device).eval()
    predicts_variance = bool(getattr(model, "predicts_variance", False))
    regression = task_kind == "regression"
    seconds = 0.0
    with PredictionAccumulator(len(dataset), disk_backed=disk_backed) as accumulated:
        for batch in _loader(dataset, collator, batch_size):
            embeddings = batch["embeddings"].to(device)  # (b, d_in)
            values = {"labels": batch["labels"].numpy()}
            if "label_mask" in batch:
                values["label_mask"] = batch["label_mask"].numpy()

            started = time.perf_counter()
            if predicts_variance:
                prediction, variance = model.predict_with_variance(embeddings)  # (b, c), (b,)
                values["variance"] = variance.float().cpu().numpy()
            else:
                prediction = model(embeddings).logits  # (b, c)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds += time.perf_counter() - started
            outputs = prediction.float().cpu().numpy()  # (b, c)
            values["outputs"] = outputs.reshape(-1) if regression else outputs

            if mc_samples:
                with mc_dropout(model):
                    passes = torch.stack([model(embeddings).logits.float() for _ in range(mc_samples)])  # (T, b, c)
                if regression:
                    values["mc_outputs"] = passes.mean(dim=0).reshape(-1).cpu().numpy()  # (b,)
                    values["mc_variance"] = passes.var(dim=0, unbiased=False).reshape(-1).cpu().numpy()  # (b,)
                else:
                    values["mc_outputs"] = _probabilities(passes, task_kind).mean(dim=0).cpu().numpy()  # (b, c)
            accumulated.append(values)
        arrays = accumulated.finish()
    predictions = SplitPredictions(
        **arrays,
        **{name: np.asarray(getattr(dataset, name), dtype=str) for name in ("observation_ids", "protein_ids", "cluster_ids") if getattr(dataset, name, None) is not None},
    )
    return predictions, seconds


@torch.no_grad()
def trunk_features(
    model: nn.Module, dataset: Dataset, collator: object, *, batch_size: int, device: torch.device
) -> Tensor:
    """The trunk output, the space the output layer and the Gaussian process read, `(n, p)`."""
    model.to(device).eval()
    blocks = [model.trunk_features(batch["embeddings"].to(device)) for batch in _loader(dataset, collator, batch_size)]
    return torch.cat(blocks).float()  # (n, p)


@torch.no_grad()
def nearest_neighbor_distance(queries: Tensor, reference: Tensor) -> np.ndarray:
    """Each query's Euclidean distance to its nearest reference row, in blocks."""
    # queries: (n, p); reference: (r, p)
    distances = []
    for start in range(0, queries.shape[0], QUERY_BLOCK):
        block = queries[start:start + QUERY_BLOCK]  # (q, p)
        nearest = torch.full((block.shape[0],), float("inf"), device=block.device)  # (q,)
        for reference_start in range(0, reference.shape[0], REFERENCE_BLOCK):
            candidates = reference[reference_start:reference_start + REFERENCE_BLOCK]  # (r_block, p)
            nearest = torch.minimum(nearest, torch.cdist(block, candidates).amin(dim=1))  # (q,)
        distances.append(nearest.cpu().numpy())
    return np.concatenate(distances)  # (n,)


def save_seed_predictions(path: str, predictions: Mapping[str, SplitPredictions]) -> None:
    """Write every split's arrays to one `.npz`, atomically."""
    arrays = {
        f"{split}__{field.name}": value
        for split, split_predictions in predictions.items()
        for field in fields(SplitPredictions)
        if (value := getattr(split_predictions, field.name)) is not None
    }
    temporary = f"{path}.partial"
    with open(temporary, "wb") as handle:
        write_prediction_archive(handle, arrays)
    os.replace(temporary, path)


def load_seed_predictions(path: str, *, mmap_mode: str | None = None) -> dict[str, SplitPredictions]:
    if mmap_mode not in {None, "r"}:
        raise ValueError("prediction mmap_mode must be None or r")
    if mmap_mode == "r":
        arrays = mapped_prediction_archive(path)
    else:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}

    predictions = {}
    for split in SPLITS:
        values = {
            field.name: arrays[f"{split}__{field.name}"]
            for field in fields(SplitPredictions)
            if f"{split}__{field.name}" in arrays
        }
        if values:
            predictions[split] = SplitPredictions(**values)
    return predictions
