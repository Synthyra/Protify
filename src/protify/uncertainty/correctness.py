"""The correctness model a package ships: the probability that each answer is right.

Logan decided on 2026-09-26 that every Oracle package ships one and that it reads latent signals,
not MMseqs identity (`docs/decisions/tooling/property_lowcode/2026-09-26_m8_without_mmseqs.md` in
the workspace). The model is `compact` of the latent signals experiment
(`experiments/tooling/property_lowcode/2026-09-26_latent_uncertainty_signals`): a standardized
logistic regression, C = 1, on the logit of the head's calibrated confidence and these signals:

- **Gaussian**, of the head's trunk (`latent_geometry.TrunkGaussians`): `maha_global`, the squared
  Mahalanobis distance to the reference mean; for single-label tasks also `maha_answer`, to the
  answered class's mean under the shared covariance, `rmd`, their difference (Ren et al. 2021),
  and `ddu`, the log density under the class-conditional Gaussians (Mukhoti et al. 2023).
- **Neighbors**, from the `NEIGHBORS` nearest reference rows in three spaces: `trunk`, the head's
  trunk, by Euclidean distance; `inpca`, the clamped standardized input whitened on its top
  `PCA_RANK` principal components, by Euclidean distance; and `saebin`, which SAE features are
  active, by Tanimoto distance. In each space: `d1` and `dk`, the nearest distance and the mean of
  the `NEAREST` nearest; `agree`, the share of the `NEAREST` nearest whose label is the answer
  (for multilabel, whose whole label set is); `trust`, log distance to the nearest row labeled
  otherwise minus log distance to the nearest labeled as answered, within the `NEIGHBORS` nearest
  (Jiang et al. 2018); and for multilabel `jacc`, the mean Jaccard overlap of the `NEAREST`
  nearest label sets with the answer.

Distances are divided by the median nearest distance of the head's own V_fit rows, and the
Mahalanobis distances by V_fit's median `maha_global`, so a fold head reads on the served head's
scale.

**Where it is fitted.** On out-of-fold rows: each fold head of a cross-fit plan (`crossfit.py`)
reads a sample of its held-out rows against a reference drawn from its own training rows, so the
model learns how the signals relate to errors on protein families the head never trained on. Fitted
on the served head's V_fit rows instead, it lost most of its gain.

**The reference** a head reads is at most `REFERENCE_ROWS` of its training rows, every row where
there are fewer; each validated cell had at most 20,000. A larger training set is sampled at
random, keeping one row of every class where it can, and a fold head's reference keeps the same
share of its own training rows. The out-of-fold rows are capped at `OUT_OF_FOLD_ROWS` a level.

Every distance and projection here runs in full float32, whatever Protify's TF32 setting: TF32
keeps about three significant digits of a matrix product.

Shapes: `n` rows, `r` reference rows, `p` trunk width, `d` input width, `s` SAE columns,
`q` PCA rank, `c` classes or labels, `k` neighbors, `f` features.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
import torch

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from scipy.special import expit, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from foundry.eval.temperature import fit_temperature
from .answering import answer_confidence, answer_correct, decode
from .latent_geometry import TrunkGaussians, nearest_rows, nearest_tanimoto
from .methods import probabilities
from .predictions import collect_predictions


SCHEME = "latent_correctness_v1"
# The cross-fit level whose out-of-fold rows the model is fitted on, as M8 was.
LEVEL = "0.3"
REFERENCE_ROWS = 20_000
OUT_OF_FOLD_ROWS = 50_000
NEIGHBORS = 256
NEAREST = 10
PCA_RANK = 256
# An SAE feature is active when its standardized value exceeds the lowest a reference row takes.
SAE_MARGIN = 1e-3
EPSILON = 1e-6
REGULARIZATION = 1.0
SINGLE_LABEL = ("binary", "multiclass")
SPACES = ("trunk", "saebin", "inpca")
# Signals read on a log scale.
LOGGED = ("maha_global", "maha_answer", *(f"{space}_{name}" for space in SPACES for name in ("d1", "dk")))
REFERENCE_FILE = "reference.npz"
MODEL_FILE = "correctness.json"


@contextmanager
def full_float32() -> Iterator[None]:
    """Full float32 matrix products on CUDA inside the block, TF32 as it was afterwards."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.set_float32_matmul_precision(precision)


def reference_sample(labels: np.ndarray, kind: str, limit: int, seed: int) -> np.ndarray:
    """At most `limit` row indices, ascending: every row when there are no more, else a seeded sample
    that keeps the first row of every class in a random order where `limit` allows."""
    # labels: (n,) or (n, c)
    rows = labels.shape[0]
    if rows <= limit:
        return np.arange(rows)  # (rows,)
    order = np.random.default_rng(seed).permutation(rows)  # (rows,)
    if kind not in SINGLE_LABEL:
        return np.sort(order[:limit])  # (m,), m <= limit
    _, first = np.unique(np.asarray(labels)[order], return_index=True)  # each class's first position in `order`
    chosen = np.zeros(rows, dtype=bool)  # by position in `order`
    chosen[np.sort(first)[:limit]] = True
    chosen[np.flatnonzero(~chosen)[:limit - int(chosen.sum())]] = True
    return np.sort(order[chosen])  # (m,), m <= limit


@torch.no_grad()
def head_features(model: torch.nn.Module, batches: Iterable[Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    """Each standardized row's trunk and clamped input, `(n, p)` and `(n, d)`, float32 on `device`.

    The trunk runs as the head predicts, under Protify's precision settings.
    """
    # batches: (b, d) per batch
    model.to(device).eval()
    trunks, inputs = [], []
    for rows in batches:
        rows = rows.to(device)  # (b, d)
        trunks.append(model.trunk_features(rows).float())  # (b, p)
        clipped = rows if model.input_clip is None else rows.clamp(-model.input_clip, model.input_clip)
        inputs.append(clipped.float())  # (b, d)
    return torch.cat(trunks), torch.cat(inputs)  # (n, p), (n, d)


def dataset_batches(dataset: torch.utils.data.Dataset, collator: object, batch_size: int) -> Iterator[Tensor]:
    """A dataset's standardized rows, a batch at a time, as its collator builds them."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    for batch in loader:
        yield batch["embeddings"]  # (b, d)


@dataclass(frozen=True)
class LatentRows:
    """Rows in the three spaces the neighbors are searched in."""

    trunk: Tensor  # (n, p)
    inpca: Tensor  # (n, q)
    bits: Tensor | None  # (n, s) bool, None without an SAE branch


@dataclass(frozen=True)
class LatentSpace:
    """The input's derived spaces, fitted on a head's reference rows: PCA-whitened, and SAE features active."""

    mean: Tensor  # (d,)
    components: Tensor  # (d, q)
    scale: Tensor  # (q,)
    sae_columns: Tensor  # (s,) int64 columns of the input
    sae_floor: Tensor  # (s,)

    @classmethod
    @torch.no_grad()
    def fit(cls, inputs: Tensor, sae_columns: Sequence[int]) -> LatentSpace:
        # inputs: (r, d), clamped and standardized
        with full_float32():
            mean = inputs.mean(dim=0)  # (d,)
            rank = min(PCA_RANK, *inputs.shape)
            # pca_lowrank starts from a random projection, seeded here without moving the caller's generator.
            devices = [inputs.device] if inputs.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(0)
                _, singular, components = torch.pca_lowrank(inputs - mean, q=rank, center=False, niter=4)  # (q,), (d, q)
            scale = singular / max(inputs.shape[0] - 1, 1) ** 0.5  # (q,)
        columns = torch.as_tensor(np.asarray(sae_columns, dtype=np.int64), device=inputs.device)  # (s,)
        # A raw SAE code of 0 standardizes to its feature's lowest value, which reference rows reach.
        floor = inputs[:, columns].amin(dim=0) if columns.numel() else torch.zeros(0, device=inputs.device)  # (s,)
        return cls(mean, components, scale, columns, floor)

    @torch.no_grad()
    def rows(self, trunk: Tensor, inputs: Tensor) -> LatentRows:
        # trunk: (n, p); inputs: (n, d)
        with full_float32():
            inpca = ((inputs - self.mean) @ self.components) / self.scale.clamp_min(1e-6)  # (n, q)
        bits = inputs[:, self.sae_columns] > self.sae_floor + SAE_MARGIN if self.sae_columns.numel() else None  # (n, s)
        return LatentRows(trunk.float(), inpca, bits)


class _AnswerMatch:
    """Whether a reference row's label is a row's answer: its class, or its whole label set."""

    def __init__(self, kind: str, reference_labels: np.ndarray) -> None:
        # reference_labels: (r,) or (r, c)
        self.kind = kind
        if kind in SINGLE_LABEL:
            self.reference = np.asarray(reference_labels, dtype=np.int64)  # (r,)
            return
        self.sets = np.asarray(reference_labels) > 0.5  # (r, c)
        packed = np.packbits(self.sets, axis=1)  # (r, c / 8)
        unique, self.reference = np.unique(packed, axis=0, return_inverse=True)  # (u, c / 8), (r,)
        self.reference = self.reference.reshape(-1)
        self.set_ids = {row.tobytes(): index for index, row in enumerate(unique)}

    def answer_ids(self, answers: np.ndarray) -> np.ndarray:
        # answers: (n,) or (n, c)
        if self.kind in SINGLE_LABEL:
            return np.asarray(answers, dtype=np.int64)  # (n,)
        packed = np.packbits(np.asarray(answers, dtype=bool), axis=1)  # (n, c / 8)
        return np.array([self.set_ids.get(row.tobytes(), -1) for row in packed], dtype=np.int64)  # (n,)


def neighbor_signals(
    index: np.ndarray, distance: np.ndarray, match: _AnswerMatch, answers: np.ndarray, scale: float
) -> dict[str, np.ndarray]:
    """`d1`, `dk`, `agree`, `trust`, and for multilabel `jacc`, from each row's nearest reference rows."""
    # index, distance: (n, k), nearest first
    # answers: (n,) or (n, c)
    distance = distance.astype(np.float64)
    same = match.reference[index] == match.answer_ids(answers)[:, None]  # (n, k)
    signals = {
        "d1": distance[:, 0] / scale,
        "dk": distance[:, :NEAREST].mean(axis=1) / scale,
        "agree": same[:, :NEAREST].mean(axis=1),
    }
    if match.kind == "multilabel":
        chosen = np.asarray(answers, dtype=bool)  # (n, c)
        overlaps = []
        for start in range(0, index.shape[0], 256):
            neighbor_sets = match.sets[index[start:start + 256, :NEAREST]]  # (b, NEAREST, c)
            answer_sets = chosen[start:start + 256, None, :]  # (b, 1, c)
            intersection = (neighbor_sets & answer_sets).sum(axis=2)  # (b, NEAREST)
            union = np.maximum((neighbor_sets | answer_sets).sum(axis=2), 1)  # (b, NEAREST)
            overlaps.append((intersection / union).mean(axis=1))
        signals["jacc"] = np.concatenate(overlaps)
    last = distance[:, -1]
    nearest_same = np.where(same.any(axis=1), np.where(same, distance, np.inf).min(axis=1), last)  # (n,)
    nearest_other = np.where((~same).any(axis=1), np.where(~same, distance, np.inf).min(axis=1), last)  # (n,)
    signals["trust"] = np.log(nearest_other + 1e-6) - np.log(nearest_same + 1e-6)
    return signals  # each (n,) by signal name


def feature_names(kind: str, has_sae: bool) -> list[str]:
    """The model's signals, in its column order after the confidence."""
    names = ["maha_global", "maha_answer", "rmd", "ddu"] if kind in SINGLE_LABEL else ["maha_global"]
    for space in SPACES:
        if space == "saebin" and not has_sae:
            continue
        names += [f"{space}_{name}" for name in ("d1", "dk", "agree", "trust")]
        names += [f"{space}_jacc"] if kind == "multilabel" else []
    return names


def feature_matrix(confidence: np.ndarray, signals: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    """The logit of the confidence, then each signal, distances on a log scale, `(n, 1 + f)`."""
    # confidence: (n,), signals: (n,) per name
    clipped = np.clip(np.asarray(confidence, dtype=np.float64), EPSILON, 1.0 - EPSILON)  # (n,)
    columns = [np.log(clipped / (1.0 - clipped))]
    for name in names:
        value = np.asarray(signals[name], dtype=np.float64)  # (n,)
        columns.append(np.log(np.maximum(value, EPSILON)) if name in LOGGED else value)
    return np.nan_to_num(np.stack(columns, axis=1), nan=0.0, posinf=50.0, neginf=-50.0)  # (n, 1 + f)


class LatentReference:
    """One head's reference rows, their spaces, Gaussians, and V_fit scales, from which any row's signals follow."""

    def __init__(
        self,
        kind: str,
        num_labels: int,
        space: LatentSpace,
        rows: LatentRows,
        labels: np.ndarray,
        scales: Mapping[str, float] | None = None,
    ) -> None:
        # labels: (r,) or (r, c)
        self.kind = kind
        self.num_labels = num_labels
        self.space = space
        self.rows = rows
        self.labels = np.asarray(labels)
        self.match = _AnswerMatch(kind, self.labels)
        single = kind in SINGLE_LABEL
        with full_float32():
            self.gaussians = TrunkGaussians(rows.trunk, self.labels if single else None, num_labels if single else None)
        self.scales = dict(scales) if scales is not None else {}

    @property
    def has_sae(self) -> bool:
        return self.rows.bits is not None

    @classmethod
    def fit(
        cls,
        kind: str,
        num_labels: int,
        trunk: Tensor,
        inputs: Tensor,
        labels: np.ndarray,
        sae_columns: Sequence[int],
        v_fit_trunk: Tensor,
        v_fit_inputs: Tensor,
    ) -> LatentReference:
        """Fit the spaces and Gaussians on reference rows, then the scales on the head's V_fit rows."""
        # trunk: (r, p); inputs: (r, d); v_fit_trunk: (v, p); v_fit_inputs: (v, d)
        # labels: (r,) or (r, c)
        space = LatentSpace.fit(inputs, sae_columns)
        reference = cls(kind, num_labels, space, space.rows(trunk, inputs), labels)
        v_fit = space.rows(v_fit_trunk, v_fit_inputs)
        for name, (_, distance) in reference.search(v_fit, 1).items():
            reference.scales[name] = max(float(np.median(distance[:, 0])), 1e-6)
        reference.scales["maha"] = float(np.median(reference.gaussians.scores(v_fit.trunk)["maha_global"]))
        return reference

    def search(self, queries: LatentRows, k: int = NEIGHBORS) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Each query's nearest reference rows in every space, indices and distances, `(n, k)` each."""
        with full_float32():
            found = {
                "trunk": nearest_rows(queries.trunk, self.rows.trunk, k),
                "inpca": nearest_rows(queries.inpca, self.rows.inpca, k),
            }
            if self.has_sae:
                found["saebin"] = nearest_tanimoto(queries.bits, self.rows.bits, k)
        return found  # (n, k), (n, k) per space, nearest first

    def signals(self, queries: LatentRows, answers: np.ndarray) -> dict[str, np.ndarray]:
        """Every signal of the model for each query row, answered as `answers`."""
        # answers: (n,) or (n, c)
        signals = {}
        for space, (index, distance) in self.search(queries).items():
            for name, value in neighbor_signals(index, distance, self.match, answers, self.scales[space]).items():
                signals[f"{space}_{name}"] = value
        with full_float32():
            gaussian = self.gaussians.scores(queries.trunk)
        maha = max(self.scales["maha"], 1e-12)
        signals["maha_global"] = gaussian["maha_global"].astype(np.float64) / maha
        if self.kind in SINGLE_LABEL:
            answered = np.asarray(answers, dtype=np.int64)  # (n,)
            maha_answer = gaussian["maha_class"].astype(np.float64)[np.arange(answered.size), answered]  # (n,)
            signals["maha_answer"] = maha_answer / maha
            signals["rmd"] = signals["maha_answer"] - signals["maha_global"]
            log_density = np.nan_to_num(gaussian["log_density_class"].astype(np.float64), nan=-np.inf)  # (n, c)
            signals["ddu"] = logsumexp(log_density, axis=1)
        return signals  # each (n,) by signal name

    def arrays(self) -> dict[str, np.ndarray]:
        """What `reference.npz` holds: the reference rows, labels, and spaces."""
        arrays = {
            "trunk": self.rows.trunk.cpu().numpy(),  # (r, p)
            "inpca": self.rows.inpca.cpu().numpy(),  # (r, q)
            "labels": self.labels,
            "pca_mean": self.space.mean.cpu().numpy(),  # (d,)
            "pca_components": self.space.components.cpu().numpy(),  # (d, q)
            "pca_scale": self.space.scale.cpu().numpy(),  # (q,)
            "sae_columns": self.space.sae_columns.cpu().numpy(),  # (s,)
            "sae_floor": self.space.sae_floor.cpu().numpy(),  # (s,)
        }
        if self.has_sae:
            arrays["bits"] = np.packbits(self.rows.bits.cpu().numpy(), axis=1)  # (r, ceil(s / 8))
        return arrays  # arrays by name, such as sae_columns (s,) and bits (r, ceil(s / 8))

    def content_sha256(self) -> str:
        """The SHA-256 of the reference's arrays, which a package pins its novelty claims to.

        Each array's name, dtype, shape, and C-ordered bytes are hashed in name order, so the digest does not move with the
        archive's zip timestamps, and a reference read back from `reference.npz` gives the digest it was written with.
        """
        digest = hashlib.sha256()
        for name, array in sorted(self.arrays().items()):
            array = np.ascontiguousarray(array)
            digest.update(f"{name}|{array.dtype.str}|{array.shape}\0".encode("ascii"))
            digest.update(array.tobytes())
        return digest.hexdigest()

    def save(self, directory: Path) -> None:
        """`reference.npz`: the reference rows, labels, and spaces; the Gaussians are refitted on load."""
        with open(directory / REFERENCE_FILE, "wb") as handle:
            np.savez(handle, **self.arrays())

    @classmethod
    def load(cls, directory: Path, kind: str, num_labels: int, scales: Mapping[str, float], device: torch.device) -> LatentReference:
        with np.load(directory / REFERENCE_FILE, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}

        def tensor(name: str) -> Tensor:
            return torch.as_tensor(arrays[name], device=device)  # (...) as stored under the name, any shape

        space = LatentSpace(tensor("pca_mean"), tensor("pca_components"), tensor("pca_scale"), tensor("sae_columns").long(), tensor("sae_floor"))
        bits = None
        if "bits" in arrays:
            width = arrays["sae_columns"].shape[0]
            bits = torch.as_tensor(np.unpackbits(arrays["bits"], axis=1, count=width).astype(bool), device=device)  # (r, s)
        rows = LatentRows(tensor("trunk"), tensor("inpca"), bits)
        return cls(kind, num_labels, space, rows, arrays["labels"], scales)


@dataclass(frozen=True)
class CorrectnessModel:
    """A standardized logistic regression of whether an answer is right, as plain arrays."""

    names: tuple[str, ...]
    mean: np.ndarray  # (1 + f,)
    scale: np.ndarray  # (1 + f,)
    coefficients: np.ndarray  # (1 + f,)
    intercept: float

    @classmethod
    def fit(cls, features: np.ndarray, correct: np.ndarray, names: Sequence[str]) -> CorrectnessModel:
        # features: (n, 1 + f); correct: (n,) bool
        correct = np.asarray(correct, dtype=bool)
        if correct.all() or not correct.any():
            raise ValueError(
                f"the out-of-fold rows are all {'right' if correct.all() else 'wrong'}, so no correctness model can be "
                f"fitted; a head this certain or this lost has nothing for one to add."
            )
        scaler = StandardScaler().fit(features)
        regression = LogisticRegression(C=REGULARIZATION, max_iter=3000).fit(scaler.transform(features), correct)
        return cls(tuple(names), scaler.mean_.copy(), scaler.scale_.copy(), regression.coef_[0].copy(), float(regression.intercept_[0]))

    def probability(self, features: np.ndarray) -> np.ndarray:
        """Each row's probability of a right answer, `(n,)`."""
        # features: (n, f)
        logits = ((features - self.mean) / self.scale) @ self.coefficients + self.intercept  # (n,)
        return expit(logits)  # (n,)

    def to_record(self) -> dict[str, Any]:
        return {"names": list(self.names), "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coefficients": self.coefficients.tolist(), "intercept": self.intercept}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> CorrectnessModel:
        return cls(tuple(record["names"]), np.asarray(record["mean"]), np.asarray(record["scale"]),
                   np.asarray(record["coefficients"]), float(record["intercept"]))


def calibrated_answers(
    outputs: np.ndarray, labels: np.ndarray, v_fit_outputs: np.ndarray, v_fit_labels: np.ndarray, kind: str, at_least_one_label: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows' answers, confidences, and correctness under a temperature fitted on the head's own V_fit."""
    # outputs: (n, c) or (n,), labels: (n,) or (n, c), v_fit_outputs: (v, c) or (v,), v_fit_labels: (v,) or (v, c)
    temperature = fit_temperature(v_fit_outputs, v_fit_labels, multiclass=kind in SINGLE_LABEL)
    probs = probabilities(np.asarray(outputs, dtype=np.float64) / temperature, kind)  # (n, c)
    answers = decode(probs, kind, at_least_one_label)
    return answers, answer_confidence(probs, answers, kind, at_least_one_label), answer_correct(answers, labels, kind)  # (n,) or (n, c) answers, (n,) confidence, (n,) correct


def out_of_fold_columns(
    model: torch.nn.Module,
    reference: torch.utils.data.Dataset,
    reference_labels: np.ndarray,
    held: torch.utils.data.Dataset,
    v_fit: torch.utils.data.Dataset,
    collator: object,
    *,
    kind: str,
    num_labels: int,
    sae_columns: Sequence[int],
    at_least_one_label: bool,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """A fold head's held-out rows as the model reads them: confidence, correctness, and every signal."""
    # reference_labels: (r,) or (r, c)
    v_fit_predictions, _ = collect_predictions(model, v_fit, collator, task_kind=kind, batch_size=batch_size, device=device)
    held_predictions, _ = collect_predictions(model, held, collator, task_kind=kind, batch_size=batch_size, device=device)
    answers, confidence, correct = calibrated_answers(
        held_predictions.outputs, held_predictions.labels, v_fit_predictions.outputs, v_fit_predictions.labels, kind, at_least_one_label
    )
    fitted = fit_reference(model, reference, reference_labels, v_fit, collator, kind=kind, num_labels=num_labels,
                           sae_columns=sae_columns, batch_size=batch_size, device=device)
    held_trunk, held_inputs = head_features(model, dataset_batches(held, collator, batch_size), device)
    signals = fitted.signals(fitted.space.rows(held_trunk, held_inputs), answers)
    return {"confidence": confidence, "correct": correct, **signals}  # each (n,) by column name


def fit_reference(
    model: torch.nn.Module,
    reference: torch.utils.data.Dataset,
    reference_labels: np.ndarray,
    v_fit: torch.utils.data.Dataset,
    collator: object,
    *,
    kind: str,
    num_labels: int,
    sae_columns: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> LatentReference:
    """A head's reference, fitted on its reference rows and scaled on its V_fit rows."""
    # reference_labels: (r,) or (r, c)
    trunk, inputs = head_features(model, dataset_batches(reference, collator, batch_size), device)
    v_fit_trunk, v_fit_inputs = head_features(model, dataset_batches(v_fit, collator, batch_size), device)
    return LatentReference.fit(kind, num_labels, trunk, inputs, reference_labels, sae_columns, v_fit_trunk, v_fit_inputs)


def model_record(model: CorrectnessModel, reference: LatentReference, fitted_on: Mapping[str, Any]) -> dict[str, Any]:
    """`correctness.json`: the scheme, the model, the reference's scales and size, and what the model was fitted on."""
    return {
        "scheme": SCHEME,
        "task_type": reference.kind,
        "num_labels": reference.num_labels,
        "model": model.to_record(),
        "scales": reference.scales,
        "reference_rows": int(reference.labels.shape[0]),
        "neighbors": NEIGHBORS,
        "nearest": NEAREST,
        "pca_rank": int(reference.space.components.shape[1]),
        "fitted_on": dict(fitted_on),
    }


@dataclass
class ShippedCorrectness:
    """A package's correctness model and reference, loaded to score new rows."""

    model: CorrectnessModel
    reference: LatentReference
    record: Mapping[str, Any]

    @classmethod
    def load(cls, directory: Path, device: torch.device | str = "cpu") -> ShippedCorrectness:
        record = json.loads((directory / MODEL_FILE).read_text(encoding="utf-8"))
        if record.get("scheme") != SCHEME:
            raise ValueError(f"{directory} holds correctness scheme {record.get('scheme')!r}; this Protify reads {SCHEME!r}.")
        reference = LatentReference.load(directory, record["task_type"], record["num_labels"], record["scales"], torch.device(device))
        return cls(CorrectnessModel.from_record(record["model"]), reference, record)

    def signals(self, model: torch.nn.Module, standardized: np.ndarray, answers: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
        """Every signal of the rows the head reads, answered as `answers`."""
        # standardized: (n, d), answers: (n,) or (n, c)
        device = self.reference.rows.trunk.device
        tensor = torch.from_numpy(np.ascontiguousarray(standardized, dtype=np.float32))  # (n, d)
        batches = (tensor[start:start + batch_size] for start in range(0, tensor.shape[0], batch_size))
        trunk, inputs = head_features(model, batches, device)
        return self.reference.signals(self.reference.space.rows(trunk, inputs), answers)  # each (n,) by signal name

    def probability(self, confidence: np.ndarray, signals: Mapping[str, np.ndarray]) -> np.ndarray:
        # confidence: (n,), signals: (n,) per name
        return self.model.probability(feature_matrix(confidence, signals, self.model.names))  # (n,)
