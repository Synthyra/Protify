"""Cluster precomputed embeddings and save reproducible diagnostics."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import (
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    KMeans,
    MiniBatchKMeans,
    OPTICS,
    SpectralClustering,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_samples,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

try:
    from sklearn.cluster import HDBSCAN
except ImportError:  # pragma: no cover - Protify pins a version that provides it.
    HDBSCAN = None

try:
    from data.dataset_classes import EmbeddingStandardizer
    from pooler import Pooler
    from seed_utils import set_global_seed
    from utils import embedding_blob_to_tensor, torch_load
except ImportError:
    from ..data.dataset_classes import EmbeddingStandardizer
    from ..pooler import Pooler
    from ..seed_utils import set_global_seed
    from ..utils import embedding_blob_to_tensor, torch_load


plt.ioff()

DEFAULT_ALGORITHMS = ("kmeans", "birch", "hdbscan")
SUPPORTED_ALGORITHMS = (
    "kmeans",
    "mini_batch_kmeans",
    "agglomerative",
    "birch",
    "spectral",
    "gaussian_mixture",
    "dbscan",
    "optics",
    "hdbscan",
)
SUPPORTED_REDUCTIONS = ("pca", "tsne", "umap")
SUPPORTED_POOLING_TYPES = ("mean", "max", "norm", "median", "std", "var", "cls")


@runtime_checkable
class VectorSource(Protocol):
    """Adapter contract for an external vector database or custom source."""

    def iter_vectors(self) -> Iterable[tuple[str, Any]]:
        """Yield stable ``(record_id, vector_or_matrix)`` pairs."""


@dataclass
class EmbeddingTable:
    """Identifiers and their fixed-width feature matrix."""

    ids: tuple[str, ...]
    vectors: np.ndarray
    source: str | None = None

    def __post_init__(self) -> None:
        self.ids = tuple(str(record_id) for record_id in self.ids)
        self.vectors = np.asarray(self.vectors, dtype=np.float32)
        if self.vectors.ndim != 2:
            raise ValueError(f"Expected a two-dimensional feature matrix, got {self.vectors.shape}.")
        if self.vectors.shape[0] == 0 or self.vectors.shape[1] == 0:
            raise ValueError("Embedding input must contain at least one nonempty vector.")
        if len(self.ids) != self.vectors.shape[0]:
            raise ValueError(
                f"Received {len(self.ids)} IDs for {self.vectors.shape[0]} vectors."
            )
        if any(not record_id for record_id in self.ids):
            raise ValueError("Embedding record IDs cannot be empty.")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Embedding record IDs must be unique.")
        if not np.isfinite(self.vectors).all():
            raise ValueError("Embedding vectors contain NaN or infinite values.")


@dataclass
class ClusteringConfig:
    """Configuration shared by the Python and command-line workflows."""

    output_dir: Path | str = Path("plots/clustering")
    algorithms: tuple[str, ...] = DEFAULT_ALGORITHMS
    reductions: tuple[str, ...] = ("pca",)
    pooling_types: tuple[str, ...] = ("mean",)
    standardize: bool = False
    n_clusters: int = 8
    eps: float = 0.5
    min_samples: int = 5
    min_cluster_size: int = 5
    seed: int = 42
    metric_sample_size: int = 5_000
    plot_sample_size: int = 3_000
    quadratic_algorithm_limit: int = 10_000
    legacy_embedding_dim: int | None = None
    csv_id_column: str | None = None
    diagnostics: bool = True
    dpi: int = 300

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.algorithms = tuple(_normalize_name(name) for name in self.algorithms)
        self.reductions = tuple(_normalize_name(name) for name in self.reductions)
        self.pooling_types = tuple(_normalize_name(name) for name in self.pooling_types)
        if self.algorithms == ("all",):
            self.algorithms = SUPPORTED_ALGORITHMS
        unknown_algorithms = sorted(set(self.algorithms) - set(SUPPORTED_ALGORITHMS))
        if unknown_algorithms:
            raise ValueError(f"Unsupported clustering algorithms: {unknown_algorithms}.")
        unknown_reductions = sorted(set(self.reductions) - set(SUPPORTED_REDUCTIONS))
        if unknown_reductions:
            raise ValueError(f"Unsupported reductions: {unknown_reductions}.")
        if not self.algorithms:
            raise ValueError("At least one clustering algorithm is required.")
        if not self.pooling_types:
            raise ValueError("At least one pooling type is required.")
        if "parti" in self.pooling_types:
            raise ValueError("Stored embeddings do not include the attentions required by parti pooling.")
        unknown_pooling = sorted(set(self.pooling_types) - set(SUPPORTED_POOLING_TYPES))
        if unknown_pooling:
            raise ValueError(f"Unsupported pooling types: {unknown_pooling}.")
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        if self.eps <= 0:
            raise ValueError("eps must be positive.")
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2.")
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2.")
        if self.metric_sample_size < 2:
            raise ValueError("metric_sample_size must be at least 2.")
        if self.plot_sample_size < 2:
            raise ValueError("plot_sample_size must be at least 2.")
        if self.quadratic_algorithm_limit < 2:
            raise ValueError("quadratic_algorithm_limit must be at least 2.")
        if self.dpi <= 0:
            raise ValueError("dpi must be positive.")
        if self.legacy_embedding_dim is not None and self.legacy_embedding_dim <= 0:
            raise ValueError("legacy_embedding_dim must be positive when provided.")


@dataclass
class AlgorithmResult:
    """Fitted estimator, labels, and validation metrics for one algorithm."""

    name: str
    labels: np.ndarray
    estimator: Any
    metrics: dict[str, int | float | str | None]


@dataclass
class ClusteringWorkflowResult:
    """Artifacts produced by :func:`run_clustering_workflow`."""

    table: EmbeddingTable
    transformed_vectors: np.ndarray
    standardizer: EmbeddingStandardizer | None
    algorithms: dict[str, AlgorithmResult]
    errors: dict[str, str]
    output_dir: Path
    plot_paths: tuple[Path, ...] = field(default_factory=tuple)


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("Embeddings must be numeric arrays or tensors.") from error
    if not np.isfinite(array).all():
        raise ValueError("Embedding contains NaN or infinite values.")
    return array


def vectorize_embedding(
    embedding: Any,
    pooling_types: Sequence[str] = ("mean",),
) -> np.ndarray:
    """Convert one vector or per-token matrix into a one-dimensional vector."""

    normalized_pooling = tuple(_normalize_name(name) for name in pooling_types)
    if not normalized_pooling:
        raise ValueError("At least one pooling type is required.")
    unknown = sorted(set(normalized_pooling) - set(SUPPORTED_POOLING_TYPES))
    if unknown:
        raise ValueError(f"Unsupported pooling types: {unknown}.")
    array = _to_numpy(embedding)
    if array.ndim == 1:
        if array.size == 0:
            raise ValueError("Embedding vectors cannot be empty.")
        return array.astype(np.float32, copy=False)
    if array.ndim != 2:
        raise ValueError(
            f"Each embedding must have shape (d,) or (l, d), got {array.shape}."
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("Embedding matrices cannot have empty token or feature axes.")

    pooler = Pooler(list(normalized_pooling))
    tensor = torch.from_numpy(array).unsqueeze(0)  # (b=1, l, d)
    attention_mask = torch.ones(1, array.shape[0], dtype=torch.long)  # (b, l)
    pooled = pooler(tensor, attention_mask=attention_mask).squeeze(0)  # (p * d,)
    vector = pooled.detach().cpu().float().numpy()
    if not np.isfinite(vector).all():
        raise ValueError("Pooling produced NaN or infinite values.")
    return vector


def _records_to_table(
    records: Iterable[tuple[str, Any]],
    pooling_types: Sequence[str],
    source: str | None,
) -> EmbeddingTable:
    record_ids: list[str] = []
    vectors: list[np.ndarray] = []
    for record_id, embedding in records:
        record_ids.append(str(record_id))
        vectors.append(vectorize_embedding(embedding, pooling_types))
    if not vectors:
        raise ValueError("Embedding source did not contain any records.")
    feature_widths = {vector.shape[0] for vector in vectors}
    if len(feature_widths) != 1:
        raise ValueError(
            "Vectorized embeddings have inconsistent feature widths: "
            f"{sorted(feature_widths)}."
        )
    return EmbeddingTable(tuple(record_ids), np.stack(vectors), source=source)


def _array_to_table(
    values: Any,
    ids: Sequence[str] | None,
    pooling_types: Sequence[str],
    source: str | None,
) -> EmbeddingTable:
    if isinstance(ids, (str, bytes, bytearray)):
        raise TypeError("ids must be a sequence of complete record IDs, not a string.")
    array = _to_numpy(values)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim == 2:
        record_ids = tuple(str(value) for value in ids) if ids is not None else tuple(
            str(index) for index in range(array.shape[0])
        )
        return EmbeddingTable(record_ids, array, source=source)
    if array.ndim == 3:
        record_ids = tuple(str(value) for value in ids) if ids is not None else tuple(
            str(index) for index in range(array.shape[0])
        )
        return _records_to_table(
            zip(record_ids, array, strict=True),
            pooling_types,
            source,
        )
    raise ValueError(
        f"Array sources must have shape (n, d) or (n, l, d), got {array.shape}."
    )


def _payload_to_table(
    payload: Any,
    ids: Sequence[str] | None,
    pooling_types: Sequence[str],
    source: str | None,
) -> EmbeddingTable:
    if isinstance(payload, Mapping):
        if "embeddings" in payload and set(payload).issubset({"embeddings", "ids"}):
            payload_ids = payload.get("ids", ids)
            return _array_to_table(payload["embeddings"], payload_ids, pooling_types, source)
        if ids is not None:
            raise ValueError("ids cannot override identifiers in a mapping source.")
        return _records_to_table(payload.items(), pooling_types, source)
    if isinstance(payload, (np.ndarray, torch.Tensor)):
        return _array_to_table(payload, ids, pooling_types, source)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return _array_to_table(payload, ids, pooling_types, source)
    raise TypeError(f"Unsupported embedding payload type: {type(payload).__name__}.")


def _load_npz(
    path: Path,
    ids: Sequence[str] | None,
    pooling_types: Sequence[str],
) -> EmbeddingTable:
    with np.load(path, allow_pickle=False) as archive:
        if "embeddings" in archive:
            values = archive["embeddings"]
        else:
            candidates = [name for name in archive.files if name != "ids"]
            if len(candidates) != 1:
                raise ValueError(
                    "NPZ input must contain an 'embeddings' array or exactly one non-ID array."
                )
            values = archive[candidates[0]]
        archive_ids = archive["ids"].tolist() if "ids" in archive else ids
    return _array_to_table(values, archive_ids, pooling_types, str(path))


def _load_delimited(
    path: Path,
    id_column: str | None,
) -> EmbeddingTable:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"Delimited embedding file is empty: {path}.") from error
        rows = [row for row in reader if row]
    if not rows:
        raise ValueError(f"Delimited embedding file has no data rows: {path}.")
    if any(len(row) != len(header) for row in rows):
        raise ValueError("Delimited embedding rows do not match the header width.")

    if id_column is not None:
        if id_column not in header:
            raise ValueError(f"ID column '{id_column}' is not present in {path}.")
        id_index = header.index(id_column)
    else:
        nonnumeric_columns = []
        for column_index in range(len(header)):
            try:
                for row in rows:
                    float(row[column_index])
            except ValueError:
                nonnumeric_columns.append(column_index)
        if len(nonnumeric_columns) > 1:
            names = [header[index] for index in nonnumeric_columns]
            raise ValueError(
                f"Found multiple nonnumeric columns {names}; select one with csv_id_column."
            )
        id_index = nonnumeric_columns[0] if nonnumeric_columns else None

    feature_indices = [index for index in range(len(header)) if index != id_index]
    if not feature_indices:
        raise ValueError("Delimited embedding input has no numeric feature columns.")
    try:
        vectors = np.asarray(
            [[float(row[index]) for index in feature_indices] for row in rows],
            dtype=np.float32,
        )
    except ValueError as error:
        raise ValueError("All non-ID columns in a delimited embedding file must be numeric.") from error
    record_ids = (
        tuple(row[id_index] for row in rows)
        if id_index is not None
        else tuple(str(index) for index in range(len(rows)))
    )
    return EmbeddingTable(record_ids, vectors, source=str(path))


def _load_sqlite(
    path: Path,
    pooling_types: Sequence[str],
    legacy_embedding_dim: int | None,
) -> EmbeddingTable:
    connection_uri = f"{path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(connection_uri, uri=True, timeout=30) as connection:
        table_row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
        ).fetchone()
        if table_row is None:
            raise ValueError(
                "SQLite input must use Protify's embeddings(sequence, embedding) schema."
            )
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(embeddings)").fetchall()
        }
        if not {"sequence", "embedding"}.issubset(columns):
            raise ValueError(
                "SQLite table 'embeddings' must contain sequence and embedding columns."
            )
        cursor = connection.execute(
            "SELECT sequence, embedding FROM embeddings ORDER BY rowid"
        )
        fallback_shape = (
            (-1, legacy_embedding_dim) if legacy_embedding_dim is not None else None
        )
        records = (
            (
                str(record_id),
                embedding_blob_to_tensor(blob, fallback_shape=fallback_shape),
            )
            for record_id, blob in cursor
        )
        return _records_to_table(records, pooling_types, str(path))


def load_embeddings(
    source: os.PathLike[str] | str | np.ndarray | torch.Tensor | Mapping[str, Any] | VectorSource,
    *,
    ids: Sequence[str] | None = None,
    pooling_types: Sequence[str] = ("mean",),
    legacy_embedding_dim: int | None = None,
    csv_id_column: str | None = None,
) -> EmbeddingTable:
    """Load vectors from memory, standard files, Protify SQLite, or an adapter."""

    if isinstance(ids, (str, bytes, bytearray)):
        raise TypeError("ids must be a sequence of complete record IDs, not a string.")
    if isinstance(source, VectorSource):
        if ids is not None:
            raise ValueError("ids cannot override identifiers from a VectorSource.")
        return _records_to_table(source.iter_vectors(), pooling_types, type(source).__name__)
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Embedding source does not exist: {path}.")
        suffix = path.suffix.lower()
        if suffix in {".pt", ".pth"}:
            return _payload_to_table(torch_load(path), ids, pooling_types, str(path))
        if suffix == ".npy":
            return _array_to_table(
                np.load(path, allow_pickle=False), ids, pooling_types, str(path)
            )
        if suffix == ".npz":
            return _load_npz(path, ids, pooling_types)
        if suffix in {".csv", ".tsv"}:
            if ids is not None:
                raise ValueError("ids cannot override identifiers in delimited files.")
            return _load_delimited(path, csv_id_column)
        if suffix in {".db", ".sqlite", ".sqlite3"}:
            if ids is not None:
                raise ValueError("ids cannot override identifiers in SQLite.")
            return _load_sqlite(path, pooling_types, legacy_embedding_dim)
        raise ValueError(f"Unsupported embedding file extension: {suffix or '<none>'}.")
    return _payload_to_table(source, ids, pooling_types, source=None)


def _cluster_count(labels: np.ndarray) -> int:
    return len(set(labels.tolist()) - {-1})


def _sample_indices(sample_count: int, limit: int, seed: int) -> np.ndarray:
    if sample_count <= limit:
        return np.arange(sample_count)
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(sample_count, size=limit, replace=False))


def evaluate_clustering(
    vectors: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int | None = None,
    seed: int = 42,
) -> dict[str, int | float | str | None]:
    """Compute internal validation metrics while excluding noise label ``-1``."""

    labels = np.asarray(labels, dtype=np.int64)
    full_labels = labels
    full_non_noise = full_labels != -1
    full_cluster_count = _cluster_count(full_labels)
    full_noise_count = int((~full_non_noise).sum())
    if sample_size is not None:
        sampled = _sample_indices(len(labels), sample_size, seed)
        vectors = vectors[sampled]
        labels = labels[sampled]
    non_noise = labels != -1
    filtered_vectors = vectors[non_noise]
    filtered_labels = labels[non_noise]
    score_cluster_count = _cluster_count(labels)
    metrics: dict[str, int | float | str | None] = {
        "n_clusters": full_cluster_count,
        "n_noise": full_noise_count,
        "noise_fraction": full_noise_count / len(full_labels),
        "clustered_fraction": float(full_non_noise.mean()),
        "metric_sample_count": len(labels),
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
        "score_reason": None,
    }
    if score_cluster_count < 2:
        metrics["score_reason"] = "fewer than two non-noise clusters"
        return metrics
    if filtered_vectors.shape[0] <= score_cluster_count:
        metrics["score_reason"] = "not enough non-noise samples for validation metrics"
        return metrics
    metrics["silhouette"] = float(silhouette_score(filtered_vectors, filtered_labels))
    metrics["calinski_harabasz"] = float(
        calinski_harabasz_score(filtered_vectors, filtered_labels)
    )
    metrics["davies_bouldin"] = float(
        davies_bouldin_score(filtered_vectors, filtered_labels)
    )
    return metrics


def fit_clusterer(
    name: str,
    vectors: np.ndarray,
    config: ClusteringConfig,
) -> AlgorithmResult:
    """Fit one supported clusterer and return its labels and metrics."""

    name = _normalize_name(name)
    if name not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported clustering algorithm: {name}.")
    n_samples = vectors.shape[0]
    fixed_cluster_algorithms = {
        "kmeans",
        "mini_batch_kmeans",
        "agglomerative",
        "birch",
        "spectral",
        "gaussian_mixture",
    }
    if name in fixed_cluster_algorithms and config.n_clusters > n_samples:
        raise ValueError(
            f"n_clusters={config.n_clusters} exceeds n_samples={n_samples} for {name}."
        )
    quadratic_algorithms = {"agglomerative", "spectral", "dbscan"}
    if name in quadratic_algorithms and n_samples > config.quadratic_algorithm_limit:
        raise ValueError(
            f"{name} is disabled above {config.quadratic_algorithm_limit} samples because "
            "its worst-case memory or runtime is quadratic. Use MiniBatchKMeans, Birch, "
            "OPTICS, or HDBSCAN, or raise quadratic_algorithm_limit explicitly."
        )

    if name == "kmeans":
        estimator = KMeans(
            n_clusters=config.n_clusters,
            n_init=10,
            random_state=config.seed,
        )
    elif name == "mini_batch_kmeans":
        estimator = MiniBatchKMeans(
            n_clusters=config.n_clusters,
            n_init=10,
            random_state=config.seed,
        )
    elif name == "agglomerative":
        estimator = AgglomerativeClustering(
            n_clusters=config.n_clusters,
            compute_distances=True,
        )
    elif name == "birch":
        estimator = Birch(n_clusters=config.n_clusters)
    elif name == "spectral":
        if n_samples < 3:
            raise ValueError("Spectral clustering requires at least three samples.")
        estimator = SpectralClustering(
            n_clusters=config.n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=min(10, n_samples - 1),
            assign_labels="kmeans",
            random_state=config.seed,
        )
    elif name == "gaussian_mixture":
        estimator = GaussianMixture(
            n_components=config.n_clusters,
            random_state=config.seed,
        )
    elif name == "dbscan":
        estimator = DBSCAN(eps=config.eps, min_samples=config.min_samples)
    elif name == "optics":
        estimator = OPTICS(min_samples=config.min_samples)
    else:
        if HDBSCAN is None:
            raise RuntimeError("hdbscan requires scikit-learn 1.3 or newer.")
        estimator = HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
        )

    labels = np.asarray(estimator.fit_predict(vectors), dtype=np.int64)
    return AlgorithmResult(
        name=name,
        labels=labels,
        estimator=estimator,
        metrics=evaluate_clustering(
            vectors,
            labels,
            sample_size=config.metric_sample_size,
            seed=config.seed,
        ),
    )


def _reduce_vectors(
    vectors: np.ndarray,
    method: str,
    seed: int,
) -> np.ndarray:
    method = _normalize_name(method)
    if method == "pca":
        if vectors.shape[1] == 1:
            return np.column_stack([vectors[:, 0], np.zeros(vectors.shape[0])])
        if np.unique(vectors, axis=0).shape[0] == 1:
            return np.zeros((vectors.shape[0], 2), dtype=np.float64)
        return PCA(n_components=2, random_state=seed).fit_transform(vectors)
    if method == "tsne":
        if vectors.shape[0] < 3:
            raise ValueError("t-SNE requires at least three samples.")
        perplexity = min(30.0, max(1.0, (vectors.shape[0] - 1) / 3.0))
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(vectors)
    if method == "umap":
        try:
            import umap
        except ImportError as error:
            raise RuntimeError(
                "UMAP plotting requires the optional umap-learn package."
            ) from error
        if vectors.shape[0] < 3:
            raise ValueError("UMAP requires at least three samples.")
        return umap.UMAP(
            n_components=2,
            n_neighbors=min(15, vectors.shape[0] - 1),
            min_dist=0.1,
            random_state=seed,
        ).fit_transform(vectors)
    raise ValueError(f"Unsupported reduction method: {method}.")


def _cluster_labels(labels: np.ndarray) -> list[int]:
    return sorted(set(labels.tolist()), key=lambda label: (label == -1, label))


def _scatter_on_axis(
    axis: Any,
    coordinates: np.ndarray,
    labels: np.ndarray,
    title: str,
) -> None:
    unique_labels = _cluster_labels(labels)
    color_map = plt.get_cmap("tab20", max(1, len(unique_labels)))
    for index, label in enumerate(unique_labels):
        selected = labels == label
        axis.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=22,
            alpha=0.75,
            color="black" if label == -1 else color_map(index),
            marker="x" if label == -1 else "o",
            label="noise" if label == -1 else f"cluster {label}",
        )
    axis.set_title(title)
    axis.set_xlabel("component 1")
    axis.set_ylabel("component 2")
    if len(unique_labels) <= 20:
        axis.legend(loc="best", fontsize=7)


def _save_figure(figure: Any, path: Path, dpi: int) -> Path:
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_scatter(
    coordinates: np.ndarray,
    labels: np.ndarray,
    algorithm: str,
    reduction: str,
    output_dir: Path,
    dpi: int,
) -> Path:
    figure, axis = plt.subplots(figsize=(8, 6))
    _scatter_on_axis(axis, coordinates, labels, f"{algorithm}: {reduction.upper()}")
    return _save_figure(figure, output_dir / f"{algorithm}_{reduction}.png", dpi)


def _plot_cluster_sizes(
    labels: np.ndarray,
    algorithm: str,
    output_dir: Path,
    dpi: int,
) -> Path:
    unique_labels, counts = np.unique(labels, return_counts=True)
    names = ["noise" if label == -1 else str(label) for label in unique_labels]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(names, counts, color="#2f6f9f")
    axis.set_title(f"{algorithm}: cluster sizes")
    axis.set_xlabel("cluster")
    axis.set_ylabel("samples")
    return _save_figure(figure, output_dir / f"{algorithm}_cluster_sizes.png", dpi)


def _plot_silhouette(
    vectors: np.ndarray,
    labels: np.ndarray,
    algorithm: str,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    non_noise = labels != -1
    filtered_vectors = vectors[non_noise]
    filtered_labels = labels[non_noise]
    unique_labels = np.unique(filtered_labels)
    if len(unique_labels) < 2 or filtered_vectors.shape[0] <= len(unique_labels):
        return None
    sample_scores = silhouette_samples(filtered_vectors, filtered_labels)
    overall_score = silhouette_score(filtered_vectors, filtered_labels)
    figure, axis = plt.subplots(figsize=(8, 6))
    y_lower = 10
    color_map = plt.get_cmap("tab20", len(unique_labels))
    for index, label in enumerate(unique_labels):
        values = np.sort(sample_scores[filtered_labels == label])
        y_upper = y_lower + len(values)
        axis.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            values,
            color=color_map(index),
            alpha=0.75,
        )
        axis.text(-0.05, y_lower + 0.5 * len(values), str(label))
        y_lower = y_upper + 10
    axis.axvline(overall_score, color="red", linestyle="--", label=f"mean={overall_score:.3f}")
    axis.set_title(f"{algorithm}: silhouette distribution")
    axis.set_xlabel("silhouette coefficient")
    axis.set_ylabel("clustered samples")
    axis.legend()
    return _save_figure(figure, output_dir / f"{algorithm}_silhouette.png", dpi)


def _plot_centroid_distances(
    vectors: np.ndarray,
    labels: np.ndarray,
    algorithm: str,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    unique_labels = np.asarray([label for label in np.unique(labels) if label != -1])
    if len(unique_labels) < 2:
        return None
    if len(unique_labels) > 100:
        raise ValueError("Centroid-distance plots are limited to 100 clusters.")
    centroids = np.stack([vectors[labels == label].mean(axis=0) for label in unique_labels])
    distances = pairwise_distances(centroids)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(distances, cmap="viridis")
    axis.set_xticks(range(len(unique_labels)), unique_labels)
    axis.set_yticks(range(len(unique_labels)), unique_labels)
    axis.set_title(f"{algorithm}: centroid distances")
    axis.set_xlabel("cluster")
    axis.set_ylabel("cluster")
    figure.colorbar(image, ax=axis, label="Euclidean distance")
    return _save_figure(
        figure,
        output_dir / f"{algorithm}_centroid_distances.png",
        dpi,
    )


def _plot_metric_comparison(
    results: Mapping[str, AlgorithmResult],
    output_dir: Path,
    dpi: int,
) -> Path:
    metric_specs = (
        ("silhouette", "silhouette (higher is better)"),
        ("calinski_harabasz", "Calinski-Harabasz (higher is better)"),
        ("davies_bouldin", "Davies-Bouldin (lower is better)"),
        ("clustered_fraction", "clustered fraction"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    algorithms = list(results)
    for axis, (metric_name, title) in zip(axes.flat, metric_specs, strict=True):
        values = [
            float(results[name].metrics[metric_name])
            if results[name].metrics[metric_name] is not None
            else math.nan
            for name in algorithms
        ]
        axis.bar(algorithms, values, color="#4c956c")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=45)
    figure.suptitle("Clustering metric comparison (validity scores exclude noise)")
    return _save_figure(figure, output_dir / "algorithm_metrics.png", dpi)


def _plot_algorithm_overview(
    coordinates: np.ndarray,
    results: Mapping[str, AlgorithmResult],
    sample_indices: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> Path:
    columns = min(3, len(results))
    rows = math.ceil(len(results) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5 * rows), squeeze=False)
    for axis, result in zip(axes.flat, results.values()):
        _scatter_on_axis(axis, coordinates, result.labels[sample_indices], result.name)
    for axis in list(axes.flat)[len(results):]:
        axis.axis("off")
    figure.suptitle("PCA overview across clustering algorithms")
    return _save_figure(figure, output_dir / "algorithm_overview_pca.png", dpi)


def _plot_kmeans_diagnostics(
    vectors: np.ndarray,
    config: ClusteringConfig,
) -> Path | None:
    distinct_vector_count = np.unique(vectors, axis=0).shape[0]
    if distinct_vector_count < 2:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.text(
            0.5,
            0.5,
            "KMeans model selection requires at least two distinct vectors.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        return _save_figure(
            figure,
            config.output_dir / "kmeans_model_selection.png",
            config.dpi,
        )

    max_k = min(10, vectors.shape[0] - 1, distinct_vector_count)
    if max_k < 2:
        return None
    cluster_counts = list(range(2, max_k + 1))
    inertias: list[float] = []
    silhouettes: list[float] = []
    for cluster_count in cluster_counts:
        estimator = KMeans(
            n_clusters=cluster_count,
            n_init=10,
            random_state=config.seed,
        )
        labels = estimator.fit_predict(vectors)
        inertias.append(float(estimator.inertia_))
        if len(np.unique(labels)) < 2:
            silhouettes.append(math.nan)
        else:
            silhouettes.append(float(silhouette_score(vectors, labels)))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(cluster_counts, inertias, marker="o")
    axes[0].set_title("KMeans elbow")
    axes[0].set_xlabel("clusters")
    axes[0].set_ylabel("inertia")
    axes[1].plot(cluster_counts, silhouettes, marker="o")
    axes[1].set_title("KMeans silhouette by k")
    axes[1].set_xlabel("clusters")
    axes[1].set_ylabel("silhouette")
    return _save_figure(
        figure,
        config.output_dir / "kmeans_model_selection.png",
        config.dpi,
    )


def _plot_density_diagnostics(
    vectors: np.ndarray,
    config: ClusteringConfig,
) -> Path | None:
    neighbor_count = min(config.min_samples, vectors.shape[0])
    if neighbor_count < 2:
        return None
    distances, _ = NearestNeighbors(n_neighbors=neighbor_count).fit(vectors).kneighbors(vectors)
    k_distances = np.sort(distances[:, -1])
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(k_distances)
    axis.axhline(config.eps, color="red", linestyle="--", label=f"eps={config.eps:g}")
    axis.set_title(f"DBSCAN {neighbor_count}-nearest-neighbor distances")
    axis.set_xlabel("samples sorted by distance")
    axis.set_ylabel("distance")
    axis.legend()
    return _save_figure(
        figure,
        config.output_dir / "dbscan_k_distance.png",
        config.dpi,
    )


def _plot_optics_reachability(
    result: AlgorithmResult,
    config: ClusteringConfig,
) -> Path | None:
    reachability = getattr(result.estimator, "reachability_", None)
    ordering = getattr(result.estimator, "ordering_", None)
    if reachability is None or ordering is None:
        return None
    ordered = reachability[ordering]
    finite = np.isfinite(ordered)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(np.arange(len(ordered))[finite], ordered[finite], color="#2f6f9f")
    axis.set_title("OPTICS reachability")
    axis.set_xlabel("ordered samples")
    axis.set_ylabel("reachability distance")
    return _save_figure(
        figure,
        config.output_dir / "optics_reachability.png",
        config.dpi,
    )


def _plot_agglomerative_dendrogram(
    result: AlgorithmResult,
    config: ClusteringConfig,
) -> Path | None:
    children = getattr(result.estimator, "children_", None)
    distances = getattr(result.estimator, "distances_", None)
    if children is None or distances is None:
        return None
    sample_count = len(result.labels)
    counts = np.zeros(children.shape[0])
    for index, merge in enumerate(children):
        count = 0
        for child_index in merge:
            count += 1 if child_index < sample_count else counts[child_index - sample_count]
        counts[index] = count
    linkage_matrix = np.column_stack([children, distances, counts]).astype(float)
    figure, axis = plt.subplots(figsize=(10, 5))
    dendrogram(linkage_matrix, truncate_mode="level", p=5, ax=axis)
    axis.set_title("Agglomerative hierarchy")
    axis.set_xlabel("sample or merged node")
    axis.set_ylabel("distance")
    return _save_figure(
        figure,
        config.output_dir / "agglomerative_dendrogram.png",
        config.dpi,
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _source_metadata(source: Any, table: EmbeddingTable) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "description": table.source or type(source).__name__,
        "records": len(table.ids),
        "features": table.vectors.shape[1],
    }
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if path.is_file():
            stat = path.stat()
            metadata.update(
                {
                    "path": str(path.resolve()),
                    "size_bytes": stat.st_size,
                    "modified_time_ns": stat.st_mtime_ns,
                }
            )
    return metadata


def _clear_previous_manifest_artifacts(output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    output_root = output_dir.resolve()
    for relative_path in manifest.get("artifacts", []):
        candidate = (output_dir / relative_path).resolve()
        if candidate == output_root or not candidate.is_relative_to(output_root):
            continue
        if candidate.is_file():
            candidate.unlink()


def _write_outputs(
    source: Any,
    config: ClusteringConfig,
    table: EmbeddingTable,
    vectors: np.ndarray,
    standardizer: EmbeddingStandardizer | None,
    results: Mapping[str, AlgorithmResult],
    errors: Mapping[str, str],
    plot_paths: Sequence[Path],
) -> None:
    np.savez_compressed(
        config.output_dir / "vectors.npz",
        ids=np.asarray(table.ids, dtype=str),
        embeddings=vectors,
    )
    if standardizer is not None:
        np.savez_compressed(
            config.output_dir / "standardizer.npz",
            mean=standardizer.mean.numpy(),
            scale=standardizer.scale.numpy(),
        )
    with (config.output_dir / "assignments.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", *results.keys()])
        for index, record_id in enumerate(table.ids):
            writer.writerow(
                [record_id, *(int(result.labels[index]) for result in results.values())]
            )

    metric_fields = (
        "algorithm",
        "n_clusters",
        "n_noise",
        "noise_fraction",
        "clustered_fraction",
        "metric_sample_count",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "score_reason",
    )
    with (config.output_dir / "metrics.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=metric_fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for name, result in results.items():
            writer.writerow({"algorithm": name, **result.metrics})

    configuration = asdict(config)
    configuration["output_dir"] = str(config.output_dir)
    configuration["source"] = table.source or type(source).__name__
    with (config.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(configuration, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (config.output_dir / "errors.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(errors), handle, indent=2, sort_keys=True)
        handle.write("\n")

    artifact_names = [
        "assignments.tsv",
        "metrics.tsv",
        "vectors.npz",
        "config.json",
        "errors.json",
        *(("standardizer.npz",) if standardizer is not None else ()),
    ]
    artifact_names.extend(
        str(path.resolve().relative_to(config.output_dir.resolve()))
        for path in plot_paths
        if path.is_file() and path.resolve().is_relative_to(config.output_dir.resolve())
    )
    manifest = {
        "artifacts": sorted(set([*artifact_names, "manifest.json"])),
        "source": _source_metadata(source, table),
        "software": {
            "numpy": np.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "torch": torch.__version__,
        },
        "estimators": {
            name: _json_safe(result.estimator.get_params(deep=False))
            for name, result in results.items()
        },
    }
    with (config.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_clustering_workflow(
    source: os.PathLike[str] | str | np.ndarray | torch.Tensor | Mapping[str, Any] | VectorSource,
    config: ClusteringConfig | None = None,
    *,
    ids: Sequence[str] | None = None,
) -> ClusteringWorkflowResult:
    """Run loading, vectorization, optional scaling, clustering, and plotting."""

    config = config or ClusteringConfig()
    set_global_seed(config.seed)
    table = load_embeddings(
        source,
        ids=ids,
        pooling_types=config.pooling_types,
        legacy_embedding_dim=config.legacy_embedding_dim,
        csv_id_column=config.csv_id_column,
    )
    if table.vectors.shape[0] < 2:
        raise ValueError("Clustering requires at least two embedding records.")

    standardizer = None
    vectors = table.vectors
    if config.standardize:
        standardizer = EmbeddingStandardizer.fit_numpy(vectors)
        vectors = standardizer.transform_numpy(vectors)

    results: dict[str, AlgorithmResult] = {}
    errors: dict[str, str] = {}
    for algorithm in config.algorithms:
        try:
            results[algorithm] = fit_clusterer(algorithm, vectors, config)
        except Exception as error:
            errors[f"algorithm:{algorithm}"] = f"{type(error).__name__}: {error}"
    if not results:
        messages = "; ".join(errors.values())
        raise RuntimeError(f"Every clustering algorithm failed: {messages}")

    plot_indices = _sample_indices(
        vectors.shape[0],
        config.plot_sample_size,
        config.seed,
    )
    plot_vectors = vectors[plot_indices]
    reductions: dict[str, np.ndarray] = {}
    for method in config.reductions:
        try:
            reductions[method] = _reduce_vectors(plot_vectors, method, config.seed)
        except Exception as error:
            errors[f"reduction:{method}"] = f"{type(error).__name__}: {error}"

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_manifest_artifacts(config.output_dir)
    _write_outputs(
        source,
        config,
        table,
        vectors,
        standardizer,
        results,
        errors,
        (),
    )

    plot_paths: list[Path] = []

    def attempt_plot(key: str, build_plot: Callable[[], Path | None]) -> None:
        try:
            path = build_plot()
        except Exception as error:
            plt.close("all")
            errors[f"plot:{key}"] = f"{type(error).__name__}: {error}"
            return
        if path is not None:
            plot_paths.append(path)

    for algorithm, result in results.items():
        plot_labels = result.labels[plot_indices]
        attempt_plot(
            f"{algorithm}:cluster_sizes",
            lambda result=result, algorithm=algorithm: _plot_cluster_sizes(
                result.labels,
                algorithm,
                config.output_dir,
                config.dpi,
            ),
        )
        attempt_plot(
            f"{algorithm}:silhouette",
            lambda plot_labels=plot_labels, algorithm=algorithm: _plot_silhouette(
                plot_vectors,
                plot_labels,
                algorithm,
                config.output_dir,
                config.dpi,
            ),
        )
        attempt_plot(
            f"{algorithm}:centroid_distances",
            lambda result=result, algorithm=algorithm: _plot_centroid_distances(
                vectors,
                result.labels,
                algorithm,
                config.output_dir,
                config.dpi,
            ),
        )
        for method, coordinates in reductions.items():
            attempt_plot(
                f"{algorithm}:{method}",
                lambda coordinates=coordinates, plot_labels=plot_labels, algorithm=algorithm, method=method: _plot_scatter(
                    coordinates,
                    plot_labels,
                    algorithm,
                    method,
                    config.output_dir,
                    config.dpi,
                ),
            )

    attempt_plot(
        "algorithm_metrics",
        lambda: _plot_metric_comparison(results, config.output_dir, config.dpi),
    )
    if "pca" in reductions:
        attempt_plot(
            "algorithm_overview_pca",
            lambda: _plot_algorithm_overview(
                reductions["pca"],
                results,
                plot_indices,
                config.output_dir,
                config.dpi,
            ),
        )
    if config.diagnostics:
        if "kmeans" in results:
            attempt_plot(
                "kmeans_model_selection",
                lambda: _plot_kmeans_diagnostics(plot_vectors, config),
            )
        if "dbscan" in results:
            attempt_plot(
                "dbscan_k_distance",
                lambda: _plot_density_diagnostics(plot_vectors, config),
            )
        if "optics" in results:
            attempt_plot(
                "optics_reachability",
                lambda: _plot_optics_reachability(results["optics"], config),
            )
        if "agglomerative" in results:
            attempt_plot(
                "agglomerative_dendrogram",
                lambda: _plot_agglomerative_dendrogram(results["agglomerative"], config),
            )

    _write_outputs(
        source,
        config,
        table,
        vectors,
        standardizer,
        results,
        errors,
        plot_paths,
    )
    return ClusteringWorkflowResult(
        table=table,
        transformed_vectors=vectors,
        standardizer=standardizer,
        algorithms=results,
        errors=errors,
        output_dir=config.output_dir,
        plot_paths=tuple(plot_paths),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster precomputed protein embeddings and save diagnostic plots."
    )
    parser.add_argument("--input", required=True, help="Embedding .pth, .npy, .npz, CSV/TSV, or SQLite file.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Artifact directory. Defaults to plots/clustering/<input stem>.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(DEFAULT_ALGORITHMS),
        choices=[*SUPPORTED_ALGORITHMS, "all"],
    )
    parser.add_argument(
        "--reductions",
        nargs="+",
        default=["pca"],
        choices=SUPPORTED_REDUCTIONS,
    )
    parser.add_argument(
        "--pooling_types",
        nargs="+",
        default=["mean"],
        choices=SUPPORTED_POOLING_TYPES,
        help="Pooling operations concatenated for per-token embedding matrices.",
    )
    parser.add_argument("--standardize", action="store_true", help="Fit and apply StandardScaler before clustering.")
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN epsilon.")
    parser.add_argument("--min_samples", type=int, default=5, help="Density-clustering minimum samples.")
    parser.add_argument("--min_cluster_size", type=int, default=5, help="HDBSCAN minimum cluster size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric_sample_size",
        type=int,
        default=5_000,
        help="Maximum deterministic sample used for internal validation metrics.",
    )
    parser.add_argument(
        "--plot_sample_size",
        type=int,
        default=3_000,
        help="Maximum deterministic sample used for reductions and scatter diagnostics.",
    )
    parser.add_argument(
        "--quadratic_algorithm_limit",
        type=int,
        default=10_000,
        help="Maximum records allowed for algorithms with quadratic worst-case cost.",
    )
    parser.add_argument(
        "--legacy_embedding_dim",
        type=int,
        default=None,
        help="Feature width required to decode legacy raw-float SQLite blobs.",
    )
    parser.add_argument(
        "--csv_id_column",
        default=None,
        help="Identifier column for CSV/TSV input; otherwise inferred.",
    )
    parser.add_argument("--no_diagnostics", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> ClusteringWorkflowResult:
    arguments = build_argument_parser().parse_args(argv)
    input_path = Path(arguments.input)
    output_dir = (
        Path(arguments.output_dir)
        if arguments.output_dir is not None
        else Path("plots") / "clustering" / input_path.stem
    )
    config = ClusteringConfig(
        output_dir=output_dir,
        algorithms=tuple(arguments.algorithms),
        reductions=tuple(arguments.reductions),
        pooling_types=tuple(arguments.pooling_types),
        standardize=arguments.standardize,
        n_clusters=arguments.n_clusters,
        eps=arguments.eps,
        min_samples=arguments.min_samples,
        min_cluster_size=arguments.min_cluster_size,
        seed=arguments.seed,
        metric_sample_size=arguments.metric_sample_size,
        plot_sample_size=arguments.plot_sample_size,
        quadratic_algorithm_limit=arguments.quadratic_algorithm_limit,
        legacy_embedding_dim=arguments.legacy_embedding_dim,
        csv_id_column=arguments.csv_id_column,
        diagnostics=not arguments.no_diagnostics,
    )
    result = run_clustering_workflow(input_path, config)
    print(
        f"Clustered {len(result.table.ids)} records with "
        f"{len(result.algorithms)} algorithms. Artifacts: {result.output_dir}"
    )
    if result.errors:
        print(f"Completed with {len(result.errors)} skipped algorithm/reduction steps.")
    return result


if __name__ == "__main__":
    main()
