import csv
import json
import sqlite3

import numpy as np
import pytest
import torch
from sklearn.datasets import make_blobs

from src.protify.utils import tensor_to_embedding_blob
from src.protify.visualization import clustering
from src.protify.visualization.clustering import (
    SUPPORTED_ALGORITHMS,
    ClusteringConfig,
    fit_clusterer,
    load_embeddings,
    run_clustering_workflow,
)


def _cluster_matrix() -> np.ndarray:
    vectors, _ = make_blobs(
        n_samples=36,
        centers=3,
        n_features=4,
        cluster_std=0.25,
        random_state=7,
    )
    return vectors.astype(np.float32)


def test_load_embeddings_pools_per_token_matrices() -> None:
    source = {
        "alpha": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # (l=2, d=2)
        "beta": torch.tensor([[5.0, 1.0], [7.0, 3.0]]),  # (l, d)
    }

    table = load_embeddings(source, pooling_types=("mean", "max"))

    assert table.ids == ("alpha", "beta")
    np.testing.assert_allclose(
        table.vectors,
        np.array(
            [
                [2.0, 3.0, 3.0, 4.0],
                [6.0, 2.0, 7.0, 3.0],
            ],
            dtype=np.float32,
        ),
    )


@pytest.mark.parametrize("pooling_types", ((), ("typo",)))
def test_vector_sources_validate_pooling_types(pooling_types: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="pooling"):
        load_embeddings(
            {"alpha": torch.tensor([1.0, 2.0])},
            pooling_types=pooling_types,
        )


def test_array_source_rejects_bare_string_ids() -> None:
    with pytest.raises(TypeError, match="complete record IDs"):
        load_embeddings(np.ones((3, 2), dtype=np.float32), ids="abc")


def test_load_embeddings_reads_protify_sqlite(tmp_path) -> None:
    database_path = tmp_path / "embeddings.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE embeddings (sequence TEXT PRIMARY KEY, embedding BLOB)"
        )
        connection.executemany(
            "INSERT INTO embeddings VALUES (?, ?)",
            [
                ("seq-b", tensor_to_embedding_blob(torch.tensor([3.0, 4.0]))),
                (
                    "seq-a",
                    tensor_to_embedding_blob(
                        torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                    ),
                ),
            ],
        )

    table = load_embeddings(database_path, pooling_types=("mean",))

    assert table.ids == ("seq-b", "seq-a")
    np.testing.assert_allclose(
        table.vectors,
        np.array([[3.0, 4.0], [2.0, 3.0]], dtype=np.float32),
    )


def test_load_embeddings_reads_pth_mapping(tmp_path) -> None:
    embedding_path = tmp_path / "embeddings.pth"
    torch.save(
        {
            "first": torch.tensor([1.0, 2.0]),
            "second": torch.tensor([3.0, 4.0]),
        },
        embedding_path,
    )

    table = load_embeddings(embedding_path)

    assert table.ids == ("first", "second")
    np.testing.assert_allclose(table.vectors, [[1.0, 2.0], [3.0, 4.0]])


def test_load_embeddings_reads_csv_with_inferred_ids(tmp_path) -> None:
    csv_path = tmp_path / "vectors.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein", "x", "y"])
        writer.writerow(["p1", 1.0, 2.0])
        writer.writerow(["p2", 3.0, 4.0])

    table = load_embeddings(csv_path)

    assert table.ids == ("p1", "p2")
    np.testing.assert_allclose(table.vectors, [[1.0, 2.0], [3.0, 4.0]])


def test_load_embeddings_accepts_vector_source_adapter() -> None:
    class ExampleVectorDatabase:
        def iter_vectors(self):
            yield "first", np.array([1.0, 2.0], dtype=np.float32)
            yield "second", np.array([3.0, 4.0], dtype=np.float32)

    table = load_embeddings(ExampleVectorDatabase())

    assert table.ids == ("first", "second")
    np.testing.assert_allclose(table.vectors, [[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize("algorithm", SUPPORTED_ALGORITHMS)
def test_supported_clusterers_return_one_label_per_record(algorithm: str, tmp_path) -> None:
    vectors = _cluster_matrix()
    config = ClusteringConfig(
        output_dir=tmp_path,
        algorithms=(algorithm,),
        reductions=(),
        n_clusters=3,
        eps=0.7,
        min_samples=3,
        diagnostics=False,
    )

    result = fit_clusterer(algorithm, vectors, config)

    assert result.labels.shape == (len(vectors),)
    assert result.metrics["n_clusters"] >= 1


def test_hdbscan_uses_distinct_density_parameters(tmp_path) -> None:
    config = ClusteringConfig(
        output_dir=tmp_path,
        algorithms=("hdbscan",),
        reductions=(),
        min_samples=3,
        min_cluster_size=7,
        diagnostics=False,
    )

    result = fit_clusterer("hdbscan", _cluster_matrix(), config)

    assert result.estimator.min_samples == 3
    assert result.estimator.min_cluster_size == 7


def test_quadratic_algorithm_limit_is_explicit(tmp_path) -> None:
    config = ClusteringConfig(
        output_dir=tmp_path,
        algorithms=("agglomerative",),
        reductions=(),
        n_clusters=3,
        quadratic_algorithm_limit=20,
        diagnostics=False,
    )

    with pytest.raises(ValueError, match="quadratic"):
        fit_clusterer("agglomerative", _cluster_matrix(), config)


def test_clustering_workflow_standardizes_and_writes_artifacts(tmp_path) -> None:
    vectors = _cluster_matrix()
    record_ids = [f"protein-{index}" for index in range(len(vectors))]
    output_dir = tmp_path / "plots" / "clustering" / "test-run"
    config = ClusteringConfig(
        output_dir=output_dir,
        algorithms=("kmeans", "agglomerative", "dbscan"),
        reductions=("pca",),
        standardize=True,
        n_clusters=3,
        eps=0.7,
        min_samples=3,
        seed=11,
        diagnostics=True,
        dpi=72,
    )

    result = run_clustering_workflow(vectors, config, ids=record_ids)

    assert result.table.ids == tuple(record_ids)
    assert result.standardizer is not None
    np.testing.assert_allclose(
        result.transformed_vectors.mean(axis=0),
        np.zeros(vectors.shape[1]),
        atol=1e-6,
    )
    assert set(result.algorithms) == {"kmeans", "agglomerative", "dbscan"}
    assert not result.errors
    assert all(path.is_file() for path in result.plot_paths)
    assert len(result.plot_paths) >= 10
    for filename in (
        "assignments.tsv",
        "metrics.tsv",
        "vectors.npz",
        "standardizer.npz",
        "config.json",
        "errors.json",
        "manifest.json",
        "algorithm_metrics.png",
        "algorithm_overview_pca.png",
        "kmeans_model_selection.png",
        "dbscan_k_distance.png",
        "agglomerative_dendrogram.png",
    ):
        assert (output_dir / filename).is_file()
    assert json.loads((output_dir / "errors.json").read_text(encoding="utf-8")) == {}
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "assignments.tsv" in manifest["artifacts"]
    assert manifest["estimators"]["kmeans"]["n_clusters"] == 3


def test_plot_failure_does_not_discard_assignments(tmp_path, monkeypatch) -> None:
    def fail_plot(*args, **kwargs):
        raise RuntimeError("synthetic plot failure")

    monkeypatch.setattr(clustering, "_plot_cluster_sizes", fail_plot)
    output_dir = tmp_path / "failed-plot"
    config = ClusteringConfig(
        output_dir=output_dir,
        algorithms=("kmeans",),
        reductions=(),
        n_clusters=3,
        diagnostics=False,
        dpi=72,
    )

    result = run_clustering_workflow(_cluster_matrix(), config)

    assert (output_dir / "assignments.tsv").is_file()
    assert (output_dir / "metrics.tsv").is_file()
    assert "plot:kmeans:cluster_sizes" in result.errors
    saved_errors = json.loads((output_dir / "errors.json").read_text(encoding="utf-8"))
    assert "plot:kmeans:cluster_sizes" in saved_errors


def test_duplicate_vectors_do_not_break_kmeans_diagnostics(tmp_path) -> None:
    vectors = np.zeros((12, 3), dtype=np.float32)
    output_dir = tmp_path / "duplicates"
    config = ClusteringConfig(
        output_dir=output_dir,
        algorithms=("kmeans",),
        reductions=("pca",),
        n_clusters=3,
        diagnostics=True,
        dpi=72,
    )

    result = run_clustering_workflow(vectors, config)

    assert "kmeans" in result.algorithms
    assert (output_dir / "assignments.tsv").is_file()
    assert (output_dir / "kmeans_model_selection.png").is_file()


def test_load_embeddings_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        load_embeddings(np.array([[1.0, np.nan]], dtype=np.float32))
