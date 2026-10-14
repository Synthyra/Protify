import csv
import json
import sqlite3
import numpy as np
import pytest
import torch

from sklearn.datasets import make_blobs

from src.protify.testing_suite.embedding_store_builder import build_store
from src.protify.testing_suite.legacy_cache_writer import write_sqlite_cache
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
    return vectors.astype(np.float32)  # (36, 4)


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
    """An old Protify SQLite cache is read through the legacy decoder, in the order it was written."""
    database_path = write_sqlite_cache(
        tmp_path / "embeddings.db",
        {
            "seq-b": torch.tensor([3.0, 4.0]),
            "seq-a": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        },
    )

    table = load_embeddings(database_path, pooling_types=("mean",))

    assert table.ids == ("seq-b", "seq-a")
    np.testing.assert_allclose(
        table.vectors,
        np.array([[3.0, 4.0], [2.0, 3.0]], dtype=np.float32),
    )


def test_a_sqlite_cache_is_not_created_or_changed_by_loading(tmp_path) -> None:
    database_path = write_sqlite_cache(tmp_path / "embeddings.db", {"seq-a": torch.tensor([1.0, 2.0])}, wal=False)
    before = sorted((path.name, path.read_bytes()) for path in tmp_path.iterdir())

    load_embeddings(database_path)

    assert sorted((path.name, path.read_bytes()) for path in tmp_path.iterdir()) == before


def test_sqlite_input_names_the_schema_it_needs(tmp_path) -> None:
    other = tmp_path / "other.db"
    with sqlite3.connect(other) as connection:
        connection.execute("CREATE TABLE unrelated (x INTEGER)")
    narrow = tmp_path / "narrow.db"
    with sqlite3.connect(narrow) as connection:
        connection.execute("CREATE TABLE embeddings (sequence TEXT)")

    with pytest.raises(ValueError, match=r"embeddings\(sequence, embedding\) schema"):
        load_embeddings(other)
    with pytest.raises(ValueError, match="sequence and embedding columns"):
        load_embeddings(narrow)


def test_sqlite_ids_cannot_override_the_stored_sequences(tmp_path) -> None:
    database_path = write_sqlite_cache(tmp_path / "embeddings.db", {"seq-a": torch.tensor([1.0, 2.0])})

    with pytest.raises(ValueError, match="ids cannot override identifiers in SQLite"):
        load_embeddings(database_path, ids=["seq-a"])


def test_load_embeddings_reads_a_feature_store_by_sequence(tmp_path) -> None:
    directory = build_store(
        tmp_path,
        {
            "seq-a": torch.tensor([1.0, 2.0]),
            "seq-b": torch.tensor([3.0, 4.0]),
            "seq-c": torch.tensor([5.0, 6.0]),
        },
    )

    table = load_embeddings(directory, ids=["seq-c", "seq-a"])

    assert table.ids == ("seq-c", "seq-a")
    np.testing.assert_allclose(table.vectors, [[5.0, 6.0], [1.0, 2.0]])


def test_a_feature_store_of_matrices_is_pooled_per_row(tmp_path) -> None:
    directory = build_store(
        tmp_path,
        {
            "alpha": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # (l=2, d=2)
            "beta": torch.tensor([[5.0, 1.0], [7.0, 3.0]]),  # (l, d)
        },
        matrix=True,
    )

    table = load_embeddings(directory, ids=["alpha", "beta"], pooling_types=("mean", "max"))

    np.testing.assert_allclose(
        table.vectors,
        np.array([[2.0, 3.0, 3.0, 4.0], [6.0, 2.0, 7.0, 3.0]], dtype=np.float32),
    )


def test_a_sparse_feature_store_loads_dense_rows(tmp_path) -> None:
    rows = {
        "seq-a": torch.tensor([0.0, 2.5, 0.0, 0.0, 1.0, 0.0]),
        "seq-b": torch.tensor([0.0, 0.0, 0.0, 4.0, 0.0, 0.0]),
    }
    # 16 of 8192 active is below the coordinate-list threshold, so this store keeps `csr` rows.
    directory = build_store(tmp_path, rows, model_name="ESMC-300-SAE-l23-k16-c8192", pooling=("max",))

    table = load_embeddings(directory, ids=["seq-a", "seq-b"])

    np.testing.assert_allclose(table.vectors, np.stack([rows["seq-a"].numpy(), rows["seq-b"].numpy()]))


def test_a_feature_store_needs_the_sequences_to_read(tmp_path) -> None:
    directory = build_store(tmp_path, {"seq-a": torch.tensor([1.0, 2.0])})

    with pytest.raises(ValueError, match="pass ids"):
        load_embeddings(directory)


def test_a_feature_store_missing_a_requested_sequence_is_refused(tmp_path) -> None:
    directory = build_store(tmp_path, {"seq-a": torch.tensor([1.0, 2.0])})

    with pytest.raises(ValueError, match="does not hold every requested id"):
        load_embeddings(directory, ids=["seq-a", "seq-absent"])


def test_the_command_line_clusters_a_feature_store_given_an_ids_file(tmp_path) -> None:
    vectors = _cluster_matrix()  # (36, 4)
    sequences = [f"seq-{index}" for index in range(len(vectors))]
    directory = build_store(
        tmp_path, {sequence: torch.from_numpy(row) for sequence, row in zip(sequences, vectors, strict=True)}
    )
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("\n".join(sequences) + "\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    workflow = clustering.main(
        [
            "--input", str(directory),
            "--ids_file", str(ids_file),
            "--output_dir", str(output_dir),
            "--algorithms", "kmeans",
            "--n_clusters", "3",
            "--no_diagnostics",
            "--reductions", "pca",
        ]
    )

    assert workflow.table.ids == tuple(sequences)
    np.testing.assert_allclose(workflow.table.vectors, vectors)
    assert (output_dir / "assignments.tsv").is_file()


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

    fitted = fit_clusterer(algorithm, vectors, config)

    assert fitted.labels.shape == (len(vectors),)
    assert fitted.metrics["n_clusters"] >= 1


def test_hdbscan_uses_distinct_density_parameters(tmp_path) -> None:
    config = ClusteringConfig(
        output_dir=tmp_path,
        algorithms=("hdbscan",),
        reductions=(),
        min_samples=3,
        min_cluster_size=7,
        diagnostics=False,
    )

    fitted = fit_clusterer("hdbscan", _cluster_matrix(), config)

    assert fitted.estimator.min_samples == 3
    assert fitted.estimator.min_cluster_size == 7


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

    workflow = run_clustering_workflow(vectors, config, ids=record_ids)

    assert workflow.table.ids == tuple(record_ids)
    assert workflow.standardizer is not None
    np.testing.assert_allclose(
        workflow.transformed_vectors.mean(axis=0),
        np.zeros(vectors.shape[1]),
        atol=1e-6,
    )
    assert set(workflow.algorithms) == {"kmeans", "agglomerative", "dbscan"}
    assert not workflow.errors
    assert all(path.is_file() for path in workflow.plot_paths)
    assert len(workflow.plot_paths) >= 10
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

    workflow = run_clustering_workflow(_cluster_matrix(), config)

    assert (output_dir / "assignments.tsv").is_file()
    assert (output_dir / "metrics.tsv").is_file()
    assert "plot:kmeans:cluster_sizes" in workflow.errors
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

    workflow = run_clustering_workflow(vectors, config)

    assert "kmeans" in workflow.algorithms
    assert (output_dir / "assignments.tsv").is_file()
    assert (output_dir / "kmeans_model_selection.png").is_file()


def test_load_embeddings_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        load_embeddings(np.array([[1.0, np.nan]], dtype=np.float32))
