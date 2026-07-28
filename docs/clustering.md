# Embedding clustering

Protify can cluster existing embedding vectors or per-token embedding matrices without loading a protein language model. The workflow reads local embedding artifacts or a Python vector-database adapter, converts every record to a fixed-width vector, optionally standardizes the features, runs one or more clustering algorithms, and saves assignments, metrics, transformed vectors, configuration, and diagnostic plots.

The implementation is in [`visualization/clustering.py`](../src/protify/visualization/clustering.py).

## Command-line workflow

Cluster a Protify `.pth` embedding dictionary or SQLite embedding database:

```bash
python -m src.protify.visualization.clustering \
  --input embeddings/ESM2-650_False_mean.pth \
  --output_dir plots/clustering/esm2_650 \
  --algorithms kmeans agglomerative dbscan optics \
  --n_clusters 8 \
  --standardize \
  --reductions pca tsne
```

If `--output_dir` is omitted, artifacts go to `plots/clustering/<input-stem>/`.

For per-token matrix embeddings, one or more pooling operations can be concatenated:

```bash
python -m src.protify.visualization.clustering \
  --input embeddings/ESM2-650_True.pth \
  --pooling_types mean var max \
  --algorithms all \
  --n_clusters 12
```

Supported pooling operations are `mean`, `max`, `norm`, `median`, `std`, `var`, and `cls`. Existing one-dimensional vectors are used unchanged. `parti` is not available because saved embedding artifacts do not contain the attention matrices it requires.

## Input sources

The loader accepts:

- `.pt` or `.pth`: a mapping from record IDs or sequences to `(d,)` vectors or `(l, d)` token matrices.
- `.npy`: an `(n, d)` vector matrix or `(n, l, d)` stack of token matrices.
- `.npz`: an `embeddings` array and optional `ids` array. A single non-ID array is also accepted.
- `.csv` or `.tsv`: one record per row, with numeric feature columns and an optional ID column. Use `--csv_id_column` when it cannot be inferred.
- `.db`, `.sqlite`, or `.sqlite3`: Protify's `embeddings(sequence, embedding)` SQLite schema.
- Direct NumPy arrays, Torch tensors, mappings, or a Python `VectorSource` adapter.

Modern Protify SQLite blobs carry their tensor shape. Legacy raw-float blobs require `--legacy_embedding_dim` so the feature width is explicit.

FastPLMs result databases and third-party services such as FAISS, Chroma, Qdrant, Pinecone, or Weaviate use different schemas and are not opened implicitly. Connect them through the small adapter protocol instead:

```python
from protify.visualization.clustering import (
    ClusteringConfig,
    run_clustering_workflow,
)


class MyVectorDatabase:
    def iter_vectors(self):
        for record in query_my_database():
            yield record.id, record.embedding


result = run_clustering_workflow(
    MyVectorDatabase(),
    ClusteringConfig(
        output_dir="plots/clustering/my_database",
        algorithms=("kmeans", "dbscan", "hdbscan"),
        n_clusters=8,
        standardize=True,
    ),
)
```

The adapter may yield either vectors or per-token matrices.

## Algorithms

The basic default set avoids algorithms with known quadratic worst-case behavior:

- `kmeans`
- `birch`
- `hdbscan`

Additional selectable algorithms are:

- `mini_batch_kmeans`
- `agglomerative`
- `spectral`
- `gaussian_mixture`
- `dbscan`
- `optics`

Use `--algorithms all` to run every algorithm. Fixed-cluster algorithms use `--n_clusters`. DBSCAN uses `--eps` and `--min_samples`; OPTICS uses `--min_samples`; HDBSCAN uses both `--min_samples` and `--min_cluster_size`. Stochastic algorithms use `--seed`.

Agglomerative, spectral, and DBSCAN clustering are guarded above 10,000 records by default because their worst-case runtime or memory can be quadratic. Use `--quadratic_algorithm_limit` to make an explicit higher-cost choice. Internal validity metrics use a deterministic sample of at most 5,000 records, while reductions and scatter diagnostics use at most 3,000. These limits are configurable with `--metric_sample_size` and `--plot_sample_size`; assignments and cluster-size counts always cover the complete input.

Standardization is opt-in. `--standardize` fits a `StandardScaler` after matrix pooling and before every clustering algorithm and dimensionality reduction.

## Outputs

Every run writes:

- `assignments.tsv`: record IDs and the cluster label from each successful algorithm. Density methods retain `-1` for noise.
- `metrics.tsv`: cluster count, noise count and fraction, silhouette score, Calinski-Harabasz score, and Davies-Bouldin score.
- `vectors.npz`: the exact fixed-width vectors used for clustering, after optional standardization.
- `standardizer.npz`: fitted feature means and scales when standardization is enabled.
- `config.json`: the resolved run configuration and source.
- `errors.json`: any algorithm or optional dimensionality-reduction step that was skipped, with its error.
- `manifest.json`: current-run artifact inventory, resolved estimator parameters, source file metadata, and relevant library versions.

Plots are PNG files saved at 300 dpi by default:

- PCA, t-SNE, or optional UMAP cluster scatter plots.
- Cluster-size bars.
- Silhouette distributions when at least two non-noise clusters exist.
- Pairwise cluster-centroid distance heatmaps.
- Cross-algorithm metric comparison and PCA overview.
- KMeans elbow and silhouette-by-k diagnostics.
- DBSCAN nearest-neighbor distance diagnostics.
- OPTICS reachability and agglomerative dendrograms when selected.

Validity scores exclude records labeled as noise; `metrics.tsv` and the comparison plot report clustered coverage so sparse density-clustering solutions remain visible. UMAP is loaded lazily and requires the optional `umap-learn` package. If it is requested but unavailable, the rest of the run completes and the reason is recorded in `errors.json`.
