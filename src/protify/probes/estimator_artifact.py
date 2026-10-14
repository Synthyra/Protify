"""Saving a fitted estimator probe with everything needed to score a new protein.

An estimator probe is a fitted booster or forest, not a torch module, so it cannot travel
inside `PackagedProbeModel` the way a neural probe does. It still has to travel with the
same three things, or a saved model silently scores a different representation than it was
trained on:

  the embedding recipe    which backbone and codebook produced the row, and how the
                          per-residue features were pooled into it
  the pair rule           how two proteins become one row, since a probe trained on
                          [min, max] cannot read [A, B]
  the column selection    which columns of that row the probe was actually fit on, after
                          the path selector and the dead-column drop

The directory this writes is self-describing: `probe.json` names all three, and the
estimator sits beside it. Reloading rebuilds the transform, and `embed_proteins` rebuilds
the representation from the recipe rather than trusting a caller to reproduce it.
"""

from __future__ import annotations

import json
import numpy as np
import torch

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from scipy import sparse

from .estimator_probe import REGRESSION_TASK_TYPES, _positive_probability
from .feature_path_selector import FeatureSelection
from ..console_output import print_message
from ..data.pair_features import pair_vector_halves


PROBE_CONFIG_NAME = "probe.json"
ESTIMATOR_NAME = "estimator.joblib"
SELECTION_NAME = "feature_selection.npz"


@dataclass(frozen=True)
class EstimatorProbeRecipe:
    """Everything about a saved estimator probe except the fitted estimator itself."""

    probe_type: str
    task_type: str
    num_labels: int
    # The resolved model name, which for an SAE carries the layer, sparsity, and codebook
    # width, so it is the whole embedding identity rather than a family name.
    model_name: str
    pooling_types: list[str]
    # The layer a dense backbone reads. An SAE fixes its own layer through the model name,
    # so this stays at the -1 the run passed and is never what selects the representation.
    hidden_state_index: int
    max_length: int
    ppi: bool
    pair_feature_mode: str
    hyperparameters: dict[str, Any]
    # Whether the estimator was fit on a CSR matrix. XGBoost reads an entry a CSR does not
    # store as missing and a stored zero as zero, so scoring must rebuild the same layout or a
    # saved probe scores different inputs than it was fit on. False in recipes saved before
    # this was recorded, which were scored dense.
    sparse_layout: bool = False


def save_estimator_probe(
    directory: str | Path,
    estimator: Any,
    recipe: EstimatorProbeRecipe,
    feature_selection: FeatureSelection | None,
) -> Path:
    """Write one fitted estimator probe and its recipe to a directory."""
    import joblib

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path / ESTIMATOR_NAME)
    (path / PROBE_CONFIG_NAME).write_text(
        json.dumps(asdict(recipe), indent=2), encoding="utf-8"
    )
    if feature_selection is not None:
        feature_selection.save(path / SELECTION_NAME)

    print_message(f"Saved {recipe.probe_type} probe to {path}")
    return path


def load_estimator_probe(
    directory: str | Path,
) -> tuple[Any, EstimatorProbeRecipe, FeatureSelection | None]:
    """Read back what `save_estimator_probe` wrote."""
    import joblib

    path = Path(directory)
    recipe = EstimatorProbeRecipe(
        **json.loads((path / PROBE_CONFIG_NAME).read_text(encoding="utf-8"))
    )
    selection_path = path / SELECTION_NAME
    selection = FeatureSelection.load(selection_path) if selection_path.is_file() else None
    return joblib.load(path / ESTIMATOR_NAME), recipe, selection


def build_feature_rows(
    pooled: np.ndarray,
    recipe: EstimatorProbeRecipe,
    feature_selection: FeatureSelection | None,
    pooled_partner: np.ndarray | None = None,
) -> np.ndarray:
    """Turn pooled protein vectors into the rows the saved estimator was fit on.

    `pooled_partner` is protein B of each pair and is required for a probe trained on a
    pair dataset, because a pair row is not something a single protein can produce.
    """
    # pooled: (n, d); pooled_partner: (n, d) or None -> (n, s)
    if recipe.ppi:
        if pooled_partner is None:
            raise ValueError(
                f"{recipe.model_name} was trained on a pair dataset, so scoring needs both "
                f"proteins. Pass the partner's pooled vectors as pooled_partner."
            )
        half_a, half_b = pair_vector_halves(
            pooled, pooled_partner, recipe.pair_feature_mode
        )  # two of (n, d)
        pooled = np.concatenate([half_a, half_b], axis=-1)  # (n, 2 * d)
    elif pooled_partner is not None:
        raise ValueError(
            f"{recipe.model_name} was trained on single proteins, so pooled_partner has "
            f"no meaning here."
        )

    if feature_selection is None:
        return pooled  # (n, d)

    return feature_selection.transform_numpy(pooled)  # (n, s)


def embed_proteins(
    sequences: Sequence[str],
    recipe: EstimatorProbeRecipe,
    batch_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Pool each protein exactly the way the recipe says it was pooled during training.

    Runs the `Embedder` the training run used rather than reimplementing pooling here.
    Reproducing it a second way is what would let a saved probe drift from the
    representation it was fit on, and nothing would raise when it did.

    Imported inside the function because a sparse autoencoder recipe loads the vendored
    FastPLMs package, and reading a saved probe's configuration must not require it.
    """
    from ..embedder import Embedder, EmbeddingArguments

    unique = list(dict.fromkeys(sequences))
    embedding_args = EmbeddingArguments(
        embedding_pooling_types=list(recipe.pooling_types),
        matrix_embed=False,
        embedding_batch_size=batch_size,
        embedding_hidden_state_index=recipe.hidden_state_index,
        max_length=recipe.max_length,
        save_embeddings=False,
        sql=False,
        download_embeddings=False,
    )
    return Embedder(embedding_args, unique)(recipe.model_name)  # (d,) pooled or (l + 2, d) token rows per sequence


def predict_features(estimator: Any, features: np.ndarray, task_type: str) -> np.ndarray:
    """What a caller wants back: a value per protein, or a probability per class.

    Deliberately not `estimator_probe._predictions`, which returns log probabilities
    because the shared metric functions read their input as raw logits. Nobody scoring new
    proteins wants logs, and nobody wants a bare class label either.
    """
    # features: (n, s)
    if task_type in REGRESSION_TASK_TYPES:
        return estimator.predict(features)  # (n,)

    if task_type == "multilabel":
        # One block per label, and one class vector per label alongside it, which is how
        # both MultiOutputClassifier and a natively multi-output forest lay them out.
        return np.stack(
            [_positive_probability(block, classes)
             for block, classes in zip(estimator.predict_proba(features), estimator.classes_, strict=True)],
            axis=1,
        )  # (n, c)

    return estimator.predict_proba(features)  # (n, c)


def score_proteins(
    estimator: Any,
    recipe: EstimatorProbeRecipe,
    feature_selection: FeatureSelection | None,
    sequences: Sequence[str],
    partners: Sequence[str] | None = None,
    batch_size: int = 8,
) -> np.ndarray:
    """Predictions for raw sequences, embedded and joined the way training did it.

    Regression returns one value per protein; classification returns one probability per
    class, and multilabel one positive probability per label.
    """
    # sequences: n strings; partners: n strings for a pair probe -> (n,) or (n, c)
    needed = list(sequences) + (list(partners) if partners is not None else [])
    pooled_by_sequence = embed_proteins(needed, recipe, batch_size=batch_size)

    def rows(names: Sequence[str]) -> np.ndarray:
        return np.concatenate(
            [pooled_by_sequence[name].reshape(1, -1).float().numpy() for name in names], axis=0
        )  # (n, p * d)

    features = build_feature_rows(
        rows(sequences),
        recipe,
        feature_selection,
        None if partners is None else rows(partners),
    )  # (n, s)
    if recipe.sparse_layout:
        # Built as training built its CSR blocks, from dense rows, so the same zeros go unstored.
        features = sparse.csr_matrix(features)  # (n, s)
    return predict_features(estimator, features, recipe.task_type)  # (n,) or (n, c)
