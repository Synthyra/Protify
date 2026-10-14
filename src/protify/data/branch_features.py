"""Join several cached representations of each protein into one probe input row.

A branch is one representation Protify already caches, written `MODEL:POOL[,POOL][@HS]`: a
model, a pooling set, and a hidden-state index. Each branch is read from its ordinary cache,
named and opened by `DataMixin` exactly as a single-model run names and opens it, so the
caches `precompute_sae_and_dense.py` and a plain embedding run write are the ones read here.
The join happens in memory, per sequence and in spec order, the way synth's Atlas mode hands
its own `emb_dict` to training. No joined cache is written. `embed_missing_branches` fills a
branch cache first; `require_cached_branches` instead refuses an incomplete one without
writing, for training containers that share the caches.

SAE branches stay sparse in memory. A max-pooled 16384-wide codebook row is 64 KiB dense, so a
dense join of a few hundred thousand proteins would not fit a 32 GiB container. The dataset
hands out row indices, and the collator densifies one batch at a time.

Shapes: `n` sequences, `w` one branch's width, `W` the joined width, `m` rows in a block,
`b` rows in a batch.
"""

from __future__ import annotations

import random
import re
import numpy as np
import torch

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from tqdm.auto import tqdm

from foundry.embedding import FeatureView
from .cluster_sampling import ClusterSamplingPlan
from .data_mixin import DataMixin
from .dataset_classes import EmbeddingStandardizer
from .feature_matrix import stack_pooled_rows
from .label_batches import SparseLabelRow, SparseObservedTargets, collate_sparse_labels
from .pair_features import DEFAULT_PAIR_FEATURE_MODE, pair_vector_halves
from ..base_models.control_bases import check_pooling, is_control_model
from ..base_models.esmc_sae import ESMC_SAE_COVERAGE, is_sae_model_name, parse_sae_model_name, resolve_sae_model_name
from ..base_models.sae_pooling import validate_sae_pooling_types
from ..embedder import Embedder, EmbeddingArguments
from ..feature_cache import has_feature, missing_sequences, resolve_embed_dtype
from ..pooler import Pooler


# A branch's model name reaches file names and result tables, so it may not carry a path or
# the characters that delimit a spec.
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
# Rows densified at once while fitting the standardizer.
STANDARDIZER_BLOCK_ROWS = 1024
# Label types whose targets are floats; every other label type is a class index.
FLOAT_LABEL_TYPES = ("multilabel", "regression", "sigmoid_regression")
# The default probe's SAE branch (docs/conventions/probes.md): sparsity and codebook width of the
# Biohub SAE at each ESMC base's depth layer, the setting the ESMC-600M ablation chose.
DEFAULT_PROBE_SAE_K = 64
DEFAULT_PROBE_SAE_CODEBOOK_DIM = 16384
# The Atlas proteome profile (`data/atlas_profile_feature.py`) is a vector per protein that no base pools to, so its branch reads
# `Atlas-Profile:profile` and the store it comes from is the cell's `probe_features.profile_root`, not a model name.
PROFILE_MODEL = "Atlas-Profile"
PROFILE_POOLING = ("profile",)


@dataclass(frozen=True)
class BranchSpec:
    """One cached representation: a resolved model name, a pooling set, a hidden state."""

    model_name: str
    pooling_types: tuple[str, ...]
    hidden_state_index: int = -1

    @property
    def sparse(self) -> bool:
        """SAE codebooks are held sparse in memory; every other branch dense."""
        return is_sae_model_name(self.model_name)

    @property
    def profile(self) -> bool:
        """The Atlas proteome profile, which only a canonical-store probe input can read."""
        return self.model_name == PROFILE_MODEL

    @property
    def slug(self) -> str:
        """This branch as a file-name-safe string."""
        slug = f"{self.model_name}_{'-'.join(self.pooling_types)}"
        if self.hidden_state_index != -1:
            slug = f"{slug}_hs{self.hidden_state_index}"
        return slug


def parse_branch_spec(token: str) -> BranchSpec:
    """Read `MODEL:POOL[,POOL][@HS]`, resolving an SAE alias to its full checkpoint name.

    An SAE branch takes only the reductions its sparse codebook supports, so `var` is refused,
    and takes no `@HS`, since the layer its codebook reads is part of the resolved name.
    """
    body, separator, hidden_state = token.partition("@")
    model_name, _, pooling = body.partition(":")
    if not model_name or not pooling:
        raise ValueError(f"branch spec {token!r} must read MODEL:POOL[,POOL][@HS].")
    if not MODEL_NAME_PATTERN.match(model_name):
        raise ValueError(f"branch spec {token!r} names a model with characters a file name cannot carry.")

    pooling_types = tuple(name.strip() for name in pooling.split(","))
    if any(not name for name in pooling_types) or len(set(pooling_types)) != len(pooling_types):
        raise ValueError(f"branch spec {token!r} has an empty or repeated pooling type.")

    hidden_state_index = -1
    if separator:
        if not re.fullmatch(r"-?\d+", hidden_state):
            raise ValueError(f"branch spec {token!r} has a hidden-state index that is not an integer.")
        hidden_state_index = int(hidden_state)

    if model_name == PROFILE_MODEL:
        if separator or pooling_types != PROFILE_POOLING:
            raise ValueError(f"branch spec {token!r}: the Atlas profile is written {PROFILE_MODEL}:{PROFILE_POOLING[0]}, with no hidden state.")
        return BranchSpec(model_name, pooling_types)

    if is_sae_model_name(model_name):
        if separator:
            raise ValueError(
                f"branch spec {token!r} gives a hidden state, but an SAE branch reads the layer "
                f"named in its model name."
            )
        model_name = resolve_sae_model_name(model_name)
        validate_sae_pooling_types(pooling_types, parse_sae_model_name(model_name).codebook_dim)
    else:
        # The Pooler's constructor is the authority on dense pooling names.
        Pooler(list(pooling_types))
        if is_control_model(model_name):
            if separator:
                raise ValueError(f"branch spec {token!r} gives a hidden state, but {model_name} has no layers.")
            check_pooling(model_name, pooling_types)

    return BranchSpec(model_name, pooling_types, hidden_state_index)


def default_branch_specs(backbone: str) -> tuple[str, str]:
    """The default branches of a probe on an ESMC base, as `MODEL:POOL` tokens.

    The Biohub SAE at the base's depth layer (23 of 30 for ESMC-300, 27 of 36 for ESMC-600, 60 of
    80 for ESMC-6B), max-pooled over residues, beside the final hidden state's mean and variance.
    A `projected` head standardizes, clamps, and projects each branch, then concatenates them.
    docs/conventions/probes.md states the rule, and
    docs/decisions/serving/production_suite/2026-09-24_oracle_featurization.md the evidence.
    """
    if backbone not in ESMC_SAE_COVERAGE:
        raise ValueError(f"default branches are defined for {sorted(ESMC_SAE_COVERAGE)}; got {backbone!r}.")

    layer = ESMC_SAE_COVERAGE[backbone].depth_layer
    sae = f"{backbone}-SAE-l{layer}-k{DEFAULT_PROBE_SAE_K}-c{DEFAULT_PROBE_SAE_CODEBOOK_DIM}:max"
    return sae, f"{backbone}:mean,var"


def display_name(specs: Sequence[BranchSpec]) -> str:
    """A file-name-safe name for a set of branches, in spec order."""
    return "+".join(spec.slug for spec in specs)


def branch_embedding_arguments(spec: BranchSpec, arguments: Mapping[str, object]) -> EmbeddingArguments:
    """The embedding settings one branch's cache was written under.

    `arguments` holds a run's settings as MainProcess keeps them, or as the parser left them: a
    parsed `embed_dtype` of None is resolved as `MainProcess` resolves it, because the dtype is
    part of the store's name. Only the pooling set and the hidden-state index differ between
    branches, so only they are overridden. Branches are pooled vectors, so `matrix_embed` is
    always off.
    """
    settings = {
        **arguments,
        "matrix_embed": False,
        "embedding_pooling_types": list(spec.pooling_types),
        "embedding_hidden_state_index": spec.hidden_state_index,
    }
    if "embed_dtype" in settings:
        settings["embed_dtype"] = resolve_embed_dtype(settings["embed_dtype"], settings.get("model_dtype"))
    return EmbeddingArguments(**settings)


def _branch_reader(spec: BranchSpec, arguments: Mapping[str, object]) -> DataMixin:
    """A DataMixin that names and opens this branch's cache as a single-model run would."""
    reader = DataMixin()
    reader.embedding_args = branch_embedding_arguments(spec, arguments)
    reader._sql = bool(arguments["sql"])
    return reader


def branch_cache_path(spec: BranchSpec, arguments: Mapping[str, object]) -> str:
    """Where a branch's ordinary cache lives: its feature store, a directory under `embedding_save_dir`.

    The path is the same whether the run streams (`sql`) or holds rows in memory, since both read the
    one store. It exists only once the branch has been embedded or an old cache has been converted.
    """
    return _branch_reader(spec, arguments)._embedding_store(spec.model_name)


def embed_missing_branches(
    specs: Sequence[BranchSpec], sequences: Sequence[str], arguments: Mapping[str, object]
) -> None:
    """Fill each branch's cache with any sequence it lacks, through Protify's Embedder.

    A complete cache loads no model: the Embedder finds nothing to embed and returns.
    """
    for spec in specs:
        embedding_arguments = branch_embedding_arguments(spec, {**arguments, "save_embeddings": True})
        Embedder(embedding_arguments, list(sequences))(spec.model_name)


def _missing_from_cache(spec: BranchSpec, sequences: Sequence[str], arguments: Mapping[str, object]) -> int:
    """How many of the distinct `sequences` one branch's cache lacks, read without writing it.

    A missing store lacks every sequence. An existing one is opened read-only and asked about
    these sequences alone, so a cache shared across datasets is never read whole.
    """
    return len(missing_sequences(branch_cache_path(spec, arguments), sequences))


def require_cached_branches(
    specs: Sequence[BranchSpec], sequences: Sequence[str], arguments: Mapping[str, object]
) -> None:
    """Refuse, naming each incomplete branch and how many sequences it lacks, rather than embed.

    The workflow calls this in place of `embed_missing_branches` when training containers
    share one cache, which none of them may write. No cache is opened for writing here.
    """
    distinct = list(dict.fromkeys(sequences))
    shortfalls = []
    for spec in specs:
        missing = _missing_from_cache(spec, distinct, arguments)
        if missing:
            path = branch_cache_path(spec, arguments)
            shortfalls.append(f"{spec.slug} lacks {missing} of {len(distinct)} sequences ({path})")

    if shortfalls:
        raise ValueError(
            "require_cached_embeddings is set, so no sequence is embedded here, and these branch "
            f"caches are incomplete: {'; '.join(shortfalls)}. Fill them with a single writer first, "
            "such as this cell run once without require_cached_embeddings."
        )


def _cached_rows(pooled_row: Callable[[str], np.ndarray], sequences: Sequence[str], path: str) -> Iterator[np.ndarray]:
    """Each sequence's pooled row from one cache, in order, as float32 of shape (1, w)."""
    # pooled_row: (1, w) output per sequence.
    for sequence in sequences:
        try:
            row = pooled_row(sequence)  # (1, w)
        except KeyError as error:
            raise KeyError(f"{path} has no embedding for a sequence of length {len(sequence)}.") from error
        # SciPy's sparse matrices hold no float16, and the join is float32 throughout.
        yield row.astype(np.float32, copy=False)  # (1, w)


class JoinedBranchEmbeddings:
    """Every branch's cached vector for a fixed list of sequences, joined on read.

    Each branch keeps its own storage, dense `(n, w)` float32 or a CSR matrix for SAE
    branches, and `block` concatenates the requested rows' branches in spec order.
    """

    def __init__(self, sequences: Sequence[str], parts: Sequence[np.ndarray | sparse.csr_matrix]) -> None:
        # parts: (n, w_i) per branch; n = len(sequences).
        self.rows = {sequence: row for row, sequence in enumerate(sequences)}
        self.parts = list(parts)
        self.widths = tuple(int(part.shape[1]) for part in self.parts)
        self.block_rows = STANDARDIZER_BLOCK_ROWS

    @property
    def width(self) -> int:
        return sum(self.widths)  # W

    def __len__(self) -> int:
        return len(self.rows)

    def block(self, rows: Sequence[int] | np.ndarray) -> np.ndarray:
        """Dense joined rows for the given row indices."""
        # rows: (m,) requested sequence indices.
        rows = np.asarray(rows, dtype=np.int64)  # (m,)
        pieces = [
            part[rows].toarray() if sparse.issparse(part) else part[rows]  # (m, w)
            for part in self.parts
        ]
        return np.concatenate(pieces, axis=1).astype(np.float32, copy=False)  # (m, W)


class IndexedBranchEmbeddings:
    """Protify's existing dataset/collator interface over a verified canonical feature view.

    The view retains sequence metadata and native reader configuration, without a corpus matrix.
    Persistent views reuse mapped parts; spawned workers reopen their own verified handles.
    `join_branch_embeddings`, which reads Protify's own embedding stores, remains an explicitly
    separate path.
    """

    def __init__(self, view: FeatureView, names: Sequence[str] | None = None) -> None:
        self.view = view
        self.names = view.names if names is None else tuple(names)
        descriptors = [view.descriptor(name) for name in self.names]
        if not self.names or len(set(self.names)) != len(self.names) or any(
            descriptor.layout not in {"dense", "csr"} for descriptor in descriptors
        ):
            raise ValueError("Indexed branches must name distinct pooled dense/CSR features.")
        self.sequences = view.sequences
        self.rows = {sequence: row for row, sequence in enumerate(self.sequences)}
        self.widths = tuple(descriptor.width for descriptor in descriptors)
        row_bytes = max(4 * self.width, *(descriptor.width * (
            torch.empty((), dtype=descriptor.dtype).element_size() + (4 if descriptor.layout == "csr" else 0)
        ) for descriptor in descriptors))
        self.block_rows = min(view.budget.max_rows, view.budget.max_bytes // row_bytes)
        if self.block_rows < 1:
            raise ValueError("Joined feature width exceeds the reader's output budget.")

    @property
    def width(self) -> int:
        return sum(self.widths)

    def __len__(self) -> int:
        return len(self.sequences)

    def block(self, rows: Sequence[int] | np.ndarray) -> np.ndarray:
        # rows: (m,) requested sequence indices.
        rows = np.asarray(rows)  # (m,)
        if (rows.ndim != 1 or (rows.size and not np.issubdtype(rows.dtype, np.integer))
                or bool(np.any(rows < 0)) or bool(np.any(rows >= len(self)))):
            raise ValueError("Feature row indices must be one-dimensional, integral and in range.")
        sequences = tuple(self.sequences[int(row)] for row in rows)
        return self.view.pooled(sequences, self.names).numpy()  # (m,W), already FP32


def join_branch_embeddings(
    specs: Sequence[BranchSpec], sequences: Sequence[str], arguments: Mapping[str, object]
) -> tuple[JoinedBranchEmbeddings, tuple[int, ...]]:
    """Read every branch's cache for `sequences` and join them in spec order.

    `arguments` holds the run's settings as MainProcess keeps them; each branch's rows are read from
    its feature store, whether or not the run streams with `sql`. Returns the joined embeddings and
    each branch's width.
    """
    parts = []
    for spec in specs:
        path = branch_cache_path(spec, arguments)
        if not has_feature(path):
            raise FileNotFoundError(f"branch {spec.slug} has no cache at {path}.")

        reader = _branch_reader(spec, arguments)
        with reader._pooled_row_lookup(spec.model_name) as pooled_row:
            part, _ = stack_pooled_rows(_cached_rows(pooled_row, sequences, path), sparse_layout=spec.sparse)  # (n, w)
        parts.append(part)

    joined = JoinedBranchEmbeddings(sequences, parts)
    return joined, joined.widths


def _pair_rows(joined: JoinedBranchEmbeddings | IndexedBranchEmbeddings, pairs: np.ndarray, pair_feature_mode: str) -> np.ndarray:
    # pairs: (m, 2) row indices
    half_a, half_b = pair_vector_halves(
        joined.block(pairs[:, 0]), joined.block(pairs[:, 1]), pair_feature_mode
    )  # two of (m, W)
    return np.concatenate([half_a, half_b], axis=1)  # (m, 2W)


def _row_indices(joined: JoinedBranchEmbeddings | IndexedBranchEmbeddings, hf_dataset: Mapping[str, Sequence[Any]], ppi: bool) -> np.ndarray:
    if ppi:
        pairs = [
            [joined.rows[seq_a], joined.rows[seq_b]]
            for seq_a, seq_b in zip(hf_dataset["SeqA"], hf_dataset["SeqB"], strict=True)
        ]
        return np.array(pairs, dtype=np.int64).reshape(-1, 2)  # (n, 2)
    singles = [[joined.rows[sequence]] for sequence in hf_dataset["seqs"]]
    return np.array(singles, dtype=np.int64).reshape(-1, 1)  # (n, 1)


def fit_joined_standardizer(
    joined: JoinedBranchEmbeddings | IndexedBranchEmbeddings,
    hf_dataset: Mapping[str, Sequence[Any]],
    ppi: bool,
    pair_feature_mode: str = DEFAULT_PAIR_FEATURE_MODE,
    flip_pairs: bool = False,
) -> EmbeddingStandardizer:
    """A standardizer fit on the training split's rows, densified a block at a time.

    Pair rows are fit as the probe reads them, both orderings when `flip_pairs`, matching the
    streaming fit `TrainerMixin._fit_embedding_standardizer` runs over an ordinary cache.
    """
    scaler = StandardScaler()
    rows = _row_indices(joined, hf_dataset, ppi)  # (n, 1) or (n, 2)
    if ppi and flip_pairs:
        rows = np.concatenate([rows, rows[:, ::-1]], axis=0)  # (2n, 2)

    block_rows = min(STANDARDIZER_BLOCK_ROWS, joined.block_rows)
    for start in range(0, rows.shape[0], block_rows):
        chunk = rows[start:start + block_rows]  # (m, 1) or (m, 2)
        features = _pair_rows(joined, chunk, pair_feature_mode) if ppi else joined.block(chunk[:, 0])  # (m, 2W) or (m, W)
        scaler.partial_fit(features)

    return EmbeddingStandardizer(scaler.mean_, scaler.scale_)


class JoinedBranchDataset(TorchDataset):
    """Probe rows as `(row indices, label[, label_mask])` items.

    An item holds one row index into the joined embeddings, or two for a pair, swapped only in
    training under `random_pair_flipping`. `JoinedBranchCollator` turns a batch of them into
    the dense `embeddings` and `labels` Protify's pooled probes read.
    Multilabel items carry one `SparseLabelRow`, including observation status, until this
    collator creates the dense batch. Scalar tasks retain their tensor tuple representation.
    """

    def __init__(
        self,
        hf_dataset: Mapping[str, Sequence[Any]],
        joined: JoinedBranchEmbeddings | IndexedBranchEmbeddings,
        task_type: str,
        ppi: bool,
        random_pair_flipping: bool = False,
        train: bool = False,
        frozen_release: bool = False,
        cluster_draws_per_epoch: int | None = None,
        cluster_class_balance: bool = False,
    ) -> None:
        self.task_type = task_type
        self.ppi = ppi
        self.random_pair_flipping = random_pair_flipping
        self.train = train
        self.frozen_release = frozen_release
        self.sampling_plan = ClusterSamplingPlan.from_dataset(
            hf_dataset, cluster_draws_per_epoch, balance_classes=cluster_class_balance,
        ) if train and frozen_release else None
        self.rows = _row_indices(joined, hf_dataset, ppi)  # (n, 1) or (n, 2)
        self.labels = hf_dataset["labels"]
        columns = getattr(hf_dataset, "column_names", hf_dataset)
        self.label_masks = hf_dataset["label_mask"] if "label_mask" in columns else None
        # Scalar Arrow columns rebuild a view for each indexed read. Keep their exact
        # Python values once; multilabel targets retain the existing sparse path.
        if task_type != "multilabel":
            self.labels = list(self.labels)
            if self.label_masks is not None:
                self.label_masks = list(self.label_masks)
        self.observation_ids = hf_dataset["observation_id"] if "observation_id" in columns else None
        self.protein_ids = hf_dataset["protein_id"] if "protein_id" in columns else None
        self.cluster_ids = hf_dataset["cluster_id"] if "cluster_id" in columns else None
        if len(self.labels) != len(self.rows) or (self.label_masks is not None and len(self.label_masks) != len(self.labels)):
            raise ValueError("sequences, labels, and label masks must have the same row count")
        self.sparse_targets = SparseObservedTargets.from_rows(self.labels, self.label_masks) if task_type == "multilabel" else None
        # Frozen scalar targets cannot change between epochs. Validate and convert once,
        # preserving the original per-row dtype and mask arithmetic exactly.
        self.dense_targets = None
        if frozen_release and self.sparse_targets is None:
            self.dense_targets = [self._dense_label(index) for index in tqdm(
                range(len(self.labels)), desc="Prepare frozen scalar labels", mininterval=15)]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[tuple[int, ...], Tensor | SparseLabelRow] | tuple[tuple[int, ...], Tensor, Tensor]:
        rows = self.rows[index]  # (1,) or (2,)
        if self.ppi and self.train and self.random_pair_flipping and random.random() < 0.5:
            rows = rows[::-1]
        indices = tuple(int(row) for row in rows)
        if self.sparse_targets is not None:
            return indices, self.sparse_targets.row(index)  # sparse label indices and values: (observed,).
        if self.dense_targets is not None:
            return (indices, *(target.clone() for target in self.dense_targets[index]))  # fresh label/mask: () or (c,)
        return (indices, *self._dense_label(index))  # fresh label/mask: () or (c,)

    def _dense_label(self, index: int) -> tuple[Tensor, ...]:
        """Validate one scalar or dense target, retaining missing-label masking before dtype conversion."""
        dtype = torch.float if self.task_type in FLOAT_LABEL_TYPES else torch.long
        if self.label_masks is None:
            return (torch.tensor(self.labels[index], dtype=dtype),)  # label: () or (c,).
        labels = torch.as_tensor(self.labels[index])  # () or (c,), validate before integer conversion
        mask = torch.as_tensor(self.label_masks[index])  # () or (c,)
        if mask.dtype != torch.bool or mask.shape != labels.shape:
            raise ValueError("label_mask must be boolean with exactly the labels shape")
        if not torch.isfinite(labels[mask]).all():
            raise ValueError("observed labels must be finite")
        if dtype == torch.long and (labels[mask] != labels[mask].floor()).any():
            raise ValueError("observed class labels must be integers")
        return torch.where(mask, labels, 0).to(dtype), mask  # labels and mask: () or (c,)


class JoinedBranchCollator:
    """Densify one batch of joined rows, then standardize it as the scaler was fit."""

    def __init__(
        self,
        joined: JoinedBranchEmbeddings | IndexedBranchEmbeddings,
        ppi: bool,
        embedding_standardizer: EmbeddingStandardizer | None,
        pair_feature_mode: str = DEFAULT_PAIR_FEATURE_MODE,
    ) -> None:
        self.joined = joined
        self.ppi = ppi
        self.embedding_standardizer = embedding_standardizer
        self.pair_feature_mode = pair_feature_mode

    def __call__(self, batch: Sequence[tuple[tuple[int, ...], Tensor | SparseLabelRow] | tuple[tuple[int, ...], Tensor, Tensor]]) -> dict[str, Tensor]:
        # batch: (b,) observation records; label/mask () or (c,), or sparse labels (observed_i,).
        rows = np.array([item[0] for item in batch], dtype=np.int64)  # (b, 1) or (b, 2)
        if any(isinstance(item[1], SparseLabelRow) for item in batch):
            if not all(isinstance(item[1], SparseLabelRow) and len(item) == 2 for item in batch):
                raise ValueError("a batch cannot mix sparse and dense label rows")
            inputs = collate_sparse_labels([item[1] for item in batch])
        else:
            inputs = {"labels": torch.stack([item[1] for item in batch])}  # (b,) or (b, c)
            if any(len(item) == 3 for item in batch):
                if not all(len(item) == 3 for item in batch):
                    raise ValueError("a batch cannot mix implicit and explicit label masks")
                inputs["label_mask"] = torch.stack([item[2] for item in batch])  # (b,) or (b, c)
        if self.ppi:
            features = _pair_rows(self.joined, rows, self.pair_feature_mode)  # (b, 2W)
        else:
            features = self.joined.block(rows[:, 0])  # (b, W)
        if self.embedding_standardizer is not None:
            features = self.embedding_standardizer.transform_numpy(features)  # unchanged shape
        inputs["embeddings"] = torch.from_numpy(np.ascontiguousarray(features))  # (b, W) or (b, 2W)
        return inputs  # embeddings: (b, W) or (b, 2W); dense labels/masks: (b,) or (b, c).
