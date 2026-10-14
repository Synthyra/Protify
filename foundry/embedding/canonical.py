"""The canonical embedding stores: which bases exist, which streams each holds, and how a reader opens one.

Logan's embedding spec of 2026-10-01: special tokens are kept. A protein of l residues, after the N-terminal crop
at 2,046, is l + 2 rows in every per-token stream (row 0 CLS, rows 1..l residues, row l + 1 EOS), padding is the
only masked position, and every pooled stream covers all l + 2 rows. A row is keyed by the SHA-256 of the
uppercased, whitespace-stripped sequence, and a store only appends. One public Hub dataset repo and one Modal
volume hold each base; the random-vector and one-hot controls are never stored (`controls.SyntheticControlView`
generates them where they are read).

Streams of a trained base, with shapes per protein (symbols below):

| stream         | shape       | dtype | what                                                                  |
|----------------|-------------|-------|-----------------------------------------------------------------------|
| layer_hidden   | (l+2, d)    | bf16  | hidden state at the SAE layer (input to block `sae_layer`)             |
| final_hidden   | (l+2, d)    | bf16  | final normalized hidden state                                          |
| structural     | (l+2, 256)  | bf16  | ESMFold2 layer-mixture projection                                      |
| sae_codes      | (l+2, k)    | f32   | top-k SAE codes per token, with int32 indices                          |
| sae_max        | (c,) sparse | f32   | SAE codes max-pooled over all l + 2 rows                               |
| final_mean_var | (2d,)       | f32   | mean then population variance of the final state over all l + 2 rows   |

The random-init control `esmc_small_random` stores `layer_hidden`, `final_hidden` and `final_mean_var` only.

Two bases are not proteins (`family`): the ligand tower of Atlas-PLI, SELFIES-TED-LARGE, and the Annotation Transformer
that reads a protein's annotation terms. They keep the same rows (`l` units of the family plus CLS and EOS for the
ligand model, `l` terms for the annotation model, which has no special tokens), the same key by the SHA-256 of the row's
canonical text, the same pooled and token stores, and the same job. `encoders` says how a row is made and run.

A base has two kinds of store (Logan, 2026-10-01: per-token streams only for interaction data, everything else pooled-only,
because full token stores are too big to download). The **token** store holds every stream above for the proteins of
interaction datasets. The **pooled** store holds `sae_max` and `final_mean_var` (the control: `final_mean_var`) for every
protein of the list, under its own Hub repo and the `canonical_pooled_v2` directory of the base's Modal volume. A pooled
stream has the same descriptor and key as the token store's stream of that name, so a pooled store is seeded by copying.

**Profile stores** (`canonical_embeddings_v3`, Logan 2026-10-06) are computed under an embedding profile (`numerics`):
pinned container, numerics, fixed batch geometry and the continuous SAE readout, so a row is bitwise the same whatever
batch made it. They live in two repositories, `Synthyra/esmc_embeddings` for public models and
`Synthyra/private_embeddings` for private ones, one store per model and kind at `<model>/<kind>/` with a shard index
(`CanonicalBase.profile_destinations`; the profile is recorded in the store), mirrored at the same path on one Modal
volume per model, `canonical-embeddings-<model>` (`profile_volume`), so a job mounts only the model it reads (Logan,
2026-10-07). The v2 stores above are retired: nothing writes one, and the repositories and legacy volumes this registry
still names (`hub_repo`, `volume`, `pooled_hub_repo`) were deleted on 2026-10-08 once the profile stores replaced them,
so the names are provenance.

Symbols: l residues of a protein after the crop; d hidden width of the base; c SAE codebook (16384); k SAE codes
kept per token (64).
"""

from __future__ import annotations

import hashlib
import json
import torch

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .contracts import CANONICAL_MAX_RESIDUES, PROTEIN_NORMALIZATION, SequenceKey, normalize_sequence
from .readers import FeatureView, ReadBudget


if TYPE_CHECKING:
    from fastplms.embeddings.taps import Tap
    from fastplms.features import StoredFeature
    from torch import nn
    from .encoders import EncoderBinding
    from .numerics import EmbeddingProfile, HardwareClass
    from .persistence import FeatureExtractionContract
    from .sae_features import Readout


MANIFEST_FILE = "canonical.json"
MANIFEST_SCHEMA = "canonical_embeddings_v2"
MANIFEST_SCHEMA_V3 = "canonical_embeddings_v3"
MANIFEST_SCHEMAS = frozenset({MANIFEST_SCHEMA, MANIFEST_SCHEMA_V3})
FETCH_RECORD = "fetch.json"  # beside the manifest of a partial copy `hub.fetch_rows` made: its repository, revision, streams
PUBLIC_REPO = "Synthyra/esmc_embeddings"  # every public model's profile stores (Logan, 2026-10-06)
PRIVATE_REPO = "Synthyra/private_embeddings"  # every private model's: SELFIES-TED, the Annotation Transformers, Atlas, Dyad
PROFILE_VOLUME_PREFIX = "canonical-embeddings-"  # one Modal volume per model's profile stores (`profile_volume`)
SEQUENCE_LIST_REPO = "Synthyra/canonical_embedding_sequences"  # the proteins the stores cover: seq_sha256, sequence, full
VISIBILITY_PUBLIC = "public"
VISIBILITY_PRIVATE = "private"
NORMALIZATION_VERSION = PROTEIN_NORMALIZATION  # uppercase, ASCII whitespace removed, every residue kept
RANDOM_INIT_SEED = 20261001  # fixed and recorded; the random-init control must be the same network on every machine

STORE_TOKEN = "token"
STORE_POOLED = "pooled"
POOLED_VOLUME_DIRECTORY = "canonical_pooled_v2"
FAMILY_PROTEIN = "protein"  # the ESMC bases: FastPLMs' engine and taps
FAMILY_LIGAND = "ligand"  # SELFIES-TED-LARGE over canonical SMILES (`ligand`)
FAMILY_ANNOTATION = "annotation"  # the Annotation Transformer over sorted annotation term ids (`annotation`)
POOLED_NAMES = (
    "sae_max", "final_mean_var", "layer_mean_var", "structural_mean_var", "encoder_mean_var", "decoder_mean_var",
)  # the streams a pooled store keeps, whichever the base has

LAYER_HIDDEN = "layer_hidden"
FINAL_HIDDEN = "final_hidden"
STRUCTURAL = "structural"
SAE_CODES = "sae_codes"
SAE_MAX = "sae_max"
FINAL_MEAN_VAR = "final_mean_var"
LAYER_MEAN_VAR = "layer_mean_var"  # the SAE layer's mean and variance over all l + 2 rows; only a profile that names it
STRUCTURAL_MEAN_VAR = "structural_mean_var"  # the structural fusion's mean and variance over all l + 2 rows; likewise
ENCODER_HIDDEN = "encoder_hidden"
DECODER_HIDDEN = "decoder_hidden"
ENCODER_MEAN_VAR = "encoder_mean_var"
DECODER_MEAN_VAR = "decoder_mean_var"
TRAINED_STREAMS = (LAYER_HIDDEN, FINAL_HIDDEN, STRUCTURAL, SAE_CODES, SAE_MAX, FINAL_MEAN_VAR)
LIGAND_STREAMS = (ENCODER_HIDDEN, DECODER_HIDDEN, ENCODER_MEAN_VAR, DECODER_MEAN_VAR)
ANNOTATION_STREAMS = (FINAL_HIDDEN, FINAL_MEAN_VAR)
RANDOM_STREAMS = (LAYER_HIDDEN, FINAL_HIDDEN, FINAL_MEAN_VAR)
SAE_STREAMS = (STRUCTURAL, SAE_CODES, SAE_MAX)

# The engine files whose bytes decide a stored value: batch building, tokenization, the taps and their pooling, the token
# executor, and (besides `attention/` and the ESM++ model) nothing else. The store and writer layer (`features/`,
# `embeddings/storage.py`, `output.py`, `runner.py`, `feature_runs.py`, `identity.py`) decides where bytes go and how they
# are measured, so an edit there, such as the page-cache flush of 2026-10-01, never changes a key.
ENGINE_EMBEDDING_FILES = (
    "batches.py", "inputs.py", "pooling.py", "taps.py", "token_batches.py", "token_runs.py", "tokens.py", "types.py",
)

BATCH_FILL = 0.5  # the share of free device memory the activations of one batch may fill
BATCH_GROWTH_LIMIT = 8  # a fitted token budget is at most this multiple of the static one, until a GPU pilot shows more is safe
BATCH_TOKEN_QUANTUM = 4096  # fitted budgets round down to this, so free memory that differs a little between runs gives one policy


@dataclass(frozen=True, slots=True)
class CanonicalBase:
    """One stored base: its model, depth layer, streams, destinations and the batch sizes that fit its memory."""

    id: str
    label: str
    registry_id: str  # the registry model whose checkpoint (or, for the control, architecture) it uses; a Hub repo id for a non-protein family
    hidden_width: int  # d
    sae_layer: int | None  # hidden-state index of the SAE's depth layer; None for a base with no SAE
    streams: tuple[str, ...]
    hub_repo: str
    volume: str
    volume_directory: str  # top-level directory of the volume that holds this store
    max_sequences: int
    max_tokens: int  # padded token rows per batch
    activation_kib_per_token: int  # peak device KiB one padded token row adds beyond the weights; see CANONICAL_BASES
    sae_chunk_rows: int
    random_seed: int | None = None
    pooled_hub_repo: str = ""  # the Hub dataset repo of the pooled store
    family: str = FAMILY_PROTEIN
    store_model: str = ""  # the folder of its profile stores, `<store_model>/<kind>/`
    visibility: str = VISIBILITY_PUBLIC  # which of the two repositories holds its profile stores

    @property
    def has_sae(self) -> bool:
        return SAE_CODES in self.streams

    def profile_destinations(self, kind: str) -> tuple[str, str, str]:
        """The Hub repo, the store's folder inside it and on the volume, and the Modal volume of a profile store.

        One canonical place per model and kind, `<model>/<kind>`, whatever the profile (Logan, 2026-10-07: "I don't want
        a v1 or v2 in the repo... I want one canonical storage"): the profile is recorded in the store's manifest, its
        descriptors and the catalog, and a store computed under a new profile replaces the old one at the same place.
        """
        if kind not in {STORE_TOKEN, STORE_POOLED} or not self.store_model:
            raise ValueError(f"{self.label} has no profile store of kind {kind!r}.")
        folder = f"{self.store_model}/{kind}"
        return PUBLIC_REPO if self.visibility == VISIBILITY_PUBLIC else PRIVATE_REPO, folder, profile_volume(self.store_model)

    @property
    def pooled_streams(self) -> tuple[str, ...]:
        """The streams of the pooled store: the max-pooled SAE vector (trained bases) and the final mean and variance."""
        return tuple(name for name in self.streams if name in POOLED_NAMES)

    def store_streams(self, kind: str) -> tuple[str, ...]:
        """The streams a ``kind`` store ("token" or "pooled") of this base holds."""
        return self.pooled_streams if kind == STORE_POOLED else self.streams

    def store_destinations(self, kind: str) -> tuple[str, str]:
        """The Hub dataset repo and the volume's top-level directory of a ``kind`` store."""
        if kind == STORE_POOLED:
            return self.pooled_hub_repo, POOLED_VOLUME_DIRECTORY
        return self.hub_repo, self.volume_directory

    @property
    def model_initialization(self) -> dict[str, Any] | None:
        """How the control's weights were made, which enters its model pin; None for a released checkpoint."""
        if self.random_seed is None:
            return None
        return {"kind": "random_init", "seed": self.random_seed, "architecture": self.registry_id,
                "initializer": "ESMplusplus init_weights, normal(0, initializer_range)", "dtype": "bfloat16"}


# The static batch budgets are the conservative starting points of the 2026-10-01 runs. Their pilots peaked at 2.0 (control),
# 3.1 (300), 3.4 (600) and 14.0 GiB (6B) of a 96 GiB card with the kernels busy 44 to 51 percent (experiments/models/fastplms/
# 2026-10-01_canonical_embeddings_v1/launch.md). `activation_kib_per_token` is the pilot's peak reserved memory less the
# weights, over its padded token budget. The weights are estimated from the parameter counts, bf16 backbone plus fp32 SAE:
# 0.62, 0.74, 1.21 and 12.14 GiB. That gives 44, 76, 93 and 238 KiB per token row, rounded up here by 3 to 9 percent. It is
# one point on a line through zero, an upper bound while a fixed share of the peak does not grow with the budget;
# `choose_batch` sizes a new store's budget from it, and a GPU pilot is due to confirm it.
CANONICAL_BASES: dict[str, CanonicalBase] = {
    base.id: base for base in (
        CanonicalBase("esmc_small", "ESMC-300", "esmc_small", 960, 23, TRAINED_STREAMS,
                      "Synthyra/esmc_300_token_embeddings", "embeddings-esmc-300", "canonical_v2",
                      max_sequences=256, max_tokens=32768, activation_kib_per_token=80, sae_chunk_rows=1024,
                      pooled_hub_repo="Synthyra/esmc_300_pooled_embeddings", store_model="esmc_300"),
        CanonicalBase("esmc_large", "ESMC-600", "esmc_large", 1152, 27, TRAINED_STREAMS,
                      "Synthyra/esmc_600_token_embeddings", "embeddings-esmc-600", "canonical_v2",
                      max_sequences=128, max_tokens=24576, activation_kib_per_token=96, sae_chunk_rows=1024,
                      pooled_hub_repo="Synthyra/esmc_600_pooled_embeddings", store_model="esmc_600"),
        CanonicalBase("esmc_6b", "ESMC-6B", "esmc_6b", 2560, 60, TRAINED_STREAMS,
                      "Synthyra/esmc_6b_token_embeddings", "embeddings-esmc-6b", "canonical_v2",
                      max_sequences=64, max_tokens=8192, activation_kib_per_token=256, sae_chunk_rows=512,
                      pooled_hub_repo="Synthyra/esmc_6b_pooled_embeddings", store_model="esmc_6b"),
        CanonicalBase("esmc_small_random", "ESMC-300 random init", "esmc_small", 960, 23, RANDOM_STREAMS,
                      "Synthyra/esmc_300_random_token_embeddings", "embeddings-esmc-300-random", "canonical_v2",
                      max_sequences=256, max_tokens=32768, activation_kib_per_token=48, sae_chunk_rows=1024,
                      random_seed=RANDOM_INIT_SEED, pooled_hub_repo="Synthyra/esmc_300_random_pooled_embeddings",
                      store_model="esmc_300_random"),
    )
}
# The towers of Atlas that are not protein models (Logan, 2026-10-02), each with a private token and pooled repo. Rows are
# canonical SMILES for the ligand model and sorted term ids for the annotation model (`encoders`). The activation figures are
# first estimates for fp32 models, to be replaced by the pilot's measurement.
ENCODER_BASES: dict[str, CanonicalBase] = {
    base.id: base for base in (
        CanonicalBase("selfies_ted_large", "SELFIES-TED-LARGE", "lhallee/SELFIES-TED-LARGE", 1024, None, LIGAND_STREAMS,
                      "Synthyra/selfies_ted_large_token_embeddings", "", "canonical_v2",
                      max_sequences=1024, max_tokens=32768, activation_kib_per_token=128, sae_chunk_rows=0,
                      pooled_hub_repo="Synthyra/selfies_ted_large_pooled_embeddings", family=FAMILY_LIGAND,
                      store_model="selfies_ted_large", visibility=VISIBILITY_PRIVATE),
        # The Annotation Transformer trained on the experimental annotation vocabulary release (CAMP). Its pins are in
        # `encoder_pins`; the tests run it as a tiny model of this id.
        CanonicalBase("at_experimental_v1", "AT-experimental-v1", "Synthyra/AT-experimental-v1", 512, None, ANNOTATION_STREAMS,
                      "Synthyra/at_experimental_v1_token_embeddings", "", "canonical_v2",
                      max_sequences=1024, max_tokens=65536, activation_kib_per_token=48, sae_chunk_rows=0,
                      pooled_hub_repo="Synthyra/at_experimental_v1_pooled_embeddings", family=FAMILY_ANNOTATION,
                      store_model="at_experimental_v1", visibility=VISIBILITY_PRIVATE),
        # The same architecture trained on the v2 release's 25,119 terms (2026-10-05); a v2 entry's ids mean other terms than v1's.
        CanonicalBase("at_experimental_v2", "AT-experimental-v2", "Synthyra/AT-experimental-v2", 512, None, ANNOTATION_STREAMS,
                      "Synthyra/at_experimental_v2_token_embeddings", "", "canonical_v2",
                      max_sequences=1024, max_tokens=65536, activation_kib_per_token=48, sae_chunk_rows=0,
                      pooled_hub_repo="Synthyra/at_experimental_v2_pooled_embeddings", family=FAMILY_ANNOTATION,
                      store_model="at_experimental_v2", visibility=VISIBILITY_PRIVATE),
    )
}
_ALIASES = {
    "300": "esmc_small", "esmc-300": "esmc_small", "esmc_300": "esmc_small",
    "600": "esmc_large", "esmc-600": "esmc_large", "esmc_600": "esmc_large",
    "6b": "esmc_6b", "esmc-6b": "esmc_6b", "esmc_6b": "esmc_6b",
    "random": "esmc_small_random", "300_random": "esmc_small_random", "esmc_300_random": "esmc_small_random",
    "selfies": "selfies_ted_large", "selfies_ted": "selfies_ted_large", "selfies-ted-large": "selfies_ted_large",
    "ligand": "selfies_ted_large", "at": "at_experimental_v1", "annotation": "at_experimental_v1",
    "at-experimental-v1": "at_experimental_v1", "at_experimental": "at_experimental_v1",
    "at_v2": "at_experimental_v2", "at-experimental-v2": "at_experimental_v2",
}


def profile_volume(store_model: str) -> str:
    """The Modal volume holding one model's profile stores, each kind at its repository path:
    `canonical-embeddings-esmc-600` for `esmc_600`, so a job reading ESMC-600 mounts that volume alone."""
    return PROFILE_VOLUME_PREFIX + store_model.replace("_", "-")


def canonical_base(name: str) -> CanonicalBase:
    """The stored base called ``name``: its id (`esmc_small`) or a short name (`300`, `600`, `6b`, `random`, `selfies`, `at`)."""
    key = _ALIASES.get(name.lower(), name)
    bases = {**CANONICAL_BASES, **ENCODER_BASES}
    if key not in bases:
        raise ValueError(f"No canonical base {name!r}; the bases are {sorted(bases)}.")
    return bases[key]


@dataclass(frozen=True, slots=True)
class BatchPolicy:
    """The sequence and padded-token budget of one batch, and where it came from."""

    max_sequences: int
    max_tokens: int  # padded token rows per batch
    source: str  # "explicit", "store", "fitted" or "static"


def recorded_batch(root: Path) -> tuple[int, int] | None:
    """The (max_sequences, max_tokens) every stream of the store at ``root`` was started with; None for a new store.

    The batch policy is part of each stream's key, so a store keeps the budget it started with: another budget would
    open new, empty streams beside the old ones, and `write_manifest` refuses it.
    """
    root = Path(root)
    if not (root / MANIFEST_FILE).is_file():
        return None
    refuse_profile_store(root, "a recorded token budget")
    policies: set[tuple[int, int]] = set()
    for record in read_manifest(root)["streams"].values():
        descriptor = root / record["key"] / "feature.json"
        if descriptor.is_file():
            batch = json.loads(descriptor.read_text(encoding="utf-8"))["descriptor"]["execution"]["batch_policy"]
            policies.add((batch["max_sequences"], batch["max_tokens"]))
    if len(policies) > 1:
        raise ValueError(f"The streams of {root} were started with different batch policies: {sorted(policies)}.")
    return next(iter(policies), None)


def choose_batch(
    base: CanonicalBase, root: Path, *, explicit: tuple[int | None, int | None], free_bytes: int | None, fill: float,
    pooled: bool = False,
) -> BatchPolicy:
    """The batch budget of a run: the flags, else the store's own, else sized to free device memory, else the base's.

    ``explicit`` is the (max_sequences, max_tokens) the operator named; naming either keeps the base's static budget
    for the other, which is the behavior before memory-aware sizing. A token store that exists keeps the policy it
    started with (`recorded_batch`). A new store on a device with ``free_bytes`` of free memory takes ``fill`` of it for
    activations, at ``activation_kib_per_token`` each: tokens round down to `BATCH_TOKEN_QUANTUM`, stay within
    `BATCH_GROWTH_LIMIT` times the static budget, and never fall below it. Sequences grow in the same ratio. Without a
    device reading, or with ``fill`` 0, the static budget stands.

    A ``pooled`` run writes no per-token row, so nothing but the key ties it to the store's budget: it sizes its batches
    to memory whatever the store holds (`key_batch` gives the budget the stream keys keep).
    """
    max_sequences, max_tokens = explicit
    if max_sequences is not None or max_tokens is not None:
        return BatchPolicy(max_sequences or base.max_sequences, max_tokens or base.max_tokens, "explicit")
    recorded = None if pooled else recorded_batch(root)
    if recorded is not None:
        return BatchPolicy(*recorded, "store")
    if free_bytes is None or fill <= 0:
        return BatchPolicy(base.max_sequences, base.max_tokens, "static")
    fitted = int(fill * free_bytes) // (base.activation_kib_per_token * 1024)
    tokens = min(fitted // BATCH_TOKEN_QUANTUM * BATCH_TOKEN_QUANTUM, base.max_tokens * BATCH_GROWTH_LIMIT)
    tokens = max(tokens, base.max_tokens)
    return BatchPolicy(base.max_sequences * tokens // base.max_tokens, tokens, "fitted")


def key_batch(root: Path, run: BatchPolicy, *, pooled: bool) -> BatchPolicy:
    """The budget a run's stream keys carry: the store's recorded one, else the run's own.

    A token store's rows come from the batches its keys name. A pooled store is seeded from a token store's streams and
    keeps their keys, so a later run may batch larger than the budget those keys state (a pooled value is the same
    reduction in any batch, within the 0.02 that `job verify` allows); each segment records the budget that made it.
    """
    recorded = recorded_batch(root) if pooled else None
    return run if recorded is None else BatchPolicy(*recorded, "store")


@dataclass(frozen=True, slots=True)
class RecordedEvidence:
    """Where a store's first rows were made, as its descriptors state it: the engine files, the libraries, the machine."""

    engine_sha256: str
    software: dict[str, str]
    hardware: dict[str, Any]


def recorded_evidence(root: Path) -> RecordedEvidence | None:
    """The evidence every stream of the store at ``root`` was keyed with; None for a new store.

    A store's keys are fixed when it starts. The engine digest, library versions and machine are evidence of where the
    first rows came from, not of what a value is, so a later run, with other engine files or on another machine,
    keeps them (`inherit_recorded_evidence`) and derives the keys the store has. Streams that disagree are an error.
    """
    root = Path(root)
    if not (root / MANIFEST_FILE).is_file():
        return None
    refuse_profile_store(root, "inheriting recorded evidence")
    found: set[str] = set()
    for record in read_manifest(root)["streams"].values():
        execution = json.loads((root / record["key"] / "feature.json").read_text(encoding="utf-8"))["descriptor"]["execution"]
        found.add(json.dumps({
            "engine_sha256": execution["engine_sha256"], "software": execution["software"],
            "hardware": execution["numerical_policy"]["hardware"],
        }, sort_keys=True))
    if len(found) != 1:
        raise ValueError(f"The streams of {root} were keyed with {len(found)} different engine and machine records; expected one.")
    return RecordedEvidence(**json.loads(next(iter(found))))


def inherit_recorded_evidence(runtime: CanonicalRuntime, root: Path) -> CanonicalRuntime:
    """``runtime`` with the engine digest, libraries and machine of the store at ``root``, so its keys are the store's.

    What decides a key is the base's weights and pins, layer, stream definitions, crop, special-token policy, dtypes,
    numerical settings, batch budget and reducer files. The engine digest (every file the run executes, writer
    included), the library versions and the hardware only say where the first rows were made, and a store that
    keeps changing them would never accept another delta. A new store keeps what this run measured. The measured
    values stay in the receipt's evidence.
    """
    recorded = recorded_evidence(root)
    if recorded is None:
        return runtime
    capture = {**runtime.capture, "engine_sha256": recorded.engine_sha256,
               "recorded_environment": {"software": recorded.software, "hardware": recorded.hardware}}
    evidence = {**runtime.evidence, "key_evidence": {"source": "store", "engine_sha256": recorded.engine_sha256}}
    return replace(runtime, capture=capture, evidence=evidence)


def changed_fields(recorded: Any, derived: Any, path: str = "") -> list[str]:
    """The dotted paths where two descriptors differ, each with both values (digests cut to 12 characters)."""
    if isinstance(recorded, Mapping) and isinstance(derived, Mapping):
        paths: list[str] = []
        for name in sorted(recorded.keys() | derived.keys()):
            paths += changed_fields(recorded.get(name), derived.get(name), f"{path}.{name}" if path else name)
        return paths
    if recorded == derived:
        return []

    def short(value: Any) -> str:
        text = json.dumps(value, sort_keys=True)
        return text[:14] + "..." if len(text) > 64 else text

    return [f"{path}: store {short(recorded)}, derived {short(derived)}"]


def check_keys_match_store(root: Path, features: Mapping[str, StoredFeature]) -> None:
    """Refuse, naming the changed descriptor fields, when the keys just derived are not the ones the store at ``root`` has.

    A new store has nothing to match. The same refusal later reads "describes a different store" and says nothing
    about which field moved, which cost a debugging session once.
    """
    root = Path(root)
    if not (root / MANIFEST_FILE).is_file():
        return
    streams = read_manifest(root)["streams"]
    problems: list[str] = []
    for name, feature in features.items():
        record = streams.get(name)
        if record is None:
            problems.append(f"{name}: the store has no such stream")
        elif feature.key != record["key"]:
            recorded = json.loads((root / record["key"] / "feature.json").read_text(encoding="utf-8"))["descriptor"]
            problems.append(f"{name} ({record['key']} against {feature.key}): "
                            + "; ".join(changed_fields(recorded, feature.descriptor)))
    if problems:
        raise ValueError(f"The keys this run derives are not the keys of {root}. Nothing was written. "
                         + " | ".join(problems))


def sequence_inventory(sequences: Sequence[str]) -> dict[str, SequenceKey]:
    """Each protein's normalized text with its identity: SHA-256 of that text, its length, and the policy.

    The text is made by `normalize_sequence` (ASCII whitespace removed, uppercased), once and here, so the key is
    the census and dataset-file key and two spellings of one protein collapse to one entry. The dictionary is
    keyed by the normalized text; the reader also accepts a raw spelling when asking for rows.
    """
    inventory: dict[str, SequenceKey] = {}
    for sequence in sequences:
        text = normalize_sequence(sequence)
        if text not in inventory:
            inventory[text] = SequenceKey(
                hashlib.sha256(text.encode("ascii")).hexdigest(), len(text), NORMALIZATION_VERSION,
            )
    return inventory


def file_identity(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Bind the exact implementation files by SHA-256, under stable logical names."""
    files = {}
    for name, path in sorted(paths.items()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        files[name] = digest.hexdigest()
    return {"sha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "files": files}


def canonical_taps(
    base: CanonicalBase, *, sae: nn.Module | None, weights: Mapping[str, torch.Tensor] | None, pooled: bool = False,
    readout: Readout | None = None, sae_chunk_rows: int | None = None, structural_dtype: torch.dtype = torch.bfloat16,
    extra_streams: Sequence[str] = (),
) -> tuple[Tap, ...]:
    """The taps that fill a base's streams, every one over all l + 2 token rows; ``pooled``: the pooled streams only.

    A ``readout`` builds the profile's SAE taps (its readout, fixed encoder chunks of ``sae_chunk_rows`` rows); without
    one the SAE taps are the plain top-k taps of the v2 stores. The structural projection computes in
    ``structural_dtype``, the backbone's dtype under a profile's precision mode. ``extra_streams`` are the profile's
    additions (`EmbeddingProfile.extra_streams`), to both stores of a trained base: `layer_mean_var` the SAE layer's
    mean and variance, `structural_mean_var` the structural fusion's (which needs ``weights`` in a pooled store too).
    """
    # weights: (n,) layer_logits, (d,) projection.0.weight and .bias, (256, d) projection.1.weight; n = depth + 1
    from fastplms.embeddings import HiddenTap

    from .structural import structural_tap
    from .taps import sae_residue_tap, sae_tap

    chunk_rows = base.sae_chunk_rows if sae_chunk_rows is None else sae_chunk_rows
    unknown = sorted(set(extra_streams) - {LAYER_MEAN_VAR, STRUCTURAL_MEAN_VAR})
    if unknown:
        raise ValueError(f"No canonical tap computes the extra streams {unknown}.")
    moments: list[Tap] = []  # the profile's extra streams, for the trained bases only, in the profile's order
    for name in extra_streams if base.has_sae else ():
        if name == LAYER_MEAN_VAR:  # (2d,) the SAE layer's mean and variance over all l+2 rows
            moments.append(HiddenTap(LAYER_MEAN_VAR, base.sae_layer, ("mean", "var"), dtype=torch.float32))
        else:  # (2 * 256,) the structural fusion's mean and variance over all l+2 rows, in float32
            if weights is None:
                raise ValueError(f"{base.label} needs its structural weights for {STRUCTURAL_MEAN_VAR}.")
            moments.append(structural_tap(STRUCTURAL_MEAN_VAR, weights, compute_dtype=structural_dtype,
                                          keep_special_tokens=True, pooling=("mean", "var")))
    if pooled:
        pooled_taps: list[Tap] = []
        if base.has_sae:
            if sae is None:
                raise ValueError(f"{base.label} needs its SAE.")
            pooled_taps.append(sae_tap(SAE_MAX, sae, base.sae_layer, ("max",), chunk_rows=chunk_rows,
                                       keep_special_tokens=True, readout=readout))  # (c,) sparse, max over all l+2 rows
        pooled_taps.append(HiddenTap(FINAL_MEAN_VAR, -1, ("mean", "var"), dtype=torch.float32))  # (2d,) over all l+2 rows
        return (*pooled_taps, *moments)
    taps: list[Tap] = [
        HiddenTap(LAYER_HIDDEN, base.sae_layer, dtype=torch.bfloat16),  # (l+2, d) bf16 at the SAE layer
        HiddenTap(FINAL_HIDDEN, -1, dtype=torch.bfloat16),  # (l+2, d) bf16 after the final norm
    ]
    if base.has_sae:
        if sae is None or weights is None:
            raise ValueError(f"{base.label} needs its SAE and structural weights.")
        taps += [
            structural_tap(STRUCTURAL, weights, compute_dtype=structural_dtype, keep_special_tokens=True),  # (l+2, 256)
            sae_residue_tap(SAE_CODES, sae, base.sae_layer, chunk_rows=chunk_rows, keep_special_tokens=True,
                            readout=readout),
            sae_tap(SAE_MAX, sae, base.sae_layer, ("max",), chunk_rows=chunk_rows, keep_special_tokens=True,
                    readout=readout),
        ]
    taps.append(HiddenTap(FINAL_MEAN_VAR, -1, ("mean", "var"), dtype=torch.float32))  # (2d,) over all l+2 rows
    return (*taps, *moments)


@dataclass(frozen=True, slots=True)
class CanonicalRuntime:
    """The loaded base and taps, the capture options they were measured with, and the evidence of their bytes."""

    base: CanonicalBase
    model: nn.Module
    taps: tuple[Tap, ...]
    capture: Mapping[str, Any]
    evidence: Mapping[str, Any]
    encoder: EncoderBinding | None = None  # a non-protein family's tokenizer and forward; its taps are empty


def random_init_model(config: Any, seed: int) -> nn.Module:
    """ESM++ built from ``config`` with every initializer draw seeded by ``seed``, in FP32: the random control's network.

    The draw follows the CPU's random stream, which no release promises to repeat on another machine or library, so a
    profile run never draws: it loads the file `save_random_weights` wrote once (`build_runtime`'s ``random_weights``).
    """
    from fastplms.models.esm_plusplus.modeling_esm_plusplus import ESMplusplusModel

    torch.manual_seed(seed)
    return ESMplusplusModel(config)


def save_random_weights(model: nn.Module, path: Path, *, base: CanonicalBase) -> str:
    """Write ``model``'s FP32 weights as safetensors at ``path``, never over a file, and return the file's SHA-256.

    The metadata names the draw (`base.model_initialization`, with the file's own dtype); nothing in the file depends on
    when or where it was written, so the same draw saves to the same bytes.
    """
    from safetensors.torch import save_file

    from foundry.digests import sha256_file

    weights = {name: tensor.detach().clone().contiguous() for name, tensor in model.state_dict().items()}
    if {tensor.dtype for tensor in weights.values()} != {torch.float32}:
        raise ValueError("The random control is saved as its FP32 draw; this model holds other dtypes.")
    if Path(path).exists():
        raise FileExistsError(f"{path} exists; a saved draw is never replaced.")
    record = {"base": base.id, **(base.model_initialization or {}), "dtype": "float32"}
    # One metadata entry, sorted JSON: safetensors writes several in a random order, so the bytes would follow the process.
    save_file(weights, str(path), metadata={"random_init": json.dumps(record, sort_keys=True)})
    return sha256_file(path)


def build_runtime(
    base: CanonicalBase, cache: Path, *, device: str = "cuda", fetch: bool = False,
    max_sequences: int | None = None, max_tokens: int | None = None, window: int = 65536, pooled: bool = False,
    random_weights: Path | None = None, profile: EmbeddingProfile | None = None, hardware: HardwareClass | None = None,
) -> CanonicalRuntime:
    """Verify the pinned checkpoints, load the base and its taps, and bind the options a contract measures.

    Without a ``profile`` this runs BF16 SDPA with an FP32 SAE and FP32 pooling and changes no global Torch setting
    (the v2 stores). With one, the caller has applied and checked the profile's numerics (`numerics.apply_profile`) and
    names the device's ``hardware`` class: the SAE taps take the profile's readout and chunk rows, and the capture binds
    the profile, the class and the base's geometry instead of a batch budget. The random-init control builds the
    ESMC-300 architecture from the pinned config and seeds its initializer with `base.random_seed`; the seed enters the
    model pin, and under a profile it must load its saved checkpoint (``random_weights``), never a fresh draw.
    ``fetch`` downloads the pinned files into ``cache`` first. A missing or altered file fails with its path before
    anything loads. A ``pooled`` runtime builds only the pooled taps, and loads the structural projection only for a
    profile that names `structural_mean_var`.
    """
    if random_weights is not None and (base.family != FAMILY_PROTEIN or base.random_seed is None):
        raise ValueError("Saved random weights require the random protein control.")
    if profile is not None:
        if hardware is None:
            raise ValueError("A profile runtime needs the device's hardware class.")
        if max_sequences is not None or max_tokens is not None:
            raise ValueError("A profile runtime takes its batch shapes from the profile's geometry.")
        if base.random_seed is not None and random_weights is None:
            raise ValueError("Under a profile the random control loads its saved checkpoint (random_weights); it is never redrawn.")
    if base.family != FAMILY_PROTEIN:
        if profile is not None:
            raise ValueError(f"{base.label} has no profile geometry yet; its encoder rows follow the v2 path.")
        from .encoders import build_encoder_runtime

        return build_encoder_runtime(base, cache, device=device, fetch=fetch, max_sequences=max_sequences,
                                     max_tokens=max_tokens, window=window, pooled=pooled)

    import fastplms

    from fastplms.models.esm_plusplus.modeling_esm_plusplus import ESMplusplusConfig, ESMplusplusModel
    from fastplms.registry import get_model_registry

    from .dependencies import (
        fetch_snapshot, load_sae, resolve_dependencies, structural_weights, verified_snapshot,
    )

    dependencies = resolve_dependencies(get_model_registry(), base.registry_id, sae_layer=base.sae_layer)
    extra_streams = () if profile is None else profile.extra_streams
    # A token store's `structural` and a profile's `structural_mean_var` both need the structural projection.
    structured = base.has_sae and (not pooled or STRUCTURAL_MEAN_VAR in extra_streams)
    sources = {"base": dependencies.base, "tokenizer": dependencies.tokenizer}
    if base.has_sae:
        sources["sae"] = dependencies.sae.checkpoint
        if structured:
            sources["structural"] = dependencies.structural
    snapshots, pins = {}, {}
    for name, source in sources.items():
        if fetch:
            fetch_snapshot(source, cache)
        snapshots[name], pins[name] = verified_snapshot(source, cache)

    backbone_dtype = torch.bfloat16 if profile is None else profile.backbone_dtype(base.id)
    initialization = base.model_initialization
    if base.random_seed is None:
        model = ESMplusplusModel.from_pretrained(
            str(snapshots["base"]), torch_dtype=backbone_dtype, attn_implementation="sdpa", local_files_only=True,
        )
    else:
        # Random weights from the ESMC-300 config: same shapes, no learning. The seed fixes every initializer draw.
        model = random_init_model(ESMplusplusConfig.from_pretrained(str(snapshots["base"])), base.random_seed).to(backbone_dtype)
        if random_weights is not None:
            from safetensors.torch import load_file

            from foundry.digests import sha256_file

            saved = load_file(str(random_weights), device="cpu")
            if {tensor.dtype for tensor in saved.values()} != {torch.float32}:
                raise ValueError(f"{random_weights} is not the FP32 draw `save_random_weights` writes.")
            model.load_state_dict(saved, strict=True)  # cast to the backbone dtype, as the draw itself was
            # The saved file, not the seed, is the network now: its digest enters every key the control's rows carry.
            initialization = {**initialization, "dtype": "float32", "weights_sha256": sha256_file(random_weights)}
        model._reset_rotary_embeddings()
        model.set_attn_implementation("sdpa")
    model = model.to(device).eval().requires_grad_(False)
    if model.config.hidden_size != base.hidden_width or model.config.hidden_size != dependencies.sae.input_width:
        raise ValueError("Loaded backbone width differs from the canonical profile.")
    model.config.fastplms_checkpoint_repo_id = dependencies.base.repo_id
    model.config.fastplms_checkpoint_revision = dependencies.base.revision

    sae = weights = None
    if base.has_sae:
        sae = load_sae(dependencies.sae, snapshots["sae"], device=device)
        if structured:
            weights = {name: value.to(device) for name, value in structural_weights(
                snapshots["structural"], width=model.config.hidden_size, depth=model.config.num_hidden_layers).items()}
    taps = canonical_taps(
        base, sae=sae, weights=weights, pooled=pooled, readout=None if profile is None else profile.readout,
        sae_chunk_rows=None if profile is None else profile.sae_chunk_rows(base.id), structural_dtype=backbone_dtype,
        extra_streams=extra_streams,
    )

    # Bind the loaded package's own files under stable logical names, never an unused tree copy.
    package = Path(fastplms.__file__).resolve().parent
    extension = Path(__file__).resolve().parent
    model_directory = package / "models/esm_plusplus"
    engine = file_identity({
        f"models/fastplms/{path.relative_to(package).as_posix()}": path
        for path in (*(package / "embeddings" / name for name in ENGINE_EMBEDDING_FILES),
                     *(package / "attention").glob("*.py"), model_directory / "modeling_esm_plusplus.py")
    })
    sae_files = file_identity({
        "foundry/embedding/taps.py": extension / "taps.py",
        "foundry/embedding/sae_features.py": extension / "sae_features.py",
        "models/fastplms/models/esm_plusplus/modeling_esm_plusplus_sae.py": model_directory / "modeling_esm_plusplus_sae.py",
    })
    rows = file_identity({
        "models/fastplms/embeddings/token_batches.py": package / "embeddings/token_batches.py",
        "models/fastplms/embeddings/tokens.py": package / "embeddings/tokens.py",
    })
    # Named apart from the `pooled` argument: reusing that name (2026-10-02, when pooled stores arrived) sent every token run
    # down the pooled branches below and stopped it, while the published token keys were made before that change.
    pooling_files = file_identity({"models/fastplms/embeddings/pooling.py": package / "embeddings/pooling.py"})
    reducers: dict[str, Any] = {LAYER_HIDDEN: rows, FINAL_HIDDEN: rows, FINAL_MEAN_VAR: pooling_files, LAYER_MEAN_VAR: pooling_files}
    options: dict[str, Any] = {}
    if base.has_sae:
        reducers.update({
            SAE_CODES: sae_files, SAE_MAX: sae_files,
            STRUCTURAL: file_identity({"foundry/embedding/structural.py": extension / "structural.py"}),
            STRUCTURAL_MEAN_VAR: file_identity({"foundry/embedding/structural.py": extension / "structural.py",
                                                "models/fastplms/embeddings/pooling.py": package / "embeddings/pooling.py"}),
        })
        sae_source = {"repository": dependencies.sae.checkpoint.repo_id, "revision": dependencies.sae.checkpoint.revision}
        sae_names = [tap.name for tap in taps if tap.name in {SAE_CODES, SAE_MAX}]
        structural_names = [tap.name for tap in taps if tap.name in {STRUCTURAL, STRUCTURAL_MEAN_VAR}]
        structural_source = {} if not structural_names else {
            "repository": dependencies.structural.repo_id, "revision": dependencies.structural.revision}
        options.update(
            sae_sources={name: sae_source for name in sae_names}, sae_layers={name: sae for name in sae_names},
            structural_sources={name: structural_source for name in structural_names},
            structural_weights={name: weights for name in structural_names},
        )
    if profile is not None:
        options.update(profile=profile, hardware_class=hardware, geometry=profile.geometry(base.id))
    else:
        options.update(batch_size=max_sequences or base.max_sequences, batch_window_size=window,
                       max_tokens_per_batch=max_tokens or base.max_tokens)
    capture = dict(
        model_source={"repository": dependencies.base.repo_id, "revision": dependencies.base.revision},
        tokenizer_source={"repository": dependencies.tokenizer.repo_id, "revision": dependencies.tokenizer.revision},
        engine_sha256=engine["sha256"],
        reducer_sha256={tap.name: reducers[tap.name]["sha256"] for tap in taps},
        normalization_version=NORMALIZATION_VERSION, max_length=CANONICAL_MAX_RESIDUES, truncate=True,
        dtype=None, keep_special_tokens=True, model_initialization=initialization, **options,
    )
    return CanonicalRuntime(base, model, taps, capture, {"checkpoints": pins, "engine": engine, "reducers": reducers})


def check_model_pin(
    root: str | Path, model_state_sha256: str, *, expected: str | None = None,
) -> None:
    """Refuse to embed with weights that differ from the ones the store was first built with.

    The random-init control is drawn from a seed, and a CPU's random draw is not promised to be bit-identical
    on another machine. The weights are never redrawn to match: the first build records their SHA-256 in the
    manifest (`model_state_sha256`), and every later command, on any machine, compares before it embeds.
    ``expected`` is a digest the operator carries from another machine's receipt, for a store not yet started.
    """
    if expected is not None and expected != model_state_sha256:
        raise ValueError(
            f"The model's weights hash to {model_state_sha256}, not the expected {expected}. A random-init draw "
            "differs between these machines; run on the machine that made the first build."
        )
    path = Path(root) / MANIFEST_FILE
    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8")).get("model_state_sha256")
        if recorded != model_state_sha256:
            raise ValueError(
                f"{path} records weights {recorded}, but this machine built {model_state_sha256}. Nothing is "
                "embedded: the store's rows belong to the recorded weights, so run where they were first built."
            )


def stream_manifest(base: CanonicalBase, contract: FeatureExtractionContract, kind: str = STORE_TOKEN) -> dict[str, Any]:
    """The record of what a ``kind`` store holds: each stream's key, layout, width and dtype, and the model's bytes.

    A token manifest has exactly the fields of the published ones. A pooled manifest adds `store_kind` and names the
    pooled repo and directory.
    """
    features: Mapping[str, StoredFeature] = contract.stored_features
    hub_repo, volume_directory = base.store_destinations(kind)
    policy: Mapping[str, Any] = {
        "special_tokens": "kept",
        "row_layout": "row 0 CLS, rows 1..l residues, row l+1 EOS; padding is the only masked position",
        "max_residues": CANONICAL_MAX_RESIDUES, "normalization": NORMALIZATION_VERSION, "sae_layer": base.sae_layer,
    } if base.family == FAMILY_PROTEIN else contract.manifest_policy  # a family says its own rows, crop and pins
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA, "base": base.id, "label": base.label, **policy,
        "model_state_sha256": contract.model_state_sha256, "random_init": base.model_initialization,
        "hub_repo": hub_repo, "volume": base.volume, "volume_directory": volume_directory,
        "streams": {
            name: {
                "key": feature.key, "layout": feature.layout, "width": feature.width,
                "dtype": str(feature.dtype).removeprefix("torch."), "sparse_count": feature.sparse_count,
                "descriptor_sha256": hashlib.sha256(
                    json.dumps(feature.payload(), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            }
            for name, feature in features.items()
        },
    }
    if kind == STORE_POOLED:
        manifest["store_kind"] = STORE_POOLED
    return manifest


def profile_manifest(
    base: CanonicalBase, contract: FeatureExtractionContract, kind: str, profile: EmbeddingProfile, hardware: HardwareClass,
) -> dict[str, Any]:
    """The record of a profile store: the streams as `stream_manifest` states them, the profile and class, and where it lives.

    ``path_in_repo`` is the store's folder in its repository (`<model>/<kind>`) and on its Modal volume. ``random_init`` is
    the record the keys carry, so the random control's names its saved weights file by SHA-256.
    """
    if contract.geometry is None:
        raise ValueError("A profile store's manifest needs a profile contract.")
    hub_repo, folder, volume = base.profile_destinations(kind)
    manifest = stream_manifest(base, contract, kind)
    initialization = contract.model_initialization
    manifest.update(schema=MANIFEST_SCHEMA_V3, hub_repo=hub_repo, path_in_repo=folder, volume=volume, volume_directory=folder,
                    store_kind=kind, visibility=base.visibility, store_model=base.store_model, profile=profile.binding(),
                    hardware_class=hardware.record(), random_init=None if initialization is None else dict(initialization))
    return manifest


def manifest_profile(manifest: Mapping[str, Any]) -> dict[str, str] | None:
    """The profile a store was computed under, `{"id", "sha256"}`; None for a v2 store."""
    return dict(manifest["profile"]) if manifest.get("schema") == MANIFEST_SCHEMA_V3 else None


def manifest_kind(manifest: Mapping[str, Any]) -> str:
    """"pooled" for a pooled store's manifest; every manifest written before pooled stores existed is a token store."""
    return str(manifest.get("store_kind", STORE_TOKEN))


def pooled_manifest(base: CanonicalBase, token_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The manifest of ``base``'s pooled store seeded from a token store: its pooled streams, each as the token store has it.

    Equal to what `stream_manifest(base, contract, "pooled")` derives for the same weights, which is what lets a later
    `embed` extend the seeded store: the stream records are copied, never rewritten.
    """
    if manifest_kind(token_manifest) != STORE_TOKEN or token_manifest["base"] != base.id:
        raise ValueError(f"A pooled store is seeded from the token store of {base.id}; this manifest is "
                         f"{manifest_kind(token_manifest)} of {token_manifest['base']}.")
    missing = [name for name in base.pooled_streams if name not in token_manifest["streams"]]
    if missing:
        raise ValueError(f"The token manifest lacks the pooled streams {missing}.")
    hub_repo, volume_directory = base.store_destinations(STORE_POOLED)
    return {**{key: value for key, value in token_manifest.items() if key != "streams"},
            "hub_repo": hub_repo, "volume_directory": volume_directory, "store_kind": STORE_POOLED,
            "streams": {name: dict(token_manifest["streams"][name]) for name in base.pooled_streams}}


def pooled_profile_manifest(base: CanonicalBase, token_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The manifest a pooled run of ``base`` writes under the profile of its token store's ``token_manifest``: the token
    store's pooled streams as it records them, the pooled store's folder and volume, everything else the same.

    It is what `profile_manifest(base, contract, "pooled", ...)` derives for the same weights and profile, so a graft can
    start a pooled store from its token store's segments and a later pooled `embed` extends it.
    """
    from .numerics import load_profile

    if token_manifest.get("schema") != MANIFEST_SCHEMA_V3 or manifest_kind(token_manifest) != STORE_TOKEN \
            or token_manifest["base"] != base.id:
        raise ValueError(f"A pooled profile store starts from a profile token store of {base.id}, not a "
                         f"{manifest_kind(token_manifest)} store of {token_manifest.get('base')}.")
    profile = load_profile(token_manifest["profile"]["id"])
    if profile.binding() != token_manifest["profile"]:
        raise ValueError(f"The token store binds {token_manifest['profile']}, not the profile file's {profile.binding()}.")
    hub_repo, folder, volume = base.profile_destinations(STORE_POOLED)
    return {**{key: value for key, value in token_manifest.items() if key != "streams"}, "hub_repo": hub_repo,
            "path_in_repo": folder, "volume": volume, "volume_directory": folder, "store_kind": STORE_POOLED,
            "streams": {name: dict(record) for name, record in token_manifest["streams"].items() if name in POOLED_NAMES}}


def write_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    """Record the manifest at the store root, or confirm the one already there is the same store."""
    path = Path(root) / MANIFEST_FILE
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded != manifest:
            raise ValueError(f"{path} describes a different store; a base's streams never change, start a new root.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_manifest(root: str | Path) -> dict[str, Any]:
    """The manifest of a canonical store root."""
    path = Path(root) / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; {root} is not a canonical embedding store.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") not in MANIFEST_SCHEMAS:
        raise ValueError(f"{path} is not a canonical embedding manifest; the schemas are {sorted(MANIFEST_SCHEMAS)}.")
    return manifest


def fetch_record(root: str | Path) -> dict[str, Any] | None:
    """What `hub.fetch_rows` recorded in a partial copy of a published store (repository, revision, streams), or None."""
    path = Path(root) / FETCH_RECORD
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def refuse_profile_store(root: Path, purpose: str) -> None:
    """Refuse a v2 operation on a profile store, whose keys name no batch budget, libraries or machine."""
    path = Path(root) / MANIFEST_FILE
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("schema") == MANIFEST_SCHEMA_V3:
        raise ValueError(f"{root} is a profile store ({MANIFEST_SCHEMA_V3}); {purpose} belongs to v2 stores. Run with --profile.")


def manifest_features(root: str | Path, manifest: Mapping[str, Any], names: Sequence[str]) -> dict[str, StoredFeature]:
    """The descriptors of the named streams, each read from its `feature.json` and held to the manifest's digest."""
    from fastplms.features import StoredFeature

    base = Path(root)
    features: dict[str, StoredFeature] = {}
    for name in names:
        record = manifest["streams"][name]
        payload = json.loads((base / record["key"] / "feature.json").read_text(encoding="utf-8"))
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if digest != record["descriptor_sha256"]:
            raise ValueError(f"The descriptor of stream {name!r} differs from the manifest that pins it.")
        features[name] = StoredFeature.from_payload(payload)
    return features


def open_canonical_view(
    root: str | Path, inventory: Mapping[str, SequenceKey], *, streams: Sequence[str] | None = None,
    budget: ReadBudget | None = None, persistent_reader: bool = True, deep: bool = False,
) -> FeatureView:
    """Open a canonical store for reading as a `FeatureView`, with nothing but its manifest and the inventory.

    Per-token streams answer `rows` and `residues` with l + 2 rows per protein (CLS, residues, EOS) and `lengths`
    with l; pooled streams answer `pooled` and `rows` with one vector. ``inventory`` maps each protein text to its
    `SequenceKey` (`sequence_inventory`). A store copied from the Hub has no `index.sqlite`; it is rebuilt here from
    the commit markers, without re-reading the parts. A partial copy (`hub.fetch_rows`, which leaves `fetch.json`) holds
    only the parts of the proteins it was fetched for: its index lists their rows and grows when a later fetch adds
    parts, and without ``streams`` the view opens the streams it fetched.

    Opening reads no row. It checks the manifest and each stream's descriptor against it, looks every requested key
    up in the index, and checks each part holding a requested row once: a part passes from the verification receipt
    kept under `<root>/.cache/open_receipts/<manifest sha256>/` when the commit marker's digests and the part's size and
    modification time match, and is otherwise hashed and validated in full and then recorded. A second open, and every
    spawned worker, skips the hashing; a changed manifest, marker or part fails or is verified again. A receipt cannot see
    a rewrite that keeps a file's size and modification time: ``deep=True`` is the check for that. It ignores the
    receipts, hashes every touched part, reads and scans every requested row, and refreshes the receipts, so use it
    where the job verifies a store (`job verify`), not to open a training view.
    """
    from fastplms.features import FeatureStore

    base = Path(root)
    manifest = read_manifest(base)
    manifest_sha256 = hashlib.sha256((base / MANIFEST_FILE).read_bytes()).hexdigest()
    fetched = fetch_record(base)
    default = manifest["streams"] if fetched is None else fetched["streams"]
    names = tuple(streams) if streams is not None else tuple(default)
    unknown = [name for name in names if name not in manifest["streams"]]
    if unknown:
        raise KeyError(f"{manifest['label']} has no stream {unknown}; it holds {sorted(manifest['streams'])}.")
    features = manifest_features(base, manifest, names)
    for feature in features.values():
        # A partial copy is indexed on every open, so the parts a later fetch added are found.
        if fetched is not None or not (base / feature.key / "index.sqlite").exists():
            FeatureStore.open(base, feature, deep_verify=False, partial=fetched is not None)
    return FeatureView(
        base, features, inventory, budget=budget, persistent_reader=persistent_reader, verify_rows=deep,
        receipt_directory=base / ".cache" / "open_receipts" / manifest_sha256, trust_receipts=not deep,
    )


__all__ = [
    "ANNOTATION_STREAMS", "BATCH_FILL", "BATCH_GROWTH_LIMIT", "BATCH_TOKEN_QUANTUM", "CANONICAL_BASES", "DECODER_HIDDEN",
    "DECODER_MEAN_VAR", "ENCODER_BASES", "ENCODER_HIDDEN", "ENCODER_MEAN_VAR", "FAMILY_ANNOTATION", "FAMILY_LIGAND", "FAMILY_PROTEIN",
    "FINAL_HIDDEN", "FINAL_MEAN_VAR", "LIGAND_STREAMS",
    "LAYER_HIDDEN", "LAYER_MEAN_VAR", "STRUCTURAL_MEAN_VAR", "MANIFEST_FILE", "CANONICAL_MAX_RESIDUES", "NORMALIZATION_VERSION", "RANDOM_INIT_SEED",
    "RANDOM_STREAMS", "SAE_CODES", "SAE_MAX", "STRUCTURAL", "TRAINED_STREAMS", "BatchPolicy", "CanonicalBase",
    "CanonicalRuntime", "build_runtime", "canonical_base", "canonical_taps", "check_model_pin", "choose_batch",
    "ENGINE_EMBEDDING_FILES", "RecordedEvidence", "changed_fields", "check_keys_match_store", "file_identity",
    "inherit_recorded_evidence", "manifest_features", "open_canonical_view", "read_manifest", "recorded_batch",
    "recorded_evidence", "sequence_inventory", "stream_manifest", "write_manifest",
    "POOLED_NAMES", "POOLED_VOLUME_DIRECTORY", "STORE_POOLED", "STORE_TOKEN", "key_batch", "manifest_kind",
    "pooled_manifest", "pooled_profile_manifest", "MANIFEST_SCHEMA_V3", "PRIVATE_REPO", "PROFILE_VOLUME_PREFIX",
    "PUBLIC_REPO", "profile_volume", "random_init_model", "save_random_weights",
    "SEQUENCE_LIST_REPO", "VISIBILITY_PRIVATE", "VISIBILITY_PUBLIC", "manifest_profile", "profile_manifest",
    "refuse_profile_store", "FETCH_RECORD", "fetch_record",
]
