"""Feature cache keys: one filename-safe name for each distinct stored feature.

The accepted decision `2026-09-22_sae_feature_cache_keys` keys a feature cache by the model, its
revision, the SAE by repository and revision, the layer, and the pooling. The stored dtype and the
residue limit also change stored values, so the key includes them. A key refuses an unpinned
revision, because a key without one serves stale features once the checkpoint's weights change.

The key assumes FastPLMs' residue-only policy. Dense features pool the biological residues that
`embed_dataset` pools. SAE features use the same biological mask in `sae_tap`, as raw encoder
magnitudes without the SAE's idf normalization. A feature computed another way, such as a canonical
stream that keeps CLS and EOS (`l + 2` rows, `feature_spec_v2`), must not be stored under this key: it
is named by its `FeatureSpec` key, which records the row policy.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import torch

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any


# Bump to rename every key on purpose; the digest covers it.
KEY_SCHEMA = 2
FEATURE_DTYPES = ("float64", "float32", "float16", "bfloat16")
DIGEST_CHARACTERS = 16
_REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?")
_REVISION = re.compile(r"[0-9a-fA-F]{40}")
_POOLING_NAME = re.compile(r"[a-z][a-z0-9_]*")
_OWNER = re.compile(r"[a-z][a-z0-9_]{0,31}")
_SLUG_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SLUG_PART_CHARACTERS = 48


@dataclass(frozen=True, slots=True)
class FeatureKey:
    """A readable slug and the SHA-256 of every field that changes stored values.

    The slug is a summary and can repeat; the digest decides identity.
    """

    slug: str
    digest: str

    @property
    def name(self) -> str:
        """The filename-safe key: the slug, then the first 16 digest characters."""

        return f"{self.slug}__{self.digest[:DIGEST_CHARACTERS]}"


def feature_key(
    model: str,
    revision: str,
    *,
    sae_repo: str | None = None,
    sae_revision: str | None = None,
    layer: int,
    pooling: str | Sequence[str] | None,
    dtype: str | torch.dtype,
    max_length: int | None,
) -> FeatureKey:
    """The cache key of one feature.

    `model` and `sae_repo` are Hugging Face repository ids, and both revisions are 40-character
    commits. The SAE fields are both None for dense features. `layer` is the FastPLMs hidden-state
    index, non-negative, so resolve -1 to the final state's index first. `pooling` is the ordered
    reductions, or None for per-residue rows. `dtype` is the stored dtype. `max_length` is the
    residue limit applied before embedding, or None for none.
    """

    model = _repository(model, "model")
    revision = _pinned_revision(revision, "revision")
    if (sae_repo is None) != (sae_revision is None):
        raise ValueError(
            "sae_repo and sae_revision go together: both for SAE features, neither for dense ones."
        )
    if sae_repo is not None:
        sae_repo = _repository(sae_repo, "sae_repo")
        sae_revision = _pinned_revision(sae_revision, "sae_revision")
    if not isinstance(layer, int) or isinstance(layer, bool):
        raise TypeError("layer must be an integer hidden-state index.")
    if layer < 0:
        raise ValueError(
            "layer must be a non-negative hidden-state index; resolve -1 to the final state's "
            "index, so one feature has one key."
        )
    pooling_names = _pooling_names(pooling)
    dtype_name = str(dtype).removeprefix("torch.")
    if dtype_name not in FEATURE_DTYPES:
        raise ValueError(f"dtype must be one of {FEATURE_DTYPES}; received {dtype!r}.")
    if max_length is not None and (
        not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0
    ):
        raise ValueError("max_length must be a positive residue count, or None for no limit.")

    fields = {
        "key_schema": KEY_SCHEMA,
        "model": model,
        "revision": revision,
        "sae_repo": sae_repo,
        "sae_revision": sae_revision,
        "layer": layer,
        "pooling": None if pooling_names is None else list(pooling_names),
        "dtype": dtype_name,
        "max_length": max_length,
    }
    digest = hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    parts = [_slug_part(model), revision[:8], f"l{layer}"]
    if sae_repo is not None and sae_revision is not None:
        parts += [_slug_part(sae_repo), sae_revision[:8]]
    parts += [
        "residues" if pooling_names is None else "-".join(pooling_names),
        dtype_name,
        "maxlen-none" if max_length is None else f"maxlen{max_length}",
    ]
    return FeatureKey(slug="__".join(parts), digest=digest)


def plain_feature_key(
    owner: str, descriptor: Mapping[str, Any], *, slug: Sequence[str] = (),
) -> FeatureKey:
    """The cache key of a feature the canonical key cannot describe.

    `feature_key` names what FastPLMs' taps produce from pinned checkpoints. A cache computed
    another way still belongs in the feature store: pooling that keeps BOS and EOS, argmax positions
    beside each code, a project's own encoder. Its whole identity is then a descriptor of plain data
    the caller states, and the key is that descriptor's digest. Two different descriptors never share
    a name, and the store refuses a second descriptor under a name it holds.

    `owner` is the project's snake_case id, `descriptor["schema"]` names the descriptor's format
    (`<owner>_<what>_v1`, and never `feature_spec_v1`, which is the canonical contract), and `slug`
    is a few readable parts of the name. The digest, not the slug, decides identity.
    """

    if not isinstance(owner, str) or not _OWNER.fullmatch(owner):
        raise ValueError(f"owner must be a snake_case project id; received {owner!r}.")
    if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("schema"), str):
        raise ValueError("A plain feature descriptor is a mapping with a string 'schema'.")
    if descriptor["schema"] == "feature_spec_v1":
        raise ValueError(
            "feature_spec_v1 is the canonical contract; describe it with FeatureSpec and feature_key."
        )
    parts = tuple(slug)
    if not all(isinstance(part, str) and _SLUG_PART.fullmatch(part) for part in parts):
        raise ValueError(f"slug parts must be filename-safe words; received {parts!r}.")
    try:
        encoded = json.dumps(
            {"key_schema": KEY_SCHEMA, "owner": owner, "descriptor": dict(descriptor)},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("A plain feature descriptor must be finite plain JSON data.") from error
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return FeatureKey(
        slug="__".join([owner, *(part[:_SLUG_PART_CHARACTERS] for part in parts)]), digest=digest,
    )


def plain_feature(
    owner: str, descriptor: Mapping[str, Any], *, layout: str, width: int, dtype: torch.dtype,
    positions: bool = False, sparse_count: int | None = None, slug: Sequence[str] = (),
    store: ModuleType | None = None,
) -> Any:
    """The `StoredFeature` of a plain feature, named by `plain_feature_key`.

    `store` is the module whose `StoredFeature` class is built, `fastplms.features` by default. A
    process that loads Hub artifacts cannot import that name (see `private_store`) and passes
    `load_private_feature_store()`, so the feature is an instance of the class it checks against.
    """

    module = importlib.import_module("fastplms.features") if store is None else store
    return module.StoredFeature(
        key=plain_feature_key(owner, descriptor, slug=slug).name, layout=layout, width=width,
        dtype=dtype, positions=positions, descriptor=descriptor, sparse_count=sparse_count,
    )


def _repository(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a Hugging Face repository id string.")
    if not _REPOSITORY.fullmatch(value):
        raise ValueError(
            f"{field} must be a Hugging Face repository id such as 'Synthyra/ESMplusplus_large'; "
            f"received {value!r}."
        )
    return value


def _pinned_revision(value: object, field: str) -> str:
    if not isinstance(value, str) or not _REVISION.fullmatch(value):
        raise ValueError(
            f"{field} must be a 40-character commit hash; received {value!r}. A key without a "
            "pinned revision serves stale features once the checkpoint changes."
        )
    return value.lower()


def _pooling_names(pooling: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if pooling is None:
        return None
    names = (pooling,) if isinstance(pooling, str) else tuple(pooling)
    if not names:
        raise ValueError("pooling must name at least one reduction, or be None for residue rows.")
    if not all(isinstance(name, str) and _POOLING_NAME.fullmatch(name) for name in names):
        raise ValueError(f"pooling names must be lowercase identifiers; received {names!r}.")
    if len(set(names)) != len(names):
        raise ValueError(f"pooling names must not repeat; received {names!r}.")
    return names


def _slug_part(repository: str) -> str:
    """A repository id as one readable slug field, `namespace--name` as the Hub cache writes it."""

    return repository.replace("/", "--")[:_SLUG_PART_CHARACTERS]
