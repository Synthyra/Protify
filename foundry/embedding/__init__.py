"""SAE and frozen structural features on the FastPLMs embedding API, their contracts, and store readers.

Importing this package imports no submodule, so it imports no torch. A name in `__all__` loads the submodule that defines it
on first access, and `foundry.embedding.<submodule>` loads that submodule the same way, so `import
foundry.embedding.dependencies` selects pinned checkpoints on a machine with no torch at all.
"""

from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .canonical import (
        CANONICAL_BASES, ENCODER_BASES, CanonicalBase, canonical_base, open_canonical_view, sequence_inventory,
    )
    from .capture import capture_feature_specs
    from .contracts import FeatureRowKey, FeatureSpec, SequenceKey
    from .controls import ONE_HOT, RANDOM_VECTOR, SyntheticControlView
    from .keys import FeatureKey, feature_key, plain_feature, plain_feature_key
    from .persistence import FeatureExtractionContract
    from .private_store import PRIVATE_STORE_NAME, load_private_feature_store
    from .readers import FeatureView, LengthBucketBatches, ReadBudget, ResidueBatch
    from .sae_features import (
        POOLING_REDUCTIONS,
        RowLayout,
        TopKFeatures,
        concatenated_rows,
        encode_topk,
        interior_tokens,
        pool_all_rows,
        pool_features,
        row_layout,
    )
    from .structural import structural_tap
    from .taps import sae_residue_tap, sae_tap


# Each exported name, by the submodule that defines it.
EXPORTS = {
    **dict.fromkeys(
        ("CANONICAL_BASES", "ENCODER_BASES", "CanonicalBase", "canonical_base", "open_canonical_view", "sequence_inventory"),
        ".canonical",
    ),
    "capture_feature_specs": ".capture",
    **dict.fromkeys(("FeatureRowKey", "FeatureSpec", "SequenceKey"), ".contracts"),
    **dict.fromkeys(("ONE_HOT", "RANDOM_VECTOR", "SyntheticControlView"), ".controls"),
    **dict.fromkeys(("FeatureKey", "feature_key", "plain_feature", "plain_feature_key"), ".keys"),
    "FeatureExtractionContract": ".persistence",
    **dict.fromkeys(("PRIVATE_STORE_NAME", "load_private_feature_store"), ".private_store"),
    **dict.fromkeys(("FeatureView", "LengthBucketBatches", "ReadBudget", "ResidueBatch"), ".readers"),
    **dict.fromkeys(
        (
            "POOLING_REDUCTIONS", "RowLayout", "TopKFeatures", "concatenated_rows", "encode_topk", "interior_tokens",
            "pool_all_rows", "pool_features", "row_layout",
        ),
        ".sae_features",
    ),
    "structural_tap": ".structural",
    **dict.fromkeys(("sae_residue_tap", "sae_tap"), ".taps"),
}


def __getattr__(name: str) -> object:
    if name in EXPORTS:
        value = getattr(import_module(EXPORTS[name], __name__), name)
        globals()[name] = value  # a later lookup finds it without this function
        return value
    # A bare `import foundry.embedding` once imported these submodules, so `foundry.embedding.keys` still resolves.
    if not name.startswith("_") and find_spec(f"{__name__}.{name}") is not None:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = [
    "CANONICAL_BASES",
    "ENCODER_BASES",
    "ONE_HOT",
    "POOLING_REDUCTIONS",
    "PRIVATE_STORE_NAME",
    "RANDOM_VECTOR",
    "CanonicalBase",
    "FeatureKey",
    "FeatureExtractionContract",
    "FeatureRowKey",
    "FeatureSpec",
    "FeatureView",
    "LengthBucketBatches",
    "ReadBudget",
    "ResidueBatch",
    "RowLayout",
    "SequenceKey",
    "SyntheticControlView",
    "TopKFeatures",
    "canonical_base",
    "capture_feature_specs",
    "concatenated_rows",
    "encode_topk",
    "feature_key",
    "interior_tokens",
    "load_private_feature_store",
    "open_canonical_view",
    "plain_feature",
    "plain_feature_key",
    "pool_all_rows",
    "pool_features",
    "row_layout",
    "sae_tap",
    "sae_residue_tap",
    "sequence_inventory",
    "structural_tap",
]
