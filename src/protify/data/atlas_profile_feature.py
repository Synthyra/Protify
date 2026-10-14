"""One Atlas proteome profile read as a probe branch: a protein's probability of interacting with each protein of a frozen axis.

`production_suite.atlas_profile` writes the store. Its root holds `profile.json`, which names the `profile` feature of the FastPLMs
feature store (`features.profile.key`, width `p`, dtype), keyed by the SHA-256 of a protein's normalized sequence, plus the pins that
make a value what it is: the Atlas package, weights and run digests, the calibration, and the digest of the proteome axis. Column `j`
of a row is the probability against protein `j` of the axis, in axis order, so a column is the same protein in every row of every
store built on that axis.

This module reads that feature and nothing else. It refuses a root that is no profile store and a sequence the store holds no profile
for (`featurize` has not run on it), because a missing row read as zeros would be a different feature that no check could see.
The probability ranks interactions and is not the chance of one in a cell: a hub scores high against everything
(docs/decisions/serving/production_suite/2026-10-02_atlas_proteome_profile_store.md).

Shapes: `b` proteins requested, `p` proteins of the axis (the columns, about 20,000).
"""

from __future__ import annotations

import hashlib
import json
import torch

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastplms.features import FeatureReader
from foundry.embedding.contracts import normalize_sequence


SCHEMA = "atlas_proteome_profile_v1"
MANIFEST_FILE = "profile.json"
FEATURE = "profile"
DTYPES = {"float32": torch.float32, "float16": torch.float16}
# The pins a package records about the model and axis that made the profiles.
MODEL_PINS = ("base", "arm", "package_sha256", "weights_sha256", "run_sha256", "max_length", "precision", "scaling")


def read_profile_manifest(root: str | Path) -> dict[str, Any]:
    """`profile.json` of a store root, refused unless it is a profile store's with a dense `profile` feature of a known dtype."""
    path = Path(root) / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; {root} is not an Atlas profile store.")

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} manifest.")

    feature = manifest.get("features", {}).get(FEATURE)
    if not isinstance(feature, dict) or feature.get("dtype") not in DTYPES or type(feature.get("width")) is not int or feature["width"] < 1:
        raise ValueError(f"{path} names no usable {FEATURE!r} feature: a key, a positive integer width and a dtype of {sorted(DTYPES)}.")
    if not isinstance(feature.get("key"), str) or not feature["key"] or Path(feature["key"]).name != feature["key"]:
        raise ValueError(f"{path} names the {FEATURE!r} feature by a key that is not one directory name.")

    return manifest


def profile_provenance(root: str | Path) -> dict[str, Any]:
    """What a package read: the store's model and axis pins, the feature, and the digest of the manifest that names them."""
    path = Path(root) / MANIFEST_FILE
    manifest = read_profile_manifest(root)
    feature = manifest["features"][FEATURE]
    return {
        "kind": "atlas_proteome_profile", "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model": {name: manifest["model"][name] for name in MODEL_PINS if name in manifest["model"]},
        "axis": manifest["axis"], "dtype": feature["dtype"], "width": feature["width"], "key": feature["key"],
    }


class AtlasProfileFeature:
    """The profile of each of a fixed list of proteins, read by sequence from one store.

    Every sequence is checked against the store when this opens, so a run that would read a missing profile stops before it trains.
    """

    def __init__(self, root: str | Path, sequences: Sequence[str]) -> None:
        manifest = read_profile_manifest(root)
        feature = manifest["features"][FEATURE]
        directory = Path(root) / feature["key"]
        if not directory.is_dir():
            raise FileNotFoundError(f"{directory} is missing; {root}/{MANIFEST_FILE} names a feature the store does not hold.")

        self.root = Path(root)
        self.width = int(feature["width"])  # p
        self.reader = FeatureReader.open(directory)
        absent = self.reader.missing([normalize_sequence(text) for text in sequences])
        if absent:
            self.reader.close()
            raise ValueError(
                f"{len(absent):,} of the {len(sequences):,} proteins have no profile in {root}; run `atlas_profile featurize` "
                f"on them first (the first lacks {len(absent[0])} residues)."
            )

    def read(self, sequences: Sequence[str]) -> torch.Tensor:
        """The profiles of `sequences`, float32 `(b, p)`, in the order asked; a stored half-precision value widens exactly."""
        rows = torch.stack(self.reader.read([normalize_sequence(text) for text in sequences])).to(torch.float32)  # (b, p)
        if rows.shape[1] != self.width or not bool(torch.isfinite(rows).all()):
            raise ValueError(f"{self.root} returned profiles that are not finite rows of width {self.width}.")

        return rows  # (b, p)

    def close(self) -> None:
        self.reader.close()
