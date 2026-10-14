"""Pinned phylum membership and probability sums shared by evaluation and serving."""

from __future__ import annotations

import hashlib
import json
import math

from collections.abc import Mapping, Sequence
from typing import Any


GROUPS = ("eukaryota", "bacteria", "archaea", "viral")
SOURCE_GROUPS = {"Eukaryota": "eukaryota", "Bacteria": "bacteria", "Archaea": "archaea", "Viruses": "viral"}


def identity(document: object) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def winner(probabilities: Mapping[str, float]) -> str | None:
    """Return a unique maximum; an exact tie has no winning label."""
    if not probabilities:
        raise ValueError("Empty probability distribution")
    values = list(probabilities.values())
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Invalid probability distribution")
    if not math.isclose(math.fsum(values), 1.0, abs_tol=1e-5):
        raise ValueError("Probability distribution does not sum to one")
    maximum = max(values)
    tied = [label for label, value in probabilities.items() if value == maximum]
    return tied[0] if len(tied) == 1 else None


class TaxonomyMapping:
    """An exhaustive mapping bound to the source dataset and ordered head vocabulary."""

    def __init__(self, document: Mapping[str, Any], labels: Sequence[str]) -> None:
        self.document = dict(document)
        self.labels = tuple(labels)
        if len(set(labels)) != len(labels) or not labels:
            raise ValueError("Taxonomy vocabulary is empty or duplicated")
        if document.get("schema_version") != "taxonomy_broad_v1" or document.get("groups") != list(GROUPS):
            raise ValueError("Unknown broad taxonomy schema")
        if document.get("vocabulary_sha256") != identity(list(labels)):
            raise ValueError("Broad taxonomy vocabulary mismatch")
        self.membership: dict[str, str] = dict(document["membership"])
        if set(self.membership) != set(labels) or set(self.membership.values()) != set(GROUPS):
            raise ValueError("Every phylum must map to one of the four groups")
        self.group_labels = {group: tuple(sorted(label for label in labels if self.membership[label] == group))
                             for group in GROUPS}
        self.sha256 = identity({key: value for key, value in document.items() if key != "sha256"})
        if document.get("sha256") != self.sha256:
            raise ValueError("Broad taxonomy digest mismatch")

    def aggregate(self, probabilities: Mapping[str, float]) -> dict[str, float]:
        """Validate the complete phylum distribution, then sum its mass without renormalization."""
        missing = set(self.labels) - set(probabilities)
        if missing:
            raise ValueError("Missing taxonomy probabilities")
        selected = {label: probabilities[label] for label in self.labels}
        winner(selected)
        return {group: math.fsum(selected[label] for label in labels) for group, labels in self.group_labels.items()}

    def describe(self, probabilities: Mapping[str, float]) -> dict[str, Any]:
        broad = self.aggregate(probabilities)
        top = sorted(self.labels, key=lambda label: (-probabilities[label], label))[:5]
        return {"probabilities": broad, "winner": winner(broad), "mapping_sha256": self.sha256,
                "top_phyla": [{"label": label, "probability": probabilities[label], "category": self.membership[label]}
                              for label in top]}
