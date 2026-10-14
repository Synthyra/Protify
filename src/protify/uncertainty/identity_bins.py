"""Rows binned by their highest sequence identity to the training split.

A useful uncertainty rises as a test protein moves away from anything the head trained on, so
every metric is also reported per identity bin. `foundry.datasets.homology` scores each
sequence's highest MMseqs identity to training, 0.0 when no hit passes the e-value cutoff,
and a pair row takes the higher of its two proteins: a pair is only as novel as its more
familiar member.

Bins: no hit (identity 0.0), then (0, 0.3), [0.3, 0.5), [0.5, 0.7), [0.7, 0.9), [0.9, 1.0].
"""

from __future__ import annotations

import hashlib
import numpy as np

from collections.abc import Mapping, Sequence


IDENTITY_BIN_NAMES = ("no_hit", "(0,0.3)", "[0.3,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,1.0]")
IDENTITY_EDGES = np.array([0.3, 0.5, 0.7, 0.9])


def sequence_digest(sequence: str) -> str:
    """A whitespace-free identifier for a sequence, usable as a FASTA header and a JSON key."""
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]


def identity_codes(identity: np.ndarray) -> np.ndarray:
    """Each row's bin, as an index into `IDENTITY_BIN_NAMES`."""
    # identity: (n,)
    identity = np.asarray(identity, dtype=np.float64)  # (n,)
    if np.isnan(identity).any() or identity.min(initial=0.0) < 0.0 or identity.max(initial=0.0) > 1.0:
        raise ValueError("identities must lie in [0, 1].")
    codes = 1 + np.searchsorted(IDENTITY_EDGES, identity, side="right")  # (n,)
    return np.where(identity <= 0.0, 0, codes)  # (n,)


def row_identity(sequence_identity: Mapping[str, float], sequence_columns: Sequence[Sequence[str]]) -> np.ndarray:
    """Each row's identity to training: its sequence's, or the higher of a pair's two.

    `sequence_identity` maps `sequence_digest` to identity. `sequence_columns` holds one list
    of sequences for single-sequence rows and two for pairs.
    """
    per_column = [
        np.array([sequence_identity[sequence_digest(sequence)] for sequence in column], dtype=np.float64)  # (n,)
        for column in sequence_columns
    ]
    return np.max(np.stack(per_column), axis=0)  # (n,)


def bin_counts(identity: np.ndarray) -> dict[str, int]:
    # identity: (n,)
    counts = np.bincount(identity_codes(identity), minlength=len(IDENTITY_BIN_NAMES))  # (bins,)
    return {name: int(count) for name, count in zip(IDENTITY_BIN_NAMES, counts, strict=True)}


def bin_rows(identity: np.ndarray) -> dict[str, np.ndarray]:
    """Row indices in each bin, empty bins included."""
    # identity: (n,)
    codes = identity_codes(identity)  # (n,)
    return {name: np.flatnonzero(codes == code) for code, name in enumerate(IDENTITY_BIN_NAMES)}  # each (m,) row indices by bin name
