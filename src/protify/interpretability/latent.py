"""Principal component views of pooled taps: how each layer or site arranges proteins that share an annotation.

For one site and layer, the mean half of every protein's pooled row is centered and projected onto its first two
principal components, fitted on the task's proteins with their original sequences. Scrambled copies of the test
proteins, read from a scrambled store, are projected onto the same components, so they show where sequence order
alone moves a protein. Proteins are grouped by their exact set of labels: the `top` most frequent sets are named and
the rest are "other".

Symbols: n proteins; w site width; k components.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from collections.abc import Sequence
from torch import Tensor

from .concepts import ConceptData
from .pooled_store import PooledTapStore


SCRAMBLED = "scrambled"
OTHER = "other"


def principal_components(X: Tensor, components: int = 2) -> tuple[Tensor, Tensor, Tensor]:
    """The mean, the leading principal axes and the share of variance each explains."""
    # X: (n, w)
    X = X.to(torch.float64)
    mean = X.mean(dim=0)  # (w,)
    _, singular, axes = torch.linalg.svd(X - mean, full_matrices=False)  # (min(n, w),), (min(n, w), w)
    explained = singular.square() / singular.square().sum()
    return mean, axes[:components].T, explained[:components]  # (w,), (w, k), (k,)


def label_sets(data: ConceptData, top: int) -> dict[str, str]:
    """Each protein sequence's group: its exact label set when that set is among the `top` most frequent, else other."""
    names = data.task.label_names
    sets: dict[str, str] = {}
    for split in data.splits.values():
        for sequence, label in zip(split.sequences, split.labels, strict=True):
            active = np.flatnonzero(np.atleast_1d(label)) if data.task.kind != "multiclass" else [int(label)]
            sets[sequence] = " + ".join(readable(names[index]) for index in active) or "none"
    frequent = set(pd.Series(list(sets.values())).value_counts().index[:top])
    return {sequence: name if name in frequent else OTHER for sequence, name in sets.items()}


def readable(label: str) -> str:
    """An InterPro or GO label without its accession: 'IPR000276__g_protein-coupled_receptor' reads as the name."""
    return label.split("__", 1)[-1].replace("_", " ")


def project_layers(
    store: PooledTapStore,
    groups: dict[str, str],
    site: str,
    layers: Sequence[int],
    scrambled: PooledTapStore | None = None,
    scrambled_sequences: Sequence[str] = (),
) -> pd.DataFrame:
    """Two principal coordinates of every protein at each layer of a site, with its group and the variance shares."""
    sequences = list(groups)
    stored_layers = store.layers(site)
    width = int(store.manifest.sites[site]["width"])
    pooled = store.load(site, store.rows_for(sequences))  # (n, L, 2 * w)
    extra = scrambled.load(site, scrambled.rows_for(list(scrambled_sequences))) if scrambled is not None and scrambled_sequences else None
    frames = []
    for layer in layers:
        slot = stored_layers.index(layer)
        X = pooled[:, slot, :width].float()  # (n, w), the mean half
        mean, axes, explained = principal_components(X)
        coordinates = ((X.double() - mean) @ axes).numpy()  # (n, 2)
        frame = pd.DataFrame({"pc1": coordinates[:, 0], "pc2": coordinates[:, 1], "group": [groups[s] for s in sequences]})
        if extra is not None:
            moved = ((extra[:, slot, :width].double() - mean) @ axes).numpy()  # (s, 2)
            frame = pd.concat([frame, pd.DataFrame({"pc1": moved[:, 0], "pc2": moved[:, 1], "group": SCRAMBLED})], ignore_index=True)
        frame["site"] = site
        frame["layer"] = layer
        frame["explained_1"] = float(explained[0])
        frame["explained_2"] = float(explained[1])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
