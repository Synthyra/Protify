"""A cell's dataset split again by identity strata: train, validation, calibration, and test.

`python -m protify.uncertainty split cell.yaml` pools the rows the cell loads, from every
published split, and splits them with `foundry.datasets.build_identity_stratified_split`, as
docs/conventions/splits.md asks of a dataset the workspace builds. Each distinct sequence is a
protein, and every row holding it follows it. The manifest goes to `uncertainty.split_manifest`,
with the MMseqs stages beside it, so a rerun reuses finished searches and an existing manifest is
kept.

A cell that names a manifest reads its rows through `apply_split_manifest`: train from the train
split, validation as V_fit, calibration as V_cal, and test from the test split. V_fit and V_cal
reach the workflow as explicit roles, so `split.json` names them by row digest as for a carve.
Calibration and test are drawn alike from the same identity strata, so they are exchangeable by
construction, which a carve of a published validation split is not.

Pair datasets are refused: a pair's split needs both proteins' clusters, which this does not build.
"""

from __future__ import annotations

import os
import numpy as np

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from foundry.datasets import (
    STRATIFIED_SPLIT_NAMES,
    StratifiedSettings,
    build_identity_stratified_split,
    quintile_labels,
    validate_identity_stratified_split,
)
from .calibration_split import REGRESSION_LABEL_TYPES, sequence_columns, stratum_codes, take_rows
from .identity_bins import sequence_digest
from .report import read_json, write_json


SPLIT_SCHEME = "identity_stratified"


def pooled_rows(splits: Sequence[Any]) -> Any:
    """The rows of every split given, in order, as one dataset of the same kind."""
    if all(hasattr(split, "select") for split in splits):
        from datasets import concatenate_datasets

        return concatenate_datasets(list(splits))
    names = list(splits[0].keys())
    return {name: [value for split in splits for value in list(split[name])] for name in names}


def _sequence_ids(rows: Any) -> list[str]:
    if sequence_columns(rows) != ("seqs",):
        raise ValueError("an identity-stratified split of pair rows is not built; it needs both proteins' clusters.")
    return [sequence_digest(sequence) for sequence in rows["seqs"]]


def protein_labels(ids: Sequence[str], labels: Sequence[Any], label_type: str) -> dict[str, str]:
    """One label per protein for the splitter's label shares: its class, quintile, or rarest label.

    A sequence on several rows takes its first row's label.
    """
    if label_type in REGRESSION_LABEL_TYPES:
        first: dict[str, float] = {}
        for protein, value in zip(ids, labels, strict=True):
            first.setdefault(protein, float(np.asarray(value, dtype=np.float64).reshape(-1)[0]))
        return quintile_labels(first)

    codes = stratum_codes(labels, label_type)  # (n,)
    chosen: dict[str, str] = {}
    for protein, code in zip(ids, codes, strict=True):
        chosen.setdefault(protein, str(int(code)))
    return chosen


def build_split_manifest(
    data_name: str, splits: Sequence[Any], label_type: str, path: str, *, seed: int, threads: int | None = None
) -> dict[str, Any]:
    """Split the pooled rows by identity strata and write the manifest to `path`; an existing one is kept."""
    if os.path.exists(path):
        print(f"{path} exists; keeping it.")
        return read_json(path)

    rows = pooled_rows(splits)
    ids = _sequence_ids(rows)
    sequences = dict(zip(ids, rows["seqs"], strict=True))
    labels = protein_labels(ids, list(rows["labels"]), label_type)
    work = Path(path).parent / f"{Path(path).stem}_mmseqs"
    options = {} if threads is None else {"threads": threads}
    hierarchy, split = build_identity_stratified_split(
        sequences, work, labels=labels, settings=StratifiedSettings(seed=seed), **options
    )
    validate_identity_stratified_split(hierarchy, split)

    manifest = {
        "scheme": SPLIT_SCHEME,
        "data_name": data_name,
        "seed": seed,
        "pooled_rows": len(ids),
        "proteins": len(sequences),
        "rows": {name: int(sum(split.protein_to_split[protein] == name for protein in ids)) for name in STRATIFIED_SPLIT_NAMES},
        "split": split.to_dict(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_json(path, manifest)
    return manifest


def split_row_indices(ids: Sequence[str], manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Each split's pooled row indices, ascending, by each row's protein."""
    if manifest.get("scheme") != SPLIT_SCHEME:
        raise ValueError(f"a split manifest of scheme {SPLIT_SCHEME!r} is expected; got {manifest.get('scheme')!r}.")
    assignment = manifest["split"]["protein_to_split"]
    missing = sorted({protein for protein in ids if protein not in assignment})
    if missing:
        raise ValueError(f"the split manifest names no split for {len(missing)} of the rows' sequences, such as {missing[:3]}.")

    names = np.asarray([assignment[protein] for protein in ids])  # (n,)
    indices = {name: np.nonzero(names == name)[0] for name in STRATIFIED_SPLIT_NAMES}
    counted = {name: int(rows.size) for name, rows in indices.items()}
    if counted != manifest["rows"]:
        raise ValueError(f"the rows fall into splits {counted}, but the manifest recorded {manifest['rows']}.")
    return indices  # each (m,) row indices by split name


def apply_split_manifest(cell_data: Any, manifest: Mapping[str, Any]) -> tuple[Any, tuple[np.ndarray, np.ndarray]]:
    """The cell's data re-split by the manifest, and the explicit V_fit and V_cal rows over its validation.

    `cell_data` is the workflow's `CellData`. Validation holds the validation rows, then the
    calibration rows, so V_fit and V_cal index them in that order.
    """
    if manifest.get("data_name") != cell_data.data_name:
        raise ValueError(f"the split manifest is for {manifest.get('data_name')!r}, but the cell reads {cell_data.data_name!r}.")

    rows = pooled_rows([cell_data.train, cell_data.valid, cell_data.test])
    indices = split_row_indices(_sequence_ids(rows), manifest)
    fit_rows, cal_rows = indices["validation"], indices["calibration"]
    train = take_rows(rows, indices["train"])
    valid = take_rows(rows, np.concatenate([fit_rows, cal_rows]))
    test = take_rows(rows, indices["test"])
    cell_data.main.datasets[cell_data.data_name] = (train, valid, test, cell_data.num_labels, cell_data.label_type, cell_data.ppi)
    explicit = (np.arange(fit_rows.size), np.arange(fit_rows.size, fit_rows.size + cal_rows.size))
    return replace(cell_data, train=train, valid=valid, test=test), explicit  # a CellData, then (f,) fit rows and (c,) calibration rows
