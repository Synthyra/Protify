"""The two roles a validation split plays: V_fit and V_cal.

V_fit early-stops training, chooses hyperparameters, and fits the temperature and variance
maps. V_cal is read only by split conformal, whose guarantee needs calibration rows that no
earlier fit has seen. Both roles are resolved here, once, into explicit row-index arrays over
the validation split, and everything downstream reads those arrays rather than deriving its
own.

Two sources resolve them:
- the default, a stratified carve of the validation split with its own seed, independent of
  every run seed, so the five training seeds of one cell share one V_fit and one V_cal;
- index lists given explicitly, as a split manifest naming a dataset-level calibration split
  would supply them.

The carve stratifies by class for single-label tasks, by label decile for regression, and by
each row's rarest positive label for multilabel tasks, so each role keeps the validation
split's label distribution.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


STRATIFIED_SOURCE = "stratified"
EXPLICIT_SOURCE = "explicit"
LABEL_DECILES = 10
REGRESSION_LABEL_TYPES = ("regression", "sigmoid_regression")


@dataclass(frozen=True)
class ValidationRoles:
    """V_fit and V_cal as ascending row indices into the validation split."""

    fit: np.ndarray  # (n_fit,)
    cal: np.ndarray  # (n_cal,)
    source: str
    fraction: float | None = None
    seed: int | None = None


def _plain(value: Any) -> Any:
    """A label as JSON can hold it, whatever container Hugging Face returned."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def column_names(dataset: Any) -> list[str]:
    return list(dataset.column_names) if hasattr(dataset, "column_names") else list(dataset.keys())


def sequence_columns(dataset: Any) -> tuple[str, ...]:
    """`("SeqA", "SeqB")` for a pair dataset, `("seqs",)` otherwise."""
    return ("SeqA", "SeqB") if "SeqA" in column_names(dataset) else ("seqs",)


def row_digests(dataset: Any) -> list[str]:
    """A short SHA-256 of sequences, labels, and any explicit observation mask."""
    sequence_names = sequence_columns(dataset)
    names = getattr(dataset, "column_names", dataset)
    masked = "label_mask" in names
    columns = [dataset[name] for name in (*sequence_names, "labels", *(["label_mask"] if masked else []))]
    digests = []
    for values in zip(*columns, strict=True):
        record = {"sequences": list(values[:len(sequence_names)]), "label": _plain(values[len(sequence_names)])}
        if masked:
            record["label_mask"] = _plain(values[-1])
        payload = json.dumps(record, sort_keys=True)
        digests.append(hashlib.sha256(payload.encode()).hexdigest()[:16])
    return digests


def take_rows(dataset: Any, rows: np.ndarray) -> Any:
    """The given rows of a Hugging Face dataset, or of a dict of columns."""
    # rows: (m,)
    if hasattr(dataset, "select"):
        return dataset.select([int(row) for row in rows])
    return {name: [list(values)[int(row)] for row in rows] for name, values in dataset.items()}


def stratum_codes(labels: Sequence[Any], label_type: str) -> np.ndarray:
    """One stratum per row: its class, its label decile, or its rarest positive label."""
    if label_type == "singlelabel":
        return np.asarray(labels, dtype=np.int64).reshape(-1)  # (n,)

    if label_type in REGRESSION_LABEL_TYPES:
        values = np.asarray(labels, dtype=np.float64).reshape(-1)  # (n,)
        order = np.argsort(values, kind="mergesort")  # (n,)
        codes = np.empty(values.shape[0], dtype=np.int64)  # (n,)
        # Equal-count deciles by rank, so tied labels cannot empty a decile.
        codes[order] = np.arange(values.shape[0]) * LABEL_DECILES // values.shape[0]
        return codes  # (n,)

    if label_type == "multilabel":
        matrix = np.asarray(labels, dtype=np.float64)  # (n, c)
        positives = matrix > 0.5  # (n, c)
        frequency = positives.sum(axis=0)  # (c,)
        # A row's rarest positive label; rows with none share stratum -1.
        rarity = np.where(positives, frequency[None, :], np.inf)  # (n, c)
        codes = np.argmin(rarity, axis=1).astype(np.int64)  # (n,)
        codes[~positives.any(axis=1)] = -1
        return codes  # (n,)

    raise ValueError(f"validation roles cannot be stratified for label type {label_type!r}.")


def carve_validation(
    labels: Sequence[Any], label_type: str, fraction: float = 0.5, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified V_fit and V_cal rows, V_cal holding `fraction` of the validation split.

    Rows are ordered by stratum, randomly within each, and every row where the running count
    `floor(position * fraction)` steps up goes to V_cal. Each stratum therefore splits within
    one row of `fraction`, and the whole split within one row, even for strata of one row.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"calibration fraction must lie strictly between 0 and 1; got {fraction}.")

    codes = stratum_codes(labels, label_type)  # (n,)
    count = codes.shape[0]
    keys = np.random.default_rng(seed).random(count)  # (n,)
    order = np.lexsort((keys, codes))  # (n,)
    position = np.arange(count)  # (n,)
    to_cal = np.floor((position + 1) * fraction) > np.floor(position * fraction)  # (n,)
    return np.sort(order[~to_cal]), np.sort(order[to_cal])  # (n_fit,), (n_cal,)


def _explicit_rows(rows: Sequence[int], name: str, count: int) -> np.ndarray:
    array = np.asarray(rows, dtype=np.int64).reshape(-1)  # (k,)
    if array.size == 0:
        raise ValueError(f"explicit {name} rows are empty.")
    if array.min() < 0 or array.max() >= count:
        raise ValueError(f"explicit {name} rows must index the {count} validation rows.")
    if np.unique(array).size != array.size:
        raise ValueError(f"explicit {name} rows repeat an index.")
    return np.sort(array)  # (k,)


def resolve_validation_roles(
    labels: Sequence[Any],
    label_type: str,
    *,
    fraction: float = 0.5,
    seed: int = 0,
    explicit_rows: tuple[Sequence[int], Sequence[int]] | None = None,
    require_calibration: bool = True,
) -> ValidationRoles:
    """V_fit and V_cal rows, carved by default or taken from `explicit_rows`.

    `explicit_rows` is `(v_fit_rows, v_cal_rows)` over the validation split. They must be
    disjoint and non-empty; rows in neither are left unused. A fit that never loads the
    calibration split passes `require_calibration=False` with explicit rows whose V_cal is
    empty, because a carve or a V_cal row would need labels that fit does not read.
    """
    if explicit_rows is None:
        if not require_calibration:
            raise ValueError("a fit without V_cal takes explicit V_fit rows; a carve reads calibration labels.")
        fit, cal = carve_validation(labels, label_type, fraction, seed)
        roles = ValidationRoles(fit, cal, STRATIFIED_SOURCE, fraction, seed)
    else:
        count = len(labels)
        fit = _explicit_rows(explicit_rows[0], "V_fit", count)
        if require_calibration:
            cal = _explicit_rows(explicit_rows[1], "V_cal", count)
        elif len(explicit_rows[1]):
            raise ValueError("a fit without V_cal cannot name V_cal rows.")
        else:
            cal = np.empty(0, dtype=np.int64)  # (0,)
        if np.intersect1d(fit, cal).size:
            raise ValueError("explicit V_fit and V_cal rows overlap.")
        roles = ValidationRoles(fit, cal, EXPLICIT_SOURCE)

    if roles.fit.size == 0 or (require_calibration and roles.cal.size == 0):
        raise ValueError(f"validation of {len(labels)} rows leaves V_fit or V_cal empty.")
    return roles


def roles_manifest(roles: ValidationRoles, digests: Sequence[str], label_type: str) -> dict[str, Any]:
    """What a run records about its roles, so a resumed run can prove it reads the same rows."""
    return {
        "source": roles.source,
        "fraction": roles.fraction,
        "seed": roles.seed,
        "label_type": label_type,
        "validation_rows": len(digests),
        "v_fit_rows": roles.fit.tolist(),
        "v_cal_rows": roles.cal.tolist(),
        "v_fit_digests": [digests[row] for row in roles.fit],
        "v_cal_digests": [digests[row] for row in roles.cal],
    }


def split_validation(
    valid_set: Any,
    label_type: str,
    fraction: float = 0.5,
    seed: int = 0,
    explicit_rows: tuple[Sequence[int], Sequence[int]] | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """V_fit, V_cal, and the manifest naming their rows and row digests."""
    roles = resolve_validation_roles(
        list(valid_set["labels"]), label_type, fraction=fraction, seed=seed, explicit_rows=explicit_rows
    )
    manifest = roles_manifest(roles, row_digests(valid_set), label_type)
    return take_rows(valid_set, roles.fit), take_rows(valid_set, roles.cal), manifest


def check_manifest(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Refuse to resume when the roles or the rows behind them have changed."""
    keys = ("v_fit_rows", "v_cal_rows", "v_fit_digests", "v_cal_digests")
    changed = [key for key in keys if list(saved.get(key, [])) != list(current[key])]
    if changed:
        raise ValueError(
            f"the validation roles differ from the ones this output directory was run with "
            f"({', '.join(changed)}). Use a new output directory, or the same data and split."
        )
