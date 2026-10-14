"""Compact multilabel truth and observation masks until a training batch needs tensors.

Values and exceptions to each row's default observation status use CSR storage. Each mask
stores the smaller of its observed and unknown column sets. Missing truth stays unknown.
"""

from __future__ import annotations

import math
import numpy as np
import torch

from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from scipy import sparse
from torch import Tensor

from ..metrics import validate_observed_labels


FLOAT32_MAX = float(np.finfo(np.float32).max)


@dataclass(frozen=True, slots=True)
class SparseLabelRow:
    width: int
    indices: np.ndarray  # (k,) nonzero observed target columns
    values: np.ndarray  # (k,) float32 target values
    mask_columns: np.ndarray  # (m,) exceptions to default_observed
    default_observed: bool
    explicit_mask: bool

    @classmethod
    def from_records(cls, labels: Mapping[str, Any], mask: Mapping[str, Any]) -> SparseLabelRow:
        """Validate a sparse source row without constructing a width-sized array."""
        if not isinstance(labels, Mapping) or set(labels) != {"width", "indices", "values"}:
            raise ValueError("sparse labels require width, indices, and values")
        if not isinstance(mask, Mapping) or set(mask) != {"default_observed", "exceptions"}:
            raise ValueError("sparse masks require default_observed and exceptions")
        width, default = labels["width"], mask["default_observed"]
        if type(width) is not int or not 0 < width <= np.iinfo(np.int32).max:
            raise ValueError("sparse label width must be a positive int32 vocabulary size")
        if type(default) is not bool:
            raise ValueError("sparse mask default_observed must be boolean")
        indices = _columns(labels["indices"], width)  # (k,)
        exceptions = _columns(mask["exceptions"], width)  # (m,)
        raw_values = labels["values"]
        if not isinstance(raw_values, list) or len(raw_values) != len(indices):
            raise ValueError("sparse values must align with nonzero target indices")
        if any(type(value) not in {int, float} or abs(value) > FLOAT32_MAX
               or not math.isfinite(value) or value == 0 for value in raw_values):
            raise ValueError("sparse values must be finite nonzero float32 numbers")
        values = np.asarray(raw_values, dtype=np.float32)  # (k,)
        if (values == 0).any():
            raise ValueError("sparse values must remain nonzero at float32 precision")
        observed_count = width - len(exceptions) if default else len(exceptions)
        if default != (observed_count > width // 2):
            raise ValueError("sparse masks must store the smaller exception set, with False on a tie")
        if exceptions.size:
            found = np.minimum(np.searchsorted(exceptions, indices), exceptions.size - 1)  # (k,) exceptions are sorted
            exceptional = exceptions[found] == indices  # (k,) whether each nonzero column is an exception
        else:
            exceptional = np.zeros(indices.size, dtype=np.bool_)  # (k,)
        if (exceptional == default).any():
            raise ValueError("nonzero sparse targets must have observed labels")
        return cls(width, indices, values, exceptions, default, True)

    @classmethod
    def from_observed(cls, width: int, observed: Mapping[int, float]) -> SparseLabelRow:
        """Encode measured targets, including zeros; absent columns retain unknown status."""
        if type(width) is not int or not 0 < width <= np.iinfo(np.int32).max:
            raise ValueError("sparse label width must be a positive int32 vocabulary size")
        if any(type(index) is not int or not 0 <= index < width for index in observed):
            raise ValueError("observed columns must index the frozen vocabulary")
        if any(type(value) not in {int, float} or abs(value) > FLOAT32_MAX
               or not math.isfinite(value) for value in observed.values()):
            raise ValueError("observed labels must be finite float32 numbers")
        default = len(observed) > width // 2
        exceptions = [index for index in range(width) if index not in observed] if default else sorted(observed)
        indices = sorted(index for index, value in observed.items() if value != 0)
        return cls.from_records(
            {"width": width, "indices": indices, "values": [observed[index] for index in indices]},
            {"default_observed": default, "exceptions": exceptions},
        )

    @property
    def observed_count(self) -> int:
        return self.width - self.mask_columns.size if self.default_observed else self.mask_columns.size

    def to_records(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """JSON/Arrow columns for an explicit sparse source row."""
        if not self.explicit_mask:
            raise ValueError("sparse source records require an explicit observation mask")
        return (
            {"width": self.width, "indices": self.indices.tolist(), "values": self.values.tolist()},
            {"default_observed": self.default_observed, "exceptions": self.mask_columns.tolist()},
        )


def _columns(values: object, width: int) -> np.ndarray:
    """A strictly increasing list of Python ints inside the vocabulary, checked in bulk: a row holds up to `width` of them."""
    inside = "sparse columns must be integer indices inside the frozen vocabulary"
    if not isinstance(values, list):
        raise ValueError(inside)
    if not values:
        return np.empty(0, dtype=np.int64)  # (0,)
    if set(map(type, values)) != {int}:  # bool and numpy integers are not indices
        raise ValueError(inside)
    try:
        columns = np.array(values, dtype=np.int64)  # (k,)
    except OverflowError:
        raise ValueError(inside) from None
    if columns.min() < 0 or columns.max() >= width:
        raise ValueError(inside)
    if (columns[1:] <= columns[:-1]).any():
        raise ValueError("sparse columns must be strictly increasing and unique")
    return columns  # (k,)


def observed_row_counts(labels: object, mask: object) -> tuple[tuple[int, ...], int, int]:
    """Validated shape, observed count, and total count without expanding sparse source rows."""
    if isinstance(labels, Mapping):
        row = SparseLabelRow.from_records(labels, mask)
        return (row.width,), row.observed_count, row.width
    targets, observed = np.asarray(labels), np.asarray(mask)  # () or (c,)
    validate_observed_labels(targets, observed)
    return targets.shape, int(observed.sum()), observed.size


@dataclass(frozen=True, slots=True)
class SparseObservedTargets:
    values: sparse.csr_matrix  # (n, c), unknown targets and measured zero both store no value
    mask_entries: sparse.csr_matrix  # (n, c), exceptions to the row's default observation status
    default_observed: np.ndarray  # (n,), choose the default that minimizes stored mask entries
    explicit_mask: bool

    @classmethod
    def from_rows(cls, labels: Sequence[Any], masks: Sequence[Any] | None = None) -> SparseObservedTargets:
        """Read dense or versioned sparse source rows without a corpus-sized dense copy.

        Legacy rows without a mask retain their fully observed meaning. Masked rows are
        validated before compression, and unknown fillers are discarded regardless of value.
        """
        rows = len(labels)
        if masks is not None and len(masks) != rows:
            raise ValueError("labels and masks must have the same row count")
        values, columns, offsets = array("f"), array("q"), array("q", [0])
        measured, measured_offsets = array("q"), array("q", [0])
        defaults = np.empty(rows, dtype=np.bool_)  # (n,)
        width = None
        mask_rows = iter(masks) if masks is not None else None
        for index, source in enumerate(labels):
            if isinstance(source, Mapping):
                if mask_rows is None:
                    raise ValueError("sparse source labels require explicit sparse masks")
                row = SparseLabelRow.from_records(source, next(mask_rows))
                if width is None:
                    width = row.width
                elif row.width != width:
                    raise ValueError("multilabel rows must have one frozen vocabulary width")
                columns.frombytes(row.indices.tobytes())  # int64 and float32 bytes append at memcpy speed; extend visits each scalar
                values.frombytes(row.values.tobytes())
                offsets.append(len(values))
                defaults[index] = row.default_observed
                measured.frombytes(row.mask_columns.tobytes())
                measured_offsets.append(len(measured))
                continue
            target = np.asarray(source)  # (c,)
            if target.ndim != 1 or target.size == 0 or target.dtype.kind not in "biuf":
                raise ValueError("multilabel targets must be nonempty numeric vectors")
            if width is None:
                width = target.size
            elif target.size != width:
                raise ValueError("multilabel rows must have one frozen vocabulary width")
            mask = np.ones(width, dtype=np.bool_) if mask_rows is None else np.asarray(next(mask_rows))  # (c,)
            if mask.dtype != np.bool_ or mask.shape != target.shape:
                raise ValueError("label_mask must be boolean with exactly the labels shape")
            target = np.where(mask, target, 0).astype(np.float32)  # (c,), matches the probe's target dtype
            if not np.isfinite(target).all():
                raise ValueError("observed labels must be finite float32 values")
            nonzero = np.flatnonzero(target)  # (k,)
            columns.extend(nonzero)
            values.extend(target[nonzero])
            offsets.append(len(values))
            defaults[index] = np.count_nonzero(mask) > width // 2
            measured.extend(np.flatnonzero(mask != defaults[index]))  # (m,), smaller mask polarity
            measured_offsets.append(len(measured))
        shape = (rows, 0 if width is None else width)
        value_matrix = sparse.csr_matrix((np.frombuffer(values, dtype=np.float32), np.frombuffer(columns, dtype=np.int64),
                                          np.frombuffer(offsets, dtype=np.int64)), shape=shape)
        observed_matrix = sparse.csr_matrix((np.ones(len(measured), dtype=np.bool_), np.frombuffer(measured, dtype=np.int64),
                                             np.frombuffer(measured_offsets, dtype=np.int64)), shape=shape)
        return cls(value_matrix, observed_matrix, defaults, masks is not None)

    @property
    def complete_rows(self) -> np.ndarray:
        return self.default_observed & (np.diff(self.mask_entries.indptr) == 0)  # (n,)

    @property
    def storage_nbytes(self) -> int:
        """Owned compact array bytes, excluding the caller's original source representation."""
        return self.default_observed.nbytes + sum(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
                                                  for matrix in (self.values, self.mask_entries))

    def row(self, index: int) -> SparseLabelRow:
        if index < 0:
            index += self.values.shape[0]
        if not 0 <= index < self.values.shape[0]:
            raise IndexError("label row is outside the dataset")
        start, stop = self.values.indptr[index:index + 2]
        observed_start, observed_stop = self.mask_entries.indptr[index:index + 2]
        measured = self.mask_entries.indices[observed_start:observed_stop]  # (m,)
        return SparseLabelRow(self.values.shape[1], self.values.indices[start:stop], self.values.data[start:stop],
                              measured, bool(self.default_observed[index]), self.explicit_mask)


def collate_sparse_labels(rows: Sequence[SparseLabelRow]) -> dict[str, Tensor]:
    """Allocate only the selected batch's `(b, c)` labels and optional observation mask."""
    if not rows:
        raise ValueError("cannot collate an empty label batch")
    width, explicit = rows[0].width, rows[0].explicit_mask
    if any(row.width != width or row.explicit_mask != explicit for row in rows):
        raise ValueError("a batch cannot mix vocabulary widths or implicit and explicit label masks")
    if not explicit and any(not row.default_observed or row.mask_columns.size for row in rows):
        raise ValueError("partially observed rows require an explicit label mask")
    labels = torch.zeros((len(rows), width), dtype=torch.float32)  # (b, c)
    masks = torch.zeros_like(labels, dtype=torch.bool) if explicit else None  # (b, c) or None
    for index, row in enumerate(rows):
        labels[index, torch.as_tensor(row.indices, dtype=torch.long)] = torch.from_numpy(row.values)  # (k,)
        if masks is not None:
            masks[index] = row.default_observed
            masks[index, torch.as_tensor(row.mask_columns, dtype=torch.long)] = not row.default_observed  # (m,)
    return {"labels": labels, **({"label_mask": masks} if masks is not None else {})}  # labels: (b, c); label_mask: (b, c) when explicit
