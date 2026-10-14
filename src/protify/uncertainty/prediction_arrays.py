"""Bounded prediction accumulation and read-only arrays inside uncompressed NumPy archives.

Temporary streams hold complete outputs on disk; only the current batch is copied. Mapped
arrays own their mapping through NumPy's base chain, so their file handle may close first.
The persisted format remains standard NPZ, including its per-member CRC checks.
"""

from __future__ import annotations

import math
import mmap
import struct
import tempfile
import zipfile
import numpy as np

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from pathlib import Path
from types import TracebackType
from typing import BinaryIO


BUFFER_BYTES = 8 * 1024**2


class DiskPredictionArray(np.ndarray):
    """Array backed by this module's read-only mapping, eligible for page-release hints."""


def release_prediction_pages(values: np.ndarray) -> None:
    """Drop clean mapped pages where supported, without touching caller-owned writable maps."""
    # values: (n, ...)
    if not isinstance(values, DiskPredictionArray) or values.flags.writeable or not hasattr(mmap.mmap, "madvise"):
        return
    base = values
    while isinstance(base, np.ndarray):
        base = base.base
    if isinstance(base, mmap.mmap) and hasattr(mmap, "MADV_DONTNEED"):
        base.madvise(mmap.MADV_DONTNEED)


def _mapped_array(stream: BinaryIO, shape: tuple[int, ...], dtype: np.dtype, *, offset: int = 0, order: str = "C") -> np.ndarray:
    size = math.prod(shape) * dtype.itemsize
    if not size:
        values = np.empty(shape, dtype=dtype, order=order)
        values.flags.writeable = False
        return values  # (n, ...) empty, read-only
    aligned = offset - offset % mmap.ALLOCATIONGRANULARITY
    mapping = mmap.mmap(stream.fileno(), size + offset - aligned, offset=aligned, access=mmap.ACCESS_READ)
    return np.ndarray(shape, dtype=dtype, buffer=mapping, offset=offset - aligned, order=order).view(DiskPredictionArray)  # (n, ...) mapped


class PredictionAccumulator:
    """Append aligned batches exactly once, into disk streams or one preallocated RAM array."""

    def __init__(self, rows: int, *, disk_backed: bool) -> None:
        if type(rows) is not int or rows < 1:
            raise ValueError("prediction collection requires a nonempty dataset")
        self.rows = rows
        self.disk_backed = disk_backed
        self.cursor = 0
        self.stack = ExitStack()
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.dtypes: dict[str, np.dtype] = {}
        self.streams: dict[str, BinaryIO] = {}
        self.arrays: dict[str, np.ndarray] = {}

    def __enter__(self) -> PredictionAccumulator:
        return self

    def __exit__(self, kind: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None) -> None:
        self.stack.close()

    def append(self, batch: Mapping[str, np.ndarray]) -> None:
        # batch: (b, ...) per field
        if not batch or any(values.ndim == 0 or values.dtype.hasobject for values in batch.values()):
            raise ValueError("prediction batches require non-object row arrays")
        count = next(iter(batch.values())).shape[0]
        if count < 1 or any(values.shape[0] != count for values in batch.values()) or self.cursor + count > self.rows:
            raise ValueError("prediction batch rows differ from their declared dataset")
        if not self.shapes:
            for name, values in batch.items():
                self.shapes[name] = (self.rows, *values.shape[1:])
                self.dtypes[name] = values.dtype
                if self.disk_backed:
                    self.streams[name] = self.stack.enter_context(tempfile.TemporaryFile(prefix="protify_predictions_"))  # noqa: SIM115 - ExitStack owns closure.
                else:
                    self.arrays[name] = np.empty(self.shapes[name], dtype=values.dtype)
        if set(batch) != set(self.shapes):
            raise ValueError("prediction batches cannot mix fields or implicit and explicit label masks")
        for name, values in batch.items():
            if values.shape[1:] != self.shapes[name][1:] or values.dtype != self.dtypes[name]:
                raise ValueError("prediction batch shapes and dtypes must remain constant")
            if self.disk_backed:
                self.streams[name].write(np.ascontiguousarray(values).tobytes())  # One (b, ...) block.
            else:
                self.arrays[name][self.cursor:self.cursor + count] = values
        self.cursor += count

    def finish(self) -> dict[str, np.ndarray]:
        if self.cursor != self.rows:
            raise ValueError("prediction collection did not consume every declared row")
        if self.disk_backed:
            for name, stream in self.streams.items():
                stream.flush()
                self.arrays[name] = _mapped_array(stream, self.shapes[name], self.dtypes[name])  # (n, ...)
        return self.arrays  # (n, ...) per field


def prediction_row_blocks(*inputs: np.ndarray) -> Iterator[tuple[np.ndarray, ...]]:
    """Yield aligned plain-array views and release owned mapping pages after each block."""
    # inputs: (n, ...) aligned on the first dimension
    if any(values.ndim == 0 for values in inputs) or not inputs or len({len(values) for values in inputs}) != 1:
        raise ValueError("prediction transforms require aligned row arrays")
    count = len(inputs[0])
    if not any(isinstance(values, DiskPredictionArray) for values in inputs):
        yield inputs
        return
    row_bytes = sum(math.prod(values.shape[1:]) * values.dtype.itemsize for values in inputs)
    batch_rows = max(1, BUFFER_BYTES // max(1, row_bytes))
    for start in range(0, count, batch_rows):
        try:
            yield tuple(np.asarray(values[start:start + batch_rows]) for values in inputs)  # aligned (b, ...)
        finally:
            for values in inputs:
                release_prediction_pages(values)


def select_prediction_rows(values: np.ndarray, rows: np.ndarray | slice) -> np.ndarray:
    """Preserve NumPy selection order without copying a complete mapped matrix into RAM."""
    # values: (n, ...); rows: (m,) integer indices, (n,) flags, or a slice.
    if not isinstance(values, DiskPredictionArray) or isinstance(rows, slice):
        return values[rows]  # (m, ...), a view for slice selection
    indices = np.arange(len(values))[rows]  # (m,), also validates NumPy's indexing rules
    if indices.ndim != 1:
        raise ValueError("prediction row selection requires one-dimensional indices")
    if not len(indices):
        return values[:0]  # (0, ...)
    row_bytes = math.prod(values.shape[1:]) * values.dtype.itemsize
    batch_rows = max(1, BUFFER_BYTES // max(1, row_bytes))
    with PredictionAccumulator(len(indices), disk_backed=True) as accumulator:
        for start in range(0, len(indices), batch_rows):
            try:
                selected = np.asarray(values[indices[start:start + batch_rows]])  # (b, ...)
                accumulator.append({"values": selected})
            finally:
                release_prediction_pages(values)
        return accumulator.finish()["values"]  # (m, ...), read-only


def transform_prediction_rows(
    transform: Callable[..., Mapping[str, np.ndarray]], *inputs: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply a row-preserving transform, keeping owned mapped inputs and outputs off the heap.

    Ordinary callers retain their original whole-array arithmetic and writable results.
    For mapped inputs, the combined input block fits BUFFER_BYTES, with one complete row
    as the minimum. The transform receives ordinary ndarray views, so nested transforms
    work on this block rather than creating more temporary mappings. Every output must
    preserve the block's first dimension; trailing dimensions and dtypes remain constant.
    """
    # inputs: (n, ...) aligned on the first dimension, transform: (b, ...) blocks in, (b, ...) fields out
    if not any(isinstance(values, DiskPredictionArray) for values in inputs):
        return dict(transform(*inputs))  # (n, ...) per field
    if any(values.ndim == 0 for values in inputs) or len({len(values) for values in inputs}) != 1:
        raise ValueError("prediction transforms require aligned row arrays")
    count = len(inputs[0])
    if not count:
        return dict(transform(*(np.asarray(values) for values in inputs)))  # (n, ...) per field
    with PredictionAccumulator(count, disk_backed=True) as accumulator:
        for blocks in prediction_row_blocks(*inputs):
            outputs = transform(*blocks)
            if any(values.ndim == 0 or len(values) != len(blocks[0]) for values in outputs.values()):
                raise ValueError("prediction transforms must preserve each block's row count")
            accumulator.append(outputs)
        return accumulator.finish()  # (n, ...) per field


def prediction_mask_is_complete(mask: np.ndarray) -> bool:
    """Check all observation flags without retaining every mapped mask page in memory."""
    # mask: (n, ...)
    if not isinstance(mask, DiskPredictionArray):
        return bool(mask.all())
    batch_rows = max(1, BUFFER_BYTES // max(1, math.prod(mask.shape[1:]) * mask.dtype.itemsize))
    for start in range(0, len(mask), batch_rows):
        complete = bool(mask[start:start + batch_rows].all())
        release_prediction_pages(mask)
        if not complete:
            return False
    return True


def write_prediction_archive(stream: BinaryIO, arrays: Mapping[str, np.ndarray]) -> None:
    """Write standard uncompressed NPZ members in bounded chunks, retaining their dtype/order."""
    # arrays: (n, ...) per name
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, values in arrays.items():
            if values.dtype.hasobject:
                raise ValueError("prediction archives cannot contain object arrays")
            header = np.lib.format.header_data_from_array_1_0(values)
            order = "F" if header["fortran_order"] else "C"
            with archive.open(name + ".npy", "w", force_zip64=True) as member:
                np.lib.format.write_array_header_2_0(member, header)
                blocks = np.nditer(values, flags=["external_loop", "buffered", "zerosize_ok"],
                                   op_flags=["readonly"], order=order, buffersize=max(1, BUFFER_BYTES // max(1, values.dtype.itemsize)))
                for block in blocks:
                    member.write(block.tobytes())
                    release_prediction_pages(values)


def mapped_prediction_archive(path: str | Path) -> dict[str, np.ndarray]:
    """Check each ZIP CRC and NPY layout, then map its exact bytes without extracting files.

    Explicit mapping requires ZIP_STORED and NPY 1.0/2.0, the layouts written by NumPy and
    this module. Compressed or object payloads fail instead of silently allocating full arrays.
    """
    arrays = {}
    # Windows mmap can reposition the file descriptor. An unbuffered stream keeps ZIP
    # seeks synchronized with that descriptor when the next member is opened.
    with Path(path).open("rb", buffering=0) as stream, zipfile.ZipFile(stream) as archive:
        for record in archive.infolist():
            if not record.filename.endswith(".npy") or record.filename[:-4] in arrays:
                raise ValueError("prediction archives require unique NPY member names")
            if record.compress_type != zipfile.ZIP_STORED or record.compress_size != record.file_size:
                raise ValueError("mapped predictions require uncompressed NPZ members")
            stream.seek(record.header_offset)
            local = stream.read(30)
            if len(local) != 30 or local[:4] != b"PK\x03\x04":
                raise ValueError("invalid prediction ZIP member header")
            name_bytes, extra_bytes = struct.unpack("<HH", local[26:30])
            payload_offset = record.header_offset + 30 + name_bytes + extra_bytes
            with archive.open(record) as member:
                version = np.lib.format.read_magic(member)
                readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
                if version not in readers:
                    raise ValueError("mapped predictions require NPY 1.0 or 2.0")
                shape, fortran, dtype = readers[version](member)
                if dtype.hasobject:
                    raise ValueError("prediction archives cannot contain object arrays")
                offset = member.tell()
                if math.prod(shape) * dtype.itemsize != record.file_size - offset:
                    raise ValueError("prediction array shape differs from its member byte count")
                # ZipExtFile verifies CRC at EOF. Reading bounded blocks also rejects truncation.
                while member.read(BUFFER_BYTES):
                    pass
            arrays[record.filename[:-4]] = _mapped_array(stream, shape, dtype, offset=payload_offset + offset,
                                                         order="F" if fortran else "C")
    return arrays  # (n, ...) per name, mapped
