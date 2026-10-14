"""The old Protify cache formats, written for tests only.

Protify's own writers of these formats were retired when embeddings moved to the feature store, and
the decoder that survives (`protify.legacy_cache`) only reads them. A test that pins a conversion has
to hold the old bytes, so this module writes them from the format's specification and shares no code
with the decoder it checks:

    dense   [0x01][dtype code:1][ndim:int32][shape: int32 each][raw values]
    sparse  [0x02][dtype code:1][index code:1][ndim:int32][shape: int32 each][nnz:int32]
            [indices: uint16 or uint32][values]

Dtype code 0 is float16, 1 is bfloat16 stored as float16 bytes (the first encoding), 2 is float32, and
3 is a bfloat16 bit pattern. Index code 0 is uint16, which addresses at most 65536 values, and 1 is
uint32.
"""

from __future__ import annotations

import sqlite3
import struct
import numpy as np
import torch

from collections.abc import Mapping
from pathlib import Path
from torch import Tensor


DTYPE_CODES = {torch.float16: 0, torch.float32: 2, torch.bfloat16: 3}
LEGACY_BFLOAT16_CODE = 1


def _value_bytes(values: Tensor, code: int) -> bytes:
    # values: (...), any shape
    if code == LEGACY_BFLOAT16_CODE:
        return values.to(torch.float16).numpy().tobytes()  # the first encoding, lossy below float16 range
    if values.dtype == torch.bfloat16:
        return values.contiguous().view(torch.int16).numpy().tobytes()
    return values.numpy().tobytes()


def dense_blob(tensor: Tensor, code: int | None = None) -> bytes:
    """One tensor as a version 0x01 blob. `code` overrides the dtype code, to write the first bfloat16 encoding."""
    # tensor: (...), any shape
    values = tensor.cpu().contiguous()
    code = DTYPE_CODES[values.dtype] if code is None else code
    header = struct.pack(f"<BBi{values.ndim}i", 0x01, code, values.ndim, *values.shape)
    return header + _value_bytes(values, code)


def sparse_blob(tensor: Tensor) -> bytes:
    """One tensor as a version 0x02 blob holding its exact nonzero entries."""
    # tensor: (...), any shape
    values = tensor.cpu().contiguous()
    code = DTYPE_CODES[values.dtype]
    flat = values.reshape(-1)
    indices = torch.nonzero(flat, as_tuple=False).reshape(-1)
    index_code = 0 if flat.numel() <= 1 << 16 else 1
    header = struct.pack(
        f"<BBBi{values.ndim}ii", 0x02, code, index_code, values.ndim, *values.shape, indices.numel(),
    )
    index_dtype = np.uint16 if index_code == 0 else np.uint32
    return header + indices.numpy().astype(index_dtype).tobytes() + _value_bytes(flat[indices], code)


def write_sqlite_cache(
    path: Path, rows: Mapping[str, bytes | Tensor], *, sparse: bool = False, wal: bool = True,
) -> Path:
    """A SQLite cache with Protify's schema. A tensor row is encoded; a bytes row is stored as given."""
    # rows: (...) per sequence, any shape
    connection = sqlite3.connect(path)
    try:
        if wal:
            connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE IF NOT EXISTS embeddings (sequence text PRIMARY KEY, embedding blob)")
        connection.executemany(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
            [
                (sequence, row if isinstance(row, bytes) else (sparse_blob(row) if sparse else dense_blob(row)))
                for sequence, row in rows.items()
            ],
        )
        connection.commit()
        if wal:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    return path


def write_torch_cache(path: Path, rows: Mapping[str, Tensor]) -> Path:
    """A pickled sequence-to-tensor dictionary."""
    # rows: (...) per sequence, any shape
    torch.save(dict(rows), path)
    return path
