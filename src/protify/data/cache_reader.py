"""Read cached embedding rows without pooling, casting, or creating a cache.

`kind="store"` opens a feature store directory, which is what Protify writes. `sqlite` and `torch`
open the two file formats Protify wrote before the feature store, through the read-only decoders of
`legacy_cache`, for code that has to inspect an old cache before converting it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from torch import Tensor

from ..feature_cache import row_lookup
from ..legacy_cache import cached_rows as legacy_rows


@contextmanager
def cached_rows(
    path: Path, *, kind: str, fallback_shape: Callable[[str], tuple[int, ...]] | None = None,
) -> Iterator[Callable[[str], Tensor]]:
    """Yield a raw lookup from sequence to the row the cache holds.

    A store gives `(d,)` for a pooled representation and `(l, d)` per token, at the store's dtype.
    An old SQLite cache is read lazily and an old Torch dictionary loads once on CPU. The caller owns
    input integrity and memory admission; this layer neither pools residue matrices nor widens
    bfloat16. Raw legacy float32 blobs need a caller-supplied `fallback_shape`, and a store ignores it.
    """
    if kind == "store":
        with row_lookup(path) as lookup:
            yield lookup
        return
    with legacy_rows(Path(path), kind=kind, fallback_shape=fallback_shape) as lookup:
        yield lookup
