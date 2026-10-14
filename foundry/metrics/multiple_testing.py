"""Benjamini-Hochberg false discovery rate adjustment of p-values.

Sorting `m` tested p-values ascending as `p_(1) <= ... <= p_(m)`, the adjusted value at rank `i` is
`min over j >= i of min(1, p_(j) * m / j)`. Adjusted values therefore rise with p, never pass 1,
and give every tied p-value one value whatever order the sort leaves the tie in. A test is a
discovery at false discovery rate `alpha` when its adjusted value is at most `alpha`.

Twelve copies of this lived in six entities, and they differed in ways this module turns into a
parameter or fixes:

- What counts as a valid p-value. `nan="raise"` (the default) refuses an entry that is not
  finite, as the copies that validated did. `nan="omit"` leaves such an entry out of the tested
  count `m` and returns NaN there, as annotation_vocabulary's association study does for a pair
  it cannot test. A finite entry outside `[0, 1]` raises either way.
- Shape. Any shape is one family of `m = size` tests and the result keeps the input's shape; the
  copies took a vector, a mapping or a list, which a caller reshapes or zips itself.
- Empty input returns an empty array, as nine copies did; Embedding Translation's raised.
- The evaluation order is `(p * m) / rank`, which every copy but one used. The abca4 screen
  computed `p * (m / rank)`, which can differ in the last bit.
"""

from __future__ import annotations

import numpy as np

from collections.abc import Sequence
from typing import Literal


def benjamini_hochberg(
    p_values: np.ndarray | Sequence[float], *, nan: Literal["raise", "omit"] = "raise"
) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, as a float64 array in the input's order and shape.

    `nan="raise"` raises `ValueError` for an entry that is not finite; `nan="omit"` leaves each
    such entry out of the tested count and returns NaN in its place. A finite entry outside
    `[0, 1]` raises either way.
    """
    # p_values: (*s,) any shape, including empty; the return has the same shape
    if nan not in ("raise", "omit"):
        raise ValueError(f"nan must be 'raise' or 'omit', got {nan!r}")
    values = np.asarray(p_values, dtype=np.float64)  # (*s,)
    flattened = values.ravel()  # (pairs,)
    testable = np.isfinite(flattened)  # (pairs,)
    if nan == "raise" and not testable.all():
        raise ValueError("p-values must be finite")
    if np.any((flattened[testable] < 0.0) | (flattened[testable] > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")

    adjusted = np.full(flattened.shape, np.nan)  # (pairs,)
    tested = np.flatnonzero(testable)  # (m,)
    if tested.size:
        order = tested[np.argsort(flattened[tested], kind="stable")]  # (m,)
        ranks = np.arange(1, tested.size + 1, dtype=np.float64)  # (m,)
        scaled = np.minimum(1.0, flattened[order] * tested.size / ranks)  # (m,)
        adjusted[order] = np.minimum.accumulate(scaled[::-1])[::-1]  # (m,)
    return adjusted.reshape(values.shape)  # (*s,)


__all__ = ["benjamini_hochberg"]
