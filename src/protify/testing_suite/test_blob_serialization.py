"""The dense blob of the old SQLite caches, decoded.

Protify no longer writes SQLite caches (embeddings live in the feature store), but old ones are read by
`legacy_cache.embedding_blob_to_tensor`, which the converter calls. Every blob here is written by the
test-local writer of the old bytes (`legacy_cache_writer`), which follows the format's specification and
shares no code with the decoder.
"""

import io
import struct
import numpy as np
import pytest
import torch

from protify import legacy_cache
from protify.legacy_cache import _COMPACT_VERSION, blob_dtype, embedding_blob_to_tensor
from protify.testing_suite.legacy_cache_writer import LEGACY_BFLOAT16_CODE, dense_blob


def test_roundtrip_float32() -> None:
    tensor = torch.randn(128)  # (d=128,)
    recovered = embedding_blob_to_tensor(dense_blob(tensor))  # (d,)
    assert recovered.shape == tensor.shape
    assert recovered.dtype == torch.float32
    assert torch.equal(recovered, tensor)


def test_roundtrip_float16() -> None:
    tensor = torch.randn(64).half()  # (d=64,)
    recovered = embedding_blob_to_tensor(dense_blob(tensor))  # (d,)
    assert recovered.shape == tensor.shape
    assert recovered.dtype == torch.float16
    assert torch.equal(recovered, tensor)


def test_roundtrip_bfloat16_is_exact() -> None:
    tensor = torch.randn(64).bfloat16()  # (d=64,)
    recovered = embedding_blob_to_tensor(dense_blob(tensor))  # (d,)
    assert recovered.shape == tensor.shape
    assert recovered.dtype == torch.bfloat16
    # The bit pattern is stored, so the values are the same bits, not merely close.
    assert torch.equal(recovered, tensor)


def test_roundtrip_2d() -> None:
    tensor = torch.randn(10, 320)  # (l=10, d=320)
    recovered = embedding_blob_to_tensor(dense_blob(tensor))  # (l, d)
    assert recovered.shape == (10, 320)
    assert torch.equal(recovered, tensor)


def test_roundtrip_1d() -> None:
    tensor = torch.randn(256)  # (d=256,)
    recovered = embedding_blob_to_tensor(dense_blob(tensor))  # (d,)
    assert recovered.shape == (256,)
    assert torch.equal(recovered, tensor)


def test_a_decoded_blob_owns_its_memory() -> None:
    blob = dense_blob(torch.arange(4, dtype=torch.float32))  # (d,)
    recovered = embedding_blob_to_tensor(blob)
    recovered += 1  # a view of the immutable blob bytes would refuse or corrupt this
    assert recovered.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert embedding_blob_to_tensor(blob).tolist() == [0.0, 1.0, 2.0, 3.0]


def test_header_names_the_dtype_without_decoding_the_values() -> None:
    assert blob_dtype(dense_blob(torch.zeros(3, dtype=torch.float16))) == torch.float16
    assert blob_dtype(dense_blob(torch.zeros(3))) == torch.float32
    assert blob_dtype(dense_blob(torch.zeros(3, dtype=torch.bfloat16))) == torch.bfloat16
    assert blob_dtype(dense_blob(torch.zeros(3, dtype=torch.bfloat16), code=LEGACY_BFLOAT16_CODE)) == torch.bfloat16
    assert blob_dtype(b"") is None
    assert blob_dtype(b"\x00\x00\x00") is None


def test_unsupported_dtype_is_a_torch_save_blob() -> None:
    """The old writer saved any other dtype with `torch.save`, and the decoder reads it back."""
    tensor = torch.randn(32).double()  # (d=32,)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    blob = buffer.getvalue()
    # A compact blob starts with its version byte; torch.save bytes do not.
    assert blob[0] != _COMPACT_VERSION
    assert blob_dtype(blob) is None
    recovered = embedding_blob_to_tensor(blob)  # (d,)
    assert recovered.shape == tensor.shape
    assert recovered.dtype == torch.float64
    assert torch.equal(recovered, tensor)


def test_legacy_raw_float32_with_fallback_shape() -> None:
    tensor = torch.randn(4, 16)  # (l=4, d=16)
    raw = tensor.numpy().tobytes()
    recovered = embedding_blob_to_tensor(raw, fallback_shape=(4, 16))  # (l, d)
    assert recovered.shape == (4, 16)
    assert torch.allclose(recovered, tensor)


def test_legacy_raw_no_fallback_raises() -> None:
    legacy_values = np.random.randn(32).astype(np.float32)  # (d=32,)
    raw = legacy_values.tobytes()
    with pytest.raises(ValueError, match="no fallback_shape"):
        embedding_blob_to_tensor(raw)


def test_legacy_bfloat16_blobs_still_read() -> None:
    """Caches written before the bit-pattern code tagged float16 bytes with dtype code 1."""
    values = torch.tensor([1.0, -2.5, 0.5], dtype=torch.bfloat16)  # (d,)
    hand_written = struct.pack("<BBii", _COMPACT_VERSION, 1, 1, 3) + values.half().numpy().tobytes()

    assert dense_blob(values, code=LEGACY_BFLOAT16_CODE) == hand_written
    recovered = embedding_blob_to_tensor(hand_written)  # (d,)
    assert recovered.dtype == torch.bfloat16
    assert torch.equal(recovered, values)


def test_bfloat16_keeps_values_below_the_float16_normal_range() -> None:
    """The bit pattern keeps small magnitudes; the first encoding, float16 bytes, changed them."""
    vector = torch.tensor([1.0e-6, 3.0e-8, 1.0e-30, 1.0, -2.5], dtype=torch.bfloat16)  # (d,)

    exact = embedding_blob_to_tensor(dense_blob(vector))  # (d,)
    first_encoding = embedding_blob_to_tensor(dense_blob(vector, code=LEGACY_BFLOAT16_CODE))  # (d,)

    assert torch.equal(exact, vector)
    assert not torch.equal(first_encoding, vector), "code 1 flushes 1e-30 to zero, which is why code 3 exists"


def test_legacy_cache_exports_the_decoder() -> None:
    assert legacy_cache.embedding_blob_to_tensor(dense_blob(torch.ones(2))).tolist() == [1.0, 1.0]


@pytest.mark.parametrize("name", ["tensor_to_embedding_blob", "batch_tensor_to_blobs"])
def test_the_blob_writers_are_gone(name: str) -> None:
    """Nothing in Protify writes the old format, so a new cache cannot be made in it by accident."""
    assert not hasattr(legacy_cache, name)
