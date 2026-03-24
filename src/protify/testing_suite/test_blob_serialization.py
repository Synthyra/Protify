import struct

import numpy as np
import torch
import pytest

try:
    from src.protify.utils import (
        _COMPACT_VERSION, _DTYPE_TO_CODE,
        tensor_to_embedding_blob, batch_tensor_to_blobs, embedding_blob_to_tensor,
    )
except ImportError:
    try:
        from protify.utils import (
            _COMPACT_VERSION, _DTYPE_TO_CODE,
            tensor_to_embedding_blob, batch_tensor_to_blobs, embedding_blob_to_tensor,
        )
    except ImportError:
        from ..utils import (
            _COMPACT_VERSION, _DTYPE_TO_CODE,
            tensor_to_embedding_blob, batch_tensor_to_blobs, embedding_blob_to_tensor,
        )


def test_roundtrip_float32() -> None:
    t = torch.randn(128)
    blob = tensor_to_embedding_blob(t)
    recovered = embedding_blob_to_tensor(blob)
    assert recovered.shape == t.shape
    assert recovered.dtype == torch.float32
    assert torch.equal(recovered, t)


def test_roundtrip_float16() -> None:
    t = torch.randn(64).half()
    blob = tensor_to_embedding_blob(t)
    recovered = embedding_blob_to_tensor(blob)
    assert recovered.shape == t.shape
    assert recovered.dtype == torch.float16
    assert torch.equal(recovered, t)


def test_roundtrip_bfloat16() -> None:
    t = torch.randn(64).bfloat16()
    blob = tensor_to_embedding_blob(t)
    recovered = embedding_blob_to_tensor(blob)
    assert recovered.shape == t.shape
    assert recovered.dtype == torch.bfloat16
    # bfloat16 goes through fp16 intermediate, so values are close but not exact
    assert torch.allclose(recovered.float(), t.float(), atol=1e-2)


def test_roundtrip_2d() -> None:
    t = torch.randn(10, 320)
    blob = tensor_to_embedding_blob(t)
    recovered = embedding_blob_to_tensor(blob)
    assert recovered.shape == (10, 320)
    assert torch.equal(recovered, t)


def test_roundtrip_1d() -> None:
    t = torch.randn(256)
    blob = tensor_to_embedding_blob(t)
    recovered = embedding_blob_to_tensor(blob)
    assert recovered.shape == (256,)
    assert torch.equal(recovered, t)


def test_batch_blob_count() -> None:
    batch = torch.randn(8, 64)
    blobs = batch_tensor_to_blobs(batch)
    assert len(blobs) == 8


def test_batch_blob_individual_shape() -> None:
    batch = torch.randn(4, 128)
    blobs = batch_tensor_to_blobs(batch)
    for blob in blobs:
        recovered = embedding_blob_to_tensor(blob)
        assert recovered.shape == (128,)


def test_batch_blob_3d() -> None:
    batch = torch.randn(5, 10, 32)
    blobs = batch_tensor_to_blobs(batch)
    assert len(blobs) == 5
    for i, blob in enumerate(blobs):
        recovered = embedding_blob_to_tensor(blob)
        assert recovered.shape == (10, 32)
        assert torch.equal(recovered, batch[i])


def test_batch_matches_individual() -> None:
    batch = torch.randn(3, 64).half()
    blobs_batch = batch_tensor_to_blobs(batch)
    blobs_individual = [tensor_to_embedding_blob(batch[i]) for i in range(3)]
    for b, ind in zip(blobs_batch, blobs_individual):
        assert b == ind


def test_unsupported_dtype_falls_back_to_torch_save() -> None:
    t = torch.randn(32).double()
    blob = tensor_to_embedding_blob(t)
    # Compact format starts with _COMPACT_VERSION; torch.save does not
    assert blob[0] != _COMPACT_VERSION
    recovered = embedding_blob_to_tensor(blob)
    assert recovered.shape == t.shape
    assert recovered.dtype == torch.float64
    assert torch.equal(recovered, t)


def test_legacy_raw_float32_with_fallback_shape() -> None:
    t = torch.randn(4, 16)
    raw = t.numpy().tobytes()
    recovered = embedding_blob_to_tensor(raw, fallback_shape=(4, 16))
    assert recovered.shape == (4, 16)
    assert torch.allclose(recovered, t)


def test_legacy_raw_no_fallback_raises() -> None:
    raw = np.random.randn(32).astype(np.float32).tobytes()
    with pytest.raises(ValueError, match="no fallback_shape"):
        embedding_blob_to_tensor(raw)
