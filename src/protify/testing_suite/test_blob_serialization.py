import numpy as np
import pytest
import torch

try:
    from src.protify.utils import (
        _COMPACT_VERSION,
        batch_tensor_to_blobs,
        embedding_blob_to_tensor,
        tensor_to_embedding_blob,
    )
except ImportError:
    try:
        from protify.utils import (
            _COMPACT_VERSION,
            batch_tensor_to_blobs,
            embedding_blob_to_tensor,
            tensor_to_embedding_blob,
        )
    except ImportError:
        from ..utils import (
            _COMPACT_VERSION,
            batch_tensor_to_blobs,
            embedding_blob_to_tensor,
            tensor_to_embedding_blob,
        )


def test_roundtrip_float32() -> None:
    tensor = torch.randn(128)  # (d=128,)
    blob = tensor_to_embedding_blob(tensor)
    recovered = embedding_blob_to_tensor(blob)  # (d,)
    assert recovered.shape == tensor.shape
    assert recovered.dtype == torch.float32
    assert torch.equal(recovered, tensor)


def test_roundtrip_float16() -> None:
    tensor = torch.randn(64).half()  # (d=64,)
    blob = tensor_to_embedding_blob(tensor)
    recovered = embedding_blob_to_tensor(blob)  # (d,)
    assert recovered.shape == tensor.shape
    assert recovered.dtype == torch.float16
    assert torch.equal(recovered, tensor)


def test_roundtrip_bfloat16() -> None:
    tensor = torch.randn(64).bfloat16()  # (d=64,)
    blob = tensor_to_embedding_blob(tensor)
    recovered = embedding_blob_to_tensor(blob)  # (d,)
    assert recovered.shape == tensor.shape
    assert recovered.dtype == torch.bfloat16
    # bfloat16 goes through fp16 intermediate, so values are close but not exact
    assert torch.allclose(recovered.float(), tensor.float(), atol=1e-2)


def test_roundtrip_2d() -> None:
    tensor = torch.randn(10, 320)  # (l=10, d=320)
    blob = tensor_to_embedding_blob(tensor)
    recovered = embedding_blob_to_tensor(blob)  # (l, d)
    assert recovered.shape == (10, 320)
    assert torch.equal(recovered, tensor)


def test_roundtrip_1d() -> None:
    tensor = torch.randn(256)  # (d=256,)
    blob = tensor_to_embedding_blob(tensor)
    recovered = embedding_blob_to_tensor(blob)  # (d,)
    assert recovered.shape == (256,)
    assert torch.equal(recovered, tensor)


def test_batch_blob_count() -> None:
    batch = torch.randn(8, 64)  # (b=8, d=64)
    blobs = batch_tensor_to_blobs(batch)
    assert len(blobs) == 8


def test_batch_blob_individual_shape() -> None:
    batch = torch.randn(4, 128)  # (b=4, d=128)
    blobs = batch_tensor_to_blobs(batch)
    for blob in blobs:
        recovered = embedding_blob_to_tensor(blob)  # (d,)
        assert recovered.shape == (128,)


def test_batch_blob_3d() -> None:
    batch = torch.randn(5, 10, 32)  # (b=5, l=10, d=32)
    blobs = batch_tensor_to_blobs(batch)
    assert len(blobs) == 5
    for i, blob in enumerate(blobs):
        recovered = embedding_blob_to_tensor(blob)  # (l, d)
        assert recovered.shape == (10, 32)
        assert torch.equal(recovered, batch[i])


def test_batch_matches_individual() -> None:
    batch = torch.randn(3, 64).half()  # (b=3, d=64)
    blobs_batch = batch_tensor_to_blobs(batch)
    blobs_individual = [tensor_to_embedding_blob(batch[i]) for i in range(3)]
    for batch_blob, individual_blob in zip(blobs_batch, blobs_individual):
        assert batch_blob == individual_blob


def test_unsupported_dtype_falls_back_to_torch_save() -> None:
    tensor = torch.randn(32).double()  # (d=32,)
    blob = tensor_to_embedding_blob(tensor)
    # Compact format starts with _COMPACT_VERSION; torch.save does not
    assert blob[0] != _COMPACT_VERSION
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
