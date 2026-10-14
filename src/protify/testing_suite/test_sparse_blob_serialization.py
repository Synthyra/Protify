"""The coordinate-list blob of the old SQLite caches, decoded, and its bfloat16 fidelity.

SAE models whose active fraction is small used to save each embedding as its nonzero coordinates. The
feature store keeps those rows as `csr` now, and old files are read by `legacy_cache.embedding_blob_to_tensor`.
Every blob here is written by the test-local writer of the old bytes (`legacy_cache_writer`), which follows
the format's specification and shares no code with the decoder.
"""

import struct
import torch

from protify.legacy_cache import _COMPACT_VERSION, _SPARSE_VERSION, blob_dtype, embedding_blob_to_tensor
from protify.testing_suite.legacy_cache_writer import LEGACY_BFLOAT16_CODE, dense_blob, sparse_blob


def _sparse_vector(width: int, density: float, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """A codebook-shaped vector with a fixed fraction of non-zero entries."""
    torch.manual_seed(width)
    vector = torch.zeros(width, dtype=dtype)  # (w,)
    active = torch.randperm(width)[: int(density * width)]  # (nnz,)
    vector[active] = torch.rand(len(active)).to(dtype) + 0.1  # (nnz,)
    return vector  # (w,)


def test_the_first_byte_names_the_coordinate_list_format() -> None:
    # Measured max-pooled density for ESMC SAE features at codebook width 131072.
    blob = sparse_blob(_sparse_vector(131072, 0.036))

    assert blob[0] == _SPARSE_VERSION
    assert dense_blob(_sparse_vector(131072, 0.036))[0] == _COMPACT_VERSION


def test_sparse_blob_roundtrips_exactly() -> None:
    vector = _sparse_vector(65536, 0.087)  # (w,)

    recovered = embedding_blob_to_tensor(sparse_blob(vector))  # (w,)

    assert recovered.dtype == vector.dtype
    assert recovered.shape == vector.shape
    assert torch.equal(recovered, vector)


def test_indices_are_two_bytes_up_to_65536_values_and_four_beyond() -> None:
    narrow = _sparse_vector(65536, 0.05)  # (w,)
    wide = _sparse_vector(65537, 0.05)  # (w,)

    assert sparse_blob(narrow)[2] == 0  # uint16 addresses 65536 values
    assert sparse_blob(wide)[2] == 1  # uint32
    assert torch.equal(embedding_blob_to_tensor(sparse_blob(narrow)), narrow)
    assert torch.equal(embedding_blob_to_tensor(sparse_blob(wide)), wide)


def test_sparse_blob_shrinks_wide_codebooks() -> None:
    vector = _sparse_vector(131072, 0.036)  # (w,)

    blob = sparse_blob(vector)

    dense_bytes = vector.numel() * vector.element_size()
    assert len(blob) < 0.15 * dense_bytes


def test_bfloat16_sparse_blob_roundtrips() -> None:
    vector = _sparse_vector(65536, 0.05, dtype=torch.bfloat16)  # (w,)

    recovered = embedding_blob_to_tensor(sparse_blob(vector))  # (w,)

    assert recovered.dtype == torch.bfloat16
    assert torch.equal(recovered, vector)


def test_every_row_of_a_batch_decodes_to_itself() -> None:
    width = 131072  # w
    batch = torch.zeros(3, width, dtype=torch.float16)  # (b, w)
    for row, count in enumerate((100, 5_000, 120_000)):
        batch[row, torch.randperm(width)[:count]] = 1.5

    blobs = [sparse_blob(row) for row in batch]

    assert [blob[0] for blob in blobs] == [_SPARSE_VERSION] * 3
    for row, blob in enumerate(blobs):
        assert torch.equal(embedding_blob_to_tensor(blob), batch[row])


def test_sparse_matrix_embedding_roundtrips() -> None:
    torch.manual_seed(0)
    matrix = torch.zeros(7, 65536, dtype=torch.float16)  # (l, w)
    matrix[torch.randint(0, 7, (400,)), torch.randint(0, 65536, (400,))] = 2.0

    recovered = embedding_blob_to_tensor(sparse_blob(matrix))  # (l, w)

    assert recovered.shape == matrix.shape
    assert torch.equal(recovered, matrix)


def test_sparse_bfloat16_keeps_small_magnitudes() -> None:
    vector = torch.zeros(8192, dtype=torch.bfloat16)  # (w,)
    vector[5] = 1.0e-6
    vector[9] = 1.0e-30

    recovered = embedding_blob_to_tensor(sparse_blob(vector))  # (w,)

    assert recovered.dtype == torch.bfloat16
    assert torch.equal(recovered, vector)


def test_legacy_bfloat16_sparse_blobs_still_read() -> None:
    """A coordinate list can carry the first bfloat16 encoding too: float16 bytes under dtype code 1."""
    indices = torch.tensor([2, 5], dtype=torch.int32)
    values = torch.tensor([1.0, -2.5], dtype=torch.bfloat16)
    hand_written = (
        struct.pack("<BBBii", _SPARSE_VERSION, LEGACY_BFLOAT16_CODE, 0, 1, 8)
        + struct.pack("<i", 2)
        + indices.numpy().astype("<u2").tobytes()
        + values.half().numpy().tobytes()
    )

    recovered = embedding_blob_to_tensor(hand_written)  # (w=8,)

    assert blob_dtype(hand_written) == torch.bfloat16
    assert recovered.dtype == torch.bfloat16
    expected = torch.zeros(8, dtype=torch.bfloat16)
    expected[2], expected[5] = 1.0, -2.5
    assert torch.equal(recovered, expected)


def test_negative_zero_survives_the_coordinate_list_format_as_zero() -> None:
    # The writer keeps only non-zero entries, and negative zero is not one, so it comes back as
    # positive zero. The two compare equal, which is all callers relied on.
    vector = torch.zeros(4096, dtype=torch.float16)  # (w,)
    vector[7] = -0.0
    vector[11] = 2.5

    recovered = embedding_blob_to_tensor(sparse_blob(vector))  # (w,)

    assert torch.equal(recovered, vector)
    assert not torch.signbit(recovered[7])


def test_all_zero_embedding_roundtrips() -> None:
    vector = torch.zeros(4096, dtype=torch.float16)  # (w,)

    blob = sparse_blob(vector)

    assert blob[0] == _SPARSE_VERSION
    assert torch.equal(embedding_blob_to_tensor(blob), vector)


def test_the_header_names_the_dtype_of_a_coordinate_list() -> None:
    for dtype in (torch.float16, torch.float32, torch.bfloat16):
        assert blob_dtype(sparse_blob(_sparse_vector(4096, 0.1, dtype=dtype))) == dtype
