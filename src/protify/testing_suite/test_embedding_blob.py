"""Tests for tensor_to_embedding_blob / embedding_blob_to_tensor serialization.

Verifies round-trip correctness across dtypes, backward compatibility with
legacy torch.save format, SQLite round-trip, and performance vs pickle.
"""
import io
import sqlite3
import tempfile
import time

import torch

from utils import tensor_to_embedding_blob, embedding_blob_to_tensor


# ---------------------------------------------------------------------------
# Round-trip correctness
# ---------------------------------------------------------------------------

class TestRoundTripCorrectness:
    """New raw binary format preserves exact tensor values for all dtypes/shapes."""

    def test_float32_vector(self):
        t = torch.randn(1280)
        recovered = embedding_blob_to_tensor(tensor_to_embedding_blob(t))
        assert torch.equal(t, recovered)
        assert recovered.dtype == torch.float32

    def test_float32_matrix(self):
        t = torch.randn(512, 1280)
        recovered = embedding_blob_to_tensor(tensor_to_embedding_blob(t))
        assert torch.equal(t, recovered)

    def test_float16_vector(self):
        t = torch.randn(1280).half()
        recovered = embedding_blob_to_tensor(tensor_to_embedding_blob(t))
        assert torch.equal(t, recovered)
        assert recovered.dtype == torch.float16

    def test_float16_matrix(self):
        t = torch.randn(256, 640).half()
        recovered = embedding_blob_to_tensor(tensor_to_embedding_blob(t))
        assert torch.equal(t, recovered)

    def test_bfloat16_vector(self):
        t = torch.randn(1280).bfloat16()
        recovered = embedding_blob_to_tensor(tensor_to_embedding_blob(t))
        assert torch.equal(t, recovered)
        assert recovered.dtype == torch.bfloat16

    def test_bfloat16_matrix(self):
        t = torch.randn(512, 1280).bfloat16()
        recovered = embedding_blob_to_tensor(tensor_to_embedding_blob(t))
        assert torch.equal(t, recovered)

    def test_float64_vector(self):
        t = torch.randn(64, dtype=torch.float64)
        recovered = embedding_blob_to_tensor(tensor_to_embedding_blob(t))
        assert torch.equal(t, recovered)
        assert recovered.dtype == torch.float64

    def test_small_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        recovered = embedding_blob_to_tensor(tensor_to_embedding_blob(t))
        assert torch.equal(t, recovered)

    def test_large_matrix_embed(self):
        """Simulates a full-length residue-level embedding (2048 residues x 1280 dims)."""
        t = torch.randn(2048, 1280).bfloat16()
        recovered = embedding_blob_to_tensor(tensor_to_embedding_blob(t))
        assert torch.equal(t, recovered)

    def test_1d_shape_preserved(self):
        t = torch.randn(100)
        recovered = embedding_blob_to_tensor(tensor_to_embedding_blob(t))
        assert recovered.shape == (100,)

    def test_2d_shape_preserved(self):
        t = torch.randn(50, 200)
        recovered = embedding_blob_to_tensor(tensor_to_embedding_blob(t))
        assert recovered.shape == (50, 200)


# ---------------------------------------------------------------------------
# Backward compatibility with legacy torch.save format
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Blobs created with old torch.save format still deserialize correctly."""

    def _legacy_serialize(self, tensor: torch.Tensor) -> bytes:
        buf = io.BytesIO()
        torch.save(tensor.cpu(), buf)
        return buf.getvalue()

    def test_legacy_float32(self):
        t = torch.randn(256, 640)
        blob = self._legacy_serialize(t)
        recovered = embedding_blob_to_tensor(blob)
        assert torch.equal(t, recovered)

    def test_legacy_bfloat16(self):
        t = torch.randn(128, 320).bfloat16()
        blob = self._legacy_serialize(t)
        recovered = embedding_blob_to_tensor(blob)
        assert torch.equal(t, recovered)

    def test_legacy_float16(self):
        t = torch.randn(128, 320).half()
        blob = self._legacy_serialize(t)
        recovered = embedding_blob_to_tensor(blob)
        assert torch.equal(t, recovered)


# ---------------------------------------------------------------------------
# SQLite round-trip
# ---------------------------------------------------------------------------

class TestSQLiteRoundTrip:
    """Full write-to-DB, read-from-DB cycle with the new format."""

    def test_sqlite_round_trip(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('CREATE TABLE embeddings (sequence TEXT PRIMARY KEY, embedding BLOB)')

        tensors = {}
        for i in range(100):
            t = torch.randn(64 + i, 320).bfloat16()
            tensors[f"SEQ{i}"] = t
            c.execute("INSERT INTO embeddings VALUES (?, ?)", (f"SEQ{i}", tensor_to_embedding_blob(t)))
        conn.commit()

        # Read back
        for seq, original in tensors.items():
            c.execute("SELECT embedding FROM embeddings WHERE sequence = ?", (seq,))
            blob = c.fetchone()[0]
            recovered = embedding_blob_to_tensor(blob)
            assert torch.equal(original, recovered), f"Mismatch for {seq}"

        conn.close()


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

class TestPerformance:
    """Verify the new raw format is faster than torch.save/load."""

    def test_serialization_speedup(self):
        t = torch.randn(512, 1280).bfloat16()
        n_iters = 1000

        # New format
        start = time.perf_counter()
        for _ in range(n_iters):
            blob = tensor_to_embedding_blob(t)
            _ = embedding_blob_to_tensor(blob)
        new_time = time.perf_counter() - start

        # Old format (torch.save/load)
        start = time.perf_counter()
        for _ in range(n_iters):
            buf = io.BytesIO()
            torch.save(t.cpu(), buf)
            buf.seek(0)
            _ = torch.load(buf, map_location='cpu', weights_only=True)
        old_time = time.perf_counter() - start

        speedup = old_time / new_time
        print(f"\nBlob serialization benchmark ({n_iters} round-trips, 512x1280 bf16):")
        print(f"  Old (torch.save/load): {old_time:.2f}s")
        print(f"  New (raw bytes):       {new_time:.2f}s")
        print(f"  Speedup:               {speedup:.1f}x")

        assert speedup > 1.3, f"Expected at least 1.3x speedup, got {speedup:.1f}x"
