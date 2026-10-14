"""Cached rows keep their representation whatever holds them, and reading never creates a cache.

`kind="store"` is the feature store Protify writes. `sqlite` and `torch` are the two formats Protify
wrote before it, read through the legacy decoder for code that inspects an old cache before it is
converted. The old files in these tests are written by the test-local writer of the old bytes.
"""

from __future__ import annotations

import sqlite3
import pytest
import torch

from contextlib import closing

from protify.data.cache_reader import cached_rows
from protify.testing_suite.embedding_store_builder import build_store
from protify.testing_suite.legacy_cache_writer import write_sqlite_cache, write_torch_cache


@pytest.mark.parametrize("kind", ["sqlite", "torch"])
@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_raw_rows_preserve_values_rank_and_dtype(tmp_path, kind, shape, dtype):
    values = torch.arange(6 if len(shape) == 2 else 4, dtype=dtype).reshape(shape)  # (*s)
    if kind == "sqlite":
        path = write_sqlite_cache(tmp_path / "cache # percent%.db", {"AC": values}, wal=False)
    else:
        path = write_torch_cache(tmp_path / "cache.pth", {"AC": values})
    before = path.read_bytes()
    with cached_rows(path, kind=kind) as lookup:
        actual = lookup("AC")  # (*s)
        assert actual.dtype == dtype
        assert actual.device.type == "cpu"
        assert torch.equal(actual, values)
        with pytest.raises(KeyError):
            lookup("missing")
    assert path.read_bytes() == before
    if kind == "sqlite":
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            lookup("AC")
    else:
        with pytest.raises(RuntimeError, match="closed"):
            lookup("AC")


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_store_rows_preserve_values_rank_and_dtype(tmp_path, matrix, dtype):
    values = torch.arange(15 if matrix else 4, dtype=dtype).reshape((5, 3) if matrix else (4,))  # (*s)
    directory = build_store(tmp_path, {"AC": values}, matrix=matrix, dtype=str(dtype).removeprefix("torch."))

    with cached_rows(directory, kind="store") as lookup:
        actual = lookup("AC")  # (*s)
        assert actual.dtype == dtype
        assert actual.device.type == "cpu"
        assert torch.equal(actual, values)
        with pytest.raises(KeyError):
            lookup("missing")
    with pytest.raises(ValueError, match="closed"):
        lookup("AC")


def test_a_missing_store_is_not_created(tmp_path):
    path = tmp_path / "absent"
    with pytest.raises(FileNotFoundError, match="No embedding store"), cached_rows(path, kind="store"):
        pass
    assert not path.exists()


def test_reading_a_store_changes_no_file_of_it(tmp_path):
    directory = build_store(tmp_path, {"AC": torch.ones(4)})
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in sorted(tmp_path.rglob("*")) if path.is_file()
    }

    with cached_rows(directory, kind="store") as lookup:
        lookup("AC")

    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in sorted(tmp_path.rglob("*")) if path.is_file()
    }
    assert after == before


def test_an_unknown_kind_is_refused(tmp_path):
    with pytest.raises(ValueError, match="cache kind"), cached_rows(tmp_path / "x", kind="json"):
        pass


def test_missing_sqlite_is_not_created(tmp_path):
    path = tmp_path / "absent.db"
    with pytest.raises(FileNotFoundError), cached_rows(path, kind="sqlite"):
        pass
    assert not path.exists()


def test_raw_legacy_float32_requires_an_explicit_shape(tmp_path):
    values = torch.arange(6, dtype=torch.float32).reshape(2, 3)  # (r=2, w=3)
    path = write_sqlite_cache(tmp_path / "raw.db", {"AC": values.numpy().tobytes()}, wal=False)  # no header
    with cached_rows(path, kind="sqlite") as lookup, pytest.raises(ValueError, match="no fallback_shape"):
        lookup("AC")
    with cached_rows(path, kind="sqlite", fallback_shape=lambda sequence: (len(sequence), 3)) as lookup:
        assert torch.equal(lookup("AC"), values)


def test_sqlite_errors_and_duplicate_rows_are_not_missing_features(tmp_path):
    path = tmp_path / "rows.db"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE embeddings (sequence TEXT, embedding BLOB)")
        connection.executemany("INSERT INTO embeddings VALUES (?, ?)", [("AC", b"x")] * 2)
        connection.commit()
    with cached_rows(path, kind="sqlite") as lookup, pytest.raises(ValueError, match="repeats"):
        lookup("AC")
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("DROP TABLE embeddings")
        connection.commit()
    with cached_rows(path, kind="sqlite") as lookup, pytest.raises(sqlite3.OperationalError):
        lookup("AC")


@pytest.mark.parametrize("contents", [[], {"AC": 5}])
def test_torch_requires_a_tensor_dictionary(tmp_path, contents):
    path = tmp_path / "rows.pth"
    torch.save(contents, path)
    with pytest.raises(ValueError), cached_rows(path, kind="torch") as lookup:
        lookup("AC")


def test_sqlite_only_needs_read_permissions(tmp_path):
    path = write_sqlite_cache(tmp_path / "rows.db", {"AC": torch.ones(2)}, wal=False)  # (w=2,)
    path.chmod(0o444)
    try:
        with cached_rows(path, kind="sqlite") as lookup:
            assert lookup("AC").tolist() == [1, 1]
        assert sorted(item.name for item in tmp_path.iterdir()) == ["rows.db"]
    finally:
        path.chmod(0o644)
