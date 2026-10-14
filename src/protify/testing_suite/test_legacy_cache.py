"""Old Protify caches convert into the feature store, and the rows that come out are the rows that went in.

Each test builds an old-format cache inside the test, with the test-local writer of the old bytes, and
converts it with `protify.legacy_cache`, which calls `fastplms.features.convert_rows`. The assertions are
on bytes: the store returns exactly what the decoder read, cast only to a dtype that holds it exactly.
"""

import sqlite3
import numpy as np
import pytest
import torch

from contextlib import closing

from fastplms.features import (
    CSR,
    RAGGED,
    ConversionMismatch,
    FeatureReader,
    FeatureStore,
    describe_file,
)


try:
    from src.protify import feature_cache, legacy_cache
    from src.protify.embedder import get_embedding_filename
    from src.protify.testing_suite.legacy_cache_writer import (
        LEGACY_BFLOAT16_CODE,
        dense_blob,
        write_sqlite_cache,
        write_torch_cache,
    )
except ImportError:
    from protify import feature_cache, legacy_cache
    from protify.embedder import get_embedding_filename
    from protify.testing_suite.legacy_cache_writer import (
        LEGACY_BFLOAT16_CODE,
        dense_blob,
        write_sqlite_cache,
        write_torch_cache,
    )


SEQUENCES = ["MKTAYIAKQR", "MKTAYIAKQRQISFVKSHFSRQ", "GGGGSGGGGS", "ACDEFGHIKL", "WWWWWWWWWWWWWW"]
SAE_SPARSE = "ESMC-300-SAE-l23-k16-c8192"  # k / codebook_dim = 0.002, so stored as coordinates
WIDTH = 8


def descriptor(model="ESM2-8", *, matrix=False, pooling=("mean",), dtype="float32", hidden=-1, max_length=2048):
    # An old SAE cache pooled the interior tokens only, so it converts under `excluded` and no other label.
    return feature_cache.embedding_descriptor(
        model, matrix_embed=matrix, pooling_types=list(pooling), hidden_state_index=hidden,
        max_length=max_length, dtype=dtype,
        special_tokens="excluded" if "-SAE-" in model else "included",
    )


def raw(tensor):
    """A tensor's bytes, so equality means bit for bit and not merely numerically equal."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def stored(directory, sequences):
    with FeatureReader.open(directory) as reader:
        return dict(zip(sequences, reader.read(sequences), strict=True))


def pooled_rows(dtype=torch.float32, width=WIDTH, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {sequence: torch.randn(width, generator=generator).to(dtype) for sequence in SEQUENCES}


def test_a_float32_sqlite_cache_converts_bit_for_bit(tmp_path):
    rows = pooled_rows()
    old = write_sqlite_cache(tmp_path / "old.db", rows)
    before = describe_file(old)

    outcome = legacy_cache.convert_legacy_cache(old, tmp_path, descriptor())

    assert outcome.dtype == "float32"
    assert (outcome.receipt.source_rows, outcome.receipt.written, outcome.receipt.verified) == (5, 5, 5)
    stored_rows = stored(outcome.directory, SEQUENCES)
    for sequence in SEQUENCES:
        assert raw(stored_rows[sequence]) == raw(rows[sequence])
    assert describe_file(old) == before  # the old cache is read, never changed


def test_the_commit_marker_records_where_the_rows_came_from(tmp_path):
    old = write_sqlite_cache(tmp_path / "old.db", pooled_rows())

    outcome = legacy_cache.convert_legacy_cache(old, tmp_path, descriptor())

    (segment,) = FeatureStore.read_only(outcome.directory).segments()
    origin = segment.metadata["conversion"]["origin"]
    assert origin["format"] == "protify_sqlite_blobs"
    assert origin["decoder"] == "protify.legacy_cache"
    assert origin["files"] == [describe_file(old)]


def test_bfloat16_blobs_of_both_encodings_convert_to_a_bfloat16_store(tmp_path):
    exact = pooled_rows(torch.bfloat16, seed=1)
    first_encoding = pooled_rows(torch.bfloat16, seed=2)
    rows = {}
    for sequence in SEQUENCES[:3]:
        rows[sequence] = dense_blob(exact[sequence])  # code 3, the bit pattern
    for sequence in SEQUENCES[3:]:
        rows[sequence] = dense_blob(first_encoding[sequence], code=LEGACY_BFLOAT16_CODE)  # float16 bytes
    old = write_sqlite_cache(tmp_path / "old.db", rows)

    outcome = legacy_cache.convert_legacy_cache(old, tmp_path, descriptor(dtype="bfloat16"))

    assert outcome.dtype == "bfloat16"
    stored_rows = stored(outcome.directory, SEQUENCES)
    for sequence in SEQUENCES[:3]:
        assert raw(stored_rows[sequence]) == raw(exact[sequence])
    for sequence in SEQUENCES[3:]:
        # The first encoding went through float16, and the decoder reads it back as it always did.
        assert raw(stored_rows[sequence]) == raw(first_encoding[sequence].to(torch.float16).to(torch.bfloat16))


def test_float16_rows_widen_exactly_when_the_run_stores_float32(tmp_path):
    rows = pooled_rows(torch.float16)
    old = write_sqlite_cache(tmp_path / "old.db", rows)

    outcome = legacy_cache.convert_legacy_cache(old, tmp_path, descriptor(dtype="float32"))

    assert outcome.dtype == "float32"
    stored_rows = stored(outcome.directory, SEQUENCES)
    for sequence in SEQUENCES:
        assert stored_rows[sequence].dtype == torch.float32
        assert raw(stored_rows[sequence]) == raw(rows[sequence].to(torch.float32))


def test_a_float16_cache_keeps_its_own_dtype_when_the_run_stores_float16(tmp_path):
    rows = pooled_rows(torch.float16)
    old = write_sqlite_cache(tmp_path / "old.db", rows)

    outcome = legacy_cache.convert_legacy_cache(old, tmp_path, descriptor(dtype="float16"))

    assert outcome.dtype == "float16"
    assert raw(stored(outcome.directory, SEQUENCES[:1])[SEQUENCES[0]]) == raw(rows[SEQUENCES[0]])


def test_a_cache_that_mixed_dtypes_across_runs_converts_to_the_dtype_that_holds_both(tmp_path):
    half = pooled_rows(torch.float16, seed=3)
    brain = pooled_rows(torch.bfloat16, seed=4)
    rows = {sequence: (half if position % 2 else brain)[sequence] for position, sequence in enumerate(SEQUENCES)}
    old = write_sqlite_cache(tmp_path / "old.db", rows)

    outcome = legacy_cache.convert_legacy_cache(old, tmp_path, descriptor(dtype="float16"))

    assert outcome.dtype == "float32"  # neither half nor brain floats hold the other
    stored_rows = stored(outcome.directory, SEQUENCES)
    for sequence in SEQUENCES:
        assert raw(stored_rows[sequence]) == raw(rows[sequence].to(torch.float32))


def test_a_store_serves_any_dtype_it_holds_exactly_and_no_narrower_one(tmp_path):
    old = write_sqlite_cache(tmp_path / "old.db", pooled_rows(torch.float32))
    outcome = legacy_cache.convert_legacy_cache(old, tmp_path, descriptor(dtype="float32"))

    for wanted in ("float32", "float16", "bfloat16"):
        assert feature_cache.locate_feature(tmp_path, descriptor(dtype=wanted)) == outcome.directory
    # float64 is wider than what the store holds, so a float64 run would need its own store.
    assert feature_cache.locate_feature(tmp_path, descriptor(dtype="float64")) != outcome.directory


def test_a_narrower_existing_store_skips_a_file_it_cannot_hold(tmp_path, capsys):
    narrow = tmp_path / "narrow"
    with feature_cache.CacheWriter(narrow, descriptor(dtype="bfloat16"), layout="dense") as writer:
        writer.add(["MAAA"], [torch.ones(WIDTH, dtype=torch.bfloat16)])
    old = write_sqlite_cache(tmp_path / "old.db", pooled_rows(torch.float32))

    outcome = legacy_cache.convert_legacy_cache(old, narrow, descriptor(dtype="bfloat16"))

    assert outcome is None
    assert "do not fit exactly" in capsys.readouterr().out
    assert feature_cache.missing_sequences(feature_cache.locate_feature(narrow, descriptor(dtype="bfloat16")), SEQUENCES) == tuple(SEQUENCES)


def test_sparse_sae_blobs_convert_to_a_csr_store(tmp_path):
    width = 2 * 8192  # max and mean pooled codebooks side by side
    rows = {}
    for position, sequence in enumerate(SEQUENCES):
        row = torch.zeros(width)
        active = torch.tensor([3 + position, 900 + position, 8192 + 7, 16000 - position])
        row[active] = torch.linspace(0.5, 2.0, 4) + position
        rows[sequence] = row
    old = write_sqlite_cache(tmp_path / "old.db", rows, sparse=True)

    outcome = legacy_cache.convert_legacy_cache(
        old, tmp_path, descriptor(SAE_SPARSE, pooling=("max", "mean")),
    )

    spec = FeatureStore.read_only(outcome.directory).spec
    assert (spec.layout, spec.width, spec.descriptor["special_tokens"]) == (CSR, width, "excluded")
    stored_rows = stored(outcome.directory, SEQUENCES)
    for sequence in SEQUENCES:
        assert raw(stored_rows[sequence]) == raw(rows[sequence])
    with FeatureReader.open(outcome.directory) as reader:
        coordinates = reader.read_sparse(SEQUENCES[:1])[0]
    assert coordinates.indices.tolist() == torch.nonzero(rows[SEQUENCES[0]]).flatten().tolist()


def test_an_old_sae_cache_is_not_converted_under_the_canonical_label(tmp_path):
    """Its rows pooled the interior tokens; the canonical descriptor says l + 2 rows, so the label would lie."""
    old = write_sqlite_cache(tmp_path / "old.db", {sequence: torch.ones(8192) for sequence in SEQUENCES}, sparse=False)
    canonical = feature_cache.embedding_descriptor(
        SAE_SPARSE, matrix_embed=False, pooling_types=["max"], hidden_state_index=-1, max_length=2048, dtype="float32",
    )

    with pytest.raises(ValueError, match="interior tokens only"):
        legacy_cache.convert_legacy_cache(old, tmp_path, canonical)


def test_a_dense_cache_of_a_sparse_storage_model_converts_to_the_same_layout(tmp_path):
    rows = {}
    for position, sequence in enumerate(SEQUENCES):
        row = torch.zeros(8192)
        row[[position, 100 + position]] = 1.5 + position
        rows[sequence] = row
    old = write_sqlite_cache(tmp_path / "old.db", rows, sparse=False)  # written before coordinates existed

    outcome = legacy_cache.convert_legacy_cache(old, tmp_path, descriptor(SAE_SPARSE, pooling=("max",)))

    assert FeatureStore.read_only(outcome.directory).spec.layout == CSR
    stored_rows = stored(outcome.directory, SEQUENCES)
    for sequence in SEQUENCES:
        assert torch.equal(stored_rows[sequence], rows[sequence])


def test_a_per_token_cache_converts_to_a_ragged_store(tmp_path):
    generator = torch.Generator().manual_seed(5)
    rows = {sequence: torch.randn(len(sequence) + 2, WIDTH, generator=generator) for sequence in SEQUENCES}
    old = write_sqlite_cache(tmp_path / "old.db", rows)

    outcome = legacy_cache.convert_legacy_cache(old, tmp_path, descriptor(matrix=True, pooling=()))

    assert FeatureStore.read_only(outcome.directory).spec.layout == RAGGED
    stored_rows = stored(outcome.directory, SEQUENCES)
    for sequence in SEQUENCES:
        assert stored_rows[sequence].shape == (len(sequence) + 2, WIDTH)
        assert raw(stored_rows[sequence]) == raw(rows[sequence])
    with FeatureReader.open(outcome.directory) as reader:
        assert reader.residue_counts(SEQUENCES) == [len(sequence) + 2 for sequence in SEQUENCES]


def test_a_pickled_dictionary_converts_and_flattens_single_row_matrices(tmp_path):
    rows = pooled_rows()
    rows[SEQUENCES[0]] = rows[SEQUENCES[0]].reshape(1, -1)  # the shape older runs pickled
    old = write_torch_cache(tmp_path / "old.pth", rows)

    outcome = legacy_cache.convert_legacy_cache(old, tmp_path, descriptor())

    assert outcome.receipt.verified == 5
    (segment,) = FeatureStore.read_only(outcome.directory).segments()
    assert segment.metadata["conversion"]["origin"]["format"] == "protify_torch_dict"
    stored_rows = stored(outcome.directory, SEQUENCES)
    for sequence in SEQUENCES:
        assert raw(stored_rows[sequence]) == raw(rows[sequence])


def test_a_raw_float32_blob_decodes_only_with_the_shape_and_converts_with_it(tmp_path):
    rows = pooled_rows()
    blobs = {sequence: row.numpy().tobytes() for sequence, row in rows.items()}  # no header, as the oldest runs wrote
    old = write_sqlite_cache(tmp_path / "old.db", blobs)
    with pytest.raises(ValueError, match="no fallback_shape"):
        legacy_cache.embedding_blob_to_tensor(blobs[SEQUENCES[0]])

    outcome = legacy_cache.convert_legacy_cache(old, tmp_path, descriptor())

    assert raw(stored(outcome.directory, SEQUENCES[:1])[SEQUENCES[0]]) == raw(rows[SEQUENCES[0]])


def test_converting_again_resumes_and_writes_nothing_twice(tmp_path):
    old = write_sqlite_cache(tmp_path / "old.db", pooled_rows())
    legacy_cache.convert_legacy_cache(old, tmp_path, descriptor())

    again = legacy_cache.convert_legacy_cache(old, tmp_path, descriptor())

    assert (again.receipt.written, again.receipt.skipped, again.receipt.verified) == (0, 5, 5)


def store_with_a_fresh_row(root, fresh):
    """A store that already holds one sequence, embedded afresh and so unlike the old cache's row for it."""
    with feature_cache.CacheWriter(root, descriptor(), layout="dense") as writer:
        writer.add(SEQUENCES[:1], [fresh])


def test_skip_existing_adds_only_what_the_store_lacks(tmp_path):
    rows = pooled_rows()
    old = write_sqlite_cache(tmp_path / "old.db", rows)
    fresh = torch.full((WIDTH,), 9.0)
    store_with_a_fresh_row(tmp_path / "store", fresh)

    outcome = legacy_cache.convert_legacy_cache(old, tmp_path / "store", descriptor(), skip_existing=True)

    assert (outcome.receipt.source_rows, outcome.receipt.written) == (4, 4)
    stored_rows = stored(outcome.directory, SEQUENCES)
    assert raw(stored_rows[SEQUENCES[0]]) == raw(fresh)
    for sequence in SEQUENCES[1:]:
        assert raw(stored_rows[sequence]) == raw(rows[sequence])


def test_a_store_that_holds_a_different_row_for_a_cached_sequence_fails_the_strict_comparison(tmp_path):
    old = write_sqlite_cache(tmp_path / "old.db", pooled_rows())
    store_with_a_fresh_row(tmp_path / "store", torch.full((WIDTH,), 9.0))

    with pytest.raises(ConversionMismatch, match="differs from the source"):
        legacy_cache.convert_legacy_cache(old, tmp_path / "store", descriptor())

    # The rows are committed before they are compared, so undoing the conversion means removing the
    # segments it named, which the commit markers identify by origin.
    segments = FeatureStore.read_only(feature_cache.locate_feature(tmp_path / "store", descriptor())).segments()
    assert sorted(segment.fingerprint.split("-")[0] for segment in segments) == ["convert", "rows"]


def test_an_empty_cache_is_refused(tmp_path):
    old = write_sqlite_cache(tmp_path / "old.db", {})

    with pytest.raises(ValueError, match="holds no embeddings"):
        legacy_cache.convert_legacy_cache(old, tmp_path, descriptor())


def test_a_per_token_row_of_the_wrong_rank_is_refused(tmp_path):
    old = write_sqlite_cache(tmp_path / "old.db", {"MAAA": torch.randn(WIDTH)})

    with pytest.raises(ValueError, match="per-token cache row must be"):
        legacy_cache.convert_legacy_cache(old, tmp_path, descriptor(matrix=True, pooling=()))


def test_the_command_line_reads_the_file_name_and_takes_the_pooling_order_from_the_caller(tmp_path):
    name = get_embedding_filename("ESMC-300", False, ["var", "mean"], "db", -1, 64)
    old = write_sqlite_cache(tmp_path / name, pooled_rows(torch.float16))

    assert legacy_cache.main([str(old), "--root", str(tmp_path / "sorted")]) == 0
    assert legacy_cache.main([str(old), "--root", str(tmp_path / "given"), "--pooling", "var", "mean"]) == 0

    sorted_store = next((tmp_path / "sorted").iterdir())
    given_store = next((tmp_path / "given").iterdir())
    assert FeatureStore.read_only(sorted_store).spec.descriptor["pooling"] == ["mean", "var"]
    assert FeatureStore.read_only(given_store).spec.descriptor["pooling"] == ["var", "mean"]
    assert sorted_store.name != given_store.name  # a different column order is a different feature
    assert FeatureStore.read_only(sorted_store).spec.dtype == torch.float16
    assert legacy_cache.main([str(old), "--root", str(tmp_path / "wide"), "--dtype", "float32"]) == 0
    assert FeatureStore.read_only(next((tmp_path / "wide").iterdir())).spec.dtype == torch.float32


@pytest.mark.parametrize(
    "arguments",
    [
        ("ESM2-8", False, ["mean"], "db", -1, 2048),
        ("ESM2-8", False, ["var", "mean", "max"], "pth", 6, 512),
        ("ESM2-8", True, ["mean"], "db", 3, 2048),
        ("ESMC-300-SAE-l23-k64-c8192", False, ["max", "mean"], "db", -1, 1024),
        ("CARBON-500M", False, ["mean"], "pth", -1, 2048),
    ],
)
def test_a_file_name_reads_back_to_the_arguments_that_named_it(arguments):
    model, matrix, pooling, extension, hidden, length = arguments
    name = get_embedding_filename(model, matrix, pooling, extension, hidden, length)

    parsed = legacy_cache.parse_cache_filename(name)

    assert parsed["model_name"] == model
    assert parsed["matrix_embed"] is matrix
    assert parsed["hidden_state_index"] == hidden
    assert parsed["max_length"] == length
    assert parsed["pooling_types"] == ([] if matrix else sorted(pooling))
    assert (parsed["carbon"] is not None) == (model == "CARBON-500M")


def test_a_carbon_cache_converts_under_the_revision_its_name_carries(tmp_path):
    name = get_embedding_filename("CARBON-500M", False, ["mean"], "db", -1, 64)
    old = write_sqlite_cache(tmp_path / name, pooled_rows())

    assert legacy_cache.main([str(old), "--root", str(tmp_path / "out")]) == 0

    spec = FeatureStore.read_only(next((tmp_path / "out").iterdir())).spec
    assert len(spec.descriptor["revision"]) == 40
    assert spec.descriptor["preprocessing"].startswith("carbon_")


def test_cached_rows_reads_both_formats_read_only(tmp_path):
    rows = pooled_rows()
    database = write_sqlite_cache(tmp_path / "old.db", rows)
    pickled = write_torch_cache(tmp_path / "old.pth", rows)

    with legacy_cache.cached_rows(database, kind="sqlite") as lookup:
        assert raw(lookup(SEQUENCES[0])) == raw(rows[SEQUENCES[0]])
        with pytest.raises(KeyError):
            lookup("NOT-THERE")
    with legacy_cache.cached_rows(pickled, kind="torch") as lookup:
        assert raw(lookup(SEQUENCES[1])) == raw(rows[SEQUENCES[1]])
    with (
        pytest.raises(ValueError, match="sqlite or torch"),
        legacy_cache.cached_rows(database, kind="json"),
    ):
        pass


def test_rows_are_decoded_in_the_order_they_were_written(tmp_path):
    # Not alphabetical, and the table's primary-key index is: a scan by that index would reorder them.
    order = ["WWWWWWWWWWWWWW", "ACDEFGHIKL", "MKTAYIAKQR", "GGGGSGGGGS"]
    rows = {sequence: pooled_rows()[sequence] for sequence in order}
    database = write_sqlite_cache(tmp_path / "old.db", rows)

    assert [sequence for sequence, _ in legacy_cache.sqlite_rows(database)] == order
    assert sorted(legacy_cache.cache_sequences(database, "sqlite")) == sorted(order)


def test_a_database_without_protifys_table_is_refused_by_every_reader(tmp_path):
    unrelated = tmp_path / "unrelated.db"
    with closing(sqlite3.connect(unrelated)) as connection:
        connection.execute("CREATE TABLE something_else (x INTEGER)")
        connection.commit()

    with pytest.raises(ValueError, match=r"embeddings\(sequence, embedding\) schema"):
        list(legacy_cache.sqlite_rows(unrelated))
    with pytest.raises(ValueError, match=r"embeddings\(sequence, embedding\) schema"):
        legacy_cache.cache_sequences(unrelated, "sqlite")
    with pytest.raises(ValueError, match=r"embeddings\(sequence, embedding\) schema"):
        legacy_cache.scan_cache(unrelated, "sqlite")
    with pytest.raises(ValueError, match=r"embeddings\(sequence, embedding\) schema"):
        legacy_cache.convert_legacy_cache(unrelated, tmp_path / "out", descriptor())
    assert not (tmp_path / "out").exists()


def test_a_table_lacking_a_column_is_refused(tmp_path):
    narrow = tmp_path / "narrow.db"
    with closing(sqlite3.connect(narrow)) as connection:
        connection.execute("CREATE TABLE embeddings (sequence TEXT)")
        connection.commit()

    with pytest.raises(ValueError, match="sequence and embedding columns"):
        list(legacy_cache.sqlite_rows(narrow))


def test_the_blob_dtype_is_read_from_the_header_alone():
    assert legacy_cache.blob_dtype(dense_blob(torch.zeros(3))) == torch.float32
    assert legacy_cache.blob_dtype(dense_blob(torch.zeros(3, dtype=torch.bfloat16))) == torch.bfloat16
    assert legacy_cache.blob_dtype(dense_blob(torch.zeros(3, dtype=torch.bfloat16), code=LEGACY_BFLOAT16_CODE)) == torch.bfloat16
    assert legacy_cache.blob_dtype(np.zeros(3, dtype=np.float32).tobytes()) is None
