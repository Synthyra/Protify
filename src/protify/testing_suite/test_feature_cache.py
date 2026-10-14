"""The feature-store layer under every Protify cache: keys, layouts, the writer, and the lookups.

These tests hold the contracts the rest of Protify leans on. A store's directory name decides whether a
later run finds an earlier run's rows, so the names are pinned. The writer commits a segment per window,
so a run that dies keeps what it finished. A store serves any dtype it holds exactly and is never
narrowed. A fingerprint is one digest for a store's committed content and refuses damaged bytes.
"""

import json
import os
import pytest
import torch

from pathlib import Path

from fastplms.features import CSR, DENSE, RAGGED, FeatureStore


try:
    from src.protify import feature_cache
    from src.protify.testing_suite.embedding_store_builder import build_store, cache_descriptor, write_store
    from src.protify.testing_suite.legacy_cache_writer import write_sqlite_cache
except ImportError:
    from protify import feature_cache
    from protify.testing_suite.embedding_store_builder import build_store, cache_descriptor, write_store
    from protify.testing_suite.legacy_cache_writer import write_sqlite_cache


SEQUENCES = ["MKTAYIAKQR", "MKTAYIAKQRQISFVKSHFSRQ", "GGGGSGGGGS", "ACDEFGHIKL", "WWWWWWWWWWWWWW"]
SPARSE_SAE = "ESMC-300-SAE-l23-k16-c8192"  # k / codebook_dim = 0.002, stored as coordinates


def test_explicit_family_roots_route_reads_and_writes_without_changing_keys(monkeypatch, tmp_path):
    logical = tmp_path / "logical"
    families = {name: str(tmp_path / name) for name in ("esmc_300", "esmc_600", "esmc_6b", "controls")}
    routing = {"logical_root": str(logical), "families": families}
    descriptor = cache_descriptor("ESMC-600", pooling=("mean",), dtype="float32")
    key = feature_cache.feature_key(descriptor)
    monkeypatch.setenv("PROTIFY_FEATURE_ROOTS", json.dumps(routing))
    rows = vectors(count=2, width=3)  # sequence -> (width,) float32
    with feature_cache.CacheWriter(logical, descriptor, layout=DENSE) as writer:
        writer.add(list(rows), list(rows.values()))
    target = Path(families["esmc_600"]) / key
    assert target.is_dir() and not logical.exists()
    assert feature_cache.locate_feature(logical, descriptor) == target
    recovered = feature_cache.read_embeddings(target, rows)
    assert all(raw(recovered[sequence]) == raw(row) for sequence, row in rows.items())
    assert feature_cache.feature_key(descriptor) == key
    for model, family in (("ESMC-300", "esmc_300"), ("ESMC-600-SAE-l27-k64-c16384", "esmc_600"),
                          ("ESMC-6B", "esmc_6b"), ("Random-Transformer", "controls")):
        request = cache_descriptor(model, pooling=("mean",))
        assert feature_cache.feature_directory(logical, request).parent == Path(families[family])
    # The two generated controls have no store, so no family root: their virtual directory stays under the logical root.
    for model in ("Random", "OneHot-Protein"):
        generated = feature_cache.feature_directory(logical, cache_descriptor(model, pooling=("mean",)))
        assert generated.parent == logical and feature_cache.parse_generated(generated) is not None
    other = tmp_path / "other"
    assert feature_cache.feature_directory(other, descriptor) == other / key
    with pytest.raises(ValueError, match="No family embedding storage"):
        feature_cache.feature_directory(logical, cache_descriptor("ESM2-8"))


def test_downloaded_legacy_conversion_also_writes_the_explicit_family_root(monkeypatch, tmp_path):
    from protify.legacy_cache import convert_legacy_cache

    logical, target = tmp_path / "logical", tmp_path / "family"
    monkeypatch.setenv("PROTIFY_FEATURE_ROOTS", json.dumps({
        "logical_root": str(logical), "families": {"esmc_600": str(target)},
    }))
    descriptor = cache_descriptor("ESMC-600", pooling=("mean",), dtype="float32")
    rows = vectors(count=2, width=3)  # sequence -> (width,) float32
    source = write_sqlite_cache(tmp_path / "old.db", rows)
    convert_legacy_cache(source, logical, descriptor)
    directory = feature_cache.feature_directory(logical, descriptor)
    assert directory.parent == target and not logical.exists()
    recovered = feature_cache.read_embeddings(directory, rows)
    assert all(raw(recovered[sequence]) == raw(row) for sequence, row in rows.items())


def vectors(count=None, width=8, dtype=torch.float32, seed=0):
    count = len(SEQUENCES) if count is None else count
    generator = torch.Generator().manual_seed(seed)
    return {sequence: torch.randn(width, generator=generator).to(dtype) for sequence in SEQUENCES[:count]}


def raw(tensor):
    """A tensor's bytes, so equality means bit for bit."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


@pytest.mark.parametrize(
    ("facts", "name"),
    [
        (
            dict(model_name="ESM2-8", pooling=("mean",), max_length=2048, dtype="float32"),
            "protify__ESM2-8__vector__mean__len2048__float32__7ce8bea5fdd085cd",
        ),
        (
            dict(model_name="ESM2-8", pooling=("mean", "var"), max_length=1024, dtype="float16"),
            "protify__ESM2-8__vector__mean-var__len1024__float16__fa78626840a79fd2",
        ),
        (
            dict(model_name="ESM2-8", matrix=True, max_length=2048, dtype="bfloat16"),
            "protify__ESM2-8__matrix__len2048__bfloat16__a659d428e8494616",
        ),
        (
            dict(model_name=SPARSE_SAE, pooling=("max",), max_length=2048, dtype="float32"),
            # SAE rows pool all l + 2 tokens, so their descriptor says special_tokens included and the key changed
            # once, with the canonical convention; an old interior-pooled cache keeps the excluded key.
            "protify__ESMC-300-SAE-l23-k16-c8192__vector__max__len2048__float32__dd9bcd07bd1286b9",
        ),
        (
            dict(model_name="ESM2-8", pooling=("mean",), hidden_state_index=4, max_length=2048, dtype="float32"),
            "protify__ESM2-8__vector__mean__hs4__len2048__float32__c7119515e9504188",
        ),
    ],
)
def test_store_directory_names_are_pinned(facts, name):
    """A renamed store would strand every cache already written under the old name."""
    assert feature_cache.feature_key(cache_descriptor(**facts)) == name


def test_the_descriptor_states_every_fact_that_changes_the_numbers():
    descriptor = cache_descriptor("ESM2-8", pooling=("mean", "var"), hidden_state_index=3, max_length=1024, dtype="float16")

    assert descriptor == {
        "schema": "protify_embedding_v1",
        "model": "ESM2-8",
        "revision": None,
        "preprocessing": None,
        "form": "vector",
        "pooling": ["mean", "var"],
        "hidden_state_index": 3,
        "max_length": 1024,
        "special_tokens": "included",
        "dtype": "float16",
    }


def test_a_matrix_has_no_pooling_and_an_sae_pools_the_special_tokens_too():
    matrix = cache_descriptor("ESM2-8", matrix=True, pooling=("mean",))
    sae = cache_descriptor(SPARSE_SAE, pooling=("max",))

    assert matrix["form"] == "matrix"
    assert matrix["pooling"] is None
    assert matrix["special_tokens"] == "included"
    # Canonical pooling covers all l + 2 rows for every representation, SAE max included.
    assert sae["special_tokens"] == "included"


@pytest.mark.parametrize(
    "change",
    [
        {"model_name": "ESM2-35"},
        {"pooling": ("var", "mean")},
        {"pooling": ("mean", "var")},
        {"hidden_state_index": 3},
        {"max_length": 1024},
        {"dtype": "float16"},
        {"matrix": True},
    ],
)
def test_every_fact_that_changes_the_numbers_changes_the_key(change):
    base = cache_descriptor("ESM2-8", pooling=("mean",))

    assert feature_cache.feature_key(cache_descriptor(**{"model_name": "ESM2-8", "pooling": ("mean",), **change})) != (
        feature_cache.feature_key(base)
    )


def test_pooling_order_is_part_of_the_key():
    """The old file name sorted the pooling types, so it could not tell these two apart."""
    forward = cache_descriptor(pooling=("mean", "var"))
    reverse = cache_descriptor(pooling=("var", "mean"))

    assert feature_cache.feature_key(forward) != feature_cache.feature_key(reverse)


def test_the_layout_follows_the_representation():
    assert feature_cache.embedding_layout(cache_descriptor("ESM2-8")) == DENSE
    assert feature_cache.embedding_layout(cache_descriptor("ESM2-8", matrix=True)) == RAGGED
    assert feature_cache.embedding_layout(cache_descriptor(SPARSE_SAE, pooling=("max",))) == CSR
    # A matrix is never sparse, even for an SAE model.
    assert feature_cache.embedding_layout(cache_descriptor(SPARSE_SAE, matrix=True)) == RAGGED


def test_only_carbon_models_have_a_recorded_revision():
    assert feature_cache.carbon_cache_identity("ESM2-8") is None
    assert feature_cache.carbon_cache_identity("ESMC-300") is None


def test_dtype_helpers_widen_exactly_and_refuse_what_no_cache_stores():
    assert feature_cache.dtype_name(torch.float32) == "float32"
    assert feature_cache.dtype_name("torch.bfloat16") == "bfloat16"
    with pytest.raises(ValueError, match="stores"):
        feature_cache.dtype_name(torch.int64)
    assert feature_cache.holds("float32", "float16") and feature_cache.holds("float32", "bfloat16")
    assert not feature_cache.holds("float16", "float32")
    assert not feature_cache.holds("float16", "bfloat16") and not feature_cache.holds("bfloat16", "float16")
    assert feature_cache.narrowest_holding("float16") == "float16"
    assert feature_cache.narrowest_holding("float16", "bfloat16") == "float32"
    assert feature_cache.narrowest_holding("float32", "float64") == "float64"


def test_a_writer_commits_one_segment_per_full_window(tmp_path):
    rows = vectors()
    descriptor = cache_descriptor()

    with feature_cache.CacheWriter(tmp_path, descriptor, layout=DENSE, window_rows=2) as writer:
        assert writer.directory is None  # nothing committed, so no store yet
        writer.add(list(rows), list(rows.values()))

    assert (writer.windows, writer.rows_written) == (3, 5)  # 2 + 2 + the closing 1
    store = FeatureStore.read_only(writer.directory)
    assert len(store.segments()) == 3
    assert store.missing(list(rows)) == ()
    assert feature_cache.missing_sequences(writer.directory, [*rows, "AAAA"]) == ("AAAA",)


def test_a_run_that_raises_keeps_its_committed_windows_and_drops_the_open_one(tmp_path):
    rows = vectors(3)
    descriptor = cache_descriptor()

    with (
        pytest.raises(RuntimeError, match="died"),
        feature_cache.CacheWriter(tmp_path, descriptor, layout=DENSE, window_rows=2) as writer,
    ):
        writer.add(list(rows), list(rows.values()))  # commits two, buffers the third
        raise RuntimeError("the run died")

    directory = feature_cache.locate_feature(tmp_path, descriptor)
    assert feature_cache.missing_sequences(directory, list(rows)) == (SEQUENCES[2],)


def test_a_restart_embeds_only_what_the_store_lacks(tmp_path):
    rows = vectors()
    descriptor = cache_descriptor()
    write_store(tmp_path, descriptor, {sequence: rows[sequence] for sequence in SEQUENCES[:2]})

    directory = feature_cache.locate_feature(tmp_path, descriptor)
    lacking = feature_cache.missing_sequences(directory, SEQUENCES)
    with feature_cache.CacheWriter(tmp_path, descriptor, layout=DENSE) as writer:
        writer.add(list(lacking), [rows[sequence] for sequence in lacking])

    assert lacking == tuple(SEQUENCES[2:])
    assert writer.rows_written == 3
    stored = feature_cache.read_embeddings(directory, SEQUENCES)
    for sequence in SEQUENCES:
        assert raw(stored[sequence]) == raw(rows[sequence])


def test_a_sequence_added_twice_in_one_window_is_refused(tmp_path):
    rows = vectors(1)
    writer = feature_cache.CacheWriter(tmp_path, cache_descriptor(), layout=DENSE)
    writer.add(list(rows), list(rows.values()))

    with pytest.raises(ValueError, match="twice"):
        writer.add(list(rows), list(rows.values()))


def test_a_window_skips_rows_another_run_committed_first(tmp_path):
    rows = vectors(2)
    first, second = SEQUENCES[:2]
    descriptor = cache_descriptor()
    late = feature_cache.CacheWriter(tmp_path, descriptor, layout=DENSE, label="late")
    late.add([first, second], [rows[first], rows[second]])  # buffered: the window is not full

    write_store(tmp_path, descriptor, {first: rows[first]})  # a parallel run commits `first`
    late.flush()

    assert late.rows_written == 1
    stored = feature_cache.read_embeddings(late.directory, [first, second])
    assert raw(stored[first]) == raw(rows[first]) and raw(stored[second]) == raw(rows[second])


def test_a_writer_rejects_settings_it_cannot_honor(tmp_path):
    with pytest.raises(ValueError, match="dense, csr, or ragged"):
        feature_cache.CacheWriter(tmp_path, cache_descriptor(), layout="coo")
    with pytest.raises(ValueError, match="positive"):
        feature_cache.CacheWriter(tmp_path, cache_descriptor(), layout=DENSE, window_rows=0)


def test_a_writer_refuses_a_store_of_another_layout(tmp_path):
    descriptor = cache_descriptor()
    write_store(tmp_path, descriptor, vectors(1))

    with (
        pytest.raises(ValueError, match="storage policy changed"),
        feature_cache.CacheWriter(tmp_path, descriptor, layout=RAGGED) as writer,
    ):
        writer.add(["ACDEFGHIKL"], [torch.randn(12, 8)])


def test_a_float32_store_serves_float16_and_bfloat16_requests(tmp_path):
    directory = build_store(tmp_path, vectors(), dtype="float32")

    assert feature_cache.locate_feature(tmp_path, cache_descriptor(dtype="float16")) == directory
    assert feature_cache.locate_feature(tmp_path, cache_descriptor(dtype="bfloat16")) == directory
    assert feature_cache.locate_feature(tmp_path, cache_descriptor(dtype="float32")) == directory


def test_a_narrow_store_never_serves_a_wider_request(tmp_path):
    for held in ("float16", "bfloat16"):
        directory = build_store(tmp_path, vectors(dtype=getattr(torch, held)), dtype=held)

        located = feature_cache.locate_feature(tmp_path, cache_descriptor(dtype="float32"))

        assert located != directory
        assert not feature_cache.has_feature(located)
        assert located == feature_cache.feature_directory(tmp_path, cache_descriptor(dtype="float32"))
    # And each serves only its own dtype.
    assert feature_cache.locate_feature(tmp_path, cache_descriptor(dtype="float16")).name.endswith(
        feature_cache.feature_key(cache_descriptor(dtype="float16")).rsplit("__", 1)[-1]
    )


def test_a_writer_widens_narrower_rows_into_a_float32_store_exactly(tmp_path):
    directory = build_store(tmp_path, vectors(1), dtype="float32")
    half = vectors(dtype=torch.float16, seed=4)
    lacking = SEQUENCES[1:]

    with feature_cache.CacheWriter(tmp_path, cache_descriptor(dtype="float16"), layout=DENSE) as writer:
        writer.add(lacking, [half[sequence] for sequence in lacking])

    assert writer.directory == directory  # the float32 store took them: no float16 store was made
    stored = feature_cache.read_embeddings(directory, lacking)
    for sequence in lacking:
        assert stored[sequence].dtype == torch.float32
        assert raw(stored[sequence]) == raw(half[sequence].to(torch.float32))


def test_a_writer_never_rounds_wider_rows_into_a_narrower_store(tmp_path):
    build_store(tmp_path, vectors(1, dtype=torch.float16), dtype="float16")

    with (
        pytest.raises(ValueError, match="would round float32 rows"),
        feature_cache.CacheWriter(tmp_path, cache_descriptor(dtype="float16"), layout=DENSE) as writer,
    ):
        writer.add([SEQUENCES[1]], [torch.randn(8)])


def test_a_sparse_sae_cache_round_trips_its_exact_nonzeros(tmp_path):
    width = 8192
    descriptor = cache_descriptor(SPARSE_SAE, pooling=("max",))
    generator = torch.Generator().manual_seed(5)
    rows = {}
    for sequence in SEQUENCES:
        vector = torch.zeros(width)
        active = torch.randperm(width, generator=generator)[:16]
        vector[active] = torch.rand(16, generator=generator) + 0.5
        rows[sequence] = vector

    directory = write_store(tmp_path, descriptor, rows)

    spec = FeatureStore.read_only(directory).spec
    assert (spec.layout, spec.width) == (CSR, width)
    stored = feature_cache.read_embeddings(directory, SEQUENCES)
    for sequence in SEQUENCES:
        assert stored[sequence].shape == (width,)
        assert raw(stored[sequence]) == raw(rows[sequence])


def test_a_matrix_cache_keeps_every_token_row_of_each_sequence(tmp_path):
    generator = torch.Generator().manual_seed(6)
    rows = {sequence: torch.randn(len(sequence) + 2, 4, generator=generator) for sequence in SEQUENCES}  # BOS and EOS kept
    directory = build_store(tmp_path, rows, matrix=True)

    assert FeatureStore.read_only(directory).spec.layout == RAGGED
    stored = feature_cache.read_embeddings(directory, SEQUENCES)
    for sequence in SEQUENCES:
        assert stored[sequence].shape == (len(sequence) + 2, 4)
        assert raw(stored[sequence]) == raw(rows[sequence])


def test_rows_read_into_memory_outlive_the_reader_and_do_not_alias_the_store(tmp_path):
    rows = vectors()
    directory = build_store(tmp_path, rows)

    stored = feature_cache.read_embeddings(directory, SEQUENCES)  # the reader closed inside
    stored[SEQUENCES[0]].zero_()

    again = feature_cache.read_embeddings(directory, SEQUENCES)
    assert raw(again[SEQUENCES[0]]) == raw(rows[SEQUENCES[0]])
    assert feature_cache.read_embeddings(directory, []) == {}


def test_a_missing_sequence_is_a_key_error_and_never_a_silent_zero(tmp_path):
    directory = build_store(tmp_path, vectors(2))

    with pytest.raises(KeyError):
        feature_cache.read_embeddings(directory, [SEQUENCES[0], "NOTSTORED"])
    with feature_cache.row_lookup(directory) as lookup:
        assert lookup(SEQUENCES[0]).shape == (8,)
        with pytest.raises(KeyError):
            lookup("NOTSTORED")


def test_a_reader_refuses_reads_after_it_closes(tmp_path):
    directory = build_store(tmp_path, vectors(2))
    reader = feature_cache.open_reader(directory)
    row = reader.read([SEQUENCES[0]])[0]
    reader.close()

    assert row.shape == (8,)  # a row taken before the close stays valid
    with pytest.raises(ValueError, match="closed"):
        reader.read([SEQUENCES[0]])


def test_missing_sequences_lists_every_distinct_sequence_when_there_is_no_store(tmp_path):
    assert feature_cache.missing_sequences(tmp_path / "none", ["B", "A", "B"]) == ("B", "A")


def test_require_feature_says_how_to_make_a_missing_store(tmp_path):
    with pytest.raises(FileNotFoundError, match="Embed the sequences first"):
        feature_cache.require_feature(tmp_path / "nothing")


def test_require_feature_names_an_old_cache_and_the_command_that_converts_it(tmp_path):
    old = write_sqlite_cache(tmp_path / "ESM2-8_False_len2048_mean.db", vectors(1))

    with pytest.raises(FileNotFoundError, match=r"python -m protify\.legacy_cache") as failure:
        feature_cache.require_feature(tmp_path / "store", [old])

    assert str(old) in str(failure.value)


def test_a_directory_without_a_feature_file_is_no_store(tmp_path):
    assert not feature_cache.has_feature(tmp_path)
    assert feature_cache.has_feature(build_store(tmp_path, vectors(1)))


def test_the_fingerprint_is_stable_and_follows_the_committed_content(tmp_path):
    rows = vectors()
    descriptor = cache_descriptor()
    directory = write_store(tmp_path, descriptor, {SEQUENCES[0]: rows[SEQUENCES[0]]})

    first = feature_cache.store_fingerprint(directory)
    assert feature_cache.store_fingerprint(directory) == first
    assert len(first) == 64

    write_store(tmp_path, descriptor, {SEQUENCES[1]: rows[SEQUENCES[1]]})  # a second segment
    assert feature_cache.store_fingerprint(directory) != first


def test_the_fingerprint_refuses_a_part_that_no_longer_matches_its_marker(tmp_path):
    directory = build_store(tmp_path, vectors())
    part = next(directory.glob("segments/*/part-*.safetensors"))
    part_bytes = bytearray(part.read_bytes())
    part_bytes[-1] ^= 0xFF
    os.chmod(part, 0o666)
    part.write_bytes(bytes(part_bytes))

    with pytest.raises(ValueError, match="digest"):
        feature_cache.store_fingerprint(directory)
