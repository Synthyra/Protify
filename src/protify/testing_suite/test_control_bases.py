"""The two control bases are generated where they are read and never stored.

A control's rows are a function of the sequence, so Protify makes them on request, through the embedding
engine's `SyntheticControlView`. These tests hold the canonical row convention (l + 2 token rows, every
statistic over all of them), the determinism that gives every split the same row for the same protein, the
distribution of the random control, the agreement of Protify's read with the engine's, and the absence of any
file: no store is written, no model is built, nothing is downloaded.

Shapes: `b` sequences, `l` residues, `v` one-hot columns, `d` random-vector columns.
"""

import pickle
import numpy as np
import pytest
import torch

from pathlib import Path


try:
    from src.protify import embedder as embedder_module
    from src.protify import feature_cache
    from src.protify.base_models import control_bases
    from src.protify.base_models.get_base_models import get_base_model
    from src.protify.data.branch_features import (
        join_branch_embeddings,
        parse_branch_spec,
        require_cached_branches,
    )
    from src.protify.embedder import Embedder, EmbeddingArguments
    from src.protify.testing_suite.embedding_store_builder import cache_descriptor
except ImportError:
    from protify import embedder as embedder_module
    from protify import feature_cache
    from protify.base_models import control_bases
    from protify.base_models.get_base_models import get_base_model
    from protify.data.branch_features import (
        join_branch_embeddings,
        parse_branch_spec,
        require_cached_branches,
    )
    from protify.embedder import Embedder, EmbeddingArguments
    from protify.testing_suite.embedding_store_builder import cache_descriptor


SEQUENCES = ["MKTAYIAKQR", "MKTAYIAKQRQISFVKSHFSRQ", "GGGGSGGGGS", "ACDEFGHIKL", "WWWWWWWWWWWWWW"]
ONE_HOT = control_bases.ONE_HOT
RANDOM = control_bases.RANDOM


def test_one_hot_mean_is_the_composition_over_all_l_plus_2_rows():
    sequence = "MKKA"  # l = 4, so 6 token rows
    pooled = control_bases.control_pooled(ONE_HOT, sequence, ("mean",), max_length=64)  # (v,)
    symbols = control_bases.ONE_HOT_SYMBOLS

    assert pooled.shape == (control_bases.ONE_HOT_WIDTH,)
    assert pooled[symbols.index("K")] == pytest.approx(2 / 6)
    assert pooled[symbols.index("M")] == pytest.approx(1 / 6)
    assert pooled[symbols.index("<cls>")] == pytest.approx(1 / 6)
    assert pooled[symbols.index("<eos>")] == pytest.approx(1 / 6)
    assert pooled.sum() == pytest.approx(1.0)


def test_one_hot_variance_is_the_population_variance_of_a_symbol_column():
    pooled = control_bases.control_pooled(ONE_HOT, "MKKA", ("mean", "var"), max_length=64)  # (2 * v,)
    mean, variance = np.split(pooled, 2)  # each (v,)

    assert np.allclose(variance, mean * (1 - mean))
    assert pooled[: control_bases.ONE_HOT_WIDTH].tolist() == control_bases.control_pooled(ONE_HOT, "MKKA", ("mean",), 64).tolist()


def test_one_hot_matrix_has_cls_first_eos_last_and_one_symbol_per_row():
    matrix = control_bases.control_matrix(ONE_HOT, "MKA", max_length=64)  # (5, v)
    symbols = control_bases.ONE_HOT_SYMBOLS

    assert matrix.shape == (5, control_bases.ONE_HOT_WIDTH)
    assert matrix.sum(dim=1).tolist() == [1.0] * 5
    assert [symbols[column] for column in matrix.argmax(dim=1)] == ["<cls>", "M", "K", "A", "<eos>"]
    assert np.allclose(matrix.mean(dim=0).numpy(), control_bases.control_pooled(ONE_HOT, "MKA", ("mean",), 64))


def test_a_long_sequence_is_cut_to_max_length_minus_2_residues_before_pooling():
    long = "ACDEFGHIKLMNPQRSTVWY" * 5

    for model in (ONE_HOT, RANDOM):
        assert control_bases.control_matrix(model, long, 12).shape[0] == 12  # 10 residues, CLS and EOS
    # A one-hot row reads only the kept residues, so the cut sequence and the long one agree.
    assert np.array_equal(control_bases.control_pooled(ONE_HOT, long, ("mean",), 12),
                          control_bases.control_pooled(ONE_HOT, long[:10], ("mean",), 12))
    with pytest.raises(ValueError, match="cannot hold one residue"):
        control_bases.control_matrix(ONE_HOT, long, max_length=2)


def test_an_unknown_residue_symbol_is_refused_not_dropped():
    with pytest.raises(ValueError, match="residue symbols"):
        control_bases.control_pooled(ONE_HOT, "MKJA", ("mean",), 64)


def test_random_rows_are_seeded_by_the_sequence_and_nothing_else():
    first = control_bases.control_pooled(RANDOM, "MKTAYIAKQR", ("mean", "var"), 2048)  # (2 * d,)
    again = control_bases.control_pooled(RANDOM, "MKTAYIAKQR", ("mean", "var"), 2048)
    other = control_bases.control_pooled(RANDOM, "MKTAYIAKQK", ("mean", "var"), 2048)

    assert first.shape == (2 * control_bases.RANDOM_WIDTH,) and first.dtype == np.float32
    assert np.array_equal(first, again)
    assert not np.allclose(first[: control_bases.RANDOM_WIDTH], other[: control_bases.RANDOM_WIDTH])
    # `mean` is the first half of `mean,var`, and the order asked decides the column order.
    mean_only = control_bases.control_pooled(RANDOM, "MKTAYIAKQR", ("mean",), 2048)
    var_first = control_bases.control_pooled(RANDOM, "MKTAYIAKQR", ("var", "mean"), 2048)
    assert np.array_equal(mean_only, first[: control_bases.RANDOM_WIDTH])
    assert np.array_equal(var_first[control_bases.RANDOM_WIDTH:], mean_only)


def test_random_pooled_statistics_follow_the_distribution_of_l_plus_2_noise_rows():
    # 200 distinct sequences of one length, so n = l + 2 = 12 rows each and the statistics are comparable.
    generator = np.random.default_rng(0)
    residues = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    sequences = ["".join(generator.choice(residues, size=10)) for _ in range(200)]
    rows = control_bases.control_rows(RANDOM, sequences, ("mean", "var"), 2048).numpy()  # (200, 2 * d)
    mean, variance = rows[:, : control_bases.RANDOM_WIDTH], rows[:, control_bases.RANDOM_WIDTH:]  # each (200, d)
    n = 12

    assert mean.mean() == pytest.approx(0.0, abs=0.01)
    assert mean.std() == pytest.approx(1 / np.sqrt(n), rel=0.02)
    # Population variance of n standard normals: (n - 1) / n in expectation.
    assert variance.mean() == pytest.approx((n - 1) / n, rel=0.01)
    assert variance.var() == pytest.approx(2 * (n - 1) / n**2, rel=0.05)


def test_the_random_scale_shrinks_with_the_row_count_so_it_carries_length_and_nothing_else():
    short = control_bases.control_pooled(RANDOM, "A" * 4, ("mean",), 2048)  # n = 6
    long = control_bases.control_pooled(RANDOM, "A" * 400, ("mean",), 2048)  # n = 402

    assert short.std() > 5 * long.std()


def test_pooled_rows_always_agree_with_the_token_rows_they_summarize():
    for model in (ONE_HOT, RANDOM):
        matrix = control_bases.control_matrix(model, "MKTAYIAKQR", 64)  # (l + 2, w)
        pooled = control_bases.control_pooled(model, "MKTAYIAKQR", ("mean", "var"), 64)  # (2 * w,)
        width = control_bases.control_width(model)

        assert matrix.shape == (12, width)
        assert np.allclose(pooled[:width], matrix.mean(dim=0).numpy(), atol=1e-6)
        assert np.allclose(pooled[width:], matrix.var(dim=0, unbiased=False).numpy(), atol=1e-6)


def test_protifys_read_is_the_engines_read():
    from foundry.embedding import ONE_HOT as ENGINE_ONE_HOT, RANDOM_VECTOR, SyntheticControlView, sequence_inventory

    inventory = sequence_inventory(SEQUENCES)
    for model, kind in ((ONE_HOT, ENGINE_ONE_HOT), (RANDOM, RANDOM_VECTOR)):
        engine = SyntheticControlView(kind, inventory, max_residues=62).pooled(SEQUENCES)  # (5, 2 * w)
        assert torch.equal(control_bases.control_rows(model, SEQUENCES, ("mean", "var"), 64), engine)


def test_pooling_a_control_cannot_ask_for_what_it_never_keeps():
    for model in (ONE_HOT, RANDOM):
        for pooling in (("max",), ("mean", "mean"), (), ("mean", "median")):
            with pytest.raises(ValueError, match="generates the pooled statistics"):
                control_bases.control_pooled(model, "MKTA", pooling, 64)
    with pytest.raises(ValueError, match="not a control base"):
        control_bases.control_width("ESMC-300")


def test_control_rows_stack_in_the_order_given_and_an_empty_list_keeps_its_width():
    rows = control_bases.control_rows(ONE_HOT, SEQUENCES, ("mean", "var"), 2048)  # (5, 2 * v)

    assert rows.shape == (5, 2 * control_bases.ONE_HOT_WIDTH) and rows.dtype == torch.float32
    assert control_bases.control_rows(ONE_HOT, [], ("mean",), 2048).shape == (0, control_bases.ONE_HOT_WIDTH)
    assert torch.equal(control_bases.control_rows(RANDOM, SEQUENCES[::-1], ("mean",), 2048).flip(0),
                       control_bases.control_rows(RANDOM, SEQUENCES, ("mean",), 2048))


# The reader: a control has a virtual directory and a generated reader, and nothing on disk.


@pytest.mark.parametrize("model", [ONE_HOT, RANDOM])
def test_a_control_directory_is_virtual_and_available_without_a_file(tmp_path, model):
    descriptor = cache_descriptor(model, pooling=("mean", "var"), max_length=512)
    directory = feature_cache.locate_feature(tmp_path, descriptor)

    assert directory == tmp_path / f"generated__{model}__vector__mean-var__len512"
    assert feature_cache.feature_directory(tmp_path, descriptor) == directory
    assert feature_cache.has_feature(directory)
    assert feature_cache.require_feature(directory) == directory
    assert feature_cache.missing_sequences(directory, SEQUENCES) == ()
    assert list(tmp_path.iterdir()) == []
    parsed = feature_cache.parse_generated(directory)
    assert (parsed.model, parsed.form, parsed.pooling, parsed.max_length) == (model, "vector", ("mean", "var"), 512)
    assert feature_cache.parse_generated(tmp_path / "protify__ESMC-300__ab12") is None


@pytest.mark.parametrize("model", [ONE_HOT, RANDOM])
def test_the_reader_makes_the_rows_a_package_will_read(tmp_path, model):
    descriptor = cache_descriptor(model, pooling=("mean",), max_length=64)
    directory = feature_cache.locate_feature(tmp_path, descriptor)
    expected = control_bases.control_rows(model, SEQUENCES, ("mean",), 64)  # (5, w)

    with feature_cache.open_reader(directory) as reader:
        assert reader.spec.width == control_bases.control_width(model) and reader.spec.layout == "dense"
        assert reader.missing(SEQUENCES) == () and SEQUENCES[0] in reader
        rows = reader.read(SEQUENCES)
    with feature_cache.row_lookup(directory) as lookup:
        single = lookup(SEQUENCES[2])
    read = feature_cache.read_embeddings(directory, SEQUENCES + SEQUENCES[:1])

    assert torch.equal(torch.stack(rows), expected)
    assert torch.equal(single, expected[2])
    assert list(read) == SEQUENCES and torch.equal(torch.stack(list(read.values())), expected)
    assert pickle.loads(pickle.dumps(feature_cache.open_reader(directory))).read(SEQUENCES[:1])[0].equal(expected[0])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("model", [ONE_HOT, RANDOM])
def test_a_control_matrix_is_generated_per_token(tmp_path, model):
    matrix = cache_descriptor(model, matrix=True, max_length=64)
    with feature_cache.open_reader(feature_cache.locate_feature(tmp_path, matrix)) as reader:
        rows = reader.read(["MKA"])
        assert reader.spec.layout == "ragged" and reader.spec.width == control_bases.control_width(model)
    assert rows[0].shape == (5, control_bases.control_width(model))  # (l + 2, w)

    with pytest.raises(ValueError, match="no layers"):
        feature_cache.feature_directory(tmp_path, cache_descriptor(model, hidden_state_index=3))


def test_nothing_writes_a_control_and_its_fingerprint_names_the_generator(tmp_path):
    descriptor = cache_descriptor(RANDOM, pooling=("mean",))
    with pytest.raises(ValueError, match="never stored"):
        feature_cache.CacheWriter(tmp_path, descriptor, layout="dense")
    directory = feature_cache.locate_feature(tmp_path, descriptor)

    first = feature_cache.store_fingerprint(directory)
    assert first == feature_cache.store_fingerprint(directory)
    assert first != feature_cache.store_fingerprint(feature_cache.locate_feature(tmp_path, cache_descriptor(ONE_HOT, pooling=("mean",))))
    assert feature_cache.parse_generated(directory).descriptor["generator"] == control_bases.CONTROL_SCHEMA


def test_an_old_control_cache_has_no_store_to_convert_into(tmp_path):
    refusal = feature_cache.legacy_conversion_refusal(cache_descriptor(ONE_HOT))

    assert "generated at load" in refusal
    assert feature_cache.legacy_conversion_refusal(cache_descriptor("ESM2-8")) is None


def test_the_embedder_embeds_a_control_without_a_model_a_download_or_a_file(tmp_path, monkeypatch):
    def refuse(*arguments, **keywords):
        raise AssertionError("a control builds no model")

    monkeypatch.setattr(embedder_module, "get_base_model", refuse)
    monkeypatch.setattr(embedder_module, "hf_hub_download", refuse)
    arguments = EmbeddingArguments(
        embedding_save_dir=str(tmp_path), embedding_pooling_types=["mean", "var"], max_length=64,
        download_embeddings=True, save_embeddings=True,
    )

    rows = Embedder(arguments, SEQUENCES)(ONE_HOT)

    assert set(rows) == set(SEQUENCES)
    assert torch.equal(torch.stack([rows[sequence] for sequence in SEQUENCES]),
                       control_bases.control_rows(ONE_HOT, SEQUENCES, ("mean", "var"), 64))
    assert list(tmp_path.iterdir()) == []


def test_a_control_has_no_base_model_to_build():
    for model in (ONE_HOT, RANDOM):
        with pytest.raises(ValueError, match="generated where they are read"):
            get_base_model(model)


# The branch path every Oracle cell reads.


def test_control_branch_specs_parse_and_refuse_what_a_control_lacks():
    spec = parse_branch_spec(f"{ONE_HOT}:mean")

    assert (spec.model_name, spec.pooling_types, spec.sparse) == (ONE_HOT, ("mean",), False)
    assert parse_branch_spec(f"{RANDOM}:mean,var").pooling_types == ("mean", "var")
    for token in (f"{RANDOM}:max", f"{ONE_HOT}:mean@3", f"{RANDOM}:median"):
        with pytest.raises(ValueError):
            parse_branch_spec(token)


@pytest.mark.parametrize("model", [ONE_HOT, RANDOM])
def test_a_control_branch_joins_with_no_cache_and_no_embedding(tmp_path, model):
    specs = [parse_branch_spec(f"{model}:mean")]
    arguments = {"embedding_save_dir": str(tmp_path), "sql": True, "max_length": 64}

    require_cached_branches(specs, SEQUENCES, arguments)  # refuses an incomplete cache; a control has none to lack
    joined, widths = join_branch_embeddings(specs, SEQUENCES, arguments)

    assert widths == (control_bases.control_width(model),)
    rows = joined.block(np.arange(len(SEQUENCES)))  # (5, w)
    assert np.array_equal(rows, control_bases.control_rows(model, SEQUENCES, ("mean",), 64).numpy())
    assert list(Path(tmp_path).iterdir()) == []
