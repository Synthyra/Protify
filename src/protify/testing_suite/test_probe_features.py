"""The probe matrix's inputs: a stored base's canonical store or a generated control, read as pooled branches.

A tiny random ESM++ fills a canonical store on CPU (`canonical_store_builder`), so every read here goes through
the engine's own `open_canonical_view`. Each branch spec must read the stream and the columns the table of
`protify.data.probe_features` says, and an arm that cannot be read (SAE max from a random-init store) must refuse.

Shapes: `b` proteins, `c` SAE codebook (8192), `d` hidden width (32), `w` columns of one branch.
"""

import pickle
import numpy as np
import pytest

from pathlib import Path

from foundry.embedding import ONE_HOT, RANDOM_VECTOR, open_canonical_view, sequence_inventory
from protify.base_models import control_bases
from protify.data.branch_features import parse_branch_spec
from protify.data.probe_features import (
    MEAN_HALF,
    WHOLE,
    branch_stream,
    open_probe_features,
    probe_features_record,
    probe_provenance,
    validate_probe_features,
)
from protify.testing_suite import canonical_store_builder as builder


SEQUENCES = ["MKTAYIAKQRQISF", "MGGGGSGGGGSAAW", "MACDEFGHIKLNPQ", "MWWWWWWWWWWWWWWKK", "MKTAYIAK", "MLLLLLLLLLL"]
SAE = "ESMC-300-SAE-l23-k64-c8192:max"
DENSE = "ESMC-300:mean,var"
MEAN = "ESMC-300:mean"


@pytest.fixture(scope="module")
def stores(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("canonical")
    return {
        "esmc_small": builder.write_canonical_store(root / "esmc_small", SEQUENCES, base_id="esmc_small"),
        "esmc_small_random": builder.write_canonical_store(root / "random", SEQUENCES, base_id="esmc_small_random", seed=7),
    }


def test_the_setting_names_exactly_one_source() -> None:
    stored = probe_features_record("esmc_small", root=Path("/stores/esmc_small").resolve())
    generated = probe_features_record("random_vectors", control=RANDOM_VECTOR)

    assert (stored["kind"], stored["control"]) == ("stored", None) and (generated["kind"], generated["root"]) == ("generated", None)
    for bad in ({**stored, "root": "relative/path"}, {**stored, "control": ONE_HOT}, {**generated, "control": "noise"},
                {**generated, "root": str(Path("/x").resolve())}, {**stored, "kind": "remote"}, {**stored, "schema": "other"},
                {key: value for key, value in stored.items() if key != "base"}):
        with pytest.raises(ValueError):
            validate_probe_features(bad)


def test_each_branch_spec_reads_its_stream_and_columns() -> None:
    assert branch_stream(parse_branch_spec(SAE)) == ("sae_max", WHOLE)
    assert branch_stream(parse_branch_spec(DENSE)) == ("final_mean_var", WHOLE)
    assert branch_stream(parse_branch_spec(MEAN)) == ("final_mean_var", MEAN_HALF)
    for token in ("ESMC-300:mean@5", "ESMC-300:var", "ESMC-300-SAE-l23-k64-c8192:mean"):
        with pytest.raises(ValueError):
            branch_stream(parse_branch_spec(token))


def test_the_both_arm_joins_sae_max_beside_the_final_mean_and_variance(stores) -> None:
    record = probe_features_record("esmc_small", root=stores["esmc_small"])
    joined = open_probe_features(record, [SAE, DENSE], SEQUENCES, batch_size=4)  # branches (c, 2d)

    assert joined.widths == (builder.CODEBOOK, 2 * builder.HIDDEN_SIZE)
    block = joined.block(np.arange(len(SEQUENCES)))  # (b, c + 2d)
    with open_canonical_view(stores["esmc_small"], sequence_inventory(SEQUENCES)) as view:
        expected = view.pooled(SEQUENCES, ["sae_max", "final_mean_var"]).numpy()  # (b, c + 2d)

    assert block.shape == (len(SEQUENCES), builder.CODEBOOK + 2 * builder.HIDDEN_SIZE)
    assert np.array_equal(block, expected)
    assert (block[:, : builder.CODEBOOK] > 0).any(), "max over l + 2 rows keeps active codes"


def test_the_mean_final_arm_is_the_first_half_of_the_final_mean_and_variance(stores) -> None:
    record = probe_features_record("esmc_small", root=stores["esmc_small"])
    joined = open_probe_features(record, [MEAN], SEQUENCES, batch_size=4)  # (d,)
    block = joined.block(np.arange(len(SEQUENCES)))  # (b, d)

    assert joined.widths == (builder.HIDDEN_SIZE,)
    with open_canonical_view(stores["esmc_small"], sequence_inventory(SEQUENCES)) as view:
        both = view.pooled(SEQUENCES, ["final_mean_var"]).numpy()  # (b, 2d)
        rows = view.rows("final_hidden", SEQUENCES)  # b of (l + 2, d)
    assert np.array_equal(block, both[:, : builder.HIDDEN_SIZE])
    # The stored mean is the mean over every l + 2 row, CLS and EOS included.
    np.testing.assert_allclose(block[0], rows[0].float().mean(dim=0).numpy(), atol=2e-2)


def test_a_block_follows_the_rows_requested_in_order_with_duplicates(stores) -> None:
    joined = open_probe_features(probe_features_record("esmc_small", root=stores["esmc_small"]), [MEAN], SEQUENCES, batch_size=4)
    ordered = joined.block(np.array([3, 0, 3]))  # (3, d)
    everything = joined.block(np.arange(len(SEQUENCES)))  # (b, d)

    assert np.array_equal(ordered, everything[[3, 0, 3]])
    assert pickle.loads(pickle.dumps(joined)).block(np.array([1])).tolist() == everything[[1]].tolist()


def test_the_random_init_store_reads_the_mean_but_has_no_sae_to_read(stores) -> None:
    record = probe_features_record("esmc_small_random", root=stores["esmc_small_random"])
    joined = open_probe_features(record, [MEAN], SEQUENCES, batch_size=4)

    assert joined.widths == (builder.HIDDEN_SIZE,)
    with pytest.raises(KeyError, match="sae_max"):
        open_probe_features(record, [SAE, DENSE], SEQUENCES, batch_size=4)


def test_a_codebook_the_store_does_not_have_is_refused(stores) -> None:
    record = probe_features_record("esmc_small", root=stores["esmc_small"])

    with pytest.raises(ValueError, match="codebook of 16384"):
        open_probe_features(record, ["ESMC-300-SAE-l23-k64-c16384:max", DENSE], SEQUENCES, batch_size=4)


@pytest.mark.parametrize(("control", "model", "width"), [(RANDOM_VECTOR, control_bases.RANDOM, 960), (ONE_HOT, control_bases.ONE_HOT, 28)])
def test_a_generated_control_reads_the_engines_rows_and_stores_nothing(tmp_path, control, model, width) -> None:
    record = probe_features_record(model, control=control)
    joined = open_probe_features(record, [f"{model}:mean"], SEQUENCES, batch_size=16)  # (w,)
    full = open_probe_features(record, [f"{model}:mean,var"], SEQUENCES, batch_size=16)  # (2w,)

    assert joined.widths == (width,) and full.widths == (2 * width,)
    block, both = joined.block(np.arange(len(SEQUENCES))), full.block(np.arange(len(SEQUENCES)))  # (b, w), (b, 2w)
    assert np.array_equal(block, both[:, :width])
    assert np.array_equal(both, control_bases.control_rows(model, SEQUENCES, ("mean", "var"), 2048).numpy())
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="no sae_max stream"):
        open_probe_features(record, [SAE, f"{model}:mean,var"], SEQUENCES, batch_size=16)


def test_a_generated_control_answers_a_batch_larger_than_one_read(tmp_path) -> None:
    sequences = [("M" + "ACDEFGHIKLMNPQRSTVWY"[index % 20] * (8 + index % 5)) + "K" * index for index in range(150)]
    joined = open_probe_features(probe_features_record("random_vectors", control=RANDOM_VECTOR), [f"{control_bases.RANDOM}:mean"], sequences, batch_size=128)

    block = joined.block(np.arange(150))  # (150, 960), read in blocks of the control's budget
    assert block.shape == (150, 960)
    assert np.array_equal(block[:3], control_bases.control_rows(control_bases.RANDOM, sequences[:3], ("mean",), 2048).numpy())


def test_a_batch_beyond_the_reader_budget_is_refused(stores) -> None:
    with pytest.raises(ValueError, match="rows a read may return"):
        open_probe_features(probe_features_record("esmc_small", root=stores["esmc_small"]), [MEAN], SEQUENCES, batch_size=100000)


def test_provenance_names_the_store_and_its_streams_or_the_controls_arithmetic(stores) -> None:
    stored = probe_provenance(probe_features_record("esmc_small", root=stores["esmc_small"]), [SAE, DENSE])
    random = probe_provenance(probe_features_record("esmc_small_random", root=stores["esmc_small_random"]), [MEAN])
    generated = probe_provenance(probe_features_record("one_hot", control=ONE_HOT), [f"{control_bases.ONE_HOT}:mean"])

    assert stored["kind"] == "stored" and stored["special_tokens"] == "kept" and set(stored["streams"]) == {"sae_max", "final_mean_var"}
    assert stored["random_init"] is None and random["random_init"]["kind"] == "random_init" and set(random["streams"]) == {"final_mean_var"}
    assert generated["kind"] == "generated" and generated["generator"]["special_tokens"] == "kept" and generated["generator"]["width"] == 28
    assert generated["generator"]["kind"] == ONE_HOT


def test_a_generated_control_makes_each_proteins_rows_once_and_in_parallel_equal_to_the_engines(tmp_path, monkeypatch) -> None:
    from protify.data import probe_features

    sequences = [("M" + "ACDEFGHIKLMNPQRSTVWY"[index % 20] * (8 + index % 5)) + "K" * index for index in range(40)]
    record = probe_features_record("random_vectors", control=RANDOM_VECTOR)
    expected = control_bases.control_rows(control_bases.RANDOM, sequences, ("mean", "var"), 2048).numpy()  # (b, 2w)
    calls = []
    original = probe_features._pooled_chunk
    with monkeypatch.context() as counting:
        counting.setattr(probe_features, "_pooled_chunk", lambda kind, texts: calls.append(len(texts)) or original(kind, texts))
        joined = open_probe_features(record, [f"{control_bases.RANDOM}:mean,var"], sequences, batch_size=16)

    after_opening = len(calls)
    for _ in range(3):
        assert np.array_equal(joined.block(np.arange(40)), expected)
        assert np.array_equal(joined.block(np.asarray([7, 3, 3])), expected[[7, 3, 3]])
    assert len(calls) == after_opening == 1, "every protein's rows are made when the cell opens, and a batch only reads them"
    assert list(tmp_path.iterdir()) == []
    # More than PARALLEL_FROM proteins go to worker processes, whose rows are the same function of the sequence.
    monkeypatch.setattr(probe_features, "PARALLEL_FROM", 8)
    monkeypatch.setattr(probe_features, "CONTROL_WORKERS", 2)
    parallel = probe_features.MaterializedControl(probe_features.SyntheticControlView(RANDOM_VECTOR, sequence_inventory(sequences), budget=probe_features.GENERATED_BUDGET), workers=2)
    assert np.array_equal(parallel.pooled(sequences).numpy(), expected)
