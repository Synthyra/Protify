"""The Atlas proteome profile as a probe branch: a fixture store in the layout `production_suite.atlas_profile` writes.

The store is a FastPLMs `profile` feature keyed by the SHA-256 of the normalized sequence, beside a `profile.json` that names it.
A tiny canonical store (`canonical_store_builder`) supplies the SAE and mean and variance branches it joins, so the whole third
branch reads through `open_probe_features` as a cell reads it.

Shapes: `b` proteins, `p` axis proteins (7 here, 20,652 in the real axis), `c` SAE codebook (8192), `d` hidden width (32).
"""

import json
import pickle
import numpy as np
import pytest
import torch

from pathlib import Path

from fastplms.features import DENSE, FeatureStore, write_rows
from foundry.embedding import plain_feature
from protify.data.atlas_profile_feature import AtlasProfileFeature, profile_provenance, read_profile_manifest
from protify.data.branch_features import PROFILE_MODEL, parse_branch_spec
from protify.data.probe_features import PROFILE_STREAM, WHOLE, branch_stream, open_probe_features, probe_features_record, probe_provenance
from protify.testing_suite import canonical_store_builder as builder


SEQUENCES = ["MKTAYIAKQRQISF", "MGGGGSGGGGSAAW", "MACDEFGHIKLNPQ", "MWWWWWWWWWWWWWWKK", "MKTAYIAK", "MLLLLLLLLLL"]
AXIS_SIZE = 7  # p
SAE = "ESMC-300-SAE-l23-k64-c8192:max"
DENSE_BRANCH = "ESMC-300:mean,var"
PROFILE = f"{PROFILE_MODEL}:profile"
AXIS_SHA256 = "a" * 64


def write_profile_store(root: Path, sequences: list[str], dtype: str = "float32") -> torch.Tensor:
    """A profile store holding one row per sequence, `(b, p)` probabilities that differ by protein and axis column."""
    generator = torch.Generator().manual_seed(11)
    rows = torch.rand(len(sequences), AXIS_SIZE, generator=generator)  # (b, p)
    stored = {"float32": torch.float32, "float16": torch.float16}[dtype]
    feature = plain_feature(
        "atlas_profile", {"schema": "atlas_proteome_profile_v1", "axis": {"sha256": AXIS_SHA256, "proteins": AXIS_SIZE}, "dtype": dtype},
        layout=DENSE, width=AXIS_SIZE, dtype=stored, slug=("profile", "esmc_small", f"p{AXIS_SIZE}"),
    )
    store = FeatureStore.open(root, feature, deep_verify=False)
    write_rows(store, {sequence: rows[index].to(stored) for index, sequence in enumerate(sequences)})
    manifest = {
        "schema": "atlas_proteome_profile_v1", "visibility": "private", "dtype": dtype,
        "model": {"base": "esmc_small", "arm": "fusion", "package_sha256": "b" * 64, "weights_sha256": "c" * 64, "run_sha256": "d" * 64,
                  "max_length": 2046, "precision": "bf16", "scaling": {"kind": "platt", "slope": 0.127, "intercept": -0.029, "report_sha256": "e" * 64}},
        "axis": {"proteome_id": "UP000005640", "release": "2026_03", "proteins": AXIS_SIZE, "order": "length_then_sequence", "sha256": AXIS_SHA256},
        "features": {"profile": {"key": feature.key, "width": AXIS_SIZE, "dtype": dtype}},
    }
    (root / "profile.json").write_text(json.dumps(manifest), encoding="utf-8")
    return rows.to(stored).to(torch.float32)  # what a read gives back, (b, p)


@pytest.fixture(scope="module")
def stores(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("profile_probe")
    canonical = builder.write_canonical_store(root / "esmc_small", SEQUENCES, base_id="esmc_small")
    profile = root / "profile"
    profile.mkdir()
    write_profile_store(profile, SEQUENCES)
    half = root / "profile_half"
    half.mkdir()
    write_profile_store(half, SEQUENCES, dtype="float16")
    return {"canonical": canonical, "profile": profile, "half": half}


def test_the_profile_spec_reads_the_profile_stream_and_refuses_other_spellings() -> None:
    spec = parse_branch_spec(PROFILE)

    assert spec.profile and not spec.sparse
    assert branch_stream(spec) == (PROFILE_STREAM, WHOLE)
    for token in (f"{PROFILE_MODEL}:mean", f"{PROFILE_MODEL}:profile@3", f"{PROFILE_MODEL}:profile,mean"):
        with pytest.raises(ValueError):
            parse_branch_spec(token)


def test_a_store_returns_each_proteins_row_by_normalized_sequence(tmp_path) -> None:
    written = write_profile_store(tmp_path, SEQUENCES)  # (b, p)
    feature = AtlasProfileFeature(tmp_path, SEQUENCES)

    assert feature.width == AXIS_SIZE
    # A lower-case spelling with a space is the same protein, so it reads the row of the normalized text.
    asked = feature.read([SEQUENCES[2].lower().replace("a", "a "), SEQUENCES[0]])  # (2, p)
    assert asked.dtype == torch.float32 and asked.shape == (2, AXIS_SIZE)
    assert torch.equal(asked, written[[2, 0]])


def test_a_half_precision_store_widens_exactly(tmp_path) -> None:
    written = write_profile_store(tmp_path, SEQUENCES, dtype="float16")  # (b, p), already widened from float16
    rows = AtlasProfileFeature(tmp_path, SEQUENCES).read(SEQUENCES)  # (b, p)

    assert rows.dtype == torch.float32
    assert torch.equal(rows, written)
    assert torch.equal(rows, rows.to(torch.float16).to(torch.float32)), "every value is one a float16 holds"


def test_a_protein_with_no_profile_stops_the_run_before_it_trains(stores) -> None:
    with pytest.raises(ValueError, match="1 of the 7 proteins have no profile"):
        AtlasProfileFeature(stores["profile"], [*SEQUENCES, "MNOPQRSTVWY"])


@pytest.mark.parametrize("damage", [
    {"schema": "other"},
    {"features": {"profile": {"key": "../x", "width": AXIS_SIZE, "dtype": "float32"}}},
    {"features": {"profile": {"key": "k", "width": 0, "dtype": "float32"}}},
    {"features": {"profile": {"key": "k", "width": AXIS_SIZE, "dtype": "bfloat16"}}},
    {"features": {}},
])
def test_a_manifest_that_is_not_a_profile_stores_is_refused(stores, tmp_path, damage) -> None:
    manifest = json.loads((stores["profile"] / "profile.json").read_text(encoding="utf-8"))
    (tmp_path / "profile.json").write_text(json.dumps({**manifest, **damage}), encoding="utf-8")

    with pytest.raises(ValueError):
        read_profile_manifest(tmp_path)
    with pytest.raises(FileNotFoundError):
        read_profile_manifest(tmp_path / "nowhere")


def test_the_provenance_pins_the_model_the_axis_and_the_manifest(stores) -> None:
    provenance = profile_provenance(stores["profile"])

    assert provenance["axis"]["sha256"] == AXIS_SHA256 and provenance["width"] == AXIS_SIZE and provenance["dtype"] == "float32"
    assert provenance["model"]["package_sha256"] == "b" * 64 and provenance["model"]["scaling"]["slope"] == 0.127
    assert len(provenance["manifest_sha256"]) == 64


def test_the_third_branch_joins_after_the_sae_and_the_final_mean_and_variance(stores) -> None:
    record = probe_features_record("esmc_small", root=stores["canonical"], profile_root=stores["profile"])
    joined = open_probe_features(record, [SAE, DENSE_BRANCH, PROFILE], SEQUENCES, batch_size=4)  # branches (c, 2d, p)

    assert joined.widths == (builder.CODEBOOK, 2 * builder.HIDDEN_SIZE, AXIS_SIZE)
    block = joined.block(np.arange(len(SEQUENCES)))  # (b, c + 2d + p)
    base = open_probe_features(probe_features_record("esmc_small", root=stores["canonical"]), [SAE, DENSE_BRANCH], SEQUENCES, batch_size=4)
    profiles = AtlasProfileFeature(stores["profile"], SEQUENCES).read(SEQUENCES).numpy()  # (b, p)

    assert block.shape == (len(SEQUENCES), builder.CODEBOOK + 2 * builder.HIDDEN_SIZE + AXIS_SIZE)
    assert np.array_equal(block[:, : base.width], base.block(np.arange(len(SEQUENCES))))
    assert np.array_equal(block[:, base.width:], profiles)
    assert pickle.loads(pickle.dumps(joined)).block(np.array([4, 1])).tolist() == block[[4, 1]].tolist()


def test_a_profile_can_come_second_among_the_canonical_branches(stores) -> None:
    record = probe_features_record("esmc_small", root=stores["canonical"], profile_root=stores["profile"])
    joined = open_probe_features(record, [DENSE_BRANCH, PROFILE], SEQUENCES, batch_size=4)  # (2d, p)

    assert joined.widths == (2 * builder.HIDDEN_SIZE, AXIS_SIZE)


def test_the_cell_must_name_the_profile_root_exactly_when_a_branch_reads_the_profile(stores) -> None:
    with_root = probe_features_record("esmc_small", root=stores["canonical"], profile_root=stores["profile"])
    without = probe_features_record("esmc_small", root=stores["canonical"])

    with pytest.raises(ValueError, match="exactly when a branch reads the Atlas profile"):
        open_probe_features(without, [SAE, DENSE_BRANCH, PROFILE], SEQUENCES, batch_size=4)
    with pytest.raises(ValueError, match="exactly when a branch reads the Atlas profile"):
        open_probe_features(with_root, [SAE, DENSE_BRANCH], SEQUENCES, batch_size=4)
    with pytest.raises(ValueError, match="not read alone"):
        open_probe_features(with_root, [PROFILE], SEQUENCES, batch_size=4)


def test_only_a_stored_base_with_an_absolute_root_reads_a_profile(stores) -> None:
    with pytest.raises(ValueError, match="only a stored base"):
        probe_features_record("random_vectors", control="random_vector", profile_root=stores["profile"])
    with pytest.raises(ValueError, match="absolute root"):
        probe_features_record("esmc_small", root=stores["canonical"], profile_root="profile")


def test_a_package_records_the_profile_beside_the_canonical_streams(stores) -> None:
    record = probe_features_record("esmc_small", root=stores["canonical"], profile_root=stores["profile"])
    provenance = probe_provenance(record, [SAE, DENSE_BRANCH, PROFILE])

    assert set(provenance["streams"]) == {"sae_max", "final_mean_var"}
    assert provenance["atlas_profile"]["axis"]["sha256"] == AXIS_SHA256
    assert "atlas_profile" not in probe_provenance(probe_features_record("esmc_small", root=stores["canonical"]), [SAE, DENSE_BRANCH])
