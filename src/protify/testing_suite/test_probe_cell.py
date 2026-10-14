"""A cell trains on a canonical store or a generated control, both arms, and packages what it read.

The data is a declared four-way release; the rows come from a tiny canonical store the engine's taps filled on CPU, or
from the control the engine generates. Nothing is embedded: `require_cached_embeddings` is true and no Protify store exists.
What must hold: both arms train and package, the package records the source and its streams, a random-init store
records its seed and no checkpoint, a generated control records its arithmetic, and a cell that asks for a source its
arm cannot read is refused before it trains.

Shapes: `n` rows of a split, `d` hidden width (32), `c` SAE codebook (8192).
"""

import json
import numpy as np
import pytest

from pathlib import Path

from foundry.embedding import ONE_HOT, RANDOM_VECTOR
from protify.base_models import control_bases
from protify.data.branch_features import PROFILE_MODEL
from protify.data.probe_features import probe_features_record
from protify.testing_suite import canonical_store_builder as builder
from protify.testing_suite import test_uncertainty_workflow as fixture
from protify.testing_suite.test_atlas_profile_feature import AXIS_SHA256, AXIS_SIZE, write_profile_store
from protify.testing_suite.test_declared_dataset import isolated_arrow_cache, release_fixture  # noqa: F401
from protify.uncertainty.cell import cell_from_mapping
from protify.uncertainty.package import build_package
from protify.uncertainty.packaged import load_package


SAE = f"{fixture.SAE}:max"
DENSE = "ESMC-300:mean,var"
MEAN = "ESMC-300:mean"


@pytest.fixture
def toy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.chdir(tmp_path)
    path, digest, manifest, rows = release_fixture(tmp_path, "binary")
    sequences = sorted({row["seqs"] for records in rows.values() for row in records})
    stores = {
        "esmc_small": builder.write_canonical_store(tmp_path / "stores" / "esmc_small", sequences),
        "esmc_small_random": builder.write_canonical_store(tmp_path / "stores" / "random", sequences, base_id="esmc_small_random", seed=7),
    }
    return {"path": path, "digest": digest, "stores": stores, "rows": rows}


def run_cell(toy: dict, tmp_path: Path, name: str, branch_specs: list[str], probe: dict, **settings):
    mapping = fixture._cell_mapping(tmp_path, "unused", name, num_runs=1)
    mapping["protify_args"].pop("data_dirs")
    mapping["uncertainty"].update(
        head="projected", branch_specs=branch_specs, dataset_manifest=str(toy["path"]), dataset_sha256=toy["digest"],
        save_heads=True, require_cached_embeddings=True, probe_features=probe, **settings,
    )
    for key in ("identity_path", "calibration_fraction"):
        mapping["uncertainty"].pop(key)
    mapping["output_dir"] = str(tmp_path / "runs" / name)
    cell = cell_from_mapping(mapping)
    return cell, fixture.workflow.run_uncertainty_cell(cell)


def test_the_both_arm_trains_and_packages_from_a_canonical_store(toy, tmp_path):
    probe = probe_features_record("esmc_small", root=toy["stores"]["esmc_small"])
    cell, report = run_cell(toy, tmp_path, "both", [SAE, DENSE], probe)

    assert [branch["width"] for branch in report["branches"]] == [builder.CODEBOOK, 2 * builder.HIDDEN_SIZE]
    assert report["input_size"] == builder.CODEBOOK + 2 * builder.HIDDEN_SIZE
    package = Path(build_package(cell.output_dir, "temperature", str(tmp_path / "package_both")))
    features = json.loads((package / "features.json").read_text(encoding="utf-8"))

    assert features["embedding_dtype"] == "float32" and features["max_length"] == 2048
    assert features["source"]["kind"] == "stored" and features["source"]["special_tokens"] == "kept"
    assert set(features["source"]["streams"]) == {"sae_max", "final_mean_var"}
    assert features["backbones"][0]["repository"] == "Synthyra/ESMplusplus_small"
    assert "engine's pooled streams" in (package / "README.md").read_text(encoding="utf-8")
    assert load_package(package).predict(np.zeros((2, features["width"]), dtype=np.float32)).probs.shape == (2, 2)


def test_the_mean_final_arm_reads_one_branch_of_the_final_mean(toy, tmp_path):
    probe = probe_features_record("esmc_small", root=toy["stores"]["esmc_small"])
    cell, report = run_cell(toy, tmp_path, "mean_final", [MEAN], probe)

    assert [branch["width"] for branch in report["branches"]] == [builder.HIDDEN_SIZE]
    assert report["branches"][0]["pooling_types"] == ["mean"]
    package = Path(build_package(cell.output_dir, "temperature", str(tmp_path / "package_mean")))
    assert json.loads((package / "features.json").read_text(encoding="utf-8"))["source"]["streams"].keys() == {"final_mean_var"}


def test_a_random_init_store_records_its_seed_and_no_checkpoint(toy, tmp_path):
    probe = probe_features_record("esmc_small_random", root=toy["stores"]["esmc_small_random"])
    cell, _ = run_cell(toy, tmp_path, "random_init", [MEAN], probe)
    features = json.loads((Path(build_package(cell.output_dir, "temperature", str(tmp_path / "package_random"))) / "features.json").read_text(encoding="utf-8"))

    assert features["source"]["random_init"]["kind"] == "random_init"
    assert [(entry["component"], entry["repository"]) for entry in features["backbones"]] == [("random_init_backbone", None)]
    assert "random init" in features["backbones"][0]["generator"]


@pytest.mark.parametrize(("control", "model", "width"), [(RANDOM_VECTOR, control_bases.RANDOM, 960), (ONE_HOT, control_bases.ONE_HOT, 28)])
def test_a_generated_control_trains_and_records_its_arithmetic(toy, tmp_path, control, model, width):
    cell, report = run_cell(toy, tmp_path, f"control_{control}", [f"{model}:mean"], probe_features_record(model, control=control))
    package = Path(build_package(cell.output_dir, "temperature", str(tmp_path / f"package_{control}")))
    features = json.loads((package / "features.json").read_text(encoding="utf-8"))

    assert [branch["width"] for branch in report["branches"]] == [width]
    assert features["source"]["kind"] == "generated" and features["source"]["generator"]["kind"] == control
    assert features["backbones"] == [{"model_name": model, "component": "generated_rows", "repository": None, "revision": None,
                                      "generator": control_bases.CONTROL_SCHEMA}]


def test_a_cell_that_asks_a_store_for_a_stream_it_lacks_is_refused_before_training(toy, tmp_path):
    probe = probe_features_record("esmc_small_random", root=toy["stores"]["esmc_small_random"])

    with pytest.raises(KeyError, match="sae_max"):
        run_cell(toy, tmp_path, "no_sae", [SAE, DENSE], probe)
    assert not (tmp_path / "runs" / "no_sae" / "seed_42.json").exists()


def test_probe_features_replaces_the_pinned_manifest_and_needs_a_declared_dataset(tmp_path):
    mapping = fixture._cell_mapping(tmp_path, "unused", "bad", num_runs=1)
    mapping["uncertainty"].update(head="projected", branch_specs=[MEAN], probe_features=probe_features_record("one_hot", control=ONE_HOT),
                                  require_cached_embeddings=True)
    with pytest.raises(ValueError, match="requires a declared dataset"):
        cell_from_mapping(mapping)


PROFILE = f"{PROFILE_MODEL}:profile"


def test_the_atlas_profile_trains_as_a_third_branch_with_its_own_projected_width(toy, tmp_path):
    sequences = sorted({row["seqs"] for records in toy["rows"].values() for row in records})
    profile_root = tmp_path / "profiles"
    profile_root.mkdir()
    write_profile_store(profile_root, sequences)
    probe = probe_features_record("esmc_small", root=toy["stores"]["esmc_small"], profile_root=profile_root)
    cell, report = run_cell(toy, tmp_path, "fusion_profile", [SAE, DENSE, PROFILE], probe, projection_size=16, branch_projection_sizes=[4, 4, 8])

    assert [branch["width"] for branch in report["branches"]] == [builder.CODEBOOK, 2 * builder.HIDDEN_SIZE, AXIS_SIZE]
    assert report["input_size"] == builder.CODEBOOK + 2 * builder.HIDDEN_SIZE + AXIS_SIZE
    package = Path(build_package(cell.output_dir, "temperature", str(tmp_path / "package_profile")))
    features = json.loads((package / "features.json").read_text(encoding="utf-8"))

    assert features["branch_sizes"] == [4, 4, 8] and features["projection_size"] == 16
    assert set(features["source"]["streams"]) == {"sae_max", "final_mean_var"}
    assert features["source"]["atlas_profile"]["axis"]["sha256"] == AXIS_SHA256
    assert ("atlas_proteome_profile", None) in [(entry["component"], entry["repository"]) for entry in features["backbones"]]
    assert "4, 4, 8 columns, joined to 16" in (package / "README.md").read_text(encoding="utf-8")
    assert load_package(package).predict(np.zeros((2, features["width"]), dtype=np.float32)).probs.shape == (2, 2)


def test_a_profile_store_that_lacks_a_protein_stops_the_cell_before_it_trains(toy, tmp_path):
    sequences = sorted({row["seqs"] for records in toy["rows"].values() for row in records})
    profile_root = tmp_path / "profiles"
    profile_root.mkdir()
    write_profile_store(profile_root, sequences[:-1])
    probe = probe_features_record("esmc_small", root=toy["stores"]["esmc_small"], profile_root=profile_root)

    with pytest.raises(ValueError, match="have no profile"):
        run_cell(toy, tmp_path, "short_profile", [SAE, DENSE, PROFILE], probe, projection_size=16, branch_projection_sizes=[4, 4, 8])
    assert not (tmp_path / "runs" / "short_profile" / "seed_42.json").exists()


@pytest.mark.parametrize("sizes", [[4, 12], [4, 4, 7], [4, 4, 0, 8], [4.0, 4, 8]])
def test_branch_projection_sizes_must_fit_the_branches_and_the_projection_size(tmp_path, sizes):
    mapping = fixture._cell_mapping(tmp_path, "unused", "bad_sizes", num_runs=1)
    mapping["uncertainty"].update(branch_specs=[SAE, DENSE, PROFILE], projection_size=16, branch_projection_sizes=sizes)

    with pytest.raises(ValueError, match="branch_projection_sizes"):
        cell_from_mapping(mapping)
