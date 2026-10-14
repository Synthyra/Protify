"""The whole-path capacity pilot runs on a tiny procedural release, and its receipt says what a capacity gate needs.

A tiny release cannot show capacity; it shows that the pilot itself is sound before a large run spends money on it:
the two storages of one truth train to the same predictions, a rerun trains nothing, the package answers, every stage
of the path is recorded, and a failure still leaves a receipt that says where it stopped.
"""

import json
import sys
import tempfile
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from collections.abc import Iterator
from pathlib import Path
from datasets import config as dataset_config

from foundry.digests import sha256_file
from protify.testing_suite import run_complete_path_pilot as pilot
from protify.testing_suite.procedural_release_fixture import Dimensions, write_release
from protify.uncertainty import workflow


SETTINGS = {
    "protify_args": {
        "seed": 42, "num_runs": 1, "num_epochs": 2, "max_length": 64, "probe_batch_size": 16, "no_compile": True, "sql": True, "patience": 3,
        "optimizer": "adamw_torch", "lr_scheduler": "cosine", "weight_decay": 0.0, "probe_warmup_steps": 2, "probe_precision": "float32",
        "probe_eval_strategy": "auto", "eval_accumulation_steps": "auto", "num_workers": 0,
    },
    "uncertainty": {
        # k over the codebook width is below the density at which Protify stores SAE codes sparsely, as in production.
        "branch_specs": ["ESMC-300-SAE-l23-k64-c16384:max", "ESMC-300:mean,var"], "head": "projected", "projection_size": 32, "learning_rate": 0.001,
        "dropout": 0.1, "split_seed": 0, "mc_dropout_samples": 0, "conformal_levels": [0.8, 0.9], "save_heads": True,
    },
}
# Every stage the fit cell owns, in the order the pilot names them; `full` mode adds the package's.
FIT_STAGES = {
    "cell.load_release", "cell.record_observations", "cell.resolve_roles", "cell.load_features", "cell.build_dataset", "seed", "seed.train",
    "seed.predict", "seed.save_predictions", "seed.primary_metric", "seed.nll", "cell.report", "report.load_seeds", "report.build_methods",
    "report.score_method",
}
PACKAGE_STAGES = {"package.build", "package.load", "package.predict"}


@pytest.fixture(autouse=True)
def short_paths(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Hugging Face's cache lock file names overflow a Windows path under a long temporary directory."""
    with tempfile.TemporaryDirectory(prefix="pd") as directory:
        monkeypatch.setattr(dataset_config, "HF_DATASETS_CACHE", Path(directory))
        monkeypatch.setattr(sys.modules[workflow.parse_arguments.__module__], "load_secrets_env", lambda: [])
        yield


def dimensions(observed_fraction: float, scored_observed_fraction: float | None = None) -> Dimensions:
    return Dimensions(
        train_rows=96, scored_rows=24, labels=41, positives=3, observed_fraction=observed_fraction, dense_width=16, sae_width=64,
        sae_nonzeros=8, seed=1, scored_observed_fraction=scored_observed_fraction,
    )


@pytest.mark.parametrize(("observed_fraction", "scored_observed_fraction"), [(1.0, None), (0.3, None), (0.3, 1.0), (0.9, 0.3)])
def test_a_full_pilot_agrees_across_storages_resumes_and_packages(tmp_path, monkeypatch, observed_fraction, scored_observed_fraction):
    monkeypatch.chdir(tmp_path)
    dims = dimensions(observed_fraction, scored_observed_fraction)
    receipt = pilot.run_pilot(tmp_path / "run", dims, SETTINGS, mode="full", epochs=2, tier="test", parity=True, package_rows=10)
    assert receipt["failure"] is None, receipt["failure"]
    assert receipt["schema"] == pilot.RECEIPT_SCHEMA and receipt["complete"]
    assert receipt["parity"]["agree"], receipt["parity"]
    assert receipt["resume"]["seeds_trained_again"] == 0 and receipt["resume"]["report_identical"]
    assert receipt["release"]["splits"] == {"train": 96, "validation": 24, "test": 24, "calibration": 24}
    assert receipt["consumed"] == {
        "split": {"train": 96, "v_fit": 24, "v_cal": 24, "test": 24}, "release": {"train": 96, "v_fit": 24, "v_cal": 24, "test": 24}, "matches_release": True,
    }
    assert (receipt["release"]["label_entries"], receipt["release"]["scored_observed_targets"]) == (dims.label_entries(), dims.scored_observed_targets())
    assert {role: (facts["dtype"], facts["shape"]) for role, facts in receipt["predictions"].items()} == {
        "v_fit": ("float32", [24, 41]), "v_cal": ("float32", [24, 41]), "test": ("float32", [24, 41]),
    }
    assert (receipt["features"]["sae"]["layout"], receipt["features"]["dense"]["layout"]) == ("csr", "dense")
    assert receipt["features"]["sae"]["rows"] == receipt["features"]["dense"]["rows"] == 168
    assert receipt["package"]["rows"] == 10 and receipt["package"]["shape"] == [10, 41] and receipt["package"]["finite"]
    names = {record["name"] for record in receipt["stages"]}
    assert names >= FIT_STAGES | PACKAGE_STAGES | {"fixture.release", "fixture.features", "cell.run", "cell.resume"}
    assert all(record["seconds"] >= 0 and "rss_peak_bytes" in record for record in receipt["stages"])
    assert {record["name"] for record in receipt["stages"] if record["depth"] == 0} >= {"fixture.release", "fixture.features", "cell.run", "cell.resume"}
    assert all(len(digest) == 64 for digest in receipt["code"].values())
    if workflow_path := pilot.workspace_path(Path(workflow.__file__), None):  # a GitHub clone is not the workspace, and records no code
        assert receipt["code"][workflow_path] == sha256_file(workflow.__file__)
        assert not [path for path in receipt["code"] if "site-packages" in path or path.startswith("/") or ":" in path]
    assert receipt["cell"]["uncertainty"]["evaluation_mode"] == "full" and receipt["cell"]["uncertainty"]["require_cached_embeddings"]
    assert receipt["throughput"]["training_steps_per_second"]
    json.dumps(receipt)


@pytest.mark.parametrize(("observed_fraction", "scored_observed_fraction"), [(1.0, None), (0.3, 1.0), (0.9, 0.3), (0.5, 0.5)])
def test_the_fixtures_counts_are_those_of_the_release_it_writes(tmp_path, observed_fraction, scored_observed_fraction):
    dims = dimensions(observed_fraction, scored_observed_fraction)
    write_release(tmp_path, dims)
    entries, observed = 0, {}
    for name in dims.splits():
        table = pq.read_table(tmp_path / f"{name}.parquet")
        labels, mask = table.column("labels").combine_chunks(), table.column("label_mask").combine_chunks()
        stored = mask.field("exceptions")
        entries += pc.sum(pc.list_value_length(labels.field("indices"))).as_py() + pc.sum(pc.list_value_length(stored)).as_py()
        held = pc.list_value_length(stored).to_numpy(zero_copy_only=False)
        polarity = mask.field("default_observed").to_numpy(zero_copy_only=False)
        observed[name] = int(np.where(polarity, dims.labels - held, held).sum())
    assert entries == dims.label_entries()
    assert observed["validation"] == observed["test"] == observed["calibration"] == dims.scored_observed_targets()
    assert observed["train"] == dims.train_rows * dims.observed_per_row(False)


@pytest.mark.parametrize(("observed_fraction", "scored_observed_fraction"), [(0.0, None), (0.3, 0.0), (0.3, 1.5), (1.2, 1.0)])
def test_every_observed_share_must_be_a_fraction(observed_fraction, scored_observed_fraction):
    with pytest.raises(ValueError, match="observed fraction"):
        dimensions(observed_fraction, scored_observed_fraction)


def test_a_validation_only_pilot_never_reads_calibration_or_test_and_records_the_seed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    receipt = pilot.run_pilot(tmp_path / "run", dimensions(0.3), SETTINGS, mode="validation_only", epochs=2, tier="test")
    assert receipt["failure"] is None, receipt["failure"]
    assert receipt["complete"] and "package" not in receipt
    assert receipt["report"]["evaluation_mode"] == "validation_only"
    (seed,) = receipt["seeds"]
    outcome = seed["training_outcome"]
    assert outcome["global_step"] > 0 and outcome["planned_optimizer_steps"] >= outcome["global_step"] and outcome["best_checkpoint"]
    assert not any(name in {record["name"] for record in receipt["stages"]} for name in PACKAGE_STAGES)


def test_a_failed_pilot_still_writes_where_it_stopped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def broken(*arguments, **options):
        raise RuntimeError("the feature join ran out of memory")

    monkeypatch.setattr(workflow, "load_cell_branches", broken)
    receipt = pilot.run_pilot(tmp_path / "run", dimensions(1.0), SETTINGS, mode="validation_only", epochs=2, tier="test")
    assert not receipt["complete"]
    assert receipt["failure"]["exception"] == "RuntimeError"
    assert receipt["failure"]["stages"] == ["cell.run", "cell.load_features"]
    assert "the feature join ran out of memory" in receipt["failure"]["traceback"]
    assert workflow.load_cell_branches is broken, "the pilot restores what it patched, including a function a test replaced"
    written = tmp_path / "receipt.json"
    pilot.write_json(written, receipt)
    assert json.loads(written.read_text(encoding="utf-8"))["failure"]["exception"] == "RuntimeError"
    assert not list(tmp_path.glob("*.partial"))


def test_the_recorder_leaves_its_progress_where_a_killed_process_can_be_read(tmp_path):
    progress = tmp_path / "progress.json"
    recorder = pilot.StageRecorder(tmp_path, progress_path=progress)
    try:
        with recorder.stage("outer"), recorder.stage("inner"):
            recorder.write_progress()
            held = json.loads(progress.read_text(encoding="utf-8"))
    finally:
        recorder.close()
    assert held["open_stages"] == ["outer", "inner"]
    finished = json.loads(progress.read_text(encoding="utf-8"))
    assert finished["open_stages"] == [] and [record["name"] for record in finished["stages"]] == ["outer", "inner"]
    assert finished["stages"][0]["depth"] == 0 and finished["stages"][1]["depth"] == 1 and "tree_bytes" in finished["stages"][0]


def test_the_pilot_refuses_a_parity_run_that_has_no_test_predictions(tmp_path):
    with pytest.raises(ValueError, match="parity"):
        pilot.run_pilot(tmp_path / "run", dimensions(1.0), SETTINGS, mode="validation_only", epochs=2, tier="test", parity=True)


def test_source_paths_are_workspace_paths_in_place_and_in_a_shipment(tmp_path):
    def write(name: str) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value = 1\n", encoding="utf-8")
        return path

    write("workspace/AGENTS.md"), write("workspace/projects/tooling/property_lowcode/src/protify/a.py")
    assert pilot.workspace_path(tmp_path / "workspace/projects/tooling/property_lowcode/src/protify/a.py", None) == "projects/tooling/property_lowcode/src/protify/a.py"
    assert pilot.workspace_path(write("site-packages/sklearn/tree/_classes.py"), None) is None

    shipment = tmp_path / "ws-runs" / "property_lowcode" / "run"
    assert pilot.workspace_path(write("ws-runs/property_lowcode/run/tree/projects/tooling/property_lowcode/src/protify/a.py"), shipment) == "projects/tooling/property_lowcode/src/protify/a.py"
    assert pilot.workspace_path(write("ws-runs/property_lowcode/run/import-roots/foundry/digests.py"), shipment) == "foundry/digests.py"
    assert pilot.workspace_path(write("ws-runs/property_lowcode/run/venv/lib/sklearn/tree/_classes.py"), shipment) is None


def test_stage_names_cover_every_function_the_workflow_still_defines():
    """A renamed workflow function must be renamed here, or its stage silently stops being measured."""
    missing = [name for name in pilot.WORKFLOW_STAGES if not hasattr(workflow, name)]
    assert not missing, f"the workflow no longer has {missing}"
