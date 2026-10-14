"""The uncertainty cell end to end, on a toy dataset with feature stores keyed as Protify keys them.

No model loads: every branch cache is complete, so the Embedder finds nothing to embed. The
caches are narrower than the real models' pooled vectors, which the workflow never checks,
so the run stays fast; the SAE branch is still sparse in memory.

What the run must prove: a report with the fields the grid reads, one prediction file and one
record per seed, the validation roles pinned by digest, temperature fitted on V_fit rows and
conformal quantiles on V_cal rows only, from either source of those roles, a rerun that
trains nothing, and a final cell that resumes from a tuning cell's copied first seed.
"""

import json
import os
import shutil
import sys
import numpy as np
import pandas as pd
import pytest
import torch

import foundry.eval.conformal as foundry_conformal

from collections.abc import Iterable
from scipy.special import log_softmax


try:
    from src.protify.testing_suite.embedding_store_builder import cache_descriptor, write_store
    from src.protify.uncertainty import conformal_sets, methods, workflow
    from src.protify.uncertainty.calibration_split import row_digests
    from src.protify.uncertainty.cell import cell_from_mapping
    from src.protify.uncertainty.report import read_json, write_json
except ImportError:
    from protify.testing_suite.embedding_store_builder import cache_descriptor, write_store
    from protify.uncertainty import conformal_sets, methods, workflow
    from protify.uncertainty.calibration_split import row_digests
    from protify.uncertainty.cell import cell_from_mapping
    from protify.uncertainty.report import read_json, write_json


SAE = "ESMC-300-SAE-l23-k64-c8192"
SAE_WIDTH = 24
DENSE_WIDTH = 16
AMINO_ACIDS = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
SPLIT_SIZES = {"train": 48, "valid": 40, "test": 30}


def _toy_dataset(root) -> tuple[dict[str, pd.DataFrame], str]:
    rng = np.random.default_rng(0)
    frames, used = {}, set()
    for split, count in SPLIT_SIZES.items():
        sequences = []
        while len(sequences) < count:
            sequence = "M" + "".join(rng.choice(AMINO_ACIDS, size=int(rng.integers(10, 30))))
            if sequence not in used:
                used.add(sequence)
                sequences.append(sequence)
        frames[split] = pd.DataFrame({"seqs": sequences, "labels": rng.integers(0, 2, size=count)})

    data_dir = root / "data" / "toyset"
    data_dir.mkdir(parents=True)
    for split, frame in frames.items():
        frame.to_csv(data_dir / f"{split}.csv", index=False)
    return frames, str(data_dir)


def _write_caches(frames: dict[str, pd.DataFrame], directory, *, omit_dense: Iterable[str] = ()) -> None:
    """Both branches' feature stores under `directory`, keyed as a plain run keys them.

    `omit_dense` names sequences left out of the dense branch's store, for a cache a run has not finished.
    """
    directory.mkdir()
    rng = np.random.default_rng(1)
    rows = pd.concat(frames.values())
    signal = np.where(rows["labels"].to_numpy()[:, None] == 1, 1.0, -1.0)  # (n, 1)
    sae = np.maximum(rng.normal(size=(len(rows), SAE_WIDTH)) - 1.0 + 0.5 * signal, 0.0)  # (n, 24)
    dense = rng.normal(size=(len(rows), DENSE_WIDTH)) + signal  # (n, 16)
    omitted = set(omit_dense)
    caches = (
        (cache_descriptor(SAE, pooling=("max",), max_length=64), sae, set()),
        (cache_descriptor("ESM2-8", pooling=("mean", "var"), max_length=64), dense, omitted),
    )
    for descriptor, values, left_out in caches:
        tensors = torch.from_numpy(values.astype(np.float32))  # (n, w)
        write_store(
            directory,
            descriptor,
            {sequence: row for sequence, row in zip(rows["seqs"], tensors, strict=True) if sequence not in left_out},
        )


def _cell_mapping(root, data_dir: str, name: str, num_runs: int = 2) -> dict:
    return {
        "cell_id": name,
        "experiment_id": "toy_experiment",
        "protify_args": {
            "data_dirs": [data_dir],
            "sql": True,
            "seed": 42,
            "num_runs": num_runs,
            "num_epochs": 2,
            "max_length": 64,
            "probe_batch_size": 16,
            "no_compile": True,
            "embedding_save_dir": str(root / "embeddings"),
            "log_dir": str(root / "logs"),
            "results_dir": str(root / "results"),
            "plots_dir": str(root / "plots"),
            "model_save_dir": str(root / "weights"),
        },
        "uncertainty": {
            "branch_specs": [f"{SAE}:max", "ESM2-8:mean,var"],
            "head": "sngp",
            "projection_size": 32,
            "learning_rate": 0.001,
            "calibration_fraction": 0.4,
            "conformal_levels": [0.8, 0.9],
            "identity_path": str(root / "identity" / "toyset.json"),
        },
        "output_dir": str(root / "runs" / name),
    }


def _cell(root, data_dir: str, name: str, num_runs: int = 2, **uncertainty):
    mapping = _cell_mapping(root, data_dir, name, num_runs)
    mapping["uncertainty"].update(uncertainty)
    return cell_from_mapping(mapping)


class Spies:
    """Record the rows every temperature fit and conformal quantile reads."""

    def __init__(self, monkeypatch) -> None:
        self.temperature_labels = []
        self.quantile_sizes = []
        fit_temperature = methods.fit_temperature
        quantiles = {module: module.conformal_quantile for module in (conformal_sets, foundry_conformal)}

        def spy_temperature(logits, labels, **options):
            self.temperature_labels.append(np.asarray(labels).copy())
            return fit_temperature(logits, labels, **options)

        def spy_quantile(module):
            def recorded(scores, alpha, weights=None):
                self.quantile_sizes.append(np.asarray(scores).shape[0])
                return quantiles[module](scores, alpha, weights)
            return recorded

        monkeypatch.setattr(methods, "fit_temperature", spy_temperature)
        for module in quantiles:
            monkeypatch.setattr(module, "conformal_quantile", spy_quantile(module))


def _spy_probes(monkeypatch) -> list:
    """Every probe the workflow builds, in order."""
    built = []
    get_probe = workflow.get_probe

    def recorded(probe_args):
        probe = get_probe(probe_args)
        built.append(probe)
        return probe

    monkeypatch.setattr(workflow, "get_probe", recorded)
    return built


def _no_training(*arguments, **options):
    raise AssertionError("a complete seed was trained again")


def _no_embedding(*arguments, **options):
    raise AssertionError("require_cached_embeddings reached the Embedder")


@pytest.fixture
def toy(tmp_path, monkeypatch):
    # Hugging Face writes checkpoints under the working directory; nothing may land in the tree.
    monkeypatch.chdir(tmp_path)
    main_module = sys.modules[workflow.parse_arguments.__module__]
    monkeypatch.setattr(main_module, "load_secrets_env", lambda: [])
    frames, data_dir = _toy_dataset(tmp_path)
    _write_caches(frames, tmp_path / "embeddings")
    return tmp_path, frames, data_dir


def _fake_identity(queries, references, directory, config=None, *, runner=None):
    """Identities spread over every bin, keyed as the search keys them."""
    return {query: (int(query[:4], 16) % 11) / 10.0 for query in queries}


def test_a_cell_runs_scores_and_resumes(toy, monkeypatch):
    root, frames, data_dir = toy
    cell = _cell(root, data_dir, "toy_sngp", gp_lengthscale_squared=64)

    monkeypatch.setattr(workflow, "max_identity_to_reference", _fake_identity)
    identity = workflow.score_identity(cell)
    assert {split: len(rows) for split, rows in identity["rows"].items()} == {"test": 30, "valid": 40, "v_fit": 24, "v_cal": 16}
    assert identity["reference_sequences"] == 48 and identity["query_sequences"] == 70

    spies = Spies(monkeypatch)
    probes = _spy_probes(monkeypatch)
    report = workflow.run_uncertainty_cell(cell)

    assert report["cell_id"] == "toy_sngp" and report["task_type"] == "binary" and report["primary_metric"] == "mcc"
    assert report["tags"] == {}
    assert [record["seed"] for record in report["per_seed"]] == [42, 43]
    assert isinstance(report["v_fit_primary"], float)
    assert {
        "sngp", "sngp_entropy", "sngp_nn_distance", "sngp_variance", "sngp_temperature", "sngp_ensemble", "sngp_ensemble_temperature",
    } <= set(report["methods"])
    assert {"mcc", "auroc", "ece", "nll", "brier", "aurc", "lac_coverage@0.9", "aps_size@0.8"} <= set(report["methods"]["sngp"]["test"])
    assert sum(report["identity_bins"]["counts"].values()) == 30
    assert report["branches"][0]["sparse"] and report["branches"][0]["width"] == SAE_WIDTH
    assert report["input_size"] == SAE_WIDTH + DENSE_WIDTH

    # The cell's squared lengthscale reaches every head it trains, and its random features.
    assert probes and {probe.config.gp_lengthscale_squared for probe in probes} == {64.0}
    assert float(probes[0].gaussian_process.feature_weight.std()) == pytest.approx(1.0 / 8.0, rel=0.05)

    output = root / "runs" / "toy_sngp"
    assert {"seed_42.npz", "seed_42.json", "seed_43.npz", "seed_43.json", "split.json", "cell.json", "report.json"} <= set(os.listdir(output))
    split = json.loads((output / "split.json").read_text())
    valid = frames["valid"]
    digests = row_digests({"seqs": list(valid["seqs"]), "labels": [int(label) for label in valid["labels"]]})
    assert split["v_fit_digests"] == [digests[row] for row in split["v_fit_rows"]]
    assert split["v_cal_digests"] == [digests[row] for row in split["v_cal_rows"]]
    assert len(split["v_fit_rows"]) == 24 and len(split["v_cal_rows"]) == 16

    # Each seed's V_fit NLL is the true class's under the softmax of its mean-field logits.
    with np.load(output / "seed_42.npz") as archive:
        logits, labels = archive["v_fit__outputs"].astype(np.float64), archive["v_fit__labels"]  # (n, c), (n,)
    log_probs = log_softmax(logits, axis=1)  # (n, c)
    expected_nll = -np.mean(log_probs[np.arange(logits.shape[0]), labels])
    assert report["per_seed"][0]["v_fit_nll"] == pytest.approx(expected_nll, rel=1e-5)
    assert report["v_fit_nll"] == pytest.approx(np.mean([record["v_fit_nll"] for record in report["per_seed"]]))

    # Two seeds' temperatures and the ensemble's, all on V_fit.
    v_fit_labels = valid["labels"].to_numpy()[split["v_fit_rows"]]
    assert len(spies.temperature_labels) == 3
    assert all(np.array_equal(labels, v_fit_labels) for labels in spies.temperature_labels)
    assert spies.quantile_sizes and set(spies.quantile_sizes) == {16}

    monkeypatch.setattr(workflow, "_train_seed", _no_training)
    assert workflow.run_uncertainty_cell(cell) == report

    # A seed recorded before `v_fit_nll` existed still loads, and reports None for it.
    record_path = str(output / "seed_43.json")
    record = read_json(record_path)
    del record["v_fit_nll"]
    write_json(record_path, record)
    older = workflow.run_uncertainty_cell(cell)
    assert older["per_seed"][1]["v_fit_nll"] is None
    assert older["v_fit_nll"] == pytest.approx(report["per_seed"][0]["v_fit_nll"])


def test_explicit_validation_rows_feed_the_same_fits(toy, monkeypatch):
    root, frames, data_dir = toy
    cell = _cell(root, data_dir, "toy_explicit", num_runs=1)
    spies = Spies(monkeypatch)
    probes = _spy_probes(monkeypatch)
    fit_rows, cal_rows = list(range(28)), list(range(30, 40))

    report = workflow.run_uncertainty_cell(cell, explicit_rows=(fit_rows, cal_rows))

    assert report["split"]["source"] == "explicit" and report["split"]["v_fit"] == 28
    assert "identity_bins" not in report
    valid_labels = frames["valid"]["labels"].to_numpy()
    assert len(spies.temperature_labels) == 1
    assert np.array_equal(spies.temperature_labels[0], valid_labels[fit_rows])
    assert spies.quantile_sizes and set(spies.quantile_sizes) == {10}
    # A cell that leaves the lengthscale unset trains at the probe's own default.
    assert {probe.config.gp_lengthscale_squared for probe in probes} == {1024.0}

    with pytest.raises(ValueError, match="differ"):
        workflow.run_uncertainty_cell(cell, explicit_rows=(fit_rows[:-1], cal_rows))


def test_a_final_cell_resumes_from_a_tuning_cells_first_seed(toy, monkeypatch):
    """The grid's final cell: a tuning cell's seed 42 and split.json copied in, then 43 to 46 trained.

    Both cells read the caches under `require_cached_embeddings`, which must leave them
    untouched and never reach the Embedder.
    """
    root, _, data_dir = toy
    monkeypatch.setattr(workflow, "embed_missing_branches", _no_embedding)
    caches = root / "embeddings"

    def cache_state() -> dict:
        return {
            path.relative_to(caches).as_posix(): (path.stat().st_mtime_ns, path.stat().st_size)
            for path in sorted(caches.rglob("*")) if path.is_file()
        }

    before = cache_state()
    assert before, "the toy caches are stores of files"

    def plain_cell(name: str, num_runs: int, role: str):
        mapping = _cell_mapping(root, data_dir, name, num_runs)
        mapping["uncertainty"].update(head="projected", dropout=0.1, mc_dropout_samples=4, require_cached_embeddings=True)
        mapping["tags"] = {"arm": "both", "group": "toyset.both", "role": role}
        return cell_from_mapping(mapping)

    tune_dir = root / "runs" / "toy_tune_lr1e-3"
    workflow.run_uncertainty_cell(plain_cell("toy_tune_lr1e-3", 1, "tune"))
    final_dir = root / "runs" / "toy_final"
    final_dir.mkdir(parents=True)
    for name in ("seed_42.npz", "seed_42.json", "split.json"):
        shutil.copyfile(tune_dir / name, final_dir / name)

    trained = []
    train_seed = workflow._train_seed

    def recorded(cell, cell_data, seed, *arguments):
        trained.append(seed)
        return train_seed(cell, cell_data, seed, *arguments)

    monkeypatch.setattr(workflow, "_train_seed", recorded)
    report = workflow.run_uncertainty_cell(plain_cell("toy_final", 5, "final"))

    assert trained == [43, 44, 45, 46]
    assert report["seeds"] == [42, 43, 44, 45, 46]
    assert [record["seed"] for record in report["per_seed"]] == [42, 43, 44, 45, 46]
    assert report["per_seed"][0] == read_json(str(tune_dir / "seed_42.json"))
    assert report["tags"] == {"arm": "both", "group": "toyset.both", "role": "final"}
    assert {"plain", "temperature", "mc_dropout", "mc_dropout_temperature", "ensemble", "ensemble_temperature"} <= set(report["methods"])
    assert report["methods"]["mc_dropout_temperature"]["seeds"] == [42, 43, 44, 45, 46]
    assert cache_state() == before


def test_an_incomplete_cache_is_refused_without_embedding(toy, monkeypatch):
    root, frames, data_dir = toy
    # The dense branch's store lacks one test sequence, as after a run that stopped before it finished.
    shutil.rmtree(root / "embeddings")
    _write_caches(frames, root / "embeddings", omit_dense=[frames["test"]["seqs"][0]])
    monkeypatch.setattr(workflow, "embed_missing_branches", _no_embedding)
    monkeypatch.setattr(workflow, "_train_seed", _no_training)

    cell = _cell(root, data_dir, "toy_guarded", num_runs=1, require_cached_embeddings=True)
    with pytest.raises(ValueError, match=r"ESM2-8_mean-var lacks 1 of 118 sequences") as refusal:
        workflow.run_uncertainty_cell(cell)
    assert SAE not in str(refusal.value)


def test_the_cell_refuses_settings_it_cannot_honor(tmp_path):
    for uncertainty, tags, message in (
        ({"head": "projected", "gp_lengthscale_squared": 128}, None, "sets the SNGP kernel"),
        ({"gp_lengthscale_squared": 0}, None, "positive and finite"),
        ({"require_cached_embeddings": "false"}, None, "true or false"),
        ({}, {"role": 1}, "strings to strings"),
        ({}, ["final"], "strings to strings"),
    ):
        mapping = _cell_mapping(tmp_path, "unused", "toy_refused")
        mapping["uncertainty"].update(uncertainty)
        if tags is not None:
            mapping["tags"] = tags
        with pytest.raises(ValueError, match=message):
            cell_from_mapping(mapping)


def test_a_directory_recorded_before_a_training_setting_existed_resumes(tmp_path):
    """An older cell.json lacks `gp_lengthscale_squared`, which then reads as its default."""
    mapping = _cell_mapping(tmp_path, "unused", "toy_recorded")
    cell = cell_from_mapping(mapping)
    os.makedirs(cell.output_dir)
    workflow._record_cell(cell)
    record_path = os.path.join(cell.output_dir, workflow.CELL_RECORD)
    record = read_json(record_path)
    del record["uncertainty"]["gp_lengthscale_squared"]
    write_json(record_path, record)

    workflow._record_cell(cell)
    # Tags and the cache guard label and gate a run; neither changes what a seed trains.
    workflow._record_cell(cell_from_mapping({
        **mapping,
        "tags": {"role": "final"},
        "uncertainty": {**mapping["uncertainty"], "require_cached_embeddings": True},
    }))
    with pytest.raises(ValueError, match="different training settings"):
        workflow._record_cell(cell_from_mapping({**mapping, "uncertainty": {**mapping["uncertainty"], "gp_lengthscale_squared": 128}}))


def test_protify_arguments_refuse_what_the_cell_owns_or_does_not_run(toy):
    root, _, data_dir = toy
    cell = _cell(root, data_dir, "toy_refused")
    for overrides, message in (
        ({"probe_type": "mlp"}, "sets itself"),
        ({"not_an_argument": 1}, "unknown keys"),
        ({"matrix_embed": True}, "pooled probes only"),
        ({"seed": None}, "seed must be set"),
    ):
        refused = cell_from_mapping({
            "cell_id": cell.cell_id,
            "experiment_id": cell.experiment_id,
            "protify_args": {**cell.protify_args, **overrides},
            "uncertainty": {"branch_specs": list(cell.uncertainty.branch_specs)},
            "output_dir": cell.output_dir,
        })
        with pytest.raises(ValueError, match=message):
            workflow.run_uncertainty_cell(refused)
