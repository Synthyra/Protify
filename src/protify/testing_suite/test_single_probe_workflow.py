"""The single-probe heads end to end: tiny cells through `run_uncertainty_cell`, then `compare`.

A binary and a regression toy dataset share one cache per branch, as the grid's datasets share
the ablation's. Every head trains two seeds: `projected` and `linear` with the Laplace variance,
the loss predictor, and the p-norm score, `faithful` and `beta_nll` on regression, and
`projected_qclip` on both. The run must save the extras each head asked for, fit every new
method on V_fit, keep every fold head off the rows it scores, record the quantile clip's
bounds from the training rows, refuse what cannot run or ship, and give `compare` what it needs
to write the single-probe verdicts with every check passing.
"""

import json
import sys
import numpy as np
import pandas as pd
import pytest
import torch


try:
    from src.protify.testing_suite.embedding_store_builder import cache_descriptor, write_store
    from src.protify.uncertainty import package, workflow
    from src.protify.uncertainty.cell import cell_from_mapping
    from src.protify.uncertainty.compare import compare_runs
    from src.protify.uncertainty.predictions import load_seed_predictions
    from src.protify.uncertainty.single_probe import OUTPUT_JSON
except ImportError:
    from protify.testing_suite.embedding_store_builder import cache_descriptor, write_store
    from protify.uncertainty import package, workflow
    from protify.uncertainty.cell import cell_from_mapping
    from protify.uncertainty.compare import compare_runs
    from protify.uncertainty.predictions import load_seed_predictions
    from protify.uncertainty.single_probe import OUTPUT_JSON


SAE = "ESMC-300-SAE-l23-k64-c8192"
SAE_WIDTH = 24
DENSE_WIDTH = 16
AMINO_ACIDS = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
SPLIT_SIZES = {"train": 60, "valid": 40, "test": 40}
PRIORS = 13


def _datasets(root) -> tuple[dict[str, str], pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    """Two toy datasets' directories, every row's latent signal, and the frames."""
    rng = np.random.default_rng(0)
    used, signals, directories, frames = set(), [], {}, {}
    for name in ("toybin", "toyreg"):
        frames[name] = {}
        for split, count in SPLIT_SIZES.items():
            sequences = []
            while len(sequences) < count:
                sequence = "M" + "".join(rng.choice(AMINO_ACIDS, size=int(rng.integers(10, 30))))
                if sequence not in used:
                    used.add(sequence)
                    sequences.append(sequence)
            signal = rng.normal(size=count)  # (n,)
            labels = (signal > 0).astype(int) if name == "toybin" else 3.0 * signal + 0.5 * rng.normal(size=count) + 10.0
            frames[name][split] = pd.DataFrame({"seqs": sequences, "labels": labels})
            signals.append(pd.DataFrame({"seqs": sequences, "signal": signal}))
        directory = root / "data" / name
        directory.mkdir(parents=True)
        for split, frame in frames[name].items():
            frame.to_csv(directory / f"{split}.csv", index=False)
        directories[name] = str(directory)
    return directories, pd.concat(signals), frames


def _write_caches(signals: pd.DataFrame, directory) -> dict[str, np.ndarray]:
    """Both branches' feature stores, and each sequence's joined features in branch order."""
    directory.mkdir()
    rng = np.random.default_rng(1)
    signal = signals["signal"].to_numpy()[:, None]  # (n, 1)
    sae = np.maximum(rng.normal(size=(len(signals), SAE_WIDTH)) - 1.0 + 0.5 * signal, 0.0)  # (n, 24)
    dense = rng.normal(size=(len(signals), DENSE_WIDTH)) + signal  # (n, 16)
    caches = (
        (cache_descriptor(SAE, pooling=("max",), max_length=64), sae),
        (cache_descriptor("ESM2-8", pooling=("mean", "var"), max_length=64), dense),
    )
    for descriptor, values in caches:
        tensors = torch.from_numpy(values.astype(np.float32))  # (n, w)
        write_store(directory, descriptor, dict(zip(signals["seqs"], tensors, strict=True)))
    joined = np.concatenate([sae, dense], axis=1).astype(np.float32)  # (n, 40)
    return dict(zip(signals["seqs"], joined, strict=True))  # (40,) per sequence


def _cell(root, data_dir: str, data_name: str, head: str, **uncertainty):
    return cell_from_mapping({
        "cell_id": f"{data_name}_{head}",
        "experiment_id": "toy_single_probe",
        "tags": {"arm": "both", "group": f"{data_name}.both.{head}", "role": "final"},
        "protify_args": {
            "data_dirs": [data_dir],
            "sql": True,
            "seed": 42,
            "num_runs": 2,
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
            "head": head,
            "projection_size": 32,
            "learning_rate": 0.001,
            "dropout": 0.1,
            "calibration_fraction": 0.5,
            "mc_dropout_samples": 0,
            "conformal_levels": [0.8, 0.9],
            "require_cached_embeddings": True,
            **uncertainty,
        },
        "output_dir": str(root / "runs" / data_name / head / "final"),
    })


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main_module = sys.modules[workflow.parse_arguments.__module__]
    monkeypatch.setattr(main_module, "load_secrets_env", lambda: [])
    directories, signals, frames = _datasets(tmp_path)
    features = _write_caches(signals, tmp_path / "embeddings")
    return tmp_path, directories, frames, features


def _spy_folds(monkeypatch) -> tuple[list, list]:
    """The joined rows each fold head trains on and each fold scores, in order."""
    trained, scored = [], []
    train_head, score_fold = workflow._train_head, workflow._score_fold

    def recorded_train(main, seed, train_dataset, *arguments):
        if arguments[3].endswith("_fold"):
            trained.append(set(map(tuple, train_dataset.rows.tolist())))
        return train_head(main, seed, train_dataset, *arguments)

    def recorded_score(model, dataset, *arguments):
        scored.append(set(map(tuple, dataset.rows.tolist())))
        return score_fold(model, dataset, *arguments)

    monkeypatch.setattr(workflow, "_train_head", recorded_train)
    monkeypatch.setattr(workflow, "_score_fold", recorded_score)
    return trained, scored


def test_every_head_runs_saves_its_extras_and_compares(toy, monkeypatch):
    root, directories, frames, features = toy
    extras = {"laplace": True, "loss_head_folds": 2, "pnorm_uncertainty": True, "weight_decay": 0.01}
    trained, scored = _spy_folds(monkeypatch)
    reports = {}
    for data_name, heads in (("toybin", ("projected", "linear", "projected_qclip")), ("toyreg", ("projected", "linear", "faithful", "beta_nll", "projected_qclip"))):
        for head in heads:
            options = extras if head in ("projected", "linear") else {}
            save_heads = head in ("projected_qclip", "linear") or (head == "projected" and data_name == "toybin")
            cell = _cell(root, directories[data_name], data_name, head, save_heads=save_heads, **options)
            reports[(data_name, head)] = workflow.run_uncertainty_cell(cell)

    # Two seeds of two folds for each of the four cells with a loss predictor.
    assert len(trained) == len(scored) == 16
    for train_rows, fold_rows in zip(trained, scored, strict=True):
        assert fold_rows and not train_rows & fold_rows

    binary = reports[("toybin", "projected")]
    assert {"plain", "temperature", "ensemble_temperature", "laplace_temperature", "loss_head"} <= set(binary["methods"])
    # A binary task's centered p-norm score is constant, so the method is multiclass only.
    assert "temperature_pnorm" not in binary["methods"] and "mc_dropout" not in binary["methods"]
    assert binary["methods"]["laplace_temperature"]["test"]["nll"] is not None
    assert {"laplace_seconds", "loss_head_seconds"} <= set(binary["per_seed"][0])
    assert {"temperature", "laplace_temperature", "loss_head"} <= set(reports[("toyreg", "linear")]["methods"])
    assert "temperature_pnorm" not in reports[("toyreg", "linear")]["methods"]
    assert {"faithful", "faithful_temperature", "faithful_ensemble_temperature"} <= set(reports[("toyreg", "faithful")]["methods"])
    assert {"beta_nll", "beta_nll_temperature"} <= set(reports[("toyreg", "beta_nll")]["methods"])
    assert reports[("toyreg", "faithful")]["methods"]["faithful"]["test"]["gaussian_nll"] is not None
    assert "laplace_temperature" not in reports[("toyreg", "faithful")]["methods"]

    seed = load_seed_predictions(str(root / "runs" / "toybin" / "projected" / "final" / "seed_42.npz"))
    assert seed["test"].laplace_variance.shape == (40, PRIORS, 2) and seed["test"].loss_prediction.shape == (40,)
    seed = load_seed_predictions(str(root / "runs" / "toyreg" / "linear" / "final" / "seed_43.npz"))
    assert seed["v_fit"].laplace_variance.shape == (20, PRIORS) and np.all(np.isfinite(seed["v_fit"].loss_prediction))
    seed = load_seed_predictions(str(root / "runs" / "toyreg" / "faithful" / "final" / "seed_42.npz"))
    assert seed["test"].variance.shape == (40,) and seed["test"].laplace_variance is None
    cell = json.loads((root / "runs" / "toybin" / "projected" / "final" / "cell.json").read_text())
    assert cell["uncertainty"]["weight_decay"] == 0.01 and cell["uncertainty"]["loss_head_folds"] == 2

    # The quantile clip's bounds are the standardized training rows' quantiles.
    standardizer, _ = workflow.load_standardizer_record(str(root / "runs" / "toybin" / "projected_qclip" / "final" / "standardizer.npz"))
    def standardized(split: str) -> np.ndarray:
        rows = np.stack([features[sequence] for sequence in frames["toybin"][split]["seqs"]])  # (n, 40)
        return (rows - standardizer.mean.numpy()) / standardizer.scale.numpy()  # (n, 40)

    expected = np.quantile(standardized("train"), [0.001, 0.999], axis=0)  # (2, 40)
    np.testing.assert_allclose(standardizer.lower.numpy(), expected[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(standardizer.upper.numpy(), expected[1], rtol=1e-4, atol=1e-4)
    assert not np.allclose(standardizer.upper.numpy(), np.quantile(standardized("valid"), 0.999, axis=0), atol=1e-3)
    head_config = json.loads((root / "runs" / "toybin" / "projected_qclip" / "final" / "heads" / "seed_42" / "config.json").read_text())
    assert head_config["input_clip"] is None

    # Packaging the new heads and methods is refused, not attempted.
    for head in ("projected_qclip", "linear"):
        with pytest.raises(ValueError, match="not built yet"):
            package.read_finished_cell(str(root / "runs" / "toybin" / head / "final"))
    finished = package.read_finished_cell(str(root / "runs" / "toybin" / "projected" / "final"))
    with pytest.raises(ValueError, match="not built yet"):
        package.select_method(finished, "laplace_temperature")

    verdicts = compare_runs(str(root / "runs"), str(root / "analysis"), replicates=10, seed=0, pattern="*/*/final/report.json", repeats=4, workers=0)
    assert verdicts["checks"]["failed"] == 0
    written = json.loads((root / "analysis" / OUTPUT_JSON).read_text(encoding="utf-8"))
    assert set(written["per_arm"]["both"]) == {"classification", "regression"}
    assert written["per_arm"]["both"]["regression"]["datasets"] == ["toyreg"]


def test_a_regression_only_head_refuses_a_classification_dataset(toy):
    root, directories, _, _ = toy
    with pytest.raises(ValueError, match="takes regression only"):
        workflow.run_uncertainty_cell(_cell(root, directories["toybin"], "toybin", "faithful"))
