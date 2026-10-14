"""The default probe through `main.py`: an ESMC base with no probe choice trains the joined branches.

`--model_names ESMC-300` alone must train the Biohub SAE at layer 23, max-pooled, beside the final
hidden state's mean and variance, joined into a `projected` head (docs/conventions/probes.md).
Naming a probe type, a pooling set, a hidden state, or another workflow keeps the run as written.

The end-to-end run uses feature stores keyed as Protify keys them for ESMC-300's two default
branches, narrower than the real ones, so no model loads and the run stays fast.
"""

import math
import sys
import numpy as np
import pandas as pd
import pytest
import torch

from types import SimpleNamespace


try:
    from src.protify import main as main_module
    from src.protify.probes.projected_probe import ProjectedProbe
    from src.protify.testing_suite.embedding_store_builder import cache_descriptor, write_store
except ImportError:
    from protify import main as main_module
    from protify.probes.projected_probe import ProjectedProbe
    from protify.testing_suite.embedding_store_builder import cache_descriptor, write_store


SAE = "ESMC-300-SAE-l23-k64-c16384"
SAE_WIDTH = 24
DENSE_WIDTH = 16
AMINO_ACIDS = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
SPLIT_SIZES = {"train": 48, "valid": 24, "test": 24}


def _parse(monkeypatch, *argv: str) -> SimpleNamespace:
    monkeypatch.setattr(main_module, "load_secrets_env", lambda: [])
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    return main_module.parse_arguments()


def test_an_esmc_run_naming_no_probe_choice_allows_the_default(monkeypatch):
    args = _parse(monkeypatch, "--model_names", "ESMC-300", "--data_names", "DeepLoc-2")

    assert args.default_probe is True


@pytest.mark.parametrize(
    "argv",
    [
        ("--probe_type", "mlp"),
        ("--probe_type=mlp",),
        ("--embedding_pooling_types", "mean"),
        ("--embedding_hidden_state_index", "20"),
        ("--no_default_probe",),
        ("--matrix_embed", "--probe_type", "transformer"),
        ("--use_scikit",),
        ("--feature_selection",),
    ],
)
def test_a_named_probe_choice_or_workflow_keeps_the_run_as_written(monkeypatch, argv):
    args = _parse(monkeypatch, "--model_names", "ESMC-300", "--data_names", "DeepLoc-2", *argv)

    assert args.default_probe is False


def test_a_yaml_probe_choice_keeps_the_run_as_written(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("data_names: [DeepLoc-2]\nmodel_names: [ESMC-300]\nprobe_type: mlp\n")
    assert _parse(monkeypatch, "--yaml_path", str(config)).default_probe is False

    config.write_text("data_names: [DeepLoc-2]\nmodel_names: [ESMC-300]\n")
    assert _parse(monkeypatch, "--yaml_path", str(config)).default_probe is True
    assert _parse(monkeypatch, "--yaml_path", str(config), "--no_default_probe").default_probe is False


@pytest.mark.parametrize(
    ("model_names", "allowed", "expected"),
    [
        (["ESMC-300"], True, True),
        (["ESMC-300", "ESMC-600", "ESMC-6B"], True, True),
        (["ESMC-300", "ESM2-8"], True, False),
        (["ESMC-300"], False, False),
    ],
)
def test_the_default_needs_every_model_to_be_an_esmc_base(model_names, allowed, expected):
    process = SimpleNamespace(
        full_args=SimpleNamespace(default_probe=allowed),
        model_args=SimpleNamespace(model_names=model_names, _model_paths=None),
    )

    assert main_module.MainProcess.uses_default_probe(process) is expected


def _toy_dataset(root) -> tuple[pd.DataFrame, str]:
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
    return pd.concat(frames.values()), str(data_dir)


def _write_default_caches(rows: pd.DataFrame, directory) -> None:
    """ESMC-300's two default branch stores, keyed as a plain run of each keys them.

    The SAE's active fraction (64 of 16384) is below the coordinate-list threshold, so its store is `csr`.
    """
    directory.mkdir()
    rng = np.random.default_rng(1)
    signal = np.where(rows["labels"].to_numpy()[:, None] == 1, 1.0, -1.0)  # (n, 1)
    sae = np.maximum(rng.normal(size=(len(rows), SAE_WIDTH)) - 1.0 + 0.5 * signal, 0.0)  # (n, 24)
    dense = rng.normal(size=(len(rows), DENSE_WIDTH)) + signal  # (n, 16)
    caches = (
        (cache_descriptor(SAE, pooling=("max",), max_length=64), sae),
        (cache_descriptor("ESMC-300", pooling=("mean", "var"), max_length=64), dense),
    )
    for descriptor, values in caches:
        tensors = torch.from_numpy(values.astype(np.float32))  # (n, w)
        write_store(directory, descriptor, dict(zip(rows["seqs"], tensors, strict=True)))


def test_esmc_300_alone_trains_the_joined_default_probe(monkeypatch, tmp_path):
    # Hugging Face writes checkpoints under the working directory; nothing may land in the tree.
    monkeypatch.chdir(tmp_path)
    rows, data_dir = _toy_dataset(tmp_path)
    _write_default_caches(rows, tmp_path / "embeddings")
    args = _parse(
        monkeypatch,
        "--model_names", "ESMC-300",
        "--data_dirs", data_dir,
        "--sql",
        "--seed", "42",
        "--num_runs", "2",
        "--num_epochs", "2",
        "--max_length", "64",
        "--probe_batch_size", "16",
        "--projection_size", "32",
        "--no_compile",
        "--embedding_save_dir", str(tmp_path / "embeddings"),
        "--log_dir", str(tmp_path / "logs"),
        "--results_dir", str(tmp_path / "results"),
        "--plots_dir", str(tmp_path / "plots"),
        "--model_save_dir", str(tmp_path / "weights"),
    )
    process = main_module.MainProcess(args, GUI=False)
    process.apply_current_settings()
    process.get_datasets()
    built = []
    get_probe = main_module.get_probe

    def recorded(probe_args):
        probe = get_probe(probe_args)
        built.append(probe)
        return probe

    monkeypatch.setattr(main_module, "get_probe", recorded)
    trainers = sys.modules[main_module.TrainerMixin.__module__]
    monkeypatch.setattr(trainers, "get_probe", recorded)

    assert process.uses_default_probe()
    process.embed_default_branches()
    process.run_default_probes()

    model_name = f"{SAE}_max+ESMC-300_mean-var"
    # main.py imports `protify`, which may be a second copy of the module this file imported.
    assert built and all(type(probe).__name__ == ProjectedProbe.__name__ for probe in built)
    # The default head is the expanded MLP: --projection_size is the width its first layer reads, and the
    # output layer reads the smallest multiple of 256 covering the labels (ProjectedProbe in probes/projected_probe.py).
    num_labels = rows["labels"].nunique()
    class_aligned_width = math.ceil(num_labels / 256) * 256  # f
    assert all(probe.trunk == "expanded_mlp" for probe in built)
    assert all(probe.blocks[0].expand.in_features == 32 for probe in built)
    assert all(probe.output.in_features == class_aligned_width for probe in built)
    assert process.probe_args.branch_widths == [SAE_WIDTH, DENSE_WIDTH]
    assert process.probe_args.input_size == SAE_WIDTH + DENSE_WIDTH
    assert process.full_args.probe_type == "projected"
    # Two seeds, each a fresh head, reported as one row under the joined name.
    assert len(built) == 3
    (data_name,) = process.datasets
    reported = process.logger_data_tracking[data_name][model_name]
    assert reported["sequential_probe_total_runs"] == 2
