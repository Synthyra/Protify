"""Branches: several ordinary Protify caches joined per sequence into one probe row.

The join is only as good as its naming. A branch must read the exact store a single-model run
writes, or it silently re-embeds or reads nothing, and SAE branches must stay sparse until a
batch is densified, or a few hundred thousand proteins do not fit in memory. The cache guard
must count what each branch lacks without writing any cache.
"""

import os
import re
import sys
import numpy as np
import pytest
import torch

from scipy import sparse
from sklearn.preprocessing import StandardScaler


try:
    from src.protify import main as main_module
    from src.protify.data.branch_features import (
        JoinedBranchCollator,
        JoinedBranchDataset,
        branch_cache_path,
        branch_embedding_arguments,
        display_name,
        fit_joined_standardizer,
        join_branch_embeddings,
        parse_branch_spec,
        require_cached_branches,
    )
    from src.protify.embedder import EmbeddingArguments
    from src.protify.testing_suite.embedding_store_builder import write_store
    from src.protify.uncertainty.package import backbones
except ImportError:
    from protify import main as main_module
    from protify.data.branch_features import (
        JoinedBranchCollator,
        JoinedBranchDataset,
        branch_cache_path,
        branch_embedding_arguments,
        display_name,
        fit_joined_standardizer,
        join_branch_embeddings,
        parse_branch_spec,
        require_cached_branches,
    )
    from protify.embedder import EmbeddingArguments
    from protify.testing_suite.embedding_store_builder import write_store
    from protify.uncertainty.package import backbones


SAE = "ESMC-600-SAE-l27-k64-c16384"
SEQUENCES = ["MKTAYIAK", "MVLSPADK", "MSTNPKPQ", "MAHHHHHH", "MQIFVKTL"]


def _write_branch_store(token: str, arguments: dict, rows: dict[str, torch.Tensor]) -> None:
    """The store a single-model run of this branch leaves: its descriptor is the run's own."""
    # rows: (w,) per sequence
    spec = parse_branch_spec(token)
    descriptor = branch_embedding_arguments(spec, arguments).cache_descriptor(spec.model_name)
    write_store(arguments["embedding_save_dir"], descriptor, rows)


def _caches(directory, sql: bool = True) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """An SAE cache 6 wide, mostly zero, and a dense cache 4 wide, both for every sequence."""
    generator = torch.Generator().manual_seed(0)
    sae = {sequence: torch.relu(torch.randn(6, generator=generator) - 1.0) for sequence in SEQUENCES}
    dense = {sequence: torch.randn(4, generator=generator) for sequence in SEQUENCES}
    arguments = {"embedding_save_dir": str(directory), "sql": sql, "max_length": 64}
    for token, rows in ((f"{SAE}:max", sae), ("ESM2-8:mean,var", dense)):
        _write_branch_store(token, arguments, rows)
    return arguments, _as_numpy(sae), _as_numpy(dense)  # the arguments, then (6,) and (4,) per sequence


def _as_numpy(rows: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    # rows: (w,) per sequence
    return {sequence: row.numpy() for sequence, row in rows.items()}  # (w,) per sequence


def test_specs_parse_resolve_and_refuse():
    sae = parse_branch_spec(f"{SAE}:max")
    assert (sae.model_name, sae.pooling_types, sae.hidden_state_index, sae.sparse) == (SAE, ("max",), -1, True)

    layer = parse_branch_spec("ESMC-600:mean,var@27")
    assert (layer.model_name, layer.pooling_types, layer.hidden_state_index, layer.sparse) == (
        "ESMC-600", ("mean", "var"), 27, False,
    )
    assert parse_branch_spec("ESMC-300-SAE:max").model_name == "ESMC-300-SAE-l23-k64-c8192"

    for token in (f"{SAE}:var", f"{SAE}:max@27", "ESMC-600:bogus", "ESMC-600", "ESMC-600:mean,mean", "../x:mean"):
        with pytest.raises(ValueError):
            parse_branch_spec(token)


@pytest.mark.parametrize(
    ("base", "layer", "backbone", "stem"),
    [
        ("ESMC-300", 23, "Synthyra/ESMplusplus_small", "ESMC-300M"),
        ("ESMC-600", 27, "Synthyra/ESMplusplus_large", "ESMC-600M"),
        ("ESMC-6B", 60, "Synthyra/ESMplusplus_6B", "ESMC-6B"),
    ],
)
def test_each_suite_base_reads_its_depth_layer_codebook_and_its_final_state(base, layer, backbone, stem):
    """The production suite's two branches for each base, and the repositories its packages record."""
    sae_name = f"{base}-SAE-l{layer}-k64-c16384"
    sae = parse_branch_spec(f"{sae_name}:max")
    dense = parse_branch_spec(f"{base}:mean,var")

    assert (sae.model_name, sae.pooling_types, sae.sparse) == (sae_name, ("max",), True)
    assert (dense.model_name, dense.pooling_types, dense.hidden_state_index, dense.sparse) == (base, ("mean", "var"), -1, False)
    assert [(entry["model_name"], entry["component"], entry["repository"]) for entry in backbones([sae, dense], None)] == [
        (sae_name, "backbone", backbone),
        (sae_name, "sparse_autoencoder", f"biohub/{stem}-sae-layer{layer}-k64-codebook16384"),
        (base, "backbone", backbone),
    ]


def test_display_name_is_file_name_safe_and_ordered():
    specs = [parse_branch_spec(f"{SAE}:max"), parse_branch_spec("ESMC-600:mean,var@27")]
    name = display_name(specs)
    assert name == f"{SAE}_max+ESMC-600_mean-var_hs27"
    assert ":" not in name and "/" not in name


def test_branch_caches_carry_the_names_single_model_runs_write(tmp_path):
    arguments = {"embedding_save_dir": str(tmp_path), "sql": True, "max_length": 2048}
    runs = {
        f"{SAE}:max": (SAE, ["max"], -1),
        "ESMC-600:mean,var@27": ("ESMC-600", ["mean", "var"], 27),
        "ESMC-600:mean,var": ("ESMC-600", ["mean", "var"], -1),
    }
    for token, (model_name, pooling, hidden_state_index) in runs.items():
        single_model_run = EmbeddingArguments(
            embedding_save_dir=str(tmp_path),
            embedding_pooling_types=pooling,
            embedding_hidden_state_index=hidden_state_index,
            max_length=2048,
        )
        assert branch_cache_path(parse_branch_spec(token), arguments) == single_model_run.feature_directory(model_name)
    # The digest at the end of each name is the descriptor's; the words before it say what is inside.
    assert [os.path.basename(branch_cache_path(parse_branch_spec(token), arguments)).rsplit("__", 1)[0] for token in runs] == [
        f"protify__{SAE}__vector__max__len2048__float32",
        "protify__ESMC-600__vector__mean-var__hs27__len2048__float32",
        "protify__ESMC-600__vector__mean-var__len2048__float32",
    ]


@pytest.mark.parametrize(
    ("flags", "stored"),
    [
        ((), "bfloat16"),
        (("--model_dtype", "fp16"), "float16"),
        (("--embed_dtype", "fp32"), "float32"),
        (("--model_dtype", "fp16", "--embed_dtype", "bf16"), "bfloat16"),
    ],
)
def test_a_parsed_namespace_names_the_store_its_run_writes(monkeypatch, tmp_path, flags, stored):
    """`cell_arguments` hands back the namespace as parsed, where an omitted `embed_dtype` is still None.

    The dtype is part of a store's name, so a branch named from those arguments must resolve it as
    `MainProcess.apply_current_settings` does, or code that binds a cache before the run cannot find it.
    """
    monkeypatch.setattr(main_module, "load_secrets_env", lambda: [])
    monkeypatch.setattr(
        sys, "argv",
        [
            "main.py", "--model_names", "ESMC-600", "--data_names", "DeepLoc-2",
            "--embedding_save_dir", str(tmp_path / "embeddings"), "--log_dir", str(tmp_path / "logs"),
            "--results_dir", str(tmp_path / "results"), "--plots_dir", str(tmp_path / "plots"),
            "--model_save_dir", str(tmp_path / "weights"), *flags,
        ],
    )
    parsed = main_module.parse_arguments()
    spec = parse_branch_spec("ESMC-600:mean,var")
    before_the_run = branch_cache_path(spec, dict(vars(parsed)))

    process = main_module.MainProcess(parsed, GUI=False)
    process.apply_current_settings()  # resolves parsed.embed_dtype in place
    assert branch_cache_path(spec, dict(vars(parsed))) == before_the_run
    assert f"__{stored}__" in os.path.basename(before_the_run)


@pytest.mark.parametrize("sql", [True, False])
def test_join_concatenates_in_spec_order_and_keeps_sae_sparse(tmp_path, sql):
    arguments, sae, dense = _caches(tmp_path, sql=sql)
    specs = [parse_branch_spec(f"{SAE}:max"), parse_branch_spec("ESM2-8:mean,var")]
    joined, widths = join_branch_embeddings(specs, SEQUENCES, arguments)

    assert widths == (6, 4)
    assert sparse.issparse(joined.parts[0]) and not sparse.issparse(joined.parts[1])
    expected = np.stack([np.concatenate([sae[sequence], dense[sequence]]) for sequence in SEQUENCES])  # (5, 10)
    assert np.allclose(joined.block(np.arange(len(SEQUENCES))), expected)


def test_a_missing_cache_or_sequence_is_refused_without_creating_a_file(tmp_path):
    arguments = {"embedding_save_dir": str(tmp_path), "sql": True, "max_length": 64}
    spec = parse_branch_spec("ESM2-8:mean,var")
    with pytest.raises(FileNotFoundError, match="no cache"):
        join_branch_embeddings([spec], SEQUENCES, arguments)
    assert os.listdir(tmp_path) == []

    arguments, _, _ = _caches(tmp_path)
    with pytest.raises(KeyError, match="no embedding"):
        join_branch_embeddings([spec], [*SEQUENCES, "MNOTCACHED"], arguments)


@pytest.mark.parametrize("sql", [True, False])
def test_the_cache_guard_names_each_branch_and_what_it_lacks_without_writing(tmp_path, sql):
    arguments, _, _ = _caches(tmp_path, sql=sql)
    specs = [parse_branch_spec(f"{SAE}:max"), parse_branch_spec("ESM2-8:mean,var")]

    def cache_state() -> dict[str, tuple[int, int]]:
        """Every file under the embedding folder, so a store opened for writing cannot go unnoticed."""
        return {
            path.relative_to(tmp_path).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(tmp_path.rglob("*")) if path.is_file()
        }

    before = cache_state()
    # Complete caches pass, and a repeated sequence is one sequence.
    require_cached_branches(specs, [*SEQUENCES, SEQUENCES[0]], arguments)

    with pytest.raises(ValueError) as refusal:
        require_cached_branches(specs, [*SEQUENCES, "MNOTCACHED", "MALSONOTCACHED"], arguments)
    for spec in specs:
        assert re.search(rf"{re.escape(spec.slug)} lacks 2 of 7 sequences", str(refusal.value))

    # A branch with no cache file lacks every sequence, and none is created for it.
    with pytest.raises(ValueError, match="ESMC-600_mean lacks 5 of 5 sequences"):
        require_cached_branches([parse_branch_spec("ESMC-600:mean")], SEQUENCES, arguments)
    assert cache_state() == before


def test_the_collator_densifies_single_rows_and_pairs(tmp_path):
    arguments, sae, dense = _caches(tmp_path)
    specs = [parse_branch_spec(f"{SAE}:max"), parse_branch_spec("ESM2-8:mean,var")]
    joined, _ = join_branch_embeddings(specs, SEQUENCES, arguments)
    row = {sequence: np.concatenate([sae[sequence], dense[sequence]]) for sequence in SEQUENCES}

    singles = {"seqs": SEQUENCES[:3], "labels": [0, 1, 0]}
    dataset = JoinedBranchDataset(singles, joined, "singlelabel", ppi=False)
    batch = JoinedBranchCollator(joined, ppi=False, embedding_standardizer=None)([dataset[0], dataset[1], dataset[2]])
    assert torch.allclose(batch["embeddings"], torch.from_numpy(np.stack([row[s] for s in SEQUENCES[:3]])))
    assert batch["labels"].dtype == torch.long

    pairs = {"SeqA": SEQUENCES[:2], "SeqB": SEQUENCES[2:4], "labels": [0.5, 1.5]}
    pair_dataset = JoinedBranchDataset(pairs, joined, "regression", ppi=True)
    pair_batch = JoinedBranchCollator(joined, ppi=True, embedding_standardizer=None)([pair_dataset[0], pair_dataset[1]])
    expected = np.stack([np.concatenate([row[a], row[b]]) for a, b in zip(pairs["SeqA"], pairs["SeqB"], strict=True)])  # (2, 20)
    assert torch.allclose(pair_batch["embeddings"], torch.from_numpy(expected))
    assert pair_batch["labels"].dtype == torch.float


def test_training_pairs_flip_and_evaluation_pairs_never_do(tmp_path):
    arguments, _, _ = _caches(tmp_path)
    joined, _ = join_branch_embeddings([parse_branch_spec("ESM2-8:mean,var")], SEQUENCES, arguments)
    pairs = {"SeqA": [SEQUENCES[0]] * 64, "SeqB": [SEQUENCES[1]] * 64, "labels": [0] * 64}

    training = JoinedBranchDataset(pairs, joined, "singlelabel", ppi=True, random_pair_flipping=True, train=True)
    evaluation = JoinedBranchDataset(pairs, joined, "singlelabel", ppi=True, random_pair_flipping=True)
    assert {training[index][0] for index in range(64)} == {(0, 1), (1, 0)}
    assert {evaluation[index][0] for index in range(64)} == {(0, 1)}


def test_the_standardizer_matches_a_scaler_fit_on_the_dense_training_rows(tmp_path):
    arguments, sae, dense = _caches(tmp_path)
    specs = [parse_branch_spec(f"{SAE}:max"), parse_branch_spec("ESM2-8:mean,var")]
    joined, _ = join_branch_embeddings(specs, SEQUENCES, arguments)
    row = {sequence: np.concatenate([sae[sequence], dense[sequence]]) for sequence in SEQUENCES}

    pairs = {"SeqA": SEQUENCES[:3], "SeqB": SEQUENCES[2:5], "labels": [0, 1, 0]}
    standardizer = fit_joined_standardizer(joined, pairs, ppi=True, flip_pairs=True)
    stacked = [np.concatenate([row[a], row[b]]) for a, b in zip(pairs["SeqA"], pairs["SeqB"], strict=True)]
    stacked += [np.concatenate([row[b], row[a]]) for a, b in zip(pairs["SeqA"], pairs["SeqB"], strict=True)]
    reference = StandardScaler().fit(np.stack(stacked))
    assert np.allclose(standardizer.mean.numpy(), reference.mean_, atol=1e-6)
    assert np.allclose(standardizer.scale.numpy(), reference.scale_, atol=1e-6)

    collator = JoinedBranchCollator(joined, ppi=True, embedding_standardizer=standardizer)
    dataset = JoinedBranchDataset(pairs, joined, "singlelabel", ppi=True)
    batch = collator([dataset[0]])
    assert np.allclose(batch["embeddings"].numpy(), reference.transform(stacked[:1]), atol=1e-5)
