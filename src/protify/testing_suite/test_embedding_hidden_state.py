import os
import sys
from types import SimpleNamespace

import pytest
import torch


PROTIFY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.dirname(PROTIFY_ROOT)
REPO_ROOT = os.path.dirname(SRC_ROOT)
for path in (REPO_ROOT, SRC_ROOT, PROTIFY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


from src.protify.base_models.utils import select_hidden_state
from src.protify.embedder import EmbeddingArguments, get_embedding_filename
from src.protify.hyperopt_utils import HyperoptModule


def test_embedding_filename_preserves_default_cache_name():
    filename = get_embedding_filename("ESM2-8", False, ["var", "mean"])

    assert filename == "ESM2-8_False_mean_var.pth"


def test_embedding_filename_adds_hidden_state_suffix_for_non_default():
    filename = get_embedding_filename(
        "ESM2-8",
        False,
        ["var", "mean"],
        hidden_state_index=6,
    )
    db_filename = get_embedding_filename(
        "ESM2-8",
        True,
        ["mean"],
        extension="db",
        hidden_state_index=3,
    )

    assert filename == "ESM2-8_False_hs6_mean_var.pth"
    assert db_filename == "ESM2-8_True_hs3.db"


def test_embedding_arguments_store_hidden_state_index():
    args = EmbeddingArguments(embedding_hidden_state_index=5)

    assert args.hidden_state_index == 5


def test_select_hidden_state_uses_final_state_by_default():
    final_state = torch.ones(2, 3, 4)
    hidden_states = (torch.zeros(2, 3, 4), torch.full((2, 3, 4), 2.0))

    selected = select_hidden_state(final_state, hidden_states, -1)

    assert selected is final_state


def test_select_hidden_state_uses_requested_tuple_index():
    final_state = torch.ones(2, 3, 4)
    hidden_states = (torch.zeros(2, 3, 4), torch.full((2, 3, 4), 2.0))

    selected = select_hidden_state(final_state, hidden_states, 1)

    assert torch.equal(selected, hidden_states[1])


def test_select_hidden_state_requires_hidden_states_for_non_default():
    final_state = torch.ones(2, 3, 4)

    with pytest.raises(AssertionError):
        select_hidden_state(final_state, None, 0)


def test_parse_arguments_yaml_hidden_state_cli_override(tmp_path, monkeypatch):
    from src.protify.main import parse_arguments

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "data_names:",
                "  - DeepLoc-2",
                "model_names:",
                "  - ESM2-8",
                "embedding_hidden_state_index: 2",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--yaml_path",
            str(yaml_path),
            "--embedding_hidden_state_index",
            "4",
        ],
    )

    args = parse_arguments()

    assert args.embedding_hidden_state_index == 4


def test_hyperopt_apply_config_updates_and_restores_embedding_args():
    main_process = SimpleNamespace(
        probe_args=SimpleNamespace(
            hidden_size=128,
            head_size=64,
            pooling_types=["mean"],
        ),
        trainer_args=SimpleNamespace(),
        embedding_args=SimpleNamespace(
            pooling_types=["mean"],
            hidden_state_index=-1,
        ),
        full_args=SimpleNamespace(random_pair_flipping=False),
    )
    module = HyperoptModule(
        main_process,
        "ESM2-8",
        "DeepLoc-2",
        dataset=None,
        emb_dict=None,
        sweep_config={},
        results_list=[],
    )

    module.apply_config(
        {
            "embedding_pooling_types": "mean-var",
            "embedding_hidden_state_index": 6,
        }
    )
    assert main_process.embedding_args.pooling_types == ["mean", "var"]
    assert main_process.embedding_args.hidden_state_index == 6

    module.apply_config({})
    assert main_process.embedding_args.pooling_types == ["mean"]
    assert main_process.embedding_args.hidden_state_index == -1
