"""Loading credentials from the gitignored `.secrets.env` into the environment."""

import os
import pytest


try:
    from src.protify.secrets_env import (
        SECRETS_FILENAME,
        find_secrets_file,
        load_secrets_env,
        parse_secrets_file,
    )
except ImportError:
    from protify.secrets_env import (
        SECRETS_FILENAME,
        find_secrets_file,
        load_secrets_env,
        parse_secrets_file,
    )


@pytest.fixture(autouse=True)
def no_named_secrets_file(monkeypatch) -> None:
    """`FOUNDRY_SECRETS` names a file ahead of the upward search these tests exercise."""
    monkeypatch.delenv("FOUNDRY_SECRETS", raising=False)


def write_secrets(directory, body: str):
    path = directory / SECRETS_FILENAME
    path.write_text(body, encoding="utf-8")
    return path


def test_pairs_parse_and_comments_are_skipped(tmp_path) -> None:
    path = write_secrets(
        tmp_path,
        "# a comment\n\nHF_TOKEN=abc123\n   \nWANDB_API_KEY=def456\n",
    )

    assert parse_secrets_file(path) == {"HF_TOKEN": "abc123", "WANDB_API_KEY": "def456"}


def test_export_prefix_is_stripped(tmp_path) -> None:
    path = write_secrets(tmp_path, "export HF_TOKEN=abc123\n")

    assert parse_secrets_file(path) == {"HF_TOKEN": "abc123"}


def test_matching_quotes_are_unwrapped(tmp_path) -> None:
    path = write_secrets(tmp_path, "A=\"quoted value\"\nB='single'\nC=bare\n")

    assert parse_secrets_file(path) == {"A": "quoted value", "B": "single", "C": "bare"}


def test_a_quoted_value_keeps_its_hash(tmp_path) -> None:
    """Only a line starting with '#' is a comment, so '#' inside a value survives."""
    path = write_secrets(tmp_path, 'TOKEN="abc#123"\n')

    assert parse_secrets_file(path) == {"TOKEN": "abc#123"}


def test_lines_without_an_equals_sign_are_ignored(tmp_path) -> None:
    path = write_secrets(tmp_path, "NOT_AN_ASSIGNMENT\nHF_TOKEN=abc\n")

    assert parse_secrets_file(path) == {"HF_TOKEN": "abc"}


def test_a_value_containing_equals_keeps_every_character_after_the_first(tmp_path) -> None:
    path = write_secrets(tmp_path, "URL=https://x/y?a=1&b=2\n")

    assert parse_secrets_file(path) == {"URL": "https://x/y?a=1&b=2"}


def test_the_file_is_found_from_a_nested_directory(tmp_path) -> None:
    write_secrets(tmp_path, "HF_TOKEN=abc\n")
    nested = tmp_path / "src" / "protify"
    nested.mkdir(parents=True)

    assert find_secrets_file(nested) == tmp_path / SECRETS_FILENAME


def test_a_missing_file_loads_nothing(tmp_path) -> None:
    assert find_secrets_file(tmp_path) is None
    assert load_secrets_env(tmp_path) == []


def test_loading_sets_the_environment_and_returns_names(tmp_path, monkeypatch) -> None:
    write_secrets(tmp_path, "PROTIFY_TEST_TOKEN=from_file\n")
    monkeypatch.delenv("PROTIFY_TEST_TOKEN", raising=False)

    assert load_secrets_env(tmp_path) == ["PROTIFY_TEST_TOKEN"]
    assert os.environ["PROTIFY_TEST_TOKEN"] == "from_file"


def test_an_exported_value_beats_the_file(tmp_path, monkeypatch) -> None:
    write_secrets(tmp_path, "PROTIFY_TEST_TOKEN=from_file\n")
    monkeypatch.setenv("PROTIFY_TEST_TOKEN", "from_environment")

    assert load_secrets_env(tmp_path) == []
    assert os.environ["PROTIFY_TEST_TOKEN"] == "from_environment"


def test_override_lets_the_file_win(tmp_path, monkeypatch) -> None:
    write_secrets(tmp_path, "PROTIFY_TEST_TOKEN=from_file\n")
    monkeypatch.setenv("PROTIFY_TEST_TOKEN", "from_environment")

    assert load_secrets_env(tmp_path, override=True) == ["PROTIFY_TEST_TOKEN"]
    assert os.environ["PROTIFY_TEST_TOKEN"] == "from_file"


def test_an_empty_exported_value_does_not_block_the_file(tmp_path, monkeypatch) -> None:
    """An exported but empty variable is not a usable credential, so the file supplies it."""
    write_secrets(tmp_path, "PROTIFY_TEST_TOKEN=from_file\n")
    monkeypatch.setenv("PROTIFY_TEST_TOKEN", "")

    assert load_secrets_env(tmp_path) == ["PROTIFY_TEST_TOKEN"]
    assert os.environ["PROTIFY_TEST_TOKEN"] == "from_file"


def test_the_module_does_not_shadow_the_standard_library(tmp_path) -> None:
    """`src/protify` sits on sys.path during normal runs, so the name must stay clear."""
    import secrets

    assert hasattr(secrets, "token_hex")
