"""No credential reaches Protify on the command line or in a YAML config.

W&B records each run's command line, so Hugging Face and W&B resolve their own credentials, from
the environment or a CLI login, and a cloud run reads SYNTHYRA_API_KEY. Every value here is an
invented placeholder.
"""

import os
import sys
import pytest

from types import SimpleNamespace


PROTIFY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.dirname(PROTIFY_ROOT)
REPO_ROOT = os.path.dirname(SRC_ROOT)
for path in (PROTIFY_ROOT, SRC_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


import cloud_backend

from src.protify.cloud_cli import _build_cloud_config_from_args, _run_on_cloud, _should_auto_run_cloud
from src.protify.main import parse_arguments


@pytest.mark.parametrize("flag", ["--hf_token", "--wandb_api_key", "--synthyra_api_key", "--cloud_api_key"])
def test_no_option_takes_a_credential(monkeypatch, flag):
    monkeypatch.setattr(sys, "argv", ["main.py", "--data_names", "DeepLoc-2", flag, "placeholder"])

    with pytest.raises(SystemExit):
        parse_arguments()


def test_a_yaml_config_holding_a_token_is_refused(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("data_names:\n  - DeepLoc-2\nhf_token: placeholder\n", encoding="utf-8")
    monkeypatch.setattr("src.protify.main.load_secrets_env", lambda: [])
    monkeypatch.setattr(sys, "argv", ["main.py", "--yaml_path", str(yaml_path)])

    with pytest.raises(SystemExit, match="hf auth login"):
        parse_arguments()


def test_a_yaml_config_may_leave_a_credential_key_empty(tmp_path, monkeypatch):
    """synth's Protify runner writes `hf_token: null` and `cloud_api_key: null` into its configs."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("data_names:\n  - DeepLoc-2\nhf_token: null\ncloud_api_key: null\n", encoding="utf-8")
    monkeypatch.setattr("src.protify.main.load_secrets_env", lambda: [])
    monkeypatch.setattr(sys, "argv", ["main.py", "--yaml_path", str(yaml_path)])

    args = parse_arguments()

    assert args.cloud is False


def test_a_cloud_run_is_asked_for_and_never_implied_by_a_key(monkeypatch):
    monkeypatch.setenv("SYNTHYRA_API_KEY", "placeholder")
    monkeypatch.delenv("PROTIFY_JOB_ID", raising=False)

    assert not _should_auto_run_cloud(SimpleNamespace(cloud=False, replay_path=None))
    assert _should_auto_run_cloud(SimpleNamespace(cloud=True, replay_path=None))


def test_a_cloud_job_config_leaves_out_the_switch():
    config = _build_cloud_config_from_args(SimpleNamespace(cloud=True, cloud_url=None, data_names=["EC"]))

    assert "cloud" not in config
    assert config["data_names"] == ["EC"]


def test_a_cloud_run_without_a_key_says_which_variable_to_set(monkeypatch):
    monkeypatch.delenv("SYNTHYRA_API_KEY", raising=False)
    monkeypatch.setattr(cloud_backend, "_CLOUD_BACKEND", None)

    with pytest.raises(AssertionError, match="SYNTHYRA_API_KEY"):
        _run_on_cloud(SimpleNamespace(cloud_url=None))
