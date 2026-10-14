"""No credential reaches Protify on the command line or in a YAML config.

W&B records each run's command line, so credentials come from the environment, which a
`.secrets.env` may fill for HF_TOKEN, WANDB_API_KEY, and SYNTHYRA_API_KEY alone. Every value here is an
invented placeholder.
"""

import os
import sys
import pytest

from types import SimpleNamespace

from protify import cloud_backend
from protify.cloud_cli import _build_cloud_config_from_args, _run_on_cloud, _should_auto_run_cloud
from protify.main import CREDENTIAL_NAMES, parse_arguments


@pytest.mark.parametrize("flag", ["--hf_token", "--wandb_api_key", "--synthyra_api_key", "--cloud_api_key"])
def test_no_option_takes_a_credential(monkeypatch, flag):
    monkeypatch.setattr(sys, "argv", ["main.py", "--data_names", "DeepLoc-2", flag, "placeholder"])

    with pytest.raises(SystemExit):
        parse_arguments()


def test_a_yaml_config_holding_a_token_is_refused(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("data_names:\n  - DeepLoc-2\nhf_token: placeholder\n", encoding="utf-8")
    monkeypatch.setattr("protify.main.load_secrets_env", lambda **_: [])
    monkeypatch.setattr(sys, "argv", ["main.py", "--yaml_path", str(yaml_path)])

    with pytest.raises(SystemExit, match="hf auth login"):
        parse_arguments()


def test_a_yaml_config_may_leave_a_credential_key_empty(tmp_path, monkeypatch):
    """synth's Protify runner writes `hf_token: null` and `cloud_api_key: null` into its configs."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("data_names:\n  - DeepLoc-2\nhf_token: null\ncloud_api_key: null\n", encoding="utf-8")
    monkeypatch.setattr("protify.main.load_secrets_env", lambda **_: [])
    monkeypatch.setattr(sys, "argv", ["main.py", "--yaml_path", str(yaml_path)])

    args = parse_arguments()

    assert args.cloud is False


def test_a_run_takes_only_its_own_credentials_from_a_secrets_file(tmp_path, monkeypatch):
    """Whatever else the file holds, such as another project's keys, stays out of the process."""
    unrelated = "PROTIFY_TEST_UNRELATED_KEY"
    secrets_file = tmp_path / "credentials.env"
    secrets_file.write_text("".join(f"{name}=placeholder\n" for name in (*CREDENTIAL_NAMES, unrelated)), encoding="utf-8")
    for name in (*CREDENTIAL_NAMES, unrelated):
        monkeypatch.setenv(name, "")  # the loader reads empty as unset, and monkeypatch restores the original
    monkeypatch.setenv("FOUNDRY_SECRETS", str(secrets_file))
    monkeypatch.setattr(sys, "argv", ["main.py", "--data_names", "DeepLoc-2"])

    parse_arguments()

    assert [os.environ[name] for name in CREDENTIAL_NAMES] == ["placeholder"] * len(CREDENTIAL_NAMES)
    assert os.environ[unrelated] == ""


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
