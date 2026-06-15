import json

import pytest

try:
    from src.protify.scripts import plan_parallel_probes as preflight
    from src.protify.scripts import run_parallel_probe_launch_manifest as runner
except ImportError:
    try:
        from protify.scripts import plan_parallel_probes as preflight
        from protify.scripts import run_parallel_probe_launch_manifest as runner
    except ImportError:
        from ..scripts import plan_parallel_probes as preflight
        from ..scripts import run_parallel_probe_launch_manifest as runner


def _preflight_report():
    args = preflight.parse_args(
        [
            "--model_names",
            "ESM2-35",
            "ESM2-8",
            "--data_names",
            "EC",
            "--input_size",
            "320",
            "--hidden_size",
            "64",
            "--num_labels",
            "3",
            "--num_runs",
            "3",
            "--n_layers",
            "0",
            "--parallel_max_group_size",
            "2",
            "--wave_max_groups",
            "2",
        ]
    )
    return preflight.build_plan_report(args)


def _over_budget_preflight_report():
    args = preflight.parse_args(
        [
            "--model_names",
            "ESM2-35",
            "ESM2-8",
            "--data_names",
            "EC",
            "--input_size",
            "320",
            "--hidden_size",
            "64",
            "--num_labels",
            "3",
            "--num_runs",
            "3",
            "--n_layers",
            "0",
            "--parallel_max_group_size",
            "2",
            "--wave_max_groups",
            "2",
            "--wave_memory_budget_gb",
            "0.000001",
            "--gpu_indices",
            "0",
            "1",
        ]
    )
    return preflight.build_plan_report(args)


def test_launch_manifest_runner_builds_parallel_dry_run_plan(tmp_path) -> None:
    manifest_path = tmp_path / "preflight.json"
    manifest_path.write_text(json.dumps(_preflight_report()), encoding="utf-8")
    args = runner.parse_args(["--manifest_path", str(manifest_path)])

    report = runner.build_report(args)
    plan = report["plan"]

    assert report["execution"]["executed"] is False
    assert plan["execute"] is False
    assert plan["phase"] == "probes"
    assert plan["variant"] == "parallel"
    assert plan["use_monitor"] is False
    assert plan["skip_completed"] is False
    assert plan["allow_baseline_concurrency"] is False
    assert plan["wave_execution_mode"] == "sequential"
    assert plan["wave_count"] == 2
    assert plan["command_count"] == 2
    assert plan["skipped_count"] == 2
    assert plan["unknown_selected_command_ids"] == []
    assert plan["missing_wave_command_ids"] == []
    assert plan["waves"][0]["wave_id"] == "wave-1"
    assert plan["waves"][0]["command_count"] == 2
    assert plan["waves"][0]["commands"][0]["command_id"] == "group-1"
    assert plan["waves"][0]["commands"][0]["variant"] == "parallel"
    assert plan["waves"][0]["commands"][0]["command_field"] == "parallel_command"
    assert plan["waves"][0]["commands"][0]["command"][:3] == ["python", "-m", "main"]
    assert plan["waves"][0]["commands"][0]["completion_summary_path"] == (
        "telemetry/group-1_parallel.summary.json"
    )
    assert plan["waves"][1]["commands"][0]["skipped"] is True
    assert plan["waves"][1]["commands"][0]["skip_reason"] == "empty_command"


def test_launch_manifest_runner_builds_embedding_prerequisite_dry_run_plan(tmp_path) -> None:
    manifest_path = tmp_path / "preflight.json"
    manifest_path.write_text(json.dumps(_preflight_report()), encoding="utf-8")
    args = runner.parse_args(
        [
            "--manifest_path",
            str(manifest_path),
            "--phase",
            "embeddings",
        ]
    )

    report = runner.build_report(args)
    plan = report["plan"]

    assert plan["phase"] == "embeddings"
    assert plan["wave_count"] == 1
    assert plan["command_count"] == 2
    assert plan["skipped_count"] == 0
    assert plan["over_budget_assignment_count"] == 0
    assert plan["waves"][0]["wave_id"] == "embedding-prerequisites"
    assert plan["waves"][0]["commands"][0]["command_id"] == "embedding-1"
    assert plan["waves"][0]["commands"][0]["variant"] == "embeddings"
    assert plan["waves"][0]["commands"][0]["command_field"] == "embedding_command"
    assert plan["waves"][0]["commands"][0]["environment"] == {"_PROTIFY_EMBED_PHASE": "1"}
    assert plan["waves"][0]["commands"][0]["completion_summary_path"] is None
    assert plan["waves"][0]["commands"][0]["command"][:3] == ["python", "-m", "main"]
    assert "--save_embeddings" in plan["waves"][0]["commands"][0]["command"]


def test_launch_manifest_runner_can_select_embedding_prerequisite_command(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_preflight_report()["launch_manifest"]), encoding="utf-8")
    args = runner.parse_args(
        [
            "--manifest_path",
            str(manifest_path),
            "--phase",
            "embeddings",
            "--command_ids",
            "embedding-2",
        ]
    )

    report = runner.build_report(args)
    plan = report["plan"]

    assert plan["selected_command_ids"] == ["embedding-2"]
    assert plan["unknown_selected_command_ids"] == []
    assert plan["wave_count"] == 1
    assert plan["command_count"] == 1
    assert plan["waves"][0]["commands"][0]["command_id"] == "embedding-2"


def test_launch_manifest_runner_can_plan_embeddings_and_probes_together(tmp_path) -> None:
    manifest_path = tmp_path / "preflight.json"
    manifest_path.write_text(json.dumps(_preflight_report()), encoding="utf-8")
    args = runner.parse_args(
        [
            "--manifest_path",
            str(manifest_path),
            "--phase",
            "all",
        ]
    )

    report = runner.build_report(args)
    plan = report["plan"]

    assert plan["phase"] == "all"
    assert plan["wave_count"] == 3
    assert plan["command_count"] == 4
    assert plan["skipped_count"] == 2
    assert plan["waves"][0]["wave_id"] == "embedding-prerequisites"
    assert plan["waves"][1]["wave_id"] == "wave-1"
    assert plan["waves"][2]["wave_id"] == "wave-2"


def test_launch_manifest_runner_selects_monitored_both_variant_by_wave_and_command(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_preflight_report()["launch_manifest"]), encoding="utf-8")
    args = runner.parse_args(
        [
            "--manifest_path",
            str(manifest_path),
            "--variant",
            "both",
            "--use_monitor",
            "--wave_ids",
            "wave-1",
            "--command_ids",
            "group-1",
        ]
    )

    report = runner.build_report(args)
    plan = report["plan"]

    assert plan["wave_count"] == 1
    assert plan["command_count"] == 2
    assert plan["skipped_count"] == 0
    assert plan["selected_command_ids"] == ["group-1"]
    assert plan["selected_wave_ids"] == ["wave-1"]
    assert [command["variant"] for command in plan["waves"][0]["commands"]] == ["sequential", "parallel"]
    for command in plan["waves"][0]["commands"]:
        assert command["command_field"].endswith("_monitor_command")
        assert command["command"][:3] == ["python", "-m", "scripts.monitor_parallel_probe_hardware"]
        assert "--command" in command["command"]
        assert command["environment"] == {}


def test_launch_manifest_runner_allows_baseline_concurrency_only_when_explicit(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_preflight_report()["launch_manifest"]), encoding="utf-8")
    args = runner.parse_args(
        [
            "--manifest_path",
            str(manifest_path),
            "--variant",
            "both",
            "--wave_execution_mode",
            "concurrent",
            "--allow_baseline_concurrency",
        ]
    )

    report = runner.build_report(args)
    plan = report["plan"]

    assert plan["variant"] == "both"
    assert plan["wave_execution_mode"] == "concurrent"
    assert plan["allow_baseline_concurrency"] is True


def test_launch_manifest_runner_preserves_manifest_environment(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = _preflight_report()["launch_manifest"]
    manifest["groups"][0]["environment"] = {"CUDA_VISIBLE_DEVICES": "2"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args = runner.parse_args(
        [
            "--manifest_path",
            str(manifest_path),
            "--command_ids",
            "group-1",
        ]
    )

    report = runner.build_report(args)
    command = report["plan"]["waves"][0]["commands"][0]

    assert command["command_id"] == "group-1"
    assert command["environment"] == {"CUDA_VISIBLE_DEVICES": "2"}


def test_launch_manifest_runner_skip_completed_uses_summary_paths(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    summary_path = tmp_path / "group-1_parallel.summary.json"
    summary_path.write_text(json.dumps({"sample_count": 2}), encoding="utf-8")
    manifest = _preflight_report()["launch_manifest"]
    manifest["groups"][0]["parallel_monitor_command"] = [
        "python",
        "-m",
        "scripts.monitor_parallel_probe_hardware",
        "--output_jsonl",
        str(tmp_path / "group-1_parallel.jsonl"),
        "--summary_json",
        str(summary_path),
        "--command",
        "--",
    ] + manifest["groups"][0]["parallel_command"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args = runner.parse_args(
        [
            "--manifest_path",
            str(manifest_path),
            "--command_ids",
            "group-1",
            "--skip_completed",
        ]
    )

    report = runner.build_report(args)
    plan = report["plan"]
    command = plan["waves"][0]["commands"][0]

    assert plan["skip_completed"] is True
    assert plan["command_count"] == 0
    assert plan["skipped_count"] == 1
    assert command["skipped"] is True
    assert command["skip_reason"] == "completed_summary_exists"
    assert command["completion_summary_path"] == str(summary_path)
    assert command["command"][:3] == ["python", "-m", "main"]


def test_launch_manifest_runner_writes_report_output_path(tmp_path) -> None:
    manifest_path = tmp_path / "preflight.json"
    output_path = tmp_path / "runner" / "dry-run.json"
    manifest_path.write_text(json.dumps(_preflight_report()), encoding="utf-8")
    args = runner.parse_args(
        [
            "--manifest_path",
            str(manifest_path),
            "--output_path",
            str(output_path),
        ]
    )

    report = runner.build_report(args)
    runner.write_report(report, args.output_path, args.json_indent)
    written_report = json.loads(output_path.read_text(encoding="utf-8"))

    assert written_report["manifest_path"] == str(manifest_path)
    assert written_report["plan"]["wave_count"] == report["plan"]["wave_count"]
    assert written_report["execution"]["executed"] is False


def test_launch_manifest_runner_reports_over_budget_waves_in_dry_run(tmp_path) -> None:
    manifest_path = tmp_path / "preflight.json"
    manifest_path.write_text(json.dumps(_over_budget_preflight_report()), encoding="utf-8")
    args = runner.parse_args(["--manifest_path", str(manifest_path)])

    report = runner.build_report(args)
    plan = report["plan"]

    assert plan["execute"] is False
    assert plan["allow_over_budget"] is False
    assert plan["blocked_by_over_budget"] is False
    assert plan["over_budget_assignment_count"] > 0
    assert len(plan["over_budget_wave_ids"]) > 0
    assert plan["waves"][0]["gpu_over_memory_budget_count"] > 0
    assert plan["waves"][0]["gpu_over_memory_budget"] is True


def test_launch_manifest_runner_blocks_over_budget_execution_by_default(tmp_path) -> None:
    manifest_path = tmp_path / "preflight.json"
    manifest_path.write_text(json.dumps(_over_budget_preflight_report()), encoding="utf-8")
    args = runner.parse_args(["--manifest_path", str(manifest_path), "--execute"])

    with pytest.raises(AssertionError, match="over the preflight GPU memory budget"):
        runner.build_report(args)


def test_launch_manifest_runner_can_execute_over_budget_when_explicitly_allowed(monkeypatch, tmp_path) -> None:
    started_commands = []

    def fake_run(command, check, env):
        started_commands.append(command)

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    manifest_path = tmp_path / "preflight.json"
    manifest_path.write_text(json.dumps(_over_budget_preflight_report()), encoding="utf-8")
    args = runner.parse_args(
        [
            "--manifest_path",
            str(manifest_path),
            "--execute",
            "--allow_over_budget",
            "--command_ids",
            "group-1",
        ]
    )

    report = runner.build_report(args)
    plan = report["plan"]

    assert plan["execute"] is True
    assert plan["allow_over_budget"] is True
    assert plan["blocked_by_over_budget"] is False
    assert plan["over_budget_assignment_count"] > 0
    assert report["execution"]["executed"] is True
    assert report["execution"]["failure_count"] == 0
    assert started_commands == [plan["waves"][0]["commands"][0]["command"]]


def test_launch_manifest_runner_reports_unknown_selected_command(tmp_path) -> None:
    manifest_path = tmp_path / "preflight.json"
    manifest_path.write_text(json.dumps(_preflight_report()), encoding="utf-8")
    args = runner.parse_args(
        [
            "--manifest_path",
            str(manifest_path),
            "--command_ids",
            "missing-group",
        ]
    )

    report = runner.build_report(args)
    plan = report["plan"]

    assert plan["wave_count"] == 0
    assert plan["command_count"] == 0
    assert plan["unknown_selected_command_ids"] == ["missing-group"]


def test_launch_manifest_runner_concurrent_execution_waits_for_wave(monkeypatch) -> None:
    started_commands = []
    started_envs = []

    class FakeProcess:
        def __init__(self, command, env):
            self.command = command
            self.env = env
            if command[0] == "fail":
                self.returncode = 7
            else:
                self.returncode = 0

        def wait(self):
            return self.returncode

    def fake_popen(command, env):
        started_commands.append(command)
        started_envs.append(env)
        return FakeProcess(command, env)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    plan = {
        "execute": True,
        "variant": "parallel",
        "use_monitor": False,
        "wave_execution_mode": "concurrent",
        "continue_on_failure": False,
        "blocked_by_over_budget": False,
        "waves": [
            {
                "wave_id": "wave-1",
                "commands": [
                    {
                        "command_id": "group-1",
                        "variant": "parallel",
                        "command_field": "parallel_command",
                        "command": ["ok", "first"],
                        "environment": {"CUDA_VISIBLE_DEVICES": "0"},
                        "skipped": False,
                        "skip_reason": None,
                    },
                    {
                        "command_id": "group-2",
                        "variant": "parallel",
                        "command_field": "parallel_command",
                        "command": ["fail", "second"],
                        "environment": {"CUDA_VISIBLE_DEVICES": "1"},
                        "skipped": False,
                        "skip_reason": None,
                    },
                ],
            },
            {
                "wave_id": "wave-2",
                "commands": [
                    {
                        "command_id": "group-3",
                        "variant": "parallel",
                        "command_field": "parallel_command",
                        "command": ["ok", "third"],
                        "environment": {"CUDA_VISIBLE_DEVICES": "0"},
                        "skipped": False,
                        "skip_reason": None,
                    },
                ],
            },
        ],
    }

    result = runner.execute_launch_plan(plan)

    assert started_commands == [["ok", "first"], ["fail", "second"]]
    assert started_envs[0]["CUDA_VISIBLE_DEVICES"] == "0"
    assert started_envs[1]["CUDA_VISIBLE_DEVICES"] == "1"
    assert result["executed"] is True
    assert result["failure_count"] == 1
    assert len(result["waves"]) == 1
    assert [command["returncode"] for command in result["waves"][0]["commands"]] == [0, 7]


def test_launch_manifest_runner_rejects_invalid_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "bad.json"
    manifest_path.write_text(json.dumps({"groups": []}), encoding="utf-8")
    args = runner.parse_args(["--manifest_path", str(manifest_path)])

    with pytest.raises(AssertionError, match="waves"):
        runner.build_report(args)


def test_launch_manifest_runner_rejects_invalid_args() -> None:
    with pytest.raises(AssertionError, match="manifest_path"):
        runner.parse_args(["--manifest_path", ""])

    with pytest.raises(AssertionError, match="json_indent"):
        runner.parse_args(["--manifest_path", "plan.json", "--json_indent", "-1"])

    with pytest.raises(AssertionError, match="output_path"):
        runner.parse_args(["--manifest_path", "plan.json", "--output_path", ""])

    with pytest.raises(SystemExit):
        runner.parse_args(["--manifest_path", "plan.json", "--wave_execution_mode", "sideways"])

    with pytest.raises(AssertionError, match="baseline timing"):
        runner.parse_args(
            [
                "--manifest_path",
                "plan.json",
                "--variant",
                "both",
                "--wave_execution_mode",
                "concurrent",
            ]
        )

    with pytest.raises(AssertionError, match="baseline timing"):
        runner.parse_args(
            [
                "--manifest_path",
                "plan.json",
                "--variant",
                "sequential",
                "--wave_execution_mode",
                "concurrent",
            ]
        )
