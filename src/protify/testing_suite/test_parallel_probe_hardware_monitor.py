import json

import pytest

try:
    from src.protify.scripts import monitor_parallel_probe_hardware as monitor
except ImportError:
    try:
        from protify.scripts import monitor_parallel_probe_hardware as monitor
    except ImportError:
        from ..scripts import monitor_parallel_probe_hardware as monitor


def test_hardware_monitor_builds_nvidia_smi_command() -> None:
    command = monitor.nvidia_smi_query_command(gpu_index=2)

    assert command[0] == "nvidia-smi"
    assert command[1] == "--id=2"
    assert "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm" in command
    assert "--format=csv,noheader,nounits" in command


def test_hardware_monitor_parses_nvidia_smi_csv_line() -> None:
    sample = monitor.parse_nvidia_smi_csv_line(
        "2026/06/10 12:00:00.000, 0, 75, 44, 40960, 81920, 512.5, 1410",
        sampled_at_unix_seconds=10.5,
    )

    assert sample["sampled_at_unix_seconds"] == pytest.approx(10.5)
    assert sample["driver_timestamp"] == "2026/06/10 12:00:00.000"
    assert sample["gpu_index"] == 0
    assert sample["gpu_utilization_percent"] == 75
    assert sample["memory_utilization_percent"] == 44
    assert sample["memory_used_mib"] == 40960
    assert sample["memory_total_mib"] == 81920
    assert sample["memory_used_fraction"] == pytest.approx(0.5)
    assert sample["power_draw_watts"] == pytest.approx(512.5)
    assert sample["sm_clock_mhz"] == 1410


def test_hardware_monitor_summarizes_jsonl_samples(tmp_path) -> None:
    path = tmp_path / "telemetry.jsonl"
    samples = [
        monitor.parse_nvidia_smi_csv_line(
            "2026/06/10 12:00:00.000, 0, 50, 30, 1000, 2000, 200.0, 1000",
            sampled_at_unix_seconds=1.0,
        ),
        monitor.parse_nvidia_smi_csv_line(
            "2026/06/10 12:00:01.000, 0, 90, 60, 1500, 2000, 300.0, 1200",
            sampled_at_unix_seconds=3.0,
        ),
    ]
    monitor.append_samples(path, samples)

    loaded = monitor.load_samples(path)
    summary = monitor.summarize_samples(loaded)

    assert len(loaded) == 2
    assert summary["sample_count"] == 2
    assert summary["gpu_indices"] == [0]
    assert summary["duration_seconds"] == pytest.approx(2.0)
    assert summary["gpu_utilization_percent_mean"] == pytest.approx(70.0)
    assert summary["gpu_utilization_percent_max"] == pytest.approx(90.0)
    assert summary["memory_used_mib_max"] == pytest.approx(1500.0)
    assert summary["memory_used_fraction_max"] == pytest.approx(0.75)
    assert summary["power_draw_watts_mean"] == pytest.approx(250.0)
    assert summary["sm_clock_mhz_max"] == pytest.approx(1200.0)


def test_hardware_monitor_dry_run_plan_strips_command_separator(tmp_path) -> None:
    output_path = tmp_path / "samples.jsonl"
    summary_path = tmp_path / "summary.json"
    args = monitor.parse_args(
        [
            "--output_jsonl",
            str(output_path),
            "--summary_json",
            str(summary_path),
            "--gpu_index",
            "1",
            "--interval_seconds",
            "0.5",
            "--dry_run",
            "--command",
            "--",
            "python",
            "-m",
            "main",
            "--parallel_probe_runs",
        ]
    )

    plan = monitor.dry_run_plan(args)

    assert plan["mode"] == "dry_run"
    assert plan["nvidia_smi_command"][1] == "--id=1"
    assert plan["wrapped_command"] == ["python", "-m", "main", "--parallel_probe_runs"]
    assert plan["output_jsonl"] == str(output_path)
    assert plan["summary_json"] == str(summary_path)
    assert plan["interval_seconds"] == pytest.approx(0.5)


def test_hardware_monitor_summary_mode_writes_summary(tmp_path) -> None:
    samples_path = tmp_path / "samples.jsonl"
    summary_path = tmp_path / "summary.json"
    monitor.append_samples(
        samples_path,
        [
            monitor.parse_nvidia_smi_csv_line(
                "2026/06/10 12:00:00.000, 0, 80, 50, 1200, 2400, 250.0, 1100",
                sampled_at_unix_seconds=1.0,
            )
        ],
    )
    args = monitor.parse_args(
        [
            "--summarize_jsonl",
            str(samples_path),
            "--summary_json",
            str(summary_path),
        ]
    )
    summary = monitor.summarize_samples(monitor.load_samples(samples_path))
    monitor.write_summary_if_requested(summary, args.summary_json)

    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted["sample_count"] == 1
    assert persisted["gpu_utilization_percent_mean"] == pytest.approx(80.0)


def test_hardware_monitor_rejects_invalid_args(tmp_path) -> None:
    with pytest.raises(AssertionError, match="interval_seconds"):
        monitor.parse_args(
            [
                "--output_jsonl",
                str(tmp_path / "samples.jsonl"),
                "--interval_seconds",
                "0",
                "--command",
                "python",
            ]
        )

    with pytest.raises(AssertionError, match="gpu_index"):
        monitor.parse_args(
            [
                "--output_jsonl",
                str(tmp_path / "samples.jsonl"),
                "--gpu_index",
                "-1",
                "--command",
                "python",
            ]
        )

    with pytest.raises(AssertionError, match="output_jsonl"):
        monitor.parse_args(["--command", "python"])

    with pytest.raises(AssertionError, match="command"):
        monitor.parse_args(
            [
                "--output_jsonl",
                str(tmp_path / "samples.jsonl"),
                "--command",
                "--",
            ]
        )
