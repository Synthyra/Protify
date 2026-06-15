import csv
import json

import pytest

try:
    from src.protify.scripts import compare_parallel_probe_runs as compare
except ImportError:
    try:
        from protify.scripts import compare_parallel_probe_runs as compare
    except ImportError:
        from ..scripts import compare_parallel_probe_runs as compare


def _write_results_tsv(path, dataset: str, model: str, metrics) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["dataset", model])
        writer.writerow([dataset, json.dumps(metrics)])


def _write_runner_report(path, returncode: int) -> None:
    path.write_text(
        json.dumps(
            {
                "manifest_path": "preflight.json",
                "plan": {
                    "execute": True,
                    "variant": "parallel",
                    "use_monitor": False,
                    "skip_completed": False,
                    "allow_over_budget": False,
                    "wave_execution_mode": "sequential",
                    "continue_on_failure": False,
                    "selected_command_ids": None,
                    "selected_wave_ids": None,
                    "wave_count": 1,
                    "command_count": 1,
                    "skipped_count": 0,
                    "over_budget_wave_ids": [],
                    "over_budget_assignment_count": 0,
                    "blocked_by_over_budget": False,
                    "unknown_selected_command_ids": [],
                    "missing_wave_command_ids": [],
                    "waves": [],
                },
                "execution": {
                    "executed": True,
                    "failure_count": 0 if returncode == 0 else 1,
                    "waves": [
                        {
                            "wave_id": "wave-1",
                            "commands": [
                                {
                                    "command_id": "group-1",
                                    "variant": "parallel",
                                    "skipped": False,
                                    "returncode": returncode,
                                }
                            ],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def _write_resume_runner_report(path) -> None:
    path.write_text(
        json.dumps(
            {
                "manifest_path": "preflight.json",
                "plan": {
                    "execute": True,
                    "variant": "parallel",
                    "use_monitor": True,
                    "skip_completed": True,
                    "allow_over_budget": False,
                    "wave_execution_mode": "sequential",
                    "continue_on_failure": False,
                    "selected_command_ids": None,
                    "selected_wave_ids": None,
                    "wave_count": 1,
                    "command_count": 1,
                    "skipped_count": 1,
                    "over_budget_wave_ids": [],
                    "over_budget_assignment_count": 0,
                    "blocked_by_over_budget": False,
                    "unknown_selected_command_ids": [],
                    "missing_wave_command_ids": [],
                    "waves": [],
                },
                "execution": {
                    "executed": True,
                    "failure_count": 0,
                    "waves": [
                        {
                            "wave_id": "wave-1",
                            "commands": [
                                {
                                    "command_id": "group-1",
                                    "variant": "parallel",
                                    "skipped": True,
                                    "skip_reason": "completed_summary_exists",
                                },
                                {
                                    "command_id": "group-2",
                                    "variant": "parallel",
                                    "skipped": False,
                                    "returncode": 0,
                                },
                            ],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def test_compare_parallel_probe_parse_metric_number_variants() -> None:
    assert compare.parse_metric_number(1) == pytest.approx(1.0)
    assert compare.parse_metric_number(1.5) == pytest.approx(1.5)
    assert compare.parse_metric_number("2.5") == pytest.approx(2.5)
    assert compare.parse_metric_number("3.0\u00b10.2") == pytest.approx(3.0)
    assert compare.parse_metric_number("4.0\u00c2\u00b10.3") == pytest.approx(4.0)
    assert compare.parse_metric_number("5.0+/-0.4") == pytest.approx(5.0)
    assert compare.parse_metric_number(True) is None
    assert compare.parse_metric_number("not-a-number") is None


def test_compare_parallel_probe_tsv_results_report_speedup_and_metric_parity(tmp_path) -> None:
    sequential_path = tmp_path / "sequential.tsv"
    parallel_path = tmp_path / "parallel.tsv"
    _write_results_tsv(
        sequential_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds": "10.0000\u00b11.0000",
            "training_time_seconds_mean": 10.0,
            "test_loss_mean": 0.20,
            "sequential_probe_ensemble_test_loss": 0.18,
            "sequential_probe_ensemble_test_accuracy": 0.90,
        },
    )
    _write_results_tsv(
        parallel_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds": 12.0,
            "parallel_probe_seconds_per_run": 3.0,
            "parallel_probe_total_runs": 4,
            "parallel_probe_trainer_invocations": 1,
            "parallel_probe_invocation_reduction": 3,
            "parallel_probe_compression_ratio": 4.0,
            "parallel_probe_group_runtime_records": [
                {
                    "group_number": 1,
                    "execution_kind": "vectorized",
                    "num_runs": 3,
                    "run_seeds": [42, 43, 44],
                    "train_runtime_seconds": 9.0,
                    "seconds_per_run": 3.0,
                    "estimated_peak_bytes": 1024,
                    "estimated_training_flops_per_batch": 900,
                },
                {
                    "group_number": 2,
                    "execution_kind": "eligible_singleton",
                    "num_runs": 1,
                    "run_seeds": [45],
                    "train_runtime_seconds": 4.0,
                    "seconds_per_run": 4.0,
                    "estimated_peak_bytes": 512,
                    "estimated_training_flops_per_batch": 300,
                },
            ],
            "parallel_probe_run_records": [
                {"seed": 42},
                {"seed": 43},
                {"seed": 44},
                {"seed": 45},
            ],
            "test_loss_mean": 0.205,
            "parallel_probe_ensemble_test_loss": 0.181,
            "parallel_probe_ensemble_test_accuracy": 0.895,
        },
    )
    args = compare.parse_args(
        [
            "--sequential_results",
            str(sequential_path),
            "--parallel_results",
            str(parallel_path),
            "--metric_abs_tolerance",
            "0.01",
        ]
    )

    report = compare.build_comparison_report(args)

    assert report["summary"]["matched_pair_count"] == 1
    assert report["summary"]["mean_wall_clock_speedup"] == pytest.approx(40.0 / 12.0)
    assert report["summary"]["parallel_group_runtime"]["available_pair_count"] == 1
    assert report["summary"]["parallel_group_runtime"]["total_group_count"] == 2
    assert report["summary"]["parallel_group_runtime"]["total_vectorized_group_count"] == 1
    assert report["summary"]["parallel_group_runtime"]["total_eligible_singleton_group_count"] == 1
    assert report["summary"]["parallel_group_runtime"]["total_group_runtime_seconds"] == pytest.approx(13.0)
    assert report["summary"]["parallel_group_runtime"]["max_group_seconds_per_run"] == pytest.approx(4.0)
    assert report["summary"]["parallel_group_runtime"]["slowest_group"]["dataset"] == "EC"
    assert report["summary"]["parallel_group_runtime"]["slowest_group"]["model"] == "ESM2-35"
    assert report["summary"]["parallel_group_runtime"]["slowest_group"]["group_number"] == 2
    assert report["summary"]["total_comparable_metric_count"] == 1
    assert report["summary"]["total_failing_metric_count"] == 0
    verdict = report["summary"]["validation_verdict"]
    assert verdict["status"] == "pass"
    assert verdict["failing_requirements"] == []
    assert verdict["missing_requirements"] == []
    assert "telemetry_coverage" in verdict["not_required_requirements"]
    pair = report["pairs"][0]
    assert pair["dataset"] == "EC"
    assert pair["model"] == "ESM2-35"
    assert pair["timing"]["num_runs"] == 4
    assert pair["timing"]["sequential_total_seconds"] == pytest.approx(40.0)
    assert pair["timing"]["parallel_total_seconds"] == pytest.approx(12.0)
    assert pair["timing"]["sequential_seconds_per_run"] == pytest.approx(10.0)
    assert pair["timing"]["parallel_seconds_per_run"] == pytest.approx(3.0)
    assert pair["timing"]["wall_clock_speedup"] == pytest.approx(40.0 / 12.0)
    assert pair["timing"]["per_run_speedup"] == pytest.approx(10.0 / 3.0)
    assert pair["parallel_group_runtime"]["available"] is True
    assert pair["parallel_group_runtime"]["group_count"] == 2
    assert pair["parallel_group_runtime"]["vectorized_group_count"] == 1
    assert pair["parallel_group_runtime"]["eligible_singleton_group_count"] == 1
    assert pair["parallel_group_runtime"]["total_group_runtime_seconds"] == pytest.approx(13.0)
    assert pair["parallel_group_runtime"]["max_group_seconds_per_run"] == pytest.approx(4.0)
    assert pair["parallel_group_runtime"]["slowest_group"]["group_number"] == 2
    assert pair["parallel_group_runtime"]["slowest_group"]["execution_kind"] == "eligible_singleton"
    assert pair["parallel_group_runtime"]["slowest_group"]["run_seeds"] == [45]
    assert pair["parallel_group_runtime"]["slowest_group"]["estimated_peak_bytes"] == pytest.approx(512)
    assert pair["parallel_trainer_invocations"] == pytest.approx(1.0)
    assert pair["parallel_invocation_reduction"] == pytest.approx(3.0)
    assert pair["parallel_compression_ratio"] == pytest.approx(4.0)
    assert pair["metric_comparisons"][0]["metric"] == "test_loss_mean"
    assert pair["metric_comparisons"][0]["within_tolerance"] is True
    assert pair["comparable_ensemble_metric_count"] == 2
    assert pair["failing_ensemble_metric_count"] == 0
    assert pair["ensemble_metric_comparisons"][0]["metric"] == "test_loss"
    assert pair["ensemble_metric_comparisons"][0]["sequential_metric"] == "sequential_probe_ensemble_test_loss"
    assert pair["ensemble_metric_comparisons"][0]["parallel_metric"] == "parallel_probe_ensemble_test_loss"
    assert pair["ensemble_metric_comparisons"][0]["within_tolerance"] is True


def test_compare_parallel_probe_json_records_report_missing_pairs(tmp_path) -> None:
    sequential_path = tmp_path / "sequential.json"
    parallel_path = tmp_path / "parallel.json"
    sequential_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "dataset": "EC",
                        "model": "ESM2-35",
                        "metrics": {
                            "training_time_seconds_mean": 5.0,
                            "test_loss_mean": 0.1,
                        },
                    },
                    {
                        "dataset": "DeepLoc-2",
                        "model": "ESM2-8",
                        "metrics": {
                            "training_time_seconds_mean": 7.0,
                            "test_loss_mean": 0.3,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    parallel_path.write_text(
        json.dumps(
            [
                {
                    "data_name": "EC",
                    "model_name": "ESM2-35",
                    "training_time_seconds": 6.0,
                    "parallel_probe_total_runs": 2,
                    "test_loss_mean": 0.12,
                }
            ]
        ),
        encoding="utf-8",
    )
    args = compare.parse_args(
        [
            "--sequential_results",
            str(sequential_path),
            "--parallel_results",
            str(parallel_path),
            "--metric_abs_tolerance",
            "0.005",
        ]
    )

    report = compare.build_comparison_report(args)

    assert report["summary"]["sequential_record_count"] == 2
    assert report["summary"]["parallel_record_count"] == 1
    assert report["summary"]["matched_pair_count"] == 1
    assert report["summary"]["missing_parallel_keys"] == [["DeepLoc-2", "ESM2-8"]]
    assert report["summary"]["missing_sequential_keys"] == []
    assert report["summary"]["total_failing_metric_count"] == 1
    verdict = report["summary"]["validation_verdict"]
    assert verdict["status"] == "fail"
    assert verdict["failing_requirements"] == ["result_pair_coverage", "metric_parity"]
    assert report["pairs"][0]["metric_comparisons"][0]["within_tolerance"] is False


def test_compare_parallel_probe_adds_telemetry_summary_comparison(tmp_path) -> None:
    sequential_path = tmp_path / "sequential.tsv"
    parallel_path = tmp_path / "parallel.tsv"
    manifest_path = tmp_path / "preflight.json"
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    _write_results_tsv(
        sequential_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds_mean": 10.0,
            "test_loss_mean": 0.20,
        },
    )
    _write_results_tsv(
        parallel_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds": 12.0,
            "parallel_probe_total_runs": 2,
            "test_loss_mean": 0.20,
        },
    )
    manifest_path.write_text(
        json.dumps(
            {
                "launch_manifest": {
                    "groups": [
                        {
                            "command_id": "group-1",
                            "data_name": "EC",
                            "model_name": "ESM2-35",
                            "sequential_monitor_command": ["python", "-m", "monitor", "group-1-seq"],
                            "parallel_monitor_command": ["python", "-m", "monitor", "group-1-par"],
                        },
                        {
                            "command_id": "group-2",
                            "data_name": "EC",
                            "model_name": "ESM2-35",
                            "sequential_monitor_command": ["python", "-m", "monitor", "group-2-seq"],
                            "parallel_monitor_command": ["python", "-m", "monitor", "group-2-par"],
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    sequential_summary_1 = telemetry_dir / "group-1_sequential.summary.json"
    sequential_summary_1.write_text(
        json.dumps(
            {
                "sample_count": 2,
                "duration_seconds": 4.0,
                "gpu_utilization_percent_mean": 50.0,
                "gpu_utilization_percent_max": 75.0,
                "memory_used_mib_max": 1000.0,
                "power_draw_watts_mean": 200.0,
            }
        ),
        encoding="utf-8",
    )
    sequential_summary_2 = telemetry_dir / "group-2_sequential.summary.json"
    sequential_summary_2.write_text(
        json.dumps(
            {
                "sample_count": 4,
                "duration_seconds": 6.0,
                "gpu_utilization_percent_mean": 70.0,
                "gpu_utilization_percent_max": 90.0,
                "memory_used_mib_max": 1200.0,
                "power_draw_watts_mean": 250.0,
            }
        ),
        encoding="utf-8",
    )
    parallel_summary = telemetry_dir / "group-1_parallel.summary.json"
    parallel_summary.write_text(
        json.dumps(
            {
                "sample_count": 3,
                "duration_seconds": 5.0,
                "gpu_utilization_percent_mean": 90.0,
                "gpu_utilization_percent_max": 98.0,
                "memory_used_mib_max": 2000.0,
                "power_draw_watts_mean": 300.0,
            }
        ),
        encoding="utf-8",
    )
    args = compare.parse_args(
        [
            "--sequential_results",
            str(sequential_path),
            "--parallel_results",
            str(parallel_path),
            "--launch_manifest",
            str(manifest_path),
            "--sequential_telemetry_summaries",
            str(sequential_summary_1),
            str(sequential_summary_2),
            "--parallel_telemetry_summaries",
            str(parallel_summary),
            "--min_parallel_gpu_utilization_percent",
            "85",
            "--min_gpu_utilization_gain_percent",
            "20",
        ]
    )

    report = compare.build_comparison_report(args)
    telemetry = report["pairs"][0]["telemetry_comparison"]
    sequential_telemetry = telemetry["sequential"]
    parallel_telemetry = telemetry["parallel"]

    assert report["summary"]["sequential_telemetry_key_count"] == 1
    assert report["summary"]["parallel_telemetry_key_count"] == 1
    assert report["summary"]["telemetry_coverage"]["sequential"]["complete"] is True
    assert report["summary"]["telemetry_coverage"]["sequential"]["missing_command_ids"] == []
    assert report["summary"]["telemetry_coverage"]["parallel"]["complete"] is False
    assert report["summary"]["telemetry_coverage"]["parallel"]["missing_command_ids"] == ["group-2"]
    verdict = report["summary"]["validation_verdict"]
    assert verdict["status"] == "pass"
    assert verdict["failing_requirements"] == []
    assert verdict["missing_requirements"] == []
    assert sequential_telemetry["summary_count"] == 2
    assert sequential_telemetry["command_ids"] == ["group-1", "group-2"]
    assert sequential_telemetry["sample_count"] == pytest.approx(6.0)
    assert sequential_telemetry["duration_seconds"] == pytest.approx(10.0)
    assert sequential_telemetry["gpu_utilization_percent_mean"] == pytest.approx((50.0 * 2.0 + 70.0 * 4.0) / 6.0)
    assert sequential_telemetry["gpu_utilization_percent_max"] == pytest.approx(90.0)
    assert parallel_telemetry["summary_count"] == 1
    assert parallel_telemetry["command_ids"] == ["group-1"]
    assert parallel_telemetry["gpu_utilization_percent_mean"] == pytest.approx(90.0)
    gpu_mean_comparison = [
        item for item in telemetry["metric_comparisons"]
        if item["metric"] == "gpu_utilization_percent_mean"
    ][0]
    assert gpu_mean_comparison["delta"] == pytest.approx(90.0 - ((50.0 * 2.0 + 70.0 * 4.0) / 6.0))
    assert gpu_mean_comparison["ratio"] == pytest.approx(90.0 / ((50.0 * 2.0 + 70.0 * 4.0) / 6.0))


def test_compare_parallel_probe_validation_verdict_reports_strict_speedup_failure(tmp_path) -> None:
    sequential_path = tmp_path / "sequential.tsv"
    parallel_path = tmp_path / "parallel.tsv"
    _write_results_tsv(
        sequential_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds_mean": 10.0,
            "test_loss_mean": 0.20,
        },
    )
    _write_results_tsv(
        parallel_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds": 12.0,
            "parallel_probe_total_runs": 4,
            "test_loss_mean": 0.20,
        },
    )
    args = compare.parse_args(
        [
            "--sequential_results",
            str(sequential_path),
            "--parallel_results",
            str(parallel_path),
            "--min_wall_clock_speedup",
            "4.0",
        ]
    )

    report = compare.build_comparison_report(args)
    verdict = report["summary"]["validation_verdict"]

    assert verdict["status"] == "fail"
    assert verdict["failing_requirements"] == ["wall_clock_speedup"]
    wall_clock_requirement = [
        item for item in verdict["requirements"]
        if item["name"] == "wall_clock_speedup"
    ][0]
    assert wall_clock_requirement["observed_value"] == pytest.approx(40.0 / 12.0)
    assert wall_clock_requirement["threshold_value"] == pytest.approx(4.0)


def test_compare_parallel_probe_validation_verdict_checks_manifest_speedup_efficiency(tmp_path) -> None:
    sequential_path = tmp_path / "sequential.tsv"
    parallel_path = tmp_path / "parallel.tsv"
    manifest_path = tmp_path / "preflight.json"
    _write_results_tsv(
        sequential_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds_mean": 10.0,
            "test_loss_mean": 0.20,
        },
    )
    _write_results_tsv(
        parallel_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds": 20.0,
            "parallel_probe_total_runs": 8,
            "test_loss_mean": 0.20,
        },
    )
    manifest_path.write_text(
        json.dumps(
            {
                "launch_manifest": {
                    "groups": [
                        {
                            "command_id": "group-1",
                            "data_name": "EC",
                            "model_name": "ESM2-35",
                            "sequential_command": ["python", "-m", "main"],
                            "parallel_command": ["python", "-m", "main", "--parallel_probe_runs"],
                        }
                    ]
                },
                "validation_comparison": {
                    "sequential_trainer_invocations": 8,
                    "parallel_trainer_invocations": 1,
                    "trainer_invocation_speedup_ceiling": 8.0,
                },
            }
        ),
        encoding="utf-8",
    )
    args = compare.parse_args(
        [
            "--sequential_results",
            str(sequential_path),
            "--parallel_results",
            str(parallel_path),
            "--launch_manifest",
            str(manifest_path),
            "--min_manifest_speedup_efficiency",
            "0.75",
        ]
    )

    report = compare.build_comparison_report(args)
    summary = report["summary"]
    verdict = summary["validation_verdict"]

    assert summary["mean_wall_clock_speedup"] == pytest.approx(4.0)
    expectation = summary["manifest_speedup_expectation"]
    assert expectation["sequential_trainer_invocations"] == pytest.approx(8.0)
    assert expectation["parallel_trainer_invocations"] == pytest.approx(1.0)
    assert expectation["trainer_invocation_speedup_ceiling"] == pytest.approx(8.0)
    assert summary["manifest_speedup_efficiency"] == pytest.approx(0.5)
    assert verdict["status"] == "fail"
    assert verdict["failing_requirements"] == ["manifest_speedup_efficiency"]
    efficiency_requirement = [
        item for item in verdict["requirements"]
        if item["name"] == "manifest_speedup_efficiency"
    ][0]
    assert efficiency_requirement["observed_value"] == pytest.approx(0.5)
    assert efficiency_requirement["threshold_value"] == pytest.approx(0.75)


def test_compare_parallel_probe_validation_verdict_requires_manifest_result_coverage(tmp_path) -> None:
    sequential_path = tmp_path / "sequential.tsv"
    parallel_path = tmp_path / "parallel.tsv"
    manifest_path = tmp_path / "preflight.json"
    _write_results_tsv(
        sequential_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds_mean": 10.0,
            "test_loss_mean": 0.20,
        },
    )
    _write_results_tsv(
        parallel_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds": 10.0,
            "parallel_probe_total_runs": 1,
            "test_loss_mean": 0.20,
        },
    )
    manifest_path.write_text(
        json.dumps(
            {
                "launch_manifest": {
                    "groups": [
                        {
                            "command_id": "group-1",
                            "data_name": "EC",
                            "model_name": "ESM2-35",
                            "sequential_command": ["python", "-m", "main"],
                            "parallel_command": ["python", "-m", "main", "--parallel_probe_runs"],
                        },
                        {
                            "command_id": "group-2",
                            "data_name": "DeepLoc-2",
                            "model_name": "ESM2-8",
                            "sequential_command": ["python", "-m", "main"],
                            "parallel_command": ["python", "-m", "main", "--parallel_probe_runs"],
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    args = compare.parse_args(
        [
            "--sequential_results",
            str(sequential_path),
            "--parallel_results",
            str(parallel_path),
            "--launch_manifest",
            str(manifest_path),
            "--require_manifest_result_coverage",
        ]
    )

    report = compare.build_comparison_report(args)
    summary = report["summary"]
    verdict = summary["validation_verdict"]

    assert summary["matched_pair_count"] == 1
    assert summary["missing_parallel_keys"] == []
    assert summary["missing_sequential_keys"] == []
    assert summary["manifest_result_key_count"] == 2
    assert summary["manifest_missing_result_keys"] == [["DeepLoc-2", "ESM2-8"]]
    assert summary["manifest_unexpected_result_keys"] == []
    assert verdict["status"] == "fail"
    assert verdict["failing_requirements"] == ["manifest_result_coverage"]
    manifest_requirement = [
        item for item in verdict["requirements"]
        if item["name"] == "manifest_result_coverage"
    ][0]
    assert manifest_requirement["observed_value"]["manifest_missing_result_keys"] == [
        ["DeepLoc-2", "ESM2-8"]
    ]


def test_compare_parallel_probe_validation_verdict_requires_manifest_probe_result_coverage(tmp_path) -> None:
    sequential_path = tmp_path / "sequential.json"
    parallel_path = tmp_path / "parallel.json"
    manifest_path = tmp_path / "preflight.json"
    sequential_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "dataset": "EC",
                        "model": "ESM2-35",
                        "probe_type": "linear",
                        "hidden_size": 32,
                        "dropout": 0.0,
                        "n_layers": 0,
                        "task_type": "singlelabel",
                        "num_labels": 3,
                        "metrics": {
                            "training_time_seconds_mean": 10.0,
                            "test_loss_mean": 0.20,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    parallel_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "dataset": "EC",
                        "model": "ESM2-35",
                        "probe_type": "linear",
                        "hidden_size": 32,
                        "dropout": 0.0,
                        "n_layers": 0,
                        "task_type": "singlelabel",
                        "num_labels": 3,
                        "metrics": {
                            "training_time_seconds": 10.0,
                            "parallel_probe_total_runs": 1,
                            "test_loss_mean": 0.20,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "launch_manifest": {
                    "groups": [
                        {
                            "command_id": "group-1",
                            "data_name": "EC",
                            "model_name": "ESM2-35",
                            "probe_type": "linear",
                            "hidden_size": 32,
                            "dropout": 0.0,
                            "n_layers": 0,
                            "task_type": "singlelabel",
                            "num_labels": 3,
                            "sequential_command": ["python", "-m", "main"],
                            "parallel_command": ["python", "-m", "main", "--parallel_probe_runs"],
                        },
                        {
                            "command_id": "group-2",
                            "data_name": "EC",
                            "model_name": "ESM2-35",
                            "probe_type": "linear",
                            "hidden_size": 64,
                            "dropout": 0.0,
                            "n_layers": 0,
                            "task_type": "singlelabel",
                            "num_labels": 3,
                            "sequential_command": ["python", "-m", "main"],
                            "parallel_command": ["python", "-m", "main", "--parallel_probe_runs"],
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    args = compare.parse_args(
        [
            "--sequential_results",
            str(sequential_path),
            "--parallel_results",
            str(parallel_path),
            "--launch_manifest",
            str(manifest_path),
            "--require_manifest_result_coverage",
            "--require_manifest_probe_result_coverage",
        ]
    )

    report = compare.build_comparison_report(args)
    summary = report["summary"]
    verdict = summary["validation_verdict"]

    assert summary["matched_pair_count"] == 1
    assert summary["manifest_missing_result_keys"] == []
    assert summary["manifest_probe_result_identity_count"] == 2
    assert summary["observed_probe_result_identity_available"] is True
    assert summary["manifest_missing_probe_result_identities"] == [
        ["EC", "ESM2-35", "linear", 64, 0, 0, "singlelabel", 3]
    ]
    assert verdict["status"] == "fail"
    assert verdict["failing_requirements"] == ["manifest_probe_result_coverage"]
    probe_requirement = [
        item for item in verdict["requirements"]
        if item["name"] == "manifest_probe_result_coverage"
    ][0]
    assert probe_requirement["observed_value"]["manifest_missing_probe_result_identities"] == [
        ["EC", "ESM2-35", "linear", 64, 0, 0, "singlelabel", 3]
    ]


def test_compare_parallel_probe_manifest_probe_result_coverage_uses_tsv_cell_identity(tmp_path) -> None:
    sequential_path = tmp_path / "sequential.tsv"
    parallel_path = tmp_path / "parallel.tsv"
    manifest_path = tmp_path / "preflight.json"
    identity_metrics = {
        "probe_type": "linear",
        "hidden_size": 32,
        "dropout": 0.0,
        "n_layers": 0,
        "task_type": "singlelabel",
        "num_labels": 3,
    }
    _write_results_tsv(
        sequential_path,
        "EC",
        "ESM2-35",
        {
            **identity_metrics,
            "training_time_seconds_mean": 10.0,
            "test_loss_mean": 0.20,
        },
    )
    _write_results_tsv(
        parallel_path,
        "EC",
        "ESM2-35",
        {
            **identity_metrics,
            "training_time_seconds": 10.0,
            "parallel_probe_total_runs": 1,
            "test_loss_mean": 0.20,
        },
    )
    manifest_path.write_text(
        json.dumps(
            {
                "launch_manifest": {
                    "groups": [
                        {
                            "command_id": "group-1",
                            "data_name": "EC",
                            "model_name": "ESM2-35",
                            "probe_type": "linear",
                            "hidden_size": 32,
                            "dropout": 0.0,
                            "n_layers": 0,
                            "task_type": "singlelabel",
                            "num_labels": 3,
                            "sequential_command": ["python", "-m", "main"],
                            "parallel_command": ["python", "-m", "main", "--parallel_probe_runs"],
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    args = compare.parse_args(
        [
            "--sequential_results",
            str(sequential_path),
            "--parallel_results",
            str(parallel_path),
            "--launch_manifest",
            str(manifest_path),
            "--require_manifest_probe_result_coverage",
        ]
    )

    report = compare.build_comparison_report(args)
    summary = report["summary"]
    verdict = summary["validation_verdict"]

    assert summary["observed_probe_result_identity_available"] is True
    assert summary["manifest_missing_probe_result_identities"] == []
    assert verdict["status"] == "pass"


def test_compare_parallel_probe_validation_verdict_requires_complete_telemetry(tmp_path) -> None:
    sequential_path = tmp_path / "sequential.tsv"
    parallel_path = tmp_path / "parallel.tsv"
    manifest_path = tmp_path / "preflight.json"
    summary_path = tmp_path / "group-1_sequential.summary.json"
    _write_results_tsv(
        sequential_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds_mean": 5.0,
            "test_loss_mean": 0.1,
        },
    )
    _write_results_tsv(
        parallel_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds": 5.0,
            "parallel_probe_total_runs": 2,
            "test_loss_mean": 0.1,
        },
    )
    manifest_path.write_text(
        json.dumps(
            {
                "launch_manifest": {
                    "groups": [
                        {
                            "command_id": "group-1",
                            "data_name": "EC",
                            "model_name": "ESM2-35",
                            "sequential_monitor_command": ["python", "-m", "monitor", "group-1-seq"],
                            "parallel_monitor_command": ["python", "-m", "monitor", "group-1-par"],
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "sample_count": 2,
                "duration_seconds": 3.0,
                "gpu_utilization_percent_mean": 45.0,
            }
        ),
        encoding="utf-8",
    )
    args = compare.parse_args(
        [
            "--sequential_results",
            str(sequential_path),
            "--parallel_results",
            str(parallel_path),
            "--launch_manifest",
            str(manifest_path),
            "--sequential_telemetry_summaries",
            str(summary_path),
            "--require_complete_telemetry",
        ]
    )

    report = compare.build_comparison_report(args)
    verdict = report["summary"]["validation_verdict"]

    assert verdict["status"] == "fail"
    assert verdict["failing_requirements"] == ["telemetry_coverage"]
    telemetry_requirement = [
        item for item in verdict["requirements"]
        if item["name"] == "telemetry_coverage"
    ][0]
    assert telemetry_requirement["observed_value"]["parallel_missing_command_ids"] == ["group-1"]


def test_compare_parallel_probe_validation_verdict_accepts_successful_runner_report(tmp_path) -> None:
    sequential_path = tmp_path / "sequential.tsv"
    parallel_path = tmp_path / "parallel.tsv"
    manifest_path = tmp_path / "preflight.json"
    runner_report_path = tmp_path / "manifest_runner_execute.report.json"
    _write_results_tsv(
        sequential_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds_mean": 5.0,
            "test_loss_mean": 0.1,
        },
    )
    _write_results_tsv(
        parallel_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds": 5.0,
            "parallel_probe_total_runs": 2,
            "test_loss_mean": 0.1,
        },
    )
    manifest_path.write_text(
        json.dumps(
            {
                "launch_manifest": {
                    "groups": [
                        {
                            "command_id": "group-1",
                            "data_name": "EC",
                            "model_name": "ESM2-35",
                            "parallel_command": ["python", "-m", "main"],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    _write_runner_report(runner_report_path, returncode=0)
    args = compare.parse_args(
        [
            "--sequential_results",
            str(sequential_path),
            "--parallel_results",
            str(parallel_path),
            "--launch_manifest",
            str(manifest_path),
            "--runner_reports",
            str(runner_report_path),
            "--require_successful_runner_reports",
        ]
    )

    report = compare.build_comparison_report(args)
    runner_reports = report["summary"]["runner_reports"]
    verdict = report["summary"]["validation_verdict"]

    assert runner_reports["provided"] is True
    assert runner_reports["report_count"] == 1
    assert runner_reports["executed_report_count"] == 1
    assert runner_reports["dry_run_report_count"] == 0
    assert runner_reports["total_plan_command_count"] == 1
    assert runner_reports["total_executed_command_count"] == 1
    assert runner_reports["total_execution_failure_count"] == 0
    assert runner_reports["successful"] is True
    assert runner_reports["manifest_coverage"]["complete"] is True
    assert verdict["status"] == "pass"
    assert verdict["failing_requirements"] == []
    runner_requirement = [
        item for item in verdict["requirements"]
        if item["name"] == "runner_reports"
    ][0]
    assert runner_requirement["status"] == "pass"


def test_compare_parallel_probe_validation_verdict_fails_failed_runner_report(tmp_path) -> None:
    sequential_path = tmp_path / "sequential.tsv"
    parallel_path = tmp_path / "parallel.tsv"
    runner_report_path = tmp_path / "manifest_runner_execute.report.json"
    _write_results_tsv(
        sequential_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds_mean": 5.0,
            "test_loss_mean": 0.1,
        },
    )
    _write_results_tsv(
        parallel_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds": 5.0,
            "parallel_probe_total_runs": 2,
            "test_loss_mean": 0.1,
        },
    )
    _write_runner_report(runner_report_path, returncode=7)
    args = compare.parse_args(
        [
            "--sequential_results",
            str(sequential_path),
            "--parallel_results",
            str(parallel_path),
            "--runner_reports",
            str(runner_report_path),
            "--require_successful_runner_reports",
        ]
    )

    report = compare.build_comparison_report(args)
    runner_reports = report["summary"]["runner_reports"]
    verdict = report["summary"]["validation_verdict"]

    assert runner_reports["total_execution_failure_count"] == 1
    assert runner_reports["failed_commands"] == [
        {
            "path": str(runner_report_path),
            "wave_id": "wave-1",
            "command_id": "group-1",
            "returncode": 7,
        }
    ]
    assert runner_reports["successful"] is False
    assert verdict["status"] == "fail"
    assert verdict["failing_requirements"] == ["runner_reports"]
    runner_requirement = [
        item for item in verdict["requirements"]
        if item["name"] == "runner_reports"
    ][0]
    assert runner_requirement["observed_value"]["failed_commands"] == [
        {
            "path": str(runner_report_path),
            "wave_id": "wave-1",
            "command_id": "group-1",
            "returncode": 7,
        }
    ]


def test_compare_parallel_probe_runner_report_requires_manifest_command_coverage(tmp_path) -> None:
    sequential_path = tmp_path / "sequential.tsv"
    parallel_path = tmp_path / "parallel.tsv"
    manifest_path = tmp_path / "preflight.json"
    runner_report_path = tmp_path / "manifest_runner_execute.report.json"
    _write_results_tsv(
        sequential_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds_mean": 5.0,
            "test_loss_mean": 0.1,
        },
    )
    _write_results_tsv(
        parallel_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds": 5.0,
            "parallel_probe_total_runs": 2,
            "test_loss_mean": 0.1,
        },
    )
    manifest_path.write_text(
        json.dumps(
            {
                "launch_manifest": {
                    "groups": [
                        {
                            "command_id": "group-1",
                            "data_name": "EC",
                            "model_name": "ESM2-35",
                            "parallel_command": ["python", "-m", "main"],
                        },
                        {
                            "command_id": "group-2",
                            "data_name": "EC",
                            "model_name": "ESM2-35",
                            "parallel_command": ["python", "-m", "main"],
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    _write_runner_report(runner_report_path, returncode=0)
    args = compare.parse_args(
        [
            "--sequential_results",
            str(sequential_path),
            "--parallel_results",
            str(parallel_path),
            "--launch_manifest",
            str(manifest_path),
            "--runner_reports",
            str(runner_report_path),
            "--require_successful_runner_reports",
        ]
    )

    report = compare.build_comparison_report(args)
    coverage = report["summary"]["runner_reports"]["manifest_coverage"]
    verdict = report["summary"]["validation_verdict"]

    assert coverage["expectation_available"] is True
    assert coverage["expected_command_variant_pairs"] == [
        ["group-1", "parallel"],
        ["group-2", "parallel"],
    ]
    assert coverage["observed_command_variant_pairs"] == [["group-1", "parallel"]]
    assert coverage["missing_command_variant_pairs"] == [["group-2", "parallel"]]
    assert coverage["complete"] is False
    assert report["summary"]["runner_reports"]["successful"] is False
    assert verdict["status"] == "fail"
    assert verdict["failing_requirements"] == ["runner_reports"]


def test_compare_parallel_probe_runner_report_counts_completed_skips_as_manifest_coverage(tmp_path) -> None:
    sequential_path = tmp_path / "sequential.tsv"
    parallel_path = tmp_path / "parallel.tsv"
    manifest_path = tmp_path / "preflight.json"
    runner_report_path = tmp_path / "manifest_runner_execute.report.json"
    _write_results_tsv(
        sequential_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds_mean": 5.0,
            "test_loss_mean": 0.1,
        },
    )
    _write_results_tsv(
        parallel_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds": 5.0,
            "parallel_probe_total_runs": 2,
            "test_loss_mean": 0.1,
        },
    )
    manifest_path.write_text(
        json.dumps(
            {
                "launch_manifest": {
                    "groups": [
                        {
                            "command_id": "group-1",
                            "data_name": "EC",
                            "model_name": "ESM2-35",
                            "parallel_command": ["python", "-m", "main"],
                        },
                        {
                            "command_id": "group-2",
                            "data_name": "EC",
                            "model_name": "ESM2-35",
                            "parallel_command": ["python", "-m", "main"],
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    _write_resume_runner_report(runner_report_path)
    args = compare.parse_args(
        [
            "--sequential_results",
            str(sequential_path),
            "--parallel_results",
            str(parallel_path),
            "--launch_manifest",
            str(manifest_path),
            "--runner_reports",
            str(runner_report_path),
            "--require_successful_runner_reports",
        ]
    )

    report = compare.build_comparison_report(args)
    runner_reports = report["summary"]["runner_reports"]
    coverage = runner_reports["manifest_coverage"]
    verdict = report["summary"]["validation_verdict"]

    assert runner_reports["executed_command_variant_pairs"] == [["group-2", "parallel"]]
    assert runner_reports["completed_summary_command_variant_pairs"] == [
        ["group-1", "parallel"]
    ]
    assert runner_reports["covered_command_variant_pairs"] == [
        ["group-2", "parallel"],
        ["group-1", "parallel"],
    ]
    assert coverage["observed_command_variant_pairs"] == [
        ["group-1", "parallel"],
        ["group-2", "parallel"],
    ]
    assert coverage["missing_command_variant_pairs"] == []
    assert coverage["complete"] is True
    assert runner_reports["successful"] is True
    assert verdict["status"] == "pass"


def test_compare_parallel_probe_runner_manifest_coverage_ignores_embedding_commands(tmp_path) -> None:
    sequential_path = tmp_path / "sequential.tsv"
    parallel_path = tmp_path / "parallel.tsv"
    manifest_path = tmp_path / "preflight.json"
    runner_report_path = tmp_path / "manifest_runner_all_execute.report.json"
    _write_results_tsv(
        sequential_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds_mean": 5.0,
            "test_loss_mean": 0.1,
        },
    )
    _write_results_tsv(
        parallel_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds": 5.0,
            "parallel_probe_total_runs": 2,
            "test_loss_mean": 0.1,
        },
    )
    manifest_path.write_text(
        json.dumps(
            {
                "launch_manifest": {
                    "embedding_prerequisites": {
                        "embedding_jobs": [
                            {
                                "command_id": "embedding-1",
                                "command": ["python", "-m", "main", "--save_embeddings"],
                                "command_environment": {"_PROTIFY_EMBED_PHASE": "1"},
                            }
                        ]
                    },
                    "groups": [
                        {
                            "command_id": "group-1",
                            "data_name": "EC",
                            "model_name": "ESM2-35",
                            "parallel_command": ["python", "-m", "main"],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    runner_report_path.write_text(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "plan": {
                    "execute": True,
                    "phase": "all",
                    "variant": "parallel",
                    "use_monitor": False,
                    "skip_completed": False,
                    "allow_over_budget": False,
                    "wave_execution_mode": "sequential",
                    "continue_on_failure": False,
                    "selected_command_ids": None,
                    "selected_wave_ids": None,
                    "wave_count": 2,
                    "command_count": 2,
                    "skipped_count": 0,
                    "over_budget_wave_ids": [],
                    "over_budget_assignment_count": 0,
                    "blocked_by_over_budget": False,
                    "unknown_selected_command_ids": [],
                    "missing_wave_command_ids": [],
                    "waves": [],
                },
                "execution": {
                    "executed": True,
                    "failure_count": 0,
                    "waves": [
                        {
                            "wave_id": "embedding-prerequisites",
                            "commands": [
                                {
                                    "command_id": "embedding-1",
                                    "variant": "embeddings",
                                    "skipped": False,
                                    "returncode": 0,
                                }
                            ],
                        },
                        {
                            "wave_id": "wave-1",
                            "commands": [
                                {
                                    "command_id": "group-1",
                                    "variant": "parallel",
                                    "skipped": False,
                                    "returncode": 0,
                                }
                            ],
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    args = compare.parse_args(
        [
            "--sequential_results",
            str(sequential_path),
            "--parallel_results",
            str(parallel_path),
            "--launch_manifest",
            str(manifest_path),
            "--runner_reports",
            str(runner_report_path),
            "--require_successful_runner_reports",
        ]
    )

    report = compare.build_comparison_report(args)
    runner_reports = report["summary"]["runner_reports"]
    coverage = runner_reports["manifest_coverage"]

    assert runner_reports["covered_command_variant_pairs"] == [
        ["embedding-1", "embeddings"],
        ["group-1", "parallel"],
    ]
    assert coverage["observed_command_variant_pairs"] == [["group-1", "parallel"]]
    assert coverage["unexpected_command_variant_pairs"] == []
    assert coverage["missing_command_variant_pairs"] == []
    assert coverage["complete"] is True
    assert runner_reports["successful"] is True


def test_compare_parallel_probe_writes_report_output_path(tmp_path) -> None:
    sequential_path = tmp_path / "sequential.tsv"
    parallel_path = tmp_path / "parallel.tsv"
    output_path = tmp_path / "reports" / "compare.json"
    _write_results_tsv(
        sequential_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds_mean": 5.0,
            "test_loss_mean": 0.1,
        },
    )
    _write_results_tsv(
        parallel_path,
        "EC",
        "ESM2-35",
        {
            "training_time_seconds": 5.0,
            "parallel_probe_total_runs": 2,
            "test_loss_mean": 0.1,
        },
    )
    args = compare.parse_args(
        [
            "--sequential_results",
            str(sequential_path),
            "--parallel_results",
            str(parallel_path),
            "--output_path",
            str(output_path),
        ]
    )

    report = compare.build_comparison_report(args)
    compare.write_report(report, args.output_path, args.json_indent)
    written_report = json.loads(output_path.read_text(encoding="utf-8"))

    assert written_report["summary"]["matched_pair_count"] == 1
    assert written_report["summary"]["validation_verdict"]["status"] == "pass"
    assert written_report["pairs"][0]["dataset"] == "EC"


def test_compare_parallel_probe_rejects_invalid_args(tmp_path) -> None:
    sequential_path = tmp_path / "sequential.tsv"
    parallel_path = tmp_path / "parallel.tsv"
    sequential_path.write_text("dataset\tmodel\n", encoding="utf-8")
    parallel_path.write_text("dataset\tmodel\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="metric_abs_tolerance"):
        compare.parse_args(
            [
                "--sequential_results",
                str(sequential_path),
                "--parallel_results",
                str(parallel_path),
                "--metric_abs_tolerance",
                "-1",
            ]
        )

    with pytest.raises(AssertionError, match="min_wall_clock_speedup"):
        compare.parse_args(
            [
                "--sequential_results",
                str(sequential_path),
                "--parallel_results",
                str(parallel_path),
                "--min_wall_clock_speedup",
                "-1",
            ]
        )

    with pytest.raises(AssertionError, match="min_parallel_gpu_utilization_percent"):
        compare.parse_args(
            [
                "--sequential_results",
                str(sequential_path),
                "--parallel_results",
                str(parallel_path),
                "--min_parallel_gpu_utilization_percent",
                "101",
            ]
        )

    with pytest.raises(AssertionError, match="min_manifest_speedup_efficiency"):
        compare.parse_args(
            [
                "--sequential_results",
                str(sequential_path),
                "--parallel_results",
                str(parallel_path),
                "--min_manifest_speedup_efficiency",
                "-0.1",
            ]
        )

    with pytest.raises(AssertionError, match="Unsupported"):
        compare.load_result_records([str(tmp_path / "metrics.csv")])

    with pytest.raises(AssertionError, match="launch_manifest"):
        compare.parse_args(
            [
                "--sequential_results",
                str(sequential_path),
                "--parallel_results",
                str(parallel_path),
                "--sequential_telemetry_summaries",
                str(tmp_path / "group-1_sequential.summary.json"),
            ]
        )

    with pytest.raises(AssertionError, match="output_path"):
        compare.parse_args(
            [
                "--sequential_results",
                str(sequential_path),
                "--parallel_results",
                str(parallel_path),
                "--output_path",
                "",
            ]
        )
