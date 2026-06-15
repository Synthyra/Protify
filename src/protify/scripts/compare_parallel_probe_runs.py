import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_METRIC_KEYS = (
    "test_loss_mean",
    "eval_loss_mean",
    "loss_mean",
    "accuracy_mean",
    "mcc_mean",
    "f1_mean",
    "roc_auc_mean",
    "spearman_rho_mean",
    "pearson_r_mean",
    "mse_mean",
    "rmse_mean",
)


DEFAULT_ENSEMBLE_METRIC_SUFFIXES = (
    "test_loss",
    "test_accuracy",
    "test_mcc",
    "test_f1",
    "test_roc_auc",
    "test_pr_auc",
    "test_spearman_rho",
    "test_pearson_rho",
    "test_mse",
    "test_rmse",
)


PROBE_IDENTITY_KEYS = (
    "probe_type",
    "hidden_size",
    "dropout",
    "n_layers",
    "task_type",
    "num_labels",
)


@dataclass(frozen=True)
class MetricsRecord:
    source_path: str
    dataset: str
    model: str
    metrics: Dict[str, object]

    @property
    def key(self) -> Tuple[str, str]:
        return (self.dataset, self.model)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare future sequential and parallel probe result files without "
            "running training."
        )
    )
    parser.add_argument("--sequential_results", nargs="+", required=True)
    parser.add_argument("--parallel_results", nargs="+", required=True)
    parser.add_argument("--metric_keys", nargs="+", default=None)
    parser.add_argument("--metric_abs_tolerance", type=float, default=0.0)
    parser.add_argument("--metric_rel_tolerance", type=float, default=0.0)
    parser.add_argument("--min_wall_clock_speedup", type=float, default=1.0)
    parser.add_argument("--min_per_run_speedup", type=float, default=0.0)
    parser.add_argument("--max_failing_metric_count", type=int, default=0)
    parser.add_argument("--max_failing_ensemble_metric_count", type=int, default=0)
    parser.add_argument("--require_ensemble_metrics", action="store_true")
    parser.add_argument("--require_manifest_result_coverage", action="store_true")
    parser.add_argument("--require_manifest_probe_result_coverage", action="store_true")
    parser.add_argument("--require_complete_telemetry", action="store_true")
    parser.add_argument("--require_successful_runner_reports", action="store_true")
    parser.add_argument("--min_manifest_speedup_efficiency", type=float, default=None)
    parser.add_argument("--min_parallel_gpu_utilization_percent", type=float, default=None)
    parser.add_argument("--min_gpu_utilization_gain_percent", type=float, default=None)
    parser.add_argument("--launch_manifest", type=str, default=None)
    parser.add_argument("--runner_reports", nargs="+", default=None)
    parser.add_argument("--sequential_telemetry_summaries", nargs="+", default=None)
    parser.add_argument("--parallel_telemetry_summaries", nargs="+", default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--json_indent", type=int, default=2)
    return validate_args(parser.parse_args(argv))


def validate_args(args):
    assert len(args.sequential_results) > 0, "At least one sequential result path is required."
    assert len(args.parallel_results) > 0, "At least one parallel result path is required."
    assert args.metric_abs_tolerance >= 0.0, "metric_abs_tolerance must be non-negative."
    assert args.metric_rel_tolerance >= 0.0, "metric_rel_tolerance must be non-negative."
    assert args.min_wall_clock_speedup >= 0.0, "min_wall_clock_speedup must be non-negative."
    assert args.min_per_run_speedup >= 0.0, "min_per_run_speedup must be non-negative."
    assert args.max_failing_metric_count >= 0, "max_failing_metric_count must be non-negative."
    assert args.max_failing_ensemble_metric_count >= 0, (
        "max_failing_ensemble_metric_count must be non-negative."
    )
    if args.min_parallel_gpu_utilization_percent is not None:
        assert 0.0 <= args.min_parallel_gpu_utilization_percent <= 100.0, (
            "min_parallel_gpu_utilization_percent must be in [0, 100]."
        )
    if args.min_gpu_utilization_gain_percent is not None:
        assert args.min_gpu_utilization_gain_percent >= 0.0, (
            "min_gpu_utilization_gain_percent must be non-negative."
        )
    if args.min_manifest_speedup_efficiency is not None:
        assert args.min_manifest_speedup_efficiency >= 0.0, (
            "min_manifest_speedup_efficiency must be non-negative."
        )
    assert args.json_indent >= 0, "json_indent must be non-negative."
    if args.sequential_telemetry_summaries is not None or args.parallel_telemetry_summaries is not None:
        assert args.launch_manifest is not None, "launch_manifest is required when telemetry summaries are provided."
    if args.sequential_telemetry_summaries is not None:
        assert len(args.sequential_telemetry_summaries) > 0, (
            "sequential_telemetry_summaries must be non-empty when provided."
        )
    if args.parallel_telemetry_summaries is not None:
        assert len(args.parallel_telemetry_summaries) > 0, (
            "parallel_telemetry_summaries must be non-empty when provided."
        )
    if args.runner_reports is not None:
        assert len(args.runner_reports) > 0, "runner_reports must be non-empty when provided."
    if args.output_path is not None:
        assert args.output_path.strip() != "", "output_path must be non-empty when provided."
    return args


def parse_metric_number(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned == "":
            return None
        for separator in ("\u00c2\u00b1", "\u00b1", "+/-"):
            if separator in cleaned:
                cleaned = cleaned.split(separator, maxsplit=1)[0].strip()
                break
        try:
            number = float(cleaned)
        except ValueError:
            return None
        if math.isfinite(number):
            return number
    return None


def normalized_identity_value(value):
    number = parse_metric_number(value)
    if number is not None:
        if float(number).is_integer():
            return int(number)
        return round(number, 12)
    if isinstance(value, str):
        return value
    return str(value)


def metric_number(metrics: Dict[str, object], key: str) -> Optional[float]:
    if key not in metrics:
        return None
    return parse_metric_number(metrics[key])


def first_metric_number(metrics: Dict[str, object], keys: Tuple[str, ...]) -> Optional[float]:
    for key in keys:
        value = metric_number(metrics, key)
        if value is not None:
            return value
    return None


def infer_num_runs(sequential_metrics: Dict[str, object], parallel_metrics: Dict[str, object]) -> int:
    parallel_run_count = first_metric_number(
        parallel_metrics,
        (
            "parallel_probe_total_runs",
            "parallel_probe_num_runs",
            "num_runs",
        ),
    )
    if parallel_run_count is not None:
        return max(1, int(round(parallel_run_count)))

    if "parallel_probe_run_records" in parallel_metrics:
        records = parallel_metrics["parallel_probe_run_records"]
        if isinstance(records, list) and len(records) > 0:
            return len(records)

    sequential_run_count = first_metric_number(sequential_metrics, ("num_runs",))
    if sequential_run_count is not None:
        return max(1, int(round(sequential_run_count)))
    return 1


def timing_summary(sequential_metrics: Dict[str, object], parallel_metrics: Dict[str, object]):
    num_runs = infer_num_runs(sequential_metrics, parallel_metrics)
    sequential_per_run = first_metric_number(
        sequential_metrics,
        (
            "training_time_seconds_mean",
            "sequential_seconds_per_run",
        ),
    )
    sequential_total = first_metric_number(
        sequential_metrics,
        (
            "training_time_seconds_total",
            "sequential_seconds",
        ),
    )
    sequential_time_assumption = "explicit_total"
    if sequential_total is None:
        raw_sequential_time = metric_number(sequential_metrics, "training_time_seconds")
        if sequential_per_run is None and raw_sequential_time is not None:
            sequential_per_run = raw_sequential_time
            sequential_time_assumption = "training_time_seconds_is_per_run"
        if sequential_per_run is not None:
            sequential_total = sequential_per_run * float(num_runs)
            if sequential_time_assumption == "explicit_total":
                sequential_time_assumption = "per_run_times_num_runs"

    if sequential_per_run is None and sequential_total is not None:
        sequential_per_run = sequential_total / float(num_runs)

    parallel_total = first_metric_number(
        parallel_metrics,
        (
            "training_time_seconds",
            "parallel_seconds",
        ),
    )
    parallel_per_run = first_metric_number(
        parallel_metrics,
        (
            "parallel_probe_seconds_per_run",
            "parallel_seconds_per_run",
        ),
    )
    if parallel_per_run is None and parallel_total is not None:
        parallel_per_run = parallel_total / float(num_runs)
    if parallel_total is None and parallel_per_run is not None:
        parallel_total = parallel_per_run * float(num_runs)

    if (
            sequential_total is not None
            and parallel_total is not None
            and parallel_total > 0.0
        ):
        wall_clock_speedup = sequential_total / parallel_total
    else:
        wall_clock_speedup = None

    if (
            sequential_per_run is not None
            and parallel_per_run is not None
            and parallel_per_run > 0.0
        ):
        per_run_speedup = sequential_per_run / parallel_per_run
    else:
        per_run_speedup = None

    return {
        "num_runs": num_runs,
        "sequential_total_seconds": sequential_total,
        "parallel_total_seconds": parallel_total,
        "sequential_seconds_per_run": sequential_per_run,
        "parallel_seconds_per_run": parallel_per_run,
        "wall_clock_speedup": wall_clock_speedup,
        "per_run_speedup": per_run_speedup,
        "sequential_time_assumption": sequential_time_assumption,
    }


def _runtime_record_number(record, key: str) -> Optional[float]:
    if not isinstance(record, dict):
        return None
    if key not in record:
        return None
    return parse_metric_number(record[key])


def _runtime_record_text(record, key: str):
    if not isinstance(record, dict):
        return None
    if key not in record:
        return None
    value = record[key]
    if value is None:
        return None
    return str(value)


def parallel_group_runtime_summary(parallel_metrics: Dict[str, object]):
    if "parallel_probe_group_runtime_records" not in parallel_metrics:
        return {
            "available": False,
            "group_count": 0,
            "vectorized_group_count": 0,
            "eligible_singleton_group_count": 0,
            "total_group_runtime_seconds": None,
            "max_group_seconds_per_run": None,
            "slowest_group": None,
            "records": [],
        }

    raw_records = parallel_metrics["parallel_probe_group_runtime_records"]
    if not isinstance(raw_records, list):
        return {
            "available": False,
            "group_count": 0,
            "vectorized_group_count": 0,
            "eligible_singleton_group_count": 0,
            "total_group_runtime_seconds": None,
            "max_group_seconds_per_run": None,
            "slowest_group": None,
            "records": [],
        }

    records = []
    total_runtime = 0.0
    runtime_available = False
    vectorized_count = 0
    singleton_count = 0
    slowest_record = None
    slowest_seconds_per_run = None
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            continue
        execution_kind = _runtime_record_text(raw_record, "execution_kind")
        if execution_kind == "vectorized":
            vectorized_count += 1
        if execution_kind == "eligible_singleton":
            singleton_count += 1
        train_runtime_seconds = _runtime_record_number(raw_record, "train_runtime_seconds")
        seconds_per_run = _runtime_record_number(raw_record, "seconds_per_run")
        if train_runtime_seconds is not None:
            total_runtime += train_runtime_seconds
            runtime_available = True
        if seconds_per_run is not None:
            if slowest_seconds_per_run is None or seconds_per_run > slowest_seconds_per_run:
                slowest_seconds_per_run = seconds_per_run
                slowest_record = raw_record
        records.append(raw_record)

    if runtime_available:
        total_group_runtime_seconds = total_runtime
    else:
        total_group_runtime_seconds = None

    if slowest_record is None:
        slowest_group = None
    else:
        slowest_group = {
            "group_number": _runtime_record_number(slowest_record, "group_number"),
            "execution_kind": _runtime_record_text(slowest_record, "execution_kind"),
            "num_runs": _runtime_record_number(slowest_record, "num_runs"),
            "run_seeds": slowest_record["run_seeds"] if "run_seeds" in slowest_record else None,
            "train_runtime_seconds": _runtime_record_number(slowest_record, "train_runtime_seconds"),
            "seconds_per_run": slowest_seconds_per_run,
            "estimated_peak_bytes": _runtime_record_number(slowest_record, "estimated_peak_bytes"),
            "estimated_training_flops_per_batch": _runtime_record_number(
                slowest_record,
                "estimated_training_flops_per_batch",
            ),
        }

    return {
        "available": len(records) > 0,
        "group_count": len(records),
        "vectorized_group_count": vectorized_count,
        "eligible_singleton_group_count": singleton_count,
        "total_group_runtime_seconds": total_group_runtime_seconds,
        "max_group_seconds_per_run": slowest_seconds_per_run,
        "slowest_group": slowest_group,
        "records": records,
    }


def load_launch_manifest_payload(path_text: str):
    path = Path(path_text)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload, dict), "Launch manifest JSON must contain an object."
    if "launch_manifest" in payload:
        manifest = payload["launch_manifest"]
    else:
        manifest = payload
    assert isinstance(manifest, dict), "launch_manifest must be an object."
    assert "groups" in manifest, "launch_manifest must include groups."
    assert isinstance(manifest["groups"], list), "launch_manifest groups must be a list."
    return payload, manifest


def load_launch_manifest(path_text: str):
    payload, manifest = load_launch_manifest_payload(path_text)
    return manifest


def load_runner_report(path_text: str):
    path = Path(path_text)
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert isinstance(report, dict), f"Runner report {path} must contain an object."
    assert "plan" in report, f"Runner report {path} must include plan."
    assert "execution" in report, f"Runner report {path} must include execution."
    assert isinstance(report["plan"], dict), f"Runner report {path} plan must be an object."
    assert isinstance(report["execution"], dict), (
        f"Runner report {path} execution must be an object."
    )
    return report


def _runner_plan_count(plan, key: str) -> int:
    assert key in plan, f"Runner report plan must include {key}."
    value = plan[key]
    assert isinstance(value, int), f"Runner report plan {key} must be an integer."
    return value


def _runner_plan_list(plan, key: str):
    assert key in plan, f"Runner report plan must include {key}."
    value = plan[key]
    assert isinstance(value, list), f"Runner report plan {key} must be a list."
    return value


def _runner_execution_count(execution, key: str) -> int:
    assert key in execution, f"Runner report execution must include {key}."
    value = execution[key]
    assert isinstance(value, int), f"Runner report execution {key} must be an integer."
    return value


def _runner_execution_waves(execution):
    assert "waves" in execution, "Runner report execution must include waves."
    waves = execution["waves"]
    assert isinstance(waves, list), "Runner report execution waves must be a list."
    return waves


def runner_report_summary(paths):
    summary = {
        "provided": paths is not None,
        "report_count": 0,
        "executed_report_count": 0,
        "dry_run_report_count": 0,
        "total_plan_command_count": 0,
        "total_plan_skipped_count": 0,
        "total_executed_command_count": 0,
        "total_missing_executed_command_count": 0,
        "total_execution_failure_count": 0,
        "blocked_report_count": 0,
        "unknown_selected_command_ids": [],
        "missing_wave_command_ids": [],
        "failed_commands": [],
        "executed_command_variant_pairs": [],
        "completed_summary_command_variant_pairs": [],
        "covered_command_variant_pairs": [],
        "duplicate_executed_command_variant_pairs": [],
        "duplicate_covered_command_variant_pairs": [],
        "reports": [],
        "manifest_coverage": {
            "expectation_available": False,
            "expected_command_variant_pairs": [],
            "observed_command_variant_pairs": [],
            "missing_command_variant_pairs": [],
            "unexpected_command_variant_pairs": [],
            "duplicate_observed_command_variant_pairs": [],
            "complete": None,
        },
        "complete_execution_available": False,
        "successful": False,
    }
    if paths is None:
        return summary

    for path_text in paths:
        report = load_runner_report(path_text)
        plan = report["plan"]
        execution = report["execution"]
        path = str(Path(path_text))

        assert "execute" in plan, "Runner report plan must include execute."
        assert isinstance(plan["execute"], bool), "Runner report plan execute must be a bool."
        assert "blocked_by_over_budget" in plan, (
            "Runner report plan must include blocked_by_over_budget."
        )
        assert isinstance(plan["blocked_by_over_budget"], bool), (
            "Runner report plan blocked_by_over_budget must be a bool."
        )
        assert "executed" in execution, "Runner report execution must include executed."
        assert isinstance(execution["executed"], bool), (
            "Runner report execution executed must be a bool."
        )

        plan_command_count = _runner_plan_count(plan, "command_count")
        plan_skipped_count = _runner_plan_count(plan, "skipped_count")
        failure_count = _runner_execution_count(execution, "failure_count")
        unknown_ids = _runner_plan_list(plan, "unknown_selected_command_ids")
        missing_ids = _runner_plan_list(plan, "missing_wave_command_ids")
        waves = _runner_execution_waves(execution)
        executed_command_count = 0
        failed_commands = []
        executed_pairs = []
        completed_summary_pairs = []
        for wave in waves:
            assert isinstance(wave, dict), "Runner report execution wave must be an object."
            assert "wave_id" in wave, "Runner report execution wave must include wave_id."
            assert "commands" in wave, "Runner report execution wave must include commands."
            assert isinstance(wave["commands"], list), (
                "Runner report execution wave commands must be a list."
            )
            for command in wave["commands"]:
                assert isinstance(command, dict), (
                    "Runner report execution command must be an object."
                )
                assert "command_id" in command, (
                    "Runner report execution command must include command_id."
                )
                assert "variant" in command, (
                    "Runner report execution command must include variant."
                )
                assert isinstance(command["variant"], str), (
                    "Runner report execution command variant must be a string."
                )
                if "skipped" in command:
                    assert isinstance(command["skipped"], bool), (
                        "Runner report execution command skipped must be a bool."
                    )
                    if command["skipped"]:
                        assert "skip_reason" in command, (
                            "Runner report skipped execution command must include skip_reason."
                        )
                        if command["skip_reason"] == "completed_summary_exists":
                            completed_summary_pairs.append(
                                (command["command_id"], command["variant"])
                            )
                        continue
                executed_command_count += 1
                executed_pairs.append((command["command_id"], command["variant"]))
                if "returncode" not in command:
                    failed_commands.append(
                        {
                            "path": path,
                            "wave_id": wave["wave_id"],
                            "command_id": command["command_id"],
                            "returncode": None,
                        }
                    )
                    continue
                returncode = command["returncode"]
                assert isinstance(returncode, int), (
                    "Runner report execution command returncode must be an integer."
                )
                if returncode != 0:
                    failed_commands.append(
                        {
                            "path": path,
                            "wave_id": wave["wave_id"],
                            "command_id": command["command_id"],
                            "returncode": returncode,
                        }
                    )

        if execution["executed"]:
            summary["executed_report_count"] += 1
        else:
            summary["dry_run_report_count"] += 1
        missing_executed_command_count = 0
        if execution["executed"] and executed_command_count < plan_command_count:
            missing_executed_command_count = plan_command_count - executed_command_count

        summary["report_count"] += 1
        summary["total_plan_command_count"] += plan_command_count
        summary["total_plan_skipped_count"] += plan_skipped_count
        summary["total_executed_command_count"] += executed_command_count
        summary["total_missing_executed_command_count"] += missing_executed_command_count
        summary["total_execution_failure_count"] += failure_count
        if plan["blocked_by_over_budget"]:
            summary["blocked_report_count"] += 1
        summary["unknown_selected_command_ids"].extend(unknown_ids)
        summary["missing_wave_command_ids"].extend(missing_ids)
        summary["failed_commands"].extend(failed_commands)
        for command_id, variant in executed_pairs:
            summary["executed_command_variant_pairs"].append([command_id, variant])
            summary["covered_command_variant_pairs"].append([command_id, variant])
        for command_id, variant in completed_summary_pairs:
            summary["completed_summary_command_variant_pairs"].append([command_id, variant])
            summary["covered_command_variant_pairs"].append([command_id, variant])
        summary["reports"].append(
            {
                "path": path,
                "execute": plan["execute"],
                "executed": execution["executed"],
                "command_count": plan_command_count,
                "skipped_count": plan_skipped_count,
                "executed_command_count": executed_command_count,
                "missing_executed_command_count": missing_executed_command_count,
                "failure_count": failure_count,
                "blocked_by_over_budget": plan["blocked_by_over_budget"],
                "unknown_selected_command_ids": list(unknown_ids),
                "missing_wave_command_ids": list(missing_ids),
            }
        )

    summary["complete_execution_available"] = summary["executed_report_count"] > 0
    pair_counts = {}
    for pair in summary["executed_command_variant_pairs"]:
        pair_key = (pair[0], pair[1])
        if pair_key not in pair_counts:
            pair_counts[pair_key] = 0
        pair_counts[pair_key] += 1
    summary["duplicate_executed_command_variant_pairs"] = [
        [pair[0], pair[1]] for pair, count in sorted(pair_counts.items())
        if count > 1
    ]
    covered_pair_counts = {}
    for pair in summary["covered_command_variant_pairs"]:
        pair_key = (pair[0], pair[1])
        if pair_key not in covered_pair_counts:
            covered_pair_counts[pair_key] = 0
        covered_pair_counts[pair_key] += 1
    summary["duplicate_covered_command_variant_pairs"] = [
        [pair[0], pair[1]] for pair, count in sorted(covered_pair_counts.items())
        if count > 1
    ]
    summary["successful"] = (
        summary["complete_execution_available"]
        and summary["blocked_report_count"] == 0
        and len(summary["unknown_selected_command_ids"]) == 0
        and len(summary["missing_wave_command_ids"]) == 0
        and len(summary["failed_commands"]) == 0
        and len(summary["duplicate_covered_command_variant_pairs"]) == 0
        and summary["total_execution_failure_count"] == 0
        and summary["total_missing_executed_command_count"] == 0
    )
    return summary


def runner_manifest_coverage(runner_reports, manifest_details):
    expected_pairs = tuple(
        (pair[0], pair[1]) for pair in manifest_details["runner_command_variant_pairs"]
    )
    observed_pairs = tuple(
        (pair[0], pair[1]) for pair in runner_reports["covered_command_variant_pairs"]
        if pair[1] in ("sequential", "parallel")
    )
    duplicate_pairs = tuple(
        (pair[0], pair[1])
        for pair in runner_reports["duplicate_covered_command_variant_pairs"]
        if pair[1] in ("sequential", "parallel")
    )
    expected_set = set(expected_pairs)
    observed_set = set(observed_pairs)
    expectation_available = len(expected_pairs) > 0
    if expectation_available:
        missing_pairs = sorted(expected_set - observed_set)
        unexpected_pairs = sorted(observed_set - expected_set)
        complete = (
            len(missing_pairs) == 0
            and len(unexpected_pairs) == 0
            and len(duplicate_pairs) == 0
        )
    else:
        missing_pairs = []
        unexpected_pairs = []
        complete = None
    return {
        "expectation_available": expectation_available,
        "expected_command_variant_pairs": [
            [pair[0], pair[1]] for pair in sorted(expected_set)
        ],
        "observed_command_variant_pairs": [
            [pair[0], pair[1]] for pair in sorted(observed_set)
        ],
        "missing_command_variant_pairs": [
            [pair[0], pair[1]] for pair in missing_pairs
        ],
        "unexpected_command_variant_pairs": [
            [pair[0], pair[1]] for pair in unexpected_pairs
        ],
        "duplicate_observed_command_variant_pairs": [
            [pair[0], pair[1]] for pair in sorted(duplicate_pairs)
        ],
        "complete": complete,
    }


def command_available(group, command_key: str) -> bool:
    if command_key not in group:
        return False
    command = group[command_key]
    assert isinstance(command, list), f"Launch manifest {command_key} must be a command array."
    return len(command) > 0


def _manifest_expected_invocations_from_commands(manifest):
    sequential_invocations = 0
    parallel_invocations = 0
    for group in manifest["groups"]:
        assert isinstance(group, dict), "Each launch manifest group must be an object."
        if command_available(group, "sequential_command"):
            sequential_invocations += 1
        if command_available(group, "parallel_command"):
            parallel_invocations += 1
        elif command_available(group, "sequential_command"):
            parallel_invocations += 1
    if sequential_invocations == 0 or parallel_invocations == 0:
        speedup_ceiling = None
    else:
        speedup_ceiling = sequential_invocations / float(parallel_invocations)
    return sequential_invocations, parallel_invocations, speedup_ceiling


def manifest_probe_result_identity(group):
    identity = [group["data_name"], group["model_name"]]
    for key in PROBE_IDENTITY_KEYS:
        if key not in group:
            return None
        identity.append(normalized_identity_value(group[key]))
    return tuple(identity)


def manifest_speedup_expectation(payload, manifest):
    sequential_invocations = None
    parallel_invocations = None
    speedup_ceiling = None
    if "validation_comparison" in payload:
        validation_comparison = payload["validation_comparison"]
        assert isinstance(validation_comparison, dict), "validation_comparison must be an object."
        sequential_invocations = metric_number(
            validation_comparison,
            "sequential_trainer_invocations",
        )
        parallel_invocations = metric_number(
            validation_comparison,
            "parallel_trainer_invocations",
        )
        speedup_ceiling = metric_number(
            validation_comparison,
            "trainer_invocation_speedup_ceiling",
        )
    if speedup_ceiling is None:
        fallback_sequential, fallback_parallel, fallback_ceiling = (
            _manifest_expected_invocations_from_commands(manifest)
        )
        if sequential_invocations is None and fallback_sequential > 0:
            sequential_invocations = fallback_sequential
        if parallel_invocations is None and fallback_parallel > 0:
            parallel_invocations = fallback_parallel
        speedup_ceiling = fallback_ceiling
    return {
        "sequential_trainer_invocations": sequential_invocations,
        "parallel_trainer_invocations": parallel_invocations,
        "trainer_invocation_speedup_ceiling": speedup_ceiling,
    }


def launch_manifest_details(path_text: str):
    payload, manifest = load_launch_manifest_payload(path_text)
    group_map = {}
    manifest_result_key_set = set()
    manifest_probe_result_identity_set = set()
    expected_sequential_command_ids = []
    expected_parallel_command_ids = []
    runner_command_variant_pairs = []
    for group in manifest["groups"]:
        assert isinstance(group, dict), "Each launch manifest group must be an object."
        assert "command_id" in group, "Each launch manifest group must include command_id."
        assert "data_name" in group, "Each launch manifest group must include data_name."
        assert "model_name" in group, "Each launch manifest group must include model_name."
        command_id = group["command_id"]
        assert command_id not in group_map, f"Duplicate launch manifest command_id: {command_id}"
        group_map[command_id] = {
            "dataset": group["data_name"],
            "model": group["model_name"],
        }
        manifest_result_key_set.add((group["data_name"], group["model_name"]))
        probe_result_identity = manifest_probe_result_identity(group)
        if probe_result_identity is not None:
            manifest_probe_result_identity_set.add(probe_result_identity)
        if command_available(group, "sequential_monitor_command"):
            expected_sequential_command_ids.append(command_id)
        if command_available(group, "parallel_monitor_command"):
            expected_parallel_command_ids.append(command_id)
        if command_available(group, "sequential_command"):
            runner_command_variant_pairs.append([command_id, "sequential"])
        if command_available(group, "parallel_command"):
            runner_command_variant_pairs.append([command_id, "parallel"])
    return {
        "group_map": group_map,
        "result_keys": [
            list(key) for key in sorted(manifest_result_key_set)
        ],
        "probe_result_identities": [
            list(identity) for identity in sorted(manifest_probe_result_identity_set)
        ],
        "expected_sequential_command_ids": expected_sequential_command_ids,
        "expected_parallel_command_ids": expected_parallel_command_ids,
        "runner_command_variant_pairs": runner_command_variant_pairs,
        "speedup_expectation": manifest_speedup_expectation(payload, manifest),
    }


def launch_manifest_group_map(path_text: str):
    details = launch_manifest_details(path_text)
    return details["group_map"]


def telemetry_coverage_report(expected_command_ids, records):
    observed_command_ids = [record["command_id"] for record in records]
    expected_set = set(expected_command_ids)
    observed_set = set(observed_command_ids)
    counts = {}
    for command_id in observed_command_ids:
        if command_id not in counts:
            counts[command_id] = 0
        counts[command_id] += 1
    duplicate_observed_ids = sorted(
        command_id for command_id, count in counts.items()
        if count > 1
    )
    expectation_available = len(expected_command_ids) > 0
    missing_ids = sorted(expected_set - observed_set)
    if expectation_available:
        unexpected_ids = sorted(observed_set - expected_set)
        complete = (
            len(missing_ids) == 0
            and len(unexpected_ids) == 0
            and len(duplicate_observed_ids) == 0
        )
    else:
        unexpected_ids = []
        complete = None
    return {
        "expectation_available": expectation_available,
        "expected_command_ids": list(expected_command_ids),
        "observed_command_ids": observed_command_ids,
        "missing_command_ids": missing_ids,
        "unexpected_command_ids": unexpected_ids,
        "duplicate_observed_command_ids": duplicate_observed_ids,
        "complete": complete,
    }


def telemetry_identity_from_path(path: Path):
    name = path.name
    assert name.endswith(".json"), f"Telemetry summary path must end in .json: {path}"
    trimmed = name[:-5]
    if trimmed.endswith(".summary"):
        trimmed = trimmed[:-8]
    separator_index = trimmed.rfind("_")
    assert separator_index > 0, (
        "Telemetry summary filenames must end with _sequential or _parallel before .summary.json."
    )
    command_id = trimmed[:separator_index]
    variant = trimmed[separator_index + 1:]
    assert variant in ("sequential", "parallel"), f"Unsupported telemetry summary variant: {variant}"
    return command_id, variant


def load_telemetry_summary(path_text: str):
    path = Path(path_text)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload, dict), f"Telemetry summary {path} must contain an object."
    return payload


def numeric_summary_value(summary: Dict[str, object], key: str) -> Optional[float]:
    if key not in summary:
        return None
    value = summary[key]
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def telemetry_records_for(paths, manifest_group_map, expected_variant: str):
    records = []
    if paths is None:
        return records
    for path_text in paths:
        path = Path(path_text)
        command_id, variant = telemetry_identity_from_path(path)
        assert variant == expected_variant, (
            f"Expected {expected_variant} telemetry summary, got {variant} for {path}."
        )
        assert command_id in manifest_group_map, (
            f"Telemetry summary command_id {command_id} is not present in launch_manifest."
        )
        group_info = manifest_group_map[command_id]
        summary = load_telemetry_summary(path_text)
        records.append(
            {
                "command_id": command_id,
                "variant": variant,
                "dataset": group_info["dataset"],
                "model": group_info["model"],
                "path": str(path),
                "summary": summary,
            }
        )
    return records


def weighted_mean_from_records(records, key: str):
    numerator = 0.0
    denominator = 0.0
    for record in records:
        summary = record["summary"]
        value = numeric_summary_value(summary, key)
        if value is None:
            continue
        sample_count = numeric_summary_value(summary, "sample_count")
        if sample_count is None or sample_count <= 0.0:
            weight = 1.0
        else:
            weight = sample_count
        numerator += value * weight
        denominator += weight
    if denominator == 0.0:
        return None
    return numerator / denominator


def max_from_records(records, key: str):
    values = []
    for record in records:
        value = numeric_summary_value(record["summary"], key)
        if value is not None:
            values.append(value)
    if len(values) == 0:
        return None
    return max(values)


def sum_from_records(records, key: str):
    total = 0.0
    found = False
    for record in records:
        value = numeric_summary_value(record["summary"], key)
        if value is not None:
            total += value
            found = True
    if not found:
        return None
    return total


def aggregate_telemetry_records(records):
    grouped = {}
    for record in records:
        key = (record["dataset"], record["model"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(record)

    aggregated = {}
    for key, group_records in grouped.items():
        aggregated[key] = {
            "summary_count": len(group_records),
            "command_ids": [record["command_id"] for record in group_records],
            "sample_count": sum_from_records(group_records, "sample_count"),
            "duration_seconds": sum_from_records(group_records, "duration_seconds"),
            "gpu_utilization_percent_mean": weighted_mean_from_records(
                group_records,
                "gpu_utilization_percent_mean",
            ),
            "gpu_utilization_percent_max": max_from_records(group_records, "gpu_utilization_percent_max"),
            "memory_utilization_percent_mean": weighted_mean_from_records(
                group_records,
                "memory_utilization_percent_mean",
            ),
            "memory_utilization_percent_max": max_from_records(group_records, "memory_utilization_percent_max"),
            "memory_used_mib_max": max_from_records(group_records, "memory_used_mib_max"),
            "memory_used_fraction_max": max_from_records(group_records, "memory_used_fraction_max"),
            "power_draw_watts_mean": weighted_mean_from_records(group_records, "power_draw_watts_mean"),
            "power_draw_watts_max": max_from_records(group_records, "power_draw_watts_max"),
            "sm_clock_mhz_mean": weighted_mean_from_records(group_records, "sm_clock_mhz_mean"),
            "sm_clock_mhz_max": max_from_records(group_records, "sm_clock_mhz_max"),
        }
    return aggregated


def telemetry_ratio(parallel_value, sequential_value):
    if sequential_value is None or parallel_value is None or sequential_value == 0.0:
        return None
    return parallel_value / sequential_value


def telemetry_delta(parallel_value, sequential_value):
    if sequential_value is None or parallel_value is None:
        return None
    return parallel_value - sequential_value


def telemetry_comparison_for_pair(key, sequential_telemetry, parallel_telemetry):
    if key not in sequential_telemetry and key not in parallel_telemetry:
        return None
    if key in sequential_telemetry:
        sequential_summary = sequential_telemetry[key]
    else:
        sequential_summary = None
    if key in parallel_telemetry:
        parallel_summary = parallel_telemetry[key]
    else:
        parallel_summary = None
    metric_keys = (
        "gpu_utilization_percent_mean",
        "gpu_utilization_percent_max",
        "memory_utilization_percent_mean",
        "memory_utilization_percent_max",
        "memory_used_mib_max",
        "memory_used_fraction_max",
        "power_draw_watts_mean",
        "power_draw_watts_max",
        "sm_clock_mhz_mean",
        "sm_clock_mhz_max",
        "duration_seconds",
    )
    metric_comparisons = []
    for metric_key in metric_keys:
        if sequential_summary is not None and metric_key in sequential_summary:
            sequential_value = sequential_summary[metric_key]
        else:
            sequential_value = None
        if parallel_summary is not None and metric_key in parallel_summary:
            parallel_value = parallel_summary[metric_key]
        else:
            parallel_value = None
        metric_comparisons.append(
            {
                "metric": metric_key,
                "sequential_value": sequential_value,
                "parallel_value": parallel_value,
                "delta": telemetry_delta(parallel_value, sequential_value),
                "ratio": telemetry_ratio(parallel_value, sequential_value),
            }
        )
    return {
        "sequential": sequential_summary,
        "parallel": parallel_summary,
        "metric_comparisons": metric_comparisons,
    }


def metric_key_set(args, sequential_metrics: Dict[str, object], parallel_metrics: Dict[str, object]):
    if args.metric_keys is not None:
        return tuple(args.metric_keys)
    keys = []
    for key in DEFAULT_METRIC_KEYS:
        if key in sequential_metrics and key in parallel_metrics:
            keys.append(key)
    return tuple(keys)


def compare_metrics(args, sequential_metrics: Dict[str, object], parallel_metrics: Dict[str, object]):
    comparisons = []
    for key in metric_key_set(args, sequential_metrics, parallel_metrics):
        sequential_value = metric_number(sequential_metrics, key)
        parallel_value = metric_number(parallel_metrics, key)
        if sequential_value is None or parallel_value is None:
            comparisons.append(
                {
                    "metric": key,
                    "sequential_value": sequential_value,
                    "parallel_value": parallel_value,
                    "comparable": False,
                    "within_tolerance": None,
                }
            )
            continue
        abs_diff = abs(parallel_value - sequential_value)
        denominator = max(abs(sequential_value), 1e-12)
        rel_diff = abs_diff / denominator
        within_tolerance = (
            abs_diff <= args.metric_abs_tolerance
            or rel_diff <= args.metric_rel_tolerance
        )
        comparisons.append(
            {
                "metric": key,
                "sequential_value": sequential_value,
                "parallel_value": parallel_value,
                "absolute_difference": abs_diff,
                "relative_difference": rel_diff,
                "comparable": True,
                "within_tolerance": within_tolerance,
            }
        )
    return comparisons


def ensemble_metric_key_pairs(sequential_metrics: Dict[str, object], parallel_metrics: Dict[str, object]):
    pairs = []
    for suffix in DEFAULT_ENSEMBLE_METRIC_SUFFIXES:
        sequential_key = f"sequential_probe_ensemble_{suffix}"
        parallel_key = f"parallel_probe_ensemble_{suffix}"
        if sequential_key in sequential_metrics and parallel_key in parallel_metrics:
            pairs.append((suffix, sequential_key, parallel_key))
    return tuple(pairs)


def compare_ensemble_metrics(args, sequential_metrics: Dict[str, object], parallel_metrics: Dict[str, object]):
    comparisons = []
    for suffix, sequential_key, parallel_key in ensemble_metric_key_pairs(sequential_metrics, parallel_metrics):
        sequential_value = metric_number(sequential_metrics, sequential_key)
        parallel_value = metric_number(parallel_metrics, parallel_key)
        if sequential_value is None or parallel_value is None:
            comparisons.append(
                {
                    "metric": suffix,
                    "sequential_metric": sequential_key,
                    "parallel_metric": parallel_key,
                    "sequential_value": sequential_value,
                    "parallel_value": parallel_value,
                    "comparable": False,
                    "within_tolerance": None,
                }
            )
            continue
        abs_diff = abs(parallel_value - sequential_value)
        denominator = max(abs(sequential_value), 1e-12)
        rel_diff = abs_diff / denominator
        within_tolerance = (
            abs_diff <= args.metric_abs_tolerance
            or rel_diff <= args.metric_rel_tolerance
        )
        comparisons.append(
            {
                "metric": suffix,
                "sequential_metric": sequential_key,
                "parallel_metric": parallel_key,
                "sequential_value": sequential_value,
                "parallel_value": parallel_value,
                "absolute_difference": abs_diff,
                "relative_difference": rel_diff,
                "comparable": True,
                "within_tolerance": within_tolerance,
            }
        )
    return comparisons


def parse_tsv_records(path: Path) -> List[MetricsRecord]:
    records: List[MetricsRecord] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = list(reader)
    if len(rows) == 0:
        return records
    header = rows[0]
    assert len(header) >= 2, f"TSV results file {path} must contain dataset and model columns."
    model_names = tuple(header[1:])
    for row in rows[1:]:
        if len(row) == 0:
            continue
        dataset = row[0]
        for model_index, model in enumerate(model_names):
            cell_index = model_index + 1
            if cell_index >= len(row):
                continue
            cell = row[cell_index].strip()
            if cell == "":
                continue
            metrics = json.loads(cell)
            assert isinstance(metrics, dict), f"TSV cell for {dataset}/{model} must contain a JSON object."
            records.append(
                MetricsRecord(
                    source_path=str(path),
                    dataset=dataset,
                    model=model,
                    metrics=metrics,
                )
            )
    return records


def _record_from_dict(path: Path, item: Dict[str, object], index: int) -> MetricsRecord:
    dataset = f"record-{index}"
    model = "model"
    if "dataset" in item and isinstance(item["dataset"], str):
        dataset = item["dataset"]
    if "data_name" in item and isinstance(item["data_name"], str):
        dataset = item["data_name"]
    if "model" in item and isinstance(item["model"], str):
        model = item["model"]
    if "model_name" in item and isinstance(item["model_name"], str):
        model = item["model_name"]

    if "metrics" in item and isinstance(item["metrics"], dict):
        metrics = dict(item["metrics"])
        for key in PROBE_IDENTITY_KEYS:
            if key in item and key not in metrics:
                metrics[key] = item[key]
    else:
        metrics = item
    return MetricsRecord(
        source_path=str(path),
        dataset=dataset,
        model=model,
        metrics=metrics,
    )


def parse_json_records(path: Path) -> List[MetricsRecord]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        records = []
        for index, item in enumerate(payload):
            assert isinstance(item, dict), f"JSON list item {index} in {path} must be an object."
            records.append(_record_from_dict(path, item, index))
        return records
    assert isinstance(payload, dict), f"JSON result file {path} must contain an object or list."
    if "records" in payload:
        record_items = payload["records"]
        assert isinstance(record_items, list), f"records in {path} must be a list."
        records = []
        for index, item in enumerate(record_items):
            assert isinstance(item, dict), f"records item {index} in {path} must be an object."
            records.append(_record_from_dict(path, item, index))
        return records
    return [_record_from_dict(path, payload, 0)]


def load_result_records(paths: List[str]) -> List[MetricsRecord]:
    records: List[MetricsRecord] = []
    for path_text in paths:
        path = Path(path_text)
        suffix = path.suffix.lower()
        if suffix == ".tsv":
            records.extend(parse_tsv_records(path))
        elif suffix == ".json":
            records.extend(parse_json_records(path))
        else:
            raise AssertionError(f"Unsupported result file extension for {path}. Use .tsv or .json.")
    return records


def record_map(records: List[MetricsRecord]) -> Dict[Tuple[object, ...], MetricsRecord]:
    mapped: Dict[Tuple[str, ...], MetricsRecord] = {}
    for record in records:
        mapped[result_record_identity(record)] = record
    return mapped


def record_probe_result_identity(record: MetricsRecord):
    identity = [record.dataset, record.model]
    for key in PROBE_IDENTITY_KEYS:
        if key not in record.metrics:
            return None
        identity.append(normalized_identity_value(record.metrics[key]))
    return tuple(identity)


def result_record_identity(record: MetricsRecord):
    probe_identity = record_probe_result_identity(record)
    if probe_identity is not None:
        return probe_identity
    return record.key


def compare_record_pair(
        args,
        sequential_record: MetricsRecord,
        parallel_record: MetricsRecord,
        sequential_telemetry,
        parallel_telemetry,
    ):
    timing = timing_summary(sequential_record.metrics, parallel_record.metrics)
    group_runtime = parallel_group_runtime_summary(parallel_record.metrics)
    metrics = compare_metrics(args, sequential_record.metrics, parallel_record.metrics)
    comparable_metrics = [item for item in metrics if item["comparable"]]
    failing_metrics = [
        item for item in comparable_metrics
        if item["within_tolerance"] is False
    ]
    ensemble_metrics = compare_ensemble_metrics(args, sequential_record.metrics, parallel_record.metrics)
    comparable_ensemble_metrics = [item for item in ensemble_metrics if item["comparable"]]
    failing_ensemble_metrics = [
        item for item in comparable_ensemble_metrics
        if item["within_tolerance"] is False
    ]
    parallel_invocations = first_metric_number(
        parallel_record.metrics,
        (
            "parallel_probe_trainer_invocations",
            "parallel_trainer_invocations",
        ),
    )
    invocation_reduction = first_metric_number(
        parallel_record.metrics,
        (
            "parallel_probe_invocation_reduction",
            "trainer_invocation_reduction",
        ),
    )
    compression_ratio = first_metric_number(
        parallel_record.metrics,
        (
            "parallel_probe_compression_ratio",
            "trainer_invocation_speedup_ceiling",
        ),
    )
    telemetry_comparison = telemetry_comparison_for_pair(
        sequential_record.key,
        sequential_telemetry,
        parallel_telemetry,
    )
    return {
        "dataset": sequential_record.dataset,
        "model": sequential_record.model,
        "result_identity": list(result_record_identity(sequential_record)),
        "sequential_source": sequential_record.source_path,
        "parallel_source": parallel_record.source_path,
        "timing": timing,
        "parallel_group_runtime": group_runtime,
        "parallel_trainer_invocations": parallel_invocations,
        "parallel_invocation_reduction": invocation_reduction,
        "parallel_compression_ratio": compression_ratio,
        "metric_comparisons": metrics,
        "comparable_metric_count": len(comparable_metrics),
        "failing_metric_count": len(failing_metrics),
        "ensemble_metric_comparisons": ensemble_metrics,
        "comparable_ensemble_metric_count": len(comparable_ensemble_metrics),
        "failing_ensemble_metric_count": len(failing_ensemble_metrics),
        "telemetry_comparison": telemetry_comparison,
    }


def validation_requirement(
        name: str,
        status: str,
        observed_value=None,
        threshold_value=None,
        comparator: Optional[str] = None,
        detail: Optional[str] = None,
    ):
    assert status in ("pass", "fail", "missing", "not_required"), (
        f"Unsupported validation requirement status: {status}"
    )
    return {
        "name": name,
        "status": status,
        "observed_value": observed_value,
        "threshold_value": threshold_value,
        "comparator": comparator,
        "detail": detail,
    }


def mean_pair_timing(pair_reports, timing_key: str):
    values = []
    for pair_report in pair_reports:
        value = pair_report["timing"][timing_key]
        if value is not None:
            values.append(value)
    if len(values) == 0:
        return None
    return sum(values) / float(len(values))


def mean_pair_telemetry(pair_reports, variant: str, telemetry_key: str):
    values = []
    for pair_report in pair_reports:
        telemetry_comparison = pair_report["telemetry_comparison"]
        if telemetry_comparison is None:
            continue
        summary = telemetry_comparison[variant]
        if summary is None:
            continue
        if telemetry_key not in summary:
            continue
        value = parse_metric_number(summary[telemetry_key])
        if value is not None:
            values.append(value)
    if len(values) == 0:
        return None
    return sum(values) / float(len(values))


def aggregate_parallel_group_runtime(pair_reports):
    available_pairs = []
    total_runtime = 0.0
    runtime_available = False
    total_group_count = 0
    total_vectorized_group_count = 0
    total_singleton_group_count = 0
    slowest_pair = None
    slowest_seconds_per_run = None
    for pair_report in pair_reports:
        group_runtime = pair_report["parallel_group_runtime"]
        if not group_runtime["available"]:
            continue
        available_pairs.append(pair_report)
        total_group_count += group_runtime["group_count"]
        total_vectorized_group_count += group_runtime["vectorized_group_count"]
        total_singleton_group_count += group_runtime["eligible_singleton_group_count"]
        total_group_runtime_seconds = group_runtime["total_group_runtime_seconds"]
        if total_group_runtime_seconds is not None:
            total_runtime += total_group_runtime_seconds
            runtime_available = True
        max_group_seconds_per_run = group_runtime["max_group_seconds_per_run"]
        if max_group_seconds_per_run is not None:
            if (
                    slowest_seconds_per_run is None
                    or max_group_seconds_per_run > slowest_seconds_per_run
                ):
                slowest_seconds_per_run = max_group_seconds_per_run
                slowest_pair = pair_report

    if runtime_available:
        total_group_runtime_seconds = total_runtime
    else:
        total_group_runtime_seconds = None

    if slowest_pair is None:
        slowest_group = None
    else:
        slowest_group = dict(slowest_pair["parallel_group_runtime"]["slowest_group"])
        slowest_group["dataset"] = slowest_pair["dataset"]
        slowest_group["model"] = slowest_pair["model"]
        slowest_group["result_identity"] = slowest_pair["result_identity"]

    return {
        "available_pair_count": len(available_pairs),
        "total_group_count": total_group_count,
        "total_vectorized_group_count": total_vectorized_group_count,
        "total_eligible_singleton_group_count": total_singleton_group_count,
        "total_group_runtime_seconds": total_group_runtime_seconds,
        "max_group_seconds_per_run": slowest_seconds_per_run,
        "slowest_group": slowest_group,
    }


def telemetry_coverage_requirement(telemetry_coverage):
    sequential_complete = telemetry_coverage["sequential"]["complete"]
    parallel_complete = telemetry_coverage["parallel"]["complete"]
    if sequential_complete is None or parallel_complete is None:
        return validation_requirement(
            "telemetry_coverage",
            "missing",
            observed_value={
                "sequential_complete": sequential_complete,
                "parallel_complete": parallel_complete,
            },
            threshold_value="complete",
            comparator="both_complete",
            detail="Telemetry coverage can only be required when launch-manifest expectations exist.",
        )
    status = "pass" if sequential_complete and parallel_complete else "fail"
    return validation_requirement(
        "telemetry_coverage",
        status,
        observed_value={
            "sequential_complete": sequential_complete,
            "parallel_complete": parallel_complete,
            "sequential_missing_command_ids": telemetry_coverage["sequential"]["missing_command_ids"],
            "parallel_missing_command_ids": telemetry_coverage["parallel"]["missing_command_ids"],
        },
        threshold_value="complete",
        comparator="both_complete",
    )


def manifest_result_coverage_requirement(summary):
    if not summary["manifest_result_expectation_available"]:
        return validation_requirement(
            "manifest_result_coverage",
            "missing",
            observed_value={
                "manifest_result_key_count": 0,
            },
            threshold_value="all_manifest_result_keys_matched",
            comparator="all_present",
            detail="Manifest result coverage requires a launch manifest with planned groups.",
        )
    missing_keys = summary["manifest_missing_result_keys"]
    status = "pass" if len(missing_keys) == 0 else "fail"
    return validation_requirement(
        "manifest_result_coverage",
        status,
        observed_value={
            "manifest_result_key_count": summary["manifest_result_key_count"],
            "manifest_missing_result_keys": missing_keys,
            "manifest_unexpected_result_keys": summary["manifest_unexpected_result_keys"],
        },
        threshold_value="all_manifest_result_keys_matched",
        comparator="all_present",
    )


def manifest_probe_result_coverage_requirement(summary):
    if not summary["manifest_probe_result_expectation_available"]:
        return validation_requirement(
            "manifest_probe_result_coverage",
            "missing",
            observed_value={
                "manifest_probe_result_identity_count": 0,
            },
            threshold_value="all_manifest_probe_result_identities_matched",
            comparator="all_present",
            detail="Probe result coverage requires a launch manifest with probe identity fields.",
        )
    if not summary["observed_probe_result_identity_available"]:
        return validation_requirement(
            "manifest_probe_result_coverage",
            "missing",
            observed_value={
                "observed_probe_result_identity_available": False,
                "manifest_probe_result_identity_count": (
                    summary["manifest_probe_result_identity_count"]
                ),
            },
            threshold_value="result_records_include_probe_identity_fields",
            comparator="identity_available",
            detail=(
                "Exact probe result coverage requires result records to include "
                "probe_type, hidden_size, dropout, n_layers, task_type, and num_labels."
            ),
        )
    missing_identities = summary["manifest_missing_probe_result_identities"]
    status = "pass" if len(missing_identities) == 0 else "fail"
    return validation_requirement(
        "manifest_probe_result_coverage",
        status,
        observed_value={
            "manifest_probe_result_identity_count": summary["manifest_probe_result_identity_count"],
            "manifest_missing_probe_result_identities": missing_identities,
            "manifest_unexpected_probe_result_identities": (
                summary["manifest_unexpected_probe_result_identities"]
            ),
        },
        threshold_value="all_manifest_probe_result_identities_matched",
        comparator="all_present",
    )


def runner_reports_requirement(summary):
    runner_reports = summary["runner_reports"]
    if not runner_reports["provided"]:
        return validation_requirement(
            "runner_reports",
            "missing",
            observed_value={
                "report_count": 0,
                "executed_report_count": 0,
            },
            threshold_value="successful_execution_report",
            comparator="all_clean",
            detail="Runner report validation requires at least one saved execution report.",
        )
    if not runner_reports["complete_execution_available"]:
        return validation_requirement(
            "runner_reports",
            "missing",
            observed_value={
                "report_count": runner_reports["report_count"],
                "dry_run_report_count": runner_reports["dry_run_report_count"],
                "executed_report_count": runner_reports["executed_report_count"],
            },
            threshold_value="successful_execution_report",
            comparator="all_clean",
            detail="Dry-run reports are summarized but do not prove launch execution succeeded.",
        )
    status = "pass" if runner_reports["successful"] else "fail"
    manifest_coverage = runner_reports["manifest_coverage"]
    if (
            status == "pass"
            and manifest_coverage["expectation_available"]
            and manifest_coverage["complete"] is not True
        ):
        status = "fail"
    return validation_requirement(
        "runner_reports",
        status,
        observed_value={
            "report_count": runner_reports["report_count"],
            "executed_report_count": runner_reports["executed_report_count"],
            "blocked_report_count": runner_reports["blocked_report_count"],
            "total_execution_failure_count": (
                runner_reports["total_execution_failure_count"]
            ),
            "total_missing_executed_command_count": (
                runner_reports["total_missing_executed_command_count"]
            ),
            "failed_commands": runner_reports["failed_commands"],
            "unknown_selected_command_ids": (
                runner_reports["unknown_selected_command_ids"]
            ),
            "missing_wave_command_ids": runner_reports["missing_wave_command_ids"],
            "manifest_coverage": manifest_coverage,
        },
        threshold_value="successful_execution_report",
        comparator="all_clean",
    )


def build_validation_verdict(args, summary, pair_reports):
    requirements = []
    missing_result_pair_count = (
        len(summary["missing_parallel_keys"])
        + len(summary["missing_sequential_keys"])
    )
    if summary["matched_pair_count"] == 0:
        result_pair_status = "missing"
    elif missing_result_pair_count == 0:
        result_pair_status = "pass"
    else:
        result_pair_status = "fail"
    requirements.append(
        validation_requirement(
            "result_pair_coverage",
            result_pair_status,
            observed_value={
                "matched_pair_count": summary["matched_pair_count"],
                "missing_result_pair_count": missing_result_pair_count,
            },
            threshold_value="all_pairs_matched",
            comparator="equals",
        )
    )

    if args.require_manifest_result_coverage:
        requirements.append(
            manifest_result_coverage_requirement(summary)
        )
    else:
        requirements.append(
            validation_requirement(
                "manifest_result_coverage",
                "not_required",
                observed_value={
                    "manifest_result_key_count": summary["manifest_result_key_count"],
                    "manifest_missing_result_keys": summary["manifest_missing_result_keys"],
                    "manifest_unexpected_result_keys": summary["manifest_unexpected_result_keys"],
                },
            )
        )

    if args.require_manifest_probe_result_coverage:
        requirements.append(
            manifest_probe_result_coverage_requirement(summary)
        )
    else:
        requirements.append(
            validation_requirement(
                "manifest_probe_result_coverage",
                "not_required",
                observed_value={
                    "manifest_probe_result_identity_count": (
                        summary["manifest_probe_result_identity_count"]
                    ),
                    "observed_probe_result_identity_available": (
                        summary["observed_probe_result_identity_available"]
                    ),
                    "manifest_missing_probe_result_identities": (
                        summary["manifest_missing_probe_result_identities"]
                    ),
                    "manifest_unexpected_probe_result_identities": (
                        summary["manifest_unexpected_probe_result_identities"]
                    ),
                },
            )
        )

    if summary["total_comparable_metric_count"] == 0:
        metric_status = "missing"
    elif summary["total_failing_metric_count"] <= args.max_failing_metric_count:
        metric_status = "pass"
    else:
        metric_status = "fail"
    requirements.append(
        validation_requirement(
            "metric_parity",
            metric_status,
            observed_value={
                "comparable_metric_count": summary["total_comparable_metric_count"],
                "failing_metric_count": summary["total_failing_metric_count"],
            },
            threshold_value=args.max_failing_metric_count,
            comparator="failing_count_lte",
        )
    )

    if summary["total_comparable_ensemble_metric_count"] == 0 and not args.require_ensemble_metrics:
        ensemble_status = "not_required"
    elif summary["total_comparable_ensemble_metric_count"] == 0:
        ensemble_status = "missing"
    elif summary["total_failing_ensemble_metric_count"] <= args.max_failing_ensemble_metric_count:
        ensemble_status = "pass"
    else:
        ensemble_status = "fail"
    requirements.append(
        validation_requirement(
            "ensemble_metric_parity",
            ensemble_status,
            observed_value={
                "comparable_ensemble_metric_count": summary["total_comparable_ensemble_metric_count"],
                "failing_ensemble_metric_count": summary["total_failing_ensemble_metric_count"],
            },
            threshold_value=args.max_failing_ensemble_metric_count,
            comparator="failing_count_lte",
        )
    )

    if args.min_wall_clock_speedup > 0.0:
        wall_clock_speedup = summary["mean_wall_clock_speedup"]
        if wall_clock_speedup is None:
            speedup_status = "missing"
        elif wall_clock_speedup >= args.min_wall_clock_speedup:
            speedup_status = "pass"
        else:
            speedup_status = "fail"
        requirements.append(
            validation_requirement(
                "wall_clock_speedup",
                speedup_status,
                observed_value=wall_clock_speedup,
                threshold_value=args.min_wall_clock_speedup,
                comparator="gte",
            )
        )
    else:
        requirements.append(
            validation_requirement("wall_clock_speedup", "not_required")
        )

    mean_per_run_speedup = mean_pair_timing(pair_reports, "per_run_speedup")
    if args.min_per_run_speedup > 0.0:
        if mean_per_run_speedup is None:
            per_run_status = "missing"
        elif mean_per_run_speedup >= args.min_per_run_speedup:
            per_run_status = "pass"
        else:
            per_run_status = "fail"
        requirements.append(
            validation_requirement(
                "per_run_speedup",
                per_run_status,
                observed_value=mean_per_run_speedup,
                threshold_value=args.min_per_run_speedup,
                comparator="gte",
            )
        )
    else:
        requirements.append(
            validation_requirement(
                "per_run_speedup",
                "not_required",
                observed_value=mean_per_run_speedup,
            )
        )

    manifest_speedup_efficiency = summary["manifest_speedup_efficiency"]
    if args.min_manifest_speedup_efficiency is not None:
        if manifest_speedup_efficiency is None:
            manifest_speedup_status = "missing"
        elif manifest_speedup_efficiency >= args.min_manifest_speedup_efficiency:
            manifest_speedup_status = "pass"
        else:
            manifest_speedup_status = "fail"
        requirements.append(
            validation_requirement(
                "manifest_speedup_efficiency",
                manifest_speedup_status,
                observed_value=manifest_speedup_efficiency,
                threshold_value=args.min_manifest_speedup_efficiency,
                comparator="gte",
                detail=(
                    "mean_wall_clock_speedup divided by the preflight "
                    "trainer_invocation_speedup_ceiling"
                ),
            )
        )
    else:
        requirements.append(
            validation_requirement(
                "manifest_speedup_efficiency",
                "not_required",
                observed_value=manifest_speedup_efficiency,
            )
        )

    if args.require_complete_telemetry:
        requirements.append(
            telemetry_coverage_requirement(summary["telemetry_coverage"])
        )
    else:
        requirements.append(
            validation_requirement("telemetry_coverage", "not_required")
        )

    if args.require_successful_runner_reports:
        requirements.append(
            runner_reports_requirement(summary)
        )
    else:
        requirements.append(
            validation_requirement(
                "runner_reports",
                "not_required",
                observed_value={
                    "provided": summary["runner_reports"]["provided"],
                    "report_count": summary["runner_reports"]["report_count"],
                    "executed_report_count": (
                        summary["runner_reports"]["executed_report_count"]
                    ),
                    "successful": summary["runner_reports"]["successful"],
                },
            )
        )

    parallel_gpu_utilization = mean_pair_telemetry(
        pair_reports,
        "parallel",
        "gpu_utilization_percent_mean",
    )
    if args.min_parallel_gpu_utilization_percent is not None:
        if parallel_gpu_utilization is None:
            utilization_status = "missing"
        elif parallel_gpu_utilization >= args.min_parallel_gpu_utilization_percent:
            utilization_status = "pass"
        else:
            utilization_status = "fail"
        requirements.append(
            validation_requirement(
                "parallel_gpu_utilization",
                utilization_status,
                observed_value=parallel_gpu_utilization,
                threshold_value=args.min_parallel_gpu_utilization_percent,
                comparator="gte",
            )
        )
    else:
        requirements.append(
            validation_requirement(
                "parallel_gpu_utilization",
                "not_required",
                observed_value=parallel_gpu_utilization,
            )
        )

    sequential_gpu_utilization = mean_pair_telemetry(
        pair_reports,
        "sequential",
        "gpu_utilization_percent_mean",
    )
    if sequential_gpu_utilization is None or parallel_gpu_utilization is None:
        gpu_utilization_gain = None
    else:
        gpu_utilization_gain = parallel_gpu_utilization - sequential_gpu_utilization
    if args.min_gpu_utilization_gain_percent is not None:
        if gpu_utilization_gain is None:
            gain_status = "missing"
        elif gpu_utilization_gain >= args.min_gpu_utilization_gain_percent:
            gain_status = "pass"
        else:
            gain_status = "fail"
        requirements.append(
            validation_requirement(
                "gpu_utilization_gain",
                gain_status,
                observed_value=gpu_utilization_gain,
                threshold_value=args.min_gpu_utilization_gain_percent,
                comparator="gte",
            )
        )
    else:
        requirements.append(
            validation_requirement(
                "gpu_utilization_gain",
                "not_required",
                observed_value=gpu_utilization_gain,
            )
        )

    failing_requirements = [
        requirement["name"] for requirement in requirements
        if requirement["status"] == "fail"
    ]
    missing_requirements = [
        requirement["name"] for requirement in requirements
        if requirement["status"] == "missing"
    ]
    not_required_requirements = [
        requirement["name"] for requirement in requirements
        if requirement["status"] == "not_required"
    ]
    if len(failing_requirements) > 0:
        verdict_status = "fail"
    elif len(missing_requirements) > 0:
        verdict_status = "incomplete"
    else:
        verdict_status = "pass"
    return {
        "status": verdict_status,
        "requirements": requirements,
        "failing_requirements": failing_requirements,
        "missing_requirements": missing_requirements,
        "not_required_requirements": not_required_requirements,
        "thresholds": {
            "metric_abs_tolerance": args.metric_abs_tolerance,
            "metric_rel_tolerance": args.metric_rel_tolerance,
            "min_wall_clock_speedup": args.min_wall_clock_speedup,
            "min_per_run_speedup": args.min_per_run_speedup,
            "min_manifest_speedup_efficiency": args.min_manifest_speedup_efficiency,
            "max_failing_metric_count": args.max_failing_metric_count,
            "max_failing_ensemble_metric_count": args.max_failing_ensemble_metric_count,
            "require_ensemble_metrics": args.require_ensemble_metrics,
            "require_manifest_result_coverage": args.require_manifest_result_coverage,
            "require_manifest_probe_result_coverage": args.require_manifest_probe_result_coverage,
            "require_complete_telemetry": args.require_complete_telemetry,
            "require_successful_runner_reports": args.require_successful_runner_reports,
            "min_parallel_gpu_utilization_percent": args.min_parallel_gpu_utilization_percent,
            "min_gpu_utilization_gain_percent": args.min_gpu_utilization_gain_percent,
        },
    }


def build_comparison_report(args):
    sequential_records = load_result_records(args.sequential_results)
    parallel_records = load_result_records(args.parallel_results)
    runner_reports = runner_report_summary(args.runner_reports)
    if args.launch_manifest is not None:
        manifest_details = launch_manifest_details(args.launch_manifest)
    else:
        manifest_details = {
            "group_map": {},
            "result_keys": [],
            "probe_result_identities": [],
            "expected_sequential_command_ids": [],
            "expected_parallel_command_ids": [],
            "runner_command_variant_pairs": [],
            "speedup_expectation": {
                "sequential_trainer_invocations": None,
                "parallel_trainer_invocations": None,
                "trainer_invocation_speedup_ceiling": None,
            },
        }
    runner_reports["manifest_coverage"] = runner_manifest_coverage(
        runner_reports,
        manifest_details,
    )
    if (
            runner_reports["manifest_coverage"]["expectation_available"]
            and runner_reports["manifest_coverage"]["complete"] is not True
        ):
        runner_reports["successful"] = False
    manifest_group_map = manifest_details["group_map"]
    sequential_telemetry_records = telemetry_records_for(
        args.sequential_telemetry_summaries,
        manifest_group_map,
        "sequential",
    )
    parallel_telemetry_records = telemetry_records_for(
        args.parallel_telemetry_summaries,
        manifest_group_map,
        "parallel",
    )
    sequential_telemetry = aggregate_telemetry_records(
        sequential_telemetry_records
    )
    parallel_telemetry = aggregate_telemetry_records(
        parallel_telemetry_records
    )
    telemetry_coverage = {
        "sequential": telemetry_coverage_report(
            manifest_details["expected_sequential_command_ids"],
            sequential_telemetry_records,
        ),
        "parallel": telemetry_coverage_report(
            manifest_details["expected_parallel_command_ids"],
            parallel_telemetry_records,
        ),
    }
    sequential_by_key = record_map(sequential_records)
    parallel_by_key = record_map(parallel_records)
    matched_keys = tuple(
        sorted(
            key for key in sequential_by_key.keys()
            if key in parallel_by_key
        )
    )
    manifest_result_keys = tuple(
        (key[0], key[1]) for key in manifest_details["result_keys"]
    )
    matched_result_keys = tuple(
        sorted(
            set(sequential_by_key[key].key for key in matched_keys)
        )
    )
    manifest_missing_result_keys = [
        list(key) for key in manifest_result_keys
        if key not in matched_result_keys
    ]
    manifest_result_key_set = set(manifest_result_keys)
    manifest_unexpected_result_keys = [
        list(key) for key in matched_result_keys
        if len(manifest_result_key_set) > 0 and key not in manifest_result_key_set
    ]
    manifest_probe_result_identities = tuple(
        tuple(identity) for identity in manifest_details["probe_result_identities"]
    )
    observed_probe_result_identities = tuple(
        key for key in matched_keys
        if len(key) == len(PROBE_IDENTITY_KEYS) + 2
    )
    observed_probe_result_identity_set = set(observed_probe_result_identities)
    manifest_probe_result_identity_set = set(manifest_probe_result_identities)
    manifest_missing_probe_result_identities = [
        list(identity) for identity in manifest_probe_result_identities
        if identity not in observed_probe_result_identity_set
    ]
    manifest_unexpected_probe_result_identities = [
        list(identity) for identity in observed_probe_result_identities
        if (
            len(manifest_probe_result_identity_set) > 0
            and identity not in manifest_probe_result_identity_set
        )
    ]
    pair_reports = [
        compare_record_pair(
            args,
            sequential_by_key[key],
            parallel_by_key[key],
            sequential_telemetry,
            parallel_telemetry,
        )
        for key in matched_keys
    ]
    speedups = [
        report["timing"]["wall_clock_speedup"]
        for report in pair_reports
        if report["timing"]["wall_clock_speedup"] is not None
    ]
    if len(speedups) == 0:
        mean_wall_clock_speedup = None
    else:
        mean_wall_clock_speedup = sum(speedups) / float(len(speedups))
    manifest_speedup_ceiling = manifest_details["speedup_expectation"]["trainer_invocation_speedup_ceiling"]
    if (
            mean_wall_clock_speedup is not None
            and manifest_speedup_ceiling is not None
            and manifest_speedup_ceiling > 0.0
        ):
        manifest_speedup_efficiency = mean_wall_clock_speedup / manifest_speedup_ceiling
    else:
        manifest_speedup_efficiency = None
    parallel_group_runtime = aggregate_parallel_group_runtime(pair_reports)
    summary = {
        "sequential_record_count": len(sequential_records),
        "parallel_record_count": len(parallel_records),
        "matched_pair_count": len(pair_reports),
        "missing_parallel_keys": [
            list(key) for key in sorted(sequential_by_key.keys())
            if key not in parallel_by_key
        ],
        "missing_sequential_keys": [
            list(key) for key in sorted(parallel_by_key.keys())
            if key not in sequential_by_key
        ],
        "manifest_result_expectation_available": len(manifest_result_keys) > 0,
        "manifest_result_key_count": len(manifest_result_keys),
        "manifest_result_keys": [
            list(key) for key in manifest_result_keys
        ],
        "manifest_missing_result_keys": manifest_missing_result_keys,
        "manifest_unexpected_result_keys": manifest_unexpected_result_keys,
        "manifest_probe_result_expectation_available": (
            len(manifest_probe_result_identities) > 0
        ),
        "observed_probe_result_identity_available": (
            len(observed_probe_result_identities) > 0
        ),
        "manifest_probe_result_identity_count": len(manifest_probe_result_identities),
        "manifest_probe_result_identities": [
            list(identity) for identity in manifest_probe_result_identities
        ],
        "observed_probe_result_identities": [
            list(identity) for identity in observed_probe_result_identities
        ],
        "manifest_missing_probe_result_identities": (
            manifest_missing_probe_result_identities
        ),
        "manifest_unexpected_probe_result_identities": (
            manifest_unexpected_probe_result_identities
        ),
        "mean_wall_clock_speedup": mean_wall_clock_speedup,
        "parallel_group_runtime": parallel_group_runtime,
        "manifest_speedup_expectation": manifest_details["speedup_expectation"],
        "manifest_speedup_efficiency": manifest_speedup_efficiency,
        "total_comparable_metric_count": sum(
            report["comparable_metric_count"] for report in pair_reports
        ),
        "total_failing_metric_count": sum(
            report["failing_metric_count"] for report in pair_reports
        ),
        "total_comparable_ensemble_metric_count": sum(
            report["comparable_ensemble_metric_count"] for report in pair_reports
        ),
        "total_failing_ensemble_metric_count": sum(
            report["failing_ensemble_metric_count"] for report in pair_reports
        ),
        "sequential_telemetry_key_count": len(sequential_telemetry),
        "parallel_telemetry_key_count": len(parallel_telemetry),
        "telemetry_coverage": telemetry_coverage,
        "runner_reports": runner_reports,
    }
    summary["validation_verdict"] = build_validation_verdict(
        args,
        summary,
        pair_reports,
    )
    return {
        "summary": summary,
        "pairs": pair_reports,
    }


def write_report(report, output_path: str, json_indent: int) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=json_indent)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    report = build_comparison_report(args)
    if args.output_path is not None:
        write_report(report, args.output_path, args.json_indent)
    print(json.dumps(report, indent=args.json_indent))


if __name__ == "__main__":
    main()
