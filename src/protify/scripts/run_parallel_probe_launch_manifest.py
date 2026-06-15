import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Dry-run or execute commands from a parallel-probe preflight launch manifest."
    )
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--phase", choices=["probes", "embeddings", "all"], default="probes")
    parser.add_argument("--variant", choices=["parallel", "sequential", "both"], default="parallel")
    parser.add_argument("--use_monitor", action="store_true")
    parser.add_argument("--command_ids", nargs="+", default=None)
    parser.add_argument("--wave_ids", nargs="+", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--allow_over_budget", action="store_true")
    parser.add_argument("--wave_execution_mode", choices=["sequential", "concurrent"], default="sequential")
    parser.add_argument("--allow_baseline_concurrency", action="store_true")
    parser.add_argument("--continue_on_failure", action="store_true")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--json_indent", type=int, default=2)
    return validate_args(parser.parse_args(argv))


def validate_args(args):
    assert args.manifest_path.strip() != "", "manifest_path must be non-empty."
    assert args.json_indent >= 0, "json_indent must be non-negative."
    if args.output_path is not None:
        assert args.output_path.strip() != "", "output_path must be non-empty when provided."
    if args.command_ids is not None:
        assert len(args.command_ids) > 0, "command_ids must be non-empty when provided."
    if args.wave_ids is not None:
        assert len(args.wave_ids) > 0, "wave_ids must be non-empty when provided."
    if (
        args.wave_execution_mode == "concurrent"
        and args.phase != "embeddings"
        and args.variant != "parallel"
    ):
        assert args.allow_baseline_concurrency, (
            "Concurrent launch with sequential or both variants can contaminate "
            "baseline timing. Use --allow_baseline_concurrency to override."
        )
    return args


def load_launch_manifest(path_text: str):
    path = Path(path_text)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload, dict), "Launch manifest file must contain a JSON object."
    if "launch_manifest" in payload:
        launch_manifest = payload["launch_manifest"]
        assert isinstance(launch_manifest, dict), "launch_manifest must be a JSON object."
        manifest = dict(launch_manifest)
        if "embedding_prerequisites" not in manifest and "embedding_prerequisites" in payload:
            manifest["embedding_prerequisites"] = payload["embedding_prerequisites"]
    else:
        manifest = payload
    assert isinstance(manifest, dict), "launch_manifest must be a JSON object."
    assert "groups" in manifest, "launch_manifest must include groups."
    assert "waves" in manifest, "launch_manifest must include waves."
    assert isinstance(manifest["groups"], list), "launch_manifest groups must be a list."
    assert isinstance(manifest["waves"], list), "launch_manifest waves must be a list."
    return manifest


def command_field_for(variant: str, use_monitor: bool) -> str:
    assert variant in ("parallel", "sequential"), "variant must be parallel or sequential."
    if use_monitor:
        return f"{variant}_monitor_command"
    return f"{variant}_command"


def selected_values(values):
    if values is None:
        return None
    return set(values)


def manifest_group_by_command_id(manifest):
    groups_by_id = {}
    for group in manifest["groups"]:
        assert isinstance(group, dict), "Each launch manifest group must be an object."
        assert "command_id" in group, "Each launch manifest group must include command_id."
        command_id = group["command_id"]
        assert command_id not in groups_by_id, f"Duplicate command_id in launch manifest: {command_id}"
        groups_by_id[command_id] = group
    return groups_by_id


def embedding_jobs_by_command_id(manifest):
    jobs_by_id = {}
    if "embedding_prerequisites" not in manifest:
        return jobs_by_id
    prerequisites = manifest["embedding_prerequisites"]
    assert isinstance(prerequisites, dict), "embedding_prerequisites must be an object."
    if "embedding_jobs" not in prerequisites:
        return jobs_by_id
    embedding_jobs = prerequisites["embedding_jobs"]
    assert isinstance(embedding_jobs, list), "embedding_prerequisites embedding_jobs must be a list."
    for job_index, job in enumerate(embedding_jobs):
        assert isinstance(job, dict), "Each embedding prerequisite job must be an object."
        if "command_id" in job:
            command_id = job["command_id"]
        else:
            command_id = f"embedding-{job_index + 1}"
        assert isinstance(command_id, str), "Embedding prerequisite command_id must be a string."
        assert command_id not in jobs_by_id, f"Duplicate embedding command_id in launch manifest: {command_id}"
        job_copy = dict(job)
        job_copy["command_id"] = command_id
        jobs_by_id[command_id] = job_copy
    return jobs_by_id


def variants_for_plan(variant: str):
    if variant == "both":
        return ("sequential", "parallel")
    return (variant,)


def wave_gpu_over_budget_assignment_count(wave):
    if "gpu_over_memory_budget_count" in wave:
        over_budget_count = wave["gpu_over_memory_budget_count"]
        assert isinstance(over_budget_count, int), "wave gpu_over_memory_budget_count must be an integer."
        assert over_budget_count >= 0, "wave gpu_over_memory_budget_count must be non-negative."
        return over_budget_count
    if "gpu_assignments" not in wave:
        return 0
    gpu_assignments = wave["gpu_assignments"]
    assert isinstance(gpu_assignments, list), "wave gpu_assignments must be a list."
    over_budget_count = 0
    for gpu_assignment in gpu_assignments:
        assert isinstance(gpu_assignment, dict), "Each wave gpu assignment must be an object."
        if "over_memory_budget" in gpu_assignment:
            over_memory_budget = gpu_assignment["over_memory_budget"]
            assert isinstance(over_memory_budget, bool), "gpu assignment over_memory_budget must be a bool."
            if over_memory_budget:
                over_budget_count += 1
    return over_budget_count


def monitor_summary_path_for_group(group, variant: str):
    monitor_field = f"{variant}_monitor_command"
    if monitor_field not in group:
        return None
    monitor_command = group[monitor_field]
    assert isinstance(monitor_command, list), f"{monitor_field} must be a command array."
    for item_index, item in enumerate(monitor_command):
        if item == "--summary_json":
            summary_index = item_index + 1
            assert summary_index < len(monitor_command), (
                f"{monitor_field} includes --summary_json without a path."
            )
            summary_path = monitor_command[summary_index]
            assert isinstance(summary_path, str), f"{monitor_field} summary path must be a string."
            return summary_path
    return None


def completion_summary_exists(summary_path):
    if summary_path is None:
        return False
    return Path(summary_path).is_file()


def command_entry_for_group(group, variant: str, use_monitor: bool, skip_completed: bool = False):
    command_field = command_field_for(variant, use_monitor)
    assert command_field in group, f"Launch manifest group missing {command_field}."
    command = group[command_field]
    assert isinstance(command, list), f"{command_field} must be a command array."
    completion_summary_path = monitor_summary_path_for_group(group, variant)
    if "environment" in group:
        environment = group["environment"]
        assert isinstance(environment, dict), "Launch manifest environment must be an object."
        for key, value in environment.items():
            assert isinstance(key, str), "Launch manifest environment keys must be strings."
            assert isinstance(value, str), "Launch manifest environment values must be strings."
    else:
        environment = {}
    if len(command) == 0:
        return {
            "command_id": group["command_id"],
            "variant": variant,
            "command_field": command_field,
            "command": [],
            "environment": dict(environment),
            "completion_summary_path": completion_summary_path,
            "skipped": True,
            "skip_reason": "empty_command",
        }
    if skip_completed and completion_summary_exists(completion_summary_path):
        return {
            "command_id": group["command_id"],
            "variant": variant,
            "command_field": command_field,
            "command": list(command),
            "environment": dict(environment),
            "completion_summary_path": completion_summary_path,
            "skipped": True,
            "skip_reason": "completed_summary_exists",
        }
    return {
        "command_id": group["command_id"],
        "variant": variant,
        "command_field": command_field,
        "command": list(command),
        "environment": dict(environment),
        "completion_summary_path": completion_summary_path,
        "skipped": False,
        "skip_reason": None,
    }


def command_entry_for_embedding_job(job):
    assert "command" in job, "Embedding prerequisite job missing command."
    command = job["command"]
    assert isinstance(command, list), "Embedding prerequisite command must be a command array."
    if "command_environment" in job:
        environment = job["command_environment"]
        assert isinstance(environment, dict), "Embedding prerequisite command_environment must be an object."
        for key, value in environment.items():
            assert isinstance(key, str), "Embedding prerequisite environment keys must be strings."
            assert isinstance(value, str), "Embedding prerequisite environment values must be strings."
    else:
        environment = {}
    if len(command) == 0:
        skipped = True
        skip_reason = "empty_command"
    else:
        skipped = False
        skip_reason = None
    return {
        "command_id": job["command_id"],
        "variant": "embeddings",
        "command_field": "embedding_command",
        "command": list(command),
        "environment": dict(environment),
        "completion_summary_path": None,
        "skipped": skipped,
        "skip_reason": skip_reason,
    }


def build_embedding_wave(jobs_by_id, selected_command_ids, selected_wave_ids):
    embedding_wave_id = "embedding-prerequisites"
    if selected_wave_ids is not None and embedding_wave_id not in selected_wave_ids:
        return None
    commands = []
    for command_id in sorted(jobs_by_id.keys()):
        if selected_command_ids is not None and command_id not in selected_command_ids:
            continue
        commands.append(command_entry_for_embedding_job(jobs_by_id[command_id]))
    if len(commands) == 0:
        return None
    return {
        "wave_id": embedding_wave_id,
        "command_count": sum(1 for command in commands if not command["skipped"]),
        "skipped_count": sum(1 for command in commands if command["skipped"]),
        "gpu_over_memory_budget_count": 0,
        "gpu_over_memory_budget": False,
        "commands": commands,
    }


def build_launch_plan(manifest, args):
    groups_by_id = manifest_group_by_command_id(manifest)
    embedding_jobs_by_id = embedding_jobs_by_command_id(manifest)
    selected_command_ids = selected_values(args.command_ids)
    selected_wave_ids = selected_values(args.wave_ids)
    waves = []
    command_count = 0
    skipped_count = 0
    missing_command_ids = []
    over_budget_wave_ids = []
    over_budget_assignment_count = 0

    if args.phase in ("embeddings", "all"):
        embedding_wave = build_embedding_wave(
            embedding_jobs_by_id,
            selected_command_ids,
            selected_wave_ids,
        )
        if embedding_wave is not None:
            command_count += embedding_wave["command_count"]
            skipped_count += embedding_wave["skipped_count"]
            waves.append(embedding_wave)

    if args.phase in ("probes", "all"):
        for wave in manifest["waves"]:
            assert isinstance(wave, dict), "Each launch manifest wave must be an object."
            assert "wave_id" in wave, "Each launch manifest wave must include wave_id."
            assert "command_ids" in wave, "Each launch manifest wave must include command_ids."
            wave_id = wave["wave_id"]
            if selected_wave_ids is not None and wave_id not in selected_wave_ids:
                continue
            wave_over_budget_assignment_count = wave_gpu_over_budget_assignment_count(wave)
            wave_commands = []
            for command_id in wave["command_ids"]:
                if selected_command_ids is not None and command_id not in selected_command_ids:
                    continue
                if command_id not in groups_by_id:
                    missing_command_ids.append(command_id)
                    continue
                group = groups_by_id[command_id]
                for variant in variants_for_plan(args.variant):
                    command_entry = command_entry_for_group(
                        group,
                        variant,
                        args.use_monitor,
                        skip_completed=args.skip_completed,
                    )
                    if command_entry["skipped"]:
                        skipped_count += 1
                    else:
                        command_count += 1
                    wave_commands.append(command_entry)
            if len(wave_commands) > 0:
                if wave_over_budget_assignment_count > 0:
                    over_budget_wave_ids.append(wave_id)
                    over_budget_assignment_count += wave_over_budget_assignment_count
                waves.append(
                    {
                        "wave_id": wave_id,
                        "command_count": sum(1 for command in wave_commands if not command["skipped"]),
                        "skipped_count": sum(1 for command in wave_commands if command["skipped"]),
                        "gpu_over_memory_budget_count": wave_over_budget_assignment_count,
                        "gpu_over_memory_budget": wave_over_budget_assignment_count > 0,
                        "commands": wave_commands,
                    }
                )

    known_ids = set(groups_by_id.keys()) | set(embedding_jobs_by_id.keys())
    if selected_command_ids is not None:
        unknown_selected_ids = sorted(selected_command_ids - known_ids)
    else:
        unknown_selected_ids = []

    return {
        "execute": args.execute,
        "phase": args.phase,
        "variant": args.variant,
        "use_monitor": args.use_monitor,
        "skip_completed": args.skip_completed,
        "allow_over_budget": args.allow_over_budget,
        "allow_baseline_concurrency": args.allow_baseline_concurrency,
        "wave_execution_mode": args.wave_execution_mode,
        "continue_on_failure": args.continue_on_failure,
        "selected_command_ids": sorted(selected_command_ids) if selected_command_ids is not None else None,
        "selected_wave_ids": sorted(selected_wave_ids) if selected_wave_ids is not None else None,
        "wave_count": len(waves),
        "command_count": command_count,
        "skipped_count": skipped_count,
        "over_budget_wave_ids": over_budget_wave_ids,
        "over_budget_assignment_count": over_budget_assignment_count,
        "blocked_by_over_budget": (
            args.execute
            and not args.allow_over_budget
            and over_budget_assignment_count > 0
        ),
        "unknown_selected_command_ids": unknown_selected_ids,
        "missing_wave_command_ids": missing_command_ids,
        "waves": waves,
    }


def subprocess_environment(command_entry):
    environment = dict(os.environ)
    for key, value in command_entry["environment"].items():
        environment[key] = value
    return environment


def run_command(command_entry):
    completed = subprocess.run(
        command_entry["command"],
        check=False,
        env=subprocess_environment(command_entry),
    )
    result = dict(command_entry)
    result["returncode"] = completed.returncode
    return result


def run_wave_sequential(wave, continue_on_failure: bool):
    wave_results = []
    failure_count = 0
    should_stop = False
    for command_entry in wave["commands"]:
        if command_entry["skipped"]:
            wave_results.append(command_entry)
            continue
        result = run_command(command_entry)
        wave_results.append(result)
        if result["returncode"] != 0:
            failure_count += 1
            if not continue_on_failure:
                should_stop = True
                break
    return wave_results, failure_count, should_stop


def run_wave_concurrent(wave):
    wave_results = [None] * len(wave["commands"])
    processes = []
    for command_index, command_entry in enumerate(wave["commands"]):
        if command_entry["skipped"]:
            wave_results[command_index] = command_entry
            continue
        process = subprocess.Popen(
            command_entry["command"],
            env=subprocess_environment(command_entry),
        )
        processes.append((command_index, command_entry, process))

    failure_count = 0
    for command_index, command_entry, process in processes:
        returncode = process.wait()
        result = dict(command_entry)
        result["returncode"] = returncode
        if returncode != 0:
            failure_count += 1
        wave_results[command_index] = result

    return wave_results, failure_count, False


def execute_launch_plan(plan):
    assert plan["execute"], "execute_launch_plan requires a plan with execute=True."
    assert not plan["blocked_by_over_budget"], (
        "Launch plan includes waves over the preflight GPU memory budget. "
        "Re-run with --allow_over_budget to execute anyway."
    )
    executed_waves = []
    failure_count = 0
    for wave in plan["waves"]:
        if plan["wave_execution_mode"] == "concurrent":
            wave_results, wave_failure_count, should_stop = run_wave_concurrent(wave)
            failure_count += wave_failure_count
            if wave_failure_count > 0 and not plan["continue_on_failure"]:
                should_stop = True
        else:
            wave_results, wave_failure_count, should_stop = run_wave_sequential(
                wave,
                continue_on_failure=plan["continue_on_failure"],
            )
            failure_count += wave_failure_count
        executed_waves.append(
            {
                "wave_id": wave["wave_id"],
                "commands": wave_results,
            }
        )
        if should_stop:
            return {
                "executed": True,
                "failure_count": failure_count,
                "waves": executed_waves,
            }
    return {
        "executed": True,
        "failure_count": failure_count,
        "waves": executed_waves,
    }


def build_report(args):
    manifest = load_launch_manifest(args.manifest_path)
    plan = build_launch_plan(manifest, args)
    report = {
        "manifest_path": args.manifest_path,
        "plan": plan,
    }
    if args.execute:
        report["execution"] = execute_launch_plan(plan)
    else:
        report["execution"] = {
            "executed": False,
            "failure_count": 0,
            "waves": [],
        }
    return report


def write_report(report, output_path: str, json_indent: int) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=json_indent)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.output_path is not None:
        write_report(report, args.output_path, args.json_indent)
    print(json.dumps(report, indent=args.json_indent))


if __name__ == "__main__":
    main()
