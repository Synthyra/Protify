import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


NVIDIA_SMI_QUERY_FIELDS = (
    "timestamp",
    "index",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "clocks.sm",
)


FIELD_ALIASES = {
    "timestamp": "driver_timestamp",
    "index": "gpu_index",
    "utilization.gpu": "gpu_utilization_percent",
    "utilization.memory": "memory_utilization_percent",
    "memory.used": "memory_used_mib",
    "memory.total": "memory_total_mib",
    "power.draw": "power_draw_watts",
    "clocks.sm": "sm_clock_mhz",
}


def parse_args(argv=None):
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    command = None
    if "--command" in raw_argv:
        command_index = raw_argv.index("--command")
        command = raw_argv[command_index + 1:]
        raw_argv = raw_argv[:command_index]
        if len(command) > 0 and command[0] == "--":
            command = command[1:]
    parser = argparse.ArgumentParser(
        description=(
            "Sample nvidia-smi while a future sequential or parallel probe command runs, "
            "or summarize an existing telemetry JSONL file."
        )
    )
    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument("--summary_json", type=str, default=None)
    parser.add_argument("--summarize_jsonl", type=str, default=None)
    parser.add_argument("--interval_seconds", type=float, default=1.0)
    parser.add_argument("--gpu_index", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--json_indent", type=int, default=2)
    args = parser.parse_args(raw_argv)
    args.command = command
    return validate_args(args)


def validate_args(args):
    assert args.interval_seconds > 0.0, "interval_seconds must be positive."
    assert args.json_indent >= 0, "json_indent must be non-negative."
    if args.gpu_index is not None:
        assert args.gpu_index >= 0, "gpu_index must be non-negative when provided."
    if args.summarize_jsonl is None:
        assert args.command is not None and len(args.command) > 0, (
            "Provide --command for monitoring mode or --summarize_jsonl for summary-only mode."
        )
        if args.command[0] == "--":
            args.command = args.command[1:]
        assert len(args.command) > 0, "--command must include at least one command token."
        assert args.output_jsonl is not None, "output_jsonl is required in monitoring mode."
    return args


def nvidia_smi_query_command(gpu_index: Optional[int] = None) -> List[str]:
    command = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(NVIDIA_SMI_QUERY_FIELDS),
        "--format=csv,noheader,nounits",
    ]
    if gpu_index is not None:
        command.insert(1, f"--id={gpu_index}")
    return command


def _parse_number(value: str):
    cleaned = value.strip()
    if cleaned in ("", "[Not Supported]", "N/A"):
        return None
    try:
        if "." in cleaned:
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return cleaned


def parse_nvidia_smi_csv_line(line: str, sampled_at_unix_seconds: float) -> Dict[str, object]:
    parts = [part.strip() for part in line.split(",")]
    assert len(parts) == len(NVIDIA_SMI_QUERY_FIELDS), (
        f"Expected {len(NVIDIA_SMI_QUERY_FIELDS)} nvidia-smi columns, got {len(parts)}."
    )
    sample: Dict[str, object] = {
        "sampled_at_unix_seconds": sampled_at_unix_seconds,
    }
    for index, field in enumerate(NVIDIA_SMI_QUERY_FIELDS):
        alias = FIELD_ALIASES[field]
        if field == "timestamp":
            sample[alias] = parts[index]
        else:
            sample[alias] = _parse_number(parts[index])
    if (
            "memory_used_mib" in sample
            and "memory_total_mib" in sample
            and isinstance(sample["memory_used_mib"], (int, float))
            and isinstance(sample["memory_total_mib"], (int, float))
            and sample["memory_total_mib"] > 0
        ):
        sample["memory_used_fraction"] = (
            float(sample["memory_used_mib"]) / float(sample["memory_total_mib"])
        )
    return sample


def parse_nvidia_smi_output(output: str, sampled_at_unix_seconds: float) -> List[Dict[str, object]]:
    samples = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        samples.append(parse_nvidia_smi_csv_line(stripped, sampled_at_unix_seconds))
    return samples


def sample_nvidia_smi(gpu_index: Optional[int] = None) -> List[Dict[str, object]]:
    sampled_at = time.time()
    completed = subprocess.run(
        nvidia_smi_query_command(gpu_index),
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_nvidia_smi_output(completed.stdout, sampled_at)


def append_samples(path: Path, samples: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")


def load_samples(path: Path) -> List[Dict[str, object]]:
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped == "":
                continue
            payload = json.loads(stripped)
            assert isinstance(payload, dict), "Telemetry JSONL rows must be JSON objects."
            samples.append(payload)
    return samples


def _numeric_values(samples: List[Dict[str, object]], key: str) -> List[float]:
    values = []
    for sample in samples:
        if key not in sample:
            continue
        value = sample[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _mean(values: List[float]) -> Optional[float]:
    if len(values) == 0:
        return None
    return sum(values) / float(len(values))


def _maximum(values: List[float]) -> Optional[float]:
    if len(values) == 0:
        return None
    return max(values)


def summarize_samples(samples: List[Dict[str, object]]) -> Dict[str, object]:
    timestamps = _numeric_values(samples, "sampled_at_unix_seconds")
    gpu_indices = sorted(
        {
            int(sample["gpu_index"])
            for sample in samples
            if "gpu_index" in sample and isinstance(sample["gpu_index"], int)
        }
    )
    if len(timestamps) >= 2:
        duration_seconds = max(timestamps) - min(timestamps)
    else:
        duration_seconds = 0.0
    return {
        "sample_count": len(samples),
        "gpu_indices": gpu_indices,
        "duration_seconds": duration_seconds,
        "gpu_utilization_percent_mean": _mean(_numeric_values(samples, "gpu_utilization_percent")),
        "gpu_utilization_percent_max": _maximum(_numeric_values(samples, "gpu_utilization_percent")),
        "memory_utilization_percent_mean": _mean(_numeric_values(samples, "memory_utilization_percent")),
        "memory_utilization_percent_max": _maximum(_numeric_values(samples, "memory_utilization_percent")),
        "memory_used_mib_max": _maximum(_numeric_values(samples, "memory_used_mib")),
        "memory_used_fraction_max": _maximum(_numeric_values(samples, "memory_used_fraction")),
        "power_draw_watts_mean": _mean(_numeric_values(samples, "power_draw_watts")),
        "power_draw_watts_max": _maximum(_numeric_values(samples, "power_draw_watts")),
        "sm_clock_mhz_mean": _mean(_numeric_values(samples, "sm_clock_mhz")),
        "sm_clock_mhz_max": _maximum(_numeric_values(samples, "sm_clock_mhz")),
    }


def dry_run_plan(args) -> Dict[str, object]:
    return {
        "mode": "dry_run",
        "nvidia_smi_command": nvidia_smi_query_command(args.gpu_index),
        "wrapped_command": list(args.command),
        "output_jsonl": args.output_jsonl,
        "summary_json": args.summary_json,
        "interval_seconds": args.interval_seconds,
    }


def write_summary_if_requested(summary: Dict[str, object], summary_json: Optional[str]) -> None:
    if summary_json is None:
        return
    path = Path(summary_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def monitor_command(args) -> Dict[str, object]:
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    process = subprocess.Popen(args.command)
    while process.poll() is None:
        append_samples(output_path, sample_nvidia_smi(args.gpu_index))
        time.sleep(args.interval_seconds)
    return_code = process.wait()
    append_samples(output_path, sample_nvidia_smi(args.gpu_index))
    samples = load_samples(output_path)
    summary = summarize_samples(samples)
    summary["return_code"] = return_code
    summary["wrapped_command"] = list(args.command)
    summary["output_jsonl"] = str(output_path)
    write_summary_if_requested(summary, args.summary_json)
    return summary


def main() -> None:
    args = parse_args()
    if args.summarize_jsonl is not None:
        samples = load_samples(Path(args.summarize_jsonl))
        summary = summarize_samples(samples)
        write_summary_if_requested(summary, args.summary_json)
        print(json.dumps(summary, indent=args.json_indent, sort_keys=True))
        return
    if args.dry_run:
        print(json.dumps(dry_run_plan(args), indent=args.json_indent, sort_keys=True))
        return
    summary = monitor_command(args)
    print(json.dumps(summary, indent=args.json_indent, sort_keys=True))


if __name__ == "__main__":
    main()
