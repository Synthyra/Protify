"""Cloud CLI dispatch for remote Protify job execution.

Backend-agnostic replacement for modal_cli.py. Uses the CloudBackend protocol
from cloud_backend.py to submit, poll, and fetch results from any compatible server.
"""

import base64
import json
import os
import sys
import time

from pathlib import Path
from typing import Any


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _to_json_safe(value: Any) -> Any:
    if _is_json_scalar(value):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_json_safe(val) for key, val in value.items()}
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _should_auto_run_cloud(args: Any) -> bool:
    """Check if we should auto-dispatch to cloud execution."""
    if os.environ.get("PROTIFY_JOB_ID", ""):
        return False
    if getattr(args, "replay_path", None) is not None:
        return False
    return bool(getattr(args, "cloud", False))


def _build_cloud_config_from_args(args: Any) -> dict[str, Any]:
    """Build a config dict from CLI args, excluding cloud-specific keys."""
    excluded_keys = {
        "cloud",
        "cloud_url",
        "cloud_gpu_type",
        "cloud_timeout_seconds",
        "cloud_poll_interval",
        "cloud_artifacts_dir",
    }
    config: dict[str, Any] = {}
    for key, value in args.__dict__.items():
        if key in excluded_keys:
            continue
        config[key] = _to_json_safe(value)
    config["replay_path"] = None
    return config


def _save_cloud_artifacts(
    result_payload: dict[str, Any],
    output_root: str,
    job_id: str,
) -> str:
    """Save fetched results (TSV, images) to local disk."""
    job_dir = Path(output_root) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save results TSV
    results_tsv = result_payload.get("results_tsv")
    if results_tsv:
        tsv_path = job_dir / "results.tsv"
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write(results_tsv)

    # Save images
    images = result_payload.get("images")
    if images and isinstance(images, list):
        plots_dir = job_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        for img in images:
            filename = img.get("filename", "plot.png")
            encoded_image = img.get("data", "")
            if encoded_image:
                img_path = plots_dir / filename
                with open(img_path, "wb") as f:
                    f.write(base64.b64decode(encoded_image))

    # Save summary
    summary_path = job_dir / "cloud_fetch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, default=str)

    return str(job_dir)


def _run_on_cloud(args: Any) -> int:
    """Main entry point for cloud execution. Returns exit code (0=success, 1=failure)."""
    from .cloud_backend import get_or_create_cloud_backend

    api_key = os.environ.get("SYNTHYRA_API_KEY")
    base_url = getattr(args, "cloud_url", None)
    backend = get_or_create_cloud_backend(api_key=api_key, base_url=base_url)
    assert backend is not None, "A cloud run needs a Synthyra API key: set SYNTHYRA_API_KEY to the key from synthyra.com."

    gpu_type = getattr(args, "cloud_gpu_type", None) or "A10"
    timeout_seconds = getattr(args, "cloud_timeout_seconds", None) or 86400
    poll_interval = getattr(args, "cloud_poll_interval", None) or 5
    artifacts_root = getattr(args, "cloud_artifacts_dir", None) or "cloud_artifacts"

    # Check if there is work to do
    has_dataset_run = len(getattr(args, "data_names", [])) > 0 or len(getattr(args, "data_dirs", [])) > 0
    proteingym = getattr(args, "proteingym", False)
    if not has_dataset_run and not proteingym:
        print("No datasets or ProteinGym specified. Nothing to submit.")
        return 0

    # Build and submit
    config = _build_cloud_config_from_args(args)

    print("Submitting job to cloud backend...")
    submit_result = backend.submit_job(
        config=config,
        gpu_type=gpu_type,
        timeout_seconds=timeout_seconds,
    )
    assert isinstance(submit_result, dict), "Cloud submit response is not a dictionary."
    job_id = submit_result["job_id"]
    print(f"Job submitted: {job_id}")

    # Poll loop
    terminal_states = {"Complete", "Failed", "Cancelled", "SUCCESS", "FAILED", "TERMINATED", "TIMEOUT"}
    poll_start_time = time.time()
    max_poll_seconds = int(timeout_seconds) + 900
    status_print_interval = 15
    last_status_print_time = 0.0
    last_status_line = ""
    log_offset = 0

    final_status: str | None = None
    final_error: str | None = None

    while True:
        # Stream logs
        try:
            log_resp = backend.get_job_logs(job_id=job_id, offset=log_offset)
            content = log_resp.get("content", "")
            if content:
                sys.stdout.write(content)
                sys.stdout.flush()
                log_offset = log_resp.get("next_offset", log_offset + len(content))
        except Exception:  # noqa: broad-except silent-except a failed log poll is skipped; the status poll below reports the job
            pass

        # Poll status
        try:
            status_resp = backend.get_job_status(job_id=job_id)
        except Exception as e:  # noqa: broad-except a failed status poll is printed and polling retries
            print(f"[Cloud] Status poll error: {e}")
            time.sleep(max(1, int(poll_interval)))
            continue

        status_value = status_resp.get("status", "Unknown")
        phase_value = status_resp.get("phase", "N/A")
        status_line = f"[Cloud] status={status_value} phase={phase_value}"

        if status_value in terminal_states:
            final_status = status_value
            final_error = status_resp.get("error")
            break

        now = time.time()
        if status_line != last_status_line or (now - last_status_print_time) >= status_print_interval:
            print(status_line)
            last_status_line = status_line
            last_status_print_time = now

        elapsed = now - poll_start_time
        if elapsed > max_poll_seconds:
            final_status = "TIMEOUT"
            final_error = f"Polling exceeded timeout window ({max_poll_seconds} seconds)."
            break

        time.sleep(max(1, int(poll_interval)))

    # Final log flush
    try:
        log_resp = backend.get_job_logs(job_id=job_id, offset=log_offset, max_chars=200000)
        content = log_resp.get("content", "")
        if content:
            sys.stdout.write(content)
            sys.stdout.flush()
    except Exception:  # noqa: broad-except silent-except the final log flush is best effort; the results fetch below still runs
        pass

    # Fetch results
    try:
        results = backend.get_results(job_id=job_id)
        if results:
            artifacts_dir = _save_cloud_artifacts(results, artifacts_root, job_id)
            print(f"Artifacts saved to {artifacts_dir}")
            hub_url = results.get("hub_url")
            if hub_url:
                print(f"Model pushed to: {hub_url}")
    except Exception as e:  # noqa: broad-except a failed results fetch is printed and the final status still reports
        print(f"Failed to fetch results: {e}")

    # Report final status
    if final_status in ("Complete", "SUCCESS"):
        print(f"Job {job_id} completed successfully.")
        return 0
    else:
        if final_error:
            print(f"Job {job_id} failed: {final_error}")
        else:
            print(f"Job {job_id} ended with status: {final_status}")
        return 1
