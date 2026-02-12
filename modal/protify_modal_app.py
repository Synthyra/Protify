"""
Protify Modal App

A Modal-based application for running Protify (Protein Representation Learning) experiments
with a Gradio web interface. This app allows you to:

1. Configure experiments through a web interface (similar to the Tkinter GUI)
2. Submit jobs to run on GPU infrastructure
3. Track job progress and view results

Usage:
    modal deploy modal/protify_modal_app.py
    
    Note: Run this command from the project root directory.
    
    Then visit the web interface URL provided by Modal.

Requirements:
    - Modal account and API key configured
    - HuggingFace token (optional, for accessing gated models)
    - All Protify dependencies installed via requirements.txt

TODO manually clean up
"""

import modal
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml
import json

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
# Get the project root (parent of modal/ directory)
PROJECT_ROOT = SCRIPT_DIR.parent

# Constants for app management
APP_NAME = "protify-app"

# Resource configuration constants
PROTIFY_DEFAULT_GPU = "A10"
PROTIFY_CPU_MIN_GPU, PROTIFY_CPU_MAX_GPU = 8.0, 16.0
PROTIFY_MEMORY_MIN_GPU, PROTIFY_MEMORY_MAX_GPU = 65536, 262144
MAX_CONTAINERS_PROTIFY = 8
SCALEDOWN_WINDOW = 10  # seconds to wait before scaling down idle containers
WEB_INTERFACE_SCALEDOWN_WINDOW = 300  # 5 minutes for web interface
ENABLE_MEMORY_SNAPSHOT = True
TIMEOUT = 86400  # 24 hours timeout for long-running jobs

# Available GPU options
AVAILABLE_GPUS = [
    "H200",
    "H100", 
    "A100-80GB",
    "A100",
    "L40S",
    "A10",
    "L4",
    "T4"
]

# CPU-only function settings
CPU_FUNCTION_MEMORY_MIN, CPU_FUNCTION_MEMORY_MAX = 4096, 8192
CPU_FUNCTION_CPU_MIN, CPU_FUNCTION_CPU_MAX = 2.0, 4.0
MAX_CONTAINERS_CPU_FUNCTIONS = 10

# Create Modal app
app = modal.App(APP_NAME)

# Build image conditionally - add local_data only if it exists
# Add all Protify files and directories explicitly, similar to example_modal_app.py
# Note: Since this file is in modal/, paths need to reference parent directory
# Modal resolves paths relative to where 'modal deploy' is run from.
# We'll use paths relative to project root - these will work when deploy is run from project root
req_file_path = "requirements.txt"  # Relative to project root
src_dir_path = "src"  # Relative to project root
readme_file_path = "README.md"  # Relative to project root
local_data_dir_path = "local_data"  # Relative to project root

# Build base image
image_base = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "curl")
    .run_commands("pip install --upgrade pip setuptools")
)

# Add requirements.txt if it exists (relative to project root where deploy is run)
if (PROJECT_ROOT / req_file_path).exists():
    image_base = image_base.add_local_file(req_file_path, "/tmp/requirements.txt", copy=True)

# Continue building image
image_base = image_base.run_commands("pip install -r /tmp/requirements.txt")
image_base = image_base.run_commands("pip install gradio fastapi")
image_base = image_base.run_commands("pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128 -U")
image_base = image_base.env({
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "TOKENIZERS_PARALLELISM": "true",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
})

# Add Protify source code - copy the entire src directory structure
# This includes: protify package, probes, base_models, data, visualization, etc.
if (PROJECT_ROOT / src_dir_path).exists():
    image_base = image_base.add_local_dir(src_dir_path, "/root/src")

# Add README.md to the image
if (PROJECT_ROOT / readme_file_path).exists():
    image_base = image_base.add_local_file(readme_file_path, "/root/README.md")

# Conditionally add local_data directory if it exists at root level
if (PROJECT_ROOT / local_data_dir_path).exists() and (PROJECT_ROOT / local_data_dir_path).is_dir():
    image = image_base.add_local_dir(local_data_dir_path, "/root/local_data")
else:
    image = image_base

# Also check if there's a local_data directory inside src/protify and add it if needed
# (This is already included in src, but being explicit ensures nothing is missed)

# Create a volume for persistent data storage
volume = modal.Volume.from_name("protify-data", create_if_missing=True)


# Helper function to safely update job storage with retry logic and volume commits
def update_job_storage(job_id: str, status: str, hf_username: Optional[str] = None, max_retries: int = 3) -> bool:
    """
    Safely update job storage with retry logic and volume commit.
    
    Args:
        job_id: Job ID to update
        status: New status (RUNNING, FINISHED, etc.)
        hf_username: Optional HuggingFace username to preserve/update
        max_retries: Maximum number of retry attempts
    
    Returns:
        True if update succeeded, False otherwise
    """
    import time
    job_storage_file = "/data/job_storage.json"
    
    for attempt in range(max_retries):
        try:
            # Load existing storage or initialize
            if os.path.exists(job_storage_file):
                with open(job_storage_file, 'r') as f:
                    job_storage = json.load(f)
            else:
                job_storage = {}
            
            # Update job entry
            if job_id in job_storage:
                # Preserve hf_username if not provided
                if hf_username is None:
                    hf_username = job_storage[job_id].get("hf_username", "Unknown")
                job_storage[job_id]["status"] = status
                job_storage[job_id]["hf_username"] = hf_username
            else:
                # Initialize new job entry
                job_storage[job_id] = {
                    "id": job_id,
                    "status": status,
                    "hf_username": hf_username or "Unknown"
                }
            
            # Write back to file
            with open(job_storage_file, 'w') as f:
                json.dump(job_storage, f, default=str)
            
            # CRITICAL: Commit volume to persist changes
            volume.commit()
            
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                continue
            else:
                print(f"Warning: Could not update job status after {max_retries} attempts: {e}")
                return False
    
    return False


# Helper function containing the actual job execution logic
def _execute_protify_job(
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run Protify with the given configuration.
    
    Args:
        config: Configuration dictionary (equivalent to CLI args)
        hf_token: HuggingFace token for private models
        job_id: Optional job ID for tracking
    
    Returns:
        Dictionary with results and file paths
    """
    import sys
    import subprocess
    import json
    import time
    import threading
    from pathlib import Path
    from types import SimpleNamespace
    
    # Add protify to Python path
    sys.path.insert(0, "/root/src")
    
    # Update job status to RUNNING
    if job_id:
        # Get hf_username from config if available
        hf_username = config.get("hf_username")
        update_job_storage(job_id, "RUNNING", hf_username)
    
    # Set up HuggingFace token - prioritize user-provided token over secret
    # User-provided token takes precedence for write operations (saving models)
    hf_token_from_secret = os.environ.get("HF_TOKEN")
    if hf_token:
        # User provided a token - use it and override environment
        print(f"Using user-provided HuggingFace token (overriding secret token)")
        os.environ["HF_TOKEN"] = hf_token
        from huggingface_hub import login
        login(hf_token)
    elif hf_token_from_secret:
        # Fall back to secret token if user didn't provide one
        print(f"Using HuggingFace token from Modal secret (read-only)")
        from huggingface_hub import login
        login(hf_token_from_secret)
    
    # Create temporary directory for this run
    run_dir = Path("/tmp/protify_run")
    run_dir.mkdir(exist_ok=True)
    
    # Fix paths in configuration for Modal container
    def fix_paths(obj):
        """Recursively fix paths in configuration."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    # Fix data/ paths - use volume path
                    if value.startswith('data/') or value.startswith('local_data/'):
                        obj[key] = f"/data/{value.split('/', 1)[-1]}"
                    # Fix relative paths for logs, results, etc.
                    elif key.endswith('_dir') and not os.path.isabs(value):
                        obj[key] = f"/data/{value}"
                elif isinstance(value, list):
                    obj[key] = [fix_paths(item) for item in value]
                elif isinstance(value, dict):
                    obj[key] = fix_paths(value)
        elif isinstance(obj, list):
            return [fix_paths(item) for item in obj]
        return obj
    
    # Apply path fixes
    config = fix_paths(config)
    
    # Set default paths if not provided
    config.setdefault("log_dir", "/data/logs")
    config.setdefault("results_dir", "/data/results")
    config.setdefault("model_save_dir", "/data/weights")
    config.setdefault("embedding_save_dir", "/data/embeddings")
    config.setdefault("plots_dir", "/data/plots")
    config.setdefault("download_dir", "/data/downloads")
    
    # Set optional attributes that main.py expects to exist
    config.setdefault("replay_path", None)
    config.setdefault("pretrained_probe_path", None)
    
    # Ensure directories exist
    for dir_key in ["log_dir", "results_dir", "model_save_dir", "embedding_save_dir", "plots_dir", "download_dir"]:
        if dir_key in config:
            Path(config[dir_key]).mkdir(parents=True, exist_ok=True)
    
    # Write config to YAML file
    # Ensure None values are explicitly included (yaml.dump includes them by default, but being explicit)
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    # Prepare command - run main.py directly from protify directory
    # This is necessary because main.py uses relative imports (from probes, base_models, etc.)
    # that expect to be run from within the protify directory
    # Add -u flag for unbuffered output
    cmd = [
        "python", "-u", "main.py",
        "--yaml_path", str(config_path)
    ]
    
    if hf_token:
        cmd.extend(["--hf_token", hf_token])
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Configuration summary:")
    print(f"  - Models: {config.get('model_names', 'N/A')}")
    print(f"  - Datasets: {config.get('data_names', 'N/A')}")
    print(f"  - Probe type: {config.get('probe_type', 'N/A')}")
    print(f"  - Training mode: {'Full finetuning' if config.get('full_finetuning') else 'Hybrid probe' if config.get('hybrid_probe') else 'Scikit' if config.get('use_scikit') else 'NN Probe'}")
    print(f"  - Save model: {config.get('save_model', False)}")
    print(f"  - HF Username: {config.get('hf_username', 'NOT SET')}")
    print(f"  - HF Token provided: {'Yes' if hf_token else 'No'}")
    
    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/src"
    env["WORKING_DIR"] = "/root"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"  # Ensure Python output is unbuffered
    # CRITICAL: Override HF_TOKEN in subprocess environment if user provided token
    # This ensures the subprocess uses the user's token, not the secret token
    if hf_token:
        env["HF_TOKEN"] = hf_token
        print(f"Set HF_TOKEN in subprocess environment to user-provided token")
    # Pass job_id to Protify so it uses it as random_id
    if job_id:
        env["PROTIFY_JOB_ID"] = job_id
    
    # Change to protify directory where main.py expects to be run from
    # This allows the relative imports (from probes, base_models, etc.) to work correctly
    os.chdir("/root/src/protify")
    
    # Run Protify with streaming output
    stdout_lines = []
    stderr_lines = []
    
    def stream_output(pipe, output_list, prefix=""):
        """Stream output from a pipe line by line."""
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    line_str = line.rstrip()
                    output_list.append(line_str)
                    # Print immediately to Modal logs
                    if prefix:
                        print(f"{prefix}{line_str}", flush=True)
                    else:
                        print(line_str, flush=True)
        except Exception as e:
            print(f"Error streaming output: {e}", flush=True)
        finally:
            pipe.close()
    
    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            cwd="/root/src/protify",
            env=env
        )
        
        # Start threads to stream stdout and stderr
        stdout_thread = threading.Thread(
            target=stream_output,
            args=(process.stdout, stdout_lines, ""),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=stream_output,
            args=(process.stderr, stderr_lines, "[STDERR] "),
            daemon=True
        )
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete with timeout
        start_time = time.time()
        timeout_seconds = TIMEOUT - 600
        
        while process.poll() is None:
            if time.time() - start_time > timeout_seconds:
                process.kill()
                process.wait()
                raise subprocess.TimeoutExpired(cmd, timeout_seconds)
            time.sleep(0.1)
        
        # Wait for output threads to finish
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        
        returncode = process.returncode
        stdout_text = "\n".join(stdout_lines)
        stderr_text = "\n".join(stderr_lines)
        
        if returncode != 0:
            # Update job status to FINISHED
            if job_id:
                hf_username = config.get("hf_username")
                update_job_storage(job_id, "FINISHED", hf_username)
            
            return {
                "success": False,
                "error": stderr_text[-5000:] if stderr_text else "Unknown error",
                "stdout": stdout_text[-5000:] if stdout_text else ""
            }
        
        # This Collect results
        results_dir = Path(config.get("results_dir", "/data/results"))
        plots_dir = Path(config.get("plots_dir", "/data/plots"))
        logs_dir = Path(config.get("log_dir", "/data/logs"))
        
        # Find the most recent result files
        result_files = []
        if results_dir.exists():
            for file in results_dir.glob("*.tsv"):
                result_files.append(str(file))
        
        # Find plot directories
        plot_dirs = []
        if plots_dir.exists():
            for item in plots_dir.iterdir():
                if item.is_dir():
                    plot_dirs.append(str(item))
        
        # Find log files
        log_files = []
        if logs_dir.exists():
            for file in logs_dir.glob("*.txt"):
                log_files.append(str(file))
        
        # Update job status to FINISHED
        if job_id:
            hf_username = config.get("hf_username")
            update_job_storage(job_id, "FINISHED", hf_username)
        
        return {
            "success": True,
            "result_files": result_files,
            "plot_dirs": plot_dirs,
            "log_files": log_files,
            "stdout": stdout_text[-5000:] if stdout_text else "",
            "job_id": job_id
        }
        
    except subprocess.TimeoutExpired:
        # Update job status to FINISHED
        if job_id:
            hf_username = config.get("hf_username")
            update_job_storage(job_id, "FINISHED", hf_username)
        
        return {
            "success": False,
            "error": f"Process timed out after {TIMEOUT // 60} minutes",
            "stdout": ""
        }
    except Exception as e:
        # Update job status to FINISHED
        if job_id:
            hf_username = config.get("hf_username") if config else None
            update_job_storage(job_id, "FINISHED", hf_username)
        
        return {
            "success": False,
            "error": str(e),
            "stdout": ""
        }


# Dictionary to store GPU-specific functions
gpu_functions = {}


# Create a Modal function for each GPU type at global scope
@app.function(
    image=image,
    gpu="H200",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("hf_read_token")],
    memory=(PROTIFY_MEMORY_MIN_GPU, PROTIFY_MEMORY_MAX_GPU),
    cpu=(PROTIFY_CPU_MIN_GPU, PROTIFY_CPU_MAX_GPU),
    max_containers=MAX_CONTAINERS_PROTIFY,
    scaledown_window=SCALEDOWN_WINDOW,
    enable_memory_snapshot=ENABLE_MEMORY_SNAPSHOT,
    timeout=TIMEOUT,
)
def run_protify_job_h200(
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    return _execute_protify_job(config, hf_token, job_id)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("hf_read_token")],
    memory=(PROTIFY_MEMORY_MIN_GPU, PROTIFY_MEMORY_MAX_GPU),
    cpu=(PROTIFY_CPU_MIN_GPU, PROTIFY_CPU_MAX_GPU),
    max_containers=MAX_CONTAINERS_PROTIFY,
    scaledown_window=SCALEDOWN_WINDOW,
    enable_memory_snapshot=ENABLE_MEMORY_SNAPSHOT,
    timeout=TIMEOUT,
)
def run_protify_job_h100(
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    return _execute_protify_job(config, hf_token, job_id)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("hf_read_token")],
    memory=(PROTIFY_MEMORY_MIN_GPU, PROTIFY_MEMORY_MAX_GPU),
    cpu=(PROTIFY_CPU_MIN_GPU, PROTIFY_CPU_MAX_GPU),
    max_containers=MAX_CONTAINERS_PROTIFY,
    scaledown_window=SCALEDOWN_WINDOW,
    enable_memory_snapshot=ENABLE_MEMORY_SNAPSHOT,
    timeout=TIMEOUT,
)
def run_protify_job_a100_80gb(
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    return _execute_protify_job(config, hf_token, job_id)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("hf_read_token")],
    memory=(PROTIFY_MEMORY_MIN_GPU, PROTIFY_MEMORY_MAX_GPU),
    cpu=(PROTIFY_CPU_MIN_GPU, PROTIFY_CPU_MAX_GPU),
    max_containers=MAX_CONTAINERS_PROTIFY,
    scaledown_window=SCALEDOWN_WINDOW,
    enable_memory_snapshot=ENABLE_MEMORY_SNAPSHOT,
    timeout=TIMEOUT,
)
def run_protify_job_a100(
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    return _execute_protify_job(config, hf_token, job_id)


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("hf_read_token")],
    memory=(PROTIFY_MEMORY_MIN_GPU, PROTIFY_MEMORY_MAX_GPU),
    cpu=(PROTIFY_CPU_MIN_GPU, PROTIFY_CPU_MAX_GPU),
    max_containers=MAX_CONTAINERS_PROTIFY,
    scaledown_window=SCALEDOWN_WINDOW,
    enable_memory_snapshot=ENABLE_MEMORY_SNAPSHOT,
    timeout=TIMEOUT,
)
def run_protify_job_l40s(
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    return _execute_protify_job(config, hf_token, job_id)


@app.function(
    image=image,
    gpu="A10",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("hf_read_token")],
    memory=(PROTIFY_MEMORY_MIN_GPU, PROTIFY_MEMORY_MAX_GPU),
    cpu=(PROTIFY_CPU_MIN_GPU, PROTIFY_CPU_MAX_GPU),
    max_containers=MAX_CONTAINERS_PROTIFY,
    scaledown_window=SCALEDOWN_WINDOW,
    enable_memory_snapshot=ENABLE_MEMORY_SNAPSHOT,
    timeout=TIMEOUT,
)
def run_protify_job_a10(
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    return _execute_protify_job(config, hf_token, job_id)


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("hf_read_token")],
    memory=(PROTIFY_MEMORY_MIN_GPU, PROTIFY_MEMORY_MAX_GPU),
    cpu=(PROTIFY_CPU_MIN_GPU, PROTIFY_CPU_MAX_GPU),
    max_containers=MAX_CONTAINERS_PROTIFY,
    scaledown_window=SCALEDOWN_WINDOW,
    enable_memory_snapshot=ENABLE_MEMORY_SNAPSHOT,
    timeout=TIMEOUT,
)
def run_protify_job_l4(
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    return _execute_protify_job(config, hf_token, job_id)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("hf_read_token")],
    memory=(PROTIFY_MEMORY_MIN_GPU, PROTIFY_MEMORY_MAX_GPU),
    cpu=(PROTIFY_CPU_MIN_GPU, PROTIFY_CPU_MAX_GPU),
    max_containers=MAX_CONTAINERS_PROTIFY,
    scaledown_window=SCALEDOWN_WINDOW,
    enable_memory_snapshot=ENABLE_MEMORY_SNAPSHOT,
    timeout=TIMEOUT,
)
def run_protify_job_t4(
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    return _execute_protify_job(config, hf_token, job_id)


# Map GPU types to their corresponding functions
gpu_functions = {
    "H200": run_protify_job_h200,
    "H100": run_protify_job_h100,
    "A100-80GB": run_protify_job_a100_80gb,
    "A100": run_protify_job_a100,
    "L40S": run_protify_job_l40s,
    "A10": run_protify_job_a10,
    "L4": run_protify_job_l4,
    "T4": run_protify_job_t4,
}

# Default function (for backward compatibility)
run_protify_job = gpu_functions.get(PROTIFY_DEFAULT_GPU, run_protify_job_a100)


@app.function(
    image=image,
    volumes={"/data": volume},
    memory=(CPU_FUNCTION_MEMORY_MIN, CPU_FUNCTION_MEMORY_MAX),
    cpu=(CPU_FUNCTION_CPU_MIN, CPU_FUNCTION_CPU_MAX),
    max_containers=MAX_CONTAINERS_CPU_FUNCTIONS,
    scaledown_window=SCALEDOWN_WINDOW,
)
def get_results(job_id: str) -> Dict[str, Any]:
    """Retrieve results from a previous run."""
    import base64
    
    results = {"success": True, "files": {}, "images": {}}
    
    # Image file extensions
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"}
    
    # Use job_id (which is the random_id) to find files directly
    # Files are named: {job_id}.tsv, {job_id}.txt, and plots are in plots/{job_id}/
    results_dir = Path("/data/results")
    plots_dir = Path("/data/plots")
    logs_dir = Path("/data/logs")
    
    # Collect files matching this job_id
    all_file_paths = set()
    
    # Result file: {job_id}.tsv
    result_file = results_dir / f"{job_id}.tsv"
    if result_file.exists():
        all_file_paths.add(result_file)
    
    # Log file: {job_id}.txt
    log_file = logs_dir / f"{job_id}.txt"
    if log_file.exists():
        all_file_paths.add(log_file)
    
    # Plot directory: plots/{job_id}/
    plot_dir = plots_dir / job_id
    if plot_dir.exists() and plot_dir.is_dir():
        for file_path in plot_dir.rglob("*"):
            if file_path.is_file():
                all_file_paths.add(file_path)
    
    # Process all collected files
    for file_path in all_file_paths:
        rel_path = str(file_path.relative_to(Path("/data")))
        file_ext = file_path.suffix.lower()
        
        try:
            if file_ext in image_extensions:
                # Read image as binary and encode as base64
                with open(file_path, "rb") as f:
                    image_data = f.read()
                base64_data = base64.b64encode(image_data).decode("utf-8")
                results["images"][rel_path] = {
                    "data": base64_data,
                    "mime_type": f"image/{file_ext[1:]}" if file_ext != ".svg" else "image/svg+xml"
                }
            else:
                # Read text files normally - return full content (no size limit)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                results["files"][rel_path] = content
        except Exception as e:
            if file_ext in image_extensions:
                results["images"][rel_path] = {"error": f"Error reading image: {str(e)}"}
            else:
                results["files"][rel_path] = f"Error reading file: {str(e)}"
    
    return results


@app.function(
    image=image,
    volumes={"/data": volume},
    memory=(CPU_FUNCTION_MEMORY_MIN, CPU_FUNCTION_MEMORY_MAX),
    cpu=(CPU_FUNCTION_CPU_MIN, CPU_FUNCTION_CPU_MAX),
    max_containers=MAX_CONTAINERS_CPU_FUNCTIONS,
    scaledown_window=SCALEDOWN_WINDOW,
)
def list_jobs() -> Dict[str, Any]:
    """List all available jobs/results."""
    results_dir = Path("/data/results")
    if not results_dir.exists():
        return {"success": True, "jobs": []}
    
    jobs = []
    for file in results_dir.glob("*.tsv"):
        mtime = file.stat().st_mtime
        jobs.append({
            "job_id": file.stem,
            "created_at": mtime,
            "file": str(file)
        })
    
    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    return {"success": True, "jobs": jobs}


# Web interface using Gradio
@app.function(
    image=image,
    volumes={"/data": volume},
    memory=(CPU_FUNCTION_MEMORY_MIN, CPU_FUNCTION_MEMORY_MAX),
    cpu=(CPU_FUNCTION_CPU_MIN, CPU_FUNCTION_CPU_MAX),
    max_containers=1,
    scaledown_window=WEB_INTERFACE_SCALEDOWN_WINDOW,
)
@modal.asgi_app()
def web_interface():
    """Serve a Gradio web interface for Protify."""
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    import time
    from datetime import datetime
    import html
    
    web_app = FastAPI()
    
    # Initialize job storage
    job_storage_file = "/data/job_storage.json"
    
    # Reset job_storage.json on deploy
    try:
        if os.path.exists(job_storage_file):
            os.remove(job_storage_file)
            print("Reset job_storage.json on deploy")
        # Initialize empty job storage
        with open(job_storage_file, 'w') as f:
            json.dump({}, f)
        volume.commit()
    except Exception as e:
        print(f"Warning: Could not reset job_storage.json: {e}")
    
    def load_job_storage():
        """Load job storage from volume."""
        if os.path.exists(job_storage_file):
            try:
                with open(job_storage_file, 'r') as f:
                    data = json.load(f)
                    # Debug: print statuses to help diagnose
                    print(f"Loaded job storage with {len(data)} jobs")
                    for job_id, job_info in data.items():
                        print(f"  Job {job_id}: status={job_info.get('status', 'MISSING')}")
                    return data
            except Exception as e:
                print(f"Error loading job storage: {e}")
                import traceback
                traceback.print_exc()
        return {}
    
    def save_job_storage(job_storage):
        """Save job storage to volume with volume commit."""
        try:
            with open(job_storage_file, 'w') as f:
                json.dump(job_storage, f, default=str)
            # CRITICAL: Commit volume to persist changes
            volume.commit()
        except Exception as e:
            print(f"Warning: Failed to save job storage: {e}")
    
    # Model and dataset lists (from Protify source)
    # All currently supported models from src/protify/base_models/get_base_models.py
    STANDARD_MODELS = [
        'ESM2-8', 'ESM2-35', 'ESM2-150', 'ESM2-650', 'ESM2-3B',
        'ESMC-300', 'ESMC-600', 
        'ESM2-diff-150', 'ESM2-diffAV-150',
        'ProtBert', 'ProtBert-BFD', 
        'ProtT5', 'ProtT5-XL-UniRef50-full-prec', 'ProtT5-XXL-UniRef50', 
        'ProtT5-XL-BFD', 'ProtT5-XXL-BFD',
        'ANKH-Base', 'ANKH-Large', 'ANKH2-Large',
        'GLM2-150', 'GLM2-650', 'GLM2-GAIA',
        'DPLM-150', 'DPLM-650', 'DPLM-3B',
        'DSM-150', 'DSM-650', 'DSM-PPI',
        'ProtCLM-1b',
        'Random', 'Random-Transformer', 
        'Random-ESM2-8', 'Random-ESM2-35', 'Random-ESM2-150', 'Random-ESM2-650',
        'OneHot-Protein', 'OneHot-DNA', 'OneHot-RNA', 'OneHot-Codon'
    ]
    
    # All supported datasets from src/protify/data/supported_datasets.py
    STANDARD_DATASETS = [
        'EC', 'GO-CC', 'GO-BP', 'GO-MF', 'MB',
        'DeepLoc-2', 'DeepLoc-10', 'Subcellular',
        'enzyme-kcat', 'solubility', 'localization', 
        'temperature-stability', 'peptide-HLA-MHC-affinity',
        'optimal-temperature', 'optimal-ph', 'material-production',
        'fitness-prediction', 'number-of-folds', 'cloning-clf',
        'stability-prediction', 'human-ppi-saprot', 
        'SecondaryStructure-3', 'SecondaryStructure-8', 
        'fluorescence-prediction', 'plastic',
        'gold-ppi', 'human-ppi-pinui', 'yeast-ppi-pinui',
        'shs27-ppi-raw', 'shs148-ppi-raw',
        'shs27-ppi-random', 'shs148-ppi-random',
        'shs27-ppi-dfs', 'shs148-ppi-dfs',
        'shs27-ppi-bfs', 'shs148-ppi-bfs',
        'string-ppi-random', 'string-ppi-dfs', 'string-ppi-bfs',
        'plm-interact', 'ppi-mutation-effect', 'PPA-ppi',
        'foldseek-fold', 'foldseek-inverse',
        'ec-active',
        'taxon_domain', 'taxon_kingdom', 'taxon_phylum', 'taxon_class',
        'taxon_order', 'taxon_family', 'taxon_genus', 'taxon_species',
        'diff_phylogeny', 'plddt', 'realness', 'million_full'
    ]
    
    def form_values_to_config(*form_values) -> str:
        """Convert form values to YAML config string."""
        (
            # Info tab
            hf_username, hf_token, wandb_api_key, synthyra_api_key,
            home_dir, hf_home, log_dir, results_dir, model_save_dir,
            plots_dir, embedding_save_dir, download_dir,
            # Data tab
            max_length, trim, delimiter, col_names, multi_column, data_names_selected, data_names_custom,
            # Model tab
            model_names_selected, model_names_custom, gpu_type,
            # Embed tab
            embedding_batch_size, num_workers, download_embeddings,
            matrix_embed, embedding_pooling_types, embed_dtype, sql,
            # Probe tab
            probe_type, tokenwise, pre_ln, n_layers, hidden_size, dropout,
            classifier_size, classifier_dropout, n_heads, rotary,
            probe_pooling_types, transformer_dropout, token_attention,
            sim_type, save_model, production_model, lora, lora_r,
            lora_alpha, lora_dropout,
            # Trainer tab
            hybrid_probe, full_finetuning, num_epochs, probe_batch_size,
            base_batch_size, probe_grad_accum, base_grad_accum, lr,
            weight_decay, patience, seed, read_scaler, deterministic,
            # ProteinGym tab
            proteingym, dms_ids, mode, scoring_method, scoring_window,
            pg_batch_size, compare_scoring_methods,
            # Scikit tab
            use_scikit, scikit_n_iter, scikit_cv, scikit_random_state,
            scikit_model_name, n_jobs,
            # Job info
            job_name
        ) = form_values
        
        # Helper functions
        def str_to_list(val):
            if not val or val.strip() == "":
                return []
            return [item.strip() for item in str(val).split(",") if item.strip()]
        
        def str_to_none(val):
            if not val or str(val).strip() == "":
                return None
            return str(val).strip()
        
        def str_to_username(val):
            """Convert username value, treating placeholder as None."""
            if not val or str(val).strip() == "":
                return None
            username = str(val).strip()
            # Treat placeholder as None
            if username == "YOUR_HF_USERNAME":
                return None
            return username
        
        def str_to_int(val):
            if not val or str(val).strip() == "":
                return None
            try:
                return int(val)
            except:
                return None
        
        # Build config dictionary
        config = {
            # Info
            "hf_username": str_to_username(hf_username) or "Synthyra",  # Default to Synthyra if not provided (matching main.py default)
            "hf_token": str_to_none(hf_token),
            "wandb_api_key": str_to_none(wandb_api_key),
            "synthyra_api_key": str_to_none(synthyra_api_key),
            "home_dir": home_dir or "/root",
            "hf_home": str_to_none(hf_home),
            "log_dir": log_dir or "logs",
            "results_dir": results_dir or "results",
            "model_save_dir": model_save_dir or "weights",
            "plots_dir": plots_dir or "plots",
            "embedding_save_dir": embedding_save_dir or "embeddings",
            "download_dir": download_dir or "Synthyra/vector_embeddings",
            # Data
            "max_length": int(max_length) if max_length else 2048,
            "trim": bool(trim),
            "delimiter": delimiter or ",",
            "col_names": str_to_list(col_names) if col_names else ["seqs", "labels"],
            "multi_column": str_to_list(multi_column) if multi_column else None,
            # Combine selected datasets and custom datasets
            "data_names": (
                (list(data_names_selected) if data_names_selected else []) +
                (str_to_list(data_names_custom) if data_names_custom else [])
            ) or ["DeepLoc-2"],  # Default if both are empty
            "data_dirs": [],
            # Model
            # Combine selected models and custom models
            "model_names": (
                (list(model_names_selected) if model_names_selected else []) +
                (str_to_list(model_names_custom) if model_names_custom else [])
            ) or ["ESM2-8"],  # Default if both are empty
            # Embed
            "embedding_batch_size": int(embedding_batch_size) if embedding_batch_size else 16,
            "num_workers": int(num_workers) if num_workers else 0,
            "download_embeddings": bool(download_embeddings),
            "matrix_embed": bool(matrix_embed),
            "embedding_pooling_types": str_to_list(embedding_pooling_types) if embedding_pooling_types else ["mean"],
            "embed_dtype": embed_dtype or "float32",
            "sql": bool(sql),
            "save_embeddings": True,  # Always save embeddings
            # Probe
            "probe_type": probe_type or "linear",
            "tokenwise": bool(tokenwise),
            "pre_ln": bool(pre_ln),
            "n_layers": int(n_layers) if n_layers else 1,
            "hidden_size": int(hidden_size) if hidden_size else 8192,
            "dropout": float(dropout) if dropout else 0.2,
            "classifier_size": int(classifier_size) if classifier_size else 4096,
            "classifier_dropout": float(classifier_dropout) if classifier_dropout else 0.2,
            "n_heads": int(n_heads) if n_heads else 4,
            "rotary": bool(rotary),
            "probe_pooling_types": str_to_list(probe_pooling_types) if probe_pooling_types else ["mean", "var"],
            "transformer_dropout": float(transformer_dropout) if transformer_dropout else 0.1,
            "token_attention": bool(token_attention),
            "sim_type": sim_type or "dot",
            "save_model": bool(save_model),
            "production_model": bool(production_model),
            "lora": bool(lora),
            "lora_r": int(lora_r) if lora_r else 8,
            "lora_alpha": float(lora_alpha) if lora_alpha else 32.0,
            "lora_dropout": float(lora_dropout) if lora_dropout else 0.01,
            # Trainer
            "hybrid_probe": bool(hybrid_probe),
            "full_finetuning": bool(full_finetuning),
            "num_epochs": int(num_epochs) if num_epochs else 200,
            "probe_batch_size": int(probe_batch_size) if probe_batch_size else 64,
            "base_batch_size": int(base_batch_size) if base_batch_size else 4,
            "probe_grad_accum": int(probe_grad_accum) if probe_grad_accum else 1,
            "base_grad_accum": int(base_grad_accum) if base_grad_accum else 8,
            "lr": float(lr) if lr else 1e-4,
            "weight_decay": float(weight_decay) if weight_decay else 0.0,
            "patience": int(patience) if patience else 1,
            "seed": str_to_int(seed),
            "read_scaler": int(read_scaler) if read_scaler else 100,
            "read_scaler": int(read_scaler) if read_scaler else 100,
            "deterministic": bool(deterministic),
            # ProteinGym
            "proteingym": bool(proteingym),
            "dms_ids": str_to_list(dms_ids) if dms_ids else ["all"],
            "mode": str(mode) if mode else "benchmark",
            "scoring_method": str(scoring_method) if scoring_method else "masked_marginal",
            "scoring_window": str(scoring_window) if scoring_window else "optimal",
            "pg_batch_size": int(pg_batch_size) if pg_batch_size else 32,
            "compare_scoring_methods": bool(compare_scoring_methods),
            # Scikit
            "use_scikit": bool(use_scikit),
            "scikit_n_iter": int(scikit_n_iter) if scikit_n_iter else 10,
            "scikit_cv": int(scikit_cv) if scikit_cv else 3,
            "scikit_random_state": str_to_int(scikit_random_state),
            "scikit_model_name": str_to_none(scikit_model_name),
            "n_jobs": int(n_jobs) if n_jobs else 1,
        }
        
        return yaml.dump(config, default_flow_style=False, sort_keys=False)
    
    def submit_job(*form_values):
        """Submit a new Protify job."""
        import random
        import string
        from datetime import datetime
        
        try:
            # Extract job_name (last value)
            job_name = form_values[-1]
            config_values = form_values[:-1]
            
            # Get hf_username (first value) and hf_token (second value)
            hf_username = form_values[0] if len(form_values) > 0 else None
            hf_token = form_values[1] if len(form_values) > 1 else None
            
            # Get GPU type (after model_names, which is after data settings)
            # Counting: hf_username(0), hf_token(1), wandb_api_key(2), synthyra_api_key(3),
            # home_dir(4), hf_home(5), log_dir(6), results_dir(7), model_save_dir(8),
            # plots_dir(9), embedding_save_dir(10), download_dir(11),
            # max_length(12), trim(13), delimiter(14), col_names(15), multi_column(16), 
            # data_names_selected(17), data_names_custom(18),
            # model_names_selected(19), model_names_custom(20), gpu_type(21)
            gpu_type = form_values[21] if len(form_values) > 21 else PROTIFY_DEFAULT_GPU
            if not gpu_type or gpu_type not in AVAILABLE_GPUS:
                gpu_type = PROTIFY_DEFAULT_GPU
            
            # Convert form values to config
            yaml_content = form_values_to_config(*form_values)
            config = yaml.safe_load(yaml_content)
            
            # Generate job ID in the same format as logger.py (date + 4 random letters)
            random_letters = ''.join(random.choices(string.ascii_uppercase, k=4))
            date_str = datetime.now().strftime('%Y-%m-%d-%H-%M')
            job_id = f"{date_str}_{random_letters}"
            timestamp = time.time()
            
            # Get the appropriate GPU function
            selected_gpu_function = gpu_functions.get(gpu_type, gpu_functions[PROTIFY_DEFAULT_GPU])
            
            # Submit the job asynchronously
            handle = selected_gpu_function.spawn(
                config=config,
                hf_token=hf_token if hf_token else None,
                job_id=job_id
            )
            
            # Store job metadata - id, status, and hf_username
            job_storage = load_job_storage()
            job_storage[job_id] = {
                "id": job_id,
                "status": "RUNNING",
                "hf_username": hf_username or "Unknown"
            }
            save_job_storage(job_storage)
            
            return f"✅ Job submitted successfully! Job ID: {job_id} (GPU: {gpu_type})"
        except Exception as e:
            import traceback
            return f"❌ Error: {str(e)}\n{traceback.format_exc()}"
    
    def format_job_queue():
        """Format the job queue as HTML."""
        # Reload job storage to get latest updates
        job_storage = load_job_storage()
        if not job_storage:
            return "No jobs submitted yet."
        
        jobs = []
        for job_id, job_info in job_storage.items():
            # Get status from job storage (updated by GPU functions)
            # job_info has "id", "status", and optionally "hf_username" keys
            status_raw = job_info.get("status", "RUNNING")
            hf_username = job_info.get("hf_username", "Unknown")
            
            # Normalize status - ensure it's uppercase and valid
            status_upper = str(status_raw).strip().upper()
            
            # Map various status values to standard ones
            if status_upper in ["FINISHED", "COMPLETED", "DONE", "SUCCESS"]:
                status = "FINISHED"
            elif status_upper in ["RUNNING", "IN_PROGRESS", "PENDING"]:
                status = "RUNNING"
            else:
                # Default to FINISHED for unknown statuses (assume job completed)
                status = "FINISHED"
            
            jobs.append({
                "job_id": job_id,
                "status": status,
                "hf_username": hf_username
            })
        
        # Sort by job_id (which contains timestamp) in reverse order
        jobs.sort(key=lambda x: x["job_id"], reverse=True)
        
        html_content = "<div style='display: flex; flex-direction: column; gap: 15px;'>"
        for job in jobs:
            status = job["status"]  # Already normalized to RUNNING or FINISHED
            job_id = job["job_id"]
            hf_username = job.get("hf_username", "Unknown")
            
            # Extract timestamp from job_id (format: YYYY-MM-DD-HH-MM_XXXX)
            try:
                date_part = job_id.split('_')[0]  # Get YYYY-MM-DD-HH-MM
                # Parse and format for display
                from datetime import datetime
                dt = datetime.strptime(date_part, '%Y-%m-%d-%H-%M')
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                time_str = job_id
            
            # Status color mapping - only RUNNING and FINISHED
            status_colors = {
                "RUNNING": "#3b82f6",  # blue
                "FINISHED": "#10b981",  # green
            }
            status_color = status_colors.get(status, "#6b7280")  # default gray for any unexpected status
            
            # Create unique button ID for this job
            button_id = f"copy-btn-{job_id.replace('-', '').replace('_', '')}"
            # Escape job_id for use in HTML attributes
            job_id_escaped = html.escape(job_id, quote=True)
            html_content += f"""
            <div style='padding: 20px; margin: 10px 0; border-radius: 8px; border: 2px solid #e5e7eb; background: white; min-width: 600px;'>
                <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;'>
                    <div style='flex: 1;'>
                        <strong style='color: #1a1a1a; font-size: 18px;'>Job {job_id}</strong>
                        <br><span style='color: #6b7280; font-size: 14px;'>ID: {job_id}</span>
                        <br><span style='color: #6b7280; font-size: 14px;'>User: {hf_username}</span>
                    </div>
                    <div style='display: flex; gap: 10px; align-items: center;'>
                        <button class="copy-job-id-btn" data-job-id="{job_id_escaped}" id="{button_id}" style='padding: 6px 12px; border-radius: 6px; border: 1px solid #d1d5db; background: white; color: #374151; font-size: 12px; cursor: pointer; transition: all 0.2s;' onmouseover="this.style.background='#f3f4f6'; this.style.borderColor='#9ca3af';" onmouseout="this.style.background='white'; this.style.borderColor='#d1d5db';" title="Copy Job ID">
                            📋 Copy ID
                        </button>
                        <span style='padding: 8px 16px; border-radius: 6px; background: {status_color}; color: white; font-weight: bold; font-size: 14px; min-width: 100px; text-align: center;'>{status}</span>
                    </div>
                </div>
                <div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid #e5e7eb;'>
                    <small style='color: #6b7280;'>Submitted: {time_str}</small>
                </div>
            </div>
            """
        html_content += "</div>"
        # Add JavaScript function for copying job IDs using event delegation
        # Use a unique ID to prevent multiple script executions
        script_id = f"copy-script-{int(time.time())}"
        html_content += f"""
        <script id="{script_id}">
        (function() {{
            // Use event delegation on document body to handle dynamically added buttons
            // This works even when Gradio updates the HTML
            function handleCopyClick(e) {{
                // Check if clicked element or its parent is a copy button
                let button = e.target;
                while (button && !button.classList.contains('copy-job-id-btn')) {{
                    button = button.parentElement;
                }}
                
                if (!button || !button.classList.contains('copy-job-id-btn')) {{
                    return;
                }}
                
                e.preventDefault();
                e.stopPropagation();
                
                const jobId = button.getAttribute('data-job-id');
                if (!jobId) {{
                    console.error('No job-id attribute found');
                    return;
                }}
                
                // Try modern clipboard API first
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(jobId).then(function() {{
                        updateButtonSuccess(button);
                    }}).catch(function(err) {{
                        console.error('Clipboard API failed:', err);
                        fallbackCopy(jobId, button);
                    }});
                }} else {{
                    fallbackCopy(jobId, button);
                }}
            }}
            
            function updateButtonSuccess(button) {{
                const originalText = button.innerHTML;
                button.innerHTML = '✓ Copied!';
                button.style.background = '#10b981';
                button.style.borderColor = '#10b981';
                button.style.color = 'white';
                setTimeout(function() {{
                    button.innerHTML = originalText;
                    button.style.background = 'white';
                    button.style.borderColor = '#d1d5db';
                    button.style.color = '#374151';
                }}, 2000);
            }}
            
            function fallbackCopy(text, button) {{
                const textArea = document.createElement('textarea');
                textArea.value = text;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                textArea.style.top = '-999999px';
                textArea.style.opacity = '0';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                try {{
                    const successful = document.execCommand('copy');
                    if (successful) {{
                        updateButtonSuccess(button);
                    }} else {{
                        alert('Failed to copy. Please manually copy: ' + text);
                    }}
                }} catch (err) {{
                    console.error('execCommand failed:', err);
                    alert('Failed to copy. Please manually copy: ' + text);
                }}
                document.body.removeChild(textArea);
            }}
            
            // Remove any existing listeners to prevent duplicates
            document.removeEventListener('click', handleCopyClick);
            // Attach event listener to document body for event delegation
            document.addEventListener('click', handleCopyClick, true);
        }})();
        </script>
        """
        return html_content
    
    def refresh_queue():
        """Refresh the job queue display."""
        return format_job_queue()
    
    def clear_pending_jobs():
        """Clear all completed/finished jobs from the job storage."""
        job_storage = load_job_storage()
        if not job_storage:
            return "No jobs to clear.", format_job_queue()
        
        # Filter out finished/completed jobs
        initial_count = len(job_storage)
        job_storage = {
            job_id: job_info 
            for job_id, job_info in job_storage.items() 
            if job_info.get("status", "").upper() not in ["FINISHED", "COMPLETED", "DONE"]
        }
        cleared_count = initial_count - len(job_storage)
        
        save_job_storage(job_storage)
        
        if cleared_count > 0:
            message = f"✅ Cleared {cleared_count} completed job(s)."
        else:
            message = "ℹ️ No completed jobs to clear."
        
        return message, format_job_queue()
    
    # Load Protify logo and convert to base64
    import base64
    logo_path = "/root/src/protify/protify_logo.png"
    logo_base64 = ""
    try:
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                logo_data = f.read()
                logo_base64 = base64.b64encode(logo_data).decode("utf-8")
    except Exception as e:
        print(f"Warning: Could not load logo: {e}")
    
    # Create header HTML with logo
    if logo_base64:
        header_html = f'<div style="display: flex; align-items: center; gap: 15px;"><img src="data:image/png;base64,{logo_base64}" style="height: 60px; width: auto;" /><h1 style="margin: 0;">Protify - Molecular property prediction made easy</h1></div>'
    else:
        header_html = "# Protify - Molecular property prediction made easy"
    
    # Create Gradio interface
    with gr.Blocks(title="Protify - Molecular property prediction made easy") as interface:
        gr.HTML(header_html)
        gr.Markdown("Configure your experiment and submit jobs to run on GPU.")
        
        with gr.Tabs():
            # Tab 1: Configuration
            with gr.Tab("⚙️ Configure Job"):
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Accordion("🔐 Authentication & Paths", open=True):
                            hf_username = gr.Textbox(label="HuggingFace Username", value="", placeholder="Enter your HuggingFace username (required for saving models)")
                            hf_token = gr.Textbox(label="HuggingFace Token", type="password", value="")
                            wandb_api_key = gr.Textbox(label="Wandb API Key (optional)", type="password", value="")
                            synthyra_api_key = gr.Textbox(label="Synthyra API Key (optional)", type="password", value="")
                            
                            home_dir = gr.Textbox(label="Home Directory", value="/root")
                            hf_home = gr.Textbox(label="HF Home Directory (optional)", value="")
                            log_dir = gr.Textbox(label="Log Directory", value="logs")
                            results_dir = gr.Textbox(label="Results Directory", value="results")
                            model_save_dir = gr.Textbox(label="Model Save Directory", value="weights")
                            plots_dir = gr.Textbox(label="Plots Directory", value="plots")
                            embedding_save_dir = gr.Textbox(label="Embedding Save Directory", value="embeddings")
                            download_dir = gr.Textbox(label="Download Directory", value="Synthyra/vector_embeddings")
                        
                        with gr.Accordion("📊 Data Settings", open=True):
                            max_length = gr.Number(label="Max Sequence Length", value=2048, precision=0)
                            trim = gr.Checkbox(label="Trim Sequences", value=False)
                            delimiter = gr.Textbox(label="Delimiter", value=",")
                            col_names = gr.Textbox(label="Column Names (comma-separated)", value="seqs,labels")
                            multi_column = gr.Textbox(label="Multi-Column Sequences (space-separated, optional)", value="")
                            data_names_selected = gr.CheckboxGroup(
                                label="Select Standard Datasets",
                                choices=STANDARD_DATASETS,
                                value=["DeepLoc-2"],
                                info="Select one or more standard datasets"
                            )
                            data_names_custom = gr.Textbox(
                                label="Custom Dataset Names (comma-separated, optional)",
                                value="",
                                placeholder="Enter custom dataset names separated by commas",
                                info="Add any custom dataset names not in the standard list above"
                            )
                        
                        with gr.Accordion("🤖 Model Settings", open=True):
                            model_names_selected = gr.CheckboxGroup(
                                label="Select Standard Models",
                                choices=STANDARD_MODELS,
                                value=["ESM2-8"],
                                info="Select one or more standard models"
                            )
                            model_names_custom = gr.Textbox(
                                label="Custom Model Names (comma-separated, optional)",
                                value="",
                                placeholder="Enter custom model names separated by commas",
                                info="Add any custom model names not in the standard list above"
                            )
                        
                        with gr.Accordion("🔢 Embedding Settings", open=False):
                            embedding_batch_size = gr.Number(label="Batch Size", value=16, precision=0)
                            num_workers = gr.Number(label="Num Workers", value=0, precision=0)
                            download_embeddings = gr.Checkbox(label="Download Embeddings", value=False)
                            matrix_embed = gr.Checkbox(label="Matrix Embedding", value=False)
                            embedding_pooling_types = gr.Textbox(label="Pooling Types (comma-separated)", value="mean")
                            embed_dtype = gr.Dropdown(choices=["float32", "float16", "bfloat16"], value="float32")
                            sql = gr.Checkbox(label="Use SQL Storage", value=False)
                        
                        with gr.Accordion("🔍 Probe Settings", open=False):
                            probe_type = gr.Dropdown(choices=["linear", "transformer", "retrievalnet", "lyra"], value="linear")
                            tokenwise = gr.Checkbox(label="Tokenwise", value=False)
                            pre_ln = gr.Checkbox(label="Pre Layer Norm", value=True)
                            n_layers = gr.Number(label="Number of Layers", value=1, precision=0)
                            hidden_size = gr.Number(label="Hidden Size", value=8192, precision=0)
                            dropout = gr.Number(label="Dropout", value=0.2, precision=3)
                            classifier_size = gr.Number(label="Classifier Size", value=4096, precision=0)
                            classifier_dropout = gr.Number(label="Classifier Dropout", value=0.2, precision=3)
                            n_heads = gr.Number(label="Number of Heads", value=4, precision=0)
                            rotary = gr.Checkbox(label="Rotary", value=True)
                            probe_pooling_types = gr.Textbox(label="Probe Pooling Types (comma-separated)", value="mean,var")
                            transformer_dropout = gr.Number(label="Transformer Dropout", value=0.1, precision=3)
                            token_attention = gr.Checkbox(label="Token Attention", value=False)
                            sim_type = gr.Dropdown(choices=["dot", "euclidean", "cosine"], value="dot")
                            save_model = gr.Checkbox(label="Save Model", value=False)
                            production_model = gr.Checkbox(label="Production Model", value=False)
                            
                            gr.Markdown("### LoRA Settings")
                            lora = gr.Checkbox(label="Use LoRA", value=False)
                            lora_r = gr.Number(label="LoRA r", value=8, precision=0)
                            lora_alpha = gr.Number(label="LoRA alpha", value=32.0, precision=3)
                            lora_dropout = gr.Number(label="LoRA Dropout", value=0.01, precision=3)
                        
                        with gr.Accordion("🏋️ Trainer Settings", open=False):
                            hybrid_probe = gr.Checkbox(label="Hybrid Probe", value=False)
                            full_finetuning = gr.Checkbox(label="Full Finetuning", value=False)
                            num_epochs = gr.Number(label="Number of Epochs", value=200, precision=0)
                            probe_batch_size = gr.Number(label="Probe Batch Size", value=64, precision=0)
                            base_batch_size = gr.Number(label="Base Batch Size", value=4, precision=0)
                            probe_grad_accum = gr.Number(label="Probe Grad Accum", value=1, precision=0)
                            base_grad_accum = gr.Number(label="Base Grad Accum", value=8, precision=0)
                            lr = gr.Number(label="Learning Rate", value=1e-4, precision=6)
                            weight_decay = gr.Number(label="Weight Decay", value=0.0, precision=3)
                            patience = gr.Number(label="Patience", value=1, precision=0)
                            seed = gr.Textbox(label="Seed (optional)", value="")
                            read_scaler = gr.Number(label="Read Scaler", value=100, precision=0)
                            deterministic = gr.Checkbox(label="Deterministic", value=False)

                        with gr.Accordion("🧬 ProteinGym Settings", open=False):
                            proteingym = gr.Checkbox(label="Run ProteinGym", value=False)
                            dms_ids = gr.Textbox(label="DMS IDs (space-separated or 'all')", value="all")
                            mode = gr.Dropdown(choices=["benchmark", "indels", "multiples", "singles"], value="benchmark", label="Mode")
                            scoring_method = gr.Dropdown(choices=["masked_marginal", "mutant_marginal", "wildtype_marginal", "pll", "global_log_prob"], value="masked_marginal", label="Scoring Method")
                            scoring_window = gr.Dropdown(choices=["optimal", "sliding"], value="optimal", label="Scoring Window")
                            pg_batch_size = gr.Number(label="ProteinGym Batch Size", value=32, precision=0)
                            compare_scoring_methods = gr.Checkbox(label="Compare Scoring Methods", value=False)
                        
                        with gr.Accordion("📈 Scikit Settings", open=False):
                            use_scikit = gr.Checkbox(label="Use Scikit", value=False)
                            scikit_n_iter = gr.Number(label="Scikit Iterations", value=10, precision=0)
                            scikit_cv = gr.Number(label="Scikit CV Folds", value=3, precision=0)
                            scikit_random_state = gr.Textbox(label="Scikit Random State (optional)", value="")
                            scikit_model_name = gr.Textbox(label="Scikit Model Name (optional)", value="")
                            n_jobs = gr.Number(label="Number of Jobs", value=1, precision=0)
                    
                    with gr.Column(scale=1):
                        job_name = gr.Textbox(label="Job Name (optional)", placeholder="My Protify Experiment")
                        gpu_type = gr.Dropdown(
                            choices=AVAILABLE_GPUS,
                            value=PROTIFY_DEFAULT_GPU,
                            label="GPU Type",
                            info="Select the GPU type for this job"
                        )
                        submit_btn = gr.Button("🚀 Submit Job", variant="primary", size="lg")
                        submit_status = gr.Markdown("")
                        
                        gr.Markdown("### Preview YAML")
                        yaml_preview = gr.Code(label="Configuration YAML", language="yaml", lines=20)
                        
                        def update_yaml_preview(*form_values):
                            try:
                                return form_values_to_config(*form_values)
                            except Exception as e:
                                return f"Error: {str(e)}"
                        
                        preview_btn = gr.Button("🔄 Preview YAML", variant="secondary")
                
                # Collect all form inputs
                form_inputs = [
                    hf_username, hf_token, wandb_api_key, synthyra_api_key,
                    home_dir, hf_home, log_dir, results_dir, model_save_dir,
                    plots_dir, embedding_save_dir, download_dir,
                    max_length, trim, delimiter, col_names, multi_column, data_names_selected, data_names_custom,
                    model_names_selected, model_names_custom, gpu_type,
                    embedding_batch_size, num_workers, download_embeddings,
                    matrix_embed, embedding_pooling_types, embed_dtype, sql,
                    probe_type, tokenwise, pre_ln, n_layers, hidden_size, dropout,
                    classifier_size, classifier_dropout, n_heads, rotary,
                    probe_pooling_types, transformer_dropout, token_attention,
                    sim_type, save_model, production_model, lora, lora_r,
                    lora_alpha, lora_dropout,
                    hybrid_probe, full_finetuning, num_epochs, probe_batch_size,
                    base_batch_size, probe_grad_accum, base_grad_accum, lr,
                    weight_decay, patience, seed, read_scaler, deterministic,
                    proteingym, dms_ids, mode, scoring_method, scoring_window, pg_batch_size, compare_scoring_methods,
                    use_scikit, scikit_n_iter, scikit_cv, scikit_random_state,
                    scikit_model_name, n_jobs,
                    job_name
                ]
                
                preview_btn.click(
                    fn=update_yaml_preview,
                    inputs=form_inputs,
                    outputs=[yaml_preview]
                )
                
                submit_btn.click(
                    fn=submit_job,
                    inputs=form_inputs,
                    outputs=[submit_status]
                )
            
            # Tab 2: Job Queue
            with gr.Tab("📊 Job Queue"):
                gr.Markdown("## Active Jobs")
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                    clear_pending_btn = gr.Button("🗑️ Clear Completed Jobs", variant="stop")
                clear_status = gr.Markdown("")
                job_queue_display = gr.HTML(value=format_job_queue())
                
                refresh_btn.click(
                    fn=refresh_queue,
                    outputs=[job_queue_display]
                )
                
                clear_pending_btn.click(
                    fn=clear_pending_jobs,
                    outputs=[clear_status, job_queue_display]
                )
            
            # Tab 3: View Results
            with gr.Tab("📋 View Results"):
                gr.Markdown("## View Job Results")
                job_id_input = gr.Textbox(label="Job ID", placeholder="Enter job ID...")
                view_result_btn = gr.Button("🔍 View Results", variant="primary")
                result_display = gr.HTML("")
                
                def view_results(job_id):
                    if not job_id:
                        return "<p style='color: #1a1a1a;'>Please enter a job ID.</p>"
                    results = get_results.remote(job_id)
                    if results.get("success"):
                        files = results.get("files", {})
                        images = results.get("images", {})
                        
                        output = ""
                        
                        # Display images in a tiled grid
                        if images:
                            output += "<h2 style='color: #1a1a1a; margin-bottom: 20px;'>Plot Images</h2>\n"
                            output += "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; margin: 20px 0;'>\n"
                            for file_path, image_info in list(images.items())[:50]:  # Limit to 50 images
                                if "error" in image_info:
                                    output += f"""
                                    <div style='padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #fef2f2;'>
                                        <strong style='font-size: 12px; color: #991b1b;'>{file_path.split('/')[-1]}</strong>
                                        <p style='font-size: 11px; color: #dc2626; margin-top: 5px;'>Error: {image_info['error']}</p>
                                    </div>
                                    """
                                else:
                                    mime_type = image_info.get("mime_type", "image/png")
                                    base64_data = image_info.get("data", "")
                                    filename = file_path.split('/')[-1]
                                    output += f"""
                                    <div style='padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                                        <strong style='font-size: 12px; color: #374151; display: block; margin-bottom: 8px;'>{filename}</strong>
                                        <img src="data:{mime_type};base64,{base64_data}" style="max-width: 100%; height: auto; border-radius: 4px; display: block;" />
                                    </div>
                                    """
                            output += "</div>\n"
                        
                        # Display text files
                        if files:
                            output += "<h2 style='color: #1a1a1a; margin-top: 30px;'>Results Files</h2>\n"
                            for file_path, content in list(files.items())[:10]:  # Limit to 10 files
                                # Escape HTML in content - show full content
                                escaped_content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                                filename = file_path.split('/')[-1]
                                output += f"""
                                <div style='margin-bottom: 30px;'>
                                    <h3 style='color: #1a1a1a; font-size: 18px; margin-bottom: 10px;'>{filename}</h3>
                                    <pre style='background: #f5f5f5; padding: 15px; border-radius: 4px; overflow-x: auto; border: 1px solid #e5e7eb; max-height: 600px; overflow-y: auto;'><code style='color: #1a1a1a; font-family: monospace; font-size: 13px; line-height: 1.5;'>{escaped_content}</code></pre>
                                </div>
                                """
                        
                        if not files and not images:
                            return "<p style='color: #ffffff;'>No result files found.</p>"
                        
                        return output
                    return f"<p style='color: #dc2626;'>Error: {results.get('error', 'Unknown error')}</p>"
                
                view_result_btn.click(
                    fn=view_results,
                    inputs=[job_id_input],
                    outputs=[result_display]
                )
        
        # Add visual separator between app and README
        gr.HTML("""
        <div style="margin: 40px 0; padding: 20px 0; border-top: 3px solid #e5e7eb; border-bottom: 1px solid #e5e7eb;">
            <div style="text-align: center; color: #6b7280; font-size: 14px; font-weight: 500; letter-spacing: 2px;">
                DOCUMENTATION
            </div>
        </div>
        """)
        
        # Load and display README content
        readme_path = "/root/README.md"
        readme_content = ""
        try:
            if os.path.exists(readme_path):
                with open(readme_path, "r", encoding="utf-8") as f:
                    readme_content = f.read()
        except Exception as e:
            print(f"Warning: Could not load README: {e}")
            readme_content = """
### About Protify

Protify is an open source platform designed to simplify and democratize workflows for chemical language models. With Protify, deep learning models can be trained to predict chemical properties at the click of a button, without requiring extensive coding knowledge or computational resources.

This interface allows you to:
- Configure Protify experiments with various models and datasets
- Submit jobs to run on GPU infrastructure
- Track job progress and view results

Jobs are processed on Modal's cloud infrastructure with GPU acceleration.
            """
        
        gr.Markdown(readme_content)
    
    return mount_gradio_app(app=web_app, blocks=interface, path="/")


if __name__ == "__main__":
    with app.run():
        pass

