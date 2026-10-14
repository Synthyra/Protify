# Cloud Compute

This page documents how to run Protify jobs on cloud GPUs via the cloud backend API.

---

## Quick Start

Get your API key at [synthyra.com](https://synthyra.com) and put it in the `SYNTHYRA_API_KEY` environment variable, never on the command line, then run:

**CLI:**
```bash
python -m protify.main --cloud --model_names ESM2-8 --data_names EC
```

**GUI:** Enter your API key in the **Info** tab under "Cloud API Key", configure your run in other tabs, then go to the **Cloud** tab and click **Submit Remote Run**.

No local GPU or cloud provider setup required. Jobs are billed through your account.

---

## CLI Options

| Argument | Default | Description |
|---|---|---|
| `--cloud` | False | Dispatch the job to the cloud backend, authenticated by `SYNTHYRA_API_KEY`. |
| `--cloud_url` | `https://api.synthyra.com` | Cloud backend URL. |
| `--cloud_gpu_type` | `A10` | GPU type (H200, H100, A100-80GB, A100, L40S, A10, L4, T4). |
| `--cloud_timeout_seconds` | `86400` | Maximum job runtime in seconds (24h default). |
| `--cloud_poll_interval` | `5` | Seconds between status polls. |
| `--cloud_artifacts_dir` | `cloud_artifacts` | Local directory to save downloaded results. |

---

## How It Works

1. **Submit:** Protify serializes the current config dict and sends it to the cloud backend via HTTP POST.
2. **Execute:** The backend runs `main.py --yaml_path <config.yaml>` on a GPU container with the same pipeline as local execution.
3. **Poll:** The CLI/GUI polls job status and streams logs from the backend.
4. **Fetch:** On completion, results (metrics TSV, plots, model weights) are downloaded to your local `cloud_artifacts/` directory.

---

## Cloud Backend Protocol

Protify uses a pluggable `CloudBackend` abstraction (see `cloud_backend.py`). The built-in `HTTPCloudBackend` talks to any server implementing these endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/v1/protify/train` | POST | Submit a training job |
| `/v1/protify/job` | GET | Poll job status |
| `/v1/protify/logs` | GET | Read log delta |
| `/v1/protify/cancel` | POST | Cancel a running job |
| `/v1/protify/results` | GET | Fetch result files |
| `/v1/protify/jobs` | GET | List all jobs |

Custom backends can be registered via `register_cloud_backend()` for non-HTTP transports or custom auth schemes.

---

## GUI Cloud Tab

The **Cloud** tab provides:

- **Cloud URL** and **GPU Type** configuration
- **Submit Remote Run** to dispatch the current config
- **Poll Status** to check job progress and view logs
- **Cancel Run** to stop a running job
- **Fetch Results** to download artifacts locally
- **Auto Poll** toggle for continuous status monitoring

---

## See Also

- [GUI](gui.md) for the Cloud tab and background tasks
- [Configuration](cli_and_config.md) for options that are sent in the config
- [Getting started](getting_started.md) for local entry points
