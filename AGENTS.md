# Protify

## Purpose and Sources

This is the Gleghorn-Lab Protify checkout embedded in synth. Reusable training and evaluation behavior belongs here; parent-platform serving and orchestration belong in synth.

- `docs/getting_started.md` and `docs/cli_and_config.md`: entrypoints and configuration
- `docs/probes_and_training.md`: probe behavior
- `docs/testing.md`: test scopes and working directories
- `src/protify/base_models/supported_models.py`: authoritative model registry
- `src/protify/data/supported_datasets.py`: authoritative dataset registry

## Architectural Invariants

- CLI, YAML, GUI, and cloud dispatch share the same `MainProcess` pipeline.
- Keep heavy model and dataset loading lazy where the registries and factories are lazy.
- `src/protify/fastplms/` is a vendored repository with its own guidance. Do not mix FastPLMs changes into a Protify task.
- `--parallel_probe_runs` applies only to compatible pooled, sequence-level linear probes. Matrix or tokenwise probes, transformer probes, Lyra probes, PPI run-specific datasets, and full fine-tuning use the sequential fallback.

## Canonical Commands

Build and run the broad CPU suite from the repository root:

```powershell
docker build -t protify-env:latest .
docker run --rm --ipc=host -v ${PWD}:/workspace -w /workspace protify-env:latest \
  python -m pytest src/protify/testing_suite -v -m "not gpu and not slow"
```

Focused parallel-probe tests run with `src/protify/` as the working directory:

```powershell
docker run --rm --gpus all --ipc=host -v ${PWD}:/workspace -e PYTHONPATH=/workspace \
  -w /workspace/src/protify protify-env:latest python -m pytest \
  testing_suite/test_parallel_probe_plan.py testing_suite/test_parallel_linear_probe.py -v
```
