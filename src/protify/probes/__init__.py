"""Public probe classes and parallel-probe planning helpers.

Each name loads its module on first use, so importing one probe module, such as the estimator
probes, does not load every probe and `transformers` with it.
"""

from importlib import import_module
from typing import Any


# Each public name, by the module that defines it.
_EXPORTS = {
    "MLPProbe": ".mlp_probe",
    "MLPProbeConfig": ".mlp_probe",
    "PackagedProbeConfig": ".packaged_probe_model",
    "PackagedProbeModel": ".packaged_probe_model",
    "ParallelLinearProbe": ".parallel_linear_probe",
    "ParallelLinearProbeConfig": ".parallel_linear_probe",
    "ParallelLinearProbeEnsemble": ".parallel_linear_probe",
    "ParallelRunDataset": ".parallel_probe_batches",
    "TransformerForSequenceClassification": ".transformer_probe",
    "TransformerForTokenClassification": ".transformer_probe",
    "TransformerProbeConfig": ".transformer_probe",
    **{name: ".parallel_probe_plan" for name in (
        "ParallelProbeExecutionPlan",
        "ParallelProbeGroup",
        "ParallelProbeGroupEstimate",
        "ParallelProbePlanEstimate",
        "ParallelProbeRunSpec",
        "ParallelProbeExecutionWave",
        "ParallelProbeWaveSchedule",
        "build_seed_run_specs",
        "estimate_parallel_probe_group",
        "estimate_parallel_probe_plan",
        "group_parallel_probe_runs",
        "linear_probe_batch_activation_count",
        "linear_probe_forward_flop_count",
        "linear_probe_parameter_count",
        "linear_probe_parameter_count_for_spec",
        "max_linear_probe_runs_for_estimated_peak_budget",
        "max_linear_probe_runs_for_training_state_budget",
        "plan_parallel_probe_runs",
        "schedule_parallel_probe_execution_waves",
    )},
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
