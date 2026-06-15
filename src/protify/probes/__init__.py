"""Probes package exports.

This enables convenient imports like:
    from protify.probes import TransformerForSequenceClassification, LinearProbe

Works both when the repository is the main module and when it is used as a
submodule in another project (so long as `src/` is on PYTHONPATH or the
package is installed).
"""

from .linear_probe import LinearProbe, LinearProbeConfig  # noqa: F401
from .parallel_probe_batches import ParallelRunDataset  # noqa: F401
from .parallel_linear_probe import (  # noqa: F401
    ParallelLinearProbe,
    ParallelLinearProbeConfig,
    ParallelLinearProbeEnsemble,
)
from .parallel_probe_plan import (  # noqa: F401
    ParallelProbeExecutionPlan,
    ParallelProbeGroup,
    ParallelProbeGroupEstimate,
    ParallelProbePlanEstimate,
    ParallelProbeRunSpec,
    ParallelProbeExecutionWave,
    ParallelProbeWaveSchedule,
    build_seed_run_specs,
    estimate_parallel_probe_group,
    estimate_parallel_probe_plan,
    group_parallel_probe_runs,
    linear_probe_batch_activation_count,
    linear_probe_forward_flop_count,
    linear_probe_parameter_count,
    linear_probe_parameter_count_for_spec,
    max_linear_probe_runs_for_estimated_peak_budget,
    max_linear_probe_runs_for_training_state_budget,
    plan_parallel_probe_runs,
    schedule_parallel_probe_execution_waves,
)
from .transformer_probe import (
    TransformerForSequenceClassification,
    TransformerForTokenClassification,
    TransformerProbeConfig,
)  # noqa: F401
from .packaged_probe_model import PackagedProbeConfig, PackagedProbeModel  # noqa: F401

__all__ = [
    "LinearProbe",
    "LinearProbeConfig",
    "ParallelRunDataset",
    "ParallelLinearProbe",
    "ParallelLinearProbeConfig",
    "ParallelLinearProbeEnsemble",
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
    "TransformerForSequenceClassification",
    "TransformerForTokenClassification",
    "TransformerProbeConfig",
    "PackagedProbeConfig",
    "PackagedProbeModel",
]


