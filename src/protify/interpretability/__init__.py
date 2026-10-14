"""Tap probing: every token-wise state inside ESMC, pooled or streamed, and linear or MLP probes trained on all of them.

`capture` hooks the states (residual stream, attention inputs and outputs, queries and keys before and after the
rotary rotation, values, the MLP's gate, up and activation states) during one forward pass. `pooled_store` writes
their per-protein mean and variance for protein-level tasks; `residue_probing` trains residue-level probes inside
the hooks, so token states are never stored; `token_store` keeps token states of chosen sites when they are needed.
`probe_bank` trains every probe of a site, layer, head and hyperparameter setting at once, and `selection` picks one
global setting from a sweep of them. Run `python -m protify.interpretability --help` for the command line.
"""

from .capture import capture_states, model_dims, tap_hooks
from .concepts import ALL_TASKS, PROTEIN_TASKS, RESIDUE_TASKS, ConceptData, ConceptTask, load_concept
from .pooled_store import PooledTapStore, embed_pooled, masked_mean_var
from .probe_bank import ProbeBank, ProbeSettings
from .protein_probing import sweep_protein_task, train_protein_task
from .residue_probing import sweep_residue_task, train_residue_task
from .scores import ResamplePlan, ScoreAccumulator
from .selection import choose_setting, rank_settings, read_sweeps
from .sites import ALL_SITES, ESMC_SITES, RESIDUAL_SITES, ModelDims, TapSpec, site_layouts
from .token_store import TokenTapStore, embed_tokens
from .training import TaskResults, TrainingSettings


__all__ = [
    "ALL_SITES",
    "ALL_TASKS",
    "ConceptData",
    "ConceptTask",
    "ESMC_SITES",
    "ModelDims",
    "PROTEIN_TASKS",
    "PooledTapStore",
    "ProbeBank",
    "ProbeSettings",
    "RESIDUAL_SITES",
    "RESIDUE_TASKS",
    "ResamplePlan",
    "ScoreAccumulator",
    "TapSpec",
    "TaskResults",
    "TokenTapStore",
    "TrainingSettings",
    "capture_states",
    "choose_setting",
    "embed_pooled",
    "embed_tokens",
    "load_concept",
    "masked_mean_var",
    "model_dims",
    "rank_settings",
    "read_sweeps",
    "site_layouts",
    "sweep_protein_task",
    "sweep_residue_task",
    "tap_hooks",
    "train_protein_task",
    "train_residue_task",
]
