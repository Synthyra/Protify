"""Serializable effective probe recipes and the checkpoint each finished seed restored.

Only named scientific and execution controls are exported. Output paths, Hub tokens, and
integration configuration never enter a recipe. Runtime versions and resolved automatic
choices are bound on resume, so parser or library default drift cannot relabel old fits.
"""

from __future__ import annotations

import json
import math
import torch

from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .validation_selection import selection_contract, validate_selection_outcome


TRAINING_FIELDS = (
    "num_train_epochs", "max_steps", "per_device_train_batch_size", "per_device_eval_batch_size",
    "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type", "lr_scheduler_kwargs",
    "optim", "optim_args", "adam_beta1", "adam_beta2", "adam_epsilon", "weight_decay",
    "warmup_steps", "max_grad_norm", "eval_strategy", "eval_steps", "eval_delay",
    "save_strategy", "save_steps", "save_total_limit", "load_best_model_at_end",
    "metric_for_best_model", "greater_is_better", "label_names", "fp16", "bf16", "tf32",
    "fp16_full_eval", "bf16_full_eval", "torch_compile",
    "torch_compile_backend", "torch_compile_mode", "full_determinism", "gradient_checkpointing",
    "dataloader_num_workers", "dataloader_prefetch_factor", "dataloader_drop_last",
    "dataloader_pin_memory", "dataloader_persistent_workers", "eval_accumulation_steps",
    "remove_unused_columns", "label_smoothing_factor", "logging_steps", "logging_strategy",
    "auto_find_batch_size", "average_tokens_across_devices",
)
PROBE_FIELDS = (
    "probe_type", "task_type", "input_size", "hidden_size", "num_labels", "n_layers", "dropout",
    "projection_size", "projected_hidden_size", "projected_trunk", "branch_widths", "branch_sizes", "input_clip", "spectral_bound", "gp_lengthscale_squared",
    "pre_ln", "use_bias", "classifier_size", "classifier_dropout", "tokenwise",
)


def validate_probe_controls(arguments: Any) -> None:
    """Reject invalid numeric controls before constructing a trainer or starting a fit."""
    for name, minimum in (
        ("num_epochs", 1), ("probe_batch_size", 1), ("probe_grad_accum", 1),
        ("probe_max_steps", -1), ("probe_warmup_steps", 0), ("probe_eval_steps", 1),
        ("num_workers", 0), ("patience", 1),
    ):
        value = getattr(arguments, name)
        if type(value) is not int or value < minimum or (name == "probe_max_steps" and value == 0):
            raise ValueError(f"invalid {name}: {value!r}")
    for name in ("weight_decay", "probe_max_grad_norm", "early_stopping_threshold", "adam_beta1", "adam_beta2", "adam_epsilon"):
        value = getattr(arguments, name)
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {name}: {value!r}")
        if (name.startswith("adam_beta") and value >= 1) or (name == "adam_epsilon" and value == 0):
            raise ValueError(f"invalid {name}: {value!r}")
    if arguments.probe_precision not in {"float32", "bfloat16", "float16"}:
        raise ValueError("invalid probe_precision")
    if arguments.probe_eval_strategy not in {"auto", "epoch", "steps"}:
        raise ValueError("invalid probe_eval_strategy")
    accumulation = arguments.eval_accumulation_steps
    if accumulation not in (None, "auto") and (
        isinstance(accumulation, bool) or not str(accumulation).isdigit() or int(accumulation) < 1
    ):
        raise ValueError("eval_accumulation_steps must be auto, null, or a positive integer")


def resolved_training_recipe(arguments: Any, probe: Any, stopping: Mapping[str, Any], sampling: Any,
                             *, checkpoint_selection: str | None = None) -> dict[str, Any]:
    """The effective seed-independent recipe after row counts, masks, and devices resolve."""
    encoded = arguments.to_dict()
    head = {name: getattr(probe, name) for name in PROBE_FIELDS if hasattr(probe, name)}
    if head.get("branch_sizes") is None:
        # An even split leaves the key out, so the recipe of a seed finished before the field existed still matches.
        head.pop("branch_sizes", None)
    recipe = {
        "schema_version": 1,
        "training_arguments": {name: encoded[name] for name in TRAINING_FIELDS},
        "head": head,
        "early_stopping": dict(stopping),
        "sampling": sampling.record() if sampling is not None else {"algorithm": "transformers_random_sampler"},
        "batch_unit": "pooled_observation_vectors",
        "token_budget": None,
        "world_size": arguments.world_size,
        "effective_batch_size": arguments.train_batch_size * arguments.gradient_accumulation_steps * arguments.world_size,
        "runtime": {
            "torch": version("torch"), "transformers": version("transformers"), "accelerate": version("accelerate"),
            "device_type": arguments.device.type,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }
    if checkpoint_selection is not None:
        recipe["checkpoint_selection"] = selection_contract(checkpoint_selection)
        recipe["early_stopping"] = {"stale_checks": 6, "criterion": "validation_mcc_then_observed_loss"}
    return json.loads(json.dumps(recipe, allow_nan=False))


def training_outcome(trainer: Any) -> dict[str, Any]:
    """Record actual optimization and best-weight restoration before temporary checkpoints go."""
    state = trainer.state
    if state.best_model_checkpoint is None or state.best_metric is None:
        raise ValueError("a recorded training recipe requires a selected validation checkpoint")
    checkpoint = Path(state.best_model_checkpoint).name
    outcome = {
        "global_step": state.global_step,
        "epoch": state.epoch,
        "planned_optimizer_steps": state.max_steps,
        "best_checkpoint": checkpoint,
        "best_step": int(checkpoint.removeprefix("checkpoint-")),
        "best_metric": float(state.best_metric),
        "restored_best_weights": bool(trainer.args.load_best_model_at_end),
    }
    _validate_outcome(outcome)
    return outcome


def _validate_outcome(outcome: object) -> None:
    fields = {"global_step", "epoch", "planned_optimizer_steps", "best_checkpoint", "best_step", "best_metric", "restored_best_weights"}
    selected = isinstance(outcome, dict) and "selection_policy" in outcome
    if selected:
        fields |= {"selection_policy", "best_observed_loss", "selection_curve"}
    if not isinstance(outcome, dict) or set(outcome) != fields:
        raise ValueError("training outcome is missing its checkpoint-selection evidence")
    for name in ("global_step", "planned_optimizer_steps", "best_step"):
        minimum = 0 if selected and name == "best_step" else 1
        if type(outcome[name]) is not int or outcome[name] < minimum:
            raise ValueError(f"invalid training outcome {name}")
    for name in ("epoch", "best_metric"):
        if isinstance(outcome[name], bool) or not isinstance(outcome[name], int | float) or not math.isfinite(outcome[name]):
            raise ValueError(f"invalid training outcome {name}")
    if not outcome["best_step"] <= outcome["global_step"] <= outcome["planned_optimizer_steps"] or outcome["epoch"] <= 0:
        raise ValueError("training outcome optimizer steps are inconsistent")
    checkpoint = "initial" if selected and outcome["best_step"] == 0 else f"checkpoint-{outcome['best_step']}"
    if outcome["best_checkpoint"] != checkpoint or outcome["restored_best_weights"] is not True:
        raise ValueError("training outcome lacks the restored best checkpoint")
    if selected:
        validate_selection_outcome(outcome)


def validate_training_receipt(recipe: Mapping[str, Any], receipt: object, seed: int) -> None:
    """Require finished-seed evidence to agree with the independently resolved recipe."""
    if not isinstance(receipt, dict) or set(receipt) != {"recipe", "seed", "outcome"}:
        raise ValueError("finished seed lacks its training recipe receipt; retrain in a new output directory")
    if receipt["recipe"] != recipe or type(receipt["seed"]) is not int or receipt["seed"] != seed:
        raise ValueError("finished seed training recipe changed; use a new output directory")
    _validate_outcome(receipt["outcome"])
    selection = recipe.get("checkpoint_selection")
    policy = receipt["outcome"].get("selection_policy")
    if (selection is None and policy is not None) or (selection is not None and selection != selection_contract(policy)):
        raise ValueError("training outcome checkpoint selection differs from its recipe")
