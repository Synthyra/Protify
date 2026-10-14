"""Opt-in validation MCC selection, including the initial head and loss tie breaking."""

from __future__ import annotations

import math
import torch

from typing import Any, TypedDict
from transformers import TrainerCallback


POLICY = "validation_mcc_then_observed_loss_initial_v1"


class SelectionReading(TypedDict):
    step: int
    mcc: float
    observed_loss: float


def selection_contract(policy: str) -> dict[str, str | int | bool]:
    """Bind the accepted comparison and stopping rules into the training recipe."""
    if policy != POLICY:
        raise ValueError(f"unsupported checkpoint selection: {policy}")
    return {"policy": POLICY, "primary": "validation_mcc", "tie_break": "validation_observed_loss",
            "initial_eligible": True, "stale_checks": 6}


class ValidationSelection(TrainerCallback):
    """Select only during training, then restore the exact selected state before scoring."""

    def __init__(self, policy: str) -> None:
        self.contract = selection_contract(policy)
        self.curve: list[SelectionReading] = []
        self.best = (-math.inf, -math.inf)
        self.weights: dict[str, torch.Tensor] = {}
        self.step = 0
        self.stale = 0
        self.training_ended = False

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: dict[str, float],
                    model: torch.nn.Module, **kwargs: Any) -> None:
        if self.training_ended:
            return
        step = state.global_step
        mcc, loss = float(metrics["eval_mcc"]), float(metrics["eval_observed_loss"])
        if not math.isfinite(mcc) or not math.isfinite(loss):
            raise ValueError("checkpoint selection requires finite validation MCC and observed loss")
        if not self.curve and step != 0:
            raise ValueError("checkpoint selection requires initial validation before training")
        if self.curve and step <= self.curve[-1]["step"]:
            raise ValueError("checkpoint selection requires increasing validation steps")
        self.curve.append({"step": step, "mcc": mcc, "observed_loss": loss})
        criterion = (mcc, -loss)
        if criterion > self.best:
            self.best, self.step, self.stale = criterion, step, 0
            self.weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}  # each tensor retains its parameter shape
        else:
            self.stale += 1
        if self.stale >= self.contract["stale_checks"]:
            control.should_training_stop = True

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.training_ended = True

    def restore(self, model: torch.nn.Module, state: Any) -> dict[str, Any]:
        """Return evidence only after loading and checking the selected weights."""
        if not self.training_ended or not self.weights or not self.curve:
            raise ValueError("cannot restore an unfinished validation selection")
        model.load_state_dict(self.weights, strict=True)
        if any(not torch.equal(value.detach().cpu(), self.weights[name]) for name, value in model.state_dict().items()):
            raise ValueError("selected checkpoint failed exact weight restoration")
        return {"global_step": state.global_step, "epoch": state.epoch, "planned_optimizer_steps": state.max_steps,
                "best_checkpoint": "initial" if self.step == 0 else f"checkpoint-{self.step}",
                "best_step": self.step, "best_metric": self.best[0], "restored_best_weights": True,
                "selection_policy": POLICY, "best_observed_loss": -self.best[1], "selection_curve": self.curve}


def validate_selection_outcome(outcome: dict[str, Any]) -> None:
    """Reject a receipt whose declared winner differs from its validation-only curve."""
    selection_contract(outcome["selection_policy"])
    loss = outcome["best_observed_loss"]
    if isinstance(loss, bool) or not isinstance(loss, int | float) or not math.isfinite(loss):
        raise ValueError("invalid selected observed loss")
    curve = outcome["selection_curve"]
    if not isinstance(curve, list) or not curve:
        raise ValueError("selection outcome requires its validation curve")
    previous = -1
    for reading in curve:
        if not isinstance(reading, dict) or set(reading) != {"step", "mcc", "observed_loss"}:
            raise ValueError("invalid selection curve reading")
        step = reading["step"]
        if type(step) is not int or not previous < step <= outcome["global_step"]:
            raise ValueError("invalid selection curve step")
        for name in ("mcc", "observed_loss"):
            value = reading[name]
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError("invalid selection curve metric")
        previous = step
    if curve[0]["step"] != 0:
        raise ValueError("selection outcome lacks initial validation")
    winner = max(curve, key=lambda reading: (reading["mcc"], -reading["observed_loss"]))
    if (outcome["best_step"], outcome["best_metric"], outcome["best_observed_loss"]) != (
        winner["step"], winner["mcc"], winner["observed_loss"]
    ):
        raise ValueError("selection outcome does not match its validation winner")
