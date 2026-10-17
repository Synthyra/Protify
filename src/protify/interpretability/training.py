"""Settings, results and bookkeeping shared by the protein-level and residue-level probe trainers."""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
import torch

from dataclasses import dataclass, field
from torch import Tensor

from .probe_bank import ProbeBank, ProbeSettings
from .scores import ScoreAccumulator
from .sites import TapSpec


# The test conditions: which model produced the states, which probe read them, and how the input was changed.
PLM_STATES = ("trained", "untrained")
PROBE_STATES = ("trained", "untrained")
CONTROLS = ("original", "l2_normalized", "mean", "scrambled", "random_gaussian")
MEAN_CONTROL_BATCH = 8  # proteins whose states and labels the mean control averages
POOLINGS = ("mean_var", "mean")


@dataclass(frozen=True)
class TrainingSettings:
    """How every probe of a task is trained, validated and scored."""

    learning_rates: tuple[float, ...] = (1e-3,)
    weight_decays: tuple[float, ...] = (1e-4,)
    batch_size: int = 64  # proteins per optimizer step, for protein tasks
    max_tokens: int = 32768  # padded tokens per optimizer step, for residue tasks
    max_epochs: int = 50
    patience: int = 5  # validations without improvement before a probe stops
    validate_every: int = 0  # optimizer steps between validations; 0 validates once per epoch
    seed: int = 24895
    resamples: int = 7
    resample_fraction: float = 0.5
    controls: bool = True
    probe: ProbeSettings = field(default_factory=ProbeSettings)
    normalize: bool = False  # every token state L2-normalized along its width before pooling or probing
    pooling: str = "mean_var"  # what a protein probe reads of a pooled state: "mean_var", or "mean" alone
    untrained_controls: bool = True  # with controls, also score the untrained model's states
    site_budget_gb: float = 8.0  # pooled rows of one site held on the device at once; larger sites train in layer chunks
    bank_budget_gb: float = 0.0  # device memory of one bank's weights and optimizer state; 0 leaves banks unbounded

    def __post_init__(self) -> None:
        if self.pooling not in POOLINGS:
            raise ValueError(f"Unknown pooling {self.pooling!r}; choose one of {POOLINGS}.")

    @property
    def plm_states(self) -> tuple[str, ...]:
        """The models whose states the controls score: the trained one, and the untrained one unless left out."""
        return PLM_STATES if self.untrained_controls else PLM_STATES[:1]

    def grid(self) -> list[tuple[float, float]]:
        """Every (learning rate, weight decay) pair; each probe keeps the pair that validates best."""
        return list(itertools.product(self.learning_rates, self.weight_decays))


def recorded_settings(record: dict[str, object]) -> TrainingSettings:
    """The training settings a task's run.json records (runs from before `normalize` and `pooling` used neither)."""
    saved = dict(record["settings"])
    probe = dict(saved.pop("probe"))
    probe["betas"] = tuple(probe["betas"])
    saved["learning_rates"], saved["weight_decays"] = tuple(saved["learning_rates"]), tuple(saved["weight_decays"])
    return TrainingSettings(**saved, probe=ProbeSettings(**probe))


def recorded_spec(record: dict[str, object]) -> TapSpec:
    spec = record["spec"]
    return TapSpec(sites=tuple(spec["sites"]), blocks=tuple(spec["blocks"]) if spec["blocks"] else None, per_head=spec["per_head"])


def probe_columns(settings: TrainingSettings) -> dict[str, object]:
    """The probe's own settings as table columns: kind, pooling, and an MLP's hidden width and dropout (NaN for a
    linear probe), so sweeps of several variants rank them together."""
    probe = settings.probe
    mlp = probe.kind == "mlp"
    return {"probe_kind": probe.kind, "pooling": settings.pooling,
            "hidden": float(probe.hidden) if mlp else np.nan, "dropout": float(probe.dropout) if mlp else np.nan}


def grid_vectors(settings: TrainingSettings, probes: int) -> tuple[Tensor, Tensor]:
    """Learning rate and weight decay of each group, setting-major: probes of setting 0 first."""
    pairs = settings.grid()
    rates = torch.tensor([rate for rate, _ in pairs for _ in range(probes)], dtype=torch.float32)  # (K * P,)
    decays = torch.tensor([decay for _, decay in pairs for _ in range(probes)], dtype=torch.float32)  # (K * P,)
    return rates, decays


def make_bank(
    names: list[str],
    width: int,
    outputs: int,
    loss: str,
    settings: TrainingSettings,
    seed: int,
    device: torch.device | str,
    first: int = 0,
    total: int | None = None,
) -> ProbeBank:
    """A bank holding every probe once per grid setting, with the same initial weights in every setting.

    `names` may be a chunk of a larger family: probes `first`.. of `total`, which keep the initial weights the whole
    family would give them."""
    rates, decays = grid_vectors(settings, len(names))
    count = len(settings.grid())
    labelled = [f"{name}|lr={rate:g}|wd={decay:g}" for rate, decay in settings.grid() for name in names]
    # A whole family's bank draws count * total groups and gives probe p row p; a chunk draws the same and starts at
    # row `first`, so its first len(names) rows, which every setting repeats below, are the whole family's.
    bank = ProbeBank(labelled, width, outputs, loss, settings.probe, rates, decays, seed, device,
                     first=first, total=None if total is None else count * total)
    # One initialization per probe shared across settings, so settings differ only in how they train.
    for name, value in bank.params.items():
        shared = value[: len(names)].repeat((count,) + (1,) * (value.dim() - 1))
        bank.params[name] = shared.clone()
        bank.initial[name] = shared.clone()
        bank.best[name] = shared.clone()
    return bank


def restore_bank(names: list[str], saved: dict[str, Tensor], outputs: int, loss: str, settings: TrainingSettings, device: torch.device | str) -> ProbeBank:
    """A trained bank of one grid setting rebuilt from its saved `state("best")`, to score and not to train: the best
    weights, which are the current weights too, and the standardization, without optimizer state. `names` are the
    probes in the order they were saved. The initial weights are drawn again from the seed, which reproduces training's
    when it ran one grid setting."""
    best = {key.removeprefix("best."): value for key, value in saved.items() if key.startswith("best.")}
    width = best["weight" if settings.probe.kind == "linear" else "weight1"].shape[1]
    bank = ProbeBank(names, width, outputs, loss, settings.probe, saved["learning_rates"], saved["weight_decays"], settings.seed, device)
    if set(best) != set(bank.params) or any(value.shape != bank.params[name].shape for name, value in best.items()):
        shapes = {name: tuple(value.shape) for name, value in best.items()}
        raise ValueError(f"Saved weights of shapes {shapes} do not fit {bank.groups} probes from {names[0]}.")
    for name, value in best.items():
        bank.best[name] = value.to(bank.device, torch.float32).clone()
        bank.params[name] = bank.best[name]  # nothing trains a restored bank: one copy, and no optimizer state
    bank.grads, bank.first_moment, bank.second_moment = {}, {}, {}
    bank.mean = saved["mean"].to(bank.device, torch.float32)
    bank.inverse_std = saved["inverse_std"].to(bank.device, torch.float32)
    bank.best_loss = saved["best_loss"].to(bank.device)
    bank.best_step = saved["best_step"].to(bank.device)
    return bank


@dataclass
class TaskResults:
    """Long tables of one task's test scores, elementwise scores, chosen training settings and validation scores.

    A sweep fills `sweep` instead: every probe under every grid setting, scored on the validation split only.
    """

    scores: list[pd.DataFrame] = field(default_factory=list)
    elementwise: list[pd.DataFrame] = field(default_factory=list)
    training: list[pd.DataFrame] = field(default_factory=list)
    sweep: list[pd.DataFrame] = field(default_factory=list)
    validation: list[pd.DataFrame] = field(default_factory=list)
    ranking: list[pd.DataFrame] = field(default_factory=list)  # ROC AUC and average precision, binary pair tasks

    def add_scores(
        self,
        task: str,
        family: str,
        names: list[str],
        accumulator: ScoreAccumulator,
        plm: str,
        probe: str,
        control: str,
        keep_elementwise: bool,
    ) -> None:
        names = [name.split("|")[0] for name in names]
        scores = accumulator.scores().cpu().numpy()  # (R + 1, G)
        rows, groups = scores.shape
        frame = pd.DataFrame({
            "task": task,
            "family": family,
            "probe_name": np.repeat(np.asarray(names), rows),
            "resample": np.tile(np.arange(rows), groups),
            "plm": plm,
            "probe": probe,
            "control": control,
            "score": scores.T.reshape(-1),
        })
        self.scores.append(frame)
        elementwise = accumulator.elementwise() if keep_elementwise else None
        if elementwise is not None:
            values = elementwise.cpu().numpy()  # (R + 1, G, C)
            rows, groups, outputs = values.shape
            self.elementwise.append(pd.DataFrame({
                "task": task,
                "family": family,
                "probe_name": np.repeat(np.asarray(names), rows * outputs),
                "resample": np.tile(np.repeat(np.arange(rows), outputs), groups),
                "label_index": np.tile(np.arange(outputs), rows * groups),
                "score": values.transpose(1, 0, 2).reshape(-1),
            }))

    def add_training(self, task: str, family: str, bank: ProbeBank, chosen: Tensor, settings: TrainingSettings, epochs: int) -> None:
        pairs = settings.grid()
        choice = chosen.cpu().numpy()
        self.training.append(pd.DataFrame({
            "task": task,
            "family": family,
            "probe_name": [name.split("|")[0] for name in bank.names],
            **probe_columns(settings),
            "learning_rate": [pairs[index][0] for index in choice],
            "weight_decay": [pairs[index][1] for index in choice],
            "best_validation_loss": bank.best_loss.cpu().numpy(),
            "best_step": bank.best_step.cpu().numpy(),
            "steps": bank.steps,
            "epochs": epochs,
        }))

    def add_sweep(
        self,
        task: str,
        family: str,
        bank: ProbeBank,
        validation: ScoreAccumulator,
        settings: TrainingSettings,
        epochs: int,
        seconds: float,
    ) -> None:
        """One row per group of a bank trained over the whole grid: its setting and its validation score."""
        pairs = settings.grid()
        setting = np.arange(bank.groups) // (bank.groups // len(pairs))  # groups are setting-major
        self.sweep.append(pd.DataFrame({
            "task": task,
            "family": family,
            "probe_name": [name.split("|")[0] for name in bank.names],
            **probe_columns(settings),
            "learning_rate": [pairs[index][0] for index in setting],
            "weight_decay": [pairs[index][1] for index in setting],
            "validation_score": validation.scores()[0].cpu().numpy(),
            "best_validation_loss": bank.best_loss.cpu().numpy(),
            "best_step": bank.best_step.cpu().numpy(),
            "steps": bank.steps,
            "epochs": epochs,
            "seconds": seconds,
        }))

    def add_validation(self, task: str, family: str, names: list[str], accumulator: ScoreAccumulator, losses: Tensor) -> None:
        """One row per trained probe: its score on the whole validation split, in the test metric, which best states
        are chosen by, and its validation loss with the weights kept."""
        self.validation.append(pd.DataFrame({
            "task": task,
            "family": family,
            "probe_name": [name.split("|")[0] for name in names],
            "validation_score": accumulator.scores()[0].cpu().numpy(),
            "validation_loss": losses.detach().cpu().numpy(),
        }))

    def add_ranking(
        self,
        task: str,
        family: str,
        names: list[str],
        split: str,
        roc_auc: Tensor,
        average_precision: Tensor,
        mcc: Tensor | None = None,
        balanced_mcc: Tensor | None = None,
    ) -> None:
        """One row per probe: ROC AUC, average precision, and, when given, MCC at probability 0.5 and balanced MCC of its
        kept weights on one split."""
        frame = pd.DataFrame({
            "task": task,
            "family": family,
            "probe_name": [name.split("|")[0] for name in names],
            "split": split,
            "roc_auc": roc_auc.detach().cpu().numpy(),
            "average_precision": average_precision.detach().cpu().numpy(),
        })
        for column, values in (("mcc", mcc), ("balanced_mcc", balanced_mcc)):
            if values is not None:
                frame[column] = values.detach().cpu().numpy()
        self.ranking.append(frame)

    def frames(self) -> dict[str, pd.DataFrame]:
        tables = {
            name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            for name, parts in (("scores", self.scores), ("elementwise", self.elementwise), ("training", self.training))
        }
        if self.validation:
            tables["validation"] = pd.concat(self.validation, ignore_index=True)
        if self.ranking:
            tables["ranking"] = pd.concat(self.ranking, ignore_index=True)
        return tables

    def sweep_frame(self) -> pd.DataFrame:
        return pd.concat(self.sweep, ignore_index=True) if self.sweep else pd.DataFrame()
