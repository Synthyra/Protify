"""Protein-level probes on pooled taps: every site, layer and head of a store, trained in parallel per site.

Each site's train, validation and test rows are read once from the pooled store onto the device. One bank holds a
full-width probe per layer (input 2w: mean then variance), and, for head-split sites, a bank holds one probe per
layer and head (input 2 d_h: that head's mean then variance columns); with `pooling="mean"` every probe reads the
mean columns alone (w, or d_h). Training is minibatch AdamW over proteins
with validation each epoch (or every `validate_every` steps), per-probe early stopping on validation loss, and the
best validation weights kept, which are then scored on the validation split in the test metric (the scores a best
state is chosen by). Test scores cover the controls when control stores are given. Linear banks run on the
probe bank's shared-input path: the P probes of a family read P slices of one row, and every grid setting reads the
same slice, so one batched bf16 GEMM a step serves all K * P probes.

Symbols: n proteins; L layers of a site; w site width; H heads; d_h head width; C outputs; K grid settings;
P probes of a family (L, or L * H).
"""

from __future__ import annotations

import math
import time
import numpy as np
import torch

from collections.abc import Callable
from dataclasses import dataclass
from torch import Tensor
from tqdm.auto import tqdm

from .concepts import ConceptData, prot_param_values, scramble, scramble_seed, shuffled_labels
from .pooled_store import PooledTapStore
from .probe_bank import ProbeBank
from .scores import ResamplePlan, ScoreAccumulator
from .sites import SiteLayout
from .training import MEAN_CONTROL_BATCH, TaskResults, TrainingSettings, make_bank


Family = tuple[str, list[str], Callable[[Tensor], Tensor]]


@dataclass
class ControlStores:
    """Pooled stores of the test proteins under each condition, keyed by (plm state, input)."""

    stores: dict[tuple[str, str], PooledTapStore]  # inputs: original, l2_normalized, scrambled

    @property
    def scramble_seed(self) -> int:
        """The seed the scrambled stores were written with, which their targets must be recomputed with."""
        seeds = {self.stores[(plm, "scrambled")].manifest.seed for plm in ("trained", "untrained")}
        if len(seeds) != 1 or None in seeds:
            raise ValueError(f"The scrambled stores record seeds {sorted(seeds, key=str)}; they must share one.")
        return seeds.pop()


def head_view(pooled: Tensor, heads: int, pooling: str = "mean_var") -> Tensor:
    """Split pooled (mean, variance) columns into per-head (mean, variance) blocks, or per-head means alone."""
    # pooled: (n, L, 2 * w) -> (n, L * H, 2 * d_h), or (n, L * H, d_h) for the mean alone
    n, layers, columns = pooled.shape
    width = columns // 2
    mean, variance = pooled.split(width, dim=-1)  # each (n, L, w)
    mean = mean.reshape(n, layers, heads, width // heads)  # (n, L, H, d_h)
    if pooling == "mean":
        return mean.reshape(n, layers * heads, width // heads)  # (n, L * H, d_h)
    variance = variance.reshape(n, layers, heads, width // heads)  # (n, L, H, d_h)
    return torch.cat([mean, variance], dim=-1).reshape(n, layers * heads, 2 * width // heads)  # (n, L * H, 2 * d_h)


def full_view(pooled: Tensor, pooling: str = "mean_var") -> Tensor:
    """The pooled columns a full-width probe reads: mean then variance, or the mean alone."""
    # pooled: (n, L, 2 * w) -> (n, L, 2 * w), or (n, L, w)
    return pooled if pooling == "mean_var" else pooled[..., : pooled.shape[-1] // 2]


def feature_moments(X: Tensor, chunk: int = 1024) -> tuple[Tensor, Tensor]:
    """Per-group, per-column mean and population variance over rows, in float64 chunks."""
    # X: (n, G, w)
    total = torch.zeros(X.shape[1:], dtype=torch.float64, device=X.device)  # (G, w)
    squares = torch.zeros_like(total)
    for start in range(0, X.shape[0], chunk):
        block = X[start: start + chunk].to(torch.float64)  # (c, G, w)
        total += block.sum(dim=0)
        squares += block.square().sum(dim=0)
    mean = total / X.shape[0]
    return mean.to(torch.float32), (squares / X.shape[0] - mean.square()).clamp_min(0).to(torch.float32)


def _shared(bank: ProbeBank) -> bool:
    """Linear banks take the shared-input path; MLP banks keep one forward per grid setting."""
    return bank.settings.kind == "linear"


def _flat(X: Tensor) -> Tensor:
    # X: (n, P, w) -> (n, P * w), the row every grid setting's probe p reads its slice p of
    return X.reshape(X.shape[0], -1)


def validation_loss(bank: ProbeBank, X: Tensor, targets: Tensor, settings_count: int, batch: int = 512, which: str = "current") -> Tensor:
    """Mean validation loss of every group, settings included, from inputs shared across settings."""
    # X: (n, P, w); targets: (n,) or (n, C)
    probes = bank.groups // settings_count
    totals = torch.zeros(bank.groups, device=bank.device)
    every_group = torch.arange(bank.groups, device=bank.device)  # (K * P,), setting-major then probe
    for start in range(0, X.shape[0], batch):
        rows = X[start: start + batch]  # (b, P, w)
        labels = targets[start: start + batch]
        if _shared(bank):
            totals += bank.shared_loss_sums(_flat(rows), labels, every_group, probes, which=which)
            continue
        for setting in range(settings_count):
            totals[setting * probes: (setting + 1) * probes] += bank.loss_sums(rows, labels, setting * probes, (setting + 1) * probes, which)
    denominator = X.shape[0] if bank.loss == "multiclass" else X.shape[0] * bank.outputs
    return totals / denominator


def train_bank(
    bank: ProbeBank,
    train_X: Tensor,
    train_y: Tensor,
    valid_X: Tensor,
    valid_y: Tensor,
    settings: TrainingSettings,
    description: str,
) -> int:
    """Minibatch training of every group with per-group early stopping. Returns the epochs run."""
    # train_X: (n, P, w); valid_X: (v, P, w)
    count = len(settings.grid())
    probes = bank.groups // count
    every_group = torch.arange(bank.groups, device=bank.device)  # (K * P,)
    rng = np.random.default_rng(settings.seed)
    steps_per_epoch = math.ceil(train_X.shape[0] / settings.batch_size)
    validate_every = settings.validate_every or steps_per_epoch
    epochs = 0
    active = bank.groups
    bar = tqdm(range(settings.max_epochs), desc=description, leave=False, mininterval=10)
    for epoch in bar:
        epochs = epoch + 1
        order = torch.from_numpy(rng.permutation(train_X.shape[0])).to(train_X.device)
        for start in range(0, len(order), settings.batch_size):
            rows = order[start: start + settings.batch_size]
            inputs, labels = train_X[rows], train_y[rows]  # (b, P, w), (b,) or (b, C)
            if _shared(bank):
                bank.shared_accumulate(_flat(inputs), labels, every_group, probes)
            else:
                for setting in range(count):
                    bank.accumulate(inputs, labels, setting * probes, (setting + 1) * probes)
            bank.step()
            if bank.steps % validate_every == 0:
                active = bank.observe(validation_loss(bank, valid_X, valid_y, count), settings.patience)
                if active == 0:
                    break
        bar.set_postfix(active=active)
        if active == 0:
            break
    return epochs


def families(layout: SiteLayout, pooling: str = "mean_var") -> list[Family]:
    """(family, probe names, view of the pooled rows) of a site: full width, and per head for head-split sites."""
    found: list[Family] = [("full", layout.full_names(), lambda pooled: full_view(pooled, pooling))]
    if layout.heads:
        found.append(("head", layout.head_names(), lambda pooled, heads=layout.heads: head_view(pooled, heads, pooling)))
    return found


def fit_family(
    names: list[str],
    inputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    data: ConceptData,
    settings: TrainingSettings,
    device: torch.device,
    description: str,
) -> tuple[ProbeBank, int]:
    """A bank of every probe of one family under every grid setting, standardized on train and trained."""
    # inputs: split -> (n, P, w)
    task = data.task
    bank = make_bank(names, inputs["train"].shape[-1], task.outputs, task.loss, settings, settings.seed, device)
    mean, variance = feature_moments(inputs["train"])  # (P, w) each
    count = len(settings.grid())
    bank.set_standardization(mean.repeat(count, 1), variance.repeat(count, 1))
    epochs = train_bank(bank, inputs["train"], targets["train"], inputs["validation"], targets["validation"], settings, description)
    return bank, epochs


def _elapsed(started: float, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.time() - started


def _mean_control(X: Tensor, targets: Tensor, kind: str) -> tuple[Tensor, Tensor]:
    """Replace each batch of MEAN_CONTROL_BATCH proteins by its mean state and its mean or most common label."""
    # X: (n, P, w); targets: (n,) or (n, C)
    states, labels = [], []
    for start in range(0, X.shape[0], MEAN_CONTROL_BATCH):
        block = X[start: start + MEAN_CONTROL_BATCH].to(torch.float32)  # (b, P, w)
        target = targets[start: start + MEAN_CONTROL_BATCH]
        states.append(block.mean(dim=0, keepdim=True).expand_as(block))
        if kind == "regression":
            labels.append(target.mean(dim=0, keepdim=True).expand_as(target))
        else:
            labels.append(torch.mode(target.reshape(-1)).values.expand_as(target))
    return torch.cat(states), torch.cat(labels)


def score_bank(bank: ProbeBank, which: str, X: Tensor, targets: Tensor, accumulator: ScoreAccumulator, budget: int = 2**30) -> None:
    # Per-protein counts take g * C * 4 float64 values a row for multilabel tasks (a few bytes per output without
    # per-label counts); batches keep them under `budget` bytes.
    proteins = torch.arange(X.shape[0], device=X.device)
    row_bytes = bank.groups * max(bank.outputs, 1) * (64 if getattr(accumulator, "per_label", None) is not None else 16)
    batch = max(1, min(512, budget // row_bytes))
    probes = X.shape[1]  # P; a sweep's bank holds G = K * P groups, a trained bank G = P
    every_group = torch.arange(bank.groups, device=bank.device)  # (G,)
    for start in range(0, X.shape[0], batch):
        rows = X[start: start + batch]  # (b, P, w)
        if _shared(bank):
            Z = bank.shared_logits(_flat(rows), every_group, probes, which)  # (b, G, C)
        else:
            Z = torch.cat([
                bank.logits(rows, first, first + probes, which) for first in range(0, bank.groups, probes)
            ], dim=1)  # (b, G, C)
        accumulator.add(Z, targets[start: start + batch], proteins[start: start + batch])


def _validate(bank: ProbeBank, family: str, inputs: Tensor, targets: Tensor, data: ConceptData, plan: ResamplePlan, results: TaskResults) -> None:
    """A trained bank's validation scores and losses with its kept weights: what best states are chosen by."""
    # inputs: (v, P, w)
    task = data.task
    accumulator = ScoreAccumulator(task.kind, bank.groups, task.outputs, plan, per_label=False)
    score_bank(bank, "best", inputs, targets, accumulator)
    results.add_validation(task.name, family, bank.names, accumulator, validation_loss(bank, inputs, targets, 1, which="best"))


def validate_protein_task(
    data: ConceptData,
    store: PooledTapStore,
    layouts: dict[str, SiteLayout],
    selected: dict[str, dict[str, ProbeBank]],
    settings: TrainingSettings,
    device: torch.device,
    results: TaskResults | None = None,
) -> TaskResults:
    """Score trained banks (`selected[site][family]`, restored from a run) on the validation split of a pooled store."""
    results = results or TaskResults()
    rows = store.rows_for(data.splits["validation"].sequences)
    targets = data.protein_targets("validation").to(device)
    plan = ResamplePlan.draw(len(rows), 0, settings.resample_fraction, settings.seed, device)
    for site, layout in layouts.items():
        loaded = store.load(site, rows, device)  # (v, L, 2 * w)
        for family, _, view in families(layout, settings.pooling):
            _validate(selected[site][family], family, view(loaded), targets, data, plan, results)
        del loaded
    return results


def scrambled_targets(data: ConceptData, sequences: list[str], seed: int, device: torch.device) -> Tensor:
    """Targets of the scrambled control: ProtParam recomputed on the scrambled sequence, else labels shuffled."""
    task = data.task
    if task.kind == "regression":
        scrambled = [scramble(sequence, scramble_seed(sequence, seed)) for sequence in sequences]
        values = data.standardize_values(prot_param_values(scrambled, task.label_names))  # (n, C)
        return torch.tensor(values, dtype=torch.float32, device=device)
    labels = data.splits["test"].labels
    shuffled = [shuffled_labels(label, scramble_seed(sequence, seed)) for label, sequence in zip(labels, sequences, strict=True)]
    if task.kind == "multiclass":
        return torch.tensor(np.asarray(shuffled, dtype=np.int64), device=device)
    return torch.tensor(np.stack([np.atleast_1d(label).astype(np.float32) for label in shuffled]), device=device)


def train_protein_task(
    data: ConceptData,
    store: PooledTapStore,
    layouts: dict[str, SiteLayout],
    settings: TrainingSettings,
    device: torch.device,
    controls: ControlStores | None = None,
    results: TaskResults | None = None,
    save_weights: dict[str, dict[str, Tensor]] | None = None,
) -> TaskResults:
    """Train and score every probe of one protein-level task on a pooled store."""
    task = data.task
    results = results or TaskResults()
    split_rows = {split: store.rows_for(data.splits[split].sequences) for split in ("train", "validation", "test")}
    targets = {split: data.protein_targets(split).to(device) for split in split_rows}
    test_sequences = data.splits["test"].sequences
    plan = ResamplePlan.draw(len(test_sequences), settings.resamples, settings.resample_fraction, settings.seed, device)
    validation_plan = ResamplePlan.draw(len(split_rows["validation"]), 0, settings.resample_fraction, settings.seed, device)
    gaussian_generator = torch.Generator(device="cpu").manual_seed(settings.seed)
    scrambled = scrambled_targets(data, test_sequences, controls.scramble_seed, device) if controls is not None else None

    for site, layout in layouts.items():
        loaded = {split: store.load(site, rows, device) for split, rows in split_rows.items()}  # each (n, L, 2 * w)
        for family, names, view in families(layout, settings.pooling):
            inputs = {split: view(values) for split, values in loaded.items()}  # (n, P, w')
            bank, epochs = fit_family(names, inputs, targets, data, settings, device, f"{task.name} {site} {family}")
            selected, chosen = bank.select_settings(len(settings.grid()))
            results.add_training(task.name, family, selected, chosen, settings, epochs)
            _validate(selected, family, inputs["validation"], targets["validation"], data, validation_plan, results)
            if save_weights is not None:
                save_weights[f"{site}.{family}"] = selected.state("best")
            accumulator = ScoreAccumulator(task.kind, selected.groups, task.outputs, plan)
            score_bank(selected, "best", inputs["test"], targets["test"], accumulator)
            results.add_scores(task.name, family, selected.names, accumulator, "trained", "trained", "original", True)
            if controls is None or family != "full":
                continue
            _score_controls(selected, site, view, data, controls, targets["test"], scrambled, plan, gaussian_generator, results, device)
        del loaded
    return results


def sweep_protein_task(
    data: ConceptData,
    store: PooledTapStore,
    layouts: dict[str, SiteLayout],
    settings: TrainingSettings,
    device: torch.device,
    results: TaskResults | None = None,
) -> TaskResults:
    """Train every probe under every grid setting and score each on the validation split; test is never read."""
    task = data.task
    results = results or TaskResults()
    split_rows = {split: store.rows_for(data.splits[split].sequences) for split in ("train", "validation")}
    targets = {split: data.protein_targets(split).to(device) for split in split_rows}
    plan = ResamplePlan.draw(len(data.splits["validation"].sequences), 0, settings.resample_fraction, settings.seed, device)
    for site, layout in layouts.items():
        loaded = {split: store.load(site, rows, device) for split, rows in split_rows.items()}  # each (n, L, 2 * w)
        for family, names, view in families(layout, settings.pooling):
            started = time.time()
            inputs = {split: view(values) for split, values in loaded.items()}  # (n, P, w')
            bank, epochs = fit_family(names, inputs, targets, data, settings, device, f"sweep {task.name} {site} {family}")
            validation = ScoreAccumulator(task.kind, bank.groups, task.outputs, plan, per_label=False)
            score_bank(bank, "best", inputs["validation"], targets["validation"], validation)
            results.add_sweep(task.name, family, bank, validation, settings, epochs, _elapsed(started, device))
        del loaded
    return results


def _score_controls(
    bank: ProbeBank,
    site: str,
    view: object,
    data: ConceptData,
    controls: ControlStores,
    test_targets: Tensor,
    scrambled: Tensor,
    plan: ResamplePlan,
    generator: torch.Generator,
    results: TaskResults,
    device: torch.device,
) -> None:
    """Every (plm, probe, control) condition but the trained original one, for one site's full-width probes."""
    task = data.task
    sequences = data.splits["test"].sequences
    for plm in ("trained", "untrained"):
        original = view(controls.stores[(plm, "original")].load(site, controls.stores[(plm, "original")].rows_for(sequences), device))
        normalized = view(controls.stores[(plm, "l2_normalized")].load(site, controls.stores[(plm, "l2_normalized")].rows_for(sequences), device))
        scrambled_states = view(controls.stores[(plm, "scrambled")].load(site, controls.stores[(plm, "scrambled")].rows_for(sequences), device))
        mean_states, mean_targets = _mean_control(original, test_targets, task.kind)
        gaussian = torch.randn(original.shape, generator=generator).to(device)  # standard normal, as the reference code draws
        conditions = {
            "original": (original, test_targets),
            "l2_normalized": (normalized, test_targets),
            "mean": (mean_states, mean_targets),
            "scrambled": (scrambled_states, scrambled),
            "random_gaussian": (gaussian, test_targets),
        }
        for probe, which in (("trained", "best"), ("untrained", "initial")):
            for control, (states, labels) in conditions.items():
                if plm == "trained" and probe == "trained" and control == "original":
                    continue
                accumulator = ScoreAccumulator(task.kind, bank.groups, task.outputs, plan)
                score_bank(bank, which, states, labels, accumulator)
                results.add_scores(task.name, "full", bank.names, accumulator, plm, probe, control, False)
