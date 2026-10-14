"""Residue-level probes trained during the forward pass: no token state is ever stored.

Each batch runs ESMC once with tap hooks. As a state arrives the hook gathers its residue rows (CLS, EOS and
padding dropped), and every probe that reads that site and layer takes its gradient there, so memory holds one
state at a time. One optimizer step follows each forward pass. Validation passes run the same hooks without
gradients, and the test passes run the model four times (trained and untrained weights, original and scrambled
sequences) to score the controls.

Symbols: b sequences; l token columns; n residue rows of a batch; w site width; L layers of a site; H heads;
d_h head width; C classes; K grid settings.
"""

from __future__ import annotations

import copy
import time
import numpy as np
import torch
import torch.nn.functional as F

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from torch import Tensor, nn
from tqdm.auto import tqdm

from .capture import tap_hooks
from .concepts import ConceptData, scramble, scramble_seed, shuffled_labels
from .probe_bank import ProbeBank
from .scores import ResamplePlan, ScoreAccumulator
from .sites import SiteLayout, TapSpec
from .token_batches import TokenBatch, token_batches
from .training import MEAN_CONTROL_BATCH, TaskResults, TrainingSettings, make_bank


@dataclass
class ResidueBatch:
    input_ids: Tensor  # (b, l)
    attention_mask: Tensor  # (b, l)
    flat_index: Tensor  # (n,) residue positions in the row-major (b * l) grid
    positions: Tensor  # (n,) token column of each residue row
    proteins: Tensor  # (n,) index of each row's protein in its split
    targets: Tensor  # (n,) class of each residue
    protein_index: tuple[Tensor, Tensor]  # torch.unique(proteins, return_inverse=True): (p,), (n,)


def residue_batch(
    tokenizer: object,
    batch: TokenBatch,
    labels: Sequence[np.ndarray],
    device: torch.device,
    inputs: Sequence[str] | None = None,
) -> ResidueBatch:
    """Encode a batch (CLS, residues, EOS) and index its residue rows; `inputs` replaces the sequences fed in."""
    fed = list(inputs) if inputs is not None else list(batch.sequences)
    encoded = tokenizer(fed, padding=True, return_tensors="pt")
    width = encoded["input_ids"].shape[1]
    flat, positions, owners, classes = [], [], [], []
    for row, index in enumerate(batch.indices):
        residues = len(batch.sequences[row])
        label = np.asarray(labels[index], dtype=np.int64)
        assert label.shape == (residues,), f"protein {index} has {residues} residues but {label.shape} labels"
        columns = np.arange(1, residues + 1)  # column 0 is CLS
        flat.append(row * width + columns)
        positions.append(columns)
        owners.append(np.full(residues, index))
        classes.append(label)
    proteins = torch.from_numpy(np.concatenate(owners))  # (n,)
    local, inverse = torch.unique(proteins, return_inverse=True)  # on the host, so scoring never waits on the device
    return ResidueBatch(
        input_ids=encoded["input_ids"].to(device, non_blocking=True),
        attention_mask=encoded["attention_mask"].to(device, non_blocking=True),
        flat_index=torch.from_numpy(np.concatenate(flat)).to(device, non_blocking=True),
        positions=torch.from_numpy(np.concatenate(positions)).to(device, non_blocking=True),
        proteins=proteins.to(device, non_blocking=True),
        targets=torch.from_numpy(np.concatenate(classes)).to(device, non_blocking=True),
        protein_index=(local.to(device, non_blocking=True), inverse.to(device, non_blocking=True)),
    )


class SiteBanks:
    """The full-width bank and, for head-split sites, the per-head bank of one site."""

    def __init__(self, layout: SiteLayout, outputs: int, settings: TrainingSettings, device: torch.device) -> None:
        self.layout = layout
        self.settings_count = len(settings.grid())
        self.slot = {layer: slot for slot, layer in enumerate(layout.layers)}
        self.full = make_bank(layout.full_names(), layout.width, outputs, "multiclass", settings, settings.seed, device)
        self.head = (
            make_bank(layout.head_names(), layout.head_width, outputs, "multiclass", settings, settings.seed, device)
            if layout.heads else None
        )
        self.shared = settings.probe.kind == "linear"
        self._groups = {layer: self._layer_groups(layer, device) for layer in layout.layers}

    def _layer_groups(self, layer: int, device: torch.device) -> list[tuple[ProbeBank, Tensor, int]]:
        slot, probes, count = self.slot[layer], self.layout.num_layers, self.settings_count
        groups = [(self.full, torch.tensor([setting * probes + slot for setting in range(count)], device=device), 1)]
        if self.head is not None:
            heads = self.layout.heads
            index = [setting * probes * heads + slot * heads + head for setting in range(count) for head in range(heads)]
            groups.append((self.head, torch.tensor(index, device=device), heads))
        return groups

    def groups(self, layer: int) -> list[tuple[ProbeBank, Tensor, int]]:
        """(bank, group index setting-major then head, heads) of every probe that reads this layer's rows."""
        return self._groups[layer]

    def center(self, rows: Tensor, layer: int) -> Tensor:
        """Rows minus the layer's training mean, in bf16, once for every bank and head that reads them."""
        # rows: (n, w); the head probes' means are the full-width mean split by head
        mean = self.full.mean[self.slot[layer]]  # (w,)
        return (rows - mean.to(rows.dtype)).to(torch.bfloat16)

    def ranges(self, layer: int) -> list[tuple[ProbeBank, int, int, int]]:
        """(bank, start, stop, heads) for the groups of one layer in every grid setting."""
        slot = self.slot[layer]
        spans = []
        probes = self.layout.num_layers
        for setting in range(self.settings_count):
            start = setting * probes + slot
            spans.append((self.full, start, start + 1, 1))
        if self.head is not None:
            heads = self.layout.heads
            probes = self.layout.num_layers * heads
            for setting in range(self.settings_count):
                start = setting * probes + slot * heads
                spans.append((self.head, start, start + heads, heads))
        return spans

    def banks(self) -> list[ProbeBank]:
        return [self.full] + ([self.head] if self.head is not None else [])


def _rows(X: Tensor, flat_index: Tensor) -> Tensor:
    # X: (b, l, w) -> (n, w)
    return X.reshape(-1, X.shape[-1]).index_select(0, flat_index)


def _grouped(rows: Tensor, heads: int) -> Tensor:
    # rows: (n, w) -> (n, 1, w), or (n, H, d_h) per head
    return rows.unsqueeze(1) if heads == 1 else rows.view(rows.shape[0], heads, rows.shape[1] // heads)


def _layer_logits(bank: ProbeBank, states: Tensor, start: int, heads: int, which: str) -> Tensor:
    """Outputs of the `heads` groups from `start` of a selected bank (one setting) for rows they all read."""
    # states: (n, w) -> (n, heads, C)
    if bank.settings.kind == "linear":
        return bank.shared_logits(states, torch.arange(start, start + heads, device=states.device), heads, which)
    return bank.logits(_grouped(states, heads), start, start + heads, which)


def _run(model: nn.Module, spec: TapSpec, consumer: Callable[[str, int, Tensor], None], batch: ResidueBatch) -> None:
    with tap_hooks(model, spec, consumer):
        model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)


@torch.no_grad()
def fit_standardization(
    model: nn.Module,
    tokenizer: object,
    spec: TapSpec,
    banks: dict[str, SiteBanks],
    data: ConceptData,
    settings: TrainingSettings,
    device: torch.device,
    max_proteins: int = 512,
) -> None:
    """Set every bank's column means and variances from the residue rows of up to `max_proteins` train proteins."""
    split = data.splits["train"]
    rng = np.random.default_rng(settings.seed)
    chosen = sorted(rng.choice(len(split.sequences), size=min(max_proteins, len(split.sequences)), replace=False).tolist())
    sequences = [split.sequences[index] for index in chosen]
    labels = [split.labels[index] for index in chosen]
    sums: dict[tuple[str, int], list[Tensor]] = {}

    def gather(site: str, layer: int, X: Tensor) -> None:
        rows = _rows(X, batch.flat_index).to(torch.float64)  # (n, w)
        moments = sums.setdefault((site, layer), [torch.zeros(rows.shape[1], dtype=torch.float64, device=device) for _ in range(2)] + [torch.zeros((), dtype=torch.float64, device=device)])
        moments[0] += rows.sum(dim=0)
        moments[1] += rows.square().sum(dim=0)
        moments[2] += rows.shape[0]

    for token_batch in token_batches(sequences, settings.max_tokens):
        batch = residue_batch(tokenizer, token_batch, labels, device)
        _run(model, spec, gather, batch)
    for site, site_banks in banks.items():
        layout = site_banks.layout
        means = []
        variances = []
        for layer in layout.layers:
            total, squares, count = sums[(site, layer)]
            mean = total / count
            means.append(mean)
            variances.append((squares / count - mean.square()).clamp_min(0))
        mean = torch.stack(means).to(torch.float32)  # (L, w)
        variance = torch.stack(variances).to(torch.float32)  # (L, w)
        count = site_banks.settings_count
        site_banks.full.set_standardization(mean.repeat(count, 1), variance.repeat(count, 1))
        if site_banks.head is not None:
            heads = layout.heads
            head_mean = mean.view(layout.num_layers * heads, layout.head_width)  # (L * H, d_h)
            head_variance = variance.view(layout.num_layers * heads, layout.head_width)  # (L * H, d_h)
            site_banks.head.set_standardization(head_mean.repeat(count, 1), head_variance.repeat(count, 1))


@torch.no_grad()
def validation_losses(
    model: nn.Module,
    tokenizer: object,
    spec: TapSpec,
    banks: dict[str, SiteBanks],
    data: ConceptData,
    settings: TrainingSettings,
    device: torch.device,
) -> dict[int, Tensor]:
    """Mean validation loss of every group of every bank, keyed by the bank's id."""
    split = data.splits["validation"]
    totals = {id(bank): torch.zeros(bank.groups, device=device) for site_banks in banks.values() for bank in site_banks.banks()}
    rows_seen = 0

    def score(site: str, layer: int, X: Tensor) -> None:
        rows = _rows(X, batch.flat_index)  # (n, w)
        if banks[site].shared:
            centered = banks[site].center(rows, layer)  # (n, w) bf16
            for bank, index, heads in banks[site].groups(layer):
                totals[id(bank)].index_add_(0, index, bank.shared_loss_sums(centered, batch.targets, index, heads, centered=True))
            return
        for bank, start, stop, heads in banks[site].ranges(layer):
            totals[id(bank)][start:stop] += bank.loss_sums(_grouped(rows, heads), batch.targets, start, stop)

    for token_batch in token_batches(split.sequences, settings.max_tokens):
        batch = residue_batch(tokenizer, token_batch, split.labels, device)
        _run(model, spec, score, batch)
        rows_seen += batch.targets.shape[0]
    return {key: value / rows_seen for key, value in totals.items()}


def fit_residue_banks(
    model: nn.Module,
    tokenizer: object,
    spec: TapSpec,
    layouts: dict[str, SiteLayout],
    data: ConceptData,
    settings: TrainingSettings,
    device: torch.device,
    watch: Callable[[Iterable[TokenBatch]], Iterable[TokenBatch]] | None = None,
) -> tuple[dict[str, SiteBanks], int]:
    """Standardize and train every probe of one residue task by streaming states. Returns the banks and epochs run.

    `watch` wraps the first epoch's batches, as `foundry.pilot.Pilot.watch` does, for a pilot: training stops after
    the batches it yields.
    """
    task = data.task
    banks = {site: SiteBanks(layout, task.outputs, settings, device) for site, layout in layouts.items()}
    fit_standardization(model, tokenizer, spec, banks, data, settings, device)
    split = data.splits["train"]
    all_banks = [bank for site_banks in banks.values() for bank in site_banks.banks()]
    epochs = 0
    steps = 0
    active = sum(bank.groups for bank in all_banks)

    def learn(site: str, layer: int, X: Tensor) -> None:
        rows = _rows(X, batch.flat_index)  # (n, w)
        if banks[site].shared:
            centered = banks[site].center(rows, layer)  # (n, w) bf16
            for bank, index, heads in banks[site].groups(layer):
                bank.shared_accumulate(centered, batch.targets, index, heads, centered=True)
            return
        for bank, start, stop, heads in banks[site].ranges(layer):
            bank.accumulate(_grouped(rows, heads), batch.targets, start, stop)

    for epoch in range(settings.max_epochs):
        epochs = epoch + 1
        batches = token_batches(split.sequences, settings.max_tokens, shuffle=True, seed=settings.seed + epoch)
        validate_every = settings.validate_every or len(batches)
        bar = tqdm(batches, desc=f"{task.name} epoch {epochs}", leave=False, mininterval=10)
        for token_batch in (watch(bar) if watch is not None else bar):
            batch = residue_batch(tokenizer, token_batch, split.labels, device)
            with torch.no_grad():
                _run(model, spec, learn, batch)
            for bank in all_banks:
                bank.step()
            steps += 1
            if steps % validate_every == 0:
                losses = validation_losses(model, tokenizer, spec, banks, data, settings, device)
                active = sum(bank.observe(losses[id(bank)], settings.patience) for bank in all_banks)
                bar.set_postfix(active=active)
                if active == 0:
                    break
        if watch is not None or active == 0:
            break
    return banks, epochs


def train_residue_task(
    model: nn.Module,
    tokenizer: object,
    spec: TapSpec,
    layouts: dict[str, SiteLayout],
    data: ConceptData,
    settings: TrainingSettings,
    device: torch.device,
    results: TaskResults | None = None,
    save_weights: dict[str, dict[str, Tensor]] | None = None,
    watch: Callable[[Iterable[TokenBatch]], Iterable[TokenBatch]] | None = None,
) -> TaskResults:
    """Train every probe of one residue-level task by streaming states from the model, then score it.

    Under a pilot's `watch`, training stops after the batches it yields and nothing is scored.
    """
    task = data.task
    results = results or TaskResults()
    banks, epochs = fit_residue_banks(model, tokenizer, spec, layouts, data, settings, device, watch)
    if watch is not None:
        return results

    selected: dict[str, dict[str, ProbeBank]] = {}
    for site, site_banks in banks.items():
        selected[site] = {}
        for family, bank in (("full", site_banks.full), ("head", site_banks.head)):
            if bank is None:
                continue
            reduced, chosen = bank.select_settings(site_banks.settings_count)
            selected[site][family] = reduced
            results.add_training(task.name, family, reduced, chosen, settings, epochs)
            if save_weights is not None:
                save_weights[f"{site}.{family}"] = reduced.state("best")
    score_residue_task(model, tokenizer, spec, layouts, selected, data, settings, device, results)
    return results


@torch.no_grad()
def score_validation_settings(
    model: nn.Module,
    tokenizer: object,
    spec: TapSpec,
    banks: dict[str, SiteBanks],
    data: ConceptData,
    settings: TrainingSettings,
    device: torch.device,
) -> dict[int, ScoreAccumulator]:
    """Validation scores of every group of every bank, each grid setting with its best weights, keyed by bank id."""
    split = data.splits["validation"]
    plan = ResamplePlan.draw(len(split.sequences), 0, settings.resample_fraction, settings.seed, device)
    accumulators = {
        id(bank): ScoreAccumulator(data.task.kind, bank.groups, bank.outputs, plan)
        for site_banks in banks.values() for bank in site_banks.banks()
    }

    def score(site: str, layer: int, X: Tensor) -> None:
        rows = _rows(X, batch.flat_index)  # (n, w)
        site_banks = banks[site]
        spans = site_banks.ranges(layer)  # (bank, start, stop, heads) for each grid setting
        if not site_banks.shared:
            for bank, start, stop, heads in spans:
                Z = bank.logits(_grouped(rows, heads), start, stop, "best")  # (n, heads, C)
                accumulators[id(bank)].add(Z, batch.targets, batch.proteins, start, batch.protein_index)
            return
        centered = site_banks.center(rows, layer)  # (n, w) bf16
        for bank, index, heads in site_banks.groups(layer):
            Z = bank.shared_logits(centered, index, heads, "best", centered=True)  # (n, K * heads, C), setting-major
            starts = [start for owner, start, _, _ in spans if owner is bank]
            for setting, start in enumerate(starts):
                accumulators[id(bank)].add(Z[:, setting * heads: (setting + 1) * heads], batch.targets, batch.proteins, start, batch.protein_index)

    for token_batch in token_batches(split.sequences, settings.max_tokens):
        batch = residue_batch(tokenizer, token_batch, split.labels, device)
        _run(model, spec, score, batch)
    return accumulators


def sweep_residue_task(
    model: nn.Module,
    tokenizer: object,
    spec: TapSpec,
    layouts: dict[str, SiteLayout],
    data: ConceptData,
    settings: TrainingSettings,
    device: torch.device,
    results: TaskResults | None = None,
) -> TaskResults:
    """Train every probe under every grid setting and score each on the validation split; test is never read."""
    results = results or TaskResults()
    started = time.time()
    banks, epochs = fit_residue_banks(model, tokenizer, spec, layouts, data, settings, device)
    validation = score_validation_settings(model, tokenizer, spec, banks, data, settings, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.time() - started  # the whole task: every site streams from the same forward passes
    for site_banks in banks.values():
        for family, bank in (("full", site_banks.full), ("head", site_banks.head)):
            if bank is not None:
                results.add_sweep(data.task.name, family, bank, validation[id(bank)], settings, epochs, seconds)
    return results


def untrained_copy(model: nn.Module, seed: int) -> nn.Module:
    """The same architecture with every module's parameters reset to PyTorch's defaults."""
    torch.manual_seed(seed)
    reset = copy.deepcopy(model)
    for module in reset.modules():
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()
    return reset.eval()


@torch.no_grad()
def score_residue_task(
    model: nn.Module,
    tokenizer: object,
    spec: TapSpec,
    layouts: dict[str, SiteLayout],
    selected: dict[str, dict[str, ProbeBank]],
    data: ConceptData,
    settings: TrainingSettings,
    device: torch.device,
    results: TaskResults,
) -> None:
    """Score the selected probes on the test split, with every control when `settings.controls` is set."""
    task = data.task
    split = data.splits["test"]
    plan = ResamplePlan.draw(len(split.sequences), settings.resamples, settings.resample_fraction, settings.seed, device)
    slots = {site: {layer: slot for slot, layer in enumerate(layout.layers)} for site, layout in layouts.items()}
    accumulators: dict[tuple[str, str, str, str, str], ScoreAccumulator] = {}

    def accumulator(site: str, family: str, plm: str, probe: str, control: str) -> ScoreAccumulator:
        key = (site, family, plm, probe, control)
        if key not in accumulators:
            bank = selected[site][family]
            accumulators[key] = ScoreAccumulator(task.kind, bank.groups, task.outputs, plan)
        return accumulators[key]

    generator = torch.Generator(device=device).manual_seed(settings.seed)
    scrambled_inputs = [scramble(sequence, scramble_seed(sequence, settings.seed)) for sequence in split.sequences]
    scrambled_labels = [shuffled_labels(label, scramble_seed(sequence, settings.seed)) for label, sequence in zip(split.labels, split.sequences, strict=True)]
    variants = [("trained", False)]
    if settings.controls:
        variants += [("trained", True), ("untrained", False), ("untrained", True)]
    untrained = untrained_copy(model, settings.seed) if settings.controls else None
    # the mean control averages batches of MEAN_CONTROL_BATCH proteins; without controls, batches fill the token budget
    batches = [
        TokenBatch(tuple(range(start, min(start + MEAN_CONTROL_BATCH, len(split.sequences)))),
                   tuple(split.sequences[start: start + MEAN_CONTROL_BATCH]))
        for start in range(0, len(split.sequences), MEAN_CONTROL_BATCH)
    ] if settings.controls else token_batches(split.sequences, settings.max_tokens)
    for plm, is_scrambled in variants:
        network = model if plm == "trained" else untrained
        labels = scrambled_labels if is_scrambled else split.labels
        for token_batch in tqdm(batches, desc=f"{task.name} test {plm}{' scrambled' if is_scrambled else ''}", leave=False, mininterval=10):
            fed = [scrambled_inputs[index] for index in token_batch.indices] if is_scrambled else None
            batch = residue_batch(tokenizer, token_batch, labels, device, inputs=fed)
            mode = torch.mode(batch.targets).values

            def score(site: str, layer: int, X: Tensor) -> None:
                rows = _rows(X, batch.flat_index)  # (n, w)
                slot = slots[site][layer]
                for family, bank in selected[site].items():
                    heads = layouts[site].heads if family == "head" else 1
                    start, stop = slot * heads, (slot + 1) * heads
                    if family == "head" or not settings.controls:
                        if plm == "trained" and not is_scrambled:
                            Z = _layer_logits(bank, rows, start, heads, "best")  # (n, g, C)
                            accumulator(site, family, plm, "trained", "original").add(Z, batch.targets, batch.proteins, start, batch.protein_index)
                        continue
                    if is_scrambled:
                        inputs = {"scrambled": (rows, batch.targets)}
                    else:
                        masked = X.to(torch.float32) * batch.attention_mask.unsqueeze(-1)  # (b, l, w)
                        positional_mean = masked.mean(dim=0)  # (l, w)
                        inputs = {
                            "original": (rows, batch.targets),
                            "l2_normalized": (F.normalize(rows.to(torch.float32), dim=-1), batch.targets),
                            "mean": (positional_mean.index_select(0, batch.positions), mode.expand_as(batch.targets)),
                            "random_gaussian": (torch.randn(rows.shape, generator=generator, device=device), batch.targets),
                        }
                    for probe, which in (("trained", "best"), ("untrained", "initial")):
                        for control, (states, targets) in inputs.items():
                            Z = _layer_logits(bank, states, start, heads, which)  # (n, g, C)
                            accumulator(site, family, plm, probe, control).add(Z, targets, batch.proteins, start, batch.protein_index)

            _run(network, spec, score, batch)
    for (site, family, plm, probe, control), scores in accumulators.items():
        keep = plm == "trained" and probe == "trained" and control == "original"
        results.add_scores(task.name, family, selected[site][family].names, scores, plm, probe, control, keep)
    del untrained
