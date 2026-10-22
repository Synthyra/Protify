"""Protein-protein interaction as a probe task: pairs of proteins, read through a symmetric row of their pooled states.

Two pair tasks share the code, each with its own stores (`stores/<task>_*`):

- `ppi`, the 2024 Bernett gold standard (`Synthyra/bernett_gold_ppi`): human protein pairs, half of them interacting,
  split so that no protein appears in two splits, with negatives drawn so that each protein's count of negative pairs
  stays close to its count of positive ones.
- `ppi2`, its 2026 successor (`Synthyra/bernett_gold_ppi_2`): HIPPIE pairs under an integer-program split, no
  protein in two splits and no held-out protein within 40% identity of a training one, with negatives matched to the
  positives on per-protein degree, self-interactions, taxa and GO similarity. Beside its balanced test split it holds
  `test_realistic`, the same positives and proteins with ten uniformly drawn negatives per positive.

A probe reads one row per pair built from the two partners' pooled states, each standardized with the moments of the
training proteins: [z_a + z_b | z_a * z_b], the two symmetric functions of each column pair, so a pair reads the same in
either order. On the sum alone a linear probe can only score each partner on its own; the product lets it weigh columns
that both partners share. Pair rows are built batch by batch from the proteins' rows, so no (pairs, layers, columns)
tensor is ever held.

Everything else follows the protein-level probes (`protein_probing`): one bank per site and family, early stopping on
validation loss, MCC on the full test split and its random halves, validation scores for choosing best states, layer
chunks for wide sites, and the controls, with each pair's label kept for scrambled partners. Each kept probe also gets
its ROC AUC, average precision, MCC at probability 0.5 and balanced MCC on the test pairs (`ranking.parquet`), and the
full-width probes the same on every extra test split. Pairs that share a protein are not independent, so intervals
(`pair_intervals`) come from bootstrap draws of the test proteins, never of the pairs.

Symbols: n pairs; u proteins of a split; P probes of a family; w columns a probe reads of one protein; G groups.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import torch

from dataclasses import dataclass
from pathlib import Path
from torch import Tensor

from .concepts import ConceptTask
from .pooled_store import PooledTapStore
from .probe_bank import ProbeBank
from .protein_probing import ControlStores, bank_logits, families, family_offset, feature_moments, fit_family, keep_state, layer_chunks, score_bank, validation_loss
from .scores import (
    INTERVAL_DRAWS,
    INTERVAL_SEED,
    ResamplePlan,
    ScoreAccumulator,
    balanced_mcc,
    protein_draws,
    ranking_metrics,
    threshold_mcc,
    weighted_ranking_metrics,
)
from .sites import SiteLayout
from .training import MEAN_CONTROL_BATCH, TaskResults, TrainingSettings


@dataclass(frozen=True)
class PairSource:
    """A pair dataset: its task name, its Hub repository, and the held-out splits it adds beside test, which pair the
    test proteins differently."""

    name: str
    repo: str
    extra_tests: tuple[str, ...] = ()


PAIR_SOURCES = {
    "ppi": PairSource("ppi", "Synthyra/bernett_gold_ppi"),
    "ppi2": PairSource("ppi2", "Synthyra/bernett_gold_ppi_2", ("test_realistic",)),
}
PAIR_TASKS = tuple(PAIR_SOURCES)
PAIR_TASK = "ppi"
PAIR_REPO = PAIR_SOURCES[PAIR_TASK].repo
# Our split names, and the datasets'; an extra test split keeps its own name.
PAIR_SPLITS = {"train": "train", "validation": "valid", "test": "test"}
# ESMC's context of 2,048 tokens, less CLS and EOS: longer proteins are cut to their first 2,046 residues.
MAX_PAIR_RESIDUES = 2046
STANDARDIZE_FLOOR = 1e-6
STANDARDIZE_ROWS = 1024  # proteins standardized at once
INTERVAL_METRICS = ("roc_auc", "average_precision")
DRAW_BUDGET = 2 ** 25  # pair weights of one probe held at once while ranking under the draws


def pair_prefix(task: str) -> str:
    """The prefix of a pair task's own stores."""
    return f"{task}_"


@dataclass
class PairSplit:
    first: list[str]  # the first partner's sequence of each pair
    second: list[str]
    labels: np.ndarray  # (n,) 1 for an interacting pair

    @property
    def sequences(self) -> list[str]:
        """Every protein of the split once, in first-seen order."""
        return list(dict.fromkeys(self.first + self.second))

    def partners(self) -> tuple[Tensor, Tensor]:
        """Each pair's two partners as indices into `sequences`."""
        position = {sequence: index for index, sequence in enumerate(self.sequences)}
        return (torch.tensor([position[sequence] for sequence in self.first], dtype=torch.long),
                torch.tensor([position[sequence] for sequence in self.second], dtype=torch.long))  # (n,), (n,)


@dataclass
class PairData:
    task: ConceptTask
    splits: dict[str, PairSplit]

    @property
    def extra_tests(self) -> tuple[str, ...]:
        """The held-out splits beside test."""
        return tuple(split for split in self.splits if split not in PAIR_SPLITS)

    def pair_targets(self, split: str) -> Tensor:
        return torch.tensor(self.splits[split].labels, dtype=torch.float32).unsqueeze(1)  # (n, 1)


def pair_task(name: str = PAIR_TASK) -> ConceptTask:
    return ConceptTask(name, "pair", "binary", 1, ("interaction",))


def load_pairs(
    task: str = PAIR_TASK,
    revision: str | None = None,
    local_dir: str | Path | None = None,
    max_residues: int = MAX_PAIR_RESIDUES,
) -> PairData:
    """The splits of a pair task (`SeqA`, `SeqB`, `labels`): train, validation, test and its extra test splits, from
    its Hub repository or from `local_dir/<split>.parquet` under our split names, with every sequence cut to
    `max_residues`. A local folder may leave the extra test splits out."""
    source = PAIR_SOURCES[task]
    wanted = {**PAIR_SPLITS, **{split: split for split in source.extra_tests}}
    if local_dir is not None:
        frames = {split: pd.read_parquet(Path(local_dir) / f"{split}.parquet") for split in wanted
                  if split in PAIR_SPLITS or (Path(local_dir) / f"{split}.parquet").exists()}
    else:
        from huggingface_hub import HfApi, hf_hub_download

        files = HfApi().list_repo_files(source.repo, repo_type="dataset", revision=revision)

        def fetch(theirs: str) -> pd.DataFrame:
            parts = sorted(name for name in files if name.startswith(f"data/{theirs}-") and name.endswith(".parquet"))
            if not parts:
                raise FileNotFoundError(f"{source.repo} has no data/{theirs}-*.parquet.")
            return pd.concat([pd.read_parquet(hf_hub_download(source.repo, name, repo_type="dataset", revision=revision)) for name in parts],
                             ignore_index=True)

        frames = {split: fetch(theirs) for split, theirs in wanted.items()}
    splits = {
        split: PairSplit(
            first=[sequence[:max_residues] for sequence in frame["SeqA"]],
            second=[sequence[:max_residues] for sequence in frame["SeqB"]],
            labels=frame["labels"].to_numpy(dtype=np.int64),
        )
        for split, frame in frames.items()
    }
    return PairData(pair_task(task), splits)


class PairRows:
    """The rows a pair probe reads, [z_a + z_b | z_a * z_b], built for the pairs asked for when they are asked for."""

    def __init__(self, proteins: Tensor, first: Tensor, second: Tensor) -> None:
        # proteins: (u, P, w) standardized; first, second: (n,) indices into proteins
        self.proteins = proteins
        self.first = first.to(proteins.device)
        self.second = second.to(proteins.device)
        self.shape = (int(first.shape[0]), proteins.shape[1], 2 * proteins.shape[2])
        self.device = proteins.device

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: slice | Tensor) -> Tensor:
        a = self.proteins[self.first[index]].to(torch.float32)  # (b, P, w)
        b = self.proteins[self.second[index]].to(torch.float32)  # (b, P, w)
        return torch.cat([a + b, a * b], dim=-1)  # (b, P, 2 w)


class MeanRows:
    """The mean control of pair rows: each row replaced by the mean row of its group of MEAN_CONTROL_BATCH
    consecutive pairs. Read in contiguous slices, as scoring reads."""

    def __init__(self, rows: PairRows, group: int = MEAN_CONTROL_BATCH) -> None:
        self.rows, self.group = rows, group
        self.shape, self.device = rows.shape, rows.device

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: slice) -> Tensor:
        start, stop, _ = index.indices(self.shape[0])
        first, last = start // self.group * self.group, min(-(-stop // self.group) * self.group, self.shape[0])
        block = self.rows[first:last]  # (m, P, 2 w), whole groups
        owner = torch.arange(last - first, device=block.device) // self.group  # (m,)
        sums = torch.zeros((int(owner[-1]) + 1,) + block.shape[1:], dtype=block.dtype, device=block.device).index_add_(0, owner, block)
        counts = torch.bincount(owner).to(block.dtype).view(-1, 1, 1)
        return (sums / counts)[owner][start - first: stop - first]  # (stop - start, P, 2 w)


class GaussianRows:
    """Standard normal rows in place of pair rows, drawn per slice from a seed and the slice's start."""

    def __init__(self, shape: tuple[int, int, int], seed: int, device: torch.device) -> None:
        self.shape, self.seed, self.device = shape, seed, device

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: slice) -> Tensor:
        start, stop, _ = index.indices(self.shape[0])
        generator = torch.Generator(device="cpu").manual_seed(self.seed * 1_000_003 + start)
        return torch.randn((stop - start,) + self.shape[1:], generator=generator).to(self.device)


def mean_targets(targets: Tensor, group: int = MEAN_CONTROL_BATCH) -> Tensor:
    """The mean control's labels: the most common label of each group of `group` consecutive pairs."""
    # targets: (n, 1)
    return torch.cat([torch.mode(targets[start: start + group].reshape(-1)).values.expand_as(targets[start: start + group])
                      for start in range(0, targets.shape[0], group)])


def pair_metrics(scores: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """ROC AUC, average precision, MCC at probability 0.5 and balanced MCC (Atlas's views, seed 0) of each column of
    logits `scores`."""
    roc_auc, average_precision = ranking_metrics(scores, labels)
    return roc_auc, average_precision, threshold_mcc(scores, labels), balanced_mcc(scores, labels)


@torch.no_grad()
def bank_scores(bank: ProbeBank, which: str, X: PairRows, batch: int = 512) -> Tensor:
    """The logit of every group of a one-output bank for every row: (n, G)."""
    out = []
    for start in range(0, X.shape[0], batch):
        rows = X[start: start + batch]  # (b, P, 2 w)
        out.append(bank_logits(bank, rows, which)[..., 0])  # (b, G)
    return torch.cat(out)  # (n, G)


@dataclass
class PairInputs:
    """One chunk of one site's protein rows for every split, standardized on the training proteins, and the pairs."""

    proteins: dict[str, Tensor]  # split -> (u, L_c, 2 w) bfloat16, standardized
    partners: dict[str, tuple[Tensor, Tensor]]  # split -> each pair's two protein indices

    def rows(self, view: object, split: str) -> PairRows:
        first, second = self.partners[split]
        return PairRows(view(self.proteins[split]), first, second)


def standardized(
    store: PooledTapStore,
    site: str,
    sequences: list[str],
    device: torch.device,
    slots: slice,
    moments: tuple[Tensor, Tensor],
    block: int = STANDARDIZE_ROWS,
) -> Tensor:
    """Pooled rows of `sequences` at one site's layer slots, standardized with the training proteins' moments in place,
    `block` proteins at a time, so no float32 copy of the whole chunk is ever held."""
    mean, scale = moments
    loaded = store.load(site, store.rows_for(sequences), device, slots)  # (u, L_c, 2 w), a copy of its own
    for start in range(0, loaded.shape[0], block):
        rows = loaded[start: start + block]  # (b, L_c, 2 w), a view
        rows.copy_(((rows.to(torch.float32) - mean) * scale).to(torch.bfloat16))
    return loaded


def pair_inputs(data: PairData, store: PooledTapStore, site: str, slots: slice, device: torch.device, splits: tuple[str, ...]) -> tuple[PairInputs, tuple[Tensor, Tensor]]:
    """A chunk's standardized protein rows for `splits` and the training proteins' moments, which controls reuse."""
    train = store.load(site, store.rows_for(data.splits["train"].sequences), device, slots)  # (u, L_c, 2 w)
    mean, variance = feature_moments(train)
    moments = (mean, variance.clamp_min(STANDARDIZE_FLOOR).rsqrt())
    del train
    proteins = {split: standardized(store, site, data.splits[split].sequences, device, slots, moments) for split in splits}
    partners = {split: data.splits[split].partners() for split in splits}
    return PairInputs(proteins, partners), moments


def _validate_pairs(bank: ProbeBank, family: str, rows: PairRows, targets: Tensor, data: PairData, plan: ResamplePlan, results: TaskResults) -> None:
    accumulator = ScoreAccumulator(data.task.kind, bank.groups, data.task.outputs, plan, per_label=False)
    score_bank(bank, "best", rows, targets, accumulator)
    results.add_validation(data.task.name, family, bank.names, accumulator, validation_loss(bank, rows, targets, 1, which="best"))


def train_pair_task(
    data: PairData,
    store: PooledTapStore,
    layouts: dict[str, SiteLayout],
    settings: TrainingSettings,
    device: torch.device,
    controls: ControlStores | None = None,
    results: TaskResults | None = None,
    save_weights: dict[str, dict[str, Tensor]] | None = None,
) -> TaskResults:
    """Train and score every probe of a pair task on a pooled store of its proteins; the full-width probes are also
    ranked on each extra test split."""
    task = data.task
    results = results or TaskResults()
    splits = tuple(PAIR_SPLITS) + data.extra_tests
    targets = {split: data.pair_targets(split).to(device) for split in splits}
    plan = ResamplePlan.draw(len(data.splits["test"].labels), settings.resamples, settings.resample_fraction, settings.seed, device)
    validation_plan = ResamplePlan.draw(len(data.splits["validation"].labels), 0, settings.resample_fraction, settings.seed, device)
    proteins_held = sum(len(data.splits[split].sequences) for split in splits)
    for site, layout in layouts.items():
        for part, slots in layer_chunks(layout, proteins_held, settings.site_budget_gb):
            inputs, moments = pair_inputs(data, store, site, slots, device, splits)
            for family, names, view in families(part, settings.pooling):
                rows = {split: inputs.rows(view, split) for split in splits}
                bank, epochs = fit_family(names, rows, targets, data, settings, device, f"{task.name} {site} {family}", family_offset(layout, slots, family))
                selected, chosen = bank.select_settings(len(settings.grid()))
                results.add_training(task.name, family, selected, chosen, settings, epochs)
                _validate_pairs(selected, family, rows["validation"], targets["validation"], data, validation_plan, results)
                if save_weights is not None:
                    keep_state(save_weights, f"{site}.{family}", selected.state("best"))
                accumulator = ScoreAccumulator(task.kind, selected.groups, task.outputs, plan)
                score_bank(selected, "best", rows["test"], targets["test"], accumulator)
                results.add_scores(task.name, family, selected.names, accumulator, "trained", "trained", "original", True)
                for split in ("test",) + (data.extra_tests if family == "full" else ()):
                    metrics = pair_metrics(bank_scores(selected, "best", rows[split]), targets[split][:, 0])
                    results.add_ranking(task.name, family, selected.names, split, *metrics)
                if controls is not None and family == "full":
                    _score_pair_controls(selected, site, slots, view, data, controls, moments, rows["test"], targets["test"], plan, settings, results, device)
            del inputs
    return results


def sweep_pair_task(
    data: PairData,
    store: PooledTapStore,
    layouts: dict[str, SiteLayout],
    settings: TrainingSettings,
    device: torch.device,
    results: TaskResults | None = None,
) -> TaskResults:
    """Train every probe under every grid setting and score each on the validation split; test is never read."""
    task = data.task
    results = results or TaskResults()
    targets = {split: data.pair_targets(split).to(device) for split in ("train", "validation")}
    plan = ResamplePlan.draw(len(data.splits["validation"].labels), 0, settings.resample_fraction, settings.seed, device)
    proteins_held = sum(len(data.splits[split].sequences) for split in ("train", "validation"))
    for site, layout in layouts.items():
        for part, slots in layer_chunks(layout, proteins_held, settings.site_budget_gb):
            inputs, _ = pair_inputs(data, store, site, slots, device, ("train", "validation"))
            for family, names, view in families(part, settings.pooling):
                started = time.time()
                rows = {split: inputs.rows(view, split) for split in ("train", "validation")}
                bank, epochs = fit_family(names, rows, targets, data, settings, device, f"sweep {task.name} {site} {family}", family_offset(layout, slots, family))
                validation = ScoreAccumulator(task.kind, bank.groups, task.outputs, plan, per_label=False)
                score_bank(bank, "best", rows["validation"], targets["validation"], validation)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                results.add_sweep(task.name, family, bank, validation, settings, epochs, time.time() - started)
            del inputs
    return results


def validate_pair_task(
    data: PairData,
    store: PooledTapStore,
    layouts: dict[str, SiteLayout],
    selected: dict[str, dict[str, ProbeBank]],
    settings: TrainingSettings,
    device: torch.device,
    results: TaskResults | None = None,
) -> TaskResults:
    """Score trained banks (`selected[site][family]`, restored from a run) on the validation pairs."""
    results = results or TaskResults()
    targets = data.pair_targets("validation").to(device)
    plan = ResamplePlan.draw(len(data.splits["validation"].labels), 0, settings.resample_fraction, settings.seed, device)
    proteins_held = sum(len(data.splits[split].sequences) for split in ("train", "validation"))
    for site, layout in layouts.items():
        for part, slots in layer_chunks(layout, proteins_held, settings.site_budget_gb):
            inputs, _ = pair_inputs(data, store, site, slots, device, ("validation",))
            for family, names, view in families(part, settings.pooling):
                first, _ = family_offset(layout, slots, family)
                bank = selected[site][family]
                chunk = bank if len(names) == bank.groups else bank.subset(torch.arange(first, first + len(names), device=bank.device))
                _validate_pairs(chunk, family, inputs.rows(view, "validation"), targets, data, plan, results)
            del inputs
    return results


def pair_intervals(
    data: PairData,
    store: PooledTapStore,
    layouts: dict[str, SiteLayout],
    selected: dict[str, dict[str, ProbeBank]],
    settings: TrainingSettings,
    device: torch.device,
    draws: int = INTERVAL_DRAWS,
    seed: int = INTERVAL_SEED,
    budget: int = DRAW_BUDGET,
) -> tuple[pd.DataFrame, dict[str, Tensor]]:
    """Bootstrap intervals of the ROC AUC and average precision of the full-width probes of a trained pair task
    (`selected[site]["full"]`, restored from a run) on each test split. A draw takes the split's proteins with
    replacement (`protein_draws`) and weighs each pair by the product of its two partners' counts, so the pairs that
    share a protein move together. Returns one row per probe, split and metric (the estimate on the split itself, the
    2.5% and 97.5% quantiles of the draws, the chance level, the draws that held both classes) and each split and
    metric's draws as `<split>.<metric>`: (probes, draws), probes in the rows' order."""
    splits = ("test",) + data.extra_tests
    proteins_held = sum(len(data.splits[split].sequences) for split in ("train",) + splits)
    counts = {split: protein_draws(len(data.splits[split].sequences), draws, seed) for split in splits}  # (B, u) each
    names: list[str] = []
    estimates: dict[tuple[str, str], list[Tensor]] = {(split, metric): [] for split in splits for metric in INTERVAL_METRICS}
    drawn: dict[tuple[str, str], list[Tensor]] = {key: [] for key in estimates}
    for site, layout in layouts.items():
        for part, slots in layer_chunks(layout, proteins_held, settings.site_budget_gb):
            inputs, _ = pair_inputs(data, store, site, slots, device, splits)
            family, chunk_names, view = families(part, settings.pooling)[0]  # the full-width probes
            first, _ = family_offset(layout, slots, family)
            bank = selected[site][family]
            chunk = bank if len(chunk_names) == bank.groups else bank.subset(torch.arange(first, first + len(chunk_names), device=bank.device))
            names += chunk_names
            for split in splits:
                scores = bank_scores(chunk, "best", inputs.rows(view, split))  # (n, G)
                labels = torch.tensor(data.splits[split].labels, device=device)  # (n,)
                partner_a, partner_b = (index.to(device) for index in data.splits[split].partners())
                point = weighted_ranking_metrics(scores, labels, torch.ones(1, scores.shape[0]))
                step = max(1, budget // scores.shape[0])
                blocks = [weighted_ranking_metrics(scores, labels, block[:, partner_a] * block[:, partner_b])
                          for block in counts[split].to(device).split(step)]
                for index, metric in enumerate(INTERVAL_METRICS):
                    estimates[(split, metric)].append(point[index][:, 0].cpu())
                    drawn[(split, metric)].append(torch.cat([block[index] for block in blocks], dim=1).cpu())
            del inputs
    rows, tensors = [], {}
    for (split, metric), parts in drawn.items():
        values = torch.cat(parts)  # (P, B)
        estimate = torch.cat(estimates[(split, metric)])  # (P,)
        low, high = torch.nanquantile(values, torch.tensor([0.025, 0.975], dtype=values.dtype), dim=1)  # (P,) each
        chance = 0.5 if metric == "roc_auc" else float(data.splits[split].labels.mean())
        kept = (~values.isnan()).sum(dim=1)
        rows += [{"task": data.task.name, "family": "full", "probe_name": name, "split": split, "metric": metric,
                  "estimate": float(estimate[index]), "low": float(low[index]), "high": float(high[index]),
                  "chance": chance, "draws": int(kept[index])} for index, name in enumerate(names)]
        tensors[f"{split}.{metric}"] = values.to(torch.float32).contiguous()
    return pd.DataFrame(rows), tensors


def _score_pair_controls(
    bank: ProbeBank,
    site: str,
    slots: slice,
    view: object,
    data: PairData,
    controls: ControlStores,
    moments: tuple[Tensor, Tensor],
    original_rows: PairRows,
    targets: Tensor,
    plan: ResamplePlan,
    settings: TrainingSettings,
    results: TaskResults,
    device: torch.device,
) -> None:
    """Every (plm, probe, control) condition but the trained original one, for one site's full-width pair probes.

    Each control's partners are standardized with the training proteins' moments, as the probe's inputs were; a
    scrambled pair keeps its label."""
    task = data.task
    split = data.splits["test"]
    first, second = split.partners()
    for plm in controls.plm_states:
        def rows_of(inputs: str) -> PairRows:
            store = controls.stores[(plm, inputs)]
            return PairRows(view(standardized(store, site, split.sequences, device, slots, moments)), first, second)

        original = original_rows if plm == "trained" else rows_of("original")
        conditions = {
            "original": (original, targets),
            "l2_normalized": (rows_of("l2_normalized"), targets),
            "mean": (MeanRows(original), mean_targets(targets)),
            "scrambled": (rows_of("scrambled"), targets),
            "random_gaussian": (GaussianRows(original.shape, settings.seed, device), targets),
        }
        for probe, which in (("trained", "best"), ("untrained", "initial")):
            for control, (states, labels) in conditions.items():
                if plm == "trained" and probe == "trained" and control == "original":
                    continue
                accumulator = ScoreAccumulator(task.kind, bank.groups, task.outputs, plan)
                score_bank(bank, which, states, labels, accumulator)
                results.add_scores(task.name, "full", bank.names, accumulator, plm, probe, control, False)
