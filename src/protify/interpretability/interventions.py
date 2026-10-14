"""Phosphomimetic scanning: can a protein's pooled states tell an active mutant from an inactive one?

For a protein with phosphosites annotated in UniProt, every serine, threonine and tyrosine is mutated in turn to
alanine, glycine or glutamate, and each mutant is labeled with the outcome a molecular biologist predicts: glutamate at
an annotated phosphosite mimics the phosphate and keeps the protein active, alanine or glycine there removes the site
and inactivates it, and any change at an unannotated residue has an unknown effect; the wild type is active. Every
sequence of a protein goes through the model once with tap hooks that keep each site's per-layer mean over tokens.

Per protein, a stratified 70:30 split trains an L2-regularized multinomial logistic regression on every site and
layer at once, with the objective of scikit-learn's default `LogisticRegression` (cross entropy summed over mutants
plus half the squared weights, C = 1, intercept unpenalized), solved by L-BFGS for at most 350 iterations. Inputs are
standardized on the protein's training mutants, because mutants of one protein differ by one residue and their pooled
states agree to a few parts in a thousand. Test MCC is recorded for the true labels, for training labels shuffled
after the split, and for the true labels on an untrained copy of the model.

Symbols: m sequences of a protein (wild type and mutants); b sequences of a batch; l token columns; w site width;
L layers of a site; G probes (site layers of one width); n training mutants; C = 3 classes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from torch import Tensor, nn

from .capture import model_dims, tap_hooks
from .concepts import ConceptData
from .pooled_store import encode_batch
from .scores import multiclass_mcc
from .sites import TapSpec, site_layouts
from .token_batches import token_batches


CLASSES = ("active", "inactive", "unknown")
ACTIVE, INACTIVE, UNKNOWN = range(3)
SUBSTITUTIONS = "AGE"
PHOSPHO_LABELS = {"S": "Phosphoserine", "T": "Phosphothreonine", "Y": "Phosphotyrosine"}


@dataclass(frozen=True)
class PhosphoProtein:
    id: str
    sequence: str
    sites: tuple[int, ...]  # 0-based positions of annotated phosphoserines, phosphothreonines and phosphotyrosines


def phosphoproteins(data: ConceptData) -> list[PhosphoProtein]:
    """Proteins of every split whose residue labels annotate a phosphorylated S, T or Y, in a stable order."""
    names = data.task.label_names
    found: dict[str, PhosphoProtein] = {}
    seen: set[str] = set()
    for split in ("train", "validation", "test"):
        part = data.splits[split]
        for protein_id, sequence, labels in zip(part.ids, part.sequences, part.labels, strict=True):
            sites = tuple(
                position for position, (residue, label) in enumerate(zip(sequence, np.asarray(labels).tolist()))
                if residue in PHOSPHO_LABELS and names[label] == PHOSPHO_LABELS[residue]
            )
            if sites and sequence not in seen and protein_id not in found:
                found[protein_id] = PhosphoProtein(protein_id, sequence, sites)
                seen.add(sequence)
    return [found[key] for key in sorted(found)]


def scanning_mutants(protein: PhosphoProtein) -> tuple[list[str], np.ndarray]:
    """The wild type and every S, T and Y substituted by A, G and E, with each sequence's predicted class."""
    sequences, classes = [protein.sequence], [ACTIVE]
    sites = set(protein.sites)
    for position, residue in enumerate(protein.sequence):
        if residue not in PHOSPHO_LABELS:
            continue
        for substitute in SUBSTITUTIONS:
            sequences.append(protein.sequence[:position] + substitute + protein.sequence[position + 1:])
            if position not in sites:
                classes.append(UNKNOWN)
            else:
                classes.append(ACTIVE if substitute == "E" else INACTIVE)
    return sequences, np.asarray(classes, dtype=np.int64)


@torch.inference_mode()
def pooled_means(
    model: nn.Module,
    tokenizer: object,
    sequences: Sequence[str],
    spec: TapSpec,
    device: torch.device,
    max_tokens: int = 131072,
) -> dict[str, Tensor]:
    """Each site's per-layer mean over every attended token (CLS and EOS included), in the order given.

    Means are summed and kept in float32: mutants of one protein differ by about a part in a thousand, below what a
    bfloat16 mean resolves.
    """
    dims = model_dims(model)
    layouts = site_layouts(spec, dims)
    slots = {name: {layer: slot for slot, layer in enumerate(layout.layers)} for name, layout in layouts.items()}
    pooled = {
        name: torch.empty(len(sequences), layout.num_layers, layout.width, dtype=torch.float32, device=device)
        for name, layout in layouts.items()
    }  # each (m, L, w)
    weights: Tensor | None = None
    rows = slice(0, 0)

    def pool(site: str, layer: int, X: Tensor) -> None:
        # X: (b, l, w); weights: (b, l, 1), each row's attended tokens weighted 1 / count
        pooled[site][rows, slots[site][layer]] = (X.float() * weights).sum(dim=1)  # (b, w)

    start = 0
    with tap_hooks(model, spec, pool):
        for batch in token_batches(list(sequences), max_tokens=max_tokens):
            encoded = encode_batch(tokenizer, batch.sequences, device)
            mask = encoded["attention_mask"].to(torch.float32)  # (b, l)
            weights = (mask / mask.sum(dim=1, keepdim=True)).unsqueeze(-1)  # (b, l, 1)
            rows = slice(start, start + len(batch.sequences))
            model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
            start += len(batch.sequences)
    return pooled


def stratified_split(classes: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Train and test indices with each class split in the same proportion, as scikit-learn's stratified split does."""
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(classes))
    train, test = train_test_split(indices, test_size=test_fraction, stratify=classes, random_state=seed)
    return np.sort(train), np.sort(test)


def fit_logistic(X: Tensor, y: Tensor, classes: int, inverse_l2: float = 1.0, iterations: int = 350) -> tuple[Tensor, Tensor]:
    """L2-regularized multinomial logistic regressions of G independent probes, fitted jointly by L-BFGS.

    The objective of each probe is scikit-learn's: summed cross entropy plus ||W||^2 / (2 C), intercept unpenalized.
    The probes share no parameters, so the sum of their objectives is minimized where each one is.
    """
    # X: (G, n, w) standardized; y: (n,) class indices
    groups, rows, width = X.shape
    weight = torch.zeros(groups, width, classes, device=X.device, requires_grad=True)  # (G, w, C)
    bias = torch.zeros(groups, 1, classes, device=X.device, requires_grad=True)  # (G, 1, C)
    targets = y.repeat(groups)  # (G * n,)
    optimizer = torch.optim.LBFGS(
        [weight, bias], lr=1.0, max_iter=iterations, history_size=20, line_search_fn="strong_wolfe",
        tolerance_grad=1e-6, tolerance_change=1e-10,
    )

    def objective() -> Tensor:
        optimizer.zero_grad()
        logits = torch.baddbmm(bias, X, weight)  # (G, n, C)
        loss = F.cross_entropy(logits.reshape(-1, classes), targets, reduction="sum") + weight.square().sum() / (2.0 * inverse_l2)
        loss.backward()
        return loss

    with torch.enable_grad():
        optimizer.step(objective)
    return weight.detach(), bias.detach()


def test_mcc(weight: Tensor, bias: Tensor, X: Tensor, y: Tensor, classes: int) -> Tensor:
    """Multiclass MCC of each probe on the test mutants."""
    # weight: (G, w, C); bias: (G, 1, C); X: (G, t, w); y: (t,)
    predicted = torch.baddbmm(bias, X, weight).argmax(dim=-1)  # (G, t)
    cells = y.unsqueeze(0) * classes + predicted  # (G, t)
    confusion = F.one_hot(cells, classes * classes).sum(dim=1).view(-1, classes, classes)  # (G, C, C)
    return multiclass_mcc(confusion)  # (G,)


def standardize(train: Tensor, test: Tensor, floor: float = 1e-12) -> tuple[Tensor, Tensor]:
    # train: (G, n, w); test: (G, t, w); moments over the training mutants of each probe
    mean = train.mean(dim=1, keepdim=True)  # (G, 1, w)
    scale = train.std(dim=1, unbiased=False, keepdim=True).clamp_min(floor)  # (G, 1, w)
    return (train - mean) / scale, (test - mean) / scale


def probe_protein(
    features: dict[str, Tensor],
    classes: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    shuffled: np.ndarray,
    iterations: int,
) -> dict[str, dict[str, Tensor]]:
    """Test MCC of every site layer, keyed by label set ("true", "shuffled") and site: each (L,)."""
    device = next(iter(features.values())).device
    train_rows, test_rows = torch.from_numpy(train).to(device), torch.from_numpy(test).to(device)
    y = torch.from_numpy(classes).to(device)
    label_sets = {"true": y[train_rows], "shuffled": torch.from_numpy(shuffled).to(device)}
    test_y = y[test_rows]
    by_width: dict[int, list[str]] = {}
    for site, values in features.items():
        by_width.setdefault(values.shape[-1], []).append(site)
    scores: dict[str, dict[str, Tensor]] = {name: {} for name in label_sets}
    for sites in by_width.values():
        stacked = torch.cat([features[site].transpose(0, 1) for site in sites])  # (G, m, w)
        X_train, X_test = standardize(stacked[:, train_rows], stacked[:, test_rows])  # (G, n, w), (G, t, w)
        for name, labels in label_sets.items():
            weight, bias = fit_logistic(X_train, labels, len(CLASSES), iterations=iterations)
            mcc = test_mcc(weight, bias, X_test, test_y, len(CLASSES))  # (G,)
            offset = 0
            for site in sites:
                layers = features[site].shape[1]
                scores[name][site] = mcc[offset: offset + layers]
                offset += layers
    return scores


def run_phospho_interventions(
    model: nn.Module,
    untrained: nn.Module,
    tokenizer: object,
    data: ConceptData,
    spec: TapSpec,
    out: Path,
    device: torch.device,
    proteins: int = 1000,
    seed: int = 184502,
    max_length: int = 1022,
    iterations: int = 350,
    max_tokens: int = 131072,
) -> pd.DataFrame:
    """Score `proteins` randomly chosen phosphoproteins, appending one parquet part per protein under `out`.

    Proteins already written are skipped, so an interrupted run resumes. Returns every row written so far.
    """
    out.mkdir(parents=True, exist_ok=True)
    candidates = [protein for protein in phosphoproteins(data) if len(protein.sequence) <= max_length]
    rng = np.random.default_rng(seed)
    chosen = [candidates[index] for index in rng.choice(len(candidates), size=min(proteins, len(candidates)), replace=False)]
    layers = {site: layout.layers for site, layout in site_layouts(spec, model_dims(model)).items()}
    for position, protein in enumerate(chosen):
        part = out / f"{position:04d}_{protein.id}.parquet"
        if part.exists():
            continue
        sequences, classes = scanning_mutants(protein)
        protein_seed = seed + position
        train, test = stratified_split(classes, 0.3, protein_seed)
        shuffled = np.random.default_rng(protein_seed).permutation(classes[train])
        rows = []
        for plm, network in (("trained", model), ("untrained", untrained)):
            features = pooled_means(network, tokenizer, sequences, spec, device, max_tokens)  # site -> (m, L, w)
            scores = probe_protein(features, classes, train, test, shuffled, iterations)
            for labels, by_site in scores.items():
                if plm == "untrained" and labels == "shuffled":
                    continue
                for site, values in by_site.items():
                    rows.extend(
                        {"protein": protein.id, "position": position, "plm": plm, "labels": labels, "site": site,
                         "layer": layer, "mcc": float(value), "mutants": len(sequences), "sites": len(protein.sites)}
                        for layer, value in zip(layers[site], values.cpu().tolist(), strict=True)
                    )
        temporary = part.with_suffix(".tmp")
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        temporary.replace(part)
    return pd.concat([pd.read_parquet(path) for path in sorted(out.glob("*.parquet"))], ignore_index=True)


def paired_tests(scores: pd.DataFrame) -> pd.DataFrame:
    """Per site and layer: mean MCC by condition, and the Wilcoxon signed-rank test of true against shuffled labels,
    paired by protein, with its Bonferroni-corrected p-value over every site layer tested."""
    from scipy.stats import wilcoxon

    table = scores.pivot_table(index=["site", "layer", "protein"], columns=["plm", "labels"], values="mcc")
    rows = []
    for (site, layer), block in table.groupby(level=["site", "layer"]):
        true, shuffled = block[("trained", "true")], block[("trained", "shuffled")]
        statistic, p_value = wilcoxon(true, shuffled) if not np.allclose(true, shuffled) else (np.nan, 1.0)
        rows.append({
            "site": site, "layer": layer, "proteins": len(block),
            "trained_true": true.mean(), "trained_true_std": true.std(),
            "trained_shuffled": shuffled.mean(), "untrained_true": block[("untrained", "true")].mean(),
            "statistic": statistic, "p_value": p_value,
        })
    tests = pd.DataFrame(rows)
    tests["p_bonferroni"] = (tests["p_value"] * len(tests)).clip(upper=1.0)
    return tests
