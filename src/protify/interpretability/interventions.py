"""Phosphomimetic scanning: can a protein's pooled states tell an active mutant from an inactive one?

For a protein with phosphosites annotated in UniProt, every serine, threonine and tyrosine is mutated in turn to
alanine, glycine or glutamate, and each mutant is labeled with the outcome a molecular biologist predicts: glutamate at
an annotated phosphosite mimics the phosphate and keeps the protein active, alanine or glycine there removes the site
and inactivates it, and any change at an unannotated residue has an unknown effect; the wild type is active. Every
sequence of a protein goes through the model once with tap hooks that keep each site's per-layer mean over tokens.

Per protein, a stratified 70:30 split trains an L2-regularized multinomial logistic regression on every site and
layer at once, with the objective of scikit-learn's default `LogisticRegression` (cross entropy summed over mutants
plus half the squared weights, C = 1, intercept unpenalized), solved to convergence by Newton's method. A protein has
far fewer training mutants than a site has columns, and the optimal weights lie in the span of the training rows, so
each regression is solved exactly in that span (`RowSpace`), where its Hessian is small enough to factor. Every regression is fit
twice: on the raw pooled means, as scikit-learn's default sees them, and on means standardized over the protein's
training mutants. Mutants of one protein differ by one residue, so their raw pooled states agree to a few parts in a
thousand and the penalty decides what a raw fit can use; standardizing puts every column's variation on one scale.
Test MCC is recorded for the true labels, for training labels shuffled after the split, and for the true labels on
an untrained copy of the model.

Symbols: m sequences of a protein (wild type and mutants); b sequences of a batch; l token columns; w site width;
L layers of a site; G probes (site layers of one width); n training mutants; t test mutants; C = 3 classes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from collections.abc import Callable, Iterable, Sequence
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
INPUTS = ("raw", "standardized")
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


def _newton_objective(features: Tensor, theta: Tensor, onehot: Tensor, penalty: Tensor) -> Tensor:
    # features: (G, n, r + 1); theta: (..., G, r + 1, C); onehot: (n, C); penalty: (r + 1, 1), 1 / C on weights, 0 on bias
    logits = features @ theta  # (..., G, n, C)
    loss = -(torch.log_softmax(logits, dim=-1) * onehot).sum(dim=(-1, -2))  # (..., G)
    return loss + 0.5 * (penalty * theta.square()).sum(dim=(-1, -2))


def _newton_fit(features: Tensor, onehot: Tensor, penalty: Tensor, iterations: int, tolerance: float) -> Tensor:
    """Damped Newton steps on every probe at once, each probe with its own backtracking line search."""
    groups, rows, size = features.shape
    classes = onehot.shape[1]
    theta = features.new_zeros(groups, size, classes)  # (G, r + 1, C)
    # the bias, whose shift across all classes leaves every probability unchanged, gets a vanishing ridge
    ridge = (penalty[:, 0] + 1e-9).repeat(classes)  # (C * (r + 1),)
    steps = features.new_tensor([0.5**k for k in range(12)]).view(-1, 1, 1, 1)  # (12, 1, 1, 1)
    for _ in range(iterations):
        P = torch.softmax(features @ theta, dim=-1)  # (G, n, C)
        gradient = features.transpose(1, 2) @ (P - onehot) + penalty * theta  # (G, r + 1, C)
        hessian = features.new_zeros(groups, classes, size, classes, size)
        for c in range(classes):
            for d in range(c, classes):
                curvature = P[..., c] * (float(c == d) - P[..., d])  # (G, n)
                block = (features * curvature.unsqueeze(-1)).transpose(1, 2) @ features  # (G, r + 1, r + 1)
                hessian[:, c, :, d] = block
                hessian[:, d, :, c] = block.transpose(1, 2)
        hessian = hessian.reshape(groups, classes * size, classes * size)
        hessian.diagonal(dim1=1, dim2=2).add_(ridge)
        flat = gradient.transpose(1, 2).reshape(groups, classes * size, 1)  # class-major, as the Hessian is laid out
        factor, _ = torch.linalg.cholesky_ex(hessian)
        direction = torch.cholesky_solve(flat, factor).reshape(groups, classes, size).transpose(1, 2)  # (G, r + 1, C)
        decrement = (gradient * direction).sum(dim=(1, 2))  # (G,) the squared Newton decrement
        if float(decrement.max()) < tolerance:
            break
        current = _newton_objective(features, theta, onehot, penalty)  # (G,)
        trial = _newton_objective(features, theta - steps * direction, onehot, penalty)  # (12, G)
        accepted = trial <= current - 1e-4 * steps.view(-1, 1) * decrement  # (12, G) the Armijo condition
        first = torch.where(accepted.any(dim=0), accepted.to(torch.int8).argmax(dim=0), steps.shape[0] - 1)  # (G,)
        theta = theta - steps.view(-1)[first].view(-1, 1, 1) * direction
    return theta


@dataclass
class RowSpace:
    """The training rows of G probes, centered, and a factor of their Gram matrix: the basis a regression is solved in.

    With K = X_c X_c^T = L L^T, weights in the span of the rows are W = X_c^T L^{-T} Z, the logits X_c W are L Z and
    ||W||^2 is ||Z||^2, so L is an exact n-column design for the penalized regression. The intercept absorbs the
    centering, which removes the rows' shared component: raw pooled means of one protein's mutants agree to a few
    parts in a thousand, and their uncentered Gram matrix is dominated by what they share.
    """

    mean: Tensor  # (G, 1, w) column means of the training rows
    centered: Tensor  # (G, n, w) float64
    factor: Tensor  # (G, n, n) lower-triangular L
    features: Tensor  # (G, n, n + 1): L and an intercept column


def row_space(X: Tensor, jitter: float = 1e-10) -> RowSpace:
    """Center each probe's training rows and factor their Gram matrix, plus `jitter` times its mean diagonal (centered
    rows are rank deficient by one; the jitter adds each row a private direction too small for the penalty to use)."""
    # X: (G, n, w)
    mean = X.to(torch.float64).mean(dim=1, keepdim=True)  # (G, 1, w)
    centered = X.to(torch.float64) - mean  # (G, n, w)
    gram = centered @ centered.transpose(1, 2)  # (G, n, n)
    diagonal = gram.diagonal(dim1=1, dim2=2)  # (G, n), a view
    diagonal.add_(jitter * diagonal.mean(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny))
    factor, _ = torch.linalg.cholesky_ex(gram)  # (G, n, n)
    features = torch.cat([factor, factor.new_ones(X.shape[0], X.shape[1], 1)], dim=-1)  # (G, n, n + 1)
    return RowSpace(mean, centered, factor, features)


def solve_logistic(
    space: RowSpace,
    y: Tensor,
    classes: int,
    inverse_l2: float = 1.0,
    iterations: int = 100,
    tolerance: float = 1e-12,
    budget: int = 2**31,
) -> tuple[Tensor, Tensor]:
    """The regressions of `fit_logistic` in a `row_space` basis, which every label set of the same rows can share.
    Returns weights and biases for uncentered inputs, in float64."""
    groups, rows, size = space.features.shape
    onehot = F.one_hot(y.to(space.features.device), classes).to(torch.float64)  # (n, C)
    penalty = space.features.new_full((size, 1), 1.0 / inverse_l2)
    penalty[-1] = 0.0
    chunk = max(1, budget // (8 * 2 * (classes * size) ** 2))
    theta = torch.cat([
        _newton_fit(space.features[start: start + chunk], onehot, penalty, iterations, tolerance)
        for start in range(0, groups, chunk)
    ])  # (G, n + 1, C)
    coefficients = torch.linalg.solve_triangular(space.factor.transpose(1, 2), theta[:, :-1], upper=True)  # (G, n, C)
    weight = space.centered.transpose(1, 2) @ coefficients  # (G, w, C)
    bias = theta[:, -1:] - space.mean @ weight  # (G, 1, C): the intercept of uncentered inputs
    return weight, bias


def fit_logistic(
    X: Tensor,
    y: Tensor,
    classes: int,
    inverse_l2: float = 1.0,
    iterations: int = 100,
    tolerance: float = 1e-12,
    budget: int = 2**31,
) -> tuple[Tensor, Tensor]:
    """L2-regularized multinomial logistic regressions of G independent probes, each solved by Newton's method.

    The objective of each probe is scikit-learn's: summed cross entropy plus ||W||^2 / (2 C), intercept unpenalized.
    The weights that minimize it lie in the span of the training rows (`row_space`), so the probe is a regression on
    n columns with the same penalty. Newton's method converges there in a few steps, in float64; probes are solved in
    chunks whose Hessians fit in `budget` bytes.
    """
    # X: (G, n, w); y: (n,) class indices
    weight, bias = solve_logistic(row_space(X), y, classes, inverse_l2, iterations, tolerance, budget)
    return weight.to(X.dtype), bias.to(X.dtype)


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
    shuffled: np.ndarray | None,
    iterations: int,
) -> dict[tuple[str, str], dict[str, Tensor]]:
    """Test MCC of every site layer, keyed by (inputs, label set) and site: each (L,).

    Inputs are "raw" and "standardized" (`INPUTS`); label sets are "true", and "shuffled" when given.
    """
    device = next(iter(features.values())).device
    train_rows, test_rows = torch.from_numpy(train).to(device), torch.from_numpy(test).to(device)
    y = torch.from_numpy(classes).to(device)
    label_sets = {"true": y[train_rows]}
    if shuffled is not None:
        label_sets["shuffled"] = torch.from_numpy(shuffled).to(device)
    test_y = y[test_rows]
    by_width: dict[int, list[str]] = {}
    for site, values in features.items():
        by_width.setdefault(values.shape[-1], []).append(site)
    scores: dict[tuple[str, str], dict[str, Tensor]] = {(inputs, name): {} for inputs in INPUTS for name in label_sets}
    for sites in by_width.values():
        stacked = torch.cat([features[site].transpose(0, 1) for site in sites])  # (G, m, w)
        layers = [features[site].shape[1] for site in sites]
        for inputs in INPUTS:
            X_train, X_test = stacked[:, train_rows], stacked[:, test_rows]  # (G, n, w), (G, t, w)
            if inputs == "standardized":
                X_train, X_test = standardize(X_train, X_test)
            space = row_space(X_train)
            for name, labels in label_sets.items():
                weight, bias = solve_logistic(space, labels, len(CLASSES), iterations=iterations)
                mcc = test_mcc(weight, bias, X_test.to(torch.float64), test_y, len(CLASSES))  # (G,)
                for site, values in zip(sites, mcc.split(layers), strict=True):
                    scores[(inputs, name)][site] = values
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
    iterations: int = 100,
    max_tokens: int = 131072,
    watch: Callable[[Iterable[tuple[int, PhosphoProtein]]], Iterable[tuple[int, PhosphoProtein]]] | None = None,
) -> pd.DataFrame:
    """Score `proteins` randomly chosen phosphoproteins, appending one parquet part per protein under `out`.

    Proteins already written are skipped, so an interrupted run resumes. Returns every row written so far. `watch`
    wraps the loop over proteins, as `foundry.pilot.Pilot.watch` does, so a pilot times whole proteins.
    """
    out.mkdir(parents=True, exist_ok=True)
    candidates = [protein for protein in phosphoproteins(data) if len(protein.sequence) <= max_length]
    rng = np.random.default_rng(seed)
    chosen = [candidates[index] for index in rng.choice(len(candidates), size=min(proteins, len(candidates)), replace=False)]
    layers = {site: layout.layers for site, layout in site_layouts(spec, model_dims(model)).items()}
    pending = [(position, protein) for position, protein in enumerate(chosen) if not (out / f"{position:04d}_{protein.id}.parquet").exists()]
    for position, protein in (watch(pending) if watch is not None else pending):
        part = out / f"{position:04d}_{protein.id}.parquet"
        sequences, classes = scanning_mutants(protein)
        protein_seed = seed + position
        train, test = stratified_split(classes, 0.3, protein_seed)
        shuffled = np.random.default_rng(protein_seed).permutation(classes[train])
        rows = []
        for plm, network in (("trained", model), ("untrained", untrained)):
            features = pooled_means(network, tokenizer, sequences, spec, device, max_tokens)  # site -> (m, L, w)
            scores = probe_protein(features, classes, train, test, shuffled if plm == "trained" else None, iterations)
            for (inputs, labels), by_site in scores.items():
                for site, values in by_site.items():
                    rows.extend(
                        {"protein": protein.id, "position": position, "plm": plm, "inputs": inputs, "labels": labels,
                         "site": site, "layer": layer, "mcc": float(value), "mutants": len(sequences),
                         "sites": len(protein.sites)}
                        for layer, value in zip(layers[site], values.cpu().tolist(), strict=True)
                    )
        temporary = part.with_suffix(".tmp")
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        temporary.replace(part)
    return pd.concat([pd.read_parquet(path) for path in sorted(out.glob("*.parquet"))], ignore_index=True)


def paired_tests(scores: pd.DataFrame) -> pd.DataFrame:
    """Per input treatment, site and layer: mean MCC by condition, and the Wilcoxon signed-rank test of true against
    shuffled labels, paired by protein, with its Bonferroni-corrected p-value over every site layer of that treatment."""
    from scipy.stats import wilcoxon

    table = scores.pivot_table(index=["inputs", "site", "layer", "protein"], columns=["plm", "labels"], values="mcc")
    rows = []
    for (inputs, site, layer), block in table.groupby(level=["inputs", "site", "layer"]):
        true, shuffled = block[("trained", "true")], block[("trained", "shuffled")]
        statistic, p_value = wilcoxon(true, shuffled) if not np.allclose(true, shuffled) else (np.nan, 1.0)
        untrained = block[("untrained", "true")]
        rows.append({
            "inputs": inputs, "site": site, "layer": layer, "proteins": len(block),
            "trained_true": true.mean(), "trained_true_std": true.std(),
            "trained_shuffled": shuffled.mean(), "trained_shuffled_std": shuffled.std(),
            "untrained_true": untrained.mean(), "untrained_true_std": untrained.std(),
            "statistic": statistic, "p_value": p_value,
        })
    tests = pd.DataFrame(rows)
    tests["p_bonferroni"] = (tests["p_value"] * tests.groupby("inputs")["p_value"].transform("size")).clip(upper=1.0)
    return tests
