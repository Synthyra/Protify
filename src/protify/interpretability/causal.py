"""From readable to used: lenses, block-write ablations and concept erasure on ESMC's masked-token prediction.

Probes say where a concept is linearly readable. These analyses ask where the model uses what is there, by reading
and changing its residual stream while it does what it was trained to do, predict masked residues:

- `lens`: at every depth of the stream (0 the token embedding, k + 1 the output of block k), the logit lens (the
  model's final norm and masked-LM head applied to the stream), a tuned lens (one affine map per depth before them,
  trained to match the model's prediction; Belrose et al. 2023), and a linear readout of each unmasked residue's own
  identity at every depth and at the final norm (depth n + 1);
- `ablate`: each block's attention write and feed-forward write in turn replaced by zero or by its mean over
  reference proteins, and what that costs the masked-token prediction;
- `erase`: a concept's linear subspace removed from the stream at one depth (LEACE, Belrose et al. 2023), against a
  random subspace of the same rank, and what that costs the masked-token prediction and the concept's linear
  readability at that depth, midway to the output and at the final norm.

Masking follows the objective: round(15%) of a protein's residues, at least one, drawn from the seed and the protein's
index in its split, become the mask token. Each analysis writes per-protein sums of its test proteins (masked residues,
correct top-1 predictions, negative log-likelihood, KL divergence from the unablated prediction), so that intervals
come from bootstrap draws of the proteins (`scores.protein_draws`), the same draws for every model, seed and condition;
erasure's readouts are scored on those draws as they are fitted.

Symbols: b sequences; l token columns (CLS, residues, EOS, padding); d model width; V vocabulary; n masked residues;
m unmasked residues; r residue rows; c concept columns; P test proteins; B bootstrap draws.
"""

from __future__ import annotations

import copy
import json
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from torch import Tensor, nn

from .capture import esmc_backbone
from .concepts import ConceptData
from .scores import ScoreAccumulator, draw_plan, protein_draws
from .solver import PENALTIES, STANDARDIZE_FLOOR, headline
from .token_batches import TokenBatch, token_batches


MASK_RATE = 0.15
LENS_TASK = "ssp_q3"  # its training proteins train the tuned lens, its test proteins (with their secondary structure) score it
LENS_KINDS = ("logit", "tuned")
STATS = ("masked", "correct", "nll", "kl")  # per-protein sums at masked residues
WRITES = ("attn", "ffn")
ABLATIONS = ("zero", "mean")
CONDITIONS = ("leace", "random")  # erasure and its control, beside the unerased `clean` stream


# -- masking and the residual stream ----------------------------------------------------------------------------------


def masked_columns(residues: int, protein: int, seed: int, rate: float = MASK_RATE) -> np.ndarray:
    """The token columns of a protein's masked residues (column 0 is CLS): round(rate * residues), at least one, drawn
    from the seed and the protein's index, so that every model and condition masks the same residues."""
    count = max(1, int(round(rate * residues)))
    rng = np.random.default_rng((seed, protein))
    return np.sort(rng.choice(residues, size=min(count, residues), replace=False)) + 1


@dataclass
class MaskedBatch:
    input_ids: Tensor  # (b, l), the masked residues replaced by the mask token
    attention_mask: Tensor  # (b, l)
    masked: Tensor  # (n,) flat positions (row * l + column) of the masked residues
    targets: Tensor  # (n,) their true tokens
    proteins: Tensor  # (n,) each masked residue's protein, by index in the split
    classes: Tensor | None  # (n,) the residue label at each masked residue, when labels are given
    kept: Tensor  # (m,) flat positions of the residues left unmasked
    kept_targets: Tensor  # (m,)
    kept_proteins: Tensor  # (m,)


def masked_batch(tokenizer: object, batch: TokenBatch, seed: int, device: torch.device, labels: Sequence[np.ndarray] | None = None,
                 rate: float = MASK_RATE) -> MaskedBatch:
    """Encode a batch and mask `rate` of each protein's residues (none at rate 0)."""
    encoded = tokenizer(list(batch.sequences), padding=True, return_tensors="pt")
    ids = encoded["input_ids"]  # (b, l)
    width = ids.shape[1]
    masked, kept, owners, kept_owners, classes = [], [], [], [], []
    for row, index in enumerate(batch.indices):
        residues = len(batch.sequences[row])
        chosen = masked_columns(residues, index, seed, rate) if rate > 0 else np.empty(0, dtype=np.int64)
        rest = np.setdiff1d(np.arange(1, residues + 1), chosen)
        masked.append(row * width + chosen)
        kept.append(row * width + rest)
        owners.append(np.full(len(chosen), index))
        kept_owners.append(np.full(len(rest), index))
        if labels is not None:
            label = np.asarray(labels[index], dtype=np.int64)
            assert label.shape == (residues,), f"protein {index} has {residues} residues but {label.shape} labels"
            classes.append(label[chosen - 1])
    masked_at = torch.from_numpy(np.concatenate(masked).astype(np.int64))  # (n,)
    kept_at = torch.from_numpy(np.concatenate(kept).astype(np.int64))  # (m,)
    flat = ids.reshape(-1)
    fed = flat.clone()
    fed[masked_at] = tokenizer.mask_token_id
    return MaskedBatch(
        input_ids=fed.view_as(ids).to(device, non_blocking=True),
        attention_mask=encoded["attention_mask"].to(device, non_blocking=True),
        masked=masked_at.to(device), targets=flat[masked_at].to(device), proteins=torch.from_numpy(np.concatenate(owners)).to(device),
        classes=torch.from_numpy(np.concatenate(classes)).to(device) if labels is not None else None,
        kept=kept_at.to(device), kept_targets=flat[kept_at].to(device), kept_proteins=torch.from_numpy(np.concatenate(kept_owners)).to(device),
    )


StreamReader = Callable[[int, Tensor], None]
StreamWriter = Callable[[int, Tensor], Tensor | None]


@contextmanager
def stream_hooks(model: nn.Module, read: StreamReader | None = None, write: StreamWriter | None = None) -> Iterator[int]:
    """Hooks on the residual stream at every depth for the duration of the block, yielding the number of blocks n:
    depth 0 is the token embedding (block 0's input), depth k + 1 block k's output and depth n + 1 the final norm.
    `write(depth, x)` sees the stream (b, l, d) first and may return a replacement, which the model then carries on
    from; `read(depth, x)` sees what the model carries on from."""
    backbone = esmc_backbone(model)
    blocks = backbone.transformer.blocks

    def hook_for(depth: int):
        def hook(module: nn.Module, inputs: tuple, output: object) -> object:
            x = output[0] if isinstance(output, tuple) else output
            if write is not None:
                replaced = write(depth, x)
                if replaced is not None:
                    x = replaced
                    output = (x, *output[1:]) if isinstance(output, tuple) else x
            if read is not None:
                read(depth, x)
            return output
        return hook

    modules = [backbone.embed, *blocks, backbone.transformer.norm]
    handles = [module.register_forward_hook(hook_for(depth)) for depth, module in enumerate(modules)]
    try:
        yield len(blocks)
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def write_hooks(model: nn.Module, block: int, write: str, replace: Callable[[Tensor], Tensor]) -> Iterator[None]:
    """Replace one block's attention write (`attn`, the output projection before the residual scaling) or feed-forward
    write (`ffn`) by `replace(output)` for the duration of the block."""
    module = getattr(esmc_backbone(model).transformer.blocks[block], write)

    def hook(module: nn.Module, inputs: tuple, output: object) -> object:
        if isinstance(output, tuple):
            return (replace(output[0]), *output[1:])
        return replace(output)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def lens_head(model: nn.Module) -> nn.Module:
    """The model's final layer norm and masked-LM head, copied in float32 and frozen: the logit lens reads any depth
    of the stream through them."""
    head = getattr(model, "sequence_head", None)
    if head is None:
        raise ValueError("The lens needs a masked-LM model, with its sequence head (get_base_model(..., masked_lm=True)).")
    norm = esmc_backbone(model).transformer.norm
    return nn.Sequential(copy.deepcopy(norm), copy.deepcopy(head)).float().eval().requires_grad_(False)


def token_stats(log_q: Tensor, targets: Tensor, log_p: Tensor) -> Tensor:
    """Per masked residue: 1, top-1 correct, negative log-likelihood of the true token, and KL(p || q) from the
    reference prediction p: (n, 4) float64."""
    # log_q, log_p: (n, V) log-probabilities; targets: (n,)
    correct = (log_q.argmax(dim=-1) == targets).to(torch.float64)
    nll = -log_q.gather(1, targets.unsqueeze(1)).squeeze(1).to(torch.float64)
    kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1).to(torch.float64)
    return torch.stack([torch.ones_like(nll), correct, nll, kl], dim=-1)


def add_by_protein(sums: Tensor, proteins: Tensor, values: Tensor) -> None:
    """Add each row of `values` (n, ...) to its protein's row of `sums` (P, ...)."""
    sums.index_add_(0, proteins, values.to(sums.dtype))


def unit_rows(x: Tensor) -> Tensor:
    """Each row scaled to unit L2 norm, as the paper's unit-norm token states."""
    return F.normalize(x.to(torch.float32), dim=-1)


# -- lenses -----------------------------------------------------------------------------------------------------------


class TunedLens(nn.Module):
    """One affine map per depth, h + A h + b, before the final norm and head; A and b start at zero, so an untrained
    tuned lens is the logit lens (Belrose et al. 2023)."""

    def __init__(self, depths: int, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(depths, width, width))
        self.bias = nn.Parameter(torch.zeros(depths, width))

    def forward(self, depth: int, h: Tensor) -> Tensor:
        # h: (n, d) float32
        return h + h @ self.weight[depth].T + self.bias[depth]


class IdentityReadout(nn.Module):
    """A softmax-linear readout of each residue's own token at every depth, on unit-norm rows standardized with
    moments fitted before training."""

    def __init__(self, depths: int, width: int, vocabulary: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(depths, width, vocabulary))
        self.bias = nn.Parameter(torch.zeros(depths, vocabulary))
        self.register_buffer("mean", torch.zeros(depths, width))
        self.register_buffer("scale", torch.ones(depths, width))

    def forward(self, depth: int, h: Tensor) -> Tensor:
        # h: (m, d) -> logits (m, V)
        return ((unit_rows(h) - self.mean[depth]) * self.scale[depth]) @ self.weight[depth] + self.bias[depth]


def _capture(model: nn.Module, batch: MaskedBatch, kept: Tensor | None) -> tuple[Tensor, dict[int, Tensor], dict[int, Tensor]]:
    """One forward pass: the model's logits at the masked residues (n, V), and every depth's stream rows there and at
    the `kept` positions."""
    at_masked: dict[int, Tensor] = {}
    at_kept: dict[int, Tensor] = {}

    def read(depth: int, x: Tensor) -> None:
        flat = x.reshape(-1, x.shape[-1])  # (b * l, d)
        at_masked[depth] = flat[batch.masked]
        if kept is not None:
            at_kept[depth] = flat[kept]

    with torch.no_grad(), stream_hooks(model, read=read):
        logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits  # (b, l, V)
    return logits.reshape(-1, logits.shape[-1])[batch.masked], at_masked, at_kept


def train_lenses(model: nn.Module, tokenizer: object, data: ConceptData, device: torch.device, seed: int, epochs: int = 4,
                 lr: float = 1e-3, max_tokens: int = 8192, moment_proteins: int = 256) -> tuple[TunedLens, IdentityReadout, list[dict[str, float]]]:
    """Train the tuned lens (KL from the model's prediction at masked residues) and the identity readout (cross
    entropy at as many unmasked residues as masked ones, drawn per batch) on the training proteins of `data`, every
    depth at once from one forward pass per batch; Adam with a cosine schedule. Returns both and the loss per epoch."""
    head = lens_head(model).to(device)
    blocks = len(esmc_backbone(model).transformer.blocks)
    width = esmc_backbone(model).embed.embedding_dim
    vocabulary = head[-1][-1].out_features
    lens = TunedLens(blocks + 1, width).to(device)
    readout = IdentityReadout(blocks + 2, width, vocabulary).to(device)
    sequences = data.splits["train"].sequences
    # the identity readout's standardization, from the unmasked residues of a few training proteins
    rng = np.random.default_rng(seed)
    sums = torch.zeros(blocks + 2, 2, width, dtype=torch.float64, device=device)
    rows = 0
    sample = sorted(rng.choice(len(sequences), size=min(moment_proteins, len(sequences)), replace=False).tolist())
    for token_batch in token_batches([sequences[index] for index in sample], max_tokens):
        token_batch = TokenBatch(tuple(sample[i] for i in token_batch.indices), token_batch.sequences)
        batch = masked_batch(tokenizer, token_batch, seed, device)
        _, _, at_kept = _capture(model, batch, batch.kept)
        for depth, h in at_kept.items():
            u = unit_rows(h).to(torch.float64)
            sums[depth, 0] += u.sum(dim=0)
            sums[depth, 1] += u.square().sum(dim=0)
        rows += len(batch.kept)
    mean = sums[:, 0] / rows
    readout.mean.copy_(mean.to(torch.float32))
    readout.scale.copy_((sums[:, 1] / rows - mean.square()).clamp_min(STANDARDIZE_FLOOR).rsqrt().to(torch.float32))
    batches_per_epoch = len(token_batches(sequences, max_tokens))
    optimizer = torch.optim.Adam([*lens.parameters(), readout.weight, readout.bias], lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * batches_per_epoch)
    history = []
    for epoch in range(epochs):
        totals = torch.zeros(2, dtype=torch.float64, device=device)  # read once an epoch, so steps never wait on the device
        count = 0
        for token_batch in token_batches(sequences, max_tokens, shuffle=True, seed=seed + epoch):
            batch = masked_batch(tokenizer, token_batch, seed + 1000 * (epoch + 1), device)
            draw = torch.randperm(len(batch.kept), generator=torch.Generator().manual_seed(seed + count))[: len(batch.masked)].to(device)
            logits, at_masked, at_kept = _capture(model, batch, batch.kept[draw])
            final = torch.log_softmax(logits.float(), dim=-1)  # (n, V)
            for depth in range(blocks + 1):
                log_q = torch.log_softmax(head(lens(depth, at_masked[depth].float())), dim=-1)
                loss = F.kl_div(log_q, final, log_target=True, reduction="batchmean")
                loss.backward()
                totals[0] += loss.detach() / (blocks + 1)
            targets = batch.kept_targets[draw]
            for depth in range(blocks + 2):
                loss = F.cross_entropy(readout(depth, at_kept[depth]), targets)
                loss.backward()
                totals[1] += loss.detach() / (blocks + 2)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            schedule.step()
            count += 1
        kl, identity = (totals / count).tolist()
        history.append({"epoch": epoch, "lens_kl": kl, "identity_loss": identity, "steps": count})
        print(f"lens epoch {epoch}: mean KL {kl:.4f}, identity loss {identity:.4f} over {count} steps", flush=True)
    return lens.eval(), readout.eval(), history


@torch.no_grad()
def score_lenses(model: nn.Module, tokenizer: object, data: ConceptData, lens: TunedLens, readout: IdentityReadout, device: torch.device,
                 seed: int, max_tokens: int = 8192) -> pd.DataFrame:
    """Every depth's logit and tuned lens at the masked residues of the test proteins, overall and by residue class
    (`STATS` per protein), and the identity readout at their unmasked residues: one row per protein, depth, lens and
    class (`all` or a label name)."""
    head = lens_head(model).to(device)
    blocks = len(esmc_backbone(model).transformer.blocks)
    split = data.splits["test"]
    classes = len(data.task.label_names)
    proteins = len(split.sequences)
    sums = torch.zeros(proteins, blocks + 2, len(LENS_KINDS), classes + 1, len(STATS), dtype=torch.float64, device=device)
    identity = torch.zeros(proteins, blocks + 2, 3, dtype=torch.float64, device=device)  # residues, correct, NLL
    for token_batch in token_batches(split.sequences, max_tokens):
        batch = masked_batch(tokenizer, token_batch, seed, device, labels=split.labels)
        logits, at_masked, at_kept = _capture(model, batch, batch.kept)
        final = torch.log_softmax(logits.float(), dim=-1)
        for depth in range(blocks + 2):
            for kind_index, kind in enumerate(LENS_KINDS):
                if depth == blocks + 1:
                    log_q = final  # the final norm: the model's own prediction
                else:
                    h = at_masked[depth].float()
                    log_q = torch.log_softmax(head(lens(depth, h) if kind == "tuned" else h), dim=-1)
                stats = token_stats(log_q, batch.targets, final)  # (n, 4)
                add_by_protein(sums[:, depth, kind_index, 0], batch.proteins, stats)
                for label in range(classes):
                    chosen = batch.classes == label
                    add_by_protein(sums[:, depth, kind_index, label + 1], batch.proteins[chosen], stats[chosen])
            log_r = torch.log_softmax(readout(depth, at_kept[depth]), dim=-1)  # (m, V)
            correct = (log_r.argmax(dim=-1) == batch.kept_targets).to(torch.float64)
            nll = -log_r.gather(1, batch.kept_targets.unsqueeze(1)).squeeze(1).to(torch.float64)
            add_by_protein(identity[:, depth], batch.kept_proteins, torch.stack([torch.ones_like(correct), correct, nll], dim=-1))
    names = ["all", *data.task.label_names]
    index = pd.MultiIndex.from_product([range(proteins), range(blocks + 2), LENS_KINDS, names], names=["protein", "depth", "lens", "class"])
    table = pd.DataFrame(sums.reshape(-1, len(STATS)).cpu().numpy(), index=index, columns=list(STATS)).reset_index()
    readouts = pd.DataFrame({"protein": np.repeat(np.arange(proteins), blocks + 2), "depth": np.tile(np.arange(blocks + 2), proteins),
                             "lens": "identity", "class": "all", "masked": identity[..., 0].reshape(-1).cpu().numpy(),
                             "correct": identity[..., 1].reshape(-1).cpu().numpy(), "nll": identity[..., 2].reshape(-1).cpu().numpy(), "kl": np.nan})
    return pd.concat([table, readouts], ignore_index=True).assign(blocks=blocks)


# -- block-write ablations ----------------------------------------------------------------------------------------------


@torch.no_grad()
def mean_writes(model: nn.Module, tokenizer: object, sequences: Sequence[str], device: torch.device, seed: int, max_tokens: int = 8192) -> Tensor:
    """Each block's attention and feed-forward write averaged over every attended token of `sequences`, masked as
    the test proteins are: (blocks, 2, d) float32."""
    blocks = esmc_backbone(model).transformer.blocks
    width = esmc_backbone(model).embed.embedding_dim
    sums = torch.zeros(len(blocks), len(WRITES), width, dtype=torch.float64, device=device)
    tokens = torch.zeros((), dtype=torch.float64, device=device)
    weights: list[Tensor] = []

    def hook_for(block: int, write: int):
        def hook(module: nn.Module, inputs: tuple, output: object) -> None:
            x = output[0] if isinstance(output, tuple) else output  # (b, l, d)
            sums[block, write] += (x.to(torch.float64) * weights[0].unsqueeze(-1)).sum(dim=(0, 1))
        return hook

    handles = [getattr(block, name).register_forward_hook(hook_for(index, position))
               for index, block in enumerate(blocks) for position, name in enumerate(WRITES)]
    try:
        for token_batch in token_batches(sequences, max_tokens):
            batch = masked_batch(tokenizer, token_batch, seed, device)
            weights[:] = [batch.attention_mask.to(torch.float64)]
            tokens += weights[0].sum()
            model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    return (sums / tokens).to(torch.float32)


@torch.no_grad()
def score_ablations(model: nn.Module, tokenizer: object, data: ConceptData, means: Tensor, device: torch.device, seed: int,
                    max_tokens: int = 8192) -> pd.DataFrame:
    """For every block, write and ablation, the cost to the masked-token prediction of the test proteins: per protein
    `STATS` against the unablated prediction, one row per protein, block, write and ablation, and the unablated
    model's own as block -1."""
    blocks = len(esmc_backbone(model).transformer.blocks)
    split = data.splits["test"]
    proteins = len(split.sequences)
    sums = torch.zeros(proteins, blocks, len(WRITES), len(ABLATIONS), len(STATS), dtype=torch.float64, device=device)
    clean_sums = torch.zeros(proteins, len(STATS), dtype=torch.float64, device=device)
    for token_batch in token_batches(split.sequences, max_tokens):
        batch = masked_batch(tokenizer, token_batch, seed, device)

        def predict() -> Tensor:
            logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
            return torch.log_softmax(logits.reshape(-1, logits.shape[-1])[batch.masked].float(), dim=-1)

        clean = predict()
        add_by_protein(clean_sums, batch.proteins, token_stats(clean, batch.targets, clean))
        for block in range(blocks):
            for write_index, write in enumerate(WRITES):
                for kind_index, kind in enumerate(ABLATIONS):
                    if kind == "zero":
                        replace = torch.zeros_like
                    else:
                        vector = means[block, write_index]
                        replace = lambda x, vector=vector: vector.to(x.dtype).expand_as(x)  # noqa: E731
                    with write_hooks(model, block, write, replace):
                        ablated = predict()
                    add_by_protein(sums[:, block, write_index, kind_index], batch.proteins, token_stats(ablated, batch.targets, clean))
    index = pd.MultiIndex.from_product([range(proteins), range(blocks), WRITES, ABLATIONS], names=["protein", "block", "write", "ablation"])
    table = pd.DataFrame(sums.reshape(-1, len(STATS)).cpu().numpy(), index=index, columns=list(STATS)).reset_index()
    clean_table = pd.DataFrame(clean_sums.cpu().numpy(), columns=list(STATS)).assign(protein=np.arange(proteins), block=-1, write="none", ablation="none")
    return pd.concat([clean_table, table], ignore_index=True).assign(blocks=blocks)


# -- concept erasure ----------------------------------------------------------------------------------------------------


@dataclass
class Moments:
    """Sums over residue rows of x, x x^T, z and x z^T, in float64."""

    count: float
    x: Tensor  # (d,)
    xx: Tensor  # (d, d)
    z: Tensor  # (c,)
    xz: Tensor  # (d, c)

    def covariances(self) -> tuple[Tensor, Tensor, Tensor]:
        mean_x, mean_z = self.x / self.count, self.z / self.count
        return mean_x, self.xx / self.count - torch.outer(mean_x, mean_x), self.xz / self.count - torch.outer(mean_x, mean_z)


@dataclass
class Eraser:
    """x -> x - (x - mean) P^T, an oblique projection removing `rank` directions."""

    mean: Tensor  # (d,)
    projection: Tensor  # (d, d)
    rank: int
    singular: list[float]  # the whitened cross-covariance's singular values, largest first

    def __call__(self, x: Tensor) -> Tensor:
        # x: (..., d)
        centred = x.to(torch.float32) - self.mean
        return (x.to(torch.float32) - centred @ self.projection.T).to(x.dtype)


def _whitening(sigma: Tensor, tolerance: float = 1e-8) -> tuple[Tensor, Tensor, Tensor]:
    """Sigma^(-1/2), Sigma^(1/2) on the span of the eigenvalues above `tolerance` times the largest, and that span's
    basis."""
    values, vectors = torch.linalg.eigh(sigma)
    keep = values > tolerance * values.max()
    basis = vectors[:, keep]  # (d, k)
    return basis * values[keep].rsqrt() @ basis.T, basis * values[keep].sqrt() @ basis.T, basis


def leace(moments: Moments, max_rank: int | None = None, tolerance: float = 1e-8) -> Eraser:
    """LEACE (Belrose et al. 2023): with W the whitening Sigma_xx^(-1/2), remove the column space of W Sigma_xz in
    whitened coordinates, P = W^+ U U^T W; afterwards no linear function of x covaries with z on the fitted rows.
    `max_rank` keeps the top singular directions only, for concepts with many labels."""
    mean, sigma, cross = moments.covariances()
    whiten, unwhiten, _ = _whitening(sigma, tolerance)
    u, s, _ = torch.linalg.svd(whiten @ cross, full_matrices=False)
    rank = int((s > tolerance * s.max()).sum()) if len(s) and s.max() > 0 else 0
    if max_rank is not None:
        rank = min(rank, max_rank)
    U = u[:, :rank]  # (d, rank)
    projection = unwhiten @ U @ U.T @ whiten
    return Eraser(mean.to(torch.float32), projection.to(torch.float32), rank, s.tolist())


def random_eraser(moments: Moments, rank: int, seed: int, tolerance: float = 1e-8) -> Eraser:
    """The control: the same oblique projection of `rank` random orthonormal directions of the whitened span."""
    mean, sigma, _ = moments.covariances()
    whiten, unwhiten, basis = _whitening(sigma, tolerance)
    draw = torch.randn(basis.shape[1], rank, generator=torch.Generator().manual_seed(seed), dtype=torch.float64).to(basis.device)
    U, _ = torch.linalg.qr(basis @ draw)  # (d, rank)
    projection = unwhiten @ U @ U.T @ whiten
    return Eraser(mean.to(torch.float32), projection.to(torch.float32), rank, [])


def protein_concepts(data: ConceptData, split: str, rows: Sequence[int]) -> Tensor:
    """Each protein's concept vector: one-hot classes, 0/1 labels, or standardized values: (p, c) float64."""
    targets = data.protein_targets(split, rows)
    if data.task.kind == "multiclass":
        targets = F.one_hot(targets, data.task.num_labels)
    return targets.to(torch.float64).reshape(len(rows), -1)


def residue_rows(batch: MaskedBatch) -> tuple[Tensor, Tensor]:
    """A batch's residue positions in the flat grid and their proteins (no residue masked at rate 0)."""
    return batch.kept, batch.kept_proteins


def concept_moments(model: nn.Module, tokenizer: object, data: ConceptData, rows: Sequence[int], depths: Sequence[int],
                    device: torch.device, max_tokens: int = 8192) -> dict[int, Moments]:
    """`Moments` of the stream's residue rows at each depth against the concept: a residue task's own label, or a
    protein task's concept vector given to every residue of the protein; unmasked training proteins `rows`."""
    split = data.splits["train"]
    width = esmc_backbone(model).embed.embedding_dim
    residue = data.task.level == "residue"
    columns = data.task.num_labels if residue or data.task.kind == "multiclass" else protein_concepts(data, "train", rows[:1]).shape[1]
    sums = {depth: Moments(0.0, torch.zeros(width, dtype=torch.float64, device=device), torch.zeros(width, width, dtype=torch.float64, device=device),
                           torch.zeros(columns, dtype=torch.float64, device=device), torch.zeros(width, columns, dtype=torch.float64, device=device))
            for depth in depths}
    sequences = [split.sequences[row] for row in rows]
    concepts = protein_concepts(data, "train", rows).to(device) if not residue else None
    for token_batch in token_batches(sequences, max_tokens):
        batch = masked_batch(tokenizer, token_batch, 0, device, rate=0.0)
        positions, owners = residue_rows(batch)
        if residue:
            labels = torch.from_numpy(np.concatenate([np.asarray(split.labels[rows[index]], dtype=np.int64) for index in token_batch.indices]))
            z = F.one_hot(labels.to(device), columns).to(torch.float64)  # (r, c), in the order of the residue rows
        else:
            z = concepts[owners]  # (r, c)
        captured: dict[int, Tensor] = {}

        def read(depth: int, x: Tensor) -> None:
            if depth in sums:
                captured[depth] = x.reshape(-1, x.shape[-1])[positions].to(torch.float64)

        with torch.no_grad(), stream_hooks(model, read=read):
            model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        for depth, x in captured.items():
            moments = sums[depth]
            moments.count += x.shape[0]
            moments.x += x.sum(dim=0)
            moments.xx += x.T @ x
            moments.z += z.sum(dim=0)
            moments.xz += x.T @ z
    return sums


@dataclass
class Readout:
    """Rows a readout fits on: pooled proteins (`proteins` is their index) or residue rows (`proteins` their owner)."""

    X: Tensor  # (n, d) float32, unit-norm rows (residues) or the mean of unit-norm tokens (proteins)
    targets: Tensor  # (n,) classes or (n, C) values
    proteins: Tensor  # (n,)


@torch.no_grad()
def readout_rows(model: nn.Module, tokenizer: object, data: ConceptData, split: str, rows: Sequence[int], depths: Sequence[int],
                 device: torch.device, writer: StreamWriter | None = None, residue_cap: int | None = None, seed: int = 0,
                 max_tokens: int = 8192) -> dict[int, Readout]:
    """The rows a linear readout of the concept reads at each depth, from unmasked forward passes of the proteins
    `rows` of `split` through `writer`: a protein's mean over its attended tokens of the unit-norm states (protein
    tasks), or the unit-norm residue rows, at most `residue_cap` of them drawn from `seed` (residue tasks)."""
    source = data.splits[split]
    residue = data.task.level == "residue"
    sequences = [source.sequences[row] for row in rows]
    parts: dict[int, list[Tensor]] = {depth: [] for depth in depths}
    targets, owners = [], []
    for token_batch in token_batches(sequences, max_tokens):
        batch = masked_batch(tokenizer, token_batch, 0, device, rate=0.0)
        positions, proteins = residue_rows(batch)
        weights = batch.attention_mask.to(torch.float32)  # (b, l)

        def read(depth: int, x: Tensor) -> None:
            if depth not in parts:
                return
            if residue:
                parts[depth].append(unit_rows(x.reshape(-1, x.shape[-1])[positions]).cpu())
            else:
                pooled = (unit_rows(x) * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True)  # (b, d)
                parts[depth].append(pooled.cpu())

        with stream_hooks(model, read=read, write=writer):
            model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        if residue:
            targets.append(torch.from_numpy(np.concatenate([np.asarray(source.labels[rows[index]], dtype=np.int64) for index in token_batch.indices])))
            owners.append(proteins.cpu())
        else:
            targets.append(data.protein_targets(split, [rows[index] for index in token_batch.indices]))
            owners.append(torch.tensor(token_batch.indices))
    target = torch.cat(targets)
    owner = torch.cat(owners)
    chosen = slice(None)
    if residue and residue_cap is not None and len(owner) > residue_cap:
        chosen = torch.from_numpy(np.sort(np.random.default_rng(seed).choice(len(owner), size=residue_cap, replace=False)))
    return {depth: Readout(torch.cat(part)[chosen], target[chosen], owner[chosen]) for depth, part in parts.items()}


def ridge_readout(data: ConceptData, readouts: dict[str, Readout], plan_counts: Tensor, device: torch.device) -> tuple[Tensor, dict[str, float]]:
    """A converged linear readout: ridge regression on the training rows (standardized on them) of one-hot classes,
    0/1 labels or standardized values, the penalty with the best validation score in the task's metric along the
    solver's grid (times the rows), scored on the test rows: the whole split and every bootstrap draw of its
    proteins, (B + 1,), and the fit's record."""
    task = data.task
    train = readouts["train"]
    mean = train.X.mean(dim=0, keepdim=True)
    scale = train.X.var(dim=0, unbiased=False, keepdim=True).clamp_min(STANDARDIZE_FLOOR).rsqrt()
    X = {split: ((readout.X - mean) * scale).to(device) for split, readout in readouts.items()}  # float32
    if task.kind == "multiclass":
        Y = F.one_hot(train.targets, task.num_labels).to(device, torch.float64)
    else:
        Y = train.targets.to(device, torch.float64).reshape(len(train.targets), -1)
    centre = Y.mean(dim=0)
    rows = X["train"].to(torch.float64)
    values, vectors = torch.linalg.eigh(rows.T @ rows)
    cross = vectors.T @ (rows.T @ (Y - centre))
    del rows
    offset = 0.5 if task.kind in ("multilabel", "binary") else 0.0  # 0/1 targets: a row's label is on above one half

    def outputs(split: str, weight: Tensor) -> Tensor:
        return X[split] @ weight.to(torch.float32) + (centre - offset).to(torch.float32)

    best = None
    for penalty in PENALTIES:
        weight = vectors @ (cross / (values.clamp_min(0).unsqueeze(1) + len(Y) * penalty))
        score = headline(task.kind, task.outputs, outputs("validation", weight), readouts["validation"].targets.to(device))
        if best is None or score > best[0]:
            best = (score, penalty, weight)
    score, penalty, weight = best
    test = readouts["test"]
    accumulator = ScoreAccumulator(task.kind, 1, task.outputs, draw_plan(plan_counts.to(device)), per_label=False)
    accumulator.add(outputs("test", weight).unsqueeze(1), test.targets.to(device), test.proteins.to(device))
    return accumulator.scores()[:, 0].cpu(), {"penalty": penalty, "validation": score, "train_rows": len(Y)}


def readout_depths(depth: int, blocks: int) -> list[int]:
    """Where an erasure at `depth` is read: there, midway to the final norm, and at the final norm (depth n + 1)."""
    return sorted({depth, (depth + blocks + 1 + 1) // 2, blocks + 1})


def erase_concept(model: nn.Module, tokenizer: object, data: ConceptData, depths: Sequence[int], device: torch.device, seed: int,
                  fit_proteins: int = 2000, validation_proteins: int = 500, test_proteins: int = 1000, residue_rows_cap: int = 100_000,
                  max_rank: int | None = None, max_tokens: int = 8192) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Erase the concept at each of `depths` (LEACE, and the random control of the same rank), then read it: the
    masked-token cost on the test proteins (`STATS` per protein against the unerased prediction) and the concept's
    linear readability where `readout_depths` say, against the unerased stream there. The training and validation
    proteins are drawn from the seed; the test proteins, whose draws every seed and model share, are the first
    `test_proteins` of a fixed permutation of the test split."""
    blocks = len(esmc_backbone(model).transformer.blocks)
    rng = np.random.default_rng(seed)

    def subset(split: str, count: int, generator: np.random.Generator) -> list[int]:
        size = len(data.splits[split].sequences)
        return sorted(generator.permutation(size)[: min(count, size)].tolist())

    rows = {"train": subset("train", fit_proteins, rng), "validation": subset("validation", validation_proteins, rng),
            "test": subset("test", test_proteins, np.random.default_rng(0))}
    counts = protein_draws(len(rows["test"]))  # (B, P)
    caps = {"train": residue_rows_cap, "validation": residue_rows_cap // 3, "test": None}
    started = time.time()
    moments = concept_moments(model, tokenizer, data, rows["train"], depths, device, max_tokens)
    erasers = {}
    for depth in depths:
        fitted = leace(moments[depth], max_rank)
        erasers[(depth, "leace")] = fitted
        erasers[(depth, "random")] = random_eraser(moments[depth], fitted.rank, seed * 1000 + depth)
    record: dict[str, object] = {"rows": {split: len(chosen) for split, chosen in rows.items()}, "erasers": {}, "readouts": [],
                                 "moments_seconds": time.time() - started}
    for (depth, condition), eraser in erasers.items():
        record["erasers"][f"{depth}/{condition}"] = {"rank": eraser.rank, "singular": eraser.singular[:64]}

    readout_rows_table = []
    wanted = sorted({read for depth in depths for read in readout_depths(depth, blocks)})
    clean = {split: readout_rows(model, tokenizer, data, split, chosen, wanted, device, None, caps[split], seed, max_tokens)
             for split, chosen in rows.items()}
    for read in wanted:
        draws, fit = ridge_readout(data, {split: clean[split][read] for split in rows}, counts, device)
        readout_rows_table.append({"depth": -1, "condition": "clean", "read": read, "draws": draws.numpy()})
        record["readouts"].append({"depth": -1, "condition": "clean", "read": read, **fit})
    mlm_parts = []
    test_split = data.splits["test"]
    test_sequences = [test_split.sequences[row] for row in rows["test"]]
    sums = torch.zeros(len(rows["test"]), len(depths), len(CONDITIONS), len(STATS), dtype=torch.float64, device=device)
    for token_batch in token_batches(test_sequences, max_tokens):
        batch = masked_batch(tokenizer, token_batch, seed, device)
        with torch.no_grad():
            logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
        reference = torch.log_softmax(logits.reshape(-1, logits.shape[-1])[batch.masked].float(), dim=-1)
        for depth_index, depth in enumerate(depths):
            for condition_index, condition in enumerate(CONDITIONS):
                eraser = erasers[(depth, condition)]
                writer = (lambda at, x, depth=depth, eraser=eraser: eraser(x) if at == depth else None)
                with torch.no_grad(), stream_hooks(model, write=writer):
                    logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
                erased = torch.log_softmax(logits.reshape(-1, logits.shape[-1])[batch.masked].float(), dim=-1)
                add_by_protein(sums[:, depth_index, condition_index], batch.proteins, token_stats(erased, batch.targets, reference))
    for depth_index, depth in enumerate(depths):
        for condition_index, condition in enumerate(CONDITIONS):
            eraser = erasers[(depth, condition)]
            writer = (lambda at, x, depth=depth, eraser=eraser: eraser(x) if at == depth else None)
            reads = readout_depths(depth, blocks)
            erased = {split: readout_rows(model, tokenizer, data, split, chosen, reads, device, writer, caps[split], seed, max_tokens)
                      for split, chosen in rows.items()}
            for read in reads:
                draws, fit = ridge_readout(data, {split: erased[split][read] for split in rows}, counts, device)
                readout_rows_table.append({"depth": depth, "condition": condition, "read": read, "draws": draws.numpy()})
                record["readouts"].append({"depth": depth, "condition": condition, "read": read, **fit})
            mlm_parts.append(pd.DataFrame(sums[:, depth_index, condition_index].cpu().numpy(), columns=list(STATS))
                             .assign(protein=np.arange(len(rows["test"])), depth=depth, condition=condition))
            print(f"{data.task.name}: depth {depth} {condition} (rank {eraser.rank}) done at {time.time() - started:.0f} s", flush=True)
    readouts = pd.DataFrame([{**{key: value for key, value in row.items() if key != "draws"}, "draw": draw, "value": float(value)}
                             for row in readout_rows_table for draw, value in enumerate(row["draws"])])
    record["seconds"] = time.time() - started
    return pd.concat(mlm_parts, ignore_index=True).assign(blocks=blocks), readouts.assign(blocks=blocks), record


def write_json(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")


def relative_depths(blocks: int, shares: Sequence[float]) -> list[int]:
    """The stream depths nearest each relative depth share (depth / (blocks + 1)), so models of different depth are
    erased at matching places."""
    return sorted({int(min(blocks, max(0, math.floor(share * (blocks + 1) + 0.5)))) for share in shares})
