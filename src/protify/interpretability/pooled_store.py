"""Mean and variance of every tapped state over each sequence's tokens, written in one forward pass.

A store holds one memory-mapped bfloat16 file per site, of shape (n, L_s, 2w): row i is sequence i, axis 1 the
site's layers, and the last axis the mean over the sequence's tokens followed by their population variance. The
tokens are every attended row, CLS and EOS included, as Protify's `mean` and `var` poolings count them. A protein
probe reads whole rows, all layers of a site at once, so a site's rows are contiguous.

Symbols: b sequences of a batch; l token columns; w site width; L_s layers of a site; n stored sequences.
"""

from __future__ import annotations

import json
import os
import numpy as np
import torch
import torch.nn.functional as F

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from torch import Tensor, nn
from tqdm.auto import tqdm

from .capture import model_dims, tap_hooks
from .sites import ModelDims, SiteLayout, TapSpec, site_layouts
from .token_batches import TokenBatch, token_batches


STORE_FORMAT = "protify_pooled_taps_v1"


def masked_mean_var(X: Tensor, mask: Tensor) -> Tensor:
    """Mean then population variance over the rows the mask keeps, in float32."""
    # X: (b, l, w); mask: (b, l), 1 on CLS, residues and EOS
    weights = mask.to(torch.float32).unsqueeze(-1)  # (b, l, 1)
    counts = weights.sum(dim=1).clamp_min(1.0)  # (b, 1)
    values = X.to(torch.float32)  # (b, l, w)
    mean = (values * weights).sum(dim=1) / counts  # (b, w)
    centered = (values - mean.unsqueeze(1)) * weights  # (b, l, w)
    variance = centered.square().sum(dim=1) / counts  # (b, w)
    return torch.cat([mean, variance], dim=-1)  # (b, 2 * w)


@dataclass(frozen=True)
class StoreManifest:
    format: str
    model: str
    variant: str  # names the weights and the input transform, such as "trained" or "untrained_scrambled"
    normalized: bool  # tokens L2-normalized along the width before pooling
    dims: dict[str, int]
    sites: dict[str, dict[str, object]]
    rows: int
    dtype: str = "bfloat16"
    seed: int | None = None  # the seed of the input transform, such as the scramble's


class PooledTapStore:
    """Read access to a pooled tap store."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["format"] != STORE_FORMAT:
            raise ValueError(f"{self.root} holds format {manifest['format']!r}, not {STORE_FORMAT!r}.")
        self.manifest = StoreManifest(**manifest)
        self.sequences = (self.root / "sequences.txt").read_text(encoding="utf-8").split("\n")
        assert len(self.sequences) == self.manifest.rows, "the sequence list and the manifest disagree"
        self._row_of = {sequence: row for row, sequence in enumerate(self.sequences)}
        self._arrays: dict[str, np.memmap] = {}

    @property
    def site_names(self) -> list[str]:
        return list(self.manifest.sites)

    def layers(self, site: str) -> list[int]:
        return list(self.manifest.sites[site]["layers"])

    def rows_for(self, sequences: Sequence[str]) -> np.ndarray:
        missing = [sequence for sequence in sequences if sequence not in self._row_of]
        if missing:
            raise KeyError(f"{len(missing)} sequences are not in {self.root}, the first of length {len(missing[0])}.")
        return np.fromiter((self._row_of[sequence] for sequence in sequences), dtype=np.int64, count=len(sequences))

    def array(self, site: str) -> np.memmap:
        if site not in self._arrays:
            info = self.manifest.sites[site]
            shape = (self.manifest.rows, len(info["layers"]), int(info["columns"]))
            self._arrays[site] = np.memmap(self.root / f"{site}.bin", dtype=np.uint16, mode="r", shape=shape)
        return self._arrays[site]

    def load(self, site: str, rows: np.ndarray, device: torch.device | str = "cpu") -> Tensor:
        """The pooled rows of one site as bfloat16."""
        block = np.ascontiguousarray(self.array(site)[np.sort(rows)])  # (n, L_s, 2 * w) as uint16 bits
        order = np.argsort(np.argsort(rows))  # back to the requested order
        values = torch.from_numpy(block).view(torch.bfloat16)  # (n, L_s, 2 * w)
        return values[torch.from_numpy(order)].to(device)  # (n, L_s, 2 * w)


class PooledStoreWriter:
    """Preallocates one file per site and fills it batch by batch."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        sequences: Sequence[str],
        layouts: dict[str, SiteLayout],
        dims: ModelDims,
        model_name: str,
        variant: str,
        normalized: bool,
        seed: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.layouts = layouts
        self.layer_slot = {name: {layer: slot for slot, layer in enumerate(layout.layers)} for name, layout in layouts.items()}
        self.arrays = {
            name: np.memmap(
                self.root / f"{name}.bin", dtype=np.uint16, mode="w+",
                shape=(len(sequences), layout.num_layers, 2 * layout.width),
            )
            for name, layout in layouts.items()
        }
        self.manifest = StoreManifest(
            format=STORE_FORMAT,
            model=model_name,
            variant=variant,
            normalized=normalized,
            dims=asdict(dims),
            sites={
                name: {"layers": list(layout.layers), "width": layout.width, "columns": 2 * layout.width}
                for name, layout in layouts.items()
            },
            rows=len(sequences),
            seed=seed,
        )
        (self.root / "sequences.txt").write_text("\n".join(sequences), encoding="utf-8", newline="\n")

    def write(self, site: str, row_start: int, pooled: Tensor) -> None:
        # pooled: (b, L_s, 2 * w) bfloat16 on any device
        bits = pooled.detach().to("cpu", torch.bfloat16).contiguous().view(torch.uint16).numpy()  # (b, L_s, 2 * w)
        self.arrays[site][row_start: row_start + bits.shape[0]] = bits

    def close(self, complete: bool = True) -> None:
        """Flush every file; only a complete store gets its manifest, which is what marks a store readable."""
        for array in self.arrays.values():
            array.flush()
        if not complete:
            return
        temporary = self.root / "manifest.json.tmp"
        temporary.write_text(json.dumps(asdict(self.manifest), indent=2), encoding="utf-8", newline="\n")
        temporary.replace(self.root / "manifest.json")


@torch.inference_mode()
def embed_pooled(
    model: nn.Module,
    tokenizer: object,
    sequences: Sequence[str],
    spec: TapSpec,
    roots: dict[bool, str | os.PathLike[str]],
    model_name: str,
    variant: str,
    max_tokens: int = 65536,
    device: torch.device | str = "cuda",
    progress: bool = True,
    transform: Callable[[str], str] | None = None,
    seed: int | None = None,
    watch: Callable[[Iterable[TokenBatch]], Iterable[TokenBatch]] | None = None,
) -> int:
    """Write pooled stores of `sequences` in one pass: raw states at `roots[False]`, L2-normalized at `roots[True]`.

    Sequences are embedded longest first in batches of at most `max_tokens` padded tokens, and stored in that order.
    A `transform` (a length-preserving change such as a scramble) changes what the model reads; rows stay keyed by
    the original sequence, and the manifest records the transform's `seed` so its targets can be recomputed. `watch`
    wraps the batches, as `foundry.pilot.Pilot.watch` does; a store it ends early gets no manifest. Returns the rows
    written.
    """
    dims = model_dims(model)
    layouts = site_layouts(spec, dims)
    ordered = sorted(dict.fromkeys(sequences), key=len, reverse=True)
    writers = {
        normalized: PooledStoreWriter(root, ordered, layouts, dims, model_name, variant, normalized, seed)
        for normalized, root in roots.items()
    }
    batches = token_batches(ordered, max_tokens=max_tokens, shuffle=False)
    pending: dict[bool, dict[str, Tensor]] = {}
    row_start = 0
    mask: Tensor | None = None

    def pool(site: str, layer: int, X: Tensor) -> None:
        # X: (b, l, w)
        slot = writers[next(iter(writers))].layer_slot[site][layer]
        for normalized in writers:
            values = F.normalize(X.to(torch.float32), dim=-1) if normalized else X  # (b, l, w)
            pending[normalized][site][:, slot] = masked_mean_var(values, mask).to(torch.bfloat16)  # (b, 2 * w)

    bar = tqdm(batches, desc=f"Pooling taps ({variant})", disable=not progress, mininterval=10)
    with tap_hooks(model, spec, pool):
        for batch in (watch(bar) if watch is not None else bar):
            fed = [transform(sequence) for sequence in batch.sequences] if transform is not None else batch.sequences
            encoded = encode_batch(tokenizer, fed, device)
            mask = encoded["attention_mask"]  # (b, l)
            size = len(batch.sequences)
            for normalized in writers:
                pending[normalized] = {
                    name: torch.empty(size, layout.num_layers, 2 * layout.width, dtype=torch.bfloat16, device=device)
                    for name, layout in layouts.items()
                }
            model(input_ids=encoded["input_ids"], attention_mask=mask)
            for normalized, writer in writers.items():
                for name, pooled in pending[normalized].items():
                    writer.write(name, row_start, pooled)
            row_start += size
    for writer in writers.values():
        writer.close(complete=row_start == len(ordered))
    return row_start


def encode_batch(tokenizer: object, sequences: Sequence[str], device: torch.device | str) -> dict[str, Tensor]:
    """Token ids with CLS and EOS, padded to the batch's longest sequence."""
    encoded = tokenizer(list(sequences), padding=True, return_tensors="pt")
    return {
        "input_ids": encoded["input_ids"].to(device, non_blocking=True),  # (b, l)
        "attention_mask": encoded["attention_mask"].to(device, non_blocking=True),  # (b, l)
    }
