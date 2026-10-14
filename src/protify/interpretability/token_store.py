"""Token-wise taps written to disk: every row of every chosen site and layer, for later reading by sequence.

The token states of a whole model are large (ESMC-300 with every site is about 1.2 MB per token in bfloat16), so
this store is for chosen sites, layers or sequence sets; residue probes over everything stream from the model
instead. Each site is one memory-mapped bfloat16 file of shape (L_s, T, w): layer-major, then the T token rows of
all sequences back to back (CLS, residues, EOS for each), so a batch writes one contiguous block per layer.

Symbols: b sequences; l token columns; w site width; L_s layers of a site; T stored token rows; n sequences.
"""

from __future__ import annotations

import json
import os
import numpy as np
import torch

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from torch import Tensor, nn
from tqdm.auto import tqdm

from .capture import model_dims, tap_hooks
from .pooled_store import StoreManifest, encode_batch
from .sites import TapSpec, site_layouts
from .token_batches import SPECIAL_TOKENS, token_batches


TOKEN_STORE_FORMAT = "protify_token_taps_v1"


class TokenTapStore:
    """Read access to a token tap store."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["format"] != TOKEN_STORE_FORMAT:
            raise ValueError(f"{self.root} holds format {manifest['format']!r}, not {TOKEN_STORE_FORMAT!r}.")
        self.manifest = StoreManifest(**manifest)
        self.sequences = (self.root / "sequences.txt").read_text(encoding="utf-8").split("\n")
        self.offsets = np.load(self.root / "offsets.npy")  # (n + 1,)
        self._row_of = {sequence: row for row, sequence in enumerate(self.sequences)}
        self._arrays: dict[str, np.memmap] = {}

    def array(self, site: str) -> np.memmap:
        if site not in self._arrays:
            info = self.manifest.sites[site]
            shape = (len(info["layers"]), int(self.offsets[-1]), int(info["width"]))
            self._arrays[site] = np.memmap(self.root / f"{site}.bin", dtype=np.uint16, mode="r", shape=shape)
        return self._arrays[site]

    def read(self, sequence: str, site: str) -> Tensor:
        """Every layer's token rows of one sequence: (L_s, r + 2, w) bfloat16, CLS first and EOS last."""
        row = self._row_of[sequence]
        start, stop = int(self.offsets[row]), int(self.offsets[row + 1])
        block = np.ascontiguousarray(self.array(site)[:, start:stop])  # (L_s, r + 2, w) as uint16 bits
        return torch.from_numpy(block).view(torch.bfloat16)


@torch.inference_mode()
def embed_tokens(
    model: nn.Module,
    tokenizer: object,
    sequences: Sequence[str],
    spec: TapSpec,
    root: str | os.PathLike[str],
    model_name: str,
    variant: str = "trained",
    max_tokens: int = 65536,
    device: torch.device | str = "cuda",
) -> None:
    """Write every token row of the spec's sites and layers for `sequences`, longest first."""
    dims = model_dims(model)
    layouts = site_layouts(spec, dims)
    ordered = sorted(dict.fromkeys(sequences), key=len, reverse=True)
    lengths = np.asarray([len(sequence) + SPECIAL_TOKENS for sequence in ordered], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])  # (n + 1,)
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: np.memmap(target / f"{name}.bin", dtype=np.uint16, mode="w+", shape=(layout.num_layers, int(offsets[-1]), layout.width))
        for name, layout in layouts.items()
    }
    slots = {name: {layer: slot for slot, layer in enumerate(layout.layers)} for name, layout in layouts.items()}
    row_start = 0
    flat_index: Tensor | None = None
    span = (0, 0)

    def write(site: str, layer: int, X: Tensor) -> None:
        # X: (b, l, w) -> the attended rows of every sequence, back to back: (t, w)
        rows = X.reshape(-1, X.shape[-1]).index_select(0, flat_index)  # (t, w)
        bits = rows.to(torch.bfloat16).cpu().contiguous().view(torch.uint16).numpy()
        arrays[site][slots[site][layer], span[0]: span[1]] = bits

    with tap_hooks(model, spec, write):
        for batch in tqdm(token_batches(ordered, max_tokens=max_tokens), desc="Writing token taps", mininterval=10):
            encoded = encode_batch(tokenizer, batch.sequences, device)
            width = encoded["input_ids"].shape[1]
            kept = [row * width + np.arange(len(sequence) + SPECIAL_TOKENS) for row, sequence in enumerate(batch.sequences)]
            flat_index = torch.from_numpy(np.concatenate(kept)).to(device)
            count = len(batch.sequences)
            span = (int(offsets[row_start]), int(offsets[row_start + count]))
            model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
            row_start += count
    for array in arrays.values():
        array.flush()
    manifest = StoreManifest(
        format=TOKEN_STORE_FORMAT,
        model=model_name,
        variant=variant,
        normalized=False,
        dims=asdict(dims),
        sites={name: {"layers": list(layout.layers), "width": layout.width, "columns": layout.width} for name, layout in layouts.items()},
        rows=len(ordered),
    )
    np.save(target / "offsets.npy", offsets)
    (target / "sequences.txt").write_text("\n".join(ordered), encoding="utf-8", newline="\n")
    (target / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8", newline="\n")
