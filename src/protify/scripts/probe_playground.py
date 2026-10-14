"""Train a binary probe on frozen pooled ESMC embeddings, with the model written out here so it can be changed.

Edit the settings and the model below, then run from `~/Dropbox/workspace`, where `ws.cmd` is (`ws run` starts in
`projects/tooling/property_lowcode`, so the script path is relative to it):

    cd ~/Dropbox/workspace
    .\\ws.cmd run property_lowcode -- python src/protify/scripts/probe_playground.py

The run loads the Hub splits, carves calibration from valid when the dataset has none, and caps each split; fetches only
these sequences' rows of the canonical pooled store by HTTP range requests, and caches them; trains the model, then each
generated control, with early stopping on valid; compares their test metrics; calibrates the model alone, with a
temperature fitted on valid and class-conditional conformal sets on calibration, scored on test; and adds every run's
row to `CACHE_ROOT/runs/runs.csv`, whose whole table is printed last. Figures (300 dpi PNG) and the epoch history go to `CACHE_ROOT/runs/<run>/`. Nothing is logged
to W&B.
"""

import json
import math
import os
import struct
import sys
import time
import zlib
import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers, hf_raise_for_status, parse_ratelimit_headers
from matplotlib.ticker import MaxNLocator
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


# The workspace root (or a Protify clone's root, which vendors `foundry`) holds `foundry`.
WORKSPACE = next(path for path in Path(__file__).resolve().parents if (path / "foundry").is_dir())
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from foundry.embedding import SyntheticControlView  # noqa: E402
from foundry.embedding.canonical import sequence_inventory  # noqa: E402
from foundry.embedding.contracts import normalize_sequence  # noqa: E402
from foundry.secrets_env import find_secrets_file, load_secrets_env  # noqa: E402
from foundry.training import clip_grad_norm, set_seed  # noqa: E402


# Settings ------------------------------------------------------------------------------------------------------------

RUN_NAME = "feature_fusion"  # names this run's row in the runs table and its directory; a rerun under a name replaces it

# A Hub dataset with a sequence column and a label column. Every sequence must be in the store below: the run stops and
# names the missing ones otherwise. The DeepLoc-2 example lacks 112 of its 8,537 in the ESMC-600 store.
DATASET = "Synthyra/benignity_v1"
DATASET_CONFIG: str | None = None  # the dataset's config (subset); None for a dataset with one
DATASET_REVISION: str | None = None  # a commit pins the rows; None reads the main branch
SEQUENCE_COLUMN = "sequence"
LABEL_COLUMN = "label_id"  # numbers or strings; mapped to 0 and 1
# The label mapped to 1, such as "soluble". With two labels, None takes the second in sorted order. ONE_VERSUS_REST maps
# POSITIVE_LABEL to 1 and every other label to 0, so a multiclass column trains a binary probe. Here: level1 class 7 is
# `non_enzyme` (the vocabulary is sorted by name, EC 1 to 7 then non_enzyme), so the probe calls non-enzyme against enzyme.
POSITIVE_LABEL: object | None = 1
ONE_VERSUS_REST = True

# The dataset's split for each role. A calibration of None carves CALIBRATION_FRACTION of valid, stratified by label.
SPLIT_NAMES: dict[str, str | None] = {"train": "train", "valid": "validation", "calibration": "calibration", "test": "test"}
CALIBRATION_FRACTION = 0.5
# Rows kept per role; None keeps every row. A split with more rows than its cap is subsampled to the cap, and a smaller
# split is kept whole. BALANCE_LABELS draws equal rows per class (a small class keeps all of its rows and its unused share
# goes to the other class); False keeps the label proportions.
SPLIT_ROWS: dict[str, int | None] = {"train": 100000, "valid": 10000, "calibration": 10000, "test": 10000}
BALANCE_LABELS = True
DATA_SEED = 0

# The canonical pooled store: a folder of the canonical repository, `esmc_300/pooled`, `esmc_600/pooled`, `esmc_6b/pooled`
# or `esmc_300_random/pooled`, at a pinned revision (None pins the repository's main commit when the run starts).
STORE_REPOSITORY = "Synthyra/esmc_embeddings"
STORE = "esmc_600/pooled"
STORE_REVISION: str | None = "d296ff9384898bf5b06ef5c901a969c6d1a810e5"  # the canonical stores as published 2026-10-08
BRANCHES = ("sae_max", "final_mean_var")  # the pooled streams the probe reads, in order
# Generated controls trained beside the run, on the same splits, model and settings, and compared with it but not
# calibrated: "random_vector" and "one_hot" (see Controls below). () trains none.
CONTROLS = ("random_vector", "one_hot")
#CONTROLS = ()
FETCH_WORKERS = 32  # parallel range requests; the Hub resets some connections above this
CACHE_ROOT = "C:/ws-cache/probe_playground"  # the store index, fetched rows and run outputs, reused by every later run

SEED = 42
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 10  # epochs without a lower validation loss before training stops
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
CONFORMAL_LEVELS = (0.8, 0.9, 0.95)

Matrix = np.ndarray | sparse.csr_matrix  # one branch's rows: dense, or CSR for the SAE codebook


# Model ---------------------------------------------------------------------------------------------------------------
#
# Protify's default feature fusion head (`ProjectedProbe` with the `expanded_mlp` trunk), written out to be changed.
# Each branch is standardized with train statistics, clamped to plus or minus 10 standard deviations, layer-normalized
# and projected to an equal share of 1024. The joined vector passes through `1024 -> 8192 -> f` with ReLU and dropout, a
# LayerNorm, and the output layer, where `f` is the output count rounded up to a multiple of 256.
#
# The model reads raw branch rows as a `{branch: (b, w)}` dictionary, so standardization and anything else done to the
# inputs belongs to the model. To try another model, edit these classes or `build_model`, and change RUN_NAME. Controls
# train the same model. Shapes: `b` rows, `w` branch width, `p` joined projection, `k` branches, `f` the class-aligned
# width, `c` outputs.


def branch_moments(X: Matrix) -> tuple[np.ndarray, np.ndarray]:
    """Column mean and standard deviation over train rows, with a zero deviation read as 1, as `StandardScaler` does."""
    # X: (n, w)
    squared = X.multiply(X) if sparse.issparse(X) else np.square(X)  # (n, w)
    mean = np.asarray(X.mean(axis=0, dtype=np.float64)).ravel()  # (w,)
    std = np.sqrt(np.maximum(np.asarray(squared.mean(axis=0, dtype=np.float64)).ravel() - mean**2, 0.0))  # (w,)
    std[std == 0.0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)  # (w,), (w,)


class BranchProjection(nn.Module):
    """Standardize one branch, clamp it, layer-normalize it, and project it."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, output_size: int, input_clip: float) -> None:
        # mean: (w,); std: (w,)
        super().__init__()
        self.input_clip = input_clip
        self.register_buffer("mean", torch.from_numpy(mean))  # (w,)
        self.register_buffer("std", torch.from_numpy(std))  # (w,)
        self.norm = nn.LayerNorm(len(mean))
        self.linear = nn.Linear(len(mean), output_size)

    def forward(self, X: Tensor) -> Tensor:
        # X: (b, w)
        X = ((X - self.mean) / self.std).clamp(-self.input_clip, self.input_clip)  # (b, w)
        return self.linear(self.norm(X))  # (b, p / k)


class FeatureFusionProbe(nn.Module):
    """Project each branch, join the projections, and read them with a two-layer MLP."""

    def __init__(
        self,
        moments: Mapping[str, tuple[np.ndarray, np.ndarray]],
        num_outputs: int,
        projection_size: int = 1024,
        hidden_size: int = 8192,
        dropout: float = 0.1,
        input_clip: float = 10.0,
    ) -> None:
        # moments: ((w,), (w,)) per branch name, its train mean and deviation
        super().__init__()
        branch_size = projection_size // len(moments)  # p / k
        self.branches = nn.ModuleDict({
            name: BranchProjection(mean, std, branch_size, input_clip) for name, (mean, std) in moments.items()
        })
        feature_size = math.ceil(num_outputs / 256) * 256  # f
        self.trunk = nn.Sequential(
            nn.Linear(branch_size * len(moments), hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, feature_size), nn.ReLU(), nn.Dropout(dropout),
            nn.LayerNorm(feature_size),
        )
        self.output = nn.Linear(feature_size, num_outputs)

    def forward(self, branches: Mapping[str, Tensor]) -> Tensor:
        # branches: (b, w) per branch name, its raw pooled rows
        H = torch.cat([projection(branches[name]) for name, projection in self.branches.items()], dim=-1)  # (b, p)
        H = self.trunk(H)  # (b, f)
        return self.output(H)  # (b, c)


def build_model(moments: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> nn.Module:
    """The model each run trains: two logits, read by a softmax, as Protify's binary heads are."""
    # moments: ((w,), (w,)) per branch name
    return FeatureFusionProbe(moments, num_outputs=2)


# Dataset -------------------------------------------------------------------------------------------------------------


@dataclass
class Split:
    """One role's normalized sequences and labels, row for row."""

    sequences: list[str]
    labels: np.ndarray  # (n,) int64, 1 for the positive label

    def subset(self, rows: np.ndarray) -> "Split":
        # rows: (m,)
        return Split([self.sequences[row] for row in rows], self.labels[rows])


def canonical_label(label: object) -> object:
    """A label in the form it is compared in: strings stripped of surrounding whitespace, whole floats as integers."""
    if isinstance(label, str):
        return label.strip()
    if isinstance(label, float) and label.is_integer():
        return int(label)
    return label


def binary_classes(labels_by_role: Mapping[str, list[object]]) -> list[object]:
    """The label each class id stands for, `[negative, positive]`, checked against Settings' label rules."""
    kinds = {type(label).__name__ for labels in labels_by_role.values() for label in labels}
    if "str" in kinds and len(kinds) > 1:
        raise ValueError(f"The labels mix strings with {sorted(kinds - {'str'})}; give the label column one type.")
    train = sorted(set(labels_by_role["train"]))
    if ONE_VERSUS_REST and POSITIVE_LABEL is None:
        raise ValueError("ONE_VERSUS_REST needs POSITIVE_LABEL, the label read as 1.")
    positive = train[-1] if POSITIVE_LABEL is None else canonical_label(POSITIVE_LABEL)
    if positive not in train:
        raise ValueError(f"POSITIVE_LABEL {POSITIVE_LABEL!r} is not a train label; the labels are {train[:20]}.")
    if ONE_VERSUS_REST:
        return [f"not {positive}", positive]

    if len(train) != 2:
        raise ValueError(f"Train has labels {train[:10]}; set ONE_VERSUS_REST and POSITIVE_LABEL to train one against the rest.")
    unseen = {label for labels in labels_by_role.values() for label in labels} - set(train)
    if unseen:
        raise ValueError(f"Labels absent from train: {sorted(unseen)}.")
    return [label for label in train if label != positive] + [positive]


def draw_rows(labels: np.ndarray, total: int, balance: bool, role: str) -> np.ndarray:
    """Ascending indices of `total` rows, at random within each class: equal per class when `balance`, else proportional.

    The draw depends only on DATA_SEED and the role, so every run keeps the same rows.
    """
    # labels: (n,)
    rng = np.random.default_rng([DATA_SEED, zlib.crc32(role.encode("ascii"))])
    sizes = np.bincount(labels, minlength=2)  # (2,)
    positives = min(sizes[1], max(total // 2, total - sizes[0])) if balance else round(total * sizes[1] / sizes.sum())
    rows = [rng.choice(np.flatnonzero(labels == label), size=quota, replace=False)
            for label, quota in ((0, total - positives), (1, positives))]
    return np.sort(np.concatenate(rows))  # (total,)


def load_splits() -> tuple[dict[str, Split], list[object]]:
    """Each role's split with labels 0 and 1, calibration carved from valid when the dataset has none, each role capped."""
    hub_splits = load_dataset(DATASET, DATASET_CONFIG, revision=DATASET_REVISION)
    sequences_by_role, labels_by_role = {}, {}
    for role, name in SPLIT_NAMES.items():
        if name is None:
            continue
        if name not in hub_splits:
            raise KeyError(f"{DATASET} has no split {name!r}; it has {sorted(hub_splits)}.")
        sequences, labels = hub_splits[name][SEQUENCE_COLUMN], hub_splits[name][LABEL_COLUMN]
        if None in sequences or None in labels:
            raise ValueError(f"{DATASET} {name}: some rows lack a sequence or a label.")
        sequences_by_role[role] = [normalize_sequence(sequence) for sequence in sequences]
        labels_by_role[role] = [canonical_label(label) for label in labels]

    classes = binary_classes(labels_by_role)
    splits = {
        role: Split(sequences_by_role[role], np.array([int(label == classes[1]) for label in labels], dtype=np.int64))
        for role, labels in labels_by_role.items()
    }
    if "calibration" not in splits:
        valid = splits["valid"]
        carved = draw_rows(valid.labels, round(CALIBRATION_FRACTION * len(valid.labels)), False, "carve")  # (m,)
        kept = np.setdiff1d(np.arange(len(valid.labels)), carved)  # (n - m,)
        splits["valid"], splits["calibration"] = valid.subset(kept), valid.subset(carved)
    for role, cap in SPLIT_ROWS.items():
        if cap is not None and len(splits[role].labels) > cap:
            splits[role] = splits[role].subset(draw_rows(splits[role].labels, cap, BALANCE_LABELS, role))
    return splits, classes


def describe_splits(splits: Mapping[str, Split], classes: Sequence[object]) -> pd.DataFrame:
    """Rows per role and per label, the positive share, and the sequences each role shares with train."""
    train = set(splits["train"].sequences)
    return pd.DataFrame.from_dict({
        role: {
            "rows": len(split.labels), f"{classes[0]}": int((split.labels == 0).sum()),
            f"{classes[1]} (positive)": int(split.labels.sum()), "positive share": split.labels.mean(),
            "also in train": sum(sequence in train for sequence in split.sequences),
        }
        for role, split in splits.items()
    }, orient="index")


# Embeddings ----------------------------------------------------------------------------------------------------------
#
# A pooled store keeps each stream as about a hundred safetensors parts (ESMC-600: 88 for `final_mean_var`, 112 for
# `sae_max`), and a part's rows are not grouped by dataset, so a few hundred sequences already touch nearly every part.
# Rows are therefore read one by one, with HTTP range requests, at a pinned store revision. The first run on a store downloads each segment's commit marker (`run.json`, about 7 MB
# each and 1.2 GB for both streams of a base) once, to index where every protein's row lies. Every later run reads only
# the index and the rows it lacks. Each protein costs about 9 KB of `final_mean_var` (ESMC-600) and 20 KB of `sae_max`.

EXPECTED_DTYPES = {"dense": {"values": "F32"}, "csr": {"indptr": "I64", "indices": "I32", "values": "F32"}}
RESOLVER_RESERVE = 4 * FETCH_WORKERS  # resolver requests left unspent, so threads already in flight cannot overrun the quota
FETCH_RETRIES = 6  # retries of one request, waiting 1, 2, 4, 8, 16 and 30 s, or until a 429's window resets


@dataclass(frozen=True)
class StorePart:
    path: str
    starts: dict[str, int]  # each tensor's first byte in the file


@dataclass
class StreamIndex:
    """Where each row of one pooled stream lies: the sorted SHA-256 digests with their part and row."""

    width: int  # w
    layout: str  # "dense" or "csr"
    parts: list[StorePart]
    digests: np.ndarray  # (n,) S32, ascending
    part_ids: np.ndarray  # (n,) int32
    rows: np.ndarray  # (n,) int64


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write an `.npz` beside its destination, then move it into place, so a killed run leaves no half-written cache."""
    # arrays: (...) each, any shape
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.stem}.partial.npz")
    np.savez(partial, **arrays)
    partial.replace(path)


def range_bytes(response: httpx.Response, path: str, start: int | None, stop: int | None) -> bytes:
    """A response's body, after checking its status and, for a range, its length."""
    hf_raise_for_status(response)
    if start is not None and len(response.content) != stop - start:
        raise OSError(f"{path}: asked for {stop - start} bytes at {start}, received {len(response.content)}.")
    return response.content


class HubStore:
    """One pinned revision of a canonical pooled store on the Hub: its index, and its rows by range request.

    Every request to the Hub's resolver carries the HF_TOKEN that `main` loads from `.secrets.env`, and counts against
    the account's resolver quota (20,000 per five minutes for a signed-in account, fewer anonymously). A part's rows are
    therefore not read through the resolver one by one: each batch resolves each part it touches once, to the signed
    CDN address the resolver redirects to, and reads the rows there, where no quota applies. When the resolver reports
    fewer than RESOLVER_RESERVE requests left, every thread waits for the window to reset, and a 429 that slips through
    waits for the reset the response names.

    Requests go through the store's own HTTP client, never `huggingface_hub`'s shared one: `http_backoff` closes that
    client on any connection error, and on Windows every other thread's request on it then fails with WinError 10038
    ("not a socket"), which with 32 threads cascades. Here a dropped connection is only retried.
    """

    def __init__(self, repo: str, folder: str, revision: str | None, token: str) -> None:
        self.repo = repo
        self.folder = folder  # the store's folder in the repository, `<model>/pooled`
        self.api = HfApi(token=token)
        self.headers = build_hf_headers(token=token)
        self.client = httpx.Client(timeout=httpx.Timeout(60.0, connect=20.0), limits=httpx.Limits(max_connections=2 * FETCH_WORKERS))
        self.resume_at = 0.0  # the `time.monotonic()` before which no resolver request is sent
        self.revision = revision or self.api.dataset_info(repo).sha
        self.manifest = json.loads(self.read(f"{folder}/canonical.json"))
        self.directory = Path(CACHE_ROOT) / f"{repo.replace('/', '--')}--{folder.replace('/', '--')}@{self.revision}"

    def get(self, url: str, headers: Mapping[str, str], path: str, follow_redirects: bool = True) -> httpx.Response:
        """A GET retried through dropped connections, timeouts, 5xx responses and 429s, with waits doubling to 30 s.

        `path` names the file in an error, because a CDN address carries its signature.
        """
        for attempt in range(FETCH_RETRIES + 1):
            wait = min(2.0**attempt, 30.0)
            try:
                response = self.client.get(url, headers=headers, follow_redirects=follow_redirects)
            except httpx.TransportError as error:
                failure = f"{type(error).__name__}: {error}"
            else:
                if response.status_code != 429 and response.status_code < 500:
                    return response
                failure = f"HTTP {response.status_code}"
                quota = parse_ratelimit_headers(response.headers)
                if response.status_code == 429 and quota is not None:
                    wait = quota.reset_in_seconds + 1.0
            if attempt < FETCH_RETRIES:
                time.sleep(wait)
        raise OSError(f"{path}: {FETCH_RETRIES + 1} attempts failed, the last with {failure}.")

    def resolve(self, path: str, headers: Mapping[str, str], follow_redirects: bool = True) -> httpx.Response:
        """One authenticated resolver request for a file, paced by the quota the Hub reports in its `RateLimit` header.

        httpx drops the token when a redirect leaves the Hub, so a followed redirect never sends it to the CDN.
        """
        time.sleep(max(0.0, self.resume_at - time.monotonic()))
        url = hf_hub_url(self.repo, path, repo_type="dataset", revision=self.revision)
        response = self.get(url, {**self.headers, **headers}, path, follow_redirects)
        quota = parse_ratelimit_headers(response.headers)
        if quota is not None and quota.remaining < RESOLVER_RESERVE and time.monotonic() >= self.resume_at:
            self.resume_at = time.monotonic() + quota.reset_in_seconds + 1
            tqdm.write(f"  Hub resolver quota: {quota.remaining} requests left, pausing {quota.reset_in_seconds + 1} s for the window to reset")
        return response

    def read(self, path: str, start: int | None = None, stop: int | None = None) -> bytes:
        """A file, or its bytes [start, stop), through the resolver."""
        response = self.resolve(path, {} if start is None else {"range": f"bytes={start}-{stop - 1}"})
        return range_bytes(response, path, start, stop)

    def locate(self, path: str) -> str:
        """The signed CDN address the resolver redirects a part to, valid for about an hour."""
        response = self.resolve(path, {"range": "bytes=0-0"}, follow_redirects=False)
        if not response.is_redirect:
            raise OSError(f"{path}: the resolver answered {response.status_code}, not a redirect to storage.")
        return response.headers["location"]

    def read_many(self, requests: Sequence[tuple[str, int, int]], desc: str) -> list[bytes]:
        """The bytes of each (path, start, stop) range, read in parallel from the parts' CDN addresses under a progress
        bar in bytes. The CDN request carries no token: the signed address grants the read."""
        paths = sorted({path for path, _, _ in requests})
        with ThreadPoolExecutor(FETCH_WORKERS) as pool:
            located = tqdm(pool.map(self.locate, paths), total=len(paths), desc=f"{desc}, locating parts", unit="part",
                           leave=False, dynamic_ncols=True)
            locations = dict(zip(paths, located, strict=True))
        progress = tqdm(total=sum(stop - start for _, start, stop in requests), desc=desc, unit="B", unit_scale=True,
                        unit_divisor=1024, dynamic_ncols=True)

        def read_one(request: tuple[str, int, int]) -> bytes:
            path, start, stop = request
            if stop == start:
                return b""
            response = self.get(locations[path], {"range": f"bytes={start}-{stop - 1}"}, path)
            block = range_bytes(response, path, start, stop)
            progress.update(len(block))
            return block

        with ThreadPoolExecutor(FETCH_WORKERS) as pool:
            blocks = list(pool.map(read_one, requests))
        progress.close()
        return blocks

    def part(self, path: str, layout: str, rows: int) -> StorePart:
        """One safetensors part's tensor offsets, after checking its dtypes and row count against its commit marker."""
        (size,) = struct.unpack("<Q", self.read(path, 0, 8))
        header = json.loads(self.read(path, 8, 8 + size))
        header.pop("__metadata__", None)
        dtypes = {name: entry["dtype"] for name, entry in header.items()}
        stored = header["values"]["shape"][0] if layout == "dense" else header["indptr"]["shape"][0] - 1
        if dtypes != EXPECTED_DTYPES[layout] or stored != rows:
            raise ValueError(f"{path} holds {dtypes} over {stored} rows; its marker expects {EXPECTED_DTYPES[layout]} over {rows}.")
        return StorePart(path, {name: 8 + size + entry["data_offsets"][0] for name, entry in header.items()})

    def stream_index(self, branch: str) -> StreamIndex:
        """Where each row of a stream lies, read from its commit markers once per store revision and then cached."""
        path = self.directory / "index" / f"{branch}_index.npz"
        if path.exists():
            with np.load(path) as saved:
                meta = json.loads(str(saved["meta"]))
                parts = [StorePart(**part) for part in meta["parts"]]
                return StreamIndex(meta["width"], meta["layout"], parts, saved["digests"], saved["part_ids"], saved["rows"])

        record = self.manifest["streams"][branch]
        stream = f"{self.folder}/{record['key']}"
        tree = self.api.list_repo_tree(self.repo, path_in_repo=stream, repo_type="dataset", revision=self.revision, recursive=True)
        markers = sorted(item.path for item in tree if item.path.startswith(f"{stream}/segments/") and item.path.endswith("/run.json"))
        progress = tqdm(total=len(markers), desc=f"{branch}: indexing commit markers, once per store revision", unit="marker",
                        dynamic_ncols=True)

        def marker_parts(marker_path: str) -> list[tuple[StorePart, list[str]]]:
            directory = marker_path.rsplit("/", 1)[0]
            parts = [
                (self.part(f"{directory}/part-{entry['part']:05d}.safetensors", record["layout"], len(entry["digests"])), entry["digests"])
                for entry in json.loads(self.read(marker_path))["parts"]
            ]
            progress.update(1)
            return parts

        with ThreadPoolExecutor(16) as pool:
            parts = [pair for segment in pool.map(marker_parts, markers) for pair in segment]
        progress.close()

        blocks = [np.frombuffer(bytes.fromhex("".join(digests)), dtype="S32") for _, digests in parts]  # each (r,)
        digests = np.concatenate(blocks)  # (n,)
        part_ids = np.repeat(np.arange(len(blocks), dtype=np.int32), [len(block) for block in blocks])  # (n,)
        rows = np.concatenate([np.arange(len(block)) for block in blocks])  # (n,)
        order = np.argsort(digests, kind="stable")  # (n,)
        index = StreamIndex(record["width"], record["layout"], [part for part, _ in parts], digests[order], part_ids[order], rows[order])
        if (index.digests[1:] == index.digests[:-1]).any():
            raise ValueError(f"{self.repo}/{self.folder} {branch} holds a protein twice.")
        meta = json.dumps({"width": index.width, "layout": index.layout, "parts": [asdict(part) for part in index.parts]})
        save_npz(path, digests=index.digests, part_ids=index.part_ids, rows=index.rows, meta=np.array(meta))
        return index

    def fetch(self, index: StreamIndex, positions: np.ndarray, desc: str) -> Matrix:
        """The rows at these index positions: dense rows directly, CSR rows by their offsets, then columns and values."""
        # positions: (u,)
        parts = [index.parts[part_id] for part_id in index.part_ids[positions]]
        rows = [int(row) for row in index.rows[positions]]
        if index.layout == "dense":
            size = 4 * index.width
            requests = [(part.path, part.starts["values"] + size * row, part.starts["values"] + size * (row + 1)) for part, row in zip(parts, rows, strict=True)]
            return np.frombuffer(b"".join(self.read_many(requests, desc)), dtype="<f4").reshape(len(rows), index.width)  # (u, w)

        requests = [(part.path, part.starts["indptr"] + 8 * row, part.starts["indptr"] + 8 * row + 16) for part, row in zip(parts, rows, strict=True)]
        spans = [np.frombuffer(block, dtype="<i8") for block in self.read_many(requests, f"{desc}, row offsets")]  # u arrays of (2,)
        requests = [(part.path, part.starts[name] + 4 * int(first), part.starts[name] + 4 * int(last))
                    for part, (first, last) in zip(parts, spans, strict=True) for name in ("indices", "values")]
        blocks = self.read_many(requests, desc)
        indptr = np.concatenate([[0], np.cumsum([last - first for first, last in spans])])  # (u + 1,)
        columns = np.frombuffer(b"".join(blocks[0::2]), dtype="<i4")  # (z,)
        values = np.frombuffer(b"".join(blocks[1::2]), dtype="<f4")  # (z,)
        return sparse.csr_matrix((values, columns, indptr), shape=(len(rows), index.width))  # (u, w)

    def rows(self, branch: str, digests: np.ndarray, sequence_of: Mapping[bytes, str]) -> Matrix:
        """The rows of `digests` in their order: cached ones from disk, the rest fetched and added to the cache."""
        # digests: (q,) S32, unique
        if branch not in self.manifest["streams"]:
            raise KeyError(f"{self.repo}/{self.folder} has no stream {branch!r}; it holds {sorted(self.manifest['streams'])}.")
        layout, width = self.manifest["streams"][branch]["layout"], self.manifest["streams"][branch]["width"]
        path = self.directory / "rows" / f"{branch}.npz"
        cached_digests = np.zeros(0, dtype="S32")  # (m,)
        cached = np.zeros((0, width), dtype=np.float32) if layout == "dense" else sparse.csr_matrix((0, width), dtype=np.float32)  # (m, w)
        if path.exists():
            with np.load(path) as saved:
                cached_digests = saved["digests"]  # (m,)
                cached = saved["rows"] if layout == "dense" else sparse.csr_matrix(
                    (saved["data"], saved["indices"], saved["indptr"]), shape=(len(cached_digests), width))  # (m, w)
        missing = np.setdiff1d(digests, cached_digests)  # (u,), sorted
        report(branch, f"{len(digests) - len(missing):,} of {len(digests):,} proteins cached, {len(missing):,} to fetch")
        if len(missing) == 0:
            return cached[np.searchsorted(cached_digests, digests)]  # (q, w)

        index = self.stream_index(branch)
        positions = np.minimum(np.searchsorted(index.digests, missing), len(index.digests) - 1)  # (u,)
        absent = missing[index.digests[positions] != missing]  # (a,)
        if len(absent):
            examples = [sequence_of[digest][:40] for digest in absent[:3]]
            raise KeyError(f"{len(absent)} sequences are not in {self.repo}@{self.revision[:12]} {self.folder} {branch}, such as {examples}.")

        fetched = self.fetch(index, positions, f"{branch}: {len(missing):,} proteins")  # (u, w)
        cached_digests = np.concatenate([cached_digests, missing])  # (m + u,)
        cached = np.vstack([cached, fetched]) if layout == "dense" else sparse.vstack([cached, fetched], format="csr")  # (m + u, w)
        order = np.argsort(cached_digests, kind="stable")  # (m + u,)
        cached_digests, cached = cached_digests[order], cached[order]  # (m + u,), (m + u, w)
        arrays = {"rows": cached} if layout == "dense" else {"data": cached.data, "indices": cached.indices, "indptr": cached.indptr}
        save_npz(path, digests=cached_digests, **arrays)
        report(branch, f"cache now holds {len(cached_digests):,} proteins, {path.stat().st_size / 2**20:,.0f} MB")
        return cached[np.searchsorted(cached_digests, digests)]  # (q, w)


def store_features(store: HubStore, splits: Mapping[str, Split]) -> dict[str, dict[str, Matrix]]:
    """Each role's rows of each branch, row for row with its split."""
    inventory = sequence_inventory([sequence for split in splits.values() for sequence in split.sequences])
    digest_of = {text: bytes.fromhex(key.sha256) for text, key in inventory.items()}
    sequence_of = {digest: text for text, digest in digest_of.items()}
    unique = np.array(sorted(sequence_of), dtype="S32")  # (u,)
    features: dict[str, dict[str, Matrix]] = {role: {} for role in splits}
    for branch in BRANCHES:
        rows = store.rows(branch, unique, sequence_of)  # (u, w)
        for role, split in splits.items():
            positions = np.searchsorted(unique, np.array([digest_of[text] for text in split.sequences], dtype="S32"))  # (n,)
            features[role][branch] = rows[positions]  # (n, w)
    return features


# Controls ------------------------------------------------------------------------------------------------------------
#
# A control carries no information about a protein beyond what its row says, so a probe that beats it has learned from
# the embedding. The embedding engine makes both from the sequence where they are read, and they are never stored
# (docs/conventions/probes.md, "Arms and controls"): `random_vector` gives each of the l + 2 token rows an independent
# standard normal row of width 960, seeded by the sequence's SHA-256, and `one_hot` gives each token one 1 over 26 residue
# symbols plus CLS and EOS (width 28), so its mean is the amino acid composition. Each control trains on the mean over
# all l + 2 rows, as the workspace's `mean_final` arm reads every control. The random control's cost is its definition:
# about 12 ms of float64 log, cos and sin per protein of 500 residues, so threads are the only speedup that keeps its
# values.

CONTROL_CHUNK = 16  # proteins per engine call, inside its default read budget
CONTROL_WORKERS = min(8, os.cpu_count() or 1)  # threads; more gave nothing on a 16-thread laptop


def control_features(kind: str, splits: Mapping[str, Split]) -> dict[str, dict[str, np.ndarray]]:
    """A generated control's `mean` branch for every role, row for row with its split.

    The engine's view makes and pools the rows, so the values are those every other reader of the control sees. Its
    chunks run on threads, because the work is numpy and torch arithmetic that releases the GIL: worker processes
    would each import torch again, and Protify's `MaterializedControl` costs about 8 s to import.
    """
    texts = sorted({sequence for split in splits.values() for sequence in split.sequences})
    view = SyntheticControlView(kind, sequence_inventory(texts))
    chunks = [texts[start : start + CONTROL_CHUNK] for start in range(0, len(texts), CONTROL_CHUNK)]
    progress = tqdm(total=len(texts), desc=f"{kind} control rows", unit="protein", leave=False, dynamic_ncols=True)

    def pool_chunk(chunk: list[str]) -> np.ndarray:
        block = view.pooled(chunk).numpy()  # (b, 2w), mean then variance
        progress.update(len(chunk))
        return block  # (b, 2w)

    with ThreadPoolExecutor(CONTROL_WORKERS) as pool:
        pooled = np.concatenate(list(pool.map(pool_chunk, chunks)))  # (u, 2w)
    progress.close()
    width = pooled.shape[1] // 2  # w
    row_of = {text: row for row, text in enumerate(texts)}
    return {
        role: {"mean": pooled[[row_of[sequence] for sequence in split.sequences], :width]}  # (n, w)
        for role, split in splits.items()
    }


# Training ------------------------------------------------------------------------------------------------------------


def branch_batch(features: Mapping[str, Matrix], rows: np.ndarray, device: torch.device) -> dict[str, Tensor]:
    """The rows of every branch as dense float32 tensors on the device."""
    # rows: (b,)
    batch = {}
    for name, X in features.items():
        block = X[rows]  # (b, w)
        batch[name] = torch.from_numpy(block.toarray() if sparse.issparse(block) else np.ascontiguousarray(block)).to(device)  # (b, w)
    return batch  # {name: (b, w)}


@torch.no_grad()
def predict(model: nn.Module, features: Mapping[str, Matrix], device: torch.device, desc: str) -> Tensor:
    """The model's outputs for every row, in evaluation mode, on the CPU, under a progress bar that clears when done."""
    model.eval()
    count = next(iter(features.values())).shape[0]  # n
    outputs = [
        model(branch_batch(features, np.arange(start, min(start + 1024, count)), device)).float().cpu()  # (b, c)
        for start in tqdm(range(0, count, 1024), desc=desc, unit="batch", leave=False, dynamic_ncols=True)
    ]
    return torch.cat(outputs)  # (n, c)


def warmup_cosine(step: int, total_steps: int) -> float:
    """Linear warmup, then cosine decay to zero, as `transformers`' cosine schedule with warmup."""
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)))


def train_probe(
    model: nn.Module, features: Mapping[str, Mapping[str, Matrix]], targets: Mapping[str, Tensor], device: torch.device,
    run_name: str,
) -> list[dict[str, float]]:
    """Train with cross-entropy until valid loss has not fallen for PATIENCE epochs, restore the best epoch, and
    return the per-epoch history."""
    # targets: (n,) per role
    set_seed(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    count = len(targets["train"])  # n
    scheduler = LambdaLR(optimizer, lambda step: warmup_cosine(step, math.ceil(count / BATCH_SIZE) * MAX_EPOCHS))
    rng = np.random.default_rng(SEED)
    history, best_state, stale = [], None, 0

    epochs = tqdm(range(1, MAX_EPOCHS + 1), desc=f"{run_name}: epochs", unit="epoch", leave=False, dynamic_ncols=True)
    for epoch in epochs:
        model.train()
        order = rng.permutation(count)  # (n,)
        train_loss = 0.0
        batches = tqdm(range(0, count, BATCH_SIZE), desc=f"epoch {epoch}", unit="batch", leave=False, dynamic_ncols=True)
        for step, start in enumerate(batches, start=1):
            rows = order[start : start + BATCH_SIZE]  # (b,)
            loss = F.cross_entropy(model(branch_batch(features["train"], rows, device)), targets["train"][torch.from_numpy(rows)].to(device))  # ()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm(model, MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * len(rows)
            if step % 10 == 0:
                batches.set_postfix(loss=f"{train_loss / (start + len(rows)):.4f}", refresh=False)

        valid_logits = predict(model, features["valid"], device, "valid loss")  # (n_valid, 2)
        valid_loss = F.cross_entropy(valid_logits, targets["valid"]).item()
        valid_auroc = roc_auc_score(targets["valid"].numpy(), valid_logits.softmax(dim=-1)[:, 1].numpy())
        history.append({"epoch": epoch, "train_loss": train_loss / count, "valid_loss": valid_loss, "valid_auroc": valid_auroc})
        best = min(history, key=lambda row: row["valid_loss"])
        if best["epoch"] == epoch:
            best_state, stale = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}, 0
        else:
            stale += 1
        epochs.set_postfix(train=f"{train_loss / count:.4f}", valid=f"{valid_loss:.4f}", auroc=f"{valid_auroc:.3f}",
                           best=best["epoch"], patience=f"{stale}/{PATIENCE}", lr=f"{scheduler.get_last_lr()[0]:.1e}")
        if stale >= PATIENCE:
            break
    epochs.close()

    model.load_state_dict(best_state)
    return history


# Evaluation ----------------------------------------------------------------------------------------------------------
#
# Threshold metrics use P(positive) of 0.5; temperature scaling cannot change them, because it keeps every row on the
# same side of 0.5. The temperature is fitted on valid (the split early stopping read), the conformal thresholds on
# calibration, and test is only scored. At a target coverage, a set holding both labels means the model cannot decide
# that protein at that confidence, and an empty set means neither label clears its threshold.

RELIABILITY_BINS = 10


def positive_probability(logits: Tensor, temperature: float = 1.0) -> np.ndarray:
    # logits: (n, 2)
    return (logits / temperature).softmax(dim=-1)[:, 1].numpy()  # (n,)


def binary_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Threshold metrics at a probability of 0.5, ranking metrics, and the two proper scores."""
    # probabilities: (n,) P(positive); labels: (n,)
    predictions = (probabilities >= 0.5).astype(np.int64)  # (n,)
    return {
        "accuracy": accuracy_score(labels, predictions), "balanced accuracy": balanced_accuracy_score(labels, predictions),
        "F1": f1_score(labels, predictions), "MCC": matthews_corrcoef(labels, predictions),
        "AUROC": roc_auc_score(labels, probabilities), "AUPRC": average_precision_score(labels, probabilities),
        "NLL": log_loss(labels, probabilities, labels=[0, 1]), "Brier": brier_score_loss(labels, probabilities),
    }


def reliability_bins(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per equal-width probability bin: the mean predicted P(positive), the observed positive share, and the row count."""
    # probabilities: (n,); labels: (n,); k bins
    bin_of_row = np.minimum((probabilities * RELIABILITY_BINS).astype(np.int64), RELIABILITY_BINS - 1)  # (n,)
    counts = np.bincount(bin_of_row, minlength=RELIABILITY_BINS)  # (k,)
    predicted = np.bincount(bin_of_row, weights=probabilities, minlength=RELIABILITY_BINS)  # (k,)
    observed = np.bincount(bin_of_row, weights=labels, minlength=RELIABILITY_BINS)  # (k,)
    filled = counts > 0  # (k,)
    return predicted[filled] / counts[filled], observed[filled] / counts[filled], counts[filled]  # (k',), (k',), (k',)


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """The row-weighted mean gap between predicted and observed positive share over the reliability bins."""
    # probabilities: (n,); labels: (n,)
    predicted, observed, counts = reliability_bins(probabilities, labels)  # (k',), (k',), (k',)
    return float(np.sum(counts * np.abs(predicted - observed)) / counts.sum())


def fit_temperature(logits: Tensor, labels: Tensor) -> float:
    """The temperature that minimizes the negative log likelihood of these logits."""
    # logits: (n, c); labels: (n,)
    log_temperature = torch.zeros(1, requires_grad=True)  # (1,)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=200)

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.exp(), labels)  # ()
        loss.backward()
        return loss  # ()

    optimizer.step(closure)
    return log_temperature.exp().item()


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """The split conformal threshold: the ceil((n + 1) level)-th smallest score, infinite when that exceeds n."""
    # scores: (n,)
    rank = math.ceil((len(scores) + 1) * level)
    return math.inf if rank > len(scores) else float(np.sort(scores)[rank - 1])


def conformal_table(
    calibration_probabilities: np.ndarray, calibration_labels: np.ndarray, test_probabilities: np.ndarray,
    test_labels: np.ndarray, classes: Sequence[object],
) -> pd.DataFrame:
    """Test coverage, overall and per true label, and set sizes of class-conditional (Mondrian) LAC sets per level.

    Label y joins a row's set when P(y) clears label y's threshold, set from the calibration rows of label y alone, so
    coverage holds within each label and therefore under any label mix. A pooled threshold would assume calibration and
    test share their label proportions, which capping with BALANCE_LABELS breaks when one split is subsampled and one is not.
    """
    # calibration_probabilities: (n_cal,); calibration_labels: (n_cal,); test_probabilities: (n_test,); test_labels: (n_test,)
    calibration = np.stack([1.0 - calibration_probabilities, calibration_probabilities], axis=1)  # (n_cal, 2)
    scores = 1.0 - calibration[np.arange(len(calibration)), calibration_labels]  # (n_cal,)
    test = np.stack([1.0 - test_probabilities, test_probabilities], axis=1)  # (n_test, 2)
    rows = {}
    for level in CONFORMAL_LEVELS:
        thresholds = np.array([conformal_quantile(scores[calibration_labels == label], level) for label in (0, 1)])  # (2,)
        sets = test >= 1.0 - thresholds  # (n_test, 2)
        covered = sets[np.arange(len(sets)), test_labels]  # (n_test,)
        sizes = sets.sum(axis=1)  # (n_test,)
        rows[f"{level:.0%}"] = {
            "coverage": covered.mean(), f"coverage, {classes[0]}": covered[test_labels == 0].mean(),
            f"coverage, {classes[1]}": covered[test_labels == 1].mean(), "mean set size": sizes.mean(),
            "single label": (sizes == 1).mean(), "both labels": (sizes == 2).mean(), "empty": (sizes == 0).mean(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


# Figures -------------------------------------------------------------------------------------------------------------
#
# Colors are slots 1 to 3 of the workspace's validated categorical palette, assigned by entity: train, valid, test in
# the results figure; raw, temperature-scaled in the calibration figure.

SERIES = {"train": "#2a78d6", "valid": "#eb6834", "test": "#1baf7a"}
SCALING = {"raw": "#2a78d6", "temperature-scaled": "#eb6834"}
REFERENCE = "#8a8984"  # chance and identity lines
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": REFERENCE, "axes.labelcolor": "#52514e",
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
    "font.size": 9, "legend.frameon": False, "lines.linewidth": 2,
})


def letter_panels(axes: Sequence[plt.Axes]) -> None:
    for letter, axis in zip("ABCD", axes, strict=False):
        axis.text(-0.16, 1.02, letter, transform=axis.transAxes, fontsize=11, fontweight="bold", va="bottom")


def plot_results(
    history: Sequence[Mapping[str, float]], probabilities: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray],
    classes: Sequence[object],
) -> plt.Figure:
    """(A) loss per epoch, (B) ROC and (C) precision-recall on valid and test, (D) the test confusion matrix."""
    # probabilities: (n,) per role; labels: (n,) per role
    figure, axes = plt.subplots(2, 2, figsize=(9, 7.5), layout="constrained")
    loss_axis, roc_axis, pr_axis, confusion_axis = axes.ravel()

    best_epoch = min(history, key=lambda row: row["valid_loss"])["epoch"]
    for role in ("train", "valid"):
        loss_axis.plot([row["epoch"] for row in history], [row[f"{role}_loss"] for row in history], color=SERIES[role], label=role)
    loss_axis.axvline(best_epoch, color=REFERENCE, linestyle="--", linewidth=1, label=f"best epoch ({best_epoch})")
    loss_axis.set(xlabel="Epoch", ylabel="Cross-entropy")
    loss_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    loss_axis.legend()

    for role in ("valid", "test"):
        false_positive, true_positive, _ = roc_curve(labels[role], probabilities[role])  # (t,), (t,)
        roc_axis.plot(false_positive, true_positive, color=SERIES[role],
                      label=f"{role} (AUROC {roc_auc_score(labels[role], probabilities[role]):.3f})")
        precision, recall, _ = precision_recall_curve(labels[role], probabilities[role])  # (t,), (t,)
        pr_axis.plot(recall, precision, color=SERIES[role],
                     label=f"{role} (AUPRC {average_precision_score(labels[role], probabilities[role]):.3f})")
    roc_axis.plot([0, 1], [0, 1], color=REFERENCE, linestyle="--", linewidth=1, label="chance")
    roc_axis.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
    roc_axis.legend(loc="lower right")
    pr_axis.axhline(labels["test"].mean(), color=REFERENCE, linestyle="--", linewidth=1, label="test positive share")
    pr_axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01))
    pr_axis.legend(loc="lower left")

    counts = confusion_matrix(labels["test"], probabilities["test"] >= 0.5, labels=[0, 1])  # (2, 2)
    confusion_axis.imshow(counts, cmap="Blues", vmin=0)
    for (true, predicted), count in np.ndenumerate(counts):
        confusion_axis.text(predicted, true, f"{count}\n{count / counts[true].sum():.0%} of row", ha="center", va="center",
                            color="white" if count > counts.max() / 2 else "#0b0b0b")
    names = [f"{classes[0]}", f"{classes[1]} (positive)"]
    confusion_axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=names, yticklabels=names, xlabel="Predicted (test)", ylabel="True")
    confusion_axis.grid(False)
    letter_panels(axes.ravel())
    return figure


def plot_calibration(probabilities: Mapping[str, np.ndarray], labels: np.ndarray) -> plt.Figure:
    """(A) test reliability before and after temperature scaling, (B) the test distribution of P(positive)."""
    # probabilities: (n,) per scaling, raw or temperature-scaled; labels: (n,)
    figure, (reliability_axis, histogram_axis) = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    reliability_axis.plot([0, 1], [0, 1], color=REFERENCE, linestyle="--", linewidth=1, label="perfect calibration")
    for name, values in probabilities.items():
        predicted, observed, _ = reliability_bins(values, labels)  # (k',), (k',), (k',)
        reliability_axis.plot(predicted, observed, color=SCALING[name], marker="o", markersize=6,
                              label=f"{name} (ECE {expected_calibration_error(values, labels):.3f})")
        histogram_axis.hist(values, bins=np.linspace(0.0, 1.0, RELIABILITY_BINS + 1), histtype="step", linewidth=2,
                            color=SCALING[name], label=name)
    reliability_axis.set(xlabel="Mean predicted P(positive)", ylabel="Observed positive share", xlim=(0, 1), ylim=(0, 1))
    reliability_axis.legend(loc="upper left")
    histogram_axis.set(xlabel="Predicted P(positive), test", ylabel="Proteins", xlim=(0, 1))
    histogram_axis.legend(loc="upper center")
    letter_panels([reliability_axis, histogram_axis])
    return figure


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Save a figure as a 300 dpi PNG and release it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)


# Printing ------------------------------------------------------------------------------------------------------------
#
# Section rules are box-drawing characters when the terminal takes UTF-8, and ASCII when output goes to a file or pipe
# in a legacy encoding.

RULE = "━" if (sys.stdout.encoding or "").lower().replace("-", "") == "utf8" else "="
COMPARISON_COLUMNS = ["store", "valid AUROC", "test accuracy", "test F1", "test MCC", "test AUROC", "test AUPRC", "test NLL",
                      "test Brier"]
RUN_COLUMNS = ["dataset", "store", "parameters", "best epoch", "valid AUROC", "test AUROC", "test AUPRC", "test MCC",
               "test ECE, scaled", "coverage at 90%"]
HIGHER_IS_BETTER = ["valid AUROC", "test AUROC", "test AUPRC", "test MCC"]


def section(title: str) -> None:
    print(f"\n{RULE * 3} {title} {RULE * max(3, 110 - len(title))}", flush=True)


def report(name: str, value: object) -> None:
    print(f"  {name:<16}{value}", flush=True)


def show_table(frame: pd.DataFrame, title: str, formats: Mapping[str, str] | None = None) -> None:
    """Print a titled, indented table with three spaces between columns: each column by its format in `formats`, else
    floats to three decimals and integers with thousands separators."""
    formats = formats or {}

    def cell(column: str, value: object) -> str:
        default = "{:.3f}" if isinstance(value, (float, np.floating)) else "{:,}" if isinstance(value, (int, np.integer)) else "{}"
        return formats.get(column, default).format(value)

    columns = [[""] + [str(name) for name in frame.index]]
    columns += [[str(column)] + [cell(column, value) for value in frame[column]] for column in frame]
    widths = [max(map(len, column)) for column in columns]
    lines = [
        "   ".join(text.ljust(width) if position == 0 else text.rjust(width) for position, (text, width) in enumerate(zip(row, widths, strict=True)))
        for row in zip(*columns, strict=True)
    ]
    print(f"\n  {title}\n" + "\n".join(f"    {line}" for line in lines), flush=True)


def record_runs(rows: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    """Add these runs to the runs table on disk, each replacing an earlier run of the same name, and return the table."""
    path = Path(CACHE_ROOT) / "runs" / "runs.csv"
    runs = pd.read_csv(path, index_col=0) if path.exists() else pd.DataFrame()
    runs = pd.concat([runs.drop(index=list(rows), errors="ignore"), pd.DataFrame.from_dict(rows, orient="index")])
    path.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(path.with_name("runs.partial.csv"))
    path.with_name("runs.partial.csv").replace(path)
    return runs


def show_runs(runs: pd.DataFrame) -> None:
    """Print every run, with an asterisk on the best value of each compared metric; a control has no calibration columns."""
    shown = runs[RUN_COLUMNS].astype(object)
    for column in [*HIGHER_IS_BETTER, "test ECE, scaled"]:
        best = runs[column].max() if column in HIGHER_IS_BETTER else runs[column].min()
        shown[column] = ["-  " if pd.isna(value) else f"{value:.3f}{' *' if value == best and len(runs) > 1 else '  '}"
                         for value in runs[column]]
    shown["coverage at 90%"] = ["-" if pd.isna(value) else f"{value:.1%}" for value in runs["coverage at 90%"]]
    shown["parameters"] = [f"{value:,}" for value in runs["parameters"]]
    show_table(shown, f"* best of each metric; every column is in {Path(CACHE_ROOT) / 'runs' / 'runs.csv'}")


# Run -----------------------------------------------------------------------------------------------------------------


def train_arm(
    run_name: str, features: Mapping[str, Mapping[str, Matrix]], splits: Mapping[str, Split], classes: Sequence[object],
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    """Train `build_model` on one set of branches, save its history and results figure, and return its outputs on the
    held-out roles and its row of the runs table."""
    # features: (n, w) per branch per role
    run_directory = Path(CACHE_ROOT) / "runs" / run_name
    branches = list(features["train"])

    set_seed(SEED)
    model = build_model({branch: branch_moments(features["train"][branch]) for branch in branches}).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    targets = {role: torch.from_numpy(split.labels) for role, split in splits.items()}  # targets: (n,) per role
    history = train_probe(model, features, targets, device, run_name)
    best = min(history, key=lambda row: row["valid_loss"])
    print(f"  {run_name:<32}{parameters:>12,} parameters   best epoch {best['epoch']} of {len(history)}   "
          f"valid loss {best['valid_loss']:.4f}   valid AUROC {best['valid_auroc']:.3f}", flush=True)

    outputs = {role: predict(model, features[role], device, f"predict {role}") for role in ("valid", "calibration", "test")}  # outputs: (n, 2) per role
    labels = {role: splits[role].labels for role in outputs}  # labels: (n,) per role
    probabilities = {role: positive_probability(outputs[role]) for role in ("valid", "test")}  # probabilities: (n,) per role
    run_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(run_directory / "history.csv", index=False)
    save_figure(plot_results(history, probabilities, labels, classes), run_directory / "results.png")

    test = binary_metrics(probabilities["test"], labels["test"])
    row = {"branches": "+".join(branches), "parameters": parameters, "best epoch": best["epoch"],
           "valid AUROC": roc_auc_score(labels["valid"], probabilities["valid"])}
    return outputs, row | {f"test {name}": value for name, value in test.items()}  # outputs: (n, 2) per role


def calibrate_run(
    run_name: str, outputs: Mapping[str, Tensor], splits: Mapping[str, Split], classes: Sequence[object],
) -> dict[str, float]:
    """Print the run's temperature scaling and conformal sets, save its calibration figure, and return its calibration
    columns of the runs table."""
    # outputs: (n, 2) per held-out role
    labels = {role: splits[role].labels for role in outputs}  # labels: (n,) per role
    temperature = fit_temperature(outputs["valid"], torch.from_numpy(labels["valid"]))
    scaled = {
        name: {role: positive_probability(outputs[role], scale) for role in outputs}
        for name, scale in (("raw", 1.0), ("temperature-scaled", temperature))
    }  # scaled: (n,) per role per scaling
    scores = pd.DataFrame.from_dict({
        name: {
            "test ECE": expected_calibration_error(probability["test"], labels["test"]),
            "test NLL": log_loss(labels["test"], probability["test"], labels=[0, 1]),
            "test Brier": brier_score_loss(labels["test"], probability["test"]),
            "valid NLL": log_loss(labels["valid"], probability["valid"], labels=[0, 1]),
        }
        for name, probability in scaled.items()
    }, orient="index")
    conformal = conformal_table(scaled["temperature-scaled"]["calibration"], labels["calibration"],
                                scaled["temperature-scaled"]["test"], labels["test"], classes)
    show_table(scores, f"Temperature {temperature:.3f}, fitted on valid", {column: "{:.4f}" for column in scores})
    show_table(conformal, f"Conformal sets on test, thresholds per label from {len(labels['calibration']):,} calibration proteins",
               {column: "{:.1%}" for column in conformal} | {"mean set size": "{:.2f}"})
    save_figure(plot_calibration({name: probability["test"] for name, probability in scaled.items()}, labels["test"]),
                Path(CACHE_ROOT) / "runs" / run_name / "calibration.png")
    return {
        "temperature": temperature, "test ECE, scaled": scores.loc["temperature-scaled", "test ECE"],
        "coverage at 90%": conformal.loc["90%", "coverage"], "both labels at 90%": conformal.loc["90%", "both labels"],
    }


def main() -> None:
    # Every Hub request, the dataset's included, signs in with the HF_TOKEN in `~/Dropbox/.secrets.env`, which replaces
    # any token already in the environment: private datasets need it, and the resolver quota is higher with it.
    if "HF_TOKEN" not in load_secrets_env(names=("HF_TOKEN",), override=True):
        raise RuntimeError(f"No HF_TOKEN in {find_secrets_file() or '~/Dropbox/.secrets.env, which was not found'}.")
    token = os.environ["HF_TOKEN"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    section("Probe playground")
    report("run", RUN_NAME)
    report("hub account", f"{HfApi(token=token).whoami()['name']}, signed in from .secrets.env")
    report("dataset", DATASET + (f" ({DATASET_CONFIG})" if DATASET_CONFIG else ""))
    report("store", f"{STORE_REPOSITORY} {STORE}, reading {' + '.join(BRANCHES)}")
    report("controls", ", ".join(CONTROLS) or "none")
    report("device", torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    report("outputs", Path(CACHE_ROOT) / "runs" / RUN_NAME)

    section("Dataset")
    splits, classes = load_splits()
    report("labels", f"{classes[0]!r} is 0, {classes[1]!r} is 1 (positive)")
    show_table(describe_splits(splits, classes), "Rows per split", {"positive share": "{:.1%}"})

    section("Embeddings")
    store = HubStore(STORE_REPOSITORY, STORE, STORE_REVISION, token)
    report("store", f"{store.manifest['label']} at {store.revision[:12]}")
    features = store_features(store, splits)
    store.client.close()
    report("widths", ", ".join(f"{branch} {features['train'][branch].shape[1]:,}" for branch in BRANCHES))

    section(f"Training on {len(splits['train'].labels):,} proteins, {math.ceil(len(splits['train'].labels) / BATCH_SIZE)} batches per epoch")
    outputs, row = train_arm(RUN_NAME, features, splits, classes, device)
    rows = {RUN_NAME: row | {"store": store.manifest["label"]}}
    for kind in CONTROLS:
        _, row = train_arm(f"{RUN_NAME}/{kind}", control_features(kind, splits), splits, classes, device)
        rows[f"{RUN_NAME}/{kind}"] = row | {"store": kind}

    section("Model and controls")
    comparison = pd.DataFrame.from_dict(rows, orient="index")[COMPARISON_COLUMNS]
    show_table(comparison.rename(columns=lambda column: column.removeprefix("test ")), "Test metrics at threshold 0.5, with valid AUROC")

    section(f"Calibration of {RUN_NAME}")
    rows[RUN_NAME] |= calibrate_run(RUN_NAME, outputs, splits, classes)

    section("Runs")
    finished = datetime.now().isoformat(timespec="seconds")
    show_runs(record_runs({name: {"dataset": DATASET.split("/")[-1], **row, "finished": finished} for name, row in rows.items()}))


if __name__ == "__main__":
    main()
