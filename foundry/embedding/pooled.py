"""Seed a base's pooled store from its token store, and check the copy against the source.

A pooled stream has the descriptor, and so the key, of the token store's stream of the same name (`canonical.pooled_manifest`),
so seeding copies stream directories and writes a new manifest: no row is embedded or rewritten. `check_seed` then accounts for
every list row in every pooled stream and compares a sample of rows with the source, value for value.

A profile store's pooled store grows instead by grafting (`graft_segments`): the token run's segments of the pooled
streams that hold none of its proteins are copied in whole, so the pooled run embeds only the proteins no token run
covered. The pooled run's golden check holds its digests to the token run's, and the store's exact `verify` recomputes
grafted rows in its sample like any other. The same graft merges a shard of the pooled store embedded on another machine.

Symbols: n proteins in the list; c SAE codebook (16384); d hidden width of the base; w stored columns of a stream.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
import torch

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from tqdm.auto import tqdm

from .canonical import (
    MANIFEST_FILE, CanonicalBase, canonical_base, fetch_record, manifest_features, open_canonical_view, pooled_manifest,
    pooled_profile_manifest, read_manifest, sequence_inventory, write_manifest,
)
from .contracts import SequenceKey
from .readers import FeatureView, ReadBudget


SEEDED_FROM_FILE = "seeded_from.json"
GRAFTS_FILE = "grafts.json"  # every graft into a profile store's pooled store, in order
GRAFT_MATCHING_FIELDS = ("schema", "base", "profile", "hardware_class", "model_state_sha256")  # the two stores' manifests agree
COPY_IGNORE = shutil.ignore_patterns("index.sqlite*", ".locks", "*.writing", "*.lock")
PROGRESS_SECONDS = 15


def copy_stream(source: Path, target: Path) -> int:
    """Copy one stream directory without its index and lock files, and return the bytes copied."""
    shutil.copytree(source, target, ignore=COPY_IGNORE)
    return sum(path.stat().st_size for path in target.rglob("*") if path.is_file())


def seed_pooled(base: CanonicalBase, source: Path, target: Path) -> dict[str, Any]:
    """Copy the pooled streams of the token store ``source`` into a new pooled store at ``target``.

    The copy is checked against the commit markers it carries (`FeatureStore.open` with deep verification hashes every
    part and sidecar, and builds the index), the manifest is the one a pooled run derives, and the record says where
    every stream came from. Refuses a ``target`` that already holds a store.
    """
    from fastplms.features import FeatureStore

    if (target / MANIFEST_FILE).exists():
        raise FileExistsError(f"{target} already holds a store; seeding never overwrites.")
    token_manifest = read_manifest(source)
    manifest = pooled_manifest(base, token_manifest)
    features = manifest_features(source, token_manifest, base.pooled_streams)  # held to the manifest's descriptor digests
    streams: dict[str, Any] = {}
    for name, feature in tqdm(features.items(), desc="seed streams", unit="stream", file=sys.stderr):
        copied = copy_stream(source / feature.key, target / feature.key)
        store = FeatureStore.open(target, feature, deep_verify=True)
        streams[name] = {"key": feature.key, "rows": len(store), "segments": len(store.segments()), "bytes": copied}
    write_manifest(target, manifest)
    record = {
        "schema": "canonical_pooled_seed_v1", "finished_utc": datetime.now(UTC).isoformat(), "base": base.id,
        "source_store": str(source), "source_manifest_sha256": hashlib.sha256((source / MANIFEST_FILE).read_bytes()).hexdigest(),
        "source_hub_repo": token_manifest["hub_repo"], "streams": streams,
        "note": "stream directories copied without re-embedding; keys, descriptors and rows are the token store's",
    }
    (target / SEEDED_FROM_FILE).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def copy_segment(source: Path, target: Path) -> int:
    """Copy one committed segment: its parts and sidecars first, each held to the digest its commit marker states, then the
    marker, so a copy cut short is an uncommitted segment that nothing reads. Returns the bytes copied."""
    from foundry.digests import sha256_file

    from .hub import COMMIT_MARKER, PART_NAME

    marker = json.loads((source / COMMIT_MARKER).read_text(encoding="utf-8"))
    expected: dict[str, str] = {}
    for part in marker["parts"]:
        expected[PART_NAME.format(int(part["part"]))] = part["sha256"]
        sidecar = part.get("row_metadata")
        if sidecar:
            expected[sidecar["file"]] = sidecar["sha256"]
    held = sorted(path.name for path in source.iterdir() if path.name != COMMIT_MARKER)
    if held != sorted(expected):
        raise ValueError(f"{source} holds {held}, but its commit marker names {sorted(expected)}.")
    target.mkdir()  # a directory already there is another segment's, or a copy cut short: `sweep` first
    copied = 0
    for name in held:
        shutil.copyfile(source / name, target / name)
        if sha256_file(target / name) != expected[name]:
            raise ValueError(f"{target / name} differs from the digest its commit marker states.")
        copied += (target / name).stat().st_size
    shutil.copyfile(source / COMMIT_MARKER, target / COMMIT_MARKER)  # the commit, last
    return copied + (target / COMMIT_MARKER).stat().st_size


def graft_segments(source: Path, target: Path, *, token: bool = False, streams: Sequence[str] | None = None) -> dict[str, Any]:
    """Copy into the pooled store ``target`` the segments of ``source``'s pooled streams that hold none of its proteins:
    ``source`` is the base's token store, whose run covered some proteins already, or a shard of the pooled store embedded
    on another machine.

    Both are profile stores of one base, profile, class and weights, and each pooled stream has the same record in both
    manifests; a pooled store that does not exist yet starts with the manifest a pooled run would write
    (`canonical.pooled_profile_manifest`). A segment is taken whole (`copy_segment`) when ``target`` holds none of its
    proteins, skipped when it holds them all (a pooled run embedded them), and refused when it holds some. Every pooled
    stream must gain the same proteins. Returns the record, which is appended to the store's `grafts.json`.

    ``token=True`` requires two token stores and grafts every token stream. A missing target takes the donor's manifest.
    This permits a verified missing-protein supplement without changing existing rows or the default pooled operation.
    A fetched partial token target may instead name its retained ``streams``. Only those streams gain rows; the fetch
    record and manifest stay unchanged, and missing parts of an already-held donor segment must be fetched separately.
    """
    from fastplms.features import FeatureStore

    from .hub import COMMIT_MARKER, SEGMENTS

    donor = read_manifest(source)
    fetched = fetch_record(target)
    chosen = None if streams is None else list(streams)
    if chosen is not None:
        if not token or fetched is None:
            raise ValueError("Selected-stream grafts require an existing fetched partial token target.")
        if not chosen or len(chosen) != len(set(chosen)) or not set(chosen) <= set(fetched["streams"]) & set(donor["streams"]):
            raise ValueError("Selected graft streams must be distinct, nonempty and retained by both stores.")
    if not (target / MANIFEST_FILE).exists() and donor.get("store_kind") == "token":
        write_manifest(target, donor if token else pooled_profile_manifest(canonical_base(donor["base"]), donor))
    pooled = read_manifest(target)
    if token and (donor.get("store_kind") != "token" or pooled.get("store_kind") != "token"):
        raise ValueError("A token graft requires token source and target stores.")
    if not token and (donor.get("store_kind") not in ("token", "pooled") or pooled.get("store_kind") != "pooled"):
        raise ValueError(f"A graft goes from a token or pooled store to a pooled store, not {donor.get('store_kind')} to "
                         f"{pooled.get('store_kind')}.")
    if token and set(donor["streams"]) != set(pooled["streams"]):
        raise ValueError("A token graft requires identical stream sets.")
    differing = [field for field in GRAFT_MATCHING_FIELDS if donor.get(field) != pooled.get(field)]
    differing += [name for name, record in pooled["streams"].items() if donor["streams"].get(name) != record]
    if differing:
        raise ValueError(f"{source} and {target} differ in {differing}; a graft needs one function and one stream record.")
    # The donor's descriptors, which the stream records equal; opening the target writes them, or checks its own.
    features = manifest_features(source, donor, list(pooled["streams"]) if chosen is None else chosen)
    partial = token and fetched is not None
    planned: dict[str, list[str]] = {}
    gained: dict[str, set[str]] = {}
    for name, feature in features.items():
        store = FeatureStore.open(target, feature, deep_verify=False, partial=partial)
        held = {path.name for path in (target / feature.key / SEGMENTS).iterdir()}
        planned[name], gained[name] = [], set()
        for marker in sorted((source / feature.key / SEGMENTS).glob(f"*/{COMMIT_MARKER}")):
            segment = marker.parent.name
            if segment in held and not partial:
                continue  # grafted before
            digests = {digest for part in json.loads(marker.read_text(encoding="utf-8"))["parts"]
                       for digest in part["digests"]}
            present = store.present_digests(sorted(digests))
            if len(present) == len(digests):
                continue  # a pooled run embedded these proteins
            if segment in held:
                raise ValueError(f"{name}: partial target already holds incomplete donor segment {segment}; fetch its missing parts first.")
            if present:
                raise ValueError(f"{name}: segment {segment} holds {len(present)} of its {len(digests)} proteins in "
                                 f"{target} already; a graft takes whole segments only.")
            if digests & gained[name]:
                raise ValueError(f"{name}: segment {segment} repeats a protein of another grafted segment.")
            planned[name].append(segment)
            gained[name] |= digests
    if len({frozenset(proteins) for proteins in gained.values()}) > 1:
        raise ValueError(f"The streams would gain different proteins: {({name: len(v) for name, v in gained.items()})}.")
    copied = 0
    for name, feature in features.items():
        for segment in tqdm(planned[name], desc=f"graft {name}", unit="segment", mininterval=PROGRESS_SECONDS, file=sys.stderr):
            copied += copy_segment(source / feature.key / SEGMENTS / segment, target / feature.key / SEGMENTS / segment)
        FeatureStore.open(target, feature, deep_verify=False, partial=partial)  # indexes the new segments from their markers
    proteins = len(next(iter(gained.values()), set()))
    record = {
        "schema": "canonical_token_graft_v1" if token else "canonical_pooled_graft_v1",
        "finished_utc": datetime.now(UTC).isoformat(), "source_store": str(source),
        "source_manifest_sha256": hashlib.sha256((source / MANIFEST_FILE).read_bytes()).hexdigest(), "proteins": proteins,
        "bytes": copied, "segments": planned,
    }
    path = target / GRAFTS_FILE
    grafts = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps([*grafts, record], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return record


def rows_equal(left: Any, right: Any) -> bool:
    """Whether two rows of a store hold exactly the same values: tensors, sparse rows (indices and values), top-k rows."""
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    pairs = [(left.indices, right.indices), (left.values, right.values)]
    if getattr(left, "positions", None) is not None:
        pairs.append((left.positions, right.positions))
    return all(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b) for a, b in pairs)


COMPARE_BUDGET = ReadBudget(max_rows=16, max_padded_residues=1 << 20, max_bytes=1 << 30)  # 8 long proteins of a 6B base fit
COMPARE_GROUP = 8  # sequences read from each view at a time


def compare_views(before: FeatureView, after: FeatureView, names: Sequence[str], texts: Sequence[str]) -> int:
    """Read ``texts`` from both views, a few at a time, and raise unless every stream's rows equal value for value.

    Returns the number of rows compared (streams times texts).
    """
    compared = 0
    for name in names:
        for start in range(0, len(texts), COMPARE_GROUP):
            group = list(texts[start:start + COMPARE_GROUP])
            old, new = before.rows(name, group), after.rows(name, group)  # b rows: (w,) or (l + 2, w) tensors, sparse or top-k rows
            different = [text for text, left, right in zip(group, old, new, strict=True) if not rows_equal(left, right)]
            if different:
                raise AssertionError(f"{name}: {len(different)} of {len(group)} compared rows differ from the source, such as {different[0][:20]}")
            compared += len(group)
    return compared


def check_seed(
    base: CanonicalBase, source: Path, target: Path, texts: Sequence[str], keys: Sequence[str], *, sample: int, seed: int,
    inventory: Callable[[Sequence[str]], Mapping[str, SequenceKey]] = sequence_inventory,
) -> dict[str, Any]:
    """Every list row is in every pooled stream of ``target``, and ``sample`` rows equal the source's value for value.

    ``texts`` and ``keys`` are the list's normalized sequences and their SHA-256 keys (`job.read_inventory`), and ``inventory``
    makes a view's `SequenceKey`s for the sampled texts (a protein's by default; a ligand or annotation base passes its own). A row that is
    missing, or differs in dtype, shape or any value, raises. Returns the counts for the receipt.
    """
    from fastplms.features import FeatureStore

    manifest = read_manifest(target)
    features = manifest_features(target, manifest, base.pooled_streams)
    accounting: dict[str, int] = {}
    for name, feature in features.items():
        store = FeatureStore.open(target, feature, deep_verify=False)
        absent = set(keys) - store.present_digests(keys)  # the list's keys that the stream lacks
        if absent:
            raise AssertionError(f"{name}: {len(absent)} of {len(keys)} list rows are not in the pooled store")
        accounting[name] = len(keys)
    drawn = random.Random(seed).sample(range(len(texts)), min(sample, len(texts)))
    chosen = [texts[index] for index in drawn]
    keyed = inventory(chosen)
    with open_canonical_view(source, keyed, streams=base.pooled_streams, budget=COMPARE_BUDGET) as before, \
            open_canonical_view(target, keyed, streams=base.pooled_streams, budget=COMPARE_BUDGET) as after:
        compared = compare_views(before, after, base.pooled_streams, chosen)
    return {"accounting": accounting, "sampled_sequences": len(chosen), "rows_compared": compared, "seed": seed}


__all__ = [
    "COMPARE_BUDGET", "GRAFTS_FILE", "SEEDED_FROM_FILE", "check_seed", "compare_views", "copy_segment", "copy_stream",
    "graft_segments", "rows_equal", "seed_pooled",
]
