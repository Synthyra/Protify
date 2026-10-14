"""The two embedding repositories: index a profile store, catalog what each repository holds, fetch single proteins.

A profile store (`canonical_embeddings_v3`) lives at `<model>/<kind>/` of `Synthyra/esmc_embeddings` (public models) or
`Synthyra/private_embeddings` (private ones), one canonical place per model and kind whatever the profile, which the
store records; a local copy keeps the same path under a mirror root, so a push uploads the mirror with the store's folder
as its only pattern. `relocate_store` moves a store from an older folder. Beside its streams a store carries two indexes:

- `index/parts.parquet`: one row per published file (manifest, descriptors, commit markers, parts, row sidecars): its path,
  kind, stream, segment, part, rows, bytes, SHA-256, and the least and greatest row key a part holds.
- `index/rows.parquet`: one row per protein: its key, and per stream the segment, part, row and stored rows that hold it;
  sorted by key in row groups of `ROW_GROUP_ROWS`, so a remote read for a few proteins touches a few row groups.

Both are built from the commit markers alone, which name every part's rows and digests. The repository root holds
`catalog.json` (every store with its profile, streams, rows, bytes and files) and the dataset card made from it; the
catalog commit names its parent revision, so two machines pushing different stores cannot overwrite each other's entry;
one that finds the head moved (any commit moves it, an upload's batches too) reads the catalog again after a pause. The
account's rate limit (3,000 requests in 5 minutes) refuses every request for minutes once a large upload has spent it, so
a push resumes an upload the library gave up on and repeats a refused request (`upload_resuming` and `hub_call` of
`foundry.downloads.hub`).

`read_rows` reads a few proteins by HTTP range requests: the rows index's row groups holding their keys, each touched part's
safetensors header and row offsets, and the rows' own bytes, so it downloads kilobytes to megabytes, not whole parts.
`fetch_rows` instead copies whole parts: it reads the rows index remotely, downloads the manifest, the descriptors and only
the commit markers, parts and sidecars that hold the requested proteins, and returns a local store root that
`open_canonical_view` opens. The copy's `fetch.json` names its repository and revision and marks it partial, so its absent
parts are expected, a later fetch into the same directory adds to it, and it is never indexed or pushed as a store.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import sys
import time

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from tqdm.auto import tqdm

from .canonical import (
    FETCH_RECORD, MANIFEST_FILE, MANIFEST_SCHEMA_V3, canonical_base, fetch_record, manifest_kind, read_manifest,
)


INDEX_DIRECTORY = "index"
PARTS_INDEX = f"{INDEX_DIRECTORY}/parts.parquet"
ROWS_INDEX = f"{INDEX_DIRECTORY}/rows.parquet"
CATALOG_FILE = "catalog.json"
CARD_FILE = "README.md"
CATALOG_SCHEMA = "embedding_catalog_v1"
COVERAGE_FILE = "coverage.json"  # the exact gate's record of the list a store holds every protein of (`write_coverage`)
COVERAGE_SCHEMA = "embedding_store_coverage_v1"
COVERAGE_LIST_FIELDS = frozenset({"repository", "revision", "file", "sha256", "keep_flags", "proteins"})
RELOCATIONS_FILE = "relocations.json"  # each move of a store to a new folder (`relocate_store`)
GROWN_FILE = "grown.json"  # a grown store's record of the published store whose metadata it holds (`grow_metadata`)
GROWN_SCHEMA = "embedding_store_grown_v1"
FETCH_SCHEMA = "embedding_fetch_v1"
ROW_GROUP_ROWS = 4096
FEATURE_FILE = "feature.json"
SEGMENTS = "segments"
COMMIT_MARKER = "run.json"
PART_NAME = "part-{:05d}.safetensors"
SIDECAR_NAME = "part-{:05d}.rows.json.gz"  # a part's row identities, which every v3 descriptor requires
KIND_MANIFEST, KIND_DESCRIPTOR, KIND_MARKER, KIND_PART, KIND_SIDECAR = "manifest", "descriptor", "commit_marker", "part", "rows_sidecar"
PROGRESS_SECONDS = 15
HUB_IGNORE = ("**/index.sqlite*", "**/.locks/**", "**/*.writing", "**/*.lock", "**/.cache/**", ".cache/**", ".locks/**",
              f"**/{GROWN_FILE}")
HUB_FOLDER_FILES = 10_000  # the Hub's hard limit of files in one folder
HUB_REPOSITORY_FILES = 100_000  # and in one repository
MOVED_PARENT_STATUS = frozenset({409, 412})  # the Hub's answers to a commit whose parent is no longer the head
CATALOG_ATTEMPTS = 12  # catalog commits tried before giving up; between them about four minutes of pauses in all
CATALOG_PAUSE_SECONDS = (2.0, 30.0)  # the first pause after a moved head and the longest, each before jitter
RANGE_WORKERS = 16  # parts `read_rows` reads at once
RANGE_GAP_BYTES = 1 << 16  # spans of one part closer than this are read in one request
RANGE_TIMEOUT_SECONDS = 120
HEADER_PROBE_BYTES = 1 << 16  # the first read of a part, which holds its whole safetensors header in practice


@dataclass(frozen=True, slots=True)
class StoreFile:
    """One published file of a store, as `index/parts.parquet` lists it."""

    path: str  # relative to the store root
    kind: str
    stream: str | None
    segment: str | None
    part: int | None
    rows: int
    bytes: int
    sha256: str | None  # as the commit marker states it; None for the manifest, descriptors and markers themselves
    first_key: str | None
    last_key: str | None


@dataclass(frozen=True, slots=True)
class RowAddress:
    """Where one protein's rows of one stream are: its segment, part, row within the part, and how many rows it stores."""

    segment: str
    part: int
    row: int
    rows: int


@dataclass(frozen=True, slots=True)
class StoreLayout:
    """Every published file of a store, and every protein's address in each stream."""

    manifest: Mapping[str, Any]
    files: tuple[StoreFile, ...]
    addresses: Mapping[str, Mapping[str, RowAddress]]  # protein key -> stream -> address


def store_layout(root: Path) -> StoreLayout:
    """Read a profile store's layout from its manifest and commit markers, without opening a part.

    A grown store (`grow_metadata`) holds every marker but only its new parts; a part or sidecar it lacks takes its size
    from the published store's parts index that `grown.json` records, and must carry the digest its marker states.
    """
    root = Path(root)
    manifest = read_manifest(root)
    if manifest["schema"] != MANIFEST_SCHEMA_V3:
        raise ValueError(f"{root} is a v2 store; only profile stores ({MANIFEST_SCHEMA_V3}) are indexed and pushed.")
    fetched = fetch_record(root)
    if fetched is not None:
        raise ValueError(f"{root} is a partial copy of {fetched['repository']}@{fetched['revision']}; only a whole store is "
                         "indexed and pushed.")
    grown = read_grown(root)
    published = {} if grown is None else grown["published_files"]

    def size(relative: str, sha256: str) -> int:
        path = root / relative
        if path.is_file():
            return path.stat().st_size
        record = published.get(relative)
        if record is None:
            raise FileNotFoundError(f"{root} lacks {relative}, and no published store it was grown from holds it.")
        if record["sha256"] != sha256:
            raise ValueError(f"{relative} is {record['sha256']} in the published store and {sha256} in its marker.")
        return int(record["bytes"])

    files = [StoreFile(MANIFEST_FILE, KIND_MANIFEST, None, None, None, 0, (root / MANIFEST_FILE).stat().st_size, None, None, None)]
    addresses: dict[str, dict[str, RowAddress]] = {}
    for stream, record in sorted(manifest["streams"].items()):
        key = record["key"]
        descriptor = root / key / FEATURE_FILE
        files.append(StoreFile(f"{key}/{FEATURE_FILE}", KIND_DESCRIPTOR, stream, None, None, 0, descriptor.stat().st_size,
                               None, None, None))
        for marker in sorted((root / key / SEGMENTS).glob(f"*/{COMMIT_MARKER}")):
            segment = marker.parent.name
            payload = json.loads(marker.read_text(encoding="utf-8"))
            files.append(StoreFile(f"{key}/{SEGMENTS}/{segment}/{COMMIT_MARKER}", KIND_MARKER, stream, segment, None,
                                   int(payload["rows"]), marker.stat().st_size, None, None, None))
            for part in payload["parts"]:
                number, digests = int(part["part"]), list(part["digests"])
                relative = f"{key}/{SEGMENTS}/{segment}/{PART_NAME.format(number)}"
                files.append(StoreFile(relative, KIND_PART, stream, segment, number, len(digests),
                                       size(relative, part["sha256"]), part["sha256"], min(digests), max(digests)))
                sidecar = part.get("row_metadata")
                if sidecar:
                    path = f"{key}/{SEGMENTS}/{segment}/{sidecar['file']}"
                    files.append(StoreFile(path, KIND_SIDECAR, stream, segment, number, len(digests),
                                           size(path, sidecar["sha256"]), sidecar["sha256"], None, None))
                for row, (digest, rows) in enumerate(zip(digests, part["residues"], strict=True)):
                    held = addresses.setdefault(digest, {})
                    if stream in held:
                        raise ValueError(f"{root}: protein {digest} is stored twice in {stream}, in segments "
                                         f"{held[stream].segment} and {segment}.")
                    held[stream] = RowAddress(segment, number, row, int(rows))
    return StoreLayout(manifest, tuple(files), addresses)


def write_store_index(root: Path) -> dict[str, Any]:
    """Write `index/parts.parquet` and `index/rows.parquet` into a profile store; returns the store's catalog entry."""
    import pyarrow as pa

    root = Path(root)
    layout = store_layout(root)
    streams = sorted(layout.manifest["streams"])
    (root / INDEX_DIRECTORY).mkdir(exist_ok=True)
    parts = pa.table({name: [getattr(item, name) for item in layout.files] for name in StoreFile.__dataclass_fields__})
    _write_parquet(parts, root / PARTS_INDEX, row_group_rows=ROW_GROUP_ROWS)
    keys = sorted(layout.addresses)
    columns: dict[str, list[Any]] = {"sha256": keys}
    for stream in streams:
        located = [layout.addresses[key].get(stream) for key in keys]
        columns[f"{stream}__segment"] = [None if item is None else item.segment for item in located]
        columns[f"{stream}__part"] = [None if item is None else item.part for item in located]
        columns[f"{stream}__row"] = [None if item is None else item.row for item in located]
        columns[f"{stream}__rows"] = [None if item is None else item.rows for item in located]
    schema = pa.schema([("sha256", pa.string()), *[
        field for stream in streams for field in (
            (f"{stream}__segment", pa.string()), (f"{stream}__part", pa.int32()), (f"{stream}__row", pa.int32()),
            (f"{stream}__rows", pa.int32()),
        )
    ]])
    _write_parquet(pa.table(columns, schema=schema), root / ROWS_INDEX, row_group_rows=ROW_GROUP_ROWS)
    return catalog_entry(root, layout)


def store_sha256(root: Path) -> str:
    """The digest of a store's committed content: its manifest, descriptors and commit markers, which name every part's
    and sidecar's SHA-256, so a part appended, removed or changed changes it. Indexes and records are not content."""
    root = Path(root)
    manifest = read_manifest(root)
    paths = [MANIFEST_FILE]
    for _, record in sorted(manifest["streams"].items()):
        key = record["key"]
        paths.append(f"{key}/{FEATURE_FILE}")
        paths.extend(marker.relative_to(root).as_posix() for marker in sorted((root / key / SEGMENTS).glob(f"*/{COMMIT_MARKER}")))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode("utf-8") + b"\0" + (root / path).read_bytes() + b"\0")
    return digest.hexdigest()


def write_coverage(root: Path, *, sequences: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    """Record that an exact gate found every protein of a sequence list in a profile store, and the store it checked.

    ``sequences`` names the list: its repository, revision, file and SHA-256, the flags whose rows the store keeps (none
    for every row) and the proteins they select. ``gate`` is what the gate measured. `check_coverage` holds a push to a
    store whose `store_sha256` is still the one recorded here.
    """
    missing = sorted(COVERAGE_LIST_FIELDS - set(sequences))
    if missing:
        raise ValueError(f"A coverage record names the list's {missing}.")
    record = {"schema": COVERAGE_SCHEMA, "sequences": dict(sequences), "store_sha256": store_sha256(root),
              "verified_utc": datetime.now(UTC).isoformat(timespec="seconds"), "gate": dict(gate)}
    _write_json(Path(root) / COVERAGE_FILE, record)
    return record


def read_coverage(root: Path) -> dict[str, Any] | None:
    """A store's coverage record, or None when no exact gate has written one."""
    path = Path(root) / COVERAGE_FILE
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != COVERAGE_SCHEMA:
        raise ValueError(f"{path} is not a coverage record of schema {COVERAGE_SCHEMA}.")
    return record


def check_coverage(root: Path) -> dict[str, Any]:
    """The store's coverage record; refused when there is none, or when the store changed after the gate wrote it."""
    record = read_coverage(root)
    if record is None:
        raise ValueError(f"{root} has no {COVERAGE_FILE}: a profile store is published only after an exact verify against "
                         "its sequence list (`job verify --list-revision ...`), which writes it.")
    if record["store_sha256"] != store_sha256(root):
        raise ValueError(f"{root} changed after the exact verify that wrote {COVERAGE_FILE}; verify it again, then push.")
    return record


def read_grown(root: Path) -> dict[str, Any] | None:
    """A grown store's record of the published store it was grown from, or None for a store that holds all its parts."""
    path = Path(root) / GROWN_FILE
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != GROWN_SCHEMA:
        raise ValueError(f"{path} is not a grown-store record of schema {GROWN_SCHEMA}.")
    return record


def committed_markers(root: Path, key: str) -> dict[str, dict[str, Any]]:
    """The commit markers of one stream directory, by segment."""
    return {marker.parent.name: json.loads(marker.read_text(encoding="utf-8"))
            for marker in sorted((Path(root) / key / SEGMENTS).glob(f"*/{COMMIT_MARKER}"))}


def grow_metadata(root: Path, *, repository: str, revision: str, source: Path,
                  download: Callable[[str], Path] | None = None, workers: int = 8) -> dict[str, Any]:
    """Build a grown store at ``root``: a published store's metadata with the segments of ``source`` linked in.

    The published store at ``revision`` contributes its manifest, descriptors and every commit marker, not its parts, and
    those must hash to the `store_sha256` its coverage record states, so they are the store an exact gate passed.
    ``source`` is a complete local store of the same manifest whose proteins the published store lacks, in every stream;
    its committed segments are hard-linked in, so their parts are the ones a gate reads there. `grown.json` records the
    published store and the size and digest of every part and sidecar it holds, which `store_layout` reads for the parts
    this store lacks. The grown store is gated (`job grow`) and pushed like any other; a push uploads only what is here,
    so the published parts stay as they are. ``download`` replaces the Hub download of one repository path (a test).
    """
    import os

    import pyarrow.parquet as parquet

    from concurrent.futures import ThreadPoolExecutor

    from foundry.downloads.hub import hub_call

    root, source = Path(root), Path(source)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"{root} is not empty; a store is grown into a new folder.")
    target = read_manifest(source)
    folder = target["path_in_repo"]
    local_dir = mirror_root(root, target)
    fetch = download or _hub_download(repository, revision, local_dir)
    get = (lambda path: hub_call(fetch, f"{folder}/{path}")) if download is None else (lambda path: fetch(f"{folder}/{path}"))
    manifest = json.loads(get(MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest != target:
        differing = sorted(name for name in set(manifest) | set(target) if manifest.get(name) != target.get(name))
        raise ValueError(f"{source} is not a store of {repository}/{folder}: their manifests differ in {differing}.")
    coverage = json.loads(get(COVERAGE_FILE).read_text(encoding="utf-8"))
    published = parquet.read_table(get(PARTS_INDEX)).to_pylist()
    metadata = [row["path"] for row in published if row["kind"] in {KIND_DESCRIPTOR, KIND_MARKER}]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in tqdm(pool.map(get, metadata), total=len(metadata), desc="markers", unit="file",
                      mininterval=PROGRESS_SECONDS, file=sys.stderr):
            pass
    (root / PARTS_INDEX).unlink()  # the published index is recorded below; the push writes the grown store's own
    (root / COVERAGE_FILE).unlink()  # the grown store is gated anew, so it starts with no coverage record of its own
    if store_sha256(root) != coverage["store_sha256"]:
        raise ValueError(f"The metadata of {repository}/{folder}@{revision} does not hash to the store its coverage record "
                         "names; it is not the store an exact gate passed.")
    if store_layout(source).manifest != target:  # a complete store: every part of its markers is present
        raise ValueError(f"{source} changed while it was read.")
    linked: dict[str, list[str]] = {}
    for stream, record in sorted(target["streams"].items()):
        key = record["key"]
        held = {digest for marker in committed_markers(root, key).values() for part in marker["parts"]
                for digest in part["digests"]}
        added = committed_markers(source, key)
        repeated = sorted(segment for segment in added if (root / key / SEGMENTS / segment).exists())
        if repeated:
            raise ValueError(f"{stream}: segments {repeated[:3]} of {source} are already in the published store.")
        overlap = sorted(held & {digest for marker in added.values() for part in marker["parts"] for digest in part["digests"]})
        if overlap:
            raise ValueError(f"{stream}: {len(overlap)} proteins of {source} are already in the published store, such as "
                             f"{overlap[:3]}; grow only with proteins it lacks.")
        for segment in added:
            directory = root / key / SEGMENTS / segment
            directory.mkdir(parents=True)
            for path in sorted((source / key / SEGMENTS / segment).iterdir()):
                if path.is_file():
                    os.link(path, directory / path.name)
        linked[stream] = sorted(added)
    record = {
        "schema": GROWN_SCHEMA, "repository": repository, "revision": revision, "folder": folder,
        "published_coverage": coverage, "source": str(source), "source_store_sha256": store_sha256(source),
        "linked_segments": linked,
        "published_files": {row["path"]: {"bytes": int(row["bytes"]), "sha256": row["sha256"]}
                            for row in published if row["kind"] in {KIND_PART, KIND_SIDECAR}},
    }
    _write_json(root / GROWN_FILE, record)
    return record


def relocate_store(root: Path) -> tuple[Path, dict[str, Any]]:
    """Move a profile store to the folder its base names now, `<mirror>/<model>/<kind>`, and point its manifest there.

    One rename inside the mirror moves it, so no file is rewritten but the manifest, where only `path_in_repo` and
    `volume_directory` change: the repository and volume must already be the base's. A coverage record that was current
    before the move is rewritten for the new digest with the move noted, keeping its list and gate, since the rows are the
    ones the exact gate passed; one that was not current stays as it was, so the store still needs an exact verify before
    a push. Each move is appended to `relocations.json`. Returns the new root and the move's record.
    """
    root = Path(root).resolve()
    manifest = read_manifest(root)
    if manifest.get("schema") != MANIFEST_SCHEMA_V3:
        raise ValueError(f"{root} is not a profile store; only profile stores have a folder to move to.")
    if fetch_record(root) is not None:
        raise ValueError(f"{root} is a partial copy of a published store; only a whole store moves.")
    hub_repo, folder, volume = canonical_base(manifest["base"]).profile_destinations(manifest_kind(manifest))
    if (manifest["hub_repo"], manifest["volume"]) != (hub_repo, volume):
        raise ValueError(f"{root} is recorded for {manifest['hub_repo']} and {manifest['volume']}, but its base names {hub_repo} "
                         f"and {volume}; only a store's folder moves.")
    previous = manifest["path_in_repo"]
    if previous == folder and manifest["volume_directory"] == folder:
        raise ValueError(f"{root} is already at {folder}.")
    target = mirror_root(root, manifest) / folder
    if target.exists():
        raise ValueError(f"{target} exists; a store moves only into a free folder.")
    coverage = read_coverage(root)
    before = store_sha256(root)
    carried = coverage is not None and coverage["store_sha256"] == before
    target.parent.mkdir(parents=True, exist_ok=True)
    root.rename(target)
    _write_json(target / MANIFEST_FILE, {**manifest, "path_in_repo": folder, "volume_directory": folder})
    move = {"from": previous, "to": folder, "utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "store_sha256_before": before, "store_sha256_after": store_sha256(target), "coverage_carried": carried}
    if carried:
        _write_json(target / COVERAGE_FILE, {**coverage, "store_sha256": move["store_sha256_after"], "relocated": move})
    earlier = json.loads((target / RELOCATIONS_FILE).read_text(encoding="utf-8")) if (target / RELOCATIONS_FILE).is_file() else []
    _write_json(target / RELOCATIONS_FILE, [*earlier, move])
    return target, move


def _write_json(path: Path, value: Any) -> None:
    """Write ``value`` as sorted, indented JSON, atomically: the form `write_manifest` gives a manifest."""
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_parquet(table: Any, path: Path, *, row_group_rows: int) -> None:
    """Write ``table`` with statistics per row group, atomically."""
    import pyarrow.parquet as parquet

    temporary = path.with_name(path.name + ".writing")
    parquet.write_table(table, temporary, row_group_size=row_group_rows, compression="zstd", write_statistics=True)
    temporary.replace(path)


def catalog_entry(root: Path, layout: StoreLayout) -> dict[str, Any]:
    """What `catalog.json` says about one store: its model, kind, profile, class, the sequence list it covers (None
    before an exact gate records one), streams, proteins, bytes and files."""
    manifest = layout.manifest
    coverage = read_coverage(root)
    records = [name for name in (PARTS_INDEX, ROWS_INDEX, COVERAGE_FILE) if (Path(root) / name).exists()]
    streams: dict[str, Any] = {}
    for stream, record in sorted(manifest["streams"].items()):
        stored = [item for item in layout.files if item.stream == stream]
        streams[stream] = {
            "key": record["key"], "layout": record["layout"], "width": record["width"], "dtype": record["dtype"],
            "proteins": sum(1 for addresses in layout.addresses.values() if stream in addresses),
            "parts": sum(item.kind == KIND_PART for item in stored), "bytes": sum(item.bytes for item in stored),
        }
    return {
        "path": manifest["path_in_repo"], "model": manifest["store_model"], "base": manifest["base"], "label": manifest["label"],
        "kind": manifest["store_kind"], "visibility": manifest["visibility"], "profile": manifest["profile"],
        "hardware_class": manifest["hardware_class"], "model_state_sha256": manifest["model_state_sha256"],
        "random_init": manifest.get("random_init"), "sequences": None if coverage is None else coverage["sequences"],
        "proteins": len(layout.addresses), "streams": streams, "files": len(layout.files) + len(records),
        "bytes": sum(item.bytes for item in layout.files) + sum((Path(root) / name).stat().st_size for name in records),
        "index": {"parts": f"{manifest['path_in_repo']}/{PARTS_INDEX}", "rows": f"{manifest['path_in_repo']}/{ROWS_INDEX}"},
    }


def merge_catalog(current: Mapping[str, Any] | None, repository: str, entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The catalog with ``entries`` replacing the stores of the same path; every other store is kept as it was."""
    stores = dict((current or {}).get("stores", {}))
    if current is not None and current.get("repository") != repository:
        raise ValueError(f"The catalog names {current.get('repository')}, not {repository}.")
    for entry in entries:
        stores[entry["path"]] = dict(entry)
    return {"schema": CATALOG_SCHEMA, "repository": repository, "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "stores": dict(sorted(stores.items()))}


def control_note(catalog: Mapping[str, Any]) -> str:
    """The card's word on each random-weight control among the stores: what it is, and the saved draw its keys name."""
    drawn = {entry["model"]: entry["random_init"] for entry in catalog["stores"].values() if entry.get("random_init")}
    return "".join(
        f"\n`{model}` is a control, not a protein model: the {record['architecture']} architecture with random weights, "
        f"drawn once from seed {record['seed']} and saved at `{model}/weights/model.safetensors` (SHA-256 "
        f"`{record.get('weights_sha256', 'unrecorded')}`), which every key of its stores names.\n"
        for model, record in sorted(drawn.items()))


def repository_card(catalog: Mapping[str, Any]) -> str:
    """The dataset card of one embedding repository: what each store holds and how to fetch one protein or one shard."""
    repository = catalog["repository"]
    rows = "\n".join(
        f"| `{path}` | {entry['label']} | {entry['kind']} | {entry['proteins']:,} | {entry['bytes'] / 2**30:,.1f} | "
        f"{', '.join(f'`{name}`' for name in entry['streams'])} | `{entry['profile']['id']}` | {sequences_cell(entry)} |"
        for path, entry in catalog["stores"].items()
    )
    public = repository.endswith("esmc_embeddings")
    return f"""---
pretty_name: {'ESMC embeddings' if public else 'Private embeddings'}
tags: [protein, embeddings{', esmc' if public else ''}]
viewer: false
---
# {repository}

{'Embeddings of the public ESMC models' if public else 'Embeddings of the private models (SELFIES-TED, the Annotation Transformers, Atlas, Dyad)'},
computed under an embedding profile: a pinned container, pinned numerics, one fixed batch shape per protein and a pinned
SAE readout, so every row is a function of its protein alone and bitwise reproducible on the hardware class that made it
(Hopper: GH200, H100, H200). Each store is a FastPLMs feature store at `<model>/<kind>/`, the one canonical place for that
model and kind; the `profile` column names the profile it was computed under, and a store computed under a new profile
replaces it there. `catalog.json` lists them all. Each store holds every protein of one revision of its sequence list: a
`pooled` store every row, a `token` store
the rows flagged `full`, which need per-token rows. The `sequences` column names that revision (and the flag), and the
store's `coverage.json` records the exact check that passed it: every protein's rows present, and a sample of proteins
recomputed in new batches, equal bit for bit.

| store | model | kind | proteins | GiB | streams | profile | sequences |
|---|---|---|---:|---:|---|---|---|
{rows}
{control_note(catalog)}
| stream | per protein |
|---|---|
| `layer_hidden` | (l + 2, d) bfloat16: the hidden state at the SAE's layer |
| `final_hidden` | (l + 2, d) bfloat16: the last layer, after the final norm |
| `structural` | (l + 2, 256): the structure fusion, a learned mixture of every layer projected to 256 |
| `sae_codes` | (l + 2, 64) indices and values: each token's top 64 SAE codes, stored sparse |
| `sae_max` | sparse (c,) float32, c the SAE codebook: each code's maximum over the tokens |
| `final_mean_var`, `layer_mean_var`, `structural_mean_var` | (2w,) float32: the mean then the variance over the l + 2 tokens |

A row is keyed by the SHA-256 of the uppercased, whitespace-stripped sequence. Per-token streams hold l + 2 rows per protein
(row 0 CLS, rows 1..l residues, row l + 1 EOS, cropped at 2046 residues); pooled streams hold one vector.

Each store carries `index/rows.parquet` (per protein: the segment, part and row of each stream, sorted by key) and
`index/parts.parquet` (per file: path, rows, bytes, SHA-256). A few proteins are read by HTTP range requests, so only their
bytes are downloaded however large the store is; a whole store is a folder download:

```python
from foundry.embedding.hub import read_rows

rows = read_rows("{repository}", "esmc_600/pooled", ["MKTAYIAKQR"], revision="<commit>",
                 streams=["sae_max", "final_mean_var"])
vector = rows["MKTAYIAKQR"]["final_mean_var"]  # a (2d,) float32 tensor

from huggingface_hub import snapshot_download
snapshot_download("{repository}", repo_type="dataset", allow_patterns=["esmc_600/pooled/**"])
```

Without the workspace: filter `index/rows.parquet` on `sha256`; each part is a safetensors file, whose first 8 bytes give
its JSON header's length and whose header gives each tensor's byte span. A dense stream's row r is row r of `values`; a
per-token or sparse stream's row r spans `offsets[r]:offsets[r + 1]` (`indptr` for `sae_max`) of `values` (and `indices`).
"""


def sequences_cell(entry: Mapping[str, Any]) -> str:
    """The card's link to the list revision a store covers, with the flag it keeps; a store from before coverage
    records says so."""
    sequences = entry.get("sequences")
    if sequences is None:
        return "not recorded"
    flags = f" `{', '.join(sequences['keep_flags'])}`" if sequences["keep_flags"] else ""
    return (f"[`{sequences['revision'][:8]}`](https://huggingface.co/datasets/{sequences['repository']}/tree/"
            f"{sequences['revision']}){flags}")


def required_files(layout_rows: Mapping[str, Mapping[str, RowAddress]], manifest: Mapping[str, Any],
                   streams: Sequence[str]) -> list[str]:
    """The store files a reader of these proteins needs: manifest, descriptors, and the markers, parts and sidecars holding them."""
    needed = {MANIFEST_FILE}
    for stream in streams:
        key = manifest["streams"][stream]["key"]
        needed.add(f"{key}/{FEATURE_FILE}")
        for addresses in layout_rows.values():
            address = addresses.get(stream)
            if address is None:
                continue
            directory = f"{key}/{SEGMENTS}/{address.segment}"
            needed.update({f"{directory}/{COMMIT_MARKER}", f"{directory}/{PART_NAME.format(address.part)}",
                           f"{directory}/{SIDECAR_NAME.format(address.part)}"})
    return sorted(needed)


def locate(repository: str, folder: str, keys: Sequence[str], *, revision: str) -> dict[str, dict[str, RowAddress]]:
    """The addresses of ``keys`` in the store at ``folder``, read from its remote rows index, row groups by key statistics."""
    import pyarrow.parquet as parquet

    from huggingface_hub import HfFileSystem

    table = parquet.read_table(f"datasets/{repository}@{revision}/{folder}/{ROWS_INDEX}", filesystem=HfFileSystem(),
                               filters=[("sha256", "in", list(keys))])
    return addresses_from_rows(table.to_pylist())


def addresses_from_rows(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, RowAddress]]:
    """Rows of `index/rows.parquet` as protein key -> stream -> address."""
    located: dict[str, dict[str, RowAddress]] = {}
    for record in records:
        streams = {name.removesuffix("__segment") for name in record if name.endswith("__segment")}
        located[record["sha256"]] = {
            stream: RowAddress(record[f"{stream}__segment"], record[f"{stream}__part"], record[f"{stream}__row"],
                               record[f"{stream}__rows"])
            for stream in sorted(streams) if record[f"{stream}__segment"] is not None
        }
    return located


def fetch_rows(repository: str, folder: str, sequences: Sequence[str], *, revision: str, local_dir: str | Path,
               streams: Sequence[str] | None = None, download: Callable[[str], Path] | None = None,
               workers: int = 8) -> Path:
    """A local partial copy of the store at ``folder`` holding ``sequences``; open it with `open_canonical_view`.

    Downloads the manifest, the descriptors, and only the commit markers, parts and row sidecars of the parts that hold
    the requested proteins, ``workers`` files at a time. ``revision`` is a commit hash, so a reader pins what it read.
    ``download`` replaces the Hub download of one repository path (a test reads a local store). A protein the store lacks
    raises, naming it. A second fetch into the same ``local_dir`` adds to the copy and must name the same repository and
    revision.
    """
    from concurrent.futures import ThreadPoolExecutor

    from foundry.serialization.atomic import write_json_atomic

    from .canonical import sequence_inventory

    inventory = sequence_inventory(sequences)
    keys = [inventory[text].sha256 for text in inventory]
    local_dir = Path(local_dir)
    root = local_dir / folder
    previous = fetch_record(root)
    if previous is not None and (previous["repository"], previous["revision"]) != (repository, revision):
        raise ValueError(f"{root} copies {previous['repository']}@{previous['revision']}; fetch {repository}@{revision} into "
                         "another directory, so one copy never mixes two revisions.")
    fetch = download or _hub_download(repository, revision, local_dir)
    manifest = json.loads(fetch(f"{folder}/{MANIFEST_FILE}").read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA_V3:
        raise ValueError(f"{repository}/{folder} is not a profile store; only those carry the rows index a fetch reads.")
    chosen = list(manifest["streams"]) if streams is None else list(streams)
    located = locate(repository, folder, keys, revision=revision) if download is None else _local_addresses(fetch, folder, keys)
    missing = [text for text in inventory if inventory[text].sha256 not in located]
    if missing:
        raise KeyError(f"{repository}/{folder} holds no rows for {len(missing)} requested proteins, such as {missing[:3]}.")
    # Recorded before the first part arrives, so a copy an interrupted fetch left is still opened as partial.
    fetched_streams = sorted(set(chosen) | set(previous["streams"] if previous else ()))
    write_json_atomic(root / FETCH_RECORD, {
        "schema": FETCH_SCHEMA, "repository": repository, "revision": revision, "folder": folder, "streams": fetched_streams,
        "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    })
    paths = required_files(located, manifest, chosen)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Iterating the results re-raises the first failed download.
        for _ in tqdm(pool.map(lambda path: fetch(f"{folder}/{path}"), paths), total=len(paths), desc="fetch", unit="file",
                      mininterval=PROGRESS_SECONDS, file=sys.stderr):
            pass
    return root


def _hub_download(repository: str, revision: str, local_dir: Path) -> Callable[[str], Path]:
    from huggingface_hub import hf_hub_download

    def download(path: str) -> Path:
        return Path(hf_hub_download(repository, path, repo_type="dataset", revision=revision, local_dir=local_dir))

    return download


def _local_addresses(fetch: Callable[[str], Path], folder: str, keys: Sequence[str]) -> dict[str, dict[str, RowAddress]]:
    import pyarrow.parquet as parquet

    table = parquet.read_table(fetch(f"{folder}/{ROWS_INDEX}"), filters=[("sha256", "in", list(keys))])
    return addresses_from_rows(table.to_pylist())


@dataclass(frozen=True, slots=True)
class PartTensor:
    """One tensor of a remote safetensors part: its dtype name, shape and byte span from the start of the file."""

    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


def read_rows(repository: str, folder: str, sequences: Sequence[str], *, revision: str,
              streams: Sequence[str] | None = None, workers: int = RANGE_WORKERS,
              download: Callable[[str], Path] | None = None,
              read_range: Callable[[str, int, int], bytes] | None = None,
              addresses: Mapping[str, Mapping[str, RowAddress]] | None = None) -> dict[str, dict[str, Any]]:
    """The rows of ``sequences`` in the store at ``folder``, read from the repository by byte range.

    Only the requested rows cross the network: the rows index's row groups that hold their keys, each touched part's
    safetensors header and row offsets, and the rows' own bytes, so a few proteins cost a few kilobytes to megabytes
    however large the store is. Returns sequence -> stream -> row, as `open_canonical_view` gives it: a (w,) tensor of
    a pooled stream, a (l + 2, w) tensor of a per-token stream, a `TopKRow` of `sae_codes`, a `SparseRow` of `sae_max`.
    ``revision`` is a commit hash, so a reader pins what it read. ``download`` (one small repository file) and
    ``read_range`` (path, start, end) replace the Hub, as a test reading a local store does.
    ``addresses`` reuses previously selected source locators for bounded repeated reads.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .canonical import sequence_inventory

    inventory = sequence_inventory(sequences)
    fetch = download or _hub_file(repository, revision)
    read = read_range or hub_range_reader(repository, revision)
    manifest = json.loads(fetch(f"{folder}/{MANIFEST_FILE}").read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA_V3:
        raise ValueError(f"{repository}/{folder} is not a profile store; only those carry the rows index a read uses.")
    chosen = list(manifest["streams"]) if streams is None else list(streams)
    unknown = sorted(set(chosen) - set(manifest["streams"]))
    if unknown:
        raise KeyError(f"{repository}/{folder} has no streams {unknown}; it has {sorted(manifest['streams'])}.")
    keys = [inventory[text].sha256 for text in inventory]
    located = addresses
    if located is None:
        located = locate(repository, folder, keys, revision=revision) if download is None else _local_addresses(fetch, folder, keys)
    groups: dict[tuple[str, str], list[tuple[str, RowAddress]]] = {}  # (stream, part path) -> the rows it holds
    for text in inventory:
        sequence_addresses = located.get(inventory[text].sha256, {})
        for stream in chosen:
            if stream not in sequence_addresses:
                raise KeyError(f"{repository}/{folder} holds no {stream} row for {text[:20]}...")
            address = sequence_addresses[stream]
            path = (f"{folder}/{manifest['streams'][stream]['key']}/{SEGMENTS}/{address.segment}/"
                    f"{PART_NAME.format(address.part)}")
            groups.setdefault((stream, path), []).append((text, address))
    rows: dict[str, dict[str, Any]] = {text: {} for text in inventory}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(_part_rows, read, path, manifest["streams"][stream]["layout"], members)
                for (stream, path), members in groups.items()]
        for (stream, _), job in zip(groups, jobs, strict=True):
            for text, row in job.result().items():
                rows[text][stream] = row
    return rows


def _part_rows(read: Callable[[str, int, int], bytes], path: str, layout: str,
               members: Sequence[tuple[str, RowAddress]]) -> dict[str, Any]:
    """The rows ``members`` hold in one part, decoded from the byte spans they occupy."""
    from fastplms.features.layouts import SparseRow, TopKRow

    tensors = _part_header(read, path)
    order = sorted(members, key=lambda member: member[1].row)
    if layout == "dense":  # values (n, w)
        values = tensors["values"]
        width_bytes = (values.end - values.start) // values.shape[0]
        spans = [(values.start + member.row * width_bytes, values.start + (member.row + 1) * width_bytes) for _, member in order]
        return {text: _decode(row_bytes, values.dtype, (values.shape[1],))
                for (text, _), row_bytes in zip(order, _read_spans(read, path, spans), strict=True)}
    pointer = "indptr" if layout == "csr" else "offsets"
    bounds = _decode(_read_spans(read, path, [(tensors[pointer].start, tensors[pointer].end)])[0],
                     tensors[pointer].dtype, tensors[pointer].shape).tolist()  # (n + 1,) row starts
    names = {"ragged": ("values",), "ragged_topk": ("indices", "values"),
             "csr": ("indices", "values", "positions") if "positions" in tensors else ("indices", "values")}[layout]
    decoded: dict[str, dict[str, Any]] = {text: {} for text, _ in order}
    for name in names:
        tensor = tensors[name]  # (sum r_i, w) or (nnz,)
        item_bytes = (tensor.end - tensor.start) // max(1, tensor.shape[0])  # one leading row's bytes
        spans = [(tensor.start + bounds[member.row] * item_bytes, tensor.start + bounds[member.row + 1] * item_bytes)
                 for _, member in order]
        for (text, member), row_bytes in zip(order, _read_spans(read, path, spans), strict=True):
            count = bounds[member.row + 1] - bounds[member.row]
            if layout != "csr" and count != member.rows:
                raise ValueError(f"{path} row {member.row} holds {count} rows; the rows index says {member.rows}.")
            decoded[text][name] = _decode(row_bytes, tensor.dtype, (count, *tensor.shape[1:]))
    if layout == "ragged":
        return {text: parts["values"] for text, parts in decoded.items()}  # (l + 2, w)
    if layout == "ragged_topk":
        return {text: TopKRow(parts["indices"], parts["values"]) for text, parts in decoded.items()}  # (l + 2, k)
    return {text: SparseRow(parts["indices"], parts["values"], parts.get("positions")) for text, parts in decoded.items()}


def _part_header(read: Callable[[str, int, int], bytes], path: str) -> dict[str, PartTensor]:
    """A safetensors part's tensors: an 8-byte little-endian header length, then the JSON header naming each span."""
    head = read(path, 0, HEADER_PROBE_BYTES)
    length = int.from_bytes(head[:8], "little")
    if len(head) < 8 + length:
        head += read(path, len(head), 8 + length)
    header = json.loads(head[8:8 + length].decode("utf-8"))
    return {name: PartTensor(spec["dtype"], tuple(spec["shape"]), 8 + length + spec["data_offsets"][0],
                             8 + length + spec["data_offsets"][1])
            for name, spec in header.items() if name != "__metadata__"}


def _read_spans(read: Callable[[str, int, int], bytes], path: str, spans: Sequence[tuple[int, int]]) -> list[bytes]:
    """The bytes of each (start, end) span of one file, with neighbours closer than `RANGE_GAP_BYTES` read together."""
    order = sorted(range(len(spans)), key=lambda index: spans[index])
    merged: list[list[int]] = []  # [start, end, first index into order, last index]
    for position, index in enumerate(order):
        start, end = spans[index]
        if merged and start - merged[-1][1] <= RANGE_GAP_BYTES:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][3] = position
        else:
            merged.append([start, end, position, position])
    found: list[bytes] = [b""] * len(spans)
    for start, end, first, last in merged:
        chunk = read(path, start, end) if end > start else b""
        if len(chunk) != end - start:
            raise ValueError(f"{path} returned {len(chunk)} bytes for the range {start}-{end}.")
        for index in order[first:last + 1]:
            found[index] = chunk[spans[index][0] - start:spans[index][1] - start]
    return found


def _decode(span_bytes: bytes, dtype: str, shape: tuple[int, ...]) -> Any:
    """Bytes of a safetensors span as a tensor of ``shape``."""
    import torch

    names = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64,
             "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "U16": torch.uint16, "BOOL": torch.bool}
    if not span_bytes:
        return torch.empty(shape, dtype=names[dtype])
    return torch.frombuffer(bytearray(span_bytes), dtype=names[dtype]).reshape(shape)


def _hub_file(repository: str, revision: str) -> Callable[[str], Path]:
    """One small repository file, through the Hub's local cache."""
    from huggingface_hub import hf_hub_download

    def download(path: str) -> Path:
        return Path(hf_hub_download(repository, path, repo_type="dataset", revision=revision))

    return download


def hub_range_reader(repository: str, revision: str) -> Callable[[str, int, int], bytes]:
    """Bytes [start, end) of one repository file at ``revision``, by an HTTP range request with the Hub's retries."""
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers, http_stream_backoff

    def read(path: str, start: int, end: int) -> bytes:
        url = hf_hub_url(repository, path, repo_type="dataset", revision=revision)
        with http_stream_backoff("GET", url, headers={**build_hf_headers(), "Range": f"bytes={start}-{end - 1}"},
                                 follow_redirects=True, timeout=RANGE_TIMEOUT_SECONDS) as response:
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(f"{repository}/{path} answered {response.status_code} to a range request.")
            bounds = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", response.headers.get("Content-Range", ""))
            if bounds is None or int(bounds[1]) != start or not start <= int(bounds[2]) < end:
                raise RuntimeError(f"{repository}/{path} returned an inconsistent Content-Range.")
            expected = int(bounds[2]) - start + 1
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) != expected:
                raise RuntimeError(f"{repository}/{path} returned an inconsistent Content-Length.")
            payload = bytearray()
            for chunk in response.iter_bytes(chunk_size=1 << 20):
                if len(payload) + len(chunk) > expected:
                    raise RuntimeError(f"{repository}/{path} exceeded its declared range bytes.")
                payload.extend(chunk)
            if len(payload) != expected:
                raise RuntimeError(f"{repository}/{path} returned an incomplete byte range.")
            return bytes(payload)

    return read


def copy_from(source_root: Path, local_dir: Path) -> Callable[[str], Path]:
    """A ``download`` that copies one repository path from a local mirror root, for tests and offline reads."""

    def download(path: str) -> Path:
        target = local_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / path, target)
        return target

    return download


def mirror_root(root: Path, manifest: Mapping[str, Any]) -> Path:
    """The local mirror a profile store sits in: the store root less its `<model>/<kind>` folder."""
    folder = Path(manifest["path_in_repo"])
    if Path(root).resolve().parts[-len(folder.parts):] != folder.parts:
        raise ValueError(f"A profile store lives at <mirror>/{folder.as_posix()}; {root} does not end with it, so a push "
                         "would put it at the wrong path. Move it under a mirror root.")
    return Path(root).resolve().parents[len(folder.parts) - 1]


def push_store(root: Path, *, visibility: str, files: Sequence[Path], staging: str | None = None) -> dict[str, Any]:
    """Index a profile store, upload its folder into its repository, check the copy, and commit the catalog and card.

    ``files`` are the store's published files (the job's `store_files`). The repository's visibility must be the store's:
    a private model never reaches the public repository. ``staging`` uploads into that repository instead, a fresh copy
    that is renamed to the store's repository once complete, so its catalog and card already name the store's
    repository. Returns the catalog entry, with the revision of the commit that recorded it under `revision`.
    """
    from huggingface_hub import HfApi

    from foundry.datasets.uploads import require_hub_token
    from foundry.downloads.hub import hub_call, hub_copy_differences, upload_resuming

    manifest = read_manifest(root)
    if manifest.get("schema") != MANIFEST_SCHEMA_V3:
        raise ValueError(f"{root} is not a profile store.")
    if visibility != manifest["visibility"]:
        raise ValueError(f"{manifest['label']} is a {manifest['visibility']} model; it goes to {manifest['hub_repo']} as "
                         f"{manifest['visibility']}, not {visibility}.")
    check_coverage(root)
    mirror = mirror_root(root, manifest)
    entry = write_store_index(root)
    require_hub_token()
    api = HfApi()
    named, folder = manifest["hub_repo"], manifest["path_in_repo"]
    repository = staging or named
    hub_call(api.create_repo, repository, repo_type="dataset", private=visibility == "private", exist_ok=True)
    upload_resuming(api, repo_id=repository, folder_path=str(mirror), repo_type="dataset", allow_patterns=[f"{folder}/**"],
                    ignore_patterns=list(HUB_IGNORE))
    published = [*files, Path(root) / PARTS_INDEX, Path(root) / ROWS_INDEX]
    differences = hub_call(hub_copy_differences, api, repository, mirror, published)
    if differences:
        raise RuntimeError(f"{repository} is not a copy of {root} after the upload: {differences[:5]}")
    return {**entry, "revision": commit_catalog(api, repository, entry, folder=folder, named=named)}


def commit_catalog(api: Any, repository: str, entry: Mapping[str, Any], *, folder: str, named: str | None = None,
                   sleep: Callable[[float], None] = time.sleep) -> str:
    """Merge one store's catalog entry into the repository's catalog and commit it with the card; returns the commit.

    The commit names the head it read as its parent, so a catalog that another push committed meanwhile is never
    overwritten: a moved head is read again and merged. Any commit moves the head, and another machine's upload into
    the same repository commits every few seconds while it sends small files, so the attempts pause between them,
    doubling and jittered, until one lands in a gap. A request refused for the account's rate limit is sent again after
    a pause (`hub_call`), the commit only on that refusal. ``named`` is the repository the catalog and card name when
    ``repository`` is a staging copy that will be renamed to it.
    """
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.errors import HfHubHTTPError

    from foundry.downloads.hub import hub_call, rate_limited

    pause = CATALOG_PAUSE_SECONDS[0]
    for attempt in range(CATALOG_ATTEMPTS):
        if attempt:
            sleep(pause * random.uniform(0.5, 1.0))
            pause = min(2 * pause, CATALOG_PAUSE_SECONDS[1])
        head = hub_call(api.repo_info, repository, repo_type="dataset", sleep=sleep).sha
        catalog = merge_catalog(hub_call(_remote_catalog, api, repository, head, sleep=sleep), named or repository, [entry])
        operations = [CommitOperationAdd(CATALOG_FILE, (json.dumps(catalog, indent=2, sort_keys=True) + "\n").encode()),
                      CommitOperationAdd(CARD_FILE, repository_card(catalog).encode())]
        try:
            return hub_call(api.create_commit, repository, operations, repo_type="dataset", parent_commit=head,
                            commit_message=f"Catalog {folder}: {entry['proteins']:,} proteins", sleep=sleep,
                            when=rate_limited).oid
        except HfHubHTTPError as error:
            if error.response is None or error.response.status_code not in MOVED_PARENT_STATUS:
                raise
    raise RuntimeError(f"The catalog of {repository} moved under {CATALOG_ATTEMPTS} commits; push the store again, which "
                       "uploads nothing new and commits the catalog.")


def check_file_counts(files: Sequence[Path], root: Path) -> dict[str, int]:
    """Refuse a store over the Hub's limits: 10,000 files in one folder, 100,000 in one repository. Returns the counts."""
    folders: dict[str, int] = {}
    for path in files:
        folder = Path(path).relative_to(root).parent.as_posix()
        folders[folder] = folders.get(folder, 0) + 1
    crowded = {folder: count for folder, count in folders.items() if count > HUB_FOLDER_FILES}
    if crowded or len(files) > HUB_REPOSITORY_FILES:
        raise ValueError(f"The store exceeds the Hub's file limits ({len(files):,} files; folders over "
                         f"{HUB_FOLDER_FILES:,}: {crowded}). Raise --part-gib or --segment-gib.")
    return {"files": len(files), "largest_folder": max(folders.values(), default=0)}


def _remote_catalog(api: Any, repository: str, revision: str) -> dict[str, Any] | None:
    from huggingface_hub.errors import EntryNotFoundError

    try:
        path = api.hf_hub_download(repository, CATALOG_FILE, repo_type="dataset", revision=revision)
    except EntryNotFoundError:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "CATALOG_FILE", "CATALOG_SCHEMA", "PARTS_INDEX", "ROWS_INDEX", "RowAddress", "StoreFile", "StoreLayout",
    "addresses_from_rows", "catalog_entry", "check_file_counts", "copy_from", "fetch_rows", "locate", "merge_catalog",
    "mirror_root", "push_store", "repository_card", "required_files", "store_layout", "write_store_index",
]
