"""Embed a parquet of proteins into a canonical store: resumable, shardable, piloted, then pushed.

Commands, all with `--base 300|600|6b|random|selfies|at` (see `canonical.CANONICAL_BASES`). The two last are not
proteins: SELFIES-TED-LARGE over canonical SMILES and the Annotation Transformer over annotation term ids (`encoders`), with
the same commands, stores and keys. Their input column is `--sequence-column` (default `smiles` or `terms`), the optional key
column `--digest-column` (default `smiles_sha256` or `entry_sha256`) must equal the SHA-256 of the canonical text when present,
and `--prepare-workers` canonicalizes over that many processes. `repack` is protein only.

- `embed`  reads `(seq_sha256, sequence)` rows, embeds every stream the store lacks with CLS and EOS kept (a protein of
  l residues is l + 2 rows in each per-token stream, shape (l+2, w)), and commits segments as it goes. Run it again
  with the same arguments to resume; `--shard i/n` takes the rows whose SHA-256 is i modulo n, so shards run on separate
  GPUs or machines against the same inventory and never overlap.
- `pilot`  times chunks of a random sample into a scratch store through `foundry.pilot`, prints the projected
  run and the first bottleneck, and writes the receipt. It is due before every full run. The sample is sized to
  hold every timed step and warmup step unless `--sample` says otherwise.
- `verify` accounts for every input row in every stream, reads a sample back through `open_canonical_view`, and
  recomputes it independently.
- `sweep`  deletes the uncommitted segment directories a killed run left in each stream; `push` refuses a store
  that has any and prints this command.
- `push`   uploads the committed segments to the base's public Hub dataset repo and its Modal volume.
- `seed`   (with `--pooled`) copies a token store's `sae_max` and `final_mean_var` streams into the base's pooled store
  without embedding, then accounts for every list row and compares a sample of values with the source.
- `graft`  (with `--pooled`, profile stores) copies into the pooled store the segments of the token store `--source`'s
  pooled streams that hold none of its proteins, so the next pooled `embed` computes only proteins no token run covered;
  the exact `verify` then samples grafted rows like any other.
- `repack` writes a token store holding only the rows of the list flagged by `--keep-flag`, every stream, from the parts of
  `--source`; `push --prune-hub` then publishes it as a new revision of the existing repo.
- `grow`   (profile stores) adds a complete local store of proteins the published store lacks (`--source`, embedded under
  the same manifest) to the store published at `--hub-revision`, without its parts: `--store` receives the published
  manifest, descriptors and markers, held to the store digest of its coverage record, and the source's segments as
  hard links (`hub.grow_metadata`). The gate then accounts for every protein of the list in the markers and recomputes
  a sample of the added proteins bit for bit; `push` uploads the new segments with the rewritten indexes and catalog.
- `compare` measures how far another store of a base (`--source`, such as the v2 store a profile store replaces) lies from
  the profile store `--store`, stream by stream, on a bucket-stratified sample of the `--input` proteins both hold, with
  the statistics `qualify` compares panels by. It loads no model, uses no GPU and changes neither store.

`--pooled` selects the pooled store of the base (`canonical.STORE_POOLED`): only `sae_max` and `final_mean_var` (the
control: `final_mean_var`) are computed and written, and nothing per token, so batches can be large (`--batch-fill`).

**Profile stores** (`--profile embedding_profile_v1`, `canonical_embeddings_v3`): before the base loads, the job attests
the container's library closure, sets and reads back the numerics flags, and names the device's hardware class; before
any row is written it holds a few full batches to the repository's golden file bit for bit (`embed`, `pilot`: one batch
in each of three buckets, expected under a second on ESMC-300), or runs the whole canary panel against the class's
golden digests (`verify`, `qualify`, `golden`). Only the profile's reference class embeds, pilots, verifies and records
goldens; every other class only qualifies (`numerics.check_reference_class`). Every batch runs at the profile's fixed
geometry, so no batch flag applies (`--max-sequences`, `--max-tokens`, `--fixed-batch-size`,
`--batch-fill` and `--window` are refused). `verify` is exact: a bucket-stratified sample recomputed in shuffled
batches must equal the stored rows bit for bit. Two commands exist only for profiles:

- `attest`  records the closure, numerics and hardware class of this container and device, loading no model.
- `qualify` runs the canary twice and the panel's companion and position invariance on this device, into a scratch
  `--store`; its receipt holds the digests a class records as golden, or that a joining device must equal.
- `golden`  records the golden file of a base (`numerics.golden_path`, or `--golden`) on the reference class after its
  canary: the golden batches' digests and reference rows, kept in the repository. Needs no `--store` or `--input`.

A draft profile pilots and qualifies but writes no store; `embed` needs a qualified profile and an enrolled device, which is
measured again before every commit, and a store made by one hardware class is neither extended nor verified by another.
`push` sends a profile store to its folder of `canonical.PUBLIC_REPO` or `PRIVATE_REPO` with its indexes (`hub`), and only
a store of the reference class under the qualified, unchanged profile (`numerics.check_publishable`).

Batch budget: a new store takes `--batch-fill` of the free device memory for activations (`canonical.choose_batch`).
The budget is part of every stream's key, so an existing token store keeps the one it started with, and `--max-sequences`
or `--max-tokens` name the old static budget of the base. A pooled store keeps the budget its seeded keys state but batches
at the fitted budget, which each segment records (`canonical.key_batch`).

Symbols: b sequences of a batch; l residues of a protein after the N-terminal crop; w stored columns of a stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shlex
import sys
import time

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from tqdm.auto import tqdm

from foundry.gpu_memory import hbm_page_cache_bytes, usable_free_bytes
from .canonical import (
    BATCH_FILL, BATCH_GROWTH_LIMIT, CANONICAL_BASES, ENCODER_BASES, FAMILY_PROTEIN, MANIFEST_FILE, MANIFEST_SCHEMA_V3,
    NORMALIZATION_VERSION, SEQUENCE_LIST_REPO, STORE_POOLED,
    STORE_TOKEN, BatchPolicy, CanonicalBase, CanonicalRuntime, build_runtime, canonical_base, check_keys_match_store,
    check_model_pin, choose_batch, inherit_recorded_evidence, key_batch, manifest_features, manifest_kind, manifest_profile,
    open_canonical_view, profile_manifest, read_manifest, sequence_inventory, stream_manifest, write_manifest,
)
from .contracts import CANONICAL_MAX_RESIDUES, SPECIAL_TOKEN_ROWS, SequenceKey, normalize_sequence
from .encoder_rows import EncoderExecutor, PreparedRow
from .encoders import (
    FAMILY_DEFAULT_COLUMNS, EncoderExtractionContract, embed_encoder_features, encoder_hub_card, family_of, inventory_of,
    prepare_rows, read_family_inputs, row_inventory, write_row_inventory,
)
from .repack import REPACK_FILE


if TYPE_CHECKING:
    from .numerics import EmbeddingProfile, HardwareClass
    from .persistence import FeatureExtractionContract
    from fastplms.embeddings.tokens import ResidueVocabulary
    from foundry.pilot import Pilot


SCHEMA = "canonical_embedding_job_v1"
PROGRESS_SECONDS = 15
HUB_IGNORE = ("**/index.sqlite*", "**/.locks/**", "**/*.writing", "**/*.lock", "**/.cache/**", ".cache/**", ".locks/**")  # root and nested
GIB = 1024**3
RELATIVE_TOLERANCE = 0.02  # bf16 sums in a different order when a batch has other company; see `verify`
VERIFY_SAMPLE = 4096  # sequences `verify` recomputes when --sample is not given
EXACT_VERIFY_SAMPLE = 1024  # proteins a profile store's exact `verify` recomputes, stratified by bucket
COMPARE_SAMPLE = 4096  # proteins whose pooled rows `compare` reads from both stores, stratified by bucket
COMPARE_TOKEN_SHARE = 8  # `compare` reads the per-token rows of one in this many of its sampled proteins


def read_inventory(
    path: Path, *, digest_column: str, sequence_column: str, shard: tuple[int, int], limit: int | None,
    keep_flags: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    """Distinct proteins of the parquet in file order: normalized text and the SHA-256 key the file states for it.

    Each text goes through `normalize_sequence` (ASCII whitespace removed, uppercased), the one rule that makes a
    row key, and the SHA-256 of the result must equal the key the file states, so the file's keys are the census
    and dataset keys. Rows with the same stated key are one protein; a row whose key disagrees is an error, never
    a second row. ``shard`` keeps the rows whose key is `index` modulo `count`. ``keep_flags`` names boolean columns
    of the list: when given, only rows with any of them true are read, as a token store's interaction rows are.
    """
    import pyarrow.compute as compute
    import pyarrow.parquet as parquet

    table = parquet.read_table(path, columns=[digest_column, sequence_column, *keep_flags])
    if keep_flags:
        flagged = table.column(keep_flags[0]).fill_null(False)  # (n,) bool
        for name in keep_flags[1:]:
            flagged = compute.or_(flagged, table.column(name).fill_null(False))
        table = table.filter(flagged)
    digests = table.column(digest_column).to_pylist()
    sequences = table.column(sequence_column).to_pylist()
    index, count = shard
    texts: list[str] = []
    keys: list[str] = []
    seen: set[str] = set()
    wrong = 0
    for digest, raw in tqdm(zip(digests, sequences, strict=True), total=len(digests), desc="verify keys", unit="seq",
                            mininterval=PROGRESS_SECONDS, file=sys.stderr):
        if digest in seen:
            continue
        try:
            text = normalize_sequence(raw)
        except ValueError:
            wrong += 1
            continue
        if hashlib.sha256(text.encode("ascii")).hexdigest() != digest:
            wrong += 1
            continue
        seen.add(digest)
        if int(digest, 16) % count == index:
            texts.append(text)
            keys.append(digest)
            if limit is not None and len(texts) >= limit:
                break
    if wrong:
        raise ValueError(
            f"{wrong} rows of {path} state a key that is not the SHA-256 of their uppercased, whitespace-stripped "
            "sequence; fix the file."
        )
    if not texts:
        raise ValueError(f"{path} has no rows in shard {index}/{count}.")
    return texts, keys


def drop_unrepresentable(
    texts: list[str], keys: list[str], vocabulary: ResidueVocabulary, *, skip: bool,
) -> tuple[list[str], list[str], list[str]]:
    """Proteins with a residue the tokenizer lacks (ESMC has no J) are refused, or set aside with ``skip``."""
    kept_texts, kept_keys, rejected = [], [], []
    for text, key in zip(texts, keys, strict=True):
        try:
            vocabulary.encode(text)
        except ValueError:
            rejected.append(key)
            continue
        kept_texts.append(text)
        kept_keys.append(key)
    if rejected and not skip:
        raise ValueError(
            f"{len(rejected)} proteins hold a residue this tokenizer lacks, such as {rejected[:3]}; "
            "rerun with --skip-unrepresentable to set them aside and record them in the receipt."
        )
    return kept_texts, kept_keys, rejected


def store_kind(options: argparse.Namespace) -> str:
    """"pooled" when the command names `--pooled`, else "token"."""
    return STORE_POOLED if getattr(options, "pooled", False) else STORE_TOKEN


@dataclass(frozen=True, slots=True)
class ProfileRun:
    """What a profile session established before anything ran: the profile, the device's class, and the run's evidence."""

    profile: EmbeddingProfile
    hardware: HardwareClass
    evidence: dict[str, Any]


class Session:
    """What every command needs: the loaded base, its contract, and the proteins of this shard.

    ``run_batch`` is the budget the batches run at. For a token store it is the budget in the stream keys; a pooled store's
    keys keep the seed's budget while its batches may be larger (`canonical.key_batch`). A profile session (``profile``)
    has no budget: its batch shapes are the profile's geometry.
    """

    def __init__(
        self, base: CanonicalBase, runtime: CanonicalRuntime, texts: list[str], keys: list[str],
        rejected: list[str], contract: FeatureExtractionContract, root: Path, run_batch: BatchPolicy | None = None,
        kind: str = STORE_TOKEN, prepared: dict[str, PreparedRow] | None = None, profile: ProfileRun | None = None,
    ) -> None:
        """``prepared`` holds each text's token ids for a non-protein base (`encoders.prepare_rows`); a protein has none."""
        self.base, self.runtime, self.texts, self.keys = base, runtime, texts, keys
        self.rejected, self.contract, self.root, self.kind = rejected, contract, root, kind
        self.prepared = prepared
        self.profile = profile
        self.canary: dict[str, Any] | None = None  # a profile session's canary result, once it has run
        self.golden: dict[str, Any] | None = None  # a profile session's golden check, once it has run
        self.run_batch = None if profile is not None else run_batch or BatchPolicy(
            runtime.capture["batch_size"], runtime.capture["max_tokens_per_batch"], "keys")

    def batch_settings(self) -> dict[str, Any]:
        """The batch shapes a receipt records: the profile's geometry, or the run's budget and the one its keys state."""
        if self.profile is not None:
            assert self.contract.geometry is not None
            return {"geometry": self.contract.geometry.describe()}
        assert self.run_batch is not None
        return {"max_sequences": self.run_batch.max_sequences, "max_tokens": self.run_batch.max_tokens,
                "keys_state_sequences": self.runtime.capture["batch_size"],
                "keys_state_tokens": self.runtime.capture["max_tokens_per_batch"]}


def family_columns(base: CanonicalBase, options: argparse.Namespace) -> tuple[str, str]:
    """The digest and value columns of the input: the flags, else the base's family default (a protein's `seq_sha256`, `sequence`)."""
    default_digest, default_value = ("seq_sha256", "sequence") if base.family == FAMILY_PROTEIN else FAMILY_DEFAULT_COLUMNS[base.family]
    return options.digest_column or default_digest, options.sequence_column or default_value


def open_session(options: argparse.Namespace) -> Session:
    """Load the base, confirm the input, and bind the contract every row of this store is held to."""
    from fastplms.embeddings.tokens import ResidueVocabulary

    from .persistence import FeatureExtractionContract

    if getattr(options, "profile", None):
        return open_profile_session(options)
    base = canonical_base(options.base)
    fixed_batch_size = getattr(options, "fixed_batch_size", None)
    if fixed_batch_size is not None:
        if base.family != FAMILY_PROTEIN or fixed_batch_size < 1:
            raise ValueError("Fixed batches require a protein base and a positive size.")
        if options.max_sequences not in {None, fixed_batch_size}:
            raise ValueError("--max-sequences must match --fixed-batch-size.")
        required_tokens = fixed_batch_size * (CANONICAL_MAX_RESIDUES + 2)
        if options.max_tokens not in {None, required_tokens}:
            raise ValueError("Fixed batches require the full 2048-token budget for every physical row.")
        options.max_sequences, options.max_tokens = fixed_batch_size, required_tokens
    kind = store_kind(options)
    pooled = kind == STORE_POOLED
    shard = parse_shard(options.shard)
    digest_column, value_column = family_columns(base, options)
    inputs = None
    if base.family == FAMILY_PROTEIN:
        texts, keys = read_inventory(
            options.input, digest_column=digest_column, sequence_column=value_column, shard=shard, limit=options.limit,
            keep_flags=getattr(options, "keep_flag", ()),
        )
    else:
        inputs = read_family_inputs(
            family_of(base), options.input, digest_column=digest_column, value_column=value_column, shard=shard,
            limit=options.limit, workers=options.prepare_workers, skip=options.skip_unrepresentable,
        )
        texts, keys = [item.text for item in inputs.inputs], inputs.keys
    runtime_options = {}
    if getattr(options, "random_weights", None) is not None:
        runtime_options["random_weights"] = options.random_weights
    runtime = build_runtime(base, options.cache, device=options.device, fetch=options.fetch, window=options.window,
                            pooled=pooled, **runtime_options)
    # The weights are resident now, so the free memory is what a batch can use. The budget enters every stream's key,
    # except that a pooled store's keys keep the budget they were seeded with.
    free_bytes = free_device_bytes(options.device)
    batch = choose_batch(base, options.store, explicit=(options.max_sequences, options.max_tokens),
                         free_bytes=free_bytes, fill=options.batch_fill, pooled=pooled)
    keyed = key_batch(options.store, batch, pooled=pooled)
    runtime = replace(runtime, capture={**runtime.capture, "batch_size": keyed.max_sequences,
                                        "max_tokens_per_batch": keyed.max_tokens})
    if getattr(options, "fixed_batch_size", None) is not None:
        runtime = replace(runtime, capture={**runtime.capture, "fixed_batch_size": options.fixed_batch_size})
    measured = runtime.capture["engine_sha256"]
    runtime = inherit_recorded_evidence(runtime, options.store)  # an existing store keeps the engine and machine it began with
    if runtime.capture["engine_sha256"] != measured:
        print(f"engine {measured[:12]} here; the store's keys keep its recorded {runtime.capture['engine_sha256'][:12]}",
              file=sys.stderr, flush=True)
    free = "" if free_bytes is None else f", {free_bytes / GIB:.1f} GiB device memory free"
    print(f"batch budget ({batch.source}): {batch.max_sequences} sequences, {batch.max_tokens} padded tokens{free}",
          file=sys.stderr, flush=True)
    if keyed != batch:
        print(f"the stream keys keep the store's budget of {keyed.max_sequences} sequences and {keyed.max_tokens} tokens",
              file=sys.stderr, flush=True)
    prepared = None
    if inputs is not None:  # a non-protein base: the family's ids for each canonical input
        binding = runtime.encoder
        assert binding is not None
        family = binding.family  # the family the model was bound with, whose crop its hooks cut at
        read = prepare_rows(family, binding.hooks, inputs.inputs, inputs.keys)
        if read.rejected and not options.skip_unrepresentable:
            raise ValueError(
                f"{len(read.rejected)} {family.id} inputs cannot be tokenized, such as {read.reasons[:3]}; rerun with "
                "--skip-unrepresentable to set them aside and record them in the receipt."
            )
        prepared = read.rows
        texts = [text for text in texts if text in prepared]
        keys = [prepared[text].sha256 for text in texts]
        rejected = [*inputs.rejected, *read.rejected]
        for reason in [*inputs.reasons, *read.reasons][:5]:
            print(f"set aside: {reason}", file=sys.stderr, flush=True)
        inventory = row_inventory(family, prepared)
        contract: FeatureExtractionContract = EncoderExtractionContract(
            runtime.model, runtime.taps, inventory, free_run_batch=pooled, **runtime.capture)
    else:
        vocabulary = ResidueVocabulary(runtime.model.tokenizer)
        texts, keys, rejected = drop_unrepresentable(texts, keys, vocabulary, skip=options.skip_unrepresentable)
        inventory = {text: SequenceKey(key, len(text), NORMALIZATION_VERSION) for text, key in zip(texts, keys, strict=True)}
        contract = FeatureExtractionContract(runtime.model, runtime.taps, inventory, free_run_batch=pooled, **runtime.capture)
    # The weights this machine built, against the store's first build and the digest the operator carries.
    print(f"model weights sha256 {contract.model_state_sha256}", file=sys.stderr, flush=True)
    check_model_pin(options.store, contract.model_state_sha256, expected=options.expect_weights_sha256)
    check_keys_match_store(options.store, contract.stored_features)  # names the changed field before anything is written
    return Session(base, runtime, texts, keys, rejected, contract, options.store, batch, kind, prepared)


PROFILE_ENROLLING_COMMANDS = frozenset({"pilot", "qualify"})  # run on a device not yet enrolled in its class
PROFILE_WRITING_COMMANDS = frozenset({"embed"})  # write a store, so need a qualified profile and an enrolled device


def refuse_batch_flags(options: argparse.Namespace) -> None:
    """A profile run's batch shapes are its geometry: every flag that sized a v2 batch is refused."""
    named = [flag for flag, value, default in (
        ("--max-sequences", options.max_sequences, None), ("--max-tokens", options.max_tokens, None),
        ("--fixed-batch-size", getattr(options, "fixed_batch_size", None), None), ("--batch-fill", options.batch_fill, BATCH_FILL),
        ("--window", options.window, 65536),
    ) if value != default]
    if named:
        raise ValueError(f"A profile run takes its batch shapes from the profile's geometry; drop {named}.")


def open_profile_session(options: argparse.Namespace) -> Session:
    """A profile session: container, numerics and device class are checked before the base loads, then its v3 contract.

    `qualify` and `golden` need no input: their proteins are the canary and golden panels. Every command but `qualify`
    runs on the reference class only. The canary or golden check runs per command (`run_canary`, `run_golden`).
    """
    import torch

    from fastplms.embeddings.tokens import ResidueVocabulary

    from .numerics import (
        STATUS_QUALIFIED, apply_profile, attest_environment, canary_panel, check_environment, check_numerics,
        check_reference_class, describe_device, device_class, golden_panel, load_profile, measure_numerics,
        profile_evidence,
    )
    from .persistence import FeatureExtractionContract

    base = canonical_base(options.base)
    if base.family != FAMILY_PROTEIN:
        raise ValueError(f"{base.label} has no profile geometry yet; its stores follow the v2 path.")
    refuse_batch_flags(options)
    profile = load_profile(options.profile)
    device = torch.device(options.device)
    attestation = attest_environment()
    check_environment(profile, attestation)
    apply_profile(profile, device)
    measured = measure_numerics(device)
    check_numerics(profile, measured, device)
    record = describe_device(device)
    hardware = device_class(profile, record, base=base.id, enrolling=options.command in PROFILE_ENROLLING_COMMANDS)
    if options.command != "qualify":
        check_reference_class(profile, hardware, options.command)
    if options.command in PROFILE_WRITING_COMMANDS and profile.status != STATUS_QUALIFIED:
        raise ValueError(f"{profile.id} is a draft: it pilots and qualifies, and writes no store until it is qualified.")
    if options.store is not None and (options.store / MANIFEST_FILE).is_file():
        stored = read_manifest(options.store).get("hardware_class", {"id": "of no profile (a v2 store)"})["id"]
        if stored != hardware.id:  # its keys would differ anyway; this names why
            raise ValueError(f"{options.store} holds rows of class {stored}; this device is class {hardware.id}, whose rows "
                             "differ, so it neither extends nor verifies that store.")
    print(f"profile {profile.id} ({profile.status}) {profile.sha256[:12]}; class {hardware.id} (tier {hardware.tier}) on "
          f"{record.name}; closure {attestation['closure_sha256'][:12]}", file=sys.stderr, flush=True)
    kind = store_kind(options)
    if options.command == "qualify":
        geometry = profile.geometry(base.id)
        texts = sorted({normalize_sequence(text) for text in canary_panel(geometry, max_residues=CANONICAL_MAX_RESIDUES)})
        keys = [hashlib.sha256(text.encode("ascii")).hexdigest() for text in texts]
    elif options.command == "golden":
        if kind == STORE_POOLED:
            raise ValueError("A golden file is recorded with every token stream, which holds the pooled ones; drop --pooled.")
        panel = golden_panel(profile.geometry(base.id), max_residues=CANONICAL_MAX_RESIDUES)
        texts = sorted({normalize_sequence(text) for batch in panel.values() for text in batch})
        keys = [hashlib.sha256(text.encode("ascii")).hexdigest() for text in texts]
    else:
        digest_column, value_column = family_columns(base, options)
        texts, keys = read_inventory(options.input, digest_column=digest_column, sequence_column=value_column,
                                     shard=parse_shard(options.shard), limit=options.limit, keep_flags=getattr(options, "keep_flag", ()))
    runtime = build_runtime(base, options.cache, device=options.device, fetch=options.fetch, pooled=kind == STORE_POOLED,
                            random_weights=getattr(options, "random_weights", None), profile=profile, hardware=hardware)
    vocabulary = ResidueVocabulary(runtime.model.tokenizer)
    texts, keys, rejected = drop_unrepresentable(texts, keys, vocabulary, skip=options.skip_unrepresentable)
    inventory = {text: SequenceKey(key, len(text), NORMALIZATION_VERSION) for text, key in zip(texts, keys, strict=True)}
    contract = FeatureExtractionContract(runtime.model, runtime.taps, inventory, **runtime.capture)
    print(f"model weights sha256 {contract.model_state_sha256}", file=sys.stderr, flush=True)
    if options.store is not None:
        check_model_pin(options.store, contract.model_state_sha256, expected=options.expect_weights_sha256)
        check_keys_match_store(options.store, contract.stored_features)
    evidence = profile_evidence(profile, attestation, record, hardware, measured)
    runtime = replace(runtime, evidence={**runtime.evidence, "profile_run": evidence})
    return Session(base, runtime, texts, keys, rejected, contract, options.store, None, kind, None,
                   profile=ProfileRun(profile, hardware, evidence))


def run_canary(session: Session) -> dict[str, Any]:
    """Run the canary panel at every bucket's full batch shape and hold its digests to the class's golden ones.

    Runs before any row is written. Returns the digests, the verdict ("matched", or "unrecorded" for a draft profile that
    has none yet) and the seconds it took; a digest that differs raises.
    """
    import torch

    from fastplms.embeddings import TokenTapExecutor
    from fastplms.embeddings.taps import plan_taps
    from fastplms.embeddings.tokens import ResidueVocabulary

    from .numerics import attention_kernel, canary_panel, check_canary, stream_digests

    assert session.profile is not None and session.contract.geometry is not None
    profile, hardware, geometry = session.profile.profile, session.profile.hardware, session.contract.geometry
    runtime = session.runtime
    started = time.monotonic()
    panel = canary_panel(geometry, max_residues=CANONICAL_MAX_RESIDUES)
    executor = TokenTapExecutor(
        runtime.model, plan_taps(list(runtime.taps), int(runtime.model.embedding_tap_state_count)),
        vocabulary=ResidueVocabulary(runtime.model.tokenizer), max_residues=runtime.capture["max_length"], dtype=None,
        geometry=geometry,
    )
    outputs = []
    start = 0
    with torch.inference_mode(), attention_kernel(profile):
        for rows in tqdm(geometry.shapes().values(), desc="canary", unit="bucket", mininterval=PROGRESS_SECONDS, file=sys.stderr):
            group = panel[start:start + rows]
            start += rows
            packed = executor.run_batch(group, [hashlib.sha256(text.encode("ascii")).hexdigest() for text in group])
            packed.wait()
            outputs.append(packed.streams)
    digests = stream_digests(outputs)
    verdict = check_canary(profile, hardware, session.base.id, digests)
    seconds = time.monotonic() - started
    print(f"canary {verdict}: {len(panel)} proteins over {len(outputs)} buckets in {seconds:.1f} s", file=sys.stderr, flush=True)
    return {"digests": digests, "verdict": verdict, "seconds": seconds, "proteins": len(panel),
            "class": hardware.id, "profile": profile.binding()}


def golden_batches(session: Session) -> tuple[dict[str, Any], float]:
    """Run the golden panel, one full batch per golden bucket; returns what `numerics.check_golden` and
    `numerics.write_golden` take (panel digest, stream digests, kept rows) and the seconds it took."""
    import torch

    from fastplms.embeddings import TokenTapExecutor
    from fastplms.embeddings.taps import plan_taps
    from fastplms.embeddings.tokens import ResidueVocabulary

    from .numerics import attention_kernel, golden_panel, golden_panel_sha256, golden_rows, stream_digests

    assert session.profile is not None and session.contract.geometry is not None
    runtime, geometry = session.runtime, session.contract.geometry
    started = time.monotonic()
    panel = golden_panel(geometry, max_residues=CANONICAL_MAX_RESIDUES)
    executor = TokenTapExecutor(
        runtime.model, plan_taps(list(runtime.taps), int(runtime.model.embedding_tap_state_count)),
        vocabulary=ResidueVocabulary(runtime.model.tokenizer), max_residues=runtime.capture["max_length"], dtype=None,
        geometry=geometry,
    )
    batches = {}
    with torch.inference_mode(), attention_kernel(session.profile.profile):
        for columns, group in panel.items():
            packed = executor.run_batch(group, [hashlib.sha256(text.encode("ascii")).hexdigest() for text in group])
            packed.wait()
            batches[columns] = (packed.streams, packed.rows)
    layouts = {name: feature.layout for name, feature in session.contract.stored_features.items()}
    golden = {"panel_sha256": golden_panel_sha256(panel), "digests": stream_digests([streams for streams, _ in batches.values()]),
              "rows": golden_rows(batches, layouts), "buckets": list(panel),
              "proteins": sum(len(group) for group in panel.values())}
    return golden, time.monotonic() - started


def run_golden(session: Session, path: Path | None = None) -> dict[str, Any]:
    """Hold the golden batches to the repository's golden file bit for bit, before any row is written.

    One full batch in each of three buckets (`numerics.golden_panel`), expected under a second on ESMC-300; a stream that
    differs raises (`numerics.check_golden`). Returns the digests, the verdict and the seconds it took.
    """
    from .numerics import check_golden, golden_path

    assert session.profile is not None
    profile, hardware = session.profile.profile, session.profile.hardware
    path = golden_path(profile, session.base.id) if path is None else path
    batches, seconds = golden_batches(session)
    verdict = check_golden(profile, hardware, session.base.id, panel_sha256=batches["panel_sha256"],
                           digests=batches["digests"], rows=batches["rows"], path=path)
    print(f"golden {verdict}: {batches['proteins']} proteins in buckets {batches['buckets']} in {seconds:.1f} s",
          file=sys.stderr, flush=True)
    return {"digests": batches["digests"], "verdict": verdict, "seconds": seconds, "file": str(path),
            "buckets": batches["buckets"], "class": hardware.id, "profile": profile.binding()}


def record_golden(session: Session, options: argparse.Namespace) -> None:
    """Write the golden file of a base on the reference class, after its canary: digests and reference rows.

    Run twice and compared first, so the file holds what this device repeats. It lands in the repository
    (`numerics.golden_path`) unless `--golden` names another path, and never replaces a file.
    """
    from foundry.digests import sha256_file

    from .numerics import golden_path, write_golden

    assert session.profile is not None
    profile, evidence = session.profile.profile, session.profile.evidence
    path = golden_path(profile, session.base.id) if options.golden is None else options.golden
    first, seconds = golden_batches(session)
    second, _ = golden_batches(session)
    if first["digests"] != second["digests"]:
        raise AssertionError(f"The golden batches did not repeat on this device: {first['digests']} then {second['digests']}.")
    write_golden(path, first["rows"], {
        "profile": profile.binding(), "hardware_class": session.profile.hardware.id, "base": session.base.id,
        "panel_sha256": first["panel_sha256"], "digests": first["digests"], "buckets": first["buckets"],
        "device": evidence["device"]["name"], "closure_sha256": evidence["closure_sha256"],
        "model_state_sha256": session.contract.model_state_sha256, "recorded_utc": datetime.now(UTC).isoformat(),
    })
    write_receipt(options.receipt, {
        "schema": SCHEMA, "command": "golden", "finished_utc": datetime.now(UTC).isoformat(), "base": session.base.id,
        "file": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "digests": first["digests"],
        "seconds": seconds, "canary": session.canary, "evidence": session.runtime.evidence,
    })
    print(f"recorded {path} ({path.stat().st_size / 1e6:.2f} MB): {first['proteins']} proteins in {seconds:.1f} s",
          file=sys.stderr, flush=True)


def regroup(session: Session, texts: Sequence[str], *, seed: int) -> list[list[str]]:
    """``texts`` in batches of their buckets, shuffled, so each protein meets other companions at another position."""
    geometry = session.contract.geometry
    assert geometry is not None
    order = list(texts)
    random.Random(seed).shuffle(order)
    buckets: dict[int, list[str]] = {}
    for text in order:
        buckets.setdefault(geometry.columns(min(len(text), CANONICAL_MAX_RESIDUES)), []).append(text)
    batches = []
    for columns, members in buckets.items():
        size = geometry.rows(columns)
        batches += [members[start:start + size] for start in range(0, len(members), size)]
    return batches


def verify_exact(session: Session, options: argparse.Namespace) -> None:
    """A profile store's gate: every row accounted for, and a bucket-stratified sample recomputed bit for bit.

    The sample runs in shuffled batches, so each protein meets new companions at a new position; every stream must
    match the stored rows exactly, values and indices. A store's `verify` then writes its coverage record; `qualify`'s
    scratch store holds the canary panel, which no sequence list names, so its gate records none.
    """
    from fastplms.features import FeatureStore

    from foundry.digests import sha256_file

    from .hub import write_coverage

    assert session.profile is not None and session.contract.geometry is not None
    covers = options.command != "qualify"
    assert not covers or options.list_revision is not None  # `main` refuses a profile verify without one
    root, contract, base, geometry = options.store, session.contract, session.base, session.contract.geometry
    manifest = read_manifest(root)
    if manifest_kind(manifest) != session.kind or set(manifest["streams"]) != set(contract.stored_features):
        raise ValueError(f"{root} holds the {manifest_kind(manifest)} streams {sorted(manifest['streams'])}, but this run "
                         f"checks the {session.kind} streams {sorted(contract.stored_features)}; add or drop --pooled.")
    accounting: dict[str, int] = {}
    for name, feature in contract.stored_features.items():
        store = FeatureStore.open(root, feature, deep_verify=False)
        missing = set(session.keys) - store.present_digests(session.keys)
        if missing:
            raise AssertionError(f"{name}: {len(missing)} of {len(session.keys)} expected proteins are not committed")
        accounting[name] = len(session.keys)
    drawn = stratified_sample(session.texts, geometry, EXACT_VERIFY_SAMPLE if options.sample is None else options.sample,
                              options.seed)
    compared, differing = recompute_exact(session, root, drawn, seed=options.seed + 1)
    keep_flags = sorted(getattr(options, "keep_flag", []))
    coverage = None if not covers else write_coverage(root, sequences={
        "repository": options.list_repository, "revision": options.list_revision, "file": options.input.name,
        "sha256": sha256_file(options.input), "keep_flags": keep_flags, "proteins": len(session.keys),
    }, gate={"accounting": accounting, "recompute_sample": {"sequences": compared, "seed": options.seed, "differing": differing}})
    write_receipt(options.receipt, {
        "schema": SCHEMA, "command": "verify", "gate": "exact", "finished_utc": datetime.now(UTC).isoformat(), "base": base.id,
        "store": str(root), "store_kind": session.kind, "input": str(options.input), "keep_flags": keep_flags,
        "expected_sequences": len(session.keys), "accounting": accounting,
        "manifest_streams": sorted(manifest["streams"]), "recompute_sample": {"sequences": compared, "differing": differing},
        "coverage": coverage, "canary": session.canary, "evidence": session.runtime.evidence,
    })


def recompute_exact(session: Session, root: Path, drawn: Sequence[str], *, seed: int) -> tuple[int, dict[str, int]]:
    """Recompute ``drawn`` in shuffled batches and compare every stream with the rows ``root`` stores, bit for bit.

    Returns the proteins compared and, per stream, how many differ; any difference raises.
    """
    import torch

    from fastplms.embeddings import TokenTapExecutor
    from fastplms.embeddings.taps import plan_taps
    from fastplms.embeddings.tokens import ResidueVocabulary

    from .numerics import attention_kernel

    assert session.profile is not None and session.contract.geometry is not None
    runtime, geometry = session.runtime, session.contract.geometry
    executor = TokenTapExecutor(
        runtime.model, plan_taps(list(runtime.taps), int(runtime.model.embedding_tap_state_count)),
        vocabulary=ResidueVocabulary(runtime.model.tokenizer), max_residues=runtime.capture["max_length"], dtype=None,
        geometry=geometry,
    )
    inventory = sequence_inventory(drawn)
    differing: dict[str, int] = {name: 0 for name in session.contract.stored_features}
    compared = 0
    with open_canonical_view(root, inventory, deep=True, budget=batch_budget(geometry)) as view, torch.inference_mode(), \
            attention_kernel(session.profile.profile):
        for group in tqdm(regroup(session, drawn, seed=seed), desc="exact verify", unit="batch",
                          mininterval=PROGRESS_SECONDS, file=sys.stderr):
            packed = executor.run_batch(group, [inventory[text].sha256 for text in group])
            packed.wait()
            compared += len(group)
            for name in view.names:
                differing[name] += count_unequal(view, name, group, packed)
    if any(differing.values()):
        raise AssertionError(f"Recomputed rows are not bitwise the stored rows (proteins differing per stream): {differing}")
    return compared, differing


def grow_exact(session: Session, options: argparse.Namespace) -> None:
    """A grown store's gate: every list protein in its markers, and a sample of the added proteins recomputed bit for bit.

    The published store's own rows passed the exact gate its coverage record names, and `grow_metadata` held its markers
    to that record's store digest; the added rows are recomputed here from the source store they were linked from. The
    coverage record names the list it now covers and carries the published one under `grown_from`.
    """
    from foundry.digests import sha256_file

    from .hub import read_grown, store_layout, store_sha256, write_coverage

    assert session.profile is not None and session.contract.geometry is not None
    root, source, geometry = options.store, options.source, session.contract.geometry
    grown = read_grown(root)
    if grown is None:
        raise ValueError(f"{root} is not a grown store; `grow` builds one from --source before its gate.")
    if store_sha256(source) != grown["source_store_sha256"]:
        raise ValueError(f"{source} changed after it was linked into {root}.")
    manifest = read_manifest(root)
    if manifest_kind(manifest) != session.kind or set(manifest["streams"]) != set(session.contract.stored_features):
        raise ValueError(f"{root} holds the {manifest_kind(manifest)} streams {sorted(manifest['streams'])}, but this run "
                         f"checks the {session.kind} streams {sorted(session.contract.stored_features)}; add or drop --pooled.")
    layout = store_layout(root)
    accounting: dict[str, int] = {}
    for name in session.contract.stored_features:
        missing = [key for key in session.keys if name not in layout.addresses.get(key, {})]
        if missing:
            raise AssertionError(f"{name}: {len(missing)} of {len(session.keys)} expected proteins are in neither the "
                                 f"published store nor {source}, such as {missing[:3]}")
        accounting[name] = len(session.keys)
    added = set(store_layout(source).addresses)
    texts = [text for text, key in zip(session.texts, session.keys, strict=True) if key in added]
    drawn = stratified_sample(texts, geometry, EXACT_VERIFY_SAMPLE if options.sample is None else options.sample, options.seed)
    compared, differing = recompute_exact(session, source, drawn, seed=options.seed + 1)
    keep_flags = sorted(options.keep_flag)
    published = grown["published_coverage"]
    coverage = write_coverage(root, sequences={
        "repository": options.list_repository, "revision": options.list_revision, "file": options.input.name,
        "sha256": sha256_file(options.input), "keep_flags": keep_flags, "proteins": len(session.keys),
    }, gate={"accounting": accounting,
             "recompute_sample": {"sequences": compared, "seed": options.seed, "differing": differing,
                                  "drawn_from": "added proteins"},
             "grown_from": {"repository": grown["repository"], "revision": grown["revision"], "folder": grown["folder"],
                            "coverage": published},
             "added": {"proteins": len(added), "listed": len(texts), "source_store_sha256": grown["source_store_sha256"],
                       "segments": {stream: len(segments) for stream, segments in grown["linked_segments"].items()}}})
    write_receipt(options.receipt, {
        "schema": SCHEMA, "command": "grow", "gate": "exact_added", "finished_utc": datetime.now(UTC).isoformat(),
        "base": session.base.id, "store": str(root), "source": str(source), "store_kind": session.kind,
        "input": str(options.input), "keep_flags": keep_flags, "expected_sequences": len(session.keys),
        "accounting": accounting, "added_proteins": len(added), "recompute_sample": {"sequences": compared, "differing": differing},
        "coverage": coverage, "canary": session.canary, "evidence": session.runtime.evidence,
    })


def stratified_sample(texts: Sequence[str], geometry: Any, sample: int, seed: int) -> list[str]:
    """About ``sample`` of ``texts``, drawn from each bucket of ``geometry`` in proportion to its size and at least one
    from each, grouped by bucket in the order the buckets first occur."""
    by_bucket: dict[int, list[str]] = {}
    for text in texts:
        by_bucket.setdefault(geometry.columns(min(len(text), CANONICAL_MAX_RESIDUES)), []).append(text)
    rng = random.Random(seed)
    share = sample / max(len(texts), 1)
    return [text for members in by_bucket.values()
            for text in rng.sample(members, min(len(members), max(1, round(share * len(members)))))]


def compare(options: argparse.Namespace) -> None:
    """How far the rows of another store of a base lie from a profile store's, stream by stream: the drift record.

    ``--store`` is the reference, a profile store; ``--source`` is another store of the same base and kind, such as the
    v2 store the profile store replaces, and may be mounted read-only. Of the ``--input`` proteins both stores hold, a
    bucket-stratified sample is read from each and compared as `qualify` compares panels (`numerics.compare_panels`):
    the pooled rows of every sampled protein and the per-token rows of one in `COMPARE_TOKEN_SHARE`. Each stream's report
    adds the quantiles of the proteins' relative L2, the proteins that moved most, and for code streams the ratio of the
    other store's values to the profile store's on shared codes (`numerics.protein_differences`). Streams only one store
    has are named in the receipt and not compared.
    """
    from fastplms.features import FeatureStore

    from .numerics import compare_panels, load_profile, panel_tensors, protein_differences

    started = time.monotonic()
    reference, other = read_manifest(options.store), read_manifest(options.source)
    binding = manifest_profile(reference)
    if binding is None:
        raise SystemExit(f"{options.store} is a {reference['schema']} store; compare measures against a profile store.")
    if (other["base"], manifest_kind(other)) != (reference["base"], manifest_kind(reference)):
        raise SystemExit(f"{options.source} is the {manifest_kind(other)} store of {other['base']} and {options.store} the "
                         f"{manifest_kind(reference)} store of {reference['base']}; compare two stores of one base and kind.")
    profile = load_profile(binding["id"])
    if profile.binding() != binding:
        raise SystemExit(f"{options.store} was computed under {binding['id']} {binding['sha256'][:12]}, and this code's "
                         f"profile file has digest {profile.sha256[:12]}.")
    geometry = profile.geometry(reference["base"])
    names = sorted(set(reference["streams"]) & set(other["streams"]))
    texts, keys = read_inventory(options.input, digest_column=options.digest_column or "seq_sha256",
                                 sequence_column=options.sequence_column or "sequence", shard=(0, 1), limit=options.limit,
                                 keep_flags=options.keep_flag)
    held = set(keys)
    for root, manifest in ((options.store, reference), (options.source, other)):
        for name in names:
            stream = FeatureStore.read_only(Path(root) / manifest["streams"][name]["key"])
            held &= stream.present_digests(sorted(held))
    drawn = stratified_sample([text for text, key in zip(texts, keys, strict=True) if key in held], geometry,
                              COMPARE_SAMPLE if options.sample is None else options.sample, options.seed)
    if not drawn:
        raise SystemExit(f"No protein of {options.input} is in both {options.store} and {options.source}.")
    selected = drawn[::COMPARE_TOKEN_SHARE]  # every bucket's members are consecutive, so this is stratified too
    inventory = sequence_inventory(drawn)
    row_keys = {text: inventory[text].sha256 for text in drawn}
    panels = []
    for root in (options.store, options.source):
        with open_canonical_view(root, inventory, streams=names, budget=batch_budget(geometry)) as view:
            layouts = {name: view.descriptor(name).layout for name in names}
            panels.append(panel_tensors(view, layouts, drawn, selected, row_keys))
    report: dict[str, dict[str, Any]] = dict(compare_panels(*panels))  # the profile store's rows are the reference
    lengths = {key: len(text) for text, key in row_keys.items()}
    for name, summary in report.items():
        summary.update(protein_differences(*panels, name, lengths))
        print(f"compare {name}: " + ", ".join(f"{field} {value:.4g}" for field, value in summary.items()
                                               if isinstance(value, float)), file=sys.stderr)
    identity = ("schema", "profile", "model_state_sha256", "hardware_class")
    write_receipt(options.receipt, {
        "schema": SCHEMA, "command": "compare", "finished_utc": datetime.now(UTC).isoformat(),
        "seconds": time.monotonic() - started, "base": reference["base"], "store_kind": manifest_kind(reference),
        "input": str(options.input), "keep_flags": options.keep_flag, "seed": options.seed,
        "reference": {"store": str(options.store), **{field: reference.get(field) for field in identity}},
        "other": {"store": str(options.source), **{field: other.get(field) for field in identity}},
        "streams_compared": names,
        "streams_unmatched": sorted(set(reference["streams"]) ^ set(other["streams"])),
        "proteins": {"input": len(keys), "in_both": len(held), "sampled": len(drawn), "per_token": len(selected)},
        "report": report,
    })


def batch_budget(geometry: Any) -> Any:
    """A read budget that holds one batch of ``geometry`` in any stream: its rows, its padded tokens, at most 1 GiB."""
    from .readers import ReadBudget

    return ReadBudget(max_rows=geometry.max_rows, max_padded_residues=geometry.token_budget, max_bytes=GIB)


def count_unequal(view: Any, name: str, group: Sequence[str], packed: Any) -> int:
    """Proteins of ``group`` whose recomputed rows of stream ``name`` differ by a single bit from the stored ones."""
    import torch

    feature = view.descriptor(name)
    stored = view.rows(name, group)  # b rows: (l_i + 2, w) per token, TopKRow (l_i + 2, k), (w,) pooled or sparse
    recomputed = packed.streams[name]  # values (n, w) | (n, k) + indices (n, k) | (b, w)
    unequal = 0
    cursor = 0  # the first packed row of the current protein, n = sum(l_i + 2) rows in all
    for index, rows in enumerate(packed.rows):
        if feature.layout == "ragged":
            unequal += not torch.equal(stored[index], recomputed["values"][cursor:cursor + rows].to(stored[index].device))
        elif feature.layout == "ragged_topk":
            unequal += not (torch.equal(stored[index].values, recomputed["values"][cursor:cursor + rows])
                            and torch.equal(stored[index].indices.long(), recomputed["indices"][cursor:cursor + rows].long()))
        else:
            row = stored[index].to_dense(feature.width) if feature.layout == "csr" else stored[index]  # (w,)
            unequal += not torch.equal(row, recomputed["values"][index])
        cursor += rows
    return unequal


def attest(options: argparse.Namespace) -> None:
    """Measure and record this container's closure, numerics and device class under a profile, loading no model."""
    import torch

    from .numerics import (
        apply_profile, attest_environment, check_environment, check_numerics, describe_device, device_class, load_profile,
        measure_numerics, profile_evidence,
    )

    profile = load_profile(options.profile)
    device = torch.device(options.device)
    attestation = attest_environment()
    check_environment(profile, attestation)
    apply_profile(profile, device)
    measured = measure_numerics(device)
    check_numerics(profile, measured, device)
    record = describe_device(device)
    hardware = device_class(profile, record, base=canonical_base(options.base).id, enrolling=True)
    evidence = profile_evidence(profile, attestation, record, hardware, measured)
    write_receipt(options.receipt, {"schema": SCHEMA, "command": "attest", "finished_utc": datetime.now(UTC).isoformat(),
                                    "base": canonical_base(options.base).id, **evidence})
    print(f"closure {attestation['closure_sha256']} on {attestation['closure']['platform']}; class {hardware.id}",
          file=sys.stderr, flush=True)


def qualify(session: Session, options: argparse.Namespace) -> None:
    """A device's qualification receipt: the canary twice, and the panel's rows invariant to companions and position.

    The canary runs once in the session and once more here, and the two must match bitwise (repeatability); the second
    run also records the largest activation of every module (`numerics.activation_peaks`, the FP16 headroom). The panel
    is then embedded into a scratch store in key order and recomputed in shuffled batches, every stream bitwise equal
    (companions, positions, partial batches). The receipt holds the class, the device and the digests a profile records
    as golden for the class when this device is its first member, or compares against when it joins.
    """
    from .numerics import activation_peaks, headroom_report

    first = session.canary or run_canary(session)
    with activation_peaks(session.runtime.model) as peaks:  # the FP16 headroom audit; it reads outputs only
        second = run_canary(session)
    if first["digests"] != second["digests"]:
        raise AssertionError(f"The canary did not repeat on this device: {first['digests']} then {second['digests']}.")
    audit = headroom_report(peaks)
    print(f"largest activation {audit['largest']:.4g} in {audit['peaks'][0]['module']}: FP16 headroom "
          f"{audit['fp16_headroom']:.1f}x", file=sys.stderr, flush=True)
    if (options.store / MANIFEST_FILE).exists():
        raise FileExistsError(f"{options.store} already holds a store; qualification writes a scratch store.")
    assert session.profile is not None
    write_manifest(options.store, profile_manifest(session.base, session.contract, session.kind, session.profile.profile,
                                                   session.profile.hardware))
    run_embedding(session, options.store, session.texts, session.keys, options, progress=None, verify_cached=False)
    options.sample = len(session.texts) if options.sample is None else options.sample
    verify_receipt = options.receipt.with_name(f"{options.receipt.stem}.verify.json")
    verify_exact(session, argparse.Namespace(**{**vars(options), "receipt": verify_receipt}))
    panel = write_panel(session, options)
    provisional = session.profile.hardware.id not in session.profile.profile.classes  # numerics.provisional_class
    write_receipt(options.receipt, {
        "schema": SCHEMA, "command": "qualify", "finished_utc": datetime.now(UTC).isoformat(), "base": session.base.id,
        "store_kind": session.kind, "canary": first, "canary_repeat_matches": True, "invariance": str(verify_receipt),
        "panel": panel, "provisional_class": provisional, "activation_audit": audit, "evidence": session.runtime.evidence,
    })
    print(f"qualified {session.profile.evidence['device']['name']} for {'provisional ' * provisional}class "
          f"{session.profile.hardware.id} on {session.base.label}; record its digests from {options.receipt}",
          file=sys.stderr, flush=True)


def write_panel(session: Session, options: argparse.Namespace) -> dict[str, Any]:
    """Write beside the receipt the rows another device's qualification is compared with, read back from the store.

    Every panel protein's pooled rows, and the per-token rows of the first protein of each bucket, so every batch shape
    is compared (`numerics.panel_selection`). Returns the file's path and SHA-256 and what it holds.
    """
    from safetensors.torch import save_file

    from foundry.digests import sha256_file

    from .launch import panel_path
    from .numerics import PANEL_SCHEMA, canary_panel, panel_selection, panel_tensors

    assert session.profile is not None and session.contract.geometry is not None
    geometry = session.contract.geometry
    keys = dict(zip(session.texts, session.keys, strict=True))
    ordered = [normalize_sequence(text) for text in canary_panel(geometry, max_residues=CANONICAL_MAX_RESIDUES)]
    selected = [text for text in dict.fromkeys(panel_selection(ordered, geometry)) if text in keys]
    layouts = {stream: record["layout"] for stream, record in read_manifest(options.store)["streams"].items()}
    inventory = {text: SequenceKey(key, len(text), NORMALIZATION_VERSION) for text, key in keys.items()}
    with open_canonical_view(options.store, inventory) as view:
        tensors = panel_tensors(view, layouts, session.texts, selected, keys)
    path = panel_path(options.receipt)
    evidence = session.profile.evidence
    save_file(tensors, str(path), metadata={
        "schema": PANEL_SCHEMA, "base": session.base.id, "profile": json.dumps(session.profile.profile.binding()),
        "hardware_class": session.profile.hardware.id, "device": evidence["device"]["name"],
        "closure_sha256": evidence["closure_sha256"],
    })
    return {"path": str(path), "sha256": sha256_file(path), "proteins": len(session.texts),
            "per_token_proteins": len(selected), "tensors": len(tensors)}


def free_device_bytes(device: str) -> int | None:
    """Memory a CUDA ``device`` can give a batch, after what other processes hold; None for any other device.

    On a GH200 the driver counts the page cache of the HBM node as used, which the kernel drops when an allocation
    needs it, so a run that follows large writes read about 1.8 GiB free beside 97 GiB of cache. That cache is added
    back (`foundry.gpu_memory`), within the device's total less this process's tensors.
    """
    import torch

    target = torch.device(device)
    if target.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(target)
    return usable_free_bytes(free, total, torch.cuda.memory_allocated(target), hbm_page_cache_bytes())


def parse_shard(value: str) -> tuple[int, int]:
    """`i/n` as (i, n), with 0 <= i < n."""
    index, _, count = value.partition("/")
    shard = (int(index), int(count or 1))
    if not 0 <= shard[0] < shard[1]:
        raise ValueError(f"--shard must be i/n with 0 <= i < n; received {value!r}.")
    return shard


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON receipt that must not exist, so a command never overwrites an earlier record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_embedding(session: Session, root: Path, texts: Sequence[str], keys: Sequence[str], options: argparse.Namespace,
                  *, progress: Any, verify_cached: bool, on_plan: Any = None, batch_watch: Any = None) -> dict[str, Any]:
    """One `embed_token_features` call (`embed_encoder_features` for a non-protein base) over ``texts`` with the session's contract and settings."""
    from fastplms.embeddings import embed_token_features

    base, runtime, contract = session.base, session.runtime, session.contract
    if session.prepared is not None:
        assert isinstance(contract, EncoderExtractionContract)
        receipts = embed_encoder_features(
            runtime, session.prepared, texts, keys, root, contract.stored_features, contract=contract,
            metadata={"base": base.id, "shard": options.shard, "job": SCHEMA, "store_kind": session.kind},
            max_sequences=session.run_batch.max_sequences, max_tokens=session.run_batch.max_tokens,
            window=runtime.capture["batch_window_size"], part_bytes=int(options.part_gib * GIB),
            segment_bytes=int(options.segment_gib * GIB), queue_bytes=int(options.queue_gib * GIB), workers=options.writers,
            verify_cached=verify_cached, progress=progress, on_plan=on_plan,
        )
        return {name: [{"fingerprint": item.fingerprint, "rows": item.rows, "parts": len(item.parts)} for item in group]
                for name, group in receipts.items()}
    common = dict(
        taps=runtime.taps, contract=contract, digests=keys,
        metadata={"base": base.id, "shard": options.shard, "job": SCHEMA, "store_kind": session.kind},
        max_residues=runtime.capture["max_length"], part_bytes=int(options.part_gib * GIB),
        segment_bytes=int(options.segment_gib * GIB), queue_bytes=int(options.queue_gib * GIB), workers=options.writers,
        verify_cached=verify_cached, model_state_fingerprint=contract.model_state_sha256, progress=progress,
        on_plan=on_plan, batch_watch=batch_watch,
    )
    if session.profile is not None:
        from .numerics import attention_kernel

        # The profile's run evidence goes into every segment's run.json beside the rows it made.
        common["metadata"] = {**common["metadata"], "profile_run": session.profile.evidence}
        with attention_kernel(session.profile.profile):
            receipts = embed_token_features(runtime.model, texts, root, contract.stored_features,
                                            geometry=contract.geometry, **common)
    else:
        assert session.run_batch is not None
        receipts = embed_token_features(
            runtime.model, texts, root, contract.stored_features, max_sequences=session.run_batch.max_sequences,
            max_tokens=session.run_batch.max_tokens, window=runtime.capture["batch_window_size"],
            fixed_batch_size=runtime.capture.get("fixed_batch_size"), **common,
        )
    return {name: [{"fingerprint": item.fingerprint, "rows": item.rows, "parts": len(item.parts)} for item in group]
            for name, group in receipts.items()}


def session_manifest(session: Session) -> dict[str, Any]:
    """The manifest a session's store has: v3 with the profile and class for a profile session, else v2."""
    if session.profile is not None:
        return profile_manifest(session.base, session.contract, session.kind, session.profile.profile,
                                session.profile.hardware)
    return stream_manifest(session.base, session.contract, session.kind)


def embed(session: Session, options: argparse.Namespace, *, batch_watch: Any = None) -> None:
    """Embed every row the store lacks and record a receipt; a rerun resumes from the last committed segment."""
    root = options.store
    write_manifest(root, session_manifest(session))
    if session.prepared is not None:
        print(f"inventory {write_row_inventory(root, session.prepared)}", file=sys.stderr, flush=True)
    started = time.monotonic()
    bar = tqdm(total=len(session.texts), desc=f"{session.base.label} {options.shard}", unit="seq",
               mininterval=PROGRESS_SECONDS, file=sys.stderr)

    def planned(count: int) -> None:
        """Rows committed by an earlier run are skipped without a tick, so the bar's total drops to the rows left."""
        bar.total = count
        bar.refresh()

    received = run_embedding(session, root, session.texts, session.keys, options, progress=bar.update,
                      verify_cached=False, on_plan=planned, batch_watch=batch_watch)
    bar.close()
    seconds = time.monotonic() - started
    write_receipt(options.receipt, {
        "schema": SCHEMA, "command": "embed", "finished_utc": datetime.now(UTC).isoformat(), "seconds": seconds,
        "base": session.base.id, "store": str(root), "shard": options.shard, "input": str(options.input),
        "keep_flags": getattr(options, "keep_flag", []), "sequences": len(session.texts), "unrepresentable": session.rejected, "segments": received,
        "store_kind": session.kind,
        "settings": {"max_residues": session.runtime.capture["max_length"], **session.batch_settings(),
                     **{key: value for key, value in vars(options).items()
                        if key in {"part_gib", "segment_gib", "queue_gib", "writers"}}},
        "evidence": session.runtime.evidence, "model_state_sha256": session.contract.model_state_sha256,
        "canary": session.canary, "golden": session.golden,
    })
    print(f"embedded {len(session.texts):,} {'proteins' if session.prepared is None else family_of(session.base).id + ' rows'} "
          f"in {seconds / 60:.1f} min", file=sys.stderr, flush=True)


def pilot_sample_size(options: argparse.Namespace, available: int) -> int:
    """Sequences a pilot draws: ``--sample``, else the fewest that fill every warmup and timed step with a chunk.

    A pilot that holds fewer chunks than ``--warmup`` plus ``--steps`` times fewer steps than asked, and its
    projection rests on them, so it stops instead.
    """
    needed = options.chunk * (options.steps + options.warmup)
    sample = min(needed if options.sample is None else options.sample, available)
    chunks = -(-sample // options.chunk)
    if chunks < options.steps + options.warmup:
        raise ValueError(
            f"A pilot of {options.steps} timed steps after {options.warmup} warmup needs {options.steps + options.warmup} "
            f"chunks of {options.chunk} sequences; {sample:,} sequences make {chunks}. Raise --sample (the default is "
            f"{needed:,}), lower --steps, or give the pilot a longer list ({available:,} sequences)."
        )
    return sample


def pilot(session: Session, watcher: Pilot, options: argparse.Namespace) -> None:
    """Time chunks of a random sample into a scratch store; the watcher reports and projects the whole run."""
    import torch

    root = options.store
    if (root / MANIFEST_FILE).exists():
        raise FileExistsError(f"{root} already holds a store; a pilot writes to a scratch directory.")
    sample = pilot_sample_size(options, len(session.texts))
    write_manifest(root, session_manifest(session))
    order = sorted(range(len(session.texts)), key=lambda index: session.keys[index])
    drawn = random.Random(options.seed).sample(order, sample)
    special = special_rows(session)  # rows a sequence adds to its units: CLS and EOS, or none
    units = [unit_count(session, text) for text in session.texts]  # (list,) units l after the crop: residues, symbols or terms
    residues = [units[index] for index in drawn]  # (sample,) units l after the crop
    write_receipt(options.receipt.with_name(f"{options.receipt.stem}.sample.json"), {
        "schema": SCHEMA, "command": "pilot-sample", "store_kind": session.kind, "base": session.base.id,
        "sequences": sample, "residues": sum(residues), "mean_residues": sum(residues) / sample,
        "token_rows": sum(residues) + special * sample, "chunk": options.chunk,
        "run_batch": session.batch_settings() if session.run_batch is None else {
            "max_sequences": session.run_batch.max_sequences, "max_tokens": session.run_batch.max_tokens,
            "source": session.run_batch.source},
        "canary": session.canary, "golden": session.golden,
        "list_sequences": len(session.texts),
        "list_residues": sum(units),
        "list_token_rows": sum(units) + special * len(units),
    })
    chunks = [drawn[start:start + options.chunk] for start in range(0, len(drawn), options.chunk)]
    for chunk in watcher.watch(chunks):
        run_embedding(session, root, [session.texts[i] for i in chunk], [session.keys[i] for i in chunk], options,
                      progress=None, verify_cached=False)
    if torch.cuda.is_available():
        print(f"pilot: peak allocated {torch.cuda.max_memory_allocated() / GIB:.1f} GiB, "
              f"reserved {torch.cuda.max_memory_reserved() / GIB:.1f} GiB", file=sys.stderr, flush=True)


def special_rows(session: Session) -> int:
    """Rows a sequence adds to its units in a per-token stream: CLS and EOS (2), or none for a family without them."""
    binding = session.runtime.encoder
    return SPECIAL_TOKEN_ROWS if binding is None else binding.family.special_rows


def unit_count(session: Session, text: str) -> int:
    """Units l of one text after the crop: residues, SELFIES symbols or annotation terms, never counting special tokens."""
    if session.prepared is None:
        return min(len(text), session.runtime.capture["max_length"])
    return session.prepared[text].rows - special_rows(session)


def relative_error(stored: Any, recomputed: Any) -> float:
    """L2 norm of the difference over the norm of the recomputed rows, in float64."""
    import torch

    # stored, recomputed: (n, w) rows of a per-token stream, (w,) of a pooled one; the same shape, or a mismatch.
    if stored.shape != recomputed.shape:
        raise AssertionError(f"Stored rows have shape {tuple(stored.shape)}, recomputed {tuple(recomputed.shape)}")
    difference = stored.double() - recomputed.double()  # same shape as stored, float64
    return (torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(recomputed.double()).clamp_min(1e-30)).item()


def recompute_executor(session: Session, texts: Sequence[str]) -> tuple[Any, dict[str, SequenceKey]]:
    """The executor that recomputes rows independently of the store, and the inventory of ``texts`` a view reads them with."""
    from fastplms.embeddings import TokenTapExecutor
    from fastplms.embeddings.taps import plan_taps
    from fastplms.embeddings.tokens import ResidueVocabulary

    runtime = session.runtime
    if session.prepared is None:
        executor = TokenTapExecutor(
            runtime.model, plan_taps(list(runtime.taps), int(runtime.model.embedding_tap_state_count)),
            vocabulary=ResidueVocabulary(runtime.model.tokenizer), max_residues=runtime.capture["max_length"], dtype=None,
            fixed_batch_size=runtime.capture.get("fixed_batch_size"),
        )
        return executor, sequence_inventory(texts)
    binding = runtime.encoder
    assert binding is not None
    streams = [stream for stream in binding.family.streams if stream.name in session.contract.stored_features]
    executor = EncoderExecutor(binding.hooks, session.prepared, streams, device=next(runtime.model.parameters()).device)
    return executor, row_inventory(binding.family, {text: session.prepared[text] for text in texts})


def verify(session: Session, options: argparse.Namespace) -> None:
    """Account for every input row, read a sample back as a consumer would, and recompute it independently."""
    import torch

    from fastplms.features import FeatureStore

    root, contract, base = options.store, session.contract, session.base
    manifest = read_manifest(root)
    if manifest_kind(manifest) != session.kind or set(manifest["streams"]) != set(contract.stored_features):
        raise ValueError(f"{root} holds the {manifest_kind(manifest)} streams {sorted(manifest['streams'])}, but this run "
                         f"checks the {session.kind} streams {sorted(contract.stored_features)}; add or drop --pooled.")
    accounting: dict[str, int] = {}
    for name, feature in contract.stored_features.items():
        store = FeatureStore.open(root, feature, deep_verify=False)
        missing = set(session.keys) - store.present_digests(session.keys)
        if missing:
            raise AssertionError(f"{name}: {len(missing)} of {len(session.keys)} expected proteins are not committed")
        accounting[name] = len(session.keys)
    sample = VERIFY_SAMPLE if options.sample is None else options.sample
    drawn = random.Random(options.seed).sample(range(len(session.texts)), min(sample, len(session.texts)))
    texts = [session.texts[i] for i in drawn]
    executor, inventory = recompute_executor(session, texts)
    parity: dict[str, dict[str, float]] = {}
    with open_canonical_view(root, inventory, deep=True) as view:  # the deep check: hash every part, scan every sampled row
        errors: dict[str, list[float]] = {name: [] for name in view.names}
        with torch.inference_mode():
            for start in range(0, len(texts), 8):
                group = texts[start:start + 8]
                packed = executor.run_batch(group, [inventory[text].sha256 for text in group])
                packed.wait()
                for name in view.names:
                    feature = view.descriptor(name)
                    stored = view.rows(name, group)  # b rows: (l_i + 2, w) per token, TopKRow (l_i + 2, k), (w,) pooled
                    recomputed = packed.streams[name]  # values (n, w) | (n, k) + indices (n, k) | (b, w) | (b, c)
                    if feature.layout in {"ragged", "ragged_topk"}:
                        cursor = 0  # first packed row of sequence `index` in recomputed, n = sum(l_i + 2) rows in all
                        for index, rows in enumerate(packed.rows):
                            expected = unit_count(session, group[index]) + special_rows(session)  # l + 2: CLS, units, EOS
                            if rows != expected:
                                raise AssertionError(f"{name}: recomputed {rows} rows for a protein needing {expected}")
                            if feature.layout == "ragged":
                                errors[name].append(relative_error(stored[index], recomputed["values"][cursor:cursor + rows]))
                            else:
                                shape = (rows, feature.width)  # (l_i + 2, c) dense codes of one sequence
                                # stored, recomputed: each (l_i + 2, c) float64 after scattering its k codes per row
                                dense = [torch.zeros(shape, dtype=torch.float64).scatter_(1, part.indices.long(), part.values.double())
                                         for part in (stored[index],
                                                      type(stored[index])(recomputed["indices"][cursor:cursor + rows],
                                                                          recomputed["values"][cursor:cursor + rows]))]
                                errors[name].append(relative_error(dense[0], dense[1]))
                            cursor += rows
                    else:
                        for index in range(len(group)):
                            row = stored[index].to_dense(feature.width) if feature.layout == "csr" else stored[index]  # (w,)
                            errors[name].append(relative_error(row, recomputed["values"][index]))
        for name, values in errors.items():
            parity[name] = {"maximum_relative_error": max(values), "mean_relative_error": sum(values) / len(values)}
    worst = {name: item for name, item in parity.items() if item["maximum_relative_error"] > options.tolerance}
    if worst:
        raise AssertionError(f"Recomputed rows differ from stored rows beyond {options.tolerance}: {worst}")
    write_receipt(options.receipt, {
        "schema": SCHEMA, "command": "verify", "finished_utc": datetime.now(UTC).isoformat(), "base": base.id,
        "store": str(root), "store_kind": session.kind, "expected_sequences": len(session.keys), "accounting": accounting,
        "manifest_streams": sorted(manifest["streams"]), "recompute_sample": {"sequences": len(texts), "parity": parity,
                                                                               "tolerance": options.tolerance},
    })


def hub_card(base: CanonicalBase, manifest: dict[str, Any], repack: dict[str, Any] | None = None) -> str:
    """The README of a base's Hub dataset repo: what the rows are, their layout, and how to read them.

    ``repack`` is the record of an interaction-only token store (`repack.json`), which says which proteins it keeps.
    """
    if base.family != FAMILY_PROTEIN:
        return encoder_hub_card(base, manifest)
    streams = "\n".join(
        f"| `{name}` | {record['layout']} | {record['width']} | {record['dtype']} |"
        for name, record in manifest["streams"].items()
    )
    random_note = "" if manifest["random_init"] is None else (
        f"\nThe weights are random: seed {manifest['random_init']['seed']}, the ESMC-300 architecture, no training. "
        "This is a control, not a protein model.\n"
    )
    if manifest_kind(manifest) == STORE_POOLED:
        return f"""---
pretty_name: {base.label} pooled embeddings
tags: [protein, embeddings, esmc]
---
# {base.label} pooled embeddings

One vector per protein from {base.label}, in the FastPLMs feature-store format, for every protein of
[Synthyra/canonical_embedding_sequences_v1](https://huggingface.co/datasets/Synthyra/canonical_embedding_sequences_v1).
The per-token streams of interaction proteins are in `{base.hub_repo}`.
{random_note}
Each vector covers all l + 2 token rows of a protein (cropped from the N-terminus at {manifest['max_residues']}): row 0 is CLS, rows 1..l
are the residues, row l + 1 is EOS, and padding is the only masked position. `final_mean_var` is the mean, then the population
variance, of the final normalized hidden state. `sae_max` is the maximum of the SAE codes over those rows.

A row is keyed by the SHA-256 of the uppercased, whitespace-stripped sequence. The store only appends.

| stream | layout | width | dtype |
|---|---|---|---|
{streams}

Read it with `foundry.embedding.open_canonical_view(root, inventory)` (a `FeatureView`), or with
`fastplms.features.FeatureStore` directly. `canonical.json` lists the streams and pins each descriptor. Model state
SHA-256: `{manifest['model_state_sha256']}`.
"""
    scope = "" if repack is None else (
        f"\nThis revision holds the {repack['kept_rows']:,} proteins that an interaction dataset needs "
        f"({', '.join(repack['keep_flags'])} of the sequence list at revision `{repack['list_revision']}`). "
        f"Every other protein has its pooled vectors in `{base.pooled_hub_repo}`.\n"
    )
    return f"""---
pretty_name: {base.label} token embeddings
tags: [protein, embeddings, esmc]
---
# {base.label} token embeddings

Per-token and pooled embeddings of protein sequences from {base.label}, in the FastPLMs feature-store format.
{random_note}{scope}
**Special tokens are kept.** A protein of l residues (cropped from the N-terminus at {manifest['max_residues']}) is l + 2
rows in every per-token stream: row 0 is CLS, rows 1..l are the residues, row l + 1 is EOS. Padding is the only masked
position, and every pooled stream (mean, variance, SAE max) covers all l + 2 rows.

A row is keyed by the SHA-256 of the uppercased, whitespace-stripped sequence. The store only appends.

| stream | layout | width | dtype |
|---|---|---|---|
{streams}

Read it with `foundry.embedding.open_canonical_view(root, inventory)` (a `FeatureView` over the streams), or with
`fastplms.features.FeatureStore` directly. `canonical.json` lists the streams and pins each descriptor. Model state
SHA-256: `{manifest['model_state_sha256']}`.
"""


def committed_segments(root: Path) -> tuple[list[Path], list[Path]]:
    """The committed and the uncommitted segment directories under a store root."""
    committed, partial = [], []
    for directory in sorted(root.glob("*/segments/*")):
        if directory.is_dir():
            (committed if (directory / "run.json").is_file() else partial).append(directory)
    return committed, partial


def read_store(options: argparse.Namespace) -> tuple[CanonicalBase, dict[str, Any]]:
    """The base ``--base`` names and the manifest of ``--store``, which must describe that base and that kind of store."""
    base = canonical_base(options.base)
    manifest = read_manifest(options.store)
    if manifest["base"] != base.id:
        raise ValueError(f"{options.store} holds {manifest['base']}, not {base.id}.")
    if manifest_kind(manifest) != store_kind(options):
        raise ValueError(f"{options.store} is a {manifest_kind(manifest)} store; "
                         f"this command asked for a {store_kind(options)} one (add or drop --pooled).")
    return base, manifest


def sweep(options: argparse.Namespace) -> None:
    """Delete the uncommitted segment directories a killed run left, in each stream that has any.

    A segment a running writer holds is skipped by `FeatureStore.sweep` and reported here as left, so stop that
    run first. With ``--dry-run`` the directories are listed and kept.
    """
    from fastplms.features import FeatureStore

    root = options.store
    _, manifest = read_store(options)
    _, partial = committed_segments(root)
    if not partial:
        print(f"sweep: no uncommitted segments under {root}", file=sys.stderr, flush=True)
        return
    keys = {directory.parent.parent.name for directory in partial}
    streams = [name for name, record in manifest["streams"].items() if record["key"] in keys]
    unknown = keys - {manifest["streams"][name]["key"] for name in streams}
    if unknown:
        raise ValueError(f"{root} holds uncommitted segments in {sorted(unknown)}, which are not streams of its manifest.")
    if options.dry_run:
        print(f"sweep: would remove {len(partial)} uncommitted segment directories: {[directory.name for directory in partial]}",
              file=sys.stderr, flush=True)
        return
    for name, feature in manifest_features(root, manifest, streams).items():
        removed = FeatureStore.open(root, feature, deep_verify=False).sweep()
        print(f"sweep: {name}: removed {len(removed)} uncommitted segment directories", file=sys.stderr, flush=True)
    _, remaining = committed_segments(root)
    if remaining:
        raise RuntimeError(
            f"{len(remaining)} uncommitted segment directories remain under {root}, such as {remaining[0].name}; "
            "a running writer holds them. Stop that run, then sweep again."
        )


def push(options: argparse.Namespace) -> None:
    """Upload the committed store to the base's Hub dataset repo and Modal volume, each resumably and incrementally.

    With ``--receipt``, a profile store's push records the catalog entry and the revision that holds it.
    """
    base, manifest = read_store(options)
    root = options.store
    committed, partial = committed_segments(root)
    if partial:
        raise ValueError(
            f"{len(partial)} uncommitted segment directories remain under {root}, such as {partial[0].name}; "
            f"a killed run left them. Run `python -m foundry.embedding.job sweep --base {base.id} "
            f"--store {shlex.quote(str(root))}`, then push again."
        )
    print(f"push: {len(committed)} committed segments of {base.label} ({manifest_kind(manifest)} store)",
          file=sys.stderr, flush=True)
    if manifest.get("schema") == MANIFEST_SCHEMA_V3:
        from .hub import check_coverage, check_file_counts, mirror_root, push_store, write_store_index
        from .numerics import check_publishable, load_profile

        check_publishable(manifest, load_profile(manifest["profile"]["id"]))
        coverage = check_coverage(root)  # an exact verify passed this store as it stands, against a named list
        sequences = coverage["sequences"]
        print(f"push: covers {sequences['proteins']:,} proteins of {sequences['repository']}@{sequences['revision'][:8]}"
              f"{' flagged ' + ', '.join(sequences['keep_flags']) if sequences['keep_flags'] else ''}", file=sys.stderr, flush=True)
        counts = check_file_counts(store_files(root), root)
        repository = options.hub_staging or manifest["hub_repo"]
        print(f"push: {counts['files']:,} files, at most {counts['largest_folder']:,} in one folder, to "
              f"{repository}/{manifest['path_in_repo']} from mirror {mirror_root(root, manifest)}", file=sys.stderr, flush=True)
        if options.dry_run:
            return
        entry = None
        if options.hub:
            entry = push_store(root, visibility=options.visibility, files=store_files(root), staging=options.hub_staging)
            print(f"pushed {entry['proteins']:,} proteins to https://huggingface.co/datasets/{repository}/tree/"
                  f"{entry['revision']}/{manifest['path_in_repo']}", file=sys.stderr, flush=True)
        if options.volume:
            if not options.hub:
                write_store_index(root)  # the Hub push writes the indexes; a volume copy carries them as well
            push_volume(base, root, manifest)
        if options.receipt is not None:
            write_receipt(options.receipt, {
                "schema": SCHEMA, "command": "push", "finished_utc": datetime.now(UTC).isoformat(), "base": base.id,
                "store": str(root), "store_kind": manifest_kind(manifest), "profile": manifest["profile"],
                "sequences": sequences, "store_sha256": coverage["store_sha256"],
                "repository": repository if options.hub else None, "path_in_repo": manifest["path_in_repo"],
                "hub": entry, "volume": manifest["volume"] if options.volume else None,
            })
        return
    if getattr(options, "prune_hub", False) and not (root / REPACK_FILE).is_file():
        raise ValueError("--prune-hub removes published files; only a repacked store (repack.json) may do it.")
    if options.dry_run:
        return
    if options.hub:
        push_hub(base, root, manifest, visibility=options.visibility, prune=getattr(options, "prune_hub", False))
    if options.volume:
        push_volume(base, root, manifest)


def store_files(root: Path) -> list[Path]:
    """The files of a store that are published: parts, sidecars, markers, descriptors and manifest, never an index, a lock
    or a grown store's record of the store it grew from."""
    from .hub import GROWN_FILE

    return [path for path in sorted(root.rglob("*")) if path.is_file()
            and not any(part in {".locks", ".cache"} for part in path.parts)
            and path.suffix not in {".writing", ".lock"} and not path.name.startswith("index.sqlite")
            and path.name != GROWN_FILE]


def prune_hub(api: Any, repo: str, root: Path) -> list[str]:
    """Delete from the Hub repo every file the local store no longer has, in commits of at most 500; returns their paths.

    The earlier revisions keep the files in the repo's history. Only a repacked store may call this (`push`).
    """
    from huggingface_hub import CommitOperationDelete

    from foundry.downloads.hub import hub_call, rate_limited

    local = {path.relative_to(root).as_posix() for path in store_files(root)}
    stale = sorted(path for path in hub_call(api.list_repo_files, repo, repo_type="dataset")
                   if path not in local and path != ".gitattributes" and not path.startswith(".cache/"))
    for start in range(0, len(stale), 500):
        group = stale[start:start + 500]
        hub_call(api.create_commit, repo_id=repo, repo_type="dataset", when=rate_limited,
                 operations=[CommitOperationDelete(path_in_repo=path) for path in group],
                 commit_message=f"Repack to the interaction proteins: remove {len(group)} files of the full store "
                                "(still in the repository history)")
    return stale


def push_hub(base: CanonicalBase, root: Path, manifest: dict[str, Any], *, visibility: str, prune: bool = False) -> None:
    """One resumable `upload_large_folder` of the committed files, with the repo created when it is missing.

    The upload runs again after the account's rate limit stops it, and each request waits the limit out
    (`foundry.downloads.hub`). With ``prune`` (a repacked token store) the files the store replaced are then deleted in
    follow-up commits.
    """
    from huggingface_hub import HfApi

    from foundry.datasets.uploads import require_hub_token
    from foundry.downloads.hub import hub_call, upload_resuming

    require_hub_token()
    api = HfApi()
    repo = manifest["hub_repo"]
    hub_call(api.create_repo, repo, repo_type="dataset", private=visibility == "private", exist_ok=True)
    repack = root / REPACK_FILE
    card = root / "README.md"
    card.write_text(hub_card(base, manifest, json.loads(repack.read_text(encoding="utf-8")) if repack.is_file() else None),
                    encoding="utf-8")
    upload_resuming(api, repo_id=repo, folder_path=str(root), repo_type="dataset", ignore_patterns=list(HUB_IGNORE))
    print(f"pushed to https://huggingface.co/datasets/{repo} ({visibility})", file=sys.stderr, flush=True)
    if prune:
        removed = prune_hub(api, repo, root)
        print(f"pruned {len(removed)} files of the replaced store from {repo}", file=sys.stderr, flush=True)


def push_volume(base: CanonicalBase, root: Path, manifest: dict[str, Any]) -> None:
    """Copy files the volume lacks (by path and size) under the store's own directory of the manifest's volume.

    A v2 store goes to its base's volume; a profile store to its model's own volume (`canonical.profile_volume`), at the
    same `profile_v<n>/<model>/<kind>` path it has in its repository.
    """
    name = manifest.get("volume") or base.volume
    if not name:
        raise ValueError(f"{base.label} has no Modal volume; its stores go to the Hub.")
    import modal

    from .hub import COVERAGE_FILE, PARTS_INDEX, ROWS_INDEX
    from .pooled import GRAFTS_FILE

    volume = modal.Volume.from_name(name, create_if_missing=True)
    prefix = manifest["volume_directory"]
    try:
        present = {entry.path: entry.size for entry in volume.listdir(prefix, recursive=True)}
    except (modal.exception.NotFoundError, FileNotFoundError):  # a new volume has no such directory yet
        present = {}  # any other failure (auth, network) stops the push instead of forcing a full upload
    files = store_files(root)
    rewritten = {MANIFEST_FILE, COVERAGE_FILE, GRAFTS_FILE, PARTS_INDEX, ROWS_INDEX}  # records a growing store rewrites
    todo = [path for path in files if path.relative_to(root).as_posix() in rewritten
            or present.get(f"{prefix}/{path.relative_to(root).as_posix()}") != path.stat().st_size]
    print(f"volume {name}: {len(todo)} of {len(files)} files to copy", file=sys.stderr, flush=True)
    with volume.batch_upload(force=True) as batch:
        for path in tqdm(todo, desc="volume", unit="file", mininterval=PROGRESS_SECONDS, file=sys.stderr):
            batch.put_file(str(path), f"/{prefix}/{path.relative_to(root).as_posix()}")
    # Modal hashes every uploaded block; listing the volume again proves every file arrived whole.
    landed = {entry.path: entry.size for entry in volume.listdir(prefix, recursive=True)}
    short = [path for path in files if landed.get(f"{prefix}/{path.relative_to(root).as_posix()}") != path.stat().st_size]
    if short:
        raise RuntimeError(f"volume {name} lacks {len(short)} of the store's {len(files)} files at their size after the "
                           f"upload, such as {[path.relative_to(root).as_posix() for path in short[:3]]}.")
    print(f"volume {name}: all {len(files)} files present at their size under {prefix}", file=sys.stderr, flush=True)


def graft(options: argparse.Namespace) -> None:
    """Grow the profile pooled store ``--store`` by the pooled segments of its base's token store ``--source`` that hold
    none of its proteins (`pooled.graft_segments`), loading no model, and record the receipt."""
    from .pooled import graft_segments

    if options.source is None or not options.pooled or options.receipt is None:
        raise SystemExit("graft needs --pooled, --source (the base's token store) and --receipt.")
    if options.receipt.exists():
        raise FileExistsError(f"{options.receipt} exists; a command never overwrites its receipt")
    base = canonical_base(options.base)
    donor = read_manifest(options.source)  # the pooled store may not exist yet; the graft starts it from the donor
    if donor.get("schema") != MANIFEST_SCHEMA_V3 or donor["base"] != base.id:
        raise ValueError(f"{options.source} is not a profile store of {base.label}; a graft takes from those only.")
    record = graft_segments(options.source, options.store)
    print(f"graft: {record['proteins']:,} proteins in {sum(map(len, record['segments'].values()))} segments "
          f"({record['bytes'] / GIB:.1f} GiB) from {options.source}", file=sys.stderr, flush=True)
    write_receipt(options.receipt, {"schema": SCHEMA, "command": "graft", "finished_utc": datetime.now(UTC).isoformat(),
                                    "base": base.id, "store": str(options.store), "graft": record})


def relocate(options: argparse.Namespace) -> None:
    """Move the profile store ``--store`` to the folder its base names now (`hub.relocate_store`), loading no model, and
    record the receipt: where it was and is, its digest before and after, and whether its coverage record carried."""
    from .hub import relocate_store

    if options.receipt is None:
        raise SystemExit("relocate needs --receipt.")
    if options.receipt.exists():
        raise FileExistsError(f"{options.receipt} exists; a command never overwrites its receipt")
    base, _ = read_store(options)
    root, move = relocate_store(options.store)
    print(f"relocate: {base.label} {store_kind(options)} store from {move['from']} to {move['to']} ({root}); coverage "
          f"{'carried' if move['coverage_carried'] else 'not current, so an exact verify comes before a push'}",
          file=sys.stderr, flush=True)
    write_receipt(options.receipt, {"schema": SCHEMA, "command": "relocate", "finished_utc": datetime.now(UTC).isoformat(),
                                    "base": base.id, "store": str(root), "store_kind": store_kind(options), "move": move})


def seed(options: argparse.Namespace) -> None:
    """Seed the base's pooled store from ``--source`` (its token store) and check every list row and a sample of values."""
    from .pooled import check_seed, seed_pooled

    if options.source is None or not options.pooled:
        raise SystemExit("seed needs --source (the token store) and --pooled.")
    base = canonical_base(options.base)
    digest_column, value_column = family_columns(base, options)
    inventory: Any = sequence_inventory
    if base.family == FAMILY_PROTEIN:
        texts, keys = read_inventory(options.input, digest_column=digest_column, sequence_column=value_column,
                                     shard=(0, 1), limit=options.limit)
    else:
        family = family_of(base)
        read = read_family_inputs(family, options.input, digest_column=digest_column, value_column=value_column,
                                  shard=(0, 1), limit=options.limit, workers=options.prepare_workers, skip=options.skip_unrepresentable)
        texts, keys = [item.text for item in read.inputs], read.keys
        known = inventory_of(family, read.inputs, read.keys)
        inventory = lambda chosen: {text: known[text] for text in chosen}  # noqa: E731 - a view's inventory of the sampled texts
    started = time.monotonic()
    record = seed_pooled(base, options.source, options.store)
    checked = check_seed(base, options.source, options.store, texts, keys,
                         sample=VERIFY_SAMPLE if options.sample is None else options.sample, seed=options.seed,
                         inventory=inventory)
    write_receipt(options.receipt, {"schema": SCHEMA, "command": "seed", "finished_utc": datetime.now(UTC).isoformat(),
                                    "seconds": time.monotonic() - started, "base": base.id, "store": str(options.store),
                                    "input": str(options.input), "list_rows": len(keys), "seeded": record, "check": checked})
    print(f"seeded {base.label} pooled store with {len(keys):,} rows per stream", file=sys.stderr, flush=True)


def repack(options: argparse.Namespace) -> None:
    """Repack ``--source`` into ``--store``: the rows of ``--input`` flagged by any ``--keep-flag``, every stream, checked."""
    import pyarrow.compute as compute
    import pyarrow.parquet as parquet

    from .repack import check_repack, repack_store

    if canonical_base(options.base).family != FAMILY_PROTEIN:
        raise SystemExit("repack is for the protein bases; a ligand or annotation token store holds every row of its list.")
    if options.source is None or not options.keep_flag or options.list_revision is None:
        raise SystemExit("repack needs --source, --keep-flag (once or more) and --list-revision.")
    table = parquet.read_table(options.input, columns=[options.digest_column, options.sequence_column, *options.keep_flag])
    flagged = table.column(options.keep_flag[0])  # (n,) bool
    for name in options.keep_flag[1:]:
        flagged = compute.or_(flagged, table.column(name))
    kept = table.filter(flagged)
    keys = kept.column(options.digest_column).to_pylist()
    texts = kept.column(options.sequence_column).to_pylist()
    print(f"repack: keeping {len(keys):,} of {table.num_rows:,} list rows ({', '.join(options.keep_flag)})",
          file=sys.stderr, flush=True)
    started = time.monotonic()
    record = repack_store(options.source, options.store, set(keys), list_revision=options.list_revision,
                          keep_flags=tuple(options.keep_flag), segment_bytes=int(options.segment_gib * GIB),
                          workers=options.workers)
    checked = check_repack(options.source, options.store, texts, keys, set(keys),
                           sample=VERIFY_SAMPLE if options.sample is None else options.sample, seed=options.seed)
    write_receipt(options.receipt, {"schema": SCHEMA, "command": "repack", "finished_utc": datetime.now(UTC).isoformat(),
                                    "seconds": time.monotonic() - started, "base": record["base"], "store": str(options.store),
                                    "input": str(options.input), "repack": record, "check": checked})


def parser() -> argparse.ArgumentParser:
    arguments = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    arguments.add_argument("command", choices=("embed", "pilot", "verify", "sweep", "push", "seed", "graft", "relocate", "repack",
                                               "attest", "qualify", "golden", "compare", "grow"))
    arguments.add_argument("--base", required=True, help=f"one of {sorted({**CANONICAL_BASES, **ENCODER_BASES})} or 300, 600, 6b, random, selfies, at")
    arguments.add_argument("--profile", help="the embedding profile (embedding_profile_v1) a profile store is computed under; "
                        "required by attest and qualify, and for every profile store")
    arguments.add_argument("--store", type=Path, help="the store root; a scratch directory for a pilot or qualification; "
                        "required by every command but attest and golden")
    arguments.add_argument("--golden", type=Path, help="golden: where to write the golden file, by default the repository's "
                        "(numerics.golden_path)")
    arguments.add_argument("--input", type=Path, help="parquet with the digest and sequence columns")
    arguments.add_argument("--digest-column", help="the column of row keys; default seq_sha256 (smiles_sha256 for the ligand base, "
                        "entry_sha256 for the annotation base, where a file without it is accepted)")
    arguments.add_argument("--sequence-column", help="the column of inputs; default sequence (smiles, terms)")
    arguments.add_argument("--prepare-workers", type=int, default=1, help="a non-protein base: processes that canonicalize the input")
    arguments.add_argument("--cache", type=Path, help="the Hub cache holding the verified pinned checkpoints")
    arguments.add_argument("--fetch", action="store_true", help="download the pinned checkpoints into --cache first")
    arguments.add_argument("--receipt", type=Path, help="the JSON the command writes; it must not exist")
    arguments.add_argument("--device", default="cuda")
    arguments.add_argument("--shard", default="0/1", help="i/n: the rows whose SHA-256 is i modulo n")
    arguments.add_argument("--limit", type=int, help="the first N rows of the shard, for a trial")
    arguments.add_argument("--skip-unrepresentable", action="store_true")
    arguments.add_argument("--max-sequences", type=int, help="sequences per batch; naming either this or --max-tokens "
                        "keeps the base's static budget for the other (the budget before memory-aware sizing)")
    arguments.add_argument("--max-tokens", type=int, help="padded token rows per batch; see --max-sequences")
    arguments.add_argument("--fixed-batch-size", type=int, help="opt-in fixed physical rows, repeat the last sequence in partial batches and discard duplicate outputs")
    arguments.add_argument("--random-weights", type=Path, help="exact saved random-control safetensors checkpoint; pretrained bases refuse this option")
    arguments.add_argument("--batch-fill", type=float, default=BATCH_FILL, help="a new store's batch budget fills this "
                        f"fraction of free device memory with activations, at most {BATCH_GROWTH_LIMIT} times the static "
                        "budget; 0 keeps the static budget. An existing store keeps the budget it started with")
    arguments.add_argument("--window", type=int, default=65536, help="sequences sorted by length together")
    arguments.add_argument("--part-gib", type=float, default=1.0)
    arguments.add_argument("--segment-gib", type=float, default=8.0)
    arguments.add_argument("--queue-gib", type=float, default=4.0, help="finished rows allowed to wait for the writer")
    arguments.add_argument("--writers", type=int, default=4, help="parts written and hashed in parallel")
    arguments.add_argument("--sample", type=int, help="pilot: sequences drawn, by default chunk * (steps + warmup) so "
                        f"every step is timed on a full chunk; verify: sequences recomputed, by default {VERIFY_SAMPLE} "
                        f"({EXACT_VERIFY_SAMPLE} for a profile store); compare: sequences read, by default {COMPARE_SAMPLE}")
    arguments.add_argument("--chunk", type=int, default=1024, help="pilot: sequences per timed step")
    arguments.add_argument("--steps", type=int, default=8)
    arguments.add_argument("--warmup", type=int, default=2)
    arguments.add_argument("--budget-hours", type=float, default=24.0)
    arguments.add_argument("--seed", type=int, default=1)
    arguments.add_argument("--tolerance", type=float, default=RELATIVE_TOLERANCE)
    arguments.add_argument("--hub", action="store_true", help="push: upload to the Hub dataset repo")
    arguments.add_argument("--volume", action="store_true", help="push: copy to the Modal volume")
    arguments.add_argument("--visibility", choices=("public", "private"), help="push --hub: the repo's visibility")
    arguments.add_argument("--hub-staging", help="push --hub of a profile store: upload into this repository instead, a "
                        "fresh copy renamed to the store's repository once complete; its catalog names the store's")
    arguments.add_argument("--expect-weights-sha256", help="refuse to run unless the model's weights hash to this; "
                        "carry it from the first machine's receipt (`model_state_sha256`), mainly for random-init")
    arguments.add_argument("--dry-run", action="store_true")
    arguments.add_argument("--pooled", action="store_true", help="work on the base's pooled store (sae_max and final_mean_var only, "
                        "no per-token write): embed, pilot, verify, sweep, push and seed")
    arguments.add_argument("--source", type=Path, help="seed, repack: the token store read from; compare: the store "
                        "compared with the profile store --store; grow: the complete store of the added proteins")
    arguments.add_argument("--hub-revision", help="grow: the 40-character revision of the published store's repository "
                        "whose store --store grows")
    arguments.add_argument("--keep-flag", action="append", default=[], help="embed, pilot, verify, repack: a boolean column of "
                        "--input; only rows with any of them true are read (repeat the option), as for a token store's "
                        "interaction proteins")
    arguments.add_argument("--list-revision", help="verify of a profile store, repack: the 40-character revision of the list "
                        "named by --input")
    arguments.add_argument("--list-repository", default=SEQUENCE_LIST_REPO, help="verify of a profile store: the dataset "
                        f"repository of the list named by --input, by default {SEQUENCE_LIST_REPO}")
    arguments.add_argument("--workers", type=int, default=3, help="repack: source parts read and packed in parallel")
    arguments.add_argument("--prune-hub", action="store_true", help="push --hub of a repacked store: delete the files it replaced "
                        "(they stay in the repository history)")
    return arguments


def main(arguments: Sequence[str] | None = None) -> None:
    options = parser().parse_args(arguments)
    if not 0 <= options.batch_fill < 1:
        raise SystemExit(f"--batch-fill must be at least 0 and under 1; received {options.batch_fill}.")
    if options.store is None and options.command not in {"attest", "golden"}:
        raise SystemExit(f"{options.command} needs --store.")
    if options.command == "sweep":
        sweep(options)
        return
    if options.command == "graft":
        graft(options)
        return
    if options.command == "relocate":
        relocate(options)
        return
    if options.command == "push":
        if options.hub and options.visibility is None:
            raise SystemExit("--hub needs --visibility public or private; it has no default.")
        if options.receipt is not None and options.receipt.exists():
            raise FileExistsError(f"{options.receipt} exists; a command never overwrites its receipt")
        push(options)
        return
    if options.command in {"attest", "qualify", "golden"} and not options.profile:
        raise SystemExit(f"{options.command} needs --profile.")
    if options.command in {"verify", "grow"} and options.profile and not re.fullmatch(r"[0-9a-f]{40}", options.list_revision or ""):
        raise SystemExit("A profile store's verify names the revision of its sequence list (--list-revision, 40 hexadecimal "
                         "characters), which the coverage record it writes publishes with the store.")
    if options.command == "grow" and (not options.profile or options.source is None
                                      or not re.fullmatch(r"[0-9a-f]{40}", options.hub_revision or "")):
        raise SystemExit("grow needs --profile, --source (the store of the added proteins) and --hub-revision (the "
                         "published repository's 40-character revision).")
    required = {"seed": ("input", "receipt"), "repack": ("input", "receipt"), "attest": ("receipt",),
                "qualify": ("cache", "receipt"), "golden": ("cache", "receipt"), "compare": ("source", "input", "receipt")}.get(
        options.command, ("input", "cache", "receipt"))
    for name in required:
        if getattr(options, name) is None:
            raise SystemExit(f"{options.command} needs --{name}.")
    if options.receipt.exists():
        raise FileExistsError(f"{options.receipt} exists; a command never overwrites its receipt")
    if options.command == "seed":
        seed(options)
        return
    if options.command == "repack":
        repack(options)
        return
    if options.command == "attest":
        attest(options)
        return
    if options.command == "compare":
        compare(options)
        return
    if options.command == "grow":
        from .hub import grow_metadata, read_grown

        recorded = read_grown(options.store) if options.store.exists() else None
        if recorded is None:
            grow_metadata(options.store, repository=read_manifest(options.source)["hub_repo"], revision=options.hub_revision,
                          source=options.source)
        elif (recorded["revision"], Path(recorded["source"])) != (options.hub_revision, options.source):
            raise SystemExit(f"{options.store} was grown from {recorded['repository']}@{recorded['revision']} and "
                             f"{recorded['source']}; grow into a new folder for another.")
    watcher = None
    if options.command == "pilot":
        from foundry.pilot import Pilot

        # Created first, so the time to the first step counts model loading, key checks and the contract.
        watcher = Pilot(steps=options.steps, warmup=options.warmup, items=len, item_unit="sequences",
                        budget_seconds=options.budget_hours * 3600, receipt=options.receipt)
    session = open_session(options)
    if session.profile is not None and options.command in {"embed", "pilot"}:
        session.golden = run_golden(session)  # before any row is written; a stream that differs stops here
    elif session.profile is not None:
        session.canary = run_canary(session)  # the whole panel, for a gate or a record
    if options.command == "golden":
        record_golden(session, options)
    elif options.command == "embed":
        embed(session, options)
    elif options.command == "pilot":
        assert watcher is not None
        watcher.total_steps = -(-len(session.texts) // options.chunk)
        pilot(session, watcher, options)
    elif options.command == "qualify":
        qualify(session, options)
    elif options.command == "grow":
        grow_exact(session, options)
    elif session.profile is not None:
        verify_exact(session, options)
    else:
        verify(session, options)


if __name__ == "__main__":
    main()
