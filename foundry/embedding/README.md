---
id: foundry.embedding
kind: library_module
status: active
aliases: [foundry.embedding]
---
# foundry.embedding

SAE and frozen structural features on top of the FastPLMs embedding API, the contract that
names a stored feature, and the canonical stores of ESMC and of Atlas's ligand and annotation towers.

**This module does not reimplement embedding.** `fastplms.embeddings` already owns the
ordered per-residue interface, dense pooling (`fastplms.embeddings.pooling.Pooler`), and
persistence (`runner.py`, `storage.py`, `pooling.py`, `types.py`). It is documented in the
FastPLMs `docs/embedding_api.md` and ships to Hugging Face as `model.embed_dataset(...)`.
FastPLMs also owns the SAE arithmetic, in `fastplms.models.esm_plusplus.modeling_esm_plusplus_sae`.

**This module owns only** what FastPLMs leaves to every caller: turning its flattened SAE
output back into sequences, pooling it without densifying, the SAE reducer for FastPLMs'
one-pass taps, the released structural projection reducer, the identity of a stored feature,
and verified training views over the existing store.

| Name | Does |
|---|---|
| `encode_topk(residue_states, sae_layer, chunk_rows=1024)` | Each residue's `k` strongest codes as `TopKFeatures`, by the SAE's own encoder arithmetic without the dense scatter its forward adds. Reads only `W_enc`, `b_dec`, `params.k`, so FastPLMs and Biohub layers both work; a tier-1 test holds it to FastPLMs' forward |
| `row_layout(token_mask, residue_mask)` | Sequence, position, and keep flag of every row FastPLMs flattened out of `token_mask`, for either padding side |
| `concatenated_rows(lengths)` | The same for residue rows already concatenated sequence by sequence |
| `interior_tokens(token_mask)` | Attended tokens other than each sequence's first and last (BOS and EOS under ESM tokenizers) |
| `pool_features(features, layout, n_sequences, pooling=("max",))` | Max, mean, or sum per sequence from dense, sparse COO, or top-k rows, reductions concatenated in order |
| `sae_tap(name, sae_layer, layer, pooling=("max",), chunk_rows=1024)` | A FastPLMs `ReducedTap` for `embed_dataset(..., taps=[...])`, so SAE features come out of the same forward pass as dense taps. It encodes hidden state `layer` over the tap's biological-residue mask with `encode_topk` and pools with `pool_features`: one float32 `(p * c,)` vector per sequence, never the dense `(n, c)` codes that `forward(compute_sae=True)` scatters. `layer` must equal the SAE's own `.layer` when it declares one. Its identity, which the run fingerprint records, hashes `W_enc` and `b_dec` when the tap is built |
| `feature_key(model, revision, *, sae_repo, sae_revision, layer, pooling, dtype, max_length)` | A `FeatureKey` for one stored feature: a readable `slug`, the SHA-256 `digest` of every field, and `name`, the filename-safe slug plus 16 digest characters. Refuses a revision that is not a 40-character commit. The SAE fields are both None for dense features. `layer` is non-negative, so resolve -1 first; `pooling` is None for per-residue rows |
| `plain_feature_key(owner, descriptor, *, slug=())` and `plain_feature(owner, descriptor, *, layout, width, dtype, positions=False, sparse_count=None, slug=(), store=None)` | The key, and the `StoredFeature`, of a feature the canonical key cannot describe: pooling that keeps BOS and EOS, argmax positions beside each code, a project's own encoder. The identity is a descriptor of plain data the caller states (`schema` names its format, never `feature_spec_v1`), and the key is `<owner>__<slug parts>__<16 digest characters>` of that descriptor, so two descriptors never share a name. Use it only where `feature_key` is wrong, and say why in the descriptor's `schema` docs. `store` is the module whose `StoredFeature` is built (`fastplms.features` by default); a process that loads Hub artifacts passes the private store below |
| `load_private_feature_store()` and `PRIVATE_STORE_NAME` | The feature store package loaded from the `fastplms/features` directory on `sys.path` under the module name `foundry_private_feature_store`, once per process, without importing `fastplms`. **For a process that loads Hub artifacts.** A published Synthyra model installs its own runtime as `fastplms`, refuses to load once another `fastplms` is imported, and ships no `features` package, so such a process can never `import fastplms.features`, in either order. Every caller gets the same module, so its classes agree; a process that also imports `fastplms.features` holds a second copy, and the two share plain data only. Everywhere else `import fastplms.features`. Atlas and confounders use it |
| `sae_residue_tap(name, sae_layer, layer, *, chunk_rows=1024)` | A `SparseResidueTap` returning paired `(r_i,k)` indices and values for every biological residue. Reuses `encode_topk`, preserves zeros, order and encoder dtype, and persists through the native `ragged_topk` layout. Pool after the consumer selects its crop |
| `FeatureSpec(specification)` | Immutable `feature_spec_v1` representation contract. Its full SHA-256 covers model/config/tokenizer contents, auxiliary checkpoints, reducer semantics, code/runtime policy, and cropping. `to_dict()` returns a detached JSON document; `.key` returns a `FeatureKey` |
| `SequenceKey(sha256, length, normalization_version)` | Identity from the already verified canonical inventory. Does not normalize or hash a second copy of the sequence |
| `FeatureSpec.bind(sequence, retained_positions)` | A `FeatureRowKey` binding the representation, original normalized sequence, inference prefix length, and exact zero-based retained positions. `FeatureRowKey.from_dict(record, feature=spec)` checks persisted rows against an independently supplied contract |
| `structural_tap(name, weights, compute_dtype=torch.bfloat16)` | A FastPLMs `StreamingTap` for the released ESMFold2 layer mixture: projects each layer to width 256, retains those narrow projections, then applies the reference einsum. The engine returns masked residue rows from the same pass as dense and SAE taps |
| `capture_feature_specs(model, taps, ...)` | Measures complete dense/residue, pooled/residue SAE and structural contracts without a forward pass. Resolves layers and rejects mismatched pins, stale taps, unresolved attention, and unsupported device/dtype arrangements |
| `FeatureExtractionContract(model, taps, inventory, **capture_options)` | Enforces those measured contracts through the existing FastPLMs writer. Supplies physical descriptors, validates cache hits, binds original residue positions, and recaptures live dependencies immediately before commit |
| `FeatureView(root, features, inventory, budget=...)` | Validates independent descriptors and sequence identities before batch reads. Supports dense/CSR pooled branches and aligned dense, structural and sparse top-k residue branches |
| `ReadBudget(max_rows, max_padded_residues, max_bytes)` | Caps returned batch tensors. `queued_bytes(workers=..., prefetch_factor=...)` estimates output queue capacity; it excludes model, interpreter, inventory and store-verification working memory |
| `LengthBucketBatches(lengths, max_rows=..., max_padded_residues=..., window_size=..., seed=..., special_rows=0)` | Batches observation indices in bounded sorting windows, retaining duplicate observations and returning indices for restoring evaluation order. `lengths` are residues `l`; pass `special_rows=2` for a canonical view, whose batches pad to `l + 2` rows |

## Embedding profile v1

Every canonical store from 2026-10-06 on is computed under one enforced profile, `profiles/embedding_profile_v1.json`
([decision](../../docs/decisions/foundry/embedding/2026-10-06_embedding_profile_v1.md),
[qualification](../../results/models/fastplms/2026-10-06_embedding_profile_qualification/README.md)). Its `function`
fixes everything besides the model and the protein that decides a stored value, and has one digest that every
`feature_spec_v3` key binds, so equal keys mean one function. Its hardware classes, their members, golden canary digests
and bounds record who may compute it; they grow without changing a key. `numerics.py` holds the profile and its checks.

| Part | Rule | Code |
|---|---|---|
| Environment | Only inside the `environments/embedding_runtime` image, whose loaded library closure must hash to the profile's digest for its platform (`linux/aarch64` on the GH200, `linux/x86_64` on Modal) | `attest_environment`, `check_environment` |
| Numerics | cuBLASLt preferred; reduced-precision and split-K reductions, FP16 accumulation, TF32 and autocast off; deterministic algorithms; pinned workspaces. Set, then read back | `apply_profile`, `measure_numerics`, `check_numerics` |
| Precision | FP16 backbone for ESMC-300, ESMC-600 and the random control, BF16 for ESMC-6B (FP16 headroom 3.5 times there); the SAE and pooling in FP32 | `EmbeddingProfile.backbone_dtype(base)` |
| Attention | memory-efficient SDPA, never a fallback | the token executor |
| Geometry | one batch shape per 64-token bucket (32,768 tokens, at most 256 rows), the same on every machine, so a protein's arithmetic never depends on its companions; every batch flag is refused | `fastplms.embeddings.BatchGeometry` |
| SAE readout | v1: `sae_topk_v2`, codes inside a band of `rho` (0.1) times the token's largest pre-activation above the cutoff fade to zero. v2: `sae_topk_canonical_v1`, the exact 64 largest, a tie at the cutoff chosen by lowest code index | `sae_features.ContinuousReadout`, `HardTopKReadout` |
| Extra streams | v2: `layer_mean_var` and `structural_mean_var`, the mean and variance (float32) of the SAE layer and of the structure fusion over all l + 2 tokens, in the pooled and token stores of each trained base | `canonical_taps(extra_streams=...)` |
| Hardware | a class is a compute capability and multiprocessor count. Only the reference class `hopper_sm90_132` (GH200, H100 SXM and NVL, H200) embeds, pilots, verifies and records goldens; v1's A100, A10, A10G, L40S and L4 classes are tier 2 and may only qualify, and v2 lists Hopper alone | `device_class`, `check_reference_class`, `check_live_class` |
| Golden | before every `embed` and `pilot`, one full batch in each of three buckets (64, 1,088 and 2,048 tokens) is held bitwise to the golden file in `profiles/golden/<profile>/<base>.safetensors`: every stream's digest, and reference rows of the first protein of each batch. On a GH200 under v2: 0.8 s on ESMC-300, 1.1 to 1.4 s on ESMC-600, 4.5 to 4.9 s on ESMC-6B. Recorded once on the reference class by `job golden` | `golden_panel`, `check_golden` |
| Canary | a fixed 1,810-protein panel filling every bucket runs before `verify`, `qualify` and `golden`; its stream digests must equal the class's golden ones | `canary_panel`, `check_canary` |
| Verify | exact: every list row present in every stream, and a bucket-stratified sample recomputed in shuffled batches equal to the stored rows bit for bit. It writes the store's `coverage.json`: the list's repository, revision (`--list-revision`, required), file SHA-256 and kept flag, and the digest of the store it passed (`hub.store_sha256`) | `job verify`, `write_coverage` |
| Grow | extends a published store with the proteins a newer list adds while holding only the new parts, for a store too large to copy (3.1 TB for the ESMC-6B token store). The grown store downloads the published metadata at `--hub-revision` (manifest, descriptors, every commit marker), checks it against the published coverage's store digest, and hard-links the segments of `--source`, a complete store of just the added proteins; `grown.json` records the published sizes and digests of the parts it does not hold. The gate accounts for every list protein through the markers, recomputes a sample of the added proteins bit for bit and writes the coverage record, with the published one under `grown_from`; `push` uploads only the local files | `job grow`, `hub.grow_metadata`, `grow_exact` |
| Publish | only stores of the reference class under the qualified, unchanged profile, with a current coverage record (`check_coverage`), to `Synthyra/esmc_embeddings` (public models) or `Synthyra/private_embeddings`, at `<model>/<kind>/`, one canonical place whatever the profile, which a store under a new profile replaces (`job relocate` moves a store from an older folder; a rebuild fills a fresh repository with `push --hub-staging`, which then replaces the old one under its name), with shard and row indexes, and to the model's own Modal volume (`canonical-embeddings-<model>`); the catalog names the list revision each store covers; with `--receipt`, the push records the catalog entry and its revision | `check_publishable`, `hub.py` |
| Compare | how far another store of a base (the v2 store a profile store replaces, or another class's) lies from a profile store, per stream on a bucket-stratified sample: the panel statistics, the quantiles of the proteins' relative L2, the proteins that moved most, and for code streams the ratio of values on shared codes. CPU only; changes neither store | `job compare`, `protein_differences` |

A device is held to its class four times: `embed` refuses a device outside the reference class or not enrolled in it, or
a base its class does not cover; the golden batches must match before the first row; the device is measured again before every commit; and a store
made by one class is neither extended nor verified by another. A device of a listed shape that is not a member, or of
no listed class, may only `attest`, `pilot` and `qualify` (the latter under a provisional class), and joins a class when an enrollment record
of its receipts is applied to the profile (`experiments/models/fastplms/2026-10-06_embedding_profile_qualification/scripts/enrollment.py`).
A change to the function is a new profile and a rebuild; bounds are never relaxed.

Run it through `python -m foundry.embedding.launch --compute ssh:<gh200>|modal --modal-gpu H100!|H200 -- <job command>`;
the launcher sets `EMBEDDING_PROFILE` in the container to the profile the job names. Modal's `H100!` may be an SXM or NVL
card, both Hopper with 132 multiprocessors; no other Modal GPU is offered.

## Canonical token stores (special tokens kept)

Logan's embedding spec of 2026-10-01 ([record](../../docs/decisions/models/fastplms/2026-10-01_canonical_embeddings_keep_special_tokens.md)):
a protein of `l` residues, cropped from the N-terminus at 2,046, is `l + 2` rows in every per-token stream
(row 0 CLS, rows 1..`l` residues, row `l + 1` EOS), padding is the only masked position, and pooling covers
all `l + 2` rows. Descriptors are `feature_spec_v2` with `special_tokens: kept`. A legacy `feature_spec_v1`
store holds `l` residue rows and stays readable; the two never share a key.

| Name | Does |
|---|---|
| `canonical_base(name)` and `CANONICAL_BASES` | The four stored bases (`esmc_small` d=960 layer 23, `esmc_large` d=1152 layer 27, `esmc_6b` d=2560 layer 60, and the random-init control `esmc_small_random`), each with its streams, public Hub dataset repo, Modal volume and top-level directory, and recommended batch sizes. The control's seed (20261001) is recorded in its model pin |
| `sae_tap(..., keep_special_tokens=True)`, `sae_residue_tap(...)`, `structural_tap(...)` | The token variants of the taps. They run in FastPLMs' token executor (`row selection` and a per-batch cache supplied), so the pooled and per-token SAE taps encode each batch once. Their identity says `rows: all_attended_tokens`, and `capture_feature_specs(keep_special_tokens=True)` refuses a residue tap |
| `capture_feature_specs(..., keep_special_tokens=True)` | Measures `feature_spec_v2` contracts. `FeatureExtractionContract` with that option binds rows by key (`row_identities`) instead of engine records |
| `open_canonical_view(root, inventory, streams=None, deep=False)` | A `FeatureView` over a base's streams from its `canonical.json`: `lengths` is `l`, `rows` and `residues` return `l + 2` rows, a crop is a residue interval that keeps CLS and EOS, `pooled` joins `final_mean_var` and `sae_max`. Opening reads no row: it costs index lookups, one sidecar parse per part, and one verification of each part holding a requested row, recorded in a receipt so the next open, and every worker, skips it ([below](#opening-cost-and-receipts)). `deep=True` is the check the job's `verify` runs |
| `sequence_inventory(texts)` | Normalized text to `SequenceKey`. The one rule that makes a row key is `normalize_sequence` (ASCII whitespace removed, uppercased; the dataset census rule), so two spellings of a protein are one row |
| `SyntheticControlView(kind, inventory)` with `RANDOM_VECTOR` or `ONE_HOT` | Controls generated where they are read and never stored: the same `l + 2` layout, `final_hidden` and `final_mean_var` streams, seeded by the row's SHA-256, so `rows` and `pooled` agree |
| `pool_all_rows(features, owner, counts, pooling)` | SAE max, mean or sum over every token row of a batch with no device read |
| `python -m foundry.embedding.job` | `embed`, `pilot`, `verify`, `grow`, `sweep` and `push` over a parquet of `(seq_sha256, sequence)` rows, below; `attest`, `qualify` and `compare` for profile stores, above |
| `choose_batch(base, root, explicit=..., free_bytes=..., fill=...)` | The batch budget of a run (`BatchPolicy`): the operator's flags, else the store's recorded one (`recorded_batch`), else a new store's share of free device memory, else the base's static one |

Streams of a trained base: `layer_hidden` `(l+2, d)` bf16, `final_hidden` `(l+2, d)` bf16, `structural`
`(l+2, 256)`, `sae_codes` `(l+2, 64)` indices and values, `sae_max` `(16384,)` fp32 sparse, and
`final_mean_var` `(2d,)` fp32. The control holds `layer_hidden`, `final_hidden` and `final_mean_var`.

**The job.** `ws run production_suite -- python -m foundry.embedding.job <command> --base 300|600|6b|random
--store ROOT --input seqs.parquet --cache HF_CACHE --receipt out.json` (`ws run foundry` cannot start:
`foundry/logging` shadows the standard library, so run it through a project that imports foundry).

- `embed` is resumable (a rerun embeds only the rows a stream lacks, from the last committed segment) and
  shardable (`--shard i/n` takes the rows whose SHA-256 is `i` modulo `n`; shards on separate machines each
  push their `segments/` and a reader rebuilds the index from the commit markers). `--fetch` downloads the
  pinned checkpoints into `--cache` first, and every file is verified against its registered digest before
  loading. `--skip-unrepresentable` sets aside proteins with a residue the tokenizer lacks (ESMC has no `J`)
  and records them; without it they stop the job. Before any model work, every command compares the weights it
  built with the store's first build (`model_state_sha256` in `canonical.json`) and with
  `--expect-weights-sha256`, so a random-init draw that differs on another machine stops the run instead of
  mixing two networks; carry the digest from the first machine's receipt to the others.
- `pilot` times chunks of a random sample into a scratch store through `foundry.pilot` (`--sample`, `--chunk`,
  `--steps`, `--warmup`), prints the projected run, GPU utilization and peak memory, and writes the pilot
  receipt. It refuses a root that already holds a store. The sample defaults to `--chunk` times `--steps` plus
  `--warmup` (10,240 sequences), so every timed step has a full chunk; a `--sample` or a list too small for them is
  refused rather than timing fewer steps.
- `sweep` deletes the uncommitted segment directories a killed run left, in each stream that has any, and does
  nothing on a clean store; `push` refuses a store with such directories and prints the exact `sweep` command. It
  needs no model, only `--base` and `--store`, and stops with an error naming a segment a running writer holds.
- A new store's batch budget (sequences and padded tokens per batch) takes `--batch-fill` (default 0.5) of the free
  device memory once the weights are loaded, at most 8 times the base's static budget and never below it; `0`
  keeps the static budget, and naming `--max-sequences` or `--max-tokens` keeps it for the other. The budget is part of
  every stream's key, so a store keeps the one it started with: `embed` and `verify` read it from the store, and
  `embed` refuses a different one. The per-token cost behind the sizing is one pilot point per base, still to be confirmed on an idle GPU
  (`experiments/models/fastplms/2026-10-01_canonical_embeddings_v1/launch.md`).
- **What a stream key holds.** A key is the SHA-256 of the whole descriptor, so only what changes a stored value may
  be in it: the model's weights, config and pin, the tokenizer, the layer, the stream definition and its reducer
  files (the bytes of the code that reduces rows), the crop, the special-token policy, the dtypes, the numerical
  settings (TF32, autocast, deterministic mode) and the batch budget. The engine digest, the library versions and the
  machine only say where the first rows were made. A new store records them, and every later run on the store keeps
  them (`inherit_recorded_evidence`), so another tree or GPU derives the keys the store has. The digest covers
  `ENGINE_EMBEDDING_FILES` (batch building, tokenization, taps, pooling, the token executor), `attention/` and the
  ESM++ model, never the store, writer, page-cache or identity code. A run whose keys still differ stops before any
  write, naming each changed descriptor field (`check_keys_match_store`). `scripts/check_store_keys.py` in the
  [launch record](../../experiments/models/fastplms/2026-10-01_canonical_embeddings_v1/scripts/check_store_keys.py)
  reads a store and compares the keys the code derives; `tests/tier1_unit/test_embedding_published_keys.py` pins the four
  published stores' keys.
- `verify` accounts for every input row in every stream, reads a sample back through `open_canonical_view`, and
  recomputes it independently with the token executor (relative L2 error at most `--tolerance`, 0.02, the
  bf16 reordering allowance).

### Numerical identity amendment, 2026-10-06

Superseded for new stores by [embedding profile v1](#embedding-profile-v1); it still describes v2 captures. New captures record the live FP16/BF16 reduced-precision reduction flags, their separate split-K flags and the
preferred BLAS library. These settings affect the computation and therefore the feature key. PyTorch 2.14's
boolean reduction setter leaves split-K enabled; `(False, False)` disables both. Reading the ordinary boolean
getter alone does not capture the split-K policy. Older Torch versions without separate split-K getters record
null rather than inventing evidence.

Legacy descriptors and their published hashes remain readable and unchanged. Missing historical flags are
unknown, not inferred defaults. New capture keys differ from those legacy keys and must not be substituted into
old manifests or appended to an old stream. Reuse under a new numerical profile requires an explicit qualified
migration; a source or machine receipt alone does not establish numerical equivalence.

The [solubility/expression diagnosis](../../results/serving/production_suite/2026-10-06_solubility_expression_probes/README.md)
reproduced batch sensitivity inside one pinned Docker image. Same-batch repetition was exact, while singleton
and changed-companion batches could exceed the unchanged 2% limit. A bounded pooled-feature candidate using
cuBLASLt, disabled BF16 reduced-precision and split-K reductions, and duplicate padding to eight passed short
and length-stratified comparisons across all three trained bases on A6000. It is not yet a shared extraction
default: larger representative samples, production batch budgets, per-token streams and cross-hardware checks
remain required before integration and store migration. Preserve failed receipts and independently qualify
each affected stream and downstream head. Docker is mandatory for all qualification and rebuild work.
- `push --hub --visibility public|private` uploads committed segments to the base's Hub dataset repo
  (`upload_large_folder`, resumable, with the README card; run again after the account's rate limit stops it,
  through `foundry.downloads.hub.upload_resuming`), and `push --volume` copies the files the Modal
  volume lacks under the base's directory. `--dry-run` counts the committed segments and checks nothing is
  half-written first.

### Pinned dependencies

`dependencies` selects and loads the four pinned artifacts behind a canonical store: the ESMC base, its tokenizer, the Biohub SAE of
the base's depth layer, and the released ESMFold2 layer mixture and projection. It reads the FastPLMs model registry (never a Hub
lookup or a fallback) and takes no suite manifest, so a job with only a base id can run.

| Name | Does |
|---|---|
| `resolve_dependencies(registry, base_id, *, sae_layer, sae_k=64, sae_codebook=16384)` | The base, tokenizer, SAE and structural `CheckpointSource`s as `CanonicalDependencies`, each checked complete and immutable. A caller's own manifest may state another `k` or codebook |
| `checkpoint_identity(source)` | The repository, 40-character revision and every file's registry digest, refusing an unresolved, mutable, duplicated or unparseable inventory |
| `fetch_snapshot(source, cache)`, `verified_snapshot(source, cache)` | Download the pinned files into a Hub-layout cache, and check every cached byte (Git blob identities included) against its digest |
| `load_sae(spec, snapshot, *, device, dtype)` | Verify the SAE's bytes and its config against the registered dimensions, then load it frozen in eval mode |
| `structural_weights(snapshot, *, width, depth)` | Only the released layer mixture and projection, shape- and finiteness-checked |

A failed contract raises `DependencyContractError`, a `ValueError`. `production_suite.feature_dependencies` calls `verified_snapshot`,
`structural_weights` and `load_sae` for its runtime functions and re-raises `DependencyContractError` as the suite's `SuiteError`.
`dependencies` imports no torch (torch and FastPLMs load inside the functions that need them), and neither does the package: see
[lazy exports](#lazy-exports). So `production_suite.feature_dependencies`, whose planning layer runs with no torch, wraps this module
instead of copying it: `FeatureDependencies` subclasses `CanonicalDependencies`, and `resolve_feature_dependencies`,
`checkpoint_identity`, `verified_snapshot` and `load_sae_dependency` call `resolve_dependencies`, `checkpoint_identity`,
`verified_snapshot` and `load_sae`, raising `SuiteError` where these raise `DependencyContractError`.

Tests: `tests/tier1_unit/test_embedding_dependencies.py`, with the suite's `tests/test_feature_dependencies_foundry.py` from that side.

### Lazy exports

`import foundry.embedding` imports no submodule, so it imports no torch. `__init__.py` exports every name in `__all__` by PEP 562:
`EXPORTS` maps each to the submodule that defines it, `__getattr__` imports that module on first access and caches the value,
and `__dir__` lists `__all__`. `foundry.embedding.<submodule>` loads the submodule the same way, so code that reaches a submodule
through a bare `import foundry.embedding` still works. The `if TYPE_CHECKING` block carries the real imports for type checkers and
for the publish import closure (`tools/workspace/import_closure.py`), and a test holds it equal to `EXPORTS`. No submodule runs
code at import beyond defining names and constants (no registry fills, backend flags, environment or `sys.path` changes), so
deferring them changes nothing a consumer could observe except which modules a bare package import loads.
Test: `tests/tier1_unit/test_embedding_lazy_exports.py` imports the package and `dependencies` in a child that refuses torch.

## Pooled stores and interaction-only token stores

Logan, 2026-10-01: per-token streams are for interaction data (PPI, PLI, protein-protein affinity, CAMP); everything else is
pooled-only, because a full per-token store is too big to download
([the record](../../docs/decisions/models/fastplms/2026-10-01_canonical_embeddings_keep_special_tokens.md#pooled-stores-and-interaction-only-token-stores-2026-10-01)).
Each base has two stores, chosen by `--pooled` on `embed`, `pilot`, `verify`, `sweep` and `push`:

| Store | Streams | Rows | Where it lives |
|---|---|---|---|
| token (default) | every stream of the base | the proteins of the list flagged `in_ppi_superset` or `in_ppa` (repacked) | `<model>/token/` of `Synthyra/esmc_embeddings` and of the `canonical-embeddings-<model>` volume |
| pooled (`--pooled`) | `sae_max` and `final_mean_var` (the random control: `final_mean_var`) | every protein of the list | `<model>/pooled/`, likewise |

`CanonicalBase.profile_destinations(kind)` names that place. `.store_destinations(kind)` still names the v2 stores'
per-base repositories and volume directories (`Synthyra/esmc_{300,600,6b,300_random}_{token,pooled}_embeddings`,
`canonical_v2/`, `canonical_pooled_v2/`), which v2 manifests record as provenance; all were deleted on 2026-10-08.

| Name | Does |
|---|---|
| `CanonicalBase.pooled_streams`, `.store_streams(kind)`, `.store_destinations(kind)` | The pooled streams of a base, the streams of a `"token"` or `"pooled"` store, and its v2 Hub repo and volume directory (deleted, provenance) |
| `canonical_taps(..., pooled=True)` and `build_runtime(..., pooled=True)` | Only the pooled taps. A pooled runtime loads no structural projection and builds no per-token tap |
| `pooled_manifest(base, token_manifest)` and `manifest_kind(manifest)` | The manifest of a pooled store seeded from a token store (its stream records copied, `store_kind: pooled`), and the kind a manifest says (every manifest from before pooled stores is `"token"`) |
| `key_batch(root, run, pooled=...)` | The batch budget a run's stream keys carry. A pooled stream has the key of the token store's stream of that name, which states the seed's budget, so a pooled run keeps that budget in its keys and batches at the budget it fitted to free memory (`choose_batch(..., pooled=True)`). Each segment's `run.json` records the budget that made it. `FeatureExtractionContract(free_run_batch=True)` is what lets the two differ: it still binds every other extraction option |
| `pooled.seed_pooled(base, source, target)` and `pooled.check_seed(...)` | Copy the pooled stream directories of a token store into a new pooled store, re-hash every part against its commit marker, write the manifest and `seeded_from.json`; then account for every list row in every pooled stream and compare a sample of values with the source, bit for bit |
| `repack.repack_store(source, target, keep, ...)` and `repack.check_repack(...)` | Write a token store holding only the rows in `keep` (every stream, from the source's parts, identities carried over, keys unchanged), then re-hash every part, require exactly the kept rows in every stream and compare a sample with the source |

`job seed --pooled --source TOKEN_STORE --store POOLED_ROOT --input list.parquet` and `job repack --source TOKEN_STORE
--store NEW_ROOT --input list.parquet --keep-flag in_ppi_superset --keep-flag in_ppa --list-revision SHA` write receipts and need no
model. `job push --prune-hub` (a repacked store only, which has `repack.json`) uploads the store and then deletes the files it
replaced from the Hub repo, in commits of at most 500; the earlier revision keeps them. A pooled embed writes nothing per token, so
`--batch-fill` can take most of the free device memory.

## Ligand and annotation stores

Logan, 2026-10-02: "just like ESMC we should have an embedding store for the SELFIES language model and Annotation Transformer
we use" ([the decision](../../docs/decisions/foundry/embedding/2026-10-02_ligand_and_annotation_stores.md)). Atlas has two towers
that are not protein models, and each is a base of the same store: the same `feature_spec` descriptors, row key by the SHA-256 of a
canonical text, token and pooled stores, `FeatureView`, writer and `job`. `canonical.ENCODER_BASES` holds them (the four ESMC bases
stay in `CANONICAL_BASES`); a base's `family` says which, and `--base selfies` or `--base at` selects one.

| | `selfies_ted_large` (`ligand`) | `at_experimental_v1` (`annotation`) |
|---|---|---|
| Model | `lhallee/SELFIES-TED-LARGE` at `259f46e840edca8fbc6963b17972f0cd004469e3`, a BART of d=1,024 and 12 plus 12 layers, fp32 | `Synthyra/AT-experimental-v1`, trained on the experimental annotation vocabulary release (`Synthyra/annotation_experimental_v1` at `ee05eb945d86bf506a8ee4753d5d4465abd69a34`, 22,464 terms, ids 0 to 22,463, then pad, bos, eos, mask and unk); the model pin is registered in `encoder_pins` when its checkpoint is chosen by validation |
| A row | one molecule | one annotation entry |
| Key text | canonical SMILES (`canonicalize_smiles`) | distinct term ids ascending, joined by commas (`annotation_terms_v1`), which is the release's `annotation_id` |
| Units `l` | SELFIES symbols, cut after a whole symbol at 1,022 (`truncate_selfies`) | terms, cut after the 1,024th smallest id |
| Rows per sequence | `l + 2`: CLS, symbols, SEP (`feature_spec_v2`) | `l`: the model has no special tokens (`feature_spec_v1`) |
| Token streams | `encoder_hidden`, `decoder_hidden` (bf16) | `final_hidden` (bf16) |
| Pooled streams | `encoder_mean_var`, `decoder_mean_var` (fp32, mean then population variance over every row) | `final_mean_var` |
| Hub repos (private) | `Synthyra/selfies_ted_large_{token,pooled}_embeddings` | `Synthyra/at_experimental_v1_{token,pooled}_embeddings` |

**What Atlas stores is the decoder.** Atlas calls `AutoModel`, which is `BartModel`, with `input_ids` and `attention_mask` only, and
reads `last_hidden_state`. `BartModel` then shifts the ids right to make `decoder_input_ids`, so that state belongs to the
decoder, teacher forced on the molecule it reads. `decoder_*` is exactly that call (a test compares them) and `encoder_*` is the
encoder's `last_hidden_state`, which is what the SAE project reads; one forward gives both. Atlas stores float32 rows; the store
keeps per-token rows in bf16 like the ESMC stores and pools in float32 from the float32 states. Atlas cut 1,024 symbols and let the
tokenizer drop the tail, so its rows differ from the store's only for a molecule over 1,022 symbols.

| Name | Does |
|---|---|
| `encoders.canonical_inventory(base, values)` | The `{canonical text: SequenceKey}` that `open_canonical_view` reads one of these stores with, from raw SMILES or term-id lists in any spelling. Ask the view for a row by `family_of(base).canonicalize(value).text` |
| `encoders.EncoderExtractionContract` | `FeatureExtractionContract` with the capture of a Hugging Face model with no taps (`encoder_feature_specs`): `layers` is the stack's depth, `reducer.parameters.stack` is `encoder`, `decoder` or `final`, and the tokenizer pin is the checkpoint's tokenizer files (or the vocabulary release) with their digests |
| `encoders.embed_encoder_features(...)` | `embed_token_features` for these families over the same `AsyncFeatureWriter`: sequences longest first inside windows under a padded-row budget, rows copied to pinned buffers, every stream committed together |
| `encoders.read_family_inputs(...)` | The distinct canonical rows of a parquet in file order, over worker processes; a stated key column must equal the SHA-256 of the canonical text, and `--skip-unrepresentable` sets aside a SMILES RDKit or SELFIES cannot write |
| `encoders.write_row_inventory(root, rows)` | `inventory/<digest>.parquet` under the store: each row's `sha256`, `text`, `units`, `rows`, `truncated` and, for a ligand, its `selfies` and `unknown_tokens`. The feature store keeps only a digest and a row count, so this is how the Hub repo says what a row is |
| `encoder_rows` | The code that decides a stored value, hashed into every reducer digest: batch planning, `EncoderExecutor`, pooling. Edit it only when the rows are meant to change |
| `ligand`, `annotation` | A family's canonicalizer, tokenizer, forward and loader: `canonical_ligand`, `SelfiesTedHooks`, `canonical_annotation`, `AnnotationHooks` |
| `encoder_pins` | The pinned checkpoint of each base: a 40-character revision and the SHA-256 of every file, in the registry's `CheckpointSource` form so `dependencies.verified_snapshot` checks them |

The `job` commands are the ESMC ones with a different input: `--input` is a parquet whose `--sequence-column` (default `smiles`
or `terms`, a list of term ids or one string) holds the inputs, and an optional `--digest-column` (default `smiles_sha256` or
`entry_sha256`) must equal the SHA-256 of the canonical text where present. `--prepare-workers N` canonicalizes over N processes.
`embed`, `pilot`, `verify`, `sweep`, `seed --pooled` and `push --hub --visibility private` work as above; `repack` and `push --volume`
are protein only. `--skip-unrepresentable` also sets aside a ligand whose fragment separator sits on a symbol outside the vocabulary
(`[O-1].[Xx+1]` reads as one `[UNK]` for two symbols, so it would not be `l + 2` rows), and the receipt lists those keys.

Not here: the ligand and annotation caches Atlas writes itself (`embed.feature_cache`) are read by their own code until the
adapter moves them to `open_canonical_view` (`TODO(suite_pli_adapter)`).

## Verified training views

`FeatureView` takes a mapping of branch names to independently qualified `StoredFeature`
descriptors, and a mapping of normalized sequence text to `SequenceKey`. Each physical
descriptor contains its complete `FeatureSpec`; dense widths must come from the verified
backbone/configuration record. Loading the candidate cache's own descriptor as the expected
contract would defeat this comparison. The view requires no loaded encoder. It verifies all
selected rows, including finiteness, before returning from construction, unless it is built
with `verify_rows=False` (what `open_canonical_view` does, [below](#opening-cost-and-receipts)).
Reconstruct its expected descriptors from the pinned extraction release, not from feature filenames.

`rows(name, sequences)` returns native tensors, compressed `SparseRow` values or residue `TopKRow` values. `pooled`
joins selected branches into an FP32 matrix, while `residues` returns padded values, a Boolean
biological-residue mask and integer lengths. Both preserve request order and duplicates.
Residue branches must share an inference-prefix length. Their independently selected
contracts establish the intended models, layers and mask policies. Empty selections retain
known widths. Sparse residue codes stay `(b, l, k)`: values in `ResidueBatch.values`, int32
indices in `ResidueBatch.indices`. The reader validates k, codebook width, dtype, row identity
and nonnegative magnitudes. See the native [store contract](../../models/fastplms/docs/feature_store.md#row-layouts).

`residues(..., crops=[(start, stop), ...])` applies the same original, zero-based half-open
interval to every branch and returns the starts, lengths and masks. Each crop must be nonempty
and inside its verified inference prefix. Values and indices pad with zero. Row and padded
batch budgets include indices; `reserve_bytes` charges a consumer's additional output before
allocation. Source rows are still read in full under their own limit, so cropping cannot bypass
that limit. Atlas's [native adapter](../../projects/interaction/atlas/atlas_retrieval/README.md)
constructs its actual `CachedFeatures` and computes the SAE maximum from precisely this crop.

Readers use `FeatureStore.read_only` and the store's physical validation methods. They neither
embed sequences nor own a second SQLite implementation. Missing indexes must be recovered
with a writable store before workers start. Missing rows, unreadable/corrupt stores, changed
descriptors, wrong row identities and incompatible inference crops raise. By default each batch reopens
the descriptor and checks committed data and metadata again; a preflight does not authorize
later unchecked reads. The shared `checked_row_identity` function also guards extraction
cache hits and commits, so the original sequence and retained-prefix checks have one owner.

`FeatureView(..., persistent_reader=True)` retains one native `FeatureReader` per branch and
the sequence identities checked during full initial qualification. It uses the native mapped
row reads and read-only index connections. Each immutable part and its metadata are verified
once per reader lifetime. Do not modify already verified part files while this view is open;
use the default mode when each batch must detect changes to those files. Independent content
pins are enforced when a part is first opened. Pickling uses the native reader contract, which
drops live connections and mapped handles; spawned workers reopen and reverify the parts they
touch, except for parts a receipt vouches for. `view.close()` or a context manager releases the
handles and refuses further reads.

### Opening cost and receipts

`foundry.embedding.subset.materialize_subset` stages only requested rows from an immutable
Hub revision when whole parts would exceed local storage. It uses the canonical range reader
and native segment writer, retains the original manifest, descriptors and complete row identities,
and supports every canonical token and pooled layout. Source part order and a tensor byte cap
bound reads; committed hits and durable range/value audits support resume. A cold canonical view
then verifies local part hashes and every copied value. Receipts distinguish referenced source
whole-part hashes from verified received ranges and local bytes. The subset is a writable local
store with no published-coverage claim; qualified canonical embedding jobs may append absent rows
under the unchanged keys. The HTTP range reader checks status and response bounds before
receiving the body, and `read_rows(addresses=...)` reuses selected locators across bounded reads.

Verifying a part hashes it and loads its tensors, and an inventory spread over a large store touches
most of its parts (the ESMC-300 `layer_hidden` stream is 158 GB in parts of 1 GiB). Verifying rows on open
therefore cost about 9.6 minutes per Dyad fusion open (272, 162 and 144 s for three streams, 29,049
proteins), and each cell opened twice. `FeatureView(..., verify_rows=False,
receipt_directory=...)`, which `open_canonical_view` uses, qualifies instead from what the store already
states, in work proportional to the request:

- every inventory key is looked up in the index and agrees with its commit marker, with the stored row count
  the contract implies (`l + 2`); a missing row raises `KeyError`;
- each part holding a requested row is verified in full once (hash against its marker, layout, sidecar),
  four parts at a time, or accepted from the receipt `<root>/.cache/open_receipts/<manifest sha256>/<key>.json`;
- each requested row's identity is checked against its part's sidecar, one parse per part;
- no tensor row is read and no value is scanned on open; the first read of a row scans it once.

A receipt records, per part, the marker's digests and the part's and sidecar's size and modification
time. A part passes when all of them match, so a changed manifest (another receipt directory), marker,
sidecar or part is verified again and fails if it differs. A receipt is one machine's cache: it cannot see
a rewrite that keeps a file's size and modification time, and a copy to another machine verifies once.
`open_canonical_view(..., deep=True)`, which `python -m foundry.embedding.job verify` uses, ignores the
receipts, hashes every touched part, reads and scans every requested row, and refreshes them. A store
opened with content pins never trusts a receipt. A read-only root cannot keep receipts and warns.

Embedding-based training uses persistent views in Dyad, the suite's `NativeAtlasFeatures`
and `ObservationFeatures`, and Protify's `FrozenFeatureInput.joined`. These cover ordinary
Atlas retrieval, projected Oracle heads and the unserved Translator's frozen-residue bridge.
Automatic Oracle preparation also uses a scoped persistent view for row qualification.
`ObservationFeatures(..., persistent_reader=False)` keeps per-read verification for mutable
inputs; the general `FeatureView` default remains unchanged. Fitting requires immutable
committed parts, with existing independent pins retained across worker reopening.

The other audited training consumers already retain native readers or materialize features
once: Atlas precomputed/fused loaders, confounders, annotation vocabulary, SAE-XGBoost,
enzyme-loop probes, base-model distillation and embedding translation. Protein-screening
training uses Protify or a native reader snapshot. Do not replace these with per-batch store
opening. When adding a frozen-feature training path, keep its verified reader for the fit
and release its handles at the end; a spawned worker must reopen and reverify pinned parts.

`view.content_pins()` freezes the selected descriptor, segment commit markers, data parts and
row-metadata sidecars after qualification. Pass the resulting mapping as `content_pins=...`
when reopening a view, retaining its independently supplied descriptors and inventory. The
native store then enforces those hashes during every default read, or on each persistent
reader's first access to a part. The mutable SQLite index and
unrelated segments are excluded; their physical associations are still checked. Pins establish
byte identity, not extraction origin. Protify's separately pinned input document persists these
pins with branch order, a dataset-manifest digest and loader limits.

Pinning selects the complete inventory in one native call, so parts shared by consecutive
output batches are hashed once in that call. Read budgets still bound feature tensors;
selection metadata scales with inventory size. Qualification views used only to prepare or
authenticate inputs are closed immediately afterward.

Only sequence metadata, checked identities, immutable descriptor JSON, optional content pins,
and native reader configuration survive pickling. Output tensor memory
is bounded by `ReadBudget`, and residue padding is charged before allocation. This is not a
total-process memory cap: immutable-part validation, commit metadata, Python/ML imports,
standardizer state, labels, model activations and transient casts consume additional memory.
Inventory and commit metadata still scale with sequence count. Keep writer parts bounded,
and include these costs plus worker prefetch queues in the training process budget.

Protify's `IndexedBranchEmbeddings` implements the existing `JoinedBranchDataset` and
`JoinedBranchCollator` interface. It retains sparse labels and densifies feature codes only
for the batch. `fit_joined_standardizer` visits training rows in blocks bounded by the view's
row and byte limits. Calling the old `join_branch_embeddings` still uses the legacy eager
cache path. Declared projected-head cells explicitly select the indexed adapter with the
paired `feature_manifest` and `feature_sha256` settings described in
[Protify's input guide](../../projects/tooling/property_lowcode/README.md#frozen-native-feature-inputs).

The suite's `ObservationFeatures` bridges a previously verified `SequenceInventory` and
ordered `(observation_id, protein_role)` references to this API. It preserves measurement
identity independently of feature deduplication, and reuses the feature-demand inventory
checks. It does not certify source labels, scientific acceptance or split eligibility.
Its `joined_branches` method supplies Protify's existing collation interface; `atlas` joins
observation roles to the selected Atlas input, including explicit crops and independent content
pins. `translator` supplies final-layer residue states and Boolean masks to the unserved
V2 frozen-input path through its [native reader](../../projects/serving/api/README.md#frozen-translator-inputs).
The generic `residues` method remains available for other residue consumers.

Maintained qualification lives in `tests/tier1_unit/test_embedding_readers.py` and the suite's
`tests/test_feature_readers.py`, `tests/test_atlas_feature_reader.py` and
`tests/test_translator_feature_reader.py`. The suite's
`tests/run_feature_reader_memory.py` (add `--atlas` or `--translator` for residue consumers) compares
aggregate parent-plus-worker Linux USS/PSS/RSS at fixed batch and prefetch sizes, with an
intentional eager-copy positive control. This is synthetic infrastructure evidence, not a
biological performance or production-throughput claim. All qualification for this run executes
on the authorized GH200; consumer rollout remains tracked under `suite_feature_readers`.

## Legacy feature keys

Key schema 2 assumes FastPLMs' biological-residue policy for both dense and SAE features.
SAE tap schema 2 drops all excluded positions, including internal special tokens, and rejects
empty residue sets. SAE encoding retains its reference arithmetic: per-residue centering and
sample standard deviation across hidden channels (correction 1), epsilon `1e-5`, then the
encoder in its weight dtype. Pooling uses raw nonnegative magnitudes in FP32; missing codes
contribute zero to max/mean/sum, and mean divides by all retained residues. This sample
standardization is distinct from dense population variance across residues (correction 0).
Low-level `pool_features` still returns zero for empty groups for existing callers; the
canonical tap rejects an empty sequence before calling it. Build a new tap after mutating
SAE weights because its identity hashes them at construction.

Use run `dtype=None` with the SAE tap, and explicit `HiddenTap` dtype overrides for FP32
pooled dense vectors and BF16 residue states. This yields all three outputs in one forward
pass without quantizing the SAE input through a sibling's output dtype. See the
[FastPLMs contract](../../models/fastplms/docs/embedding_api.md#one-pass-taps) for layer
numbering and prefix-crop behavior. Key schema 1 and run-fingerprint schema 3 outputs cannot
silently become new outputs: the bumped identities deliberately invalidate their reuse.
A feature computed another way, such as legacy dense pooling that keeps BOS/EOS, must not
be stored under the new key. New canonical work uses the complete contract below. The smaller
`feature_key` API remains available for historical consumers; its schema-2 keys do not establish
compatibility with `feature_spec_v1`.

## Complete representation and row contracts

The structural reducer accepts exactly four already verified, frozen FP32 tensors on the
backbone device: `layer_logits`, `projection.0.weight`, `projection.0.bias`, and
`projection.1.weight`. These are the `StructuralMixture` state names; the checkpoint adapter
maps them from ESMFold2's `language_model.base_z_combine` and `language_model.base_z_linear.*`.
It applies layer normalization with epsilon `1e-5`, then the released bias-free projection,
then softmax mixture weights cast to projection dtype. It retains `(n, b, l, 256)` projections,
never `(n, b, l, d)` encoder states. The final stack temporarily duplicates projected storage;
this is bounded by the batch policy, not a constant-memory sum. The retained stack preserves
the reference einsum reduction order. BF16 and FP32 arithmetic are explicit; the run's output
dtype does not quantize the structural input. The layer count and width must match the base.
There is no checkpoint loader or separate encoder here.

[contracts.py](contracts.py) accepts only the exact versioned fields, finite JSON values, full
lowercase content digests, and immutable 40-character checkpoint revisions. It separates three
identities: the normalized input sequence, a representation independent of dataset membership,
and a row containing that representation for the sequence's retained positions. The full digest
decides equality; a readable slug or its 16-character suffix alone is not sufficient evidence.
Unknown fields and omitted dependencies fail rather than silently receiving defaults.

[capture.py](capture.py) is the bridge from a loaded FastPLMs model. Supply `model_source` and
`tokenizer_source` as `{repository, revision}` mappings; the model source names its **checkpoint**,
not its registry alias or runtime package. Supply the actual external SAE layer and its source
by tap name in `sae_layers` and `sae_sources`. Supply structural tensor mappings and pins by
tap name in `structural_weights` and `structural_sources`. Capture rehashes all four tensors,
records the entire structural dependency separately from the projection-only digest, and
rejects stale tap identities. Dense keys exclude both auxiliary dependencies. The
adapter measures current backbone tensors, serialized model config, tokenizer class and content,
and SAE encoder/bias tensors. It records effective dtypes, precision settings, hardware class,
software versions, attention implementation, masks, ordered pooling, and batch/crop policy.
It does not run the model, download artifacts, or authenticate declared source pins.

Call `capture_feature_specs` inside the same precision/autocast context as extraction, after
resolving attention, with explicit `max_length`, `truncate`, `batch_size`, `batch_window_size`,
`max_tokens_per_batch`, and `dtype`. These must be the exact options passed to `embed_dataset`.
Use `dtype=None` and per-`HiddenTap` overrides for the canonical mixed outputs. The required
`normalization_version` comes from the canonical inventory, currently `protein_ascii_v1`.
`engine_sha256` must hash the verified shared inference implementation; `reducer_sha256` maps
each tap name to its verified reducer implementation closure. Keep SAE-only code in its reducer
digest so changing it does not invalidate dense features. Neither value may be a label, a
mutable branch, or an unrelated whole-workspace hash.

The returned mapping has one `FeatureSpec` per tap name. Bind a result using
`spec.bind(SequenceKey(sequence_sha256, original_length, normalization_version), positions)`,
where `positions` comes from the actual retained biological-residue mask in original-sequence
coordinates. Do not infer positions from tensor length when a special token may occur internally.
The contract computes inference context length from the declared prefix crop. It rejects empty,
unordered, duplicate, or out-of-context positions and rejects overlength input under the reject
policy. On readback, validate the sequence against the original inventory before trusting its
declared digest, then call `FeatureRowKey.from_dict(row_record, feature=expected_spec)`.

This adapter qualifies frozen, single-device CPU/CUDA FastPLMs tap runs with one backbone
parameter dtype and a loaded tokenizer. It supports hidden-state taps, current pooled
`sae_tap` reducers, and `structural_tap`. Custom forward hooks, arbitrary model keyword overrides, dynamically modified
module behavior, and adapters are outside that contract. Do not mutate modules while extracting;
recapture and compare **every full contract** immediately before committing a segment. A mismatch
must prevent publication of that segment. The residue-SAE runtime capture adapter remains
part of the pending Atlas feature integration.

### Enforced persistence

Build `inventory` as `{normalized_text: SequenceKey(digest, len(normalized_text), policy)}` from
the independently verified canonical inventory. Do not normalize again or substitute accessions.
Create `FeatureExtractionContract(model, taps, inventory, **capture_options)` with the same
explicit arguments described above, then call:

```python
receipts = embed_into_features(
    model, sequences, store_root, contract.stored_features,
    taps=taps, contract=contract, **contract.embedding_options,
)
```

The adapter checks the exact model, tap objects, extraction options, physical descriptor and
full contract before creating stores. Every normalized residue must map to one non-special
token. The engine then checks the actual token IDs and biological mask before inference and
returns verified original-coordinate `TapRecord.retained_positions`. The pinned small ESMC
tokenizer lacks J, so J is rejected without replacing it or changing canonical sequence identity.
Canonical prefix crops preserve the original full-sequence digest and length alongside the
retained prefix positions; arbitrary internal special tokens are rejected on this path.

`bind_rows` validates shape, dtype and finite output, then supplies one `FeatureRowKey` per
written row. FastPLMs keeps these opaque JSON records in checksummed gzip sidecars. On a cache
hit, `validate_cached` checks the descriptor, row contract and inventory, index/commit-marker
association, and sidecar/data digests. It refuses old rows lacking this evidence. Immediately
before commit, the writer checks staged files and its descriptor; the adapter recaptures every
live dependency and checks the caller's descriptors again. Failure leaves the segment without
a commit marker or newly published index rows. Valid complete cache hits run no encoder.

This enforcement applies to `embed_into_features(..., contract=...)`. Historical descriptors
remain usable through the legacy API; they do not become canonical by renaming. The low-level
`FeatureStore.read` method does not itself enforce scientific contracts. Training readers must
validate the expected contract and `row_metadata(..., verify_data=True)` before using decoded
rows. FastPLMs now supplies read-only physical handles, window-bounded tensor persistence,
and transaction recovery for concurrent writers on one host's local filesystem;
training-view memory qualification and the
residue-SAE adapter remain separate tasks. The adapter does not authenticate
caller-declared source pins or code digests; use verified artifacts and implementation closures.

Importing this package needs nothing (it loads no submodule), and the first use of an export needs only torch. Both tap
factories import FastPLMs when called, never at import time, because Protify ships `foundry/embedding` inside exported probes that
carry no FastPLMs source. A tier-1 test holds that.

Consumers: every project that keeps a cache of per-sequence features names its features with
this module, and each lists `foundry.embedding` under `github.includes`, so `ws publish` vendors it
into the projection. enzyme_loop uses `row_layout`, and Protify's `EsmcSaeForEmbedding` uses the SAE
reducers. [The rollout record](../../docs/decisions/models/fastplms/2026-09-24_feature_store.evidence.md#rollout-2026-09-28)
lists each project's keys and what did not convert.

Everything else delegates. Adding a second embedding implementation here is the single
most damaging thing that can happen to this workspace, because every downstream project
would then have two to choose from.

## Caches and their keys

The decision
[`2026-09-22_sae_feature_cache_keys`](../../docs/decisions/foundry/embedding/2026-09-22_sae_feature_cache_keys.md)
adopts one key, and [one feature store](../../docs/decisions/models/fastplms/2026-09-24_feature_store.md)
moved every project's caches to it on 2026-09-28. A feature made by FastPLMs' taps from checkpoints
pinned by commit is named by `feature_key`. Every other cache, meaning a legacy cache converted
with `fastplms.features.convert_rows`, a pooling that keeps BOS and EOS, or a project's own encoder,
is named by `plain_feature`, whose descriptor states what the old cache could show and `None` for
what it never recorded. A converted legacy cache never claims a pinned revision it cannot show, and
is never renamed to a canonical key. The procedure is
[the adoption guide](../../docs/guides/feature_store_adoption.md).

## Not here

- **Dense pooling.** Stays in FastPLMs. A project that pools `(b, l, d)` hidden states uses
  `Pooler`, not this module.

Tests: `tests/tier1_unit/test_embedding_sae_features.py`, `test_embedding_sae_tap.py` (the tap
against FastPLMs' own SAE forward and live identity capture on a tiny random ESM++),
`test_embedding_contracts.py`, `test_embedding_keys.py`, `test_embedding_encoders.py` (the ligand and annotation stores on a tiny
random BART and Annotation Transformer), and `test_embedding_private_store.py` (the
private store writes and reads beside a stand-in artifact runtime in a fresh interpreter, and never
imports `fastplms`). They
need torch, so the tooling environment skips them. `ws test --projects foundry` runs them under
the research interpreter.
