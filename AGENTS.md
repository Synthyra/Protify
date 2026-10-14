# Protify

## Purpose and Sources

This is Synthyra Protify. Its canonical source is `projects/tooling/property_lowcode` in the research foundry workspace, and `Synthyra/Protify` on GitHub is a projection of it. Keep its public training, evaluation, packaging, and cloud interfaces independent of synth (`projects/serving/api`), which imports it; `embedding_translation`, `property_oracles`, and `protein_screening` build on it too.

- `docs/getting_started.md` and `docs/cli_and_config.md`: entrypoints and configuration
- `docs/probes_and_training.md`: probe behavior
- `docs/testing.md`: test scopes and working directories
- `src/protify/base_models/supported_models.py`: authoritative model registry
- `src/protify/data/supported_datasets.py`: authoritative dataset registry

## Architectural Invariants

- CLI, YAML, GUI, and cloud dispatch share the same `MainProcess` pipeline.
- Keep heavy model and dataset loading lazy where the registries and factories are lazy.
- FastPLMs is a separate repository with its own guidance. A GitHub clone holds it as the `src/protify/fastplms` submodule; the workspace holds one FastPLMs, at `models/fastplms`, and `ensure_fastplms_submodule_on_path` uses the submodule when it is present and the importable `fastplms` otherwise. Do not mix FastPLMs changes into a Protify task.
- `model_components` takes its attention, rotary tables (and `RotaryEmbedding.rebuild_tables`, which `_init_weights` calls after `from_pretrained`), transformer stack, and SwiGLU block from the `interpnet` family, and `probes/losses.py` takes its loss modules from `interpnet.losses`, so the probes and InterpNet run one implementation. Fix that code in `interpnet`. `_copy_runtime_code` ships `interpnet` beside `protify` in every exported probe repository, as it ships `foundry.embedding` and `foundry.layers`, whose feed-forward block `interpnet.mlp` builds on, and the kernels backend's loaded kernel lives in `interpnet.attention_utils`.
- `--parallel_probe_runs` applies only to compatible pooled, sequence-level MLP probes. Matrix or tokenwise probes, transformer probes, Lyra probes, PPI run-specific datasets, and full fine-tuning use the sequential fallback.
- A resolved model name carries everything that changes its embeddings. ESMC sparse autoencoder aliases expand to names such as `ESMC-300-SAE-l23-k64-c8192`, so the layer, sparsity, and codebook width reach the embedding store's key (its `model` field) and the results table without any separate cache key.
- FastPLMs owns the sparse autoencoder runtime, through `ESMplusplusModel.load_sae_models` and `compute_sae=True`, which returns one sparse `(valid_tokens, codebook_dim)` tensor per attached layer. `src/protify/base_models/esmc_sae.py` owns only what is Protify's: the published-coverage registry, model-name resolution, which reductions and which tokens pool to one vector per protein, and the storage policy. The pooling arithmetic is `foundry.embedding`'s, vendored at the repository root. Do not reimplement the SAE encoder or the pooling here.
- SAE models return one vector per sequence. They accept `max`, `mean`, and `sum` only, and reject `--matrix_embed`, because the dense per-residue codebook tensor does not fit the cache. Pooling covers every attended token, CLS and EOS included, so a protein of `l` residues pools over `l + 2` rows as the canonical store does; earlier Protify dropped the two special tokens, as the Biohub reference workflow does.
- Pair datasets swap protein A and B only on the training split, and only under `--random_pair_flipping`. Swapping a validation or test pair would make a reported score depend on the random state.
- Sparse storage (the store's `csr` layout, which the old cache called coordinate-list blobs) is selected by the model, through `EsmcSaeSelection.sparse_storage`, not per embedding. Every other model writes dense and pays nothing for the feature. The store keeps bfloat16 as bfloat16. Blob dtype code 3 (the bit pattern) and code 1 (the older float16-byte encoding) exist only in old `.db` caches, which `legacy_cache.py` decodes and never writes.
- `probe_type` is normalized once in `ProbeArguments` and `ParallelProbeRunSpec`. `mlp` is the current name; `linear` is accepted because saved probe configs, exported repositories, and existing YAML carry it, and `MLPProbeConfig.model_type` stays `linear_probe`.
- Estimator probes (`xgboost`, `lightgbm`, `random_forest`) run through `run_estimator_probes`, share the embedding cache and results table with the neural probes, and skip embedding standardization because it cannot change a tree's splits and would densify sparse features.
- The shared metric functions read their input as raw logits, because neural probes produce logits: the single-label one softmaxes and the multi-label one sigmoids before thresholding at 0.5. `estimator_probe._predictions` therefore returns log probabilities and log odds, never the probabilities `predict_proba` gives back.
- Matrix embeddings are cached with the leading and trailing special tokens, so a stored per-residue embedding has `len(seq) + 2` rows. Nothing may reshape one against `len(seq)`.
- Every embedding cache is a feature store directory under `--embedding_save_dir`, named by a key computed from a descriptor (`feature_cache.embedding_descriptor`): the resolved model name, `form`, `pooling` in column order, `hidden_state_index`, `max_length`, `special_tokens`, `dtype`, and for CARBON the resolved model revision and preprocessing schema, because each changes the stored values. `EmbeddingArguments.cache_descriptor` with `feature_directory`, and `DataMixin._embedding_store`, are the only ways to name a store, so writers and readers cannot drift. `feature_cache.py` is the one module that opens stores, `CacheWriter` is the one writer, and `open_reader` and `row_lookup` are the readers. Nothing writes an old cache format. `--sql` keeps its name and means streaming rows from the store; a parameter named `db_path` names the store directory.
- `EmbeddingArguments.cache_filename`, `DataMixin._embedding_cache_filename`, `get_embedding_filename`, and `legacy_embedding_filename` name old cache files, which only the converter and `_download_embeddings` read. `legacy_embedding_filename` reproduces the pre-`max_length` naming and is used only at `PUBLISHED_EMBEDDING_MAX_LENGTH`. A file name loses the pooling order, so an old file is adopted or converted from a download only when the run's pooling types are sorted, and `python -m protify.legacy_cache <file> --pooling ...` converts one pooled in another order.
- `legacy_cache.py` holds the read-only decoders of the old SQLite and `.pth` caches and converts them with `fastplms.features.convert_rows`, which compares every stored row with the decoded one bit for bit. It has no writer. A test that needs an old cache builds it with `testing_suite/legacy_cache_writer.py`, never with a writer in the source tree. A store serves any dtype it holds exactly, so a float32 store serves float16 and bfloat16 runs, and rows widen into it and never narrow.
- `Pooler.__call__` hands every pooling function the full keyword set, including `input_ids` and `eos_token_id` that only `eos` reads. Every pooling function therefore takes `**kwargs`, and a new one that does not will raise only through `__call__`, never through a direct call in a test.
- Protify reimplements CARBON's DNA tokenizer in `base_models/carbon_tokenizer.py` instead of executing the model repository's `tokenizer.py`. That reimplementation is pinned to the published layout: base order `A, T, C, G`, specials `<dna>`, `</dna>`, `<oov>` ahead of the 4096 6-mers, `<unused_i>` padding to a multiple of 128, a final 1-5 bp remainder right-padded with `A`, and `<oov>` for any chunk containing a non-ATCG base. `_check_vocabulary_agreement` compares the checkpoint's `vocab_size` against the reconstruction at load time, because a silent divergence would index the wrong embedding rows.
- CARBON reads DNA as 6-mers between two boundary tokens, so `--max_length` is a token budget and the raw budget is `6 * (max_length - 2)`. `resolve_data_max_length` turns the selected models into the single limit the dataset must satisfy: the widest under truncation, the narrowest under `--trim`, and the token budget when no model is selected.
- `wrap_lora` takes the model name and adds the Llama projection targets only for CARBON. Adding them unconditionally would attach adapters inside any other Llama-derived checkpoint loaded through `--model_types custom`, silently changing that model's trainable parameters.
- `data/pair_features.py` owns the single rule for turning two pooled protein vectors into one pair row. Every join site reads it: the two pair datasets, the streaming and row-addressed readers in `trainers.py`, the numpy builder in `data_mixin.py`, and `PackagedProbeModel`. A mode is symmetric or it is not, and `pair_order_matters` is what makes `--random_pair_flipping` a no-op under `minmax` rather than an augmentation that emits duplicates. `minmax` needs one vector per protein and is rejected with `--matrix_embed`.
- Estimator feature matrices choose dense or CSR from the measured density of a row prefix, and the choice is pinned across train, validation, and test so all three read a zero the same way. Standardizing forces dense, because centering fills in every zero. XGBoost treats an entry a CSR does not store as missing rather than as zero, which is the intended reading for a max-pooled codebook.
- A max-pooled codebook's sparsity comes from `k`, not from codebook width: at layer 60, k=16 with c=131072 pools to about 1% non-zero while k=512 at the same width reaches 35%. Any claim that SAE features are cheap holds only at low k.
- Estimator probes drop columns that are zero for every training row unless `--keep_dead_features`. The drop is composed into the `FeatureSelection` a saved probe carries, so inference applies one index array over the original pooled width rather than replaying two narrowing steps in order.
- `--tune_estimator` searches with Optuna against the validation split only, warm-started from `ESTIMATOR_DEFAULTS` so a search cannot land below the untuned configuration on validation. `search_space_names` reads the parameter names off the search space itself, so a new parameter cannot silently go unenqueued. The default objective is the metric the results table reports, because a search scored on one metric and reported on another can move the reported one backwards: `scale_pos_weight` shifts every predicted probability, which AUPRC cannot see and MCC is punished by.
- `precompute_sae_and_dense.py` fills one cache per (configuration, pooling set) from a single backbone pass. Every set is a subset of the union the encoder reduces to once, so a second pooling set costs a column slice. Each cache is named exactly as its own single-configuration run would have named it.
- An estimator probe is not a torch module, so it is saved through `probes/estimator_artifact.py` rather than `PackagedProbeModel`. The recipe it carries is the resolved model name, the pooling set, the hidden-state index, the pair mode, and the column selection: without all five a saved probe scores a different representation than it was fit on, and nothing raises.
- `projected` and `sngp` build `ProjectedProbe`: each branch is clamped to plus or minus `input_clip`, layer-normalized, and projected to `projection_size / k`, so the trunk and output after the projections are identical for every arm of a representation comparison. `branch_widths` are one protein's branch widths; a pair row is two slots of them, and a branch reads its columns from every slot. Branches select columns by view (`BranchLayout.gather`), never through a stored index buffer, because `from_pretrained` leaves non-persistent buffers uninitialized.
- `ProjectedProbeConfig.branch_sizes` gives each branch its own projected width, summing to `projection_size`; `None` is the even split and is left out of the training recipe, so a seed finished before the field existed still resumes. The Atlas proteome profile is a third branch, `Atlas-Profile:profile` (`BranchSpec.profile`), whose store is the cell's `probe_features.profile_root`, read by `data/atlas_profile_feature.py` from the `profile` feature of a `profile.json` store. It rides with at least one canonical stream, refuses a protein the store has no row for before any training, and puts the store's model, axis and manifest digests in the package's `source.atlas_profile`. A cell names `profile_root` exactly when a branch reads the profile, and the key is left out of every other cell's record.
- `sngp` bounds every projection and trunk Linear through `SpectralNormBound`, whose power iteration advances only in training mode, so MC dropout, which switches only Dropout modules on, cannot move it. Its Laplace posterior is fitted once, after training with the restored best weights, by `fit_posterior`, which `TrainerMixin._train` calls on any probe that defines it. Logits are MAP until then and mean-field after; the posterior is persistent buffers, so it travels with the state dict. The Gaussian process output takes single-label and regression tasks only.
- The default probe on an ESMC base is `branch_features.default_branch_specs(backbone)`, the Biohub SAE at the base's depth layer max-pooled beside the final state's mean and variance, under the `projected` head (`docs/conventions/probes.md` in the workspace). `main.py` trains it whenever every model is an ESMC base and the run names no probe type, pooling, hidden-state index, or other workflow, on the command line or in YAML; `--no_default_probe` turns it off. Code that needs the default calls `default_branch_specs` rather than spelling the tokens out.
- The `Random` and `OneHot-Protein` controls are generated where they are read and never embedded, stored, or downloaded: `feature_cache.GeneratedReader` and `base_models/control_bases.py` delegate to the embedding engine's `SyntheticControlView`, so a control's rows are `(l + 2, w)` and its pooled statistics always agree with them. The engine owns the arithmetic and `CONTROL_SCHEMA`; Protify must not draw pooled statistics of its own. An old control cache has no store to convert into.
- An uncertainty cell's `probe_features` (`data/probe_features.py`) reads a base's rows from the embedding engine's canonical store, or a generated control, with no pinned feature manifest and no Protify cache: `<base>-SAE-...:max` is `sae_max`, `<base>:mean,var` is `final_mean_var`, and `<base>:mean` is the first half of `final_mean_var`. It joins them through `IndexedBranchEmbeddings` over `ProbeBranchView`. Never hash a sequence here: key through `foundry.embedding.sequence_inventory`.
- `data/branch_features.py` joins several ordinary caches per sequence in memory. Each branch's cache is named and opened by `DataMixin`, exactly as its single-model run writes it, and no joined cache is ever written. SAE branches stay CSR in memory, and `JoinedBranchCollator` densifies one batch at a time.
- `python -m protify.uncertainty` runs one cell outside `main.py`. V_fit and V_cal are resolved once by `calibration_split.resolve_validation_roles`, a stratified carve with its own seed unless explicit rows are given, recorded by row digest in `split.json`, and checked on every resume. Temperature and variance maps read V_fit only, conformal quantiles read V_cal only, and test is only ever scored. The data seed stays fixed across run seeds, every seed is kept rather than the lowest test loss, and each finished seed's files make a rerun resume.
- Hugging Face reads its cache locations on import, and importing `protify` imports it, so under `python -m protify.uncertainty` an `hf_home` in `protify_args` cannot move the caches. The workflow refuses one the environment does not already match; set `HF_HOME`, `HF_HUB_CACHE`, and `HF_DATASETS_CACHE` instead.
- `protify.uncertainty compare` depends on its seed alone: every draw comes from `resampling.dataset_generator(seed, data_name, stream)` and the dataset's test labels, so its outputs are byte-identical for any `--workers`. Keep timings out of `verdicts.json`.
- Scheme T reuses `methods.build_methods` unchanged and checks repeat 0 against `report.score_method`, so a change to a method's fit or scoring must keep `test_uncertainty_exchangeable.py` passing. A grid's `--pattern` must match one `report.json` per (dataset, arm, head), as `*/*/final/report.json` does.
- Probabilities are clipped off 0 and 1 in float64 before any logarithm. In float32, `1 - 1e-12` rounds to 1.0, and an MC dropout mean of saturated sigmoids reaches exactly 1.0: `methods.mean_binary_entropy` and the multilabel `metrics.negative_log_likelihood` returned NaN on EC and crashed a fit cell until they clipped in float64.
- `uncertainty.save_heads` keeps each seed's head in `heads/seed_<seed>/`, and the standardizer, with the training labels' standard deviation, in `standardizer.npz`. `python -m protify.uncertainty package` ships one method by rebuilding it with `build_methods`, never refitting, and refuses a package that does not reproduce the saved predictions and `report.json`.
- `_train_seed` predicts on CUDA when one is visible, and Protify turns on TF32 there, so a package reproduces its cell only on the same device type. `ws run` does not hide the GPU as `ws test --projects` does; hide it for CPU-only checks.
- A package ships LAC sets alone (`package.SHIPPED_SET_SCORES`). Reports still measure APS, but on a binary task at a high level its threshold sits at a cumulative probability of 1.0, where floating-point rounding, which differs between machines, decides whether a set holds one class or two.
- A classification package's answer rule (`answering.py`) reads V_cal only: its wrong answers' confidences give each new answer a conformal p-value, and `PackagedHead.select` runs Benjamini-Hochberg over the rows passed together. A multilabel answer is never the empty set when every V_fit and V_cal row carries a label; test is never read to decide that. The confidence is the probability the whole answer is right, which the per-label `uncertainty` is not.
- Under `uncertainty.split_manifest` a cell's rows are pooled from its three splits and split again by identity strata (`uncertainty/stratified_split.py`), after row caps, because the caches hold only the capped rows. Every row follows its sequence's split, and the manifest's per-split row counts are checked on every load, so a manifest from other rows raises rather than silently re-splitting. V_fit and V_cal then reach `resolve_validation_roles` as explicit rows, so `split.json` records them by digest as for a carve.
- `uncertainty/crossfit.py` never changes the served head: it trains fold heads only for their out-of-fold rows. Full research mode refuses to write `served.npz` unless the reloaded head reproduces the cell's saved V_fit, V_cal, and test outputs within 1e-3. Explicit `--calibration-only --correctness-only` instead requires a declared validation-only cell, checks saved V_fit rows/outputs before fold training, regenerates V_cal, and never opens test payloads; the README defines its full-digest five-fold plan and hashed resume contract. Its completion is not a suite calibration or release gate. Fold neighbors come from the fold's own training rows, remapped to train indices, so a held-out row never finds itself or its cluster. `nearest_rows` uses `torch.cdist` only to choose candidates, then re-measures each chosen neighbor from differences: the matrix product runs in TF32 on CUDA, and in float32 anywhere it cancels |q|^2 + |r|^2 against 2 q.r, so a near-duplicate's distance was off by up to 0.3.
- Every Oracle classification package ships a correctness model (`uncertainty/correctness.py`, Logan's decision of 2026-09-26): a logistic regression of whether each answer is right, on the confidence and latent signals read from the head's trunk, its PCA-whitened input, and its active SAE features, against a reference of at most `REFERENCE_ROWS` training rows. It is fitted on out-of-fold rows at level `LEVEL` of the cross-fit, never on V_fit or V_cal, and `package --crossfit` refuses a cross-fit whose served seed is not the shipped seed or whose answers differ from the package's. The package's answer rule then ranks by reliability (`"score": "correctness"`), and `load_package` refuses a rule and a `correctness/` directory that disagree. `crossfit --correctness-only` skips the per-row M8 neighbors, which hold every training row's input on the device.
- The correctness model's distances, projections, and Gaussians run in full float32 (`correctness.full_float32`), and its class Gaussians use the Woodbury and determinant lemmas (`latent_geometry.TrunkGaussians`) so that thousands of classes fit in memory; `test_latent_geometry.py` checks them against the direct computation. A package predicted on another device than its cross-fit agrees to a median of 1e-3 in reliability, which `package_correctness.check_shipped_correctness` enforces before the package is written.
- A package with a correctness model also ships a novelty cutoff and the `answer_or_abstain_v1` policy (`uncertainty/applicability.py`), and `load_package` refuses a policy that is not the one its answer rule and applicability record give, because it names both by digest. `PackagedHead.decide` runs it: a row with a non-finite output is unassessed, a novel row or one in a V_cal cell of fewer than `MINIMUM_STRATUM_ROWS` rows abstains, and Benjamini-Hochberg over the remaining rows of one request answers or abstains. The false-answer-rate statement is not restated for the rows that survive the screens; a request that holds many rows the package is unsure of is stricter for every row in it, so a row answered in a request is answered alone and never the reverse, and `metrics.json` reports the test rows both as one request and one row per request (`decide_alone`); a cell without support states no rate; and nothing escalates to a larger base model. `docs/uncertainty_packages.md` says why. `PackagedHead.fixed_width_interval` names the regression packages whose interval has one width for every row, the only ones an API may return as a split-conformal interval.
- Probes compile with Inductor by default, which needs a C compiler to build Triton's launcher. Images without one, such as Modal's `fastplms_runtime`, pass `--no_compile`: the ablation's first fit launch failed on it.
- Open work in `src/protify` is marked under the README's Known issues, because `testing_suite/test_marker_cleanup.py` fails on any `TODO(...)` marker in that tree.
- `precompute_sae_and_dense.py --data_revisions` aligns one for one with `--data_names`, as in `main.py`, because the revision decides the sequences and so the cache keys.
- The wheel ships `protify` and, as top-level packages, the `foundry` and `interpnet` code vendored at the repository root, because `protify` imports both. A newly vendored package needs its own `packages` entry in `pyproject.toml`, and a new third-party import needs a dependency there as well as in `requirements.txt`. The release workflow checks out the FastPLMs submodule, which the wheel carries under `protify/fastplms`.

## Python Coding Standard

Write direct, readable Python for human maintainers. Prefer domain language and visible data flow over cleverness, speculative generality, or abstraction for its own sake.

Apply this priority order when guidance conflicts:

1. Follow the user's explicit scope and requested behavior.
2. Preserve correctness, security, public interfaces, data schemas, tested behavior, and scientific conventions.
3. Follow repository instructions and enforced formatter, linter, and type-checker configuration.
4. Apply the preferences below.
5. Minimize unrelated diff churn.

Treat cleanup and refactoring as behavior-preserving unless behavior changes are explicitly requested. Observable behavior includes exceptions, import side effects, CLI output, serialization, random-number use, dtype, device, tensor shape, and documented performance guarantees. Do not silently fix unrelated bugs or expand an ambiguous cleanup into a repository-wide rewrite. Exclude generated, vendored, migration, build, and external code unless it is explicitly in scope.

### Validation and Workflow

- Read the relevant repository guidance, configuration, implementation, and focused tests before editing.
- Establish a behavioral baseline before structural changes and run the same focused tests afterward. Record pre-existing failures and compare like with like; never weaken tests to make a refactor pass.
- Keep validation proportional. Use formatter, linter, type checker, compile checks, or focused CPU-only tests for mechanical changes; require meaningful behavioral tests for structural changes.
- Do not trigger downloads, CUDA, large datasets, remote machines, or slow integration suites unless requested.
- Make safe mechanical improvements first. Make structural changes only when they materially improve ownership, cohesion, or visible data flow. Review the final diff for behavior drift, stale imports, unnecessary movement, and formatter conflicts.

### Imports, Types, and Control Flow

- Keep the module docstring first and `from __future__` imports immediately after it. Then group direct `import` statements before ordinary `from ... import ...` statements. Within those blocks, place standard-library names before third-party names and keep repository-local imports in a distinct final section. Leave two blank lines after the complete import block.
- Preserve guarded, optional, registration, and initialization-sensitive import ordering. Do not fight an enforced formatter.
- Give each function one narrow, typed responsibility. Prefer precise domain types and parameterized containers; use `Any` only at genuinely dynamic boundaries.
- Respect the minimum Python version and runtime annotation behavior. Do not alter framework-discovered fields, import cycles, or serialization schemas merely to add types.
- Use domain-specific names instead of vague names such as `data`, `result`, `info`, or `manager` when a clearer term exists.
- Prefer explicit sequential control flow. Separate distinct guards, loops, exception handling, and logical phases with blank lines when that improves scanning.
- Raise ordinary exceptions for invalid external input and use assertions only for internal invariants or programmer assumptions.
- Avoid broad `except Exception`, silent fallback cascades, unsupported compatibility shims, wrappers that merely rename one call, and one-use helpers that do not name a real concept or isolate a testable phase.
- Keep argument parser declarations on one line when they remain comfortable to read.

### Comments and Numerical Code

- Comment intent, assumptions, units, biological conventions, tensor shapes, non-obvious mechanics, and design rationale. Remove comments and docstrings that merely narrate syntax or repeat a precise name and type.
- For NumPy, PyTorch, JAX, and similar code, maintain a complete, accurate shape trace through tensor-valued inputs, assignments, transformations, layer calls, reductions, mutations, and returns.
- Use lowercase shape symbols such as `b` (batch), `l` (sequence length), `d` (hidden width), `h` (heads), `c` (classes), and `n` (generic count). Define nonstandard or dynamic symbols near first use, use `()` for scalars, and use an ellipsis when rank cannot be proven. Never invent a fixed shape.
- Put shape comments on the same line as the operation when practical. Treat stale shape comments as correctness defects.
- Use uppercase mathematical tensor names such as `X`, `Q`, `K`, `V`, and `W_q`; keep descriptive tensor names such as `hidden_states`, `attention_mask`, and `logits` lowercase. Preserve established domain notation when clearer.

### Modules and Interfaces

- Give each module one primary responsibility. Split by ownership and dependency direction, not by line count, and keep reusable helpers near the feature that owns them. Do not create catch-all `utils.py` modules.
- Keep extracted modules' public surfaces small and intentional. Avoid circular imports and implicit shared state.
- Use a class when state and behavior form a meaningful domain object or workflow. Prefer a function, typed value, or dataclass for simple transformations and records.
- Keep entry points thin: parse arguments, construct configuration and collaborators, then invoke a short sequence whose names expose the workflow.
- Extract abstractions only when they remove real complexity, reduce meaningful duplication, name a cohesive concept, or isolate a separately testable phase.

## Canonical Commands

In the workspace, `ws test --projects property_lowcode` runs the CPU suite on the research interpreter, with `foundry`, `interpnet`, and `fastplms` importable, and `ws run property_lowcode -- <command>` runs a command with the same imports.

In a GitHub clone, build and run the broad CPU suite from the repository root:

```powershell
docker build -t protify-env:latest .
docker run --rm --ipc=host -v ${PWD}:/workspace -w /workspace protify-env:latest \
  python -m pytest src/protify/testing_suite -v -m "not gpu and not slow"
```

Focused parallel-probe tests run with `src/protify/` as the working directory:

```powershell
docker run --rm --gpus all --ipc=host -v ${PWD}:/workspace -e PYTHONPATH=/workspace \
  -w /workspace/src/protify protify-env:latest python -m pytest \
  testing_suite/test_parallel_probe_plan.py testing_suite/test_parallel_linear_probe.py -v
```

Treat the Python coding standard imported above as mandatory repository guidance for implementation, refactoring, and review. Preserve behavior and scope first; apply its readability preferences to all first-party Python touched by the task.

Maintain agent instructions only in AGENTS.md. Treat AGENTS.md as the canonical, cumulative source of repository guidance; do not create or expand separate CLAUDE.md guidance. Preserve existing information when updating this file.

---

# Projection notice

Appended by `ws publish` from the research foundry workspace. **Do not edit this section.**

This repository is a projection. The canonical source is the workspace entity `property_lowcode`
at `projects/tooling/property_lowcode/README.md`.

## If you change code here

Changes made in this clone do not reach the workspace on their own. Report what you changed
and at which commit so it can be brought back with `ws import Synthyra/Protify <sha>`. Do not
assume a later publish preserves your change: publish rebuilds this tree from the workspace
and refuses when GitHub HEAD is not the SHA it last recorded.

## Vendored code

`fastplms.features, foundry.checkpoint_loading, foundry.datasets, foundry.digests, foundry.embedding, foundry.eval, foundry.logging, foundry.metrics, foundry.secrets_env, foundry.training, interpnet, foundry.downloads, foundry.serialization, foundry.gpu_memory, foundry.pilot, foundry.compute, foundry.chemistry, foundry.layers, foundry.retry, foundry.compression` is vendored here from the workspace so this clone runs standalone. Do not edit a
vendored directory; edit it in the workspace and republish.

## Results

Log runs to Weights and Biases with the group set to the experiment id. That is how results
reach the workspace, through `ws capture`. Do not copy result files by hand.
