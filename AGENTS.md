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
- `model_components` takes its attention, rotary tables (and `RotaryEmbedding.rebuild_tables`, which `_init_weights` calls after `from_pretrained`), transformer stack, and SwiGLU block from `foundry.layers` (`probe_attention`, `attention_backends`, `probe_transformer`, `feedforward`), and `probes/losses.py` takes its loss modules from `foundry.layers.losses`, so the probes and the workspace's InterpNet run one implementation. Fix that code in `foundry.layers`. InterpNet is proprietary: nothing in Protify imports `interpnet`, and a projection never carries it (Logan, 2026-10-14). The kernels backend's loaded kernel lives in `foundry.layers.attention_backends`.
- An exported probe repository holds weights, tokenizer, card, and `packaged_probe_model.py` alone, whose imports are absolute on the installed `protify` (`_write_packaged_module`); a consumer installs Protify, and the config records `protify_version`, which a mismatched install warns about on load. The training-time embedding standardizer travels as the persistent `feature_mean` and `feature_scale` buffers, applied after the feature selection, so a reload reproduces the logits the probe was evaluated on.
- Protify's probes read one stream: a pooled vector, per-residue embeddings, or an estimator's matrix. It holds no feature fusion, fusion head, default probe, uncertainty workflow, or Oracle package runtime; the research workspace keeps those (Logan, 2026-10-14). Code outside Protify extends it through two seams only: `probes.get_probe.register_probe_type`, after which `ProbeArguments`, `get_probe`, and `rebuild_probe_from_saved_config` accept the type and store its settings for it alone, and `probes.fit_hooks.FitHooks`, which a training dataset carries as `fit_hooks` to change how `TrainerMixin._train` builds its trainer, stops, records, and fits a posterior. `_train` calls `fit_posterior` on any probe that defines it, after training, on the rows `FitHooks.posterior_rows` gives. Neither seam reads anything from the workspace, and the public test suite covers both.
- What Protify keeps of calibration is `calibration_report.py`: a dataset that has a `calibration` split (aliases in `data_mixin.SPLIT_ALIASES`) carries it through every step of `process_datasets` except the row caps, into `DataMixin.calibration_sets` beside the unchanged train, valid, test tuple, and `trainer_probe` reads it with the probe it keeps and adds `test_calibration_*` to the test metrics: a temperature, ECE and NLL before and after it, and 90% conformal sets for single-label classification, and 90% conformal intervals for regression, all from `foundry.eval` and `foundry.metrics`. The split is never required, and no dataset without one behaves differently (Logan, 2026-10-14).
- Importing `protify`, `protify.probes`, or `protify.base_models` loads nothing until a name is used (PEP 562), and nothing in the package changes `sys.path`. `main.py`, `gui.py`, and `embedder.py` call `entrypoint_setup.configure_process()` before their other imports, so Hugging Face reads its settings, and `main.py`'s `__main__` block sets `HF_HOME` from `--hf_home`, before Hugging Face loads. Run the CLI as `python -m protify.main`; modules import one another relatively.
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
- `--tune_estimator` searches with Optuna against the validation split only. Its first trial is exactly the untuned configuration, `ESTIMATOR_DEFAULTS` and, for every searched parameter those leave to the library, `LIBRARY_DEFAULTS`, so a search cannot land below the untuned fit on validation; `default_trial_params` raises for a searched parameter with neither. `search_space_names` reads the parameter names off the search space itself, so a new parameter cannot silently go unenqueued. XGBoost's L1 penalty is switched on before its log-scale strength is searched, because the untuned value is 0, and the search returns the space's estimator parameters rather than the trial's switches. LightGBM's space sets `subsample_freq` to 1, without which LightGBM never reads `subsample`. The default objective is the metric the results table reports, because a search scored on one metric and reported on another can move the reported one backwards: `scale_pos_weight` shifts every predicted probability, which AUPRC cannot see and MCC is punished by.
- An estimator probe is not a torch module, so it is saved through `probes/estimator_artifact.py` rather than `PackagedProbeModel`. The recipe it carries is the resolved model name, the pooling set, the hidden-state index, the pair mode, the column selection, and whether it was fit on a CSR matrix (`sparse_layout`), which scoring rebuilds because XGBoost reads an unstored entry as missing and a stored zero as zero: without all six a saved probe scores a different representation than it was fit on, and nothing raises.
- The `Random` and `OneHot-Protein` controls are generated where they are read and never embedded, stored, or downloaded: `feature_cache.GeneratedReader` and `base_models/control_bases.py` delegate to the embedding engine's `SyntheticControlView`, so a control's rows are `(l + 2, w)` and its pooled statistics always agree with them. The engine owns the arithmetic and `CONTROL_SCHEMA`; Protify must not draw pooled statistics of its own. An old control cache has no store to convert into.
- Probes compile with Inductor by default, which needs a C compiler to build Triton's launcher. Images without one, such as Modal's `fastplms_runtime`, pass `--no_compile`: the ablation's first fit launch failed on it.
- Open work in `src/protify` is marked under the README's Known issues, because `testing_suite/test_marker_cleanup.py` fails on any `TODO(...)` marker in that tree.
- The wheel ships `protify` and, as top-level packages, the `foundry` code vendored at the repository root, because `protify` imports it. A newly vendored package needs its own `packages` entry in `pyproject.toml`, and a new third-party import needs a dependency there as well as in `requirements.txt`. The release workflow checks out the FastPLMs submodule, which the wheel carries under `protify/fastplms`. The projection vendors no part of `fastplms`: the feature store comes from the submodule, so its pin must hold `fastplms.features` (FastPLMs `6554a56` or later), and a pin bump ships with any change that needs newer FastPLMs code.

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

In the workspace, `ws test --projects property_lowcode` runs the CPU suite on the research interpreter, with `foundry` and `fastplms` importable, and `ws run property_lowcode -- <command>` runs a command with the same imports.

In a GitHub clone, build and run the broad CPU suite from the repository root:

```powershell
docker build -t protify-env:latest .
docker run --rm --ipc=host -v ${PWD}:/workspace -w /workspace protify-env:latest \
  python -m pytest src/protify/testing_suite -v -m "not gpu and not slow"
```

Focused parallel-probe tests run on a GPU from the repository root, as everything does; the image puts `src/` and the root on `PYTHONPATH`:

```powershell
docker run --rm --gpus all --ipc=host -v ${PWD}:/workspace -w /workspace protify-env:latest python -m pytest \
  src/protify/testing_suite/test_parallel_probe_plan.py src/protify/testing_suite/test_parallel_linear_probe.py -v
```

Outside Docker, `pip install -e .` from the root makes `protify` and the vendored `foundry` importable; run the CLI as `python -m protify.main` and the GUI as `python -m protify.gui`. There is no CI test workflow: the team runs the suite itself, and none is added (Logan, 2026-10-14). `.github/workflows/upload.yml` only builds and publishes the wheel on a release.

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

`foundry.checkpoint_loading, foundry.datasets, foundry.digests, foundry.embedding, foundry.eval, foundry.layers, foundry.logging, foundry.metrics, foundry.secrets_env, foundry.training, foundry.downloads, foundry.serialization, foundry.gpu_memory, foundry.pilot, foundry.compute, foundry.chemistry, foundry.retry, foundry.compression` is vendored here from the workspace so this clone runs standalone. Do not edit a
vendored directory; edit it in the workspace and republish.

## Results

Log runs to Weights and Biases with the group set to the experiment id. That is how results
reach the workspace, through `ws capture`. Do not copy result files by hand.
