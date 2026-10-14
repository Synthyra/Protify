"""Command line for tap probing: pooled stores, token stores, and probe training on the concept tasks.

    python -m protify.interpretability embed  --root RUN [--tasks protein] [--sites all] [--controls]
    python -m protify.interpretability sweep  --root RUN --label B256 --lr 1e-4 1e-3 1e-2 --wd 0 0.1 --batch-size 256
    python -m protify.interpretability select --root RUN --labels B256 B1024
    python -m protify.interpretability train  --root RUN [--tasks all] [--lr 1e-3 ...] [--probe linear]
    python -m protify.interpretability validate --root RUN [--label linear]
    python -m protify.interpretability phospho --root RUN [--proteins 1000]
    python -m protify.interpretability pca    --root RUN --tasks interpro_family --sites resid_post --layers 0,5,10,15,20
    python -m protify.interpretability figures --roots RUN [RUN_2 ...] [--random-roots RANDOM ...] --out FIGURES
    python -m protify.interpretability compare --variants raw=RUN normalized=NORM mean=RUN:mean --out FIGURES
    python -m protify.interpretability tokens --sequences FILE --out DIR --sites resid_post

`embed` writes the pooled stores the protein tasks read: every protein-task sequence through the trained model,
and, with `--controls`, the test proteins under the untrained model, scrambled, and token-normalized. `sweep` trains
every probe under every grid setting and scores each on the validation split, and `select` picks the one setting
that costs the probes least validation score, preferring the fastest batch when several are close. `train` trains
and scores every probe of each task: protein tasks from the stores, residue tasks by streaming from the model.
Results go to `RUN/results/<label>/<task>/` as parquet tables and a safetensors file of the selected probe weights;
`validation.parquet` holds each probe's validation score, which best states are chosen by, and `validate` adds it to
runs trained before `train` wrote it.
`phospho` runs the phosphomimetic scanning test, `pca` writes principal coordinates of pooled states by label set,
`figures` draws every figure and summary table the runs' tables support, and `compare` sets variants of the analysis
(normalized inputs, mean-only pooling) against a reference. `--normalize` on `embed` and `train` L2-normalizes every
token state before pooling or probing; `--pooling mean` makes protein probes read pooled means alone.

`--model` takes any ESMC (ESMC-300, ESMC-600, ESMC-6B) or one of Protify's baseline encodings, `OneHot-Protein` and
`Random`, whose token rows are their one state. `--tasks ppi` adds the protein-protein interaction task (the Bernett
gold-standard pairs, `--pair-repo`), whose proteins get stores of their own (`stores/ppi_*`) and whose probes read a
symmetric row of the two partners' pooled states (`pairs`). `--no-untrained` leaves the untrained model out of the
controls, and `--site-budget-gb` bounds the pooled rows a protein or pair probe holds on the device at once.
"""

from __future__ import annotations

import argparse
import json
import time
import pandas as pd
import torch

from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from safetensors.torch import load_file, save_file

from .baselines import BASELINE_SITES, BaselineModel, is_baseline
from .concepts import ALL_TASKS, DEFAULT_REPO, PROTEIN_TASKS, RESIDUE_TASKS, ConceptData, load_concept, scramble, scramble_seed
from .capture import model_dims
from .interventions import paired_tests, run_phospho_interventions
from .latent import label_sets, project_layers
from .pairs import PAIR_REPO, PAIR_TASK, PairData, load_pairs, sweep_pair_task, train_pair_task, validate_pair_task
from .pooled_store import PooledTapStore, embed_pooled
from .probe_bank import ProbeBank, ProbeSettings
from .protein_probing import ControlStores, sweep_protein_task, train_protein_task, validate_protein_task
from .residue_probing import sweep_residue_task, train_residue_task, untrained_copy, validate_residue_task
from .selection import choose_setting, rank_settings, read_sweeps, task_optima
from .sites import ModelDims, SiteLayout, TapSpec, parse_sites, site_layouts
from .token_batches import TokenBatch, token_batches
from .token_store import embed_tokens
from .training import TaskResults, TrainingSettings, restore_bank


STORE_VARIANTS = (
    ("trained", "original"),
    ("trained", "l2_normalized"),
    ("trained", "scrambled"),
    ("untrained", "original"),
    ("untrained", "l2_normalized"),
    ("untrained", "scrambled"),
)
# The pair task's proteins have stores of their own, named with this prefix.
PAIR_PREFIX = f"{PAIR_TASK}_"


def parse_tasks(value: str) -> tuple[str, ...]:
    """Task names from a CLI value: 'all' (the 22 concepts), 'protein', 'residue', or comma-separated names, among
    which the protein-protein interaction task 'ppi', which no group includes."""
    if value == "all":
        return ALL_TASKS
    if value == "protein":
        return PROTEIN_TASKS
    if value == "residue":
        return RESIDUE_TASKS
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = sorted(set(names) - set(ALL_TASKS) - {PAIR_TASK})
    if unknown:
        raise SystemExit(f"Unknown tasks {unknown}; known tasks are {list(ALL_TASKS) + [PAIR_TASK]}.")
    return names


def parse_blocks(value: str) -> tuple[int, ...] | None:
    return None if value == "all" else tuple(int(part) for part in value.split(","))


def load_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    random_init: bool = False,
    seed: int = 0,
) -> tuple[torch.nn.Module, object]:
    """The model and its tokenizer; with `random_init`, every module reset to PyTorch's default initialization, as the
    untrained-model control is (`untrained_copy` with the same seed gives the same weights). A baseline encoding
    (`OneHot-Protein`, `Random`) reads ESMC's tokens."""
    if is_baseline(model_name):
        from ..base_models.esmc import get_esmc_tokenizer

        if random_init:
            raise SystemExit(f"{model_name} has no weights to initialize at random.")
        tokenizer = get_esmc_tokenizer("ESMC-300")
        return BaselineModel(model_name, tokenizer).to(device).eval(), tokenizer
    from ..base_models.get_base_models import get_base_model

    model, tokenizer = get_base_model(model_name, dtype=dtype)
    if random_init:
        model = untrained_copy(model, seed)
    return model.to(device, dtype).eval(), tokenizer


def model_label(args: argparse.Namespace) -> str:
    """The model's name as stores and records give it: a randomly initialized model says so."""
    return f"{args.model} random init (seed {args.seed})" if args.random_init else args.model


def tap_spec(args: argparse.Namespace, per_head: bool = True) -> TapSpec:
    """The states a command reads: the sites and blocks asked for, or a baseline's one state."""
    if is_baseline(args.model):
        return TapSpec(sites=BASELINE_SITES, per_head=False)
    return TapSpec(sites=parse_sites(args.sites), blocks=parse_blocks(args.blocks), per_head=per_head and not args.no_heads)


def load_task(name: str, args: argparse.Namespace) -> ConceptData | PairData:
    """One concept task, or the protein-protein interaction pairs."""
    if name == PAIR_TASK:
        return load_pairs(repo=args.pair_repo, local_dir=args.pair_local_data)
    return load_concept(name, repo=args.repo, revision=args.revision, local_dir=args.local_data)


def load_tasks(tasks: Sequence[str], args: argparse.Namespace) -> dict[str, ConceptData | PairData]:
    return {name: load_task(name, args) for name in tasks}


def store_path(root: Path, plm: str, view: str, prefix: str = "") -> Path:
    return root / "stores" / f"{prefix}{plm}_{view}"


def stored_layouts(store: PooledTapStore, spec: TapSpec) -> dict[str, SiteLayout]:
    """The layouts of the spec's sites as the store holds them: the store's layers, whatever blocks the spec names."""
    dims = ModelDims(**store.manifest.dims)
    layouts = site_layouts(replace(spec, sites=tuple(site for site in spec.sites if site in store.site_names)), dims)
    return {name: replace(layout, layers=tuple(store.layers(name))) for name, layout in layouts.items()}


def control_stores(root: Path, prefix: str = "", untrained: bool = True) -> ControlStores:
    """The stores the controls read; without `untrained`, the trained model's alone."""
    keys = [key for key in STORE_VARIANTS if untrained or key[0] == "trained"]
    missing = [str(store_path(root, *key, prefix)) for key in keys if not (store_path(root, *key, prefix) / "manifest.json").exists()]
    if missing:
        raise SystemExit(f"Control stores are missing ({', '.join(missing)}); run `embed --controls` or train with --no-controls.")
    return ControlStores({key: PooledTapStore(store_path(root, *key, prefix)) for key in keys})


StorePass = tuple[str, list[str], dict[bool, Path], Callable[[str], str] | None]


def store_passes(
    datasets: dict[str, ConceptData | PairData],
    root: Path,
    controls: bool,
    seed: int,
    normalize: bool = False,
    untrained: bool = True,
    prefix: str = "",
) -> list[StorePass]:
    """(model state, sequences, store roots, input transform) of each forward pass the stores need.

    With `normalize`, every store pools L2-normalized token states, so the `original` and `scrambled` stores are the
    normalized variant's inputs and the `l2_normalized` control equals `original`. Without `untrained`, the control
    stores are the trained model's alone. `prefix` names a task family's own stores, such as the pair task's."""
    every = sorted({sequence for data in datasets.values() for split in data.splits.values() for sequence in split.sequences})
    tested = sorted({sequence for data in datasets.values() for sequence in data.splits["test"].sequences})

    def scrambled(sequence: str) -> str:
        return scramble(sequence, scramble_seed(sequence, seed))

    def path(plm: str, view: str) -> Path:
        return store_path(root, plm, view, prefix)

    passes: list[StorePass] = [("trained", every, {normalize: path("trained", "original")}, None)]
    if controls:
        passes += [
            ("trained", tested, {True: path("trained", "l2_normalized")}, None),
            ("trained", tested, {normalize: path("trained", "scrambled")}, scrambled),
        ]
        if not untrained:
            return passes
        if normalize:  # both untrained stores are normalized, so they take two passes
            passes += [
                ("untrained", tested, {True: path("untrained", "original")}, None),
                ("untrained", tested, {True: path("untrained", "l2_normalized")}, None),
            ]
        else:
            passes.append(("untrained", tested, {False: path("untrained", "original"), True: path("untrained", "l2_normalized")}, None))
        passes.append(("untrained", tested, {normalize: path("untrained", "scrambled")}, scrambled))
    return passes


def write_stores(
    model: torch.nn.Module,
    tokenizer: object,
    model_name: str,
    datasets: dict[str, ConceptData | PairData],
    root: Path,
    spec: TapSpec,
    controls: bool,
    seed: int,
    max_tokens: int,
    device: torch.device,
    pilot: object | None = None,
    normalize: bool = False,
    untrained: bool = True,
    prefix: str = "",
) -> None:
    """The pooled stores of the protein tasks (or, with a `prefix`, of the pair task's proteins): every sequence under
    the trained model, and the control stores.

    With a `foundry.pilot.Pilot`, only the first pass runs, under the pilot's watch, and its store is left incomplete.
    """
    passes = store_passes(datasets, root, controls, seed, normalize, untrained, prefix)
    if pilot is not None:
        pilot.total_steps = sum(len(token_batches(sorted(set(sequences), key=len, reverse=True), max_tokens)) for _, sequences, _, _ in passes)
        _, sequences, roots, _ = passes[0]
        embed_pooled(model, tokenizer, sequences, spec, roots, model_name, "trained", max_tokens, device, watch=pilot.watch)
        return
    untrained = None
    for plm, sequences, roots, transform in passes:
        if all((path / "manifest.json").exists() for path in roots.values()):
            print(f"have {', '.join(str(path) for path in roots.values())}")
            continue
        if plm == "untrained" and untrained is None:
            untrained = untrained_copy(model, seed)
        network = model if plm == "trained" else untrained
        variant = f"{plm}_scrambled" if transform is not None else plm
        started = time.time()
        embed_pooled(
            network, tokenizer, sequences, spec, roots, model_name, variant, max_tokens, device,
            transform=transform, seed=seed if transform is not None else None,
        )
        print(f"{variant}: {len(sequences)} sequences in {time.time() - started:.0f} s")


def make_pilot(folder: Path, name: str, item_unit: str, items: Callable[[TokenBatch], int] | int, **options: object) -> object:
    from foundry.pilot import Pilot

    folder.mkdir(parents=True, exist_ok=True)
    return Pilot(items=items, item_unit=item_unit, receipt=folder / f"pilot_{name}.json", **options)


def command_embed(args: argparse.Namespace) -> None:
    pilot = None
    if args.pilot:
        pilot = make_pilot(Path(args.root), "embed", "tokens", lambda batch: len(batch.sequences) * batch.padded_length)
    device = torch.device(args.device)
    spec = tap_spec(args)
    tasks = parse_tasks(args.tasks)
    model, tokenizer = load_model(args.model, device, random_init=args.random_init, seed=args.seed)
    untrained = not args.no_untrained
    protein = [name for name in tasks if name in PROTEIN_TASKS]
    if protein:
        write_stores(model, tokenizer, model_label(args), load_tasks(protein, args), Path(args.root), spec, args.controls, args.seed,
                     args.max_tokens, device, pilot, args.normalize, untrained)
    if PAIR_TASK in tasks and pilot is None:
        write_stores(model, tokenizer, model_label(args), load_tasks([PAIR_TASK], args), Path(args.root), spec, args.controls, args.seed,
                     args.max_tokens, device, None, args.normalize, untrained, PAIR_PREFIX)


def training_settings(args: argparse.Namespace) -> TrainingSettings:
    probe = ProbeSettings(kind=args.probe, hidden=args.hidden, dropout=args.dropout)
    return TrainingSettings(
        learning_rates=tuple(args.lr),
        weight_decays=tuple(args.wd),
        batch_size=args.batch_size,
        max_tokens=args.train_tokens,
        max_epochs=args.max_epochs,
        patience=args.patience,
        validate_every=args.validate_every,
        seed=args.seed,
        controls=not args.no_controls,
        probe=probe,
        normalize=args.normalize,
        pooling=args.pooling,
        untrained_controls=not args.no_untrained,
        site_budget_gb=args.site_budget_gb,
    )


def write_results(folder: Path, results: TaskResults, weights: dict[str, dict[str, torch.Tensor]], meta: dict[str, object]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in results.frames().items():
        frame.to_parquet(folder / f"{name}.parquet", index=False)
    flat = {f"{prefix}.{key}": value.contiguous() for prefix, tensors in weights.items() for key, value in tensors.items()}
    save_file(flat, str(folder / "probes.safetensors"))
    # run.json last: its presence marks a finished task
    (folder / "run.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8", newline="\n")


def train_tasks(
    tasks: Sequence[str],
    root: Path,
    label: str,
    spec: TapSpec,
    settings: TrainingSettings,
    device: torch.device,
    read_task: Callable[[str], ConceptData],
    read_model: Callable[[], tuple[torch.nn.Module, object]],
    overwrite: bool = False,
    meta: dict[str, object] | None = None,
    pilot: object | None = None,
    sweep: bool = False,
) -> None:
    """Train and score every probe of each task, writing `root/results/<label>/<task>/`; finished tasks are skipped.

    With `sweep`, every grid setting is kept and scored on the validation split alone, and each task writes
    `root/sweeps/<label>/<task>/sweep.parquet` (see `selection`). With a `foundry.pilot.Pilot`, the first residue
    task trains under the pilot's watch for one window, and nothing is written.
    """
    model = tokenizer = None
    if pilot is not None:
        name = next(name for name in tasks if read_task(name).task.level == "residue")
        data = read_task(name)
        model, tokenizer = read_model()
        pilot.total_steps = len(token_batches(data.splits["train"].sequences, settings.max_tokens))
        layouts = site_layouts(spec, model_dims(model))
        train_residue_task(model, tokenizer, spec, layouts, data, settings, device, watch=pilot.watch)
        return
    for name in tasks:
        folder = root / ("sweeps" if sweep else "results") / label / name
        if (folder / "run.json").exists() and not overwrite:
            print(f"have {folder}")
            continue
        data = read_task(name)
        weights: dict[str, dict[str, torch.Tensor]] = {}
        started = time.time()
        if data.task.level in ("protein", "pair"):
            prefix = PAIR_PREFIX if data.task.level == "pair" else ""
            store = PooledTapStore(store_path(root, "trained", "original", prefix))
            if store.manifest.normalized != settings.normalize:
                state = "normalized" if store.manifest.normalized else "raw"
                raise SystemExit(f"{store.root} pools {state} token states; embed and train must agree on --normalize.")
            layouts = stored_layouts(store, spec)
            sweeper, trainer = (sweep_pair_task, train_pair_task) if data.task.level == "pair" else (sweep_protein_task, train_protein_task)
            if sweep:
                results = sweeper(data, store, layouts, settings, device)
            else:
                controls = control_stores(root, prefix, settings.untrained_controls) if settings.controls else None
                results = trainer(data, store, layouts, settings, device, controls, save_weights=weights)
        else:
            if model is None:
                model, tokenizer = read_model()
            layouts = site_layouts(spec, model_dims(model))
            if sweep:
                results = sweep_residue_task(model, tokenizer, spec, layouts, data, settings, device)
            else:
                results = train_residue_task(model, tokenizer, spec, layouts, data, settings, device, save_weights=weights)
        elapsed = time.time() - started
        record = {
            "task": asdict(data.task),
            "spec": asdict(spec),
            "settings": asdict(settings),
            "seconds": elapsed,
            # proteins, or pairs for the pair task (whose proteins are counted apart)
            "rows": {split: len(part.labels) for split, part in data.splits.items()},
            **({"proteins": {split: len(part.sequences) for split, part in data.splits.items()}} if data.task.level == "pair" else {}),
            **(meta or {}),
        }
        if sweep:
            write_sweep(folder, results, record)
        else:
            write_results(folder, results, weights, record)
        print(f"{name}: {elapsed:.0f} s -> {folder}")


def recorded_settings(record: dict[str, object]) -> TrainingSettings:
    """The training settings a task's run.json records (runs from before `normalize` and `pooling` used neither)."""
    saved = dict(record["settings"])
    probe = dict(saved.pop("probe"))
    probe["betas"] = tuple(probe["betas"])
    saved["learning_rates"], saved["weight_decays"] = tuple(saved["learning_rates"]), tuple(saved["weight_decays"])
    return TrainingSettings(**saved, probe=ProbeSettings(**probe))


def recorded_spec(record: dict[str, object]) -> TapSpec:
    spec = record["spec"]
    return TapSpec(sites=tuple(spec["sites"]), blocks=tuple(spec["blocks"]) if spec["blocks"] else None, per_head=spec["per_head"])


def restore_selected(
    weights: dict[str, torch.Tensor],
    layouts: dict[str, SiteLayout],
    data: ConceptData,
    settings: TrainingSettings,
    device: torch.device,
) -> dict[str, dict[str, ProbeBank]]:
    """Every site's trained banks from a results folder's `probes.safetensors`, named as the trainers name them."""
    task = data.task
    loss = task.loss if task.level != "residue" else "multiclass"  # a residue bank scores one class per residue
    selected: dict[str, dict[str, ProbeBank]] = {}
    for site, layout in layouts.items():
        selected[site] = {}
        for family, names in (("full", layout.full_names()), ("head", layout.head_names() if layout.heads else None)):
            if names is None:
                continue
            prefix = f"{site}.{family}."
            saved = {key.removeprefix(prefix): value for key, value in weights.items() if key.startswith(prefix)}
            selected[site][family] = restore_bank(names, saved, task.outputs, loss, settings, device)
    return selected


def validate_tasks(
    root: Path,
    label: str,
    tasks: Sequence[str],
    device: torch.device,
    read_task: Callable[[str], ConceptData],
    read_model: Callable[[bool, int], tuple[torch.nn.Module, object]],
    overwrite: bool = False,
) -> list[Path]:
    """Score the kept probes of every finished task under `root/results/<label>/` on its validation split, writing
    `validation.parquet` beside the scores (`train` writes it itself; this serves runs trained before it did). Each
    probe's recomputed validation loss is compared with the one its training recorded, as a check that the weights,
    inputs and model were all restored."""
    written = []
    loaded: tuple[tuple[bool, int], tuple[torch.nn.Module, object]] | None = None
    for name in tasks:
        folder = root / "results" / label / name
        if not (folder / "run.json").exists():
            continue
        if (folder / "validation.parquet").exists() and not overwrite:
            print(f"have {folder / 'validation.parquet'}")
            continue
        record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        settings, spec, data = recorded_settings(record), recorded_spec(record), read_task(name)
        weights = load_file(str(folder / "probes.safetensors"))
        started = time.time()
        if data.task.level in ("protein", "pair"):
            store = PooledTapStore(store_path(root, "trained", "original", PAIR_PREFIX if data.task.level == "pair" else ""))
            if store.manifest.normalized != settings.normalize:
                raise SystemExit(f"{store.root} does not pool the token states {folder} was trained on (normalize={settings.normalize}).")
            layouts = stored_layouts(store, spec)
            validator = validate_pair_task if data.task.level == "pair" else validate_protein_task
            results = validator(data, store, layouts, restore_selected(weights, layouts, data, settings, device), settings, device)
        else:
            key = (bool(record.get("random_init", False)), settings.seed)
            if loaded is None or loaded[0] != key:
                loaded = (key, read_model(*key))
            model, tokenizer = loaded[1]
            layouts = site_layouts(spec, model_dims(model))
            selected = restore_selected(weights, layouts, data, settings, device)
            results = validate_residue_task(model, tokenizer, spec, layouts, selected, data, settings, device)
        table = results.frames()["validation"]
        recorded = pd.read_parquet(folder / "training.parquet")[["family", "probe_name", "best_validation_loss"]]
        check = table.merge(recorded, on=["family", "probe_name"], validate="one_to_one")
        gap = ((check["validation_loss"] - check["best_validation_loss"]).abs() / check["best_validation_loss"].abs().clip(lower=1e-8)).max()
        table.to_parquet(folder / "validation.parquet", index=False)
        written.append(folder / "validation.parquet")
        print(f"{name}: {len(table)} probes in {time.time() - started:.0f} s; largest relative gap to the recorded validation loss {gap:.1e}"
              f"{'  (CHECK: inputs or model differ from training)' if gap > 0.02 else ''} -> {folder / 'validation.parquet'}")
    return written


def write_sweep(folder: Path, results: TaskResults, meta: dict[str, object]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    results.sweep_frame().to_parquet(folder / "sweep.parquet", index=False)
    # run.json last: its presence marks a finished task
    (folder / "run.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8", newline="\n")


def command_train(args: argparse.Namespace) -> None:
    pilot = None
    if args.pilot:
        pilot = make_pilot(Path(args.root), args.command, "tokens", lambda batch: len(batch.sequences) * batch.padded_length)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    train_tasks(
        parse_tasks(args.tasks),
        Path(args.root),
        args.label,
        tap_spec(args),
        training_settings(args),
        device,
        read_task=lambda name: load_task(name, args),
        read_model=lambda: load_model(args.model, device, random_init=args.random_init, seed=args.seed),
        overwrite=args.overwrite,
        meta={"model": model_label(args), "random_init": args.random_init, "normalize": args.normalize, "pooling": args.pooling,
              "repo": args.repo, "revision": args.revision, "local_data": args.local_data, "pair_repo": args.pair_repo},
        pilot=pilot,
        sweep=args.command == "sweep",
    )


def command_validate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True  # as in training
    validate_tasks(
        Path(args.root),
        args.label,
        parse_tasks(args.tasks),
        device,
        read_task=lambda name: load_task(name, args),
        read_model=lambda random_init, seed: load_model(args.model, device, random_init=random_init, seed=seed),
        overwrite=args.overwrite,
    )


def command_select(args: argparse.Namespace) -> None:
    root = Path(args.root)
    sweeps = read_sweeps(root, args.labels)
    if args.loss != "all":
        # squared error and cross entropy want different step sizes, so each loss can have its own global setting
        sweeps = sweeps[(sweeps["kind"] == "regression") == (args.loss == "regression")]
    ranked = rank_settings(sweeps)
    chosen = choose_setting(ranked, args.tolerance)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(ranked.head(args.show).to_string(index=False))
        print("\nEach task's own best setting:")
        print(task_optima(sweeps).to_string(index=False))
    choice = {name: (value.item() if hasattr(value, "item") else value) for name, value in chosen.items()}
    print(f"\nchosen: {choice}")
    suffix = "" if args.loss == "all" else f"_{args.loss}"
    out = Path(args.out) if args.out else root / "sweeps" / f"selection_{'_'.join(args.labels)}{suffix}.json"
    record = {"labels": list(args.labels), "loss": args.loss, "tasks": sorted(set(sweeps["task"])), "tolerance": args.tolerance,
              "chosen": choice, "ranking": ranked.to_dict(orient="records")}
    out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8", newline="\n")


def command_phospho(args: argparse.Namespace) -> None:
    pilot = None
    if args.pilot:
        # a protein is a step: two forward passes over its mutants and the regressions of every site and layer
        pilot = make_pilot(Path(args.root), "phospho", "proteins", 1, steps=4, warmup=1, total_steps=args.proteins)
    device = torch.device(args.device)
    # float32 weights and activations: mutants of one protein differ by about a part in a thousand in their pooled
    # states. TF32 products (--tf32) round each state by about 5e-4, which pooling over the tokens shrinks below the
    # one-residue difference by about a hundredfold, at twice the speed.
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    spec = tap_spec(args, per_head=False)
    data = load_concept("phosphorylation", repo=args.repo, revision=args.revision, local_dir=args.local_data)
    model, tokenizer = load_model(args.model, device, torch.float32, random_init=args.random_init, seed=args.seed)
    untrained = untrained_copy(model, args.seed)
    folder = Path(args.root) / "interventions" / args.label
    scores = run_phospho_interventions(
        model, untrained, tokenizer, data, spec, folder / "proteins", device,
        proteins=args.proteins, seed=args.data_seed, split_seed=args.split_seed, iterations=args.iterations, max_tokens=args.max_tokens,
        watch=pilot.watch if pilot is not None else None,
    )
    if pilot is not None:
        return
    scores.to_parquet(folder / "scores.parquet", index=False)
    paired_tests(scores).to_parquet(folder / "tests.parquet", index=False)
    print(f"{scores['protein'].nunique()} proteins -> {folder}")


def command_pca(args: argparse.Namespace) -> None:
    root = Path(args.root)
    store = PooledTapStore(store_path(root, "trained", "original"))
    scrambled_path = store_path(root, "trained", "scrambled")
    scrambled = PooledTapStore(scrambled_path) if (scrambled_path / "manifest.json").exists() else None
    folder = root / "latent" / args.label
    folder.mkdir(parents=True, exist_ok=True)
    for name in parse_tasks(args.tasks):
        data = load_concept(name, repo=args.repo, revision=args.revision, local_dir=args.local_data)
        groups = label_sets(data, args.top)
        tested = data.splits["test"].sequences if scrambled is not None else []
        frames = [
            project_layers(store, groups, site, [layer for layer in parse_blocks(args.layers) or store.layers(site) if layer in store.layers(site)], scrambled, tested)
            for site in parse_sites(args.sites)
        ]
        table = pd.concat(frames, ignore_index=True)
        table.insert(0, "task", name)
        table.to_parquet(folder / f"{name}.parquet", index=False)
        print(f"{name}: {len(groups)} proteins, {table['group'].nunique()} groups -> {folder / f'{name}.parquet'}")


def command_figures(args: argparse.Namespace) -> None:
    from . import figures

    roots, random_roots = [Path(root) for root in args.roots], [Path(root) for root in args.random_roots]
    first = roots[0]
    metadata = {path.parent.name: json.loads(path.read_text(encoding="utf-8"))["task"] for path in (first / "results" / args.label).glob("*/run.json")}
    rankings = {regime: rank_settings(read_sweeps(first, labels)) for regime, labels in (("protein", args.protein_sweeps), ("residue", args.residue_sweeps)) if labels}
    phospho = [first / "interventions" / label for label in args.interventions if (first / "interventions" / label / "scores.parquet").exists()]
    models = [("trained model", first)] + [("random-init model", root) for root in random_roots[:1]]
    latent = {name: root / "latent" / args.latent for name, root in models if (root / "latent" / args.latent).exists()}
    written = figures.draw(roots, Path(args.out), args.blocks, args.label, random_roots, phospho, latent, rankings, metadata)
    for path in written:
        print(path)


def parse_variant(item: str, label: str) -> tuple[str, list[tuple[Path, str]]]:
    """NAME=ROOT[:LABEL][,ROOT[:LABEL]...] as the variant's name and its replicate (root, results label) pairs."""
    name, _, spec = item.partition("=")
    if not name or not spec:
        raise SystemExit(f"--variants takes NAME=ROOT[:LABEL][,ROOT[:LABEL]...], not {item!r}.")
    replicates = []
    for part in spec.split(","):
        root, separator, named = part.rpartition(":")
        replicates.append((Path(root), named) if separator and "/" not in named and "\\" not in named else (Path(part), label))
    return name, replicates


def command_compare(args: argparse.Namespace) -> None:
    from . import figures

    variants = dict(parse_variant(item, args.label) for item in args.variants)
    for path in figures.compare(variants, Path(args.out), args.blocks):
        print(path)


def command_tokens(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    spec = tap_spec(args, per_head=False)
    sequences = [line.strip() for line in Path(args.sequences).read_text(encoding="utf-8").splitlines() if line.strip()]
    model, tokenizer = load_model(args.model, device, random_init=args.random_init, seed=args.seed)
    embed_tokens(model, tokenizer, sequences, spec, args.out, model_label(args), max_tokens=args.max_tokens, device=device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m protify.interpretability", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--model", default="ESMC-300")
        sub.add_argument("--sites", default="all", help="'all', 'residual', or comma-separated site names")
        sub.add_argument("--blocks", default="all", help="'all' or comma-separated block indices")
        sub.add_argument("--no-heads", action="store_true", help="skip the per-head probes of head-split sites")
        sub.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        sub.add_argument("--seed", type=int, default=24895)
        sub.add_argument("--max-tokens", type=int, default=65536, help="padded tokens per embedding batch")
        sub.add_argument("--pilot", action="store_true", help="time the first steps (foundry.pilot), write the report under --root, and stop")
        sub.add_argument("--random-init", action="store_true", help="reset every module to PyTorch's default initialization (seeded by --seed): the random-weights control")

    def normalizing(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--normalize", action="store_true", help="L2-normalize every token state along its width before pooling (embed) or probing (train)")
        sub.add_argument("--no-untrained", action="store_true", help="leave out the untrained model's control stores (embed) and conditions (train)")

    def data_sources(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--repo", default=DEFAULT_REPO)
        sub.add_argument("--revision", default=None)
        sub.add_argument("--local-data", default=None)
        sub.add_argument("--pair-repo", default=PAIR_REPO, help="the pairs of the protein-protein interaction task 'ppi'")
        sub.add_argument("--pair-local-data", default=None, help="a folder holding the pair task's train, validation and test parquet files")

    embed = commands.add_parser("embed", help="write the pooled tap stores of the protein tasks, and of the pair task's proteins with --tasks ppi")
    common(embed)
    normalizing(embed)
    data_sources(embed)
    embed.add_argument("--root", required=True)
    embed.add_argument("--tasks", default="protein")
    embed.add_argument("--controls", action="store_true", help="also write the control stores of the test proteins")
    embed.set_defaults(handler=command_embed)

    def training(sub: argparse.ArgumentParser) -> None:
        common(sub)
        normalizing(sub)
        data_sources(sub)
        sub.add_argument("--root", required=True)
        sub.add_argument("--label", default="linear", help="results subfolder, so settings can be compared")
        sub.add_argument("--tasks", default="all")
        sub.add_argument("--site-budget-gb", type=float, default=8.0, help="pooled rows of one site held on the device at once; wider sites train in layer chunks")
        sub.add_argument("--lr", type=float, nargs="+", default=[1e-3])
        sub.add_argument("--wd", type=float, nargs="+", default=[1e-4])
        sub.add_argument("--batch-size", type=int, default=64, help="proteins per step, protein tasks")
        sub.add_argument("--train-tokens", type=int, default=32768, help="padded tokens per step, residue tasks")
        sub.add_argument("--max-epochs", type=int, default=50)
        sub.add_argument("--patience", type=int, default=5)
        sub.add_argument("--validate-every", type=int, default=0, help="steps between validations; 0 is once per epoch")
        sub.add_argument("--probe", choices=("linear", "mlp"), default="linear")
        sub.add_argument("--hidden", type=int, default=512)
        sub.add_argument("--dropout", type=float, default=0.1)
        sub.add_argument("--no-controls", action="store_true")
        sub.add_argument("--pooling", choices=("mean_var", "mean"), default="mean_var", help="what protein probes read of a pooled state")
        sub.add_argument("--overwrite", action="store_true")
        sub.set_defaults(handler=command_train)

    training(commands.add_parser("train", help="train and score every probe of each task"))
    training(commands.add_parser("sweep", help="train every probe under every --lr and --wd setting and score each on validation"))

    validated = commands.add_parser("validate", help="score the kept probes of finished tasks on their validation split")
    validated.add_argument("--root", required=True)
    validated.add_argument("--label", default="linear", help="results label")
    validated.add_argument("--tasks", default="all")
    validated.add_argument("--model", default="ESMC-300", help="the model residue tasks stream from; each run.json gives its seed and random init")
    validated.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    data_sources(validated)
    validated.add_argument("--overwrite", action="store_true")
    validated.set_defaults(handler=command_validate)

    select = commands.add_parser("select", help="choose one global setting from the sweeps of one or more labels")
    select.add_argument("--root", required=True)
    select.add_argument("--labels", nargs="+", required=True, help="sweep labels to compare, such as one per batch size")
    select.add_argument("--tolerance", type=float, default=0.005, help="mean validation shortfall a faster label may give up")
    select.add_argument("--show", type=int, default=20, help="ranked settings to print")
    select.add_argument("--loss", choices=("all", "classification", "regression"), default="all",
                        help="choose among the tasks of one loss: regression (squared error) or classification (cross entropy)")
    select.add_argument("--out", default=None, help="where to write the selection; default root/sweeps/selection_<labels>.json")
    select.set_defaults(handler=command_select)

    phospho = commands.add_parser("phospho", help="phosphomimetic scanning: logistic probes of mutant activity on every site and layer")
    common(phospho)
    phospho.add_argument("--root", required=True)
    phospho.add_argument("--label", default="phospho")
    phospho.add_argument("--repo", default=DEFAULT_REPO)
    phospho.add_argument("--revision", default=None)
    phospho.add_argument("--local-data", default=None)
    phospho.add_argument("--proteins", type=int, default=1000, help="phosphoproteins drawn at random")
    phospho.add_argument("--data-seed", type=int, default=184502, help="seed of the protein draw, and of each protein's split unless --split-seed")
    phospho.add_argument("--split-seed", type=int, default=None, help="seed of each protein's split and label shuffle, for replicates on the same proteins")
    phospho.add_argument("--iterations", type=int, default=100, help="most Newton steps of a logistic regression")
    phospho.add_argument("--tf32", action="store_true", help="TF32 matrix products in the float32 forward passes")
    phospho.set_defaults(handler=command_phospho)

    pca = commands.add_parser("pca", help="principal coordinates of pooled states, proteins grouped by label set")
    pca.add_argument("--root", required=True)
    pca.add_argument("--label", default="pca")
    pca.add_argument("--tasks", default="interpro_family")
    pca.add_argument("--sites", default="resid_post")
    pca.add_argument("--layers", default="0,5,10,15,20", help="'all' or comma-separated layers of each site")
    pca.add_argument("--top", type=int, default=5, help="label sets named; the rest are 'other'")
    pca.add_argument("--repo", default=DEFAULT_REPO)
    pca.add_argument("--revision", default=None)
    pca.add_argument("--local-data", default=None)
    pca.set_defaults(handler=command_pca)

    drawn = commands.add_parser("figures", help="draw the figures and summary tables of one or more replicate runs")
    drawn.add_argument("--roots", nargs="+", required=True, help="one root per seed replicate; sweeps, interventions and latent views are read from the first")
    drawn.add_argument("--random-roots", nargs="*", default=[], help="the roots of the same analysis on a randomly initialized model, one per seed")
    drawn.add_argument("--out", required=True)
    drawn.add_argument("--label", default="linear", help="results label")
    drawn.add_argument("--blocks", type=int, default=30, help="blocks of the model, for the depth axis")
    drawn.add_argument("--protein-sweeps", nargs="*", default=[], help="protein sweep labels to plot")
    drawn.add_argument("--residue-sweeps", nargs="*", default=[], help="residue sweep labels to plot")
    drawn.add_argument("--interventions", nargs="*", default=["phospho"], help="interventions labels, one per replicate, averaged per protein")
    drawn.add_argument("--latent", default="pca", help="principal coordinates label")
    drawn.set_defaults(handler=command_figures)

    compared = commands.add_parser("compare", help="compare variants of the analysis, such as raw and normalized inputs")
    compared.add_argument("--variants", nargs="+", required=True, help="NAME=ROOT[:LABEL][,ROOT[:LABEL]...], one root per replicate; the first is the reference")
    compared.add_argument("--out", required=True)
    compared.add_argument("--label", default="linear", help="results label of the roots that name none")
    compared.add_argument("--blocks", type=int, default=30, help="blocks of the model, for the depth axis")
    compared.set_defaults(handler=command_compare)

    tokens = commands.add_parser("tokens", help="write token-wise taps of chosen sites for a list of sequences")
    common(tokens)
    tokens.add_argument("--sequences", required=True, help="a text file with one sequence per line")
    tokens.add_argument("--out", required=True)
    tokens.set_defaults(handler=command_tokens)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)
