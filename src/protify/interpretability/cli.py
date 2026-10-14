"""Command line for tap probing: pooled stores, token stores, and probe training on the concept tasks.

    python -m protify.interpretability embed  --root RUN [--tasks protein] [--sites all] [--controls]
    python -m protify.interpretability train  --root RUN [--tasks all] [--lr 1e-3 ...] [--probe linear]
    python -m protify.interpretability tokens --root RUN --sequences FILE --sites resid_post

`embed` writes the pooled stores the protein tasks read: every protein-task sequence through the trained model,
and, with `--controls`, the test proteins under the untrained model, scrambled, and token-normalized. `train` trains
and scores every probe of each task: protein tasks from the stores, residue tasks by streaming from the model.
Results go to `RUN/results/<task>/` as parquet tables and a safetensors file of the selected probe weights.
"""

from __future__ import annotations

import argparse
import json
import time
import torch

from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from safetensors.torch import save_file

from .concepts import ALL_TASKS, DEFAULT_REPO, PROTEIN_TASKS, RESIDUE_TASKS, ConceptData, load_concept, scramble, scramble_seed
from .capture import model_dims
from .pooled_store import PooledTapStore, embed_pooled
from .probe_bank import ProbeSettings
from .protein_probing import ControlStores, train_protein_task
from .residue_probing import train_residue_task, untrained_copy
from .sites import ModelDims, SiteLayout, TapSpec, parse_sites, site_layouts
from .token_store import embed_tokens
from .training import TaskResults, TrainingSettings


STORE_VARIANTS = (
    ("trained", "original"),
    ("trained", "l2_normalized"),
    ("trained", "scrambled"),
    ("untrained", "original"),
    ("untrained", "l2_normalized"),
    ("untrained", "scrambled"),
)


def parse_tasks(value: str) -> tuple[str, ...]:
    if value == "all":
        return ALL_TASKS
    if value == "protein":
        return PROTEIN_TASKS
    if value == "residue":
        return RESIDUE_TASKS
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = sorted(set(names) - set(ALL_TASKS))
    if unknown:
        raise SystemExit(f"Unknown tasks {unknown}; known tasks are {list(ALL_TASKS)}.")
    return names


def parse_blocks(value: str) -> tuple[int, ...] | None:
    return None if value == "all" else tuple(int(part) for part in value.split(","))


def load_model(model_name: str, device: torch.device) -> tuple[torch.nn.Module, object]:
    from ..base_models.get_base_models import get_base_model

    model, tokenizer = get_base_model(model_name, dtype=torch.bfloat16)
    return model.to(device).eval(), tokenizer


def load_tasks(tasks: Sequence[str], args: argparse.Namespace) -> dict[str, ConceptData]:
    return {
        name: load_concept(name, repo=args.repo, revision=args.revision, local_dir=args.local_data)
        for name in tasks
    }


def store_path(root: Path, plm: str, view: str) -> Path:
    return root / "stores" / f"{plm}_{view}"


def stored_layouts(store: PooledTapStore, spec: TapSpec) -> dict[str, SiteLayout]:
    """The layouts of the spec's sites as the store holds them: the store's layers, whatever blocks the spec names."""
    dims = ModelDims(**store.manifest.dims)
    layouts = site_layouts(replace(spec, sites=tuple(site for site in spec.sites if site in store.site_names)), dims)
    return {name: replace(layout, layers=tuple(store.layers(name))) for name, layout in layouts.items()}


def control_stores(root: Path) -> ControlStores:
    missing = [str(store_path(root, *key)) for key in STORE_VARIANTS if not (store_path(root, *key) / "manifest.json").exists()]
    if missing:
        raise SystemExit(f"Control stores are missing ({', '.join(missing)}); run `embed --controls` or train with --no-controls.")
    return ControlStores({key: PooledTapStore(store_path(root, *key)) for key in STORE_VARIANTS})


def write_stores(
    model: torch.nn.Module,
    tokenizer: object,
    model_name: str,
    datasets: dict[str, ConceptData],
    root: Path,
    spec: TapSpec,
    controls: bool,
    seed: int,
    max_tokens: int,
    device: torch.device,
) -> None:
    """The pooled stores of the protein tasks: every sequence under the trained model, and the control stores."""
    every = sorted({sequence for data in datasets.values() for split in data.splits.values() for sequence in split.sequences})
    tested = sorted({sequence for data in datasets.values() for sequence in data.splits["test"].sequences})

    def scrambled(sequence: str) -> str:
        return scramble(sequence, scramble_seed(sequence, seed))

    passes = [("trained", every, {False: store_path(root, "trained", "original")}, None)]
    if controls:
        passes += [
            ("trained", tested, {True: store_path(root, "trained", "l2_normalized")}, None),
            ("trained", tested, {False: store_path(root, "trained", "scrambled")}, scrambled),
            ("untrained", tested, {False: store_path(root, "untrained", "original"), True: store_path(root, "untrained", "l2_normalized")}, None),
            ("untrained", tested, {False: store_path(root, "untrained", "scrambled")}, scrambled),
        ]
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


def command_embed(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    spec = TapSpec(sites=parse_sites(args.sites), blocks=parse_blocks(args.blocks), per_head=not args.no_heads)
    tasks = [name for name in parse_tasks(args.tasks) if name in PROTEIN_TASKS]
    datasets = load_tasks(tasks, args)
    model, tokenizer = load_model(args.model, device)
    write_stores(model, tokenizer, args.model, datasets, Path(args.root), spec, args.controls, args.seed, args.max_tokens, device)


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
) -> None:
    """Train and score every probe of each task, writing `root/results/<label>/<task>/`; finished tasks are skipped."""
    model = tokenizer = None
    for name in tasks:
        folder = root / "results" / label / name
        if (folder / "run.json").exists() and not overwrite:
            print(f"have {folder}")
            continue
        data = read_task(name)
        weights: dict[str, dict[str, torch.Tensor]] = {}
        started = time.time()
        if data.task.level == "protein":
            store = PooledTapStore(store_path(root, "trained", "original"))
            controls = control_stores(root) if settings.controls else None
            results = train_protein_task(data, store, stored_layouts(store, spec), settings, device, controls, save_weights=weights)
        else:
            if model is None:
                model, tokenizer = read_model()
            layouts = site_layouts(spec, model_dims(model))
            results = train_residue_task(model, tokenizer, spec, layouts, data, settings, device, save_weights=weights)
        elapsed = time.time() - started
        record = {
            "task": asdict(data.task),
            "spec": asdict(spec),
            "settings": asdict(settings),
            "seconds": elapsed,
            "rows": {split: len(part.sequences) for split, part in data.splits.items()},
            **(meta or {}),
        }
        write_results(folder, results, weights, record)
        print(f"{name}: {elapsed:.0f} s -> {folder}")


def command_train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    spec = TapSpec(sites=parse_sites(args.sites), blocks=parse_blocks(args.blocks), per_head=not args.no_heads)
    train_tasks(
        parse_tasks(args.tasks),
        Path(args.root),
        args.label,
        spec,
        training_settings(args),
        device,
        read_task=lambda name: load_concept(name, repo=args.repo, revision=args.revision, local_dir=args.local_data),
        read_model=lambda: load_model(args.model, device),
        overwrite=args.overwrite,
        meta={"model": args.model, "repo": args.repo, "revision": args.revision, "local_data": args.local_data},
    )


def command_tokens(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    spec = TapSpec(sites=parse_sites(args.sites), blocks=parse_blocks(args.blocks), per_head=False)
    sequences = [line.strip() for line in Path(args.sequences).read_text(encoding="utf-8").splitlines() if line.strip()]
    model, tokenizer = load_model(args.model, device)
    embed_tokens(model, tokenizer, sequences, spec, args.out, args.model, max_tokens=args.max_tokens, device=device)


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

    embed = commands.add_parser("embed", help="write the pooled tap stores of the protein tasks")
    common(embed)
    embed.add_argument("--root", required=True)
    embed.add_argument("--tasks", default="protein")
    embed.add_argument("--controls", action="store_true", help="also write the control stores of the test proteins")
    embed.add_argument("--repo", default=DEFAULT_REPO)
    embed.add_argument("--revision", default=None)
    embed.add_argument("--local-data", default=None)
    embed.set_defaults(handler=command_embed)

    train = commands.add_parser("train", help="train and score every probe of each task")
    common(train)
    train.add_argument("--root", required=True)
    train.add_argument("--label", default="linear", help="results subfolder, so settings can be compared")
    train.add_argument("--tasks", default="all")
    train.add_argument("--repo", default=DEFAULT_REPO)
    train.add_argument("--revision", default=None)
    train.add_argument("--local-data", default=None)
    train.add_argument("--lr", type=float, nargs="+", default=[1e-3])
    train.add_argument("--wd", type=float, nargs="+", default=[1e-4])
    train.add_argument("--batch-size", type=int, default=64, help="proteins per step, protein tasks")
    train.add_argument("--train-tokens", type=int, default=32768, help="padded tokens per step, residue tasks")
    train.add_argument("--max-epochs", type=int, default=50)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--validate-every", type=int, default=0, help="steps between validations; 0 is once per epoch")
    train.add_argument("--probe", choices=("linear", "mlp"), default="linear")
    train.add_argument("--hidden", type=int, default=512)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--no-controls", action="store_true")
    train.add_argument("--overwrite", action="store_true")
    train.set_defaults(handler=command_train)

    tokens = commands.add_parser("tokens", help="write token-wise taps of chosen sites for a list of sequences")
    common(tokens)
    tokens.add_argument("--sequences", required=True, help="a text file with one sequence per line")
    tokens.add_argument("--out", required=True)
    tokens.set_defaults(handler=command_tokens)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)
