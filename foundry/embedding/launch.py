"""Run an embedding job in the profile's container: on a GH200 over SSH, on one named Modal GPU, or on this machine.

    python -m foundry.embedding.launch --compute ssh:atlas_gh200 -- qualify --base 300 --profile embedding_profile_v1 \
        --cache /home/ubuntu/hf-cache/hub --store /home/ubuntu/embedding-runs/qualify-300 \
        --receipt /home/ubuntu/embedding-runs/qualify-300.json
    python -m foundry.embedding.launch --compute modal --modal-gpu H100! -- qualify --base 300 \
        --profile embedding_profile_v1 --cache /hf/huggingface/hub --store /tmp/qualify-300 --receipt C:/ws-cache/q.json

Everything after `--` is `foundry.embedding.job`'s command line. Where it runs (`--compute`, `foundry.compute`):

- `ssh:<machine>`: `ws remote run` ships FastPLMs, foundry and `environments/embedding_runtime` to the machine and starts
  this module there with `--compute local`, detached; follow it with `ws remote status`.
- `local`, on a Linux GPU machine: builds the image from `environments/embedding_runtime` once, tagged by the digest of
  its Dockerfile and lock, and runs the job in it through `sudo -n docker`, the GPU machines' entrypoint
  (docs/conventions/compute.md): one GPU, `--ipc=host`, no network unless the command needs it (`push`, `--fetch`), the
  code read-only, and each path the job names mounted at the same path, so no argument is rewritten. Files are written
  as the calling user, and credentials pass by name only.
- `modal`: `modal/embedding_profile.py` builds the same Dockerfile on Modal and runs the job on the one GPU
  `--modal-gpu` names. The receipt comes back to the `--receipt` path here; every other path is a container path.

This module only orchestrates containers and imports nothing beyond the standard library and `foundry.compute`; the
arithmetic runs in the image, where the job attests the closure it finds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath

from foundry.compute import add_compute_argument, choose_compute, dispatch, launch_on_modal, launch_over_ssh, workspace_root


ENTITY = "fastplms"  # what `ws remote run` ships whole; foundry always goes with it
IMAGE_DIRECTORY = "environments/embedding_runtime"
IMAGE_FILES = ("Dockerfile", "requirements.lock")
IMAGE_NAME = "embedding_runtime"
MODAL_APP = "modal/embedding_profile.py"
# One GPU type each, never a list, and Hopper only: the reference class alone computes canonical rows
# (`numerics.check_reference_class`). `H100!` stops Modal moving a job to an H200.
MODAL_GPUS = ("H100!", "H200")
CONTAINER_RECEIPT = "/tmp/embedding_receipt.json"  # where a Modal run's job writes the receipt it returns
PANEL_SUFFIX = ".panel.safetensors"  # beside a `qualify` receipt: the rows another class is compared with
JOB_MODULE = "foundry.embedding.job"
ONLINE_COMMANDS = frozenset({"push"})  # every other command runs with no network unless it fetches checkpoints
MIRROR_COMMANDS = frozenset({"push", "relocate"})  # these mount the store's whole mirror: a push uploads it, a move renames in it
READ_PATHS = ("--input", "--source", "--random-weights")
WRITE_DIRECTORIES = ("--store",)
CREDENTIALS = ("HF_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
MANIFEST_FILE = "canonical.json"  # `foundry.embedding.canonical.MANIFEST_FILE`; this module imports no embedding code


def option_values(arguments: Sequence[str], names: Sequence[str]) -> dict[str, str]:
    """The values of ``names`` among the job's arguments, written `--name value` or `--name=value`."""
    found: dict[str, str] = {}
    for index, token in enumerate(arguments):
        name, equals, value = token.partition("=")
        if name not in names:
            continue
        if not equals:
            if index + 1 >= len(arguments):
                raise SystemExit(f"{name} needs a value.")
            value = arguments[index + 1]
        found[name] = value
    return found


def with_option(arguments: Sequence[str], name: str, value: str) -> list[str]:
    """The job's arguments with ``name`` set to ``value``, in place, in either spelling."""
    rewritten, skip = [], False
    for index, token in enumerate(arguments):
        if skip:
            skip = False
            continue
        if token == name:
            rewritten += [name, value]
            skip = index + 1 < len(arguments)
        elif token.startswith(name + "="):
            rewritten.append(f"{name}={value}")
        else:
            rewritten.append(token)
    return rewritten


def panel_path(receipt: str | Path) -> Path:
    """The panel file a `qualify` run writes beside its receipt."""
    receipt = Path(receipt)
    return receipt.with_name(receipt.stem + PANEL_SUFFIX)


def needs_network(arguments: Sequence[str]) -> bool:
    """Whether the job talks to the Hub or Modal: a push, or a run that downloads its pinned checkpoints."""
    return bool(arguments) and (arguments[0] in ONLINE_COMMANDS or "--fetch" in arguments)


def mirror_of(store: Path) -> Path | None:
    """The mirror root a profile store sits in, `<mirror>/<model>/<kind>` (an older store `<mirror>/profile_v2/<model>/
    <kind>`), read from its manifest's `path_in_repo`; None for a store without one (v2) or not at that path, which the
    job's own checks then name. `push` uploads the whole mirror with the store's folder as its only pattern, and the Hub
    client keeps its upload state in the mirror, so a push mounts the mirror rather than the store alone; `relocate`
    renames the store inside it."""
    manifest = store / MANIFEST_FILE
    if not manifest.is_file():
        return None
    folder = json.loads(manifest.read_text(encoding="utf-8")).get("path_in_repo")
    parts = PurePosixPath(folder).parts if isinstance(folder, str) and folder else ()
    if not parts or len(store.parts) <= len(parts) or store.parts[-len(parts):] != parts:
        return None
    return store.parents[len(parts) - 1]


def image_tag(directory: Path) -> str:
    """The image's tag: the digest of the Dockerfile and the hash lock, so a changed definition is a new image."""
    digest = hashlib.sha256()
    for name in IMAGE_FILES:
        digest.update(name.encode("utf-8") + b"\0" + (directory / name).read_bytes() + b"\0")
    return f"{IMAGE_NAME}:{digest.hexdigest()[:16]}"


def docker_prefix(arguments: Sequence[str], environment: Mapping[str, str]) -> list[str]:
    """`sudo -n docker`, keeping the credential names an online job needs, whose values docker copies by name."""
    present = [name for name in CREDENTIALS if environment.get(name)] if needs_network(arguments) else []
    return ["sudo", "-n", *([f"--preserve-env={','.join(present)}"] if present else []), "docker"]


def path_mounts(arguments: Sequence[str]) -> tuple[list[str], list[Path]]:
    """`-v` options mounting every path the job names at the same path, and the directories to create first.

    Inputs mount read-only; the store and the receipt's directory are writable, and created here as the calling user,
    since a directory docker creates for a mount belongs to root. The checkpoint cache is writable only for `--fetch`.
    """
    values = option_values(arguments, (*READ_PATHS, *WRITE_DIRECTORIES, "--cache", "--receipt"))
    mounts: dict[Path, str] = {}
    created: list[Path] = []
    for name, value in values.items():
        path = Path(value)
        if not path.is_absolute():
            raise SystemExit(f"{name} {value} is relative; a containerized job needs absolute paths.")
        if name == "--receipt":
            path = path.parent
        if name == "--store" and arguments and arguments[0] in MIRROR_COMMANDS:
            path = mirror_of(path) or path
        writable = name in WRITE_DIRECTORIES or name == "--receipt" or (name == "--cache" and "--fetch" in arguments)
        if writable:
            created.append(path)
        elif not path.exists():
            raise SystemExit(f"{name} {value} does not exist on this machine.")
        mode = "rw" if writable else "ro"
        mounts[path] = "rw" if mounts.get(path) == "rw" else mode
    options = [argument for path, mode in sorted(mounts.items()) for argument in ("-v", f"{path}:{path}:{mode}")]
    return options, created


def code_mount(environment: Mapping[str, str]) -> tuple[Path, str]:
    """The directory holding the code, and the PYTHONPATH the container gets.

    Under `ws remote run` that is the shipped run directory and its import roots, so the job runs exactly the files its
    tree hash names; elsewhere it is the workspace root with `models` for FastPLMs.
    """
    shipped = environment.get("WS_RUN_DIRECTORY")
    if shipped:
        roots = [entry for entry in environment.get("PYTHONPATH", "").split(os.pathsep) if entry.startswith(shipped)]
        return Path(shipped), os.pathsep.join(roots)
    root = workspace_root()
    return root, os.pathsep.join([str(root), str(root / "models")])


def profile_variables(arguments: Sequence[str]) -> dict[str, str]:
    """`EMBEDDING_PROFILE` for the profile the job names (an id, or a path to its file), which the job holds to the
    profile's own value; nothing for a job that names none. The image's default names `embedding_profile_v1`."""
    profile = option_values(arguments, ("--profile",)).get("--profile")
    return {} if profile is None else {"EMBEDDING_PROFILE": PurePosixPath(profile.replace("\\", "/")).stem}


def docker_command(arguments: Sequence[str], *, tag: str, device_index: int, environment: Mapping[str, str],
                   user: str) -> tuple[list[str], list[Path]]:
    """The `docker run` that executes the job in the image, and the directories to create before it starts."""
    code, pythonpath = code_mount(environment)
    mounts, created = path_mounts(arguments)
    online = needs_network(arguments)
    credentials = [argument for name in CREDENTIALS if online and environment.get(name) for argument in ("-e", name)]
    isolation = [] if online else ["--network", "none", "-e", "HF_HUB_OFFLINE=1"]
    selected = [argument for name, value in profile_variables(arguments).items() for argument in ("-e", f"{name}={value}")]
    command = [
        *docker_prefix(arguments, environment), "run", "--rm", "--gpus", f"device={device_index}", "--ipc=host",
        "--user", user, "-e", "HOME=/tmp", "-e", f"PYTHONPATH={pythonpath}", "-v", f"{code}:{code}:ro", "-w", str(code),
        *selected, *credentials, *isolation, *mounts, tag, "python", "-m", JOB_MODULE, *arguments,
    ]
    return command, created


def ensure_image(directory: Path, prefix: Sequence[str], run: Callable[[Sequence[str]], int]) -> str:
    """The image's tag, built here from ``directory`` when this machine does not hold it yet."""
    tag = image_tag(directory)
    held = subprocess.run([*prefix, "image", "inspect", tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          check=False).returncode == 0
    if not held:
        print(f"building {tag} from {directory}", file=sys.stderr)
        if run([*prefix, "build", "-t", tag, str(directory)]) != 0:
            raise SystemExit(f"Building {tag} failed; nothing ran.")
    return tag


def run_here(arguments: Sequence[str], *, device_index: int, dry_run: bool,
             run: Callable[[Sequence[str]], int]) -> int:
    """Run the job in the image on this Linux machine's GPU ``device_index``."""
    if sys.platform != "linux" or not hasattr(os, "getuid"):
        raise SystemExit("--compute local runs on a Linux GPU machine; from here use ssh:<machine> or modal.")
    code, _ = code_mount(os.environ)
    directory = code / "tree" / IMAGE_DIRECTORY if os.environ.get("WS_RUN_DIRECTORY") else code / IMAGE_DIRECTORY
    prefix = docker_prefix(arguments, os.environ)
    tag = image_tag(directory) if dry_run else ensure_image(directory, prefix, run)
    command, created = docker_command(arguments, tag=tag, device_index=device_index, environment=os.environ,
                                      user=f"{os.getuid()}:{os.getgid()}")
    print(shlex.join(command), file=sys.stderr)  # names only: no value of a credential is on this line
    if dry_run:
        return 0
    for path in created:
        path.mkdir(parents=True, exist_ok=True)
    return run(command)


def run_on_modal(arguments: Sequence[str], *, gpu: str | None, dry_run: bool,
                 run: Callable[[Sequence[str]], int] | None) -> int:
    """Run the job on Modal on one GPU class; its receipt comes back to the `--receipt` path on this machine."""
    if gpu not in MODAL_GPUS:
        raise SystemExit(f"--compute modal needs --modal-gpu, one of {', '.join(MODAL_GPUS)}; received {gpu}.")
    receipt = option_values(arguments, ("--receipt",)).get("--receipt")
    if receipt is None:
        raise SystemExit("A Modal run needs --receipt: the path on this machine its receipt comes back to.")
    if Path(receipt).exists():
        raise SystemExit(f"{receipt} exists; a command never overwrites its receipt.")
    entrypoint = ["--gpu", gpu, "--arguments", shlex.join(with_option(arguments, "--receipt", CONTAINER_RECEIPT)),
                  "--receipt", str(Path(receipt).resolve())]
    if dry_run:
        print(shlex.join(["modal", "run", MODAL_APP, *entrypoint]), file=sys.stderr)
        return 0
    root = workspace_root()
    return launch_on_modal(root / MODAL_APP, entrypoint, cwd=root, run=run)


def run_over_ssh(machine: str, arguments: Sequence[str], *, device_index: int, dry_run: bool) -> int:
    """Ship the code and the image definition to ``machine`` and start this launcher there with `--compute local`."""
    remote = ["python", "-m", "foundry.embedding.launch", "--compute", "local", "--device-index", str(device_index),
              "--", *arguments]
    return launch_over_ssh(machine, ENTITY, remote, dry_run=dry_run, include=[IMAGE_DIRECTORY])


def parser() -> argparse.ArgumentParser:
    arguments = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_compute_argument(arguments, default="modal")
    arguments.add_argument("--device-index", type=int, default=0, help="local and ssh: the GPU the container gets")
    arguments.add_argument("--modal-gpu", help=f"modal: the one GPU class, {', '.join(MODAL_GPUS)}")
    arguments.add_argument("--dry-run", action="store_true", help="print what would run and start nothing")
    arguments.add_argument("job", nargs=argparse.REMAINDER, help="-- then foundry.embedding.job's arguments")
    return arguments


def main(argv: Sequence[str] | None = None, *, run: Callable[[Sequence[str]], int] | None = None) -> int:
    options = parser().parse_args(argv)
    job = options.job[1:] if options.job[:1] == ["--"] else options.job
    if not job:
        raise SystemExit("Name the job after --, such as: -- qualify --base 300 --profile embedding_profile_v1 ...")
    runner = run or (lambda command: subprocess.run(list(command), check=False).returncode)
    return dispatch(
        choose_compute(options.compute, default="modal"),
        modal=lambda: run_on_modal(job, gpu=options.modal_gpu, dry_run=options.dry_run, run=run),
        local=lambda: run_here(job, device_index=options.device_index, dry_run=options.dry_run, run=runner),
        ssh=lambda machine: run_over_ssh(machine, job, device_index=options.device_index, dry_run=options.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
