"""`foundry.compute`: the one flag that runs a job on Modal, over SSH, or on this machine (principle 12).

    from foundry.compute import add_compute_argument, dispatch, launch_on_modal, launch_over_ssh, local_arguments

    parser = argparse.ArgumentParser()
    add_compute_argument(parser, default="modal")          # --compute modal|local|ssh:<machine>
    arguments = parser.parse_args()
    raise SystemExit(dispatch(
        choose_compute(arguments.compute, default="modal"),
        modal=lambda: launch_on_modal(APP, ["--stage", arguments.stage]),
        local=lambda: run_here(arguments),
        ssh=lambda machine: launch_over_ssh(machine, "my_project", ["python", "run.py", *local_arguments(sys.argv[1:])]),
    ))

The flag says where the job runs and nothing else, so it is never inferred from what a machine happens to
hold: not from Modal credentials, not from a GPU. The value is `modal`, `local` (the machine the command runs
on, which is the GPU machine when `ws remote run` shipped it there) or `ssh:<machine>`, a machine an
environment manifest names under `machines` or an address SSH accepts. `ssh:<machine>` runs the same command
there with `--compute local`, through `ws remote run`, so one code path serves the machine and the shipped run.

The module imports only the standard library, so a launcher that has no Modal installed, or that a workstation
run directory holds, can parse the flag. Nothing here reads or prints a credential: a launcher loads the
credentials it needs with `foundry.secrets_env` before it calls `launch_on_modal`.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar


R = TypeVar("R")
Kind = Literal["modal", "local", "ssh"]

FLAG = "--compute"
# The same variable `foundry.jobs` reads, so one setting moves every job and every entry point.
WHERE_VARIABLE = "WS_WHERE"
# The credentials a launcher forwards by name: the Hub for downloads and pushes, W&B for a run.
CREDENTIALS = ("HF_TOKEN", "WANDB_API_KEY")
# Settings a launch sets for one run, which a pilot and a W&B mode need on the machine too.
PASSED_THROUGH = ("WS_PILOT", "WANDB_MODE")
MACHINE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
SSH_PREFIX = "ssh:"
VALUES = "modal, local, or ssh:<machine>"


class ComputeUnavailable(ValueError):
    """The entry point has no way to run on the requested compute, and says why."""


@dataclass(frozen=True, slots=True)
class Compute:
    """Where a job runs: `modal`, `local`, or `ssh` on `machine`."""

    kind: Kind
    machine: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("modal", "local", "ssh"):
            raise ValueError(f"{self.kind!r} is not {VALUES}")
        if self.kind == "ssh" and (self.machine is None or MACHINE.fullmatch(self.machine) is None):
            raise ValueError(f"{SSH_PREFIX}{self.machine or ''} is not {VALUES}")
        if self.kind != "ssh" and self.machine is not None:
            raise ValueError(f"{self.kind} takes no machine")

    def __str__(self) -> str:
        return f"{SSH_PREFIX}{self.machine}" if self.kind == "ssh" else self.kind


def parse_compute(text: str) -> Compute:
    """`modal`, `local`, or `ssh:<machine>` as a `Compute`; anything else is a ValueError naming the three."""
    if text in ("modal", "local"):
        return Compute(text)
    if text.startswith(SSH_PREFIX) and MACHINE.fullmatch(text.removeprefix(SSH_PREFIX)):
        return Compute("ssh", text.removeprefix(SSH_PREFIX))
    raise ValueError(f"{text!r} is not {VALUES}")


def compute_argument(text: str) -> Compute:
    """`parse_compute` as argparse's `type`, which reports the refusal as a usage error."""
    try:
        return parse_compute(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def add_compute_argument(parser: argparse.ArgumentParser, *, default: str | None = None) -> None:
    """Add `--compute modal|local|ssh:<machine>` to `parser`; `choose_compute` applies the default.

    The parsed value is a `Compute`, or None when the flag was not given, so the launcher can tell a request
    from a default. `default` only words the help; it is the compute `choose_compute` falls back to.
    """
    fallback = f" (default: {default}, or the {WHERE_VARIABLE} setting)" if default else ""
    parser.add_argument(
        FLAG, type=compute_argument, default=None, metavar="COMPUTE",
        help=f"where the job runs: {VALUES}{fallback}",
    )


def choose_compute(requested: Compute | str | None, *, default: str) -> Compute:
    """The flag when given, else `WS_WHERE`, else the entry point's own `default`.

    The default belongs to the entry point: `modal` for a job, `local` for a tool that only makes sense
    on the machine it runs on, and never a guess from the credentials present.
    """
    if requested is not None:
        return requested if isinstance(requested, Compute) else parse_compute(requested)
    return parse_compute(os.environ.get(WHERE_VARIABLE) or default)


def take_compute(arguments: Sequence[str]) -> tuple[Compute | None, list[str]]:
    """The `--compute` value among `arguments`, and the arguments without it.

    For a launcher that cannot give its parser the flag, such as one that runs before the entry point's
    own parser, or one that forwards the rest of its arguments untouched. Both `--compute X` and
    `--compute=X` are read; a missing or invalid value is a ValueError. The last occurrence wins.
    """
    remaining: list[str] = []
    requested: Compute | None = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == FLAG:
            if index + 1 >= len(arguments):
                raise ValueError(f"{FLAG} needs a value: {VALUES}")
            requested = parse_compute(arguments[index + 1])
            index += 2
            continue
        if argument.startswith(FLAG + "="):
            requested = parse_compute(argument.removeprefix(FLAG + "="))
            index += 1
            continue
        remaining.append(argument)
        index += 1
    return requested, remaining


def local_arguments(arguments: Sequence[str]) -> list[str]:
    """`arguments` with `--compute` set to `local`, which is what a run on the machine itself receives."""
    _, remaining = take_compute(arguments)
    return [*remaining, FLAG, "local"]


def dispatch(
    compute: Compute,
    *,
    modal: Callable[[], R] | None = None,
    local: Callable[[], R] | None = None,
    ssh: Callable[[str], R] | None = None,
    unavailable: Mapping[Kind, str] | None = None,
) -> R:
    """Call the entry point's handler for `compute`; `ssh` receives the machine.

    A kind with no handler raises `ComputeUnavailable` naming the reason `unavailable` gives for it, so an
    entry point that cannot run somewhere fails loudly and says what would make it possible.
    """
    reasons = unavailable or {}
    if compute.kind == "modal" and modal is not None:
        return modal()
    if compute.kind == "local" and local is not None:
        return local()
    if compute.kind == "ssh" and ssh is not None and compute.machine is not None:
        return ssh(compute.machine)
    reason = reasons.get(compute.kind, "this entry point has no such path")
    raise ComputeUnavailable(f"{FLAG} {compute} is not available: {reason}")


def run_command(argv: Sequence[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> int:
    """Run a command with its output passed straight through, and return its exit code."""
    return subprocess.run(list(argv), check=False, cwd=cwd, env=None if env is None else dict(env)).returncode


def modal_run_command(app: Path, arguments: Sequence[str], *, detach: bool = False) -> list[str]:
    """`python -m modal run [--detach] <app> <arguments>`, with this interpreter."""
    return [sys.executable, "-m", "modal", "run", *(["--detach"] if detach else []), str(app), *arguments]


def launch_on_modal(
    app: Path,
    arguments: Sequence[str],
    *,
    detach: bool = False,
    cwd: Path | None = None,
    run: Callable[[Sequence[str]], int] | None = None,
) -> int:
    """Run a Modal app's local entrypoint from this machine and wait; returns its exit code.

    The Modal CLI prints Unicode status glyphs that a Windows console's code page cannot encode, so the
    child runs with UTF-8 output. The caller loads any credentials into the environment first.
    """
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    command = modal_run_command(app, arguments, detach=detach)
    return run(command) if run is not None else run_command(command, cwd=cwd, env=environment)


def workspace_root() -> Path:
    """The workspace root above this package, where `ws.cmd` is; FileNotFoundError in a vendored copy."""
    for directory in Path(__file__).resolve().parents:
        if (directory / "ws.cmd").is_file():
            return directory
    raise FileNotFoundError("ws.cmd is not above foundry here, so this copy cannot launch over SSH; run it from the workspace")


def forwarded_settings() -> dict[str, str]:
    """The launch settings in this environment that a run on another machine needs (`WS_PILOT`, `WANDB_MODE`)."""
    return {name: os.environ[name] for name in PASSED_THROUGH if os.environ.get(name)}


def ssh_command(
    workspace: Path,
    machine: str,
    entity: str,
    command: Sequence[str],
    *,
    dry_run: bool = False,
    launch_receipt: Path | None = None,
    include: Sequence[str] = (),
    settings: Mapping[str, str] | None = None,
) -> list[str]:
    """`ws remote run <machine> <entity> -- <command>`, through `ws.cmd`.

    A launch detaches, so the run outlives this connection and `ws remote status` follows it; a dry run
    lists what would ship and connects to nothing. Each `include` is a workspace-relative file shipped beside
    the entity, and each of `settings` reaches the run as an environment variable. Credentials are not sent
    here: `ws remote run` forwards its named defaults on stdin.
    """
    return [
        os.environ.get("COMSPEC", "cmd.exe"), "/c", str(workspace / "ws.cmd"), "remote", "run", machine, entity,
        "--dry-run" if dry_run else "--detach",
        *(["--launch-receipt", str(launch_receipt)] if launch_receipt is not None else []),
        *(argument for path in include for argument in ("--include", path)),
        *(argument for name, value in (settings or {}).items() for argument in ("--env", f"{name}={value}")),
        "--", *command,
    ]


def launch_over_ssh(
    machine: str,
    entity: str,
    command: Sequence[str],
    *,
    workspace: Path | None = None,
    dry_run: bool = False,
    include: Sequence[str] = (),
    run: Callable[[Sequence[str]], int] = run_command,
) -> int:
    """Ship `entity` to `machine` and start `command` there, detached; returns `ws remote run`'s exit code.

    `command` is the whole command line to run there, whose last compute is `local`: build it with
    `local_arguments` so the machine runs the code and does not forward it again. The launch settings in this
    environment (`WS_PILOT`, `WANDB_MODE`) travel with it.
    """
    root = workspace if workspace is not None else workspace_root()
    return run(ssh_command(root, machine, entity, command, dry_run=dry_run, include=include, settings=forwarded_settings()))


__all__ = [
    "CREDENTIALS", "FLAG", "PASSED_THROUGH", "WHERE_VARIABLE", "Compute", "ComputeUnavailable", "Kind",
    "add_compute_argument", "choose_compute", "compute_argument", "dispatch", "forwarded_settings", "launch_on_modal",
    "launch_over_ssh", "local_arguments", "modal_run_command", "parse_compute", "run_command", "ssh_command",
    "take_compute", "workspace_root",
]
