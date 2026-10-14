---
id: foundry.compute
kind: library_module
status: active
aliases: [foundry.compute]
---
# foundry.compute

The one flag that runs a job on Modal, over SSH, or on this machine (principle 12), chosen by
`--compute modal|local|ssh:<machine>` ([the decision](../../docs/decisions/foundry/compute/2026-10-05_one_compute_flag.md)).
The flag says where the job runs and nothing else: it is never inferred from the credentials or the GPU a machine
holds. Standard library only, so a launcher without Modal installed can parse it.

```python
parser = argparse.ArgumentParser()
add_compute_argument(parser, default="modal")        # --compute modal|local|ssh:<machine>
arguments = parser.parse_args()
raise SystemExit(dispatch(
    choose_compute(arguments.compute, default="modal"),
    modal=lambda: launch_on_modal(APP, ["--stage", arguments.stage]),
    local=lambda: run_here(arguments),
    ssh=lambda machine: launch_over_ssh(machine, "my_project", ["python", "run.py", *local_arguments(sys.argv[1:])]),
    unavailable={"local": "the stages read Modal volumes"},   # what a missing handler says
))
```

| Value | Runs |
|---|---|
| `modal` | The Modal app, started from this machine (`python -m modal run`) |
| `local` | On the machine the command runs on: the GPU machine when `ws remote run` shipped it there |
| `ssh:<machine>` | On `<machine>`, an environment manifest's `machines` name or an address SSH accepts: `ws remote run` ships the entity and what it imports and runs the same command with `--compute local` |

The default is the entry point's own: `modal` for a job, and `local` only for a tool with no other meaning. The
flag wins over `WS_WHERE` (the variable [`foundry.jobs`](../jobs/README.md) reads), which wins over the default.

| Name | Does |
|---|---|
| `Compute(kind, machine=None)`, `parse_compute(text)`, `compute_argument(text)` | A validated value, its parser (a ValueError that names `modal, local, or ssh:<machine>`), and the parser as argparse's `type` |
| `add_compute_argument(parser, *, default=None)` | Adds `--compute`; the parsed value is a `Compute`, or None when the flag was absent. `default` only words the help |
| `choose_compute(requested, *, default)` | The flag, else `WS_WHERE`, else `default` |
| `take_compute(arguments)` | The `--compute` value among `arguments` and the arguments without it, for a launcher that runs before its entry point's parser or forwards the rest untouched |
| `local_arguments(arguments)` | `arguments` with `--compute local`, what a machine receives |
| `dispatch(compute, *, modal, local, ssh, unavailable)` | Calls the handler for `compute` (`ssh` gets the machine). A kind with no handler raises `ComputeUnavailable` (a ValueError) with the reason `unavailable` gives, so an entry point that cannot run somewhere says what would make it possible, never falls back |
| `modal_run_command(app, arguments, *, detach=False)`, `launch_on_modal(app, arguments, *, detach, cwd, run)` | `python -m modal run [--detach] <app> ...` with this interpreter, run with UTF-8 output (the Modal CLI prints glyphs a Windows code page cannot encode); returns the exit code |
| `ssh_command(workspace, machine, entity, command, *, dry_run, launch_receipt, include, settings)`, `launch_over_ssh(machine, entity, command, *, workspace, dry_run, include, run)` | The `ws remote run <machine> <entity> --detach -- <command>` command through `ws.cmd`, and running it. `launch_over_ssh` forwards `WS_PILOT` and `WANDB_MODE` when set; credentials are never on the command line, since `ws remote run` sends its named defaults on stdin |
| `workspace_root()` | The directory above foundry that holds `ws.cmd`; a vendored copy has none and says so |
| `run_command(argv, *, cwd, env)`, `forwarded_settings()`, `CREDENTIALS`, `PASSED_THROUGH`, `WHERE_VARIABLE`, `FLAG` | The subprocess runner with output passed through, the launch settings that travel, the two credential names launchers forward, and the constants |

Nothing here reads or prints a credential. A launcher loads the ones it needs with
[`foundry.secrets_env`](../secrets_env/README.md) before `launch_on_modal`.

## Who uses it

`production_suite` (`train --compute`), `serving_api`'s `modal_dispatch` (`--compute`, with its credential-based
default kept as that one entry point's own), `folding_service`'s `fold.esmfold2_cli` and the workstation runner,
`base_model_distillation`'s `scripts/run_local.py` and `run_modal.py`, and `sae_xgboost_ppi`'s `modal_gp.py`.
FastPLMs' runtime source vendors no foundry (it is copied into Hugging Face repositories), so its Modal apps stay as
they are; its GitHub projection vendors only `foundry.artifact_archive` and `foundry.modal_image`. A projection that imports this module lists `foundry.compute` under `github.includes`. A run shipped to a
Modal image or a workstation directory has to carry `foundry/compute/__init__.py` if the shipped code imports it;
none of the shipped runners do.

Tests: `tests/tier1_unit/test_compute.py`, with no network and no Modal.
