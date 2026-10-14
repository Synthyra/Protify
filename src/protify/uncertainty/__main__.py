"""`python -m protify.uncertainty`: split a cell's rows, run a cell, score its identity to training, compare cells, package one, or score calibration neighbors.

    python -m protify.uncertainty split cell.yaml
    python -m protify.uncertainty run cell.yaml
    python -m protify.uncertainty identity cell.yaml
    python -m protify.uncertainty compare --runs <runs dir> --pattern "*/*/final/report.json" --output <analysis dir>
    python -m protify.uncertainty package <cell output_dir> --method ensemble_temperature --output <package dir>
    python -m protify.uncertainty neighbors --runs <fit dir> --pattern "*/*/final/report.json" --output <analysis dir>
    python -m protify.uncertainty crossfit <cell output_dir> --plan <plan.json> --output <crossfit dir>
    python -m protify.uncertainty crossfit-identity <cell output_dir> --plan <plan.json> --output <crossfit dir>
"""

from __future__ import annotations

import argparse

from collections.abc import Sequence

from .cell import load_cell
from .compare import DEFAULT_PATTERN, DEFAULT_REPEATS, DEFAULT_REPLICATES, compare_runs
from .verdicts import DecisionRules
from .workflow import run_uncertainty_cell, score_identity, split_cell


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m protify.uncertainty", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    split = commands.add_parser("split", help="Split the cell's rows again by identity strata into uncertainty.split_manifest.")
    split.add_argument("cell", help="Path to the cell YAML; its uncertainty.split_manifest receives the manifest.")
    split.add_argument("--threads", type=int, default=None, help="MMseqs2 threads; MMseqsConfig's default otherwise.")

    run = commands.add_parser("run", help="Train a cell's seeds, resuming past finished ones, and write report.json.")
    run.add_argument("cell", help="Path to the cell YAML.")

    identity = commands.add_parser("identity", help="Write each test and validation row's highest identity to training.")
    identity.add_argument("cell", help="Path to the cell YAML; its uncertainty.identity_path receives the result.")

    rules = DecisionRules()
    compare = commands.add_parser("compare", help="Answer the ablation's two questions across finished cells, with every table.")
    compare.add_argument("--runs", required=True, help="Directory the pattern is matched under.")
    compare.add_argument("--output", required=True, help="Directory the answers and tables are written to.")
    compare.add_argument(
        "--pattern", default=DEFAULT_PATTERN,
        help="Glob under --runs matching one report.json per finished cell, such as '*/*/final/report.json'.",
    )
    compare.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES, help="Bootstrap replicates, for question 1.")
    compare.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Scheme T repeats, for question 2.")
    compare.add_argument("--seed", type=int, default=0, help="Seed of every resampling.")
    compare.add_argument(
        "--workers", type=int, default=None,
        help="Worker processes; by default the CPUs available, at most 8, and 0 analyzes in this process.",
    )
    compare.add_argument("--margin", type=float, default=rules.margin, help="Question 1's verdict and recommendation margin.")
    compare.add_argument("--cost-order", nargs="+", default=list(rules.cost_order), help="The real arms, cheapest first.")

    package = commands.add_parser("package", help="Write a finished cell's heads, calibration, and conformal quantiles as one package.")
    package.add_argument("cell_dir", help="The finished cell's output_dir; the cell must have run with uncertainty.save_heads.")
    package.add_argument("--method", required=True, help="The method to ship, such as temperature or sngp_ensemble_temperature.")
    package.add_argument("--output", required=True, help="The package directory to create; it must not exist yet.")
    package.add_argument("--levels", nargs="+", type=float, default=None, help="Conformal coverage levels; by default the ones report.json scored.")
    package.add_argument("--revisions", default=None, help="A JSON file mapping each backbone repository to the revision its embeddings came from.")
    package.add_argument("--crossfit", default=None, help="The cell's crossfit output; the package then ships the correctness model fitted on it.")

    neighbors = commands.add_parser("neighbors", help="Score calibration-set neighbor estimates of correctness across finished cells.")
    neighbors.add_argument("--runs", required=True, help="Directory the pattern is matched under.")
    neighbors.add_argument("--output", required=True, help="Directory the tables, rules, and figures are written to.")
    neighbors.add_argument("--pattern", default="*/*/final/report.json", help="Glob under --runs matching one report.json per cell.")
    neighbors.add_argument("--replicates", type=int, default=2000, help="Bootstrap replicates over test rows.")
    neighbors.add_argument("--seed", type=int, default=0, help="Seed of the bootstrap.")

    crossfit = commands.add_parser("crossfit", help="Cross-fit a finished cell's head by protein cluster, and record each head's signals.")
    crossfit.add_argument("cell_dir", help="The finished cell's output_dir; the cell must have run with uncertainty.save_heads.")
    crossfit.add_argument("--plan", required=True, help="The cross-fit plan: each training protein's fold at each identity level.")
    crossfit.add_argument("--output", required=True, help="Directory each fold's and the served head's arrays go to; a rerun resumes.")
    crossfit.add_argument("--correctness-only", action="store_true", help="Record only the correctness model's columns, not every training row's neighbors.")
    crossfit.add_argument("--calibration-only", action="store_true", help="Read a declared validation-only classification fit without opening test; requires --correctness-only and five folds at 0.3.")

    identity_folds = commands.add_parser("crossfit-identity", help="Each out-of-fold row's identity to its fold's training proteins.")
    identity_folds.add_argument("cell_dir", help="The finished cell's output_dir.")
    identity_folds.add_argument("--plan", required=True, help="The cross-fit plan.")
    identity_folds.add_argument("--output", required=True, help="Directory identity.npz and the search go to.")
    identity_folds.add_argument("--threads", type=int, default=None, help="MMseqs2 threads; the search default otherwise.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    if arguments.command == "split":
        split_cell(load_cell(arguments.cell), threads=arguments.threads)
    elif arguments.command == "run":
        run_uncertainty_cell(load_cell(arguments.cell))
    elif arguments.command == "identity":
        score_identity(load_cell(arguments.cell))
    elif arguments.command == "package":
        # Imported here, so run, identity, and compare never load the packaging code.
        from .package import build_package

        build_package(
            arguments.cell_dir, arguments.method, arguments.output,
            levels=arguments.levels, revisions_path=arguments.revisions, crossfit_dir=arguments.crossfit,
        )
    elif arguments.command in ("crossfit", "crossfit-identity"):
        from .crossfit import crossfit_cell, crossfit_identity

        if arguments.command == "crossfit":
            crossfit_cell(arguments.cell_dir, arguments.plan, arguments.output,
                          m8_signals=not arguments.correctness_only, calibration_only=arguments.calibration_only)
        else:
            crossfit_identity(arguments.cell_dir, arguments.plan, arguments.output, threads=arguments.threads)
    elif arguments.command == "neighbors":
        from .neighbor_evaluation import evaluate_neighbors

        evaluate_neighbors(arguments.runs, arguments.output, pattern=arguments.pattern, replicates=arguments.replicates, seed=arguments.seed)
    else:
        rules = DecisionRules(margin=arguments.margin, cost_order=tuple(arguments.cost_order))
        compare_runs(
            arguments.runs, arguments.output, arguments.replicates, arguments.seed,
            pattern=arguments.pattern, repeats=arguments.repeats, workers=arguments.workers, rules=rules,
        )


if __name__ == "__main__":
    main()
