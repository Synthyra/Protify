"""Grid-run Protify over pooling configurations and plot cross-pooling comparisons.

Each --pooling_types token is one pooling configuration. Dashes concatenate:
    mean       -> --embedding_pooling_types mean
    mean-var   -> --embedding_pooling_types mean var
    max        -> --embedding_pooling_types max
    cls        -> --embedding_pooling_types cls

One docker invocation of `py -m main` per pooling configuration; main.py grids
models x datasets internally. Results TSVs and logs are collected, aggregated
into a long-form DataFrame, and rendered as bar/heatmap/rank plots.

Any flags not recognized by this script are forwarded verbatim to `py -m main`,
so you can set --probe_type, --hybrid_probe, --lr, --trainer, --token_probe, etc.

Example (run from repo root):
    py src/protify/scripts/compare_pooling.py \
        --model_names ESM2-8 ESM2-150 \
        --data_names FLUO BLAC remote_homology DPI \
        --pooling_types mean mean-var max cls \
        --num_epochs 100 \
        --probe_type transformer --hybrid_probe
"""

import argparse
import datetime
import glob
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
PROTIFY_DIR = SCRIPT_DIR.parent                  # src/protify
REPO_ROOT = PROTIFY_DIR.parent.parent            # repo root
RESULTS_DIR = PROTIFY_DIR / 'results'
LOGS_DIR = PROTIFY_DIR / 'logs'
PLOTS_DIR = REPO_ROOT / 'plots'

sys.path.insert(0, str(PROTIFY_DIR))
from visualization.utils import MODEL_NAMES, DATASET_NAMES, CLS_PREFS, REG_PREFS


def _timestamp() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')


def detect_docker() -> list[str]:
    base = ['docker']
    if platform.system() == 'Linux':
        try:
            subprocess.run(['docker', 'ps'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            base = ['sudo', 'docker']
    return base


def build_docker_prefix(image: str) -> list[str]:
    cmd = detect_docker() + ['run', '--rm', '--gpus', 'all']
    cmd += ['-v', f'{REPO_ROOT}:/workspace', '-w', '/workspace/src/protify', image, 'python', '-m', 'main']
    return cmd


def expand_pooling(spec: str) -> list[str]:
    parts = [p for p in spec.split('-') if p]
    assert parts, f'empty pooling spec: {spec!r}'
    return parts


def snapshot_tsvs() -> set:
    return set(glob.glob(str(RESULTS_DIR / '*.tsv')))


def find_log_for_tsv(tsv_path: str) -> str | None:
    stem = Path(tsv_path).stem
    candidate = LOGS_DIR / f'{stem}.txt'
    return str(candidate) if candidate.exists() else None


def run_one(
    prefix: list[str],
    model_names: list[str],
    data_names: list[str],
    expanded_pooling: list[str],
    num_epochs: int,
    passthrough: list[str],
) -> tuple[str | None, str | None, int, float]:
    before = snapshot_tsvs()
    main_args = [
        '--model_names', *model_names,
        '--data_names', *data_names,
        '--embedding_pooling_types', *expanded_pooling,
        '--probe_pooling_types', *expanded_pooling,
        '--num_epochs', str(num_epochs),
    ]
    if passthrough:
        main_args += passthrough
    full = prefix + main_args
    print('\n' + '=' * 80)
    print('RUN:', ' '.join(full))
    print('=' * 80, flush=True)
    t0 = time.time()
    proc = subprocess.run(full)
    wall = time.time() - t0
    after = snapshot_tsvs()
    new = sorted(after - before, key=os.path.getmtime)
    tsv = new[-1] if new else None
    log = find_log_for_tsv(tsv) if tsv else None
    return tsv, log, proc.returncode, wall


def load_tsv_long(tsv_path: str, pooling_label: str) -> pd.DataFrame:
    df_raw = pd.read_csv(tsv_path, sep='\t')
    models = [c for c in df_raw.columns if c != 'dataset']
    rows = []
    for _, r in df_raw.iterrows():
        dataset = r['dataset']
        for m in models:
            cell = r[m]
            if pd.isna(cell):
                continue
            try:
                metrics = json.loads(cell)
            except (json.JSONDecodeError, TypeError):
                continue
            for metric_name, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    rows.append((pooling_label, m, dataset, metric_name, float(v)))
    return pd.DataFrame(rows, columns=['pooling', 'model', 'dataset', 'metric', 'value'])


def pick_primary_metric(metrics_available: set[str]) -> tuple[str, str]:
    for key, label in CLS_PREFS + REG_PREFS:
        for m in metrics_available:
            if key in m.lower():
                return m, label
    any_metric = sorted(metrics_available)[0]
    return any_metric, any_metric


def pretty_model(m: str) -> str:
    return MODEL_NAMES.get(m, m)


def pretty_dataset(d: str) -> str:
    return DATASET_NAMES.get(d, d)


def plot_bars(df: pd.DataFrame, out_dir: Path) -> None:
    datasets = sorted(df['dataset'].unique())
    n = len(datasets)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), squeeze=False)
    for i, dataset in enumerate(datasets):
        ax = axes[i // ncols][i % ncols]
        sub = df[df['dataset'] == dataset]
        metric, metric_label = pick_primary_metric(set(sub['metric'].unique()))
        sub_m = sub[sub['metric'] == metric]
        if sub_m.empty:
            ax.set_visible(False)
            continue
        pivot_models = [pretty_model(m) for m in sub_m['model'].unique()]
        sub_m = sub_m.assign(model_disp=sub_m['model'].map(pretty_model))
        sns.barplot(data=sub_m, x='pooling', y='value', hue='model_disp', ax=ax)
        ax.set_title(f'{pretty_dataset(dataset)}  ({metric_label})')
        ax.set_xlabel('pooling')
        ax.set_ylabel(metric_label)
        ax.tick_params(axis='x', rotation=30)
        ax.legend(fontsize=7, loc='best')
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / 'bar_per_dataset.png', dpi=300)
    plt.close(fig)


def plot_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    datasets = sorted(df['dataset'].unique())
    n = len(datasets)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), squeeze=False)
    for i, dataset in enumerate(datasets):
        ax = axes[i // ncols][i % ncols]
        sub = df[df['dataset'] == dataset]
        metric, metric_label = pick_primary_metric(set(sub['metric'].unique()))
        sub_m = sub[sub['metric'] == metric]
        if sub_m.empty:
            ax.set_visible(False)
            continue
        pivot = sub_m.pivot_table(index='model', columns='pooling', values='value', aggfunc='mean')
        pivot.index = [pretty_model(m) for m in pivot.index]
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax, cbar_kws={'label': metric_label})
        ax.set_title(f'{pretty_dataset(dataset)}  ({metric_label})')
        ax.set_xlabel('pooling')
        ax.set_ylabel('model')
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / 'heatmap_model_x_pooling.png', dpi=300)
    plt.close(fig)


def plot_ranks(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    # For each (dataset, model) choose primary metric and rank pooling methods (1 = best).
    rank_rows = []
    for (dataset, model), sub in df.groupby(['dataset', 'model']):
        metric, _ = pick_primary_metric(set(sub['metric'].unique()))
        sub_m = sub[sub['metric'] == metric]
        if sub_m.empty:
            continue
        # Higher-is-better for everything we use; for loss/rmse/mse invert
        higher_is_better = not any(k in metric.lower() for k in ('loss', 'rmse', 'mse', 'hamming'))
        values = sub_m.set_index('pooling')['value']
        ranks = values.rank(ascending=not higher_is_better, method='average')
        for pooling, r in ranks.items():
            rank_rows.append((pooling, dataset, model, float(r)))
    rank_df = pd.DataFrame(rank_rows, columns=['pooling', 'dataset', 'model', 'rank'])
    mean_ranks = rank_df.groupby('pooling')['rank'].mean().sort_values()

    fig, ax = plt.subplots(figsize=(6, max(3, 0.5 * len(mean_ranks))))
    sns.barplot(x=mean_ranks.values, y=mean_ranks.index, ax=ax, orient='h', color='steelblue')
    for i, v in enumerate(mean_ranks.values):
        ax.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=9)
    ax.set_xlabel('mean rank (lower is better)')
    ax.set_ylabel('pooling')
    ax.set_title(f'Average rank across {rank_df[["dataset","model"]].drop_duplicates().shape[0]} (dataset, model) pairs')
    fig.tight_layout()
    fig.savefig(out_dir / 'rank_plot.png', dpi=300)
    plt.close(fig)
    return rank_df


def write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    # Pivot pooling x dataset, primary metric averaged across models
    rows = []
    for dataset, sub in df.groupby('dataset'):
        metric, _ = pick_primary_metric(set(sub['metric'].unique()))
        sub_m = sub[sub['metric'] == metric]
        for pooling, ssub in sub_m.groupby('pooling'):
            rows.append((pooling, dataset, metric, float(ssub['value'].mean())))
    summary = pd.DataFrame(rows, columns=['pooling', 'dataset', 'metric', 'mean_value'])
    pivot = summary.pivot_table(index='pooling', columns='dataset', values='mean_value')
    pivot.to_csv(out_dir / 'summary.tsv', sep='\t')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Unrecognized flags are forwarded verbatim to `py -m main` '
               '(e.g. --probe_type transformer --hybrid_probe --lr 1e-4 --trainer hf).',
    )
    parser.add_argument('--model_names', nargs='+', required=True)
    parser.add_argument('--data_names', nargs='+', required=True)
    parser.add_argument('--pooling_types', nargs='+', required=True, help='Pooling configs; dashes concatenate (mean-var).')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--image', type=str, default='protify-env:latest')
    parser.add_argument('--fail_fast', action='store_true')
    parser.add_argument('--skip_run', action='store_true', help='Reuse existing manifest.json; only aggregate and plot.')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--manifest', type=str, default=None, help='For --skip_run: path to an existing manifest.json to re-plot.')
    args, passthrough = parser.parse_known_args()

    out_dir = Path(args.out_dir) if args.out_dir else PLOTS_DIR / f'compare_pooling_{_timestamp()}'
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_run:
        assert args.manifest, '--skip_run requires --manifest <path/to/manifest.json>'
        manifest = json.loads(Path(args.manifest).read_text())
    else:
        if passthrough:
            print(f'[info] forwarding to py -m main: {passthrough}')
        prefix = build_docker_prefix(args.image)
        manifest = []
        for spec in args.pooling_types:
            expanded = expand_pooling(spec)
            tsv, log, rc, wall = run_one(
                prefix=prefix,
                model_names=args.model_names,
                data_names=args.data_names,
                expanded_pooling=expanded,
                num_epochs=args.num_epochs,
                passthrough=passthrough,
            )
            entry = {
                'pooling_label': spec,
                'expanded': expanded,
                'tsv': tsv,
                'log': log,
                'returncode': rc,
                'wall_s': round(wall, 1),
            }
            manifest.append(entry)
            print(f'[done] pooling={spec}  rc={rc}  wall={wall:.1f}s  tsv={tsv}')
            if rc != 0:
                print(f'[warn] non-zero return code for pooling={spec}')
                assert not args.fail_fast, f'--fail_fast: run for {spec!r} failed'
            (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    frames = []
    for entry in manifest:
        if not entry['tsv'] or not os.path.exists(entry['tsv']):
            print(f'[skip] missing tsv for pooling={entry["pooling_label"]}')
            continue
        frames.append(load_tsv_long(entry['tsv'], entry['pooling_label']))
    assert frames, 'no result TSVs found; cannot aggregate'
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_dir / 'combined_long.csv', index=False)
    print(f'combined_long.csv: {len(df)} rows')

    plot_bars(df, out_dir)
    plot_heatmaps(df, out_dir)
    plot_ranks(df, out_dir)
    write_summary(df, out_dir)

    print(f'\nOutputs written to {out_dir}')


if __name__ == '__main__':
    main()
