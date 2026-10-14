"""Trends across models: where each kind of concept becomes linearly readable, which states read it best, what
training adds over the controls and the baseline encodings, how the answers change with model size, and where
protein-protein interaction becomes readable.

A model is a set of replicate roots (one per seed) sharing a results label, read as `figures` reads one model. Depth
is put on a relative axis, 0 at the token embedding and 1 at the final norm, so models of 30, 36 and 80 blocks share
it. A randomly initialized model is read the same way. The baseline encodings (`OneHot-Protein`, `Random`) have one
state each, the token rows, and are drawn as reference levels. Best states are chosen on validation, as in `figures`,
and reported on test. Every figure is a 300 dpi PNG with no title, beside the table that holds its numbers.

Symbols: L blocks of a model; T tasks; S sites.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import figures
from .figures import BLOCK_SITES, GRID, INK, INK_SECONDARY, MUTED, PALETTE, RESIDUAL, SITE_LABELS


PAIR_TASK = "ppi"
# The concept categories the trend figures group tasks by, in drawing order.
CATEGORIES = (
    ("composition", ("prot_param",)),
    ("secondary structure", ("ssp_q3", "ssp_q8", "secondary_structure")),
    ("sorting and topology", ("localization", "peptide", "topology")),
    ("InterPro membership", ("interpro_family", "interpro_domain", "interpro_homologous_superfamily", "interpro_repeat",
                             "interpro_conserved_site", "interpro_active_site", "interpro_binding_site")),
    ("function", ("go_mf", "go_bp", "go_cc", "metal_ion_binding")),
    ("residue sites and modifications", ("functional_sites", "post_translational_modification", "phosphorylation", "lipidation")),
)
CATEGORY_OF = {task: category for category, tasks in CATEGORIES for task in tasks}
CONCEPT_ORDER = tuple(task for _, tasks in CATEGORIES for task in tasks)
# Parameters of the models we know, which order them; others keep the order they are given in.
MODEL_PARAMETERS = {"ESMC-300": 3e8, "ESMC-600": 6e8, "ESMC-6B": 6e9}
ONSET_SHARE = 0.95  # a curve's onset: the first depth within 5% of its highest score
SHARE_FLOOR = 0.5  # where the emergence maps' color scale starts, as a share of a task's highest score


@dataclass(frozen=True)
class ModelRuns:
    """One model's replicate roots, one per seed, and the results label they share."""

    name: str
    roots: tuple[Path, ...]
    label: str = "linear"

    def blocks(self) -> int | None:
        """The model's blocks from a store manifest (0 for a baseline encoding), or None when no root holds a store."""
        for root in self.roots:
            for store in ("trained_original", f"{PAIR_TASK}_trained_original"):
                manifest = Path(root) / "stores" / store / "manifest.json"
                if manifest.exists():
                    return int(json.loads(manifest.read_text(encoding="utf-8"))["dims"]["blocks"])
        return None


def parse_runs(item: str, label: str) -> ModelRuns:
    """NAME=ROOT[,ROOT...] as a model's runs."""
    name, _, roots = item.partition("=")
    if not name or not roots:
        raise SystemExit(f"Models are NAME=ROOT[,ROOT...], not {item!r}.")
    return ModelRuns(name, tuple(Path(root) for root in roots.split(",")), label)


def _with_relative_depth(table: pd.DataFrame, blocks: int) -> pd.DataFrame:
    depths = np.array([figures.depth(site, layer, blocks) for site, layer in zip(table["site"], table["layer"])], dtype=int)
    return table.assign(blocks=blocks, depth=depths, relative_depth=depths / (blocks + 1))


def model_summary(runs: ModelRuns) -> pd.DataFrame:
    """`figures.summarize` of every replicate, with the model's name, its blocks and each state's relative depth. The
    blocks come from a store manifest, or else from the deepest block a probe reads."""
    scores = figures.read_replicates(runs.roots, runs.label)
    validation = figures.read_validation([(root, runs.label) for root in runs.roots])
    summary = figures.summarize(scores, validation)
    blocks = runs.blocks()
    if blocks is None:
        inside = summary["site"].isin(BLOCK_SITES)
        blocks = int(summary.loc[inside, "layer"].max()) + 1 if inside.any() else 0
    return _with_relative_depth(summary, blocks).assign(model=runs.name)


def model_ranking(runs: ModelRuns, blocks: int) -> pd.DataFrame | None:
    """The pair task's test ROC AUC and average precision per probe, averaged over replicates; None without any."""
    tables = [pd.read_parquet(path) for root in runs.roots if (path := Path(root) / "results" / runs.label / PAIR_TASK / "ranking.parquet").exists()]
    if not tables:
        return None
    ranking = pd.concat(tables, ignore_index=True)
    ranking = ranking[ranking["split"] == "test"].groupby(["task", "family", "probe_name"])[["roc_auc", "average_precision"]].mean().reset_index()
    return _with_relative_depth(figures.with_probe_parts(ranking), blocks).assign(model=runs.name)


def _ordered_models(names: Sequence[str]) -> list[str]:
    known = [name for name in names if name in MODEL_PARAMETERS]
    return sorted(known, key=MODEL_PARAMETERS.get) + [name for name in names if name not in MODEL_PARAMETERS]


def _stream(summary: pd.DataFrame) -> pd.DataFrame:
    trained = figures.headline(summary)
    return trained[(trained["family"] == "full") & trained["site"].isin(RESIDUAL)]


# -- tables ----------------------------------------------------------------------------------------------------------


def stream_profiles(summary: pd.DataFrame) -> pd.DataFrame:
    """Per model and task, the residual stream's shape: its scores at the embedding and the final norm, its highest
    score and where it lies, its onset (the first depth within 5% of the highest), and the best stream state chosen on
    validation, with its score and depth. Depths are relative."""
    rows = []
    for (model, task), curve in _stream(summary).groupby(["model", "task"]):
        curve = curve.sort_values("depth")
        peak = curve.loc[curve["full"].idxmax()]
        onset = curve[curve["full"] >= ONSET_SHARE * peak["full"]]["relative_depth"].min() if peak["full"] > 0 else np.nan
        best = figures._choose(curve)
        rows.append({
            "model": model, "task": task, "category": CATEGORY_OF.get(task, task), "embedding": float(curve["full"].iloc[0]),
            "final_norm": float(curve["full"].iloc[-1]), "peak": float(peak["full"]), "peak_depth": float(peak["relative_depth"]),
            "onset_depth": float(onset), "best": float(best["full"]), "best_depth": float(best["relative_depth"]),
            "best_replicate_std": float(best["full_std"]),
        })
    return pd.DataFrame(rows)


def category_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """Per model and concept category, over its tasks: the median onset, highest-score and best-state depths, and the
    median share of the highest stream score already at the embedding and still at the final norm."""
    shares = profiles.assign(embedding_share=profiles["embedding"] / profiles["peak"], final_share=profiles["final_norm"] / profiles["peak"])
    table = shares.groupby(["model", "category"]).agg(
        tasks=("task", "count"), onset_depth=("onset_depth", "median"), peak_depth=("peak_depth", "median"),
        best_depth=("best_depth", "median"), embedding_share=("embedding_share", "median"), final_share=("final_share", "median"),
    ).reset_index()
    models = {model: index for index, model in enumerate(_ordered_models(list(dict.fromkeys(table["model"]))))}
    categories = {category: index for index, (category, _) in enumerate(CATEGORIES)}
    keys = table.assign(model_order=table["model"].map(models), category_order=table["category"].map(categories))
    return keys.sort_values(["model_order", "category_order"]).drop(columns=["model_order", "category_order"]).reset_index(drop=True)


def site_rankings(summaries: dict[str, pd.DataFrame], tasks: Sequence[str] = CONCEPT_ORDER) -> pd.DataFrame:
    """Per model and site: each site's best state over depth minus the task's best residual-stream state (both chosen
    on validation), summarized over tasks by median and quartiles."""
    rows = []
    for model, summary in summaries.items():
        gaps = figures.site_gaps(summary, tasks)
        for site in gaps.columns:
            values = gaps[site].dropna()
            if values.empty:
                continue
            rows.append({"model": model, "site": site, "median": float(values.median()), "q25": float(values.quantile(0.25)),
                         "q75": float(values.quantile(0.75)), "above": int((values > 0).sum()), "tasks": len(values)})
    return pd.DataFrame(rows)


def best_table(summaries: dict[str, pd.DataFrame], kinds: dict[str, str]) -> pd.DataFrame:
    """Per model (of each kind: trained, random-init or baseline) and task: the best residual-stream and any-state
    probes chosen on validation, their relative depths, and the trained models' controls at the stream best."""
    tables = []
    for model, summary in summaries.items():
        blocks = int(summary["blocks"].iloc[0])
        table = figures.best_states(summary, blocks)
        for name in ("residual", "any"):
            if f"{name}_depth" in table:
                table[f"{name}_relative_depth"] = table[f"{name}_depth"] / (blocks + 1)
        tables.append(table.assign(model=model, kind=kinds[model], category=table["task"].map(CATEGORY_OF)))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


# -- figures ---------------------------------------------------------------------------------------------------------


def _model_colors(models: Sequence[str]) -> dict[str, str]:
    return {model: PALETTE[index % len(PALETTE)] for index, model in enumerate(models)}


def _task_rows(tasks: set[str]) -> list[str]:
    return [task for task in CONCEPT_ORDER if task in tasks]


def _category_bands(axis, rows: list[str], positions: np.ndarray, right: bool = False) -> None:
    """Alternate shading by concept category, and each category named once beside its rows."""
    start = 0
    for index in range(1, len(rows) + 1):
        if index == len(rows) or CATEGORY_OF.get(rows[index]) != CATEGORY_OF.get(rows[start]):
            top, bottom = positions[start], positions[index - 1]
            if len({CATEGORY_OF.get(task) for task in rows[:start]}) % 2 == 0:
                axis.axhspan(min(top, bottom) - 0.5, max(top, bottom) + 0.5, color="#f6f5f2", lw=0, zorder=0)
            if right:
                axis.text(1.01, (top + bottom) / 2, CATEGORY_OF.get(rows[start], ""), transform=axis.get_yaxis_transform(),
                          fontsize=5.2, color=INK_SECONDARY, va="center", ha="left")
            start = index


def emergence_maps(summaries: dict[str, pd.DataFrame], profiles: pd.DataFrame, path: Path) -> Path:
    """One panel per model: each task's residual stream along relative depth, as a share of its highest score (the
    scale starts at one half, below which most tasks fall only at the embedding), tasks grouped by concept category; a
    bar marks the onset (the first depth within 5% of the highest) and a dot the best stream state chosen on validation."""
    plt = figures._pyplot()
    models = _ordered_models(list(summaries))
    rows = _task_rows(set().union(*(set(_stream(summary)["task"]) for summary in summaries.values())))
    figure, axes = plt.subplots(1, len(models), figsize=(1.9 * len(models) + 1.7, 0.17 * len(rows) + 1.1), squeeze=False, sharey=True)
    colormap = figures._colormap(("#ffffff",) + figures.BLUE_RAMP, "emergence")
    image = None
    for index, (axis, model) in enumerate(zip(axes[0], models)):
        stream = _stream(summaries[model])
        blocks = int(summaries[model]["blocks"].iloc[0])
        edges = (np.arange(blocks + 3) - 0.5) / (blocks + 1)  # one cell per depth 0..L + 1
        matrix = np.full((len(rows), blocks + 2), np.nan)
        for row, task in enumerate(rows):
            curve = stream[stream["task"] == task]
            peak = curve["full"].max() if not curve.empty else np.nan
            if not np.isfinite(peak) or peak <= 0:
                continue
            matrix[row, curve["depth"].to_numpy()] = np.clip(curve["full"].to_numpy() / peak, 0.0, 1.0)
        image = axis.pcolormesh(edges, np.arange(len(rows) + 1) - 0.5, matrix, cmap=colormap, vmin=SHARE_FLOOR, vmax=1.0, shading="flat")
        marks = profiles[profiles["model"] == model].set_index("task")
        for row, task in enumerate(rows):
            if task in marks.index:
                axis.plot([marks.at[task, "onset_depth"]] * 2, [row - 0.42, row + 0.42], color=INK, lw=0.9)
                axis.scatter(marks.at[task, "best_depth"], row, s=7, facecolors="white", edgecolors=INK, linewidths=0.6, zorder=3)
        axis.set_ylim(len(rows) - 0.5, -0.5)
        axis.set_xlim(edges[0], edges[-1])
        axis.set_xticks([0.0, 0.5, 1.0])
        axis.set_xlabel("relative depth")
        figures._tag(axis, model)
        figures._bare(axis)
        if index == 0:
            axis.set_yticks(range(len(rows)))
            axis.set_yticklabels([figures.SHORT_LABELS.get(task, task) for task in rows])
    for row in range(1, len(rows)):
        if CATEGORY_OF.get(rows[row]) != CATEGORY_OF.get(rows[row - 1]):
            for axis in axes[0]:
                axis.axhline(row - 0.5, color="white", lw=1.6)
    last = axes[0][-1]
    for row, task in enumerate(rows):
        if row == 0 or CATEGORY_OF.get(task) != CATEGORY_OF.get(rows[row - 1]):
            span = [other for other in range(row, len(rows)) if CATEGORY_OF.get(rows[other]) == CATEGORY_OF.get(task)]
            last.text(1.03, (span[0] + span[-1]) / 2, CATEGORY_OF.get(task, ""), transform=last.get_yaxis_transform(), fontsize=5.2,
                      color=INK_SECONDARY, va="center", ha="left")
    reserve = 0.3 / figure.get_figheight()  # inches kept under the panels for the colorbar
    figure.tight_layout(w_pad=0.6, rect=(0, reserve, 1, 1))
    left = axes[0][0].get_position().x0
    bar = figure.colorbar(image, cax=figure.add_axes((left, 0.3 * reserve, min(0.35, 0.9 - left), 0.07 / figure.get_figheight())),
                          orientation="horizontal", extend="min")
    bar.set_label("score / task's highest residual-stream score", fontsize=5.5)
    bar.ax.tick_params(labelsize=5)
    return figures._save(figure, path)


def state_rankings(rankings: pd.DataFrame, path: Path) -> Path:
    """Each site's best state over depth minus the best residual-stream state, as a median over tasks with quartiles,
    one mark per model; sites ordered by the first model's median."""
    plt = figures._pyplot()
    models = _ordered_models(list(dict.fromkeys(rankings["model"])))
    colors = _model_colors(models)
    first = rankings[rankings["model"] == models[0]].set_index("site")["median"].sort_values(ascending=False)
    sites = list(first.index) + [site for site in dict.fromkeys(rankings["site"]) if site not in first.index]
    figure, axis = plt.subplots(figsize=(4.4, 0.24 * len(sites) + 0.9))
    positions = np.arange(len(sites))[::-1]
    offsets = np.linspace(-0.22, 0.22, len(models)) if len(models) > 1 else [0.0]
    for position in positions[::2]:
        axis.axhspan(position - 0.5, position + 0.5, color="#f6f5f2", lw=0, zorder=0)
    axis.axvline(0.0, color=INK_SECONDARY, lw=0.7, zorder=1)
    for offset, model in zip(offsets, models):
        table = rankings[rankings["model"] == model].set_index("site")
        kept = [(position, site) for position, site in zip(positions, sites) if site in table.index]
        y = np.array([position for position, _ in kept]) + offset
        rows = table.loc[[site for _, site in kept]]
        axis.hlines(y, rows["q25"], rows["q75"], color=colors[model], lw=0.9, zorder=2)
        axis.scatter(rows["median"], y, s=14, color=colors[model], zorder=3, label=model)
    axis.set_yticks(positions)
    axis.set_yticklabels([SITE_LABELS.get(site, site) for site in sites])
    axis.set_ylim(-0.6, len(sites) - 0.4)
    axis.set_xlabel("best state at the site minus best residual-stream state (median over tasks)")
    axis.grid(axis="x", color=GRID, lw=0.5)
    if len(models) > 1:
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(models), frameon=False, fontsize=5.5)
    figure.tight_layout()
    return figures._save(figure, path)


def controls_summary(best: pd.DataFrame, path: Path) -> Path:
    """Per task, the best residual-stream score of every trained model, of the randomly initialized model and of each
    baseline encoding, and the first trained model's probes at that state on scrambled sequences and on an untrained
    model's states; tasks grouped by concept category."""
    plt = figures._pyplot()
    trained = _ordered_models([model for model in dict.fromkeys(best["model"]) if (best[best["model"] == model]["kind"] == "trained").all()])
    others = [model for model in dict.fromkeys(best["model"]) if model not in trained]
    rows = _task_rows(set(best["task"]))
    colors = _model_colors(trained)
    figure, axis = plt.subplots(figsize=(4.8, 0.2 * len(rows) + 1.1))
    positions = np.arange(len(rows))
    _category_bands(axis, rows, positions, right=True)
    axis.axvline(0.0, color=INK_SECONDARY, lw=0.6, zorder=1)

    def scores(model: str, column: str = "residual_score") -> np.ndarray:
        table = best[best["model"] == model].set_index("task")
        return np.array([table.at[task, column] if task in table.index and column in table else np.nan for task in rows], dtype=float)

    styles = {"random-init": {"marker": "s", "s": 13, "color": MUTED}, "baseline": {"marker": "x", "s": 14, "linewidths": 0.9}}
    baseline_colors = iter((INK, INK_SECONDARY))
    for model in others:
        kind = best[best["model"] == model]["kind"].iloc[0]
        style = dict(styles.get(kind, styles["baseline"]))
        if kind == "baseline":
            style["color"] = next(baseline_colors, INK)
            style["marker"] = "x" if style["color"] == INK else "+"
        axis.scatter(scores(model), positions, zorder=3, label=model, **style)
    if trained:
        reference = trained[0]
        for column, legend, marker in (("control_trained_trained_scrambled", "scrambled sequences", "o"),
                                       ("control_untrained_trained_original", "untrained model's states", "v")):
            values = scores(reference, column)
            if np.isfinite(values).any():
                axis.scatter(values, positions, s=12, marker=marker, facecolors="none", edgecolors=PALETTE[3] if marker == "o" else PALETTE[4],
                             linewidths=0.8, zorder=3, label=f"{reference} probes, {legend}")
    for model in trained:
        axis.scatter(scores(model), positions, s=16, color=colors[model], zorder=4, label=model)
    axis.set_yticks(positions)
    axis.set_yticklabels([figures.SHORT_LABELS.get(task, task) for task in rows])
    axis.set_ylim(len(rows) - 0.5, -0.5)
    axis.set_xlabel("best residual-stream score (MCC; cosine for ProtParam)")
    axis.grid(axis="x", color=GRID, lw=0.5)
    axis.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False, fontsize=5.3)
    figure.tight_layout()
    return figures._save(figure, path)


def model_size(best: pd.DataFrame, path: Path) -> Path:
    """A: each task's best state over all sites, chosen on validation, for every trained model, joined across models.
    B: the relative depth of each model's best residual-stream state."""
    plt = figures._pyplot()
    trained = best[best["kind"] == "trained"]
    models = _ordered_models(list(dict.fromkeys(trained["model"])))
    colors = _model_colors(models)
    rows = _task_rows(set(trained["task"]))
    positions = np.arange(len(rows))
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.0, 0.2 * len(rows) + 1.1), sharey=True, gridspec_kw={"width_ratios": [1.4, 1.0]})
    for axis in (left, right):
        _category_bands(axis, rows, positions, right=axis is right)
    for axis, column in ((left, "any_score"), (right, "residual_relative_depth")):
        values = {model: trained[trained["model"] == model].set_index("task").reindex(rows)[column].to_numpy(dtype=float) for model in models}
        stacked = np.vstack([values[model] for model in models])
        axis.hlines(positions, np.nanmin(stacked, axis=0), np.nanmax(stacked, axis=0), color=MUTED, lw=0.8, zorder=2)
        for model in models:
            axis.scatter(values[model], positions, s=15, color=colors[model], zorder=3, label=model)
        axis.grid(axis="x", color=GRID, lw=0.5)
    left.set_yticks(positions)
    left.set_yticklabels([figures.SHORT_LABELS.get(task, task) for task in rows])
    left.set_ylim(len(rows) - 0.5, -0.5)
    left.set_xlabel("best score over all states")
    right.set_xlabel("relative depth of the best residual-stream state")
    right.set_xlim(-0.03, 1.03)
    figures._letter(left, "A", -0.02)
    figures._letter(right, "B", -0.02)
    left.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(models), frameon=False, fontsize=5.5)
    figure.tight_layout(w_pad=0.6)
    return figures._save(figure, path)


def pair_figure(summaries: dict[str, pd.DataFrame], kinds: dict[str, str], rankings: dict[str, pd.DataFrame], gaps: pd.DataFrame, path: Path) -> Path:
    """A: the pair task's MCC along the relative depth of the residual stream, for every trained model, the randomly
    initialized model and the baseline encodings (levels), with the first trained model's probes on scrambled partners
    and on an untrained model's states. B: ROC AUC along the same axis. C: each site's best state minus the best
    residual-stream state, per model."""
    plt = figures._pyplot()
    trained = _ordered_models([model for model in summaries if kinds[model] == "trained"])
    colors = _model_colors(trained)
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1.0, 1.0, 1.1]})
    for axis, letter in zip(axes, "ABC"):
        figures._letter(axis, letter)

    def curve(summary: pd.DataFrame, plm: str = "trained", control: str = "original") -> pd.DataFrame:
        rows = summary[(summary["task"] == PAIR_TASK) & (summary["family"] == "full") & summary["site"].isin(RESIDUAL)
                       & (summary["plm"] == plm) & (summary["probe"] == "trained") & (summary["control"] == control)]
        return rows.sort_values("depth")

    def level(axis, value: float, model: str) -> None:
        """A baseline encoding's one state, as a named level."""
        axis.axhline(value, color=INK_SECONDARY, lw=0.8, ls=":")
        axis.text(1.0, value, f" {model}", fontsize=5, color=INK_SECONDARY, va="center", ha="left", transform=axis.get_yaxis_transform())

    for model in trained + [model for model in summaries if model not in trained]:
        line = curve(summaries[model])
        if line.empty:
            continue
        ranking = rankings.get(model)
        auc = pd.DataFrame()
        if ranking is not None:
            auc = ranking[(ranking["family"] == "full") & ranking["site"].isin(RESIDUAL)].sort_values("relative_depth")
        if kinds[model] == "trained":
            axes[0].plot(line["relative_depth"], line["full"], color=colors[model], lw=1.3, label=model)
            axes[1].plot(auc.get("relative_depth", []), auc.get("roc_auc", []), color=colors[model], lw=1.3)
        elif kinds[model] == "random-init":
            axes[0].plot(line["relative_depth"], line["full"], color=MUTED, lw=1.1, ls="--", label=model)
            axes[1].plot(auc.get("relative_depth", []), auc.get("roc_auc", []), color=MUTED, lw=1.1, ls="--")
        else:
            level(axes[0], float(line["full"].iloc[0]), model)
            if not auc.empty:
                level(axes[1], float(auc["roc_auc"].iloc[0]), model)
    if trained:
        reference = summaries[trained[0]]
        for plm, control, legend, color in (("trained", "scrambled", "scrambled partners", PALETTE[3]), ("untrained", "original", "untrained model's states", PALETTE[4])):
            line = curve(reference, plm, control)
            if not line.empty:
                axes[0].plot(line["relative_depth"], line["full"], color=color, lw=0.9, label=f"{trained[0]} probes, {legend}")
    axes[0].set_ylabel("MCC")
    axes[1].set_ylabel("ROC AUC")
    for axis in axes[:2]:
        axis.set_xlabel("relative depth")
        axis.set_xlim(0, 1)
        axis.grid(axis="y", color=GRID, lw=0.5)
    axes[1].axhline(0.5, color=MUTED, lw=0.6)
    if not gaps.empty:
        models = _ordered_models([model for model in dict.fromkeys(gaps["model"]) if kinds.get(model) == "trained"])
        first = gaps[gaps["model"] == models[0]].set_index("site")["median"].sort_values(ascending=False) if models else pd.Series(dtype=float)
        sites = list(first.index)
        positions = np.arange(len(sites))[::-1]
        offsets = np.linspace(-0.2, 0.2, len(models)) if len(models) > 1 else [0.0]
        axes[2].axvline(0.0, color=INK_SECONDARY, lw=0.6)
        for offset, model in zip(offsets, models):
            table = gaps[gaps["model"] == model].set_index("site").reindex(sites)
            axes[2].scatter(table["median"], positions + offset, s=10, color=colors[model], zorder=3)
        axes[2].set_yticks(positions)
        axes[2].set_yticklabels([SITE_LABELS.get(site, site) for site in sites], fontsize=5.2)
        axes[2].set_xlabel("best at the site minus best residual stream")
        axes[2].grid(axis="x", color=GRID, lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    figures._legend_top(figure, handles, labels, min(len(labels), 4) or 1, 0.35)
    return figures._save(figure, path)


def trends(
    models: Sequence[ModelRuns],
    out: Path,
    random: Sequence[ModelRuns] = (),
    baselines: Sequence[ModelRuns] = (),
) -> list[Path]:
    """Every trend table and figure the runs support, under `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    kinds = {runs.name: "trained" for runs in models} | {runs.name: "random-init" for runs in random} | {runs.name: "baseline" for runs in baselines}
    every = list(models) + list(random) + list(baselines)
    summaries = {runs.name: model_summary(runs) for runs in every}
    rankings = {runs.name: ranking for runs in every if (ranking := model_ranking(runs, int(summaries[runs.name]["blocks"].iloc[0]))) is not None}
    concepts = {name: summary[summary["task"] != PAIR_TASK] for name, summary in summaries.items()}
    trained = {name: summary for name, summary in concepts.items() if kinds[name] == "trained"}
    written = []
    profiles = stream_profiles(pd.concat([concepts[name] for name in trained], ignore_index=True)) if trained else pd.DataFrame()
    if not profiles.empty:
        written.append(figures.write_summary(profiles, out / "stream_profiles.csv"))
        written.append(figures.write_summary(category_profiles(profiles), out / "category_profiles.csv"))
        written.append(emergence_maps(trained, profiles, out / "emergence.png"))
    blocked = {name: summary for name, summary in trained.items() if int(summary["blocks"].iloc[0]) > 0}
    if blocked:
        rankings_table = site_rankings(blocked)
        written.append(figures.write_summary(rankings_table, out / "site_rankings.csv"))
        written.append(state_rankings(rankings_table, out / "state_rankings.png"))
    best = best_table({name: summary for name, summary in concepts.items() if not summary.empty}, kinds)
    if not best.empty:
        written.append(figures.write_summary(best, out / "best_states_by_model.csv"))
        written.append(controls_summary(best, out / "controls_summary.png"))
        if best[best["kind"] == "trained"]["model"].nunique() > 1:
            written.append(model_size(best, out / "model_size.png"))
    pairs = {name: summary[summary["task"] == PAIR_TASK] for name, summary in summaries.items() if (summary["task"] == PAIR_TASK).any()}
    if pairs:
        pair_gaps = site_rankings({name: summary for name, summary in pairs.items() if int(summary["blocks"].iloc[0]) > 0}, (PAIR_TASK,))
        pair_best = best_table(pairs, kinds)
        ranked = pd.concat(list(rankings.values()), ignore_index=True) if rankings else pd.DataFrame()
        written.append(figures.write_summary(pair_best, out / "pair_best_states.csv"))
        if not ranked.empty:
            written.append(figures.write_summary(ranked, out / "pair_ranking.csv"))
        written.append(pair_figure(pairs, kinds, rankings, pair_gaps, out / "ppi.png"))
    return written
