"""Figures and summary tables of a tap-probing run.

Every figure reads the long tables the commands write (`results/<label>/<task>/scores.parquet` and
`elementwise.parquet`, `sweeps/<label>/<task>/sweep.parquet`, `interventions/<label>/tests.parquet`) and is saved as a
300 dpi PNG with no title, since the caption belongs to the document that shows the figure. Multi-panel figures are
lettered in reading order; small multiples name their task inside the axes.

A probe name is `site:layer` or `site:layer:h<head>`. Depth puts the residual stream on one axis: the embedding at 0,
block k's output at k + 1, and the final norm one past the last block. Scores are MCC, or cosine similarity for
ProtParam; a curve is the mean over the test halves with one standard deviation shaded, and a single value is the
full test split.
"""

from __future__ import annotations

import textwrap
import numpy as np
import pandas as pd

from collections.abc import Sequence
from pathlib import Path


# The reference categorical palette, in its fixed order, and the ink and ramps that go with it.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK_SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29d", "#e4e3df"
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
DIVERGING = ("#184f95", "#6da7ec", "#f0efec", "#f19a96", "#c42f2e")

TASK_LABELS = {
    "prot_param": "ProtParam", "peptide": "Peptide", "interpro_conserved_site": "Conserved site",
    "localization": "Localization", "interpro_repeat": "Repeat", "ssp_q3": "SSP3", "ssp_q8": "SSP8",
    "secondary_structure": "Secondary structure", "interpro_domain": "Domain", "interpro_family": "Family",
    "interpro_homologous_superfamily": "Homologous superfamily", "functional_sites": "Functional sites",
    "interpro_binding_site": "Binding site", "metal_ion_binding": "Metal ion binding",
    "interpro_active_site": "Active site", "topology": "Topology", "post_translational_modification": "PTM",
    "phosphorylation": "Phosphorylation", "lipidation": "Lipidation", "go_cc": "GO CC", "go_mf": "GO MF",
    "go_bp": "GO BP",
}
TASK_ORDER = tuple(TASK_LABELS)
SHORT_LABELS = {**TASK_LABELS, "interpro_homologous_superfamily": "Hom. superfamily", "secondary_structure": "Sec. structure"}
SITE_LABELS = {
    "embedding": "embedding", "attn_ln": "attn LN", "q": "Q", "k": "K", "q_rot": "Q rotated", "k_rot": "K rotated",
    "v": "V", "attn_context": "attn context", "attn_out": "attn out", "resid_mid": "resid mid", "mlp_ln": "MLP LN",
    "mlp_gate": "MLP gate", "mlp_up": "MLP up", "mlp_act": "MLP act", "mlp_out": "MLP out",
    "resid_post": "resid post", "final_norm": "final norm",
}
BLOCK_SITES = tuple(site for site in SITE_LABELS if site not in ("embedding", "final_norm"))
RESIDUAL = ("embedding", "resid_post", "final_norm")
ATTENTION = ("attn_ln", "q", "k", "q_rot", "k_rot", "v", "attn_context", "attn_out")
MLP = ("resid_mid", "mlp_ln", "mlp_gate", "mlp_up", "mlp_act", "mlp_out")
CONDITIONS = (  # (plm, probe, control, legend), drawn in this order
    ("trained", "trained", "original", "trained model"),
    ("trained", "trained", "l2_normalized", "normalized tokens"),
    ("trained", "trained", "scrambled", "scrambled sequence"),
    ("trained", "trained", "mean", "batch mean state"),
    ("trained", "trained", "random_gaussian", "Gaussian input"),
    ("untrained", "trained", "original", "untrained model"),
    ("trained", "untrained", "original", "untrained probe"),
)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
        "savefig.dpi": 300, "figure.dpi": 100, "font.family": "DejaVu Sans",
    })
    return plt


def _colormap(colors: Sequence[str], name: str):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, list(colors))


def _letter(axis, letter: str) -> None:
    axis.text(-0.08, 1.04, letter, transform=axis.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right", color=INK)


def _tag(axis, text: str) -> None:
    """A task or site named over its small multiple, left aligned."""
    axis.text(0.0, 1.04, text, transform=axis.transAxes, fontsize=6.5, va="bottom", ha="left", color=INK)


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path


# -- tables ----------------------------------------------------------------------------------------------------------


def read_table(root: Path, label: str, name: str, folder: str = "results") -> pd.DataFrame:
    """One table of every task under `root/<folder>/<label>/`, concatenated."""
    paths = sorted((root / folder / label).glob(f"*/{name}.parquet"))
    if not paths:
        raise FileNotFoundError(f"No {name}.parquet under {root / folder / label}.")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def with_probe_parts(scores: pd.DataFrame) -> pd.DataFrame:
    """The table with `site`, `layer` and `head` (-1 for a full-width probe) split out of `probe_name`."""
    parts = scores["probe_name"].str.split(":", expand=True)
    head = parts[2].str.lstrip("h").astype(float).fillna(-1).astype(int) if parts.shape[1] > 2 else -1
    return scores.assign(site=parts[0], layer=parts[1].astype(int), head=head)


def depth(site: str, layer: int, blocks: int) -> int:
    """The residual-stream axis: embedding 0, block k's states k + 1, the final norm blocks + 1."""
    if site == "embedding":
        return 0
    if site == "final_norm":
        return blocks + 1
    return layer + 1


def summarize_halves(scores: pd.DataFrame) -> pd.DataFrame:
    """Per probe and condition: the full-test score and the mean and standard deviation over the test halves."""
    keys = ["task", "family", "probe_name", "plm", "probe", "control"]
    full = scores[scores["resample"] == 0].set_index(keys)["score"].rename("full")
    halves = scores[scores["resample"] > 0].groupby(keys)["score"].agg(["mean", "std"])
    return with_probe_parts(halves.join(full).reset_index())


def headline(summary: pd.DataFrame) -> pd.DataFrame:
    """The trained model read by trained probes on original inputs: the scores the depth curves show."""
    kept = (summary["plm"] == "trained") & (summary["probe"] == "trained") & (summary["control"] == "original")
    return summary[kept]


def best_states(summary: pd.DataFrame, blocks: int) -> pd.DataFrame:
    """Per task: the best residual-stream, attention, MLP and head probes, and the controls at the residual best."""
    rows = []
    trained = headline(summary)
    for task, block in trained.groupby("task"):
        full = block[block["family"] == "full"]
        row = {"task": task}
        for name, sites in (("residual", RESIDUAL), ("attention", ATTENTION), ("mlp", MLP), ("any", tuple(SITE_LABELS))):
            candidates = full[full["site"].isin(sites)]
            if candidates.empty:
                continue
            best = candidates.loc[candidates["full"].idxmax()]
            row[f"{name}_site"], row[f"{name}_layer"], row[f"{name}_score"] = best["site"], int(best["layer"]), float(best["full"])
            row[f"{name}_std"] = float(best["std"])
            row[f"{name}_depth"] = depth(best["site"], int(best["layer"]), blocks)
        heads = block[block["family"] == "head"]
        if not heads.empty:
            best = heads.loc[heads["full"].idxmax()]
            row["head_site"], row["head_layer"], row["head"], row["head_score"] = best["site"], int(best["layer"]), int(best["head"]), float(best["full"])
        if "residual_site" in row:
            same = summary[(summary["task"] == task) & (summary["probe_name"] == f"{row['residual_site']}:{row['residual_layer']}")]
            for plm, probe, control, legend in CONDITIONS[1:]:
                match = same[(same["plm"] == plm) & (same["probe"] == probe) & (same["control"] == control)]
                if not match.empty:
                    row[f"control_{plm}_{probe}_{control}"] = float(match["full"].iloc[0])
        rows.append(row)
    table = pd.DataFrame(rows)
    order = {task: position for position, task in enumerate(TASK_ORDER)}
    return table.sort_values("task", key=lambda column: column.map(order)).reset_index(drop=True)


# -- figures ---------------------------------------------------------------------------------------------------------


def _grid(plt, count: int, columns: int, width: float = 7.2, row_height: float = 1.25, **kwargs):
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(width, row_height * rows), squeeze=False, **kwargs)
    axes = list(axes.flat)
    for axis in axes[count:]:
        axis.set_visible(False)
    return figure, axes


def depth_curves(summary: pd.DataFrame, blocks: int, path: Path, tasks: Sequence[str] = TASK_ORDER, controls: bool = True) -> Path:
    """The residual stream of every task along depth: the trained model, and the untrained model and scrambled
    sequences as controls, mean over test halves with one standard deviation shaded."""
    plt = _pyplot()
    tasks = [task for task in tasks if task in set(summary["task"])]
    series = [CONDITIONS[0]] + ([CONDITIONS[5], CONDITIONS[2]] if controls else [])
    colors = {CONDITIONS[0]: PALETTE[0], CONDITIONS[5]: MUTED, CONDITIONS[2]: PALETTE[1]}
    figure, axes = _grid(plt, len(tasks), 6)
    stream = summary[summary["site"].isin(RESIDUAL) & (summary["family"] == "full")]
    for axis, task in zip(axes, tasks):
        block = stream[stream["task"] == task]
        for condition in series:
            plm, probe, control, legend = condition
            line = block[(block["plm"] == plm) & (block["probe"] == probe) & (block["control"] == control)].copy()
            if line.empty:
                continue
            line["depth"] = [depth(site, layer, blocks) for site, layer in zip(line["site"], line["layer"])]
            body = line[line["site"] != "final_norm"].sort_values("depth")
            axis.plot(body["depth"], body["mean"], color=colors[condition], label=legend, lw=1.2)
            axis.fill_between(body["depth"], body["mean"] - body["std"], body["mean"] + body["std"], color=colors[condition], alpha=0.18, lw=0)
            norm = line[line["site"] == "final_norm"]
            if not norm.empty:
                axis.plot(norm["depth"], norm["mean"], marker="o", ms=3, mfc="white", mec=colors[condition], mew=1.0, ls="none")
        _tag(axis, SHORT_LABELS.get(task, task))
        axis.set_xlim(-0.5, blocks + 1.5)
        axis.set_xticks([0, blocks // 2, blocks])
        axis.grid(axis="y", color=GRID, lw=0.5)
    for axis in axes[: len(tasks)]:
        if axis.get_subplotspec().is_first_col():
            axis.set_ylabel("score")
        if axis.get_subplotspec().is_last_row() or axis.get_subplotspec().rowspan.stop == axis.get_gridspec().nrows:
            axis.set_xlabel("depth")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.tight_layout(h_pad=0.8, w_pad=0.4, rect=(0, 0, 1, 1 - 0.3 / figure.get_figheight()))
    figure.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.0))
    return _save(figure, path)


def site_maps(summary: pd.DataFrame, path: Path, tasks: Sequence[str] = TASK_ORDER) -> Path:
    """Every per-block site of every task by block: the full-test score over the task's best probe."""
    plt = _pyplot()
    tasks = [task for task in tasks if task in set(summary["task"])]
    trained = headline(summary)
    trained = trained[(trained["family"] == "full") & trained["site"].isin(BLOCK_SITES)]
    sites = [site for site in BLOCK_SITES if site in set(trained["site"])]
    figure, axes = _grid(plt, len(tasks), 4, width=7.2, row_height=1.55)
    colormap = _colormap(BLUE_RAMP, "blue")
    image = None
    for axis, task in zip(axes, tasks):
        block = trained[trained["task"] == task]
        grid = block.pivot_table(index="site", columns="layer", values="full").reindex(sites)
        best = float(np.nanmax(grid.to_numpy()))
        relative = grid.to_numpy() / best if best > 0 else grid.to_numpy()
        image = axis.imshow(relative, aspect="auto", cmap=colormap, vmin=max(0.0, float(np.nanmin(relative))), vmax=1.0, interpolation="nearest")
        row, column = np.unravel_index(np.nanargmax(relative), relative.shape)
        axis.plot(column, row, marker="o", ms=4.5, mfc="none", mec="white", mew=1.1)
        axis.set_yticks(range(len(sites)))
        axis.set_yticklabels([SITE_LABELS[site] for site in sites] if axis.get_subplotspec().is_first_col() else [], fontsize=4.6)
        axis.set_xticks([0, grid.shape[1] // 2, grid.shape[1] - 1])
        axis.tick_params(length=1.5)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.text(0.0, 1.02, SHORT_LABELS.get(task, task), transform=axis.transAxes, fontsize=6, va="bottom", color=INK)
        axis.text(1.0, 1.02, f"best {best:.2f}", transform=axis.transAxes, fontsize=5.5, va="bottom", ha="right", color=INK_SECONDARY)
        if axis.get_subplotspec().rowspan.stop == axis.get_gridspec().nrows or task == tasks[-1]:
            axis.set_xlabel("block")
    figure.tight_layout(h_pad=1.0, w_pad=0.6, rect=(0, 0, 0.93, 1))
    if image is not None:
        bar = figure.colorbar(image, cax=figure.add_axes((0.945, 0.35, 0.012, 0.3)))
        bar.set_label("score / task best", fontsize=6)
        bar.ax.tick_params(labelsize=5.5)
    return _save(figure, path)


def best_state_dots(table: pd.DataFrame, path: Path) -> Path:
    """Per task: the best residual-stream, attention and MLP probes, the best single head, and the untrained model."""
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(4.6, 0.22 * len(table) + 0.8))
    positions = np.arange(len(table))[::-1]
    marks = (
        ("residual_score", "residual stream", PALETTE[0], "o"),
        ("attention_score", "attention sites", PALETTE[1], "o"),
        ("mlp_score", "MLP sites", PALETTE[2], "o"),
        ("head_score", "best single head", PALETTE[6], "D"),
        ("control_untrained_trained_original", "untrained model", MUTED, "x"),
    )
    for column, legend, color, marker in marks:
        if column not in table:
            continue
        style = {"facecolors": "none", "edgecolors": color, "linewidths": 1.0} if marker == "D" else {"color": color, "linewidths": 0.8}
        axis.scatter(table[column], positions, s=16 if marker != "x" else 14, marker=marker, label=legend, zorder=3, **style)
    for position, (_, row) in zip(positions, table.iterrows()):
        values = [row[column] for column, *_ in marks[:3] if column in table and pd.notna(row[column])]
        axis.plot([min(values), max(values)], [position, position], color=GRID, lw=2.0, zorder=1)
    axis.set_yticks(positions)
    axis.set_yticklabels([TASK_LABELS.get(task, task) for task in table["task"]])
    axis.set_xlabel("full test score (MCC; cosine for ProtParam)")
    axis.grid(axis="x", color=GRID, lw=0.5)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    figure.tight_layout()
    return _save(figure, path)


def peak_layers(summary: pd.DataFrame, path: Path, tasks: Sequence[str] = TASK_ORDER) -> Path:
    """The block at which each site's probe peaks, per task, with the peak score's share of the task's best."""
    plt = _pyplot()
    trained = headline(summary)
    trained = trained[(trained["family"] == "full") & trained["site"].isin(BLOCK_SITES)]
    tasks = [task for task in tasks if task in set(trained["task"])]
    sites = [site for site in BLOCK_SITES if site in set(trained["site"])]
    peaks = trained.loc[trained.groupby(["task", "site"])["full"].idxmax()]
    grid = peaks.pivot_table(index="task", columns="site", values="layer").reindex(index=tasks, columns=sites)
    figure, axis = plt.subplots(figsize=(0.32 * len(sites) + 1.6, 0.22 * len(tasks) + 0.9))
    image = axis.imshow(grid.to_numpy(), aspect="auto", cmap=_colormap(BLUE_RAMP, "blue"), interpolation="nearest")
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            value = grid.iat[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{int(value)}", ha="center", va="center", fontsize=4.8,
                          color="white" if value > np.nanmax(grid.to_numpy()) * 0.55 else INK)
    axis.set_xticks(range(len(sites)))
    axis.set_xticklabels([SITE_LABELS[site] for site in sites], rotation=60, ha="right")
    axis.set_yticks(range(len(tasks)))
    axis.set_yticklabels([TASK_LABELS.get(task, task) for task in tasks])
    for spine in axis.spines.values():
        spine.set_visible(False)
    bar = figure.colorbar(image, ax=axis, shrink=0.6, pad=0.02)
    bar.set_label("block of the peak", fontsize=6)
    figure.tight_layout()
    return _save(figure, path)


def control_curves(summary: pd.DataFrame, blocks: int, path: Path, tasks: Sequence[str] = TASK_ORDER) -> Path:
    """The residual stream of every task under every control, full test split."""
    plt = _pyplot()
    stream = summary[(summary["site"].isin(("embedding", "resid_post"))) & (summary["family"] == "full")]
    tasks = [task for task in tasks if task in set(stream["task"])]
    figure, axes = _grid(plt, len(tasks), 6)
    colors = dict(zip(CONDITIONS, PALETTE[:5] + (MUTED, PALETTE[6])))
    for axis, task in zip(axes, tasks):
        block = stream[stream["task"] == task]
        for condition in CONDITIONS:
            plm, probe, control, legend = condition
            line = block[(block["plm"] == plm) & (block["probe"] == probe) & (block["control"] == control)]
            if line.empty:
                continue
            ordered = line.assign(depth=[depth(s, l, blocks) for s, l in zip(line["site"], line["layer"])]).sort_values("depth")
            axis.plot(ordered["depth"], ordered["full"], color=colors[condition], label=legend, lw=1.0)
        _tag(axis, SHORT_LABELS.get(task, task))
        axis.set_xticks([0, blocks // 2, blocks])
        axis.grid(axis="y", color=GRID, lw=0.5)
        if axis.get_subplotspec().is_first_col():
            axis.set_ylabel("score")
        if axis is axes[len(tasks) - 1] or axis.get_subplotspec().rowspan.stop == axis.get_gridspec().nrows:
            axis.set_xlabel("depth")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.tight_layout(h_pad=0.8, w_pad=0.4, rect=(0, 0, 1, 1 - 0.45 / figure.get_figheight()))
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    return _save(figure, path)


def head_maps(summary: pd.DataFrame, tasks: Sequence[str], path: Path) -> Path:
    """Per task (rows) and head-split site (columns): every head's probe by block, full-test score."""
    plt = _pyplot()
    heads = headline(summary)
    heads = heads[heads["family"] == "head"]
    sites = [site for site in ATTENTION if site in set(heads["site"])]
    tasks = [task for task in tasks if task in set(heads["task"])]
    figure, axes = plt.subplots(len(tasks), len(sites), figsize=(1.2 * len(sites) + 0.6, 1.25 * len(tasks) + 0.3), squeeze=False)
    colormap = _colormap(BLUE_RAMP, "blue")
    for row, task in enumerate(tasks):
        block = heads[heads["task"] == task]
        low, high = float(block["full"].min()), float(block["full"].max())
        for column, site in enumerate(sites):
            axis = axes[row, column]
            grid = block[block["site"] == site].pivot_table(index="head", columns="layer", values="full")
            image = axis.imshow(grid.to_numpy(), aspect="auto", cmap=colormap, vmin=low, vmax=high, interpolation="nearest")
            axis.set_xticks([0, grid.shape[1] - 1])
            axis.set_yticks([0, grid.shape[0] - 1] if column == 0 else [])
            axis.tick_params(length=1.5, labelsize=5)
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.text(0.5, 1.04, SITE_LABELS[site], transform=axis.transAxes, ha="center", fontsize=6, color=INK)
            if column == 0:
                axis.set_ylabel(f"{TASK_LABELS.get(task, task)}\nhead", fontsize=6)
            if row == len(tasks) - 1:
                axis.set_xlabel("block", fontsize=6)
        bar = figure.colorbar(image, ax=list(axes[row]), shrink=0.8, pad=0.01, aspect=12)
        bar.ax.tick_params(labelsize=5)
    return _save(figure, path)


def pca_panels(frame: pd.DataFrame, path: Path, by: str = "layer") -> Path:
    """Two principal coordinates per layer (or site), proteins colored by label set; scrambled copies in green."""
    plt = _pyplot()
    panels = list(dict.fromkeys(frame[by]))
    groups = [group for group in frame["group"].value_counts().index if group not in ("other", "scrambled")]
    colors = {group: PALETTE[index] for index, group in enumerate(groups[:5])}
    colors["scrambled"] = PALETTE[6]
    figure, axes = plt.subplots(1, len(panels), figsize=(1.45 * len(panels), 1.75), squeeze=False)
    for axis, panel in zip(axes.flat, panels):
        block = frame[frame[by] == panel]
        other = block[~block["group"].isin(colors)]
        axis.scatter(other["pc1"], other["pc2"], s=1.2, color=GRID, lw=0, rasterized=True)
        for group, color in colors.items():
            chosen = block[block["group"] == group]
            axis.scatter(chosen["pc1"], chosen["pc2"], s=2.0 if group != "scrambled" else 1.6, color=color, lw=0, rasterized=True, label=group)
        name = f"block {panel}" if by == "layer" else SITE_LABELS.get(panel, panel)
        axis.text(0.03, 0.97, name, transform=axis.transAxes, va="top", fontsize=6, color=INK)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel(f"PC1 ({100 * block['explained_1'].iloc[0]:.0f}%)", fontsize=5.5)
        axis.set_ylabel(f"PC2 ({100 * block['explained_2'].iloc[0]:.0f}%)", fontsize=5.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    wrapped = [textwrap.fill(label, 44) for label in labels]
    figure.legend(handles, wrapped, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.22), markerscale=4)
    figure.tight_layout(w_pad=0.3)
    return _save(figure, path)


def phospho_figure(tests: pd.DataFrame, blocks: int, path: Path) -> Path:
    """A, B: the residual stream's mutant-activity MCC by depth, on raw (A) and standardized (B) pooled means, against
    shuffled labels and the untrained model. C: the trained model's mean MCC on raw means at every site and block."""
    plt = _pyplot()
    treatments = [inputs for inputs in ("raw", "standardized") if inputs in set(tests["inputs"])]
    figure, axes = plt.subplots(1, len(treatments) + 1, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1.0] * len(treatments) + [1.35]})
    curves = (("trained_true", "trained model, true labels", PALETTE[0]),
              ("trained_shuffled", "trained model, shuffled labels", PALETTE[1]),
              ("untrained_true", "untrained model, true labels", MUTED))
    for axis, inputs, letter in zip(axes, treatments, "AB"):
        stream = tests[(tests["inputs"] == inputs) & tests["site"].isin(("embedding", "resid_post"))].copy()
        stream["depth"] = [depth(site, layer, blocks) for site, layer in zip(stream["site"], stream["layer"])]
        stream = stream.sort_values("depth")
        for column, legend, color in curves:
            axis.plot(stream["depth"], stream[column], color=color, label=legend)
            axis.fill_between(stream["depth"], stream[column] - stream[f"{column}_std"], stream[column] + stream[f"{column}_std"],
                              color=color, alpha=0.12, lw=0)
        axis.set_xlabel("depth")
        axis.grid(axis="y", color=GRID, lw=0.5)
        _letter(axis, letter)
    axes[0].set_ylabel("mean test MCC over proteins")
    handles, labels = axes[0].get_legend_handles_labels()
    right = axes[-1]
    raw = tests[(tests["inputs"] == treatments[0]) & tests["site"].isin(BLOCK_SITES)]
    grid = raw.pivot_table(index="site", columns="layer", values="trained_true")
    grid = grid.reindex([site for site in BLOCK_SITES if site in grid.index])
    image = right.imshow(grid.to_numpy(), aspect="auto", cmap=_colormap(BLUE_RAMP, "blue"), interpolation="nearest")
    right.set_yticks(range(len(grid)))
    right.set_yticklabels([SITE_LABELS[site] for site in grid.index], fontsize=5.5)
    right.set_xlabel("block")
    for spine in right.spines.values():
        spine.set_visible(False)
    bar = figure.colorbar(image, ax=right, shrink=0.85, pad=0.02)
    bar.set_label("mean test MCC", fontsize=6)
    _letter(right, "ABC"[len(treatments)])
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    return _save(figure, path)


def prot_param_curves(elementwise: pd.DataFrame, label_names: Sequence[str], blocks: int, path: Path) -> Path:
    """Each ProtParam target's cosine similarity along the residual stream, mean over test halves."""
    plt = _pyplot()
    rows = with_probe_parts(elementwise[(elementwise["task"] == "prot_param") & (elementwise["family"] == "full")])
    rows = rows[rows["site"].isin(("embedding", "resid_post")) & (rows["resample"] > 0)]
    rows = rows.assign(depth=[depth(s, l, blocks) for s, l in zip(rows["site"], rows["layer"])])
    curves = rows.groupby(["label_index", "depth"])["score"].agg(["mean", "std"]).reset_index()
    figure, axes = _grid(plt, len(label_names), 8, width=7.2, row_height=1.05, sharey=True)
    low = min(0.0, float((curves["mean"] - curves["std"]).min()))
    for axis, (index, name) in zip(axes, enumerate(label_names)):
        line = curves[curves["label_index"] == index]
        axis.plot(line["depth"], line["mean"], color=PALETTE[0], lw=1.0)
        axis.fill_between(line["depth"], line["mean"] - line["std"], line["mean"] + line["std"], color=PALETTE[0], alpha=0.18, lw=0)
        axis.set_ylim(low, 1.02)
        axis.set_xticks([0, blocks])
        axis.grid(axis="y", color=GRID, lw=0.5)
        _tag(axis, name.replace("_", " "))
        if axis.get_subplotspec().is_first_col():
            axis.set_ylabel("cosine")
    figure.tight_layout(h_pad=0.6, w_pad=0.3)
    return _save(figure, path)


def sweep_curves(rankings: dict[str, pd.DataFrame], path: Path) -> Path:
    """Mean validation shortfall of every global setting by learning rate, one line per batch and weight decay."""
    plt = _pyplot()
    figure, axes = plt.subplots(1, len(rankings), figsize=(3.4 * len(rankings), 2.3), squeeze=False)
    for axis, (letter, (regime, ranked)) in zip(axes.flat, zip("ABCD", rankings.items())):
        column = "batch_size" if regime == "protein" else "max_tokens"
        unit = "proteins" if regime == "protein" else "tokens"
        batches = sorted(set(ranked[column]))
        decays = sorted(set(ranked["weight_decay"]))
        for (batch, decay), line in ranked.groupby([column, "weight_decay"]):
            line = line.sort_values("learning_rate")
            axis.plot(line["learning_rate"], line["mean_shortfall"], marker="o", ms=2.6, lw=1.1,
                      color=PALETTE[decays.index(decay)], ls=("-", "--", ":")[batches.index(batch) % 3],
                      label=f"{batch:,} {unit}, wd {decay:g}")
        axis.set_xscale("log")
        axis.set_xlabel("learning rate")
        axis.set_ylabel("mean validation shortfall")
        axis.grid(color=GRID, lw=0.5)
        axis.legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
        if len(rankings) > 1:
            _letter(axis, letter)
    figure.tight_layout()
    return _save(figure, path)


def write_summary(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.4f")
    return path


def all_figures(root: Path, label: str, out: Path, blocks: int, metadata: dict[str, dict] | None = None) -> list[Path]:
    """Every figure the result tables of `root` support, written under `out`; `metadata` maps each task to the task
    record its `run.json` holds, whose label names title the ProtParam panels."""
    written = []
    summary = summarize_halves(read_table(root, label, "scores"))
    out.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(out / "summary.parquet", index=False)
    written.append(out / "summary.parquet")
    written.append(depth_curves(summary, blocks, out / "depth_curves.png"))
    written.append(site_maps(summary, out / "site_maps.png"))
    table = best_states(summary, blocks)
    written.append(write_summary(table, out / "best_states.csv"))
    written.append(best_state_dots(table, out / "best_states.png"))
    written.append(peak_layers(summary, out / "peak_layers.png"))
    if set(summary["control"]) - {"original"}:
        written.append(control_curves(summary, blocks, out / "controls.png"))
    if (summary["family"] == "head").any():
        written.append(head_maps(summary, ["secondary_structure", "localization", "interpro_domain"], out / "heads.png"))
    elementwise_paths = sorted((root / "results" / label).glob("prot_param/elementwise.parquet"))
    if elementwise_paths and metadata and "prot_param" in metadata:
        elementwise = pd.read_parquet(elementwise_paths[0])
        written.append(prot_param_curves(elementwise, metadata["prot_param"]["label_names"], blocks, out / "prot_param_targets.png"))
    return written

