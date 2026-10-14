"""Figures and summary tables of tap-probing runs, over seed replicates and against a randomly initialized model.

Every figure reads the long tables the commands write (`results/<label>/<task>/scores.parquet` and
`elementwise.parquet`, `sweeps/<label>/<task>/sweep.parquet`, `interventions/<label>/scores.parquet`,
`latent/<label>/<task>.parquet`) and is saved as a 300 dpi PNG with no title, since the caption belongs to the document
that shows the figure. Multi-panel figures are lettered in reading order; small multiples name their task over the axes.

A replicate is one run of the whole analysis under its own seed, in its own root. A score is averaged over replicates;
a curve's band is one standard deviation over every replicate's test halves, and `full_std` is the standard deviation
of the full-test score over replicates alone. A probe name is `site:layer` or `site:layer:h<head>`. Depth puts the
residual stream on one axis: the embedding at 0, block k's states at k + 1, and the final norm one past the last block.
Scores are MCC, or cosine similarity for ProtParam.
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
DIVERGING = ("#c42f2e", "#f19a96", "#f0efec", "#6da7ec", "#184f95")  # negative red, zero gray, positive blue

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
MLP = ("mlp_ln", "mlp_gate", "mlp_up", "mlp_act", "mlp_out")
SITE_FAMILIES = (("attention", ("attn_ln", "q", "k", "v", "attn_context", "attn_out")), ("MLP", MLP), ("residual stream", ("resid_mid", "resid_post")))
HEAD_SITES = ("q", "k", "q_rot", "k_rot", "v", "attn_context")
CONDITIONS = (  # (plm, probe, control, legend) of the conditions the summary figures show, in drawing order
    ("trained", "trained", "original", "trained model"),
    ("trained", "trained", "l2_normalized", "normalized tokens"),
    ("trained", "trained", "scrambled", "scrambled sequence"),
    ("trained", "trained", "mean", "batch mean state"),
    ("trained", "trained", "random_gaussian", "Gaussian input"),
    ("untrained", "trained", "original", "untrained model, same probes"),
    ("trained", "untrained", "original", "untrained probe"),
)
CONTROL_INPUTS = (("original", "original input"), ("scrambled", "scrambled sequence"), ("l2_normalized", "normalized tokens"),
                  ("mean", "batch mean state"), ("random_gaussian", "Gaussian input"))
STATE_STYLES = ((("trained", "trained"), "-", "trained model, trained probe"), (("trained", "untrained"), "--", "trained model, untrained probe"),
                (("untrained", "trained"), ":", "untrained model, trained probe"), (("untrained", "untrained"), "-.", "untrained model, untrained probe"))
SELECTED_TASKS = ("prot_param", "peptide", "ssp_q8", "interpro_family", "go_bp")
UNTRAINED_LEGEND = "untrained model, same probes (control)"
RANDOM_LEGEND = "random-init model, own probes"
HEAD_TASKS = ("secondary_structure", "localization", "interpro_domain")


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


def _letter(axis, letter: str, x: float = -0.08) -> None:
    axis.text(x, 1.04, letter, transform=axis.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right", color=INK)


def _tag(axis, text: str) -> None:
    """A task or site named over its small multiple, left aligned."""
    axis.text(0.0, 1.04, text, transform=axis.transAxes, fontsize=6.5, va="bottom", ha="left", color=INK)


def _bare(axis) -> None:
    for spine in axis.spines.values():
        spine.set_visible(False)


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path


def _ordered(tasks: Sequence[str], present: set[str]) -> list[str]:
    return [task for task in tasks if task in present]


def _chosen(tasks: Sequence[str], present: set[str], count: int) -> list[str]:
    """The chosen tasks that are present, or the first `count` present tasks when none of them is."""
    return _ordered(tasks, present) or _ordered(TASK_ORDER, present)[:count]


def _grid(plt, count: int, columns: int, width: float = 7.2, row_height: float = 1.25, **kwargs):
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(width, row_height * rows), squeeze=False, **kwargs)
    axes = list(axes.flat)
    for axis in axes[count:]:
        axis.set_visible(False)
    return figure, axes


def _bottom(axis, axes: list, count: int) -> bool:
    """Whether a small multiple is the lowest visible one in its column."""
    spec = axis.get_subplotspec()
    below = axes.index(axis) + spec.get_gridspec().ncols
    return spec.rowspan.stop == spec.get_gridspec().nrows or below >= count


def _legend_top(figure, handles, labels, columns: int, reserve: float) -> None:
    figure.tight_layout(h_pad=0.8, w_pad=0.4, rect=(0, 0, 1, 1 - reserve / figure.get_figheight()))
    figure.legend(handles, labels, loc="upper center", ncol=columns, frameon=False, bbox_to_anchor=(0.5, 1.0))


# -- tables ----------------------------------------------------------------------------------------------------------


def read_table(root: Path, label: str, name: str, folder: str = "results") -> pd.DataFrame:
    """One table of every task under `root/<folder>/<label>/`, concatenated."""
    paths = sorted((root / folder / label).glob(f"*/{name}.parquet"))
    if not paths:
        raise FileNotFoundError(f"No {name}.parquet under {root / folder / label}.")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def read_replicates(roots: Sequence[Path], label: str, name: str = "scores") -> pd.DataFrame:
    """The table of every replicate (one root per seed), with a `replicate` column numbering the roots."""
    return pd.concat([read_table(Path(root), label, name).assign(replicate=index) for index, root in enumerate(roots)], ignore_index=True)


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


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """Per probe and condition: `full`, the full-test score averaged over replicates, and `full_std`, its standard
    deviation over replicates (0 for one); `mean` and `std` over every replicate's test halves; `replicates`."""
    if "replicate" not in scores:
        scores = scores.assign(replicate=0)
    keys = ["task", "family", "probe_name", "plm", "probe", "control"]
    full = scores[scores["resample"] == 0].groupby(keys)["score"].agg(full="mean", full_std="std", replicates="count")
    full["full_std"] = full["full_std"].fillna(0.0)
    halves = scores[scores["resample"] > 0].groupby(keys)["score"].agg(["mean", "std"])
    return with_probe_parts(halves.join(full).reset_index())


def full_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Each replicate's full-test scores, with the probe parts: the rows a per-replicate statistic reads."""
    if "replicate" not in scores:
        scores = scores.assign(replicate=0)
    return with_probe_parts(scores[scores["resample"] == 0])


def headline(summary: pd.DataFrame) -> pd.DataFrame:
    """The model read by trained probes on original inputs: the scores the depth curves show."""
    kept = (summary["plm"] == "trained") & (summary["probe"] == "trained") & (summary["control"] == "original")
    return summary[kept]


def _with_depth(frame: pd.DataFrame, blocks: int) -> pd.DataFrame:
    return frame.assign(depth=[depth(site, layer, blocks) for site, layer in zip(frame["site"], frame["layer"])])


def best_states(summary: pd.DataFrame, blocks: int) -> pd.DataFrame:
    """Per task: the best residual-stream, attention, MLP and any-site probes, the best single head, and the controls at
    the residual-stream best, by full-test score averaged over replicates."""
    rows = []
    trained = headline(summary)
    for task, block in trained.groupby("task"):
        full = block[block["family"] == "full"]
        row = {"task": task}
        for name, sites in (("residual", RESIDUAL), ("attention", ATTENTION), ("mlp", MLP + ("resid_mid",)), ("any", tuple(SITE_LABELS))):
            candidates = full[full["site"].isin(sites)]
            if candidates.empty:
                continue
            best = candidates.loc[candidates["full"].idxmax()]
            row[f"{name}_site"], row[f"{name}_layer"], row[f"{name}_score"] = best["site"], int(best["layer"]), float(best["full"])
            row[f"{name}_std"], row[f"{name}_replicate_std"] = float(best["std"]), float(best["full_std"])
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


def site_bests(summary: pd.DataFrame) -> pd.DataFrame:
    """Per task and site: the best full-test score over depth (layers) of its full-width probes, and where it peaks."""
    trained = headline(summary)
    trained = trained[trained["family"] == "full"]
    peaks = trained.loc[trained.groupby(["task", "site"])["full"].idxmax()]
    return peaks[["task", "site", "layer", "full", "full_std"]].reset_index(drop=True)


def replicate_spread(summary: pd.DataFrame) -> pd.DataFrame:
    """Per task: the median, over its full-width probes, of the standard deviation over replicates and over test halves."""
    trained = headline(summary)
    trained = trained[trained["family"] == "full"]
    table = trained.groupby("task").agg(replicate_std=("full_std", "median"), halves_std=("std", "median"),
                                        replicates=("replicates", "max")).reset_index()
    order = {task: position for position, task in enumerate(TASK_ORDER)}
    return table.sort_values("task", key=lambda column: column.map(order)).reset_index(drop=True)


def random_init_table(trained: pd.DataFrame, random: pd.DataFrame, blocks: int) -> pd.DataFrame:
    """Per task: the best residual-stream and any-state probe of each model, and the trained model's margin."""
    best = {name: best_states(summary, blocks).set_index("task") for name, summary in (("trained", trained), ("random", random))}
    columns = ["residual_site", "residual_layer", "residual_score", "residual_replicate_std", "any_site", "any_layer", "any_score", "any_replicate_std"]
    table = best["trained"][columns].join(best["random"][columns], lsuffix="_trained", rsuffix="_random", how="inner")
    table["residual_margin"] = table["residual_score_trained"] - table["residual_score_random"]
    table["any_margin"] = table["any_score_trained"] - table["any_score_random"]
    order = {task: position for position, task in enumerate(TASK_ORDER)}
    return table.reset_index().sort_values("task", key=lambda column: column.map(order)).reset_index(drop=True)


def write_summary(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.4f")
    return path


# -- residual stream along depth -------------------------------------------------------------------------------------


def _depth_line(axis, frame: pd.DataFrame, blocks: int, color: str, legend: str, style: str = "-", band: bool = True) -> None:
    """One residual-stream curve: the blocks as a line with its band, the final norm as a hollow marker."""
    frame = _with_depth(frame, blocks)
    body = frame[frame["site"] != "final_norm"].sort_values("depth")
    axis.plot(body["depth"], body["mean"], color=color, label=legend, lw=1.2, ls=style)
    if band:
        axis.fill_between(body["depth"], body["mean"] - body["std"], body["mean"] + body["std"], color=color, alpha=0.16, lw=0)
    norm = frame[frame["site"] == "final_norm"]
    if not norm.empty:
        axis.plot(norm["depth"], norm["mean"], marker="o", ms=3, mfc="white", mec=color, mew=1.0, ls="none")


def depth_curves(summary: pd.DataFrame, blocks: int, path: Path, random: pd.DataFrame | None = None, tasks: Sequence[str] = TASK_ORDER) -> Path:
    """The residual stream of every task along depth: the trained model; its untrained-model and scrambled-sequence
    controls; and, with `random`, probes trained on a randomly initialized model's states."""
    plt = _pyplot()
    tasks = _ordered(tasks, set(summary["task"]))
    figure, axes = _grid(plt, len(tasks), 6)
    stream = summary[summary["site"].isin(RESIDUAL) & (summary["family"] == "full")]
    series = [((("trained", "trained", "original"), stream), PALETTE[0], "trained model", "-", True),
              ((("untrained", "trained", "original"), stream), MUTED, UNTRAINED_LEGEND, "-", False),
              ((("trained", "trained", "scrambled"), stream), PALETTE[1], "scrambled sequence (control)", "-", False)]
    if random is not None:
        random_stream = random[random["site"].isin(RESIDUAL) & (random["family"] == "full")]
        series.insert(1, ((("trained", "trained", "original"), random_stream), INK_SECONDARY, RANDOM_LEGEND, "--", True))
    for axis, task in zip(axes, tasks):
        for ((plm, probe, control), table), color, legend, style, band in series:
            line = table[(table["task"] == task) & (table["plm"] == plm) & (table["probe"] == probe) & (table["control"] == control)]
            if not line.empty:
                _depth_line(axis, line, blocks, color, legend, style, band)
        _tag(axis, SHORT_LABELS.get(task, task))
        axis.set_xlim(-0.5, blocks + 1.5)
        axis.set_xticks([0, blocks // 2, blocks])
        axis.grid(axis="y", color=GRID, lw=0.5)
        if axis.get_subplotspec().is_first_col():
            axis.set_ylabel("score")
        if _bottom(axis, axes, len(tasks)):
            axis.set_xlabel("depth")
    handles, labels = axes[0].get_legend_handles_labels()
    _legend_top(figure, handles, labels, len(labels), 0.3)
    return _save(figure, path)


def selected_tasks(summary: pd.DataFrame, per_replicate: pd.DataFrame, blocks: int, path: Path, tasks: Sequence[str] = SELECTED_TASKS) -> Path:
    """A few tasks' residual streams, the dashed line at the best depth averaged over replicates, its spread shaded."""
    plt = _pyplot()
    tasks = _chosen(tasks, set(summary["task"]), len(SELECTED_TASKS))
    figure, axes = plt.subplots(1, len(tasks), figsize=(7.2, 1.6), squeeze=False)
    trained = headline(summary)
    stream = trained[trained["site"].isin(RESIDUAL) & (trained["family"] == "full")]
    replicates = headline(per_replicate)
    replicates = _with_depth(replicates[replicates["site"].isin(RESIDUAL) & (replicates["family"] == "full")], blocks)
    for axis, task in zip(axes.flat, tasks):
        _depth_line(axis, stream[stream["task"] == task], blocks, PALETTE[0], "trained model")
        rows = replicates[replicates["task"] == task]
        best = rows.loc[rows.groupby("replicate")["score"].idxmax(), "depth"].to_numpy(dtype=float)
        center, spread = float(best.mean()), float(best.std(ddof=1)) if len(best) > 1 else 0.0
        axis.axvspan(center - spread, center + spread, color=GRID, alpha=0.9, lw=0, zorder=0)
        axis.axvline(center, color=INK_SECONDARY, ls="--", lw=0.8)
        _tag(axis, SHORT_LABELS.get(task, task))
        axis.set_xlim(-0.5, blocks + 1.5)
        axis.set_xticks([0, blocks // 2, blocks])
        axis.set_xlabel("depth")
        axis.grid(axis="y", color=GRID, lw=0.5)
    axes.flat[0].set_ylabel("score")
    figure.tight_layout(w_pad=0.6)
    return _save(figure, path)


def control_grid(summary: pd.DataFrame, blocks: int, path: Path, tasks: Sequence[str] = TASK_ORDER) -> Path:
    """The residual stream of every task under all twenty conditions: color by input, line style by model and probe."""
    plt = _pyplot()
    stream = summary[summary["site"].isin(("embedding", "resid_post")) & (summary["family"] == "full")]
    tasks = _ordered(tasks, set(stream["task"]))
    figure, axes = _grid(plt, len(tasks), 6, row_height=1.3)
    for axis, task in zip(axes, tasks):
        block = _with_depth(stream[stream["task"] == task], blocks)
        for (control, _), color in zip(CONTROL_INPUTS, PALETTE):
            for (plm, probe), style, _ in STATE_STYLES:
                line = block[(block["plm"] == plm) & (block["probe"] == probe) & (block["control"] == control)].sort_values("depth")
                if not line.empty:
                    axis.plot(line["depth"], line["full"], color=color, ls=style, lw=0.9)
        _tag(axis, SHORT_LABELS.get(task, task))
        axis.set_xticks([0, blocks // 2, blocks])
        axis.grid(axis="y", color=GRID, lw=0.5)
        if axis.get_subplotspec().is_first_col():
            axis.set_ylabel("score")
        if _bottom(axis, axes, len(tasks)):
            axis.set_xlabel("depth")
    from matplotlib.lines import Line2D

    colors = [Line2D([], [], color=color, lw=1.4) for color in PALETTE[: len(CONTROL_INPUTS)]]
    styles = [Line2D([], [], color=INK_SECONDARY, ls=style, lw=1.1) for _, style, _ in STATE_STYLES]
    height = figure.get_figheight()
    figure.tight_layout(h_pad=0.8, w_pad=0.4, rect=(0, 0, 1, 1 - 0.42 / height))
    spacing = {"frameon": False, "loc": "upper center", "columnspacing": 1.2, "handlelength": 2.2}
    figure.legend(colors, [legend for _, legend in CONTROL_INPUTS], ncol=len(colors), bbox_to_anchor=(0.5, 1.0), **spacing)
    figure.legend(styles, [legend for _, _, legend in STATE_STYLES], ncol=len(styles), bbox_to_anchor=(0.5, 1.0 - 0.17 / height), **spacing)
    return _save(figure, path)


def target_label(name: str) -> str:
    """A ProtParam target's name for a figure: `percent_a` is "% A", `charge_at_ph4_7` is "charge at pH 4.7"."""
    if name.startswith("percent_"):
        return f"% {name.removeprefix('percent_').upper()}"
    if name.startswith("charge_at_ph"):
        return f"charge at pH {name.removeprefix('charge_at_ph').replace('_', '.')}"
    if name.startswith("fraction_") and name.endswith("_aas"):
        return f"{name.removeprefix('fraction_').removesuffix('_aas')} fraction"
    return {"gravy": "GRAVY"}.get(name, name.replace("_", " "))


def prot_param_curves(elementwise: pd.DataFrame, label_names: Sequence[str], blocks: int, path: Path) -> Path:
    """Each ProtParam target's cosine similarity along the residual stream, over replicates and test halves."""
    plt = _pyplot()
    rows = with_probe_parts(elementwise[(elementwise["task"] == "prot_param") & (elementwise["family"] == "full")])
    rows = _with_depth(rows[rows["site"].isin(("embedding", "resid_post")) & (rows["resample"] > 0)], blocks)
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
        _tag(axis, target_label(name))
        if axis.get_subplotspec().is_first_col():
            axis.set_ylabel("cosine")
        if _bottom(axis, axes, len(label_names)):
            axis.set_xlabel("depth")
    figure.tight_layout(h_pad=0.6, w_pad=0.3)
    return _save(figure, path)


# -- inside the block --------------------------------------------------------------------------------------------------


def site_maps(summary: pd.DataFrame, path: Path, tasks: Sequence[str] = TASK_ORDER) -> Path:
    """Every per-block site of every task by block: the full-test score over the task's best probe."""
    plt = _pyplot()
    tasks = _ordered(tasks, set(summary["task"]))
    trained = headline(summary)
    trained = trained[(trained["family"] == "full") & trained["site"].isin(BLOCK_SITES)]
    sites = [site for site in BLOCK_SITES if site in set(trained["site"])]
    figure, axes = _grid(plt, len(tasks), 4, width=7.2, row_height=1.55)
    colormap = _colormap(BLUE_RAMP, "blue")
    image = None
    for axis, task in zip(axes, tasks):
        grid = trained[trained["task"] == task].pivot_table(index="site", columns="layer", values="full").reindex(sites)
        best = float(np.nanmax(grid.to_numpy()))
        relative = grid.to_numpy() / best if best > 0 else grid.to_numpy()
        image = axis.imshow(relative, aspect="auto", cmap=colormap, vmin=0.0, vmax=1.0, interpolation="nearest")
        row, column = np.unravel_index(np.nanargmax(relative), relative.shape)
        axis.plot(column, row, marker="o", ms=4.5, mfc="none", mec="white", mew=1.1)
        axis.set_yticks(range(len(sites)))
        axis.set_yticklabels([SITE_LABELS[site] for site in sites] if axis.get_subplotspec().is_first_col() else [], fontsize=4.6)
        axis.set_xticks([0, grid.shape[1] // 2, grid.shape[1] - 1])
        axis.tick_params(length=1.5)
        _bare(axis)
        axis.text(0.0, 1.02, SHORT_LABELS.get(task, task), transform=axis.transAxes, fontsize=6, va="bottom", color=INK)
        axis.text(1.0, 1.02, f"best {best:.2f}", transform=axis.transAxes, fontsize=5.5, va="bottom", ha="right", color=INK_SECONDARY)
        if _bottom(axis, axes, len(tasks)):
            axis.set_xlabel("block")
    figure.tight_layout(h_pad=1.0, w_pad=0.6, rect=(0, 0, 0.93, 1))
    if image is not None:
        bar = figure.colorbar(image, cax=figure.add_axes((0.945, 0.35, 0.012, 0.3)))
        bar.set_label("score / task best", fontsize=6)
        bar.ax.tick_params(labelsize=5.5)
    return _save(figure, path)


def site_gap_map(summary: pd.DataFrame, path: Path, tasks: Sequence[str] = TASK_ORDER) -> Path:
    """A: every site's best score over depth minus the task's best residual-stream score (the embedding, lower in every
    task, is left out); cells at least 0.05 from zero are labeled. B: the block at which each site's probe peaks."""
    plt = _pyplot()
    bests = site_bests(summary)
    tasks = _ordered(tasks, set(bests["task"]))
    sites = [site for site in SITE_LABELS if site in set(bests["site"])]
    scores = bests.pivot_table(index="task", columns="site", values="full").reindex(index=tasks, columns=sites)
    residual = scores[[site for site in RESIDUAL if site in scores]].max(axis=1)
    gaps = scores.sub(residual, axis=0).drop(columns="embedding", errors="ignore")
    blocks_at = bests.pivot_table(index="task", columns="site", values="layer").reindex(index=tasks, columns=[s for s in sites if s in BLOCK_SITES])
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.2, 0.2 * len(tasks) + 1.3), gridspec_kw={"width_ratios": [gaps.shape[1], blocks_at.shape[1] * 0.9]})
    limit = float(np.nanmax(np.abs(gaps.to_numpy()))) or 1.0
    image = left.imshow(gaps.to_numpy(), aspect="auto", cmap=_colormap(DIVERGING, "diverging"), vmin=-limit, vmax=limit, interpolation="nearest")
    for row in range(gaps.shape[0]):
        for column in range(gaps.shape[1]):
            value = gaps.iat[row, column]
            if np.isfinite(value) and abs(value) >= 0.05:
                left.text(column, row, f"{value:+.2f}".replace("0.", "."), ha="center", va="center", fontsize=4.2,
                          color="white" if abs(value) > 0.6 * limit else INK)
    left.set_xticks(range(gaps.shape[1]))
    left.set_xticklabels([SITE_LABELS[site] for site in gaps.columns], rotation=60, ha="right")
    left.set_yticks(range(len(tasks)))
    left.set_yticklabels([TASK_LABELS.get(task, task) for task in tasks])
    _bare(left)
    bar = figure.colorbar(image, ax=left, shrink=0.5, pad=0.01, aspect=25)
    bar.set_label("best score minus best residual-stream score", fontsize=5.5)
    bar.ax.tick_params(labelsize=5)
    _letter(left, "A", -0.02)
    image = right.imshow(blocks_at.to_numpy(), aspect="auto", cmap=_colormap(BLUE_RAMP, "blue"), interpolation="nearest")
    for row in range(blocks_at.shape[0]):
        for column in range(blocks_at.shape[1]):
            value = blocks_at.iat[row, column]
            if np.isfinite(value):
                right.text(column, row, f"{int(value)}", ha="center", va="center", fontsize=4.2,
                           color="white" if value > np.nanmax(blocks_at.to_numpy()) * 0.55 else INK)
    right.set_xticks(range(blocks_at.shape[1]))
    right.set_xticklabels([SITE_LABELS[site] for site in blocks_at.columns], rotation=60, ha="right")
    right.set_yticks([])
    _bare(right)
    bar = figure.colorbar(image, ax=right, shrink=0.5, pad=0.01, aspect=25)
    bar.set_label("block of the peak", fontsize=5.5)
    bar.ax.tick_params(labelsize=5)
    _letter(right, "B", -0.02)
    figure.tight_layout(w_pad=0.8)
    return _save(figure, path)


def site_family_curves(summary: pd.DataFrame, blocks: int, path: Path, tasks: Sequence[str] = ("ssp_q8", "topology", "localization", "go_mf")) -> Path:
    """Rows of tasks, columns of site families: every site's probe along the blocks, the block output dashed for scale."""
    plt = _pyplot()
    trained = headline(summary)
    trained = trained[(trained["family"] == "full") & trained["site"].isin(BLOCK_SITES)]
    tasks = _chosen(tasks, set(trained["task"]), 4)
    figure, axes = plt.subplots(len(tasks), len(SITE_FAMILIES), figsize=(7.2, 1.35 * len(tasks) + 0.5), squeeze=False, sharex=True)
    for row, task in enumerate(tasks):
        block = trained[trained["task"] == task]
        reference = block[block["site"] == "resid_post"].sort_values("layer")
        low, high = float(block["mean"].min()), float(block["mean"].max())
        for column, (family, sites) in enumerate(SITE_FAMILIES):
            axis = axes[row, column]
            for index, site in enumerate(sites):
                line = block[block["site"] == site].sort_values("layer")
                if not line.empty:
                    axis.plot(line["layer"], line["mean"], color=PALETTE[index % len(PALETTE)], lw=1.0, label=SITE_LABELS[site])
            if family != "residual stream":
                axis.plot(reference["layer"], reference["mean"], color=INK_SECONDARY, ls="--", lw=0.8, label="resid post")
            axis.set_ylim(low - 0.03 * (high - low), high + 0.05 * (high - low))
            axis.grid(axis="y", color=GRID, lw=0.5)
            if column == 0:
                axis.set_ylabel(f"{SHORT_LABELS.get(task, task)}\nscore", fontsize=6)
            if row == len(tasks) - 1:
                axis.set_xlabel("block")
            if row == 0:
                axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=5, handlelength=1.4, columnspacing=0.8)
    figure.tight_layout(h_pad=0.6, w_pad=0.8)
    return _save(figure, path)


def rotation_effect(summary: pd.DataFrame, path: Path, tasks: Sequence[str] = TASK_ORDER) -> Path:
    """Per task, the mean over blocks of the rotated minus the unrotated score, for queries and keys: full-width probes
    and the mean over single-head probes."""
    plt = _pyplot()
    trained = headline(summary)
    tasks = _ordered(tasks, set(trained["task"]))
    figure, axis = plt.subplots(figsize=(3.6, 0.2 * len(tasks) + 0.9))
    positions = np.arange(len(tasks))[::-1]
    marks = []
    for family, marker in (("full", "o"), ("head", "D")):
        rows = trained[trained["family"] == family]
        if rows.empty:
            continue
        per_layer = rows.groupby(["task", "site", "layer"])["full"].mean().unstack("site")
        for base, rotated, color in (("q", "q_rot", PALETTE[0]), ("k", "k_rot", PALETTE[1])):
            if base not in per_layer or rotated not in per_layer:
                continue
            effect = (per_layer[rotated] - per_layer[base]).groupby("task").mean().reindex(tasks)
            label = f"{SITE_LABELS[base]}, {'full width' if family == 'full' else 'single heads'}"
            hollow = family == "head"
            axis.scatter(effect.to_numpy(), positions, marker=marker, s=14, label=label, zorder=3,
                         **({"facecolors": "none", "edgecolors": color, "linewidths": 1.0} if hollow else {"color": color}))
            marks.append(effect)
    axis.axvline(0.0, color=INK_SECONDARY, lw=0.7)
    axis.set_yticks(positions)
    axis.set_yticklabels([TASK_LABELS.get(task, task) for task in tasks])
    axis.set_xlabel("rotated minus unrotated score, mean over blocks")
    axis.grid(axis="x", color=GRID, lw=0.5)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    figure.tight_layout()
    return _save(figure, path)


def best_state_dots(table: pd.DataFrame, path: Path, random: pd.DataFrame | None = None) -> Path:
    """Per task: the best residual-stream, attention and MLP probes and the best single head, dodged vertically; the
    untrained-model control at the residual best; and, with `random`, the best probe of a randomly initialized model."""
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(4.6, 0.24 * len(table) + 0.8))
    positions = np.arange(len(table))[::-1]
    marks = [
        ("residual_score", "residual stream", PALETTE[0], "o", 0.24),
        ("attention_score", "attention sites", PALETTE[1], "o", 0.08),
        ("mlp_score", "MLP sites", PALETTE[2], "o", -0.08),
        ("head_score", "best single head", PALETTE[6], "D", -0.24),
        ("control_untrained_trained_original", UNTRAINED_LEGEND, MUTED, "x", 0.0),
    ]
    if random is not None:
        table = table.merge(random.rename(columns={"any_score": "random_score"})[["task", "random_score"]], on="task", how="left")
        marks.append(("random_score", RANDOM_LEGEND, INK_SECONDARY, "s", 0.0))
    for position in positions[::2]:
        axis.axhspan(position - 0.5, position + 0.5, color="#f6f5f2", lw=0, zorder=0)
    for column, legend, color, marker, offset in marks:
        if column not in table:
            continue
        style = {"facecolors": "none", "edgecolors": color, "linewidths": 0.9} if marker in "Ds" else {"color": color, "linewidths": 0.8}
        axis.scatter(table[column], positions + offset, s=13 if marker != "s" else 18, marker=marker, label=legend, zorder=3, **style)
    axis.set_yticks(positions)
    axis.set_yticklabels([TASK_LABELS.get(task, task) for task in table["task"]])
    axis.set_ylim(-0.6, len(table) - 0.4)
    axis.set_xlabel("full test score (MCC; cosine for ProtParam)")
    axis.grid(axis="x", color=GRID, lw=0.5)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=5.5, columnspacing=1.0)
    figure.tight_layout()
    return _save(figure, path)


def head_maps(summary: pd.DataFrame, path: Path, tasks: Sequence[str] = HEAD_TASKS) -> Path:
    """Per task (rows) and head-split site (columns): every head's probe by block, full-test score."""
    plt = _pyplot()
    heads = headline(summary)
    heads = heads[heads["family"] == "head"]
    sites = [site for site in HEAD_SITES if site in set(heads["site"])]
    tasks = _chosen(tasks, set(heads["task"]), len(HEAD_TASKS))
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
            _bare(axis)
            if row == 0:
                axis.text(0.5, 1.04, SITE_LABELS[site], transform=axis.transAxes, ha="center", fontsize=6, color=INK)
            if column == 0:
                axis.set_ylabel(f"{SHORT_LABELS.get(task, task)}\nhead", fontsize=6)
            if row == len(tasks) - 1:
                axis.set_xlabel("block", fontsize=6)
        bar = figure.colorbar(image, ax=list(axes[row]), shrink=0.8, pad=0.01, aspect=12)
        bar.ax.tick_params(labelsize=5)
    return _save(figure, path)


# -- against a randomly initialized model ------------------------------------------------------------------------------


def diff_maps(
    first: pd.DataFrame,
    second: pd.DataFrame,
    path: Path,
    tasks: Sequence[str] = TASK_ORDER,
    label: str = "trained minus random-init score",
) -> Path:
    """Every per-block site of every task by block: the first summary's score minus the second's, both averaged over
    replicates, on one color scale for all tasks."""
    plt = _pyplot()
    first, second = (headline(summary) for summary in (first, second))
    first = first[(first["family"] == "full") & first["site"].isin(BLOCK_SITES)]
    second = second[(second["family"] == "full") & second["site"].isin(BLOCK_SITES)]
    tasks = _ordered(tasks, set(first["task"]) & set(second["task"]))
    sites = [site for site in BLOCK_SITES if site in set(first["site"])]
    grids = {}
    for task in tasks:
        a = first[first["task"] == task].pivot_table(index="site", columns="layer", values="full").reindex(sites)
        b = second[second["task"] == task].pivot_table(index="site", columns="layer", values="full").reindex(index=sites, columns=a.columns)
        grids[task] = a - b
    limit = float(np.nanpercentile(np.abs(np.concatenate([grid.to_numpy().ravel() for grid in grids.values()])), 99)) or 1.0
    figure, axes = _grid(plt, len(tasks), 4, width=7.2, row_height=1.55)
    colormap = _colormap(DIVERGING, "diverging")
    image = None
    for axis, task in zip(axes, tasks):
        grid = grids[task]
        image = axis.imshow(grid.to_numpy(), aspect="auto", cmap=colormap, vmin=-limit, vmax=limit, interpolation="nearest")
        axis.set_yticks(range(len(sites)))
        axis.set_yticklabels([SITE_LABELS[site] for site in sites] if axis.get_subplotspec().is_first_col() else [], fontsize=4.6)
        axis.set_xticks([0, grid.shape[1] // 2, grid.shape[1] - 1])
        axis.tick_params(length=1.5)
        _bare(axis)
        axis.text(0.0, 1.02, SHORT_LABELS.get(task, task), transform=axis.transAxes, fontsize=6, va="bottom", color=INK)
        axis.text(1.0, 1.02, f"mean {np.nanmean(grid.to_numpy()):+.2f}", transform=axis.transAxes, fontsize=5.5, va="bottom", ha="right", color=INK_SECONDARY)
        if _bottom(axis, axes, len(tasks)):
            axis.set_xlabel("block")
    figure.tight_layout(h_pad=1.0, w_pad=0.6, rect=(0, 0, 0.93, 1))
    if image is not None:
        bar = figure.colorbar(image, cax=figure.add_axes((0.945, 0.35, 0.012, 0.3)))
        bar.set_label(label, fontsize=6)
        bar.ax.tick_params(labelsize=5.5)
    return _save(figure, path)


def diff_summary(
    first: pd.DataFrame,
    second: pd.DataFrame,
    path: Path,
    tasks: Sequence[str] = TASK_ORDER,
    names: tuple[str, str] = ("trained model", "random-init model"),
) -> Path:
    """A: per task and site, the best score over depth under the first summary minus that under the second.
    B: the best probe of each task under each, with the spread over replicates."""
    plt = _pyplot()
    first, second = site_bests(first), site_bests(second)
    tasks = _ordered(tasks, set(first["task"]) & set(second["task"]))
    sites = [site for site in SITE_LABELS if site in set(first["site"])]
    a = first.pivot_table(index="task", columns="site", values="full").reindex(index=tasks, columns=sites)
    b = second.pivot_table(index="task", columns="site", values="full").reindex(index=tasks, columns=sites)
    gaps = a - b
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.2, 0.2 * len(tasks) + 1.3), gridspec_kw={"width_ratios": [len(sites), 6]})
    limit = float(np.nanmax(np.abs(gaps.to_numpy()))) or 1.0
    image = left.imshow(gaps.to_numpy(), aspect="auto", cmap=_colormap(DIVERGING, "diverging"), vmin=-limit, vmax=limit, interpolation="nearest")
    for row in range(gaps.shape[0]):
        for column in range(gaps.shape[1]):
            value = gaps.iat[row, column]
            if np.isfinite(value):
                label = f"{value:.2f}".replace("-0.00", "0.00")
                left.text(column, row, label, ha="center", va="center", fontsize=3.8, color="white" if abs(value) > 0.6 * limit else INK)
    left.set_xticks(range(len(sites)))
    left.set_xticklabels([SITE_LABELS[site] for site in sites], rotation=60, ha="right")
    left.set_yticks(range(len(tasks)))
    left.set_yticklabels([TASK_LABELS.get(task, task) for task in tasks])
    _bare(left)
    bar = figure.colorbar(image, ax=left, shrink=0.5, pad=0.01, aspect=25)
    bar.set_label(f"{names[0]} minus {names[1]}, best score", fontsize=5.5)
    bar.ax.tick_params(labelsize=5)
    _letter(left, "A", -0.02)
    best_first = first.loc[first.groupby("task")["full"].idxmax()].set_index("task").reindex(tasks)
    best_second = second.loc[second.groupby("task")["full"].idxmax()].set_index("task").reindex(tasks)
    positions = np.arange(len(tasks))[::-1]
    for frame, color, marker, size, legend in ((best_second, INK_SECONDARY, "s", 4.6, names[1]), (best_first, PALETTE[0], "o", 3.0, names[0])):
        right.errorbar(frame["full"], positions, xerr=frame["full_std"], fmt=marker, ms=size, color=color, ecolor=color,
                       elinewidth=0.8, capsize=0, label=legend, mfc=color if marker == "o" else "white", mec=color, mew=0.9)
    for position, value_first, value_second in zip(positions, best_first["full"], best_second["full"]):
        right.plot([value_second, value_first], [position, position], color=GRID, lw=2.0, zorder=0)
    right.set_yticks([])
    right.set_ylim(-0.6, len(tasks) - 0.4)
    right.set_xlabel("best full test score")
    right.grid(axis="x", color=GRID, lw=0.5)
    right.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False, fontsize=5.5)
    _letter(right, "B", -0.02)
    figure.tight_layout(w_pad=0.8)
    return _save(figure, path)


# -- variants of the analysis ------------------------------------------------------------------------------------------


def variant_table(variants: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per task and variant: the best residual-stream and any-state full-test scores (with their spread over
    replicates) and the mean over every full-width probe; a last row per variant averages the tasks all variants share."""
    rows = []
    for name, summary in variants.items():
        trained = headline(summary)
        trained = trained[trained["family"] == "full"]
        for task, block in trained.groupby("task"):
            stream = block[block["site"].isin(RESIDUAL)]
            best_stream, best_any = stream.loc[stream["full"].idxmax()], block.loc[block["full"].idxmax()]
            rows.append({"variant": name, "task": task, "residual_best": best_stream["full"], "residual_best_std": best_stream["full_std"],
                         "any_best": best_any["full"], "any_best_std": best_any["full_std"], "mean_probe": block["full"].mean(),
                         "replicates": int(block["replicates"].max())})
    table = pd.DataFrame(rows)
    shared = set.intersection(*(set(table[table["variant"] == name]["task"]) for name in variants))
    means = table[table["task"].isin(shared)].groupby("variant", sort=False)[["residual_best", "any_best", "mean_probe"]].mean().reset_index()
    order = {task: position for position, task in enumerate(TASK_ORDER)}
    table = table.sort_values(["task", "variant"], key=lambda column: column.map(order) if column.name == "task" else column, kind="stable")
    return pd.concat([table, means.assign(task="mean over shared tasks")], ignore_index=True)


def variant_depth_curves(variants: dict[str, pd.DataFrame], blocks: int, path: Path, tasks: Sequence[str] = TASK_ORDER) -> Path:
    """The residual stream of every task along depth under each variant of the analysis, one color per variant."""
    plt = _pyplot()
    present = set.intersection(*(set(summary["task"]) for summary in variants.values()))
    tasks = _ordered(tasks, present)
    figure, axes = _grid(plt, len(tasks), 6)
    for axis, task in zip(axes, tasks):
        for (name, summary), color in zip(variants.items(), PALETTE):
            trained = headline(summary)
            line = trained[(trained["task"] == task) & (trained["family"] == "full") & trained["site"].isin(RESIDUAL)]
            _depth_line(axis, line, blocks, color, name)
        _tag(axis, SHORT_LABELS.get(task, task))
        axis.set_xlim(-0.5, blocks + 1.5)
        axis.set_xticks([0, blocks // 2, blocks])
        axis.grid(axis="y", color=GRID, lw=0.5)
        if axis.get_subplotspec().is_first_col():
            axis.set_ylabel("score")
        if _bottom(axis, axes, len(tasks)):
            axis.set_xlabel("depth")
    handles, labels = axes[0].get_legend_handles_labels()
    _legend_top(figure, handles, labels, len(labels), 0.3)
    return _save(figure, path)


def compare(variants: dict[str, Sequence[tuple[Path, str]]], out: Path, blocks: int) -> list[Path]:
    """Figures and a table comparing variants of the analysis, each a list of replicate (root, label) pairs; the
    first variant is the reference the others are drawn against."""
    out.mkdir(parents=True, exist_ok=True)
    summaries = {
        name: summarize(pd.concat([read_table(Path(root), label, "scores").assign(replicate=index) for index, (root, label) in enumerate(replicates)],
                                  ignore_index=True))
        for name, replicates in variants.items()
    }
    names = list(summaries)
    written = [write_summary(variant_table(summaries), out / "variants.csv"), variant_depth_curves(summaries, blocks, out / "variants_depth.png")]
    reference = names[0]
    for name in names[1:]:
        slug = "".join(character if character.isalnum() else "_" for character in name).strip("_")
        written.append(diff_maps(summaries[name], summaries[reference], out / f"variants_maps_{slug}.png", label=f"{name} minus {reference}"))
        written.append(diff_summary(summaries[name], summaries[reference], out / f"variants_summary_{slug}.png", names=(name, reference)))
    return written


# -- latent views, interventions, settings and spread ------------------------------------------------------------------


def pca_grid(frames: dict[str, pd.DataFrame], path: Path, top: int = 5) -> Path:
    """Two principal coordinates of pooled states per layer (columns), one row per model; proteins colored by label set
    (the most frequent sets of the first model fix the colors), scrambled copies in indigo, the rest gray."""
    plt = _pyplot()
    names = list(frames)
    first = frames[names[0]]
    layers = list(dict.fromkeys(first["layer"]))
    groups = [group for group in first["group"].value_counts().index if group not in ("other", "scrambled")][:top]
    shown = {group: group[:1].upper() + group[1:] for group in groups} | {"scrambled": "scrambled test sequences"}
    colors = {group: PALETTE[index] for index, group in enumerate(groups)}
    colors["scrambled"] = PALETTE[6]
    figure, axes = plt.subplots(len(names), len(layers), figsize=(1.05 * len(layers) + 0.4, 1.15 * len(names) + 0.6), squeeze=False)
    for row, name in enumerate(names):
        frame = frames[name]
        for column, layer in enumerate(layers):
            axis = axes[row, column]
            block = frame[frame["layer"] == layer]
            other = block[~block["group"].isin(colors)]
            axis.scatter(other["pc1"], other["pc2"], s=0.8, color=GRID, lw=0, rasterized=True)
            for group, color in colors.items():
                chosen = block[block["group"] == group]
                axis.scatter(chosen["pc1"], chosen["pc2"], s=1.4, color=color, lw=0, rasterized=True, label=group)
            axis.set_xticks([])
            axis.set_yticks([])
            if not block.empty:
                axis.set_xlabel(f"PC1 {100 * block['explained_1'].iloc[0]:.0f}%", fontsize=5, labelpad=1)
                axis.set_ylabel(f"PC2 {100 * block['explained_2'].iloc[0]:.0f}%", fontsize=5, labelpad=1)
            if row == 0:
                axis.text(0.5, 1.04, f"block {layer}", transform=axis.transAxes, ha="center", fontsize=6, color=INK)
            if column == 0:
                axis.text(-0.32, 0.5, name, transform=axis.transAxes, rotation=90, ha="center", va="center", fontsize=6, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.tight_layout(w_pad=0.2, h_pad=0.4, rect=(0, 0, 1, 1 - 0.55 / figure.get_figheight()))
    figure.legend(handles, [textwrap.fill(shown.get(label, label), 40) for label in labels], loc="upper center", ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, 1.0), markerscale=4, fontsize=5.5)
    return _save(figure, path)


def phospho_figure(tests: pd.DataFrame, blocks: int, path: Path) -> Path:
    """A, B: the residual stream's mutant-activity MCC by depth, averaged over proteins, on raw (A) and standardized (B)
    pooled means, against shuffled labels and the untrained model, with the 95% confidence interval of the mean over
    proteins shaded. C: the trained model's MCC on raw means at every site and block. D: the Bonferroni-corrected
    p-value of true against shuffled labels along the residual stream, per input treatment."""
    plt = _pyplot()
    treatments = [inputs for inputs in ("raw", "standardized") if inputs in set(tests["inputs"])]
    figure, axes = plt.subplots(1, len(treatments) + 2, figsize=(7.2, 2.2), gridspec_kw={"width_ratios": [1.0] * len(treatments) + [1.35, 0.8]})
    curves = (("trained_true", "trained model, true labels", PALETTE[0]),
              ("trained_shuffled", "trained model, shuffled labels", PALETTE[1]),
              ("untrained_true", "untrained model, true labels", MUTED))
    stream = _with_depth(tests[tests["site"].isin(("embedding", "resid_post"))], blocks).sort_values("depth")
    for axis, inputs, letter in zip(axes, treatments, "AB"):
        rows = stream[stream["inputs"] == inputs]
        for column, legend, color in curves:
            interval = 1.96 * rows[f"{column}_std"] / np.sqrt(rows["proteins"])
            axis.plot(rows["depth"], rows[column], color=color, label=legend, lw=1.2)
            axis.fill_between(rows["depth"], rows[column] - interval, rows[column] + interval, color=color, alpha=0.2, lw=0)
        axis.set_xlabel("depth")
        axis.grid(axis="y", color=GRID, lw=0.5)
        _letter(axis, letter)
    axes[0].set_ylabel("mean test MCC over proteins")
    handles, labels = axes[0].get_legend_handles_labels()
    middle = axes[len(treatments)]
    raw = tests[(tests["inputs"] == treatments[0]) & tests["site"].isin(BLOCK_SITES)]
    grid = raw.pivot_table(index="site", columns="layer", values="trained_true")
    grid = grid.reindex([site for site in BLOCK_SITES if site in grid.index])
    image = middle.imshow(grid.to_numpy(), aspect="auto", cmap=_colormap(BLUE_RAMP, "blue"), interpolation="nearest")
    middle.set_yticks(range(len(grid)))
    middle.set_yticklabels([SITE_LABELS[site] for site in grid.index], fontsize=5)
    middle.set_xlabel("block")
    _bare(middle)
    bar = figure.colorbar(image, ax=middle, shrink=0.85, pad=0.02)
    bar.set_label(f"mean test MCC, {treatments[0]}", fontsize=5.5)
    bar.ax.tick_params(labelsize=5)
    _letter(middle, "ABC"[len(treatments)])
    right = axes[-1]
    for inputs, style in zip(treatments, ("-", "--")):
        rows = stream[stream["inputs"] == inputs]
        right.plot(rows["depth"], -np.log10(rows["p_bonferroni"].clip(lower=1e-300)), color=PALETTE[0], ls=style, lw=1.1, label=inputs)
    right.axhline(-np.log10(0.05), color=INK_SECONDARY, ls=":", lw=0.7)
    right.set_xlabel("depth")
    right.set_ylabel("-log10 corrected p", fontsize=6)
    right.grid(axis="y", color=GRID, lw=0.5)
    right.legend(frameon=False, fontsize=5.5, loc="center")
    _letter(right, "ABCD"[len(treatments) + 1])
    figure.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.6)
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
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


def spread_dots(spreads: dict[str, pd.DataFrame], path: Path) -> Path:
    """Per task, the median standard deviation of a probe's score over replicates and over test halves, per model."""
    plt = _pyplot()
    first = next(iter(spreads.values()))
    tasks = list(first["task"])
    figure, axis = plt.subplots(figsize=(3.8, 0.2 * len(tasks) + 0.9))
    positions = np.arange(len(tasks))[::-1]
    for index, (name, table) in enumerate(spreads.items()):
        table = table.set_index("task").reindex(tasks)
        color = (PALETTE[0], INK_SECONDARY)[index % 2]
        axis.scatter(table["replicate_std"], positions, s=16, color=color, marker="o", label=f"{name}: over replicates", zorder=3)
        axis.scatter(table["halves_std"], positions, s=16, facecolors="none", edgecolors=color, marker="o", linewidths=0.9,
                     label=f"{name}: over test halves", zorder=3)
    axis.set_yticks(positions)
    axis.set_yticklabels([TASK_LABELS.get(task, task) for task in tasks])
    axis.set_xscale("log")
    axis.set_xlabel("median standard deviation of a probe's score")
    axis.grid(axis="x", color=GRID, lw=0.5)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=5.5)
    figure.tight_layout()
    return _save(figure, path)


# -- everything at once ------------------------------------------------------------------------------------------------


def average_phospho(scores: pd.DataFrame) -> pd.DataFrame:
    """Each protein's MCC per condition, input treatment, site and layer, averaged over replicates."""
    keys = ["protein", "plm", "inputs", "labels", "site", "layer"]
    return scores.groupby(keys, as_index=False)["mcc"].mean()


def phospho_replicates(scores: pd.DataFrame) -> pd.DataFrame:
    """Per input treatment, condition, site and layer: each replicate's mean MCC over proteins, then the mean and the
    standard deviation of those over replicates."""
    keys = ["inputs", "plm", "labels", "site", "layer"]
    means = scores.groupby(keys + ["replicate"])["mcc"].mean()
    table = means.groupby(keys).agg(mcc="mean", replicate_std="std", replicates="count").reset_index()
    table["replicate_std"] = table["replicate_std"].fillna(0.0)
    return table


def result_figures(
    roots: Sequence[Path],
    out: Path,
    blocks: int,
    label: str = "linear",
    random_roots: Sequence[Path] = (),
    metadata: dict[str, dict] | None = None,
) -> list[Path]:
    """The figures and tables of the probes' scores, over the replicates of the trained model and, with
    `random_roots`, against those of a randomly initialized model."""
    written: list[Path] = []
    scores = read_replicates(roots, label)
    summary = summarize(scores)
    summary.to_parquet(out / "summary.parquet", index=False)
    written.append(out / "summary.parquet")
    random = summarize(read_replicates(random_roots, label)) if random_roots else None
    if random is not None:
        random.to_parquet(out / "summary_random_init.parquet", index=False)
        written.append(out / "summary_random_init.parquet")
    table = best_states(summary, blocks)
    written.append(write_summary(table, out / "best_states.csv"))
    written.append(write_summary(site_bests(summary), out / "site_bests.csv"))
    spreads = {"trained": replicate_spread(summary)}
    written.append(depth_curves(summary, blocks, out / "depth_curves.png", random))
    written.append(selected_tasks(summary, full_scores(scores), blocks, out / "selected_tasks.png"))
    if set(summary["control"]) - {"original"}:
        written.append(control_grid(summary, blocks, out / "controls.png"))
    written.append(site_maps(summary, out / "site_maps.png"))
    written.append(site_gap_map(summary, out / "site_gaps.png"))
    written.append(site_family_curves(summary, blocks, out / "site_families.png"))
    if (summary["family"] == "head").any():
        written.append(head_maps(summary, out / "heads.png"))
    written.append(rotation_effect(summary, out / "rotation.png"))
    if random is not None:
        written.append(best_state_dots(table, out / "best_states.png", best_states(random, blocks)))
        written.append(diff_maps(summary, random, out / "random_init_maps.png"))
        written.append(diff_summary(summary, random, out / "random_init_summary.png"))
        written.append(write_summary(random_init_table(summary, random, blocks), out / "random_init.csv"))
        spreads["random-init"] = replicate_spread(random)
    else:
        written.append(best_state_dots(table, out / "best_states.png"))
    written.append(spread_dots(spreads, out / "replicate_spread.png"))
    written.append(write_summary(pd.concat([frame.assign(model=name) for name, frame in spreads.items()]), out / "replicate_spread.csv"))
    if metadata and "prot_param" in metadata:
        elementwise = [Path(root) / "results" / label / "prot_param" / "elementwise.parquet" for root in roots]
        tables = [pd.read_parquet(path) for path in elementwise if path.exists()]
        if tables:
            written.append(prot_param_curves(pd.concat(tables), metadata["prot_param"]["label_names"], blocks, out / "prot_param_targets.png"))
    return written


def draw(
    roots: Sequence[Path],
    out: Path,
    blocks: int,
    label: str = "linear",
    random_roots: Sequence[Path] = (),
    phospho: Sequence[Path] = (),
    latent: dict[str, Path] | None = None,
    rankings: dict[str, pd.DataFrame] | None = None,
    metadata: dict[str, dict] | None = None,
) -> list[Path]:
    """Every figure and table the runs support, under `out`. `roots` are the replicates of the trained model's analysis,
    `random_roots` those of a randomly initialized model, `phospho` the folders of the phosphomimetic replicates (each
    protein's scores are averaged over them before the paired tests), `latent` maps a model's name to a folder of
    principal coordinate tables, `rankings` maps a regime to its ranked sweep settings, and `metadata` maps a task to
    its task record (the label names of ProtParam's targets)."""
    from .interventions import paired_tests

    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if (Path(roots[0]) / "results" / label).exists():
        written += result_figures(roots, out, blocks, label, random_roots, metadata)
    if phospho:
        combined = pd.concat([pd.read_parquet(Path(folder) / "scores.parquet").assign(replicate=index) for index, folder in enumerate(phospho)])
        tests = paired_tests(average_phospho(combined))
        tests.to_parquet(out / "phospho_tests.parquet", index=False)
        written += [out / "phospho_tests.parquet", write_summary(phospho_replicates(combined), out / "phospho_replicates.csv")]
        written.append(phospho_figure(tests, blocks, out / "phospho.png"))
    for task in sorted({path.stem for folder in (latent or {}).values() for path in Path(folder).glob("*.parquet")}):
        frames = {name: pd.read_parquet(Path(folder) / f"{task}.parquet") for name, folder in latent.items() if (Path(folder) / f"{task}.parquet").exists()}
        for site in sorted(set(next(iter(frames.values()))["site"])):
            written.append(pca_grid({name: frame[frame["site"] == site] for name, frame in frames.items()}, out / f"pca_{task}_{site}.png"))
    for regime, ranked in (rankings or {}).items():
        written.append(sweep_curves({regime: ranked}, out / f"sweeps_{regime}.png"))
    return written
