#!/usr/bin/env python3
"""
Ablation AUROC figures organised by scientific question.

Styles (``--style``):
  dot_2x2      — 2×2 task grid, horizontal dots, sectioned y, pair connectors
  dot_1x4      — 1×4 wide; matched tEHR/code share a row (no connectors)
  matrix       — rows = configs, cols = tasks, one marker per cell
  bar_2x2      — same organisation as dots, horizontal bars
  dot_two_task — ICU mortality + ICU readmission only (1×2)
  all          — write every style

Example:
  python -m src.analysis.plot_ablation_auroc --style all
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "analysis_out" / "ablations" / "results.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "analysis_out" / "ablations"

AUROC_LIM = (0.55, 0.95)

# Semantic encodings — colour = family only
COLOR_TEHR = "#4C78A8"
COLOR_CODE = "#E45756"
COLOR_SURFACE = "#717171"
COLOR_CONNECTOR = "#9A9A9A"
COLOR_SECTION = "#333333"

TASKS_ALL: List[Tuple[str, str]] = [
    ("ICU mortality AUROC", "ICU mortality"),
    ("ICU readmission AUROC", "ICU readmission"),
    ("Hospital mortality AUROC", "Hospital mortality"),
    ("Hospital readmission AUROC", "Hospital readmission"),
]
TASKS_ICU = TASKS_ALL[:2]

STYLES = ("dot_2x2", "dot_1x4", "matrix", "bar_2x2", "dot_two_task")

MARKER_SIZE = 8.0


@dataclass(frozen=True)
class Entry:
    section: str
    label: str
    csv_key: str
    kind: str  # tehr | code | surface
    size: str = "normal"  # normal | small (legacy; ignored when uniform=True)
    pair_id: Optional[str] = None


@dataclass(frozen=True)
class RowGroup:
    """One y-tick row; may hold both tEHR and code markers."""

    section: str
    label: str
    members: Tuple[Entry, ...]


# Scientific organisation (order within each section is intentional).
# Matched pairs are adjacent so connectors stay nearly vertical.
ENTRIES: List[Entry] = [
    # Adaptation
    Entry("Adaptation", "Full", "tEHR-Mistral-7B", "tehr", pair_id="full"),
    Entry("Adaptation", "Code full", "TE", "code", pair_id="full"),
    Entry("Adaptation", "Supervised only", "tEHR: supervised only", "tehr", pair_id="sup"),
    Entry("Adaptation", "Code supervised", "TE: supervised only", "code", pair_id="sup"),
    Entry("Adaptation", "CPT + frozen", "tEHR: pretrained+frozen", "tehr", pair_id="frozen"),
    Entry("Adaptation", "Code MLM + frozen", "TE: pretrained+frozen", "code", pair_id="frozen"),
    Entry("Adaptation", "Base + frozen", "tEHR: no pretrain+frozen", "tehr"),
    # Semantics
    Entry("Semantics", "Original text", "tEHR-Mistral-7B", "tehr"),
    Entry("Semantics", "Raw codes", "surface hash", "surface"),
    Entry("Semantics", "Scrambled", "surface scramble", "surface"),
    # Model
    Entry("Model", "tEHR-Mistral-7B", "tEHR-Mistral-7B", "tehr"),
    Entry("Model", "tEHR-Qwen-8B", "tEHR-Qwen-8B", "tehr"),
    Entry("Model", "tEHR-Qwen-0.6B", "tEHR-Qwen-0.6B", "tehr", size="small"),
]

# Compact rows for dot_1x4: matched tEHR / code share one y line.
PAIRED_ROWS: List[RowGroup] = [
    RowGroup(
        "Adaptation",
        "Full",
        (
            Entry("Adaptation", "Full", "tEHR-Mistral-7B", "tehr"),
            Entry("Adaptation", "Code full", "TE", "code"),
        ),
    ),
    RowGroup(
        "Adaptation",
        "Supervised only",
        (
            Entry("Adaptation", "Supervised only", "tEHR: supervised only", "tehr"),
            Entry("Adaptation", "Code supervised", "TE: supervised only", "code"),
        ),
    ),
    RowGroup(
        "Adaptation",
        "CPT + frozen",
        (
            Entry("Adaptation", "CPT + frozen", "tEHR: pretrained+frozen", "tehr"),
            Entry("Adaptation", "Code MLM + frozen", "TE: pretrained+frozen", "code"),
        ),
    ),
    RowGroup(
        "Adaptation",
        "Base + frozen",
        (Entry("Adaptation", "Base + frozen", "tEHR: no pretrain+frozen", "tehr"),),
    ),
    RowGroup(
        "Semantics",
        "Original text",
        (Entry("Semantics", "Original text", "tEHR-Mistral-7B", "tehr"),),
    ),
    RowGroup(
        "Semantics",
        "Hash codes",
        (Entry("Semantics", "Hash codes", "surface hash", "surface"),),
    ),
    RowGroup(
        "Semantics",
        "Scrambled",
        (Entry("Semantics", "Scrambled", "surface scramble", "surface"),),
    ),
    RowGroup(
        "Model",
        "tEHR-Mistral-7B",
        (Entry("Model", "tEHR-Mistral-7B", "tEHR-Mistral-7B", "tehr"),),
    ),
    RowGroup(
        "Model",
        "tEHR-Qwen-8B",
        (Entry("Model", "tEHR-Qwen-8B", "tEHR-Qwen-8B", "tehr"),),
    ),
    RowGroup(
        "Model",
        "tEHR-Qwen-0.6B",
        (Entry("Model", "tEHR-Qwen-0.6B", "tEHR-Qwen-0.6B", "tehr"),),
    ),
]

SECTION_ORDER = ("Adaptation", "Semantics", "Model")


def load_results(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """model -> {column: value}, missing cells omitted."""
    out: Dict[str, Dict[str, float]] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            model = (row.get("Models") or "").strip()
            if not model:
                continue
            vals: Dict[str, float] = {}
            for k, v in row.items():
                if k == "Models" or v is None:
                    continue
                s = str(v).strip()
                if not s:
                    continue
                vals[k] = float(s)
            out[model] = vals
    return out


def value_for(data: Dict[str, Dict[str, float]], entry: Entry, col: str) -> Optional[float]:
    row = data.get(entry.csv_key)
    if not row:
        return None
    return row.get(col)


def marker_kwargs(entry: Entry, *, uniform: bool = False) -> dict:
    if uniform:
        ms = MARKER_SIZE
    else:
        ms = MARKER_SIZE if entry.size == "normal" else MARKER_SIZE - 2.5
    if entry.kind == "tehr":
        return dict(
            marker="o",
            markersize=ms,
            color=COLOR_TEHR,
            markerfacecolor=COLOR_TEHR,
            markeredgecolor="black",
            markeredgewidth=0.8,
            linestyle="None",
        )
    if entry.kind == "code":
        return dict(
            marker="o",
            markersize=ms,
            color=COLOR_CODE,
            markerfacecolor="white",
            markeredgecolor=COLOR_CODE,
            markeredgewidth=1.4,
            linestyle="None",
        )
    # surface / semantic perturbation
    return dict(
        marker="^",
        markersize=ms,
        color=COLOR_SURFACE,
        markerfacecolor=COLOR_SURFACE,
        markeredgecolor="black",
        markeredgewidth=0.8,
        linestyle="None",
    )


def bar_color(entry: Entry) -> str:
    if entry.kind == "tehr":
        return COLOR_TEHR
    if entry.kind == "code":
        return COLOR_CODE
    return COLOR_SURFACE


def build_y_layout(
    entries: Sequence[Entry] = ENTRIES,
    *,
    section_gap: float = 0.9,
    row_gap: float = 1.0,
) -> Tuple[List[dict], float]:
    """
    Top-to-bottom layout: Adaptation → Semantics → Model.
    Returns (rows, y_max) with rows
      {kind: 'header'|'data', y, section?, entry?}
    """
    by_section: Dict[str, List[Entry]] = {s: [] for s in SECTION_ORDER}
    for e in entries:
        by_section[e.section].append(e)

    placed: List[dict] = []
    y = 0.0
    first = True
    for section in SECTION_ORDER:
        ents = by_section.get(section) or []
        if not ents:
            continue
        if not first:
            y += section_gap
        first = False
        placed.append({"kind": "header", "section": section, "y": y})
        y += row_gap
        for e in ents:
            placed.append({"kind": "data", "entry": e, "section": section, "y": y})
            y += row_gap

    ymax = max(r["y"] for r in placed) if placed else 1.0
    for r in placed:
        r["y"] = ymax - r["y"]
    return placed, ymax


def build_y_layout_paired(
    rows: Sequence[RowGroup] = PAIRED_ROWS,
    *,
    section_gap: float = 0.9,
    row_gap: float = 1.15,
) -> Tuple[List[dict], float]:
    """Like build_y_layout, but each data row is a RowGroup (possibly multi-marker)."""
    by_section: Dict[str, List[RowGroup]] = {s: [] for s in SECTION_ORDER}
    for rg in rows:
        by_section[rg.section].append(rg)

    placed: List[dict] = []
    y = 0.0
    first = True
    for section in SECTION_ORDER:
        groups = by_section.get(section) or []
        if not groups:
            continue
        if not first:
            y += section_gap
        first = False
        placed.append({"kind": "header", "section": section, "y": y})
        y += row_gap
        for rg in groups:
            placed.append({"kind": "data", "row": rg, "section": section, "y": y})
            y += row_gap

    ymax = max(r["y"] for r in placed) if placed else 1.0
    for r in placed:
        r["y"] = ymax - r["y"]
    return placed, ymax


def legend_handles(*, include_connector: bool = True) -> List:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_TEHR,
            markeredgecolor="black",
            markersize=8,
            label="tEHR",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=COLOR_CODE,
            markeredgewidth=1.5,
            markersize=8,
            label="Code Transformer",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=COLOR_SURFACE,
            markeredgecolor="black",
            markersize=8,
            label="Semantic perturbation",
        ),
    ]
    if include_connector:
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="-",
                color=COLOR_CONNECTOR,
                linewidth=1.0,
                label="Matched pair",
            )
        )
    return handles


def _style_ax(
    ax,
    layout: List[dict],
    *,
    xlabel: str = "AUROC",
    paired: bool = False,
    show_sections: bool = True,
) -> None:
    data_rows = [r for r in layout if r["kind"] == "data"]
    ys = [r["y"] for r in layout]
    ax.set_xlim(*AUROC_LIM)
    # Extra bottom pad when labels sit under markers
    pad = 0.85 if paired else 0.6
    ax.set_ylim(min(ys) - pad, max(ys) + 0.6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_yticks([r["y"] for r in data_rows])
    if paired:
        labels = [r["row"].label for r in data_rows]
    else:
        labels = [r["entry"].label for r in data_rows]
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.xaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not show_sections:
        return
    for row in layout:
        if row["kind"] != "header":
            continue
        ax.annotate(
            row["section"],
            xy=(AUROC_LIM[0], row["y"]),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=COLOR_SECTION,
            annotation_clip=False,
        )


def _plot_dots_panel(
    ax,
    data: Dict[str, Dict[str, float]],
    col: str,
    layout: List[dict],
    *,
    annotate: bool,
    show_ylabel_side: bool,
) -> None:
    # Pair connectors first (behind markers)
    by_pair: Dict[str, List[Tuple[float, float]]] = {}
    for row in layout:
        if row["kind"] != "data":
            continue
        e: Entry = row["entry"]
        v = value_for(data, e, col)
        if v is None or e.pair_id is None:
            continue
        by_pair.setdefault(e.pair_id, []).append((v, row["y"]))
    for pts in by_pair.values():
        if len(pts) != 2:
            continue
        (x0, y0), (x1, y1) = pts
        ax.plot([x0, x1], [y0, y1], color=COLOR_CONNECTOR, linewidth=1.0, zorder=1)

    for row in layout:
        if row["kind"] != "data":
            continue
        e: Entry = row["entry"]
        v = value_for(data, e, col)
        if v is None:
            continue
        ax.plot([v], [row["y"]], zorder=3, **marker_kwargs(e))
        # Commented out: no longer show numbers next to dots
        # if annotate:
        #     ax.text(
        #         v + 0.008,
        #         row["y"],
        #         f"{v:.3f}",
        #         va="center",
        #         ha="left",
        #         fontsize=6,
        #         color="#333333",
        #     )

    _style_ax(ax, layout, show_sections=show_ylabel_side)
    if not show_ylabel_side:
        ax.tick_params(axis="y", labelleft=False)


def _plot_dots_panel_paired(
    ax,
    data: Dict[str, Dict[str, float]],
    col: str,
    layout: List[dict],
    *,
    annotate: bool,
    show_ylabel_side: bool,
) -> None:
    """Paired markers on one y-row; values printed under each marker; no connectors."""
    for row in layout:
        if row["kind"] != "data":
            continue
        rg: RowGroup = row["row"]
        y = row["y"]
        plotted: List[Tuple[Entry, float]] = []
        for e in rg.members:
            v = value_for(data, e, col)
            if v is None:
                continue
            ax.plot([v], [y], zorder=3, **marker_kwargs(e, uniform=True))
            plotted.append((e, v))

        # Commented out: no longer show numbers below dots
        # if annotate and plotted:
        #     # Stagger labels when two markers share a row (avoids overlap for close AUROCs).
        #     for i, (_e, v) in enumerate(plotted):
        #         dy = 0.28 if len(plotted) == 1 else (0.28 if i == 0 else 0.52)
        #         ax.text(
        #             v,
        #             y - dy,
        #             f"{v:.3f}",
        #             va="top",
        #             ha="center",
        #             fontsize=5.5,
        #             color="#333333",
        #             zorder=4,
        #         )

    _style_ax(ax, layout, paired=True, show_sections=show_ylabel_side)
    if not show_ylabel_side:
        ax.tick_params(axis="y", labelleft=False)


def plot_dot_grid(
    data: Dict[str, Dict[str, float]],
    out_path: Path,
    *,
    tasks: Sequence[Tuple[str, str]],
    annotate: bool,
    ncols: int,
    figsize: Tuple[float, float],
    title: str,
) -> None:
    layout, _ = build_y_layout()
    n = len(tasks)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for i, (col, task_title) in enumerate(tasks):
        ax = axes_flat[i]
        left_col = (i % ncols) == 0
        _plot_dots_panel(ax, data, col, layout, annotate=annotate, show_ylabel_side=left_col)
        ax.set_title(task_title, fontsize=11)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(left=0.16 if ncols == 2 else 0.14)
    _save(fig, out_path)


def plot_dot_1x4(
    data: Dict[str, Dict[str, float]],
    out_path: Path,
    *,
    annotate: bool,
) -> None:
    layout, _ = build_y_layout_paired()
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.8), sharex=True, sharey=True)

    for i, (col, task_title) in enumerate(TASKS_ALL):
        ax = axes[i]
        _plot_dots_panel_paired(
            ax,
            data,
            col,
            layout,
            annotate=annotate,
            show_ylabel_side=(i == 0),
        )
        ax.set_title(task_title, fontsize=11)

    fig.legend(
        handles=legend_handles(include_connector=False),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.06),
    )
    # fig.suptitle("Ablations on MIMIC-IV", fontsize=12, y=1.04)
    fig.tight_layout()
    fig.subplots_adjust(left=0.14, wspace=0.12, bottom=0.14)
    _save(fig, out_path)


def plot_bar_2x2(
    data: Dict[str, Dict[str, float]],
    out_path: Path,
    *,
    annotate: bool,
) -> None:
    layout, _ = build_y_layout()
    data_rows = [r for r in layout if r["kind"] == "data"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for i, (col, task_title) in enumerate(TASKS_ALL):
        ax = axes_flat[i]
        for row in data_rows:
            e: Entry = row["entry"]
            v = value_for(data, e, col)
            if v is None:
                continue
            ax.barh(
                row["y"],
                v - AUROC_LIM[0],
                left=AUROC_LIM[0],
                height=0.65,
                color=bar_color(e),
                edgecolor="black",
                linewidth=0.8,
                alpha=0.9 if e.kind != "code" else 0.55,
            )
            # Commented out: no longer show numbers next to bars
            # if annotate:
            #     ax.text(v + 0.005, row["y"], f"{v:.3f}", va="center", fontsize=6)

        # Pair connectors at bar tips
        by_pair: Dict[str, List[Tuple[float, float]]] = {}
        for row in data_rows:
            e = row["entry"]
            v = value_for(data, e, col)
            if v is None or e.pair_id is None:
                continue
            by_pair.setdefault(e.pair_id, []).append((v, row["y"]))
        for pts in by_pair.values():
            if len(pts) != 2:
                continue
            (x0, y0), (x1, y1) = pts
            ax.plot([x0, x1], [y0, y1], color=COLOR_CONNECTOR, linewidth=1.0, zorder=2)

        _style_ax(ax, layout)
        if i % 2 == 1:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(task_title, fontsize=11)

    # Open-circle legend still applies conceptually; for bars use patches
    handles = [
        mpatches.Patch(facecolor=COLOR_TEHR, edgecolor="black", label="tEHR"),
        mpatches.Patch(facecolor=COLOR_CODE, edgecolor="black", alpha=0.55, label="Code Transformer"),
        mpatches.Patch(facecolor=COLOR_SURFACE, edgecolor="black", label="Semantic perturbation"),
        Line2D([0], [0], color=COLOR_CONNECTOR, linewidth=1.0, label="Matched pair"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Ablations by scientific question (bars)", fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, out_path)


def plot_matrix(
    data: Dict[str, Dict[str, float]],
    out_path: Path,
    *,
    annotate: bool,
) -> None:
    layout, _ = build_y_layout()
    data_rows = [r for r in layout if r["kind"] == "data"]
    n_tasks = len(TASKS_ALL)
    # One axes per task column, shared y
    fig_h = 0.42 * len(layout) + 1.8
    fig, axes = plt.subplots(
        1,
        n_tasks,
        figsize=(11.5, fig_h),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    for i, (col, task_title) in enumerate(TASKS_ALL):
        ax = axes[i]
        by_pair: Dict[str, List[Tuple[float, float]]] = {}
        for row in data_rows:
            e: Entry = row["entry"]
            v = value_for(data, e, col)
            if v is None:
                continue
            if e.pair_id is not None:
                by_pair.setdefault(e.pair_id, []).append((v, row["y"]))
            ax.plot([v], [row["y"]], zorder=3, **marker_kwargs(e))
            # Commented out: no longer show numbers next to dots
            # if annotate:
            #     ax.text(v + 0.006, row["y"], f"{v:.3f}", va="center", fontsize=5.5)

        for pts in by_pair.values():
            if len(pts) != 2:
                continue
            (x0, y0), (x1, y1) = pts
            ax.plot([x0, x1], [y0, y1], color=COLOR_CONNECTOR, linewidth=0.9, zorder=1)

        ax.set_xlim(*AUROC_LIM)
        ys = [r["y"] for r in layout]
        ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
        ax.set_title(task_title.replace(" ", "\n"), fontsize=9)
        ax.xaxis.grid(True, alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("AUROC", fontsize=8)
        if i == 0:
            ax.set_yticks([r["y"] for r in data_rows])
            ax.set_yticklabels([r["entry"].label for r in data_rows], fontsize=7.5)
            for row in layout:
                if row["kind"] != "header":
                    continue
                ax.annotate(
                    row["section"],
                    xy=(AUROC_LIM[0], row["y"]),
                    xytext=(-8, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=COLOR_SECTION,
                    annotation_clip=False,
                )
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle("Ablation matrix (configs × tasks)", fontsize=12, y=1.02)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.9, bottom=0.12)
    _save(fig, out_path)


def _save(fig, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.pdf')}")


def render(
    data: Dict[str, Dict[str, float]],
    out_dir: Path,
    style: str,
    *,
    annotate: bool,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if style == "dot_2x2":
        plot_dot_grid(
            data,
            out_dir / "auroc_dot_2x2.png",
            tasks=TASKS_ALL,
            annotate=annotate,
            ncols=2,
            figsize=(10.5, 8.5),
            title="Ablations by scientific question",
        )
    elif style == "dot_1x4":
        # Paired layout always prints values under markers (readable when two dots share a row).
        # plot_dot_1x4(data, out_dir / "auroc_dot_1x4.png", annotate=True)
        plot_dot_1x4(data, out_dir / "auroc_dot_1x4.png", annotate=False)
    elif style == "dot_two_task":
        plot_dot_grid(
            data,
            out_dir / "auroc_dot_icu.png",
            tasks=TASKS_ICU,
            annotate=annotate,
            ncols=2,
            figsize=(10.5, 5.2),
            title="Ablations by scientific question (ICU tasks)",
        )
    elif style == "bar_2x2":
        plot_bar_2x2(data, out_dir / "auroc_bar_2x2.png", annotate=annotate)
    elif style == "matrix":
        plot_matrix(data, out_dir / "auroc_matrix.png", annotate=annotate)
    else:
        raise ValueError(f"Unknown style: {style}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--style",
        default="all",
        choices=[*STYLES, "all"],
        help="Figure layout to render (default: all)",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Print AUROC values next to markers",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    data = load_results(args.csv)
    missing = sorted({e.csv_key for e in ENTRIES} - set(data))
    if missing:
        print(f"Warning: CSV missing models referenced by layout: {missing}")

    styles = list(STYLES) if args.style == "all" else [args.style]
    for style in styles:
        render(data, args.out_dir, style, annotate=args.annotate)


if __name__ == "__main__":
    main()
