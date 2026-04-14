#!/usr/bin/env python3
"""
4×5 grid: rows=models, cols=benchmarks.
Each subplot: scatter of x-metric vs |MdR|, colored by quantization family.

Two modes:
  - "bound":   x = Bound_I, with y=x line + safe zone (theoretical validity)
  - "feature":  x = delta_I, no safe zone (predictive correlation)

Usage:
  python plot_grid_4x5.py              # both PDFs
  python plot_grid_4x5.py bound        # Bound_I only
  python plot_grid_4x5.py feature      # delta_I only
"""

import csv
import math
import re
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════
# Shared config
# ═══════════════════════════════════════════════════════════════════
CSV_PATH = Path("quantization_merged_slim.csv")

ROW_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B",
    "mistralai/Ministral-3-8B-Base-2512",
    "Qwen/Qwen3-8B-Base",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
]
COL_DATASETS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]

ROW_DISPLAY = {
    "meta-llama/Meta-Llama-3.1-8B": "Llama-3.1-8B",
    "mistralai/Ministral-3-8B-Base-2512": "Ministral-3-8B",
    "Qwen/Qwen3-8B-Base": "Qwen3-8B-Base",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-8B",
}
COL_DISPLAY = {
    "arc": "ARC", "mmlu": "MMLU", "squad": "SQuAD",
    "triviaqa": "TriviaQA", "gsm8k": "GSM8K",
}

YCOL = "|MdR|"

# ═══════════════════════════════════════════════════════════════════
# Per-mode config
# ═══════════════════════════════════════════════════════════════════
MODE_CONFIG = {
    "bound": {
        "xcol": "Bound_I",
        "xlabel": "PRISM Bound  $\\mathcal{B}$",
        "xlim": (2e-2, 3e3),
        "ylim": (3e-6, 3e0),
        "safe_zone": True,
        "outfile": "prism_grid_bound.pdf",
    },
    "feature": {
        "xcol": "delta_I",
        "xlabel": "Feature Alignment Error  $\\delta$",
        "xlim": (2e-3, 5e2),
        "ylim": (3e-6, 3e0),
        "safe_zone": False,
        "outfile": "prism_grid_feature.pdf",
    },
}


# ═══════════════════════════════════════════════════════════════════
# Label parsing
# ═══════════════════════════════════════════════════════════════════
def parse_method(label: str) -> str:
    rhs = label.replace("BF16 vs ", "").strip()
    if rhs == "FP16":
        return "FP16"
    for g in ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]:
        if rhs == g:
            return g
    if rhs in ("INT8", "NF4", "FP4"):
        return rhs
    m = re.match(r"GPTQ\((.+)\)", rhs)
    if m:
        inner = m.group(1).lower()
        if "int8" in inner or "8bit" in inner:
            return "GPTQ-8bit"
        return "GPTQ-4bit"
    if "AWQ" in rhs:
        return "AWQ"
    return rhs


# ═══════════════════════════════════════════════════════════════════
# Visual style
# ═══════════════════════════════════════════════════════════════════
_blues = sns.color_palette("Blues_r", 8)

ALL_STYLE = {
    "Q8_0":      {"color": _blues[0], "marker": "o"},
    "Q6_K":      {"color": _blues[1], "marker": "o"},
    "Q5_K_M":    {"color": _blues[2], "marker": "o"},
    "Q4_K_M":    {"color": _blues[3], "marker": "o"},
    "Q3_K_M":    {"color": _blues[4], "marker": "o"},
    "Q2_K":      {"color": _blues[5], "marker": "o"},
    "INT8":      {"color": "#d62728",  "marker": "s"},
    "NF4":       {"color": "#d62728",  "marker": "^"},
    "FP4":       {"color": "#d62728",  "marker": "v"},
    "GPTQ-4bit": {"color": "#2ca02c",  "marker": "D"},
    "GPTQ-8bit": {"color": "#1b7a1b",  "marker": "D"},
    "FP16":      {"color": "#888888",  "marker": "o"},
}

LEGEND_ORDER = [
    "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K",
    "INT8", "NF4", "FP4",
    "GPTQ-4bit", "GPTQ-8bit",
    "FP16",
]


# ═══════════════════════════════════════════════════════════════════
# Correlation
# ═══════════════════════════════════════════════════════════════════
def _rankdata(values):
    order = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and order[j][1] == order[i][1]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k][0]] = avg
        i = j
    return ranks


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════
_CACHE = None

def load_data():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    all_num = {"Bound_I", "delta_I", YCOL}
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        item = dict(r)
        for k in all_num:
            try:
                item[k] = float(r[k])
            except (TypeError, ValueError, KeyError):
                item[k] = float("nan")
        item["_method"] = parse_method(r["Label"])
        out.append(item)
    _CACHE = out
    return out


# ═══════════════════════════════════════════════════════════════════
# Core plotting function
# ═══════════════════════════════════════════════════════════════════
def plot_grid(mode: str):
    cfg = MODE_CONFIG[mode]
    xcol = cfg["xcol"]
    rows = load_data()

    nrow, ncol = len(ROW_MODELS), len(COL_DATASETS)
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(ncol * 3.6, nrow * 2.8 + 1.2),
        squeeze=False,
    )
    plt.subplots_adjust(
        hspace=0.22, wspace=0.24,
        top=0.91, bottom=0.08, left=0.07, right=0.97,
    )

    seen_methods = set()

    # ── First pass: collect per-cell data ranges ──────────────────
    cell_data = {}
    for ri, model in enumerate(ROW_MODELS):
        for ci, ds in enumerate(COL_DATASETS):
            sub = [r for r in rows
                   if r["target_model"] == model
                   and r["dataset"] == ds
                   and not math.isnan(r[xcol])
                   and not math.isnan(r[YCOL])]
            xs = [r[xcol] for r in sub if r[xcol] > 0]
            ys = [r[YCOL] for r in sub if r[YCOL] > 0]
            cell_data[(ri, ci)] = (sub, xs, ys)

    # ── Second pass: plot ─────────────────────────────────────────
    for ri, model in enumerate(ROW_MODELS):
        for ci, ds in enumerate(COL_DATASETS):
            ax = axes[ri][ci]
            ax.set_xscale("log")
            ax.set_yscale("log")

            sub, xs, ys = cell_data[(ri, ci)]

            # ── Per-cell axis limits (half-decade padding) ────────
            if xs and ys:
                x_lo_d = np.floor(np.log10(min(xs))) - 0.5
                x_hi_d = np.ceil(np.log10(max(xs))) + 0.5
                y_lo_d = np.floor(np.log10(min(ys))) - 0.5
                y_hi_d = np.ceil(np.log10(max(ys))) + 0.5
                xlim = (10**x_lo_d, 10**x_hi_d)
                ylim = (10**y_lo_d, 10**y_hi_d)
            else:
                xlim = cfg["xlim"]
                ylim = cfg["ylim"]

            # ── Safe zone + y=x line (bound mode only) ────────────
            if cfg["safe_zone"]:
                diag = [ylim[0], xlim[1]]
                ax.plot(diag, diag, color="#d62728", ls="--", lw=1.3,
                        alpha=0.55, zorder=1, clip_on=True)
                ax.fill_between(diag, diag, ylim[0], color="#2ca02c",
                                alpha=0.05, zorder=0, clip_on=True)

            # ── Scatter data ──────────────────────────────────────
            for pt in sub:
                method = pt["_method"]
                seen_methods.add(method)
                style = ALL_STYLE.get(method, {"color": "gray", "marker": "+"})
                ax.scatter(
                    pt[xcol], pt[YCOL],
                    color=style["color"], marker=style["marker"],
                    s=70, alpha=0.9, edgecolors="k", linewidth=0.5,
                    zorder=3,
                )

            # ── Set limits + ticks at every integer power ─────────
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            xt_lo = int(np.floor(np.log10(xlim[0])))
            xt_hi = int(np.ceil(np.log10(xlim[1])))
            ax.set_xticks([10**i for i in range(xt_lo, xt_hi + 1)])

            yt_lo = int(np.floor(np.log10(ylim[0])))
            yt_hi = int(np.ceil(np.log10(ylim[1])))
            ax.set_yticks([10**i for i in range(yt_lo, yt_hi + 1)])

            # ── Spearman r_s (bottom-right) ───────────────────────
            if len(xs) >= 3:
                rho = spearman(xs, ys)
                rho_str = f"{rho:.2f}".lstrip("0")
                ax.text(
                    0.96, 0.04, f"$r_s$={rho_str}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=13, fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              alpha=0.85, ec="0.7", lw=0.5),
                )

            ax.tick_params(labelsize=9)
            ax.tick_params(axis="both", which="minor", length=0)
            ax.grid(True, which="major", ls=":", alpha=0.35)

            # ── Column title (top row) ────────────────────────────
            if ri == 0:
                ax.set_title(COL_DISPLAY[ds], fontsize=16,
                             fontweight="bold", pad=5)

            # ── Y label: model name on left col, $|ΔR|$ on all ───
            if ci == 0:
                ax.set_ylabel(
                    ROW_DISPLAY[model] + "\n$|\\Delta\\mathcal{R}|$",
                    fontsize=12, fontweight="bold", labelpad=2,
                )
            else:
                ax.set_ylabel("$|\\Delta\\mathcal{R}|$",
                              fontsize=10, labelpad=2)

            # ── X label on all bottom row subplots ────────────────
            if ri == nrow - 1:
                ax.set_xlabel(cfg["xlabel"], fontsize=13, labelpad=2)

    # ── Legend ─────────────────────────────────────────────────────
    legend_entries = []
    for method in LEGEND_ORDER:
        if method not in seen_methods:
            continue
        style = ALL_STYLE[method]
        handle = Line2D(
            [0], [0], marker=style["marker"], color="w",
            markerfacecolor=style["color"], markeredgecolor="k",
            markeredgewidth=0.5, markersize=8, linestyle="None",
        )
        legend_entries.append((handle, method))

    if cfg["safe_zone"]:
        legend_entries.append((
            Line2D([0], [0], color="#d62728", ls="--", lw=1.5, alpha=0.6),
            "$|\\Delta\\mathcal{R}|=\\mathcal{B}$",
        ))
        legend_entries.append((
            Patch(facecolor="#2ca02c", alpha=0.15, edgecolor="none"),
            "Safe zone",
        ))

    if legend_entries:
        handles, labels = zip(*legend_entries)
        fig.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.52, 0.99),
            ncol=min(len(labels), 14), fontsize=11,
            frameon=True, fancybox=True,
            handletextpad=0.3, columnspacing=1.0, borderpad=0.4,
        )

    outpath = Path(cfg["outfile"])
    fig.savefig(str(outpath), format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {outpath}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def main():
    args = sys.argv[1:]
    if not args:
        modes = ["bound", "feature"]
    else:
        modes = [a for a in args if a in MODE_CONFIG]
        if not modes:
            print(f"Usage: {sys.argv[0]} [bound|feature]")
            sys.exit(1)

    for m in modes:
        plot_grid(m)


if __name__ == "__main__":
    main()
