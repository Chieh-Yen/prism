#!/usr/bin/env python3
"""
4×5 grid: rows=models, cols=benchmarks.
Each subplot: scatter of delta_I vs |MdR|, colored by quantization family.
Outputs a single PDF.
"""

import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

# ── Config ──────────────────────────────────────────────────────────
CSV_PATH = Path("quantization_merged_slim.csv")
OUT_PDF = Path("prism_grid_4x5.pdf")

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

XCOL = "Bound_I"
YCOL = "|MdR|"


# ── Parse Label → unified method name ──────────────────────────────
def parse_method(label: str) -> str:
    rhs = label.replace("BF16 vs ", "").strip()
    if rhs == "FP16":
        return "FP16"
    for g in ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]:
        if rhs == g:
            return g
    if rhs == "INT8":
        return "INT8"
    if rhs == "NF4":
        return "NF4"
    if rhs == "FP4":
        return "FP4"
    m = re.match(r"GPTQ\((.+)\)", rhs)
    if m:
        inner = m.group(1).lower()
        if "int8" in inner or "8bit" in inner:
            return "GPTQ-8bit"
        return "GPTQ-4bit"
    if "AWQ" in rhs:
        return "AWQ"
    return rhs


# ── Visual style per method ────────────────────────────────────────
import seaborn as sns

_blues = sns.color_palette("Blues_r", 8)
GGUF_STYLE = {
    "Q8_0":   {"color": _blues[0], "marker": "o"},
    "Q6_K":   {"color": _blues[1], "marker": "o"},
    "Q5_K_M": {"color": _blues[2], "marker": "o"},
    "Q4_K_M": {"color": _blues[3], "marker": "o"},
    "Q3_K_M": {"color": _blues[4], "marker": "o"},
    "Q2_K":   {"color": _blues[5], "marker": "o"},
}

BNB_COLOR = "#d62728"
BNB_STYLE = {
    "INT8": {"color": BNB_COLOR, "marker": "s"},
    "NF4":  {"color": BNB_COLOR, "marker": "^"},
    "FP4":  {"color": BNB_COLOR, "marker": "v"},
}

GPTQ_COLOR = "#2ca02c"
GPTQ_STYLE = {
    "GPTQ-4bit": {"color": GPTQ_COLOR, "marker": "D"},
    "GPTQ-8bit": {"color": "#1b7a1b", "marker": "D"},
}

FP16_STYLE = {
    "FP16": {"color": "#888888", "marker": "o"},  # neutral circle, not X
}

ALL_STYLE = {**GGUF_STYLE, **BNB_STYLE, **GPTQ_STYLE, **FP16_STYLE}

LEGEND_ORDER = [
    "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K",
    "INT8", "NF4", "FP4",
    "GPTQ-4bit", "GPTQ-8bit",
    "FP16",
]


# ── Correlation helpers ────────────────────────────────────────────
def rankdata(values):
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
    rx, ry = rankdata(x), rankdata(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


# ── Load and parse ─────────────────────────────────────────────────
def load():
    numcols = [XCOL, YCOL]
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        item = dict(r)
        for k in numcols:
            try:
                item[k] = float(r[k])
            except:
                item[k] = float("nan")
        item["_method"] = parse_method(r["Label"])
        out.append(item)
    return out


# ── Main plotting ──────────────────────────────────────────────────
def main():
    rows = load()

    nrow, ncol = len(ROW_MODELS), len(COL_DATASETS)

    # (5) Compress vertical: shorter per-row height
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(ncol * 3.0, nrow * 2.5 + 1.0),
        squeeze=False,
    )
    plt.subplots_adjust(
        hspace=0.18, wspace=0.28,
        top=0.90, bottom=0.08, left=0.08, right=0.97,
    )

    # ── Fixed axis limits (data-driven, shared across all subplots) ──
    XLIM = (2e-2, 3e3)
    YLIM = (3e-6, 3e0)

    seen_methods = set()

    for ri, model in enumerate(ROW_MODELS):
        for ci, ds in enumerate(COL_DATASETS):
            ax = axes[ri][ci]

            sub = [r for r in rows
                   if r["target_model"] == model
                   and r["dataset"] == ds
                   and not math.isnan(r[XCOL])
                   and not math.isnan(r[YCOL])]

            # (2) Draw y=x line + safe zone FIRST (behind data)
            ax.set_xscale("log")
            ax.set_yscale("log")
            diag = [YLIM[0], XLIM[1]]  # line from bottom to right edge
            ax.plot(
                diag, diag,
                color="#d62728", ls="--", lw=1.3, alpha=0.55,
                zorder=1, clip_on=True,
            )
            ax.fill_between(
                diag, diag, YLIM[0],
                color="#2ca02c", alpha=0.05, zorder=0, clip_on=True,
            )

            # Scatter data points
            xs, ys = [], []
            for pt in sub:
                method = pt["_method"]
                seen_methods.add(method)
                style = ALL_STYLE.get(method, {"color": "gray", "marker": "+"})
                ax.scatter(
                    pt[XCOL], pt[YCOL],
                    color=style["color"], marker=style["marker"],
                    s=70, alpha=0.9, edgecolors="k", linewidth=0.5,
                    zorder=3,
                )
                xs.append(pt[XCOL])
                ys.append(pt[YCOL])

            # Fix axis limits (uniform across all subplots)
            ax.set_xlim(XLIM)
            ax.set_ylim(YLIM)

            # (1) Spearman annotation → top-left
            if len(xs) >= 3:
                rho = spearman(xs, ys)
                ax.text(
                    0.04, 0.96, f"$\\rho$={rho:.2f}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=8.5, fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              alpha=0.85, ec="0.7", lw=0.5),
                )

            ax.tick_params(labelsize=7)
            ax.grid(True, which="major", ls=":", alpha=0.35)

            # Column title (top row only)
            if ri == 0:
                ax.set_title(COL_DISPLAY[ds], fontsize=11, fontweight="bold",
                             pad=4)

            # Row label (left column only)
            if ci == 0:
                ax.set_ylabel(
                    ROW_DISPLAY[model] + "\n$|\\Delta\\mathcal{R}|$",
                    fontsize=8.5, fontweight="bold", labelpad=2,
                )
            else:
                ax.set_ylabel("")

            # (4) X label: bottom row only, formal name
            if ri == nrow - 1:
                ax.set_xlabel(
                    "PRISM Bound  $\\mathcal{B}_I$",
                    fontsize=8.5, labelpad=2,
                )

    # ── Build legend ───────────────────────────────────────────────
    legend_entries = []
    for method in LEGEND_ORDER:
        if method not in seen_methods:
            continue
        style = ALL_STYLE[method]
        handle = Line2D(
            [0], [0],
            marker=style["marker"], color="w",
            markerfacecolor=style["color"], markeredgecolor="k",
            markeredgewidth=0.5, markersize=7, linestyle="None",
        )
        legend_entries.append((handle, method))

    # Add y=x bound line to legend
    bound_handle = Line2D([0], [0], color="#d62728", ls="--", lw=1.5, alpha=0.6)
    legend_entries.append((bound_handle, "$|\\Delta\\mathcal{R}|=\\mathcal{B}_I$"))

    # Add safe zone to legend
    safe_handle = Patch(facecolor="#2ca02c", alpha=0.15, edgecolor="none")
    legend_entries.append((safe_handle, "Safe zone"))

    if legend_entries:
        handles, labels = zip(*legend_entries)
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.52, 0.99),
            ncol=min(len(labels), 14),
            fontsize=8,
            frameon=True, fancybox=True,
            handletextpad=0.2, columnspacing=0.8,
            borderpad=0.4,
        )

    fig.savefig(str(OUT_PDF), format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved → {OUT_PDF}")


if __name__ == "__main__":
    main()
