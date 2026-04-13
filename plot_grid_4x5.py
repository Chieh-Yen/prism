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
import matplotlib.ticker as ticker
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

XCOL = "delta_I"
YCOL = "|MdR|"


# ── Parse Label → unified method name ──────────────────────────────
def parse_method(label: str) -> str:
    """Map raw Label to a canonical quantization method name."""
    # Strip the 'BF16 vs ' prefix
    rhs = label.replace("BF16 vs ", "").strip()

    # FP16 baseline
    if rhs == "FP16":
        return "FP16"

    # GGUF variants
    gguf_names = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
    for g in gguf_names:
        if rhs == g:
            return g

    # BitsAndBytes
    if rhs == "INT8":
        return "INT8"
    if rhs == "NF4":
        return "NF4"
    if rhs == "FP4":
        return "FP4"

    # GPTQ — extract bit width from the parenthesized name
    m = re.match(r"GPTQ\((.+)\)", rhs)
    if m:
        inner = m.group(1).lower()
        if "int8" in inner or "8bit" in inner:
            return "GPTQ-8bit"
        # default: 4-bit GPTQ (all our GPTQ variants are 4-bit unless stated)
        return "GPTQ-4bit"

    # AWQ
    if "AWQ" in rhs:
        return "AWQ"

    return rhs  # fallback


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
    "FP16": {"color": "#999999", "marker": "X"},
}

ALL_STYLE = {**GGUF_STYLE, **BNB_STYLE, **GPTQ_STYLE, **FP16_STYLE}

# Legend ordering
LEGEND_ORDER = [
    "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K",
    "INT8", "NF4", "FP4",
    "GPTQ-4bit", "GPTQ-8bit",
    "FP16",
]

FAMILY_LABELS = {
    "Q8_0": "GGUF", "Q6_K": "", "Q5_K_M": "", "Q4_K_M": "", "Q3_K_M": "", "Q2_K": "",
    "INT8": "BnB", "NF4": "", "FP4": "",
    "GPTQ-4bit": "GPTQ", "GPTQ-8bit": "",
    "FP16": "Other",
}


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
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(ncol * 3.2, nrow * 3.0 + 1.6),
        squeeze=False,
    )
    plt.subplots_adjust(
        hspace=0.35, wspace=0.30,
        top=0.88, bottom=0.06, left=0.07, right=0.97,
    )

    seen_methods = set()

    for ri, model in enumerate(ROW_MODELS):
        for ci, ds in enumerate(COL_DATASETS):
            ax = axes[ri][ci]

            # Filter data
            sub = [r for r in rows
                   if r["target_model"] == model
                   and r["dataset"] == ds
                   and not math.isnan(r[XCOL])
                   and not math.isnan(r[YCOL])]

            xs, ys = [], []
            for pt in sub:
                method = pt["_method"]
                seen_methods.add(method)
                style = ALL_STYLE.get(method, {"color": "gray", "marker": "+"})
                ax.scatter(
                    pt[XCOL], pt[YCOL],
                    color=style["color"], marker=style["marker"],
                    s=80, alpha=0.9, edgecolors="k", linewidth=0.6,
                    zorder=3,
                )
                xs.append(pt[XCOL])
                ys.append(pt[YCOL])

            # Spearman annotation
            if len(xs) >= 3:
                rho = spearman(xs, ys)
                ax.text(
                    0.97, 0.05, f"$\\rho$={rho:.2f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="0.7"),
                )

            # Axis formatting
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.tick_params(labelsize=8)
            ax.grid(True, which="major", ls=":", alpha=0.4)

            # Column title (top row only)
            if ri == 0:
                ax.set_title(COL_DISPLAY[ds], fontsize=12, fontweight="bold")

            # Row label (left column only)
            if ci == 0:
                ax.set_ylabel(
                    ROW_DISPLAY[model] + "\n$|\\Delta\\mathcal{R}|$",
                    fontsize=9, fontweight="bold",
                )
            else:
                ax.set_ylabel("")

            # X label (bottom row only)
            if ri == nrow - 1:
                ax.set_xlabel("$\\delta_I$", fontsize=10)

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
            markeredgewidth=0.6, markersize=8, linestyle="None",
        )
        legend_entries.append((handle, method))

    if legend_entries:
        handles, labels = zip(*legend_entries)
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.52, 0.97),
            ncol=min(len(labels), 12),
            fontsize=9,
            frameon=True, fancybox=True,
            handletextpad=0.3, columnspacing=1.0,
            borderpad=0.5,
        )

    fig.savefig(str(OUT_PDF), format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved → {OUT_PDF}")

    # ── Print summary table for verification ───────────────────────
    print(f"\n{'Model':<25} {'Dataset':<10} {'n':>3}  methods present")
    print("-" * 80)
    for model in ROW_MODELS:
        for ds in COL_DATASETS:
            sub = [r for r in rows
                   if r["target_model"] == model and r["dataset"] == ds
                   and not math.isnan(r[XCOL]) and not math.isnan(r[YCOL])]
            methods = sorted(set(r["_method"] for r in sub))
            name = ROW_DISPLAY[model]
            print(f"  {name:<23} {ds:<10} {len(sub):>3}  {', '.join(methods)}")


if __name__ == "__main__":
    main()
