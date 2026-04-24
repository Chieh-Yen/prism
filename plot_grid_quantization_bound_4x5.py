#!/usr/bin/env python3
"""
Grid scatter of x-metric vs |MdR|; rows = models, cols = benchmarks.
Default grid is 2×5 (llama + qwen over the 5 benchmarks); the model set
is selectable via --models.

Two modes:
  - "bound":   x = Bound_I, with y=x line + safe zone (theoretical validity)
  - "feature":  x = delta_I, no safe zone (predictive correlation)

Usage:
  python plot_grid_4x5.py                             # both modes × all corrs
  python plot_grid_4x5.py bound                       # Bound_I only
  python plot_grid_4x5.py feature                     # delta_I only
  python plot_grid_4x5.py --corr spearman             # r_s only
  python plot_grid_4x5.py --corr pearson              # r_p (raw) only
  python plot_grid_4x5.py --corr pearson_log          # r_p on log(x), log(y)
  python plot_grid_4x5.py --corr pearson_both         # r_p AND r_p^log stacked
  python plot_grid_4x5.py --corr spearman,pearson_log # pick a subset
  python plot_grid_4x5.py --models llama,qwen,mistral,deepseek  # 4-model grid

Default --corr:   spearman,pearson,pearson_log,pearson_both
Default --models: llama,qwen

Output files:
  spearman     → prism_grid_{bound|feature}.pdf
  pearson      → prism_grid_{bound|feature}_rp.pdf
  pearson_log  → prism_grid_{bound|feature}_rplog.pdf
  pearson_both → prism_grid_{bound|feature}_rpboth.pdf
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
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "exp_result" / "quantization" / "quantization_merged_slim.csv"
FIG_DIR = ROOT / "paper" / "figures" / "quantization"

# short_id -> (target_model path in CSV, display label)
MODEL_CATALOG = {
    "llama":    ("meta-llama/Meta-Llama-3.1-8B",             "Llama-3.1-8B"),
    "mistral":  ("mistralai/Ministral-3-8B-Base-2512",       "Ministral-3-8B"),
    "qwen":     ("Qwen/Qwen3-8B-Base",                       "Qwen3-8B-Base"),
    "deepseek": ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-8B"),
}
# Default selection when --models is not passed. Override at CLI.
DEFAULT_MODELS = ["llama", "qwen"]

# Populated from DEFAULT_MODELS here and overwritten in main() once the CLI
# is parsed; plot_grid() reads these module-level globals at call time.
ROW_MODELS = [MODEL_CATALOG[m][0] for m in DEFAULT_MODELS]
ROW_DISPLAY = {MODEL_CATALOG[m][0]: MODEL_CATALOG[m][1] for m in DEFAULT_MODELS}
# Short ids (in CLI order) — used for output filename suffix.
MODEL_IDS = list(DEFAULT_MODELS)
COL_DATASETS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]
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
        "outfile": "prism_grid_bound_{models}{suffix}.pdf",
    },
    "feature": {
        "xcol": "delta_I",
        "xlabel": "Feature Alignment Error  $\\delta$",
        "xlim": (2e-3, 5e2),
        "ylim": (3e-6, 3e0),
        "safe_zone": False,
        "outfile": "prism_grid_feature_{models}{suffix}.pdf",
    },
}

CORR_CONFIG = {
    "spearman":    {"labels": ["r_s"],         "methods": ["spearman"],
                    "suffix": ""},
    "pearson":     {"labels": ["r_p"],         "methods": ["pearson"],
                    "suffix": "_rp"},
    "pearson_log": {"labels": ["r_p^{\\log}"], "methods": ["pearson_log"],
                    "suffix": "_rplog"},
    "pearson_both":{"labels": ["r_p", "r_p^{\\log}"],
                    "methods": ["pearson", "pearson_log"],
                    "suffix": "_rpboth"},
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


def pearson(x, y):
    if len(x) < 3:
        return float("nan")
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x)
                    * sum((b - my) ** 2 for b in y))
    return num / den if den > 0 else float("nan")


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    return pearson(rx, ry)


def pearson_log(x, y):
    """Pearson on log-transformed values — slope-like summary for log-log
    scatter. Non-positive pairs are dropped defensively."""
    pairs = [(a, b) for a, b in zip(x, y) if a > 0 and b > 0]
    if len(pairs) < 3:
        return float("nan")
    lx = [math.log(a) for a, _ in pairs]
    ly = [math.log(b) for _, b in pairs]
    return pearson(lx, ly)


def compute_corr(method: str, x, y):
    if method == "spearman":
        return spearman(x, y)
    if method == "pearson":
        return pearson(x, y)
    if method == "pearson_log":
        return pearson_log(x, y)
    raise ValueError(f"Unknown correlation method: {method}")


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
def plot_grid(mode: str, corr_method: str = "spearman"):
    cfg = MODE_CONFIG[mode]
    corr_cfg = CORR_CONFIG[corr_method]
    xcol = cfg["xcol"]
    rows = load_data()

    nrow, ncol = len(ROW_MODELS), len(COL_DATASETS)
    # Reserve an absolute vertical band at the top for the legend + column
    # titles so small grids (e.g., 2 rows) don't have the legend collide
    # with the header text.
    legend_reserve_inch = 1.1
    fig_height = nrow * 2.8 + legend_reserve_inch
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(ncol * 3.6, fig_height),
        squeeze=False,
    )
    top_frac = (fig_height - legend_reserve_inch) / fig_height
    plt.subplots_adjust(
        hspace=0.22, wspace=0.24,
        top=top_frac, bottom=0.08, left=0.07, right=0.97,
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

            # ── Correlation (bottom-right) — one line per method ──
            if len(xs) >= 3:
                lines = []
                for lbl, meth in zip(corr_cfg["labels"], corr_cfg["methods"]):
                    rho = compute_corr(meth, xs, ys)
                    rho_str = (f"{rho:.2f}".lstrip("0") if rho >= 0
                               else f"{rho:.2f}")
                    lines.append(f"${lbl}$={rho_str}")
                ax.text(
                    0.96, 0.04, "\n".join(lines),
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=17, fontstyle="italic",
                    linespacing=1.15,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              alpha=0.85, ec="0.7", lw=0.5),
                )

            ax.tick_params(labelsize=12)
            ax.tick_params(axis="both", which="minor", length=0)
            ax.grid(True, which="major", ls=":", alpha=0.35)

            # ── Column title (top row) ────────────────────────────
            if ri == 0:
                ax.set_title(COL_DISPLAY[ds], fontsize=20,
                             fontweight="bold", pad=7)

            # ── Y label: model name + $|ΔR|$ on left col only ────
            if ci == 0:
                ax.set_ylabel(
                    ROW_DISPLAY[model] + "\n$|\\Delta\\mathcal{R}|$",
                    fontsize=16, fontweight="bold", labelpad=2,
                )

            # ── X label on all bottom row subplots ────────────────
            if ri == nrow - 1:
                ax.set_xlabel(cfg["xlabel"], fontsize=17, labelpad=2)

    # ── Legend ─────────────────────────────────────────────────────
    legend_entries = []
    for method in LEGEND_ORDER:
        if method not in seen_methods:
            continue
        style = ALL_STYLE[method]
        handle = Line2D(
            [0], [0], marker=style["marker"], color="w",
            markerfacecolor=style["color"], markeredgecolor="k",
            markeredgewidth=0.5, markersize=11, linestyle="None",
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
            ncol=min(len(labels), 14), fontsize=14,
            frameon=True, fancybox=True,
            handletextpad=0.3, columnspacing=1.0, borderpad=0.4,
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outpath = FIG_DIR / cfg["outfile"].format(
        suffix=corr_cfg["suffix"],
        models="_".join(MODEL_IDS),
    )
    fig.savefig(str(outpath), format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {outpath}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def main():
    args = sys.argv[1:]

    corr_methods = None
    model_ids = None
    filtered = []
    i = 0
    while i < len(args):
        if args[i] == "--corr" and i + 1 < len(args):
            corr_methods = [c.strip() for c in args[i + 1].split(",")]
            i += 2
        elif args[i] == "--models" and i + 1 < len(args):
            model_ids = [m.strip() for m in args[i + 1].split(",") if m.strip()]
            i += 2
        else:
            filtered.append(args[i])
            i += 1

    if not filtered:
        modes = ["bound", "feature"]
    else:
        modes = [a for a in filtered if a in MODE_CONFIG]
        if not modes:
            print(f"Usage: {sys.argv[0]} [bound|feature] "
                  f"[--corr spearman,pearson,pearson_log] "
                  f"[--models llama,qwen,...]")
            sys.exit(1)

    if corr_methods is None:
        corr_methods = ["spearman", "pearson", "pearson_log", "pearson_both"]
    for c in corr_methods:
        if c not in CORR_CONFIG:
            print(f"Unknown --corr value: {c} (expected: {list(CORR_CONFIG)})")
            sys.exit(1)

    if model_ids is None:
        model_ids = list(DEFAULT_MODELS)
    unknown = [m for m in model_ids if m not in MODEL_CATALOG]
    if unknown:
        print(f"Unknown --models value(s): {unknown} "
              f"(expected: {list(MODEL_CATALOG)})")
        sys.exit(1)
    # Mutate in-place so plot_grid() and helpers pick up the CLI-selected set.
    ROW_MODELS[:] = [MODEL_CATALOG[m][0] for m in model_ids]
    ROW_DISPLAY.clear()
    ROW_DISPLAY.update(
        {MODEL_CATALOG[m][0]: MODEL_CATALOG[m][1] for m in model_ids}
    )
    MODEL_IDS[:] = model_ids

    for m in modes:
        for c in corr_methods:
            plot_grid(m, corr_method=c)


if __name__ == "__main__":
    main()
