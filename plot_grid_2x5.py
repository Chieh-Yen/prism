#!/usr/bin/env python3
"""
2×5 grid: rows=fine-tune datasets (truthfulqa, social_iqa),
          cols=eval benchmarks (ARC, MMLU, SQuAD, TriviaQA, GSM8K).
Each subplot: scatter of x-metric vs |ΔR|, colored by training step.

Two modes:
  - "bound": x = Bound_total (PRISM Bound B), with y=x line + safe zone
  - "omega": x = Ω_I, no safe zone

Usage:
  python plot_grid_2x5.py                             # both modes × {r_s, r_p}
  python plot_grid_2x5.py bound                       # Bound only
  python plot_grid_2x5.py omega                       # Omega only
  python plot_grid_2x5.py --model mistral             # switch model dir
  python plot_grid_2x5.py --corr spearman             # r_s only (no _rp file)
  python plot_grid_2x5.py --corr pearson              # r_p only
  python plot_grid_2x5.py --corr spearman,pearson     # both (default)

Output files:
  Spearman: forgetting_grid_{bound|omega}_{model}.pdf
  Pearson : forgetting_grid_{bound|omega}_{model}_rp.pdf
"""

import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# Shared config
# ═══════════════════════════════════════════════════════════════════
ROW_TASKS = ["truthfulqa", "social_iqa", "bbq"]
COL_BENCHMARKS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]

ROW_DISPLAY = {
    "truthfulqa": "FT: TruthfulQA",
    "social_iqa": "FT: Social IQA",
    "bbq": "FT: BBQ",
}
COL_DISPLAY = {
    "arc": "ARC", "mmlu": "MMLU", "squad": "SQuAD",
    "triviaqa": "TriviaQA", "gsm8k": "GSM8K",
}

# ═══════════════════════════════════════════════════════════════════
# Per-mode config
# ═══════════════════════════════════════════════════════════════════
MODE_CONFIG = {
    "bound": {
        "xcol": "bound_total",
        "xlabel": "PRISM Bound  $\\mathcal{B}$",
        "safe_zone": True,
        "log_x": True,
        "log_y": True,
        "outfile": "forgetting_grid_bound_{model}{suffix}.pdf",
    },
    "omega": {
        "xcol": "omega",
        "xlabel": "$1 - \\Omega_I$",
        "safe_zone": False,
        "log_x": True,
        "log_y": True,
        "transform_x": lambda v: 1 - v,
        "outfile": "forgetting_grid_omega_{model}{suffix}.pdf",
    },
}

CORR_CONFIG = {
    "spearman": {"label": "r_s", "suffix": ""},
    "pearson":  {"label": "r_p", "suffix": "_rp"},
}


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


def compute_corr(method: str, x, y):
    if method == "spearman":
        return spearman(x, y)
    if method == "pearson":
        return pearson(x, y)
    raise ValueError(f"Unknown correlation method: {method}")


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════
def load_json(model_dir: str, ft_task: str) -> dict:
    path = Path("forgetting_exp_log_safety") / model_dir / f"prism_forgetting_metrics_{ft_task}.json"
    with path.open() as f:
        return json.load(f)


def extract_points(data: dict, eval_task: str, xcol: str, max_steps: int = 0,
                   transform_x=None):
    """Return list of (step, x, y) tuples for an eval benchmark."""
    points = []
    for ckpt in data["checkpoints"]:
        step = ckpt["step"]
        if max_steps > 0 and step > max_steps:
            continue
        task_data = ckpt["tasks"].get(eval_task)
        if task_data is None:
            continue
        x_val = task_data.get(xcol)
        y_val = task_data.get("delta_risk")
        if x_val is None or y_val is None:
            continue
        if math.isnan(x_val) or math.isnan(y_val):
            continue
        if transform_x is not None:
            x_val = transform_x(x_val)
        points.append((step, x_val, y_val))
    return points


# ═══════════════════════════════════════════════════════════════════
# Core plotting function
# ═══════════════════════════════════════════════════════════════════
def plot_grid(mode: str, model_dir: str, max_steps: int = 500,
              corr_method: str = "spearman"):
    cfg = MODE_CONFIG[mode]
    corr_cfg = CORR_CONFIG[corr_method]
    xcol = cfg["xcol"]
    transform_x = cfg.get("transform_x")

    nrow, ncol = len(ROW_TASKS), len(COL_BENCHMARKS)
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(ncol * 3.6, nrow * 3.0 + 1.4),
        squeeze=False,
    )
    plt.subplots_adjust(
        hspace=0.28, wspace=0.28,
        top=0.88, bottom=0.10, left=0.08, right=0.95,
    )

    # Collect all steps across both tasks for a unified colormap
    all_steps = set()
    json_cache = {}
    for ft_task in ROW_TASKS:
        d = load_json(model_dir, ft_task)
        json_cache[ft_task] = d
        for ckpt in d["checkpoints"]:
            if max_steps > 0 and ckpt["step"] > max_steps:
                continue
            all_steps.add(ckpt["step"])
    all_steps = sorted(all_steps)
    step_min, step_max = all_steps[0], all_steps[-1]

    cmap = cm.cividis
    norm = mcolors.Normalize(vmin=step_min, vmax=step_max)
    MARKER_SIZE = 50

    for ri, ft_task in enumerate(ROW_TASKS):
        data = json_cache[ft_task]
        for ci, bench in enumerate(COL_BENCHMARKS):
            ax = axes[ri][ci]

            points = extract_points(data, bench, xcol, max_steps, transform_x)
            steps = [p[0] for p in points]
            xs = [p[1] for p in points]
            ys = [p[2] for p in points]

            # Axis scale
            if cfg["log_x"]:
                ax.set_xscale("log")
            if cfg["log_y"]:
                ax.set_yscale("log")

            # Per-cell axis limits
            xs_pos = [v for v in xs if v > 0]
            ys_pos = [v for v in ys if v > 0]

            if cfg["log_x"]:
                if xs_pos:
                    x_lo_d = np.floor(np.log10(min(xs_pos))) - 0.5
                    x_hi_d = np.ceil(np.log10(max(xs_pos))) + 0.5
                    xlim = (10**x_lo_d, 10**x_hi_d)
                else:
                    xlim = None
            elif xs:
                margin = (max(xs) - min(xs)) * 0.1 or 0.01
                xlim = (min(xs) - margin, max(xs) + margin)
            else:
                xlim = None

            if cfg["log_y"] and ys_pos:
                y_lo_d = np.floor(np.log10(min(ys_pos))) - 0.5
                y_hi_d = np.ceil(np.log10(max(ys_pos))) + 0.5
                ylim = (10**y_lo_d, 10**y_hi_d)
            elif ys:
                margin = (max(ys) - min(ys)) * 0.1 or 0.001
                ylim = (min(ys) - margin, max(ys) + margin)
            else:
                ylim = None

            # Safe zone + y=x line (bound mode only)
            if cfg["safe_zone"] and xlim and ylim:
                diag_lo = min(xlim[0], ylim[0])
                diag_hi = max(xlim[1], ylim[1])
                diag = [diag_lo, diag_hi]
                ax.plot(diag, diag, color="#d62728", ls="--", lw=1.3,
                        alpha=0.55, zorder=1, clip_on=True)
                ax.fill_between(diag, diag, ylim[0], color="#2ca02c",
                                alpha=0.05, zorder=0, clip_on=True)

            # Filter valid points for plotting
            plot_pts = [(s, x, y) for s, x, y in zip(steps, xs, ys)
                        if (not cfg["log_x"] or x > 0)
                        and (not cfg["log_y"] or y > 0)]
            plot_pts.sort(key=lambda p: p[0])  # sort by step

            # Trajectory line connecting sequential points
            if len(plot_pts) >= 2:
                lx = [p[1] for p in plot_pts]
                ly = [p[2] for p in plot_pts]
                ax.plot(lx, ly, color="0.65", lw=0.8, alpha=0.6,
                        zorder=2, solid_capstyle="round")

            # Scatter with size + color encoding
            for idx, (s, x, y) in enumerate(plot_pts):
                is_first = (idx == 0)
                is_last = (idx == len(plot_pts) - 1)
                if is_first:
                    marker = "*"      # star = start
                elif is_last:
                    marker = "D"      # diamond = end
                else:
                    marker = "o"
                sz = MARKER_SIZE
                if is_first:
                    sz *= 1.5         # star reads smaller, compensate
                ax.scatter(
                    x, y,
                    color=cmap(norm(s)), marker=marker,
                    s=sz, alpha=0.9,
                    edgecolors="k", linewidth=0.4, zorder=3,
                )

            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)

            # Ticks at every integer power for log axes
            if cfg["log_x"] and xlim and xlim[0] > 0:
                xt_lo = int(np.floor(np.log10(xlim[0])))
                xt_hi = int(np.ceil(np.log10(xlim[1])))
                ax.set_xticks([10**i for i in range(xt_lo, xt_hi + 1)])

            if cfg["log_y"] and ylim:
                yt_lo = int(np.floor(np.log10(ylim[0])))
                yt_hi = int(np.ceil(np.log10(ylim[1])))
                ax.set_yticks([10**i for i in range(yt_lo, yt_hi + 1)])

            # Correlation (bottom-right)
            if len(plot_pts) >= 3:
                vx, vy = [p[1] for p in plot_pts], [p[2] for p in plot_pts]
                rho = compute_corr(corr_method, vx, vy)
                rho_str = f"{rho:.2f}".lstrip("0") if rho >= 0 else f"{rho:.2f}"
                ax.text(
                    0.96, 0.04, f"${corr_cfg['label']}$={rho_str}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=12, fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              alpha=0.85, ec="0.7", lw=0.5),
                )

            ax.tick_params(labelsize=9)
            ax.tick_params(axis="both", which="minor", length=0)
            ax.grid(True, which="major", ls=":", alpha=0.35)

            # Column title (top row)
            if ri == 0:
                ax.set_title(COL_DISPLAY[bench], fontsize=16,
                             fontweight="bold", pad=5)

            # Y label
            if ci == 0:
                ax.set_ylabel(
                    ROW_DISPLAY[ft_task] + "\n$|\\Delta\\mathcal{R}|$",
                    fontsize=12, fontweight="bold", labelpad=2,
                )
            else:
                ax.set_ylabel("$|\\Delta\\mathcal{R}|$",
                              fontsize=10, labelpad=2)

            # X label on bottom row
            if ri == nrow - 1:
                ax.set_xlabel(cfg["xlabel"], fontsize=13, labelpad=2)

    # ── Colorbar for step ─────────────────────────────────────────
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), location="right",
                        shrink=0.75, pad=0.02, aspect=30)
    cbar.set_label("Training Step", fontsize=12)
    # Discrete ticks at every other actual step for readability
    tick_stride = max(1, len(all_steps) // 6)
    cbar_ticks = all_steps[::tick_stride]
    if all_steps[-1] not in cbar_ticks:
        cbar_ticks.append(all_steps[-1])
    cbar.set_ticks(cbar_ticks)
    cbar.ax.tick_params(labelsize=9)

    # ── Legend ─────────────────────────────────────────────────────
    legend_entries = [
        (Line2D([0], [0], marker="*", color="w",
                markerfacecolor=cmap(0.0), markeredgecolor="k",
                markeredgewidth=0.5, markersize=10, linestyle="None"),
         "Start"),
        (Line2D([0], [0], marker="D", color="w",
                markerfacecolor=cmap(1.0), markeredgecolor="k",
                markeredgewidth=0.5, markersize=7, linestyle="None"),
         "End"),
        (Line2D([0], [0], color="0.65", lw=1.0, alpha=0.6),
         "Trajectory"),
    ]
    if cfg["safe_zone"]:
        legend_entries.append(
            (Line2D([0], [0], color="#d62728", ls="--", lw=1.5, alpha=0.6),
             "$|\\Delta\\mathcal{R}|=\\mathcal{B}$"))
        legend_entries.append(
            (Patch(facecolor="#2ca02c", alpha=0.15, edgecolor="none"),
             "Safe zone"))

    handles, labels = zip(*legend_entries)
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.45, 0.99),
        ncol=min(len(labels), 7), fontsize=10,
        frameon=True, fancybox=True,
        handletextpad=0.3, columnspacing=1.0, borderpad=0.4,
    )

    outpath = Path(cfg["outfile"].format(
        model=model_dir, suffix=corr_cfg["suffix"]))
    fig.savefig(str(outpath), format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {outpath}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def main():
    args = sys.argv[1:]

    # Parse flags
    model_dir = "llama"
    max_steps = 300
    corr_methods = None
    filtered = []
    i = 0
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            model_dir = args[i + 1]
            i += 2
        elif args[i] == "--max_steps" and i + 1 < len(args):
            max_steps = int(args[i + 1])
            i += 2
        elif args[i] == "--corr" and i + 1 < len(args):
            corr_methods = [c.strip() for c in args[i + 1].split(",")]
            i += 2
        else:
            filtered.append(args[i])
            i += 1

    modes = [a for a in filtered if a in MODE_CONFIG]
    if not modes:
        if filtered:
            print(f"Usage: {sys.argv[0]} [bound|omega] [--model MODEL_DIR] "
                  f"[--max_steps N] [--corr spearman,pearson]")
            sys.exit(1)
        modes = ["bound", "omega"]

    if corr_methods is None:
        corr_methods = ["spearman", "pearson"]
    for c in corr_methods:
        if c not in CORR_CONFIG:
            print(f"Unknown --corr value: {c} (expected: {list(CORR_CONFIG)})")
            sys.exit(1)

    for m in modes:
        for c in corr_methods:
            plot_grid(m, model_dir, max_steps, corr_method=c)


if __name__ == "__main__":
    main()
