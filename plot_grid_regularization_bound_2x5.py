#!/usr/bin/env python3
"""
2×5 grid of forgetting trajectories overlaying multiple regularization methods.

Rows: fine-tune datasets (truthfulqa, bbq)
Cols: eval benchmarks (ARC, MMLU, SQuAD, TriviaQA, GSM8K)
Each subplot overlays trajectories of (bound, |ΔR|) from multiple methods,
each shown in a distinct color, with star=start checkpoint, diamond=end.

Default methods (recommended pair, each method at sweep |ΔR|-best):
  - baseline      (replay λ=0.0)
  - replay λ=0.01 (lowest 4-cell mean |ΔR| in replay sweep)
  - trace λ=1.0   (lowest 4-cell mean |ΔR| in trace sweep; sweep max)

Two modes:
  - "bound": x = bound_total (PRISM bound 𝓑), with y=x line + safe zone
  - "omega": x = 1 - Ω_I, no safe zone

Data sources:
  regularization_exp/exp_result/regularization/<lam>/...        (trace norm)
  regularization_exp/exp_result/regularization_replay/<lam>/... (replay CE)

Output:
  paper/figures/forgetting/forgetting_grid_reg_<mode>_<model>.pdf

Usage:
  python plot_grid_regularization_bound_2x5.py                    # both modes, both models
  python plot_grid_regularization_bound_2x5.py bound              # bound mode only
  python plot_grid_regularization_bound_2x5.py --model llama      # llama only
  python plot_grid_regularization_bound_2x5.py \
      --config "replay:0.0:baseline:tab:gray" \
      --config "replay:0.01:Replay 0.01:tab:blue" \
      --config "trace:0.5:Trace 0.5:tab:red"
"""

import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════
ROOT       = Path(__file__).resolve().parent
TRACE_DIR  = ROOT / "regularization_exp" / "exp_result" / "regularization"
REPLAY_DIR = ROOT / "regularization_exp" / "exp_result" / "regularization_replay"
FIG_DIR    = ROOT / "paper" / "figures" / "forgetting"

ROW_TASKS = ["truthfulqa", "bbq"]
COL_BENCHMARKS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]

ROW_DISPLAY = {"truthfulqa": "FT: TruthfulQA", "bbq": "FT: BBQ"}
COL_DISPLAY = {"arc": "ARC", "mmlu": "MMLU", "squad": "SQuAD",
               "triviaqa": "TriviaQA", "gsm8k": "GSM8K"}

MODE_CONFIG = {
    "bound": {
        "xcol":      "bound_total",
        "xlabel":    r"PRISM Bound $\mathcal{B}$",
        "safe_zone": True,
        "log_x":     True,
        "log_y":     True,
        "outfile":   "forgetting_grid_reg_bound_{model}.pdf",
    },
    "omega": {
        "xcol":      "omega",
        "xlabel":    r"$1 - \Omega_I$",
        "safe_zone": False,
        "log_x":     True,
        "log_y":     True,
        "transform_x": lambda v: 1 - v,
        "outfile":   "forgetting_grid_reg_omega_{model}.pdf",
    },
}

# Each method: (method_type, lambda, label, color).
# Recommended pair: each method at its sweep |ΔR|-best
#   - replay λ=0.01: lowest 4-cell mean |ΔR| in replay sweep
#   - trace  λ=1.0:  lowest 4-cell mean |ΔR| in trace sweep; sweep max
DEFAULT_METHODS = [
    ("replay", "0.0",  "no reg",                       "tab:gray"),
    ("replay", "0.01", "replay ($\\lambda{=}0.01$)",   "tab:blue"),
    ("trace",  "1.0",  "trace ($\\lambda{=}1.0$)",     "tab:red"),
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


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════
def get_dir(method_type):
    return TRACE_DIR if method_type == "trace" else REPLAY_DIR


def load_run(method_type, lam, model, ft_task):
    p = (get_dir(method_type) / lam / model
         / f"prism_forgetting_metrics_{ft_task}.json")
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def extract_trajectory(data, eval_task, xcol, max_steps=0, transform_x=None):
    """Return list of (step, x, y) tuples sorted by step."""
    points = []
    if data is None:
        return points
    for ckpt in data["checkpoints"]:
        step = ckpt["step"]
        if max_steps > 0 and step > max_steps:
            continue
        t = ckpt["tasks"].get(eval_task)
        if t is None:
            continue
        x_val = t.get(xcol)
        y_val = t.get("delta_risk")
        if x_val is None or y_val is None:
            continue
        if isinstance(x_val, float) and math.isnan(x_val):
            continue
        if isinstance(y_val, float) and math.isnan(y_val):
            continue
        if transform_x is not None:
            x_val = transform_x(x_val)
        points.append((step, x_val, abs(y_val)))
    points.sort(key=lambda p: p[0])
    return points


# ═══════════════════════════════════════════════════════════════════
# Core plotting function
# ═══════════════════════════════════════════════════════════════════
def plot_grid(mode, model, methods, max_steps=300, outfile_name=None):
    cfg = MODE_CONFIG[mode]
    xcol = cfg["xcol"]
    transform_x = cfg.get("transform_x")

    nrow, ncol = len(ROW_TASKS), len(COL_BENCHMARKS)
    legend_reserve_inch = 1.4
    fig_height = nrow * 2.8 + legend_reserve_inch
    fig_width = ncol * 3.6
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    top_frac = (fig_height - legend_reserve_inch) / fig_height
    plt.subplots_adjust(
        hspace=0.28, wspace=0.28,
        top=top_frac, bottom=0.10, left=0.08, right=0.97,
    )

    # Pre-load all trajectories per (ft_task, eval, method)
    trajectories = {}
    for ft_task in ROW_TASKS:
        for ev in COL_BENCHMARKS:
            for (mt, lam, label, color) in methods:
                run = load_run(mt, lam, model, ft_task)
                trajectories[(ft_task, ev, mt, lam)] = extract_trajectory(
                    run, ev, xcol, max_steps, transform_x)

    MARKER_SIZE = 60

    for ri, ft_task in enumerate(ROW_TASKS):
        for ci, ev in enumerate(COL_BENCHMARKS):
            ax = axes[ri][ci]

            if cfg["log_x"]:
                ax.set_xscale("log")
            if cfg["log_y"]:
                ax.set_yscale("log")

            # Compute global axis limits across all methods in this cell
            all_xs, all_ys = [], []
            for (mt, lam, _label, _color) in methods:
                pts = trajectories[(ft_task, ev, mt, lam)]
                all_xs.extend(p[1] for p in pts)
                all_ys.extend(p[2] for p in pts)

            xs_pos = [v for v in all_xs if v > 0]
            ys_pos = [v for v in all_ys if v > 0]
            xlim, ylim = None, None
            if cfg["log_x"] and xs_pos:
                x_lo_d = np.floor(np.log10(min(xs_pos))) - 0.3
                x_hi_d = np.ceil(np.log10(max(xs_pos))) + 0.3
                xlim = (10**x_lo_d, 10**x_hi_d)
            if cfg["log_y"] and ys_pos:
                y_lo_d = np.floor(np.log10(min(ys_pos))) - 0.3
                y_hi_d = np.ceil(np.log10(max(ys_pos))) + 0.3
                ylim = (10**y_lo_d, 10**y_hi_d)

            # Safe zone (bound mode only)
            if cfg["safe_zone"] and xlim and ylim:
                diag_lo = min(xlim[0], ylim[0])
                diag_hi = max(xlim[1], ylim[1])
                diag = [diag_lo, diag_hi]
                ax.plot(diag, diag, color="#d62728", ls="--", lw=1.0,
                        alpha=0.45, zorder=1, clip_on=True)
                ax.fill_between(diag, diag, ylim[0], color="#2ca02c",
                                alpha=0.05, zorder=0, clip_on=True)

            # Plot each method's trajectory
            corr_lines = []
            for (mt, lam, label, color) in methods:
                pts = trajectories[(ft_task, ev, mt, lam)]
                plot_pts = [(s, x, y) for s, x, y in pts
                            if (not cfg["log_x"] or x > 0)
                            and (not cfg["log_y"] or y > 0)]
                if not plot_pts:
                    continue

                # Trajectory line
                lx = [p[1] for p in plot_pts]
                ly = [p[2] for p in plot_pts]
                if len(plot_pts) >= 2:
                    ax.plot(lx, ly, color=color, lw=0.9, alpha=0.5,
                            zorder=2, solid_capstyle="round")

                # Markers: star=start, diamond=end, circle=middle
                for idx, (s, x, y) in enumerate(plot_pts):
                    is_first = (idx == 0)
                    is_last  = (idx == len(plot_pts) - 1)
                    if is_first:
                        marker, sz = "*", MARKER_SIZE * 1.5
                    elif is_last:
                        marker, sz = "D", MARKER_SIZE
                    else:
                        marker, sz = "o", MARKER_SIZE * 0.5
                    ax.scatter(x, y, color=color, marker=marker, s=sz,
                               alpha=0.85, edgecolors="k", linewidth=0.4,
                               zorder=3)

                # Per-method Spearman
                if len(plot_pts) >= 3:
                    rs = spearman([p[1] for p in plot_pts],
                                  [p[2] for p in plot_pts])
                    rs_str = (f"{rs:.2f}".lstrip("0").lstrip("-")
                              if abs(rs) < 1 else f"{rs:.2f}")
                    sign = "" if rs >= 0 else "-"
                    corr_lines.append((color, f"$r_s$={sign}{rs_str}"))

            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)

            # Per-method correlation annotation (color-coded, stacked)
            if corr_lines:
                for ki, (color, txt) in enumerate(corr_lines):
                    ax.text(
                        0.96, 0.04 + ki * 0.10, txt,
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=11, fontstyle="italic", color=color,
                    )

            ax.tick_params(labelsize=10)
            ax.tick_params(axis="both", which="minor", length=0)
            ax.grid(True, which="major", ls=":", alpha=0.30)

            if ri == 0:
                ax.set_title(COL_DISPLAY[ev], fontsize=15,
                             fontweight="bold", pad=6)
            if ci == 0:
                ax.set_ylabel(
                    ROW_DISPLAY[ft_task] + "\n" + r"$|\Delta\mathcal{R}|$",
                    fontsize=13, fontweight="bold", labelpad=2,
                )
            if ri == nrow - 1:
                ax.set_xlabel(cfg["xlabel"], fontsize=13, labelpad=2)

    # ── Legend ─────────────────────────────────────────────────────
    legend_entries = []
    for (mt, lam, label, color) in methods:
        legend_entries.append(
            (Line2D([0], [0], color=color, marker="o", lw=1.0,
                    markersize=8, alpha=0.85), label))
    legend_entries.append(
        (Line2D([0], [0], marker="*", color="w", markerfacecolor="0.5",
                markeredgecolor="k", markeredgewidth=0.5, markersize=13,
                linestyle="None"), "Start"))
    legend_entries.append(
        (Line2D([0], [0], marker="D", color="w", markerfacecolor="0.5",
                markeredgecolor="k", markeredgewidth=0.5, markersize=9,
                linestyle="None"), "End"))
    if cfg["safe_zone"]:
        legend_entries.append(
            (Line2D([0], [0], color="#d62728", ls="--", lw=1.5, alpha=0.5),
             r"$|\Delta\mathcal{R}|=\mathcal{B}$"))
        legend_entries.append(
            (Patch(facecolor="#2ca02c", alpha=0.15, edgecolor="none"),
             "Safe zone"))

    handles, labels = zip(*legend_entries)
    leg = fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=min(len(labels), 4), fontsize=18,
        frameon=True, fancybox=True, edgecolor="black",
        handletextpad=0.3, columnspacing=1.0, borderpad=0.4,
    )
    leg.get_frame().set_linewidth(0.8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if outfile_name is None:
        outfile_name = cfg["outfile"].format(model=model)
    outpath = FIG_DIR / outfile_name
    fig.savefig(str(outpath), format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {outpath}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def parse_methods(specs):
    """Parse 'method:lambda:label:color' format strings."""
    out = []
    for s in specs:
        parts = s.split(":", 3)
        if len(parts) != 4:
            raise ValueError(
                f"Invalid method spec {s!r}: expected method:lambda:label:color")
        out.append(tuple(parts))
    return out


def main():
    args = sys.argv[1:]
    model = None
    max_steps = 300
    method_specs = []
    filtered = []
    i = 0
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]; i += 2
        elif args[i] == "--max_steps" and i + 1 < len(args):
            max_steps = int(args[i + 1]); i += 2
        elif args[i] == "--config" and i + 1 < len(args):
            method_specs.append(args[i + 1]); i += 2
        else:
            filtered.append(args[i]); i += 1

    modes = [a for a in filtered if a in MODE_CONFIG]
    if not modes:
        if filtered:
            print(f"Usage: {sys.argv[0]} [bound|omega] "
                  f"[--model llama|qwen] [--max_steps N] "
                  f"[--config method:lambda:label:color]...")
            sys.exit(1)
        modes = ["bound", "omega"]

    methods = parse_methods(method_specs) if method_specs else DEFAULT_METHODS
    models = [model] if model else ["llama", "qwen"]

    print(f"=== methods ===")
    for m in methods:
        print(f"  {m[0]:6s} λ={m[1]:5s} → {m[2]}  ({m[3]})")

    for m in models:
        for mode in modes:
            plot_grid(mode, m, methods, max_steps=max_steps)


if __name__ == "__main__":
    main()
