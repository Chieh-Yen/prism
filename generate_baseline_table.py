#!/usr/bin/env python3
"""
Generate paper/tables/quantization/baseline.tex: a baseline-comparison table
that measures the variant-ranking power of three geometric metrics against
the empirical cross-entropy risk gap |MdR|, across the 4-model x 5-benchmark
PTQ grid (20 cells).

Metrics compared (rows of the table) — all in the W=I operational family, so
the ablation isolates *component-by-component* contribution under the
paper's consistent alignment convention (no W* confound):

  1. Omega_I     - trace-form Procrustes similarity (shape only)
  2. delta_I     - feature alignment error: shape + scale term (no head)
                   = K_feat * sqrt( (Delta rho)^2 + 2 rho_T rho_P (1-Omega_I) )
  3. Bound_I     - full PRISM bound: shape + scale + covariance-weighted head
                   -- the paper's main proposed metric

Reported per metric:

  - Mean |r_s|       over all 20 (model, benchmark) cells
  - Per-model mean   averaged across the 5 benchmarks of that lineage
                     (serves as dispersion signal: uniform = cross-lineage robust)
  - Winner count     in how many of the 20 cells is this metric's |r_s| the
                     largest (ties counted for all tied winners)

Median is omitted: per-model mean already exposes dispersion, and Mean is the
primary aggregate of interest.

Sign convention: Omega_I, Omega_W are similarity scores (higher = more
similar = smaller |MdR|), so their raw Spearman with |MdR| is negative; the
PRISM bound has the opposite sign. We report |r_s| so that larger is
uniformly better and the three metrics are magnitude-comparable.

Usage:
  python generate_baseline_table.py
"""

import csv
import math
import statistics
from pathlib import Path

# ======================================================================
# Config
# ======================================================================
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "exp_result" / "quantization" / "quantization_merged_slim.csv"
OUT_PATH = ROOT / "paper" / "tables" / "quantization" / "baseline.tex"

ROW_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B",
    "mistralai/Ministral-3-8B-Base-2512",
    "Qwen/Qwen3-8B-Base",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
]
COL_DATASETS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]

ROW_DISPLAY = {
    "meta-llama/Meta-Llama-3.1-8B": "Llama",
    "mistralai/Ministral-3-8B-Base-2512": "Ministral",
    "Qwen/Qwen3-8B-Base": "Qwen3",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "DeepSeek",
}

YCOL = "|MdR|"

# (display label, csv column, short id for internal use)
# All metrics use W=I operational alignment; rows form a progressive
# ablation ladder under a consistent alignment convention:
#   shape only -> + scale (=> feature term) -> + head (=> full bound).
METRICS = [
    (r"$\Omega_I$ \; {\scriptsize (shape only)}",                "Omega_I", "omega_i"),
    (r"$\delta_I$ \; {\scriptsize (+ scale; no head)}",          "delta_I", "delta_i"),
    (r"$\mathcal{B}_I$ \; {\scriptsize (+ head; PRISM, ours)}",  "Bound_I", "bound_i"),
]


# ======================================================================
# Rank-based correlation (self-contained, no scipy dep)
# ======================================================================
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


# ======================================================================
# Data loading
# ======================================================================
def load_data():
    needed = {col for _, col, _ in METRICS} | {YCOL}
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        item = {"target_model": r["target_model"], "dataset": r["dataset"]}
        for k in needed:
            try:
                item[k] = float(r[k])
            except (TypeError, ValueError, KeyError):
                item[k] = float("nan")
        out.append(item)
    return out


# ======================================================================
# Per-cell Spearman
# ======================================================================
def compute_per_cell_spearman(rows, metric_col):
    """Return nested dict {model: {dataset: signed r_s}} and {model: {dataset: |r_s|}}."""
    signed = {m: {} for m in ROW_MODELS}
    absval = {m: {} for m in ROW_MODELS}
    for m in ROW_MODELS:
        for d in COL_DATASETS:
            sub = [r for r in rows
                   if r["target_model"] == m
                   and r["dataset"] == d
                   and not math.isnan(r[metric_col])
                   and not math.isnan(r[YCOL])]
            xs = [r[metric_col] for r in sub]
            ys = [r[YCOL] for r in sub]
            rs = spearman(xs, ys) if len(xs) >= 3 else float("nan")
            signed[m][d] = rs
            absval[m][d] = abs(rs) if not math.isnan(rs) else float("nan")
    return signed, absval


def _gather(absval, model=None):
    vals = []
    if model is not None:
        vals = [v for v in absval[model].values() if not math.isnan(v)]
    else:
        for m in ROW_MODELS:
            vals += [v for v in absval[m].values() if not math.isnan(v)]
    return vals


# ======================================================================
# Formatting
# ======================================================================
def _fmt(x, bold=False):
    if math.isnan(x):
        return "--"
    s = f"{x:.3f}".lstrip("0") if 0 <= x < 1 else f"{x:.3f}"
    return rf"\textbf{{{s}}}" if bold else s


# ======================================================================
# Main
# ======================================================================
def main():
    rows = load_data()

    # Compute per-cell |r_s| for every metric.
    all_signed = {}
    all_abs = {}
    for label, col, mid in METRICS:
        signed, absval = compute_per_cell_spearman(rows, col)
        all_signed[mid] = signed
        all_abs[mid] = absval

    # Per-cell winner: metric with largest |r_s| in each (model, dataset) cell.
    winners = {mid: 0 for _, _, mid in METRICS}
    n_cells_with_data = 0
    for m in ROW_MODELS:
        for d in COL_DATASETS:
            per_metric = {mid: all_abs[mid][m][d] for _, _, mid in METRICS}
            finite = {k: v for k, v in per_metric.items() if not math.isnan(v)}
            if not finite:
                continue
            n_cells_with_data += 1
            best = max(finite.values())
            # Count ties as multi-winners.
            for k, v in finite.items():
                if abs(v - best) < 1e-9:
                    winners[k] += 1

    # Aggregate stats per metric.
    stats = {}
    for _, _, mid in METRICS:
        all_vals = _gather(all_abs[mid])
        stats[mid] = {
            "mean": statistics.mean(all_vals) if all_vals else float("nan"),
            "median": statistics.median(all_vals) if all_vals else float("nan"),
            "per_model_mean": {
                m: (statistics.mean(_gather(all_abs[mid], m))
                    if _gather(all_abs[mid], m) else float("nan"))
                for m in ROW_MODELS
            },
            "winners": winners[mid],
        }

    # ----------------------------------------------------------------
    # Find per-column winners for bold-facing
    # ----------------------------------------------------------------
    best_cols = {}
    col_keys = (["mean"]
                + [f"model_{m}" for m in ROW_MODELS]
                + ["winners"])

    def _colval(mid, key):
        s = stats[mid]
        if key == "mean":
            return s["mean"]
        if key == "winners":
            return s["winners"]
        if key.startswith("model_"):
            return s["per_model_mean"][key[len("model_"):]]
        raise KeyError(key)

    for k in col_keys:
        vals = {mid: _colval(mid, k) for _, _, mid in METRICS}
        finite = {mid: v for mid, v in vals.items() if v is not None and not math.isnan(v)}
        if finite:
            best_val = max(finite.values())
            best_cols[k] = {mid for mid, v in finite.items() if abs(v - best_val) < 1e-9}
        else:
            best_cols[k] = set()

    # ----------------------------------------------------------------
    # Emit LaTeX
    # ----------------------------------------------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{\textbf{Component-wise ablation of the PRISM bound.} "
        r"Per-cell Spearman rank correlation $|r_s|$ with the empirical "
        rf"cross-entropy risk gap $|\Delta\mathcal{{R}}|$ across the $4{{\times}}5$ "
        rf"PTQ grid ({n_cells_with_data} cells; larger $|r_s|$ = better variant "
        r"ranking). Rows add bound components cumulatively: shape, then "
        r"$+$ scale, then $+$ head. Per-lineage columns average $|r_s|$ over "
        r"the 5 benchmarks of that lineage. ``Wins'' counts cells where a "
        r"metric's $|r_s|$ is the largest (ties counted for each tied "
        r"metric); bold = best in column.}"
    )
    lines.append(r"\label{tab:baseline}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    # Column spec: Metric | Mean | Llama | Ministral | Qwen3 | DeepSeek | Wins
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    header_cells = (
        [r"Metric"]
        + [r"Mean $|r_s|$"]
        + [ROW_DISPLAY[m] for m in ROW_MODELS]
        + [rf"Wins / {n_cells_with_data}"]
    )
    lines.append(" & ".join(header_cells) + r" \\")
    subheader = ([""]
                 + [rf"\scriptsize ({n_cells_with_data} cells)"]
                 + [r"\scriptsize (5 benchmarks)"] * len(ROW_MODELS)
                 + [""])
    lines.append(" & ".join(subheader) + r" \\")
    lines.append(r"\midrule")
    for label, _, mid in METRICS:
        s = stats[mid]
        row = [label]
        row.append(_fmt(s["mean"], bold=(mid in best_cols["mean"])))
        for m in ROW_MODELS:
            key = f"model_{m}"
            row.append(_fmt(s["per_model_mean"][m], bold=(mid in best_cols[key])))
        wins_val = s["winners"]
        wins_str = (rf"\textbf{{{wins_val}}}" if mid in best_cols["winners"]
                    else f"{wins_val}")
        row.append(wins_str)
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    OUT_PATH.write_text("\n".join(lines) + "\n")

    # ----------------------------------------------------------------
    # Human-readable summary on stdout
    # ----------------------------------------------------------------
    print(f"Wrote {OUT_PATH}")
    print()
    print(f"Summary (|r_s| on {n_cells_with_data} cells):")
    print("=" * 96)
    hdr = f"{'Metric':<14}  {'Mean':>6}  " + "  ".join(
        f"{ROW_DISPLAY[m]:>10}" for m in ROW_MODELS) + f"  {'Wins':>5}"
    print(hdr)
    print("-" * 96)
    plain_map = {
        "omega_i":  "Omega_I",
        "omega_w":  "Omega_W",
        "delta_i":  "delta_I",
        "bound_i":  "Bound_I (ours)",
    }
    for _, _, mid in METRICS:
        s = stats[mid]
        row = f"{plain_map[mid]:<14}  {s['mean']:>6.3f}  "
        row += "  ".join(f"{s['per_model_mean'][m]:>10.3f}" for m in ROW_MODELS)
        row += f"  {s['winners']:>5}"
        print(row)
    print("=" * 96)


if __name__ == "__main__":
    main()
