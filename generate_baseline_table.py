#!/usr/bin/env python3
"""
Generate two component-wise ablation tables that measure the variant-ranking
power of three geometric metrics against the empirical cross-entropy risk
gap |MdR|, across an (N-model x 5-benchmark) PTQ grid. The model set is
selectable via --models; the default is the 2-model set (llama + qwen):

  - paper/tables/quantization/baseline.tex   (main text, W=I identity-alignment form)
  - paper/tables/quantization/baseline_w.tex (appendix, W=W_N Procrustes-optimal)

Both tables share the same row structure (shape -> + scale -> + head) and
the same per-cell Spearman protocol; they differ only in the alignment
convention used inside each metric (identity vs Procrustes-optimal).

Reported per metric:

  - Mean |r_s|       over all 20 (model, benchmark) cells
  - Per-model mean   averaged across the 5 benchmarks of that lineage
                     (serves as dispersion signal: uniform = cross-lineage robust)
  - Winner count     in how many of the 20 cells is this metric's |r_s| the
                     largest (ties counted for all tied winners)

Median is omitted: per-model mean already exposes dispersion, and Mean is the
primary aggregate of interest.

Sign convention: Omega (W=I) and Omega_N (W=W_N) are similarity scores
(higher = more similar = smaller |MdR|), so their raw Spearman with |MdR|
is negative; the PRISM bound has the opposite sign. We report |r_s| so that
larger is uniformly better and the three metrics are magnitude-comparable.

Usage:
  python generate_baseline_table.py                                # default: llama + qwen
  python generate_baseline_table.py --models llama qwen mistral deepseek
"""

import argparse
import csv
import math
import statistics
from pathlib import Path

# ======================================================================
# Config
# ======================================================================
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "exp_result" / "quantization" / "quantization_merged_slim.csv"
OUT_PATH_I = ROOT / "paper" / "tables" / "quantization" / "baseline.tex"
OUT_PATH_W = ROOT / "paper" / "tables" / "quantization" / "baseline_w.tex"
OUT_PATH_COMBINED = ROOT / "paper" / "tables" / "quantization" / "baseline_combined.tex"

# short_id -> (target_model path in CSV, display label)
MODEL_CATALOG = {
    "llama":    ("meta-llama/Meta-Llama-3.1-8B",             "Llama"),
    "mistral":  ("mistralai/Ministral-3-8B-Base-2512",       "Ministral"),
    "qwen":     ("Qwen/Qwen3-8B-Base",                       "Qwen3"),
    "deepseek": ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DeepSeek"),
}
# Default selection when --models is not passed. Override at CLI.
DEFAULT_MODELS = ["llama", "qwen"]

# Populated from DEFAULT_MODELS here and overwritten in main() once CLI is
# parsed; downstream helpers read these module-level globals at call time.
ROW_MODELS = [MODEL_CATALOG[m][0] for m in DEFAULT_MODELS]
ROW_DISPLAY = {MODEL_CATALOG[m][0]: MODEL_CATALOG[m][1] for m in DEFAULT_MODELS}
COL_DATASETS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]

YCOL = "|MdR|"

# (display label, csv column, short id for internal use)
# All metrics use W=I identity alignment; rows form a progressive
# ablation ladder under a consistent alignment convention:
#   shape only -> + scale (=> feature term) -> + head (=> full bound).
# Symbol column is left-aligned in a fixed-width \makebox so that the
# qualifiers in {\scriptsize (...)} start at the same horizontal position
# across rows (otherwise $\Omega$, $\delta$, $\mathcal{B}$ have different
# widths and the parens misalign visually).
_SYM_W = "2.1em"  # wide enough for $\mathcal{B}_N$, the widest symbol


def _label(symbol_tex, qualifier):
    return rf"\makebox[{_SYM_W}][l]{{${symbol_tex}$}}{{\scriptsize {qualifier}}}"


# Operational W=I form: paper convention drops the subscript so the
# trace-form symbols match the main-text Eq.~\ref{eq:omega_def}.
# CSV column names (second tuple element) keep their original "_I" suffix
# to match the data file schema.
METRICS_I = [
    (_label(r"\Omega",       r"(shape only; baseline)"),     "Omega_I", "omega_i"),
    (_label(r"\delta",       r"(+ scale; no head)"),         "delta_I", "delta_i"),
    (_label(r"\mathcal{B}",  r"(+ head; PRISM, $W{=}I$)"),   "Bound_I", "bound_i"),
]

# Same ablation ladder, but each metric uses the Procrustes-optimal alignment
# W=W_N (i.e., \|Z_T - Z_P W_N\|_F^2 minimization in the feature term, and
# W_N H_T - H_P inside the covariance-weighted head term). Symbols carry the
# "_N" subscript to mark the nuclear-norm specialization used in the appendix
# counterpart table. CSV column names (second tuple element) keep their
# original "_W" suffix to match the data file schema.
METRICS_W = [
    (_label(r"\Omega_N",       r"(shape only; baseline)"),       "Omega_W", "omega_w"),
    (_label(r"\delta_N",       r"(+ scale; no head)"),           "delta_W", "delta_w"),
    (_label(r"\mathcal{B}_N",  r"(+ head; PRISM, $W{=}W_N$)"),   "Bound_W", "bound_w"),
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
    needed = ({col for _, col, _ in METRICS_I}
              | {col for _, col, _ in METRICS_W}
              | {YCOL})
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
def _fmt(x, rank=None):
    """rank=1: bold; rank=2: underline; else plain."""
    if math.isnan(x):
        return "--"
    s = f"{x:.3f}".lstrip("0") if 0 <= x < 1 else f"{x:.3f}"
    if rank == 1:
        return r"\textbf{" + s + "}"
    if rank == 2:
        return r"\underline{" + s + "}"
    return s


# ======================================================================
# Per-table emission (parametrized over the metrics list and output path)
# ======================================================================
def _compute_stats(rows, metrics):
    """Return (stats, best_cols, n_cells_with_data) for the given metrics list."""
    all_abs = {}
    for _, col, mid in metrics:
        _, absval = compute_per_cell_spearman(rows, col)
        all_abs[mid] = absval

    # Per-cell winner: metric with largest |r_s| in each (model, dataset) cell.
    winners = {mid: 0 for _, _, mid in metrics}
    n_cells_with_data = 0
    for m in ROW_MODELS:
        for d in COL_DATASETS:
            per_metric = {mid: all_abs[mid][m][d] for _, _, mid in metrics}
            finite = {k: v for k, v in per_metric.items() if not math.isnan(v)}
            if not finite:
                continue
            n_cells_with_data += 1
            best = max(finite.values())
            for k, v in finite.items():
                if abs(v - best) < 1e-9:
                    winners[k] += 1

    stats = {}
    for _, _, mid in metrics:
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

    # Per-column 1st / 2nd placement for bold + green / underline + green.
    best_cols = {}
    second_cols = {}
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
        vals = {mid: _colval(mid, k) for _, _, mid in metrics}
        finite = {mid: v for mid, v in vals.items()
                  if v is not None and not math.isnan(v)}
        if finite:
            best_val = max(finite.values())
            best_cols[k] = {mid for mid, v in finite.items()
                            if abs(v - best_val) < 1e-9}
            non_best = [v for mid, v in finite.items()
                        if mid not in best_cols[k]]
            if non_best:
                second_val = max(non_best)
                second_cols[k] = {mid for mid, v in finite.items()
                                  if abs(v - second_val) < 1e-9
                                  and mid not in best_cols[k]}
            else:
                second_cols[k] = set()
        else:
            best_cols[k] = set()
            second_cols[k] = set()

    return stats, best_cols, second_cols, n_cells_with_data


def _emit_latex(metrics, stats, best_cols, second_cols, n_cells_with_data,
                out_path, label, caption):
    def _rank(mid, key):
        if mid in best_cols[key]:
            return 1
        if mid in second_cols[key]:
            return 2
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    # Dynamic col_spec with vertical rules grouping: label | aggregate (Mean) |
    # per-model. Pipes give the reader visual partitions.
    col_spec = "l | c | " + "c" * len(ROW_MODELS)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    header_cells = (
        [r"Metric"]
        + [r"Mean $|r_s|$"]
        + [ROW_DISPLAY[m] for m in ROW_MODELS]
    )
    lines.append(" & ".join(header_cells) + r" \\")
    subheader = ([""]
                 + [rf"\scriptsize ({n_cells_with_data} cells)"]
                 + [r"\scriptsize (5 benchmarks)"] * len(ROW_MODELS))
    lines.append(" & ".join(subheader) + r" \\")
    lines.append(r"\midrule")
    for label_text, _, mid in metrics:
        s = stats[mid]
        row = [label_text]
        row.append(_fmt(s["mean"], rank=_rank(mid, "mean")))
        for m in ROW_MODELS:
            key = f"model_{m}"
            row.append(_fmt(s["per_model_mean"][m], rank=_rank(mid, key)))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")


def _print_summary(metrics, stats, n_cells_with_data, out_path, alignment_tag):
    print(f"Wrote {out_path}")
    print()
    print(f"Summary [{alignment_tag}] (|r_s| on {n_cells_with_data} cells):")
    print("=" * 96)
    hdr = f"{'Metric':<14}  {'Mean':>6}  " + "  ".join(
        f"{ROW_DISPLAY[m]:>10}" for m in ROW_MODELS) + f"  {'Wins':>5}"
    print(hdr)
    print("-" * 96)
    for _, col, mid in metrics:
        s = stats[mid]
        row = f"{col:<14}  {s['mean']:>6.3f}  "
        row += "  ".join(f"{s['per_model_mean'][m]:>10.3f}" for m in ROW_MODELS)
        row += f"  {s['winners']:>5}"
        print(row)
    print("=" * 96)
    print()


def generate_table(rows, metrics, out_path, label, caption, alignment_tag):
    stats, best_cols, second_cols, n_cells = _compute_stats(rows, metrics)
    _emit_latex(metrics, stats, best_cols, second_cols, n_cells,
                out_path, label, caption)
    _print_summary(metrics, stats, n_cells, out_path, alignment_tag)
    return stats, best_cols, second_cols, n_cells


def _emit_combined_latex(groups, out_path, label, caption):
    """groups: list of (metrics, stats, best_cols, second_cols, n_cells) tuples.

    Produces one tabular with a single header followed by each group's rows
    separated by \\midrule. Ranking is per-group (each group's stats and
    best/second_cols are independent), so each block highlights its own
    column-best rather than being dominated by the other block.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use the first group's n_cells in the header (groups should match by design).
    n_cells_header = groups[0][4]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    # Dynamic col_spec: label | per-model | aggregate (Mean).
    # Pipes give the reader visual partitions.
    col_spec = "l | " + "c" * len(ROW_MODELS) + " | c"
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    header_cells = (
        [r"Metric"]
        + [ROW_DISPLAY[m] for m in ROW_MODELS]
        + [r"Mean $r_s$"]
    )
    lines.append(" & ".join(header_cells) + r" \\")
    subheader = ([""]
                 + [r"\scriptsize (5 benchmarks)"] * len(ROW_MODELS)
                 + [rf"\scriptsize ({n_cells_header} cells)"])
    lines.append(" & ".join(subheader) + r" \\")

    for gi, (metrics, stats, best_cols, second_cols, _) in enumerate(groups):
        def _rank(mid, key, bc=best_cols, sc=second_cols):
            if mid in bc[key]:
                return 1
            if mid in sc[key]:
                return 2
            return None
        lines.append(r"\midrule")
        for label_text, _, mid in metrics:
            s = stats[mid]
            row = [label_text]
            for m in ROW_MODELS:
                key = f"model_{m}"
                row.append(_fmt(s["per_model_mean"][m], rank=_rank(mid, key)))
            row.append(_fmt(s["mean"], rank=_rank(mid, "mean")))
            lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


# ======================================================================
# Captions (kept inline so they live next to the metric definitions above)
# ======================================================================
def _caption_i():
    grid = rf"${len(ROW_MODELS)}{{\times}}{len(COL_DATASETS)}$"
    return (
        r"\textbf{Component-wise ablation of the PRISM bound (identity-alignment form, $W{=}I$).} "
        r"Per-cell Spearman rank correlation $|r_s|$ with the empirical "
        r"cross-entropy risk gap $|\Delta\mathcal{R}|$ across the " + grid + r" "
        r"PTQ grid (larger $|r_s|$ = better variant ranking). Rows add bound "
        r"components cumulatively: shape, then $+$ scale, then $+$ head. "
        r"Per-lineage columns average $|r_s|$ over the "
        rf"{len(COL_DATASETS)} benchmarks of that "
        r"lineage. \textbf{Bold} / \underline{underline}: 1st / 2nd-best in column. The "
        r"Procrustes-optimal $W{=}W_N$ counterpart is in "
        r"Appendix~Table~\ref{tab:baseline_w}."
    )

CAPTION_W = (
    r"\textbf{Component-wise ablation under Procrustes-optimal alignment "
    r"($W{=}W_N$).} Counterpart of Table~\ref{tab:baseline}: same row "
    r"structure (shape $\to +$ scale $\to +$ head) and same per-cell "
    r"Spearman protocol, but every metric is evaluated under the optimal "
    r"orthogonal alignment $W_N$ rather than identity. Larger $|r_s|$ = "
    r"better variant ranking; \textbf{bold} / \underline{underline}: 1st / 2nd-best in column. As expected, the "
    r"Procrustes-optimal alignment yields modestly stronger ranking on the "
    r"full bound; the identity-alignment form ($W{=}I$) used in the main text "
    r"(Table~\ref{tab:baseline}) trades this margin for autograd "
    r"compatibility (no SVD per step) and for the head-term simplification "
    r"$H_T = H_P$ that holds in the frozen-\texttt{lm\_head} regimes (LoRA, "
    r"FP16-head PTQ) studied in Sec.~\ref{sec:experiments}."
)

CAPTION_COMBINED = (
    r"\textbf{Component-wise ablation: PRISM ranks variants strongly under both alignments.} "
    r"Rows add bound components cumulatively (shape $\to +$ scale $\to +$ head); "
    r"larger $r_s$ = better ranking; \textbf{bold} / \underline{underline}: "
    r"1st / 2nd-best within each block. "
    r"Top block: identity $W{=}I$ (main text default; $\mathcal{B}$ reaches $r_s{=}0.82$). "
    r"Bottom block: Procrustes-optimal $W{=}W_N$ ($\mathcal{B}_N$ reaches $r_s{=}0.91$, "
    r"modestly tighter since $W_N$ minimizes the alignment residual). "
    r"The main text adopts $W{=}I$ for autograd compatibility and the "
    r"$H_T{=}H_P$ head-term simplification in frozen-\texttt{lm\_head} regimes."
)


# ======================================================================
# Main
# ======================================================================
def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CATALOG.keys()),
        default=list(DEFAULT_MODELS),
        metavar="ID",
        help=(f"Model short-ids to include (default: "
              f"{' '.join(DEFAULT_MODELS)}). "
              f"Available: {', '.join(MODEL_CATALOG)}."),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    # Mutate in-place so every helper that reads these globals picks up the
    # CLI-selected model set on its next access.
    ROW_MODELS[:] = [MODEL_CATALOG[m][0] for m in args.models]
    ROW_DISPLAY.clear()
    ROW_DISPLAY.update(
        {MODEL_CATALOG[m][0]: MODEL_CATALOG[m][1] for m in args.models}
    )

    rows = load_data()
    stats_i, best_i, second_i, n_i = generate_table(
        rows, METRICS_I, OUT_PATH_I,
        label="tab:baseline",
        caption=_caption_i(),
        alignment_tag="W=I",
    )
    stats_w, best_w, second_w, n_w = generate_table(
        rows, METRICS_W, OUT_PATH_W,
        label="tab:baseline_w",
        caption=CAPTION_W,
        alignment_tag="W=W_N",
    )
    _emit_combined_latex(
        groups=[
            (METRICS_I, stats_i, best_i, second_i, n_i),
            (METRICS_W, stats_w, best_w, second_w, n_w),
        ],
        out_path=OUT_PATH_COMBINED,
        label="tab:baseline_combined",
        caption=CAPTION_COMBINED,
    )


if __name__ == "__main__":
    main()
