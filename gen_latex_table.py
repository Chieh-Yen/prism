#!/usr/bin/env python3
"""
Generate a LaTeX table for Meta-Llama-3.1-8B × {mmlu, triviaqa, gsm8k}.

Rows: (benchmark, family, method)
Columns: rho_T, rho_P, Omega, delta, gamma, Bound, |dR|

Highlights:
  - Omega < 0.85: \cellcolor{red!10}  (structural collapse)
  - gamma structurally zero (BnB/GPTQ): $0^{\dagger}$
  - r_s per benchmark: bold
"""

import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path("quantization_merged_slim.csv")
TARGET_MODEL = "meta-llama/Meta-Llama-3.1-8B"
DATASETS = ["mmlu", "triviaqa", "gsm8k"]
DS_DISPLAY = {"mmlu": "MMLU", "triviaqa": "TriviaQA", "gsm8k": "GSM8K"}

# (method, family) in display order
METHOD_TABLE = [
    ("FP16",      "--"),
    ("Q8_0",      "GGUF"),
    ("Q6_K",      "GGUF"),
    ("Q5_K_M",    "GGUF"),
    ("Q4_K_M",    "GGUF"),
    ("Q3_K_M",    "GGUF"),
    ("Q2_K",      "GGUF"),
    ("INT8",      "BnB"),
    ("NF4",       "BnB"),
    ("FP4",       "BnB"),
    ("GPTQ-4bit", "GPTQ"),
]

# Families where gamma is structurally zero (head preserved in FP16)
GAMMA_ZERO_FAMILIES = {"BnB", "GPTQ", "--"}

# Omega thresholds for red highlight
OMEGA_LIGHT = 0.95   # light red
OMEGA_DEEP = 0.80    # deep red

# Columns to highlight red when Omega collapses
RED_COLS = {"Omega_I", "delta_I", "Bound_I", "|MdR|"}

# CSV column key -> LaTeX header
COLUMNS = [
    ("rho_T",   r"$\rho_T$"),
    ("rho_P",   r"$\rho_P$"),
    ("Omega_I", r"$\Omega$"),
    ("delta_I", r"$\delta$"),
    ("gamma_I", r"$\gamma$"),
    ("Bound_I", r"$\mathcal{B}$"),
    ("|MdR|",   r"$|\Delta\mathcal{R}|$"),
]


def parse_method(label):
    rhs = label.replace("BF16 vs ", "").strip()
    if rhs == "FP16":
        return "FP16"
    for g in ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]:
        if rhs == g:
            return g
    if rhs in ("INT8", "NF4", "FP4"):
        return rhs
    if re.match(r"GPTQ\(", rhs):
        return "GPTQ-4bit"
    return rhs


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


def fmt4(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "--"
    return f"{v:.4f}"


def fmt_cell(col_key, val, family, omega_level=None):
    """Format a cell with conditional highlighting.
    omega_level: None, "light", or "deep"
    """
    if val is None:
        return "--"

    # gamma: structural zero for BnB/GPTQ/FP16 → sky blue
    if col_key == "gamma_I" and family in GAMMA_ZERO_FAMILIES:
        return r"\cellcolor{cyan!12} $0$"

    # Omega collapse → two-level red on Omega, delta, Bound, |MdR|
    if col_key in RED_COLS and omega_level:
        color = "red!35" if omega_level == "deep" else "red!18"
        return r"\cellcolor{" + color + "} " + f"{val:.4f}"

    return f"{val:.4f}"


def main():
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    # Exclude shuyuej GPTQ variant, keep only ModelCloud
    EXCLUDE_PROXY = {"shuyuej/Meta-Llama-3.1-8B-GPTQ [GPTQ]"}

    # Index data: (dataset, method) -> list of row dicts
    data = defaultdict(list)
    for r in rows:
        if r["target_model"] != TARGET_MODEL:
            continue
        if r.get("proxy_model", "") in EXCLUDE_PROXY:
            continue
        method = parse_method(r["Label"])
        ds = r["dataset"]
        if ds in DATASETS and method in [m for m, _ in METHOD_TABLE]:
            data[(ds, method)].append(r)

    # Compute Spearman r_s(delta_I, |MdR|) per benchmark
    rho_per_ds = {}
    for ds in DATASETS:
        xs, ys = [], []
        for r in rows:
            if r["target_model"] != TARGET_MODEL or r["dataset"] != ds:
                continue
            try:
                x, y = float(r["delta_I"]), float(r["|MdR|"])
                if not math.isnan(x) and not math.isnan(y):
                    xs.append(x)
                    ys.append(y)
            except (ValueError, KeyError):
                pass
        rho_per_ds[ds] = spearman(xs, ys)

    # Build LaTeX
    n_data_cols = len(COLUMNS)
    col_spec = "ll l " + "r" * n_data_cols
    header_cols = " & ".join(h for _, h in COLUMNS)

    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Geometric decomposition for \textbf{Llama-3.1-8B} under identity alignment ($W{=}I$). "
             r"Each benchmark section reports Spearman's $r_s(\delta,\,|\Delta\mathcal{R}|)$ across all quantization variants.}")
    L.append(r"\label{tab:llama_decomposition}")
    L.append(r"\resizebox{\textwidth}{!}{%")
    L.append(r"\begin{tabular}{" + col_spec + "}")
    L.append(r"\toprule")
    L.append(r"Dataset & Family & Method & " + header_cols + r" \\")
    L.append(r"\midrule")

    for di, ds in enumerate(DATASETS):
        if di > 0:
            L.append(r"\midrule")

        methods_present = [(m, f) for m, f in METHOD_TABLE if data.get((ds, m))]
        n_rows_ds = len(methods_present)

        prev_family = None

        for row_i, (method, family) in enumerate(methods_present):
            entries = data[(ds, method)]

            # Average values across entries (for multiple GPTQ variants)
            avg = {}
            for col_key, _ in COLUMNS:
                vals = []
                for e in entries:
                    try:
                        vals.append(float(e[col_key]))
                    except (TypeError, ValueError, KeyError):
                        pass
                avg[col_key] = sum(vals) / len(vals) if vals else None

            # Dataset label on first row; bold r_s on second row
            if row_i == 0:
                ds_cell = DS_DISPLAY[ds]
            elif row_i == 1:
                rho = rho_per_ds[ds]
                ds_cell = r"{\small ($\mathbf{r_s{=}" + f"{rho:.2f}" + r"}$)}"
            else:
                ds_cell = ""

            # Family label: first row of each family group only
            if family != prev_family:
                fam_cell = "--" if family == "--" else family
                prev_family = family
            else:
                fam_cell = ""

            # Method display
            method_disp = method.replace("_", r"\_")

            # Detect Omega collapse level for this row
            omega_val = avg.get("Omega_I")
            if omega_val is not None and omega_val < OMEGA_DEEP:
                omega_level = "deep"
            elif omega_val is not None and omega_val < OMEGA_LIGHT:
                omega_level = "light"
            else:
                omega_level = None

            # Format each data cell with conditional highlighting
            vals_str = " & ".join(
                fmt_cell(ck, avg.get(ck), family, omega_level) for ck, _ in COLUMNS
            )

            L.append(f"{ds_cell} & {fam_cell} & {method_disp} & {vals_str} " + r"\\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    L.append(r"\end{table}")

    table_body = "\n".join(L)

    # Includable table
    out_path = Path("table_llama_decomposition.tex")
    out_path.write_text(table_body)
    print(f"Saved → {out_path}")

    # Standalone preview
    preamble = (
        r"\documentclass[11pt]{article}" "\n"
        r"\usepackage[utf8]{inputenc}" "\n"
        r"\usepackage[T1]{fontenc}" "\n"
        r"\usepackage{booktabs,amsmath,amssymb,graphicx}" "\n"
        r"\usepackage[table]{xcolor}" "\n"
        r"\usepackage[margin=1cm,landscape]{geometry}" "\n"
        r"\newcommand{\subsection}[1]{}" "\n"
        r"\pagestyle{empty}" "\n"
        r"\begin{document}" "\n"
    )
    preview_path = Path("table_llama_preview.tex")
    preview_path.write_text(preamble + table_body + "\n" + r"\end{document}" + "\n")
    print(f"Saved → {preview_path}  (standalone)")

    print(table_body)


if __name__ == "__main__":
    main()
