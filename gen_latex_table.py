#!/usr/bin/env python3
"""
Generate LaTeX tables per (model × task group).

Rows: (benchmark, family, method)
Columns: rho_T, rho_P, Omega, delta, gamma, Bound, |dR|

Output files land in ./paper/tables/ and are named by model + task group:
  paper/tables/table_{model_short}_{task_group}.tex

Task groups:
  main  = mmlu, triviaqa, gsm8k          (reasoning / QA / math, paper body)
  ext   = arc, squad, fineweb_edu,       (additional eval: MC, extractive,
           wikitext                        perplexity-style)

Models covered (bases used in the quantization grid plots):
  llama    = meta-llama/Meta-Llama-3.1-8B
  mistral  = mistralai/Ministral-3-8B-Base-2512
  qwen     = Qwen/Qwen3-8B-Base
  deepseek = deepseek-ai/DeepSeek-R1-Distill-Llama-8B

Highlights:
  - Omega < 0.95 / 0.80 → two-level red on (Omega, delta, Bound, |MdR|)
  - gamma structurally zero (BnB / GPTQ / FP16) → sky-blue $0$
  - r_s(delta, |MdR|) per benchmark printed under the dataset label
"""

import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "exp_result" / "quantization" / "quantization_merged_slim.csv"
OUT_DIR = ROOT / "paper" / "tables" / "quantization"


# ═══════════════════════════════════════════════════════════════════
# Model / task config
# ═══════════════════════════════════════════════════════════════════
MODELS = [
    {"path": "meta-llama/Meta-Llama-3.1-8B",
     "short": "llama",             "display": "Llama-3.1-8B"},
    {"path": "mistralai/Ministral-3-8B-Base-2512",
     "short": "mistral",           "display": "Ministral-3-8B-Base"},
    {"path": "Qwen/Qwen3-8B-Base",
     "short": "qwen",              "display": "Qwen3-8B-Base"},
    {"path": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
     "short": "deepseek",          "display": "DeepSeek-R1-Distill-Llama-8B"},
    # Instruct counterparts
    {"path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
     "short": "llama_instruct",    "display": "Llama-3.1-8B-Instruct"},
    {"path": "mistralai/Ministral-3-8B-Instruct-2512",
     "short": "mistral_instruct",  "display": "Ministral-3-8B-Instruct"},
    {"path": "Qwen/Qwen3-8B",
     "short": "qwen_instruct",     "display": "Qwen3-8B-Instruct"},
]

TASK_GROUPS = [
    {"name": "main", "layout": "table",
     "datasets": ["mmlu", "triviaqa"]},
    {"name": "ext",  "layout": "longtable",
     "datasets": ["arc", "squad", "gsm8k", "wikitext", "fineweb_edu"]},
    {"name": "all",  "layout": "longtable",
     "datasets": ["arc", "mmlu", "squad", "triviaqa", "gsm8k",
                  "wikitext", "fineweb_edu"]},
]

DS_DISPLAY = {
    "arc": "ARC", "mmlu": "MMLU", "squad": "SQuAD",
    "triviaqa": "TriviaQA", "gsm8k": "GSM8K",
    "fineweb_edu": "FineWeb-Edu", "wikitext": "WikiText",
}

# Per-model proxy exclusions (keep ModelCloud GPTQ as canonical 4-bit variant)
EXCLUDE_PROXY = {
    "meta-llama/Meta-Llama-3.1-8B": {
        "shuyuej/Meta-Llama-3.1-8B-GPTQ [GPTQ]",
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "shuyuej/Meta-Llama-3.1-8B-Instruct-GPTQ [GPTQ]",
        "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 [GPTQ]",
    },
}

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
    ("GPTQ-8bit", "GPTQ"),
]

GAMMA_ZERO_FAMILIES = {"BnB", "GPTQ", "--"}

OMEGA_LIGHT = 0.95
OMEGA_DEEP = 0.80

RED_COLS = {"Omega_I", "delta_I", "Bound_I", "|MdR|"}

COLUMNS = [
    ("rho_T",   r"$\rho_T$"),
    ("rho_P",   r"$\rho_P$"),
    ("Omega_I", r"$\Omega$"),
    ("delta_I", r"$\delta$"),
    ("gamma_I", r"$\gamma$"),
    ("Bound_I", r"$\mathcal{B}$"),
    ("|MdR|",   r"$|\Delta\mathcal{R}|$"),
]


# ═══════════════════════════════════════════════════════════════════
# Parsing helpers
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


def fmt_cell(col_key, val, family, omega_level=None):
    if val is None:
        return "--"
    if col_key == "gamma_I" and family in GAMMA_ZERO_FAMILIES:
        return r"\cellcolor{cyan!12} $0$"
    prec = 2 if col_key in ("rho_T", "rho_P") else 4
    if col_key in RED_COLS and omega_level:
        color = "red!35" if omega_level == "deep" else "red!18"
        return r"\cellcolor{" + color + "} " + f"{val:.{prec}f}"
    return f"{val:.{prec}f}"


# ═══════════════════════════════════════════════════════════════════
# Table builder
# ═══════════════════════════════════════════════════════════════════
def _collect(model_cfg, datasets, rows):
    """Return (data, rho_per_ds) filtered for this (model, dataset-list)."""
    target_model = model_cfg["path"]
    exclude = EXCLUDE_PROXY.get(target_model, set())

    data = defaultdict(list)
    valid_methods = {m for m, _ in METHOD_TABLE}
    for r in rows:
        if r["target_model"] != target_model:
            continue
        if r.get("proxy_model", "") in exclude:
            continue
        if r["dataset"] not in datasets:
            continue
        m = parse_method(r["Label"])
        if m in valid_methods:
            data[(r["dataset"], m)].append(r)

    rho_per_ds = {}
    for ds in datasets:
        xs, ys = [], []
        for r in rows:
            if r["target_model"] != target_model or r["dataset"] != ds:
                continue
            if r.get("proxy_model", "") in exclude:
                continue
            try:
                x, y = float(r["delta_I"]), float(r["|MdR|"])
                if not math.isnan(x) and not math.isnan(y):
                    xs.append(x)
                    ys.append(y)
            except (ValueError, KeyError):
                pass
        rho_per_ds[ds] = spearman(xs, ys)

    return data, rho_per_ds


def _render_body_rows(data, rho_per_ds, datasets):
    """Render the content rows (no header, no rules outside inter-dataset)."""
    L = []
    for di, ds in enumerate(datasets):
        if di > 0:
            L.append(r"\midrule")

        methods_present = [(m, f) for m, f in METHOD_TABLE if data.get((ds, m))]

        # Pre-compute averages so we can filter rows before rendering.
        rendered = []
        for method, family in methods_present:
            entries = data[(ds, method)]
            avg = {}
            for col_key, _ in COLUMNS:
                vals = []
                for e in entries:
                    try:
                        vals.append(float(e[col_key]))
                    except (TypeError, ValueError, KeyError):
                        pass
                avg[col_key] = sum(vals) / len(vals) if vals else None

            # Drop GPTQ rows whose rho_T / rho_P ratio exceeds 1.5x.
            # TODO: this is an access issue (unable to pull GPTQ activations
            #   cleanly for some proxies); remove this skip once upstream
            #   access / extraction is fixed.
            if family == "GPTQ":
                rt, rp = avg.get("rho_T"), avg.get("rho_P")
                if rt is not None and rp is not None and min(abs(rt), abs(rp)) > 0:
                    ratio = max(abs(rt), abs(rp)) / min(abs(rt), abs(rp))
                    if ratio > 1.5:
                        continue

            rendered.append((method, family, avg))

        prev_family = None
        for row_i, (method, family, avg) in enumerate(rendered):
            if row_i == 0:
                ds_cell = DS_DISPLAY.get(ds, ds)
            elif row_i == 1:
                rho = rho_per_ds[ds]
                rho_str = f"{rho:.2f}" if not math.isnan(rho) else "--"
                ds_cell = r"{\small ($\mathbf{r_s{=}" + rho_str + r"}$)}"
            else:
                ds_cell = ""

            if family != prev_family:
                fam_cell = "--" if family == "--" else family
                prev_family = family
            else:
                fam_cell = ""

            method_disp = method.replace("_", r"\_")

            omega_val = avg.get("Omega_I")
            if omega_val is not None and omega_val < OMEGA_DEEP:
                omega_level = "deep"
            elif omega_val is not None and omega_val < OMEGA_LIGHT:
                omega_level = "light"
            else:
                omega_level = None

            vals_str = " & ".join(
                fmt_cell(ck, avg.get(ck), family, omega_level)
                for ck, _ in COLUMNS
            )

            L.append(f"{ds_cell} & {fam_cell} & {method_disp} & "
                     f"{vals_str} " + r"\\")
    return L


def _wrap_regular_table(body_rows, caption, label, col_spec, header_cols):
    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{" + caption + r"}")
    L.append(r"\label{" + label + r"}")
    L.append(r"\resizebox{\textwidth}{!}{%")
    L.append(r"\begin{tabular}{" + col_spec + "}")
    L.append(r"\toprule")
    L.append(r"Dataset & Family & Method & " + header_cols + r" \\")
    L.append(r"\midrule")
    L.extend(body_rows)
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    L.append(r"\end{table}")
    return "\n".join(L)


def _wrap_longtable(body_rows, caption, label, col_spec, header_cols):
    """Multi-page longtable with repeating header + continuation markers.

    Uses \\small + tight \\tabcolsep instead of \\resizebox (which longtable
    does not support). 10 data/label columns fit NeurIPS textwidth at \\small.
    """
    ncols = col_spec.count("r") + col_spec.count("l")
    L = []
    L.append(r"\begingroup")
    L.append(r"\small")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(r"\renewcommand{\arraystretch}{1.05}")
    # longtable caption defaults to 4in; widen to textwidth
    L.append(r"\setlength{\LTcapwidth}{\textwidth}")
    L.append(r"\begin{longtable}{" + col_spec + "}")
    L.append(r"\caption{" + caption + r"}\label{" + label + r"}\\")
    # First-page header
    L.append(r"\toprule")
    L.append(r"Dataset & Family & Method & " + header_cols + r" \\")
    L.append(r"\midrule")
    L.append(r"\endfirsthead")
    # Subsequent-page header
    L.append(r"\multicolumn{" + str(ncols) + r"}{l}"
             r"{\small\itshape (continued from previous page)}\\")
    L.append(r"\toprule")
    L.append(r"Dataset & Family & Method & " + header_cols + r" \\")
    L.append(r"\midrule")
    L.append(r"\endhead")
    # Mid-page footer (when breaking)
    L.append(r"\midrule")
    L.append(r"\multicolumn{" + str(ncols) + r"}{r}"
             r"{\small\itshape continued on next page}\\")
    L.append(r"\endfoot")
    # Last-page footer
    L.append(r"\bottomrule")
    L.append(r"\endlastfoot")
    L.extend(body_rows)
    L.append(r"\end{longtable}")
    L.append(r"\endgroup")
    return "\n".join(L)


def build_table(model_cfg, group_cfg, rows):
    display = model_cfg["display"]
    short = model_cfg["short"]
    group_name = group_cfg["name"]
    datasets = group_cfg["datasets"]
    layout = group_cfg.get("layout", "table")

    data, rho_per_ds = _collect(model_cfg, datasets, rows)
    body_rows = _render_body_rows(data, rho_per_ds, datasets)

    col_spec = "ll l " + "r" * len(COLUMNS)
    header_cols = " & ".join(h for _, h in COLUMNS)
    caption = (
        r"Geometric decomposition for \textbf{" + display + r"} under identity "
        r"alignment ($W{=}I$) --- " + group_name + r" task group. "
        r"Each benchmark section reports Spearman's "
        r"$r_s(\delta,\,|\Delta\mathcal{R}|)$ across all quantization variants."
    )
    label = f"tab:{short}_decomposition_{group_name}"

    if layout == "longtable":
        return _wrap_longtable(body_rows, caption, label, col_spec, header_cols)
    return _wrap_regular_table(body_rows, caption, label, col_spec, header_cols)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stems = []
    for model_cfg in MODELS:
        for group_cfg in TASK_GROUPS:
            table = build_table(model_cfg, group_cfg, rows)
            stem = f"table_{model_cfg['short']}_{group_cfg['name']}"
            out_path = OUT_DIR / f"{stem}.tex"
            out_path.write_text(table)
            print(f"Saved → {out_path}")
            stems.append(stem)

    preamble = (
        r"\documentclass[11pt]{article}" "\n"
        r"\usepackage[utf8]{inputenc}" "\n"
        r"\usepackage[T1]{fontenc}" "\n"
        r"\usepackage{booktabs,amsmath,amssymb,graphicx}" "\n"
        r"\usepackage{longtable}" "\n"
        r"\usepackage[table]{xcolor}" "\n"
        r"\usepackage[margin=1cm,landscape]{geometry}" "\n"
        r"\pagestyle{empty}" "\n"
        r"\begin{document}" "\n"
    )
    body = "\n\n\\clearpage\n\n".join(
        f"\\input{{{stem}}}" for stem in stems)
    preview_path = OUT_DIR / "preview_all.tex"
    preview_path.write_text(preamble + body + "\n" + r"\end{document}" + "\n")
    print(f"Saved → {preview_path}  "
          f"(\\input{{...}} references each table_*.tex)")


if __name__ == "__main__":
    main()
