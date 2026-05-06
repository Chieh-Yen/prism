#!/usr/bin/env python3
"""
Generate LaTeX tables per (model × task group).

Rows: (benchmark, family, method)
Columns: rho_T, rho_P, Omega, delta, gamma, Bound, |dR|

Output files land in ./paper/tables/ and are named by model + task group:
  paper/tables/table_{model_short}_{task_group}.tex

Task groups:
  main  = mmlu                            (paper body, Sec.~5.2 Table 1)
  ext   = arc, triviaqa, squad, gsm8k,   (appendix: extended eval set
           wikitext, fineweb_edu           excluding mmlu)
  all   = mmlu + ext                      (appendix: full eval set incl. mmlu)

Models covered (bases used in the quantization grid plots):
  llama    = meta-llama/Meta-Llama-3.1-8B
  mistral  = mistralai/Ministral-3-8B-Base-2512
  qwen     = Qwen/Qwen3-8B-Base
  deepseek = deepseek-ai/DeepSeek-R1-Distill-Llama-8B

Highlights:
  - Omega < 0.95 / 0.80 → two-level red on (Omega, delta, Bound, |MdR|)
  - gamma structurally zero (BnB / GPTQ / FP16) → sky-blue $0$
  - r_s(B, |MdR|) per benchmark printed under the dataset label
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
    {"name": "main", "layout": "table", "caption_label": "MMLU",
     "datasets": ["mmlu"]},
    {"name": "ext",  "layout": "longtable",
     "caption_label": "extended benchmarks (excl.\\ MMLU)",
     "datasets": ["arc", "triviaqa", "squad", "gsm8k",]},
    #              "wikitext", "fineweb_edu"]},
    {"name": "all",  "layout": "longtable",
     "caption_label": "all benchmarks",
     "datasets": ["mmlu", "arc", "triviaqa", "squad", "gsm8k",]},
    #              "wikitext", "fineweb_edu"]},
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
    ("rho_T",   r"$\boldsymbol{\rho_T}$"),
    ("rho_P",   r"$\boldsymbol{\rho_P}$"),
    ("Omega_I", r"$\boldsymbol{\Omega}$"),
    ("delta_I", r"$\boldsymbol{\delta}$"),
    ("gamma_I", r"$\boldsymbol{\gamma}$"),
    ("Bound_I", r"\textbf{PRISM} $\boldsymbol{\mathcal{B}}$"),
    ("|MdR|",   r"$\boldsymbol{|\Delta\mathcal{R}|}$"),
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

    # NB: Spearman intentionally does NOT apply EXCLUDE_PROXY -- the exclusion
    # is a body-display dedup (one canonical GPTQ-4bit row), but the rank
    # correlation should consume all raw proxies so it matches the figure.
    rho_per_ds = {}
    for ds in datasets:
        xs, ys = [], []
        for r in rows:
            if r["target_model"] != target_model or r["dataset"] != ds:
                continue
            try:
                x, y = float(r["Bound_I"]), float(r["|MdR|"])
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
                        v = float(e[col_key])
                        # Scale delta and gamma so that delta + gamma = Bound
                        # in the displayed table (per-row scaling, then avg).
                        # CSV stores delta_I, gamma_I as raw geometric residuals;
                        # Bound_I = K_f * delta_I + K_p * gamma_I per row.
                        if col_key == "delta_I":
                            v *= float(e["K_f"])
                        elif col_key == "gamma_I":
                            v *= float(e["K_p"])
                        vals.append(v)
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
    L.append(r"\multicolumn{1}{c}{\textbf{Dataset}} & \multicolumn{1}{c}{\textbf{Family}} & \multicolumn{1}{c}{\textbf{Method}} & " + header_cols + r" \\")
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
    L.append(r"\multicolumn{1}{c}{\textbf{Dataset}} & \multicolumn{1}{c}{\textbf{Family}} & \multicolumn{1}{c}{\textbf{Method}} & " + header_cols + r" \\")
    L.append(r"\midrule")
    L.append(r"\endfirsthead")
    # Subsequent-page header
    L.append(r"\multicolumn{" + str(ncols) + r"}{l}"
             r"{\small\itshape (continued from previous page)}\\")
    L.append(r"\toprule")
    L.append(r"\multicolumn{1}{c}{\textbf{Dataset}} & \multicolumn{1}{c}{\textbf{Family}} & \multicolumn{1}{c}{\textbf{Method}} & " + header_cols + r" \\")
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
    caption_label = group_cfg.get("caption_label", group_name)

    data, rho_per_ds = _collect(model_cfg, datasets, rows)
    body_rows = _render_body_rows(data, rho_per_ds, datasets)

    col_spec = "ll l " + "r" * len(COLUMNS)
    header_cols = " & ".join(rf"\multicolumn{{1}}{{c}}{{{h}}}" for _, h in COLUMNS)
    caption = (
        r"Geometric decomposition for \textbf{" + display + r"} under identity "
        r"alignment ($W{=}I$) on " + caption_label + r". "
        r"Each benchmark section reports Spearman's "
        r"$r_s(\mathcal{B},\,|\Delta\mathcal{R}|)$ across all quantization variants. "
        r"Shading: \colorbox{red!35}{$\Omega{<}0.80$} / "
        r"\colorbox{red!18}{$\Omega{<}0.95$} on $(\Omega,\delta,\mathcal{B},|\Delta\mathcal{R}|)$; "
        r"\colorbox{cyan!12}{$\gamma{=}0$} when the head is preserved ($H_T{=}H_P$)."
    )
    label = f"tab:{short}_decomposition_{group_name}_bound"

    if layout == "longtable":
        return _wrap_longtable(body_rows, caption, label, col_spec, header_cols)
    return _wrap_regular_table(body_rows, caption, label, col_spec, header_cols)


# ═══════════════════════════════════════════════════════════════════
# Cross-model summary: |dR| and Spearman per (model, benchmark)
# ═══════════════════════════════════════════════════════════════════
SUMMARY_BENCHMARKS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]


def _short_display(name: str) -> str:
    """Shorten display names so the summary table fits textwidth."""
    return (name.replace("-Base", "")
                .replace("-Distill-Llama", "")
                .replace("-2512", ""))


def build_summary_table(rows):
    """Per-(model x benchmark) mean |dR| and per-benchmark mean Spearman r_s.

    Demonstrates the GSM8K outlier: its small |dR| (long teacher-forced answer
    spans dilute per-token loss) yields low SNR, depressing rank correlation.

    Layout
        Columns: 5 benchmarks (ARC, MMLU, SQuAD, TriviaQA, GSM8K)
        Rows:    7 model families
                 + row mean of mean |dR| per benchmark
                 + row mean of Spearman r_s per benchmark
    """
    cell_dr = {}     # (short, ds) -> mean |dR| (matches body-table aggregation)
    cell_rs = {}     # (short, ds) -> Spearman r_s(B, |dR|) over raw rows

    for model_cfg in MODELS:
        target = model_cfg["path"]
        for ds in SUMMARY_BENCHMARKS:
            # Pass 1: collect raw rows (no method/quality filter) for Spearman,
            # which intentionally consumes every variant so it matches the
            # figure's per-subplot r_s and the body-table convention.
            xs, ys = [], []
            # Pass 2: collect rows for the |dR| mean using the same filters
            # as _collect(): per-method aggregation + GPTQ ratio>1.5x drop
            # (a data-quality filter for GPTQ extraction failures, marked
            # TODO in _collect; removing it would let measurement artefacts
            # leak into the displayed mean).
            method_rows = defaultdict(list)
            for r in rows:
                if r["target_model"] != target or r["dataset"] != ds:
                    continue
                try:
                    bnd = float(r["Bound_I"])
                    dr = float(r["|MdR|"])
                    if math.isnan(bnd) or math.isnan(dr):
                        continue
                except (ValueError, KeyError):
                    continue
                xs.append(bnd)
                ys.append(dr)
                m = parse_method(r.get("Label", ""))
                if m in {meth for meth, _ in METHOD_TABLE}:
                    method_rows[m].append(r)

            method_means = []
            for m, group in method_rows.items():
                # Per-method aggregation: average |dR| (and rho) across multiple
                # checkpoints labelled the same way (e.g., several GPTQ-4bit).
                rt = sum(float(g["rho_T"]) for g in group) / len(group)
                rp = sum(float(g["rho_P"]) for g in group) / len(group)
                # Drop GPTQ rows where rho_T / rho_P ratio exceeds 1.5x
                # (data-quality filter; matches _collect()).
                if m.startswith("GPTQ") and min(abs(rt), abs(rp)) > 0:
                    if max(abs(rt), abs(rp)) / min(abs(rt), abs(rp)) > 1.5:
                        continue
                dr_avg = sum(float(g["|MdR|"]) for g in group) / len(group)
                method_means.append(dr_avg)

            cell_dr[(model_cfg["short"], ds)] = (
                sum(method_means) / len(method_means)
                if method_means else float("nan"))
            cell_rs[(model_cfg["short"], ds)] = spearman(xs, ys)

    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(
        r"\caption{Per-benchmark mean empirical risk gap "
        r"$|\Delta\mathcal{R}|$ (averaged over per-method aggregated "
        r"quantization rows, matching the per-model body tables) and "
        r"per-benchmark mean Spearman $r_s(\mathcal{B},|\Delta\mathcal{R}|)$ "
        r"(over raw variants, matching the figure's per-subplot $r_s$) "
        r"across seven 8B model families. GSM8K's small "
        r"$|\Delta\mathcal{R}|$---driven by long teacher-forced answer spans "
        r"diluting per-token loss---yields a low signal-to-noise ratio that "
        r"depresses its rank correlation.}")
    L.append(r"\label{tab:gsm8k_outlier}")
    L.append(r"\setlength{\tabcolsep}{5pt}")
    L.append(r"\begin{tabular}{l " + "c" * len(SUMMARY_BENCHMARKS) + "}")
    L.append(r"\toprule")
    L.append(r" & \multicolumn{" + str(len(SUMMARY_BENCHMARKS)) +
             r"}{c}{\textbf{Empirical risk gap} $\boldsymbol{|\Delta\mathcal{R}|}$} \\")
    L.append(r"\cmidrule(lr){2-" + str(len(SUMMARY_BENCHMARKS) + 1) + "}")
    L.append(r"\multicolumn{1}{c}{\textbf{Model}} & " + " & ".join(
        rf"\textbf{{{DS_DISPLAY[b]}}}" for b in SUMMARY_BENCHMARKS) + r" \\")
    L.append(r"\midrule")

    for model_cfg in MODELS:
        short, disp = model_cfg["short"], _short_display(model_cfg["display"])
        cells = []
        for ds in SUMMARY_BENCHMARKS:
            v = cell_dr[(short, ds)]
            cells.append(f"{v:.4f}" if not math.isnan(v) else "--")
        L.append(disp + " & " + " & ".join(cells) + r" \\")

    L.append(r"\midrule")

    def _hl(ds, cell, bold=False):
        """Highlight GSM8K summary cells (red background) to flag the outlier;
        optionally bold the value to emphasize the dominant cause (small |dR|)."""
        if ds == "gsm8k":
            inner = r"\textbf{" + cell + "}" if bold else cell
            return r"\cellcolor{red!18} " + inner
        return cell

    mean_dr = []
    for ds in SUMMARY_BENCHMARKS:
        vals = [cell_dr[(m["short"], ds)] for m in MODELS]
        vals = [v for v in vals if not math.isnan(v)]
        mean_dr.append(sum(vals) / len(vals) if vals else float("nan"))
    L.append(r"\textbf{Mean} $\boldsymbol{|\Delta\mathcal{R}|}$ & " + " & ".join(
        _hl(ds, f"{v:.4f}" if not math.isnan(v) else "--", bold=True)
        for ds, v in zip(SUMMARY_BENCHMARKS, mean_dr)) + r" \\")
    L.append(r"\midrule")

    mean_rs = []
    for ds in SUMMARY_BENCHMARKS:
        vals = [cell_rs[(m["short"], ds)] for m in MODELS]
        vals = [v for v in vals if not math.isnan(v)]
        mean_rs.append(sum(vals) / len(vals) if vals else float("nan"))
    L.append(r"\textbf{Mean} $\boldsymbol{r_s}$ & " + " & ".join(
        _hl(ds, f"{v:.3f}" if not math.isnan(v) else "--")
        for ds, v in zip(SUMMARY_BENCHMARKS, mean_rs)) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


def build_lipschitz_table(rows):
    """Per-model Lipschitz constants K_feat (model-specific) and K_pred (= sqrt(2)
    universally). Pulled from the BF16-vs-FP16 row of each model (K is constant
    per model since it depends only on the head H_T's token-embedding spread)."""
    per_model = {}
    for model_cfg in MODELS:
        target = model_cfg["path"]
        for r in rows:
            if r["target_model"] != target:
                continue
            try:
                kf = float(r["K_f"])
                kp = float(r["K_p"])
                if not (math.isnan(kf) or math.isnan(kp)):
                    per_model[model_cfg["short"]] = (kf, kp,
                                                     model_cfg["display"])
                    break
            except (ValueError, KeyError):
                continue

    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(
        r"\caption{Empirical Lipschitz constants of the PRISM bound "
        r"(Theorem~\ref{thm:unified_bound}) for each evaluated model. "
        r"$K_{\mathrm{pred}} = \sqrt{2}$ holds universally "
        r"(Appendix~\ref{app:head_bound}, simplex polarization), independent "
        r"of model. $K_{\mathrm{feat}}$ is empirical per model, depending on "
        r"the spread of token embeddings in $H_T$; for the 8B-scale models "
        r"studied here, $K_{\mathrm{feat}}$ ranges from $\approx 0.93$ "
        r"(Mistral family) to $\approx 3.46$ (Qwen3-Base). Since "
        r"$K_{\mathrm{feat}}$ is a constant per model, it scales the bound's "
        r"magnitude but does not affect the within-model rank correlation "
        r"that PRISM is calibrated to.}")
    L.append(r"\label{tab:lipschitz_constants}")
    L.append(r"\setlength{\tabcolsep}{8pt}")
    L.append(r"\begin{tabular}{l cc}")
    L.append(r"\toprule")
    L.append(r"\multicolumn{1}{c}{\textbf{Model}} & \multicolumn{1}{c}{$\boldsymbol{K_{\mathrm{feat}}}$} & \multicolumn{1}{c}{$\boldsymbol{K_{\mathrm{pred}}}$} \\")
    L.append(r"\midrule")

    for model_cfg in MODELS:
        short = model_cfg["short"]
        if short not in per_model:
            continue
        kf, kp, disp = per_model[short]
        kp_str = (r"$\sqrt{2}$" if abs(kp - math.sqrt(2)) < 1e-6
                  else f"{kp:.3f}")
        L.append(f"{_short_display(disp)} & {kf:.2f} & {kp_str} " + r"\\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


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
            stem = f"table_{model_cfg['short']}_{group_cfg['name']}_bound"
            out_path = OUT_DIR / f"{stem}.tex"
            out_path.write_text(table)
            print(f"Saved → {out_path}")
            stems.append(stem)

    summary_table = build_summary_table(rows)
    summary_stem = "table_summary_gsm8k_outlier"
    summary_path = OUT_DIR / f"{summary_stem}.tex"
    summary_path.write_text(summary_table)
    print(f"Saved → {summary_path}")
    stems.append(summary_stem)

    lipschitz_table = build_lipschitz_table(rows)
    lipschitz_stem = "table_lipschitz_constants"
    lipschitz_path = OUT_DIR / f"{lipschitz_stem}.tex"
    lipschitz_path.write_text(lipschitz_table)
    print(f"Saved → {lipschitz_path}")
    stems.append(lipschitz_stem)

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
