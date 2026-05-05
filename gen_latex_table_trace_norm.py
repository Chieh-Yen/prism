#!/usr/bin/env python3
"""
Generate LaTeX tables for the trace-norm forgetting experiment.

Framing: target = base model.  Proxies = fine-tuned models at
λ ∈ {0.0, 0.1, 0.5}.  Metrics in the JSON are already between the
fine-tuned (proxy) and the base (target).

One LaTeX table per (model × ft_task):
  Rows: (eval benchmark, λ) blocks — each benchmark has 3 sub-rows for λ.
  Cols: λ | ρ_T | ρ_P | Ω | δ | γ | B | |ΔR|
  Metric values are taken from the last checkpoint at or before step=300.

Highlights (inherited from gen_latex_table.py):
  Ω < 0.95 → light red on (Ω, δ, B, |ΔR|)
  Ω < 0.80 → deep  red on (Ω, δ, B, |ΔR|)
  γ = 0   → cyan "$0$" (LoRA config without lm_head → γ structurally 0)

Within each benchmark (3 rows for λ∈{0.0,0.1,0.5}), the row with the smallest
|ΔR| gets |ΔR| and the corresponding Ω rendered in bold.

Output: paper/tables/regularization/table_trace_norm_{model}_{ft_task}.tex
         paper/tables/regularization/preview_trace_norm.tex  (\\input aggregator)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IN_DIR = ROOT / "exp_result" / "regularization"
OUT_DIR = ROOT / "paper" / "tables" / "regularization"

MODELS = [
    {"short": "llama", "display": "Llama-3.1-8B"},
    {"short": "qwen",  "display": "Qwen3-8B-Base"},
]

FT_TASKS = [
    {"short": "bbq",         "display": "BBQ"},
    {"short": "lima",        "display": "LIMA"},
    {"short": "no_robots",   "display": "No-Robots"},
    {"short": "social_iqa",  "display": "Social IQA"},
    {"short": "truthfulqa",  "display": "TruthfulQA"},
]

DS_DISPLAY = {
    "arc": "ARC", "mmlu": "MMLU", "squad": "SQuAD",
    "triviaqa": "TriviaQA", "gsm8k": "GSM8K",
    "bbq": "BBQ (self)", "lima": "LIMA (self)",
    "no_robots": "No-Robots (self)", "social_iqa": "Social IQA (self)",
    "truthfulqa": "TruthfulQA (self)",
}
DOWNSTREAM = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]

LAMS = ["0.0", "0.1", "0.5"]
ALIGN_STEP = 300

# Whether to include the fine-tuning (target) task itself as one of the
# evaluation benchmarks.  Default: False (only downstream tasks are shown).
INCLUDE_TARGET_TASK = False

OMEGA_LIGHT = 0.95
OMEGA_DEEP  = 0.80

RED_COLS = {"omega", "delta", "bound_total", "delta_risk"}

COLUMNS = [
    ("rho_T",        r"$\rho_T$"),
    ("rho_P",        r"$\rho_P$"),
    ("omega",        r"$\Omega$"),
    ("delta",        r"$\delta$"),
    ("gamma",        r"$\gamma$"),
    ("bound_total",  r"$\mathcal{B}$"),
    ("delta_risk",   r"$|\Delta\mathcal{R}|$"),
]


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════
def load_run(lam: str, model_short: str, ft_task: str):
    p = IN_DIR / lam / model_short / f"prism_forgetting_metrics_{ft_task}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def metrics_at_step(run: dict, eval_task: str, step: int = ALIGN_STEP):
    """Return metrics from the last checkpoint at or before `step`."""
    if run is None:
        return None
    best = None
    for cp in run["checkpoints"]:
        if cp["step"] > step:
            continue
        if best is None or cp["step"] > best["step"]:
            best = cp
    if best is None:
        return None
    t = best["tasks"].get(eval_task)
    if not t:
        return None
    out = {"step": best["step"]}
    for k in ("omega", "rho_T", "rho_P", "delta", "gamma",
             "bound_total", "loss_T", "loss_P", "delta_risk"):
        out[k] = t.get(k)
    # normalize to |ΔR|
    if out["delta_risk"] is not None:
        out["delta_risk"] = abs(out["delta_risk"])
    return out


# ═══════════════════════════════════════════════════════════════════
# Cell formatting (gen_latex_table.py style)
# ═══════════════════════════════════════════════════════════════════
def fmt_cell(col_key: str, val, gamma_zero: bool, omega_level, bold: bool = False):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "--"
    if col_key == "gamma" and gamma_zero:
        return r"\cellcolor{cyan!12} $0$"
    if col_key in ("rho_T", "rho_P"):
        num = f"{val:.2f}"
    else:
        num = f"{val:.4f}"
    if bold:
        num = r"\textbf{" + num + "}"
    if col_key in RED_COLS and omega_level:
        color = "red!35" if omega_level == "deep" else "red!18"
        return r"\cellcolor{" + color + "} " + num
    return num


def omega_level(omega_val):
    if omega_val is None:
        return None
    if omega_val < OMEGA_DEEP:
        return "deep"
    if omega_val < OMEGA_LIGHT:
        return "light"
    return None


# ═══════════════════════════════════════════════════════════════════
# Body + wrapper
# ═══════════════════════════════════════════════════════════════════
def _render_body_rows(model_short, ft_task, include_target=INCLUDE_TARGET_TASK):
    downstream = [d for d in DOWNSTREAM if d != ft_task]
    eval_tasks = ([ft_task] + downstream) if include_target else downstream
    L = []
    for di, ev in enumerate(eval_tasks):
        if di > 0:
            L.append(r"\midrule")

        # pre-compute metrics for the 3 λ rows and pick the one with
        # minimum |ΔR|; |ΔR| and Ω on that row get bolded
        ms = {lam: metrics_at_step(load_run(lam, model_short, ft_task), ev)
              for lam in LAMS}
        best_lam = None
        best_dr = math.inf
        for lam in LAMS:
            m = ms[lam]
            if m is None:
                continue
            dr = m.get("delta_risk")
            if dr is None or (isinstance(dr, float) and math.isnan(dr)):
                continue
            if dr < best_dr:
                best_dr = dr
                best_lam = lam

        for li, lam in enumerate(LAMS):
            m = ms[lam]
            ds_cell = DS_DISPLAY.get(ev, ev) if li == 0 else ""
            lam_disp = f"$\\lambda{{=}}{lam}$"

            if m is None:
                # missing run — emit placeholder row
                blanks = " & ".join(["--"] * len(COLUMNS))
                L.append(f"{ds_cell} & {lam_disp} & {blanks} " + r"\\")
                continue

            ov = m.get("omega")
            lvl = omega_level(ov)
            gz = (m.get("gamma") is not None
                  and abs(m.get("gamma")) < 1e-9)
            is_best = (lam == best_lam)

            cells = []
            for ck, _ in COLUMNS:
                bold = is_best and ck in ("omega", "delta_risk")
                cells.append(fmt_cell(ck, m.get(ck), gz, lvl, bold=bold))
            vals_str = " & ".join(cells)
            L.append(f"{ds_cell} & {lam_disp} & {vals_str} " + r"\\")
    return L


def _wrap_regular_table(body_rows, caption, label, col_spec, header_cols):
    # Scale target set below \textwidth so the resize factor matches
    # gen_latex_table.py's main tables (which are naturally wider because
    # they include a Family column); this keeps font sizes visually aligned
    # instead of letting \resizebox stretch the narrower table more.
    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{" + caption + r"}")
    L.append(r"\label{" + label + r"}")
    L.append(r"\resizebox{0.85\textwidth}{!}{%")
    L.append(r"\begin{tabular}{" + col_spec + "}")
    L.append(r"\toprule")
    L.append(r"\textbf{Dataset} & \textbf{$\lambda$} & " + header_cols + r" \\")
    L.append(r"\midrule")
    L.extend(body_rows)
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    L.append(r"\end{table}")
    return "\n".join(L)


def build_table(model_cfg, ft_cfg, include_target=INCLUDE_TARGET_TASK):
    display_model = model_cfg["display"]
    display_ft    = ft_cfg["display"]
    short_model   = model_cfg["short"]
    short_ft      = ft_cfg["short"]

    body_rows = _render_body_rows(short_model, short_ft, include_target=include_target)

    col_spec = "l l " + "r" * len(COLUMNS)
    header_cols = " & ".join(rf"\textbf{{{h}}}" for _, h in COLUMNS)

    caption = (
        r"Trace-norm forgetting decomposition for \textbf{" + display_model
        + r"} fine-tuned on \textbf{" + display_ft + r"} under identity "
        r"alignment ($W{=}I$).  Rows group by evaluation benchmark; each "
        r"benchmark lists metrics at step " + str(ALIGN_STEP)
        + r" for $\lambda \in \{0.0, 0.1, 0.5\}$ (target $=$ base model).  "
        r"Within each benchmark the row with the smallest $|\Delta\mathcal{R}|$ "
        r"is marked: both $|\Delta\mathcal{R}|$ and the corresponding $\Omega$ "
        r"are in bold."
    )
    label = f"tab:trace_norm_{short_model}_{short_ft}"
    return _wrap_regular_table(body_rows, caption, label, col_spec, header_cols)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--include-target", dest="include_target",
                       action="store_true",
                       help="Include the fine-tuning target task as an "
                            "evaluation row (default: off).")
    group.add_argument("--no-include-target", dest="include_target",
                       action="store_false",
                       help="Exclude the fine-tuning target task from the "
                            "evaluation rows (default).")
    parser.set_defaults(include_target=INCLUDE_TARGET_TASK)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stems = []
    for model_cfg in MODELS:
        for ft_cfg in FT_TASKS:
            # skip (model × ft) combinations that are completely missing
            # across all three λ values (e.g. llama missing lima/no_robots at
            # λ=1.0 — but we only use 0.0/0.1/0.5 here so all should be OK).
            any_run = False
            for lam in LAMS:
                if load_run(lam, model_cfg["short"], ft_cfg["short"]) is not None:
                    any_run = True
                    break
            if not any_run:
                print(f"Skipping {model_cfg['short']}/{ft_cfg['short']} "
                      f"(no runs found)")
                continue
            tex = build_table(model_cfg, ft_cfg,
                              include_target=args.include_target)
            stem = f"table_trace_norm_{model_cfg['short']}_{ft_cfg['short']}"
            out = OUT_DIR / f"{stem}.tex"
            out.write_text(tex)
            print(f"Saved → {out}")
            stems.append(stem)

    preamble = (
        r"\documentclass[11pt]{article}" "\n"
        r"\usepackage[utf8]{inputenc}" "\n"
        r"\usepackage[T1]{fontenc}" "\n"
        r"\usepackage{booktabs,amsmath,amssymb,graphicx}" "\n"
        r"\usepackage[table]{xcolor}" "\n"
        r"\usepackage[margin=1.5cm,landscape]{geometry}" "\n"
        r"\pagestyle{empty}" "\n"
        r"\begin{document}" "\n"
    )
    body = "\n\n".join(f"\\input{{{stem}}}" for stem in stems)
    preview_path = OUT_DIR / "preview_trace_norm.tex"
    preview_path.write_text(preamble + body + "\n" + r"\end{document}" + "\n")
    print(f"Saved → {preview_path}")


if __name__ == "__main__":
    main()
