#!/usr/bin/env python3
"""
Generate LaTeX tables for the regularization experiments (post gradient-bug
fix; both trace-norm and replay).

Data sources:
  regularization_exp/exp_result/regularization/<lam>/...        (trace norm)
  regularization_exp/exp_result/regularization_replay/<lam>/... (replay CE)

Two modes:

  compare (default)
    Side-by-side comparison of methods at specific λ values.
    Default configs: baseline (replay λ=0.0) + replay λ=0.01 + trace λ=1.0
    (each near per-method sweep |ΔR|-best).
    One table per (model × ft_task); rows = (eval benchmark × config).
    Outputs:
      paper/tables/regularization/table_compare_<model>_<ft_task>.tex
      paper/tables/regularization/preview_compare.tex

  sweep
    Per-method tables showing all λ values discovered on disk.
    One table per (method × model × ft_task).
    Outputs:
      paper/tables/regularization/table_sweep_<method>_<model>_<ft_task>.tex
      paper/tables/regularization/preview_sweep_<method>.tex

Highlights (both modes):
  Ω < 0.95 → light red on (Ω, δ, B, |ΔR|)
  Ω < 0.80 → deep  red on (Ω, δ, B, |ΔR|)
  γ = 0   → cyan "$0$" (LoRA frozen lm_head ⇒ γ structurally 0)
  Within each benchmark block, smallest |ΔR| → bold (and corresponding Ω).

Usage:
  # Compare mode (default; recommended pair):
  python gen_latex_table_regularization.py
  python gen_latex_table_regularization.py --mode compare \
      --config "replay:0.0:baseline" \
      --config "replay:0.01:Replay \$\\lambda{=}0.01\$" \
      --config "trace:0.5:Trace \$\\lambda{=}0.5\$"

  # Sweep mode (full λ sweep, one method at a time):
  python gen_latex_table_regularization.py --mode sweep
  python gen_latex_table_regularization.py --mode sweep --methods trace
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
IN_ROOT = ROOT / "regularization_exp" / "exp_result"
OUT_DIR = ROOT / "paper" / "tables" / "regularization"

# ── Method registry ───────────────────────────────────────────────────────
METHODS = {
    "trace": {
        "subdir": "regularization",
        "display": r"Trace-norm shape regularizer ($\lambda \cdot (1 - \Omega_I)$)",
        "short_display": "trace",
    },
    "replay": {
        "subdir": "regularization_replay",
        "display": r"Replay-CE baseline ($\lambda \cdot \mathrm{CE}_{\mathrm{ref}}$)",
        "short_display": "replay",
    },
}

# ── Models / tasks ────────────────────────────────────────────────────────
MODELS = [
    {"short": "llama", "display": "Llama-3.1-8B"},
    {"short": "qwen",  "display": "Qwen3-8B-Base"},
]

FT_TASKS = [
    {"short": "bbq",         "display": "BBQ"},
    {"short": "truthfulqa",  "display": "TruthfulQA"},
]

DS_DISPLAY = {
    "arc": "ARC", "mmlu": "MMLU", "squad": "SQuAD",
    "triviaqa": "TriviaQA", "gsm8k": "GSM8K",
    "bbq": "BBQ (self)", "truthfulqa": "TruthfulQA (self)",
}
DOWNSTREAM = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]

ALIGN_STEP = 300

# Default compare-mode configurations (recommended pair).
# Each method at its sweep |ΔR|-best:
#   - replay λ=0.01  (lowest 4-cell mean |ΔR| in replay sweep; replay
#                     degrades for λ > 0.01)
#   - trace  λ=1.0   (lowest 4-cell mean |ΔR| in trace sweep; sweep max)
DEFAULT_COMPARE_CONFIGS = [
    ("replay", "0.0",  r"no reg"),
    ("replay", "0.01", r"replay ($\lambda{=}0.01$)"),
    ("trace",  "1.0",  r"trace ($\lambda{=}1.0$)"),
]

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
# λ discovery + data loading
# ═══════════════════════════════════════════════════════════════════
def discover_lambdas(method: str) -> list[str]:
    """Return all λ subdir names under IN_ROOT/<method.subdir>/, sorted asc."""
    method_root = IN_ROOT / METHODS[method]["subdir"]
    if not method_root.is_dir():
        return []
    lams: list[str] = []
    for child in method_root.iterdir():
        if not child.is_dir():
            continue
        try:
            float(child.name)
        except ValueError:
            continue
        lams.append(child.name)
    return sorted(lams, key=float)


def load_run(method: str, lam: str, model_short: str, ft_task: str):
    p = (IN_ROOT / METHODS[method]["subdir"] / lam / model_short
         / f"prism_forgetting_metrics_{ft_task}.json")
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def metrics_at_step(run: dict, eval_task: str, step: int = ALIGN_STEP):
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
    if out["delta_risk"] is not None:
        out["delta_risk"] = abs(out["delta_risk"])
    return out


# ═══════════════════════════════════════════════════════════════════
# Cell formatting
# ═══════════════════════════════════════════════════════════════════
def fmt_cell(col_key: str, val, gamma_zero: bool, omega_level,
             bold: bool = False, underline: bool = False):
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
    elif underline:
        num = r"\underline{" + num + "}"
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
# Table assembly (shared body builder)
# ═══════════════════════════════════════════════════════════════════
def _render_body(rows_per_block, model_short, ft_task, include_target):
    """rows_per_block: list of (method, lam, label) tuples, one per sub-row.
    Returns LaTeX body lines."""
    downstream = [d for d in DOWNSTREAM if d != ft_task]
    eval_tasks = ([ft_task] + downstream) if include_target else downstream
    L: list[str] = []
    for di, ev in enumerate(eval_tasks):
        if di > 0:
            L.append(r"\midrule")

        ms = {(method, lam): metrics_at_step(
                  load_run(method, lam, model_short, ft_task), ev)
              for method, lam, _ in rows_per_block}

        # Symmetric ranking: per block, mark 1st (bold) and 2nd (underline)
        # for both |ΔR| (smallest first) and Ω (largest first), selected
        # independently. Bold and underline are mutually exclusive on a cell.
        dr_sorted = sorted(
            ((k, m["delta_risk"]) for k, m in ms.items()
             if m is not None and m.get("delta_risk") is not None
             and not (isinstance(m["delta_risk"], float)
                      and math.isnan(m["delta_risk"]))),
            key=lambda x: x[1])
        om_sorted = sorted(
            ((k, m["omega"]) for k, m in ms.items()
             if m is not None and m.get("omega") is not None
             and not (isinstance(m["omega"], float)
                      and math.isnan(m["omega"]))),
            key=lambda x: -x[1])
        best_dr_key   = dr_sorted[0][0] if len(dr_sorted) >= 1 else None
        second_dr_key = dr_sorted[1][0] if len(dr_sorted) >= 2 else None
        best_om_key   = om_sorted[0][0] if len(om_sorted) >= 1 else None
        second_om_key = om_sorted[1][0] if len(om_sorted) >= 2 else None

        for ri, (method, lam, label) in enumerate(rows_per_block):
            m = ms[(method, lam)]
            ds_cell = DS_DISPLAY.get(ev, ev) if ri == 0 else ""

            if m is None:
                blanks = " & ".join(["--"] * len(COLUMNS))
                L.append(f"{ds_cell} & {label} & {blanks} " + r"\\")
                continue

            ov = m.get("omega")
            lvl = omega_level(ov)
            gz = (m.get("gamma") is not None and abs(m.get("gamma")) < 1e-9)

            cells = []
            for ck, _ in COLUMNS:
                bold = underline = False
                if ck == "delta_risk":
                    if (method, lam) == best_dr_key:
                        bold = True
                    elif (method, lam) == second_dr_key:
                        underline = True
                elif ck == "omega":
                    if (method, lam) == best_om_key:
                        bold = True
                    elif (method, lam) == second_om_key:
                        underline = True
                cells.append(fmt_cell(ck, m.get(ck), gz, lvl,
                                      bold=bold, underline=underline))
            vals_str = " & ".join(cells)
            L.append(f"{ds_cell} & {label} & {vals_str} " + r"\\")
    return L


def _wrap_table(body_rows, caption, label, second_col_header):
    col_spec = "l l " + "r" * len(COLUMNS)
    header_cols = " & ".join(h for _, h in COLUMNS)
    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{" + caption + r"}")
    L.append(r"\label{" + label + r"}")
    L.append(r"\resizebox{0.95\textwidth}{!}{%")
    L.append(r"\begin{tabular}{" + col_spec + "}")
    L.append(r"\toprule")
    L.append(r"Dataset & " + second_col_header + r" & " + header_cols + r" \\")
    L.append(r"\midrule")
    L.extend(body_rows)
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    L.append(r"\end{table}")
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════
# Compare mode
# ═══════════════════════════════════════════════════════════════════
def build_compare_table(model_cfg, ft_cfg, configs, include_target):
    body_rows = _render_body(configs, model_cfg["short"], ft_cfg["short"],
                             include_target)
    config_summary = ", ".join(label for _, _, label in configs)
    caption = (
        r"Regularization comparison for \textbf{" + model_cfg["display"]
        + r"} fine-tuned on \textbf{" + ft_cfg["display"] + r"} under identity "
        r"alignment ($W{=}I$). Rows group by evaluation benchmark; each "
        r"benchmark block lists metrics at step " + str(ALIGN_STEP)
        + r" for: " + config_summary + r". "
        r"Bold / underline: 1st / 2nd-best smallest $|\Delta\mathcal{R}|$ and largest $\Omega$ per block."
    )
    label = f"tab:reg_compare_{model_cfg['short']}_{ft_cfg['short']}"
    return _wrap_table(body_rows, caption, label, "Config")


def parse_configs(specs):
    configs = []
    for s in specs:
        parts = s.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Invalid config spec {s!r}: expected method:lambda:label")
        method, lam, label = parts
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r} in {s!r}")
        configs.append((method, lam, label))
    return configs


def run_compare_mode(args):
    configs = parse_configs(args.config) if args.config else DEFAULT_COMPARE_CONFIGS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== compare mode | configs ===")
    for method, lam, label in configs:
        print(f"  {method:6s} λ={lam:5s} → {label}")

    stems: list[str] = []
    for model_cfg in MODELS:
        for ft_cfg in FT_TASKS:
            any_run = any(load_run(m, l, model_cfg["short"], ft_cfg["short"])
                          is not None for m, l, _ in configs)
            if not any_run:
                print(f"  [skip] {model_cfg['short']}/{ft_cfg['short']} "
                      f"(no runs found)")
                continue
            tex = build_compare_table(model_cfg, ft_cfg, configs,
                                      args.include_target)
            stem = f"table_compare_{model_cfg['short']}_{ft_cfg['short']}"
            out = OUT_DIR / f"{stem}.tex"
            out.write_text(tex)
            print(f"  saved {out.relative_to(ROOT)}")
            stems.append(stem)

    if stems:
        _write_preview(stems, OUT_DIR / "preview_compare.tex")


# ═══════════════════════════════════════════════════════════════════
# Sweep mode (preserves earlier per-method full-λ behavior)
# ═══════════════════════════════════════════════════════════════════
def build_sweep_table(method, lams, model_cfg, ft_cfg, include_target):
    rows_per_block = [(method, lam, f"$\\lambda{{=}}{lam}$") for lam in lams]
    body_rows = _render_body(rows_per_block, model_cfg["short"],
                             ft_cfg["short"], include_target)
    lam_str = ", ".join(lams)
    caption = (
        METHODS[method]["display"] + r" decomposition for \textbf{"
        + model_cfg["display"] + r"} fine-tuned on \textbf{"
        + ft_cfg["display"] + r"} under identity alignment ($W{=}I$). "
        r"Rows group by evaluation benchmark; each benchmark lists metrics "
        r"at step " + str(ALIGN_STEP) + r" for $\lambda \in \{"
        + lam_str + r"\}$. "
        r"Bold / underline: 1st / 2nd-best smallest $|\Delta\mathcal{R}|$ and largest $\Omega$ per benchmark."
    )
    label = f"tab:reg_sweep_{method}_{model_cfg['short']}_{ft_cfg['short']}"
    return _wrap_table(body_rows, caption, label, r"$\lambda$")


def run_sweep_mode(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    methods = args.methods or list(METHODS.keys())

    for method in methods:
        lams = discover_lambdas(method)
        if not lams:
            print(f"[skip method={method}] no λ subdirs under "
                  f"{IN_ROOT / METHODS[method]['subdir']}")
            continue
        print(f"\n=== sweep mode | method={method} | "
              f"λ ∈ {{{', '.join(lams)}}} ===")

        stems: list[str] = []
        for model_cfg in MODELS:
            for ft_cfg in FT_TASKS:
                any_run = any(
                    load_run(method, lam, model_cfg["short"],
                             ft_cfg["short"]) is not None for lam in lams)
                if not any_run:
                    print(f"  [skip] {model_cfg['short']}/{ft_cfg['short']} "
                          f"(no runs)")
                    continue
                tex = build_sweep_table(method, lams, model_cfg, ft_cfg,
                                        args.include_target)
                stem = (f"table_sweep_{method}_"
                        f"{model_cfg['short']}_{ft_cfg['short']}")
                out = OUT_DIR / f"{stem}.tex"
                out.write_text(tex)
                print(f"  saved {out.relative_to(ROOT)}")
                stems.append(stem)

        if stems:
            _write_preview(stems, OUT_DIR / f"preview_sweep_{method}.tex")


# ═══════════════════════════════════════════════════════════════════
# Preview wrapper
# ═══════════════════════════════════════════════════════════════════
def _write_preview(stems, preview_path):
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
    body = "\n\n\\clearpage\n\n".join(f"\\input{{{s}}}" for s in stems)
    preview_path.write_text(preamble + body + "\n" + r"\end{document}" + "\n")
    print(f"  saved {preview_path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["compare", "sweep"],
                        default="compare",
                        help="compare: side-by-side methods at specific λ "
                             "(default). sweep: full λ sweep per method.")
    parser.add_argument("--config", action="append", metavar="METHOD:LAM:LABEL",
                        help="(compare mode) Add a config row. Repeatable. "
                             "Default: baseline + replay 0.01 + trace 1.0.")
    parser.add_argument("--methods", nargs="+",
                        choices=list(METHODS.keys()),
                        help="(sweep mode) Methods to render. Default: all.")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--include-target", dest="include_target",
                     action="store_true",
                     help="Include the fine-tuning target task as eval row.")
    grp.add_argument("--no-include-target", dest="include_target",
                     action="store_false",
                     help="Exclude the fine-tuning target task (default).")
    parser.set_defaults(include_target=INCLUDE_TARGET_TASK)
    args = parser.parse_args()

    if args.mode == "compare":
        run_compare_mode(args)
    else:
        run_sweep_mode(args)


if __name__ == "__main__":
    main()
