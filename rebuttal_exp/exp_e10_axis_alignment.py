#!/usr/bin/env python3
"""
E10 — Do the shape-regularizer failure cells align with PRISM's own axis
diagnosis?  (8VrD-W3: Table 22 negative cells; zero GPU, existing data only.)

Hypothesis under test (Google-Doc rule: use ONLY if the data supports it):
    The trace (shape) regularizer helps exactly where the no-reg drift is
    shape-dominated; the cells where it fails are those whose no-reg drift
    PRISM itself diagnoses as least shape-driven — small 1-Omega (nothing
    for a shape penalty to fix) and/or a large scale-axis share.

Per (model, ft_task) cell, merged across result roots:
    benefit(lambda) = 100 * (m0 - m_l) / m0,   m = mean downstream |dR|
    axis stats from the lambda=0 run at the analysis step:
        one_minus_omega  (mean over the 5 downstream tasks)
        scale_share = scale / (scale + shape)      [gamma == 0, frozen head]
        drho_signed = rho_P - rho_T                [scale-axis direction]

Outputs per-cell table, pooled + within-model Spearman at the
best-covered lambda, and an explicit honest verdict.

Inputs (first root that has a given (lambda, model, task) file wins):
    exp_result/regularization/{lam}/{model}/prism_forgetting_metrics_{task}.json
    forgetting_exp_log_safety_clear/{lam}/{model}/...
    forgetting_exp_log_safety/{lam}/{model}/...
  (lambda dir names are normalized: "0" == "0.0", "1" == "1.0")

Output: rebuttal_exp/out/E10/E10_results.md  (+ cells.csv)

Stdlib only.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOTS = [
    REPO / "exp_result" / "regularization",
    REPO / "forgetting_exp_log_safety_clear",
    REPO / "forgetting_exp_log_safety",
]
OUT_DIR = Path(__file__).resolve().parent / "out" / "E10"

DOWNSTREAM = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]
ANALYSIS_STEP = 300          # paper: analysis at step 300


# ----------------------------------------------------------------------
def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y)
             if not (math.isnan(a) or math.isnan(b))]
    if len(pairs) < 3:
        return float("nan"), len(pairs)
    xs, ys = zip(*pairs)
    rx, ry = rankdata(list(xs)), rankdata(list(ys))
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return (num / den if den > 0 else float("nan")), len(pairs)


def checkpoint_at(data, step):
    """Checkpoint dict at `step`, or the latest one <= step."""
    best = None
    for ck in data["checkpoints"]:
        if ck["step"] <= step and (best is None or ck["step"] > best["step"]):
            best = ck
    return best


def mean_downstream(ck, key):
    vals = [ck["tasks"][t][key] for t in DOWNSTREAM if t in ck["tasks"]]
    return statistics.mean(vals) if vals else float("nan")


def discover():
    """index[(lam_norm, model, ft_task)] = Path (first root wins)."""
    index = {}
    for root in ROOTS:
        if not root.is_dir():
            continue
        for lam_dir in root.iterdir():
            if not lam_dir.is_dir():
                continue
            try:
                lam = f"{float(lam_dir.name):g}"
            except ValueError:
                continue
            for model_dir in lam_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                for f in model_dir.glob("prism_forgetting_metrics_*.json"):
                    ft = f.stem.replace("prism_forgetting_metrics_", "")
                    key = (lam, model_dir.name, ft)
                    index.setdefault(key, f)
    return index


# ----------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = discover()
    lambdas = sorted({k[0] for k in index}, key=float)
    if "0" not in lambdas:
        raise SystemExit("No lambda=0 (no-reg) runs found in any root.")
    reg_lams = [l for l in lambdas if l != "0"]
    cells = sorted({(m, t) for (l, m, t) in index if l == "0"})

    rows = []
    for model, ft in cells:
        d0 = json.load(open(index[("0", model, ft)]))
        ck0 = checkpoint_at(d0, ANALYSIS_STEP)
        if ck0 is None:
            continue
        m0 = mean_downstream(ck0, "delta_risk")

        one_minus_omega, scale_share, drho = [], [], []
        for t in DOWNSTREAM:
            if t not in ck0["tasks"]:
                continue
            tk = ck0["tasks"][t]
            one_minus_omega.append(1.0 - tk["omega"])
            tot = tk["scale"] + tk["shape"]
            scale_share.append(tk["scale"] / tot if tot > 0 else float("nan"))
            drho.append(tk["rho_P"] - tk["rho_T"])

        row = {
            "model": model, "ft_task": ft, "step": ck0["step"],
            "noreg_mean_dR": m0,
            "one_minus_omega": statistics.mean(one_minus_omega),
            "scale_share_pct": 100 * statistics.mean(scale_share),
            "drho_mean": statistics.mean(drho),
            "target_loss_noreg": ck0["tasks"].get(ft, {}).get("loss_P"),
        }

        best = None
        for lam in reg_lams:
            key = (lam, model, ft)
            col = f"benefit%@{lam}"
            row[col] = None
            if key not in index:
                continue
            dl = json.load(open(index[key]))
            ckl = checkpoint_at(dl, ANALYSIS_STEP)
            if ckl is None or ckl["step"] != ck0["step"] or m0 <= 0:
                continue
            ml = mean_downstream(ckl, "delta_risk")
            row[col] = 100 * (m0 - ml) / m0
            if best is None or row[col] > best[1]:
                best = (lam, row[col],
                        mean_downstream(ckl, "omega") - mean_downstream(ck0, "omega"),
                        ckl["tasks"].get(ft, {}).get("loss_P"))
        if best:
            row["best_lambda"] = best[0]
            row["benefit%@best"] = best[1]
            row["omega_lift@best"] = best[2]
            row["target_loss@best"] = best[3]
        rows.append(row)

    if not rows:
        raise SystemExit("No cells found.")

    # lambda with maximal coverage drives the headline correlations
    coverage = {l: sum(1 for r in rows if r.get(f"benefit%@{l}") is not None)
                for l in reg_lams}
    lam_star = max(coverage, key=lambda l: (coverage[l], float(l)))
    key = f"benefit%@{lam_star}"

    def corr(rows_subset, stat):
        b = [r[key] for r in rows_subset if r.get(key) is not None]
        s = [r[stat] for r in rows_subset if r.get(key) is not None]
        return spearman(b, s)

    pooled_shape, n_pooled = corr(rows, "one_minus_omega")
    pooled_scale, _ = corr(rows, "scale_share_pct")
    per_model = {}
    for m in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == m]
        per_model[m] = (corr(sub, "one_minus_omega"), corr(sub, "scale_share_pct"))

    # ── Report ──────────────────────────────────────────────────────────
    md = ["# E10 results — regularizer benefit vs PRISM axis diagnosis", ""]
    md.append(f"Analysis step {ANALYSIS_STEP}; downstream mean over "
              f"{', '.join(DOWNSTREAM)}; benefit% > 0 = trace reduces "
              f"forgetting vs no-reg. Coverage per lambda: "
              + ", ".join(f"{l}: {coverage[l]}/{len(rows)}" for l in reg_lams)
              + f". Headline stats use lambda*={lam_star}.")
    md.append("")
    hdr = ("| model | FT task | no-reg mean dR | "
           + " | ".join(f"b%@{l}" for l in reg_lams)
           + " | b%@best (lam) | 1-Omega | scale share % | drho |")
    md.append(hdr)
    md.append("|" + "---|" * (hdr.count("|") - 1))
    for r in sorted(rows, key=lambda r: (r["model"], r["ft_task"])):
        lam_txt = " | ".join(
            f"{r[f'benefit%@{l}']:+.1f}" if r.get(f"benefit%@{l}") is not None
            else "-" for l in reg_lams)
        best_txt = (f"{r['benefit%@best']:+.1f} ({r['best_lambda']})"
                    if r.get("benefit%@best") is not None else "-")
        md.append(f"| {r['model']} | {r['ft_task']} | {r['noreg_mean_dR']:.4f} "
                  f"| {lam_txt} | {best_txt} | {r['one_minus_omega']:.4f} "
                  f"| {r['scale_share_pct']:.2f} | {r['drho_mean']:+.2f} |")

    md += ["", f"## Spearman at lambda*={lam_star} (n={n_pooled} cells)", "",
           f"- pooled  benefit vs (1-Omega)   : rs = {pooled_shape:+.2f}",
           f"- pooled  benefit vs scale share : rs = {pooled_scale:+.2f}"]
    for m, ((rs_o, n_o), (rs_s, _)) in per_model.items():
        md.append(f"- {m:<6s} benefit vs (1-Omega)   : rs = {rs_o:+.2f} (n={n_o}); "
                  f"vs scale share: rs = {rs_s:+.2f}")

    md += ["", "## Verdict", ""]
    within = [v[0][0] for v in per_model.values() if not math.isnan(v[0][0])]
    mean_within = statistics.mean(within) if within else float("nan")
    if mean_within >= 0.5:
        md.append(
            f"SUPPORTED (mean within-model rs = {mean_within:+.2f}): cells with "
            f"larger unregularized shape drift (1-Omega) gain more from the shape "
            f"regularizer, and the failure cells are those PRISM diagnoses as "
            f"least shape-driven (small 1-Omega, i.e. nothing for a shape penalty "
            f"to fix) or with elevated scale share. Usable for the 8VrD-W3 "
            f"axis-specificity response — cite per-cell numbers, flag exceptions "
            f"explicitly.")
    elif mean_within >= 0.2:
        md.append(
            f"WEAK SUPPORT (mean within-model rs = {mean_within:+.2f}): "
            f"directionally consistent but not decisive; use per-cell contrasts "
            f"only, no general claim.")
    else:
        md.append(
            f"NOT SUPPORTED (mean within-model rs = {mean_within:+.2f}): do NOT "
            f"use the axis-specificity reframing; fall back to honest all-cell "
            f"reporting + scoping.")
    md += ["",
           "Caveats for the rebuttal text: (i) reconcile metric/step/lambda with "
           "the exact Table 22 protocol before quoting; (ii) this analysis uses "
           "answer-only delta_risk mean over the 5 downstream benchmarks at "
           f"step {ANALYSIS_STEP}; (iii) coverage is partial for some lambdas "
           "(see header)."]

    (OUT_DIR / "E10_results.md").write_text("\n".join(md))
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(OUT_DIR / "cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
