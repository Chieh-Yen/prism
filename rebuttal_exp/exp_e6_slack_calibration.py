#!/usr/bin/env python3
"""
E6 — Slack quantification + leave-one-out isotonic calibration.

Answers pCi8-W2, G3T9-W4, 8VrD-W2/Q1 from existing data only (zero GPU):

  1. Slack distribution  s = B / |dR|  per (family, benchmark) cell,
     pooled per family and overall (median / IQR / min / max, log10 stats).
  2. Two-step slack attribution using the empirical-K bound (EBound):
         log10(B/|dR|) = log10(B/EBound)  +  log10(EBound/|dR|)
     The first term is the Lipschitz-constant relaxation (K_f vs empirical
     grad norm), the second the alignment/triangle/Jensen remainder.
     Also reported per quantization tier (Q8_0 ... Q2_K, INT8, NF4, ...).
  3. Leave-one-out isotonic (PAV) calibration B -> |dR| per cell:
     MAE in nats vs a predict-the-mean baseline, plus the operational rule
     "calibrated prediction < eps  =>  true |dR| < eps" precision/recall
     for eps in {0.05, 0.1, 0.5}.

Input : exp_result/quantization/quantization_merged_slim.csv  (repo data)
Output: rebuttal_exp/out/E6/slack_summary.csv
        rebuttal_exp/out/E6/calibration.csv
        rebuttal_exp/out/E6/E6_results.md

Stdlib only (csv/json/math/statistics) — same style as analyze_bound_tightness.py.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "exp_result" / "quantization" / "quantization_merged_slim.csv"
OUT_DIR = Path(__file__).resolve().parent / "out" / "E6"

# Main-text protocol: identity alignment W = I.
BOUND_COL, EBOUND_COL, RISK_COL = "Bound_I", "EBound_I", "|MdR|"

# Order used for the per-tier slack table.
TIER_ORDER = ["FP16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K",
              "INT8", "NF4", "FP4"]

EPSILONS = [0.05, 0.1, 0.5]
MIN_RISK = 1e-6          # rows below this |dR| get slack reported separately


# ----------------------------------------------------------------------
# Small numeric helpers (no numpy)
# ----------------------------------------------------------------------
def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def quantiles(xs, qs=(0.25, 0.5, 0.75)):
    if not xs:
        return [float("nan")] * len(qs)
    s = sorted(xs)
    out = []
    for q in qs:
        idx = q * (len(s) - 1)
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        out.append(s[lo] + (s[hi] - s[lo]) * (idx - lo))
    return out


def tier_of(label: str) -> str:
    """Map 'BF16 vs Q4_K_M' / 'BF16 vs GPTQ(...)' to a short tier tag."""
    tag = label.split(" vs ", 1)[-1]
    return "GPTQ" if tag.startswith("GPTQ") else tag


# ----------------------------------------------------------------------
# Isotonic regression (pool-adjacent-violators) + LOO prediction
# ----------------------------------------------------------------------
def pav_fit(points):
    """points: list of (x, y) — returns list of blocks (x_lo, x_hi, value)."""
    pts = sorted(points)
    blocks = [[x, x, y, 1.0] for x, y in pts]        # lo, hi, mean, weight
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][2] > blocks[i + 1][2] + 1e-15:  # violator: merge
            lo = blocks[i][0]
            hi = blocks[i + 1][1]
            w = blocks[i][3] + blocks[i + 1][3]
            m = (blocks[i][2] * blocks[i][3] + blocks[i + 1][2] * blocks[i + 1][3]) / w
            blocks[i:i + 2] = [[lo, hi, m, w]]
            i = max(i - 1, 0)
        else:
            i += 1
    return [(b[0], b[1], b[2]) for b in blocks]


def pav_predict(blocks, x):
    """Piecewise-constant prediction with clamping outside the fitted range."""
    if not blocks:
        return float("nan")
    if x <= blocks[0][1]:
        return blocks[0][2]
    for lo, hi, v in blocks:
        if lo <= x <= hi:
            return v
    # between blocks: linear interpolation of block values by midpoint
    for (l0, h0, v0), (l1, h1, v1) in zip(blocks, blocks[1:]):
        if h0 < x < l1:
            t = (x - h0) / max(l1 - h0, 1e-12)
            return v0 + t * (v1 - v0)
    return blocks[-1][2]


def loo_isotonic(cell_points):
    """cell_points: [(B, dR, label)] — returns [(label, B, true, pred)]."""
    preds = []
    for i, (b, r, lab) in enumerate(cell_points):
        train = [(bb, rr) for j, (bb, rr, _) in enumerate(cell_points) if j != i]
        blocks = pav_fit(train)
        preds.append((lab, b, r, pav_predict(blocks, b)))
    return preds


# ----------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(CSV_PATH)))

    # Keep every (target, dataset) cell; parse numerics once.
    cells = defaultdict(list)       # (target, dataset) -> row dicts
    for r in rows:
        rec = {
            "target": r["target_model"],
            "dataset": r["dataset"],
            "label": r["Label"],
            "tier": tier_of(r["Label"]),
            "B": to_float(r[BOUND_COL]),
            "EB": to_float(r[EBOUND_COL]),
            "dR": to_float(r[RISK_COL]),
        }
        if math.isnan(rec["B"]) or math.isnan(rec["dR"]):
            continue
        cells[(rec["target"], rec["dataset"])].append(rec)

    # ── 1+2. Slack distribution and attribution ────────────────────────
    slack_rows = []
    tier_slacks = defaultdict(list)          # tier -> log10 slack list
    tier_ksteps = defaultdict(list)          # tier -> log10(B/EB)
    tier_rsteps = defaultdict(list)          # tier -> log10(EB/dR)
    family_slacks = defaultdict(list)
    near_zero = 0
    all_logs = []

    for (target, dataset), recs in sorted(cells.items()):
        logs = []
        for rec in recs:
            if rec["dR"] < MIN_RISK:
                near_zero += 1
                continue
            s = rec["B"] / rec["dR"]
            logs.append(math.log10(s))
            all_logs.append(math.log10(s))
            family_slacks[target].append(math.log10(s))
            tier_slacks[rec["tier"]].append(math.log10(s))
            if rec["EB"] > 0:
                tier_ksteps[rec["tier"]].append(math.log10(rec["B"] / rec["EB"]))
                tier_rsteps[rec["tier"]].append(math.log10(rec["EB"] / rec["dR"]))
        if not logs:
            continue
        q1, med, q3 = quantiles([10 ** v for v in logs])
        slack_rows.append({
            "target": target, "dataset": dataset, "n": len(logs),
            "slack_median": med, "slack_q1": q1, "slack_q3": q3,
            "slack_min": 10 ** min(logs), "slack_max": 10 ** max(logs),
            "log10_mean": statistics.mean(logs),
            "log10_sd": statistics.pstdev(logs),
        })

    # ── 3. LOO isotonic calibration per cell ───────────────────────────
    calib_rows = []
    for (target, dataset), recs in sorted(cells.items()):
        pts = [(r["B"], r["dR"], r["label"]) for r in recs if r["dR"] >= 0]
        if len(pts) < 5:
            continue
        preds = loo_isotonic(pts)
        abs_err = [abs(t - p) for (_, _, t, p) in preds]
        mean_r = statistics.mean([t for (_, _, t, _) in preds])
        base_err = [abs(t - mean_r) for (_, _, t, _) in preds]
        row = {
            "target": target, "dataset": dataset, "n": len(preds),
            "risk_min": min(t for (_, _, t, _) in preds),
            "risk_max": max(t for (_, _, t, _) in preds),
            "mae_isotonic": statistics.mean(abs_err),
            "mae_predict_mean": statistics.mean(base_err),
        }
        for eps in EPSILONS:
            tp = sum(1 for (_, _, t, p) in preds if p < eps and t < eps)
            fp = sum(1 for (_, _, t, p) in preds if p < eps and t >= eps)
            fn = sum(1 for (_, _, t, p) in preds if p >= eps and t < eps)
            row[f"prec@{eps}"] = tp / (tp + fp) if (tp + fp) else float("nan")
            row[f"rec@{eps}"] = tp / (tp + fn) if (tp + fn) else float("nan")
        calib_rows.append(row)

    # ── Write CSVs ──────────────────────────────────────────────────────
    with open(OUT_DIR / "slack_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(slack_rows[0].keys()))
        w.writeheader()
        w.writerows(slack_rows)
    with open(OUT_DIR / "calibration.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(calib_rows[0].keys()))
        w.writeheader()
        w.writerows(calib_rows)

    # ── Markdown report ─────────────────────────────────────────────────
    md = ["# E6 results — slack quantification + LOO isotonic calibration", ""]
    md.append(f"Rows analysed: {sum(len(v) for v in cells.values())} "
              f"({near_zero} with |dR| < {MIN_RISK} excluded from slack stats).")
    q1, med, q3 = quantiles([10 ** v for v in all_logs])
    md += ["",
           f"**Pooled slack B/|dR| (all families x benchmarks):** "
           f"median {med:.1f}x, IQR [{q1:.1f}, {q3:.1f}]x, "
           f"log10 mean {statistics.mean(all_logs):.2f} "
           f"(sd {statistics.pstdev(all_logs):.2f}).", ""]

    md.append("## Per-family slack (log10 units)\n")
    md.append("| family | n | median x | log10 mean | log10 sd |")
    md.append("|---|---|---|---|---|")
    for fam, logs in sorted(family_slacks.items()):
        md.append(f"| {fam} | {len(logs)} | {10**statistics.median(logs):.1f} "
                  f"| {statistics.mean(logs):.2f} | {statistics.pstdev(logs):.2f} |")

    md.append("\n## Slack attribution per tier "
              "(median log10; total = K-step + remainder)\n")
    md.append("| tier | n | log10 slack | K-step log10(B/EB) | remainder log10(EB/dR) |")
    md.append("|---|---|---|---|---|")
    tiers = TIER_ORDER + sorted(t for t in tier_slacks if t not in TIER_ORDER)
    for t in tiers:
        if t not in tier_slacks:
            continue
        md.append(f"| {t} | {len(tier_slacks[t])} "
                  f"| {statistics.median(tier_slacks[t]):.2f} "
                  f"| {statistics.median(tier_ksteps[t]):.2f} "
                  f"| {statistics.median(tier_rsteps[t]):.2f} |")

    md.append("\n## LOO isotonic calibration per (family, benchmark)\n")
    md.append("| family | benchmark | n | risk range (nats) | MAE isotonic "
              "| MAE mean-baseline | prec@0.1 | rec@0.1 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in calib_rows:
        md.append(f"| {r['target'].split('/')[-1]} | {r['dataset']} | {r['n']} "
                  f"| [{r['risk_min']:.4f}, {r['risk_max']:.4f}] "
                  f"| {r['mae_isotonic']:.4f} | {r['mae_predict_mean']:.4f} "
                  f"| {r['prec@0.1']:.2f} | {r['rec@0.1']:.2f} |")

    mae_iso = statistics.mean([r["mae_isotonic"] for r in calib_rows])
    mae_base = statistics.mean([r["mae_predict_mean"] for r in calib_rows])
    md += ["",
           f"**Aggregate LOO MAE:** isotonic {mae_iso:.4f} nats vs "
           f"predict-mean {mae_base:.4f} nats "
           f"({100 * (1 - mae_iso / mae_base):.0f}% reduction).", ""]

    (OUT_DIR / "E6_results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
