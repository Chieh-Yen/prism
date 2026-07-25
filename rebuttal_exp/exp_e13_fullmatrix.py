#!/usr/bin/env python3
"""
E13 — Full 20-cell correlation matrix for the paper's forgetting grid
(8VrD-W4(1) and AC-E: "report the full correlation matrix with all-cell
aggregates, not the Llama-only 0.831").

Zero GPU. For the paper grid — 2 models (llama, qwen) x 2 FT tasks
(truthfulqa, bbq) x 5 downstream benchmarks (arc/mmlu/squad/triviaqa/gsm8k)
— read the lambda=0 (no-reg) trajectory JSONs and compute, per cell,
Spearman rs between the PRISM bound and the measured risk gap across
training checkpoints:

    rs( bound_total_t , |delta_risk|_t ),   t = checkpoints of the run.

Deliverables:
  1. the 20-cell matrix + aggregates (mean/median overall and per model);
  2. a PROTOCOL-SCAN checksum: the paper reports mean rs = 0.831 +/- 0.0722
     over the Llama 2x5 grid and negative Qwen-BBQ cells (~ -0.34 / -0.66,
     Fig. 7). We scan the small protocol space
        risk key   in {delta_risk (answer-only), delta_risk_full}
        window     in {steps <= 300 (paper analysis window), all steps}
        include t0 in {yes, no}
     and report which protocol reproduces the paper's Llama-grid mean; the
     headline table uses that protocol (choice disclosed in the output).
     If NO protocol reproduces 0.831 closely, the local tree is a different
     round from the paper's — then present these numbers as an INDEPENDENT
     RERUN aggregate (same framing as E10; do not mix with Fig. 7 numbers).

Inputs (first root that has a given (model, task) lambda-0 file wins):
    exp_result/regularization/0.0/{model}/prism_forgetting_metrics_{task}.json
    forgetting_exp_log_safety_clear/{0,0.0}/...
    forgetting_exp_log_safety/{0,0.0}/...

Output: rebuttal_exp/out/E13/E13_results.md  (+ matrix.csv)

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
OUT_DIR = Path(__file__).resolve().parent / "out" / "E13"

MODELS = ["llama", "qwen"]
FT_TASKS = ["truthfulqa", "bbq"]           # the paper's 2x2 grid
DOWNSTREAM = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]
LAM0_DIRS = ["0.0", "0"]

# Paper anchors for the checksum (Sec. 5 / Fig. 7).
PAPER_LLAMA_MEAN = 0.831
PAPER_QWEN_BBQ_NEG = (-0.34, -0.66)


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
        return float("nan")
    xs, ys = zip(*pairs)
    rx, ry = rankdata(list(xs)), rankdata(list(ys))
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


def find_lam0(model, task):
    for root in ROOTS:
        for lam in LAM0_DIRS:
            p = root / lam / model / f"prism_forgetting_metrics_{task}.json"
            if p.exists():
                return p
    return None


def cell_rs(data, bench, risk_key, max_step, include_t0):
    bs, rs_ = [], []
    for ck in data["checkpoints"]:
        if ck["step"] > max_step:
            continue
        if not include_t0 and ck["step"] == 0:
            continue
        t = ck["tasks"].get(bench)
        if t is None:
            continue
        bs.append(t["bound_total"])
        rs_.append(abs(t[risk_key]))
    return spearman(bs, rs_), len(bs)


# ----------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = {}
    for m in MODELS:
        for t in FT_TASKS:
            p = find_lam0(m, t)
            if p is None:
                raise SystemExit(f"missing lambda=0 run for ({m}, {t})")
            runs[(m, t)] = json.load(open(p))
            print(f"[load] {m}/{t}: {p}")

    protocols = [
        (risk_key, max_step, inc0)
        for risk_key in ("delta_risk", "delta_risk_full")
        for max_step in (300, 10**9)
        for inc0 in (False, True)
    ]

    # ── Protocol scan (checksum vs the paper's Llama-grid mean) ────────
    scan_lines = ["| risk key | window | t0 | llama mean | qwen-bbq min two |",
                  "|---|---|---|---|---|"]
    best, best_gap = None, float("inf")
    for risk_key, max_step, inc0 in protocols:
        llama_vals, qwen_bbq = [], []
        for (m, t), data in runs.items():
            for b in DOWNSTREAM:
                r, _ = cell_rs(data, b, risk_key, max_step, inc0)
                if m == "llama":
                    llama_vals.append(r)
                if m == "qwen" and t == "bbq":
                    qwen_bbq.append(r)
        lm = statistics.mean([v for v in llama_vals if not math.isnan(v)])
        qb = sorted(v for v in qwen_bbq if not math.isnan(v))[:2]
        gap = abs(lm - PAPER_LLAMA_MEAN)
        win = "<=300" if max_step == 300 else "all"
        scan_lines.append(f"| {risk_key} | {win} | {'y' if inc0 else 'n'} "
                          f"| {lm:+.3f} | {', '.join(f'{v:+.2f}' for v in qb)} |")
        if gap < best_gap:
            best, best_gap = (risk_key, max_step, inc0), gap

    risk_key, max_step, inc0 = best
    win = "<=300" if max_step == 300 else "all steps"
    reproduced = best_gap <= 0.05

    # ── Headline matrix under the selected protocol ─────────────────────
    rows, all_vals = [], []
    per_model = {m: [] for m in MODELS}
    for (m, t), data in runs.items():
        for b in DOWNSTREAM:
            r, n = cell_rs(data, b, risk_key, max_step, inc0)
            rows.append({"model": m, "ft_task": t, "benchmark": b,
                         "rs": r, "n_checkpoints": n})
            if not math.isnan(r):
                all_vals.append(r)
                per_model[m].append(r)

    with open(OUT_DIR / "matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = ["# E13 — full 20-cell correlation matrix (paper grid, lambda=0)",
          "",
          f"Protocol selected by checksum: risk={risk_key}, window={win}, "
          f"include step-0={'yes' if inc0 else 'no'} "
          f"(llama-grid mean gap to paper's 0.831: {best_gap:.3f}).",
          "",
          f"**Checksum verdict: {'REPRODUCED' if reproduced else 'NOT reproduced'}** "
          + ("— cite these aggregates as the paper grid's own numbers."
             if reproduced else
             "— the local lambda=0 tree is a DIFFERENT ROUND from the "
             "paper's Fig. 7 runs. Present these aggregates as an "
             "independent-rerun extension (same framing as E10 §4); do "
             "NOT mix with Fig. 7 numbers in the same sentence."),
          "",
          "| model | FT task | " + " | ".join(DOWNSTREAM) + " |",
          "|---|---|" + "---|" * len(DOWNSTREAM)]
    for m in MODELS:
        for t in FT_TASKS:
            cells = [f"{r['rs']:+.2f}" for r in rows
                     if r["model"] == m and r["ft_task"] == t]
            md.append(f"| {m} | {t} | " + " | ".join(cells) + " |")

    md += ["",
           "## Aggregates (fill 8VrD-W4(1) / AC-E)",
           "",
           f"- all 20 cells: mean rs = {statistics.mean(all_vals):+.3f}, "
           f"median = {statistics.median(all_vals):+.3f}, "
           f"min = {min(all_vals):+.2f}, {sum(v < 0 for v in all_vals)} negative",
           f"- llama (10 cells): mean {statistics.mean(per_model['llama']):+.3f} "
           f"(paper anchor 0.831)",
           f"- qwen  (10 cells): mean {statistics.mean(per_model['qwen']):+.3f}",
           "",
           "## Protocol scan (checksum detail)", ""] + scan_lines

    (OUT_DIR / "E13_results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
