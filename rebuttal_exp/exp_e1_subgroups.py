#!/usr/bin/env python3
"""
E1 subgroup analysis — why does the full bound trail similarity scores on
pooled rank correlation? (Mechanism check for AC-C / pCi8-W3 / G3T9-W3.)

Hypothesis = the paper's own Sec. 5.5 sentence: gamma engages only on GGUF
k-quant tiers, so POOLING heterogeneous gamma-regimes adds ordering variance
while preserving validity. Test: recompute rs(B, |dR|) and rs(1-CKA, |dR|)
within fixed-head-protocol subgroups (GGUF-only / non-GGUF) from E1's
metrics CSV. If the gap closes within subgroups, the pooled deficit is the
cross-protocol gamma effect — the same certified head term that makes B
LEAD on SQuAD where head damage actually expresses.

Zero GPU; reads rebuttal_exp/out/E1/{family}_metrics.csv.
Output: rebuttal_exp/out/E1/subgroup_analysis.md
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "E1"

GGUF = {"Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"}
BENCHMARKS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]


def rankdata(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and v[order[j]] == v[order[i]]:
            j += 1
        for k in range(i, j):
            r[order[k]] = (i + j + 1) / 2
        i = j
    return r


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


def main():
    md = ["# E1 subgroup analysis — fixed-head-protocol rank correlations",
          "",
          "rs(score, |dR|) within head-protocol subgroups. If B ties 1-CKA "
          "inside each subgroup, the pooled deficit is the cross-protocol "
          "gamma-pooling effect of Sec. 5.5 (head term certified for "
          "validity; only expresses in |dR| where head damage matters).", ""]
    for fam in ("llama", "qwen"):
        path = OUT / f"{fam}_metrics.csv"
        if not path.exists():
            md.append(f"({fam}: metrics CSV missing — run E1 first)")
            continue
        rows = list(csv.DictReader(open(path)))
        md += [f"## {fam}", "",
               "| benchmark | ALL: B / 1-CKA | GGUF-only: B / 1-CKA "
               "| non-GGUF: B / 1-CKA |", "|---|---|---|---|"]
        for bench in BENCHMARKS:
            sub = [r for r in rows if r["dataset"] == bench
                   and not math.isnan(float(r["|MdR|"]))]
            cells = []
            for name, grp in (
                    ("ALL", sub),
                    ("GGUF", [r for r in sub
                              if r["label"].split(" vs ")[-1] in GGUF]),
                    ("nonGGUF", [r for r in sub
                                 if r["label"].split(" vs ")[-1] not in GGUF])):
                if len(grp) < 4:
                    cells.append("n<4")
                    continue
                d = [float(r["|MdR|"]) for r in grp]
                b = spearman([float(r["bound_I"]) for r in grp], d)
                c = spearman([1 - float(r["cka"]) for r in grp], d)
                cells.append(f"{b:+.2f} / {c:+.2f} (n={len(grp)})")
            md.append(f"| {bench} | " + " | ".join(cells) + " |")
        md.append("")
    (OUT / "subgroup_analysis.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
