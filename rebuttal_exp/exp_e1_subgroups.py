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
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "out" / "E1"
QCSV = REPO / "exp_result" / "quantization" / "quantization_merged_slim.csv"

GGUF = {"Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"}
BENCHMARKS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]
FAM_ID = {"llama": "meta-llama/Meta-Llama-3.1-8B",
          "qwen": "Qwen/Qwen3-8B-Base"}


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
                cells.append(f"{b:+.3f} / {c:+.3f} (n={len(grp)})")
            md.append(f"| {bench} | " + " | ".join(cells) + " |")
        md.append("")

        # ── bootstrap: is the pooled mean rs difference significant? ──
        # (variants resampled within each cell, 5000 reps, seed 0 — the
        # draft's "differences are within Spearman noise" claim source)
        rng = random.Random(0)
        cells_d = {}
        for bench in BENCHMARKS:
            sub = [r for r in rows if r["dataset"] == bench
                   and not math.isnan(float(r["|MdR|"]))]
            cells_d[bench] = [(float(r["bound_I"]), 1 - float(r["cka"]),
                               float(r["|MdR|"])) for r in sub]

        def mean_diff(cd, benches):
            ds = []
            for bench in benches:
                pts = cd[bench]
                ds.append(spearman([p[0] for p in pts], [p[2] for p in pts])
                          - spearman([p[1] for p in pts],
                                     [p[2] for p in pts]))
            return statistics.mean(ds)

        # ── bound-gauge comparison: W=I vs W=W_N (App C.1 family) ──────
        # Bound_W is the ROTATION-INVARIANT gauge — the invariance class
        # CKA/SVCCA live in — and comes from FULL paper features (no
        # token-cap issue). Its gsm8k signal is carried by gamma_W
        # (deviation of the optimal rotation from identity, expressed
        # through the head) — an isometry-violation reading (App C.2).
        gauges = {}
        for r in csv.DictReader(open(QCSV)):
            if r["target_model"] == FAM_ID[fam]:
                try:
                    gauges[(r["Label"], r["dataset"])] = (
                        float(r["Bound_I"]), float(r["Bound_W"]))
                except ValueError:
                    pass
        md += ["Bound-gauge comparison (full-feature CSV; same E1 subset):",
               "", "| benchmark | B_I | B_W | 1-CKA (E1 feats) |",
               "|---|---|---|---|"]
        mi, mw, mc = [], [], []
        for bench in BENCHMARKS:
            sub = [r for r in rows if r["dataset"] == bench
                   and not math.isnan(float(r["|MdR|"]))]
            d = [float(r["|MdR|"]) for r in sub]
            bi = spearman([gauges[(r["label"], bench)][0] for r in sub], d)
            bw = spearman([gauges[(r["label"], bench)][1] for r in sub], d)
            ck = spearman([1 - float(r["cka"]) for r in sub], d)
            mi.append(bi); mw.append(bw); mc.append(ck)
            md.append(f"| {bench} | {bi:+.3f} | {bw:+.3f} | {ck:+.3f} |")
        md.append(f"| **mean** | **{statistics.mean(mi):+.3f}** "
                  f"| **{statistics.mean(mw):+.3f}** "
                  f"| **{statistics.mean(mc):+.3f}** |")
        md.append("")

        md.append(f"Bootstrap (B − 1-CKA mean rs, 5000 reps, {fam}):")
        for tag, benches in (("all-5", BENCHMARKS),
                             ("ex-gsm8k", [b for b in BENCHMARKS
                                           if b != "gsm8k"])):
            obs = mean_diff(cells_d, benches)
            boots = []
            for _ in range(5000):
                samp = {b: [cells_d[b][rng.randrange(len(cells_d[b]))]
                            for _ in cells_d[b]] for b in benches}
                boots.append(mean_diff(samp, benches))
            boots.sort()
            lo = boots[int(0.025 * len(boots))]
            hi = boots[int(0.975 * len(boots))]
            verdict = ("significant" if hi < 0 or lo > 0
                       else "NOT significant (CI covers 0)")
            md.append(f"- {tag}: {obs:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] "
                      f"— {verdict}")
        md.append("")
    (OUT / "subgroup_analysis.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
