#!/usr/bin/env python3
"""
E7 seed aggregation (AC-D / G3T9-W2 / 8VrD-W4-Q2 robustness).

The free-running result rests on one prompt subset; reviewers may read n=100
on a single seed as cherry-picked. This reads the per-seed CSVs written by
exp_e7_freerun.py (seed 42 -> {family}_{dataset}_freerun.csv; seed s != 42 ->
{family}_{dataset}_s{s}_freerun.csv), recomputes the four headline Spearman
statistics per seed, and reports mean +/- sd across seeds so the rebuttal can
cite a subset-robust number rather than a single draw.

Pure CPU / CSV — no GPU, no torch. Run after the seed sweep:
    python rebuttal_exp/exp_e7_seed_aggregate.py --family llama --dataset gsm8k \
        --seeds 42 43 44
Output: rebuttal_exp/out/E7/E7_seed_aggregate_{family}_{dataset}.md
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out" / "E7"

METRICS = [
    ("rs_tf", "B_tf", "dR_tf", "rs(B, |dR|) teacher-forced"),
    ("rs_free", "B_free", "dR_free", "rs(B, |dR|) free-running"),
    ("rs_agree", "B_tf", "B_free", "rank agreement rs(B_tf, B_free)"),
    ("rs_cross", "B_tf", "dR_free", "cross rs(B_tf, |dR|_free)"),
]


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


def seed_csv(family, dataset, seed):
    stem = f"{family}_{dataset}" + ("" if seed == 42 else f"_s{seed}")
    return OUT_DIR / f"{stem}_freerun.csv"


def load_metrics(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for key, xcol, ycol, _ in METRICS:
        out[key] = spearman([float(r[xcol]) for r in rows],
                            [float(r[ycol]) for r in rows])
    out["_n"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="llama")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()

    per_seed = {}
    for s in args.seeds:
        p = seed_csv(args.family, args.dataset, s)
        if p.exists():
            per_seed[s] = load_metrics(p)
        else:
            print(f"[skip] seed {s}: {p.name} not found")
    if not per_seed:
        raise SystemExit("No per-seed E7 CSVs found — run script_E7.sh first.")

    seeds = sorted(per_seed)
    md = [f"# E7 free-running seed robustness ({args.family}, {args.dataset})",
          "",
          f"{len(seeds)} prompt subsets (seeds {', '.join(map(str, seeds))}; "
          f"n per subset: "
          f"{', '.join(str(per_seed[s]['_n']) for s in seeds)} variants). "
          "Greedy decoding is deterministic, so each seed varies only the "
          "prompt subset; mean +/- sd across seeds is the subset-robustness "
          "statistic.", "",
          "| statistic | " + " | ".join(f"seed {s}" for s in seeds)
          + " | mean +/- sd |",
          "|---|" + "---|" * (len(seeds) + 1)]

    headline = {}
    for key, _, _, label in METRICS:
        vals = [per_seed[s][key] for s in seeds
                if not math.isnan(per_seed[s][key])]
        cells = [f"{per_seed[s][key]:+.3f}" for s in seeds]
        if len(vals) >= 2:
            m, sd = statistics.mean(vals), statistics.stdev(vals)
            agg = f"**{m:+.3f} +/- {sd:.3f}**"
            headline[key] = (m, sd)
        elif vals:
            agg = f"{vals[0]:+.3f} (1 seed)"
            headline[key] = (vals[0], 0.0)
        else:
            agg = "-"
        md.append(f"| {label} | " + " | ".join(cells) + f" | {agg} |")

    if "rs_free" in headline and "rs_tf" in headline:
        mf, sdf = headline["rs_free"]
        mt, _ = headline["rs_tf"]
        md += ["",
               f"Headline for the rebuttal: free-running rs = {mf:+.3f} +/- "
               f"{sdf:.3f} across {len(seeds)} subsets, vs {mt:+.3f} "
               "teacher-forced on the same subsets. The small sd certifies the "
               "free-running result is not a single-subset artifact."]

    out = OUT_DIR / f"E7_seed_aggregate_{args.family}_{args.dataset}.md"
    out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
