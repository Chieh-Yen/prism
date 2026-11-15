#!/usr/bin/env python3
"""Forgetting of the anchor-draw runs of launch_anchor_draw.sh (Appendix I).

For every anchor-set size n and every run below OUT_ROOT/n<n>_off<offset>/,
prints the downstream forgetting (mean |delta_risk| over the five benchmarks at
step 300, as protocol.read_forgetting defines it), then the mean, population
sd, minimum and maximum over the runs of each n.

Example
-------
    python scripts/response/forgetting/anchor_draw_table.py --out_root out/anchor_draw
"""

from __future__ import annotations

import argparse
import re
import statistics as st
from pathlib import Path

import protocol as P


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", required=True, type=Path)
    args = ap.parse_args()
    rows = []
    for metrics in sorted(args.out_root.glob(
            "n*_off*/trace/lam1000/seed*/llama-3.1-8b/truthfulqa/" + P.METRICS_FILE)):
        m = re.match(r"n(\d+)_off(\d+)", metrics.parts[len(args.out_root.parts)])
        seed = int(metrics.parent.parent.parent.name[len("seed"):])
        forgetting = P.read_forgetting(metrics.parent)
        if m and forgetting is not None:
            rows.append((int(m.group(1)), int(m.group(2)), seed, forgetting))
    print("| n | offset | seed | forgetting |\n|--:|--:|--:|--:|")
    for n, off, seed, f in sorted(rows, key=lambda r: (-r[0], r[1], r[2])):
        print(f"| {n} | {off} | {seed} | {f:.3f} |")
    print("\n| n | runs | mean ± population sd | min | max |\n|--:|--:|--:|--:|--:|")
    for n in sorted({r[0] for r in rows}, reverse=True):
        v = [r[3] for r in rows if r[0] == n]
        print(f"| {n} | {len(v)} | {st.mean(v):.3f} ± {st.pstdev(v):.3f} | {min(v):.3f} | {max(v):.3f} |")


if __name__ == "__main__":
    main()
