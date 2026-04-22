#!/usr/bin/env python3
"""Check which model×dataset cells have mean |MdR| < threshold (low signal)."""

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path("/Users/cylin/github/prism/quantization_merged_slim.csv")
DS_ORDER = ["arc", "mmlu", "gsm8k", "squad", "triviaqa", "fineweb_edu", "wikitext"]

BASE = {
    "Qwen/Qwen2.5-7B", "Qwen/Qwen3-8B-Base",
    "meta-llama/Meta-Llama-3.1-8B",
    # "mistralai/Mistral-7B-v0.3",  # excluded from current study
    "mistralai/Ministral-3-8B-Base-2512",
}
INSTRUCT = {
    "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3",  # excluded from current study
    "mistralai/Ministral-3-8B-Instruct-2512",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}


def to_float(v):
    try:
        return float(v)
    except:
        return float("nan")


def short(name):
    return name.split("/")[-1]


def load():
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        item = dict(r)
        item["|MdR|"] = to_float(r.get("|MdR|", ""))
        out.append(item)
    return out


def main():
    rows = load()

    by_md = defaultdict(list)
    for r in rows:
        v = r["|MdR|"]
        if not math.isnan(v):
            by_md[(r["target_model"], r["dataset"])].append(v)

    # ── Table: mean |MdR| heatmap, Base ──
    for label, model_set in [("BASE", BASE), ("INSTRUCT", INSTRUCT)]:
        models = sorted(model_set, key=lambda m: short(m))

        print(f"\n{'='*120}")
        print(f"  {label}: mean |MdR| per model×dataset    (! = mean<0.01,  !! = mean<0.005)")
        print(f"{'='*120}")

        hdr = f"  {'Model':<35}"
        for ds in DS_ORDER:
            hdr += f" {ds[:8]:>9}"
        hdr += f" {'| mean':>9}"
        print(f"\n{hdr}")
        print(f"  {'-'*(35 + 10*len(DS_ORDER) + 10)}")

        ds_vals = defaultdict(list)
        for m in models:
            line = f"  {short(m):<35}"
            row_vals = []
            for ds in DS_ORDER:
                vals = by_md.get((m, ds), [])
                if not vals:
                    line += f" {'—':>9}"
                    continue
                mn = statistics.mean(vals)
                row_vals.append(mn)
                ds_vals[ds].append(mn)
                flag = "!!" if mn < 0.005 else ("!" if mn < 0.01 else "")
                line += f" {mn:>8.4f}{flag}"
            if row_vals:
                line += f" {statistics.mean(row_vals):>9.4f}"
            print(line)

        # Column mean
        line = f"  {'mean':<35}"
        all_col = []
        for ds in DS_ORDER:
            if ds_vals[ds]:
                v = statistics.mean(ds_vals[ds])
                all_col.append(v)
                line += f" {v:>9.4f}"
            else:
                line += f" {'':>9}"
        if all_col:
            line += f" {statistics.mean(all_col):>9.4f}"
        print(f"  {'-'*(35 + 10*len(DS_ORDER) + 10)}")
        print(line)

    # ── Detailed list: all cells with mean < 0.01 ──
    print(f"\n\n{'='*120}")
    print(f"  ALL CELLS with mean |MdR| < 0.01  (sorted by mean)")
    print(f"{'='*120}")
    print(f"\n  {'Model':<35} {'type':>5} {'Dataset':<12} {'n':>3} {'mean':>9} {'std':>9} {'min':>9} {'max':>9} {'range':>9}")
    print(f"  {'-'*100}")

    flagged = []
    for (m, ds), vals in sorted(by_md.items()):
        mn = statistics.mean(vals)
        if mn < 0.01:
            tag = "Base" if m in BASE else "Inst"
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            flagged.append((mn, m, ds, tag, len(vals), sd, min(vals), max(vals)))

    for mn, m, ds, tag, n, sd, lo, hi in sorted(flagged):
        print(f"  {short(m):<35} {tag:>5} {ds:<12} {n:>3} {mn:>9.6f} {sd:>9.6f} {lo:>9.6f} {hi:>9.6f} {hi-lo:>9.6f}")

    # ── Per-row detail for the worst cells (mean < 0.005) ──
    print(f"\n\n{'='*120}")
    print(f"  DETAIL: cells with mean |MdR| < 0.005 — individual quantization variants")
    print(f"{'='*120}")

    rows_full = load()
    worst_cells = [(m, ds) for mn, m, ds, *_ in flagged if mn < 0.005]
    for m, ds in worst_cells:
        cell_rows = [(r["Label"], r["|MdR|"]) for r in rows_full
                     if r["target_model"] == m and r["dataset"] == ds]
        print(f"\n  {short(m)} / {ds}:")
        for lbl, v in sorted(cell_rows, key=lambda x: x[1]):
            print(f"    {lbl:<45} |MdR|={v:.6f}")


if __name__ == "__main__":
    main()
