#!/usr/bin/env python3
"""
E12 — GPU-cost table: PRISM diagnosis vs running the benchmark suite
(G3T9-W1 "a cost analysis is needed"; also feeds 8VrD's Limitations item
(iv) feature-extraction cost).

No dedicated GPU run. Three ingredients, all measured on the same hardware:

  1. PRISM side (MEASURED): per-variant wall time from E1's log —
       [load] <label>: <s>s              (proxy load, incl. GGUF dequant)
       <bench>: cka=... (<s>s)           (one teacher-forced forward pass
                                          per benchmark, 512 samples)
     PRISM's diagnostic cost for a variant = load + K_bench x extract.
  2. Decode throughput (MEASURED): from E7's log —
       [gen] <label>: <N> new tokens in <s>s (<r> tok/s)
     greedy decoding on this hardware, per variant.
  3. Benchmark-suite decode volume (COUNTED, CPU-only): tokenize each
     benchmark's answer/CoT spans (same 512-sample splits as the paper) and
     count the tokens an evaluation harness must GENERATE. The benchmark
     cost estimate  sum_b n_b * mean_answer_tokens_b / throughput  is a
     deliberate LOWER bound (greedy, single pass, no retries/self-
     consistency) — the honest-conservative direction for our comparison.

Run AFTER E1 and E7 (their logs supply 1. and 2.):
    python rebuttal_exp/exp_e12_cost_table.py                # defaults
    python rebuttal_exp/exp_e12_cost_table.py --family llama \
        --model meta-llama/Meta-Llama-3.1-8B

Output: rebuttal_exp/out/E12/cost_table.md
"""

from __future__ import annotations

import argparse
import glob
import re
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_DIR = HERE / "out" / "E12"

BENCHMARKS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]

LOAD_RE = re.compile(r"^\s*\[load\]\s+(?P<label>.+?):\s+(?P<sec>\d+)s\s*$")
BENCH_RE = re.compile(
    r"^\s*(?P<bench>" + "|".join(BENCHMARKS) + r"):\s+cka=.*\((?P<sec>\d+)s\)\s*$")
GEN_RE = re.compile(
    r"^\s*\[gen\]\s+(?P<label>.+?):\s+(?P<tok>\d+) new tokens in "
    r"(?P<sec>\d+)s \((?P<rate>[\d.]+) tok/s\)\s*$")


def parse_e1(paths):
    """label -> {'load': s, 'extract': {bench: s}} from E1 screen logs."""
    per = defaultdict(lambda: {"load": None, "extract": {}})
    cur = None
    for path in paths:
        for line in open(path, errors="replace"):
            m = LOAD_RE.match(line)
            if m:
                cur = m.group("label")
                per[cur]["load"] = int(m.group("sec"))
                continue
            m = BENCH_RE.match(line)
            if m and cur is not None:
                per[cur]["extract"][m.group("bench")] = int(m.group("sec"))
    return {k: v for k, v in per.items() if v["extract"]}


def parse_e7(paths):
    """Measured greedy decode throughput (tok/s) per variant."""
    rates = {}
    for path in paths:
        for line in open(path, errors="replace"):
            m = GEN_RE.match(line)
            if m:
                rates[m.group("label")] = float(m.group("rate"))
    return rates


def answer_token_counts(model_id, num_samples, max_length, seed):
    """CPU-only: tokens the benchmark harness must GENERATE per dataset
    (answer/CoT span lengths of the same splits the paper evaluates)."""
    import sys
    sys.path.insert(0, str(REPO))
    from prism.data.loaders import load_task_data          # noqa: E402
    from transformers import AutoTokenizer                 # noqa: E402

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = {}
    for b in BENCHMARKS:
        dl = load_task_data(b, split="test", num_samples=num_samples,
                            batch_size=8, tokenizer=tok,
                            max_length=max_length, seed=seed)
        lens = []
        for batch in dl:
            pl = batch["prompt_length"]
            for i in range(batch["input_ids"].shape[0]):
                n = int(batch["attention_mask"][i].sum().item())
                lens.append(max(n - int(pl[i].item()), 1))
        out[b] = {"n": len(lens), "mean_answer_tok": statistics.mean(lens),
                  "total_tok": sum(lens)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="llama")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--num_samples", type=int, default=512)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--e1_logs", default=str(HERE / "out" / "E1" / "screen.E1.*.log"))
    ap.add_argument("--e7_logs", default=str(HERE / "out" / "E7" / "screen.E7.*.log"))
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    e1 = parse_e1(sorted(glob.glob(args.e1_logs)))
    e7 = parse_e7(sorted(glob.glob(args.e7_logs)))
    if not e1:
        raise SystemExit(f"No E1 timing lines found in {args.e1_logs} — "
                         "run script_E1.sh first (it emits [load]/per-bench "
                         "timings that this table is built from).")
    if not e7:
        raise SystemExit(f"No E7 [gen] lines found in {args.e7_logs} — "
                         "run script_E7.sh first (measured decode tok/s).")

    counts = answer_token_counts(args.model, args.num_samples,
                                 args.max_length, args.seed)
    rate = statistics.median(e7.values())        # tok/s, same hardware

    md = [f"# E12 — GPU cost: PRISM diagnosis vs benchmark evaluation "
          f"({args.family}, {args.num_samples} samples/benchmark)", "",
          f"Decode throughput (measured, E7 greedy, median over "
          f"{len(e7)} variants): **{rate:.1f} tok/s**.", "",
          "## Per-variant cost", "",
          "| variant | load (min) | PRISM extract, 5 benchmarks (min) "
          "| PRISM total (min) | benchmark-suite decode estimate (min) |",
          "|---|---|---|---|---|"]

    est_decode_min = sum(c["total_tok"] for c in counts.values()) / rate / 60
    prism_mins, ratios = [], []
    for label, t in sorted(e1.items()):
        ext = sum(t["extract"].values()) / 60
        load = (t["load"] or 0) / 60
        total = load + ext
        prism_mins.append(total)
        ratios.append(est_decode_min / max(total, 1e-9))
        md.append(f"| {label} | {load:.1f} | {ext:.1f} | **{total:.1f}** "
                  f"| {est_decode_min:.0f} |")

    md += ["",
           "## Benchmark decode volume (counted from the paper's own splits)",
           "", "| benchmark | n | mean answer tokens | total tokens |",
           "|---|---|---|---|"]
    for b in BENCHMARKS:
        c = counts[b]
        md.append(f"| {b} | {c['n']} | {c['mean_answer_tok']:.0f} "
                  f"| {c['total_tok']:,} |")

    md += ["",
           f"**Summary:** PRISM diagnosis median "
           f"{statistics.median(prism_mins):.1f} min/variant "
           f"(one teacher-forced forward pass per benchmark, no decoding, "
           f"no labels) vs ~{est_decode_min:.0f} min/variant to DECODE the "
           f"same 5-benchmark suite once "
           f"(median ratio {statistics.median(ratios):.1f}x). The decode "
           f"figure is a deliberate lower bound: greedy, single pass, no "
           f"retries or self-consistency, and it still needs labels to "
           f"score. Screening mode (32-sequence generic reference) drops "
           f"PRISM's extraction cost by a further ~{args.num_samples // 32}x "
           f"and is load-dominated.", ""]

    out = OUT_DIR / "cost_table.md"
    out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
