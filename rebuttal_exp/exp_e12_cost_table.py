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

    # Three-tier benchmark-side estimate. The gold-length floor alone is a
    # LOWER BOUND OF A LOWER BOUND: real MC evaluation scores every choice
    # (~4 forwards/question, not 1 decoded token) and generative evaluation
    # decodes to a budget/stop, not to the gold length.
    #   floor    : decode exactly the gold answer tokens once (absurdly
    #              generous to the benchmark side; kept as the anchor)
    #   standard : arc/mmlu -> 4-choice log-likelihood scoring, costed at
    #              4 x our own MEASURED per-benchmark forward time;
    #              gsm8k -> 256-token generation budget; squad/triviaqa ->
    #              64-token budgets; 0-shot, no few-shot prefill counted
    #   maj8     : standard, with maj@8 self-consistency on GSM8K only
    GEN_BUDGET = {"squad": 64, "triviaqa": 64, "gsm8k": 256}
    MC_CHOICES = {"arc": 4, "mmlu": 4}

    floor_s = sum(c["total_tok"] for c in counts.values()) / rate

    # E12b: if the dedicated GSM8K measurement exists, its gsm8k component
    # replaces the 256-token-budget ASSUMPTION with a MEASURED wall-clock
    # (one greedy decode to natural EOS, generous ceiling with truncation
    # check; scaled linearly to the paper's sample count).
    measured = None
    mpath = OUT_DIR / "gsm8k_measured.json"
    if mpath.exists():
        import json
        measured = json.loads(mpath.read_text())
        # decode_s_min == the single timed run under the x1 protocol;
        # fall back to mean for older json versions
        measured["_dec_s"] = measured.get("decode_s_min",
                                          measured.get("decode_s_mean"))
        print(f"[measured] gsm8k decode {measured['_dec_s']:.1f}s "
              f"@ n={measured['n']} (E12b, greedy x1 to natural EOS) — "
              f"replaces the 256-token budget assumption")

    def standard_s(t, maj8=False):
        mc = sum(MC_CHOICES[b] * t["extract"].get(b, 0)
                 for b in MC_CHOICES)
        gen = 0.0
        for b, budget in GEN_BUDGET.items():
            reps = 8 if (maj8 and b == "gsm8k") else 1
            if b == "gsm8k" and measured is not None:
                gen += (reps * measured["_dec_s"]
                        * counts[b]["n"] / measured["n"])
            else:
                gen += reps * counts[b]["n"] * budget / rate
        return mc + gen

    md = [f"# E12 — GPU cost: PRISM diagnosis vs benchmark evaluation "
          f"({args.family}, {args.num_samples} samples/benchmark)", "",
          f"Decode throughput (measured, E7 greedy, median over "
          f"{len(e7)} variants): **{rate:.1f} tok/s**. Benchmark-side "
          f"tiers: floor = decode gold answers once; standard = 4-choice "
          f"LL scoring on ARC/MMLU (at 4x our measured forward time) + "
          + ("MEASURED gsm8k decode (E12b: one greedy decode to natural "
             "EOS, truncation-checked) + 64-token budgets (SQuAD, TriviaQA)"
             if measured is not None else
             "generation budgets 256 (GSM8K) / 64 (SQuAD, TriviaQA)")
          + f", 0-shot, no few-shot prefill counted; maj8 = standard + "
          f"maj@8 on GSM8K. All tiers exclude retries and still require "
          f"labels to score.",
          "",
          "## Per-variant cost (seconds, 1 decimal)", "",
          "| variant | load (s) | PRISM 5-bench (s) | PRISM total (s) "
          "| bench floor (s) | bench standard (s) | bench maj@8 (s) |",
          "|---|---|---|---|---|---|---|"]

    prism_s, std_ratios = [], []
    for label, t in sorted(e1.items()):
        ext = sum(t["extract"].values())
        load = float(t["load"] or 0)
        total = load + ext
        std = standard_s(t)
        m8 = standard_s(t, maj8=True)
        prism_s.append(total)
        std_ratios.append(std / max(total, 1e-9))
        md.append(f"| {label} | {load:.1f} | {ext:.1f} | **{total:.1f}** "
                  f"| {floor_s:.1f} | {std:.1f} | {m8:.1f} |")

    md += ["",
           "## Gold answer-span volume (the floor tier's token counts)",
           "", "| benchmark | n | mean answer tokens | total tokens |",
           "|---|---|---|---|"]
    for b in BENCHMARKS:
        c = counts[b]
        md.append(f"| {b} | {c['n']} | {c['mean_answer_tok']:.0f} "
                  f"| {c['total_tok']:,} |")

    med_p = statistics.median(prism_s)
    med_ext = statistics.median(
        [sum(t["extract"].values()) for t in e1.values()])
    screen = statistics.median(
        [float(t["load"] or 0) for t in e1.values()]) \
        + med_ext / 5 * 32 / args.num_samples
    med_std = statistics.median([standard_s(t) for t in e1.values()])
    med_m8 = statistics.median([standard_s(t, True) for t in e1.values()])
    md += ["",
           f"**Summary (medians):** PRISM full 5-benchmark diagnosis "
           f"{med_p:.1f} s/variant; PRISM screening mode (32-sequence "
           f"generic reference, load-dominated) ~{screen:.1f} s/variant. "
           f"Benchmark side: {floor_s:.1f} s (floor) / {med_std:.1f} s "
           f"(standard) / {med_m8:.1f} s (maj@8 GSM8K) — i.e., "
           f"{floor_s / med_p:.1f}x / {med_std / med_p:.0f}x / "
           f"{med_m8 / med_p:.0f}x the full diagnosis, and "
           f"{med_std / screen:.0f}x-{med_m8 / screen:.0f}x the screening "
           f"mode. Even the floor tier still requires labels; PRISM "
           f"requires none.", ""]

    out = OUT_DIR / "cost_table.md"
    out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
