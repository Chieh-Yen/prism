#!/usr/bin/env python3
"""
E12b — MEASURED GSM8K decode-vs-teacher-forced cost (G3T9-W1).

GSM8K dominates the benchmark-suite cost estimate (~90% of the standard
tier), so this experiment replaces its assumption with a measurement on the
same hardware, model, prompts, and batch size:

    TF pass   x1 : the exact PRISM extraction call
                   (extract_features_and_loss_per_sample, concat) — what
                   PRISM actually pays on this benchmark;
    decode    x1 : greedy generation to NATURAL stopping (EOS), timed
                   once — greedy is deterministic, and the claim is a
                   coarse ratio (TF is MUCH faster than decoding), so one
                   run suffices. The max_new_tokens CEILING MUST BE LARGE
                   enough that generations end at EOS, not at the ceiling
                   (default 1024); the script counts ceiling hits and
                   warns if >5% — a truncated decode would UNDERSTATE the
                   benchmark side's true cost.

A GPU warmup (small forward + 8-token generate) runs before any timing.

Output: rebuttal_exp/out/E12/gsm8k_measured.json  (consumed by
        exp_e12_cost_table.py, which swaps its gsm8k standard-tier
        assumption for this measurement, scaled to the paper's 512)
        + gsm8k_measured.md (human-readable).

Cost: n=256 -> TF ~0.5-1 min + one ~4-7 min decode ~= 5-8 min on the
RTX 5090. SAMPLES=512 for the zero-footnote version (~2x).

Usage (repo root, GPU box):
    python rebuttal_exp/exp_e12_gsm8k_measure.py --num_samples 256
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from common_quant import load_target                      # noqa: E402
from exp_e7_freerun import (collect_prompts,               # noqa: E402
                            generate_trajectories)
from prism.data.loaders import load_task_data              # noqa: E402
from prism.models.extractors import LLMExtractor           # noqa: E402
from transformers import AutoTokenizer                     # noqa: E402

OUT_DIR = HERE / "out" / "E12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=1024,
                    help="ceiling only — decoding stops at EOS naturally; "
                         "MUST be large enough that generations are not "
                         "truncated (ceiling hits are counted and warned)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="greedy is deterministic; one timed decode "
                         "suffices for the coarse TF-vs-decode ratio")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gold_batches = list(load_task_data(
        "gsm8k", split="test", num_samples=args.num_samples,
        batch_size=args.batch_size, tokenizer=tokenizer,
        max_length=512, seed=args.seed))
    prompts = collect_prompts(gold_batches)
    print(f"{len(prompts)} gsm8k prompts, batch {args.batch_size}")

    model = load_target(args.model, args.device)
    extractor = LLMExtractor()

    # ── warmup (excluded from every timing) ────────────────────────────
    _ = extractor.extract_features_and_loss_per_sample(
        model, gold_batches[:1], args.device, z_mode="concat")
    _ = generate_trajectories(model, prompts[:args.batch_size], tokenizer,
                              args.device, 8, args.batch_size)
    torch.cuda.synchronize()

    # ── teacher-forced pass x1 (the exact PRISM extraction call) ──────
    t0 = time.time()
    Z, stats = extractor.extract_features_and_loss_per_sample(
        model, gold_batches, args.device, z_mode="concat")
    torch.cuda.synchronize()
    tf_s = time.time() - t0
    tok = stats["token_losses"]
    print(f"TF x1: {tf_s:.1f}s  (Z={list(Z.shape)}, "
          f"loss={tok.mean().item():.4f})")
    del Z
    torch.cuda.empty_cache()

    # ── greedy decode (natural EOS; ceiling hits counted) ─────────────
    dec_s, gen_tok, ceiling_hits = [], None, 0
    for rep in range(args.repeats):
        t0 = time.time()
        trajs = generate_trajectories(
            model, prompts, tokenizer, args.device,
            args.max_new_tokens, args.batch_size)
        torch.cuda.synchronize()
        dt = time.time() - t0
        n_tok = sum(len(seq) - pl for seq, pl in trajs)
        if gen_tok is None:
            gen_tok = n_tok
            ceiling_hits = sum(1 for seq, pl in trajs
                               if len(seq) - pl >= args.max_new_tokens)
        dec_s.append(dt)
        print(f"decode (run {rep + 1}/{args.repeats}): {dt:.1f}s, "
              f"{n_tok} tokens ({n_tok / dt:.1f} tok/s), "
              f"{ceiling_hits} ceiling hits")

    hit_rate = ceiling_hits / len(prompts)
    if hit_rate > 0.05:
        print(f"  [WARN] {100 * hit_rate:.0f}% of generations hit the "
              f"{args.max_new_tokens}-token ceiling — the decode cost is "
              f"UNDERSTATED; rerun with a larger --max_new_tokens")

    dec_min = min(dec_s)
    ratio = dec_min / tf_s
    mean_len = gen_tok / len(prompts)

    result = {
        "model": args.model, "n": len(prompts),
        "batch_size": args.batch_size,
        "max_new_tokens_ceiling": args.max_new_tokens,
        "tf_s": tf_s,
        "decode_s": dec_s,
        "decode_s_min": dec_min,             # headline (conservative)
        "decode_s_mean": statistics.mean(dec_s),
        "gen_tok_total": gen_tok, "gen_tok_mean": mean_len,
        "ceiling_hits": ceiling_hits, "ceiling_hit_rate": hit_rate,
        "tok_per_s": gen_tok / dec_min,
        "decode_over_tf": ratio,
    }
    (OUT_DIR / "gsm8k_measured.json").write_text(json.dumps(result, indent=2))

    md = ["# E12b — measured GSM8K decode vs teacher-forced "
          f"({args.model.split('/')[-1]}, n={len(prompts)}, "
          f"batch {args.batch_size})", "",
          f"- teacher-forced x1 (PRISM's exact extraction call): "
          f"**{tf_s:.1f}s**",
          f"- greedy decode x1 to natural EOS (ceiling "
          f"{args.max_new_tokens}, hit by {ceiling_hits}/{len(prompts)} "
          f"generations): **{dec_min:.1f}s**",
          f"- natural generation length: mean **{mean_len:.1f} tok/prompt** "
          f"({gen_tok} total; throughput {gen_tok / dec_min:.1f} tok/s)",
          f"- **decode / TF = {ratio:.1f}x** on identical prompts, model, "
          f"batch size — a single greedy decode of this ONE benchmark "
          f"costs {ratio:.1f}x PRISM's whole teacher-forced pass on it; "
          f"maj@8 makes it {8 * ratio:.0f}x, and scoring still needs "
          f"labels.", "",
          "Consumed by exp_e12_cost_table.py: the gsm8k component of the "
          "'standard' tier switches from an assumed 256-token budget to "
          "this measurement (scaled linearly to 512 prompts).", ""]
    (OUT_DIR / "gsm8k_measured.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
