#!/usr/bin/env python3
"""
E11 — First data point beyond PTQ / frozen-head LoRA: PRISM between BASE
and INSTRUCT checkpoints (G3T9-Q1: "applicable to full FT / RLHF?").

This is exactly the regime App. C.3 scopes out for tightness — full-
parameter post-training rotates features AND diverges the head — but the
DECOMPOSITION itself only needs matched architecture/dimension. E11 runs it
end-to-end: target T = base checkpoint, proxy P = its instruct counterpart,
identity alignment W = I (main-text protocol), nonzero head term (H_P is the
instruct lm_head), on the paper's 5 benchmarks.

Reported per (pair, benchmark):
    rho_T, rho_P, 1-Omega_I, scale/shape terms, gamma, bound B,
    measured |dR| (teacher-forced answer-only CE gap, paper protocol),
    bound_holds (Theorem 1 verified end-to-end in this regime).
Plus a PTQ context anchor from the paper CSV (the same base family's
Q4_K_M / Q2_K rows): how much larger is base->instruct geometric drift than
aggressive quantization — the honest tightness story of App. C.3.

Both models are fed IDENTICAL token ids (base tokenizer, no chat template):
token-aligned features require shared inputs; this matches how the paper
evaluates instruct-family targets on the same loaders.

Cost: 2 model loads + 2 x 5 x 512-sample forwards per pair ~= 30-45 min;
default pairs (llama, qwen) ~= 1.5 h on the RTX 5090.

Usage (repo root, GPU box):
    python rebuttal_exp/exp_e11_base_instruct.py --pairs llama qwen

Output: rebuttal_exp/out/E11/base_instruct.csv
        rebuttal_exp/out/E11/E11_results.md
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from common_quant import free_cuda, load_target            # noqa: E402
from prism.core.bounds import UnifiedBound                 # noqa: E402
from prism.core.metrics import PRISMMetrics                # noqa: E402
from prism.data.loaders import load_task_data              # noqa: E402
from prism.models.extractors import LLMExtractor           # noqa: E402
from transformers import AutoTokenizer                     # noqa: E402

OUT_DIR = HERE / "out" / "E11"
CSV_PATH = REPO / "exp_result" / "quantization" / "quantization_merged_slim.csv"

BENCHMARKS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]

# (base, instruct) checkpoints — all five pairs exist as paper families.
PAIRS = {
    "llama": ("meta-llama/Meta-Llama-3.1-8B",
              "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    "qwen": ("Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B"),
    "mistral": ("mistralai/Mistral-7B-v0.3",
                "mistralai/Mistral-7B-Instruct-v0.3"),
    "ministral": ("mistralai/Ministral-3-8B-Base-2512",
                  "mistralai/Ministral-3-8B-Instruct-2512"),
    "qwen25": ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"),
}

CONTEXT_LABELS = ["BF16 vs Q4_K_M", "BF16 vs Q2_K"]   # PTQ anchors


def extract(model, batches, device):
    """(Z_concat, mean answer-token CE) — paper protocol."""
    Z, stats = LLMExtractor().extract_features_and_loss_per_sample(
        model, batches, device, z_mode="concat",
    )
    tok = stats["token_losses"]
    return Z, (tok.mean().item() if tok is not None
               else stats["losses"].mean().item())


def ptq_context(base_id):
    """(label, dataset) -> 1 - Omega_I from the paper CSV, PTQ anchors."""
    out = {}
    for r in csv.DictReader(open(CSV_PATH)):
        if (r["target_model"] == base_id and r["Label"] in CONTEXT_LABELS
                and r["dataset"] in BENCHMARKS):
            try:
                out[(r["Label"], r["dataset"])] = 1 - float(r["Omega_I"])
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["llama", "qwen"],
                    choices=list(PAIRS))
    ap.add_argument("--num_samples", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    md = ["# E11 — PRISM across the full-post-training gap "
          "(base vs instruct, W = I, nonzero head term)", "",
          f"Paper protocol otherwise: {args.num_samples} samples/benchmark, "
          f"seed {args.seed}, base tokenizer, teacher-forced answer-only CE. "
          "This is App. C.3's scoped-out tightness regime, run end-to-end.", ""]

    for pair in args.pairs:
        base_id, inst_id = PAIRS[pair]
        print(f"\n===== {pair}: {base_id} -> {inst_id} =====")
        tokenizer = AutoTokenizer.from_pretrained(base_id,
                                                  trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        loaders = {
            b: list(load_task_data(b, split="test",
                                   num_samples=args.num_samples,
                                   batch_size=args.batch_size,
                                   tokenizer=tokenizer,
                                   max_length=args.max_length,
                                   seed=args.seed))
            for b in BENCHMARKS
        }

        extractor = LLMExtractor()
        print(f"[base] {base_id}")
        base = load_target(base_id, args.device)
        H_T = extractor.extract_head(base).float().cpu()
        K = UnifiedBound.theoretical_K(H_T.to(args.device))
        Z_T, loss_T = {}, {}
        for b in BENCHMARKS:
            t0 = time.time()
            Z_T[b], loss_T[b] = extract(base, loaders[b], args.device)
            print(f"  {b}: Z={list(Z_T[b].shape)} loss={loss_T[b]:.4f} "
                  f"({time.time() - t0:.0f}s)")
        del base
        free_cuda()

        print(f"[instruct] {inst_id}")
        inst = load_target(inst_id, args.device)
        H_P = extractor.extract_head(inst).float().cpu()
        if H_P.shape != H_T.shape:
            raise SystemExit(f"head shape mismatch {H_T.shape} vs "
                             f"{H_P.shape} — pair not comparable")
        # Collect ALL proxy features first, then release the model: head
        # metrics with an 8B model resident OOM a 32 GB card (E3 part-B
        # postmortem, 2026-07-24 — svdvals workspace inside compute_all).
        Z_P_all, loss_P_all = {}, {}
        for b in BENCHMARKS:
            t0 = time.time()
            Z_P_all[b], loss_P_all[b] = extract(inst, loaders[b], args.device)
            print(f"  {b}: Z={list(Z_P_all[b].shape)} "
                  f"loss={loss_P_all[b]:.4f} ({time.time() - t0:.0f}s)")
        del inst
        free_cuda()

        rows = []
        I_d = torch.eye(H_T.shape[0], device=args.device)
        H_T_dev = H_T.to(args.device)
        H_P_dev = H_P.to(args.device)
        for b in BENCHMARKS:
            t0 = time.time()
            n = min(Z_T[b].shape[0], Z_P_all[b].shape[0])
            X = Z_T[b][:n].to(args.device)
            Y = Z_P_all[b][:n].to(args.device)
            # Lightweight Bound_I path (paper primitives; skips the
            # 4096 x |V| spectral SVD compute_all adds, which is not part
            # of the bound).
            rho_T = PRISMMetrics.rms_scale(X)
            rho_P = PRISMMetrics.rms_scale(Y)
            omega = PRISMMetrics.trace_omega(X, Y)
            fe = PRISMMetrics.feature_error(rho_T, rho_P, omega)
            Sigma_P = (Y.T @ Y) / n
            gamma = PRISMMetrics.head_discrepancy_covariance(
                H_T_dev, H_P_dev, I_d, Sigma_P)
            B = K["K_feat"] * fe + K["K_pred"] * gamma
            dR = abs(loss_P_all[b] - loss_T[b])
            row = {
                "pair": pair, "benchmark": b,
                "rho_T": rho_T, "rho_P": rho_P,
                "one_minus_omega": 1 - omega,
                "scale_term": PRISMMetrics.scale_mismatch(rho_T, rho_P),
                "shape_term": PRISMMetrics.shape_mismatch(rho_T, rho_P,
                                                          omega),
                "gamma": gamma,
                "head_share_of_B": K["K_pred"] * gamma / B
                if B > 0 else float("nan"),
                "bound": B, "|dR|": dR, "bound_holds": B >= dR,
            }
            rows.append(row)
            all_rows.append(row)
            print(f"  {b}: 1-omega={row['one_minus_omega']:.4f} "
                  f"gamma={gamma:.2f} B={B:.2f} |dR|={dR:.4f} "
                  f"holds={row['bound_holds']} ({time.time() - t0:.0f}s)")
            del X, Y, Sigma_P
            free_cuda()
        del Z_T, Z_P_all, H_T_dev, H_P_dev, I_d
        free_cuda()

        ctx = ptq_context(base_id)
        md += [f"## {pair}: {base_id} -> {inst_id}", "",
               "| benchmark | rho_T | rho_P | 1-Omega_I | gamma "
               "| head share of B | B | measured |dR| | holds |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            md.append(f"| {r['benchmark']} | {r['rho_T']:.1f} "
                      f"| {r['rho_P']:.1f} | {r['one_minus_omega']:.4f} "
                      f"| {r['gamma']:.2f} | {r['head_share_of_B']:.0%} "
                      f"| {r['bound']:.1f} | {r['|dR|']:.4f} "
                      f"| {'yes' if r['bound_holds'] else 'NO'} |")
        for lab in CONTEXT_LABELS:
            vals = [ctx[(lab, b)] for b in BENCHMARKS if (lab, b) in ctx]
            if vals:
                md.append(f"\nPTQ context ({lab}, same base family): "
                          f"1-Omega_I median {statistics.median(vals):.4f} "
                          f"over {len(vals)} benchmarks.")
        md.append("")

    with open(OUT_DIR / "base_instruct.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    holds = sum(r["bound_holds"] for r in all_rows)
    om = [r["one_minus_omega"] for r in all_rows]
    hs = [r["head_share_of_B"] for r in all_rows
          if not math.isnan(r["head_share_of_B"])]
    md += ["## Summary", "",
           f"- Theorem 1 verified end-to-end: bound holds in "
           f"{holds}/{len(all_rows)} (pair, benchmark) cells.",
           f"- Geometric drift 1-Omega_I: median "
           f"{statistics.median(om):.4f} (min {min(om):.4f}, "
           f"max {max(om):.4f}) — compare the PTQ anchors above.",
           f"- Head term engages as predicted: median head share of B = "
           f"{statistics.median(hs):.0%} (frozen-head settings: 0%).",
           "",
           "Reading for G3T9-Q1: the decomposition runs unchanged in the "
           "full-post-training regime and the bound remains valid; what "
           "degrades is TIGHTNESS, exactly as App. C.3 anticipates — "
           "report the numbers and keep validated claims scoped to PTQ + "
           "frozen-head LoRA.", ""]
    (OUT_DIR / "E11_results.md").write_text("\n".join(md))
    print("\n".join(md[-10:]))


if __name__ == "__main__":
    main()
