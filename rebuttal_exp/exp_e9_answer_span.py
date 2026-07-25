#!/usr/bin/env python3
"""
E9 — GSM8K final-answer-span variant (pCi8-W4: "GSM8K correlation notably
lower — structural limitation for reasoning models").

Paper diagnosis (App. F.3): long teacher-forced CoT answer spans dilute the
per-token loss, collapsing GSM8K's mean |dR| to ~0.019 nats — per-variant
differences drown in measurement noise. Proposed mitigation: restrict
features AND losses to the FINAL-ANSWER span (GSM8K answers end with
"#### <number>"), where the graded content lives, instead of averaging over
the whole chain of thought.

Implementation trick: the repo's extractor pairs concat features with
answer-token losses for positions >= prompt_length. Rewriting
``prompt_length`` to the position of the LAST "####" marker therefore
restricts BOTH features and per-token CE to the final-answer span with the
otherwise-identical pipeline (same extractor, same bound, same variants).
Samples without a marker fall back to the last ``--fallback_last_k`` tokens
(count reported).

Per variant of the family, both protocols in the same load:
    full span (paper protocol)  ->  B_full, |dR|_full
    final-answer span (E9)      ->  B_span, |dR|_span
Deliverable: rs(B, |dR|) across variants under both protocols + the |dR|
scale/dynamic-range change — does conditioning on the graded span restore
rank signal that CoT averaging washes out?

Head convention follows the paper: only GGUF k-quants alter the served
lm_head (gamma > 0); BnB/GPTQ/dtype proxies keep the FP16 head.

Cost: ~12 variants x (load 2-5 min + two GSM8K extractions) ~= 1 h for the
Llama family on the RTX 5090.

Usage (repo root, GPU box):
    python rebuttal_exp/exp_e9_answer_span.py --family llama

Output: rebuttal_exp/out/E9/{family}_gsm8k_span.csv
        rebuttal_exp/out/E9/E9_results_{family}.md
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

from common_quant import (FAMILIES, free_cuda, load_proxy, load_target,  # noqa: E402
                          risk_gaps_from_csv, variants_from_csv)
from prism.core.bounds import UnifiedBound                # noqa: E402
from prism.core.metrics import PRISMMetrics               # noqa: E402
from prism.data.loaders import load_task_data             # noqa: E402
from prism.models.extractors import LLMExtractor          # noqa: E402
from transformers import AutoTokenizer                    # noqa: E402

OUT_DIR = HERE / "out" / "E9"
DATASET = "gsm8k"


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


# ----------------------------------------------------------------------
def marker_patterns(tokenizer):
    """Token-id patterns that realise '####' under common BPE contexts."""
    pats = set()
    for pre in ("", " ", "\n"):
        ids = tuple(tokenizer.encode(pre + "####", add_special_tokens=False))
        if ids:
            pats.add(ids)
    return [list(p) for p in pats]


def make_span_batches(gold_batches, patterns, fallback_k):
    """Copy of the gold batches with prompt_length moved to the LAST '####'
    marker of each sample's answer region (fallback: last K tokens)."""
    span_batches, matched, total = [], 0, 0
    tok_full, tok_span = 0, 0
    for batch in gold_batches:
        ids, mask, pl = (batch["input_ids"], batch["attention_mask"],
                         batch["prompt_length"])
        new_pl = pl.clone()
        for i in range(ids.shape[0]):
            total += 1
            n = int(mask[i].sum().item())
            p = int(pl[i].item())
            seq = ids[i, :n].tolist()
            best = None
            for pat in patterns:
                L = len(pat)
                for s in range(max(p, 0), n - L + 1):
                    if seq[s:s + L] == pat:
                        best = s if best is None else max(best, s)
            if best is not None and n - best >= 2:
                new_pl[i] = best
                matched += 1
            else:
                new_pl[i] = max(p, n - fallback_k)
            tok_full += max(n - p, 0)
            tok_span += max(n - int(new_pl[i].item()), 0)
        nb = dict(batch)
        nb["prompt_length"] = new_pl
        span_batches.append(nb)
    return span_batches, matched, total, tok_full / total, tok_span / total


def extract(model, batches, device):
    """(Z_concat, mean per-token CE over the span the batches define)."""
    Z, stats = LLMExtractor().extract_features_and_loss_per_sample(
        model, batches, device, z_mode="concat",
    )
    tok = stats["token_losses"]
    return Z, (tok.mean().item() if tok is not None
               else stats["losses"].mean().item())


def prism_bound(Z_T, Z_P, H_T, H_P, K, device, label):
    n = min(Z_T.shape[0], Z_P.shape[0])
    res = PRISMMetrics.compute_all(
        Z_T[:n].to(device), H_T.to(device),
        Z_P[:n].to(device), H_P.to(device),
        W=torch.eye(H_T.shape[0], device=device), label=label,
    )
    return K["K_feat"] * res.feature_error + K["K_pred"] * res.head_discrepancy


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(FAMILIES), default="llama")
    ap.add_argument("--num_samples", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--fallback_last_k", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_id = FAMILIES[args.family]
    specs = variants_from_csv(args.family)
    risk_csv = risk_gaps_from_csv(args.family)

    tokenizer = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gold_batches = list(load_task_data(
        DATASET, split="test", num_samples=args.num_samples,
        batch_size=args.batch_size, tokenizer=tokenizer,
        max_length=args.max_length, seed=args.seed))

    patterns = marker_patterns(tokenizer)
    span_batches, matched, total, mean_full, mean_span = \
        make_span_batches(gold_batches, patterns, args.fallback_last_k)
    print(f"'####' marker matched in {matched}/{total} samples "
          f"({100 * matched / total:.1f}%); mean answer tokens "
          f"full={mean_full:.1f} -> span={mean_span:.1f}")

    print(f"Target {target_id} ...")
    target = load_target(target_id, args.device)
    extractor = LLMExtractor()
    H_T = extractor.extract_head(target).float().cpu()
    K = UnifiedBound.theoretical_K(H_T.to(args.device))
    Z_T_full, loss_T_full = extract(target, gold_batches, args.device)
    Z_T_span, loss_T_span = extract(target, span_batches, args.device)
    print(f"  loss_T full={loss_T_full:.4f}  span={loss_T_span:.4f}")
    del target
    free_cuda()

    rows = []
    for spec in specs:
        label = spec["label"]
        print(f"\n=== {label} ===")
        t0 = time.time()
        try:
            proxy = load_proxy(spec, args.device)
        except Exception as exc:                          # noqa: BLE001
            print(f"  [FAIL load] {exc}")
            continue
        try:
            Z_P_full, loss_P_full = extract(proxy, gold_batches, args.device)
            Z_P_span, loss_P_span = extract(proxy, span_batches, args.device)
        except Exception as exc:                          # noqa: BLE001
            print(f"  [FAIL extract] {exc}")
            del proxy
            free_cuda()
            continue
        H_P = (extractor.extract_head(proxy).float().cpu()
               if spec["kind"] == "gguf" else H_T)
        del proxy
        free_cuda()

        B_full = prism_bound(Z_T_full, Z_P_full, H_T, H_P, K,
                             args.device, label)
        B_span = prism_bound(Z_T_span, Z_P_span, H_T, H_P, K,
                             args.device, label)
        row = {
            "label": label,
            "B_full": B_full, "dR_full": abs(loss_P_full - loss_T_full),
            "B_span": B_span, "dR_span": abs(loss_P_span - loss_T_span),
            "dR_csv": risk_csv.get((label, DATASET), float("nan")),
        }
        rows.append(row)
        print(f"  full: B={B_full:.2f} |dR|={row['dR_full']:.4f}   "
              f"span: B={B_span:.2f} |dR|={row['dR_span']:.4f} "
              f"({time.time() - t0:.0f}s)")
        del Z_P_full, Z_P_span
        free_cuda()

        with open(OUT_DIR / f"{args.family}_gsm8k_span.csv", "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ── Summary ─────────────────────────────────────────────────────────
    dr_f = [r["dR_full"] for r in rows]
    dr_s = [r["dR_span"] for r in rows]
    rs_full = spearman([r["B_full"] for r in rows], dr_f)
    rs_span = spearman([r["B_span"] for r in rows], dr_s)
    rs_csv = spearman([r["B_full"] for r in rows],
                      [r["dR_csv"] for r in rows])
    nz_f = [v for v in dr_f if v > 1e-9]
    nz_s = [v for v in dr_s if v > 1e-9]

    md = [f"# E9 — GSM8K final-answer-span variant ({args.family}, "
          f"n={args.num_samples}, seed {args.seed})", "",
          f"'####' marker matched in {matched}/{total} samples; mean answer "
          f"tokens {mean_full:.1f} (full CoT) -> {mean_span:.1f} "
          f"(final-answer span; fallback last {args.fallback_last_k}).", "",
          "| protocol | rs(B, |dR|) | mean |dR| (nats) | |dR| max/min |",
          "|---|---|---|---|",
          f"| full CoT span (paper) | {rs_full:+.3f} "
          f"| {statistics.mean(dr_f):.4f} "
          f"| {max(nz_f) / min(nz_f):.0f}x |",
          f"| final-answer span (E9) | {rs_span:+.3f} "
          f"| {statistics.mean(dr_s):.4f} "
          f"| {max(nz_s) / min(nz_s):.0f}x |",
          "",
          f"Sanity: rs of recomputed B_full against the paper CSV's |MdR| "
          f"column = {rs_csv:+.3f} (protocol reproduction anchor).",
          "",
          "| variant | B_full | dR_full | B_span | dR_span | dR_csv |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['label']} | {r['B_full']:.2f} | {r['dR_full']:.4f} "
                  f"| {r['B_span']:.2f} | {r['dR_span']:.4f} "
                  f"| {r['dR_csv']:.4f} |")
    md += ["",
           "Reading: if rs_span > rs_full with a larger |dR| scale/range, "
           "conditioning on the graded span restores the rank signal that "
           "full-CoT averaging washes out (pCi8-W4 mitigation). If not, "
           "report honestly and keep GSM8K as a first-class limitation — "
           "the SNR diagnosis of App. F.3 stands either way.", ""]
    (OUT_DIR / f"E9_results_{args.family}.md").write_text("\n".join(md))
    print("\n".join(md[:14]))


if __name__ == "__main__":
    main()
