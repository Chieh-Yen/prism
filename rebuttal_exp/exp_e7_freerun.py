#!/usr/bin/env python3
"""
E7 — Free-running generation subset experiment (AC-D; G3T9-W2(3),
8VrD-W4(2)/Q2: "surrogate errors affect subsequent context").

Protocol (one benchmark subset, default MMLU, N prompts):
  For each variant P:
    1. teacher-forced pass on the gold (prompt + answer) sequences
       -> B_tf, |dR|_tf on this subset (apples-to-apples anchor);
    2. P GREEDILY GENERATES its own continuation of every prompt
       (max_new_tokens tokens) — its errors compound into its own context;
    3. both T and P are then scored on P's generated trajectories
       (features + per-token CE on the generated region only)
       -> B_free, |dR|_free.
  Deliverable: Spearman rs(B, |dR|) across variants, teacher-forced vs
  free-running, plus the rank agreement between the two bound columns.

The target model stays resident in CPU RAM and is swapped onto the GPU
after each proxy is released — no double-VRAM, no feature dumps.

Head convention follows the paper: only GGUF k-quants alter the served
lm_head (gamma > 0); BnB/GPTQ/dtype proxies keep the FP16 head.

Cost: ~8-12 min per variant (load + generate N x max_new_tokens + three
extraction passes) -> ~2-2.5 h for the Llama family at N=100 on the 5090.

Usage (repo root, GPU box):
    python rebuttal_exp/exp_e7_freerun.py --family llama \
        --dataset mmlu --num_prompts 100 --max_new_tokens 128

Output: rebuttal_exp/out/E7/{family}_{dataset}_freerun.csv
        rebuttal_exp/out/E7/E7_results_{family}_{dataset}.md
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
                          variants_from_csv)
from prism.core.bounds import UnifiedBound                # noqa: E402
from prism.core.metrics import PRISMMetrics               # noqa: E402
from prism.data.loaders import load_task_data             # noqa: E402
from prism.models.extractors import LLMExtractor          # noqa: E402
from transformers import AutoTokenizer                    # noqa: E402

OUT_DIR = HERE / "out" / "E7"


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
def collect_prompts(dataloader):
    """(prompt_ids list) from a gold dataloader that carries prompt_length."""
    prompts = []
    for batch in dataloader:
        pl = batch.get("prompt_length")
        if pl is None:
            raise SystemExit("Dataset has no prompt_length — pick a QA task "
                             "(mmlu/arc/triviaqa/squad/gsm8k).")
        for i in range(batch["input_ids"].shape[0]):
            n = int(batch["attention_mask"][i].sum().item())
            p = min(int(pl[i].item()), n)
            prompts.append(batch["input_ids"][i, :p].clone())
    return prompts


@torch.no_grad()
def generate_trajectories(model, prompts, tokenizer, device,
                          max_new_tokens, batch_size=4):
    """Greedy continuations; returns list of (ids, prompt_len)."""
    model.eval()
    pad_id = tokenizer.pad_token_id
    out = []
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s:s + batch_size]
        maxlen = max(len(p) for p in chunk)
        ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for i, p in enumerate(chunk):                      # left padding
            ids[i, maxlen - len(p):] = p
            mask[i, maxlen - len(p):] = 1
        gen = model.generate(
            input_ids=ids.to(device), attention_mask=mask.to(device),
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=pad_id,
        ).cpu()
        for i, p in enumerate(chunk):
            row = gen[i]
            seq = torch.cat([p, row[maxlen:]])             # strip left pad
            if pad_id is not None:                          # strip right pad
                keep = (seq != pad_id)
                keep[:len(p)] = True                        # prompts may contain pad_id==eos
                seq = seq[:int(keep.nonzero().max().item()) + 1]
            if len(seq) > len(p):                           # has generated region
                out.append((seq, len(p)))
    return out


def make_batches(trajectories, batch_size=4):
    """Right-padded batches with prompt_length, extractor-compatible."""
    batches = []
    for s in range(0, len(trajectories), batch_size):
        chunk = trajectories[s:s + batch_size]
        maxlen = max(len(t[0]) for t in chunk)
        ids = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        mask = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        pls = torch.zeros(len(chunk), dtype=torch.long)
        for i, (seq, pl) in enumerate(chunk):
            ids[i, :len(seq)] = seq
            mask[i, :len(seq)] = 1
            pls[i] = pl
        batches.append({"input_ids": ids, "attention_mask": mask,
                        "prompt_length": pls})
    return batches


def extract_on(model, batches, device):
    """(Z_concat, mean generated-region token loss)."""
    Z, stats = LLMExtractor().extract_features_and_loss_per_sample(
        model, batches, device, z_mode="concat",
    )
    tok = stats["token_losses"]
    return Z, (tok.mean().item() if tok is not None
               else stats["losses"].mean().item())


def prism_bound(Z_T, Z_P, H_T, H_P, K, device):
    n = min(Z_T.shape[0], Z_P.shape[0])
    res = PRISMMetrics.compute_all(
        Z_T[:n].to(device), H_T.to(device),
        Z_P[:n].to(device), H_P.to(device),
        W=torch.eye(H_T.shape[0], device=device),
    )
    return (K["K_feat"] * res.feature_error
            + K["K_pred"] * res.head_discrepancy), res.omega


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(FAMILIES), default="llama")
    ap.add_argument("--dataset", default="mmlu")
    ap.add_argument("--num_prompts", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_id = FAMILIES[args.family]
    specs = variants_from_csv(args.family)

    tokenizer = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gold_dl = load_task_data(args.dataset, split="test",
                             num_samples=args.num_prompts,
                             batch_size=args.batch_size, tokenizer=tokenizer,
                             max_length=args.max_length, seed=args.seed)
    gold_batches = list(gold_dl)
    prompts = collect_prompts(gold_batches)
    print(f"{len(prompts)} prompts from {args.dataset}")

    print(f"Loading target {target_id} (CPU-resident) ...")
    target = load_target(target_id, device="cpu")
    H_T = LLMExtractor().extract_head(target).float()
    K = UnifiedBound.theoretical_K(H_T.to(args.device))
    H_T = H_T.cpu()
    Z_T_tf, loss_T_tf = None, None

    rows = []
    for spec in specs:
        label = spec["label"]
        print(f"\n=== {label} ===")
        try:
            proxy = load_proxy(spec, args.device)
        except Exception as exc:                          # noqa: BLE001
            print(f"  [FAIL load] {exc}")
            continue
        t0 = time.time()
        Z_P_tf, loss_P_tf = extract_on(proxy, gold_batches, args.device)
        t_gen = time.time()
        trajs = generate_trajectories(proxy, prompts, tokenizer,
                                      args.device, args.max_new_tokens,
                                      args.batch_size)
        gen_tok = sum(len(seq) - pl for seq, pl in trajs)
        dt_gen = max(time.time() - t_gen, 1e-9)
        # Timing line harvested by E12's cost table (G3T9-W1):
        # measured greedy-decode throughput on this hardware.
        print(f"  [gen] {label}: {gen_tok} new tokens in {dt_gen:.0f}s "
              f"({gen_tok / dt_gen:.1f} tok/s)")
        gen_batches = make_batches(trajs, args.batch_size)
        Z_P_fr, loss_P_fr = extract_on(proxy, gen_batches, args.device)
        H_P = (LLMExtractor().extract_head(proxy).float().cpu()
               if spec["kind"] == "gguf" else H_T)
        print(f"  proxy passes done: {len(trajs)} trajectories "
              f"({time.time() - t0:.0f}s)")
        del proxy
        free_cuda()

        target.to(args.device)
        if Z_T_tf is None:
            Z_T_tf, loss_T_tf = extract_on(target, gold_batches, args.device)
        Z_T_fr, loss_T_fr = extract_on(target, gen_batches, args.device)
        target.to("cpu")
        free_cuda()

        B_tf, omega_tf = prism_bound(Z_T_tf, Z_P_tf, H_T, H_P, K, args.device)
        B_fr, omega_fr = prism_bound(Z_T_fr, Z_P_fr, H_T, H_P, K, args.device)
        row = {
            "label": label, "n_traj": len(trajs),
            "B_tf": B_tf, "dR_tf": abs(loss_P_tf - loss_T_tf),
            "B_free": B_fr, "dR_free": abs(loss_P_fr - loss_T_fr),
            "omega_tf": omega_tf, "omega_free": omega_fr,
        }
        rows.append(row)
        print(f"  TF:   B={B_tf:.2f}  |dR|={row['dR_tf']:.4f}")
        print(f"  FREE: B={B_fr:.2f}  |dR|={row['dR_free']:.4f}")
        del Z_P_tf, Z_P_fr, Z_T_fr
        free_cuda()

        stem = f"{args.family}_{args.dataset}"
        with open(OUT_DIR / f"{stem}_freerun.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    rs_tf = spearman([r["B_tf"] for r in rows], [r["dR_tf"] for r in rows])
    rs_fr = spearman([r["B_free"] for r in rows], [r["dR_free"] for r in rows])
    rs_bb = spearman([r["B_tf"] for r in rows], [r["B_free"] for r in rows])
    rs_xx = spearman([r["B_tf"] for r in rows], [r["dR_free"] for r in rows])

    md = [f"# E7 — free-running subset ({args.family}, {args.dataset}, "
          f"n={args.num_prompts} prompts, {args.max_new_tokens} new tokens, "
          f"greedy)", "",
          "Free-run protocol: the PROXY generates its own continuation "
          "(errors compound into its own context); both models are then "
          "scored on those trajectories.", "",
          f"- rs(B, |dR|)  teacher-forced (same subset): {rs_tf:+.3f}",
          f"- rs(B, |dR|)  free-running                : {rs_fr:+.3f}",
          f"- rank agreement rs(B_tf, B_free)          : {rs_bb:+.3f}",
          f"- cross rs(B_tf, |dR|_free)                : {rs_xx:+.3f}",
          "", "| variant | B_tf | dR_tf | B_free | dR_free |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['label']} | {r['B_tf']:.2f} | {r['dR_tf']:.4f} "
                  f"| {r['B_free']:.2f} | {r['dR_free']:.4f} |")
    stem = f"{args.family}_{args.dataset}"
    (OUT_DIR / f"E7_results_{stem}.md").write_text("\n".join(md))
    print("\n".join(md[:12]))


if __name__ == "__main__":
    main()
