#!/usr/bin/env python3
"""
E8 — Does near-isometry survive aggressive quantization? (pCi8-W6)

Question posed by the reviewer: PRISM restricts alignment to rotation x
scale — is that restriction still adequate after aggressive quantization?
Direct measurement: how much residual does the O(d)-restriction COST versus
the best arbitrary linear alignment?

    iso_gain = 1 - res_lin / res_orth   in [0, 1)

    res_orth = min_{s, W in O(r)} ||X - s Y W||_F   (rotation x scale)
    res_lin  = min_{A}            ||X - Y A||_F     (arbitrary linear)

iso_gain ~ 0  => the orthogonal family is already (near-)optimal — the
near-isometry premise holds; growth with bit-width aggressiveness would
quantify its erosion.

Design note (why a subspace): token counts on answer-only benchmarks are
SMALLER than d=4096 (mmlu ~511 tokens), so an unconstrained d x d regression
is underdetermined (res_lin = 0 exactly — meaningless; this is what broke
the first-cut iso_dev column inside E1, whose FP16 control scored 18-50).
We therefore pose both problems in the top-r principal subspace of X
(r = min(256, n_tokens // 4), variance share reported): well-conditioned
(n >= 4r), FP16 control lands near 0 by construction, and the question —
"is the OPTIMAL alignment near-isometric where the features actually
live?" — is answered where it matters.

Reuses E1's cached target features (out/E1/{family}_ZT); loads only the
requested proxies. Default: llama x {Q8_0, Q4_K_M, Q2_K, NF4} + FP16
control x {mmlu, squad} ~= 15-25 min on the RTX 5090.

Usage (repo root, GPU box):
    python rebuttal_exp/exp_e8_isometry.py --family llama

Output: rebuttal_exp/out/E8/E8_results_{family}.md  (+ .csv)
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from common_quant import (FAMILIES, extract_Z, free_cuda, load_proxy,  # noqa: E402
                          load_target, subsample_tokens, variants_from_csv)
from prism.data.loaders import load_task_data                          # noqa: E402
from transformers import AutoTokenizer                                 # noqa: E402

OUT_DIR = HERE / "out" / "E8"
E1_CACHE = HERE / "out" / "E1"

DEFAULT_TIERS = ["FP16", "Q8_0", "Q4_K_M", "Q2_K", "NF4"]


def subspace_residuals(X: torch.Tensor, Y: torch.Tensor, cap: int = 256):
    """(res_orth, res_lin, r, var_share): alignment-family comparison with
    EACH side in its OWN top-r principal basis (rotation-invariant — any
    rotation between the two subspaces is absorbed by W / A, exactly like
    SVCCA's PCA-then-CCA structure; a shared basis would spuriously punish
    the orthogonal family whenever quantization rotates the subspace)."""
    n = X.shape[0]
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    _, Sx, Vtx = torch.linalg.svd(Xc, full_matrices=False)
    _, _, Vty = torch.linalg.svd(Yc, full_matrices=False)
    # r must not exceed the EFFECTIVE rank (90% energy): beyond it the extra
    # basis directions are noise, which an unconstrained A can simply zero
    # out while O(r) x single-scale cannot — inflating iso_gain spuriously
    # (identity control fails without this guard).
    ev = (Sx ** 2).cumsum(0) / (Sx ** 2).sum().clamp(min=1e-12)
    r90 = int((ev < 0.90).sum().item()) + 1
    r = max(8, min(cap, n // 4, r90))
    var_share = ((Sx[:r] ** 2).sum() / (Sx ** 2).sum().clamp(min=1e-12)).item()
    Xr = Xc @ Vtx[:r].T                 # (n, r) — X in its own basis
    Yr = Yc @ Vty[:r].T                 # (n, r) — Y in its own basis
    # scaled-orthogonal residual: min_{s, W in O(r)} ||Xr - s Yr W||
    nuc = torch.linalg.svdvals(Yr.T @ Xr).sum()
    y2 = (Yr ** 2).sum().clamp(min=1e-12)
    res_orth2 = (Xr ** 2).sum() - nuc ** 2 / y2
    # unconstrained linear residual: min_A ||Xr - Yr A||  (n >= 4r)
    sol = torch.linalg.lstsq(Yr, Xr)
    res_lin2 = (Xr - Yr @ sol.solution).pow(2).sum()
    # relative residual: when this sits at the noise floor (FP16 control),
    # the lin/orth RATIO is dominated by the r/n overfitting advantage and
    # iso_gain is not meaningful — flag such rows instead of citing them.
    rel_orth = (res_orth2.clamp(min=0).sqrt()
                / (Xr ** 2).sum().sqrt().clamp(min=1e-12)).item()
    return (res_orth2.clamp(min=0).sqrt().item(),
            res_lin2.clamp(min=0).sqrt().item(), r, var_share, rel_orth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(FAMILIES), default="llama")
    ap.add_argument("--tiers", nargs="+", default=DEFAULT_TIERS,
                    help="variant tags to load (matched against 'BF16 vs X')")
    ap.add_argument("--benchmarks", nargs="+", default=["mmlu", "squad"])
    ap.add_argument("--num_samples", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--token_cap", type=int, default=16384)
    ap.add_argument("--subspace_cap", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    target_id = FAMILIES[args.family]
    specs = [s for s in variants_from_csv(args.family)
             if s["label"].split(" vs ", 1)[-1] in args.tiers]
    print(f"{args.family}: {len(specs)} variants x {args.benchmarks}")

    tokenizer = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    loaders = {
        b: load_task_data(b, split="test", num_samples=args.num_samples,
                          batch_size=args.batch_size, tokenizer=tokenizer,
                          max_length=args.max_length, seed=args.seed)
        for b in args.benchmarks
    }

    # target features: reuse E1's cache when present
    Z_T = {}
    missing = []
    for b in args.benchmarks:
        p = E1_CACHE / f"{args.family}_ZT" / f"{b}.pt"
        if p.exists():
            try:
                Z_T[b] = torch.load(p).float()
                continue
            except Exception:                              # noqa: BLE001
                pass
        missing.append(b)
    if missing:
        print(f"extracting target features for {missing} ...")
        target = load_target(target_id, args.device)
        for b in missing:
            Z_T[b] = extract_Z(target, loaders[b], args.device)
        del target
        free_cuda()

    rows = []
    for spec in specs:
        label = spec["label"]
        print(f"\n=== {label} ===")
        t0 = time.time()
        try:
            proxy = load_proxy(spec, args.device)
        except Exception as exc:                           # noqa: BLE001
            print(f"  [FAIL load] {exc}")
            continue
        for b in args.benchmarks:
            Z_P = extract_Z(proxy, loaders[b], args.device)
            Xc, Yc = subsample_tokens(Z_T[b], Z_P, args.token_cap,
                                      seed=args.seed)
            ro, rl, r, vs, rel = subspace_residuals(
                Xc.to(args.device), Yc.to(args.device), args.subspace_cap)
            gain = 1 - rl / max(ro, 1e-12)
            floor = rel < 0.02          # noise-floor rows: gain unreliable
            rows.append({"label": label, "benchmark": b, "n_tokens": Xc.shape[0],
                         "r": r, "var_share": vs, "rel_orth": rel,
                         "res_orth": ro, "res_lin": rl, "iso_gain": gain,
                         "at_floor": floor})
            print(f"  {b}: r={r} var={vs:.2f} rel_orth={rel:.3f} "
                  f"iso_gain={gain:.3f}{' [floor]' if floor else ''}")
            del Z_P
            free_cuda()
        del proxy
        free_cuda()
        print(f"  ({time.time() - t0:.0f}s)")

        with open(OUT_DIR / f"{args.family}_isometry.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    md = [f"# E8 — isometry-restriction cost vs bit-width "
          f"({args.family}, top-r subspace, r<=256)", "",
          "iso_gain = 1 - res_lin/res_orth: residual reduction an ARBITRARY "
          "linear alignment achieves over the best rotation x scale. "
          "~0 => the O(d) restriction costs nothing (near-isometry holds).", "",
          "| variant | benchmark | r | var share | rel res (orth) "
          "| iso_gain |", "|---|---|---|---|---|---|"]
    for row in rows:
        tag = " [noise floor]" if row["at_floor"] else ""
        md.append(f"| {row['label']} | {row['benchmark']} | {row['r']} "
                  f"| {row['var_share']:.2f} | {row['rel_orth']:.3f} "
                  f"| {row['iso_gain']:.3f}{tag} |")
    by_lab = {}
    for row in rows:
        if not row["at_floor"]:
            by_lab.setdefault(row["label"], []).append(row["iso_gain"])
    md += ["", "Mean iso_gain per variant (floor rows excluded): " + ", ".join(
        f"{lab.split(' vs ')[-1]} {statistics.mean(v):.3f}"
        for lab, v in by_lab.items()),
        "", "Rows at the noise floor (rel res < 2%, e.g. the FP16 control) "
        "have ratio-of-tiny-numbers instability; the claim rests on the "
        "quantized rows, whose residuals sit far above the floor.", ""]
    (OUT_DIR / f"E8_results_{args.family}.md").write_text("\n".join(md))
    print("\n".join(md[-4:]))


if __name__ == "__main__":
    main()
