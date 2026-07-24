#!/usr/bin/env python3
"""
E1 — Direct same-protocol comparison of PRISM against CKA / SVCCA /
Procrustes distance for variant ranking.  (G3T9-W3; also cited in
pCi8-W3's "why not CKA" and the Global Response.)

Protocol
--------
For each family (Llama-3.1-8B, Qwen3-8B-Base) and each of the 5 paper
benchmarks: reload the exact proxies of the paper (parsed from
exp_result/quantization/quantization_merged_slim.csv), re-extract paired
token-level features (concat z-mode, 512 samples — identical to the paper),
and compute on the SAME features:

    cka        linear CKA (centered)
    svcca      mean canonical correlation after 99%-variance PCA (cap 256)
    procr_dist size-and-shape Procrustes distance  (aligned residual / sqrt(n))
    omega_I    PRISM trace similarity (sanity link to the paper's tables)

|dR| per (variant, benchmark) is joined from the existing CSV, so the
Spearman table is directly comparable with the paper's PRISM numbers.

Ranking convention: similarity metrics (cka, svcca, omega_I) correlate
NEGATIVELY with degradation, distances positively; the report uses
rs(metric-as-degradation-score, |dR|) with the sign convention fixed so
that "higher = better ranking" for every column.

Cost: one load per proxy (~2-5 min incl. GGUF dequant), 5 forward passes
per proxy. ~12 proxies + 1 target per family -> ~1.5-2.5 h/family on the
RTX 5090.

Usage (repo root, GPU box):
    python rebuttal_exp/exp_e1_similarity_baselines.py --family llama
    python rebuttal_exp/exp_e1_similarity_baselines.py --family qwen

Output: rebuttal_exp/out/E1/{family}_metrics.csv
        rebuttal_exp/out/E1/{family}_spearman.md
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

from common_quant import (BENCHMARKS, FAMILIES, extract_Z, free_cuda,   # noqa: E402
                          load_proxy, load_target, risk_gaps_from_csv,
                          subsample_tokens, variants_from_csv)
from prism.data.loaders import load_task_data                          # noqa: E402
from transformers import AutoTokenizer                                 # noqa: E402

OUT_DIR = HERE / "out" / "E1"


# ----------------------------------------------------------------------
# Similarity metrics on paired token features (n, d) — all on GPU fp32
# ----------------------------------------------------------------------
def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xty = (X.T @ Y)
    num = (xty ** 2).sum()
    den = torch.linalg.matrix_norm(X.T @ X) * torch.linalg.matrix_norm(Y.T @ Y)
    return (num / den.clamp(min=1e-12)).item()


def _pca_basis(X: torch.Tensor, ev_keep: float = 0.99, cap: int = 256):
    X = X - X.mean(0, keepdim=True)
    # economy SVD via covariance (d x d) — n >> d never happens here (d=4096)
    U, S, _ = torch.linalg.svd(X, full_matrices=False)
    ev = (S ** 2) / (S ** 2).sum().clamp(min=1e-12)
    k = int(torch.searchsorted(ev.cumsum(0), torch.tensor(ev_keep, device=S.device)).item()) + 1
    k = max(2, min(k, cap, S.shape[0]))
    return U[:, :k]          # orthonormal columns spanning the kept subspace


def svcca(X: torch.Tensor, Y: torch.Tensor) -> float:
    Qx, Qy = _pca_basis(X), _pca_basis(Y)
    rho = torch.linalg.svdvals(Qx.T @ Qy)
    return rho.mean().item()


def procrustes_distance(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Size-and-shape distance: min_W ||X - Y W||_F / sqrt(n)  over O(d)."""
    nuc = torch.linalg.svdvals(X.T @ Y).sum()
    sq = (X ** 2).sum() + (Y ** 2).sum() - 2 * nuc
    return math.sqrt(max(sq.item(), 0.0) / X.shape[0])


def omega_trace(X: torch.Tensor, Y: torch.Tensor) -> float:
    denom = torch.linalg.matrix_norm(X) * torch.linalg.matrix_norm(Y)
    return ((X * Y).sum() / denom.clamp(min=1e-12)).item()


# ----------------------------------------------------------------------
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
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(FAMILIES), required=True)
    ap.add_argument("--num_samples", type=int, default=512,
                    help="samples per benchmark (paper: 512)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--token_cap", type=int, default=16384,
                    help="paired-token subsample for CKA/SVCCA tractability")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_id = FAMILIES[args.family]
    specs = variants_from_csv(args.family)
    risk = risk_gaps_from_csv(args.family)
    print(f"Family {args.family}: target={target_id}, {len(specs)} proxies, "
          f"{len(BENCHMARKS)} benchmarks")

    tokenizer = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loaders = {
        b: load_task_data(b, split="test", num_samples=args.num_samples,
                          batch_size=args.batch_size, tokenizer=tokenizer,
                          max_length=args.max_length, seed=args.seed)
        for b in BENCHMARKS
    }

    # ── Target features once per benchmark, cached to disk ─────────────
    zt_cache = OUT_DIR / f"{args.family}_ZT"
    zt_cache.mkdir(exist_ok=True)
    Z_T: dict = {}
    missing = [b for b in BENCHMARKS if not (zt_cache / f"{b}.pt").exists()]
    if missing:
        print(f"Extracting target features for {missing} ...")
        target = load_target(target_id, args.device)
        for b in missing:
            t0 = time.time()
            Z = extract_Z(target, loaders[b], args.device)
            torch.save(Z.half(), zt_cache / f"{b}.pt")
            print(f"  {b}: Z={list(Z.shape)}  ({time.time() - t0:.0f}s)")
        del target
        free_cuda()
    for b in BENCHMARKS:
        Z_T[b] = torch.load(zt_cache / f"{b}.pt").float()

    # ── Per-proxy loop: load once, all benchmarks ───────────────────────
    rows = []
    for spec in specs:
        label = spec["label"]
        if spec["kind"] == "dtype":
            pass  # FP16 control row — keep, it anchors the near-zero end
        print(f"\n=== {label} ({spec['kind']}) ===")
        try:
            proxy = load_proxy(spec, args.device)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [FAIL load] {exc}")
            continue
        for b in BENCHMARKS:
            t0 = time.time()
            try:
                Z_P = extract_Z(proxy, loaders[b], args.device)
            except Exception as exc:                   # noqa: BLE001
                print(f"  [FAIL extract {b}] {exc}")
                continue
            Xc, Yc = subsample_tokens(Z_T[b], Z_P, args.token_cap, seed=args.seed)
            X = Xc.to(args.device)
            Y = Yc.to(args.device)
            row = {
                "family": args.family, "label": label, "dataset": b,
                "n_tokens": X.shape[0],
                "cka": linear_cka(X, Y),
                "svcca": svcca(X, Y),
                "procr_dist": procrustes_distance(X, Y),
                "omega_I": omega_trace(X, Y),
                "|MdR|": risk.get((label, b), float("nan")),
            }
            rows.append(row)
            del X, Y, Z_P
            free_cuda()
            print(f"  {b}: cka={row['cka']:.4f} svcca={row['svcca']:.4f} "
                  f"procr={row['procr_dist']:.3f} omega={row['omega_I']:.4f} "
                  f"|dR|={row['|MdR|']:.4f} ({time.time() - t0:.0f}s)")
        del proxy
        free_cuda()

        # Incremental write — survive interruptions.
        with open(OUT_DIR / f"{args.family}_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ── Spearman table: degradation-score convention ────────────────────
    md = [f"# E1 — similarity-baseline ranking, family {args.family}", "",
          "Score convention: rs( degradation-score, |dR| ), where the "
          "degradation score is (1 - cka), (1 - svcca), procr_dist, "
          "(1 - omega_I). Higher rs = better ranking. PRISM full-bound "
          "column comes from the paper CSV (Bound_I).", ""]
    md.append("| benchmark | n | 1-CKA | 1-SVCCA | Procrustes dist | 1-Omega_I |")
    md.append("|---|---|---|---|---|---|")
    per_metric = {m: [] for m in ("cka", "svcca", "procr_dist", "omega_I")}
    for b in BENCHMARKS:
        sub = [r for r in rows if r["dataset"] == b
               and not math.isnan(r["|MdR|"])]
        if len(sub) < 3:
            continue
        dr = [r["|MdR|"] for r in sub]
        cells = []
        for m in ("cka", "svcca", "procr_dist", "omega_I"):
            score = [r[m] if m == "procr_dist" else 1 - r[m] for r in sub]
            rs = spearman(score, dr)
            per_metric[m].append(rs)
            cells.append(f"{rs:+.3f}")
        md.append(f"| {b} | {len(sub)} | " + " | ".join(cells) + " |")
    md.append("| **mean** | | " + " | ".join(
        f"**{statistics.mean(per_metric[m]):+.3f}**"
        for m in ("cka", "svcca", "procr_dist", "omega_I")) + " |")
    (OUT_DIR / f"{args.family}_spearman.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
