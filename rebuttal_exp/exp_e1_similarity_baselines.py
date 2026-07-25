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

from common_quant import (BENCHMARKS, CSV_PATH, FAMILIES, extract_Z,    # noqa: E402
                          free_cuda, load_proxy, load_target,
                          risk_gaps_from_csv, subsample_tokens,
                          variants_from_csv)
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


def isometry_deviation(X: torch.Tensor, Y: torch.Tensor) -> float:
    """E8 (pCi8-W6): how far the OPTIMAL UNCONSTRAINED alignment is from a
    scaled isometry. A* = argmin_A ||X A - Y||_F (ridge-regularised normal
    equations); returns CV of A*'s singular values — 0 iff A* is exactly a
    scalar x orthogonal map, growing as quantization distorts geometry
    anisotropically. Zero extra GPU cost on E1's paired features."""
    d = X.shape[1]
    XtX = X.T @ X
    ridge = 1e-6 * XtX.diagonal().mean()
    A = torch.linalg.solve(XtX + ridge * torch.eye(d, device=X.device), X.T @ Y)
    sv = torch.linalg.svdvals(A)
    return (sv.std(unbiased=False) / sv.mean().clamp(min=1e-12)).item()


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
    # Paper bound joined on the same (Label, dataset) keys — gives a PRISM
    # column computed on EXACTLY the variant subset that loads, so every
    # column of the Spearman table shares the same n (fair comparison even
    # if a proxy fails to load).
    bound_csv = {}
    for r in csv.DictReader(open(CSV_PATH)):
        if r["target_model"] == target_id:
            try:
                bound_csv[(r["Label"], r["dataset"])] = float(r["Bound_I"])
            except ValueError:
                pass
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
    # Self-healing cache: a run killed mid-torch.save leaves a truncated
    # .pt that EXISTS but EOFErrors on load (seen 2026-07-24, qwen/arc).
    # (a) validate every existing file up front, deleting unreadable ones;
    # (b) write atomically (tmp -> rename) so it can never recur.
    zt_cache = OUT_DIR / f"{args.family}_ZT"
    zt_cache.mkdir(exist_ok=True)
    Z_T: dict = {}
    missing = []
    for b in BENCHMARKS:
        p = zt_cache / f"{b}.pt"
        if not p.exists():
            missing.append(b)
            continue
        try:
            Z_T[b] = torch.load(p).float()
        except Exception as exc:                       # noqa: BLE001
            print(f"  [cache] {p.name} unreadable "
                  f"({type(exc).__name__}: truncated save?) — deleting "
                  f"and re-extracting")
            p.unlink()
            missing.append(b)
    if missing:
        print(f"Extracting target features for {missing} ...")
        target = load_target(target_id, args.device)
        for b in missing:
            t0 = time.time()
            Z = extract_Z(target, loaders[b], args.device)
            tmp = zt_cache / f"{b}.pt.tmp"
            torch.save(Z.half(), tmp)
            tmp.rename(zt_cache / f"{b}.pt")           # atomic publish
            # fp16 roundtrip in memory too — identical precision whether a
            # benchmark came from cache or from this fresh extraction.
            Z_T[b] = Z.half().float()
            print(f"  {b}: Z={list(Z.shape)}  ({time.time() - t0:.0f}s)")
        del target
        free_cuda()

    # ── Per-proxy loop: load once, all benchmarks ───────────────────────
    rows = []
    for spec in specs:
        label = spec["label"]
        if spec["kind"] == "dtype":
            pass  # FP16 control row — keep, it anchors the near-zero end
        print(f"\n=== {label} ({spec['kind']}) ===")
        t_load = time.time()
        try:
            proxy = load_proxy(spec, args.device)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [FAIL load] {exc}")
            continue
        # Timing line harvested by E12's cost table (G3T9-W1).
        print(f"  [load] {label}: {time.time() - t_load:.0f}s")
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
                "iso_dev": isometry_deviation(X, Y),
                "bound_I": bound_csv.get((label, b), float("nan")),
                "|MdR|": risk.get((label, b), float("nan")),
            }
            rows.append(row)
            del X, Y, Z_P
            free_cuda()
            print(f"  {b}: cka={row['cka']:.4f} svcca={row['svcca']:.4f} "
                  f"procr={row['procr_dist']:.3f} omega={row['omega_I']:.4f} "
                  f"iso_dev={row['iso_dev']:.4f} "
                  f"|dR|={row['|MdR|']:.4f} ({time.time() - t0:.0f}s)")
        del proxy
        free_cuda()

        # Incremental write — survive interruptions.
        with open(OUT_DIR / f"{args.family}_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ── Spearman table: degradation-score convention ────────────────────
    METRICS = ("bound_I", "cka", "svcca", "procr_dist", "omega_I")
    AS_IS = {"bound_I", "procr_dist"}      # already degradation-oriented
    md = [f"# E1 — similarity-baseline ranking, family {args.family}", "",
          "Score convention: rs( degradation-score, |dR| ), where the "
          "degradation score is Bound_I (PRISM, joined from the paper CSV "
          "on the same keys), (1 - cka), (1 - svcca), procr_dist, "
          "(1 - omega_I). Higher rs = better ranking. All columns are "
          "computed over the IDENTICAL variant subset (same n), so the "
          "comparison stays fair even if a proxy fails to load.", ""]
    md.append("| benchmark | n | PRISM B | 1-CKA | 1-SVCCA "
              "| Procrustes dist | 1-Omega_I |")
    md.append("|---|---|---|---|---|---|---|")
    per_metric = {m: [] for m in METRICS}
    for b in BENCHMARKS:
        sub = [r for r in rows if r["dataset"] == b
               and not math.isnan(r["|MdR|"])
               and not math.isnan(r["bound_I"])]
        if len(sub) < 3:
            continue
        dr = [r["|MdR|"] for r in sub]
        cells = []
        for m in METRICS:
            score = [r[m] if m in AS_IS else 1 - r[m] for r in sub]
            rs = spearman(score, dr)
            per_metric[m].append(rs)
            cells.append(f"{rs:+.3f}")
        md.append(f"| {b} | {len(sub)} | " + " | ".join(cells) + " |")
    md.append("| **mean** | | " + " | ".join(
        f"**{statistics.mean(per_metric[m]):+.3f}**"
        for m in METRICS) + " |")

    # ── E8 (pCi8-W6): isometry deviation vs variant/bit-width ──────────
    md += ["", "## E8 — isometry deviation of the optimal unconstrained "
           "alignment (pCi8-W6)", "",
           "CV of singular values of A* = argmin ||Z_T A - Z_P||; 0 = "
           "exactly scaled-orthogonal. Mean over benchmarks per variant:",
           "", "| variant | mean iso_dev |", "|---|---|"]
    seen_labels = []
    for r in rows:
        if r["label"] not in seen_labels:
            seen_labels.append(r["label"])
    for lab in seen_labels:
        vals = [r["iso_dev"] for r in rows if r["label"] == lab]
        md.append(f"| {lab} | {statistics.mean(vals):.4f} |")

    (OUT_DIR / f"{args.family}_spearman.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
