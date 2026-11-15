#!/usr/bin/env python3
"""
Similarity scores on the quantized variants, at the size of the main grid
(Appendix F.1): linear CKA, SVCCA and the Procrustes distance next to the
PRISM quantities, on the same features.

Protocol
--------
For one family (Llama-3.1-8B or Qwen3-8B-Base) and each of the five
benchmarks, the script loads the target and every variant of the manifest
(scripts/rounds/data/variant_manifest.csv), extracts paired answer-token
features (512 sequences, seed 42) and computes on these features:

    cka        linear CKA (centered)
    svcca      mean canonical correlation after a 99%-variance PCA (at most
               256 components)
    procr_dist size-and-shape Procrustes distance (aligned residual / sqrt(n))
    omega_I    trace similarity under the identity alignment
    omega_W    nuclear-norm similarity (Procrustes alignment)

The risk gap |dR| and the bounds Bound_I and Bound_W of each (variant,
benchmark) are not recomputed here: they are joined, on the variant label and
the benchmark, from the result tables written by run_quantization.py, which
writes one CSV per model and dataset
(prism_<model>_<dataset>_n<N>_concat.csv).  --results-csv takes those files, a
directory that holds them, or a glob pattern; only the rows of the family's
target model with z_mode "concat" are read.  The risk gap is the column
`|AdR|` of those files (the answer-only gap); where a row has no `|AdR|` value, the full-sequence `|dR|`
is used (common_quant.risk_gaps_from_csv).  In this script's own output CSV
the joined gap is stored in a column named `|MdR|`.

Ranking convention: the table gives the Spearman r_s between a degradation
score and |dR| across the variants of one benchmark.  Similarities enter as
1 - similarity, distances and bounds as they are, so a higher r_s means a
better ranking in every column.

The functions linear_cka and svcca are also imported by three_seed_round.py.

Usage (from the supplementary root)
    python scripts/rounds/similarity_scores.py --family llama --results-csv RESULTS_DIR
    python scripts/rounds/similarity_scores.py --family qwen  --results-csv 'RESULTS_DIR/prism_qwen3-8b-base_*_n512_concat.csv'

RESULTS_DIR is the output directory of run_quantization.py.  The files passed
must come from one run (one sample size); a variant and benchmark found twice
with different values is an error.

Output: out/rounds/similarity/{family}_metrics.csv
        out/rounds/similarity/{family}_spearman.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent          # supplementary root (holds prism/)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from common_quant import (BENCHMARKS, FAMILIES, extract_Z,              # noqa: E402
                          free_cuda, load_proxy, load_target,
                          risk_gaps_from_csv, subsample_tokens,
                          variants_from_csv)
from prism.data.loaders import load_task_data                          # noqa: E402
from prism.experiments.quantization import (_TRUST_REMOTE_CODE,        # noqa: E402
                                            pinned_revision)
from transformers import AutoTokenizer                                 # noqa: E402

OUT_DIR = REPO / "out" / "rounds" / "similarity"


# ----------------------------------------------------------------------
# Similarity metrics on paired token features (n, d)
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
    n, d = X.shape
    if n > 32768:
        # covariance path (d x d eigh) for large n; it keeps the same
        # subspace as the SVD of the (n x d) matrix below.
        C = X.T @ X
        evals, evecs = torch.linalg.eigh(C)             # ascending order
        evals = evals.flip(0).clamp(min=0)
        evecs = evecs.flip(1)
        ev = evals / evals.sum().clamp(min=1e-12)
        k = int(torch.searchsorted(ev.cumsum(0),
                                   torch.tensor(ev_keep,
                                                device=ev.device)).item()) + 1
        k = max(2, min(k, cap, d))
        S = evals[:k].sqrt().clamp(min=1e-12)
        return (X @ evecs[:, :k]) / S    # left singular vectors (n, k)
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


def omega_nuclear(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Omega under the Procrustes alignment: ||X^T Y||_* / (||X||_F ||Y||_F).

    Same denominator as omega_trace; the trace is replaced by the nuclear
    norm, which is the value at the optimal orthogonal alignment."""
    nuc = torch.linalg.svdvals(X.T @ Y).sum()
    denom = torch.linalg.matrix_norm(X) * torch.linalg.matrix_norm(Y)
    return (nuc / denom.clamp(min=1e-12)).item()


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


def result_files(specs) -> list:
    """Expand --results-csv: each entry is a CSV file, a directory (its
    prism_*.csv files, else its *.csv files) or a glob pattern."""
    files = []
    for spec in specs:
        p = Path(spec)
        if p.is_dir():
            found = sorted(p.glob("prism_*.csv")) or sorted(p.glob("*.csv"))
        elif p.is_file():
            found = [p]
        else:
            found = sorted(Path(f) for f in glob.glob(str(spec)))
        if not found:
            raise SystemExit(f"--results-csv: no CSV file found for {spec!r}")
        files.extend(f for f in found if f not in files)
    return files


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=list(FAMILIES), required=True)
    ap.add_argument("--results-csv", dest="results_csv", required=True, nargs="+",
                    help="result tables written by run_quantization.py (one CSV "
                         "per model and dataset): files, a directory holding "
                         "them, or a glob pattern. Columns read: target_model, "
                         "Label, dataset, z_mode, Bound_I, Bound_W and the risk "
                         "gap |AdR| (|dR| where |AdR| is empty); the risk gaps "
                         "and the bounds are joined from them")
    ap.add_argument("--num_samples", type=int, default=512,
                    help="sequences per benchmark (main grid: 512)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--token_cap", type=int, default=65536,
                    help="upper limit on paired token rows; the default lies "
                         "above the largest count of the grid (GSM8K), so no "
                         "benchmark is subsampled")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_id = FAMILIES[args.family]
    specs = variants_from_csv(args.family)
    csv_files = result_files(args.results_csv)
    risk = risk_gaps_from_csv(args.family, csv_files)
    # Bounds joined on the same (Label, dataset) keys: Bound_I under the
    # identity alignment and Bound_W under the Procrustes alignment.  Every
    # column of the Spearman table is computed over the variants that have a
    # row here, so all columns share the same n.
    bound_csv, bound_csv_W = {}, {}
    for path in csv_files:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("target_model") != target_id:
                    continue
                if r.get("z_mode") not in (None, "", "concat"):
                    continue
                try:
                    bound_csv[(r["Label"], r["dataset"])] = float(r["Bound_I"])
                    bound_csv_W[(r["Label"], r["dataset"])] = float(r["Bound_W"])
                except ValueError:
                    pass
    print(f"Family {args.family}: target={target_id}, {len(specs)} proxies, "
          f"{len(BENCHMARKS)} benchmarks")
    print(f"Result tables: {len(csv_files)} file(s); risk gaps for {len(risk)} "
          f"and bounds for {len(bound_csv)} (variant, benchmark) pairs of this "
          f"family")
    if not risk:
        raise SystemExit(f"no row of {target_id} with a risk gap in the result "
                         f"tables passed with --results-csv")

    tokenizer = AutoTokenizer.from_pretrained(
        target_id, trust_remote_code=_TRUST_REMOTE_CODE,
        revision=pinned_revision(target_id))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The extraction assumes right padding.
    tokenizer.padding_side = "right"

    loaders = {
        b: load_task_data(b, split="test", num_samples=args.num_samples,
                          batch_size=args.batch_size, tokenizer=tokenizer,
                          max_length=args.max_length, seed=args.seed)
        for b in BENCHMARKS
    }

    # ── Target features once per benchmark, cached to disk ─────────────
    # An unreadable cache file (for example from an interrupted save) is
    # deleted and re-extracted; files are written with an atomic rename.
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
                  f"({type(exc).__name__}); deleting and re-extracting")
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
            tmp.rename(zt_cache / f"{b}.pt")
            Z_T[b] = Z.half().float()
            print(f"  {b}: Z={list(Z.shape)}  ({time.time() - t0:.0f}s)")
        del target
        free_cuda()

    # ── Resume: keep rows already in the output CSV, skip completed variants ──
    csv_path = OUT_DIR / f"{args.family}_metrics.csv"
    rows = []
    done: dict = {}
    if csv_path.exists():
        for r in csv.DictReader(open(csv_path)):
            for k in ("cka", "svcca", "procr_dist", "omega_I", "omega_W",
                      "bound_I", "bound_W", "|MdR|"):
                r[k] = float(r[k])
            r["n_tokens"] = int(r["n_tokens"])
            rows.append(r)
            done.setdefault(r["label"], set()).add(r["dataset"])
        print(f"[resume] {csv_path.name}: {len(rows)} rows kept, "
              f"{len(done)} variants present")

    # ── Per-variant loop: load once, all benchmarks ─────────────────────
    for spec in specs:
        label = spec["label"]
        done_b = done.get(label, set())
        if done_b >= set(BENCHMARKS):
            print(f"=== {label}: complete in the output CSV, skipped ===")
            continue
        print(f"\n=== {label} ({spec['kind']}) ===")
        t_load = time.time()
        try:
            proxy = load_proxy(spec, args.device)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [FAIL load] {exc}")
            continue
        print(f"  [load] {label}: {time.time() - t_load:.0f}s")
        for b in BENCHMARKS:
            if b in done_b:
                continue
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
                "omega_W": omega_nuclear(X, Y),
                "bound_I": bound_csv.get((label, b), float("nan")),
                "bound_W": bound_csv_W.get((label, b), float("nan")),
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

        # Rewrite the output CSV after every variant.
        with open(OUT_DIR / f"{args.family}_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ── Spearman table ──────────────────────────────────────────────────
    METRICS = ("bound_I", "bound_W", "cka", "svcca", "procr_dist",
               "omega_I", "omega_W")
    AS_IS = {"bound_I", "bound_W", "procr_dist"}   # degradation-oriented
    md = [f"# Similarity scores and PRISM quantities, family {args.family}", "",
          "Entries are the Spearman r_s between a degradation score and |dR| "
          "across the variants of a benchmark: Bound_I and Bound_W (joined "
          "from the result table), 1 - cka, 1 - svcca, procr_dist, "
          "1 - omega_I, 1 - omega_W. Higher is better. Every column is "
          "computed over the same variants (column n).", ""]
    md.append("| benchmark | n | PRISM B_I | PRISM B_W | 1-CKA | 1-SVCCA "
              "| Procrustes dist | 1-Omega_I | 1-Omega_W |")
    md.append("|---|---|---|---|---|---|---|---|---|")
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

    (OUT_DIR / f"{args.family}_spearman.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
