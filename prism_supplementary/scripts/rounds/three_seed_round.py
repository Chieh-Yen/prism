#!/usr/bin/env python3
"""Three-seed round of the reference-set-size comparison (Appendix F.2).

For each family (Llama-3.1-8B, Qwen3-8B-Base), each seed, each reference size
and each of the five benchmarks, the script extracts the answer-token features
of the target and of every quantized variant in the manifest
(scripts/rounds/data/variant_manifest.csv) and writes, per (variant,
benchmark, seed, size), one CSV row with

    cka, svcca            linear CKA and SVCCA (functions of similarity_scores.py)
    omega_I               trace similarity under the identity alignment
    omega_W               nuclear-norm similarity (Procrustes alignment W_N)
    rho_T, rho_P          RMS feature scales
    delta_W               feature term under W_N
    gamma_W               head term under W_N
    K_feat, K_pred        constants of the bound (UnifiedBound.theoretical_K)
    bound_W               B_N = K_feat * delta_W + K_pred * gamma_W
    loss_T, loss_P, |MdR| answer-token losses of target and variant, and the
                          risk gap |loss_T - loss_P|

Every column of a row comes from the same forward pass on the same sequences.
The risk gap is the absolute difference of the two models' means, over
sequences, of each sequence's mean answer-token cross-entropy.

Seeds and sizes
    Seeds 43, 44, 45 (the main quantization grid uses seed 42).  A seed fixes
    which test sequences are drawn: the loader shuffles the split with the
    seed and keeps the first n.  Sizes 8, 32, 128, 512 are therefore nested
    prefixes of one shuffle per seed.  The run writes, per family, the
    pairwise overlap between the seeds' 512-sequence draws
    ({family}_seed_draws.md); on small test splits the draws overlap.

Usage (from the supplementary root)
    python3 scripts/rounds/three_seed_round.py --dry-run
    python3 scripts/rounds/three_seed_round.py --family llama --seeds 43 44 45
    python3 scripts/rounds/three_seed_round.py --check
    python3 scripts/rounds/size_study.py --family llama        # the table

Output (out/rounds/three_seed/)
    {family}_seed{S}.csv        one row per (variant, benchmark, size)
    {family}_seed_draws.md      overlap between the seeds' draws
    {family}_ZT/                cached target features and losses
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent          # supplementary root (holds prism/)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

OUT_DIR = ROOT / "out" / "rounds" / "three_seed"
PAPER5 = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]

# Above every token count of the grid, so no feature matrix is subsampled.
TOKEN_CAP = 131072

# ----------------------------------------------------------------------
# metrics on one feature pair
# ----------------------------------------------------------------------
def fresh_metrics(Z_T, Z_P, H_T, H_P, chunk: int = 8192):
    """CKA, SVCCA and the PRISM quantities under W_N on one feature pair; sums run over chunks of `chunk` token rows."""
    import torch

    from similarity_scores import linear_cka, svcca
    from prism.core.bounds import UnifiedBound
    from prism.core.metrics import PRISMMetrics

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        # --- similarity baselines (functions of similarity_scores.py) ---
        cka = linear_cka(Z_T, Z_P)
        sv = svcca(Z_T, Z_P)

        n, d = Z_T.shape
        dev = Z_T.device
        nx2 = torch.zeros((), dtype=torch.float64, device=dev)
        ny2 = torch.zeros((), dtype=torch.float64, device=dev)
        dot = torch.zeros((), dtype=torch.float64, device=dev)
        cross = torch.zeros((d, d), dtype=torch.float64, device=dev)
        sig_p = torch.zeros((d, d), dtype=torch.float64, device=dev)
        for i in range(0, n, chunk):
            xc = Z_T[i:i + chunk].double()
            yc = Z_P[i:i + chunk].double()
            nx2 += (xc * xc).sum()
            ny2 += (yc * yc).sum()
            dot += (xc * yc).sum()
            cross += yc.T @ xc
            sig_p += yc.T @ yc
            del xc, yc
        nx = nx2.sqrt()
        ny = ny2.sqrt()

        cross_hat = cross / (nx * ny).clamp(min=1e-300)
        U, S, Vt = torch.linalg.svd(cross_hat, full_matrices=False)
        W_N = U @ Vt
        omega_raw = S.sum().item()
        omega_i = (dot / (nx * ny).clamp(min=1e-300)).item()
        omega_N = min(omega_raw, 1.0)

        rho_T = (nx / math.sqrt(n)).item()
        rho_P = (ny / math.sqrt(n)).item()
        delta_N = math.sqrt(max((rho_T - rho_P) ** 2
                                + 2 * rho_T * rho_P * (1 - omega_N), 0.0))

        # --- head term at W_N, and the constants of the bound ---
        Sigma_P = (sig_p / n).float()
        gamma_N = PRISMMetrics.head_discrepancy_covariance(
            H_T.float(), H_P.float(), W_N.float(), Sigma_P)
        K = UnifiedBound.theoretical_K(H_T.float())
        bound_N = K["K_feat"] * delta_N + K["K_pred"] * float(gamma_N)
        del cross, cross_hat, sig_p, U, S, Vt, W_N, Sigma_P
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    return {
        "n_tokens": n,
        "cka": cka, "svcca": sv,
        "omega_I": omega_i,
        "omega_W": omega_N,
        "rho_T": rho_T, "rho_P": rho_P,
        "delta_W": delta_N, "gamma_W": float(gamma_N),
        "K_feat": K["K_feat"], "K_pred": K["K_pred"],
        "bound_W": bound_N,
    }


# ----------------------------------------------------------------------
# GPU pass
# ----------------------------------------------------------------------
def loader_fingerprint(loader):
    """Hash of every selected example, to measure what the seed changes.

    The seed's only effect in this script is which examples are drawn:
    prism/data/loaders.py does `shuffle(seed)` then `select(range(num_samples))`
    (subsample_tokens is a no-op because TOKEN_CAP lies above every token
    count).  The hashes are used for two checks:
      * if num_samples >= len(split), every seed selects the same examples and
        the metrics, which are invariant to row order, coincide across seeds;
      * on a small split, two draws share part of their examples, so the seeds
        are overlapping draws.
    """
    import hashlib

    ds = loader.dataset
    ids = ds.encodings["input_ids"]
    # Hash only the non-pad tokens: the loader pads to the longest text of the
    # set it is given, so the same example has a different padded width at
    # n = 8 and at n = 512.
    mask = ds.encodings.get("attention_mask")
    h = set()
    for i, row in enumerate(ids):
        real = row[mask[i] == 1] if mask is not None else row
        h.add(hashlib.blake2b(real.numpy().tobytes(), digest_size=8).hexdigest())
    return {"n_selected": int(ids.shape[0]), "hashes": h}


def run(args) -> None:
    """One target load and one load per variant, for all seeds.

    The variant loop is outside the seed loop, so each variant is loaded once.
    Target features are extracted once per (seed, size, benchmark) and cached
    to disk with an atomic rename.  Rows are appended per (seed, variant), so
    an interrupted run resumes.
    """
    import torch
    from common_quant import (FAMILIES, free_cuda, load_proxy, load_target,
                              subsample_tokens, variants_from_csv)
    from prism.data.loaders import load_task_data
    from prism.experiments.quantization import (_TRUST_REMOTE_CODE,
                                                pinned_revision)
    from prism.models.extractors import LLMExtractor
    from transformers import AutoTokenizer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_id = FAMILIES[args.family]
    specs = variants_from_csv(args.family)
    benches = args.benchmarks or PAPER5
    seeds = list(args.seeds)
    ex = LLMExtractor()

    COLS = ["family", "seed", "n_samples", "label", "dataset", "n_tokens",
            "cka", "svcca", "omega_I", "omega_W",
            "rho_T", "rho_P", "delta_W", "gamma_W",
            "K_feat", "K_pred", "bound_W", "loss_T", "loss_P", "|MdR|"]

    # ── resume: which (seed, label) pairs are already complete? ──
    def csv_path(seed):
        return OUT_DIR / f"{args.family}_seed{seed}.csv"

    # Resume per cell: a cell is skipped iff its (seed, size, label, benchmark)
    # row already exists, so no row is ever written twice.
    have = set()
    if args.force:
        # --force moves the existing files aside: rows are appended, so
        # recomputing on top of an existing file would duplicate every cell.
        for seed in seeds:
            fp_ = csv_path(seed)
            if fp_.exists():
                bak = fp_.with_suffix(".csv.bak")
                fp_.rename(bak)
                print(f"[force] {fp_.name} moved aside to {bak.name}")
    if not args.force:
        for seed in seeds:
            p = csv_path(seed)
            if not p.exists():
                continue
            for r in csv.DictReader(open(p)):
                have.add((seed, int(r["n_samples"]),
                          r["label"], r["dataset"].lower()))
        if have:
            print(f"[resume] {len(have)} cells already on disk; they will be "
                  f"skipped, not re-appended")
    need_cells = {(n, b) for n in sorted(args.sizes) for b in benches}

    def missing_for(seed, label):
        return sorted(c for c in need_cells
                      if (seed, c[0], label, c[1]) not in have)

    tok = AutoTokenizer.from_pretrained(
        target_id, trust_remote_code=_TRUST_REMOTE_CODE,
        revision=pinned_revision(target_id))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # The extraction assumes right padding.
    tok.padding_side = "right"
    sizes = sorted(args.sizes)
    loaders = {(s, n, b): load_task_data(b, split="test", num_samples=n,
                                         batch_size=args.batch_size,
                                         tokenizer=tok,
                                         max_length=args.max_length, seed=s)
               for s in seeds for n in sizes for b in benches}

    # ── overlap between the seeds' draws ──
    nmax = max(sizes)
    fp = {(s, b): loader_fingerprint(loaders[(s, nmax, b)])
          for s in seeds for b in benches}
    # check: select(range(n)) after a fixed shuffle makes the small draws
    # prefixes of the large one, so the sizes are nested.
    for s in seeds:
        for b in benches:
            big = loader_fingerprint(loaders[(s, nmax, b)])["hashes"]
            for n in sizes[:-1]:
                small = loader_fingerprint(loaders[(s, n, b)])["hashes"]
                if not small <= big:
                    print(f"  [warn] seed{s} {b}: size {n} is not a subset of "
                          f"size {nmax}; the sizes are not nested")
    draw_lines = ["# Overlap between the seeds' draws", "",
                  f"family {args.family}; seeds {seeds}; {nmax} sequences "
                  f"per draw", "",
                  "| benchmark | selected | pairwise example overlap |",
                  "|---|--:|--:|"]
    degenerate = []
    for b in benches:
        nsel = fp[(seeds[0], b)]["n_selected"]
        ovs = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                a, c = fp[(seeds[i], b)]["hashes"], fp[(seeds[j], b)]["hashes"]
                ovs.append(len(a & c) / max(len(a), 1))
        lo, hi = (min(ovs), max(ovs)) if ovs else (float("nan"),) * 2
        draw_lines.append(f"| {b} | {nsel} | {100*lo:.0f}-{100*hi:.0f}% |")
        if ovs and min(ovs) > 0.999:
            degenerate.append(b)
        print(f"  [draw] {b}: {nsel} examples, seed-pair overlap "
              f"{100*lo:.0f}-{100*hi:.0f}%")
    if degenerate:
        msg = ("!! the selected example set is identical across seeds for "
               + ", ".join(degenerate) + " (num_samples >= split size). The "
               "metrics are invariant to row order, so these benchmarks have "
               "sd = 0 across seeds; they are a single draw.")
        print(msg)
        draw_lines += ["", msg]
    draw_lines += ["", "On a small test split two draws of the same size share "
                   "part of their examples, so the seeds are overlapping draws."]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{args.family}_seed_draws.md").write_text("\n".join(draw_lines) + "\n")

    def feats_and_loss(model, loader):
        """One forward pass -> (answer-token features, mean answer loss).

        Features and loss come from the same answer tokens.  The loss is the
        mean over sequences of each sequence's mean answer-token cross-entropy.
        """
        Z, stats = ex.extract_features_and_loss_per_sample(
            model, loader, args.device, z_mode="concat")
        al = stats.get("answer_losses")
        loss = float(al.mean()) if al is not None else float(stats["losses"].mean())
        return Z.float().cpu(), loss

    # ── target: one load, all (seed, size, benchmark), cached to disk ──
    cache = OUT_DIR / f"{args.family}_ZT"
    cache.mkdir(exist_ok=True)
    loss_T = {}
    lossfile = cache / "target_loss.json"
    if lossfile.exists():
        try:
            loss_T = {tuple(k.split("|")): v
                      for k, v in json.loads(lossfile.read_text()).items()}
            loss_T = {(int(s), int(n), b): v for (s, n, b), v in loss_T.items()}
        except Exception:                                    # noqa: BLE001
            loss_T = {}

    def zt_path(seed, n, b):
        return cache / f"seed{seed}_n{n}_{b}.pt"

    missing = [(s, n, b) for s in seeds for n in sizes for b in benches
               if not zt_path(s, n, b).exists() or (s, n, b) not in loss_T]
    if missing:
        print(f"[target] one load, extracting {len(missing)} (seed, size, benchmark) cells")
        tgt = load_target(target_id, args.device)
        H_T = ex.extract_head(tgt).float().cpu()
        torch.save(H_T, cache / "H_T.pt")
        for s, n, b in missing:
            Z, l = feats_and_loss(tgt, loaders[(s, n, b)])
            tmp = zt_path(s, n, b).with_suffix(".pt.tmp")
            torch.save(Z, tmp)
            tmp.rename(zt_path(s, n, b))
            loss_T[(s, n, b)] = l
            print(f"  seed{s} n{n} {b}: Z{tuple(Z.shape)} answer-CE {l:.5f}")
            del Z
        lossfile.write_text(json.dumps({f"{s}|{n}|{b}": v
                                        for (s, n, b), v in loss_T.items()}))
        del tgt
        free_cuda()
    else:
        print("[target] all (seed, size, benchmark) features cached")
    H_T = torch.load(cache / "H_T.pt").float()

    # ── variants: one load each, inner loops over seed, size and benchmark ──
    failed_loads = []
    for spec in specs:
        label = spec["label"]
        pending = [s for s in seeds if missing_for(s, label)]
        if not pending:
            print(f"\n=== {label}: complete for all seeds, not loaded ===")
            continue
        print(f"\n=== {label}  (seeds {pending}) ===")
        try:
            proxy = load_proxy(spec, args.device)
        except Exception as exc:                             # noqa: BLE001
            # Print the traceback: the variant is otherwise absent from the
            # output without a visible cause.
            import traceback
            print(f"  [FAIL load] {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failed_loads.append((label, f"{type(exc).__name__}: {exc}"))
            continue
        try:
            H_P = ex.extract_head(proxy).float()
        except Exception as exc:                             # noqa: BLE001
            print(f"  [FAIL head] {exc}")
            del proxy
            free_cuda()
            continue

        for seed in pending:
            rows = []
            for n, b in missing_for(seed, label):
                try:
                    Z_P, loss_P = feats_and_loss(proxy, loaders[(seed, n, b)])
                except Exception as exc:                     # noqa: BLE001
                    print(f"  [FAIL extract seed{seed} n{n} {b}] {exc}")
                    continue
                Z_Tb = torch.load(zt_path(seed, n, b)).float()
                Xc, Yc = subsample_tokens(Z_Tb, Z_P, TOKEN_CAP, seed=seed)
                m = fresh_metrics(Xc.to(args.device), Yc.to(args.device),
                                  H_T.to(args.device), H_P.to(args.device),
                                  chunk=args.chunk)
                m.update({"family": args.family, "seed": seed, "n_samples": n,
                          "label": label, "dataset": b,
                          "loss_T": loss_T[(seed, n, b)], "loss_P": loss_P,
                          "|MdR|": abs(loss_T[(seed, n, b)] - loss_P)})
                rows.append(m)
                print(f"  seed{seed} n{n:<4} {b:<9} 1-Om_N={1-m['omega_W']:.6f} "
                      f"d_N={m['delta_W']:.3f} B_N={m['bound_W']:.2f} "
                      f"|dR|={m['|MdR|']:.5f}")
                del Z_P, Z_Tb, Xc, Yc
                free_cuda()
            # append this (seed, variant) block immediately
            if rows:
                p = csv_path(seed)
                new = not p.exists()
                with open(p, "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
                    if new:
                        w.writeheader()
                    w.writerows(rows)
                print(f"  [append] {p.name}  +{len(rows)} rows")
        del proxy, H_P
        free_cuda()

    if failed_loads:
        print(f"\n!! {len(failed_loads)} of {len(specs)} variants did not load and "
              f"are absent from the output:")
        for lab, why in failed_loads:
            print(f"   {lab}: {why}")
        print("   --check and size_study.py refuse an incomplete grid.")
    print("\nNext: python3 scripts/rounds/three_seed_round.py --check")


# ----------------------------------------------------------------------
# completeness check (no GPU)
# ----------------------------------------------------------------------
def incomplete_files(directory: Path):
    """[(file name, reason)] for every seed file that does not hold the full grid.

    The expected number of variants comes from the manifest, so a file that is
    uniformly short (a variant that never loaded) is reported as well.
    """
    from common_quant import variants_from_csv

    files = sorted(Path(directory).glob("*_seed*.csv"))
    if not files:
        raise SystemExit(f"no *_seed*.csv files in {directory}; run the GPU pass first")
    problems = []
    for p in files:
        fam = p.name.split("_seed")[0]
        expected = len(variants_from_csv(fam))
        rows = list(csv.DictReader(open(p)))
        cells = defaultdict(set)
        for r in rows:
            cells[(int(r["n_samples"]), r["dataset"].lower())].add(r["label"])
        if not cells:
            problems.append((p.name, "empty"))
            continue
        sizes = sorted({k[0] for k in cells})
        why = []
        missing = [(n, b) for n in sizes for b in PAPER5 if (n, b) not in cells]
        if missing:
            why.append(f"missing (size, benchmark) cells {missing}")
        short = {k: len(v) for k, v in cells.items() if len(v) != expected}
        if short:
            why.append(f"{len(short)} cells without all {expected} variants, e.g. "
                       f"{dict(list(short.items())[:3])}")
        if len(rows) != sum(len(v) for v in cells.values()):
            why.append("duplicate rows")
        if why:
            problems.append((p.name, "; ".join(why)))
    return problems


def check(args) -> None:
    problems = incomplete_files(OUT_DIR)
    if problems:
        print("incomplete seed files:")
        for name, why in problems:
            print(f"   {name}: {why}")
        raise SystemExit(1)
    print(f"every *_seed*.csv file in {OUT_DIR} holds the full grid")


def dry_run(args) -> None:
    """No GPU: print the plan."""
    benches = args.benchmarks or PAPER5
    print("[dry-run] plan")
    print(f"  family      : {args.family}")
    print(f"  seeds       : {args.seeds}")
    print(f"  benchmarks  : {benches}")
    print(f"  sizes       : {sorted(args.sizes)}")
    print(f"  max_length  : {args.max_length}")
    print(f"  token cap   : {TOKEN_CAP}")
    ncell = len(args.seeds) * len(args.sizes) * len(benches)
    print(f"  per family  : 1 target load, 1 load per variant in the manifest, "
          f"{ncell} forward passes per model")
    print(f"  output      : {OUT_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="llama", choices=["llama", "qwen"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[43, 44, 45])
    ap.add_argument("--benchmarks", nargs="*", default=None)
    ap.add_argument("--sizes", type=int, nargs="*", default=[8, 32, 128, 512],
                    help="reference sizes in sequences; the largest is the size of "
                         "the main quantization grid")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=8192,
                    help="token rows per accumulation chunk (memory setting)")
    ap.add_argument("--force", action="store_true",
                    help="move existing seed files aside and recompute everything")
    ap.add_argument("--check", action="store_true",
                    help="no GPU: verify that every seed file holds the full grid")
    ap.add_argument("--dry-run", dest="dry", action="store_true",
                    help="no GPU: print the plan")
    args = ap.parse_args()

    if args.dry:
        dry_run(args)
    elif args.check:
        check(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
