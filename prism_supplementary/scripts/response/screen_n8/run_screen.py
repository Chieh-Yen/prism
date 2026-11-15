#!/usr/bin/env python3
"""Eight-sequence screen: many reference sets of eight sequences per cell, each scored
against the risk gap measured on its 512-sequence pool.

Design (fixed before any set was scored; every set is written out):
  families    llama, qwen (12 / 11 quantized variants: the manifest of the three-seed round,
              scripts/rounds/data/variant_manifest.csv)
  benchmarks  arc, mmlu, squad, triviaqa, gsm8k
  pools       the 512-sequence pools of the three-seed round, seeds 43, 44, 45: the same
              sequences.  gap512 is recomputed here on each pool and is not copied from
              that round's output.
  sets        100 per (seed, benchmark): set 0 = the first eight sequences of the pool, i.e.
              the n = 8 set of the three-seed round (Appendix F.2); sets 1..99 = eight of
              the 512 drawn uniformly without replacement, numpy Generator seeded by
              [seed, benchmark index, 2026]; the same set is used for the target and for
              every variant
  -> 300 sets per (family, benchmark) cell.
Per set and variant: CKA (the score is 1-CKA), delta_N, gamma_N,
B_N = K_feat*delta_N + K_pred*gamma_N under the Procrustes alignment, and gap8; per pool and
variant: gap512.

Risk gap.  gap8 and gap512 are the risk gap on the eight sequences of the set (gap8) and
on the 512 of the pool (gap512), as in the three-seed round.

Arithmetic.  The omega / Procrustes / gamma arithmetic is that of fresh_metrics in the
three-seed round's script (scripts/rounds/three_seed_round.py), rewritten in the n x n (thin) form, which is
exact when n_tokens <= d: ||Z_T^T Z_P||_* via QR + an n x n SVD, and
||Sigma_P^{1/2} A||_F = ||Z_P A||_F / sqrt(n).  Sets with more token rows than feature
dimensions fall back to fresh_metrics itself.

Output (under --out):
  {family}_sets.json     the 100 index sets of every (seed, benchmark) pool
  {family}_K.json        K_feat, K_pred of the target head
  {family}_ZT/           cached target features and losses
  {family}_screen.csv    one row per (variant, seed, benchmark, set)

A variant whose model cannot be loaded (for example a GPTQ variant in an environment
without a GPTQ loader) is skipped; the skipped variants are listed at the end and the
script exits with status 1, so an incomplete CSV is not mistaken for a finished pass.
"""
import argparse, csv, json, math, os, sys, time
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("MKL_NUM_THREADS", "8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent          # supplementary root (holds prism/)
sys.path.insert(0, str(ROOT / "scripts" / "rounds"))
import numpy as np
import torch
torch.set_num_threads(8)
from common_quant import FAMILIES, free_cuda, load_proxy, load_target, variants_from_csv
from prism.data.loaders import load_task_data
from prism.models.extractors import LLMExtractor
from prism.core.bounds import UnifiedBound
from prism.experiments.quantization import _TRUST_REMOTE_CODE, pinned_revision

PAPER5 = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]
SEEDS = [43, 44, 45]; NPOOL = 512; NDRAW = 100; K8 = 8
ALIAS = {"meta-llama/Meta-Llama-3.1-8B": "meta-llama/Llama-3.1-8B"}   # renamed repo, same weights
COLS = ["family", "seed", "draw", "label", "dataset", "n_tokens", "cka", "omega_W", "rho_T", "rho_P",
        "delta_W", "gamma_W", "bound_W", "gap8", "gap512"]

# The extractor returns the answer-token rows of all sequences concatenated.  The wrapper
# records how many rows each sequence contributed, so the rows of a set can be selected.
ex = LLMExtractor()
_counts = []
_orig = ex._extract_z
def _rec(hidden, masks, zm, prompt_lens):
    out = _orig(hidden, masks, zm, prompt_lens)
    if zm == "concat":
        for i in range(hidden.shape[0]):
            length = masks[i].sum().long().item() if masks is not None else hidden.shape[1]
            start = max(int(prompt_lens[i].item()) - 1, 0) if prompt_lens is not None else 0
            _counts.append(max(length - 1 - start, 0))
    return out
ex._extract_z = _rec

def feats(model, loader, dev):
    """One forward pass: answer-token features, answer losses, and the rows of each sequence."""
    _counts.clear()
    Z, st = ex.extract_features_and_loss_per_sample(model, loader, dev, z_mode="concat")
    al = st["answer_losses"]; al = (al if al is not None else st["losses"]).double().numpy()
    cnt = np.array(_counts, dtype=np.int64)
    assert cnt.sum() == Z.shape[0] and len(cnt) == len(al), (cnt.sum(), Z.shape, len(cnt), len(al))
    return Z.float().cpu(), al, cnt

def fast_metrics(X, Y, H_T, H_P, K_feat, K_pred):
    """X = Z_T rows, Y = Z_P rows, n <= d.  Same quantities as fresh_metrics."""
    n, d = X.shape
    nx, ny = X.norm(), Y.norm()
    Xc, Yc = X - X.mean(0, keepdim=True), Y - Y.mean(0, keepdim=True)
    Kx, Ky = Xc @ Xc.T, Yc @ Yc.T
    cka = ((Kx * Ky).sum() / (Kx.norm() * Ky.norm()).clamp(min=1e-300)).item()
    Qy, Ry = torch.linalg.qr(Y.T)            # Y^T = Qy Ry
    Qx, Rx = torch.linalg.qr(X.T)            # X^T = Qx Rx
    M = (Ry @ Rx.T) / (nx * ny).clamp(min=1e-300)          # Y^T X / (nx ny) = Qy M Qx^T
    Um, S, Vmh = torch.linalg.svd(M)
    omega_raw = S.sum().item(); omega = min(omega_raw, 1.0)
    rho_T, rho_P = (nx / math.sqrt(n)).item(), (ny / math.sqrt(n)).item()
    delta = math.sqrt(max((rho_T - rho_P) ** 2 + 2 * rho_T * rho_P * (1 - omega), 0.0))
    # Y W = Ry^T Um Vmh Qx^T  (W = polar factor of Y^T X); gamma = ||Y (W H_T - H_P)||_F / sqrt(n)
    A = (Ry.T @ Um @ Vmh).float()            # n x n
    B = (Qx.T).float() @ H_T                 # n x V
    D = A @ B - Y.float() @ H_P
    gamma = (D.norm() / math.sqrt(n)).item()
    return dict(n_tokens=n, cka=cka, omega_W=omega, rho_T=rho_T, rho_P=rho_P, delta_W=delta,
                gamma_W=gamma, bound_W=K_feat * delta + K_pred * gamma)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=["llama", "qwen"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", nargs="*", default=None, help="variant labels to run (default: all in the manifest)")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--benchmarks", nargs="*", default=PAPER5)
    ap.add_argument("--out", default=str(ROOT / "out" / "response" / "screen_n8"))
    a = ap.parse_args()
    out = Path(a.out)
    from transformers import AutoTokenizer
    tid = ALIAS.get(FAMILIES[a.family], FAMILIES[a.family])
    specs = variants_from_csv(a.family)
    for s in specs:
        s["repo"] = ALIAS.get(s.get("repo"), s.get("repo"))
    if a.only:
        known = [s["label"] for s in specs]
        unknown = [l for l in a.only if l not in known]
        if unknown:
            ap.error(f"--only: no variant of family '{a.family}' has the label(s) "
                     f"{unknown}; the labels of this family are {known}")
        specs = [s for s in specs if s["label"] in a.only]
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(tid, trust_remote_code=_TRUST_REMOTE_CODE, revision=pinned_revision(tid))
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    # The extraction assumes right padding.
    tok.padding_side = "right"
    loaders = {(s, b): load_task_data(b, split="test", num_samples=NPOOL, batch_size=4, tokenizer=tok,
                                      max_length=512, seed=s) for s in a.seeds for b in a.benchmarks}
    # sets: fixed before any variant is scored, shared by target and every variant
    draws = {}
    for s in a.seeds:
        for bi, b in enumerate(PAPER5):
            if b not in a.benchmarks: continue
            npool = len(loaders[(s, b)].dataset)
            rng = np.random.default_rng([s, bi, 2026])
            draws[(s, b)] = [np.arange(K8)] + [np.sort(rng.choice(npool, K8, replace=False)) for _ in range(NDRAW - 1)]
    (out / f"{a.family}_sets.json").write_text(json.dumps({f"{s}|{b}": [d.tolist() for d in v] for (s, b), v in draws.items()}))
    # target: one load, cached per (seed, benchmark)
    zdir = out / f"{a.family}_ZT"; zdir.mkdir(exist_ok=True)
    need = [(s, b) for s in a.seeds for b in a.benchmarks if not (zdir / f"seed{s}_{b}.pt").exists()]
    if need or not (zdir / "H_T.pt").exists():
        tgt = load_target(tid, a.device)
        torch.save(ex.extract_head(tgt).float().cpu(), zdir / "H_T.pt")
        for s, b in need:
            Z, al, cnt = feats(tgt, loaders[(s, b)], a.device)
            torch.save({"Z": Z, "al": al, "cnt": cnt}, zdir / f"seed{s}_{b}.pt.tmp")
            os.replace(zdir / f"seed{s}_{b}.pt.tmp", zdir / f"seed{s}_{b}.pt")
            print(f"[target] seed{s} {b}: Z{tuple(Z.shape)} loss512 {al.mean():.5f}", flush=True)
        del tgt; free_cuda()
    H_T = torch.load(zdir / "H_T.pt").float().to(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    kf = out / f"{a.family}_K.json"
    if kf.exists(): K = json.loads(kf.read_text())
    else:
        K = UnifiedBound.theoretical_K(H_T); K = {"K_feat": float(K["K_feat"]), "K_pred": float(K["K_pred"])}
        kf.write_text(json.dumps(K))
    print(f"[K] {K}", flush=True)
    T = {(s, b): torch.load(zdir / f"seed{s}_{b}.pt", weights_only=False) for s in a.seeds for b in a.benchmarks}
    csvp = out / f"{a.family}_screen.csv"
    done = set()
    failed = []          # variants whose model could not be loaded
    if csvp.exists():
        for r in csv.DictReader(open(csvp)): done.add((r["label"], int(r["seed"]), r["dataset"]))
    for spec in specs:
        lab = spec["label"]
        pending = [(s, b) for s in a.seeds for b in a.benchmarks if (lab, s, b) not in done]
        if not pending:
            print(f"=== {lab}: complete", flush=True); continue
        t0 = time.time()
        print(f"=== {lab} ({len(pending)} cells)", flush=True)
        try:
            proxy = load_proxy(spec, a.device)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[FAIL load] {lab}: {e}", flush=True)
            failed.append(lab); continue
        H_P = ex.extract_head(proxy).float().to(a.device)
        for s, b in pending:
            Z_P, alP, cntP = feats(proxy, loaders[(s, b)], a.device)
            t = T[(s, b)]
            assert np.array_equal(cntP, t["cnt"]), f"token layout differs {lab} {s} {b}"
            off = np.concatenate([[0], np.cumsum(t["cnt"])])
            gap512 = abs(t["al"].mean() - alP.mean())
            rows = []
            for di, idx in enumerate(draws[(s, b)]):
                rr = np.concatenate([np.arange(off[i], off[i + 1]) for i in idx])
                X = t["Z"][rr].to(a.device, torch.float64); Y = Z_P[rr].to(a.device, torch.float64)
                if X.shape[0] <= X.shape[1]:
                    m = fast_metrics(X, Y, H_T, H_P, K["K_feat"], K["K_pred"])
                else:
                    from three_seed_round import fresh_metrics
                    f = fresh_metrics(X.float(), Y.float(), H_T, H_P)
                    m = {k: f[k] for k in ("n_tokens", "cka", "omega_W", "rho_T", "rho_P", "delta_W", "gamma_W", "bound_W")}
                m.update(family=a.family, seed=s, draw=di, label=lab, dataset=b,
                         gap8=abs(t["al"][idx].mean() - alP[idx].mean()), gap512=gap512)
                rows.append(m)
            new = not csvp.exists()
            with open(csvp, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
                if new: w.writeheader()
                w.writerows(rows)
            d0 = rows[0]
            print(f"  seed{s} {b:<9} gap512 {gap512:.5f} | set0 1-CKA {1-d0['cka']:.6f} d {d0['delta_W']:.3f} "
                  f"B {d0['bound_W']:.3f} gap8 {d0['gap8']:.5f}", flush=True)
            del Z_P
        del proxy, H_P; free_cuda()
        print(f"  {lab} done in {time.time()-t0:.0f}s", flush=True)
    if failed:
        print(f"screen INCOMPLETE for {a.family}: {len(failed)} variant(s) could not be loaded and have "
              f"no rows in {csvp}:", flush=True)
        for lab in failed:
            print(f"  {lab}", flush=True)
        print("re-run them with --only in an environment that can load them, e.g.\n"
              f"  python3 {Path(__file__).name} --family {a.family} --out {out} --only "
              + " ".join(f'"{lab}"' for lab in failed), flush=True)
        sys.exit(1)
    print("screen done", a.family, flush=True)

if __name__ == "__main__":
    main()
