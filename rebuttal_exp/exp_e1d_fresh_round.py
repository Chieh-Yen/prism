#!/usr/bin/env python3
"""E1-D: a fully self-contained FRESH ROUND of the similarity comparison.

Produces exactly this table, per family and pooled, as mean +- sd over three
seeds that the paper never used:

    method          Llama    Qwen3    mean r_s
    1-CKA
    1-SVCCA
    1-Omega_N
    feature arm_N   (= delta_N, the Procrustes size-and-shape distance)
    PRISM B_N       (= K_feat * delta_N + K_pred * gamma_N)

WHY A FRESH ROUND, AND WHY IT IS THE CLEAN FIX
----------------------------------------------
E1 mixes two pipelines: cka/svcca/procr_dist are recomputed on fresh features
while bound_I/bound_W are JOINED from the paper CSV, and |dR| is joined too.
That forced the draft into "within-block comparisons only" and left two
different delta_N numbers (0.901 re-extracted vs 0.873 in the paper's Table 3)
side by side.  Here EVERY column, including B_N and |dR|, is computed on the
same features from the same forward pass, so the table is internally paired and
needs no provenance caveat.

Two independent problems in the paper's round are also fixed here:

  (a) THE CLAMP.  prism/core/metrics.py:244 does
          omega = max(min(omega, 1.0), -1.0)
      At n = 52184 tokens the float32 accumulation of (W * Z_P^T Z_T).sum()
      (~1.7e7 terms, magnitude ~1e9) can land slightly above 1, and the clamp
      then stores exactly 1.0.  That is why the paper CSV holds a literal "1.0"
      for 11/12 GSM8K Omega cells while mmlu (511 tokens) holds full precision
      (0.9998420930774202).  An exact 1.0 cannot be a true value: Omega is a
      normalised inner product, so Omega = 1 iff Z_P = c * Z_T, which a Q2_K
      backbone is not.  This script accumulates in float64 and records whether
      the clamp WOULD have fired (`clamp_would_fire`), instead of clamping.

  (b) THE JOINED |dR|.  Changing the extraction seed changes which examples are
      drawn, so reusing the paper's |dR| would pair a score from sample A with a
      risk from sample B.  Here |dR| is recomputed from the answer-span losses of
      the same forward pass (loss_mode "answer", matching the paper's protocol:
      features and CE both come from the gold span).

SEEDS
    Default 43,44,45.  The paper's round is seed 42, so these are disjoint from
    it: the resulting numbers are a fresh replication, NOT an attempt to
    reproduce Table 3, and they should be reported as such.

WHAT COULD COME OUT DIFFERENTLY FROM THE PAPER (read before running)
    Fixing (a) RAISES the Omega/delta ranking on long-token benchmarks, because
    the corrupted cells were the ones dragging Omega down.  The paper's Table 3
    W_N ladder is Omega_N 0.806 -> delta_N 0.873 (+0.067) -> B_N 0.912 (+0.039).
    On corrected features the scale-arm gain may be much smaller (a fresh
    trace-gauge shape core already ranks +0.895 versus a feature arm of +0.901,
    i.e. +0.006).  If that carries over to the nuclear gauge, the "the machinery
    buys +0.106" argument currently made to pCi8-W3 no longer holds and must be
    rewritten around the four-outputs framing instead.  Decide that BEFORE
    running, and report whatever comes out.

USAGE
    # zero GPU: check the plan, the API wiring and the clamp diagnosis
    python3 rebuttal_exp/exp_e1d_fresh_round.py --dry-run

    # GPU: one family, three seeds  (~3 x 25 min; target extracted once/seed)
    python3 rebuttal_exp/exp_e1d_fresh_round.py --family llama --seeds 43 44 45

    # zero GPU: build the table from whatever seeds are on disk
    python3 rebuttal_exp/exp_e1d_fresh_round.py --report

SANITY CHECK BUILT INTO --report
    K_feat depends only on H_T, so a recomputed value must match the paper's or
    B_N is not the paper's bound (the two arms would be weighted differently).
    Paper values: llama K_f 2.6137, qwen K_f 3.4583, K_p 1.4142 = sqrt(2) for
    both, exactly as Proposition 1 states.

OUTPUT
    out/E1D/{family}_seed{S}.csv   per (variant, benchmark): all 5 metrics + |dR|
    out/E1D/table.md               the 5-row table, mean +- sd over seeds
    out/E1D/diagnostics.md         clamp-would-fire counts, token counts, drift
                                   of |dR| across seeds
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT_DIR = HERE / "out" / "E1D"
PAPER5 = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]

# GSM8K needs the full token set; 52184 (llama) / 63056 (qwen) were observed, so
# the cap must sit above both or the concat features get truncated mid-benchmark.
TOKEN_CAP = 131072

ROWS = ["1-CKA", "1-SVCCA", "1-Omega_N", "feature arm_N", "PRISM B_N"]
# (csv column, higher-means-more-drift?) -- every row is oriented so that
# "larger = more degradation", which is what |dR| is.
ROW_COL = {
    "1-CKA": ("cka", True),            # similarity -> 1 - x
    "1-SVCCA": ("svcca", True),
    "1-Omega_N": ("omega_W", True),
    "feature arm_N": ("delta_W", False),
    "PRISM B_N": ("bound_W", False),
}


# ----------------------------------------------------------------------
# metrics, all in float64, no clamp (the clamp is only DIAGNOSED)
# ----------------------------------------------------------------------
def fresh_metrics(Z_T, Z_P, H_T, H_P, chunk: int = 8192):
    """All five quantities on one feature pair.

    Precision and memory, both load-bearing:

    * The omega path accumulates in FLOAT64, CHUNKED over tokens.  Chunking is
      not a shortcut: the alternative (a full float64 copy of a 52184x4096
      feature matrix) costs 1.7 GB per side, and at 8 GB of transient buffers
      the metric would have to run with the model unloaded.  Chunked
      accumulation is the same arithmetic at ~700 MB peak.
    * Both sides are normalised by their Frobenius norm BEFORE the cross
      product, so omega is read off directly as a nuclear norm of an O(1)
      matrix instead of as a ratio of two ~1e9 quantities.  That is what makes
      1 - omega ~ 1e-5 resolvable; the paper round computed the ratio in float32
      and the clamp at prism/core/metrics.py:244 then stored a literal 1.0.
    * TF32 is pinned off for these matmuls.  TF32 keeps 10 mantissa bits, i.e.
      ~1e-3 relative, which would destroy the cross product outright.  The
      default is already off in torch 2.5, but it is environment-dependent
      (torch.set_float32_matmul_precision can flip it), so we set it here.

    `clamp_would_fire` records whether the paper's clamp WOULD have triggered.
    """
    import torch

    from exp_e1_similarity_baselines import linear_cka, svcca
    from prism.core.bounds import UnifiedBound
    from prism.core.metrics import PRISMMetrics

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        # --- similarity baselines: unchanged E1 code path, so these numbers are
        #     produced by exactly the same function as the round they replace ---
        cka = linear_cka(Z_T, Z_P)
        sv = svcca(Z_T, Z_P)

        # --- chunked float64 accumulation ---
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

        # normalise: omega is then the nuclear norm of an O(1) matrix
        cross_hat = cross / (nx * ny).clamp(min=1e-300)
        U, S, Vt = torch.linalg.svd(cross_hat, full_matrices=False)
        W_N = U @ Vt
        omega_raw = S.sum().item()                 # = ||Z_T^T Z_P||_* / (nx*ny)
        omega_i = (dot / (nx * ny).clamp(min=1e-300)).item()
        clamp_fire = int(omega_raw > 1.0)
        omega_N = min(omega_raw, 1.0)              # only for the delta formula

        rho_T = (nx / math.sqrt(n)).item()
        rho_P = (ny / math.sqrt(n)).item()
        delta_N = math.sqrt(max((rho_T - rho_P) ** 2
                                + 2 * rho_T * rho_P * (1 - omega_N), 0.0))

        # --- head term at W_N, and the paper's own Lipschitz constants ---
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
        "omega_W": omega_N, "omega_W_raw": omega_raw,
        "clamp_would_fire": clamp_fire,
        "rho_T": rho_T, "rho_P": rho_P,
        "delta_W": delta_N, "gamma_W": float(gamma_N),
        "K_feat": K["K_feat"], "K_pred": K["K_pred"],
        "bound_W": bound_N,
    }


# ----------------------------------------------------------------------
# GPU pass
# ----------------------------------------------------------------------
def loader_fingerprint(loader):
    """Hash of every selected example, so we can MEASURE what the seed changed.

    The seed's only effect in this script is which examples are drawn:
    prism/data/loaders.py does `shuffle(seed)` then `select(range(num_samples))`.
    (The other seed use, subsample_tokens, is a no-op here because TOKEN_CAP sits
    above every observed token count.)

    Two failure modes this catches, both of which would make "three seeds"
    meaningless without saying so:
      * num_samples >= len(split): `select` never fires, so all seeds see the SAME
        examples in a different order -- and every metric here (CKA, omega,
        Procrustes, Sigma_P) is row-permutation invariant, so the three seeds
        would return bit-identical numbers and a spurious sd of 0.000.
      * a small split (GSM8K test and ARC test are ~1.2-1.3k rows): two draws of
        512 then share ~40% of their examples, so the seeds are CORRELATED draws
        and the sd understates true sampling variability.
    """
    import hashlib

    ds = loader.dataset
    ids = ds.encodings["input_ids"]
    h = set()
    for row in ids:
        h.add(hashlib.blake2b(row.numpy().tobytes(), digest_size=8).hexdigest())
    return {"n_selected": int(ids.shape[0]), "hashes": h}


def run(args) -> None:
    """One target load + one load per proxy, for ALL seeds.

    The naive order (seed outer, proxy inner) reloads every proxy once per seed.
    Loading a GGUF/GPTQ 8B proxy costs 1-2 min, i.e. 36 loads/family at 3 seeds,
    which dominates everything else (a 5-benchmark forward pass is ~5-8 s and the
    float64 metric ~2 s).  Hoisting the proxy loop outside the seed loop makes it
    13 loads/family: same arithmetic, same rows, ~3-4x less wall clock.

    Target features are extracted once per (seed, benchmark) and cached to disk
    atomically, so they are shared across proxies and survive a kill.  Rows are
    appended per (seed, proxy) so an interrupted run resumes instead of redoing.
    """
    import torch
    from common_quant import (FAMILIES, free_cuda, load_proxy, load_target,
                              subsample_tokens, variants_from_csv)
    from prism.data.loaders import load_task_data
    from prism.models.extractors import LLMExtractor
    from transformers import AutoTokenizer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_id = FAMILIES[args.family]
    specs = variants_from_csv(args.family)
    benches = args.benchmarks or PAPER5
    seeds = list(args.seeds)
    ex = LLMExtractor()

    COLS = ["family", "seed", "label", "dataset", "n_tokens",
            "cka", "svcca", "omega_I", "omega_W", "omega_W_raw",
            "clamp_would_fire", "rho_T", "rho_P", "delta_W", "gamma_W",
            "K_feat", "K_pred", "bound_W", "loss_T", "loss_P", "|MdR|"]

    # ── resume: which (seed, label) pairs are already complete? ──
    def csv_path(seed):
        return OUT_DIR / f"{args.family}_seed{seed}.csv"

    done = set()
    if not args.force:
        for seed in seeds:
            p = csv_path(seed)
            if not p.exists():
                continue
            per = defaultdict(set)
            for r in csv.DictReader(open(p)):
                per[r["label"]].add(r["dataset"].lower())
            for label, dss in per.items():
                if set(benches) <= dss:            # every benchmark present
                    done.add((seed, label))
        if done:
            print(f"[resume] {len(done)} (seed, proxy) pairs already complete")

    tok = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    loaders = {s: {b: load_task_data(b, split="test",
                                     num_samples=args.num_samples,
                                     batch_size=args.batch_size, tokenizer=tok,
                                     max_length=args.max_length, seed=s)
                   for b in benches} for s in seeds}

    # ── what did the seed actually change?  Measure, do not assume. ──
    fp = {(s, b): loader_fingerprint(loaders[s][b]) for s in seeds for b in benches}
    draw_lines = ["# E1-D seed draw overlap (what the seed changed)", "",
                  f"family {args.family}; seeds {seeds}; num_samples "
                  f"{args.num_samples}", "",
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
        msg = ("!! seeds are DEGENERATE for " + ", ".join(degenerate) +
               ": the selected example set is identical across seeds (num_samples "
               ">= split size), and every metric here is row-permutation "
               "invariant, so those benchmarks will show sd = 0 for a reason that "
               "has nothing to do with stability. Lower --num_samples for them or "
               "report them as single-draw.")
        print(msg)
        draw_lines += ["", msg]
    draw_lines += ["", "Overlap is expected to be high on small splits (a 512-draw "
                   "from a ~1.2k-row split shares ~40% with another draw), so the "
                   "seeds are CORRELATED draws and the sd is a lower bound on "
                   "sampling variability. Report it that way."]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{args.family}_seed_draws.md").write_text("\n".join(draw_lines) + "\n")

    def feats_and_loss(model, loader):
        """One forward pass -> (answer-region concat features, mean answer CE).

        loss_mode "answer" matches the paper: Z and the CE come from the same
        gold-span tokens, so score and risk are paired by construction.
        """
        Z, stats = ex.extract_features_and_loss_per_sample(
            model, loader, args.device, z_mode="concat")
        al = stats.get("answer_losses")
        loss = float(al.mean()) if al is not None else float(stats["losses"].mean())
        return Z.float().cpu(), loss

    # ── target: ONE load, all (seed, benchmark), disk-cached atomically ──
    cache = OUT_DIR / f"{args.family}_ZT"
    cache.mkdir(exist_ok=True)
    loss_T = {}
    lossfile = cache / "target_loss.json"
    if lossfile.exists():
        try:
            loss_T = {tuple(k.split("|")): v
                      for k, v in json.loads(lossfile.read_text()).items()}
            loss_T = {(int(s), b): v for (s, b), v in loss_T.items()}
        except Exception:                                    # noqa: BLE001
            loss_T = {}

    def zt_path(seed, b):
        return cache / f"seed{seed}_{b}.pt"

    missing = [(s, b) for s in seeds for b in benches
               if not zt_path(s, b).exists() or (s, b) not in loss_T]
    if missing:
        print(f"[target] one load, extracting {len(missing)} (seed, benchmark) cells")
        tgt = load_target(target_id, args.device)
        H_T = ex.extract_head(tgt).float().cpu()
        torch.save(H_T, cache / "H_T.pt")
        for s, b in missing:
            Z, l = feats_and_loss(tgt, loaders[s][b])
            tmp = zt_path(s, b).with_suffix(".pt.tmp")
            torch.save(Z, tmp)
            tmp.rename(zt_path(s, b))
            loss_T[(s, b)] = l
            print(f"  seed{s} {b}: Z{tuple(Z.shape)} answer-CE {l:.5f}")
            del Z
        lossfile.write_text(json.dumps({f"{s}|{b}": v
                                        for (s, b), v in loss_T.items()}))
        del tgt
        free_cuda()
    else:
        print("[target] all (seed, benchmark) features cached")
    H_T = torch.load(cache / "H_T.pt").float()

    # ── proxies: ONE load each, inner loops over seed x benchmark ──
    for spec in specs:
        label = spec["label"]
        todo = [s for s in seeds if (s, label) not in done]
        if not todo:
            print(f"\n=== {label}: complete for all seeds, not loaded ===")
            continue
        print(f"\n=== {label}  (seeds {todo}) ===")
        try:
            proxy = load_proxy(spec, args.device)
        except Exception as exc:                             # noqa: BLE001
            print(f"  [FAIL load] {exc}")
            continue
        try:
            H_P = ex.extract_head(proxy).float()
        except Exception as exc:                             # noqa: BLE001
            print(f"  [FAIL head] {exc}")
            del proxy
            free_cuda()
            continue

        for seed in todo:
            rows = []
            for b in benches:
                try:
                    Z_P, loss_P = feats_and_loss(proxy, loaders[seed][b])
                except Exception as exc:                     # noqa: BLE001
                    print(f"  [FAIL extract seed{seed} {b}] {exc}")
                    continue
                Z_Tb = torch.load(zt_path(seed, b)).float()
                Xc, Yc = subsample_tokens(Z_Tb, Z_P, TOKEN_CAP, seed=seed)
                m = fresh_metrics(Xc.to(args.device), Yc.to(args.device),
                                  H_T.to(args.device), H_P.to(args.device),
                                  chunk=args.chunk)
                m.update({"family": args.family, "seed": seed, "label": label,
                          "dataset": b, "loss_T": loss_T[(seed, b)],
                          "loss_P": loss_P,
                          "|MdR|": abs(loss_T[(seed, b)] - loss_P)})
                rows.append(m)
                print(f"  seed{seed} {b}: 1-CKA={1-m['cka']:.4f} "
                      f"1-Om_N={1-m['omega_W']:.6f} d_N={m['delta_W']:.3f} "
                      f"B_N={m['bound_W']:.2f} |dR|={m['|MdR|']:.5f}"
                      + ("  [clamp would fire]" if m["clamp_would_fire"] else ""))
                del Z_P, Z_Tb, Xc, Yc
                free_cuda()
            # append this (seed, proxy) block immediately: a kill costs one proxy
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

    print("\nNext: python3 rebuttal_exp/exp_e1d_fresh_round.py --report")


# ----------------------------------------------------------------------
# report (no GPU)
# ----------------------------------------------------------------------
def spearman(x, y) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    a, b = rank(x), rank(y)
    n = len(x)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** 0.5
    return num / den if den else float("nan")


def family_mean(path: Path, row: str) -> float:
    """mean r_s over the 5 benchmarks for one metric row in one seed file."""
    col, inv = ROW_COL[row]
    rows = list(csv.DictReader(open(path)))
    per = []
    for ds in PAPER5:
        rs = [r for r in rows if r["dataset"].lower() == ds]
        if not rs:
            continue
        dr = [abs(float(r["|MdR|"])) for r in rs]
        v = [(1 - float(r[col])) if inv else float(r[col]) for r in rs]
        per.append(spearman(v, dr))
    return statistics.mean(per) if per else float("nan")


def fmt(vals) -> str:
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:+.3f}"
    return f"{statistics.mean(vals):+.3f} ± {statistics.stdev(vals):.3f}"


def report(args) -> None:
    files = defaultdict(list)
    for p in sorted(OUT_DIR.glob("*_seed*.csv")):
        fam = p.name.split("_seed")[0]
        files[fam].append(p)
    if not files:
        raise SystemExit(f"no seed CSVs in {OUT_DIR} -- run the GPU pass first")

    seeds = sorted({p.name.split("_seed")[1].split(".")[0]
                    for ps in files.values() for p in ps})

    # ── completeness gate: a partially written seed file would silently give a
    #    Spearman over fewer variants, which is not comparable across rows ──
    incomplete = []
    for fam, ps in files.items():
        for p in ps:
            rows = list(csv.DictReader(open(p)))
            per = defaultdict(set)
            for r in rows:
                per[r["dataset"].lower()].add(r["label"])
            counts = {ds: len(v) for ds, v in per.items()}
            if not counts:
                incomplete.append((p.name, "empty"))
                continue
            nmax = max(counts.values())
            short = {ds: c for ds, c in counts.items() if c < nmax}
            miss = [ds for ds in PAPER5 if ds not in counts]
            if short or miss:
                incomplete.append((p.name,
                                   f"variants/benchmark {counts}"
                                   + (f", missing benchmarks {miss}" if miss else "")))
    if incomplete:
        print("!! INCOMPLETE seed files -- the table below is NOT final:")
        for name, why in incomplete:
            print(f"   {name}: {why}")
        if not args.allow_incomplete:
            raise SystemExit("refusing to emit a table from incomplete data; "
                             "finish the run or pass --allow-incomplete")
    lines = ["# E1-D fresh round (seeds " + ", ".join(seeds) + ")", "",
             "Every column computed on the SAME features from one forward pass:",
             "similarity baselines, the shape core, the feature arm and the full",
             "certified bound, with |dR| recomputed from the answer-span CE of that",
             "same pass. Accumulation in float64, no omega clamp. These seeds are",
             "disjoint from the paper's round, so the numbers are a fresh",
             "replication and not a reproduction of Table 3.", "",
             "| method | Llama | Qwen3 | mean r_s |", "|:--|--:|--:|--:|"]
    pooled = {}
    for row in ROWS:
        cells = []
        for fam in ("llama", "qwen"):
            vals = [family_mean(p, row) for p in files.get(fam, [])]
            cells.append(fmt(vals))
            pooled.setdefault(row, {})[fam] = vals
        lv, qv = pooled[row]["llama"], pooled[row]["qwen"]
        both = [statistics.mean([a, b]) for a, b in zip(lv, qv)] if lv and qv else (lv or qv)
        lines.append(f"| {row} | {cells[0]} | {cells[1]} | {fmt(both)} |")
    lines.append("")

    # paired differences, the load-bearing quantity
    lines += ["## Paired differences (per seed, then mean +- sd)", ""]
    for base in ("1-CKA", "1-SVCCA"):
        for probe in ("feature arm_N", "PRISM B_N"):
            per_seed = []
            for i in range(max(len(v) for v in
                               [pooled[probe]["llama"], pooled[probe]["qwen"]] or [[]])):
                acc = []
                for fam in ("llama", "qwen"):
                    a, b = pooled[probe][fam], pooled[base][fam]
                    if i < len(a) and i < len(b):
                        acc.append(a[i] - b[i])
                if acc:
                    per_seed.append(statistics.mean(acc))
            lines.append(f"- {probe} minus {base}: {fmt(per_seed)}")
    lines.append("")

    # diagnostics
    dl = ["# E1-D diagnostics", ""]
    # paper K_f / K_p per family, for the B_N sanity check: K_feat depends only
    # on H_T, so a recomputed value must match the paper's or B_N is not the
    # paper's bound (the two arms would be weighted differently).
    paperK = {}
    try:
        from common_quant import CSV_PATH, FAMILIES
        for fam_key, target in FAMILIES.items():
            vals = [(float(r["K_f"]), float(r["K_p"]))
                    for r in csv.DictReader(open(CSV_PATH))
                    if r["target_model"] == target and r["K_f"] and r["K_p"]]
            if vals:
                paperK[fam_key] = (statistics.mean(v[0] for v in vals),
                                   statistics.mean(v[1] for v in vals))
    except Exception as exc:                             # noqa: BLE001
        dl += [f"(paper K lookup failed: {exc})", ""]
    for fam, ps in files.items():
        for p in ps:
            rows = list(csv.DictReader(open(p)))
            fire = sum(int(r["clamp_would_fire"]) for r in rows)
            ntok = {r["dataset"]: r["n_tokens"] for r in rows}
            dl += [f"## {p.name}",
                   f"- rows: {len(rows)}; omega clamp WOULD have fired in "
                   f"{fire}/{len(rows)} cells (float32 + clamp is the paper-round "
                   f"artefact; float64 here avoids it)",
                   f"- tokens: " + ", ".join(f"{k} {v}" for k, v in sorted(ntok.items()))]
            if rows and fam in paperK:
                kf = statistics.mean(float(r["K_feat"]) for r in rows)
                kp = statistics.mean(float(r["K_pred"]) for r in rows)
                pkf, pkp = paperK[fam]
                ok = abs(kf - pkf) / max(pkf, 1e-9) < 0.02
                note = ("MATCH: B_N here is the paper's bound evaluated on fresh "
                        "features" if ok else
                        "MISMATCH: B_N weights the two arms differently from the "
                        "paper, so do NOT compare B_N across rounds")
                dl.append(f"- K sanity: recomputed K_feat {kf:.3f} vs paper "
                          f"{pkf:.3f}, K_pred {kp:.3f} vs paper {pkp:.3f} "
                          f"-> {note}")
            # |dR| drift across seeds is what tells us whether the fresh
            # targets are stable; a large spread means the ranking target itself
            # is noisy for that benchmark.
            per_ds = defaultdict(list)
            for r in rows:
                per_ds[r["dataset"]].append(abs(float(r["|MdR|"])))
            dl.append("- mean |dR| by benchmark: " + ", ".join(
                f"{k} {statistics.mean(v):.4f}" for k, v in sorted(per_ds.items())))
            dl.append("")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "table.md").write_text("\n".join(lines) + "\n")
    (OUT_DIR / "diagnostics.md").write_text("\n".join(dl) + "\n")
    print("\n".join(lines))
    print("\n".join(dl))
    print(f"[write] {OUT_DIR/'table.md'}  {OUT_DIR/'diagnostics.md'}")


def dry_run(args) -> None:
    """No GPU: show the plan and prove the clamp diagnosis from source."""
    import re
    src = (HERE.parent / "prism" / "core" / "metrics.py").read_text()
    m = re.search(r"omega = max\(min\(omega, 1\.0\), -1\.0\)", src)
    print("[dry-run] plan")
    print(f"  family      : {args.family}")
    print(f"  seeds       : {args.seeds}   (paper round = 42, so disjoint)")
    print(f"  benchmarks  : {args.benchmarks or PAPER5}")
    print(f"  token cap   : {TOKEN_CAP}  (> gsm8k 52184 llama / 63056 qwen, so no truncation)")
    print(f"  num_samples : {args.num_samples}   max_length: {args.max_length}")
    print(f"  rows        : {ROWS}")
    print("\n[dry-run] clamp present in the paper's metric path: "
          f"{'YES  prism/core/metrics.py' if m else 'NOT FOUND'}")
    print("  -> that clamp plus float32 accumulation is why the paper CSV holds a")
    print("     literal 1.0 for 11/12 gsm8k Omega cells. This script accumulates in")
    print("     float64 and only RECORDS whether the clamp would have fired.")
    print("\n[dry-run] every column here is recomputed, including |dR| and B_N, so")
    print("  the table is internally paired and needs no provenance caveat.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="llama", choices=["llama", "qwen"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[43, 44, 45],
                    help="fresh seeds, disjoint from the paper's 42")
    ap.add_argument("--benchmarks", nargs="*", default=None)
    ap.add_argument("--num_samples", type=int, default=512, help="paper: 512")
    ap.add_argument("--max_length", type=int, default=512, help="paper: 512")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=8192,
                    help="tokens per float64 accumulation chunk (memory knob; "
                         "does not change the result)")
    ap.add_argument("--force", action="store_true",
                    help="ignore the resume set and recompute everything")
    ap.add_argument("--report", action="store_true", help="no GPU: build the table")
    ap.add_argument("--allow-incomplete", dest="allow_incomplete",
                    action="store_true",
                    help="emit the table even if some (seed, proxy) cells are "
                         "missing (marked NOT final)")
    ap.add_argument("--dry-run", dest="dry", action="store_true",
                    help="no GPU: show the plan + the clamp diagnosis")
    args = ap.parse_args()

    if args.dry:
        dry_run(args)
    elif args.report:
        report(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
