#!/usr/bin/env python3
"""
E3 — Reference-set ablation for the PRISM diagnostic
(G3T9-W2: benchmark-independent reference set + sample-size ablation;
 8VrD-Q3's regularizer-side sweep is driven by script_E3.sh part B via
 rebuttal_exp/train_forgetting_baselines.py).

Part A (--csv-only, ZERO GPU — uses existing paper data):
    The quantization CSV already contains bounds computed on two generic
    corpora (wikitext, fineweb_edu) for every variant. Cross-referencing:
        rs( Bound_I computed on generic corpus,  |dR| on benchmark X )
    answers "does a benchmark-independent reference set preserve the
    ranking?" for every family without loading a single model.

Part B (GPU): reference-set SIZE ablation {8, 16, 32, 64, 128} sequences of
    wikitext for one family: reload each proxy once, extract features per
    size, recompute Bound_I, correlate against the existing benchmark |dR|.
    ~1-1.5 h for the Llama family on the RTX 5090.

Usage (repo root):
    python rebuttal_exp/exp_e3_refset_ablation.py --csv-only
    python rebuttal_exp/exp_e3_refset_ablation.py --family llama   # GPU box

Output: rebuttal_exp/out/E3/partA_cross_reference.md
        rebuttal_exp/out/E3/partB_{family}_size_ablation.{csv,md}
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_DIR = HERE / "out" / "E3"
CSV_PATH = REPO / "exp_result" / "quantization" / "quantization_merged_slim.csv"

BENCHMARKS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]
GENERIC = ["wikitext", "fineweb_edu"]
SIZES = [8, 16, 32, 64, 128]


# ── tiny stats helpers (stdlib) ────────────────────────────────────────
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
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ── Part A: cross-reference-set ranking from the existing CSV ──────────
def part_a():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = defaultdict(dict)   # (target, label) -> {dataset: (B, dR)}
    for r in csv.DictReader(open(CSV_PATH)):
        data[(r["target_model"], r["Label"])][r["dataset"]] = (
            to_float(r["Bound_I"]), to_float(r["|MdR|"]))

    families = sorted({t for (t, _) in data})
    md = ["# E3 part A — benchmark-independent reference set (existing data)",
          "",
          "rs( Bound_I on reference corpus, |dR| on benchmark ). "
          "`same-bench` = the paper's default (bound and risk on the same "
          "benchmark inputs); wikitext/fineweb rows are fully "
          "benchmark-independent references.", ""]

    summary = defaultdict(list)   # ref -> list of rs (pooled over fam x bench)
    for fam in families:
        variants = [(t, l) for (t, l) in data if t == fam]
        md.append(f"\n## {fam}\n")
        md.append("| benchmark | n | same-bench | " +
                  " | ".join(GENERIC) + " |")
        md.append("|---|---|---|" + "---|" * len(GENERIC))
        for bench in BENCHMARKS:
            pairs = []
            for key in variants:
                cell = data[key]
                if bench in cell and not math.isnan(cell[bench][1]):
                    pairs.append((key, cell[bench][1]))
            if len(pairs) < 4:
                continue
            drs = [p[1] for p in pairs]
            cells = []
            b_same = [data[k][bench][0] for k, _ in pairs]
            rs_same = spearman(b_same, drs)
            summary["same-bench"].append(rs_same)
            cells.append(f"{rs_same:+.3f}")
            for ref in GENERIC:
                sub = [(data[k][ref][0], dr) for k, dr in pairs
                       if ref in data[k] and not math.isnan(data[k][ref][0])]
                if len(sub) < 4:
                    cells.append("-")
                    continue
                rs = spearman([s[0] for s in sub], [s[1] for s in sub])
                summary[ref].append(rs)
                cells.append(f"{rs:+.3f}")
            md.append(f"| {bench} | {len(pairs)} | " + " | ".join(cells) + " |")

    md += ["", "## Pooled means (all families x benchmarks)", ""]
    for ref in ["same-bench"] + GENERIC:
        vals = summary[ref]
        md.append(f"- {ref:<12s}: mean rs = {statistics.mean(vals):+.3f} "
                  f"(n={len(vals)} cells, sd {statistics.pstdev(vals):.3f})")
    out = OUT_DIR / "partA_cross_reference.md"
    out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n[written] {out}")


# ── Part B: size ablation (GPU) ────────────────────────────────────────
def part_b(args):
    import torch                                        # noqa: WPS433
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(HERE))
    from common_quant import (FAMILIES, extract_Z, free_cuda, load_proxy,  # noqa: E402
                              load_target, risk_gaps_from_csv,
                              variants_from_csv)
    from prism.core.bounds import UnifiedBound          # noqa: E402
    from prism.core.metrics import PRISMMetrics         # noqa: E402
    from prism.data.loaders import load_task_data       # noqa: E402
    from prism.models.extractors import LLMExtractor    # noqa: E402
    from transformers import AutoTokenizer              # noqa: E402

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_id = FAMILIES[args.family]
    specs = variants_from_csv(args.family)
    risk = risk_gaps_from_csv(args.family)

    # (size, draw) grid: draw 0 is the paper seed; the extra draws answer
    # REFERENCE-SET STABILITY ("did the 32 sequences just get lucky?") at
    # near-zero cost — forwards over <=128 sequences are seconds; the cost
    # of this experiment is all in model loading.
    keys = [(k, d) for k in SIZES for d in range(args.n_draws)]
    csv_path = OUT_DIR / f"partB_{args.family}_size_ablation.csv"

    # Resume: a variant is complete when all (size, draw) rows are present.
    rows, done = [], set()
    if csv_path.exists():
        with open(csv_path) as f:
            legacy = list(csv.DictReader(f))
        if legacy and "draw" in legacy[0]:
            counts = {}
            for r in legacy:
                counts[r["label"]] = counts.get(r["label"], 0) + 1
            done = {lb for lb, c in counts.items() if c >= len(keys)}
            rows = [{"label": r["label"], "ref_size": int(r["ref_size"]),
                     "draw": int(r["draw"]), "n_tokens": int(r["n_tokens"]),
                     "bound_I": float(r["bound_I"]),
                     "omega_I": float(r["omega_I"])}
                    for r in legacy if r["label"] in done]
            if done:
                print(f"[resume] {len(done)} variants already complete "
                      f"({len(rows)} rows) — skipping them")
        else:
            bak = csv_path.with_suffix(".csv.predraws")
            csv_path.rename(bak)
            print(f"[resume] legacy single-draw csv moved to {bak.name}; "
                  "restarting on the (size x draw) grid")

    todo = [s for s in specs if s["label"] not in done]
    Z_T = {}
    if todo:
        tokenizer = AutoTokenizer.from_pretrained(target_id,
                                                  trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        loaders = {
            (k, d): load_task_data(args.ref_task, split="test",
                                   num_samples=k, batch_size=min(4, k),
                                   tokenizer=tokenizer,
                                   max_length=args.max_length,
                                   seed=args.seed + 1000 * d)
            for (k, d) in keys
        }

        print(f"Extracting target features ({target_id}) ...")
        target = load_target(target_id, args.device)
        extractor = LLMExtractor()
        H_T = extractor.extract_head(target).float().cpu()
        K = UnifiedBound.theoretical_K(H_T.to(args.device))
        Z_T = {key: extract_Z(target, loaders[key], args.device)
               for key in keys}
        del target
        free_cuda()

    for spec in todo:
        label = spec["label"]
        print(f"\n=== {label} ===")
        try:
            proxy = load_proxy(spec, args.device)
        except Exception as exc:                        # noqa: BLE001
            print(f"  [FAIL load] {exc}")
            continue
        # Collect features for EVERY size first, then release the model.
        # Head metrics must never run with an 8B model resident: the
        # 2026-07-24 run OOMed a 32 GB card inside compute_all's spectral
        # SVD (4096 x |V| workspace) during the first proxy.
        Z_P_by = {}
        for key in keys:
            Z_P_by[key] = extract_Z(proxy, loaders[key], args.device)
        # Paper convention: only GGUF k-quants alter the served lm_head
        # (gamma > 0); BnB/GPTQ/dtype proxies keep the FP16 head, and their
        # quantized weight wrappers make extract_head unusable anyway.
        H_P = (extractor.extract_head(proxy).float().cpu()
               if spec["kind"] == "gguf" else None)
        del proxy
        free_cuda()

        for (k, d) in keys:
            n = min(Z_T[(k, d)].shape[0], Z_P_by[(k, d)].shape[0])
            X = Z_T[(k, d)][:n].to(args.device)
            Y = Z_P_by[(k, d)][:n].to(args.device)
            # Lightweight Bound_I path — the paper's own primitives.
            # (compute_all additionally runs a 4096 x |V| spectral SVD
            # that is NOT part of the bound; skip it.)
            rho_T = PRISMMetrics.rms_scale(X)
            rho_P = PRISMMetrics.rms_scale(Y)
            omega = PRISMMetrics.trace_omega(X, Y)
            fe = PRISMMetrics.feature_error(rho_T, rho_P, omega)
            if H_P is None:
                gamma = 0.0                    # FP16 head kept -> gamma == 0
            else:
                Sigma_P = (Y.T @ Y) / n
                gamma = PRISMMetrics.head_discrepancy_covariance(
                    H_T.to(args.device), H_P.to(args.device),
                    torch.eye(H_T.shape[0], device=args.device), Sigma_P)
            bound = K["K_feat"] * fe + K["K_pred"] * gamma
            rows.append({"label": label, "ref_size": k, "draw": d,
                         "n_tokens": n, "bound_I": bound,
                         "omega_I": omega})
            print(f"  size={k:<4d} draw={d} B={bound:.2f} omega={omega:.4f}")
            del X, Y
            free_cuda()
        del Z_P_by, H_P
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["label", "ref_size", "draw",
                                              "n_tokens", "bound_I",
                                              "omega_I"])
            w.writeheader()
            w.writerows(rows)

    # rank correlation vs benchmark |dR| per (size, draw); cells report
    # mean +/- sd ACROSS DRAWS — the sd is the stability answer.
    md = [f"# E3 part B — reference-size ablation ({args.family}, "
          f"{args.ref_task} reference)", "",
          f"{args.n_draws} independent draws per size (draw 0 = paper seed "
          f"{args.seed}; draw d uses seed {args.seed}+1000d). Cells: "
          "mean rs +/- sd across draws.", "",
          "| ref size | " + " | ".join(BENCHMARKS) + " | mean |",
          "|---|" + "---|" * (len(BENCHMARKS) + 1)]
    sds = {}
    for k in SIZES:
        cells, bench_means = [], []
        for bench in BENCHMARKS:
            rs_draws = []
            for d in range(args.n_draws):
                sub = {r["label"]: r["bound_I"] for r in rows
                       if r["ref_size"] == k and r["draw"] == d}
                pairs = [(sub[lb], risk[(lb, bench)]) for lb in sub
                         if (lb, bench) in risk
                         and not math.isnan(risk[(lb, bench)])]
                if len(pairs) >= 4:
                    rs_draws.append(spearman([p[0] for p in pairs],
                                             [p[1] for p in pairs]))
            if rs_draws:
                m = statistics.mean(rs_draws)
                sd = (statistics.stdev(rs_draws)
                      if len(rs_draws) > 1 else 0.0)
                sds[(k, bench)] = sd
                cells.append(f"{m:+.3f}±{sd:.3f}")
                bench_means.append(m)
            else:
                cells.append("-")
        md.append(f"| {k} | " + " | ".join(cells) +
                  (f" | {statistics.mean(bench_means):+.3f} |"
                   if bench_means else " | - |"))
    if sds:
        sd32 = [v for (k, b), v in sds.items() if k == 32]
        md += ["", f"Stability: max rs sd across draws = "
               f"{max(sds.values()):.3f} (all sizes/benchmarks)"
               + (f"; at the paper size 32: {max(sd32):.3f}" if sd32 else "")
               + ". A small sd at 32 certifies the paper's 32-sequence "
                 "reference is not a lucky draw."]
    out = OUT_DIR / f"partB_{args.family}_size_ablation.md"
    out.write_text("\n".join(md))
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-only", action="store_true",
                    help="run part A only (no GPU, existing data)")
    ap.add_argument("--family", choices=["llama", "qwen"], default="llama")
    ap.add_argument("--ref_task", default="wikitext")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_draws", type=int, default=3,
                    help="independent reference draws per size (draw 0 = "
                         "the paper seed; draw d uses seed+1000d). The sd "
                         "of rs across draws answers reference-set "
                         "STABILITY; extra draws cost only forwards over "
                         "<=128 sequences (seconds per variant)")
    args = ap.parse_args()

    part_a()
    if not args.csv_only:
        part_b(args)


if __name__ == "__main__":
    main()
