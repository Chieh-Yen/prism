#!/usr/bin/env python3
"""Analysis of head experiment 1 (output of head_swap.py). No GPU.

Per cell (model, benchmark, GGUF variant) the input stores three gaps to the
target, each |R_target - R| in nats:
  V  the variant           (quantized backbone + quantized head)   field dR
  A  hybrid A              (quantized backbone + original head)    field dR_fp16head
  B  hybrid B              (original backbone + quantized head)    field dR_quanthead

Quantities printed
  integrity   hybrid A has the variant's features and hybrid B the target's,
              entry for entry (backbone_identical, head_only_identical).
  bound       K_feat * delta + K_pred * gamma >= gap, for V, A and B, with
              each model's own delta and gamma. K_feat is read from the row
              when present, otherwise from K_FEAT below; K_pred = sqrt(2).
  additivity  per cell with V >= NOISE_FLOOR, the additivity ratio
                r = ((V - A) + (V - B)) / V,
              where V - A is what restoring the original head removes from the
              variant's gap and V - B is what restoring the original backbone
              removes. r = 1 means the two reductions add up to the variant's
              gap. The reported number is the median of r over these cells
              (ordinary median: mean of the two middle values for an even
              count). A second line gives the median of the sum-of-gaps form
              (A + B) / V, which equals 2 - r cell by cell. V, A and B are
              absolute gaps. A third line uses the signed gaps
              s = R - R_target instead and gives, over all cells, the largest
              residual |s_A + s_B - s_V| in nats.
  ranking     in hybrid A the head term is 0, so the bound is the feature
              term alone. For every (model, benchmark) with at least two
              variants: does K_feat * delta of hybrid A order the variants as
              the gap A does? Ties count as wrong. The pair count is given
              twice: over all pairs of variants, and over the pairs whose
              hybrid-A gaps differ by at least NOISE_FLOOR.

NOISE_FLOOR is fixed at 0.02 nats and applies to gaps and to differences of
gaps (see the constant below).

usage:
  python analyze_head_swap.py FILE [FILE ...] [--targets ID[,ID ...]]

--targets gives the target model of each FILE, in order; it is needed only for
files whose rows do not store "target".
"""
from __future__ import annotations
import argparse, itertools, json, math, statistics

# K_feat of the target head (UnifiedBound.theoretical_K: the largest distance
# between two columns of H_T), as the quantization experiment reports it for
# these targets. Used only when an input row does not store K_feat.
K_FEAT = {"Qwen/Qwen3-8B-Base": 3.4582768324560638,
          "meta-llama/Llama-3.1-8B": 2.613742029368554,
          "meta-llama/Meta-Llama-3.1-8B": 2.613742029368554}
K_PRED = math.sqrt(2.0)
# Noise floor of the head experiments, in nats: gaps and differences of gaps
# below it are treated as noise. It is a constant of the analysis, not an
# option.
NOISE_FLOOR = 0.02
QUANT_ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]


def load(paths, targets):
    rows = []
    for i, p in enumerate(paths):
        for r in json.load(open(p)):
            if "target" not in r:
                if targets is None or i >= len(targets):
                    raise SystemExit(f"{p}: rows do not store the target model; pass --targets")
                r["target"] = targets[i]
            if r.get("dR_quanthead") is None:
                raise SystemExit(f"{p}: {r['task']}/{r['quant']} has no hybrid B")
            if "K_feat" not in r:
                if r["target"] not in K_FEAT:
                    raise SystemExit(f"{p}: no K_feat stored and none tabulated for {r['target']}")
                r["K_feat"], r["K_pred"], r["K_source"] = K_FEAT[r["target"]], K_PRED, "tabulated"
            else:
                r["K_source"] = "stored"
            r["model"] = r["target"].split("/")[-1]
            rows.append(r)
    keys = [(r["model"], r["task"], r["quant"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate cells in the input files")
    return rows


def qrank(q):
    return QUANT_ORDER.index(q) if q in QUANT_ORDER else len(QUANT_ORDER)


def main():
    ap = argparse.ArgumentParser(description="Analysis of head experiment 1 (hybrids).")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--targets", default=None,
                    help="comma-separated target model ids, one per file, for files "
                         "whose rows do not store the target")
    a = ap.parse_args()
    a.floor = NOISE_FLOOR
    rows = load(a.files, a.targets.split(",") if a.targets else None)
    rows.sort(key=lambda r: (r["model"], r["task"], qrank(r["quant"])))
    models = sorted({r["model"] for r in rows})

    print(f"cells: {len(rows)}")
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        print(f"  {m}: {len(rs)} cells, variants {sorted({r['quant'] for r in rs}, key=qrank)}, "
              f"benchmarks {sorted({r['task'] for r in rs})}, n = {sorted({r['n_samples'] for r in rs})}, "
              f"K_feat = {rs[0]['K_feat']:.4f} ({rs[0]['K_source']}), K_pred = {rs[0]['K_pred']:.4f}")

    ok = sum(bool(r["backbone_identical"]) and bool(r["head_only_identical"]) for r in rows)
    print(f"\nintegrity (features identical entry for entry in both hybrids): {ok} / {len(rows)}")

    print("\nbound  K_feat * delta + K_pred * gamma >= gap")
    trip = [("variant", "dR", "delta", "gamma"),
            ("hybrid A", "dR_fp16head", "delta_fp16head", "gamma_fp16head"),
            ("hybrid B", "dR_quanthead", "delta_quanthead", "gamma_quanthead")]
    for name, g, d, h in trip:
        holds = sum(r[g] <= r["K_feat"] * r[d] + r["K_pred"] * r[h] for r in rows)
        print(f"  {name:9s} holds in {holds} / {len(rows)} cells")

    print(f"\nper cell (gaps in nats; * marks V >= {a.floor})")
    print(f"  {'model':14s} {'benchmark':9s} {'variant':7s} {'V':>8s} {'A':>8s} {'B':>8s} "
          f"{'r':>8s} {'(A+B)/V':>8s}")
    for r in rows:
        V, A, B = r["dR"], r["dR_fp16head"], r["dR_quanthead"]
        r["ratio_reductions"] = ((V - A) + (V - B)) / V
        r["ratio_gaps"] = (A + B) / V
        ratios = (f"{r['ratio_reductions']:8.3f} {r['ratio_gaps']:8.3f} *" if V >= a.floor
                  else f"{'-':>8s} {'-':>8s}")
        print(f"  {r['model']:14s} {r['task']:9s} {r['quant']:7s} {V:8.4f} {A:8.4f} {B:8.4f} {ratios}")

    above = [r for r in rows if r["dR"] >= a.floor]
    print(f"\nadditivity over the cells whose variant gap is >= {a.floor} nats: "
          f"{len(above)} of {len(rows)} cells, on {sorted({r['model'] for r in above})}")
    if above:
        red = [r["ratio_reductions"] for r in above]
        gap = [r["ratio_gaps"] for r in above]
        print(f"  additivity ratio r = ((V - A) + (V - B)) / V : median {statistics.median(red):.3f}")
        print(f"  sum-of-gaps form (A + B) / V = 2 - r          : median {statistics.median(gap):.3f}")
    if all(k in r for r in rows for k in ("loss_T", "loss_P", "loss_P_fp16head", "loss_T_quanthead")):
        resid = [abs((r["loss_P_fp16head"] - r["loss_T"]) + (r["loss_T_quanthead"] - r["loss_T"])
                     - (r["loss_P"] - r["loss_T"])) for r in rows]
        print(f"  signed gaps s = R - R_target, all {len(rows)} cells: "
              f"largest |s_A + s_B - s_V| = {max(resid):.4f} nats")

    print("\nranking of the variants by the feature term in hybrid A (head term = 0)")
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["task"]), []).append(r)
    n_groups = n_right = conc = tot = conc_f = tot_f = 0
    for (m, t), rs in sorted(groups.items()):
        if len(rs) < 2:
            continue
        n_groups += 1
        by_term = sorted(rs, key=lambda r: r["K_feat"] * r["delta_fp16head"])
        right = True
        for x, y in itertools.combinations(rs, 2):
            c = (x["K_feat"] * x["delta_fp16head"] - y["K_feat"] * y["delta_fp16head"]) \
                * (x["dR_fp16head"] - y["dR_fp16head"]) > 0
            tot += 1; conc += c; right = right and c
            if abs(x["dR_fp16head"] - y["dR_fp16head"]) >= a.floor:
                tot_f += 1; conc_f += c
        n_right += right
        print(f"  {m:14s} {t:9s} " + "  ".join(
            f"{r['quant']}: term {r['K_feat'] * r['delta_fp16head']:7.2f}, gap {r['dR_fp16head']:.4f}"
            for r in by_term) + f"   {'same order' if right else 'ORDER DIFFERS'}")
    print(f"  benchmarks ranked without error: {n_right} / {n_groups}")
    print(f"  pairs of variants ranked correctly: {conc} / {tot} (all pairs), "
          f"{conc_f} / {tot_f} (pairs whose hybrid-A gaps differ by >= {a.floor} nats)")


if __name__ == "__main__":
    main()
