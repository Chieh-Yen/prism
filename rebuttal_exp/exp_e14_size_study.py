#!/usr/bin/env python3
"""E14: reference-slice size study (analysis only, zero GPU).

Consumes the CSVs that exp_e1d_fresh_round.py writes and produces the four
tables of the E14 plan.  It deliberately does NOT re-implement the GPU pass:
E1D already extracts at every (seed, size, benchmark), with cell-granularity
resume, atomic feature caching, TF32 pinned off and chunked float64
accumulation.  Duplicating that would duplicate the risk.

    # GPU, once (Llama first; decide on Qwen after seeing the wall clock)
    bash rebuttal_exp/script_E1D.sh run          # FAMILIES=llama SIZES="8 32 128 512"
    # analysis, seconds
    python3 rebuttal_exp/exp_e14_size_study.py --family llama

A ZEROTH TABLE THAT THE OTHER FOUR DEPEND ON
    Sizes are counted in SEQUENCES, but the metrics live on ANSWER-REGION tokens,
    and the answer is a single token for multiple-choice tasks. Observed at 512
    sequences on Llama: MMLU 511 tokens, ARC 527, TriviaQA 1524, SQuAD 2330,
    GSM8K 52184. So "8 sequences" is ~8 tokens on MMLU and ~815 on GSM8K, and an
    8 x 4096 feature matrix has rank 8 against d = 4096. Table (0) prints the
    token counts so no correlation below is read as if the sizes were comparable
    across benchmarks. The useful conclusion may well be "the requirement is
    ~N tokens", which is a different and more portable statement than
    "N sequences".

FOUR QUESTIONS, IN THE ORDER THAT MAKES THEM INTERPRETABLE
    (1) Is the TARGET itself stable as the slice grows?  |dR| per (variant, size),
        mean +- sd over seeds, ONE TABLE PER BENCHMARK.  Not averaged across
        benchmarks: mean |dR| runs 0.031 (GSM8K) to 0.190 (TriviaQA) on Llama, so a
        cross-benchmark mean is just TriviaQA plus noise.  The target's own CE is
        printed with it, since |dR| is a difference of two CE values and a moving
        base would explain a moving difference.
    (2) At a FIXED size, do the three seeds agree on the ORDERING?  Measured as
        pairwise Spearman between the seeds' own orderings of the variants, not as
        the sd of a correlation against |dR|.  Those are different things: E3-B
        reported sd = 0.000 across draws while the underlying bound values moved by
        up to 10.8, because Spearman only sees ranks and ten of the eleven adjacent
        pairs were separated by more than the jitter.  Seed-to-seed ordering
        agreement is the honest statistic.
    (3) Diagonal: rs(score at n, |dR| at n).  Fully self-contained at each size,
        i.e. "on this slice, does the score order this slice's gap?"
    (4) Fixed target: rs(score at n, |dR| at 512).  The decision-relevant one:
        "can a SMALL slice order the variants the way the full protocol does?"
        Note the sizes are nested (shuffle(seed) then select(range(n)) makes
        8 subset 32 subset 128 subset 512), so a small slice is a subset of the
        target set and there is no distribution shift between them; the script
        checks the nesting claim against the token counts it finds.

WHAT WOULD FALSIFY THE "A SMALL SLICE IS ENOUGH" CLAIM
    (4) dropping off at small n, or (2) showing seeds disagreeing at small n.
    Both are reported for every size and benchmark, and the summary prints the
    worst cell rather than the mean, so a bad corner cannot hide behind an
    average.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
E1D = HERE / "out" / "E1D"
OUT = HERE / "out" / "E14"
PAPER5 = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]
SCORES = {                       # display name -> (column, similarity?)
    "1-CKA": ("cka", True),
    "1-SVCCA": ("svcca", True),
    "1-Omega_N": ("omega_W", True),
    "feature arm_N": ("delta_W", False),
    "PRISM B_N": ("bound_W", False),
}


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


def load(family: str):
    """rows keyed by (seed, size, benchmark, variant)."""
    files = sorted(E1D.glob(f"{family}_seed*.csv"))
    if not files:
        raise SystemExit(f"no {family}_seed*.csv in {E1D}; run the E1D GPU pass first")
    data = {}
    for p in files:
        for r in csv.DictReader(open(p)):
            key = (int(r["seed"]), int(r.get("n_samples", 512) or 512),
                   r["dataset"].lower(), r["label"])
            data[key] = r
    seeds = sorted({k[0] for k in data})
    sizes = sorted({k[1] for k in data})
    benches = [b for b in PAPER5 if any(k[2] == b for k in data)]
    variants = sorted({k[3] for k in data})
    print(f"[load] {family}: {len(data)} cells | seeds {seeds} | sizes {sizes} | "
          f"{len(benches)} benchmarks | {len(variants)} variants")
    return data, seeds, sizes, benches, variants


def fmt(vals, prec=4) -> str:
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.{prec}f}"
    return f"{statistics.mean(vals):.{prec}f}±{statistics.stdev(vals):.{prec}f}"


def get(data, seed, n, b, v, col):
    r = data.get((seed, n, b, v))
    if r is None or r.get(col) in (None, ""):
        return None
    try:
        return float(r[col])
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="llama", choices=["llama", "qwen"])
    ap.add_argument("--target-size", type=int, default=None,
                    help="fixed target for question 4 (default: the largest size)")
    args = ap.parse_args()

    data, seeds, sizes, benches, variants = load(args.family)
    ntarget = args.target_size or sizes[-1]
    OUT.mkdir(parents=True, exist_ok=True)
    L = [f"# E14 reference-slice size study ({args.family})", "",
         f"seeds {seeds}; sizes {sizes}; fixed target for Q4 = n={ntarget}.",
         "Sizes are nested by construction (one shuffle per seed, then the first n),",
         "so each slice is a subset of the target set.", ""]

    # ── nesting sanity: token counts must be non-decreasing in n ──
    bad = []
    for s in seeds:
        for b in benches:
            tk = [(n, get(data, s, n, b, variants[0], "n_tokens")) for n in sizes]
            tk = [(n, t) for n, t in tk if t is not None]
            if any(tk[i][1] > tk[i + 1][1] for i in range(len(tk) - 1)):
                bad.append((s, b, tk))
    L += ([f"WARNING: token counts are not monotone in n for {len(bad)} "
           f"(seed, benchmark) pairs, so the slices may not be nested: {bad[:3]}", ""]
          if bad else ["Nesting check: token counts are monotone in n everywhere.", ""])

    # ── (0) token counts: the size axis means very different things per
    #    benchmark, so this table has to come first ──
    L += ["## (0) What a slice size actually buys, in TOKENS", "",
          "The features and the CE both come from the ANSWER region, and the answer",
          "is one token for multiple-choice tasks, so a fixed number of sequences is",
          "a wildly different number of tokens per benchmark. Since d = 4096, a slice",
          "with fewer than a few thousand tokens gives a rank-deficient feature",
          "matrix, and the similarity scores are computed inside that subspace. Read",
          "every table below against this one.", "",
          "| benchmark | " + " | ".join(f"n={n}" for n in sizes) + " | tokens/sequence |",
          "|:--|" + "--:|" * (len(sizes) + 1)]
    for b in benches:
        tk = [get(data, seeds[0], n, b, variants[0], "n_tokens") for n in sizes]
        per = (tk[-1] / sizes[-1]) if tk[-1] else float("nan")
        L.append(f"| {b} | " + " | ".join("-" if x is None else f"{int(x)}" for x in tk)
                 + f" | {per:.1f} |")
    L += ["", "A row whose smallest entries are in the tens is telling you that the",
          "requirement should be stated in tokens, not sequences.", ""]

    # ── (1) is the target stable? per benchmark, no cross-benchmark averaging ──
    L += ["## (1) Measured gap by variant and slice size", "",
          "Cells are mean±sd of |dR| over seeds. `base CE` is the target's own",
          "answer-span loss at that size, listed because |dR| is a difference of",
          "two CE values.", ""]
    for b in benches:
        L += [f"### {b}", "",
              "| variant | " + " | ".join(f"n={n}" for n in sizes) + " |",
              "|:--|" + "--:|" * len(sizes)]
        for v in variants:
            cells = [fmt([get(data, s, n, b, v, "|MdR|") for s in seeds])
                     for n in sizes]
            L.append(f"| `{v[:26]}` | " + " | ".join(cells) + " |")
        L.append("| **base CE (target)** | " + " | ".join(
            fmt([get(data, s, n, b, variants[0], "loss_T") for s in seeds])
            for n in sizes) + " |")
        L.append("")

    # ── (2) do seeds agree on the ordering, at fixed size? ──
    L += ["## (2) Seed agreement on the ordering, at fixed size", "",
          "Pairwise Spearman BETWEEN seeds' own orderings of the variants (not",
          "against |dR|). 1.000 means the three seeds rank the variants",
          "identically. The worst cell over benchmarks is what matters, so it is",
          "printed next to the mean.", ""]
    for name, (col, inv) in SCORES.items():
        L += [f"### {name}", "",
              "| size | " + " | ".join(benches) + " | mean | worst |",
              "|:--|" + "--:|" * (len(benches) + 2)]
        for n in sizes:
            cells, allv = [], []
            for b in benches:
                pair = []
                for i in range(len(seeds)):
                    for j in range(i + 1, len(seeds)):
                        a = [get(data, seeds[i], n, b, v, col) for v in variants]
                        c = [get(data, seeds[j], n, b, v, col) for v in variants]
                        ok = [k for k in range(len(variants))
                              if a[k] is not None and c[k] is not None]
                        if len(ok) >= 3:
                            pair.append(spearman([a[k] for k in ok],
                                                 [c[k] for k in ok]))
                cells.append(fmt(pair, 3) if pair else "-")
                allv += pair
            L.append(f"| {n} | " + " | ".join(cells) +
                     f" | {statistics.mean(allv):.3f} | {min(allv):.3f} |"
                     if allv else f"| {n} | " + " | ".join(cells) + " | - | - |")
        L.append("")

    # ── (3) diagonal and (4) fixed target ──
    for tag, target_of in (("(3) Diagonal: score at n against |dR| at the SAME n",
                            lambda n: n),
                           (f"(4) Fixed target: score at n against |dR| at n={ntarget}",
                            lambda n: ntarget)):
        L += [f"## {tag}", "",
              "Cells are mean±sd over seeds of the per-benchmark Spearman; `worst`",
              "is the lowest single (seed, benchmark) value at that size.", "",
              "| size | " + " | ".join(SCORES) + " | worst cell |",
              "|:--|" + "--:|" * (len(SCORES) + 1)]
        for n in sizes:
            cells, worst = [], []
            for name, (col, inv) in SCORES.items():
                per_seed = []
                for s in seeds:
                    rs = []
                    for b in benches:
                        sc, tg = [], []
                        for v in variants:
                            a = get(data, s, n, b, v, col)
                            y = get(data, s, target_of(n), b, v, "|MdR|")
                            if a is None or y is None:
                                continue
                            sc.append(1 - a if inv else a)
                            tg.append(abs(y))
                        if len(sc) >= 3:
                            r = spearman(sc, tg)
                            rs.append(r)
                            worst.append(r)
                    if rs:
                        per_seed.append(statistics.mean(rs))
                cells.append(fmt(per_seed, 3))
            L.append(f"| {n} | " + " | ".join(cells) +
                     (f" | {min(worst):.3f} |" if worst else " | - |"))
        L.append("")

    L += ["## How to read this", "",
          f"* If (4) at n=8 matches n={ntarget}, eight sequences already order the",
          "  variants the way the full protocol does, and that is the claim to make.",
          "* If (2) is below 1.000 at small n, the ordering itself is seed-dependent",
          "  there, and any size claim must be stated with that spread.",
          "* If (1) shows |dR| moving strongly with n while (4) stays flat, the",
          "  ranking is robust even though the target is not, which is the more",
          "  interesting result and should be said explicitly.", ""]

    p = OUT / f"{args.family}_size_study.md"
    p.write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"\n... [write] {p}")


if __name__ == "__main__":
    main()
