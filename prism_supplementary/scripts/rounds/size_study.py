#!/usr/bin/env python3
"""Reference-set size table (Appendix F.2) from the CSVs of three_seed_round.py. No GPU.

For one family the script prints two tables and writes them to
out/rounds/size_study/{family}_size_study.md.

(0) Token rows per reference size
    Sizes are counted in sequences, while the features are the answer-token
    rows of those sequences.  A multiple-choice answer is one token, so the
    same number of sequences gives very different numbers of rows per
    benchmark.  The table lists the row counts (first seed, first variant).

(1) Ranking against the gap at the largest size
    For each seed, benchmark and reference size n, the Spearman r_s across the
    variants between a score computed on the n-sequence slice and the risk gap
    measured on the largest slice of the same seed (512 sequences).  The
    slices are nested (one shuffle per seed, then the first n sequences), so
    the n-sequence slice is a subset of the largest one.  Per seed, r_s is
    averaged over the five benchmarks; the table gives the mean and the
    SAMPLE standard deviation (``statistics.stdev``, divisor n - 1) of these
    per-seed means over the seeds.

    Scores (CSV columns of three_seed_round.py):
        gap on the slice   |MdR| measured on the n-sequence slice itself; at
                           the largest size it is the target, so r_s = 1
        1-CKA              1 - cka
        1-SVCCA            1 - svcca
        delta (W = I)      feature term under the identity alignment,
                           sqrt((rho_T - rho_P)^2 + 2 rho_T rho_P (1 - omega_I))
        delta_N            feature term under the Procrustes alignment (delta_W)
        B_N                the bound under the Procrustes alignment (bound_W)

The script refuses an incomplete grid: every (seed, size, benchmark) cell must
hold every variant of the manifest.

Usage (from the supplementary root)
    python3 scripts/rounds/size_study.py --family llama
    python3 scripts/rounds/size_study.py --family qwen --dir OTHER_DIR
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent          # supplementary root
DEFAULT_DIR = ROOT / "out" / "rounds" / "three_seed"
OUT = ROOT / "out" / "rounds" / "size_study"
MANIFEST = HERE / "data" / "variant_manifest.csv"
BENCHMARKS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]


def delta_identity(r) -> float:
    rho_t, rho_p, omega = float(r["rho_T"]), float(r["rho_P"]), float(r["omega_I"])
    return math.sqrt(max((rho_t - rho_p) ** 2 + 2.0 * rho_t * rho_p * (1.0 - omega), 0.0))


SCORES = {
    "gap on the slice": lambda r: abs(float(r["|MdR|"])),
    "1-CKA": lambda r: 1.0 - float(r["cka"]),
    "1-SVCCA": lambda r: 1.0 - float(r["svcca"]),
    "delta (W = I)": delta_identity,
    "delta_N": lambda r: float(r["delta_W"]),
    "B_N": lambda r: float(r["bound_W"]),
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


def manifest_labels(family: str):
    with open(MANIFEST, newline="") as fh:
        return sorted({r["Label"] for r in csv.DictReader(fh) if r["family"] == family})


def load(family: str, directory: Path):
    """Rows keyed by (seed, size, benchmark, variant); the grid must be complete."""
    files = sorted(directory.glob(f"{family}_seed*.csv"))
    if not files:
        raise SystemExit(f"no {family}_seed*.csv in {directory}; run three_seed_round.py first")
    data = {}
    for p in files:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (int(r["seed"]), int(r["n_samples"]), r["dataset"].lower(), r["label"])
                if key in data:
                    raise SystemExit(f"{p.name}: duplicate row {key}")
                data[key] = r
    seeds = sorted({k[0] for k in data})
    sizes = sorted({k[1] for k in data})
    variants = manifest_labels(family)
    missing = [(s, n, b, v) for s in seeds for n in sizes for b in BENCHMARKS for v in variants
               if (s, n, b, v) not in data]
    if missing:
        raise SystemExit(f"incomplete grid: {len(missing)} of "
                         f"{len(seeds) * len(sizes) * len(BENCHMARKS) * len(variants)} "
                         f"(seed, size, benchmark, variant) rows are missing, e.g. {missing[:3]}")
    return data, seeds, sizes, variants


def fmt(vals) -> str:
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="llama", choices=["llama", "qwen"])
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                    help="directory with the {family}_seed*.csv files of three_seed_round.py")
    args = ap.parse_args()

    data, seeds, sizes, variants = load(args.family, args.dir)
    target = sizes[-1]
    L = [f"# Reference-set size ({args.family})", "",
         f"seeds {seeds}; sizes {sizes}; {len(variants)} variants; "
         f"target: the risk gap at n = {target}.", ""]

    L += ["## (0) Token rows per reference size", "",
          "Answer-token rows of the feature matrix (first seed, first variant).", "",
          "| benchmark | " + " | ".join(f"n = {n}" for n in sizes) + " |",
          "|:--|" + "--:|" * len(sizes)]
    for b in BENCHMARKS:
        L.append(f"| {b} | " + " | ".join(
            data[(seeds[0], n, b, variants[0])]["n_tokens"] for n in sizes) + " |")
    L.append("")

    L += [f"## (1) r_s of the score at n against the risk gap at n = {target}", "",
          "Per seed, the mean over the five benchmarks of the Spearman r_s across the",
          f"variants; cells are mean ± sample sd (divisor n - 1) over the {len(seeds)} seeds.", "",
          "| n | " + " | ".join(SCORES) + " |",
          "|--:|" + "--:|" * len(SCORES)]
    for n in sizes:
        cells = []
        for score in SCORES.values():
            per_seed = []
            for s in seeds:
                rs = []
                for b in BENCHMARKS:
                    x = [score(data[(s, n, b, v)]) for v in variants]
                    y = [abs(float(data[(s, target, b, v)]["|MdR|"])) for v in variants]
                    rs.append(spearman(x, y))
                per_seed.append(statistics.mean(rs))
            cells.append(fmt(per_seed))
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    L.append("")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{args.family}_size_study.md"
    out_path.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
