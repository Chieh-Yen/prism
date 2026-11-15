#!/usr/bin/env python3
"""Eight-sequence screen: tables and tests from the CSVs written by run_screen.py (no GPU).

For every reference set (family, benchmark, pool seed, set index) the script computes the
Spearman r_s, across the quantized variants, between each score computed on the set's eight
sequences and the risk gap measured on the pool's 512 sequences (column gap512):

    gap@8    risk gap measured on the same eight sequences            (column gap8)
    1-CKA    one minus linear CKA                                     (1 - cka)
    B_N      PRISM bound under the Procrustes alignment               (column bound_W)

A cell is a (family, benchmark) pair; it holds 300 sets (3 pools x 100 sets).

Each r_s is rounded to six decimals before anything else is computed.

Printed:

  1. summary over the ten cells: mean r_s (mean of the ten cell means), the worst single
     set, and an exact two-sided paired Wilcoxon signed-rank test across the ten cell means
     of B_N against each other score;
  2. per cell: mean and standard deviation of r_s over the 300 sets.  The standard
     deviation is the population one (ddof = 0);
  3. within each cell: paired two-sided Wilcoxon signed-rank test over the 300 sets of B_N
     against 1-CKA (normal approximation with tie correction, zero differences dropped),
     Holm-corrected over the ten cells, and the paired t-test as a check;
  4. bound coverage: how many (variant, set) rows have B_N >= gap8 and B_N >= gap512;
  5. with --size-round DIR, from the *_seed*.csv files that the three-seed round
     (scripts/rounds/three_seed_round.py, default out/rounds/three_seed) writes: the same
     coverage count at every reference size.

The screen does not compute or score 1-SVCCA; the CSVs of the three-seed round hold an
``svcca`` column.

--exclude-label LABEL (repeatable) drops every row of the variant with that ``label``
before anything is computed, and all sections are then printed for the remaining variants.
The default excludes nothing.

Usage:
    python scripts/response/screen_n8/analyze_screen.py
    python scripts/response/screen_n8/analyze_screen.py --csv A/llama_screen.csv A/qwen_screen.csv
    python scripts/response/screen_n8/analyze_screen.py --exclude-label "<label>"
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent.parent      # supplementary root
DEFAULT_DIR = ROOT / "out" / "response" / "screen_n8"

FAMILY_NAME = {"llama": "Llama", "qwen": "Qwen"}
BENCH_NAME = {"arc": "ARC", "gsm8k": "GSM8K", "mmlu": "MMLU", "squad": "SQuAD", "triviaqa": "TriviaQA"}
SCORES = [
    ("gap@8", lambda r: float(r["gap8"])),
    ("1-CKA", lambda r: 1.0 - float(r["cka"])),
    ("B_N", lambda r: float(r["bound_W"])),
]
NAMES = [name for name, _ in SCORES]
SETS_PER_CELL = 300
RS_DECIMALS = 6
VARIANTS = {"llama": 12, "qwen": 11}


def cell_name(cell):
    return f"{FAMILY_NAME.get(cell[0], cell[0])} {BENCH_NAME.get(cell[1], cell[1])}"


def holm(pvalues):
    """Holm step-down adjusted p-values, in the order given."""
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def load_rows(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def per_set_rs(rows):
    """{cell: {(seed, set): {score: r_s}}}, r_s rounded to RS_DECIMALS, plus the variant labels of each cell."""
    by_set = defaultdict(list)
    for r in rows:
        by_set[(r["family"], r["dataset"], int(r["seed"]), int(r["draw"]))].append(r)
    rs = defaultdict(dict)
    labels = defaultdict(set)
    for (fam, ds, seed, k), v in by_set.items():
        target = [float(r["gap512"]) for r in v]
        rs[(fam, ds)][(seed, k)] = {name: round(float(stats.spearmanr([f(r) for r in v], target)[0]), RS_DECIMALS)
                                    for name, f in SCORES}
        labels[(fam, ds)].add(tuple(sorted(r["label"] for r in v)))
    return rs, labels


def check_grid(rows, rs, labels, expected=VARIANTS):
    """Refuse an incomplete grid: every cell must hold the full set of sets and variants.

    ``expected`` is the number of variants per family (VARIANTS minus the excluded ones)."""
    problems = []
    for cell in sorted(rs):
        if len(rs[cell]) != SETS_PER_CELL:
            problems.append(f"{cell_name(cell)}: {len(rs[cell])} sets, expected {SETS_PER_CELL}")
        if len(labels[cell]) != 1:
            problems.append(f"{cell_name(cell)}: the variant list differs between sets")
        elif cell[0] in expected and len(next(iter(labels[cell]))) != expected[cell[0]]:
            problems.append(f"{cell_name(cell)}: {len(next(iter(labels[cell])))} variants, expected {expected[cell[0]]}")
    seen = set()
    target = defaultdict(set)
    for r in rows:
        key = (r["family"], r["dataset"], r["seed"], r["draw"], r["label"])
        if key in seen:
            problems.append(f"duplicate row {key}")
        seen.add(key)
        target[key[:3] + key[4:]].add(r["gap512"])
    if any(len(v) != 1 for v in target.values()):
        problems.append("gap512 is not constant within a (family, benchmark, seed, variant) pool")
    return problems


def size_round_checks(directory):
    """Bound coverage per reference size, from the three-seed round's CSVs."""
    rows = load_rows(sorted(Path(directory).glob("*_seed*.csv")))
    if not rows:
        raise SystemExit(f"no *_seed*.csv files in {directory}")
    sizes = sorted({int(r["n_samples"]) for r in rows})
    full = {(r["family"], r["seed"], r["label"], r["dataset"]): float(r["|MdR|"])
            for r in rows if int(r["n_samples"]) == sizes[-1]}
    print(f"\n5. Three-seed round ({directory}): bound coverage by reference size")
    print(f"   rows are (variant, benchmark, seed); the full-size gap is the one at n = {sizes[-1]}\n")
    print("| n | rows | B_N >= gap on the same n | B_N >= gap at full size |")
    print("|--:|--:|--:|--:|")
    for n in sizes:
        sub = [r for r in rows if int(r["n_samples"]) == n]
        same = sum(float(r["bound_W"]) >= float(r["|MdR|"]) for r in sub)
        ref = sum(float(r["bound_W"]) >= full[(r["family"], r["seed"], r["label"], r["dataset"])] for r in sub)
        print(f"| {n} | {len(sub)} | {same} / {len(sub)} | {ref} / {len(sub)} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", default=None,
                    help="screen CSVs (default: llama_screen.csv and qwen_screen.csv under out/response/screen_n8/)")
    ap.add_argument("--size-round", default=None,
                    help="directory with the three-seed round's *_seed*.csv files; adds section 5")
    ap.add_argument("--allow-incomplete", action="store_true", help="print the tables even if the grid is incomplete")
    ap.add_argument("--exclude-label", action="append", default=[], metavar="LABEL",
                    help="drop the variant with this value in the CSV column 'label' before the analysis; "
                         "repeatable (default: no variant is dropped)")
    a = ap.parse_args()
    paths = a.csv or [DEFAULT_DIR / f"{fam}_screen.csv" for fam in ("llama", "qwen")]
    rows = load_rows(paths)
    expected = dict(VARIANTS)
    if a.exclude_label:
        for lab in a.exclude_label:
            families = sorted({r["family"] for r in rows if r["label"] == lab})
            if not families:
                raise SystemExit(f"--exclude-label: no row has label {lab!r}")
            for fam in families:
                if fam in expected:
                    expected[fam] -= 1
            print(f"excluded variant: {lab} ({', '.join(FAMILY_NAME.get(f, f) for f in families)})")
        rows = [r for r in rows if r["label"] not in a.exclude_label]
    rs, labels = per_set_rs(rows)
    cells = sorted(rs)

    print(f"{len(rows)} (variant, set) rows; {len(cells)} cells; {sum(len(rs[c]) for c in cells)} reference sets")
    problems = check_grid(rows, rs, labels, expected)
    for p in problems:
        print(f"!! {p}")
    if problems and not a.allow_incomplete:
        raise SystemExit("incomplete grid: finish the run or pass --allow-incomplete")

    per_cell = {c: {n: np.array([rs[c][k][n] for k in sorted(rs[c])]) for n in NAMES} for c in cells}
    cell_mean = {n: np.array([per_cell[c][n].mean() for c in cells]) for n in NAMES}

    print(f"\n1. Summary at n = 8 ({len(cells)} cells x {SETS_PER_CELL} reference sets)\n")
    print("| | " + " | ".join(NAMES) + " |")
    print("|---|" + "--:|" * len(NAMES))
    print("| mean r_s (mean of the cell means) | " + " | ".join(f"{cell_mean[n].mean():.3f}" for n in NAMES) + " |")
    print("| worst single reference set | " + " | ".join(f"{min(per_cell[c][n].min() for c in cells):.3f}" for n in NAMES) + " |")
    line = "| cells where the mean r_s of B_N is higher (exact two-sided paired Wilcoxon p) | "
    for n in NAMES[:-1]:
        higher = int((cell_mean["B_N"] > cell_mean[n]).sum())
        p = stats.wilcoxon(cell_mean["B_N"], cell_mean[n], alternative="two-sided", method="exact").pvalue
        line += f"{higher} / {len(cells)} (p = {p:.5f}) | "
    print(line + " |")

    print("\n2. Per cell: r_s, mean ± population sd (ddof = 0) over the 300 sets\n")
    print("| model | benchmark | variants | " + " | ".join(NAMES) + " |")
    print("|---|---|--:|" + "--:|" * len(NAMES))
    for c in cells:
        v = per_cell[c]
        print(f"| {FAMILY_NAME.get(c[0], c[0])} | {BENCH_NAME.get(c[1], c[1])} | {len(next(iter(labels[c])))} | "
              + " | ".join(f"{v[n].mean():.3f} ± {v[n].std(ddof=0):.3f}" for n in NAMES) + " |")
    for c in cells:
        for n in NAMES[:-1]:
            if f"{per_cell[c][n].mean():.3f}" == f"{per_cell[c]['B_N'].mean():.3f}":
                print(f"   {cell_name(c)}: {n} {per_cell[c][n].mean():.4f}, B_N {per_cell[c]['B_N'].mean():.4f} (four decimals)")

    other = "1-CKA"
    diff = {c: per_cell[c]["B_N"] - per_cell[c][other] for c in cells}
    p_w = np.array([stats.wilcoxon(per_cell[c]["B_N"], per_cell[c][other], zero_method="wilcox", correction=False,
                                   alternative="two-sided", method="approx").pvalue for c in cells])
    p_t = np.array([stats.ttest_rel(per_cell[c]["B_N"], per_cell[c][other]).pvalue for c in cells])
    h_w, h_t = holm(p_w), holm(p_t)
    print(f"\n3. Within each cell, B_N against {other}: paired tests over the 300 sets, two-sided\n")
    print(f"| model | benchmark | mean of r_s(B_N) - r_s({other}) | Wilcoxon p, Holm | t-test p, Holm |")
    print("|---|---|--:|--:|--:|")
    for i, c in enumerate(cells):
        print(f"| {FAMILY_NAME.get(c[0], c[0])} | {BENCH_NAME.get(c[1], c[1])} | {diff[c].mean():+.3f} | "
              f"{h_w[i]:.1e} | {h_t[i]:.1e} |")
    print(f"   largest Holm-corrected p: Wilcoxon {h_w.max():.1e}, t-test {h_t.max():.1e}; "
          f"cells with a positive mean difference: {sum(diff[c].mean() > 0 for c in cells)} / {len(cells)}")

    bound = np.array([float(r["bound_W"]) for r in rows])
    gap8 = np.array([float(r["gap8"]) for r in rows])
    gap512 = np.array([float(r["gap512"]) for r in rows])
    print("\n4. Bound coverage at n = 8, over all (variant, set) rows\n")
    print(f"   B_N >= gap on the same eight sequences : {int((bound >= gap8).sum())} / {len(rows)}")
    print(f"   B_N >= gap on the pool's 512 sequences : {int((bound >= gap512).sum())} / {len(rows)}")
    cells_all = sum(all(float(r["bound_W"]) >= float(r["gap512"]) for r in rows if (r["family"], r["dataset"]) == c)
                    for c in cells)
    print(f"   cells in which every row satisfies B_N >= gap512: {cells_all} / {len(cells)}")

    if a.size_round:
        size_round_checks(a.size_round)


if __name__ == "__main__":
    main()
