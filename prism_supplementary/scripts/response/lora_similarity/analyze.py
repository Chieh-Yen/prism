#!/usr/bin/env python3
"""Checkpoint ranking on the LoRA series: PRISM bound against 1-CKA and 1-SVCCA.

Input: the runs written by ``run_series.py`` (one
``prism_forgetting_metrics.json`` per run, 12 checkpoints at steps 25..300).

Definitions
-----------
cell    one (fine-tuning task, downstream benchmark) pair; two tasks x five
        benchmarks = ten cells, each trained with three seeds.
gap     |delta_risk|: the absolute difference between the fine-tuned
        checkpoint's and the base model's answer-only cross-entropy on the
        256 evaluation sequences of the benchmark.
bound   the PRISM bound under the identity alignment W = I
        (``bound_total``), computed on those 256 sequences.  LoRA leaves the
        output head frozen, so the head term is zero and the bound equals the
        feature term K_feat * sqrt(scale + shape).
1-CKA, 1-SVCCA   one minus linear CKA / SVCCA between the base model's and
        the checkpoint's final-layer features, the features the bound uses
        (one row per answer token of the 256 sequences).  SVCCA keeps at most
        256 principal directions per model.

The runs are trained by the library's default trainer (``--lambda_shape 0
--lambda_replay 0``).  Each seed sets the training data order and also the
draw of the 256 evaluation sequences of every benchmark.

Ranking
-------
Within one run and one benchmark, the 12 checkpoints are ranked by a score and
by the gap; r_s is the Spearman correlation of the two rankings (ties receive
average ranks; nan if either is constant).  Per cell the table gives the mean
of r_s over the three seeds and the SAMPLE standard deviation
(``statistics.stdev``, divisor 2).  The last row is the mean over the ten
cells of the per-cell means; the script also counts the cells in which the
bound's per-cell mean is higher than that of 1-CKA.

Validity
--------
The bound is compared with the gap at every checkpoint of every cell and seed
(12 x 10 x 3 = 360 cases); the script reports in how many the bound is at
least the gap and in how many the head term is zero.

Example
-------
    python scripts/response/lora_similarity/analyze.py --out_root out/lora_series
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List

METRICS_FILE = "prism_forgetting_metrics.json"
STEPS = list(range(25, 301, 25))
TASKS = ("truthfulqa", "bbq")
BENCHMARKS = ("arc", "gsm8k", "mmlu", "squad", "triviaqa")
LABELS = {"truthfulqa": "TruthfulQA", "bbq": "BBQ", "arc": "ARC", "gsm8k": "GSM8K",
          "mmlu": "MMLU", "squad": "SQuAD", "triviaqa": "TriviaQA"}
SCORES: Dict[str, Callable[[dict], float]] = {
    "PRISM bound": lambda rec: rec["bound_total"],
    "1-CKA": lambda rec: 1.0 - rec["cka"],
    "1-SVCCA": lambda rec: 1.0 - rec["svcca"],
}


def average_ranks(xs: List[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: List[float], y: List[float]) -> float:
    rx, ry = average_ranks(x), average_ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    den = math.sqrt(sum((p - mx) ** 2 for p in rx) * sum((q - my) ** 2 for q in ry))
    return num / den if den > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", required=True, type=Path,
                    help="directory run_series.py wrote the runs to")
    args = ap.parse_args()

    r_s: Dict[tuple, Dict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    cases = holds = head_zero = 0
    files = sorted(args.out_root.glob(f"*/{METRICS_FILE}"))
    if not files:
        raise SystemExit(f"no {METRICS_FILE} below {args.out_root}")
    for f in files:
        metrics = json.loads(f.read_text())
        task = metrics["experiment"]["trained_task"]
        seed = metrics["experiment"]["seed"]
        checkpoints = [c for c in metrics["checkpoints"] if c["step"] > 0]
        if [c["step"] for c in checkpoints] != STEPS:
            raise SystemExit(f"{f}: expected checkpoints at steps {STEPS[0]}..{STEPS[-1]} "
                             f"every 25, found {[c['step'] for c in checkpoints]}")
        for bench in BENCHMARKS:
            rows = [c["tasks"][bench] for c in checkpoints]
            gaps = [abs(rec["delta_risk"]) for rec in rows]
            for name, score in SCORES.items():
                r_s[(task, bench)][name][seed] = spearman([score(rec) for rec in rows], gaps)
            for rec, gap in zip(rows, gaps):
                cases += 1
                holds += rec["bound_total"] >= gap
                head_zero += rec["bound_head"] == 0

    names = list(SCORES)
    print("Spearman r_s between each score and the risk gap over the 12 checkpoints of a run;")
    print("mean ± sample sd (divisor n - 1) over the seeds of each cell.\n")
    print("| task | benchmark | seeds | " + " | ".join(names) + " |")
    print("|---|---|--:|" + "--:|" * len(names))
    cell_means: Dict[str, List[float]] = defaultdict(list)
    for task in TASKS:
        for bench in BENCHMARKS:
            cell = r_s.get((task, bench))
            if cell is None:
                continue
            seeds = sorted(cell[names[0]])
            cols = []
            for name in names:
                values = [cell[name][s] for s in seeds]
                cell_means[name].append(st.mean(values))
                cols.append(f"{st.mean(values):.3f} ± {st.stdev(values):.3f}")
            print(f"| {LABELS[task]} | {LABELS[bench]} | {len(seeds)} | " + " | ".join(cols) + " |")
    n_cells = len(cell_means[names[0]])
    print(f"| mean over the {n_cells} cells | | | "
          + " | ".join(f"{st.mean(cell_means[name]):.3f}" for name in names) + " |")
    print()
    higher = sum(b > o for b, o in zip(cell_means["PRISM bound"], cell_means["1-CKA"]))
    print(f"PRISM bound higher than 1-CKA in {higher} of {n_cells} cells (per-cell means)")
    print(f"\nbound >= gap in {holds} of {cases} (checkpoint, cell, seed) cases; "
          f"head term zero in {head_zero} of {cases}")


if __name__ == "__main__":
    main()
