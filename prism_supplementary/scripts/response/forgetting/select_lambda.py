#!/usr/bin/env python3
"""Selection rule of the unified forgetting protocol, and its tables.

Quantities (per run, at step 300)
---------------------------------
forgetting    mean over the five downstream benchmarks of |delta_risk|, the
              absolute change of the answer-only cross-entropy from the base
              model to the fine-tuned model on 256 evaluation sequences per
              benchmark (``prism_forgetting_metrics.json``).
target loss   answer-only cross-entropy on the fine-tuning task's evaluation
              rows outside the run's 32 anchor rows
              (``target_loss`` of ``eval_target_disjoint.py``).  A run without
              this record is not used.

Rule (per (model, task) cell; the same for every method)
--------------------------------------------------------
reference   no regularization trained in the same loop (``noreg_sameloop``),
            three seeds.
budget      mean target loss of the reference over its three seeds, plus
            ``--budget`` nats (default 0.01).
selection   among the grid values of lambda at which the method has all three
            seeds, the largest one whose mean target loss over the three seeds
            is within the budget.  Forgetting is not read by the selection.

The main table, the least-forgetting table and the budget-sensitivity table
cover no regularization, anchor-CE, generic replay and the shape regularizer.
LwF and representation KD have a table of their own, and so have L2-SP and
EWC, the weight-space baselines of Appendix I ("Appendix I baselines" below),
which are selected by the same rule on the same grid.  Layer freezing has no
weight to select; its runs are listed in a last table next to the reference.

Only lambda values on the grid {1, 5} x 10^k (1e-4 .. 1e4) are considered.
Runs of the baselines script's ``trace`` method are the shape regularizer and
are analysed as ``shape``; where ``train_forgetting.py`` ran the same
(cell, lambda, seed), that run is used.

Reported spread: "± sd" is the POPULATION standard deviation over the three
seeds (``statistics.pstdev``, divisor 3), not the sample standard deviation.

Search (``--emit``)
-------------------
``--emit`` prints the runs the search needs next, one ``method model task
lambda seed`` per line (the job format of ``run_unified.py --jobs``), and
nothing once the search has finished.  For every cell whose reference has all
three seeds, and every method that has at least one run on the grid:

1. with the runs available so far, a grid point counts as within budget if the
   mean target loss over its available seeds is within the budget;
2. if no point is within budget, request seed 42 at the grid value just below
   the lowest one run;
3. otherwise let b be the largest point within budget.  If the grid value just
   above b has not been run, request it at seed 42;
4. otherwise request the missing seeds of 42/43/44 at b.  For every method
   other than the shape regularizer, also request them at the within-budget
   point with the least mean forgetting; these runs do not enter the selection
   and serve only the "least-forgetting lambda" comparison below.

The search therefore only adds the grid value adjacent to the current
boundary and stops when that value is out of budget (or the boundary is the
top of the grid).  The per-lambda listing printed for every method shows which
points exist and with how many seeds.

Example
-------
    python scripts/response/forgetting/select_lambda.py --out_root out/unified
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import protocol as P

Cell = Tuple[str, str]
SeedValues = Dict[int, Tuple[float, float]]            # seed -> (target loss, forgetting)
Points = Dict[Tuple[str, str, str], Dict[float, SeedValues]]   # (model, task, method) -> lambda -> seeds

MAIN_METHODS = ("anchor_ce", "generic_replay", "shape")
EXTRA_BASELINES = ("lwf", "hint_mse")
APPENDIX_BASELINES = ("l2sp", "ewc")
SEARCHED = MAIN_METHODS + EXTRA_BASELINES + APPENDIX_BASELINES


def load_points(out_root: Path) -> Tuple[Points, Dict[str, int]]:
    """Collect (target loss, forgetting) of every usable run below ``out_root``."""
    rest = {}
    for f in sorted(Path(out_root).glob("target_loss_disjoint_*.json")):
        for r in json.loads(f.read_text()):
            rest[(r["method"], float(r["lam"]), int(r["seed"]), r["model"], r["task"])] = r["target_loss"]
    points: Points = defaultdict(lambda: defaultdict(dict))
    skipped = dict(off_grid=0, no_target_loss=0, no_last_step=0, duplicate_shape=0)
    runs = sorted(P.iter_runs(out_root), key=lambda r: P.METHODS[r.method].trainer != "main")
    for run in runs:
        no_weight = run.method in P.NO_WEIGHT
        if not no_weight and not P.on_grid(run.lam):
            skipped["off_grid"] += 1
            continue
        key = (run.method, run.lam, run.seed, run.model, run.task)
        if key not in rest:
            skipped["no_target_loss"] += 1
            continue
        forgetting = P.read_forgetting(run.path)
        if forgetting is None:
            skipped["no_last_step"] += 1
            continue
        method = P.METHODS[run.method].report_as
        seeds = points[(run.model, run.task, method)][0.0 if no_weight else run.lam]
        if run.seed in seeds:
            skipped["duplicate_shape"] += 1
            continue
        seeds[run.seed] = (rest[key], forgetting)
    return points, skipped


def mean_target(values: SeedValues) -> float:
    return st.mean(v[0] for v in values.values())


def mean_forgetting(values: SeedValues) -> float:
    return st.mean(v[1] for v in values.values())


def complete(values: SeedValues, seeds: Sequence[int]) -> Optional[SeedValues]:
    """The values restricted to ``seeds``, or None if one of them is missing."""
    if any(s not in values for s in seeds):
        return None
    return {s: values[s] for s in seeds}


def reference(points: Points, cell: Cell, seeds: Sequence[int]) -> Optional[SeedValues]:
    return complete(points.get(cell + (P.REFERENCE,), {}).get(0.0, {}), seeds)


def within_budget(points: Points, cell: Cell, method: str, budget: float,
                  seeds: Sequence[int]) -> Dict[float, SeedValues]:
    """Grid points with all seeds whose mean target loss is within the budget."""
    out = {}
    for lam, values in points.get(cell + (method,), {}).items():
        full = complete(values, seeds)
        if full is not None and mean_target(full) <= budget:
            out[lam] = full
    return out


def select(points: Points, cell: Cell, method: str, budget: float,
           seeds: Sequence[int]) -> Optional[Tuple[float, SeedValues]]:
    """The rule: largest lambda with all seeds and mean target loss within budget."""
    ok = within_budget(points, cell, method, budget, seeds)
    if not ok:
        return None
    lam = max(ok)
    return lam, ok[lam]


def least_forgetting(points: Points, cell: Cell, method: str, budget: float,
                     seeds: Sequence[int]) -> Optional[Tuple[float, SeedValues]]:
    """Within-budget point (all seeds) with the least mean forgetting."""
    ok = within_budget(points, cell, method, budget, seeds)
    if not ok:
        return None
    lam = min(ok, key=lambda l: mean_forgetting(ok[l]))
    return lam, ok[lam]


def next_jobs(points: Points, tol: float, seeds: Sequence[int]) -> List[str]:
    """Runs the search needs next (see the module docstring, 'Search')."""
    jobs = []
    for cell in P.CELLS:
        ref = reference(points, cell, seeds)
        if ref is None:
            continue
        budget = mean_target(ref) + tol
        for method in SEARCHED:
            pts = points.get(cell + (method,), {})
            if not pts:
                continue
            line = f"{method} {cell[0]} {cell[1]}"
            inside = [lam for lam, values in pts.items() if mean_target(values) <= budget]
            if not inside:
                below = [g for g in P.GRID if g < min(pts) - 1e-12]
                if below:
                    jobs.append(f"{line} {P.lam_tag(below[-1])} {seeds[0]}")
                continue
            b = max(inside)
            above = [g for g in P.GRID if g > b + 1e-12]
            if above and not any(abs(lam - above[0]) < 1e-9 for lam in pts):
                jobs.append(f"{line} {P.lam_tag(above[0])} {seeds[0]}")
                continue
            wanted = {b}
            if method != "shape":
                wanted.add(min(inside, key=lambda lam: mean_forgetting(pts[lam])))
            jobs += [f"{line} {P.lam_tag(lam)} {s}"
                     for lam in sorted(wanted) for s in seeds if s not in pts[lam]]
    return jobs


def fmt(values: SeedValues) -> str:
    forgetting = [v[1] for v in values.values()]
    return f"{st.mean(forgetting):.3f} ± {st.pstdev(forgetting):.3f} ({mean_target(values):.3f})"


def seeds_lower(a: SeedValues, b: SeedValues) -> str:
    """In how many seeds run ``a`` forgets less than run ``b``."""
    return f"{sum(a[s][1] < b[s][1] for s in a)}/{len(a)}"


def cell_name(cell: Cell) -> str:
    return f"{P.MODEL_LABELS[cell[0]]} | {P.TASK_LABELS[cell[1]]}"


def print_listing(points: Points, tol: float, seeds: Sequence[int]) -> None:
    print("## Runs and selection per cell")
    print("Per lambda: [number of seeds] mean target loss / mean forgetting; "
          "'x' marks a mean target loss above the budget.\n")
    for cell in P.CELLS:
        ref = reference(points, cell, seeds)
        if ref is None:
            print(f"{cell_name(cell)}: reference incomplete, cell skipped\n")
            continue
        budget = mean_target(ref) + tol
        print(f"{cell_name(cell)}: reference {fmt(ref)}, budget {budget:.4f}")
        for method in SEARCHED:
            pts = points.get(cell + (method,), {})
            if not pts:
                continue
            chosen = select(points, cell, method, budget, seeds)
            head = (f"selected lambda {chosen[0]:g}: {fmt(chosen[1])}" if chosen
                    else "no lambda with all seeds within budget")
            print(f"  {method:15s} {head}")
            print("      " + "  ".join(
                f"{lam:g}[{len(v)}]:{mean_target(v):.4f}/{mean_forgetting(v):.3f}"
                f"{'' if mean_target(v) <= budget else 'x'}"
                for lam, v in sorted(pts.items())))
        print()


def print_main_table(points: Points, tol: float, seeds: Sequence[int]) -> None:
    print(f"## Main table (budget +{tol:g} nats)")
    print("Entries: forgetting, mean ± population sd over the seeds "
          f"{', '.join(map(str, seeds))}, with the mean target loss in parentheses.\n")
    print("| model | task | no regularization | anchor-CE | generic replay | shape regularizer "
          "| shape vs. no regularization | shape target loss minus reference |")
    print("|---|---|--:|--:|--:|--:|--:|--:|")
    notes = []
    for cell in P.CELLS:
        ref = reference(points, cell, seeds)
        if ref is None:
            continue
        budget = mean_target(ref) + tol
        chosen = {m: select(points, cell, m, budget, seeds) for m in MAIN_METHODS}
        cols = [fmt(ref)] + [f"{fmt(chosen[m][1])} [lambda {chosen[m][0]:g}]" if chosen[m] else "none"
                             for m in MAIN_METHODS]
        shape = chosen["shape"]
        if shape:
            change = 100 * (mean_forgetting(shape[1]) / mean_forgetting(ref) - 1)
            cost = mean_target(shape[1]) - mean_target(ref)
            cols += [f"{change:+.0f}%", f"{cost:+.3f}"]
            checks = [f"no regularization {seeds_lower(shape[1], ref)}"]
            checks += [f"{P.METHODS[m].label} {seeds_lower(shape[1], chosen[m][1])}"
                       for m in ("anchor_ce", "generic_replay") if chosen[m]]
            notes.append(f"{cell_name(cell)}: " + ", ".join(checks))
        else:
            cols += ["", ""]
        print(f"| {cell_name(cell)} | " + " | ".join(cols) + " |")
    print("\nSeeds in which the shape regularizer forgets less than:")
    for note in notes:
        print(f"  {note}")
    print()


def print_least_forgetting(points: Points, tol: float, seeds: Sequence[int]) -> None:
    print(f"## Each baseline at its least-forgetting lambda within budget (+{tol:g} nats)")
    print("Among the grid points with all seeds and mean target loss within budget, the one "
          "with the least mean forgetting; the shape regularizer stays at its selected lambda.\n")
    print("| model | task | anchor-CE | generic replay | shape regularizer (selected) |")
    print("|---|---|--:|--:|--:|")
    for cell in P.CELLS:
        ref = reference(points, cell, seeds)
        if ref is None:
            continue
        budget = mean_target(ref) + tol
        cols = []
        for method in ("anchor_ce", "generic_replay"):
            best = least_forgetting(points, cell, method, budget, seeds)
            cols.append(f"{mean_forgetting(best[1]):.3f} [lambda {best[0]:g}]" if best else "none")
        shape = select(points, cell, "shape", budget, seeds)
        cols.append(f"{mean_forgetting(shape[1]):.3f} [lambda {shape[0]:g}]" if shape else "none")
        print(f"| {cell_name(cell)} | " + " | ".join(cols) + " |")
    print()


def print_sensitivity(points: Points, tols: Sequence[float], seeds: Sequence[int]) -> None:
    print("## Budget sensitivity")
    print("Selection repeated under other budgets. Entries: mean forgetting [selected lambda]; "
          "last column: whether the shape regularizer's mean forgetting is below that of "
          "no regularization, anchor-CE and generic replay.\n")
    print("| budget | model | task | no regularization | anchor-CE | generic replay "
          "| shape regularizer | shape lowest |")
    print("|--:|---|---|--:|--:|--:|--:|---|")
    for tol in tols:
        for cell in P.CELLS:
            ref = reference(points, cell, seeds)
            if ref is None:
                continue
            budget = mean_target(ref) + tol
            chosen = {m: select(points, cell, m, budget, seeds) for m in MAIN_METHODS}
            cols = [f"{mean_forgetting(ref):.3f}"]
            cols += [f"{mean_forgetting(chosen[m][1]):.3f} [{chosen[m][0]:g}]" if chosen[m] else "none"
                     for m in MAIN_METHODS]
            if all(chosen.values()):
                shape = mean_forgetting(chosen["shape"][1])
                others = [mean_forgetting(ref)] + [mean_forgetting(chosen[m][1])
                                                   for m in ("anchor_ce", "generic_replay")]
                cols.append("yes" if all(shape < o for o in others) else "NO")
            else:
                cols.append("n/a")
            print(f"| +{tol:g} | {cell_name(cell)} | " + " | ".join(cols) + " |")
    print()


def print_extra_baselines(points: Points, tol: float, seeds: Sequence[int]) -> None:
    rows = []
    for cell in P.CELLS:
        ref = reference(points, cell, seeds)
        if ref is None or not any(points.get(cell + (m,)) for m in EXTRA_BASELINES):
            continue
        budget = mean_target(ref) + tol
        for method in EXTRA_BASELINES + ("shape",):
            chosen = select(points, cell, method, budget, seeds)
            if chosen is None:
                continue
            forgetting = [v[1] for v in chosen[1].values()]
            rows.append(f"| {cell_name(cell)} | {P.METHODS[method].label} | {chosen[0]:g} | "
                        f"{st.mean(forgetting):.3f} ± {st.pstdev(forgetting):.3f} | "
                        f"{mean_target(chosen[1]):.3f} |")
    if not rows:
        return
    print(f"## LwF and representation KD under the same rule (budget +{tol:g} nats)")
    print("Forgetting: mean ± population sd over the seeds.\n")
    print("| model | task | method | selected lambda | forgetting | target loss |")
    print("|---|---|---|--:|--:|--:|")
    print("\n".join(rows))
    print()


def print_appendix_baselines(points: Points, tol: float, seeds: Sequence[int]) -> None:
    """L2-SP and EWC under the rule, next to the reference and the shape regularizer."""
    if not any(points.get(cell + (m,)) for cell in P.CELLS for m in APPENDIX_BASELINES):
        return
    methods = APPENDIX_BASELINES + ("shape",)
    print(f"## Appendix I baselines (L2-SP, EWC) under the same rule (budget +{tol:g} nats)")
    print("Forgetting: mean ± population sd over the seeds; 'seeds' is the number of seeds "
          "behind the entry.\n")
    print("| model | task | method | selected lambda | forgetting | target loss | seeds |")
    print("|---|---|---|--:|--:|--:|--:|")
    notes = []
    for cell in P.CELLS:
        ref = reference(points, cell, seeds)
        if ref is None:
            continue
        budget = mean_target(ref) + tol
        forgetting = [v[1] for v in ref.values()]
        print(f"| {cell_name(cell)} | {P.METHODS[P.REFERENCE].label} | | "
              f"{st.mean(forgetting):.3f} ± {st.pstdev(forgetting):.3f} | "
              f"{mean_target(ref):.3f} | {len(ref)} |")
        chosen = {m: select(points, cell, m, budget, seeds) for m in methods}
        for method in methods:
            if chosen[method] is None:
                print(f"| {cell_name(cell)} | {P.METHODS[method].label} | none | | | |")
                continue
            lam, values = chosen[method]
            forgetting = [v[1] for v in values.values()]
            print(f"| {cell_name(cell)} | {P.METHODS[method].label} | {lam:g} | "
                  f"{st.mean(forgetting):.3f} ± {st.pstdev(forgetting):.3f} | "
                  f"{mean_target(values):.3f} | {len(values)} |")
        if chosen["shape"]:
            checks = [f"{P.METHODS[m].label} {seeds_lower(chosen['shape'][1], chosen[m][1])}"
                      for m in APPENDIX_BASELINES if chosen[m]]
            notes.append(f"{cell_name(cell)}: " + ", ".join(checks))
    print("\nSeeds in which the shape regularizer forgets less than:")
    for note in notes:
        print(f"  {note}")

    print()


def print_layer_freezing(points: Points, seeds: Sequence[int]) -> None:
    """Layer freezing (no weight to select) next to the reference."""
    if not any(points.get(cell + ("layer_freeze",)) for cell in P.CELLS):
        return
    print("## Layer freezing (no weight to select)")
    print("Forgetting: mean ± population sd over the seeds; last column: mean target loss "
          "minus that of the reference.\n")
    print("| model | task | forgetting | target loss | target loss minus reference | seeds |")
    print("|---|---|--:|--:|--:|--:|")
    for cell in P.CELLS:
        ref = reference(points, cell, seeds)
        values = points.get(cell + ("layer_freeze",), {}).get(0.0)
        if ref is None or not values:
            continue
        forgetting = [v[1] for v in values.values()]
        print(f"| {cell_name(cell)} | {st.mean(forgetting):.3f} ± {st.pstdev(forgetting):.3f} | "
              f"{mean_target(values):.3f} | {mean_target(values) - mean_target(ref):+.3f} | "
              f"{len(values)} |")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", required=True, type=Path,
                    help="directory holding the runs and target_loss_disjoint_*.json")
    ap.add_argument("--budget", type=float, default=0.01,
                    help="tolerance in nats added to the reference's target loss (default 0.01)")
    ap.add_argument("--sensitivity", type=float, nargs="*", default=[0.005, 0.02],
                    help="other tolerances for the budget-sensitivity table (default 0.005 0.02)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(P.SEEDS))
    ap.add_argument("--emit", action="store_true",
                    help="print the runs the search needs next and exit")
    args = ap.parse_args()

    points, skipped = load_points(args.out_root)
    if args.emit:
        jobs = next_jobs(points, args.budget, args.seeds)
        if jobs:
            print("\n".join(jobs))
        return

    print("# Unified forgetting protocol")
    print(f"runs used: {sum(len(v) for lams in points.values() for v in lams.values())}; "
          f"not used: {skipped['off_grid']} at a lambda off the grid, "
          f"{skipped['no_target_loss']} without a disjoint target loss, "
          f"{skipped['no_last_step']} without a step-{P.MAX_STEPS} record, "
          f"{skipped['duplicate_shape']} baselines-script shape runs duplicating a "
          f"train_forgetting.py run")
    print("forgetting = mean |delta_risk| over five benchmarks; target loss = answer-only CE on "
          "the evaluation rows outside the anchor set; sd = population sd (divisor = number of "
          "seeds)")
    pending = next_jobs(points, args.budget, args.seeds)
    print(f"search status at +{args.budget:g} nats: "
          + ("finished (no further runs requested)" if not pending
             else f"{len(pending)} further runs requested, see --emit"))
    print()
    print_listing(points, args.budget, args.seeds)
    print_main_table(points, args.budget, args.seeds)
    print_least_forgetting(points, args.budget, args.seeds)
    print_sensitivity(points, [args.budget] + list(args.sensitivity), args.seeds)
    print_extra_baselines(points, args.budget, args.seeds)
    print_appendix_baselines(points, args.budget, args.seeds)
    print_layer_freezing(points, args.seeds)
    if pending:
        sys.exit("search not finished: the tables above are provisional")


if __name__ == "__main__":
    main()
