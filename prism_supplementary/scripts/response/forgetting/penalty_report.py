#!/usr/bin/env python3
"""Weighted penalty lambda * P at the selected lambda of each method.

P is the unweighted penalty a method adds to the loss, as the trainer logged
it at the end of training:

* shape regularizer: 1 - Omega_I between the fine-tuned and the base model's
  final-layer features on the anchor set (log key ``shape_loss``);
* LwF: KL divergence from the frozen base model's next-token distribution to
  the fine-tuned model's on the anchor set
  (log key ``lwf``);
* representation KD: mean squared error between the fine-tuned and the base
  model's final-layer features on the anchor set (log key ``hint_mse``);
* L2-SP: sum over the adapter weights of the squared difference from their
  initial values (log key ``l2sp``);
* EWC: the same sum with every squared difference multiplied by the Fisher
  diagonal, which is estimated on the anchor set at the adapter
  initialisation and normalised to unit mean (log key ``ewc``).

The trainers evaluate the penalty once per optimizer step and log its mean
over each window of 10 optimizer steps.  P of a run is the last such entry in
``checkpoint-300/trainer_state.json``, i.e. the mean penalty over optimizer
steps 291-300.  The reported value is lambda times the mean of P over the
three seeds, at the lambda ``select_lambda.py`` selects.  L2-SP and EWC are
listed in a second table, printed when runs of these methods exist.

The cross-entropy penalties of anchor-CE and generic replay are not listed:
they are not zero for the base model, so lambda * P does not measure a
departure from it.  The trainer logs them under the key ``replay_ce`` in the
same ``trainer_state.json``.

Example
-------
    python scripts/response/forgetting/penalty_report.py --out_root out/unified
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Dict, Optional, Tuple

import protocol as P
import select_lambda as S

METHODS = ("lwf", "hint_mse", "shape")
APPENDIX_METHODS = ("l2sp", "ewc")


def last_logged_penalty(run_path: Path, key: str) -> Optional[float]:
    state = run_path / f"checkpoint-{P.MAX_STEPS}" / "trainer_state.json"
    if not state.exists():
        return None
    entries = [e for e in json.loads(state.read_text())["log_history"] if key in e]
    return entries[-1][key] if entries else None


def run_paths(out_root: Path) -> Dict[Tuple[str, str, str, float, int], Tuple[Path, str]]:
    """(model, task, analysed method, lambda, seed) -> (run directory, penalty log key).

    Uses the precedence of ``select_lambda.load_points``: a train_forgetting.py
    run wins over a baselines-script run of the same shape-regularizer point.
    """
    paths: Dict[Tuple[str, str, str, float, int], Tuple[Path, str]] = {}
    runs = sorted(P.iter_runs(out_root), key=lambda r: P.METHODS[r.method].trainer != "main")
    for run in runs:
        spec = P.METHODS[run.method]
        key = (run.model, run.task, spec.report_as, run.lam, run.seed)
        paths.setdefault(key, (run.path, spec.penalty_key))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", required=True, type=Path,
                    help="directory holding the runs and target_loss_disjoint_*.json")
    ap.add_argument("--budget", type=float, default=0.01,
                    help="tolerance in nats of the selection rule (default 0.01)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(P.SEEDS))
    args = ap.parse_args()

    points, _ = S.load_points(args.out_root)
    paths = run_paths(args.out_root)

    def table(methods) -> None:
        print("| model | task | method | selected lambda | P | lambda * P |")
        print("|---|---|---|--:|--:|--:|")
        for cell in P.CELLS:
            ref = S.reference(points, cell, args.seeds)
            if ref is None:
                continue
            budget = S.mean_target(ref) + args.budget
            for method in methods:
                chosen = S.select(points, cell, method, budget, args.seeds)
                if chosen is None:
                    continue
                lam = chosen[0]
                values = [last_logged_penalty(*paths[cell + (method, lam, seed)])
                          for seed in args.seeds]
                if any(v is None for v in values):
                    cols = "trainer_state.json missing | "
                else:
                    cols = f"{st.mean(values):.3g} | {lam * st.mean(values):.2f}"
                print(f"| {S.cell_name(cell)} | {P.METHODS[method].label} | {lam:g} | {cols} |")

    print(f"# Weighted penalty at the selected lambda (budget +{args.budget:g} nats)")
    print("P = unweighted penalty, mean over optimizer steps 291-300, averaged over the seeds "
          f"{', '.join(map(str, args.seeds))}\n")
    table(METHODS)
    if any(points.get(cell + (m,)) for cell in P.CELLS for m in APPENDIX_METHODS):
        print("\n## Appendix I baselines (L2-SP, EWC)\n")
        table(APPENDIX_METHODS)


if __name__ == "__main__":
    main()
