#!/usr/bin/env python3
"""Table of the single-seed lambda sweeps of run_forgetting.sh (anchor-CE and
shape regularizer). No GPU.

Reads the prism_forgetting_metrics.json files that run_forgetting.sh (through
run_forgetting_one.sh and train_forgetting.py) writes,

    <root>/forgetting_replay_lam<lambda>/<model>/<task>/prism_forgetting_metrics.json
    <root>/forgetting_shape_lam<lambda>/<model>/<task>/prism_forgetting_metrics.json
    <root>/forgetting/<model>/<task>/prism_forgetting_metrics.json

(<root> is ./checkpoints by default), and prints, per model and fine-tuning
task, one line per method and lambda with two quantities, both taken from
the record of the final step (the run's `max_steps`, 300 in the sweeps; or
--step):

  target loss       `eval_loss` of the checkpoint record at the final step:
                    the Trainer's evaluation loss, i.e. the answer-only
                    cross-entropy of the fine-tuned model on held-out rows of
                    the fine-tuning task, as aggregated by the library.
                    When the sweep is run through run_forgetting.sh, the
                    trainer is started with --eval_keep_anchor_rows and this
                    is the loss on the whole held-out slice (the last 20% of
                    the task's data), which includes the 32 anchor rows the
                    regularizers read; the column "eval rows" shows the number
                    of rows each run was evaluated on when the file records it.

  downstream |dR|   unweighted mean, over the five downstream benchmarks
                    (ARC, MMLU, SQuAD, TriviaQA, GSM8K), of `delta_risk` in
                    the checkpoint record at the final step. `delta_risk` of
                    a benchmark is |R(theta_t) - R(theta_0)|, the absolute
                    difference between the fine-tuned and the base model in
                    answer-only cross-entropy on that benchmark's evaluation
                    sequences. Every task of the record other than the
                    fine-tuning task enters the mean.

The method and lambda of a run are read from the `experiment` block of its
file (`lambda_shape`, `lambda_replay`), not from the directory name:
"none" is a run with both at 0 (the library's default Trainer), "anchor-CE" a
run with lambda_replay > 0 (cross-entropy on the anchor set), "shape" a run
with lambda_shape > 0. The values
are single runs (one seed per configuration).

Usage (from the supplementary root):
    python scripts/sweep_table.py                       # ./checkpoints
    python scripts/sweep_table.py --root DIR [--step 300] [--digits 4]

Standard library only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

METRICS_FILE = "prism_forgetting_metrics.json"
METHOD_ORDER = {"none": 0, "anchor-CE": 1, "shape": 2}


def read_run(path, step):
    """One table row from a metrics file, or a (row, problem) pair."""
    with open(path) as f:
        d = json.load(f)
    exp = d.get("experiment", {})
    lam_shape = float(exp.get("lambda_shape") or 0.0)
    lam_replay = float(exp.get("lambda_replay") or 0.0)
    if lam_shape > 0:
        method, lam = "shape", lam_shape
    elif lam_replay > 0:
        method, lam = "anchor-CE", lam_replay
    else:
        method, lam = "none", 0.0
    task = exp.get("trained_task", "?")
    row = {
        "model": exp.get("model", "?"), "task": task, "method": method,
        "lam": lam, "seed": exp.get("seed"),
        "n_eval_rows": exp.get("n_eval_rows"),
        "dir": os.path.dirname(path),
    }
    final = step if step is not None else exp.get("max_steps")
    row["step"] = final
    checkpoints = d.get("checkpoints", [])
    record = next((c for c in checkpoints if c.get("step") == final), None)
    if record is None:
        last = max((c.get("step", 0) for c in checkpoints), default=None)
        row["problem"] = f"no record at step {final} (last recorded step: {last})"
        return row
    benchmarks = [b for b in record.get("tasks", {}) if b != task]
    gaps = [record["tasks"][b]["delta_risk"] for b in benchmarks]
    row["target_loss"] = record.get("eval_loss")
    row["dR"] = sum(gaps) / len(gaps) if gaps else None
    row["n_bench"] = len(benchmarks)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="checkpoints",
                    help="directory that holds the forgetting*/ run directories "
                         "(default: ./checkpoints, where run_forgetting.sh writes)")
    ap.add_argument("--step", type=int, default=None,
                    help="step of the record to read (default: each run's max_steps)")
    ap.add_argument("--digits", type=int, default=4,
                    help="decimals printed (default: 4)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "*", "*", "*", METRICS_FILE)))
    if not files:
        raise SystemExit(f"no {METRICS_FILE} found under {args.root}/<run>/<model>/<task>/")

    cells = defaultdict(list)
    for path in files:
        try:
            row = read_run(path, args.step)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"unreadable: {path} ({type(exc).__name__}: {exc})")
            continue
        cells[(row["model"], row["task"])].append(row)

    nd = args.digits
    for (model, task) in sorted(cells):
        rows = sorted(cells[(model, task)],
                      key=lambda r: (METHOD_ORDER.get(r["method"], 3), r["method"],
                                     r["lam"], r["dir"]))
        print(f"\n{model} / {task}")
        print(f"  {'method':<16}{'lambda':<10}{'target loss':<13}{'downstream |dR|':<17}"
              f"{'step':<6}{'benchmarks':<12}{'eval rows':<11}run directory")
        for r in rows:
            head = f"  {r['method']:<16}{r['lam']:<10g}"
            if "problem" in r:
                print(f"{head}{r['problem']}   {r['dir']}")
                continue
            tl = f"{r['target_loss']:.{nd}f}" if r["target_loss"] is not None else "-"
            dr = f"{r['dR']:.{nd}f}" if r["dR"] is not None else "-"
            rows_s = str(r["n_eval_rows"]) if r["n_eval_rows"] is not None else "-"
            print(f"{head}{tl:<13}{dr:<17}{r['step']:<6}{r['n_bench']:<12}"
                  f"{rows_s:<11}{r['dir']}")

    print("\ntarget loss = eval_loss at the final step; downstream |dR| = unweighted "
          "mean of delta_risk over the benchmarks at that step.\n"
          "eval rows = number of rows eval_loss was computed on, when the file "
          "records it ('-' otherwise).")


if __name__ == "__main__":
    main()
