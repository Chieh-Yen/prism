#!/usr/bin/env python3
"""Launcher for the LoRA checkpoint series behind the similarity comparison.

Six unregularized LoRA fine-tuning runs of Llama-3.1-8B: two fine-tuning tasks
(TruthfulQA, BBQ) x three seeds (42, 43, 44).  Each run is 300 optimizer steps
at learning rate 1e-5 (LoRA r = 32, alpha = 64, dropout 0.05, batch size 2 x 8
accumulation steps) and is evaluated every 25 steps, which gives 12
checkpoints per run.  At every checkpoint the trainer records, for each of the
five downstream benchmarks (ARC, GSM8K, MMLU, SQuAD, TriviaQA; 256 evaluation
sequences each), the PRISM terms and the risk gap; with
``PRISM_EVAL_SIMILARITY=1`` it also records linear CKA and SVCCA computed on
the same features.  Two tasks x five benchmarks are the ten cells that
``analyze.py`` ranks.

"Unregularized" here means ``--lambda_shape 0 --lambda_replay 0``: no penalty
and no anchor set, trained with the library's default trainer.

The launcher passes ``--eval_keep_anchor_rows`` (``eval_loss`` on the whole
held-out slice; not read by ``analyze.py``).

Runs are written to ``<out_root>/<task>_seed<seed>/`` and run one after
another on the visible GPU.  A run is skipped only when the last checkpoint
recorded in its metrics file is the final step (300); a run that was
interrupted earlier is started again from step 0 and its metrics file is
overwritten.

Example
-------
    python scripts/response/lora_similarity/run_series.py --out_root out/lora_series
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

SUPP_ROOT = Path(__file__).resolve().parents[3]
TRAINER = SUPP_ROOT / "train_forgetting.py"
MODEL = "meta-llama/Llama-3.1-8B"
TASKS = ("truthfulqa", "bbq")
SEEDS = (42, 43, 44)
METRICS_FILE = "prism_forgetting_metrics.json"

MAX_STEPS = 300

TRAIN_ARGS = (
    "--lambda_shape", "0",
    "--lambda_replay", "0",
    "--replay_source", "task",
    "--eval_keep_anchor_rows",
    "--max_steps", str(MAX_STEPS),
    "--save_steps", "25",
    "--lr", "1e-5",
    "--batch_size", "2",
    "--grad_accum", "8",
    "--max_length", "512",
    "--warmup_ratio", "0.05",
    "--lora_r", "32",
    "--lora_alpha", "64",
    "--lora_dropout", "0.05",
    "--prism_eval_samples", "256",
    "--prism_eval_batch_size", "4",
    "--logging_steps", "10",
)


def command(task: str, seed: int, run_dir: Path, python: str) -> List[str]:
    return [python, str(TRAINER), "--model", MODEL, "--task", task, "--seed", str(seed),
            "--output_dir", str(run_dir)] + list(TRAIN_ARGS)


def finished(run_dir: Path) -> bool:
    """True when the last checkpoint recorded in the run's metrics file is the
    final step.  The file is written from the first checkpoint on, so its
    existence alone does not mean the run finished."""
    try:
        with open(run_dir / METRICS_FILE) as f:
            checkpoints = json.load(f).get("checkpoints", [])
        return bool(checkpoints) and checkpoints[-1].get("step") == MAX_STEPS
    except (OSError, ValueError, AttributeError):
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", required=True, type=Path,
                    help="directory the runs are written below")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter used for the training script")
    ap.add_argument("--dry_run", action="store_true", help="print the commands and exit")
    args = ap.parse_args()

    out_root = args.out_root.resolve()
    env = dict(os.environ)
    env["PRISM_EVAL_SIMILARITY"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SUPP_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    failed = 0
    for task, seed in itertools.product(args.tasks, args.seeds):
        run_dir = out_root / f"{task}_seed{seed}"
        cmd = command(task, seed, run_dir, args.python)
        if args.dry_run:
            print("PRISM_EVAL_SIMILARITY=1 " + " ".join(cmd))
            continue
        if finished(run_dir):
            print(f"skip (finished): {run_dir.name}", flush=True)
            continue
        if (run_dir / METRICS_FILE).exists():
            print(f"restart (metrics file has no record of step {MAX_STEPS}): "
                  f"{run_dir.name}", flush=True)
        (out_root / "logs").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"start {run_dir.name}", flush=True)
        with open(out_root / "logs" / f"{run_dir.name}.log", "w") as log:
            rc = subprocess.call(cmd, cwd=SUPP_ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
        failed += rc != 0
        print(f"{'ok' if rc == 0 else f'FAILED (exit {rc})'} {run_dir.name} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    if failed:
        raise SystemExit(f"{failed} runs failed; see {out_root / 'logs'}")


if __name__ == "__main__":
    main()
