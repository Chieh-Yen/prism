#!/usr/bin/env python3
"""Launcher for the unified forgetting protocol.

One training run per (method, model, task, lambda, seed).  Every run is LoRA
fine-tuning (r = 32, alpha = 64, dropout 0.05) for 300 optimizer steps at
learning rate 1e-5 with batch size 2 x 8 accumulation steps, a 32-sequence
anchor set revisited every 8 micro-steps, one checkpoint and one evaluation at
step 300 (``--save_steps 25`` evaluates and checkpoints every 25 steps instead,
which gives the trajectory of a run); the full argument list is
``protocol.COMMON_ARGS``.  Runs differ only in the method, its weight lambda
and the seed.

Methods
-------
noreg_sameloop   no regularization, trained in the same loop as the other
                 methods: ``train_forgetting.py --lambda_shape 1e-9``.  The
                 regularized trainer class is used with penalty weight 1e-9
                 (weighted penalty at most 2e-9 nats); lambda is fixed and
                 need not be given.
shape            shape regularizer: ``train_forgetting.py --lambda_shape L``
anchor_ce        cross-entropy on the anchor set:
                 ``train_forgetting.py --lambda_replay L --replay_source task``
generic_replay   cross-entropy on 32 WikiText sequences:
                 ``train_forgetting.py --lambda_replay L --replay_source generic``
lwf              ``train_forgetting_baselines.py --method lwf --lambda_reg L``
hint_mse         ``train_forgetting_baselines.py --method hint_mse --lambda_reg L``
trace            the shape regularizer as implemented in the baselines script:
                 ``train_forgetting_baselines.py --method trace --lambda_reg L``
l2sp             L2-SP: ``train_forgetting_baselines.py --method l2sp --lambda_reg L``
ewc              EWC: ``train_forgetting_baselines.py --method ewc --lambda_reg L``
layer_freeze     layer freezing, LoRA adapters on the top 16 layers only, no
                 penalty, trained in the same loop: ``train_forgetting_baselines.py
                 --method layer_freeze --lora_top_layers 16 --same_loop``.  It
                 has no weight; lambda need not be given and the runs are
                 stored under the reference weight 1e-9.

Jobs are given either as a product (``--methods --models --tasks --lambdas
--seeds``) or as a job file (``--jobs``; ``-`` reads standard input) with one
``method model task lambda seed`` per line, the format ``select_lambda.py
--emit`` prints.  Jobs run one after another on the visible GPU; a job whose
metrics file exists is skipped, so the launcher can be re-run.  To use several
GPUs, start one launcher per GPU with ``CUDA_VISIBLE_DEVICES`` set and disjoint
job lists.

Example
-------
    python scripts/response/forgetting/run_unified.py --out_root out/unified \\
        --methods shape --models llama --tasks truthfulqa \\
        --lambdas 500 1000 --seeds 42
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, NamedTuple, Optional

import protocol as P


class Job(NamedTuple):
    method: str
    model: str
    task: str
    lam: float
    seed: int


def make_job(method: str, model: str, task: str, lam: float, seed: int) -> Job:
    if method not in P.METHODS:
        raise SystemExit(f"unknown method {method!r}; choose from {list(P.METHODS)}")
    if model not in P.MODELS:
        raise SystemExit(f"unknown model {model!r}; choose from {list(P.MODELS)}")
    if task not in P.TASKS:
        raise SystemExit(f"unknown task {task!r}; choose from {list(P.TASKS)}")
    if method in P.NO_WEIGHT:
        lam = P.REFERENCE_LAMBDA
    elif not P.on_grid(lam):
        raise SystemExit(f"lambda {lam:g} is not on the grid {{1, 5}} x 10^k, 1e-4 .. 1e4")
    return Job(method, model, task, float(lam), int(seed))


def read_jobs(path: str) -> List[Job]:
    lines = sys.stdin.read().splitlines() if path == "-" else Path(path).read_text().splitlines()
    jobs = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        method, model, task, lam, seed = line.split()
        jobs.append(make_job(method, model, task, float(lam), int(seed)))
    return jobs


def command(job: Job, out_root: Path, python: str,
            save_steps: int = P.MAX_STEPS,
            save_total_limit: Optional[int] = None) -> List[str]:
    spec = P.METHODS[job.method]
    flags = [f.format(lam=P.lam_tag(job.lam)) for f in spec.flags]
    head = [python, str(P.MAIN_TRAINER if spec.trainer == "main" else P.BASELINE_TRAINER),
            "--model", P.MODELS[job.model], "--task", job.task, "--seed", str(job.seed)]
    if spec.trainer == "main":
        where = ["--output_dir",
                 str(P.run_dir(out_root, job.method, job.lam, job.seed, job.model, job.task))]
    else:
        where = ["--output_root", str(out_root)]
    keep = (["--save_total_limit", str(save_total_limit)]
            if save_total_limit is not None and spec.trainer == "main" else [])
    return (head + flags + where + ["--save_steps", str(save_steps)] + keep
            + list(P.COMMON_ARGS))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", required=True, type=Path,
                    help="directory the runs are written below")
    ap.add_argument("--jobs", default=None,
                    help="job file, one 'method model task lambda seed' per line ('-' = stdin)")
    ap.add_argument("--methods", nargs="+", default=[], choices=list(P.METHODS))
    ap.add_argument("--models", nargs="+", default=list(P.MODELS), choices=list(P.MODELS))
    ap.add_argument("--tasks", nargs="+", default=list(P.TASKS), choices=list(P.TASKS))
    ap.add_argument("--lambdas", nargs="+", type=float, default=[],
                    help="grid values; ignored for noreg_sameloop and layer_freeze")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(P.SEEDS))
    ap.add_argument("--save_steps", type=int, default=P.MAX_STEPS,
                    help="steps between evaluations and checkpoints (default: "
                         "one evaluation at the last step)")
    ap.add_argument("--save_total_limit", type=int, default=None,
                    help="keep only the last N checkpoints of a train_forgetting.py "
                         "run (default: keep all)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter used for the training scripts")
    ap.add_argument("--dry_run", action="store_true", help="print the commands and exit")
    args = ap.parse_args()

    jobs: List[Job] = read_jobs(args.jobs) if args.jobs else []
    for method, model, task, seed in itertools.product(
            args.methods, args.models, args.tasks, args.seeds):
        lams = [P.REFERENCE_LAMBDA] if method in P.NO_WEIGHT else args.lambdas
        if not lams:
            raise SystemExit(f"--lambdas is required for method {method}")
        jobs += [make_job(method, model, task, lam, seed) for lam in lams]
    jobs = list(dict.fromkeys(jobs))
    if not jobs:
        raise SystemExit("no jobs: give --jobs or --methods")

    out_root = args.out_root.resolve()
    log_dir = out_root / "logs"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(P.SUPP_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"{len(jobs)} jobs -> {out_root}")
    failed = 0
    for i, job in enumerate(jobs, 1):
        cmd = command(job, out_root, args.python, args.save_steps, args.save_total_limit)
        tag = f"{job.method}_lam{P.lam_tag(job.lam)}_seed{job.seed}_{job.model}_{job.task}"
        if args.dry_run:
            print(" ".join(cmd))
            continue
        done = P.run_dir(out_root, job.method, job.lam, job.seed, job.model, job.task) / P.METRICS_FILE
        if done.exists():
            print(f"[{i}/{len(jobs)}] skip (finished): {tag}", flush=True)
            continue
        log_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"[{i}/{len(jobs)}] start {tag}", flush=True)
        with open(log_dir / f"{tag}.log", "w") as log:
            rc = subprocess.call(cmd, cwd=P.SUPP_ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
        failed += rc != 0
        print(f"[{i}/{len(jobs)}] {'ok' if rc == 0 else f'FAILED (exit {rc})'} {tag} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    if failed:
        raise SystemExit(f"{failed} of {len(jobs)} jobs failed; see {log_dir}")


if __name__ == "__main__":
    main()
