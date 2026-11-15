"""Shared definitions of the unified forgetting protocol.

Every script in this folder imports the constants below, so the launcher, the
target-loss evaluation and the selection read one description of the runs:

* which (model, task) cells, seeds and methods exist,
* the lambda grid,
* the hyper-parameters every run is launched with,
* the directory a run is written to, and how finished runs are found again.

Run directory (identical for both training scripts)::

    <out_root>/<method>/lam<lambda>/seed<seed>/<model_short>/<task>/

``<lambda>`` is formatted with ``%g`` and ``<model_short>`` is the lower-cased
last component of the Hugging Face model id.  This is the layout
``scripts/rounds/train_forgetting_baselines.py`` writes below ``--output_root``;
runs of ``train_forgetting.py`` are given the same path through
``--output_dir``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

SUPP_ROOT = Path(__file__).resolve().parents[3]
MAIN_TRAINER = SUPP_ROOT / "train_forgetting.py"
BASELINE_TRAINER = SUPP_ROOT / "scripts" / "rounds" / "train_forgetting_baselines.py"

MODELS: Dict[str, str] = {
    "llama": "meta-llama/Llama-3.1-8B",
    "qwen": "Qwen/Qwen3-8B-Base",
}
MODEL_LABELS = {"llama": "Llama-3.1-8B", "qwen": "Qwen3-8B-Base"}
TASKS = ("truthfulqa", "bbq")
TASK_LABELS = {"truthfulqa": "TruthfulQA", "bbq": "BBQ"}
CELLS: Tuple[Tuple[str, str], ...] = (
    ("llama", "truthfulqa"), ("llama", "bbq"),
    ("qwen", "truthfulqa"), ("qwen", "bbq"),
)
SEEDS = (42, 43, 44)

MAX_STEPS = 300
METRICS_FILE = "prism_forgetting_metrics.json"

# Lambda grid: {1, 5} x 10^k for k = -4 .. 3, plus the top point 10^4.
GRID: Tuple[float, ...] = tuple(sorted(
    {round(b * 10.0 ** e, 6) for e in range(-4, 4) for b in (1, 5)} | {1e4}
))

# The unregularized reference is trained by the regularized trainer class with
# the shape penalty weighted by this value, so that it shares the training loop
# of every other method (``--lambda_shape 0`` would select the library's
# default trainer instead).  The penalty 1 - Omega_I is at most 2, so the
# weighted penalty is at most 2e-9 nats.
REFERENCE = "noreg_sameloop"
REFERENCE_LAMBDA = 1e-9
# Methods without a weight to select; their runs carry REFERENCE_LAMBDA.
NO_WEIGHT = (REFERENCE, "layer_freeze")


class Method(NamedTuple):
    label: str
    trainer: str            # "main": train_forgetting.py; "baselines": train_forgetting_baselines.py
    flags: Tuple[str, ...]  # how lambda and the method are passed; "{lam}" is substituted
    penalty_key: Optional[str]  # key of the unweighted penalty in the trainer's log history
    report_as: str          # method the runs are analysed as


METHODS: Dict[str, Method] = {
    REFERENCE: Method(
        "no regularization", "main",
        ("--lambda_shape", "{lam}", "--lambda_replay", "0", "--replay_source", "task"),
        "shape_loss", REFERENCE),
    "shape": Method(
        "shape regularizer", "main",
        ("--lambda_shape", "{lam}", "--lambda_replay", "0", "--replay_source", "task"),
        "shape_loss", "shape"),
    "anchor_ce": Method(
        "anchor-CE", "main",
        ("--lambda_shape", "0", "--lambda_replay", "{lam}", "--replay_source", "task"),
        "replay_ce", "anchor_ce"),
    "generic_replay": Method(
        "generic replay", "main",
        ("--lambda_shape", "0", "--lambda_replay", "{lam}", "--replay_source", "generic"),
        "replay_ce", "generic_replay"),
    "lwf": Method(
        "LwF", "baselines", ("--method", "lwf", "--lambda_reg", "{lam}"),
        "lwf", "lwf"),
    "hint_mse": Method(
        "representation KD (hint MSE)", "baselines",
        ("--method", "hint_mse", "--lambda_reg", "{lam}"),
        "hint_mse", "hint_mse"),
    # The shape regularizer as implemented in the baselines script (same
    # penalty 1 - Omega_I, same anchor set and schedule).  Its runs are analysed
    # as "shape"; where both scripts ran the same (cell, lambda, seed), the
    # train_forgetting.py run is used.
    "trace": Method(
        "shape regularizer (baselines script)", "baselines",
        ("--method", "trace", "--lambda_reg", "{lam}"),
        "shape_loss", "shape"),
    # The weight-space baselines of Appendix I.  Their penalties are computed
    # on the adapter weights, not on the anchor set; EWC estimates its Fisher
    # diagonal on the anchor set before training.
    "l2sp": Method(
        "L2-SP", "baselines", ("--method", "l2sp", "--lambda_reg", "{lam}"),
        "l2sp", "l2sp"),
    "ewc": Method(
        "EWC", "baselines", ("--method", "ewc", "--lambda_reg", "{lam}"),
        "ewc", "ewc"),
    # Layer freezing (Appendix I): LoRA on the top 16 layers only, no penalty,
    # trained in the regularized loop.  It has no weight to select; its runs
    # are listed with the reference weight 1e-9.
    "layer_freeze": Method(
        "layer freezing (top 16)", "baselines",
        ("--method", "layer_freeze", "--lora_top_layers", "16", "--same_loop"),
        None, "layer_freeze"),
}

# Hyper-parameters passed explicitly to every run.  ``--save_steps`` is added
# by the launcher (300 for the unified protocol: one checkpoint and one
# evaluation at the last step).  Not exposed as flags by the trainers and
# therefore fixed in their code: LoRA target modules (q/k/v/o/gate/up/down
# projections), cosine schedule, weight decay 0.01, gradient clipping 1.0,
# bf16, optimizer (paged 8-bit AdamW of bitsandbytes).
COMMON_ARGS: Tuple[str, ...] = (
    "--max_steps", str(MAX_STEPS),
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
    "--reg_samples", "32",
    "--reg_every_k", "8",
    "--reg_batch_size", "8",
    "--reg_max_length", "512",
    "--logging_steps", "10",
)


def lam_tag(lam: float) -> str:
    return f"{float(lam):g}"


def model_short(model_key: str) -> str:
    return MODELS[model_key].split("/")[-1].lower()


def on_grid(lam: float) -> bool:
    return any(abs(lam - g) <= 1e-9 * max(1.0, g) for g in GRID)


def run_dir(out_root: Path, method: str, lam: float, seed: int,
            model_key: str, task: str) -> Path:
    return (Path(out_root) / method / f"lam{lam_tag(lam)}" / f"seed{seed}"
            / model_short(model_key) / task)


class Run(NamedTuple):
    method: str     # directory name, a key of METHODS
    lam: float
    seed: int
    model: str      # key of MODELS
    task: str
    path: Path


def iter_runs(out_root: Path) -> Iterator[Run]:
    """Finished runs below ``out_root`` (those with a metrics file), sorted."""
    short_to_key = {model_short(k): k for k in MODELS}
    found: List[Run] = []
    for method in METHODS:
        for f in (Path(out_root) / method).glob(f"lam*/seed*/*/*/{METRICS_FILE}"):
            lam_dir, seed_dir, short, task = f.parts[-5:-1]
            if short not in short_to_key or task not in TASKS:
                continue
            found.append(Run(method, float(lam_dir[3:]), int(seed_dir[4:]),
                             short_to_key[short], task, f.parent))
    return iter(sorted(found))


def read_forgetting(path: Path) -> Optional[float]:
    """Downstream forgetting of one run at the last step.

    Mean over the five downstream benchmarks of |delta_risk|, where delta_risk
    is the fine-tuned model's answer-only cross-entropy minus the base model's
    on the 256 evaluation sequences of that benchmark.  The fine-tuning task
    itself is excluded.  Returns None if the run has no record at the last step.
    """
    with open(Path(path) / METRICS_FILE) as fh:
        metrics = json.load(fh)
    last = [c for c in metrics["checkpoints"] if c["step"] == MAX_STEPS]
    if not last:
        return None
    trained = metrics["experiment"]["trained_task"]
    gaps = [abs(rec["delta_risk"]) for bench, rec in last[0]["tasks"].items()
            if bench != trained]
    return sum(gaps) / len(gaps)
