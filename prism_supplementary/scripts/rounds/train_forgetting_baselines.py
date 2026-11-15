#!/usr/bin/env python3
"""
LoRA forgetting runs with a set of regularizer baselines, one entry point.

Every method fine-tunes with the answer-only cross-entropy L_CE; the
regularized ones add lambda * P, where P is evaluated on the anchor set
(--reg_samples held-out sequences of the fine-tuning task; see main()):

    none          no penalty
    trace         P = 1 - Omega_I                            (shape regularizer)
    replay        P = CE on the anchor set                   (anchor-CE)
    l2sp          P = sum ||theta - theta_0||^2              (parameter space)
    ewc           P = sum F_i (theta_i - theta_0i)^2         (Fisher-weighted)
    lwf           P = forward KL(p_base || p_theta) at temperature 1, on the
                  anchor sequences, at the non-padding positions
                  t >= prompt_length (see LwFTrainer)        (output space)
    hint_mse      P = mean squared error between the final-layer answer-token
                  features of the fine-tuned and the base model (feature space;
                  the same features the shape regularizer reads)
    layer_freeze  no penalty; LoRA on the top --lora_top_layers layers only

--seed and --ref_offset expose the training seed and the anchor draw, so
multi-seed runs and anchor-set ablations go through this script as well.
--seed also sets the initial LoRA weights and the anchor draw (see its help
text).

Which trainer runs. trace, replay, l2sp, ewc, lwf and hint_mse run the
regularized training loop (subclasses of ShapeRegularizedTrainer). `none` and
`layer_freeze` have no penalty and run the library's default Trainer; with
--same_loop they run the regularized loop instead (the L2-SP trainer at
weight 1e-9), which is how the unified protocol
(scripts/response/forgetting/) runs layer freezing.

LoRA config, data, collator, PRISM checkpoint callback and the regularized
training loop are imported unchanged from train_forgetting_multitask.py.
The regularized methods (trace, replay, l2sp, ewc, lwf, hint_mse) subclass
its ShapeRegularizedTrainer and share its loss scaling under gradient
accumulation (see that class's docstring). The unregularized reference that
shares the regularized loop is `python train_forgetting.py --lambda_shape 1e-9`
(see that script; 1 - Omega_I lies in [0, 2], so the weighted penalty
contributes at most 2e-9 nats).

Target-task evaluation rows. By default the rows of the anchor draw are
removed from the Trainer's eval set, for every method, so `eval_loss` is on
held-out rows outside the anchor set. With --eval_keep_anchor_rows the whole
held-out slice is evaluated; scripts/rounds/run_regularizer_baselines.sh
passes the flag, so the target-task loss column of regularizer_table.py is
computed on the whole slice. The flag changes `eval_loss` only.

Resume. `--resume auto` continues an interrupted run from its last covered
checkpoint, except for l2sp and ewc: their reference point (the initial
adapter weights, and for ewc the Fisher estimated at them) is not stored in
the checkpoints, so an interrupted l2sp or ewc run has to be restarted.

EWC note: the Fisher diagonal is estimated on the anchor set at adapter
initialisation, i.e. at the base model itself. The estimate is the square of
mini-batch gradients (see estimate_fisher). The Fisher is normalised to unit
mean so lambda is on the same scale as l2sp.

Usage (from the supplementary root, GPU required):
    python scripts/rounds/train_forgetting_baselines.py \
        --model meta-llama/Llama-3.1-8B --task truthfulqa \
        --method l2sp --lambda_reg 1e-3 --seed 42 --max_steps 300

Outputs: {output_root}/{method}/{tag}/seed{seed}/{model_short}/{task}/
             prism_forgetting_metrics.json   (same schema as train_forgetting.py)
         {tag} is lam{lambda_reg:g} (e.g. lam0.01, lam1, lam0 for `none`), or
         top{K} for layer_freeze with --lora_top_layers K (e.g. top16).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent          # supplementary root (holds prism/)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))      # train_forgetting_multitask.py lives beside this file

from transformers import (AutoModelForCausalLM, AutoTokenizer,      # noqa: E402
                          Trainer, TrainingArguments, set_seed)

from train_forgetting_multitask import (                            # noqa: E402
    AnswerOnlyDataCollator,
    LORA_TARGET_MODULES,
    PRISMCheckpointCallback,
    ReplayCETrainer,
    ShapeRegularizedTrainer,
    TASK_CONFIGS,
    build_dataset,
    get_eval_tasks,
    pre_compute_base_features,
)
from prism.data.loaders import get_task_metadata, load_task_data    # noqa: E402
from prism.experiments.quantization import pinned_revision          # noqa: E402
from prism.models.extractors import LLMExtractor                    # noqa: E402

# Remote code is opt-in (PRISM_TRUST_REMOTE_CODE=1). Every model family used in
# the paper is native to the pinned transformers release, so the default is off.
# Model revisions are pinned through configs/model_revisions.yaml.
_TRUST_REMOTE_CODE = os.environ.get("PRISM_TRUST_REMOTE_CODE", "0") == "1"


# ══════════════════════════════════════════════════════════════════════
# Baseline trainers. Each overrides only the penalty of
# ShapeRegularizedTrainer; scheduling, backward path and loss scaling are
# inherited (penalty backward every reg_every_k micro-steps).
# ══════════════════════════════════════════════════════════════════════
class L2SPTrainer(ShapeRegularizedTrainer):
    """L2-SP on the trainable (LoRA) parameters: sum ||p - p_init||^2."""

    LOG_KEY = "l2sp"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_params: Dict[str, Tensor] = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters() if p.requires_grad
        }

    def _lookup(self, model):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            key = n if n in self._init_params else n.removeprefix("module.")
            if key in self._init_params:
                yield key, p

    def _penalty(self, name: str, p: Tensor, p0: Tensor) -> Tensor:
        return ((p - p0) ** 2).sum()

    def _compute_shape_loss(self, model) -> Tuple[Tensor, float]:
        loss = torch.zeros((), device=self._device_str)
        for name, p in self._lookup(model):
            p0 = self._init_params[name].to(p.device)
            loss = loss + self._penalty(name, p.float(), p0.float())
        return loss, loss.item()

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if self._count > 0:
            logs[self.LOG_KEY] = round(self._shape_sum / self._count, 8)
            self._shape_sum = self._omega_sum = 0.0
            self._count = 0
        Trainer.log(self, logs, *args, **kwargs)


class LwFTrainer(ShapeRegularizedTrainer):
    """Learning without Forgetting (Li & Hoiem, 2016) on the anchor set.

    Penalty: forward KL(p_base || p_current) between next-token distributions
    at temperature 1 (no temperature scaling), computed on the anchor
    sequences only (32 by default), and averaged over their non-padding
    positions t >= prompt_length, pooled over all anchor sequences.

    Positions: the non-padding positions t >= prompt_length of each anchor
    sequence.

    The base logits are produced on the fly by the same network with the
    LoRA adapters disabled, under no_grad; logits are cast to float32 before
    the softmax.
    """

    def _compute_shape_loss(self, model) -> Tuple[Tensor, float]:
        tot = torch.zeros((), device=self._device_str)
        n = torch.zeros((), device=self._device_str)
        for batch in self._ref_dl:
            b = {k: v.to(self._device_str) for k, v in batch.items()}
            prompt_lens = b.pop("prompt_length", None)
            mask = b.get("attention_mask")
            with torch.no_grad():
                with model.disable_adapter():
                    ref_logits = model(**b).logits
            logits = model(**b).logits
            lp = torch.log_softmax(logits.float(), dim=-1)
            rp = torch.softmax(ref_logits.float(), dim=-1)
            kl = (rp * (torch.log(rp.clamp_min(1e-12)) - lp)).sum(-1)  # (B, L)
            m = mask.bool() if mask is not None else torch.ones_like(kl, dtype=torch.bool)
            if prompt_lens is not None:
                pos = torch.arange(kl.shape[1], device=kl.device)[None, :]
                m = m & (pos >= prompt_lens.to(kl.device)[:, None])
            tot = tot + (kl * m).sum()
            n = n + m.sum()
            del ref_logits, logits, lp, rp
        loss = tot / n.clamp(min=1)
        return loss, loss.item()

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if self._count > 0:
            logs["lwf"] = round(self._shape_sum / self._count, 8)
            self._shape_sum = self._omega_sum = 0.0
            self._count = 0
        Trainer.log(self, logs, *args, **kwargs)


class EWCTrainer(L2SPTrainer):
    """Diagonal-Fisher EWC on the trainable parameters."""

    LOG_KEY = "ewc"

    def __init__(self, *args, fisher: Dict[str, Tensor], **kwargs):
        super().__init__(*args, **kwargs)
        self._fisher = fisher

    def _penalty(self, name: str, p: Tensor, p0: Tensor) -> Tensor:
        F = self._fisher[name].to(p.device)
        return (F * (p - p0) ** 2).sum()


class HintMSETrainer(ShapeRegularizedTrainer):
    """Representation distillation with the hint loss of FitNets (Romero et
    al., 2015) on the anchor set.

    Penalty: torch.nn.MSELoss (mean over all elements) between Z_P, the
    fine-tuned model's final-layer hidden states at the answer-token
    positions of the anchor set, and Z_T, the base model's features at the
    same positions (pre-computed, float32). These are the features the shape
    regularizer reads (LLMExtractor._extract_z, "concat" mode); only the
    penalty differs from the trace term.
    """

    def _compute_shape_loss(self, model) -> Tuple[Tensor, float]:
        parts = []
        for batch in self._ref_dl:
            b = {k: v.to(self._device_str) for k, v in batch.items()}
            prompt_lens = b.pop("prompt_length", None)
            masks = b.get("attention_mask")
            out = model(**b, output_hidden_states=True)
            parts.append(LLMExtractor._extract_z(out.hidden_states[-1], masks, "concat", prompt_lens))
        Z_P = torch.cat(parts, dim=0).float()
        Z_T = self.Z_T_ref.to(Z_P.device)
        loss = torch.nn.MSELoss()(Z_P, Z_T)
        return loss, loss.item()

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if self._count > 0:
            logs["hint_mse"] = round(self._shape_sum / self._count, 8)
            self._shape_sum = self._omega_sum = 0.0
            self._count = 0
        Trainer.log(self, logs, *args, **kwargs)


# ══════════════════════════════════════════════════════════════════════
def estimate_fisher(model, ref_dl, device: str) -> Dict[str, Tensor]:
    """Diagonal Fisher estimate for EWC, from the answer-only CE on the anchor set.

    For each anchor batch (batch size --reg_batch_size, 8 by default) the
    gradient of the batch-mean answer-only CE is taken and squared
    element-wise; the squares are averaged over the anchor batches. The
    estimate is taken at the current adapter state, which is
    the adapter initialisation when called from main(), and is normalised to
    unit mean over all trainable entries."""
    fisher = {n: torch.zeros_like(p, dtype=torch.float32)
              for n, p in model.named_parameters() if p.requires_grad}
    n_batches = 0
    model.zero_grad(set_to_none=True)
    for batch in ref_dl:
        b = {k: v.to(device) for k, v in batch.items()}
        prompt_lens = b.pop("prompt_length", None)
        labels = b["input_ids"].clone()
        if prompt_lens is not None:
            for i, plen in enumerate(prompt_lens):
                labels[i, :int(plen)] = -100
        attn = b.get("attention_mask")
        if attn is not None:
            labels = labels.masked_fill(attn == 0, -100)
        b["labels"] = labels
        out = model(**b)
        out.loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach().float() ** 2
        model.zero_grad(set_to_none=True)
        n_batches += 1

    total, numel = 0.0, 0
    for f in fisher.values():
        f /= max(n_batches, 1)
        total += f.sum().item()
        numel += f.numel()
    mean = total / max(numel, 1)
    if mean <= 0:
        print("  [warn] Fisher is identically zero — falling back to L2-SP "
              "weighting (F = 1).")
        return {n: torch.ones_like(f) for n, f in fisher.items()}
    for f in fisher.values():
        f /= mean
    nz = sum((f > 0).sum().item() for f in fisher.values())
    print(f"  Fisher: {n_batches} batches, unit-mean normalised, "
          f"{nz}/{numel} nonzero entries")
    return fisher


# ══════════════════════════════════════════════════════════════════════
# Mid-run resume (an interrupted run restarts from its last covered checkpoint)
# ══════════════════════════════════════════════════════════════════════
# Methods that cannot be resumed. L2SPTrainer snapshots the trainable
# parameters when it is constructed, and the EWC Fisher is estimated right
# after the adapter is created. Neither is stored in the checkpoints. A
# resumed process would rebuild both from a new adapter created with
# --seed; whether they equal those of the interrupted run is not checked,
# so the run is refused.
_NO_RESUME = ("l2sp", "ewc")
# Weight of the L2-SP penalty in a --same_loop run (numerically no penalty);
# the value train_forgetting.py uses for its same-loop reference.
SAME_LOOP_WEIGHT = 1e-9

# Fields that must agree before we resume into an existing directory. A
# mismatch means two different configs share an output_root; refuse, don't
# merge.
_RESUME_GUARD = ("method", "lambda_reg", "ref_task", "reg_samples",
                 "ref_seed", "ref_offset", "reg_every_k", "lr", "seed",
                 "model", "trained_task", "max_steps", "lora_r",
                 "eval_keep_anchor_rows")


def _env_stamp() -> Dict[str, str]:
    """Library versions, recorded in the metrics JSON for provenance."""
    import datasets
    import peft
    import transformers
    return {"torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "datasets": datasets.__version__,
            "cuda": str(torch.version.cuda),
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "cpu")}


def plan_resume(output_dir: str, cfg: Dict[str, Any],
                ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Decide where to restart and which PRISM records to keep.

    HF writes the checkpoint directory BEFORE the callback runs its PRISM
    evaluation, so a crash inside that evaluation leaves a checkpoint whose
    metrics never reached the JSON. Resuming from the newest directory would
    silently drop that step from the record, so we resume from the newest
    checkpoint the JSON actually covers and let training redo the rest.

    Returns (checkpoint_path or None, records to pre-seed the callback with).
    """
    steps = sorted(int(m.group(1)) for p in glob.glob(
        os.path.join(output_dir, "checkpoint-*"))
        if (m := re.search(r"checkpoint-(\d+)$", p)) and os.path.isdir(p))
    if not steps:
        return None, []

    jp = os.path.join(output_dir, "prism_forgetting_metrics.json")
    records: List[Dict[str, Any]] = []
    if os.path.exists(jp):
        try:
            prev = json.load(open(jp))
        except Exception as e:
            sys.exit(f"resume: {jp} is unreadable ({e}) — move it aside or "
                     f"pass --resume no")
        old = prev.get("experiment", {})
        # Keys absent from the file on disk are not compared.
        bad = [k for k in _RESUME_GUARD
               if k in old and k in cfg and old[k] != cfg[k]]
        if bad:
            sys.exit("resume: refusing to resume — "
                     + "; ".join(f"{k}: on disk {old[k]!r} vs requested "
                                 f"{cfg[k]!r}" for k in bad)
                     + f"\n  ({output_dir} holds a DIFFERENT config. Use a "
                       f"fresh --output_root, or --resume no to overwrite.)")
        records = prev.get("checkpoints", [])

    json_max = max((r.get("step", -1) for r in records), default=-1)
    covered = [s for s in steps if s <= json_max]
    if covered:
        step = max(covered)
    else:
        step = min(steps)
        print(f"  [warn] no checkpoint is covered by the metrics JSON "
              f"(json max step {json_max}, checkpoints {steps}); resuming "
              f"from {step} — steps evaluated before it are not recoverable")
    kept = sorted((r for r in records if r.get("step", 0) <= step),
                  key=lambda r: r["step"])
    ckpt = os.path.join(output_dir, f"checkpoint-{step}")
    print(f"\n=== RESUMING from {ckpt} "
          f"({len(kept)} PRISM checkpoint record(s) retained: "
          f"{[r['step'] for r in kept]}) ===")
    return ckpt, kept


# ══════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True, choices=list(TASK_CONFIGS))
    p.add_argument("--method", required=True,
                   choices=["none", "trace", "replay", "l2sp", "ewc",
                            "layer_freeze", "lwf", "hint_mse"])
    p.add_argument("--lambda_reg", type=float, default=0.0,
                   help="penalty weight; must be > 0 for the regularized "
                        "methods, ignored by none and layer_freeze; negative "
                        "values are rejected")
    p.add_argument("--lora_top_layers", type=int, default=8,
                   help="layer_freeze only: apply LoRA to the top-K layers "
                        "and freeze the rest (a low-cost continual-learning "
                        "baseline)")
    p.add_argument("--same_loop", action="store_true",
                   help="none and layer_freeze only: train in the regularized "
                        "loop (the L2-SP trainer with weight 1e-9, numerically "
                        "no penalty) instead of the library's default Trainer; "
                        "the run is written under lam1e-09")
    p.add_argument("--seed", type=int, default=42,
                   help="Sets the initial LoRA weights, the Trainer seed "
                        "(order of the training data, dropout), the default "
                        "anchor draw (shuffle seed: seed + 1000) and the "
                        "benchmark subsets of the PRISM evaluation.")
    p.add_argument("--max_steps", type=int, default=300,
                   help="number of optimizer steps (reported runs: 300)")
    p.add_argument("--save_steps", type=int, default=25)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--prism_eval_samples", type=int, default=256)
    p.add_argument("--prism_eval_batch_size", type=int, default=4)
    p.add_argument("--reg_samples", type=int, default=32)
    p.add_argument("--ref_offset", type=int, default=0,
                   help="row offset into the shuffled anchor pool; "
                        "distinct multiples of --reg_samples give disjoint "
                        "same-size draws of one fixed shuffle. "
                        "0 = the default draw.")
    p.add_argument("--reg_batch_size", type=int, default=8)
    p.add_argument("--reg_max_length", type=int, default=512)
    p.add_argument("--reg_every_k", type=int, default=8,
                   help="compute the penalty every K micro-batches (default: "
                        "8, a constant; it is not derived from --grad_accum. "
                        "The reported runs use 8 = --grad_accum, i.e. one "
                        "penalty evaluation per optimizer step)")
    p.add_argument("--eval_keep_anchor_rows", action="store_true",
                   help="Evaluation only: keep the rows of the anchor draw in "
                        "the Trainer's eval set, so eval_loss is computed on "
                        "the whole held-out slice of the fine-tuning task (the "
                        "target-task loss column of regularizer_table.py; "
                        "run_regularizer_baselines.sh passes this "
                        "flag). Default: those rows are removed (unified "
                        "protocol). Does not affect training, the anchor draw "
                        "or the PRISM records.")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--output_root", default=str(REPO / "out" / "rounds" / "regularizers"))
    p.add_argument("--resume", choices=["auto", "no"], default="auto",
                   help="auto (default): if output_dir already holds "
                        "checkpoints with a MATCHING config, restart from the "
                        "newest one the metrics JSON covers and keep the "
                        "earlier PRISM records; refused for l2sp and ewc, "
                        "whose reference point is not stored in the "
                        "checkpoints (delete the run directory and restart). "
                        "no: ignore them and overwrite.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.lambda_reg < 0:
        sys.exit(f"--lambda_reg must be >= 0 (got {args.lambda_reg})")
    if args.method not in ("none", "layer_freeze") and args.lambda_reg <= 0:
        sys.exit("--lambda_reg must be > 0 for a regularized method")
    if args.same_loop and args.method not in ("none", "layer_freeze"):
        sys.exit("--same_loop applies to the methods none and layer_freeze")
    # The anchor set is drawn from the fine-tuning task with shuffle seed
    # --seed + 1000 (as in train_forgetting.py).
    ref_task = args.task
    ref_seed = args.seed + 1000
    task_cfg = TASK_CONFIGS[args.task]
    # Learning rate of the paper's forgetting runs.
    lr = args.lr if args.lr is not None else 1e-5

    model_short = args.model.split("/")[-1].lower()
    sweep_tag = (f"lam{SAME_LOOP_WEIGHT:g}" if args.same_loop
                 else f"top{args.lora_top_layers}" if args.method == "layer_freeze"
                 else f"lam{args.lambda_reg:g}")
    output_dir = os.path.join(
        args.output_root, args.method, sweep_tag,
        f"seed{args.seed}", model_short, args.task,
    )
    os.makedirs(output_dir, exist_ok=True)
    eval_tasks = get_eval_tasks(args.task)

    # Revision the model repository is loaded at (configs/model_revisions.yaml);
    # None = not pinned, the repository head is loaded.
    revision = pinned_revision(args.model)

    experiment_config = {
        "script": "scripts/rounds/train_forgetting_baselines.py",
        "method": args.method, "lambda_reg": args.lambda_reg,
        "ref_task": ref_task, "reg_samples": args.reg_samples,
        "ref_seed": ref_seed, "ref_offset": args.ref_offset,
        "reg_every_k": args.reg_every_k,
        "model": args.model, "trained_task": args.task,
        "eval_tasks": eval_tasks, "seed": args.seed,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": LORA_TARGET_MODULES,
        "lr": lr, "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_steps": args.max_steps, "save_steps": args.save_steps,
        "max_length": args.max_length,
        "train_loss_mode": "answer_only",
        # `replay` rehearses the anchor set (source "task" of train_forgetting.py).
        "replay_source": ("task" if args.method == "replay" else None),
        "same_loop": args.same_loop,
        "revision": revision,
        "eval_keep_anchor_rows": args.eval_keep_anchor_rows,
        # Added below, once known: n_eval_rows, eval_anchor_rows_removed,
        # optim, weight_decay, lr_scheduler, max_grad_norm, K_feat, K_pred.
        # Provenance: record the library versions in the metrics file that
        # gets analysed, not just in the console log.
        "env": _env_stamp(),
    }
    print("=" * 78)
    for k, v in experiment_config.items():
        print(f"  {k:<22s}: {v}")
    print(f"  output_dir            : {output_dir}")
    print("=" * 78)

    # Decided BEFORE the 8B load so a config collision fails in seconds.
    if (args.resume == "auto" and (args.method in _NO_RESUME or args.same_loop)
            and glob.glob(os.path.join(output_dir, "checkpoint-*"))):
        sys.exit(
            f"resume: {output_dir} holds checkpoints of an earlier "
            f"{args.method} run, and {args.method} runs cannot be resumed: the "
            f"penalty is measured from the adapter initialisation of the "
            f"original process"
            + (" and weighted by the Fisher estimated at it"
               if args.method == "ewc" else "")
            + ", which is not stored in the checkpoints.\n"
              "  To run it again from step 0, delete this "
              "directory and launch the same command.")
    resume_ckpt, kept_records = (
        plan_resume(output_dir, experiment_config)
        if args.resume == "auto" else (None, []))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=_TRUST_REMOTE_CODE,
        revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=_TRUST_REMOTE_CODE,
        revision=revision,
    )
    model.config.use_cache = False

    extractor = LLMExtractor()
    device = "cuda"
    base_features, eval_dataloaders, K_theory = pre_compute_base_features(
        model, tokenizer, extractor, eval_tasks=eval_tasks,
        num_samples=args.prism_eval_samples,
        batch_size=args.prism_eval_batch_size,
        max_length=args.max_length, seed=args.seed, device=device,
    )
    experiment_config["K_feat"] = K_theory["K_feat"]
    experiment_config["K_pred"] = K_theory["K_pred"]

    # ── Anchor set ──────────────────────────────────────────────────────
    # --reg_samples sequences of the held-out slice of the fine-tuning task
    # (ref_task; prism.data.loaders split "test"): rows
    # [ref_offset, ref_offset + reg_samples) after shuffling that slice with
    # seed ref_seed (--seed + 1000, as in train_forgetting.py). The
    # draw depends only on (ref_task, ref_seed, ref_offset, reg_samples), so
    # every method reads the same rows at a given seed, and the rows are
    # redrawn for each seed.
    needs_ref = args.method in ("trace", "replay", "ewc", "lwf", "hint_mse")
    ref_dataloader = None
    Z_T_ref: Optional[Tensor] = None
    if needs_ref:
        ref_dataloader = load_task_data(
            ref_task, split="test", num_samples=args.reg_samples,
            batch_size=args.reg_batch_size, tokenizer=tokenizer,
            max_length=min(args.max_length, args.reg_max_length),
            seed=ref_seed, offset=args.ref_offset,
        )
        n_kept = len(ref_dataloader.dataset)
        print(f"\nAnchor draw: task={ref_task} seed={ref_seed} "
              f"offset={args.ref_offset} requested={args.reg_samples} "
              f"kept={n_kept}")
        if n_kept < args.reg_samples:
            # Empty/whitespace rows are dropped inside TextDataset, so a
            # requested size is not a delivered size on raw-text corpora;
            # keep it visible here.
            print(f"  [warn] {args.reg_samples - n_kept} of the requested rows "
                  f"were blank and dropped — the delivered reference size is "
                  f"{n_kept}, not {args.reg_samples}")
    if args.method in ("trace", "hint_mse"):
        z_mode_ref = get_task_metadata(ref_task)["z_mode"]
        print(f"\nPre-computing Z_T_ref (task={ref_task}, "
              f"n={args.reg_samples}, z_mode={z_mode_ref}) ...")
        Z_T_ref = extractor.extract_features(
            model, ref_dataloader, device, z_mode=z_mode_ref,
        ).float().cpu()
        print(f"  Z_T_ref: {list(Z_T_ref.shape)}")

    # ── LoRA ────────────────────────────────────────────────────────────
    model.gradient_checkpointing_enable()
    from peft import LoraConfig, TaskType, get_peft_model
    lora_kwargs = dict(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    if args.method == "layer_freeze":
        n_layers = model.config.num_hidden_layers
        top = list(range(n_layers - args.lora_top_layers, n_layers))
        lora_kwargs["layers_to_transform"] = top
        experiment_config["lora_top_layers"] = args.lora_top_layers
        print(f"  layer_freeze: LoRA on layers {top[0]}..{top[-1]} "
              f"of {n_layers} (bottom {top[0]} layers frozen entirely)")
    lora_config = LoraConfig(**lora_kwargs)
    # --seed sets the adapter initialisation: the generators are seeded
    # immediately before the adapter is created.
    set_seed(args.seed)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    fisher = None
    if args.method == "ewc":
        print("\nEstimating diagonal Fisher on the anchor set ...")
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        fisher = estimate_fisher(model, ref_dataloader, device)

    # ── Data + callback + training args ────────────────────────────────
    train_dataset = build_dataset(args.task, "train", tokenizer,
                                  max_length=args.max_length,
                                  max_samples=task_cfg["max_train_samples"],
                                  seed=args.seed)
    eval_dataset = build_dataset(args.task, "eval", tokenizer,
                                 max_length=args.max_length,
                                 max_samples=task_cfg["max_eval_samples"],
                                 seed=args.seed)
    print(f"  Train {len(train_dataset):,} / eval {len(eval_dataset):,}")
    # Target-task evaluation rows. When the Trainer's eval set is the whole
    # held-out slice (max_eval_samples is None: TruthfulQA, BBQ), that slice is
    # also the one the anchor set is drawn from, and it is disjoint from the
    # training slice.
    #
    # Default (unified protocol): the rows of the anchor draw (same shuffle
    # seed, offset and size as above) are removed from the eval set, for every
    # method, so eval_loss is computed on held-out rows outside the anchor
    # set. For `l2sp`, `none` and `layer_freeze`, which build no anchor set
    # and read no row of the slice during training, the removed rows are the
    # rows the anchor-based methods use at this seed, so that eval_loss is on
    # the same rows for every method.
    #
    # --eval_keep_anchor_rows: nothing is removed and eval_loss is computed on
    # the whole slice (163 rows for TruthfulQA), as for the target-task loss
    # column of regularizer_table.py. The flag only selects the rows
    # of eval_loss; training, the anchor draw and the PRISM records are the
    # same with and without it.
    #
    # The PRISM record of the fine-tuning task itself (first entry of
    # eval_tasks) is drawn separately from the whole slice and is not the
    # target-task loss.
    n_removed = 0
    if task_cfg["max_eval_samples"] is None and not args.eval_keep_anchor_rows:
        _n = len(eval_dataset)
        _lo = min(args.ref_offset, _n)
        _hi = min(args.ref_offset + args.reg_samples, _n)
        _ref = set(
            eval_dataset.add_column("_row", list(range(_n)))
            .shuffle(seed=ref_seed).select(range(_lo, _hi))["_row"]
        )
        eval_dataset = eval_dataset.select([i for i in range(_n) if i not in _ref])
        n_removed = len(_ref)
        print(f"  Eval size after removing the {len(_ref)} anchor rows: "
              f"{len(eval_dataset):,}")
    else:
        print(f"  Eval set as built, anchor rows not removed: "
              f"{len(eval_dataset):,}")
    experiment_config["n_eval_rows"] = len(eval_dataset)
    experiment_config["eval_anchor_rows_removed"] = n_removed

    prism_callback = PRISMCheckpointCallback(
        model=model, base_features=base_features,
        eval_dataloaders=eval_dataloaders, extractor=extractor,
        trained_task=args.task, eval_tasks=eval_tasks, model_id=args.model,
        output_dir=output_dir, device=device,
        experiment_config=experiment_config, K_theory=K_theory,
    )
    # The callback rewrites the whole JSON at every save from its in-memory
    # list, so a resumed run must inherit the earlier records or it truncates
    # the file to the post-resume tail.
    if kept_records:
        prism_callback.all_checkpoints = list(kept_records)
    collator = AnswerOnlyDataCollator(tokenizer=tokenizer)
    try:
        import bitsandbytes  # noqa: F401
        optim = "paged_adamw_8bit"
    except ImportError:
        optim = "adamw_torch"
        print("  [bitsandbytes not found] Using standard AdamW optimizer.")
    # The optimizer depends on whether bitsandbytes can be imported.
    print(f"  Optimizer: {optim}")

    training_args = TrainingArguments(
        output_dir=output_dir, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr, lr_scheduler_type="cosine",
        warmup_steps=int(args.warmup_ratio * args.max_steps),
        weight_decay=0.01, max_grad_norm=1.0, optim=optim,
        save_steps=args.save_steps, save_total_limit=2,
        logging_steps=args.logging_steps,
        eval_strategy="steps", eval_steps=args.save_steps,
        bf16=True, dataloader_num_workers=0, report_to="none",
        remove_unused_columns=False, seed=args.seed,
    )
    # Recorded for the metrics JSON (the dict is shared with the callback).
    experiment_config["optim"] = optim
    experiment_config["weight_decay"] = training_args.weight_decay
    experiment_config["lr_scheduler"] = str(
        getattr(training_args.lr_scheduler_type, "value",
                training_args.lr_scheduler_type))
    experiment_config["max_grad_norm"] = training_args.max_grad_norm

    common = dict(model=model, args=training_args,
                  train_dataset=train_dataset, eval_dataset=eval_dataset,
                  data_collator=collator, callbacks=[prism_callback])
    reg = dict(ref_dataloader=ref_dataloader, lambda_shape=args.lambda_reg,
               reg_every_k=args.reg_every_k, device_str=device)

    if args.method == "trace":
        trainer = ShapeRegularizedTrainer(**common, Z_T_ref=Z_T_ref, **reg)
    elif args.method == "replay":
        trainer = ReplayCETrainer(**common, Z_T_ref=None, **reg)
    elif args.method == "lwf":
        trainer = LwFTrainer(**common, Z_T_ref=None, **reg)
    elif args.method == "hint_mse":
        trainer = HintMSETrainer(**common, Z_T_ref=Z_T_ref, **reg)
    elif args.method == "l2sp":
        trainer = L2SPTrainer(**common, Z_T_ref=None, **reg)
    elif args.method == "ewc":
        trainer = EWCTrainer(**common, Z_T_ref=None, fisher=fisher, **reg)
    elif args.same_loop:
        # none / layer_freeze in the regularized loop: the L2-SP trainer with
        # a weight that makes the penalty numerically zero.
        trainer = L2SPTrainer(**common, Z_T_ref=None,
                              **{**reg, "lambda_shape": SAME_LOOP_WEIGHT})
    else:
        # none / layer_freeze: no penalty; the library's default Trainer.
        # The unregularized run in the regularized loop is
        # `train_forgetting.py --lambda_shape 1e-9`.
        trainer = Trainer(**common)

    print(f"\nTraining ({args.method}, lambda={args.lambda_reg})"
          + (f", resuming at {os.path.basename(resume_ckpt)}"
             if resume_ckpt else "") + " ...")
    if resume_ckpt:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()
    print(f"\nDone. PRISM log: {prism_callback.json_path}")


if __name__ == "__main__":
    main()
