#!/usr/bin/env python3
"""
Stage 1 — LoRA fine-tune a base model on one of 5 tasks.

At every checkpoint, computes PRISM forgetting metrics on ALL 5 eval tasks
and saves the full results to JSON.  This eliminates the need for a separate
Stage 2 inference pass for most analysis.

Fine-tuning tasks: ARC, MMLU, SQuAD, TriviaQA, GSM8K
Models: meta-llama/Llama-3.1-8B, Qwen/Qwen3-8B-Base (or any causal LM)

Under LoRA the lm_head is frozen (H_t = H_0), so PRISM's head divergence
term vanishes under identity alignment, isolating forgetting entirely in
backbone geometry (Eq. 8).

Usage:
    python train_forgetting_multitask.py \\
        --model meta-llama/Llama-3.1-8B --task gsm8k

    python train_forgetting_multitask.py \\
        --model Qwen/Qwen3-8B-Base --task arc --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from prism.core.bounds import UnifiedBound
from prism.core.metrics import PRISMMetrics
from prism.data.loaders import get_task_metadata, load_task_data
from prism.models.extractors import LLMExtractor


# ── Formatters — identical to prism/data/loaders.py for consistency ───────

def _format_arc(row: dict) -> str:
    q = row["question"]
    labels = row["choices"]["label"]
    texts = row["choices"]["text"]
    opts = "\n".join(f"{labels[i]}. {texts[i]}" for i in range(len(labels)))
    key = row["answerKey"]
    return f"Question: {q}\n{opts}\nAnswer: {key}"


def _format_mmlu(row: dict) -> str:
    q = row["question"]
    choices = row["choices"]
    ans = row["answer"]
    if isinstance(ans, int):
        ans_label = ["A", "B", "C", "D"][ans]
    else:
        ans_label = ans
    labels = ["A", "B", "C", "D"]
    opts = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
    return f"Question: {q}\n{opts}\nAnswer: {ans_label}"


def _format_squad(row: dict) -> str:
    context = row["context"]
    q = row["question"]
    ans = row["answers"]["text"][0]
    return f"Context: {context}\nQuestion: {q}\nAnswer: {ans}"


def _format_triviaqa(row: dict) -> str:
    q = row["question"]
    ans = row["answer"]["value"]
    return f"Question: {q}\nAnswer: {ans}"


def _format_gsm8k(row: dict) -> str:
    return f"Question: {row['question']}\nAnswer: {row['answer']}"


FORMATTERS = {
    "arc": _format_arc,
    "mmlu": _format_mmlu,
    "squad": _format_squad,
    "triviaqa": _format_triviaqa,
    "gsm8k": _format_gsm8k,
}

# ── Task-specific dataset configuration ───────────────────────────────────

TASK_CONFIGS = {
    "arc": {
        "hf_id": "allenai/ai2_arc",
        "hf_subset": "ARC-Challenge",
        "train_split": "train",
        "eval_split": "test",
        "max_train_samples": None,   # use all 1,119
        "max_eval_samples": 256,
        "default_max_steps": 700,
        "default_save_steps": 50,
    },
    "mmlu": {
        "hf_id": "cais/mmlu",
        "hf_subset": "all",
        "train_split": "auxiliary_train",
        "eval_split": "validation",
        "max_train_samples": 8000,
        "max_eval_samples": 256,
        "default_max_steps": 1500,
        "default_save_steps": 100,
    },
    "squad": {
        "hf_id": "rajpurkar/squad",
        "hf_subset": None,
        "train_split": "train",
        "eval_split": "validation",
        "max_train_samples": 8000,
        "max_eval_samples": 256,
        "default_max_steps": 1500,
        "default_save_steps": 100,
    },
    "triviaqa": {
        "hf_id": "trivia_qa",
        "hf_subset": "rc.nocontext",
        "train_split": "train",
        "eval_split": "validation",
        "max_train_samples": 8000,
        "max_eval_samples": 256,
        "default_max_steps": 1500,
        "default_save_steps": 100,
    },
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "hf_subset": "main",
        "train_split": "train",
        "eval_split": "test",
        "max_train_samples": None,   # use all 7,473
        "max_eval_samples": 256,
        "default_max_steps": 1400,
        "default_save_steps": 100,
    },
}

ALL_EVAL_TASKS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ======================================================================
# PRISM Checkpoint Callback
# ======================================================================

class PRISMCheckpointCallback(TrainerCallback):
    """Evaluate PRISM forgetting metrics on all 5 tasks at each checkpoint.

    Pre-computes base-model features once, then at every save step:
      1. Switches the PEFT model to eval mode
      2. Extracts proxy features + loss for each eval task
      3. Computes PRISM metrics (Ω, Δρ, bound) against the base features
      4. Prints a detailed table and appends to a running JSON log
      5. Restores training mode

    The JSON is overwritten at each checkpoint so partial results survive
    crashes.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        base_features: Dict[str, Dict[str, Any]],
        eval_dataloaders: Dict[str, Tuple[Any, str]],
        extractor: LLMExtractor,
        trained_task: str,
        model_id: str,
        output_dir: str,
        device: str,
        experiment_config: Dict[str, Any],
    ):
        super().__init__()
        self.model = model
        self.base_features = base_features      # {task: {Z, H, loss_full, loss_answer}}
        self.eval_dataloaders = eval_dataloaders  # {task: (dataloader, z_mode)}
        self.extractor = extractor
        self.trained_task = trained_task
        self.model_id = model_id
        self.output_dir = output_dir
        self.device = device
        self.experiment_config = experiment_config

        self.all_checkpoints: List[Dict[str, Any]] = []
        self.json_path = os.path.join(output_dir, "prism_forgetting_metrics.json")

    # ------------------------------------------------------------------
    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        step = state.global_step
        print(f"\n{'=' * 78}")
        print(f"  PRISM evaluation @ step {step}")
        print(f"{'=' * 78}")
        t0 = time.time()

        # Retrieve latest train_loss and eval_loss from Trainer log
        train_loss = self._latest_metric(state, "loss")
        eval_loss = self._latest_metric(state, "eval_loss")

        # Run PRISM on all tasks
        task_results = self._evaluate_all_tasks(step)

        # Assemble checkpoint record
        checkpoint_record = {
            "step": step,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "tasks": task_results,
        }
        self.all_checkpoints.append(checkpoint_record)

        elapsed = time.time() - t0

        # Print table
        self._print_table(step, train_loss, eval_loss, task_results, elapsed)

        # Save JSON (overwrite — survives crashes)
        self._save_json()

    # ------------------------------------------------------------------
    def _evaluate_all_tasks(self, step: int) -> Dict[str, Dict[str, Any]]:
        """Extract proxy features and compute PRISM metrics for each task."""
        self.model.eval()
        task_results: Dict[str, Dict[str, Any]] = {}

        with torch.no_grad():
            for task in ALL_EVAL_TASKS:
                dl, z_mode = self.eval_dataloaders[task]

                # Proxy features + loss (single forward pass)
                Z_P, loss_stats_P = self.extractor.extract_features_and_loss_per_sample(
                    self.model, dl, self.device, z_mode=z_mode,
                )
                H_P = self.extractor.extract_head(self.model)

                # Base features
                Z_T = self.base_features[task]["Z"]
                H_T = self.base_features[task]["H"]
                loss_T_full = self.base_features[task]["loss_full"]
                loss_T_answer = self.base_features[task]["loss_answer"]

                # PRISM metrics (Procrustes alignment, W = W_opt)
                prism = PRISMMetrics.compute_all(
                    Z_T.float(), H_T.float(),
                    Z_P.float().cpu(), H_P.float().cpu(),
                    W=None,
                    label=f"step-{step}_{task}",
                )
                UnifiedBound.fill_result(prism, K_feat=1.0, K_pred=1.0)

                # Proxy losses
                loss_P_full = loss_stats_P["losses"].mean().item()
                loss_P_answer = (
                    loss_stats_P["answer_losses"].mean().item()
                    if loss_stats_P.get("answer_losses") is not None
                    else None
                )

                # Delta risks
                delta_risk_full = abs(loss_P_full - loss_T_full)
                delta_risk_answer = (
                    abs(loss_P_answer - loss_T_answer)
                    if loss_P_answer is not None and loss_T_answer is not None
                    else None
                )

                task_results[task] = {
                    # Geometry
                    "omega": prism.omega,
                    "rho_target": prism.rho_target,
                    "rho_proxy": prism.rho_proxy,
                    "scale_mismatch": prism.scale_mismatch,
                    "shape_mismatch": prism.shape_mismatch,
                    "feature_error": prism.feature_error,
                    "head_discrepancy": prism.head_discrepancy,
                    # Bound
                    "risk_bound_feature": prism.risk_bound_feature,
                    "risk_bound_head": prism.risk_bound_head,
                    "risk_bound_total": prism.risk_bound_total,
                    # Empirical risk (full-sequence)
                    "loss_target": loss_T_full,
                    "loss_proxy": loss_P_full,
                    "delta_risk": delta_risk_full,
                    # Empirical risk (answer-only)
                    "loss_target_answer": loss_T_answer,
                    "loss_proxy_answer": loss_P_answer,
                    "delta_risk_answer": delta_risk_answer,
                    # Bound validation
                    "bound_holds": (
                        prism.risk_bound_total >= delta_risk_full
                        if prism.risk_bound_total is not None
                        else None
                    ),
                }

        self.model.train()
        return task_results

    # ------------------------------------------------------------------
    def _print_table(
        self,
        step: int,
        train_loss: Optional[float],
        eval_loss: Optional[float],
        task_results: Dict[str, Dict],
        elapsed: float,
    ):
        tl_s = f"{train_loss:.4f}" if train_loss is not None else "—"
        el_s = f"{eval_loss:.4f}" if eval_loss is not None else "—"

        print(f"  train_loss={tl_s}  eval_loss({self.trained_task})={el_s}  "
              f"prism_eval={elapsed:.1f}s")
        print()

        # Header
        hdr = (
            f"  {'Task':<11s}  {'Omega':>8s}  {'Drho^2':>10s}  {'1-Omega':>10s}"
            f"  {'FeatErr':>8s}  {'HeadDisc':>8s}  {'Bound':>8s}"
            f"  {'Loss_T':>8s}  {'Loss_P':>8s}  {'|dR|':>8s}"
        )
        print(hdr)
        print(f"  {'─' * (len(hdr) - 2)}")

        for task in ALL_EVAL_TASKS:
            r = task_results[task]
            marker = " *" if task == self.trained_task else "  "
            one_minus_omega = 1.0 - r["omega"]

            dr_s = f"{r['delta_risk']:8.4f}"
            bound_s = f"{r['risk_bound_total']:8.4f}" if r["risk_bound_total"] is not None else "       —"

            print(
                f"  {task:<9s}{marker}"
                f"  {r['omega']:8.6f}"
                f"  {r['scale_mismatch']:10.6f}"
                f"  {one_minus_omega:10.6f}"
                f"  {r['feature_error']:8.4f}"
                f"  {r['head_discrepancy']:8.4f}"
                f"  {bound_s}"
                f"  {r['loss_target']:8.4f}"
                f"  {r['loss_proxy']:8.4f}"
                f"  {dr_s}"
            )

        # Answer-only delta risk (if available)
        has_answer = any(
            r.get("delta_risk_answer") is not None for r in task_results.values()
        )
        if has_answer:
            print()
            print(f"  {'Task':<11s}  {'ALoss_T':>8s}  {'ALoss_P':>8s}  {'|AdR|':>8s}")
            print(f"  {'─' * 40}")
            for task in ALL_EVAL_TASKS:
                r = task_results[task]
                marker = " *" if task == self.trained_task else "  "
                if r.get("delta_risk_answer") is not None:
                    print(
                        f"  {task:<9s}{marker}"
                        f"  {r['loss_target_answer']:8.4f}"
                        f"  {r['loss_proxy_answer']:8.4f}"
                        f"  {r['delta_risk_answer']:8.4f}"
                    )

        print(f"{'=' * 78}")

    # ------------------------------------------------------------------
    def _save_json(self):
        """Overwrite the JSON file with all accumulated results."""
        os.makedirs(self.output_dir, exist_ok=True)
        payload = {
            "experiment": self.experiment_config,
            "base_model_losses": {
                task: {
                    "loss_full": self.base_features[task]["loss_full"],
                    "loss_answer": self.base_features[task]["loss_answer"],
                    "Z_shape": list(self.base_features[task]["Z"].shape),
                }
                for task in ALL_EVAL_TASKS
            },
            "checkpoints": self.all_checkpoints,
        }
        with open(self.json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  [saved] {self.json_path}")

    # ------------------------------------------------------------------
    @staticmethod
    def _latest_metric(state: TrainerState, key: str) -> Optional[float]:
        for entry in reversed(state.log_history):
            if key in entry:
                return entry[key]
        return None


# ======================================================================
# Pre-compute base model features
# ======================================================================

def pre_compute_base_features(
    model: torch.nn.Module,
    tokenizer,
    extractor: LLMExtractor,
    num_samples: int,
    batch_size: int,
    max_length: int,
    seed: int,
    device: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Tuple[Any, str]]]:
    """Extract base model features + loss on all 5 eval tasks.

    Returns:
        base_features:    {task: {Z, H, loss_full, loss_answer, Z_shape}}
        eval_dataloaders: {task: (DataLoader, z_mode)}  — reused in callback
    """
    print(f"\n{'─' * 78}")
    print(f"  Pre-computing base model features on {len(ALL_EVAL_TASKS)} eval tasks")
    print(f"  (n={num_samples}, batch_size={batch_size}, max_length={max_length})")
    print(f"{'─' * 78}")

    model.eval()
    base_features: Dict[str, Dict[str, Any]] = {}
    eval_dataloaders: Dict[str, Tuple[Any, str]] = {}

    for task in ALL_EVAL_TASKS:
        meta = get_task_metadata(task)
        z_mode = meta["z_mode"]
        print(f"  {task:<10s} (z_mode={z_mode}) ... ", end="", flush=True)
        t0 = time.time()

        dl = load_task_data(
            task, split="test", num_samples=num_samples,
            batch_size=batch_size, tokenizer=tokenizer,
            max_length=max_length, seed=seed,
        )

        Z_T, loss_stats = extractor.extract_features_and_loss_per_sample(
            model, dl, device, z_mode=z_mode,
        )
        H_T = extractor.extract_head(model)

        loss_full = loss_stats["losses"].mean().item()
        loss_answer = (
            loss_stats["answer_losses"].mean().item()
            if loss_stats.get("answer_losses") is not None
            else None
        )

        base_features[task] = {
            "Z": Z_T.cpu(),
            "H": H_T.cpu(),
            "loss_full": loss_full,
            "loss_answer": loss_answer,
        }
        eval_dataloaders[task] = (dl, z_mode)

        elapsed = time.time() - t0
        la_s = f"  loss_answer={loss_answer:.4f}" if loss_answer is not None else ""
        print(f"Z={list(Z_T.shape)}  loss={loss_full:.4f}{la_s}  ({elapsed:.1f}s)")

    print(f"{'─' * 78}")
    return base_features, eval_dataloaders


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1: LoRA fine-tune on a single task for PRISM forgetting",
    )
    # Required
    p.add_argument("--model", required=True,
                   help="HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B)")
    p.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()),
                   help="Fine-tuning task")

    # Output
    p.add_argument("--output_dir", default=None,
                   help="Override checkpoint output directory")

    # Training
    p.add_argument("--max_steps", type=int, default=None,
                   help="Max training steps (default: task-specific)")
    p.add_argument("--save_steps", type=int, default=None,
                   help="Save checkpoint every N steps (default: task-specific)")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (default: 2e-4 for LLaMA, 1e-4 for Qwen)")
    p.add_argument("--batch_size", type=int, default=2,
                   help="Per-device train batch size")
    p.add_argument("--grad_accum", type=int, default=8,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--max_length", type=int, default=512,
                   help="Maximum sequence length")
    p.add_argument("--warmup_ratio", type=float, default=0.05,
                   help="Warmup ratio (fraction of max_steps)")

    # LoRA
    p.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Data
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Override max training samples (default: task-specific)")
    p.add_argument("--max_eval_samples", type=int, default=None,
                   help="Override max eval samples for Trainer validation (default: 256)")

    # PRISM eval
    p.add_argument("--prism_eval_samples", type=int, default=256,
                   help="Number of samples per task for PRISM evaluation")
    p.add_argument("--prism_eval_batch_size", type=int, default=4,
                   help="Batch size for PRISM feature extraction")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=10)
    return p.parse_args()


# ======================================================================
# Data loading  (for training only — PRISM eval uses prism.data.loaders)
# ======================================================================

def _load_hf_dataset(task_name: str, split: str):
    cfg = TASK_CONFIGS[task_name]
    hf_args = [cfg["hf_id"]]
    if cfg["hf_subset"] is not None:
        hf_args.append(cfg["hf_subset"])
    actual_split = cfg["train_split"] if split == "train" else cfg["eval_split"]
    return load_dataset(*hf_args, split=actual_split, trust_remote_code=True)


def build_dataset(task_name, split, tokenizer, max_length, max_samples, seed):
    raw = _load_hf_dataset(task_name, split)
    if max_samples is not None and len(raw) > max_samples:
        raw = raw.shuffle(seed=seed).select(range(max_samples))
    formatter = FORMATTERS[task_name]
    original_columns = raw.column_names

    def tokenize_fn(batch):
        keys = list(batch.keys())
        n = len(batch[keys[0]])
        texts = []
        for i in range(n):
            row = {k: batch[k][i] for k in keys}
            texts.append(formatter(row))
        return tokenizer(texts, truncation=True, max_length=max_length, padding=False)

    return raw.map(
        tokenize_fn, batched=True, remove_columns=original_columns,
        desc=f"Tokenizing {task_name} ({split})",
    )


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()
    task_cfg = TASK_CONFIGS[args.task]

    # ── Resolve defaults ─────────────────────────────────────────────
    max_steps = args.max_steps or task_cfg["default_max_steps"]
    save_steps = args.save_steps or task_cfg["default_save_steps"]
    lr = args.lr if args.lr is not None else (1e-4 if "qwen" in args.model.lower() else 2e-4)
    max_train_samples = (
        args.max_train_samples
        if args.max_train_samples is not None
        else task_cfg["max_train_samples"]
    )
    max_eval_samples = (
        args.max_eval_samples
        if args.max_eval_samples is not None
        else task_cfg["max_eval_samples"]
    )

    model_short = args.model.split("/")[-1].lower()
    output_dir = args.output_dir or os.path.join(
        "checkpoints", "forgetting_multitask", model_short, args.task,
    )
    os.makedirs(output_dir, exist_ok=True)

    experiment_config = {
        "model": args.model,
        "trained_task": args.task,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": LORA_TARGET_MODULES,
        "lr": lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_steps": max_steps,
        "save_steps": save_steps,
        "max_length": args.max_length,
        "warmup_ratio": args.warmup_ratio,
        "max_train_samples": max_train_samples,
        "prism_eval_samples": args.prism_eval_samples,
        "seed": args.seed,
    }

    print(f"{'=' * 78}")
    print(f"  PRISM Forgetting — Stage 1: LoRA Fine-Tuning + Online Monitoring")
    print(f"{'=' * 78}")
    for k, v in experiment_config.items():
        print(f"  {k:<25s}: {v}")
    print(f"  output_dir              : {output_dir}")
    print(f"{'=' * 78}")

    # ── Tokenizer ────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model (before LoRA — used for base feature extraction) ───────
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── Pre-compute base model features ──────────────────────────────
    extractor = LLMExtractor()
    tensor_device = "cuda"

    base_features, eval_dataloaders = pre_compute_base_features(
        model, tokenizer, extractor,
        num_samples=args.prism_eval_samples,
        batch_size=args.prism_eval_batch_size,
        max_length=args.max_length,
        seed=args.seed,
        device=tensor_device,
    )

    # ── Apply LoRA ───────────────────────────────────────────────────
    model.gradient_checkpointing_enable()
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        print("ERROR: peft is required. Install with: pip install peft", file=sys.stderr)
        sys.exit(1)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training datasets ────────────────────────────────────────────
    print(f"\nBuilding training dataset: {args.task} ...")
    train_dataset = build_dataset(
        args.task, "train", tokenizer,
        max_length=args.max_length,
        max_samples=max_train_samples,
        seed=args.seed,
    )
    print(f"  Train size: {len(train_dataset):,} examples")

    print(f"Building eval dataset: {args.task} ...")
    eval_dataset = build_dataset(
        args.task, "eval", tokenizer,
        max_length=args.max_length,
        max_samples=max_eval_samples,
        seed=args.seed,
    )
    print(f"  Eval size:  {len(eval_dataset):,} examples")

    # ── PRISM callback ───────────────────────────────────────────────
    prism_callback = PRISMCheckpointCallback(
        model=model,
        base_features=base_features,
        eval_dataloaders=eval_dataloaders,
        extractor=extractor,
        trained_task=args.task,
        model_id=args.model,
        output_dir=output_dir,
        device=tensor_device,
        experiment_config=experiment_config,
    )

    # ── Trainer ──────────────────────────────────────────────────────
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    try:
        import bitsandbytes  # noqa: F401
        optim = "paged_adamw_8bit"
    except ImportError:
        optim = "adamw_torch"
        print("  [bitsandbytes not found] Using standard AdamW optimizer.")

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim=optim,
        save_steps=save_steps,
        save_total_limit=None,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=save_steps,
        bf16=True,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=True,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[prism_callback],
    )

    # ── Train ────────────────────────────────────────────────────────
    print(f"\nStarting training ...")
    trainer.train()

    # ── Final summary ────────────────────────────────────────────────
    saved_steps = sorted(
        int(d.name.split("-")[-1])
        for d in Path(output_dir).glob("checkpoint-*")
        if d.name.split("-")[-1].isdigit()
    )
    print(f"\n{'=' * 78}")
    print(f"  Training complete.")
    print(f"  Checkpoints : {output_dir}")
    print(f"  Saved steps : {saved_steps}")
    print(f"  PRISM log   : {prism_callback.json_path}")
    print(f"  Total ckpts : {len(saved_steps)}")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
