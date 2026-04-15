#!/usr/bin/env python3
"""
Stage 1 — LoRA fine-tune a base model on one of 5 tasks.

Saves LoRA adapter checkpoints at regular intervals for PRISM forgetting
analysis (Stage 2: infer_forgetting_multitask.py).

Fine-tuning tasks: ARC, MMLU, SQuAD, TriviaQA, GSM8K
Models: meta-llama/Llama-3.1-8B, Qwen/Qwen3-8B-Base (or any causal LM)

Under LoRA the lm_head is frozen (H_t = H_0), so PRISM's head divergence
term vanishes, isolating forgetting entirely in backbone geometry (Eq. 8).

Usage:
    python train_forgetting_multitask.py \\
        --model meta-llama/Llama-3.1-8B --task gsm8k

    python train_forgetting_multitask.py \\
        --model Qwen/Qwen3-8B-Base --task arc --lr 1e-4

    python train_forgetting_multitask.py \\
        --model meta-llama/Llama-3.1-8B --task mmlu \\
        --max_train_samples 8000 --max_steps 1500
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ── Formatters — reuse from prism/data/loaders.py for consistency ─────────
# These produce a single string per example, matching the PRISM eval format.


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
# Each task specifies HuggingFace IDs, splits, and default training parameters.

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

# ── LoRA target modules (same for LLaMA and Qwen) ────────────────────────

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ── CLI ───────────────────────────────────────────────────────────────────

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
                   help="Override checkpoint output directory "
                        "(default: checkpoints/forgetting_multitask/<model_short>/<task>)")

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
                   help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    p.add_argument("--max_length", type=int, default=512,
                   help="Maximum sequence length")
    p.add_argument("--warmup_ratio", type=float, default=0.05,
                   help="Warmup ratio (fraction of max_steps)")

    # LoRA
    p.add_argument("--lora_r", type=int, default=32,
                   help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=64,
                   help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05,
                   help="LoRA dropout")

    # Data
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Override max training samples (default: task-specific)")
    p.add_argument("--max_eval_samples", type=int, default=None,
                   help="Override max eval samples (default: 256)")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=10)
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────

def _load_hf_dataset(task_name: str, split: str):
    """Load a HuggingFace dataset for the given task and split."""
    cfg = TASK_CONFIGS[task_name]
    hf_args = [cfg["hf_id"]]
    if cfg["hf_subset"] is not None:
        hf_args.append(cfg["hf_subset"])
    actual_split = cfg["train_split"] if split == "train" else cfg["eval_split"]
    return load_dataset(*hf_args, split=actual_split, trust_remote_code=True)


def build_dataset(
    task_name: str,
    split: str,
    tokenizer,
    max_length: int,
    max_samples: int | None,
    seed: int,
):
    """Load, format, and tokenize a dataset for causal LM fine-tuning."""
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
        return tokenizer(
            texts, truncation=True, max_length=max_length, padding=False,
        )

    dataset = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=original_columns,
        desc=f"Tokenizing {task_name} ({split})",
    )
    return dataset


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    task_cfg = TASK_CONFIGS[args.task]

    # ── Resolve defaults ─────────────────────────────────────────────────
    max_steps = args.max_steps or task_cfg["default_max_steps"]
    save_steps = args.save_steps or task_cfg["default_save_steps"]

    # Model-specific learning rate default
    if args.lr is not None:
        lr = args.lr
    elif "qwen" in args.model.lower():
        lr = 1e-4
    else:
        lr = 2e-4

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

    # Derive a short model name for directory paths
    model_short = args.model.split("/")[-1].lower()
    output_dir = args.output_dir or os.path.join(
        "checkpoints", "forgetting_multitask", model_short, args.task,
    )

    print(f"{'=' * 72}")
    print(f"  PRISM Forgetting — Stage 1: LoRA Fine-Tuning")
    print(f"{'=' * 72}")
    print(f"  Model       : {args.model}")
    print(f"  Task        : {args.task}")
    print(f"  LoRA r/α    : {args.lora_r}/{args.lora_alpha}")
    print(f"  LR          : {lr}")
    print(f"  Eff. batch  : {args.batch_size * args.grad_accum}")
    print(f"  Max steps   : {max_steps}")
    print(f"  Save every  : {save_steps} steps")
    print(f"  Max length  : {args.max_length}")
    print(f"  Train samples: {max_train_samples or 'all'}")
    print(f"  Output      : {output_dir}")
    print(f"{'=' * 72}")

    # ── Tokenizer ────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # required for training

    # ── Model ────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False          # required for gradient checkpointing
    model.gradient_checkpointing_enable()

    # ── Apply LoRA ───────────────────────────────────────────────────────
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

    # ── Datasets ─────────────────────────────────────────────────────────
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

    # ── Training ─────────────────────────────────────────────────────────
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Prefer paged 8-bit Adam for memory efficiency (requires bitsandbytes)
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
        save_total_limit=None,       # keep ALL checkpoints
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
    )

    print(f"\nStarting training ...")
    trainer.train()

    # ── Summary ──────────────────────────────────────────────────────────
    saved_steps = sorted(
        int(d.name.split("-")[-1])
        for d in Path(output_dir).glob("checkpoint-*")
        if d.name.split("-")[-1].isdigit()
    )
    print(f"\n{'=' * 72}")
    print(f"  Training complete.")
    print(f"  Checkpoints saved under: {output_dir}")
    print(f"  Saved steps: {saved_steps}")
    print(f"  Total checkpoints: {len(saved_steps)}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
