#!/usr/bin/env python3
"""
Stage 1 — Fine-tune Llama-3-8B on GSM8K (catastrophic forgetting experiment).

Three trainable modes, selectable via --mode:
  full           : all parameters trainable
  head_only      : freeze backbone (model.model), train lm_head only
  backbone_only  : freeze lm_head, train backbone only

Saves a full HuggingFace checkpoint every --save_steps steps so that
Stage 2 (run_forgetting.sh) can load them for PRISM inference.

Usage:
  python train_forgetting.py --mode full --output_dir checkpoints/forgetting/full
  python train_forgetting.py --mode head_only --output_dir checkpoints/forgetting/head_only
  python train_forgetting.py --mode backbone_only --output_dir checkpoints/forgetting/backbone_only
"""

from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ── GSM8K formatter ────────────────────────────────────────────────────────────
def _format(row: dict) -> str:
    return f"Question: {row['question']}\nAnswer: {row['answer']}"


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Llama-3-8B on GSM8K")
    p.add_argument("--model",       default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--mode",        choices=["full", "head_only", "backbone_only"],
                   default="full")
    p.add_argument("--output_dir",  default="./checkpoints/forgetting/full")
    p.add_argument("--max_steps",   type=int, default=500)
    p.add_argument("--save_steps",  type=int, default=50)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--batch_size",  type=int, default=4,
                   help="Per-device train batch size")
    p.add_argument("--grad_accum",  type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    p.add_argument("--max_length",  type=int, default=512)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--grad_checkpointing", action="store_true",
                   help="Enable gradient checkpointing to reduce VRAM usage")
    p.add_argument("--eval_samples",  type=int, default=256,
                   help="Number of GSM8K test samples for eval; 0 = use all 1319")
    p.add_argument("--seed",          type=int, default=42,
                   help="Random seed for shuffling the eval test split")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ─────────────────────────────────────────────────────────────────
    # device_map={"": 0} pins the whole model to GPU 0 (respects CUDA_VISIBLE_DEVICES).
    # This is compatible with HuggingFace Trainer on single-GPU setups.
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()

    # ── Freeze parameters per mode ────────────────────────────────────────────
    if args.mode == "head_only":
        for name, param in model.named_parameters():
            param.requires_grad = "lm_head" in name
        print("Mode: head_only — backbone frozen, lm_head trainable")

    elif args.mode == "backbone_only":
        for name, param in model.named_parameters():
            param.requires_grad = "lm_head" not in name
        print("Mode: backbone_only — lm_head frozen, backbone trainable")

    else:  # full
        print("Mode: full — all parameters trainable")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    # ── Dataset ───────────────────────────────────────────────────────────────
    def tokenize(batch: dict) -> dict:
        texts = [_format({"question": q, "answer": a})
                 for q, a in zip(batch["question"], batch["answer"])]
        return tokenizer(texts, truncation=True, max_length=args.max_length, padding=False)

    print("Loading GSM8K train split ...")
    raw_train = load_dataset("openai/gsm8k", "main", split="train")
    dataset = raw_train.map(tokenize, batched=True, remove_columns=raw_train.column_names)
    print(f"Train size: {len(dataset):,} examples")

    print("Loading GSM8K test split ...")
    raw_test = load_dataset("openai/gsm8k", "main", split="test")
    raw_test = raw_test.shuffle(seed=args.seed)
    if args.eval_samples and args.eval_samples < len(raw_test):
        raw_test = raw_test.select(range(args.eval_samples))
    eval_dataset = raw_test.map(tokenize, batched=True, remove_columns=raw_test.column_names)
    print(f"Test size:  {len(eval_dataset):,} examples")

    # ── Training ──────────────────────────────────────────────────────────────
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        save_total_limit=None,      # keep ALL checkpoints for PRISM inference
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.save_steps,  # eval at every checkpoint
        bf16=True,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()
    print(f"\nDone. Checkpoints saved under: {args.output_dir}")
    print("Saved steps:", sorted(
        int(d.name.split("-")[-1])
        for d in __import__("pathlib").Path(args.output_dir).glob("checkpoint-*")
        if d.name.split("-")[-1].isdigit()
    ))


if __name__ == "__main__":
    main()
