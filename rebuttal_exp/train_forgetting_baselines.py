#!/usr/bin/env python3
"""
E2 / E3(b) — LoRA forgetting runs with the extended regularizer zoo.

Adds three canonical baselines on top of the paper's trace / replay pair
(pCi8-W5, G3T9-W3, 8VrD-W3/Q4, eQL6-W4):

    l2sp        L = L_CE + lambda * sum ||theta - theta_0||^2       (parameter space)
    ewc         L = L_CE + lambda * sum F_i (theta_i - theta_0i)^2  (Fisher-weighted)
    feature_kd  L = L_CE + lambda * ||Z_t - Z_0||_F^2 / ||Z_0||_F^2 (feature space)

plus `trace` / `replay` / `none` re-exposed with --seed and --ref_task
overrides, so multi-seed reruns (8VrD-W3) and the regularizer-side
reference-set sweep (8VrD-Q3, E3 part B-reg) all go through ONE entry point.

Everything else — LoRA config, data, collator, PRISM checkpoint callback,
schedules — is imported unchanged from train_forgetting_multitask.py, so runs
are protocol-identical to the paper.

EWC note: the Fisher diagonal is estimated on the reference set at adapter
initialisation, i.e. at the base model itself — the correct "old-task
optimum" for the pre-trained model. At that point LoRA-B matrices are zero,
so grads w.r.t. lora_A vanish and the Fisher mass sits on lora_B; this is a
property of EWC-on-LoRA-parameters, stated openly in E2.md. Fisher is
normalised to unit mean so lambda is on the same scale as l2sp.

Usage (repo root, GPU box):
    python rebuttal_exp/train_forgetting_baselines.py \
        --model meta-llama/Llama-3.1-8B --task truthfulqa \
        --method l2sp --lambda_reg 1e-3 --seed 42 --max_steps 300

Outputs: {output_root}/{method}/lam{lambda}/seed{seed}/{model_short}/{task}/
             prism_forgetting_metrics_{task}.json   (same schema as the paper)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from transformers import (AutoModelForCausalLM, AutoTokenizer,      # noqa: E402
                          Trainer, TrainingArguments)

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
from prism.models.extractors import LLMExtractor                    # noqa: E402


# ══════════════════════════════════════════════════════════════════════
# New baseline trainers (same scheduling/backward path as the paper's
# ShapeRegularizedTrainer: separate backward once per optimizer step)
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


class EWCTrainer(L2SPTrainer):
    """Diagonal-Fisher EWC on the trainable parameters."""

    LOG_KEY = "ewc"

    def __init__(self, *args, fisher: Dict[str, Tensor], **kwargs):
        super().__init__(*args, **kwargs)
        self._fisher = fisher

    def _penalty(self, name: str, p: Tensor, p0: Tensor) -> Tensor:
        F = self._fisher[name].to(p.device)
        return (F * (p - p0) ** 2).sum()


class FeatureKDTrainer(ShapeRegularizedTrainer):
    """Feature-space distillation: ||Z_t - Z_0||_F^2 / ||Z_0||_F^2 on D_ref.

    The simplest feature-preserving regularizer G3T9 asks for — identical
    machinery to the trace penalty, only the loss functional differs
    (unnormalised L2 matching instead of the scale-free 1 - Omega_I).
    """

    def _compute_shape_loss(self, model) -> Tuple[Tensor, float]:
        parts = []
        for batch in self._ref_dl:
            b = {k: v.to(self._device_str) for k, v in batch.items()}
            prompt_lens = b.pop("prompt_length", None)
            masks = b.get("attention_mask")
            out = model(**b, output_hidden_states=True)
            hidden = out.hidden_states[-1]
            parts.append(LLMExtractor._extract_z(hidden, masks, "concat",
                                                 prompt_lens))
        Z_P = torch.cat(parts, dim=0).float()
        Z_T = self.Z_T_ref.to(Z_P.device)
        denom = (Z_T ** 2).sum().clamp(min=1e-12)
        kd = ((Z_P - Z_T) ** 2).sum() / denom
        return kd, kd.item()

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if self._count > 0:
            logs["feature_kd"] = round(self._shape_sum / self._count, 8)
            self._shape_sum = self._omega_sum = 0.0
            self._count = 0
        Trainer.log(self, logs, *args, **kwargs)


# ══════════════════════════════════════════════════════════════════════
def estimate_fisher(model, ref_dl, device: str) -> Dict[str, Tensor]:
    """Diagonal empirical Fisher of the answer-only CE on D_ref, at the
    current (initial) adapter state, normalised to unit mean."""
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
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True, choices=list(TASK_CONFIGS))
    p.add_argument("--method", required=True,
                   choices=["none", "trace", "replay", "l2sp", "ewc",
                            "feature_kd", "layer_freeze"])
    p.add_argument("--lambda_reg", type=float, default=0.0)
    p.add_argument("--lora_top_layers", type=int, default=8,
                   help="layer_freeze only: apply LoRA to the top-K layers "
                        "and freeze the rest (the AC-named low-cost "
                        "continual-learning baseline)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=300,
                   help="paper analysis window: 300")
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
    p.add_argument("--ref_task", default=None,
                   help="reference-set source task (default: --task; "
                        "e.g. wikitext for the E3(b) domain sweep)")
    p.add_argument("--reg_samples", type=int, default=32)
    p.add_argument("--reg_batch_size", type=int, default=4)
    p.add_argument("--reg_max_length", type=int, default=512)
    p.add_argument("--reg_every_k", type=int, default=8)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--output_root", default=str(HERE / "out" / "E2"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.method not in ("none", "layer_freeze") and args.lambda_reg <= 0:
        sys.exit("--lambda_reg must be > 0 for a regularized method")
    ref_task = args.ref_task or args.task
    task_cfg = TASK_CONFIGS[args.task]
    task_max_length = task_cfg.get("max_length")
    if task_max_length is not None and args.max_length == 512:
        args.max_length = task_max_length
    lr = args.lr if args.lr is not None else \
        (1e-4 if "qwen" in args.model.lower() else 2e-4)

    model_short = args.model.split("/")[-1].lower()
    sweep_tag = (f"top{args.lora_top_layers}" if args.method == "layer_freeze"
                 else f"lam{args.lambda_reg:g}")
    output_dir = os.path.join(
        args.output_root, args.method, sweep_tag,
        f"seed{args.seed}", model_short, args.task,
    )
    os.makedirs(output_dir, exist_ok=True)
    eval_tasks = get_eval_tasks(args.task)

    experiment_config = {
        "script": "rebuttal_exp/train_forgetting_baselines.py",
        "method": args.method, "lambda_reg": args.lambda_reg,
        "ref_task": ref_task, "reg_samples": args.reg_samples,
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
    }
    print("=" * 78)
    for k, v in experiment_config.items():
        print(f"  {k:<22s}: {v}")
    print(f"  output_dir            : {output_dir}")
    print("=" * 78)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True,
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

    # ── Reference set (fixed; seed offset matches the paper script) ────
    needs_ref = args.method in ("trace", "replay", "feature_kd", "ewc")
    ref_dataloader = None
    Z_T_ref: Optional[Tensor] = None
    if needs_ref:
        ref_dataloader = load_task_data(
            ref_task, split="test", num_samples=args.reg_samples,
            batch_size=args.reg_batch_size, tokenizer=tokenizer,
            max_length=min(args.max_length, args.reg_max_length),
            seed=args.seed + 1000,
        )
    if args.method in ("trace", "feature_kd"):
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
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    fisher = None
    if args.method == "ewc":
        print("\nEstimating diagonal Fisher on the reference set ...")
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        fisher = estimate_fisher(model, ref_dataloader, device)

    # ── Data + callback + training args (paper-identical) ──────────────
    train_dataset = build_dataset(args.task, "train", tokenizer,
                                  max_length=args.max_length,
                                  max_samples=task_cfg["max_train_samples"],
                                  seed=args.seed)
    eval_dataset = build_dataset(args.task, "eval", tokenizer,
                                 max_length=args.max_length,
                                 max_samples=task_cfg["max_eval_samples"],
                                 seed=args.seed)
    print(f"  Train {len(train_dataset):,} / eval {len(eval_dataset):,}")

    prism_callback = PRISMCheckpointCallback(
        model=model, base_features=base_features,
        eval_dataloaders=eval_dataloaders, extractor=extractor,
        trained_task=args.task, eval_tasks=eval_tasks, model_id=args.model,
        output_dir=output_dir, device=device,
        experiment_config=experiment_config, K_theory=K_theory,
    )
    collator = AnswerOnlyDataCollator(tokenizer=tokenizer)
    try:
        import bitsandbytes  # noqa: F401
        optim = "paged_adamw_8bit"
    except ImportError:
        optim = "adamw_torch"

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

    common = dict(model=model, args=training_args,
                  train_dataset=train_dataset, eval_dataset=eval_dataset,
                  data_collator=collator, callbacks=[prism_callback])
    reg = dict(ref_dataloader=ref_dataloader, lambda_shape=args.lambda_reg,
               reg_every_k=args.reg_every_k, device_str=device)

    if args.method == "trace":
        trainer = ShapeRegularizedTrainer(**common, Z_T_ref=Z_T_ref, **reg)
    elif args.method == "replay":
        trainer = ReplayCETrainer(**common, Z_T_ref=None, **reg)
    elif args.method == "feature_kd":
        trainer = FeatureKDTrainer(**common, Z_T_ref=Z_T_ref, **reg)
    elif args.method == "l2sp":
        trainer = L2SPTrainer(**common, Z_T_ref=None, **reg)
    elif args.method == "ewc":
        trainer = EWCTrainer(**common, Z_T_ref=None, fisher=fisher, **reg)
    else:  # none / layer_freeze — plain CE training
        trainer = Trainer(**common)

    print(f"\nTraining ({args.method}, lambda={args.lambda_reg}) ...")
    trainer.train()
    print(f"\nDone. PRISM log: {prism_callback.json_path}")


if __name__ == "__main__":
    main()
