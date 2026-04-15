#!/usr/bin/env python3
"""
Stage 2 — PRISM forgetting inference on LoRA checkpoints.

Loads each LoRA checkpoint produced by Stage 1 (train_forgetting_multitask.py),
merges the adapter into the base model, then computes PRISM geometric metrics
and empirical loss on every evaluation task.

Output: a JSON file per (model, trained_task) with all checkpoint × eval_task
metrics, ready for plotting forgetting curves and scatter plots.

Usage:
    # Single run (specified via YAML config):
    python infer_forgetting_multitask.py --config configs/forgetting_multitask.yaml

    # Override via CLI:
    python infer_forgetting_multitask.py --config configs/forgetting_multitask.yaml \\
        --base_model meta-llama/Llama-3.1-8B \\
        --checkpoint_dir checkpoints/forgetting_multitask/llama-3.1-8b/gsm8k \\
        --eval_tasks gsm8k mmlu arc

    # Standalone (no config file):
    python infer_forgetting_multitask.py \\
        --base_model meta-llama/Llama-3.1-8B \\
        --checkpoint_dir checkpoints/forgetting_multitask/llama-3.1-8b/gsm8k \\
        --eval_tasks gsm8k mmlu arc squad triviaqa \\
        --output_dir results/forgetting_multitask/llama-3.1-8b/trained_gsm8k
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from torch import Tensor

from prism.core.bounds import UnifiedBound
from prism.core.metrics import PRISMMetrics, PRISMResult
from prism.data.loaders import get_task_metadata, load_task_data
from prism.models.extractors import LLMExtractor


# ── Helpers ───────────────────────────────────────────────────────────────

def sorted_checkpoints(checkpoint_dir: str) -> List[str]:
    """Return checkpoint directories sorted by step number."""
    ckpt_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    pairs = []
    for p in ckpt_paths:
        name = os.path.basename(p)
        parts = name.split("-")
        if len(parts) >= 2 and parts[-1].isdigit():
            pairs.append((int(parts[-1]), p))
    pairs.sort(key=lambda x: x[0])
    return [p for _, p in pairs]


def step_from_path(ckpt_path: str) -> int:
    """Extract step number from checkpoint-NNN directory name."""
    name = os.path.basename(ckpt_path.rstrip("/"))
    return int(name.split("-")[-1])


def load_base_model(model_id: str, device_map="auto"):
    """Load a base model in bf16."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_merged_model(base_model_id: str, adapter_path: str, device_map="auto"):
    """Load base model, apply LoRA adapter, merge, and return a standard model."""
    from transformers import AutoModelForCausalLM
    try:
        from peft import PeftModel
    except ImportError:
        print("ERROR: peft is required. Install with: pip install peft", file=sys.stderr)
        sys.exit(1)

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    return model


def free_model(model):
    """Delete a model and free GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Feature extraction + loss (single forward pass) ──────────────────────

def extract_all(
    model,
    dataloader,
    extractor: LLMExtractor,
    device: str,
    z_mode: str,
    chunk_size: int = 2048,
) -> Tuple[Tensor, Tensor, Dict]:
    """Extract features Z, head H, and loss stats in a single forward pass.

    Returns:
        Z: (n_tokens, d) feature matrix
        H: (d, vocab) head weights
        loss_stats: dict with 'losses', 'answer_losses', etc.
    """
    Z, loss_stats = extractor.extract_features_and_loss_per_sample(
        model, dataloader, device,
        z_mode=z_mode,
        chunk_size=chunk_size,
    )
    H = extractor.extract_head(model)
    return Z, H, loss_stats


# ── PRISM metric computation ─────────────────────────────────────────────

def compute_prism_metrics(
    Z_T: Tensor, H_T: Tensor,
    Z_P: Tensor, H_P: Tensor,
    label: str,
) -> PRISMResult:
    """Compute PRISM metrics with Procrustes alignment (W = W_opt)."""
    result = PRISMMetrics.compute_all(
        Z_T.float(), H_T.float(),
        Z_P.float(), H_P.float(),
        W=None,         # Procrustes W_opt (rotational regime)
        label=label,
    )
    # Fill risk bound with K_feat=1, K_pred=1 as placeholders.
    # The actual K values are computed from H_T if needed.
    UnifiedBound.fill_result(result, K_feat=1.0, K_pred=1.0)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2: PRISM forgetting inference on LoRA checkpoints",
    )
    p.add_argument("--config", "-c", type=str, default=None,
                   help="YAML config file (optional; CLI args override config)")
    p.add_argument("--base_model", type=str, default=None,
                   help="HuggingFace base model ID")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Directory containing checkpoint-* subdirectories")
    p.add_argument("--eval_tasks", nargs="+", default=None,
                   help="Evaluation tasks (e.g. arc mmlu squad triviaqa gsm8k)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory for results JSON")
    p.add_argument("--num_samples", type=int, default=256,
                   help="Number of samples per eval task")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Batch size for feature extraction")
    p.add_argument("--max_length", type=int, default=512,
                   help="Maximum sequence length")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge YAML config with CLI overrides. CLI takes precedence."""
    config: Dict[str, Any] = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    # CLI overrides
    target = config.setdefault("target", {})
    proxy = config.setdefault("proxy", {})
    data = config.setdefault("data", {})
    output = config.setdefault("output", {})

    if args.base_model:
        target["model"] = args.base_model
    if args.checkpoint_dir:
        proxy["checkpoint_dir"] = args.checkpoint_dir
    if args.eval_tasks:
        data["eval_tasks"] = args.eval_tasks
    if args.output_dir:
        output["dir"] = args.output_dir
    if args.num_samples:
        data["num_samples"] = args.num_samples
    if args.batch_size:
        data["batch_size"] = args.batch_size
    if args.max_length:
        data["max_length"] = args.max_length

    config.setdefault("seed", args.seed)
    config.setdefault("device", args.device)

    # Validate required fields
    if not target.get("model"):
        print("ERROR: base model not specified (--base_model or target.model in config)",
              file=sys.stderr)
        sys.exit(1)
    if not proxy.get("checkpoint_dir"):
        print("ERROR: checkpoint_dir not specified (--checkpoint_dir or proxy.checkpoint_dir)",
              file=sys.stderr)
        sys.exit(1)

    # Defaults
    data.setdefault("eval_tasks", ["arc", "mmlu", "squad", "triviaqa", "gsm8k"])
    data.setdefault("num_samples", 256)
    data.setdefault("batch_size", 4)
    data.setdefault("max_length", 512)
    output.setdefault("dir", "./results/forgetting_multitask")

    return config


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    config = resolve_config(args)

    base_model_id = config["target"]["model"]
    checkpoint_dir = config["proxy"]["checkpoint_dir"]
    eval_tasks = config["data"]["eval_tasks"]
    num_samples = config["data"]["num_samples"]
    batch_size = config["data"]["batch_size"]
    max_length = config["data"]["max_length"]
    output_dir = config["output"]["dir"]
    seed = config.get("seed", 42)
    device = config.get("device", "cuda")

    os.makedirs(output_dir, exist_ok=True)
    extractor = LLMExtractor()

    # Discover checkpoints
    checkpoints = sorted_checkpoints(checkpoint_dir)
    if not checkpoints:
        print(f"ERROR: No checkpoints found in {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    # Derive trained_task from checkpoint_dir path
    trained_task = os.path.basename(checkpoint_dir.rstrip("/"))

    print(f"{'=' * 72}")
    print(f"  PRISM Forgetting — Stage 2: Inference")
    print(f"{'=' * 72}")
    print(f"  Base model    : {base_model_id}")
    print(f"  Trained task  : {trained_task}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Checkpoints   : {len(checkpoints)}")
    print(f"  Eval tasks    : {eval_tasks}")
    print(f"  Samples/task  : {num_samples}")
    print(f"  Output        : {output_dir}")
    print(f"{'=' * 72}")

    # ── Load tokenizer ───────────────────────────────────────────────────
    from transformers import AutoTokenizer
    print(f"\nLoading tokenizer from {base_model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Build dataloaders for all eval tasks ─────────────────────────────
    dataloaders = {}
    for task in eval_tasks:
        meta = get_task_metadata(task)
        print(f"  Loading eval data: {task} (n={num_samples}, z_mode={meta['z_mode']}) ...")
        dl = load_task_data(
            task, split="test", num_samples=num_samples,
            batch_size=batch_size, tokenizer=tokenizer,
            max_length=max_length, seed=seed,
        )
        dataloaders[task] = (dl, meta)

    # ── Phase 1: Extract base model (target) features ────────────────────
    print(f"\n{'─' * 40}")
    print(f"  Phase 1: Extracting base model features")
    print(f"{'─' * 40}")

    base_data: Dict[str, Dict] = {}
    tensor_device = "cuda" if device == "auto" else device

    print(f"  Loading base model: {base_model_id} ...")
    t0 = time.time()
    base_model = load_base_model(base_model_id, device_map={"": 0})

    for task in eval_tasks:
        dl, meta = dataloaders[task]
        z_mode = meta["z_mode"]
        print(f"    Extracting features for {task} (z_mode={z_mode}) ...")

        Z_T, H_T, loss_stats_T = extract_all(
            base_model, dl, extractor, tensor_device,
            z_mode=z_mode,
        )
        # Move to CPU to free GPU for proxy models
        base_data[task] = {
            "Z": Z_T.cpu(),
            "H": H_T.cpu(),
            "loss_full": loss_stats_T["losses"].mean().item(),
            "loss_answer": (
                loss_stats_T["answer_losses"].mean().item()
                if loss_stats_T.get("answer_losses") is not None
                else None
            ),
        }
        print(f"      Z shape: {Z_T.shape}, loss: {base_data[task]['loss_full']:.4f}")

    elapsed_base = time.time() - t0
    print(f"  Base model features extracted in {elapsed_base:.1f}s")
    free_model(base_model)

    # ── Phase 2: Iterate over checkpoints ────────────────────────────────
    print(f"\n{'─' * 40}")
    print(f"  Phase 2: Evaluating {len(checkpoints)} checkpoints")
    print(f"{'─' * 40}")

    all_results: List[Dict[str, Any]] = []

    for ckpt_idx, ckpt_path in enumerate(checkpoints):
        step = step_from_path(ckpt_path)
        print(f"\n  [{ckpt_idx + 1}/{len(checkpoints)}] checkpoint-{step}")

        # Load and merge LoRA adapter
        t0 = time.time()
        print(f"    Loading merged model ...")
        try:
            proxy_model = load_merged_model(
                base_model_id, ckpt_path, device_map={"": 0},
            )
        except Exception as e:
            print(f"    ERROR loading {ckpt_path}: {e}")
            continue

        # Evaluate on all tasks
        for task in eval_tasks:
            dl, meta = dataloaders[task]
            z_mode = meta["z_mode"]

            Z_P, H_P, loss_stats_P = extract_all(
                proxy_model, dl, extractor, tensor_device,
                z_mode=z_mode,
            )

            loss_proxy_full = loss_stats_P["losses"].mean().item()
            loss_proxy_answer = (
                loss_stats_P["answer_losses"].mean().item()
                if loss_stats_P.get("answer_losses") is not None
                else None
            )

            # Compute PRISM metrics
            Z_T = base_data[task]["Z"]
            H_T = base_data[task]["H"]
            loss_target_full = base_data[task]["loss_full"]
            loss_target_answer = base_data[task]["loss_answer"]

            prism = compute_prism_metrics(
                Z_T, H_T, Z_P.cpu(), H_P.cpu(),
                label=f"step-{step}_{task}",
            )

            delta_risk_full = abs(loss_proxy_full - loss_target_full)
            delta_risk_answer = (
                abs(loss_proxy_answer - loss_target_answer)
                if loss_proxy_answer is not None and loss_target_answer is not None
                else None
            )

            # Primary |ΔR|: answer-only (aligned with Z extraction region)
            primary_dr = delta_risk_answer if delta_risk_answer is not None else delta_risk_full

            result_row = {
                "base_model": base_model_id,
                "trained_task": trained_task,
                "eval_task": task,
                "step": step,
                "checkpoint": ckpt_path,
                # Geometry (paper notation)
                "omega": prism.omega,
                "rho_T": prism.rho_target,
                "rho_P": prism.rho_proxy,
                "scale": prism.scale_mismatch,
                "shape": prism.shape_mismatch,
                "delta": prism.feature_error,
                "gamma": prism.head_discrepancy,
                "bound_feature": prism.risk_bound_feature,
                "bound_head": prism.risk_bound_head,
                "bound_total": prism.risk_bound_total,
                # Empirical risk — answer-only (primary)
                "loss_T": loss_target_answer,
                "loss_P": loss_proxy_answer,
                "delta_risk": primary_dr,
                # Empirical risk — full-sequence (supplementary)
                "loss_T_full": loss_target_full,
                "loss_P_full": loss_proxy_full,
                "delta_risk_full": delta_risk_full,
                # Bound validation (answer-only)
                "bound_holds": (
                    prism.risk_bound_total >= primary_dr
                    if prism.risk_bound_total is not None
                    else None
                ),
            }
            all_results.append(result_row)

            # Log summary (use answer-only as primary)
            is_same = (task == trained_task)
            marker = " (trained)" if is_same else ""
            lt = loss_target_answer if loss_target_answer is not None else loss_target_full
            lp = loss_proxy_answer if loss_proxy_answer is not None else loss_proxy_full
            print(
                f"    {task}{marker}: "
                f"Ω={prism.omega:.4f}  δ={prism.feature_error:.4f}  γ={prism.head_discrepancy:.4f}  "
                f"|ΔR|={primary_dr:.4f}  Loss_T={lt:.4f}  Loss_P={lp:.4f}"
            )

        elapsed_ckpt = time.time() - t0
        print(f"    (checkpoint done in {elapsed_ckpt:.1f}s)")
        free_model(proxy_model)

    # ── Save results ─────────────────────────────────────────────────────
    result_path = os.path.join(output_dir, "forgetting_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'=' * 72}")
    print(f"  Results saved to: {result_path}")
    print(f"  Total evaluations: {len(all_results)}")
    print(f"{'=' * 72}")

    # ── Print summary table ──────────────────────────────────────────────
    print(f"\n  Summary (final checkpoint, step {step_from_path(checkpoints[-1])}):")
    final_step = step_from_path(checkpoints[-1])
    header = (
        f"  {'Task':<11s}  {'ρ_T':>7s}  {'ρ_P':>7s}  {'Ω':>10s}"
        f"  {'δ':>8s}  {'γ':>8s}  {'Bound':>8s}"
        f"  {'Loss_T':>8s}  {'Loss_P':>8s}  {'|ΔR|':>8s}  {'Holds':>5s}"
    )
    print(header)
    print(f"  {'─' * (len(header) - 2)}")
    for r in all_results:
        if r["step"] == final_step:
            marker = " *" if r["eval_task"] == trained_task else "  "
            bound_s = f"{r['bound_total']:8.4f}" if r["bound_total"] else "       —"
            holds = r.get("bound_holds")
            holds_s = "  yes" if holds is True else "   no" if holds is False else "    —"
            lt = r["loss_T"] if r["loss_T"] is not None else r["loss_T_full"]
            lp = r["loss_P"] if r["loss_P"] is not None else r["loss_P_full"]
            print(
                f"  {r['eval_task']:<9s}{marker}"
                f"  {r['rho_T']:7.2f}  {r['rho_P']:7.2f}  {r['omega']:10.6f}"
                f"  {r['delta']:8.4f}  {r['gamma']:8.4f}  {bound_s}"
                f"  {lt:8.4f}  {lp:8.4f}  {r['delta_risk']:8.4f}  {holds_s}"
            )


if __name__ == "__main__":
    main()
