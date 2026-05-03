#!/usr/bin/env python3
"""
LoRA fine-tune a base model on a single task and compute PRISM forgetting
metrics online at every checkpoint (no separate inference stage needed).

The paper (Sec.~5.4) fine-tunes on **TruthfulQA** and **BBQ**; additional
task configurations are kept in TASK_CONFIGS for users who wish to extend
the experiment to other fine-tuning sources, but the paper's reported
results use only those two.

Models in the paper: meta-llama/Meta-Llama-3.1-8B, Qwen/Qwen3-8B-Base.
The script accepts any causal LM whose backbone exposes hidden states.

Under LoRA the lm_head is frozen (H_t = H_0), so PRISM's head divergence
term vanishes under identity alignment, isolating forgetting entirely in
backbone geometry (Eq.~8 in the paper).

Usage:
    python train_forgetting.py \\
        --model meta-llama/Meta-Llama-3.1-8B --task truthfulqa

    python train_forgetting.py \\
        --model Qwen/Qwen3-8B-Base --task bbq \\
        --lambda_shape 1.0
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


def _format_truthfulqa(row: dict) -> str:
    return f"Question: {row['question']}\nAnswer: {row['best_answer']}"


def _format_bbq(row: dict) -> str:
    choices = row["choices"]
    labels = ["A", "B", "C"]
    opts = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
    ans_label = labels[row["gold_index"]]
    return f"Context: {row['context']}\nQuestion: {row['question']}\n{opts}\nAnswer: {ans_label}"


def _format_social_iqa(row: dict) -> str:
    answers = [row["answerA"], row["answerB"], row["answerC"]]
    labels = ["A", "B", "C"]
    opts = "\n".join(f"{labels[i]}. {answers[i]}" for i in range(3))
    ans_label = labels[int(row["label"]) - 1]  # 1-indexed → 0-indexed
    return f"Context: {row['context']}\nQuestion: {row['question']}\n{opts}\nAnswer: {ans_label}"


FORMATTERS = {
    "arc": _format_arc,
    "mmlu": _format_mmlu,
    "squad": _format_squad,
    "triviaqa": _format_triviaqa,
    "gsm8k": _format_gsm8k,
    "truthfulqa": _format_truthfulqa,
    "bbq": _format_bbq,
    "social_iqa": _format_social_iqa,
}


# ── Prompt-only formatters (question prefix, no answer) ──────────────────
# Used to compute prompt_length so that loss is only on answer tokens.
# Must be a strict prefix of the corresponding FORMATTER output.

def _prompt_arc(row: dict) -> str:
    q = row["question"]
    labels = row["choices"]["label"]
    texts = row["choices"]["text"]
    opts = "\n".join(f"{labels[i]}. {texts[i]}" for i in range(len(labels)))
    return f"Question: {q}\n{opts}\nAnswer:"


def _prompt_mmlu(row: dict) -> str:
    q = row["question"]
    choices = row["choices"]
    labels = ["A", "B", "C", "D"]
    opts = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
    return f"Question: {q}\n{opts}\nAnswer:"


def _prompt_squad(row: dict) -> str:
    return f"Context: {row['context']}\nQuestion: {row['question']}\nAnswer:"


def _prompt_triviaqa(row: dict) -> str:
    return f"Question: {row['question']}\nAnswer:"


def _prompt_gsm8k(row: dict) -> str:
    return f"Question: {row['question']}\nAnswer:"


def _prompt_truthfulqa(row: dict) -> str:
    return f"Question: {row['question']}\nAnswer:"


def _prompt_bbq(row: dict) -> str:
    choices = row["choices"]
    labels = ["A", "B", "C"]
    opts = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
    return f"Context: {row['context']}\nQuestion: {row['question']}\n{opts}\nAnswer:"


def _prompt_social_iqa(row: dict) -> str:
    answers = [row["answerA"], row["answerB"], row["answerC"]]
    labels = ["A", "B", "C"]
    opts = "\n".join(f"{labels[i]}. {answers[i]}" for i in range(3))
    return f"Context: {row['context']}\nQuestion: {row['question']}\n{opts}\nAnswer:"


PROMPT_FORMATTERS = {
    "arc": _prompt_arc,
    "mmlu": _prompt_mmlu,
    "squad": _prompt_squad,
    "triviaqa": _prompt_triviaqa,
    "gsm8k": _prompt_gsm8k,
    "truthfulqa": _prompt_truthfulqa,
    "bbq": _prompt_bbq,
    "social_iqa": _prompt_social_iqa,
}


# ── Answer-only data collator ────────────────────────────────────────────

class AnswerOnlyDataCollator:
    """Collator that masks prompt tokens in labels → answer-only loss.

    Each dataset sample must contain a ``prompt_length`` field indicating
    how many leading tokens belong to the prompt.  The collator:

      1. Pops ``prompt_length`` from each sample.
      2. Pads ``input_ids`` / ``attention_mask`` dynamically per batch.
      3. Creates ``labels`` = ``input_ids``.clone() with:
         - labels[:prompt_length] = -100   (ignore prompt)
         - labels[padding]        = -100   (ignore padding)

    Result: the model only back-propagates through answer tokens.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list) -> dict:
        # Pop prompt_length before padding (tokenizer.pad doesn't know it)
        prompt_lengths = [f.pop("prompt_length") for f in features]

        # Dynamic padding
        batch = self.tokenizer.pad(
            features, return_tensors="pt", padding=True,
        )

        # Build labels: answer-only
        labels = batch["input_ids"].clone()
        for i, pl in enumerate(prompt_lengths):
            labels[i, :pl] = -100
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels

        return batch


# ── Shape-regularized Trainer (W=I, trace form) ────────────────────────────

class ShapeRegularizedTrainer(Trainer):
    """HF Trainer with PRISM shape regularization: L = L_CE + λ·(1 − Ω_I).

    Uses trace-based Ω_I (identity alignment, W=I):
        Ω_I = tr(Z_T^T Z_P) / (‖Z_T‖_F · ‖Z_P‖_F)

    Every ``reg_every_k`` micro-steps, a forward pass is run on a fixed
    reference set to extract Z_P, and the shape loss 1 − Ω_I is backprop'd
    into the current parameter gradient buffer.  Z_T (base model features)
    is pre-computed and frozen.

    CE and shape losses have **separate backward paths** (see ``training_step``):
    CE goes through parent ``Trainer``'s normal flow and is divided by
    ``gradient_accumulation_steps`` before backward, so the per-micro-batch
    CE gradient averages correctly.  Shape loss is backprop'd once per
    optimizer step with **no** grad-accumulation division, so the gradient
    contribution is exactly ``self._lambda`` × ∂(1-Ω_I)/∂θ — the nominal λ
    matches the effective signal.
    """

    def __init__(
        self,
        *args,
        Z_T_ref: Optional[Tensor] = None,   # required for shape; unused by ReplayCETrainer
        ref_dataloader,
        lambda_shape: float = 0.1,
        reg_every_k: int = 8,
        device_str: str = "cuda",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.Z_T_ref = Z_T_ref          # (n_tokens, d) CPU float32 detached, or None
        self._ref_dl = ref_dataloader
        self._lambda = lambda_shape
        self._reg_k = reg_every_k
        self._device_str = device_str
        self._micro = 0

        # Logging accumulators (reset at each log() call)
        self._shape_sum = 0.0
        self._omega_sum = 0.0
        self._count = 0

    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Pure CE loss.  Shape regularizer lives in ``training_step`` so it
        can bypass HF's gradient-accumulation division (see class docstring)."""
        outputs = model(**inputs)
        ce_loss = outputs.loss
        if return_outputs:
            return ce_loss, outputs
        return ce_loss

    # ------------------------------------------------------------------
    def training_step(self, model, inputs, *args, **kwargs):
        """Run parent's CE training step, then add a separately-scaled
        shape-loss backward once per optimizer step.

        Parent's ``training_step`` does CE forward + backward with the
        standard ``loss /= gradient_accumulation_steps`` division.  We then
        compute the shape loss and call ``accelerator.backward`` directly
        on ``λ · (1 − Ω_I)`` — no grad-accum division — so the shape
        gradient is accumulated into the same buffer as the averaged CE
        gradient at its nominal weight.
        """
        loss_tensor = super().training_step(model, inputs, *args, **kwargs)

        if self._lambda > 0 and (self._micro % self._reg_k == 0):
            with self.compute_loss_context_manager():
                shape_loss, omega_val = self._compute_shape_loss(model)
            self.accelerator.backward(self._lambda * shape_loss)
            self._shape_sum += shape_loss.item()
            self._omega_sum += omega_val
            self._count += 1
        self._micro += 1

        return loss_tensor

    # ------------------------------------------------------------------
    def _compute_shape_loss(self, model) -> Tuple[Tensor, float]:
        """Differentiable shape loss: 1 − Ω_I on reference data.

        Forward-passes the reference DataLoader (with gradients enabled)
        to build Z_P, then computes the trace-based Ω_I against the
        frozen Z_T_ref.

        Returns:
            shape_loss: scalar tensor with grad_fn
            omega_val:  float for logging
        """
        Z_P_parts: List[Tensor] = []

        for batch in self._ref_dl:
            batch_gpu = {k: v.to(self._device_str) for k, v in batch.items()}
            prompt_lens = batch_gpu.pop("prompt_length", None)
            masks = batch_gpu.get("attention_mask")

            out = model(**batch_gpu, output_hidden_states=True)
            hidden = out.hidden_states[-1]                   # (bsz, seq, d)

            z = LLMExtractor._extract_z(hidden, masks, "concat", prompt_lens)
            Z_P_parts.append(z)

        Z_P = torch.cat(Z_P_parts, dim=0).float()           # (n_tok, d) GPU
        Z_T = self.Z_T_ref.to(Z_P.device)                   # (n_tok, d) GPU

        # Trace-based Ω_I (W = I): Frobenius cosine similarity
        trace = (Z_T * Z_P).sum()                            # ⟨Z_T, Z_P⟩_F
        denom = Z_T.norm("fro") * Z_P.norm("fro")
        omega_I = trace / denom.clamp(min=1e-12)

        shape_loss = 1.0 - omega_I
        return shape_loss, omega_I.item()

    # ------------------------------------------------------------------
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """Inject shape regularizer metrics into log entries.

        HF Trainer.log() added ``start_time`` as a second positional arg
        in newer versions; accept *args/**kwargs and pass them through so
        this override stays compatible across transformers releases.
        """
        if self._count > 0:
            logs["shape_loss"] = round(self._shape_sum / self._count, 6)
            logs["omega_ref"] = round(self._omega_sum / self._count, 6)
            self._shape_sum = 0.0
            self._omega_sum = 0.0
            self._count = 0
        super().log(logs, *args, **kwargs)


# ── Replay-CE Trainer (data-replay baseline) ──────────────────────────────

class ReplayCETrainer(ShapeRegularizedTrainer):
    """Data-replay baseline: L = L_CE + λ · CE_ref.

    Same scheduling and backward path as ``ShapeRegularizedTrainer`` (every
    optimizer step, separate backward outside grad-accum division), but the
    regularizer is the next-token cross-entropy on a fixed reference set
    rather than 1 − Ω_I. This is the apples-to-apples baseline for the
    PRISM shape-regularizer claim: same 32 reference instances, same compute,
    only the loss form differs.

    Z_T_ref is unused (CE only depends on the current-model output).
    Prompt tokens are masked from the CE target (set to -100) so the loss
    measures response-token reconstruction, matching how the fine-tuning CE
    is computed by ``AnswerOnlyDataCollator``.
    """

    def _compute_shape_loss(self, model) -> Tuple[Tensor, float]:
        """Override: mean next-token CE on the reference set (response tokens only)."""
        total = torch.zeros((), device=self._device_str)
        n_batches = 0
        for batch in self._ref_dl:
            b = {k: v.to(self._device_str) for k, v in batch.items()}
            prompt_lens = b.pop("prompt_length", None)
            labels = b["input_ids"].clone()
            if prompt_lens is not None:
                # Mask prompt tokens so CE measures response-token loss only.
                for i, plen in enumerate(prompt_lens):
                    labels[i, :int(plen)] = -100
            attn = b.get("attention_mask")
            if attn is not None:
                labels = labels.masked_fill(attn == 0, -100)
            b["labels"] = labels
            out = model(**b)
            total = total + out.loss
            n_batches += 1
        replay_ce = total / max(n_batches, 1)
        # Second return slot is repurposed for logging (no Ω here).
        return replay_ce, replay_ce.item()

    # ------------------------------------------------------------------
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """Rename log keys: ``shape_loss``/``omega_ref`` → ``replay_ce``."""
        if self._count > 0:
            logs["replay_ce"] = round(self._shape_sum / self._count, 6)
            self._shape_sum = 0.0
            self._omega_sum = 0.0
            self._count = 0
        # Skip ShapeRegularizedTrainer.log to avoid duplicate key injection;
        # call grandparent (HF Trainer) directly.
        Trainer.log(self, logs, *args, **kwargs)


# ── Task-specific dataset configuration ───────────────────────────────────

TASK_CONFIGS = {
    # ── New tasks (safety / truthfulness / social reasoning) ─────────
    "truthfulqa": {
        "hf_id": "truthful_qa",
        "hf_subset": "generation",
        "train_split": "validation[:80%]",   # only split; first 80% for train (~653)
        "eval_split": "validation[80%:]",    # last 20% for eval (~164)
        "max_train_samples": None,
        "max_eval_samples": None,
        "default_max_steps": 700,
        "default_save_steps": 25,
    },
    "bbq": {
        "hf_id": "lighteval/bbq_helm",
        "hf_subset": "all",
        "train_split": "test[:80%]",         # only split; first 80% for train (~800)
        "eval_split": "test[80%:]",          # last 20% for eval (~200)
        "max_train_samples": None,
        "max_eval_samples": None,
        "default_max_steps": 700,
        "default_save_steps": 25,
    },
    "social_iqa": {
        "hf_id": "allenai/social_i_qa",
        "hf_subset": None,
        "train_split": "train",
        "eval_split": "validation",
        "max_train_samples": 8000,
        "max_eval_samples": 256,
        "default_max_steps": 1500,
        "default_save_steps": 25,
    },
    # ── Original tasks ───────────────────────────────────────────────
    "arc": {
        "hf_id": "allenai/ai2_arc",
        "hf_subset": "ARC-Challenge",
        "train_split": "train",
        "eval_split": "test",
        "max_train_samples": None,   # use all 1,119
        "max_eval_samples": 256,
        "default_max_steps": 700,
        "default_save_steps": 25,
    },
    "mmlu": {
        "hf_id": "cais/mmlu",
        "hf_subset": "all",
        "train_split": "auxiliary_train",
        "eval_split": "validation",
        "max_train_samples": 8000,
        "max_eval_samples": 256,
        "default_max_steps": 1500,
        "default_save_steps": 25,
    },
    "squad": {
        "hf_id": "rajpurkar/squad",
        "hf_subset": None,
        "train_split": "train",
        "eval_split": "validation",
        "max_train_samples": 8000,
        "max_eval_samples": 256,
        "default_max_steps": 1500,
        "default_save_steps": 25,
    },
    "triviaqa": {
        "hf_id": "trivia_qa",
        "hf_subset": "rc.nocontext",
        "train_split": "train",
        "eval_split": "validation",
        "max_train_samples": 8000,
        "max_eval_samples": 256,
        "default_max_steps": 1500,
        "default_save_steps": 25,
    },
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "hf_subset": "main",
        "train_split": "train",
        "eval_split": "test",
        "max_train_samples": None,   # use all 7,473
        "max_eval_samples": 256,
        "default_max_steps": 1400,
        "default_save_steps": 25,
    },
}

BASE_EVAL_TASKS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]


def get_eval_tasks(trained_task: str) -> list:
    """Return the downstream eval task list.

    Forgetting is measured on benchmarks **disjoint** from the fine-tuning
    task, so we evaluate only on BASE_EVAL_TASKS regardless of which task
    was used for fine-tuning. (If the trained task happens to be in
    BASE_EVAL_TASKS, it is excluded from the eval list to keep the
    evaluation strictly downstream.)
    """
    return [t for t in BASE_EVAL_TASKS if t != trained_task]

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ======================================================================
# PRISM Checkpoint Callback
# ======================================================================

class PRISMCheckpointCallback(TrainerCallback):
    """Evaluate PRISM forgetting metrics on eval tasks at each checkpoint.

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
        eval_tasks: List[str],
        model_id: str,
        output_dir: str,
        device: str,
        experiment_config: Dict[str, Any],
        K_theory: Dict[str, float],
    ):
        super().__init__()
        self.model = model
        self.base_features = base_features      # {task: {Z, H, loss_full, loss_answer}}
        self.eval_dataloaders = eval_dataloaders  # {task: (dataloader, z_mode)}
        self.extractor = extractor
        self.trained_task = trained_task
        self.eval_tasks = eval_tasks             # tasks to evaluate at each checkpoint
        self.model_id = model_id
        self.output_dir = output_dir
        self.device = device
        self.experiment_config = experiment_config

        # Tight Lipschitz constants for paper Eq. 8 bound (see Appendix A):
        #   K_feat = max_{j,k} ||h_j - h_k||_2   (simplex polarisation)
        #   K_pred = sqrt(2)                     (CE gradient in logit space)
        # Stored once; H_T is the frozen base head and is identical across tasks.
        self.K_theory = K_theory
        self.K_feat = K_theory["K_feat"]
        self.K_pred = K_theory["K_pred"]

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
            total_tasks = len(self.eval_tasks)
            for idx, task in enumerate(self.eval_tasks, start=1):
                dl, z_mode = self.eval_dataloaders[task]
                marker = " *" if task == self.trained_task else ""
                task_t0 = time.time()
                print(f"  [{idx}/{total_tasks}] {task:<10s} (z_mode={z_mode}){marker} ... ", end="", flush=True)

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

                # PRISM metrics (identity alignment, W = I)
                d = Z_T.shape[1]
                prism = PRISMMetrics.compute_all(
                    Z_T.float(), H_T.float(),
                    Z_P.float().cpu(), H_P.float().cpu(),
                    W=torch.eye(d, dtype=Z_T.dtype),
                    label=f"step-{step}_{task}",
                )
                UnifiedBound.fill_result(
                    prism, K_feat=self.K_feat, K_pred=self.K_pred)

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

                # Primary |ΔR|: use answer-only loss because Z is extracted
                # from answer-region tokens only (concat mode + prompt mask).
                # The bound applies to the same token positions as Z.
                primary_dr = delta_risk_answer if delta_risk_answer is not None else delta_risk_full
                loss_s = (
                    f"loss_answer={loss_P_answer:.4f}"
                    if loss_P_answer is not None
                    else f"loss_full={loss_P_full:.4f}"
                )
                bound_s = (
                    f"Bound={prism.risk_bound_total:.4f}"
                    if prism.risk_bound_total is not None
                    else "Bound=—"
                )
                elapsed = time.time() - task_t0
                print(
                    f"Z={list(Z_P.shape)}  {loss_s}  |ΔR|={primary_dr:.4f}  "
                    f"{bound_s}  ({elapsed:.1f}s)"
                )

                task_results[task] = {
                    # Geometry  (paper notation)
                    "omega": prism.omega,                       # Ω_I  (identity-aligned similarity)
                    "rho_T": prism.rho_target,                  # ρ_T  (target RMS scale)
                    "rho_P": prism.rho_proxy,                   # ρ_P  (proxy RMS scale)
                    "scale": prism.scale_mismatch,              # (ρ_T − ρ_P)²
                    "shape": prism.shape_mismatch,              # 2 ρ_T ρ_P (1 − Ω)
                    "delta": prism.feature_error,               # δ/K_feat = √(Scale + Shape)
                    "gamma": prism.head_discrepancy,            # γ/K_pred = ‖Σ^½(WH_T−H_P)‖_F
                    # Tight Lipschitz constants (paper Eq. 8, Appendix A)
                    "K_feat": self.K_feat,                      # max_{j,k} ‖h_j − h_k‖₂  (from base H_T)
                    "K_pred": self.K_pred,                      # √2
                    # Bound (using tight K_feat, K_pred above)
                    "bound_feature": prism.risk_bound_feature,  # K_feat × δ
                    "bound_head": prism.risk_bound_head,        # K_pred × γ
                    "bound_total": prism.risk_bound_total,      # K_feat·δ + K_pred·γ
                    # Empirical risk — answer-only (primary)
                    "loss_T": loss_T_answer,                    # R(θ₀) on answer tokens
                    "loss_P": loss_P_answer,                    # R(θ_t) on answer tokens
                    "delta_risk": primary_dr,                   # |R(θ_t) − R(θ₀)|
                    # Empirical risk — full-sequence (supplementary)
                    "loss_T_full": loss_T_full,
                    "loss_P_full": loss_P_full,
                    "delta_risk_full": delta_risk_full,
                    # Bound validation (answer-only |ΔR| vs bound)
                    "bound_holds": (
                        prism.risk_bound_total >= primary_dr
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
        print(f"  (All losses are answer-only CE. eval_loss is Trainer validation on")
        print(f"   training-task data; PRISM metrics below use separate eval data.)")

        # ── Table A: Geometry ────────────────────────────────────────
        print()
        print(f"  [Geometry]  δ = √(Scale+Shape),  γ = ‖Σ^½(WH_T−H_P)‖,  Bound = δ+γ  (K=1)")
        hdr_a = (
            f"  {'Task':<11s}"
            f"  {'ρ_T':>8s}  {'ρ_P':>8s}"
            f"  {'Ω':>10s}  {'Scale':>10s}  {'Shape':>10s}"
            f"  {'δ':>8s}  {'γ':>8s}  {'Bound':>8s}"
        )
        print(hdr_a)
        print(f"  {'─' * (len(hdr_a) - 2)}")

        for task in self.eval_tasks:
            r = task_results[task]
            marker = " *" if task == self.trained_task else "  "
            bound_s = f"{r['bound_total']:8.4f}" if r["bound_total"] is not None else "       —"

            print(
                f"  {task:<9s}{marker}"
                f"  {r['rho_T']:8.2f}  {r['rho_P']:8.2f}"
                f"  {r['omega']:10.6f}  {r['scale']:10.6f}  {r['shape']:10.6f}"
                f"  {r['delta']:8.4f}  {r['gamma']:8.4f}  {bound_s}"
            )

        # ── Table B: Empirical Risk (answer-only) ────────────────────
        print()
        print(f"  [Empirical Risk]  answer-only CE  (loss on answer tokens aligned with Z)")
        hdr_b = (
            f"  {'Task':<11s}"
            f"  {'Loss_T':>8s}  {'Loss_P':>8s}  {'|ΔR|':>8s}"
            f"  {'Bound':>8s}  {'Holds':>5s}"
        )
        print(hdr_b)
        print(f"  {'─' * (len(hdr_b) - 2)}")

        for task in self.eval_tasks:
            r = task_results[task]
            marker = " *" if task == self.trained_task else "  "
            bound_s = f"{r['bound_total']:8.4f}" if r["bound_total"] is not None else "       —"
            holds = r.get("bound_holds")
            holds_s = "  yes" if holds is True else "   no" if holds is False else "    —"

            if r["loss_T"] is not None:
                print(
                    f"  {task:<9s}{marker}"
                    f"  {r['loss_T']:8.4f}  {r['loss_P']:8.4f}  {r['delta_risk']:8.4f}"
                    f"  {bound_s}  {holds_s}"
                )
            else:
                # Fallback to full-sequence if answer-only unavailable
                print(
                    f"  {task:<9s}{marker}"
                    f"  {r['loss_T_full']:8.4f}  {r['loss_P_full']:8.4f}  {r['delta_risk_full']:8.4f}"
                    f"  {bound_s}  {holds_s}  (full-seq)"
                )

        print(f"{'=' * 78}")

    # ------------------------------------------------------------------
    def _save_json(self):
        """Overwrite the JSON file with all accumulated results."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Compute base rho from stored Z for completeness
        base_summary = {}
        for task in self.eval_tasks:
            Z = self.base_features[task]["Z"]
            import math
            rho = Z.norm("fro").item() / math.sqrt(Z.shape[0])
            base_summary[task] = {
                "rho": rho,
                "loss_full": self.base_features[task]["loss_full"],
                "loss_answer": self.base_features[task]["loss_answer"],
                "Z_shape": list(Z.shape),
            }

        payload = {
            "experiment": self.experiment_config,
            "field_definitions": {
                "omega": "Ω_I — identity-aligned similarity ∈ [-1,1]",
                "rho_T": "ρ_T — RMS feature scale of target (base model)",
                "rho_P": "ρ_P — RMS feature scale of proxy (fine-tuned)",
                "scale": "(ρ_T − ρ_P)² — scale mismatch",
                "shape": "2 ρ_T ρ_P (1 − Ω_I) — shape mismatch under W=I",
                "delta": "√(scale + shape) — feature alignment error (δ/K_feat) under W=I",
                "gamma": "‖Σ_P^½ (H_T − H_P)‖_F — head discrepancy (γ/K_pred) under W=I",
                "K_feat": "max_{j,k} ‖h_j − h_k‖₂ — simplex-polarization Lipschitz of CE in features (from base H_T, Appendix A)",
                "K_pred": "√2 — Lipschitz of CE in logits (Proposition 1)",
                "bound_feature": "K_feat × δ — feature term of the unified bound",
                "bound_head": "K_pred × γ — head term of the unified bound",
                "bound_total": "K_feat·δ + K_pred·γ — unified risk bound (Eq. 8)",
                "loss_T": "R(θ₀) — base model answer-only CE loss",
                "loss_P": "R(θ_t) — fine-tuned model answer-only CE loss",
                "delta_risk": "|R(θ_t) − R(θ₀)| — empirical forgetting (answer-only)",
                "loss_T_full": "full-sequence CE loss (supplementary)",
                "loss_P_full": "full-sequence CE loss (supplementary)",
                "train_loss": "Trainer training loss (answer-only CE, training task)",
                "eval_loss": "Trainer eval loss (answer-only CE, training task val split)",
            },
            "base_model": base_summary,
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
    eval_tasks: List[str],
    num_samples: int,
    batch_size: int,
    max_length: int,
    seed: int,
    device: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Tuple[Any, str]], Dict[str, float]]:
    """Extract base model features + loss on eval tasks.

    Also computes the tight Lipschitz constants K_feat, K_pred from the
    base model's lm_head once (identical across tasks since H_T is the
    same frozen base head). These are required for the paper's bound
    |R_T - R_P| <= K_feat * delta + K_pred * gamma to hold; K = 1 is
    not a valid assumption for cross-entropy loss.

    Returns:
        base_features:    {task: {Z, H, loss_full, loss_answer, Z_shape}}
        eval_dataloaders: {task: (DataLoader, z_mode)}  — reused in callback
        K_theory:         {K_feat, K_pred, K_feat_naive, ...} from
                          UnifiedBound.theoretical_K (paper Eq. 8 constants)
    """
    print(f"\n{'─' * 78}")
    print(f"  Pre-computing base model features on {len(eval_tasks)} eval tasks")
    print(f"  (n={num_samples}, batch_size={batch_size}, max_length={max_length})")
    print(f"{'─' * 78}")

    model.eval()
    base_features: Dict[str, Dict[str, Any]] = {}
    eval_dataloaders: Dict[str, Tuple[Any, str]] = {}
    K_theory: Dict[str, float] = None  # filled on first iter, reused

    for task in eval_tasks:
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

        # Compute K_feat/K_pred once on GPU (paper Eq. 8, Appendix A).
        # H_T is the same frozen base head for every task, so the tight
        # bound K_feat = max_{j,k} ||h_j - h_k||_2 is task-independent.
        if K_theory is None:
            print(f"\n  [computing tight Lipschitz constants on H_T "
                  f"{tuple(H_T.shape)} ...] ", end="", flush=True)
            kt0 = time.time()
            K_theory = UnifiedBound.theoretical_K(H_T)
            print(f"K_feat={K_theory['K_feat']:.4f}  "
                  f"K_pred={K_theory['K_pred']:.4f}  "
                  f"({time.time() - kt0:.1f}s)\n  {task:<10s} ... ",
                  end="", flush=True)

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
    return base_features, eval_dataloaders, K_theory


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1: LoRA fine-tune on a single task for PRISM forgetting",
    )
    # Required
    p.add_argument("--model", required=True,
                   help="HuggingFace model ID (e.g. meta-llama/Meta-Llama-3.1-8B)")
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

    # Shape regularization (W=I trace form). Mutually exclusive with replay.
    p.add_argument("--lambda_shape", type=float, default=0.0,
                   help="Weight for shape regularizer 1-Ω_I (0 = disabled). "
                        "Mutually exclusive with --lambda_replay.")
    p.add_argument("--lambda_replay", type=float, default=0.0,
                   help="Weight for data-replay CE regularizer on the reference "
                        "set (0 = disabled). Apples-to-apples baseline for "
                        "--lambda_shape: same 32 instances, same scheduling, "
                        "only the loss form (CE vs 1-Ω_I) differs. "
                        "Mutually exclusive with --lambda_shape.")
    p.add_argument("--reg_every_k", type=int, default=8,
                   help="Compute regularizer (shape or replay) every K "
                        "micro-steps (default: match grad_accum)")
    p.add_argument("--reg_samples", type=int, default=32,
                   help="Number of reference samples for the regularizer")
    p.add_argument("--reg_batch_size", type=int, default=8,
                   help="Batch size for reference forward pass")
    p.add_argument("--reg_max_length", type=int, default=512,
                   help="Max sequence length for the reference forward pass. "
                        "Kept separate from --max_length so long-form tasks "
                        "don't OOM the grad-retained hidden-state concat "
                        "inside _compute_shape_loss.")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=10)
    return p.parse_args()


# ======================================================================
# Data loading  (for training only — PRISM eval uses prism.data.loaders)
# ======================================================================

_SOCIAL_IQA_URL = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"
_SOCIAL_IQA_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "social_iqa")


def _load_social_iqa(split: str):
    """Download Social IQa from the original source and return an HF Dataset.

    The allenai/social_i_qa HuggingFace repo uses a deprecated loading script,
    so we download the raw JSONL + label files directly from AI2's public bucket.
    """
    import io
    import json
    import zipfile
    import requests
    from datasets import Dataset as HFDataset

    cache_dir = _SOCIAL_IQA_CACHE
    os.makedirs(cache_dir, exist_ok=True)

    split_name = "train" if split == "train" else "dev"
    jsonl_path = os.path.join(cache_dir, f"{split_name}.jsonl")
    labels_path = os.path.join(cache_dir, f"{split_name}-labels.lst")

    # Download and extract if not cached
    if not os.path.exists(jsonl_path) or not os.path.exists(labels_path):
        print(f"  Downloading Social IQa from {_SOCIAL_IQA_URL} ...")
        r = requests.get(_SOCIAL_IQA_URL, timeout=60)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        for name in ["train.jsonl", "train-labels.lst", "dev.jsonl", "dev-labels.lst"]:
            src = f"socialiqa-train-dev/{name}"
            dst = os.path.join(cache_dir, name)
            with z.open(src) as zf, open(dst, "wb") as out:
                out.write(zf.read())
        print(f"  Cached to {cache_dir}")

    # Load JSONL + labels
    with open(jsonl_path) as f:
        rows = [json.loads(line) for line in f]
    with open(labels_path) as f:
        labels = [line.strip() for line in f]

    for row, label in zip(rows, labels):
        row["label"] = label

    return HFDataset.from_list(rows)


def _load_hf_dataset(task_name: str, split: str):
    cfg = TASK_CONFIGS[task_name]

    # Social IQa requires custom loading (HF script is deprecated)
    if task_name == "social_iqa":
        actual_split = cfg["train_split"] if split == "train" else cfg["eval_split"]
        return _load_social_iqa(actual_split)

    hf_args = [cfg["hf_id"]]
    if cfg.get("hf_subset") is not None:
        hf_args.append(cfg["hf_subset"])
    actual_split = cfg["train_split"] if split == "train" else cfg["eval_split"]
    return load_dataset(*hf_args, split=actual_split)


def build_dataset(task_name, split, tokenizer, max_length, max_samples, seed):
    """Tokenize a dataset and compute prompt_length for answer-only loss.

    Each tokenized example gets a ``prompt_length`` field (int) indicating
    where the prompt ends, determined by longest-common-prefix matching
    between the full-text and prompt-only tokenizations.  This guards
    against context-dependent BPE divergence.
    """
    raw = _load_hf_dataset(task_name, split)
    if max_samples is not None and len(raw) > max_samples:
        raw = raw.shuffle(seed=seed).select(range(max_samples))

    formatter = FORMATTERS[task_name]
    prompt_formatter = PROMPT_FORMATTERS[task_name]
    original_columns = raw.column_names

    def tokenize_fn(batch):
        keys = list(batch.keys())
        n = len(batch[keys[0]])
        full_texts, prompt_texts = [], []
        for i in range(n):
            row = {k: batch[k][i] for k in keys}
            full_texts.append(formatter(row))
            prompt_texts.append(prompt_formatter(row))

        # Tokenize full text (what the model trains on)
        full_enc = tokenizer(
            full_texts, truncation=True, max_length=max_length, padding=False,
        )
        # Tokenize prompt only (to find where the answer starts)
        prompt_enc = tokenizer(
            prompt_texts, truncation=True, max_length=max_length,
            add_special_tokens=True,
        )

        # Compute prompt_length via longest-common-prefix (LCP) matching
        # between full-text and prompt-only token IDs.
        prompt_lengths = []
        for i in range(n):
            f_ids = full_enc["input_ids"][i]
            p_ids = prompt_enc["input_ids"][i]
            pl = 0
            for k in range(min(len(p_ids), len(f_ids))):
                if p_ids[k] == f_ids[k]:
                    pl = k + 1
                else:
                    break
            prompt_lengths.append(pl)

        full_enc["prompt_length"] = prompt_lengths
        return full_enc

    return raw.map(
        tokenize_fn, batched=True, remove_columns=original_columns,
        desc=f"Tokenizing {task_name} ({split})",
    )


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()

    # ── Mutual exclusion: shape reg and replay reg cannot both be on ──
    if args.lambda_shape > 0 and args.lambda_replay > 0:
        print("ERROR: --lambda_shape and --lambda_replay are mutually exclusive; "
              "set at most one to a positive value.", file=sys.stderr)
        sys.exit(2)

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

    # Per-task max_length override (e.g. instruction-tuning datasets need longer).
    # Only applied when the user did not pass --max_length explicitly (i.e. it
    # still equals the argparse default of 512).
    task_max_length = task_cfg.get("max_length")
    if task_max_length is not None and args.max_length == 512:
        args.max_length = task_max_length

    model_short = args.model.split("/")[-1].lower()
    output_dir = args.output_dir or os.path.join(
        "checkpoints", "forgetting", model_short, args.task,
    )
    os.makedirs(output_dir, exist_ok=True)

    warmup_steps = int(args.warmup_ratio * max_steps)

    # ── Eval tasks: 5 downstream benchmarks (disjoint from trained task) ─
    eval_tasks = get_eval_tasks(args.task)

    experiment_config = {
        "model": args.model,
        "trained_task": args.task,
        "eval_tasks": eval_tasks,
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
        "warmup_steps": warmup_steps,
        "max_train_samples": max_train_samples,
        "prism_eval_samples": args.prism_eval_samples,
        "train_loss_mode": "answer_only",
        "seed": args.seed,
        "lambda_shape": args.lambda_shape,
        "lambda_replay": args.lambda_replay,
        "reg_every_k": (args.reg_every_k
                        if (args.lambda_shape > 0 or args.lambda_replay > 0)
                        else None),
        "reg_samples": (args.reg_samples
                        if (args.lambda_shape > 0 or args.lambda_replay > 0)
                        else None),
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

    base_features, eval_dataloaders, K_theory = pre_compute_base_features(
        model, tokenizer, extractor,
        eval_tasks=eval_tasks,
        num_samples=args.prism_eval_samples,
        batch_size=args.prism_eval_batch_size,
        max_length=args.max_length,
        seed=args.seed,
        device=tensor_device,
    )

    # Record tight Lipschitz constants in the experiment config so they're
    # written to prism_forgetting_metrics.json for reproducibility.
    experiment_config["K_feat"] = K_theory["K_feat"]
    experiment_config["K_pred"] = K_theory["K_pred"]
    experiment_config["K_feat_naive"] = K_theory.get("K_feat_naive")

    # ── Pre-compute reference data + (for shape only) Z_T_ref ────────
    # Both regularizers share the same fixed reference set; only the shape
    # regularizer needs Z_T (base-model features) pre-extracted, since the
    # replay baseline uses the current model's CE on the same inputs.
    Z_T_ref = None
    ref_dataloader = None
    use_reg = args.lambda_shape > 0 or args.lambda_replay > 0
    if use_reg:
        ref_dataloader = load_task_data(
            args.task, split="test",
            num_samples=args.reg_samples,
            batch_size=args.reg_batch_size,
            tokenizer=tokenizer,
            max_length=min(args.max_length, args.reg_max_length),
            seed=args.seed + 1000,
        )

    if args.lambda_shape > 0:
        reg_meta = get_task_metadata(args.task)
        z_mode_ref = reg_meta["z_mode"]

        print(f"\nPre-computing Z_T_ref for shape regularizer ...")
        print(f"  task={args.task}, n={args.reg_samples}, z_mode={z_mode_ref}")
        Z_T_ref = extractor.extract_features(
            model, ref_dataloader, tensor_device, z_mode=z_mode_ref,
        )
        Z_T_ref = Z_T_ref.float().cpu()
        print(f"  Z_T_ref shape: {list(Z_T_ref.shape)}")

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
        eval_tasks=eval_tasks,
        model_id=args.model,
        output_dir=output_dir,
        device=tensor_device,
        experiment_config=experiment_config,
        K_theory=K_theory,
    )

    # ── Trainer ──────────────────────────────────────────────────────
    collator = AnswerOnlyDataCollator(tokenizer=tokenizer)
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
        warmup_steps=warmup_steps,
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
        remove_unused_columns=False,   # keep prompt_length for AnswerOnlyDataCollator
        seed=args.seed,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[prism_callback],
    )
    if args.lambda_shape > 0:
        print(f"\n  Shape regularizer: λ={args.lambda_shape}, "
              f"every_k={args.reg_every_k}, samples={args.reg_samples}")
        trainer = ShapeRegularizedTrainer(
            **trainer_kwargs,
            Z_T_ref=Z_T_ref,
            ref_dataloader=ref_dataloader,
            lambda_shape=args.lambda_shape,
            reg_every_k=args.reg_every_k,
            device_str=tensor_device,
        )
    elif args.lambda_replay > 0:
        print(f"\n  Replay-CE regularizer: λ={args.lambda_replay}, "
              f"every_k={args.reg_every_k}, samples={args.reg_samples}")
        trainer = ReplayCETrainer(
            **trainer_kwargs,
            Z_T_ref=None,                        # unused by replay baseline
            ref_dataloader=ref_dataloader,
            lambda_shape=args.lambda_replay,     # parent's slot for the weight
            reg_every_k=args.reg_every_k,
            device_str=tensor_device,
        )
    else:
        trainer = Trainer(**trainer_kwargs)

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
