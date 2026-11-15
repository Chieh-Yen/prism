#!/usr/bin/env python3
"""
LoRA fine-tune a base model on a single task and compute PRISM forgetting
metrics online at every checkpoint (no separate inference stage needed).

Fine-tuning tasks: **TruthfulQA** and **BBQ** (paper Sec. 4.5).

Models in the paper: meta-llama/Meta-Llama-3.1-8B, Qwen/Qwen3-8B-Base.
The script accepts any causal LM whose backbone exposes hidden states.

Under LoRA the lm_head is frozen (H_t = H_0), so PRISM's head divergence
term vanishes under identity alignment and the bound of Theorem 1 (Eq. 4 in
the paper) reduces to its feature term (Sec. 3.4).

Usage:
    python train_forgetting.py \\
        --model meta-llama/Meta-Llama-3.1-8B --task truthfulqa \\
        --lr 1e-5 --max_steps 300

    python train_forgetting.py \\
        --model Qwen/Qwen3-8B-Base --task bbq \\
        --lr 1e-5 --max_steps 300 --lambda_shape 1.0

Methods (see the trainer selection in main() and --help):
    --lambda_shape L                           shape regularizer 1 - Omega_I
    --lambda_replay L --replay_source task     anchor-CE (CE on the anchor set)
    --lambda_replay L --replay_source generic  generic replay (CE on WikiText)
    --lambda_shape 1e-9                        unregularized, same training loop
    (both lambdas 0)                           unregularized, stock Trainer

Target-task evaluation rows.  The Trainer's eval set is the held-out slice of
the fine-tuning task (TASK_CONFIGS eval_split).  By default the 32 rows of the
anchor draw are removed from it (unified protocol, scripts/response/forgetting/);
with --eval_keep_anchor_rows the whole slice is evaluated, as in the λ sweeps of
run_forgetting.sh (which passes the flag).  The flag changes `eval_loss`
only; see the comment at the eval-set construction in main().

Set PRISM_EVAL_SIMILARITY=1 to add linear CKA and SVCCA (similarity_baselines.py)
to every PRISM evaluation record.
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
    set_seed,
)

from prism.core.bounds import UnifiedBound
from prism.core.metrics import PRISMMetrics
from prism.data.loaders import get_task_metadata, load_task_data
from prism.models.extractors import LLMExtractor

# Remote code is opt-in (PRISM_TRUST_REMOTE_CODE=1). Every model family used in
# the paper is native to the pinned transformers release, so the default is off.
_TRUST_REMOTE_CODE = os.environ.get("PRISM_TRUST_REMOTE_CODE", "0") == "1"
from prism.experiments.quantization import pinned_revision  # configs/model_revisions.yaml


# ── Formatters; identical to prism/data/loaders.py for consistency ───────

def _format_truthfulqa(row: dict) -> str:
    return f"Question: {row['question']}\nAnswer: {row['best_answer']}"


def _format_bbq(row: dict) -> str:
    choices = row["choices"]
    labels = ["A", "B", "C"]
    opts = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
    ans_label = labels[row["gold_index"]]
    return f"Context: {row['context']}\nQuestion: {row['question']}\n{opts}\nAnswer: {ans_label}"


FORMATTERS = {
    "truthfulqa": _format_truthfulqa,
    "bbq": _format_bbq,
}


# ── Prompt-only formatters (question prefix, no answer) ──────────────────
# Used to compute prompt_length so that loss is only on answer tokens.
# Must be a strict prefix of the corresponding FORMATTER output.

def _prompt_truthfulqa(row: dict) -> str:
    return f"Question: {row['question']}\nAnswer:"


def _prompt_bbq(row: dict) -> str:
    choices = row["choices"]
    labels = ["A", "B", "C"]
    opts = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
    return f"Context: {row['context']}\nQuestion: {row['question']}\n{opts}\nAnswer:"


PROMPT_FORMATTERS = {
    "truthfulqa": _prompt_truthfulqa,
    "bbq": _prompt_bbq,
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
    anchor set to extract Z_P, and the shape loss 1 − Ω_I is backprop'd
    into the current parameter gradient buffer.  Z_T (base model features)
    is pre-computed and frozen.

    Loss scaling under gradient accumulation
    ----------------------------------------
    The cross-entropy and the penalty have separate backward calls (see
    ``training_step``).  With G = ``gradient_accumulation_steps`` and the
    default ``reg_every_k == G`` (one penalty evaluation per optimizer
    step), the gradient accumulated for one optimizer step is

        (1/G) · [ Σ_{m=1..G} ∇CE_m  +  λ · ∇P ]

    where CE_m is the mean answer-token cross-entropy of micro-batch m and
    P is the penalty (here 1 − Ω_I).  This follows from three facts, stated
    for the pinned library versions (transformers 5.2.0, accelerate 1.12.0;
    see requirements.txt):

      * ``compute_loss`` returns the model's own micro-batch mean CE and
        does not forward ``num_items_in_batch`` to the model.
      * ``Trainer.training_step`` is nevertheless called with
        ``num_items_in_batch``, and in that case it applies no division by
        ``gradient_accumulation_steps`` of its own.
      * Both backward calls go through ``accelerator.backward``, which
        divides by the accumulation steps registered with the Accelerator
        (G).

    Every subclass (``ReplayCETrainer`` here, and the baseline trainers in
    scripts/rounds/train_forgetting_baselines.py) overrides only the penalty
    P, so all regularized methods, and the same-loop unregularized reference
    (this class with a negligible λ; see ``main``), share this scaling.
    """

    def __init__(
        self,
        *args,
        Z_T_ref: Optional[Tensor] = None,
        ref_dataloader,
        lambda_shape: float = 0.1,
        reg_every_k: int = 8,
        device_str: str = "cuda",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.Z_T_ref = Z_T_ref          # (n_tokens, d) on CPU, detached, or None
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
        """Micro-batch mean CE as returned by the model.

        ``num_items_in_batch`` (passed by the ``Trainer`` in ``kwargs``) is not
        forwarded to the model.  The penalty is added in ``training_step``
        with its own backward call (see class docstring for the scaling)."""
        outputs = model(**inputs)
        ce_loss = outputs.loss
        if return_outputs:
            return ce_loss, outputs
        return ce_loss

    # ------------------------------------------------------------------
    def training_step(self, model, inputs, *args, **kwargs):
        """Run the parent's CE training step, then add the penalty with a
        separate backward call every ``reg_every_k`` micro-steps.

        The parent's ``training_step`` does the CE forward and
        ``accelerator.backward(CE_m)``.  On micro-steps 0, k, 2k, … the
        penalty is then computed on the anchor set and
        ``accelerator.backward(λ · P)`` accumulates its gradient into the
        same buffer.  The resulting scaling is given in the class docstring.
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
        """Differentiable shape loss: 1 − Ω_I on the anchor set.

        Forward-passes the anchor-set DataLoader (with gradients enabled)
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
        trace = (Z_T * Z_P).sum(dim=1).double().sum()
        denom = (Z_T * Z_T).sum(dim=1).double().sum().sqrt() * (Z_P * Z_P).sum(dim=1).double().sum().sqrt()
        omega_I = (trace / denom.clamp(min=1e-12)).float()

        shape_loss = 1.0 - omega_I
        return shape_loss, omega_I.item()

    # ------------------------------------------------------------------
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """Add the penalty averages to the log entry."""
        if self._count > 0:
            logs["shape_loss"] = round(self._shape_sum / self._count, 6)
            logs["omega_ref"] = round(self._omega_sum / self._count, 6)
            self._shape_sum = 0.0
            self._omega_sum = 0.0
            self._count = 0
        super().log(logs, *args, **kwargs)


# ── Cross-entropy penalty trainer (anchor-CE, generic replay) ─────────────

class ReplayCETrainer(ShapeRegularizedTrainer):
    """Cross-entropy rehearsal on a fixed set: L = L_CE + λ · CE_ref.

    Same scheduling, backward path and loss scaling as
    ``ShapeRegularizedTrainer`` (see its docstring); only the penalty
    differs: the next-token cross-entropy on a fixed set of sequences
    instead of 1 − Ω_I.  Which sequences depends on ``--replay_source``
    (see ``build_ref_dataloader``):

      * ``task``:     the anchor set itself, i.e. the same sequences the
                      shape regularizer reads ("anchor-CE"; the same-data
                      control, differing from the regularizer only in the
                      penalty);
      * ``generic``:  the same number of WikiText sequences ("generic
                      replay").

    Z_T_ref is unused (CE only depends on the current-model output).
    When the batch carries ``prompt_length`` (the task source), prompt
    tokens are masked from the CE target (set to -100), matching how the
    fine-tuning CE is computed by ``AnswerOnlyDataCollator``; WikiText rows
    have no prompt, so all their non-padding tokens are targets.
    """

    def _compute_shape_loss(self, model) -> Tuple[Tensor, float]:
        """Override: next-token CE on the fixed set, averaged over its batches.

        Targets are the answer tokens when the batch carries ``prompt_length``
        (source ``task``), and all non-padding tokens otherwise (source
        ``generic``)."""
        total = torch.zeros((), device=self._device_str)
        n_batches = 0
        for batch in self._ref_dl:
            b = {k: v.to(self._device_str) for k, v in batch.items()}
            prompt_lens = b.pop("prompt_length", None)
            labels = b["input_ids"].clone()
            if prompt_lens is not None:
                # Mask prompt tokens: the CE targets are the answer tokens.
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
        Trainer.log(self, logs, *args, **kwargs)


# ── Anchor set / generic-replay set construction ──────────────────────────

def build_ref_dataloader(args, tokenizer):
    """Build the fixed set of sequences the penalty is computed on.

    ``--replay_source task`` (default), and always when the shape regularizer
    is active: the anchor set, ``args.reg_samples`` sequences of the
    fine-tuning task's held-out slice (prism.data.loaders split "test"),
    taken as the first rows after shuffling that slice with seed
    ``args.seed + 1000``.  The draw depends only on (task, seed,
    reg_samples), so every method sees the same rows at a given seed, and
    the rows are redrawn for each seed.

    ``--replay_source generic`` (generic replay): ``args.reg_samples``
    WikiText sequences (wikitext-2-raw-v1, train split).

    Both sources yield the same number of sequences, truncated at the same
    length, min(max_length, reg_max_length).
    """
    from torch.utils.data import DataLoader, Subset

    max_len = min(args.max_length, args.reg_max_length)
    src = getattr(args, "replay_source", "task")

    if src == "task" or args.lambda_shape > 0:
        # The shape regularizer always reads the fine-tuning task's held-out
        # slice: its anchor set defines the geometry being preserved.
        if args.lambda_shape > 0 and src != "task":
            print(f"[ref] shape regularizer active: forcing replay_source=task "
                  f"(requested '{src}' applies to the cross-entropy penalty only)")
        return load_task_data(
            args.task, split="test",
            num_samples=args.reg_samples,
            batch_size=args.reg_batch_size,
            tokenizer=tokenizer,
            max_length=max_len,
            seed=args.seed + 1000,
        )

    # src == "generic".  WikiText rows include blank lines and section
    # headers; draw a larger pool (reg_samples * 40 rows, shuffled with seed
    # args.seed + 1000) and keep the first `reg_samples` rows with at least
    # MIN_TOK tokens, so this source delivers as many usable sequences as the
    # task source.
    MIN_TOK = 64
    pool = load_task_data(
        "wikitext", split="train",
        num_samples=args.reg_samples * 40,
        batch_size=args.reg_batch_size,
        tokenizer=tokenizer,
        max_length=max_len,
        seed=args.seed + 1000,
    )
    ds = pool.dataset
    keep, tok_total = [], 0
    for i in range(len(ds)):
        ex = ds[i]
        n_tok = int(ex["attention_mask"].sum())
        if n_tok >= MIN_TOK:
            keep.append(i); tok_total += n_tok
        if len(keep) == args.reg_samples:
            break
    if len(keep) < args.reg_samples:
        raise RuntimeError(
            f"generic replay pool too small: {len(keep)} usable rows "
            f"(need {args.reg_samples}); raise the pool multiplier")
    print(f"[ref] generic old-domain replay: wikitext train, "
          f"{len(keep)} sequences, {tok_total} tokens "
          f"(>= {MIN_TOK} tokens each)")
    return DataLoader(Subset(ds, keep), batch_size=args.reg_batch_size,
                      shuffle=False, num_workers=0)


# ── Task-specific dataset configuration ───────────────────────────────────

TASK_CONFIGS = {
    # ── Fine-tuning tasks (paper Sec. 4.5) ───────────────────────────
    "truthfulqa": {
        "hf_id": "truthfulqa/truthful_qa",
        "hf_subset": "generation",
        "train_split": "validation[:80%]",   # only split; first 80% for train (654 rows)
        "eval_split": "validation[80%:]",    # last 20% for eval (163 rows)
        "max_train_samples": None,
        "max_eval_samples": None,
        "default_max_steps": 300,
        "default_save_steps": 25,
    },
    "bbq": {
        "hf_id": "lighteval/bbq_helm",
        "hf_subset": "all",
        "train_split": "test[:80%]",         # only split; first 80% for train (~800)
        "eval_split": "test[80%:]",          # last 20% for eval (~200)
        "max_train_samples": None,
        "max_eval_samples": None,
        "default_max_steps": 300,
        "default_save_steps": 25,
    },
}

BASE_EVAL_TASKS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]


def get_eval_tasks(trained_task: str) -> list:
    """Return the downstream eval task list.

    Forgetting is measured on benchmarks **disjoint** from the fine-tuning
    task (TruthfulQA or BBQ), so we evaluate on BASE_EVAL_TASKS.
    """
    return list(BASE_EVAL_TASKS)

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

        # Tight Lipschitz constants of the bound of Theorem 1 (Eq. 4; Appendix A):
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

        # Save JSON (overwrite; survives crashes)
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
                task_t0 = time.time()
                print(f"  [{idx}/{total_tasks}] {task:<10s} (z_mode={z_mode}) ... ", end="", flush=True)

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

                sim_scores = {}
                if os.environ.get("PRISM_EVAL_SIMILARITY", "0") == "1":
                    # Linear CKA and SVCCA on the same features (Z_T, Z_P) the
                    # PRISM terms above are computed from, with the same
                    # estimators as the quantization-cell similarity baselines.
                    from similarity_baselines import linear_cka, svcca
                    _zt = Z_T.float()
                    _zp = Z_P.float().cpu()
                    sim_scores["cka"] = linear_cka(_zt, _zp)
                    sim_scores["svcca"] = svcca(_zt, _zp)

                task_results[task] = {
                    **sim_scores,
                    # Geometry  (paper notation)
                    "omega": prism.omega,                       # Ω_I  (identity-aligned similarity)
                    "rho_T": prism.rho_target,                  # ρ_T  (target RMS scale)
                    "rho_P": prism.rho_proxy,                   # ρ_P  (proxy RMS scale)
                    "scale": prism.scale_mismatch,              # (ρ_T − ρ_P)²
                    "shape": prism.shape_mismatch,              # 2 ρ_T ρ_P (1 − Ω)
                    "delta": prism.feature_error,               # δ/K_feat = √(Scale + Shape)
                    "gamma": prism.head_discrepancy,            # γ/K_pred = ‖Σ^½(WH_T−H_P)‖_F
                    # Tight Lipschitz constants (Theorem 1 / Eq. 4, Appendix A)
                    "K_feat": self.K_feat,                      # max_{j,k} ‖h_j − h_k‖₂  (from base H_T)
                    "K_pred": self.K_pred,                      # √2
                    # Bound (using tight K_feat, K_pred above)
                    "bound_feature": prism.risk_bound_feature,  # K_feat × δ
                    "bound_head": prism.risk_bound_head,        # K_pred × γ
                    "bound_total": prism.risk_bound_total,      # K_feat·δ + K_pred·γ
                    # Empirical risk; answer-only (primary)
                    "loss_T": loss_T_answer,                    # R(θ₀) on answer tokens
                    "loss_P": loss_P_answer,                    # R(θ_t) on answer tokens
                    "delta_risk": primary_dr,                   # |R(θ_t) − R(θ₀)|
                    # Empirical risk; full-sequence (supplementary)
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
        print(f"   the fine-tuning task's held-out rows; PRISM metrics below use the benchmarks.)")

        # ── Table A: Geometry ────────────────────────────────────────
        print()
        print(f"  [Geometry]  δ = √(Scale+Shape),  γ = ‖Σ^½(WH_T−H_P)‖,  Bound = K_feat·δ + K_pred·γ")
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
            bound_s = f"{r['bound_total']:8.4f}" if r["bound_total"] is not None else "       —"

            print(
                f"  {task:<9s}  "
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
            bound_s = f"{r['bound_total']:8.4f}" if r["bound_total"] is not None else "       —"
            holds = r.get("bound_holds")
            holds_s = "  yes" if holds is True else "   no" if holds is False else "    —"

            if r["loss_T"] is not None:
                print(
                    f"  {task:<9s}  "
                    f"  {r['loss_T']:8.4f}  {r['loss_P']:8.4f}  {r['delta_risk']:8.4f}"
                    f"  {bound_s}  {holds_s}"
                )
            else:
                # Fallback to full-sequence if answer-only unavailable
                print(
                    f"  {task:<9s}  "
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
            rho = PRISMMetrics.rms_scale(Z.float())
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
                "K_pred": "√2: Lipschitz of CE in logits (Appendix A)",
                "bound_feature": "K_feat × δ — feature term of the unified bound",
                "bound_head": "K_pred × γ — head term of the unified bound",
                "bound_total": "K_feat·δ + K_pred·γ: unified risk bound (Theorem 1, Eq. 4)",
                "loss_T": "R(θ₀) — base model answer-only CE loss",
                "loss_P": "R(θ_t) — fine-tuned model answer-only CE loss",
                "delta_risk": "|R(θ_t) − R(θ₀)| — empirical forgetting (answer-only)",
                "loss_T_full": "full-sequence CE loss (supplementary)",
                "loss_P_full": "full-sequence CE loss (supplementary)",
                "train_loss": "most recent `loss` logged by the Trainer at or before this "
                              "step: answer-only CE on the fine-tuning task, averaged over "
                              "the optimizer steps of the last logging window; the regularized trainers "
                              "log the sum of the G = grad_accum micro-batch mean CEs "
                              "(README note (c)); the penalty is not included",
                "eval_loss": ("Trainer eval loss: answer-only CE on the held-out slice of the "
                              "fine-tuning task; without the "
                              "anchor rows by default, on the whole slice with "
                              "--eval_keep_anchor_rows (experiment.eval_keep_anchor_rows, "
                              "experiment.n_eval_rows)"),
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
    |R_T - R_P| <= K_feat * delta + K_pred * gamma to hold.

    Returns:
        base_features:    {task: {Z, H, loss_full, loss_answer, Z_shape}}
        eval_dataloaders: {task: (DataLoader, z_mode)} ; reused in callback
        K_theory:         {K_feat, K_pred, K_feat_naive, ...} from
                          UnifiedBound.theoretical_K (constants of Theorem 1 / Eq. 4)
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

        # Compute K_feat/K_pred once on GPU (Theorem 1 / Eq. 4, Appendix A).
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
        description="LoRA fine-tune on a single task with online PRISM forgetting metrics",
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
                   help="Max training steps (default: task-specific, 300 for "
                        "truthfulqa and bbq)")
    p.add_argument("--save_steps", type=int, default=None,
                   help="Save checkpoint every N steps (default: task-specific)")
    p.add_argument("--save_total_limit", type=int, default=None,
                   help="Keep only the last N checkpoints on disk "
                        "(default: keep all)")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Learning rate (default: 1e-5)")
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
    p.add_argument("--eval_keep_anchor_rows", action="store_true",
                   help="Evaluation only: keep the anchor rows in the Trainer's "
                        "eval set, so eval_loss is computed on the whole held-out "
                        "slice of the fine-tuning task (as in the λ sweeps of "
                        "run_forgetting.sh, which passes this flag). Default: "
                        "the --reg_samples rows of the anchor draw (seed + 1000) "
                        "are removed from the eval set (unified protocol). Does "
                        "not affect training, the anchor draw or the PRISM "
                        "records.")

    # PRISM eval
    p.add_argument("--prism_eval_samples", type=int, default=256,
                   help="Number of samples per task for PRISM evaluation")
    p.add_argument("--prism_eval_batch_size", type=int, default=4,
                   help="Batch size for PRISM feature extraction")

    # Shape regularization (W=I trace form). Mutually exclusive with
    # --lambda_replay.
    #
    # Two unregularized references (see the trainer selection in main()):
    #   --lambda_shape 0 --lambda_replay 0   stock transformers Trainer
    #   --lambda_shape 1e-9                  regularized training loop with a
    #                                        numerically negligible penalty
    p.add_argument("--lambda_shape", type=float, default=0.0,
                   help="Weight for shape regularizer 1-Ω_I. Mutually "
                        "exclusive with --lambda_replay. Any value > 0 selects "
                        "the regularized training loop. With both λ at 0 the "
                        "stock transformers Trainer is used (the unregularized "
                        "run of the λ sweeps in run_forgetting.sh). The "
                        "same-loop unregularized reference of the unified "
                        "protocol is --lambda_shape 1e-9: the regularized loop "
                        "runs, including the anchor-set forward/backward, and "
                        "1 − Ω_I lies in [0, 2], so the weighted penalty "
                        "contributes at most 2e-9 nats. Negative values are "
                        "rejected.")
    p.add_argument("--lambda_replay", type=float, default=0.0,
                   help="Weight of the cross-entropy penalty on a "
                        "fixed set of --reg_samples sequences chosen by "
                        "--replay_source: anchor-CE with 'task', generic replay "
                        "with 'generic' (0 = disabled). Same scheduling and "
                        "loss scaling as --lambda_shape; only the penalty "
                        "differs. Mutually exclusive with --lambda_shape. "
                        "Negative values are rejected.")
    p.add_argument("--replay_source", default="task",
                   choices=["task", "generic"],
                   help="Sequences the --lambda_replay penalty is computed on: "
                        "'task' = the anchor set, i.e. the held-out sequences of "
                        "the fine-tuning task that the shape regularizer reads "
                        "(default; anchor-CE); "
                        "'generic' = the same number of sequences of a "
                        "pre-training-like corpus (WikiText train; generic "
                        "replay). Ignored by the shape regularizer, which "
                        "always reads the anchor set.")
    p.add_argument("--reg_every_k", type=int, default=8,
                   help="Compute the penalty (shape or cross-entropy) every K "
                        "micro-batches (default: 8, a constant; it is not "
                        "derived from --grad_accum. The reported runs use "
                        "8 = --grad_accum, i.e. one penalty evaluation per "
                        "optimizer step)")
    p.add_argument("--reg_samples", type=int, default=32,
                   help="Number of sequences in the anchor set (and in the "
                        "generic-replay set)")
    p.add_argument("--reg_batch_size", type=int, default=8,
                   help="Batch size for the anchor-set forward pass")
    p.add_argument("--reg_max_length", type=int, default=512,
                   help="Max sequence length for the anchor-set forward pass.")

    # Misc
    p.add_argument("--seed", type=int, default=42,
                   help="Sets the initial LoRA weights, the Trainer seed "
                        "(order of the training data, dropout), the anchor "
                        "draw (seed + 1000) and the benchmark subsets of the "
                        "PRISM evaluation.")
    p.add_argument("--logging_steps", type=int, default=10)
    return p.parse_args()


# ======================================================================
# Data loading  (for training only; PRISM eval uses prism.data.loaders)
# ======================================================================

def _load_hf_dataset(task_name: str, split: str):
    cfg = TASK_CONFIGS[task_name]
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

    # ── Mutual exclusion: shape penalty and CE penalty cannot both be on ──
    if args.lambda_shape > 0 and args.lambda_replay > 0:
        print("ERROR: --lambda_shape and --lambda_replay are mutually exclusive; "
              "set at most one to a positive value.", file=sys.stderr)
        sys.exit(2)

    if args.lambda_shape < 0 or args.lambda_replay < 0:
        print("ERROR: --lambda_shape and --lambda_replay must be >= 0 "
              f"(got {args.lambda_shape} and {args.lambda_replay}).",
              file=sys.stderr)
        sys.exit(2)

    task_cfg = TASK_CONFIGS[args.task]

    # ── Resolve defaults ─────────────────────────────────────────────
    max_steps = args.max_steps or task_cfg["default_max_steps"]
    save_steps = args.save_steps or task_cfg["default_save_steps"]
    lr = args.lr
    max_train_samples = task_cfg["max_train_samples"]
    max_eval_samples = task_cfg["max_eval_samples"]

    model_short = args.model.split("/")[-1].lower()
    output_dir = args.output_dir or os.path.join(
        "checkpoints", "forgetting", model_short, args.task,
    )
    os.makedirs(output_dir, exist_ok=True)

    warmup_steps = int(args.warmup_ratio * max_steps)

    # ── Eval tasks: 5 downstream benchmarks (disjoint from trained task) ─
    eval_tasks = get_eval_tasks(args.task)

    # Revision the model repository is loaded at (configs/model_revisions.yaml);
    # None = not pinned, the repository head is loaded.
    revision = pinned_revision(args.model)

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
        "replay_source": (args.replay_source if args.lambda_replay > 0 else None),
        "revision": revision,
        "eval_keep_anchor_rows": args.eval_keep_anchor_rows,
        # Added below, once known: n_eval_rows, eval_anchor_rows_removed,
        # optim, weight_decay, lr_scheduler, max_grad_norm, K_feat, K_pred.
    }

    print(f"{'=' * 78}")
    print(f"  PRISM Forgetting: LoRA Fine-Tuning + Online Monitoring")
    print(f"{'=' * 78}")
    for k, v in experiment_config.items():
        print(f"  {k:<25s}: {v}")
    print(f"  output_dir              : {output_dir}")
    print(f"{'=' * 78}")

    # ── Tokenizer ────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=_TRUST_REMOTE_CODE,
        revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model (before LoRA; used for base feature extraction) ───────
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=_TRUST_REMOTE_CODE,
        revision=revision,
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

    # ── Pre-compute anchor / generic-replay data + (for shape only) Z_T_ref ──
    # The shape regularizer and anchor-CE (--replay_source task) read the
    # same anchor set (see build_ref_dataloader); only the shape regularizer
    # needs Z_T (base-model features) pre-extracted, since the cross-entropy
    # penalty uses the current model's CE on its inputs.
    Z_T_ref = None
    ref_dataloader = None
    use_reg = args.lambda_shape > 0 or args.lambda_replay > 0
    if use_reg:
        ref_dataloader = build_ref_dataloader(args, tokenizer)

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
    # --seed also sets the adapter initialisation: the generators are seeded
    # immediately before the adapter is created.
    set_seed(args.seed)
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
    # Target-task evaluation rows. The anchor set (build_ref_dataloader, source
    # "task") is drawn from this same held-out slice: prism.data.loaders split
    # "test" is the slice TASK_CONFIGS calls eval_split, shuffled with seed
    # args.seed + 1000, first --reg_samples rows. The slice is disjoint from
    # train_split, so the anchor rows are never fine-tuning data.
    #
    # Default (unified protocol): the rows of that draw are removed from the
    # eval set, so eval_loss is computed on held-out rows outside the anchor
    # set. The selection below replays the loader's draw (same shuffle seed over
    # the same number of rows), so it depends only on (task, seed, reg_samples):
    # identical across methods at a given seed, redrawn for each seed. The
    # removal does not depend on the method:
    #   * shape regularizer, anchor-CE (--replay_source task): the removed rows
    #     are the rows the penalty is computed on;
    #   * generic replay (--replay_source generic): the penalty reads WikiText; the
    #     removed rows are the rows the task source would draw at this seed;
    #   * both λ = 0 (library Trainer): no anchor set is built and no row of the
    #     slice is read during training; the removed rows are the rows a
    #     regularized run at this seed would use as anchors, so that eval_loss
    #     is on the same rows for every method.
    #
    # --eval_keep_anchor_rows: nothing is removed and eval_loss is computed on
    # the whole slice (163 rows for TruthfulQA), as in the λ sweeps of
    # run_forgetting.sh. The flag only selects the rows of eval_loss; training, the
    # anchor draw and the PRISM records are the same with and without it.
    #
    # The removal applies when the eval slice is used whole (max_eval_samples is
    # None: TruthfulQA and BBQ, the paper's fine-tuning tasks).
    n_removed = 0
    if max_eval_samples is None and not args.eval_keep_anchor_rows:
        _n = len(eval_dataset)
        _ref = set(
            eval_dataset.add_column("_row", list(range(_n)))
            .shuffle(seed=args.seed + 1000).select(range(args.reg_samples))["_row"]
        )
        eval_dataset = eval_dataset.select([i for i in range(_n) if i not in _ref])
        n_removed = len(_ref)
        print(f"  Eval size after removing the {len(_ref)} anchor rows: "
              f"{len(eval_dataset):,} examples")
    else:
        print(f"  Eval set as built, anchor rows not removed: "
              f"{len(eval_dataset):,} examples")
    experiment_config["n_eval_rows"] = len(eval_dataset)
    experiment_config["eval_anchor_rows_removed"] = n_removed

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
    # The optimizer depends on whether bitsandbytes can be imported.
    print(f"  Optimizer: {optim}")

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
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=save_steps,
        bf16=True,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,   # keep prompt_length for AnswerOnlyDataCollator
        seed=args.seed,
    )
    # Recorded for the metrics JSON (the dict is shared with the callback).
    experiment_config["optim"] = optim
    experiment_config["weight_decay"] = training_args.weight_decay
    experiment_config["lr_scheduler"] = str(
        getattr(training_args.lr_scheduler_type, "value",
                training_args.lr_scheduler_type))
    experiment_config["max_grad_norm"] = training_args.max_grad_norm

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[prism_callback],
    )
    # ── Trainer selection ────────────────────────────────────────────
    #   --lambda_shape  > 0                 ShapeRegularizedTrainer
    #   --lambda_replay > 0                 ReplayCETrainer (same loop, CE penalty)
    #   both 0                              stock transformers Trainer
    #
    # There are two unregularized references:
    #
    #   (a) stock-Trainer reference: `--lambda_shape 0 --lambda_replay 0`.
    #       This is the unregularized run of the λ sweeps launched by
    #       run_forgetting.sh. It does not build the anchor set and runs no
    #       anchor forward pass.
    #
    #   (b) same-loop reference: `--lambda_shape 1e-9`. The run goes through
    #       ShapeRegularizedTrainer exactly like a regularized run (anchor-set
    #       forward and backward every reg_every_k micro-steps); 1 − Ω_I lies
    #       in [0, 2], so the weighted penalty contributes at most 2e-9 nats.
    #       This is the "no regularization" reference of the unified protocol, in which
    #       every method and the reference share one training loop.
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
        print(f"\n  Cross-entropy penalty (source: {args.replay_source}): "
              f"λ={args.lambda_replay}, "
              f"every_k={args.reg_every_k}, samples={args.reg_samples}")
        trainer = ReplayCETrainer(
            **trainer_kwargs,
            Z_T_ref=None,                        # unused by the CE penalty
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
