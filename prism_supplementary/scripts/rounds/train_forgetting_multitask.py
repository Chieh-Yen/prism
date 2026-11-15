#!/usr/bin/env python3
"""
Trainer classes, data pipeline and PRISM checkpoint callback imported by
train_forgetting_baselines.py. At every checkpoint the callback computes the
PRISM forgetting metrics on the fine-tuning task and the five benchmarks (see
get_eval_tasks) and writes them to prism_forgetting_metrics.json.
Under LoRA the lm_head is frozen (H_t = H_0), so PRISM's head divergence term
vanishes under identity alignment and the bound of Theorem 1 (Eq. 4 in the
paper) reduces to its feature term (Sec. 3.4).
"""

from __future__ import annotations

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
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

_ROOT = Path(__file__).resolve().parents[2]      # supplementary root (holds prism/)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prism.core.bounds import UnifiedBound                              # noqa: E402
from prism.core.metrics import PRISMMetrics                             # noqa: E402
from prism.data.loaders import get_task_metadata, load_task_data        # noqa: E402
from prism.models.extractors import LLMExtractor                        # noqa: E402

# Remote code is opt-in (PRISM_TRUST_REMOTE_CODE=1). Every model family used in
# the paper is native to the pinned transformers release, so the default is off.
# Model revisions are pinned through configs/model_revisions.yaml.
_TRUST_REMOTE_CODE = os.environ.get("PRISM_TRUST_REMOTE_CODE", "0") == "1"


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

    For comparison, the stock ``Trainer`` passes ``num_items_in_batch`` to
    the model, so its CE is the mean over all G micro-batches together,
    CE_all, and its accumulated gradient is (1/G) · ∇CE_all.  The CE term of
    this class is therefore about G times that of the stock ``Trainer``.

    Every subclass (``ReplayCETrainer`` here, and the baseline trainers in
    train_forgetting_baselines.py) overrides only the penalty P, so all
    regularized methods share this scaling.
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
        trace = (Z_T * Z_P).sum()                            # ⟨Z_T, Z_P⟩_F
        denom = Z_T.norm("fro") * Z_P.norm("fro")
        omega_I = trace / denom.clamp(min=1e-12)

        shape_loss = 1.0 - omega_I
        return shape_loss, omega_I.item()

    # ------------------------------------------------------------------
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """Inject shape regularizer metrics into log entries."""
        if self._count > 0:
            logs["shape_loss"] = round(self._shape_sum / self._count, 6)
            logs["omega_ref"] = round(self._omega_sum / self._count, 6)
            self._shape_sum = 0.0
            self._omega_sum = 0.0
            self._count = 0
        super().log(logs, *args, **kwargs)


# ── Anchor-CE trainer (cross-entropy on the anchor set) ───────────────────

class ReplayCETrainer(ShapeRegularizedTrainer):
    """Anchor-CE: L = L_CE + λ · CE_ref.

    Same scheduling, backward path and loss scaling as
    ``ShapeRegularizedTrainer`` (see its docstring); only the penalty
    differs: the next-token cross-entropy on the anchor set, i.e. the same
    sequences the shape regularizer reads, instead of 1 − Ω_I.

    Z_T_ref is unused (CE only depends on the current-model output).
    Prompt tokens are masked from the CE target (set to -100), so the
    targets are the answer tokens, matching how the fine-tuning CE is
    computed by ``AnswerOnlyDataCollator``.
    """

    def _compute_shape_loss(self, model) -> Tuple[Tensor, float]:
        """Override: next-token CE on the anchor set (answer tokens only),
        averaged over its batches."""
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
        # Skip ShapeRegularizedTrainer.log to avoid duplicate key injection;
        # call grandparent (HF Trainer) directly.
        Trainer.log(self, logs, *args, **kwargs)


# ── Task-specific dataset configuration ───────────────────────────────────

TASK_CONFIGS = {
    # ── Fine-tuning tasks of the paper ───────────────────────────────
    "truthfulqa": {
        "hf_id": "truthfulqa/truthful_qa",
        "hf_subset": "generation",
        "train_split": "validation[:80%]",   # only split; first 80% for train (654 rows)
        "eval_split": "validation[80%:]",    # last 20% for eval (163 rows)
        "max_train_samples": None,
        "max_eval_samples": None,
    },
    "bbq": {
        "hf_id": "lighteval/bbq_helm",
        "hf_subset": "all",
        "train_split": "test[:80%]",         # only split; first 80% for train (~800)
        "eval_split": "test[80%:]",          # last 20% for eval (~200)
        "max_train_samples": None,
        "max_eval_samples": None,
    },
}

BASE_EVAL_TASKS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]


def get_eval_tasks(trained_task: str) -> list:
    """Return eval task list: trained task (first) + the five base tasks."""
    return [trained_task] + BASE_EVAL_TASKS

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
        print(f"  (All losses are answer-only CE. eval_loss is on the fine-tuning task's")
        print(f"   held-out rows; PRISM metrics below use the benchmarks.)")

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
                "K_pred": "√2 — Lipschitz of CE in logits (Appendix A)",
                "bound_feature": "K_feat × δ — feature term of the unified bound",
                "bound_head": "K_pred × γ — head term of the unified bound",
                "bound_total": "K_feat·δ + K_pred·γ — unified risk bound (Theorem 1, Eq. 4)",
                "loss_T": "R(θ₀) — base model answer-only CE loss",
                "loss_P": "R(θ_t) — fine-tuned model answer-only CE loss",
                "delta_risk": "|R(θ_t) − R(θ₀)| — empirical forgetting (answer-only)",
                "loss_T_full": "full-sequence CE loss (supplementary)",
                "loss_P_full": "full-sequence CE loss (supplementary)",
                "train_loss": "most recent `loss` logged by the Trainer at or before this "
                              "step: answer-only CE on the fine-tuning task, averaged over "
                              "the optimizer steps of the last logging window. Per optimizer "
                              "step the library Trainer logs the mean over "
                              "the G = grad_accum micro-batches; the regularized trainers "
                              "log the sum of the G micro-batch means (about G times as "
                              "large); the penalty is not included",
                "eval_loss": ("Trainer eval loss: answer-only CE on the held-out slice of the "
                              "fine-tuning task, as aggregated by the library. Rows: "
                              "experiment.n_eval_rows; the whole slice when "
                              "experiment.eval_keep_anchor_rows is true, without the "
                              "anchor rows otherwise"),
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
    same frozen base head), the constants of the paper's bound.

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
