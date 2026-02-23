"""
Geometric Monitoring of Catastrophic Forgetting — Rotational Regime (W ∈ O(d)).

Target = frozen base model (θ₀).
Proxy  = model at training step t (θ_t).

The forgetting bound (Eq. 8):
    Forgetting(t) ≤ K_feat · √[ (ρ₀ − ρ_t)² + 2 ρ₀ ρ_t (1 − Ω) ] + K_pred · γ

Also provides the Ω-regulariser (Eq. 9):
    L_geom = 1 − Ω(Z₀, Z_t)
"""

from __future__ import annotations

import gc
import time
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..core.metrics import PRISMMetrics
from ..data.loaders import load_task_data
from ..models.extractors import LLMExtractor
from .base import BaseExperiment


class ForgettingExperiment(BaseExperiment):
    """Track geometric drift of a model away from its pre-training manifold.

    Config shape::

        target:
          model: meta-llama/Llama-2-7b-hf       # base / pre-trained
          extractor: llm
        proxy:
          checkpoints:                            # list of fine-tuned checkpoints
            - path/to/step_100
            - path/to/step_500
            - path/to/step_1000
          extractor: llm
        data:
          task: wikitext
          num_samples: 256
    """

    def setup_pairs(self) -> List[Dict[str, Any]]:
        cfg_target = self.config.get("target", {})
        cfg_proxy = self.config.get("proxy", {})
        cfg_data = self.config.get("data", {})

        base_model_id = cfg_target.get("model")
        checkpoint_paths: List[str] = cfg_proxy.get("checkpoints", [])
        task_name = cfg_data.get("task", "wikitext")
        num_samples = cfg_data.get("num_samples", 256)
        batch_size = cfg_data.get("batch_size", 8)
        max_length = cfg_data.get("max_length", 512)

        if not checkpoint_paths:
            raise ValueError("proxy.checkpoints must list at least one checkpoint path.")

        print(f"Loading tokenizer from {base_model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading data: {task_name} (n={num_samples}) ...")
        dataloader = load_task_data(
            task_name, split="test", num_samples=num_samples,
            batch_size=batch_size, tokenizer=tokenizer, max_length=max_length,
        )

        print(f"Loading base model (target): {base_model_id} ...")
        target_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, dtype=torch.float16,
            device_map=self.device,
        )
        target_model.eval()
        extractor = LLMExtractor()

        # Pre-extract target features once
        print("Pre-extracting target features ...")
        Z_T = extractor.extract_features(target_model, dataloader, self.device)
        H_T = extractor.extract_head(target_model)

        pairs: List[Dict[str, Any]] = []
        for ckpt_path in checkpoint_paths:
            label = ckpt_path.rsplit("/", 1)[-1] if "/" in ckpt_path else ckpt_path
            pairs.append({
                "label": f"base vs {label}",
                "target_model": target_model,
                "proxy_model": None,
                "dataloader": dataloader,
                "extractor_target": extractor,
                "extractor_proxy": extractor,
                "force_identity": False,
                "_ckpt_path": ckpt_path,
                "_Z_T": Z_T,
                "_H_T": H_T,
                "_tokenizer": tokenizer,
            })

        return pairs

    def run(self) -> list:
        print(f"{'=' * 72}")
        print(f"  PRISM Experiment: Catastrophic Forgetting Monitoring")
        print(f"{'=' * 72}")

        pairs = self.setup_pairs()
        results = []
        extractor = LLMExtractor()

        for i, pair in enumerate(pairs):
            label = pair["label"]
            ckpt_path = pair.pop("_ckpt_path")
            Z_T = pair.pop("_Z_T")
            H_T = pair.pop("_H_T")

            print(f"\n--- [{i+1}/{len(pairs)}] {label} ---")
            print(f"    Loading checkpoint: {ckpt_path} ...")
            t0 = time.time()

            proxy_model = AutoModelForCausalLM.from_pretrained(
                ckpt_path, dtype=torch.float16,
                device_map=self.device,
            )
            proxy_model.eval()

            dl = pair["dataloader"]
            Z_P = extractor.extract_features(proxy_model, dl, self.device)
            H_P = extractor.extract_head(proxy_model)

            result = self.compute_metrics(Z_T, H_T, Z_P, H_P, label=label)

            # LM loss on pre-training distribution
            result.loss_target = self.compute_lm_loss(pair["target_model"], dl, self.device)
            result.loss_proxy = self.compute_lm_loss(proxy_model, dl, self.device)

            omega_reg = 1.0 - result.omega
            result.extra["omega_regulariser"] = omega_reg
            result.extra["forgetting_delta_loss"] = (
                result.loss_proxy - result.loss_target if result.loss_target is not None else None
            )

            results.append(result)
            elapsed = time.time() - t0
            dr = abs(result.loss_target - result.loss_proxy)
            bt = result.risk_bound_total
            bt_s = f"Bound={bt:.4f}  " if bt is not None else ""
            print(f"    Omega={result.omega:.4f}  L_geom={omega_reg:.4f}  "
                  f"{bt_s}|dR|={dr:.4f}  "
                  f"Loss_base={result.loss_target:.4f}  Loss_ft={result.loss_proxy:.4f}  "
                  f"({elapsed:.1f}s)")

            del proxy_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.report(results)
        self.save(results)
        return results

    # ------------------------------------------------------------------
    # Utility: compute the Ω-regulariser loss  (Eq. 9)
    # ------------------------------------------------------------------
    @staticmethod
    def omega_regulariser_loss(
        Z_base: Tensor,
        Z_current: Tensor,
    ) -> Tensor:
        """Differentiable Ω-regulariser: L_geom = 1 − Ω(Z₀, Z_t).

        Can be added to a training loop as:
            loss_total = loss_task + lambda * omega_regulariser_loss(Z_0, Z_t)
        """
        cross = Z_base.T @ Z_current
        nuclear_norm = torch.linalg.svdvals(cross).sum()
        denom = Z_base.norm("fro") * Z_current.norm("fro")
        omega = nuclear_norm / denom.clamp(min=1e-12)
        return 1.0 - omega
