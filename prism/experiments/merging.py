"""
Cross-Scale Coefficient Transfer in Model Merging — Projective Regime (d_P ≠ d_T).

Target = large model family (e.g. ViT-L-14).
Proxy  = small model family (e.g. ViT-B-32).

We search for the optimal merging coefficient α on the cheap proxy, then
transfer it to the expensive target.  The transfer regret bound (Sec. 5.4):

    R_T(α̂_P) − R_T(α*_T) ≤ 2 [K_feat·δ + ε_head] + 2ξ
"""

from __future__ import annotations

import copy
import gc
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from transformers import AutoModel, AutoProcessor

from ..core.metrics import PRISMMetrics
from ..core.bounds import UnifiedBound
from ..data.loaders import load_task_data
from ..models.extractors import CLIPExtractor
from .base import BaseExperiment


def _state_dict_cpu(model_name: str) -> Dict[str, Tensor]:
    """Load a model's state dict onto CPU."""
    model = AutoModel.from_pretrained(model_name)
    sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    return sd


def _task_vector(base_sd: Dict[str, Tensor], ft_sd: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Compute θ_ft − θ_base for shared keys."""
    tv = {}
    for k in base_sd:
        if k in ft_sd and base_sd[k].shape == ft_sd[k].shape:
            tv[k] = ft_sd[k] - base_sd[k]
    return tv


def _merge_into(
    model: torch.nn.Module,
    base_sd: Dict[str, Tensor],
    task_vectors: List[Dict[str, Tensor]],
    alpha: np.ndarray,
    device: str,
) -> None:
    """In-place merge:  θ = θ_base + Σ_i α_i Δ_i."""
    merged = {}
    for k, v in base_sd.items():
        val = v.clone()
        for i, tv in enumerate(task_vectors):
            if k in tv:
                val = val + alpha[i] * tv[k]
        merged[k] = val.to(device)
    model.load_state_dict(merged, strict=False)
    model.eval()


class MergingExperiment(BaseExperiment):
    """Cross-scale coefficient transfer via task arithmetic.

    Config shape::

        target:
          model: openai/clip-vit-large-patch14
          extractor: clip
          fine_tuned:                        # one per task
            - hf-org/vit-l-14-sun397
            - hf-org/vit-l-14-stanford-cars
        proxy:
          model: openai/clip-vit-base-patch32
          extractor: clip
          fine_tuned:
            - hf-org/vit-b-32-sun397
            - hf-org/vit-b-32-stanford-cars
        data:
          tasks:
            - sun397
            - stanford-cars
          num_samples_per_class: 10
          batch_size: 64
        merging:
          n_alphas: 20
          alpha_range: [0.0, 1.0]
          seed: 42
    """

    def setup_pairs(self) -> List[Dict[str, Any]]:
        """Not used directly — see ``run()``."""
        return []

    def run(self) -> list:
        print(f"{'=' * 72}")
        print(f"  PRISM Experiment: Cross-Scale Coefficient Transfer")
        print(f"{'=' * 72}")

        cfg_target = self.config.get("target", {})
        cfg_proxy = self.config.get("proxy", {})
        cfg_data = self.config.get("data", {})
        cfg_merge = self.config.get("merging", {})

        target_base_id = cfg_target["model"]
        proxy_base_id = cfg_proxy["model"]
        target_ft_ids: List[str] = cfg_target.get("fine_tuned", [])
        proxy_ft_ids: List[str] = cfg_proxy.get("fine_tuned", [])
        tasks: List[str] = cfg_data.get("tasks", [])
        num_samples = cfg_data.get("num_samples_per_class", 10)
        batch_size = cfg_data.get("batch_size", 64)

        n_alphas = cfg_merge.get("n_alphas", 20)
        alpha_lo, alpha_hi = cfg_merge.get("alpha_range", [0.0, 1.0])
        merge_seed = cfg_merge.get("seed", self.seed)
        num_tasks = len(tasks)

        if num_tasks == 0:
            raise ValueError("data.tasks must list at least one task.")
        if len(target_ft_ids) != num_tasks or len(proxy_ft_ids) != num_tasks:
            raise ValueError("Number of fine_tuned checkpoints must match number of tasks.")

        # Sample alpha vectors
        rng = np.random.RandomState(merge_seed)
        alphas = rng.uniform(alpha_lo, alpha_hi, size=(n_alphas, num_tasks))

        # Image transform
        processor = AutoProcessor.from_pretrained(target_base_id)

        def image_transform(img):
            return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        # Combined dataloader (all tasks)
        from torch.utils.data import ConcatDataset, DataLoader
        all_ds = []
        for task in tasks:
            dl = load_task_data(task, num_samples=num_samples, batch_size=1, transform=image_transform)
            all_ds.append(dl.dataset)

        combined_dl = DataLoader(
            ConcatDataset(all_ds), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        # Task vectors (CPU)
        print("\nComputing task vectors ...")
        target_base_sd = _state_dict_cpu(target_base_id)
        proxy_base_sd = _state_dict_cpu(proxy_base_id)

        target_tvs = [_task_vector(target_base_sd, _state_dict_cpu(ft)) for ft in target_ft_ids]
        proxy_tvs = [_task_vector(proxy_base_sd, _state_dict_cpu(ft)) for ft in proxy_ft_ids]

        # Reusable GPU models
        target_model = AutoModel.from_pretrained(target_base_id).to(self.device)
        proxy_model = AutoModel.from_pretrained(proxy_base_id).to(self.device)
        ext = CLIPExtractor()

        results = []
        for idx, alpha in enumerate(alphas):
            t0 = time.time()
            label = f"alpha[{idx}]={np.array2string(alpha, precision=2, separator=',')}"
            print(f"\n--- [{idx+1}/{n_alphas}] {label} ---")

            _merge_into(target_model, target_base_sd, target_tvs, alpha, self.device)
            _merge_into(proxy_model, proxy_base_sd, proxy_tvs, alpha, self.device)

            Z_T = ext.extract_features(target_model, combined_dl, self.device)
            Z_P = ext.extract_features(proxy_model, combined_dl, self.device)

            # Use dummy head (identity-like) since head discrepancy is evaluated
            # separately per task in full pipeline.  Here we focus on backbone Omega.
            d_T, d_P = Z_T.shape[1], Z_P.shape[1]
            d_min = min(d_T, d_P)
            H_T = torch.eye(d_T, d_min)
            H_P = torch.eye(d_P, d_min)

            result = self.compute_metrics(Z_T, H_T, Z_P, H_P, label=label)
            result.extra["alpha"] = alpha.tolist()

            results.append(result)
            elapsed = time.time() - t0
            print(f"    Omega={result.omega:.4f}  Scale={result.scale_mismatch:.6f}  "
                  f"Shape={result.shape_mismatch:.6f}  ({elapsed:.1f}s)")

        # Best alpha (highest Omega)
        best_idx = int(np.argmax([r.omega for r in results]))
        print(f"\nBest alpha (Omega={results[best_idx].omega:.4f}): "
              f"{results[best_idx].extra['alpha']}")

        self.report(results)
        self.save(results)
        return results
