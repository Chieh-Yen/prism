"""
Out-of-Distribution Detection via Geometric Consistency — Rotational Regime.

Two models (Target & Proxy) form a *consistent geometric view* for ID data.
The alignment map W learned on ID data breaks for OOD inputs, producing a
high per-sample consistency score  s(x) = ‖φ_T(x) − φ_P(x) W‖₂.

Reports:
  - Per-sample s(x) for ID and OOD data
  - AUROC for OOD detection
  - Global Omega and discrepancy on both distributions
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from ..core.metrics import PRISMMetrics
from ..models.extractors import FeatureExtractor, get_extractor
from ..data.loaders import load_task_data
from .base import BaseExperiment


def auroc_from_scores(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    """Compute AUROC where *higher score = more likely OOD*."""
    labels = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])
    scores = np.concatenate([scores_id, scores_ood])

    # Sort by score descending
    order = np.argsort(-scores)
    labels_sorted = labels[order]

    n_pos = labels_sorted.sum()
    n_neg = len(labels_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    tpr = tp / n_pos
    fpr = fp / n_neg

    # Trapezoidal integration
    auroc = np.trapz(tpr, fpr)
    return float(auroc)


class OODExperiment(BaseExperiment):
    """Detect OOD inputs via multi-view geometric consistency.

    Config shape::

        target:
          model: openai/clip-vit-large-patch14
          extractor: clip
        proxy:
          model: openai/clip-vit-base-patch32
          extractor: clip
        data:
          id_task: cifar10           # in-distribution task
          ood_tasks:                  # one or more OOD tasks
            - svhn
            - mnist
          num_samples: 1000
          batch_size: 64
    """

    def setup_pairs(self) -> List[Dict[str, Any]]:
        """Not used directly — see ``run()``."""
        return []

    def run(self) -> list:
        print(f"{'=' * 72}")
        print(f"  PRISM Experiment: OOD Detection via Geometric Consistency")
        print(f"{'=' * 72}")

        cfg_target = self.config.get("target", {})
        cfg_proxy = self.config.get("proxy", {})
        cfg_data = self.config.get("data", {})

        target_model_id = cfg_target.get("model")
        proxy_model_id = cfg_proxy.get("model")
        target_ext_name = cfg_target.get("extractor", "clip")
        proxy_ext_name = cfg_proxy.get("extractor", "clip")

        id_task = cfg_data.get("id_task", "cifar10")
        ood_tasks: List[str] = cfg_data.get("ood_tasks", ["svhn"])
        num_samples = cfg_data.get("num_samples", 1000)
        batch_size = cfg_data.get("batch_size", 64)

        # Load models
        from transformers import AutoModel, AutoProcessor

        print(f"Loading target: {target_model_id} ...")
        target_model = AutoModel.from_pretrained(target_model_id).to(self.device)
        target_model.eval()

        print(f"Loading proxy:  {proxy_model_id} ...")
        proxy_model = AutoModel.from_pretrained(proxy_model_id).to(self.device)
        proxy_model.eval()

        ext_t = get_extractor(target_ext_name)
        ext_p = get_extractor(proxy_ext_name)

        # Shared image transform from target's processor
        processor = AutoProcessor.from_pretrained(target_model_id)

        def image_transform(img):
            return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        # ---- ID data ----
        print(f"\nLoading ID data: {id_task} (n={num_samples}) ...")
        id_dl = load_task_data(id_task, num_samples=num_samples, batch_size=batch_size, transform=image_transform)

        Z_T_id = ext_t.extract_features(target_model, id_dl, self.device)
        Z_P_id = ext_p.extract_features(proxy_model, id_dl, self.device)

        # Learn alignment W on ID data
        W = PRISMMetrics.orthogonal_procrustes(Z_T_id, Z_P_id)
        omega_id = PRISMMetrics.procrustes_omega(Z_T_id, Z_P_id)
        scores_id = PRISMMetrics.consistency_scores(Z_T_id, Z_P_id, W).numpy()

        print(f"  ID Omega={omega_id:.4f}  mean_s={scores_id.mean():.4f}  std_s={scores_id.std():.4f}")

        # ---- OOD data ----
        results = []
        for ood_task in ood_tasks:
            t0 = time.time()
            print(f"\nLoading OOD data: {ood_task} (n={num_samples}) ...")
            ood_dl = load_task_data(ood_task, num_samples=num_samples, batch_size=batch_size, transform=image_transform)

            Z_T_ood = ext_t.extract_features(target_model, ood_dl, self.device)
            Z_P_ood = ext_p.extract_features(proxy_model, ood_dl, self.device)

            omega_ood = PRISMMetrics.procrustes_omega(Z_T_ood, Z_P_ood)
            scores_ood = PRISMMetrics.consistency_scores(Z_T_ood, Z_P_ood, W).numpy()

            auc = auroc_from_scores(scores_id, scores_ood)

            from ..core.metrics import PRISMResult
            result = PRISMResult(
                omega=omega_ood,
                rho_target=PRISMMetrics.rms_scale(Z_T_ood),
                rho_proxy=PRISMMetrics.rms_scale(Z_P_ood),
                scale_mismatch=0.0,
                shape_mismatch=0.0,
                feature_error=0.0,
                head_discrepancy=0.0,
                head_discrepancy_spectral=0.0,
                label=f"ID:{id_task} vs OOD:{ood_task}",
            )
            result.extra["omega_id"] = omega_id
            result.extra["omega_ood"] = omega_ood
            result.extra["auroc"] = auc
            result.extra["mean_score_id"] = float(scores_id.mean())
            result.extra["mean_score_ood"] = float(scores_ood.mean())
            result.extra["std_score_id"] = float(scores_id.std())
            result.extra["std_score_ood"] = float(scores_ood.std())

            results.append(result)
            elapsed = time.time() - t0
            print(f"  OOD Omega={omega_ood:.4f}  mean_s={scores_ood.mean():.4f}  "
                  f"AUROC={auc:.4f}  ({elapsed:.1f}s)")

        # Report
        self._report_ood(results, omega_id, scores_id)
        self.save(results)
        return results

    @staticmethod
    def _report_ood(results, omega_id: float, scores_id: np.ndarray) -> None:
        print(f"\n{'=' * 72}")
        print(f"  OOD Detection Results  (ID Omega={omega_id:.4f})")
        print(f"{'=' * 72}")
        header = f"{'OOD Task':<30s} {'Omega_OOD':>10s} {'mean_s_OOD':>10s} {'AUROC':>8s}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r.label:<30s} {r.extra['omega_ood']:>10.4f} "
                f"{r.extra['mean_score_ood']:>10.4f} {r.extra['auroc']:>8.4f}"
            )
        print(f"{'=' * 72}\n")
