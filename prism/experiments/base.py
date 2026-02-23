"""
BaseExperiment — shared pipeline for all PRISM experiments.

Subclasses override ``setup_pairs()`` to define which (target, proxy) model
pairs to compare.  Everything else (feature extraction, metric computation,
reporting) is handled uniformly.
"""

from __future__ import annotations

import gc
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..core.bounds import UnifiedBound
from ..core.metrics import PRISMMetrics, PRISMResult
from ..models.extractors import FeatureExtractor, get_extractor


class BaseExperiment(ABC):
    """Template for a PRISM experiment.

    Lifecycle::

        exp = SomeExperiment(config)
        results = exp.run()          # setup → iterate pairs → compute metrics
        exp.report(results)          # pretty-print table
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.seed = config.get("seed")
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.output_dir = config.get("output", {}).get("dir", "./results")
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def setup_pairs(self) -> List[Dict[str, Any]]:
        """Return a list of pair descriptors to evaluate.

        Each dict must contain at least:
            - ``"label"``: str      — human-readable tag, e.g. "INT4"
            - ``"target_model"``    — loaded nn.Module (on device)
            - ``"proxy_model"``     — loaded nn.Module (on device)
            - ``"dataloader"``      — DataLoader for feature extraction
            - ``"extractor_target"``: FeatureExtractor
            - ``"extractor_proxy"``:  FeatureExtractor

        Optional keys:
            - ``"H_target"``, ``"H_proxy"`` — precomputed head tensors
            - ``"force_identity"``  — bool, default False
            - ``"eval_dataloader"`` — separate DataLoader for loss eval
            - ``"eval_fn_target"``, ``"eval_fn_proxy"``
        """

    def cleanup_pair(self, pair: Dict[str, Any]) -> None:
        """Free GPU memory after processing a pair.  Override if needed."""
        for key in ("target_model", "proxy_model"):
            if key in pair:
                del pair[key]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Core pipeline steps
    # ------------------------------------------------------------------
    def extract(
        self,
        extractor: FeatureExtractor,
        model: torch.nn.Module,
        dataloader: DataLoader,
        head_kwargs: Optional[Dict] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Extract (Z, H) from a model."""
        Z = extractor.extract_features(model, dataloader, self.device)
        H = extractor.extract_head(model, **(head_kwargs or {}))
        return Z, H

    def compute_metrics(
        self,
        Z_T: Tensor,
        H_T: Tensor,
        Z_P: Tensor,
        H_P: Tensor,
        *,
        force_identity: bool = False,
        label: str = "",
        K_feat: float = 1.0,
        K_pred: float = 1.0,
        absorbed: bool = False,
    ) -> PRISMResult:
        """Run full PRISM metric suite and fill risk bound.

        Args:
            absorbed: If True, use scale-absorbed reparameterisation where
                feature error = √(2(1−Ω)) and scale is folded into head.
        """
        compute_fn = PRISMMetrics.compute_all_absorbed if absorbed else PRISMMetrics.compute_all
        result = compute_fn(
            Z_T, H_T, Z_P, H_P,
            force_identity=force_identity,
            label=label,
        )
        UnifiedBound.fill_result(result, K_feat=K_feat, K_pred=K_pred)
        return result

    # ------------------------------------------------------------------
    # Optional: loss evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def compute_lm_loss(
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str,
    ) -> float:
        """Compute average cross-entropy for a causal LM."""
        stats = BaseExperiment.compute_lm_loss_per_sample(model, dataloader, device)
        return stats["losses"].mean().item()

    @staticmethod
    def compute_lm_loss_per_sample(
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str,
    ) -> Dict[str, Tensor]:
        """Compute per-sample cross-entropy and per-token gradient norms.

        For each sample, also computes the per-token logit-gradient norm
        ``||softmax(v) - e_y||_2`` which is the local Lipschitz constant
        of cross-entropy w.r.t. logits (K_pred).  These are aggregated
        into summary statistics without materialising huge tensors.

        Returns:
            Dict with:
                ``losses``       — (n,) per-sample average loss
                ``grad_norm_p95``— 95th percentile of per-token ||p-e_y||
                ``grad_norm_max``— max of per-token ||p-e_y||
                ``grad_norm_mean``— mean of per-token ||p-e_y||
        """
        model.eval()
        sample_losses: list = []
        all_grad_norms: list = []
        CHUNK = 512

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    batch_on_device = {k: v.to(device) for k, v in batch.items()}
                else:
                    batch_on_device = {"input_ids": batch.to(device)}

                bsz = batch_on_device["input_ids"].shape[0]
                for j in range(bsz):
                    single = {k: v[j : j + 1] for k, v in batch_on_device.items()}
                    labels = single["input_ids"].clone()
                    outputs = model(**single, labels=labels)
                    sample_losses.append(outputs.loss.item())

                    logits = outputs.logits[0, :-1, :]   # (seq-1, V)
                    targets = labels[0, 1:]               # (seq-1,)
                    mask = single.get("attention_mask")
                    if mask is not None:
                        token_mask = mask[0, 1:].bool()
                        logits = logits[token_mask]
                        targets = targets[token_mask]

                    seq_len = logits.shape[0]
                    for start in range(0, seq_len, CHUNK):
                        end = min(start + CHUNK, seq_len)
                        p = torch.softmax(logits[start:end].float(), dim=-1)
                        one_hot = torch.zeros_like(p)
                        one_hot.scatter_(1, targets[start:end].unsqueeze(1), 1.0)
                        gnorms = (p - one_hot).norm(dim=-1)  # (chunk,)
                        all_grad_norms.append(gnorms.cpu())

        all_gn = torch.cat(all_grad_norms)
        return {
            "losses": torch.tensor(sample_losses),
            "grad_norm_p95": torch.quantile(all_gn, 0.95).item(),
            "grad_norm_max": all_gn.max().item(),
            "grad_norm_mean": all_gn.mean().item(),
        }

    @staticmethod
    def compute_classification_loss(
        model: torch.nn.Module,
        dataloader: DataLoader,
        head_weights: Tensor,
        logit_scale: float,
        logit_bias: float,
        device: str,
    ) -> Tuple[float, float]:
        """Compute (avg_loss, accuracy) for a CLIP-style zero-shot classifier."""
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        head_weights = head_weights.to(device)

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                feats = model.get_image_features(pixel_values=images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                logits = (feats @ head_weights.T) * logit_scale + logit_bias
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------
    def run(self) -> List[PRISMResult]:
        """Execute the full experiment pipeline."""
        print(f"{'=' * 72}")
        print(f"  PRISM Experiment: {self.__class__.__name__}")
        print(f"{'=' * 72}")

        pairs = self.setup_pairs()
        results: List[PRISMResult] = []

        for i, pair in enumerate(pairs):
            label = pair.get("label", f"pair-{i}")
            print(f"\n--- [{i+1}/{len(pairs)}] {label} ---")
            t0 = time.time()

            ext_t = pair["extractor_target"]
            ext_p = pair["extractor_proxy"]
            dl = pair["dataloader"]
            force_id = pair.get("force_identity", False)

            Z_T, H_T = self.extract(ext_t, pair["target_model"], dl, pair.get("head_kwargs_target"))
            Z_P, H_P = self.extract(ext_p, pair["proxy_model"], dl, pair.get("head_kwargs_proxy"))

            if "H_target" in pair:
                H_T = pair["H_target"]
            if "H_proxy" in pair:
                H_P = pair["H_proxy"]

            result = self.compute_metrics(
                Z_T, H_T, Z_P, H_P,
                force_identity=force_id,
                label=label,
                K_feat=pair.get("K_feat", 1.0),
                K_pred=pair.get("K_pred", 1.0),
            )

            if "eval_fn_target" in pair:
                result.loss_target = pair["eval_fn_target"]()
            if "eval_fn_proxy" in pair:
                result.loss_proxy = pair["eval_fn_proxy"]()

            results.append(result)
            elapsed = time.time() - t0
            dr = self._delta_risk(result)
            dr_s = f"|dR|={dr:.4f}  " if dr is not None else ""
            print(f"    Omega={result.omega:.4f}  Scale={result.scale_mismatch:.6f}  "
                  f"Shape={result.shape_mismatch:.6f}  Head={result.head_discrepancy:.6f}  "
                  f"Bound={result.risk_bound_total:.4f}  {dr_s}({elapsed:.1f}s)")

            self.cleanup_pair(pair)

        self.report(results)
        self.save(results)
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @staticmethod
    def _delta_risk(r: PRISMResult) -> Optional[float]:
        """Compute |R_T − R_P| if both losses are available."""
        if r.loss_target is not None and r.loss_proxy is not None:
            return abs(r.loss_target - r.loss_proxy)
        return None

    @staticmethod
    def report(results: List[PRISMResult]) -> None:
        """Print a formatted comparison table."""
        if not results:
            print("(no results)")
            return

        header = (
            f"{'Label':<20s} {'Omega':>8s} {'Scale':>10s} {'Shape':>10s} "
            f"{'Head':>10s} {'Bound':>8s} {'|dR|':>8s} {'Loss_T':>8s} {'Loss_P':>8s}"
        )
        sep = "-" * len(header)
        print(f"\n{'=' * len(header)}")
        print("  PRISM Results Summary")
        print(f"{'=' * len(header)}")
        print(header)
        print(sep)

        for r in results:
            lt = f"{r.loss_target:.4f}" if r.loss_target is not None else "—"
            lp = f"{r.loss_proxy:.4f}" if r.loss_proxy is not None else "—"
            bt = f"{r.risk_bound_total:.4f}" if r.risk_bound_total is not None else "—"
            dr = BaseExperiment._delta_risk(r)
            dr_s = f"{dr:.4f}" if dr is not None else "—"
            print(
                f"{r.label:<20s} {r.omega:>8.4f} {r.scale_mismatch:>10.6f} "
                f"{r.shape_mismatch:>10.6f} {r.head_discrepancy:>10.6f} "
                f"{bt:>8s} {dr_s:>8s} {lt:>8s} {lp:>8s}"
            )

        # Bound validation summary
        valid = [(r, BaseExperiment._delta_risk(r)) for r in results]
        valid = [(r, dr) for r, dr in valid if dr is not None and r.risk_bound_total is not None]
        if valid:
            print(sep)
            holds = sum(1 for r, dr in valid if r.risk_bound_total >= dr)
            print(f"  Bound holds: {holds}/{len(valid)}  "
                  f"({'ALL PASS' if holds == len(valid) else 'SOME VIOLATED'})")

        print(f"{'=' * len(header)}\n")

    def save(self, results: List[PRISMResult], filename: str = "prism_results.json") -> None:
        """Persist results as JSON."""
        path = os.path.join(self.output_dir, filename)
        serialised = []
        for r in results:
            dr = self._delta_risk(r)
            bound_holds = (
                r.risk_bound_total >= dr
                if dr is not None and r.risk_bound_total is not None
                else None
            )
            d = {
                "label": r.label,
                "omega": r.omega,
                "rho_target": r.rho_target,
                "rho_proxy": r.rho_proxy,
                "scale_mismatch": r.scale_mismatch,
                "shape_mismatch": r.shape_mismatch,
                "feature_error": r.feature_error,
                "head_discrepancy": r.head_discrepancy,
                "head_discrepancy_spectral": r.head_discrepancy_spectral,
                "risk_bound_feature": r.risk_bound_feature,
                "risk_bound_head": r.risk_bound_head,
                "risk_bound_total": r.risk_bound_total,
                "loss_target": r.loss_target,
                "loss_proxy": r.loss_proxy,
                "delta_risk": dr,
                "bound_holds": bound_holds,
            }
            d.update(r.extra)
            serialised.append(d)

        with open(path, "w") as f:
            json.dump(serialised, f, indent=2)
        print(f"Results saved to {path}")
