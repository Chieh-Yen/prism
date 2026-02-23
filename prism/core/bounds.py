"""
Unified Risk Discrepancy Bound (Theorem 2) and Lipschitz estimation.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor

from .metrics import PRISMMetrics, PRISMResult


class UnifiedBound:
    """Compute the PRISM Unified Risk Discrepancy Bound.

    |R_T − R_P| ≤ K_feat · δ  +  K_pred · γ

    where
        δ = √[ (ρ_T − ρ_P)² + 2 ρ_T ρ_P (1 − Ω) ]   (feature error)
        γ = ‖Σ_P^{1/2}(W H_T − H_P)‖_F                (head discrepancy)
    """

    @staticmethod
    def compute_bound(
        omega: float,
        rho_T: float,
        rho_P: float,
        head_discrepancy: float,
        K_feat: float = 1.0,
        K_pred: float = 1.0,
    ) -> Dict[str, float]:
        feat_err = PRISMMetrics.feature_error(rho_T, rho_P, omega)
        rb_feat = K_feat * feat_err
        rb_head = K_pred * head_discrepancy

        return {
            "risk_bound_total": rb_feat + rb_head,
            "risk_bound_feature": rb_feat,
            "risk_bound_head": rb_head,
            "feature_error": feat_err,
            "scale_mismatch": PRISMMetrics.scale_mismatch(rho_T, rho_P),
            "shape_mismatch": PRISMMetrics.shape_mismatch(rho_T, rho_P, omega),
            "K_feat": K_feat,
            "K_pred": K_pred,
        }

    @staticmethod
    def fill_result(
        result: PRISMResult,
        K_feat: float = 1.0,
        K_pred: float = 1.0,
    ) -> PRISMResult:
        """Populate the optional risk-bound fields of an existing PRISMResult."""
        bound = UnifiedBound.compute_bound(
            omega=result.omega,
            rho_T=result.rho_target,
            rho_P=result.rho_proxy,
            head_discrepancy=result.head_discrepancy,
            K_feat=K_feat,
            K_pred=K_pred,
        )
        result.risk_bound_feature = bound["risk_bound_feature"]
        result.risk_bound_head = bound["risk_bound_head"]
        result.risk_bound_total = bound["risk_bound_total"]
        return result

    # ------------------------------------------------------------------
    # Empirical Lipschitz estimation
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_lipschitz(
        features: Tensor,
        labels: Tensor,
        head_weights: Tensor,
        logit_scale: float = 1.0,
        logit_bias: float = 0.0,
        num_pairs: int = 1000,
        percentile: float = 95.0,
    ) -> Dict[str, float]:
        """Estimate K_feat and K_pred empirically via finite differences.

        Args:
            features:     (n, d) feature matrix.
            labels:       (n,) integer labels.
            head_weights: (C, d) or (d, C) classification head.
            logit_scale:  Scalar multiplier on logits.
            logit_bias:   Additive bias on logits.
            num_pairs:    Number of random pairs to sample.
            percentile:   Use this percentile of ratios as the estimate.

        Returns:
            Dict with keys ``K_feat``, ``K_pred``.
        """
        features = features.float()
        head_weights = head_weights.float()

        if head_weights.shape[0] != features.shape[1]:
            head_weights = head_weights.T  # ensure (d, C)

        n = features.shape[0]
        num_pairs = min(num_pairs, n * (n - 1) // 2)

        idx_a = torch.randint(0, n, (num_pairs,))
        idx_b = torch.randint(0, n, (num_pairs,))
        mask = idx_a != idx_b
        idx_a, idx_b = idx_a[mask], idx_b[mask]

        z_a, z_b = features[idx_a], features[idx_b]
        y_a, y_b = labels[idx_a], labels[idx_b]

        logits_a = z_a @ head_weights * logit_scale + logit_bias
        logits_b = z_b @ head_weights * logit_scale + logit_bias

        ce = torch.nn.functional.cross_entropy
        loss_a = torch.stack([ce(la.unsqueeze(0), ya.unsqueeze(0)) for la, ya in zip(logits_a, y_a)])
        loss_b = torch.stack([ce(lb.unsqueeze(0), yb.unsqueeze(0)) for lb, yb in zip(logits_b, y_b)])

        feat_dists = (z_a - z_b).norm(dim=1).clamp(min=1e-10)
        loss_diffs = (loss_a - loss_b).abs()

        ratios = loss_diffs / feat_dists
        K_feat = torch.quantile(ratios, percentile / 100.0).item()

        K_pred = logit_scale  # reasonable default for softmax head

        return {"K_feat": K_feat, "K_pred": K_pred}
