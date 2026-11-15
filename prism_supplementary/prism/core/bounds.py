"""
Unified risk bound (Theorem 1) and its Lipschitz constants.

Tight Lipschitz constants (Appendix A)
=======================================
K_feat  --  via simplex polarization:
    The gradient of CE loss w.r.t. features z decomposes as a non-negative
    linear combination of pairwise class-prototype differences:

        nabla_z g_T = sum_{i!=k}  p_hat_i (h_i - h_k)

    where h_i are columns of H_T.  Therefore:

        K_feat = max_{j,k} ||h_j - h_k||_2

    This is at most equal to, and typically much tighter than, the naive
    sqrt(2) * ||H_T||_op bound.  Proof: ||h_j - h_k|| = ||H(e_j - e_k)||
    <= sigma_1(H) * sqrt(2).  Equality is rare; the tight bound depends
    on relative prototype distances, not absolute magnitude.

K_pred  --  from the logit-space gradient:
    nabla_v ell = p_hat - y   =>   K_pred <= sqrt(2).

The scale-absorbed variant (PRISMMetrics.compute_all_absorbed) rescales only
the proxy; H_T, and therefore K_feat, is the same as in the original mode.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import Tensor

from .metrics import PRISMMetrics, PRISMResult

CROSS_ENTROPY_LIPSCHITZ: float = math.sqrt(2.0)


def _max_pairwise_column_distance(H: Tensor, *, exact: bool = True) -> float:
    """Compute  max_{j,k} ||h_j - h_k||_2  for columns h_j of H in R^{d x C}.

    Args:
        H: (d, C) matrix whose columns are class prototypes.
        exact: If True (default), compute the exact diameter via a chunked
            Gram-matrix sweep in O(C^2 d) time and O(CHUNK^2) memory.
            If False, use a fast farthest-point heuristic in
            O(restarts * iters * C * d) — not guaranteed exact but
            very reliable in practice.

    The exact path (``_diameter_exact``) visits every pair of column chunks
    with CHUNK = 4096 columns per chunk, i.e. ceil(C / 4096) *
    (ceil(C / 4096) + 1) / 2 Gram blocks; it is computed once per target head.
    """
    H = H.float()
    C = H.shape[1]
    if C <= 1:
        return 0.0

    col_norms_sq = (H * H).sum(dim=0)  # (C,)

    if exact:
        return _diameter_exact(H, col_norms_sq)
    return _diameter_heuristic(H, col_norms_sq)


def _spectral_norm_power_iter(H: Tensor, n_iter: int = 32) -> float:
    """Largest singular value via power iteration.

    O(d * C * n_iter) vs O(d² * C) for full SVD — ~100× faster for
    large-vocab heads (Qwen3 151K, Gemma 256K).
    """
    v = torch.randn(H.shape[1], dtype=H.dtype, device=H.device)
    v = v / v.norm()
    for _ in range(n_iter):
        u = H @ v;  u = u / u.norm()   # (d,)
        v = H.T @ u; v = v / v.norm()  # (C,)
    return (H @ v).norm().item()


def _diameter_exact(H: Tensor, col_norms_sq: Tensor) -> float:
    """Exact column-set diameter via chunked ||h_j-h_k||^2 = ||h_j||^2 + ||h_k||^2 - 2 h_j·h_k."""
    C = H.shape[1]
    CHUNK = 4096
    best_sq = 0.0

    for i in range(0, C, CHUNK):
        Hi = H[:, i : i + CHUNK]
        ni = col_norms_sq[i : i + CHUNK]
        for j in range(i, C, CHUNK):
            Hj = H[:, j : j + CHUNK]
            nj = col_norms_sq[j : j + CHUNK]

            gram = Hi.T @ Hj                                       # (ci, cj)
            dsq = ni.unsqueeze(1) + nj.unsqueeze(0) - 2.0 * gram
            if i == j:
                dsq.fill_diagonal_(0.0)
            dsq.clamp_(min=0.0)

            cand = dsq.max().item()
            if cand > best_sq:
                best_sq = cand

    return math.sqrt(best_sq)


def _diameter_heuristic(H: Tensor, col_norms_sq: Tensor) -> float:
    """Fast approximate diameter via farthest-point iteration (4 restarts × 4 iters)."""
    C = H.shape[1]
    best = 0.0
    starts = [0, C // 3, 2 * C // 3, C - 1]
    for start in starts:
        cur = start
        for _ in range(4):
            h_cur = H[:, cur]
            dots = H.T @ h_cur
            dists_sq = col_norms_sq + col_norms_sq[cur] - 2.0 * dots
            dists_sq.clamp_(min=0.0)
            nxt = dists_sq.argmax().item()
            d = math.sqrt(dists_sq[nxt].item())
            if d > best:
                best = d
            cur = nxt

    return best


class UnifiedBound:
    """Compute the PRISM Unified Risk Discrepancy Bound.

    |R_T - R_P| <= K_feat * delta  +  K_pred * gamma

    where
        delta = sqrt[ (rho_T - rho_P)^2 + 2 rho_T rho_P (1 - Omega) ]
        gamma = ||Sigma_P^{1/2}(W H_T - H_P)||_F
    """

    # ------------------------------------------------------------------
    # Theoretical K  (tight, from Appendix A)
    # ------------------------------------------------------------------
    @staticmethod
    def theoretical_K(
        H_T: Tensor,
    ) -> Dict[str, float]:
        """Compute tight Lipschitz constants for cross-entropy.

        K_feat uses the simplex-polarization bound:
            K_feat = max_{j,k} ||h_j - h_k||_2

        K_pred uses the logit-gradient bound:
            K_pred = sqrt(2)

        These constants are properties of the target head and do not depend
        on the scale-absorbed reparameterisation, which leaves H_T untouched.

        Args:
            H_T: (d, C) target head weights.

        Returns:
            Dict with K_feat, K_pred, K_feat_naive, max_pairwise_dist,
            H_T_spectral, L_loss.
        """
        H_T = H_T.float()
        H_T_spec = _spectral_norm_power_iter(H_T)
        L = CROSS_ENTROPY_LIPSCHITZ

        max_pw = _max_pairwise_column_distance(H_T)

        return {
            "K_feat": max_pw,
            "K_feat_naive": L * H_T_spec,
            "K_pred": L,
            "max_pairwise_dist": max_pw,
            "H_T_spectral": H_T_spec,
            "L_loss": L,
        }

    # ------------------------------------------------------------------
    # Bound computation
    # ------------------------------------------------------------------
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
        """Populate the optional risk-bound fields of an existing PRISMResult.

        Works for both original and scale-absorbed modes: uses the
        ``feature_error`` already stored in the result rather than
        recomputing it, so the absorbed sqrt(2(1-Omega)) value is respected.
        """
        rb_feat = K_feat * result.feature_error
        rb_head = K_pred * result.head_discrepancy
        result.risk_bound_feature = rb_feat
        result.risk_bound_head = rb_head
        result.risk_bound_total = rb_feat + rb_head
        return result
