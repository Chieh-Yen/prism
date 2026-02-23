"""
Unified Risk Discrepancy Bound (Theorem 2) and Lipschitz estimation.

Tight Lipschitz constants (Appendix A)
=======================================
K_feat  --  via Simplex Polarization (Lemma 1):
    The gradient of CE loss w.r.t. features z decomposes as a convex
    combination of pairwise class-prototype differences:

        nabla_z g_T = sum_{i!=k}  p_hat_i (h_i - h_k)

    where h_i are columns of H_T.  Therefore:

        K_feat = max_{j,k} ||h_j - h_k||_2

    This is at most equal to, and typically much tighter than, the naive
    sqrt(2) * ||H_T||_op bound.  Proof: ||h_j - h_k|| = ||H(e_j - e_k)||
    <= sigma_1(H) * sqrt(2).  Equality is rare; the tight bound depends
    on relative prototype distances, not absolute magnitude.

K_pred  --  from logit-space gradient (Proposition 1):
    nabla_v ell = p_hat - y   =>   K_pred <= sqrt(2).

For the scale-absorbed variant H_bar_T = rho_T * H_T, columns scale by
rho_T and K_feat^{abs} = rho_T * K_feat^{orig}.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

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

    For d=4096, C=32000, CHUNK=2048 the exact path needs ~136 matmuls
    of shape (2048, 2048) and takes roughly 15-40 s on a modern GPU
    (one-time cost per experiment).
    """
    H = H.float()
    C = H.shape[1]
    if C <= 1:
        return 0.0

    col_norms_sq = (H * H).sum(dim=0)  # (C,)

    if exact:
        return _diameter_exact(H, col_norms_sq)
    return _diameter_heuristic(H, col_norms_sq)


def _diameter_exact(H: Tensor, col_norms_sq: Tensor) -> float:
    """Exact column-set diameter via chunked ||h_j-h_k||^2 = ||h_j||^2 + ||h_k||^2 - 2 h_j·h_k."""
    C = H.shape[1]
    CHUNK = 2048
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
        *,
        absorbed: bool = False,
        rho_T: float = 1.0,
    ) -> Dict[str, float]:
        """Compute tight Lipschitz constants for cross-entropy.

        K_feat uses the simplex-polarization bound (Lemma 1):
            K_feat = max_{j,k} ||h_j - h_k||_2

        K_pred uses the logit-gradient bound (Proposition 1):
            K_pred = sqrt(2)

        Args:
            H_T: (d, C) target head weights (original, not absorbed).
            absorbed: Whether scale-absorbed mode is used.
            rho_T: Original rho_T (needed only when absorbed=True).

        Returns:
            Dict with K_feat, K_pred, K_feat_naive, max_pw_dist,
            H_T_spectral, L_loss.
        """
        H_T = H_T.float()
        H_T_spec = torch.linalg.svdvals(H_T)[0].item()
        L = CROSS_ENTROPY_LIPSCHITZ

        max_pw = _max_pairwise_column_distance(H_T)

        K_feat_naive = L * H_T_spec
        K_feat_tight = max_pw

        if absorbed:
            K_feat_tight *= rho_T
            K_feat_naive *= rho_T

        return {
            "K_feat": K_feat_tight,
            "K_feat_naive": K_feat_naive,
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

    # ------------------------------------------------------------------
    # Empirical Lipschitz estimation -- LLM  (per-sample losses)
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_lipschitz_lm(
        features: Tensor,
        per_sample_losses: Tensor,
        *,
        rho_T: Optional[float] = None,
        num_pairs: int = 2000,
        percentile: float = 95.0,
    ) -> Dict[str, float]:
        """Estimate K_feat empirically from target-model features and losses.

        Samples random pairs (i, j) from the *same model* and computes::

            K_hat = percentile_p( |l_i - l_j| / ||z_i - z_j||_2 )

        When ``rho_T`` is provided (scale-absorbed mode), features are
        normalised to Z_bar = Z / rho_T first so the ratio lives in the same
        coordinate system as eps_bar_feat = sqrt(2(1-Omega)).

        Args:
            features:          (n, d) feature matrix (CPU, float).
            per_sample_losses: (n,) per-sample losses (CPU, float).
            rho_T:             If given, divide features by this value
                               (scale-absorbed mode).
            num_pairs:         Number of random pairs.
            percentile:        Percentile of the ratio distribution to use.

        Returns:
            Dict with ``K_feat_empirical``, ``median``, ``p95``, ``max``.
        """
        features = features.float()
        if rho_T is not None:
            features = features / max(rho_T, 1e-12)
        per_sample_losses = per_sample_losses.float()
        n = features.shape[0]
        num_pairs = min(num_pairs, n * (n - 1) // 2)

        idx_a = torch.randint(0, n, (num_pairs,))
        idx_b = torch.randint(0, n, (num_pairs,))
        mask = idx_a != idx_b
        idx_a, idx_b = idx_a[mask], idx_b[mask]

        feat_dists = (features[idx_a] - features[idx_b]).norm(dim=1).clamp(min=1e-10)
        loss_diffs = (per_sample_losses[idx_a] - per_sample_losses[idx_b]).abs()

        ratios = loss_diffs / feat_dists
        p50 = torch.quantile(ratios, 0.50).item()
        p95 = torch.quantile(ratios, percentile / 100.0).item()
        p_max = ratios.max().item()

        return {
            "K_feat_empirical": p95,
            "median": p50,
            "p95": p95,
            "max": p_max,
        }

    # ------------------------------------------------------------------
    # Empirical Lipschitz estimation -- CLIP  (classification logits)
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_lipschitz_clip(
        features: Tensor,
        labels: Tensor,
        head_weights: Tensor,
        logit_scale: float = 1.0,
        logit_bias: float = 0.0,
        num_pairs: int = 1000,
        percentile: float = 95.0,
    ) -> Dict[str, float]:
        """Estimate K_feat empirically for CLIP-style zero-shot classification."""
        features = features.float()
        head_weights = head_weights.float()

        if head_weights.shape[0] != features.shape[1]:
            head_weights = head_weights.T

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
        K_pred = logit_scale

        return {"K_feat": K_feat, "K_pred": K_pred}
