"""
PRISM Core Metrics — Procrustes Similarity, Scale/Shape/Head Discrepancy.

All functions operate on raw PyTorch tensors with no model-loading logic,
making this module testable and reusable across experiment types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


@dataclass
class PRISMResult:
    """Standardised output of a single PRISM comparison."""

    omega: float                    # Procrustes Similarity Ω ∈ [0, 1]
    rho_target: float               # RMS scale of target features
    rho_proxy: float                # RMS scale of proxy features
    scale_mismatch: float           # (ρ_T − ρ_P)²
    shape_mismatch: float           # 2 ρ_T ρ_P (1 − Ω)
    feature_error: float            # √(scale + shape)
    head_discrepancy: float         # ‖Σ_P^{1/2}(W H_T − H_P)‖_F
    head_discrepancy_spectral: float  # ‖W H_T − H_P‖_op  (spectral norm)

    # Optional risk bound (filled when K_feat / K_pred are known)
    risk_bound_feature: Optional[float] = None
    risk_bound_head: Optional[float] = None
    risk_bound_total: Optional[float] = None

    # Optional evaluation losses
    loss_target: Optional[float] = None
    loss_proxy: Optional[float] = None
    label: str = ""                 # Free-form label, e.g. "INT4" or "step-500"

    extra: dict = field(default_factory=dict)


class PRISMMetrics:
    """Pure-function collection of all PRISM geometric metrics."""

    # ------------------------------------------------------------------
    # Procrustes Similarity  (Eq. 3)
    # ------------------------------------------------------------------
    @staticmethod
    def procrustes_omega(Z_T: Tensor, Z_P: Tensor) -> float:
        """Ω(Z_T, Z_P) = ‖Z_T^T Z_P‖_* / (‖Z_T‖_F ‖Z_P‖_F)."""
        cross = Z_T.T @ Z_P  # (d_T, d_P)
        nuclear_norm = torch.linalg.svdvals(cross).sum().item()
        denom = Z_T.norm("fro").item() * Z_P.norm("fro").item()
        if denom < 1e-12:
            return 0.0
        return nuclear_norm / denom

    # ------------------------------------------------------------------
    # Optimal alignment W  (Orthogonal Procrustes)
    # ------------------------------------------------------------------
    @staticmethod
    def orthogonal_procrustes(Z_T: Tensor, Z_P: Tensor) -> Tensor:
        """Solve  min_{W: W W^T = I}  ‖Z_T − Z_P W‖_F  via SVD.

        Returns W ∈ R^{d_P × d_T}  (semi-orthogonal).
        """
        cross = Z_P.T @ Z_T  # (d_P, d_T)
        U, _, Vt = torch.linalg.svd(cross, full_matrices=False)
        W = U @ Vt  # (d_P, d_T)
        return W

    # ------------------------------------------------------------------
    # RMS scale  ρ = (1/√n) ‖Z‖_F
    # ------------------------------------------------------------------
    @staticmethod
    def rms_scale(Z: Tensor) -> float:
        n = Z.shape[0]
        return Z.norm("fro").item() / math.sqrt(n)

    # ------------------------------------------------------------------
    # Decomposed mismatch terms  (Theorem 1)
    # ------------------------------------------------------------------
    @staticmethod
    def scale_mismatch(rho_T: float, rho_P: float) -> float:
        return (rho_T - rho_P) ** 2

    @staticmethod
    def shape_mismatch(rho_T: float, rho_P: float, omega: float) -> float:
        return 2.0 * rho_T * rho_P * (1.0 - omega)

    @staticmethod
    def feature_error(rho_T: float, rho_P: float, omega: float) -> float:
        sm = PRISMMetrics.scale_mismatch(rho_T, rho_P)
        shm = PRISMMetrics.shape_mismatch(rho_T, rho_P, omega)
        return math.sqrt(max(sm + shm, 0.0))

    # ------------------------------------------------------------------
    # Head discrepancy  (Proposition 1 — covariance-adjusted)
    # ------------------------------------------------------------------
    @staticmethod
    def head_discrepancy_covariance(
        H_T: Tensor,
        H_P: Tensor,
        W: Tensor,
        Sigma_P: Tensor,
    ) -> float:
        """‖Σ_P^{1/2} (W H_T − H_P)‖_F

        Args:
            H_T: (d_T, C) target head weights.
            H_P: (d_P, C) proxy head weights.
            W:   (d_P, d_T) alignment map.
            Sigma_P: (d_P, d_P) uncentered covariance  (1/n) Z_P^T Z_P.
        """
        delta_H = W @ H_T - H_P  # (d_P, C)

        evals, evecs = torch.linalg.eigh(Sigma_P)
        evals = evals.clamp(min=0.0)
        sqrt_sigma = evecs @ torch.diag(evals.sqrt()) @ evecs.T

        projected = sqrt_sigma @ delta_H  # (d_P, C)
        return projected.norm("fro").item()

    # ------------------------------------------------------------------
    # Head discrepancy  (spectral-norm fallback)
    # ------------------------------------------------------------------
    @staticmethod
    def head_discrepancy_spectral(
        H_T: Tensor,
        H_P: Tensor,
        W: Tensor,
    ) -> float:
        """‖W H_T − H_P‖_op   (largest singular value)."""
        delta_H = W @ H_T - H_P
        return torch.linalg.svdvals(delta_H)[0].item()

    # ------------------------------------------------------------------
    # Per-sample geometric consistency score  (for OOD, Sec 5.3)
    # ------------------------------------------------------------------
    @staticmethod
    def consistency_scores(
        Z_T: Tensor,
        Z_P: Tensor,
        W: Tensor,
    ) -> Tensor:
        """s(x_i) = ‖z_T^i − z_P^i W‖_2   per sample.

        Returns: (n,) tensor of scores.
        """
        residuals = Z_T - Z_P @ W  # (n, d_T)
        return residuals.norm(dim=1)

    # ------------------------------------------------------------------
    # Convenience: compute everything at once
    # ------------------------------------------------------------------
    @staticmethod
    def compute_all(
        Z_T: Tensor,
        H_T: Tensor,
        Z_P: Tensor,
        H_P: Tensor,
        *,
        W: Optional[Tensor] = None,
        force_identity: bool = False,
        label: str = "",
    ) -> PRISMResult:
        """Run full PRISM analysis on a (target, proxy) pair.

        Args:
            Z_T: (n, d_T) target features.
            H_T: (d_T, C) target head weights.
            Z_P: (n, d_P) proxy features.
            H_P: (d_P, C) proxy head weights.
            W:   Optional pre-computed alignment.  Computed if None.
            force_identity: If True, use W = I  (quantization regime).
            label: Human-readable label for this comparison.
        """
        Z_T = Z_T.float()
        Z_P = Z_P.float()
        H_T = H_T.float()
        H_P = H_P.float()

        omega = PRISMMetrics.procrustes_omega(Z_T, Z_P)
        rho_T = PRISMMetrics.rms_scale(Z_T)
        rho_P = PRISMMetrics.rms_scale(Z_P)

        if force_identity:
            assert Z_T.shape[1] == Z_P.shape[1], (
                f"force_identity requires d_T == d_P, got {Z_T.shape[1]} vs {Z_P.shape[1]}"
            )
            W_use = torch.eye(Z_P.shape[1], Z_T.shape[1], dtype=Z_T.dtype, device=Z_T.device)
        elif W is not None:
            W_use = W.float()
        else:
            W_use = PRISMMetrics.orthogonal_procrustes(Z_T, Z_P)

        n = Z_P.shape[0]
        Sigma_P = (Z_P.T @ Z_P) / n

        hd_cov = PRISMMetrics.head_discrepancy_covariance(H_T, H_P, W_use, Sigma_P)
        hd_spec = PRISMMetrics.head_discrepancy_spectral(H_T, H_P, W_use)

        return PRISMResult(
            omega=omega,
            rho_target=rho_T,
            rho_proxy=rho_P,
            scale_mismatch=PRISMMetrics.scale_mismatch(rho_T, rho_P),
            shape_mismatch=PRISMMetrics.shape_mismatch(rho_T, rho_P, omega),
            feature_error=PRISMMetrics.feature_error(rho_T, rho_P, omega),
            head_discrepancy=hd_cov,
            head_discrepancy_spectral=hd_spec,
            label=label,
        )
