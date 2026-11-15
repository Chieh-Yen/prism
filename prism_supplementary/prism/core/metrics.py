"""
PRISM Core Metrics; Procrustes Similarity, Scale/Shape/Head Discrepancy.

All functions operate on raw PyTorch tensors with no model-loading logic,
making this module testable and reusable across experiment types.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


@dataclass
class PRISMResult:
    """Standardised output of a single PRISM comparison."""

    omega: float                    # Ω(W) ∈ [-1, 1]; ≥ 0 for W = W_opt
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

    @staticmethod
    def _metric_dtype_cast(*tensors: Tensor):
        """Cast the inputs of the metric path.

        dtype from ``PRISM_METRIC_DTYPE`` (float32 by default, or float64).
        """
        name = os.environ.get("PRISM_METRIC_DTYPE", "float32").strip().lower()
        if name in ("float32", "float", "single"):
            return tuple(t.float() for t in tensors)
        if name in ("float64", "double"):
            return tuple(t.double() for t in tensors)
        raise ValueError(
            f"PRISM_METRIC_DTYPE must be float64 or float32, got {name!r}")

    @staticmethod
    def _rowdot_sum(A: Tensor, B: Tensor) -> float:
        """sum_i <A_i, B_i>."""
        return (A * B).sum(dim=1).double().sum().item()

    @staticmethod
    def _fro(Z: Tensor) -> float:
        """Frobenius norm, sqrt(sum_i <Z_i, Z_i>)."""
        return math.sqrt(PRISMMetrics._rowdot_sum(Z, Z))

    @staticmethod
    def _unit_clip(x: float) -> float:
        """Restrict Omega to [-1, 1] (Eq. 3)."""
        if x > 1.0:
            return 1.0
        if x < -1.0:
            return -1.0
        return x

    # ------------------------------------------------------------------
    # RMS scale  ρ = (1/√n) ‖Z‖_F
    # ------------------------------------------------------------------
    @staticmethod
    def rms_scale(Z: Tensor) -> float:
        n = Z.shape[0]
        return PRISMMetrics._fro(Z) / math.sqrt(n)

    # ------------------------------------------------------------------
    # Scale and shape terms  (Proposition 1)
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
    # Head term of the bound  (covariance-adjusted)
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
        label: str = "",
    ) -> PRISMResult:
        """Run full PRISM analysis on a (target, proxy) pair.

        Args:
            Z_T: (n, d_T) target features.
            H_T: (d_T, C) target head weights.
            Z_P: (n, d_P) proxy features.
            H_P: (d_P, C) proxy head weights.
            W:   Alignment matrix.  Pass ``torch.eye(d)`` for the
                 identity regime (quantization) or ``None`` to compute
                 the optimal Procrustes alignment W_opt.
            label: Human-readable label for this comparison.
        """
        Z_T, Z_P, H_T, H_P = PRISMMetrics._metric_dtype_cast(Z_T, Z_P, H_T, H_P)

        rho_T = PRISMMetrics.rms_scale(Z_T)
        rho_P = PRISMMetrics.rms_scale(Z_P)

        # Cross-product; reused for Procrustes solve and Ω computation.
        cross = Z_P.T @ Z_T  # (d_P, d_T)

        if W is None:
            U, _, Vt = torch.linalg.svd(cross, full_matrices=False)
            W_use = (U @ Vt).to(dtype=Z_T.dtype)
        else:
            W_use = W.to(dtype=Z_T.dtype, device=Z_T.device)

        # Ω(W) = tr(W^T · cross) / (‖Z_T‖_F ‖Z_P‖_F)
        denom = PRISMMetrics._fro(Z_T) * PRISMMetrics._fro(Z_P)
        omega = (W_use * cross).sum().item() / max(denom, 1e-12)
        omega_raw = omega
        omega = PRISMMetrics._unit_clip(omega)

        n = Z_P.shape[0]
        Sigma_P = (Z_P.T @ Z_P) / n

        # Identical heads under W = I: W H_T - H_P is the zero matrix, so both
        # head terms are exactly 0.0 (the general path returns the same 0.0
        # after an eigh, a (d x |V|) matmul and an SVD).  This is the frozen-head
        # LoRA case, evaluated at every checkpoint.
        _eye = torch.eye(W_use.shape[0], dtype=W_use.dtype, device=W_use.device)
        if (H_T.shape == H_P.shape and torch.equal(H_T, H_P)
                and W_use.shape[0] == W_use.shape[1] and torch.equal(W_use, _eye)):
            hd_cov, hd_spec = 0.0, 0.0
        else:
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
            extra={"omega_raw": omega_raw},
        )
