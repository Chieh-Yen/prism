"""
PRISM Core Metrics — Procrustes Similarity, Scale/Shape/Head Discrepancy.

All functions operate on raw PyTorch tensors with no model-loading logic,
making this module testable and reusable across experiment types.
"""

from __future__ import annotations

import math
import os
import warnings
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

    # Reductions used for the Frobenius norms: each row is reduced in the dtype
    # of the inputs and the n row sums are accumulated in float64.
    _UNIT_TOL = 1e-6

    @staticmethod
    def _metric_dtype_cast(*tensors: Tensor):
        """Cast the inputs of the metric path.

        float32 by default; ``PRISM_METRIC_DTYPE=float64`` selects double
        precision.
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
        """sum_i <A_i, B_i>: per-row dot in the input dtype, float64 accumulation over rows."""
        return (A * B).sum(dim=1).double().sum().item()

    @staticmethod
    def _fro(Z: Tensor) -> float:
        """Frobenius norm via the row-wise accumulation above."""
        return math.sqrt(PRISMMetrics._rowdot_sum(Z, Z))

    @staticmethod
    def _unit_clip(x: float) -> float:
        """Clip a normalised inner product to [-1, 1].

        Rounding can push the ratio marginally past 1.  The clip is applied in
        every case (the returned value is the same as a plain clamp); an
        overshoot larger than ``_UNIT_TOL`` additionally raises a warning, and
        callers keep the unclipped value in ``PRISMResult.extra["omega_raw"]``.
        """
        if x > 1.0:
            if x - 1.0 > PRISMMetrics._UNIT_TOL:
                warnings.warn(f"Omega = {x:.6f} exceeds 1 by more than "
                              f"{PRISMMetrics._UNIT_TOL}; check reductions")
            return 1.0
        if x < -1.0:
            if -1.0 - x > PRISMMetrics._UNIT_TOL:
                warnings.warn(f"Omega = {x:.6f} below -1; check reductions")
            return -1.0
        return x

    # ------------------------------------------------------------------
    # Procrustes similarity
    # ------------------------------------------------------------------
    @staticmethod
    def procrustes_omega(Z_T: Tensor, Z_P: Tensor) -> float:
        """Ω(Z_T, Z_P) = ‖Z_T^T Z_P‖_* / (‖Z_T‖_F ‖Z_P‖_F), clipped at 1 (it is ≥ 0).

        Corresponds to optimal Procrustes alignment W = W_opt.
        """
        cross = Z_T.T @ Z_P  # (d_T, d_P)
        nuclear_norm = torch.linalg.svdvals(cross).sum().item()
        denom = PRISMMetrics._fro(Z_T) * PRISMMetrics._fro(Z_P)
        if denom < 1e-12:
            return 0.0
        return PRISMMetrics._unit_clip(nuclear_norm / denom)

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
    # General Ω for any orthogonal W
    # ------------------------------------------------------------------
    @staticmethod
    def omega_for_W(Z_T: Tensor, Z_P: Tensor, W: Tensor) -> float:
        """Ω(W) = tr(W^T · Z_P^T Z_T) / (‖Z_T‖_F ‖Z_P‖_F).

        Unified formula that subsumes the W = W_opt and W = I special cases.
        Uses O(d²) element-wise product instead of O(d³) matrix multiply.
        """
        cross = Z_P.T @ Z_T                         # (d_P, d_T)
        denom = PRISMMetrics._fro(Z_T) * PRISMMetrics._fro(Z_P)
        if denom < 1e-12:
            return 0.0
        omega = (W * cross).sum().item() / denom
        return PRISMMetrics._unit_clip(omega)

    # ------------------------------------------------------------------
    # Convenience: compute everything at once
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_W(
        Z_T: Tensor, Z_P: Tensor, W: Optional[Tensor],
    ) -> Tensor:
        """Return W for alignment. If None, solve Orthogonal Procrustes."""
        if W is not None:
            return W.to(dtype=Z_T.dtype, device=Z_T.device)
        return PRISMMetrics.orthogonal_procrustes(Z_T, Z_P)

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

        # Cross-product — reused for Procrustes solve and Ω computation.
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

    # ------------------------------------------------------------------
    # Scale-absorbed variant
    # ------------------------------------------------------------------
    @staticmethod
    def compute_all_absorbed(
        Z_T: Tensor,
        H_T: Tensor,
        Z_P: Tensor,
        H_P: Tensor,
        *,
        W: Optional[Tensor] = None,
        label: str = "",
    ) -> PRISMResult:
        """Scale-absorbed PRISM analysis.

        Rescales only the proxy:  Z_P' = c Z_P,  H_P' = H_P / c  with
        c = ρ_T / ρ_P, forcing ρ_P' = ρ_T and (Δρ)² = 0.  The target
        is left untouched, so K_feat is the same as in the original mode.

        The scale mismatch is perfectly conserved and transferred into the
        head discrepancy:  ‖Σ_P^{1/2}(c W H_T − H_P)‖_F.

        Feature error becomes  ρ_T √(2(1−Ω)).
        Predictions are unchanged:  Z_P' H_P' = Z_P H_P.
        """
        Z_T, Z_P, H_T, H_P = PRISMMetrics._metric_dtype_cast(Z_T, Z_P, H_T, H_P)

        rho_T = PRISMMetrics.rms_scale(Z_T)
        rho_P = PRISMMetrics.rms_scale(Z_P)
        c = rho_T / max(rho_P, 1e-12)

        cross = Z_P.T @ Z_T  # (d_P, d_T)

        if W is None:
            U, _, Vt = torch.linalg.svd(cross, full_matrices=False)
            W_use = (U @ Vt).to(dtype=Z_T.dtype)
        else:
            W_use = W.to(dtype=Z_T.dtype, device=Z_T.device)

        denom = PRISMMetrics._fro(Z_T) * PRISMMetrics._fro(Z_P)
        omega = (W_use * cross).sum().item() / max(denom, 1e-12)
        omega_raw = omega
        omega = PRISMMetrics._unit_clip(omega)

        n = Z_P.shape[0]
        Sigma_P = (Z_P.T @ Z_P) / n

        shape = 2.0 * rho_T * rho_T * (1.0 - omega)
        feat_err = math.sqrt(max(shape, 0.0))

        hd_cov = PRISMMetrics.head_discrepancy_covariance(
            c * H_T, H_P, W_use, Sigma_P,
        )
        hd_spec = PRISMMetrics.head_discrepancy_spectral(
            c * H_T, H_P, W_use,
        )

        return PRISMResult(
            omega=omega,
            rho_target=rho_T,
            rho_proxy=rho_T,
            scale_mismatch=0.0,
            shape_mismatch=shape,
            feature_error=feat_err,
            head_discrepancy=hd_cov,
            head_discrepancy_spectral=hd_spec,
            label=label,
            extra={
                "omega_raw": omega_raw,
                "mode": "scale_absorbed",
                "rho_target_original": rho_T,
                "rho_proxy_original": rho_P,
                "c": c,
            },
        )
