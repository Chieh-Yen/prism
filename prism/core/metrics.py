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


_AGG_CHUNK = 4096


def fro_norm_f32(Z: Tensor, chunk_size: int = _AGG_CHUNK) -> float:
    """Frobenius norm in float32, memory-efficient for float16 inputs."""
    if Z.dtype == torch.float32:
        return Z.norm("fro").item()
    total = 0.0
    for i in range(0, Z.shape[0], chunk_size):
        total += Z[i:i + chunk_size].float().pow(2).sum().item()
    return math.sqrt(total)


def _aggregate_features(
    Z_T: Tensor, Z_P: Tensor, *, chunk_size: int = _AGG_CHUNK,
) -> tuple:
    """Compute cross product, proxy covariance, and Frobenius norms.

    When inputs are float16, processes rows in chunks to avoid
    materialising full float32 copies (saves ~50% peak memory for
    large concat-mode feature matrices).

    Returns:
        (cross, Sigma_P, fro_T, fro_P) all in float32.
    """
    n = Z_T.shape[0]
    d_T, d_P = Z_T.shape[1], Z_P.shape[1]
    device = Z_T.device

    if Z_T.dtype == torch.float32 and Z_P.dtype == torch.float32:
        cross = Z_P.T @ Z_T
        Sigma_P = (Z_P.T @ Z_P) / n
        return cross, Sigma_P, Z_T.norm("fro").item(), Z_P.norm("fro").item()

    cross = torch.zeros(d_P, d_T, dtype=torch.float32, device=device)
    cov_P = torch.zeros(d_P, d_P, dtype=torch.float32, device=device)
    norm_T_sq = 0.0
    norm_P_sq = 0.0

    for i in range(0, n, chunk_size):
        ct = Z_T[i:i + chunk_size].float()
        cp = Z_P[i:i + chunk_size].float()
        cross += cp.T @ ct
        cov_P += cp.T @ cp
        norm_T_sq += ct.pow(2).sum().item()
        norm_P_sq += cp.pow(2).sum().item()

    return cross, cov_P / n, math.sqrt(norm_T_sq), math.sqrt(norm_P_sq)


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
        """Ω(Z_T, Z_P) = ‖Z_T^T Z_P‖_* / (‖Z_T‖_F ‖Z_P‖_F), clamped to [0, 1].

        Corresponds to optimal Procrustes alignment W = W_opt.
        """
        cross = Z_T.T @ Z_P  # (d_T, d_P)
        nuclear_norm = torch.linalg.svdvals(cross).sum().item()
        denom = Z_T.norm("fro").item() * Z_P.norm("fro").item()
        if denom < 1e-12:
            return 0.0
        return min(nuclear_norm / denom, 1.0)

    @staticmethod
    def trace_omega(Z_T: Tensor, Z_P: Tensor) -> float:
        """Ω_I(Z_T, Z_P) = tr(Z_T^T Z_P) / (‖Z_T‖_F ‖Z_P‖_F).

        Identity-consistent alignment score — corresponds to W = I.
        By Cauchy-Schwarz for the Frobenius inner product, |Ω_I| ≤ 1.
        Always ≤ procrustes_omega in absolute value.
        """
        trace_val = (Z_T * Z_P).sum().item()
        denom = Z_T.norm("fro").item() * Z_P.norm("fro").item()
        if denom < 1e-12:
            return 0.0
        return max(min(trace_val / denom, 1.0), -1.0)

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
    # General Ω for any orthogonal W
    # ------------------------------------------------------------------
    @staticmethod
    def omega_for_W(Z_T: Tensor, Z_P: Tensor, W: Tensor) -> float:
        """Ω(W) = tr(W · Z_P^T Z_T) / (‖Z_T‖_F ‖Z_P‖_F).

        Unified formula that subsumes ``procrustes_omega`` (W = W_opt)
        and ``trace_omega`` (W = I) as special cases.
        Uses O(d²) element-wise product instead of O(d³) matrix multiply.
        """
        cross = Z_P.T @ Z_T                         # (d_P, d_T)
        denom = Z_T.norm("fro").item() * Z_P.norm("fro").item()
        if denom < 1e-12:
            return 0.0
        omega = (W * cross).sum().item() / denom
        return max(min(omega, 1.0), -1.0)

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
        H_T = H_T.float()
        H_P = H_P.float()

        # Chunked float32 aggregation — avoids materialising full float32
        # copies of Z_T / Z_P when they are stored in float16.
        cross, Sigma_P, fro_T, fro_P = _aggregate_features(Z_T, Z_P)
        n = Z_T.shape[0]
        rho_T = fro_T / math.sqrt(n)
        rho_P = fro_P / math.sqrt(n)

        if W is None:
            U, _, Vt = torch.linalg.svd(cross, full_matrices=False)
            W_use = U @ Vt
        else:
            W_use = W.float()

        # Ω(W) = tr(W · cross) / (‖Z_T‖_F ‖Z_P‖_F)
        denom = fro_T * fro_P
        omega = (W_use * cross).sum().item() / max(denom, 1e-12)
        omega = max(min(omega, 1.0), -1.0)

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

    # ------------------------------------------------------------------
    # Scale-absorbed variant  (Corollary: Invariance under Scalar Norm.)
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
        H_T = H_T.float()
        H_P = H_P.float()

        cross, Sigma_P, fro_T, fro_P = _aggregate_features(Z_T, Z_P)
        n = Z_T.shape[0]
        rho_T = fro_T / math.sqrt(n)
        rho_P = fro_P / math.sqrt(n)
        c = rho_T / max(rho_P, 1e-12)

        if W is None:
            U, _, Vt = torch.linalg.svd(cross, full_matrices=False)
            W_use = U @ Vt
        else:
            W_use = W.float()

        denom = fro_T * fro_P
        omega = (W_use * cross).sum().item() / max(denom, 1e-12)
        omega = max(min(omega, 1.0), -1.0)

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
                "mode": "scale_absorbed",
                "rho_target_original": rho_T,
                "rho_proxy_original": rho_P,
                "c": c,
            },
        )
