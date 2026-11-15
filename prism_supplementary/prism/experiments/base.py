"""
BaseExperiment; shared scaffolding for PRISM experiments.

Provides config bootstrap, the PRISM metric/bound computation, and JSON/CSV
result persistence.  Subclasses (e.g. ``QuantizationExperiment``) implement
their own ``run()`` pipeline on top.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from ..core.bounds import UnifiedBound
from ..core.metrics import PRISMMetrics, PRISMResult


class BaseExperiment:
    """Shared base: config + metric computation + result persistence."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        # "auto" is valid for HuggingFace device_map but not for PyTorch .to().
        # tensor_device is used wherever a concrete device string is required.
        self.tensor_device = "cuda" if self.device == "auto" else self.device
        self.seed = config.get("seed")
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.output_dir = config.get("output", {}).get("dir", "./results")
        os.makedirs(self.output_dir, exist_ok=True)

        cfg_computing = config.get("computing", {})
        self.model_dtype = getattr(torch, cfg_computing.get("model_dtype", "float16"))
        self.use_flash_attention = cfg_computing.get("use_flash_attention", False)

    # ------------------------------------------------------------------
    # Flash Attention 2 helper
    # ------------------------------------------------------------------
    def _attn_impl_kwargs(self) -> dict:
        """Return ``{"attn_implementation": "flash_attention_2"}`` when enabled.

        Checks that the ``flash-attn`` package is importable before adding the
        flag, so the experiment degrades gracefully on machines without it.
        Flash Attention 2 requires bf16 or fp16 weights; fp32 models are
        silently excluded.
        """
        if not self.use_flash_attention:
            return {}
        try:
            import flash_attn  # noqa: F401; availability probe only
        except ImportError:
            print("  [flash_attn] package not found — running without Flash Attention 2.")
            return {}
        if self.model_dtype == torch.float32:
            print("  [flash_attn] skipped — flash_attention_2 requires fp16/bf16, not fp32.")
            return {}
        return {"attn_implementation": "flash_attention_2"}

    # ------------------------------------------------------------------
    # Core metric computation
    # ------------------------------------------------------------------
    def compute_metrics(
        self,
        Z_T: Tensor,
        H_T: Tensor,
        Z_P: Tensor,
        H_P: Tensor,
        *,
        W: Optional[Tensor] = None,
        label: str = "",
        K_feat: float = 1.0,
        K_pred: float = 1.0,
    ) -> PRISMResult:
        """Run full PRISM metric suite and fill risk bound.

        Args:
            W:       Alignment matrix.  ``torch.eye(d)`` for identity regime,
                     ``None`` to compute the optimal Procrustes W_opt.
        """
        result = PRISMMetrics.compute_all(
            Z_T, H_T, Z_P, H_P,
            W=W,
            label=label,
        )
        UnifiedBound.fill_result(result, K_feat=K_feat, K_pred=K_pred)
        return result

    # ------------------------------------------------------------------
    # Result persistence
    # ------------------------------------------------------------------
    @staticmethod
    def _delta_risk(r: PRISMResult) -> Optional[float]:
        """Compute |R_T − R_P| if both losses are available."""
        if r.loss_target is not None and r.loss_proxy is not None:
            return abs(r.loss_target - r.loss_proxy)
        return None

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

    def save_csv(
        self,
        results: List[PRISMResult],
        filename: str = "prism_results.csv",
        loss_mode: str = "full",
    ) -> None:
        """Persist results as a flat CSV with dual-W metrics.

        Full-text loss columns (|dR|, Loss_T/P, PPL_T/P) are always
        included.  Answer-only columns (|AdR|, ALoss_T/P, APPL_T/P) are
        appended automatically when any result has answer loss data.

        Suffix _I: W = I; suffix _W: Procrustes alignment W_N (Omega_W is
        Omega_N, Bound_W is B_N).  Bound is the smaller of Bound_I and
        Bound_W.  delta_* and gamma_* are the two norms without K_f and K_p.
        |AdR| is the risk gap of the paper's tables; |dR| is the full-text gap.
        """
        path = os.path.join(self.output_dir, filename)

        geo_fields = ["target_model", "proxy_model", "dataset", "z_mode",
                       "Label", "rho_T", "rho_P", "Scale"]

        dual_w_fields = [
            "Omega_I", "Omega_W",
            "delta_I", "delta_W",
            "gamma_I", "gamma_W",
            "Bound_I", "Bound_W", "Bound",
        ]

        loss_fields = ["|dR|", "Loss_T", "Loss_P", "PPL_T", "PPL_P"]
        has_answer = any(r.extra.get("answer_loss_target") is not None for r in results)
        if has_answer:
            loss_fields += ["|AdR|", "ALoss_T", "ALoss_P", "APPL_T", "APPL_P"]

        k_fields = ["K_f", "K_p"]
        fieldnames = geo_fields + dual_w_fields + loss_fields + k_fields

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row: dict = {
                    "target_model": r.extra.get("target_model", ""),
                    "proxy_model": r.extra.get("proxy_model", ""),
                    "dataset": r.extra.get("dataset", ""),
                    "z_mode": r.extra.get("z_mode", ""),
                    "Label": r.label,
                    "rho_T": r.extra.get("rho_T", ""),
                    "rho_P": r.rho_proxy,
                    # Dual-W geometry
                    "Omega_I": r.extra.get("omega_I", r.omega),
                    "Omega_W": r.extra.get("omega_W", r.omega),
                    "delta_I": r.extra.get("delta_I", r.feature_error),
                    "delta_W": r.extra.get("delta_W", r.feature_error),
                    "gamma_I": r.extra.get("gamma_I", r.head_discrepancy),
                    "gamma_W": r.extra.get("gamma_W", r.head_discrepancy),
                    "Bound_I": r.extra.get("bound_I_th", ""),
                    "Bound_W": r.extra.get("bound_W_th", ""),
                    "Bound": r.extra.get("bound_th", r.risk_bound_total if r.risk_bound_total is not None else ""),
                    # Lipschitz constants
                    "K_f": r.extra.get("K_feat_tight", ""),
                    "K_p": r.extra.get("K_pred_theory", ""),
                    "Scale": r.scale_mismatch,
                }

                # Full-text loss (always present)
                flt = r.extra.get("full_loss_target", r.loss_target)
                flp = r.extra.get("full_loss_proxy", r.loss_proxy)
                dr = abs(flt - flp) if flt is not None and flp is not None else None
                row["|dR|"] = f"{dr:.6f}" if dr is not None else ""
                row["Loss_T"] = flt if flt is not None else ""
                row["Loss_P"] = flp if flp is not None else ""
                row["PPL_T"] = r.extra.get("full_ppl_target", r.extra.get("perplexity_target", ""))
                row["PPL_P"] = r.extra.get("full_ppl_proxy", r.extra.get("perplexity_proxy", ""))

                # Answer-only loss (Q&A datasets)
                if has_answer:
                    alt = r.extra.get("answer_loss_target")
                    alp = r.extra.get("answer_loss_proxy")
                    adr = abs(alt - alp) if alt is not None and alp is not None else None
                    row["|AdR|"] = f"{adr:.6f}" if adr is not None else ""
                    row["ALoss_T"] = alt if alt is not None else ""
                    row["ALoss_P"] = alp if alp is not None else ""
                    row["APPL_T"] = r.extra.get("answer_ppl_target", "")
                    row["APPL_P"] = r.extra.get("answer_ppl_proxy", "")

                writer.writerow(row)

        print(f"CSV saved to {path}  [{loss_mode}]")
