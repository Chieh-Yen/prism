"""
Quantization Quality Estimation — Identity Regime (W = I).

Target = full-precision model (FP16).
Proxy  = quantized model at each bit width.
The alignment map W degenerates to the identity.

The simplified bound (Eq. 6) becomes:
    |R_F − R_Q| ≈ K_feat · √[ (ρ_F − ρ_Q)² + 2 ρ_F ρ_Q (1 − Ω) ]
"""

from __future__ import annotations

import gc
import math
import time
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..core.bounds import UnifiedBound
from ..data.loaders import load_task_data
from ..models.extractors import LLMExtractor
from .base import BaseExperiment


_LLAMA2_GGUF_MAP = {
    "Q8_0": "llama-2-7b.Q8_0.gguf",
    "Q6_K": "llama-2-7b.Q6_K.gguf",
    "Q5_K_M": "llama-2-7b.Q5_K_M.gguf",
    "Q4_K_M": "llama-2-7b.Q4_K_M.gguf",
    "Q3_K_M": "llama-2-7b.Q3_K_M.gguf",
    "Q2_K": "llama-2-7b.Q2_K.gguf",
}


class QuantizationExperiment(BaseExperiment):
    """Compare FP16 (target) with quantised variants (proxy), W = I."""

    def setup_pairs(self) -> List[Dict[str, Any]]:
        """Not used directly — see ``run()``."""
        return []

    def _load_proxy(self, quant_repo: str, filename: str, fallback_model_id: str) -> torch.nn.Module:
        """Load a single quantised proxy model."""
        # Ensure VRAM is clean before loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  Loading proxy: {filename} from {quant_repo} ...")
        try:
            proxy = AutoModelForCausalLM.from_pretrained(
                quant_repo,
                gguf_file=filename,
                dtype=torch.float16,
                device_map=self.device,
            )
        except Exception as e:
            print(f"    GGUF load failed ({e}), falling back to bitsandbytes ...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            from transformers import BitsAndBytesConfig
            is_4bit = any(tag in filename for tag in ("Q4", "Q3", "Q2"))
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=is_4bit,
                load_in_8bit=not is_4bit,
            )
            proxy = AutoModelForCausalLM.from_pretrained(
                fallback_model_id,
                quantization_config=bnb_config,
                device_map=self.device,
            )
        proxy.eval()
        return proxy

    def run(self) -> list:
        """Cache target features/loss, free VRAM, then load proxies one at a time."""
        cfg_target = self.config.get("target", {})
        cfg_proxy = self.config.get("proxy", {})
        cfg_data = self.config.get("data", {})
        cfg_align = self.config.get("alignment", {})

        target_model_id = cfg_target.get("model", "NousResearch/Llama-2-7b-hf")
        quant_repo = cfg_proxy.get("model", "TheBloke/Llama-2-7b-GGUF")
        quant_bits: List[str] = cfg_proxy.get("quantization_bits", list(_LLAMA2_GGUF_MAP.keys()))
        absorbed: bool = cfg_align.get("scale_absorbed", False)

        task_name = cfg_data.get("task", "wikitext")
        num_samples = cfg_data.get("num_samples", 256)
        batch_size = cfg_data.get("batch_size", 8)
        max_length = cfg_data.get("max_length", 512)

        mode_tag = " (scale-absorbed)" if absorbed else ""
        print(f"{'=' * 72}")
        print(f"  PRISM Experiment: Quantization Quality Estimation{mode_tag}")
        print(f"{'=' * 72}")

        print(f"Loading tokenizer from {target_model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(target_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading data: {task_name} (n={num_samples}) ...")
        dataloader = load_task_data(
            task_name, split="test", num_samples=num_samples,
            batch_size=batch_size, tokenizer=tokenizer, max_length=max_length,
        )

        # --- Phase 1: load target, extract everything, then free VRAM ---
        print(f"Loading target (FP16): {target_model_id} ...")
        target_model = AutoModelForCausalLM.from_pretrained(
            target_model_id, dtype=torch.float16, device_map=self.device,
        )
        target_model.eval()
        extractor = LLMExtractor()

        print("Caching target features and loss (will free target before loading proxies) ...")
        Z_T = extractor.extract_features(target_model, dataloader, self.device)
        H_T = extractor.extract_head(target_model)
        loss_stats = self.compute_lm_loss_per_sample(target_model, dataloader, self.device)
        losses_T = loss_stats["losses"]
        loss_target = losses_T.mean().item()
        ppl_target = math.exp(loss_target)
        print(f"  Target: Loss={loss_target:.4f}  PPL={ppl_target:.2f}  "
              f"Z={tuple(Z_T.shape)}  H={tuple(H_T.shape)}")

        del target_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  Target model freed from VRAM.")

        # --- Lipschitz constants (invariant under Corollary reparameterisation) ---
        rho_T_orig = Z_T.norm("fro").item() / math.sqrt(Z_T.shape[0])
        K_theory = UnifiedBound.theoretical_K(H_T)
        K_empirical_feat = UnifiedBound.estimate_lipschitz_lm(Z_T, losses_T)
        K_feat = K_theory["K_feat"]
        K_pred = K_theory["K_pred"]
        K_pred_empirical = loss_stats["grad_norm_p95"]

        print(f"\n  ρ_T = {rho_T_orig:.6f}")
        print(f"\n  Lipschitz constants (invariant across original / absorbed):")
        print(f"    ‖H_T‖_op = {K_theory['H_T_spectral']:.4f}")
        print(f"    K_feat (tight)     = max_jk ‖h_j−h_k‖       = {K_feat:.4f}")
        print(f"    K_feat (naive)     = √2·‖H_T‖_op            = {K_theory['K_feat_naive']:.4f}")
        print(f"    K_feat (empirical) = p95(|Δℓ|/‖Δz‖)         = {K_empirical_feat['p95']:.4f}  "
              f"(median={K_empirical_feat['median']:.4f}, max={K_empirical_feat['max']:.4f})")
        print(f"    K_pred (theory)    = √2                      = {K_pred:.4f}")
        print(f"    K_pred (empirical) = p95(‖p̂−e_y‖)           = {K_pred_empirical:.4f}  "
              f"(mean={loss_stats['grad_norm_mean']:.4f}, max={loss_stats['grad_norm_max']:.4f})")

        # --- Lipschitz summary dict (shared across all pairs) ---
        lipschitz_info = {
            "K_feat_tight": K_feat,
            "K_feat_naive": K_theory["K_feat_naive"],
            "K_feat_empirical": K_empirical_feat["K_feat_empirical"],
            "K_feat_empirical_median": K_empirical_feat["median"],
            "K_feat_empirical_max": K_empirical_feat["max"],
            "K_pred_theory": K_pred,
            "K_pred_empirical": K_pred_empirical,
            "K_pred_empirical_mean": loss_stats["grad_norm_mean"],
            "K_pred_empirical_max": loss_stats["grad_norm_max"],
            "H_T_spectral": K_theory["H_T_spectral"],
            "max_pairwise_dist": K_theory["max_pairwise_dist"],
            "rho_T": rho_T_orig,
        }

        # --- Phase 2: load each proxy one at a time ---
        results = []
        for i, bit_label in enumerate(quant_bits):
            filename = _LLAMA2_GGUF_MAP.get(bit_label, bit_label)
            label = f"FP16 vs {bit_label}"
            print(f"\n--- [{i+1}/{len(quant_bits)}] {label} ---")
            t0 = time.time()

            proxy_model = self._load_proxy(quant_repo, filename, target_model_id)

            Z_P = extractor.extract_features(proxy_model, dataloader, self.device)
            H_P = extractor.extract_head(proxy_model)

            result = self.compute_metrics(
                Z_T, H_T, Z_P, H_P,
                force_identity=True, label=label, absorbed=absorbed,
                K_feat=K_feat, K_pred=K_pred,
            )

            result.loss_target = loss_target
            result.loss_proxy = self.compute_lm_loss(proxy_model, dataloader, self.device)
            result.extra["perplexity_target"] = ppl_target
            result.extra["perplexity_proxy"] = math.exp(result.loss_proxy)
            result.extra.update(lipschitz_info)
            result.extra["dataset"] = task_name
            result.extra["num_samples"] = num_samples

            results.append(result)

            dr = abs(result.loss_target - result.loss_proxy)
            ppl_p = result.extra["perplexity_proxy"]
            elapsed = time.time() - t0
            scale_s = "" if absorbed else f"Scale={result.scale_mismatch:.6f}  "
            print(f"    ρ_T={rho_T_orig:.4f}  Ω={result.omega:.4f}  {scale_s}"
                  f"Shape={result.shape_mismatch:.6f}  Head={result.head_discrepancy:.6f}  "
                  f"Bound={result.risk_bound_total:.4f}  |dR|={dr:.4f}  "
                  f"Loss_T={loss_target:.4f}  Loss_P={result.loss_proxy:.4f}  "
                  f"PPL_T={ppl_target:.2f}  PPL_P={ppl_p:.2f}  "
                  f"K_f={K_feat:.4f}({K_empirical_feat['p95']:.4f})  "
                  f"K_p={K_pred:.4f}({K_pred_empirical:.4f})  ({elapsed:.1f}s)")

            del proxy_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.report(results)
        abs_tag = "_absorbed" if absorbed else ""
        stem = f"prism_{task_name}_n{num_samples}{abs_tag}"
        self.save(results, filename=f"{stem}.json")
        self.save_csv(results, filename=f"{stem}.csv")
        return results

    @staticmethod
    def report(results: list) -> None:
        """Print table with perplexity and |ΔR| columns.

        Automatically detects scale-absorbed mode and omits the Scale column.
        Prints Lipschitz constants in the header.
        """
        if not results:
            print("(no results)")
            return

        absorbed = results[0].extra.get("mode") == "scale_absorbed"
        r0 = results[0]

        kf_hdr = "K_f(emp)"
        kp_hdr = "K_p(emp)"
        if absorbed:
            header = (
                f"{'Label':<20s} {'ρ_T':>8s} {'Omega':>8s} {'Shape':>10s} "
                f"{'Head':>10s} {'Bound':>10s} {'|dR|':>8s} "
                f"{'Loss_T':>8s} {'Loss_P':>8s} {'PPL_T':>8s} {'PPL_P':>8s} "
                f"{'K_f':>8s} {kf_hdr:>8s} {'K_p':>8s} {kp_hdr:>8s}"
            )
        else:
            header = (
                f"{'Label':<20s} {'ρ_T':>8s} {'Omega':>8s} {'Scale':>10s} {'Shape':>10s} "
                f"{'Head':>10s} {'Bound':>10s} {'|dR|':>8s} "
                f"{'Loss_T':>8s} {'Loss_P':>8s} {'PPL_T':>8s} {'PPL_P':>8s} "
                f"{'K_f':>8s} {kf_hdr:>8s} {'K_p':>8s} {kp_hdr:>8s}"
            )

        mode_tag = " (scale-absorbed)" if absorbed else ""
        ds_tag = ""
        if "dataset" in r0.extra:
            ds_tag = f"  [data: {r0.extra['dataset']}, n={r0.extra.get('num_samples', '?')}]"
        sep = "-" * len(header)
        print(f"\n{'=' * len(header)}")
        print(f"  PRISM Quantization Results{mode_tag}{ds_tag}")
        print(f"{'=' * len(header)}")

        rho_T_val = r0.extra.get("rho_T")
        if rho_T_val is not None:
            print(f"  ρ_T = {rho_T_val:.6f}")

        if "K_feat_tight" in r0.extra:
            K_f = r0.extra["K_feat_tight"]
            K_fn = r0.extra.get("K_feat_naive")
            K_fe = r0.extra.get("K_feat_empirical")
            K_p = r0.extra.get("K_pred_theory")
            K_pe = r0.extra.get("K_pred_empirical")
            print(f"  K_feat:  {K_f:.4f} (tight)  "
                  + (f"{K_fn:.4f} (naive)  " if K_fn is not None else "")
                  + (f"{K_fe:.4f} (emp)" if K_fe is not None else ""))
            print(f"  K_pred:  {K_p:.4f} (theory) "
                  + (f"{K_pe:.4f} (emp)" if K_pe is not None else ""))
            print(sep)

        print(header)
        print(sep)

        for r in results:
            dr = abs(r.loss_target - r.loss_proxy) if r.loss_target is not None and r.loss_proxy is not None else None
            dr_s = f"{dr:.4f}" if dr is not None else "—"
            bt = f"{r.risk_bound_total:.4f}" if r.risk_bound_total is not None else "—"
            lt = f"{r.loss_target:.4f}" if r.loss_target is not None else "—"
            lp = f"{r.loss_proxy:.4f}" if r.loss_proxy is not None else "—"
            ppl_t = f"{r.extra.get('perplexity_target', 0):.2f}" if "perplexity_target" in r.extra else "—"
            ppl_p = f"{r.extra.get('perplexity_proxy', 0):.2f}" if "perplexity_proxy" in r.extra else "—"
            kf = f"{r.extra.get('K_feat_tight', 0):.4f}" if "K_feat_tight" in r.extra else "—"
            kfe = f"{r.extra.get('K_feat_empirical', 0):.4f}" if "K_feat_empirical" in r.extra else "—"
            kp = f"{r.extra.get('K_pred_theory', 0):.4f}" if "K_pred_theory" in r.extra else "—"
            kpe = f"{r.extra.get('K_pred_empirical', 0):.4f}" if "K_pred_empirical" in r.extra else "—"
            rho_s = f"{r.extra.get('rho_T', 0):.4f}" if "rho_T" in r.extra else "—"
            if absorbed:
                print(
                    f"{r.label:<20s} {rho_s:>8s} {r.omega:>8.4f} "
                    f"{r.shape_mismatch:>10.6f} {r.head_discrepancy:>10.6f} "
                    f"{bt:>10s} {dr_s:>8s} "
                    f"{lt:>8s} {lp:>8s} {ppl_t:>8s} {ppl_p:>8s} "
                    f"{kf:>8s} {kfe:>8s} {kp:>8s} {kpe:>8s}"
                )
            else:
                print(
                    f"{r.label:<20s} {rho_s:>8s} {r.omega:>8.4f} {r.scale_mismatch:>10.6f} "
                    f"{r.shape_mismatch:>10.6f} {r.head_discrepancy:>10.6f} "
                    f"{bt:>10s} {dr_s:>8s} "
                    f"{lt:>8s} {lp:>8s} {ppl_t:>8s} {ppl_p:>8s} "
                    f"{kf:>8s} {kfe:>8s} {kp:>8s} {kpe:>8s}"
                )

        valid = [(r, abs(r.loss_target - r.loss_proxy))
                 for r in results
                 if r.loss_target is not None and r.loss_proxy is not None and r.risk_bound_total is not None]
        if valid:
            print(sep)
            holds = sum(1 for r, dr in valid if r.risk_bound_total >= dr)
            print(f"  Bound holds: {holds}/{len(valid)}  "
                  f"({'ALL PASS' if holds == len(valid) else 'SOME VIOLATED'})")

        print(f"{'=' * len(header)}\n")
