"""
Instruction-Tuning Forgetting Experiment — Rotational Regime (W ∈ O(d)).

Target = instruct / chat model (fine-tuned).
Proxy  = base / pre-trained model.

Uses Procrustes alignment to solve for W, then measures geometric drift
and the unified risk bound:

    |R_instruct − R_base| ≤ K_feat · δ  +  K_pred · γ

where δ = √[(ρ_T−ρ_P)² + 2ρ_Tρ_P(1−Ω)]  and  γ = ‖Σ_P^{1/2}(WH_T−H_P)‖_F.

A positive signed delta_loss (Loss_T − Loss_P) indicates the instruct model
performs *worse* (forgetting on general text), while negative indicates
*improvement* (better Q&A ability).
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


class FinetuningExperiment(BaseExperiment):
    """Compare instruct (target) with base (proxy) using Procrustes alignment."""

    def setup_pairs(self) -> List[Dict[str, Any]]:
        return []

    def run(self) -> list:
        cfg_target = self.config.get("target", {})
        cfg_proxy = self.config.get("proxy", {})
        cfg_data = self.config.get("data", {})
        cfg_align = self.config.get("alignment", {})

        target_model_id = cfg_target.get("model")
        proxy_model_id = cfg_proxy.get("model")
        absorbed: bool = cfg_align.get("scale_absorbed", False)

        task_name = cfg_data.get("task", "wikitext")
        num_samples = cfg_data.get("num_samples", 256)
        batch_size = cfg_data.get("batch_size", 8)
        max_length = cfg_data.get("max_length", 512)

        if not target_model_id or not proxy_model_id:
            raise ValueError(
                "Both target.model (instruct) and proxy.model (base) must be set."
            )

        mode_tag = " (scale-absorbed)" if absorbed else ""
        print(f"{'=' * 72}")
        print(f"  PRISM Experiment: Instruction-Tuning Forgetting{mode_tag}")
        print(f"{'=' * 72}")
        print(f"  Target (instruct): {target_model_id}")
        print(f"  Proxy  (base)    : {proxy_model_id}")
        print(f"  Dataset          : {task_name}  (n={num_samples}, max_len={max_length})")

        print(f"\nLoading tokenizer from {target_model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            target_model_id, trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading data: {task_name} (n={num_samples}) ...")
        dataloader = load_task_data(
            task_name, split="test", num_samples=num_samples,
            batch_size=batch_size, tokenizer=tokenizer, max_length=max_length,
        )

        extractor = LLMExtractor()

        # =============================================================
        # Phase 1 — Target (instruct): extract, compute loss, free VRAM
        # =============================================================
        print(f"\n--- Phase 1: Target (instruct) ---")
        print(f"Loading target (FP16): {target_model_id} ...")
        target_model = AutoModelForCausalLM.from_pretrained(
            target_model_id, dtype=torch.float16, device_map=self.device,
            trust_remote_code=True,
        )
        target_model.eval()

        print("Extracting target features ...")
        Z_T = extractor.extract_features(target_model, dataloader, self.device)
        H_T = extractor.extract_head(target_model)

        print("Computing target loss ...")
        loss_stats_T = self.compute_lm_loss_per_sample(
            target_model, dataloader, self.device,
        )
        losses_T = loss_stats_T["losses"]
        loss_target = losses_T.mean().item()
        ppl_target = math.exp(loss_target)

        has_answer = loss_stats_T["has_answer_loss"]
        aloss_target = (
            loss_stats_T["answer_losses"].mean().item() if has_answer else None
        )
        appl_target = math.exp(aloss_target) if aloss_target is not None else None

        info = f"  Target: Loss={loss_target:.4f}  PPL={ppl_target:.2f}"
        if aloss_target is not None:
            info += f"  ALoss={aloss_target:.4f}  APPL={appl_target:.2f}"
        info += f"  Z={tuple(Z_T.shape)}  H={tuple(H_T.shape)}"
        print(info)

        del target_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  Target model freed from VRAM.")

        # --- Lipschitz constants (computed from target head) ---
        rho_T = Z_T.norm("fro").item() / math.sqrt(Z_T.shape[0])
        K_theory = UnifiedBound.theoretical_K(H_T)
        K_empirical_feat = UnifiedBound.estimate_lipschitz_lm(Z_T, losses_T)
        K_feat = K_theory["K_feat"]
        K_pred = K_theory["K_pred"]
        K_pred_empirical = loss_stats_T["grad_norm_p95"]

        print(f"\n  rho_T = {rho_T:.6f}")
        print(f"\n  Lipschitz constants:")
        print(f"    K_feat (tight)     = {K_feat:.4f}")
        print(f"    K_feat (naive)     = {K_theory['K_feat_naive']:.4f}")
        print(f"    K_feat (empirical) = {K_empirical_feat['p95']:.4f}  "
              f"(median={K_empirical_feat['median']:.4f}, "
              f"max={K_empirical_feat['max']:.4f})")
        print(f"    K_pred (theory)    = {K_pred:.4f}")
        print(f"    K_pred (empirical) = {K_pred_empirical:.4f}  "
              f"(mean={loss_stats_T['grad_norm_mean']:.4f}, "
              f"max={loss_stats_T['grad_norm_max']:.4f})")

        lipschitz_info = {
            "K_feat_tight": K_feat,
            "K_feat_naive": K_theory["K_feat_naive"],
            "K_feat_empirical": K_empirical_feat["K_feat_empirical"],
            "K_feat_empirical_median": K_empirical_feat["median"],
            "K_feat_empirical_max": K_empirical_feat["max"],
            "K_pred_theory": K_pred,
            "K_pred_empirical": K_pred_empirical,
            "K_pred_empirical_mean": loss_stats_T["grad_norm_mean"],
            "K_pred_empirical_max": loss_stats_T["grad_norm_max"],
            "H_T_spectral": K_theory["H_T_spectral"],
            "max_pairwise_dist": K_theory["max_pairwise_dist"],
            "rho_T": rho_T,
        }

        # =============================================================
        # Phase 2 — Proxy (base): extract, compute loss, free VRAM
        # =============================================================
        print(f"\n--- Phase 2: Proxy (base) ---")
        print(f"Loading proxy (FP16): {proxy_model_id} ...")
        t0 = time.time()

        proxy_model = AutoModelForCausalLM.from_pretrained(
            proxy_model_id, dtype=torch.float16, device_map=self.device,
            trust_remote_code=True,
        )
        proxy_model.eval()

        print("Extracting proxy features ...")
        Z_P = extractor.extract_features(proxy_model, dataloader, self.device)

        if not torch.isfinite(Z_P).all():
            print("  WARNING: proxy features contain NaN/Inf — aborting.")
            del proxy_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return []

        H_P = extractor.extract_head(proxy_model)

        print("Computing proxy loss ...")
        loss_stats_P = self.compute_lm_loss_per_sample(
            proxy_model, dataloader, self.device,
        )
        loss_proxy = loss_stats_P["losses"].mean().item()
        ppl_proxy = math.exp(loss_proxy)

        aloss_proxy = (
            loss_stats_P["answer_losses"].mean().item() if has_answer else None
        )
        appl_proxy = math.exp(aloss_proxy) if aloss_proxy is not None else None

        info = f"  Proxy: Loss={loss_proxy:.4f}  PPL={ppl_proxy:.2f}"
        if aloss_proxy is not None:
            info += f"  ALoss={aloss_proxy:.4f}  APPL={appl_proxy:.2f}"
        info += f"  Z={tuple(Z_P.shape)}  H={tuple(H_P.shape)}"
        print(info)

        del proxy_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  Proxy model freed from VRAM.")

        # =============================================================
        # Phase 3 — Metrics (Procrustes alignment, force_identity=False)
        # =============================================================
        print(f"\n--- Phase 3: PRISM Metrics (Procrustes) ---")

        target_short = target_model_id.split("/")[-1]
        proxy_short = proxy_model_id.split("/")[-1]
        label = f"{proxy_short} -> {target_short}"

        result = self.compute_metrics(
            Z_T, H_T, Z_P, H_P,
            force_identity=False, label=label, absorbed=absorbed,
            K_feat=K_feat, K_pred=K_pred,
        )

        result.loss_target = loss_target
        result.loss_proxy = loss_proxy
        result.extra["perplexity_target"] = ppl_target
        result.extra["perplexity_proxy"] = ppl_proxy
        if has_answer:
            result.extra["answer_loss_target"] = aloss_target
            result.extra["answer_loss_proxy"] = aloss_proxy
            result.extra["answer_ppl_target"] = appl_target
            result.extra["answer_ppl_proxy"] = appl_proxy
        result.extra.update(lipschitz_info)
        result.extra["dataset"] = task_name
        result.extra["num_samples"] = num_samples
        result.extra["target_model"] = target_model_id
        result.extra["proxy_model"] = proxy_model_id

        dr = abs(loss_target - loss_proxy)
        delta_loss = loss_target - loss_proxy
        result.extra["delta_loss"] = delta_loss
        direction = "forgetting" if delta_loss > 0 else "improvement"

        elapsed = time.time() - t0
        scale_s = "" if absorbed else f"Scale={result.scale_mismatch:.6f}  "
        print(
            f"  rho_T={rho_T:.4f}  Omega={result.omega:.4f}  {scale_s}"
            f"Shape={result.shape_mismatch:.6f}  Head={result.head_discrepancy:.6f}"
        )
        print(
            f"  Bound={result.risk_bound_total:.4f}  |dR|={dr:.4f}  "
            f"delta_loss={delta_loss:+.4f} ({direction})"
        )
        print(
            f"  Loss_T(instruct)={loss_target:.4f}  "
            f"Loss_P(base)={loss_proxy:.4f}  "
            f"PPL_T={ppl_target:.2f}  PPL_P={ppl_proxy:.2f}"
        )
        print(
            f"  K_f={K_feat:.4f}({K_empirical_feat['p95']:.4f})  "
            f"K_p={K_pred:.4f}({K_pred_empirical:.4f})"
        )
        bound_ok = result.risk_bound_total >= dr
        print(
            f"  Bound {'HOLDS' if bound_ok else 'VIOLATED'}  "
            f"(Bound={result.risk_bound_total:.4f} vs |dR|={dr:.4f})"
        )

        if has_answer:
            adr = abs(aloss_target - aloss_proxy)
            a_delta = aloss_target - aloss_proxy
            a_dir = "forgetting" if a_delta > 0 else "improvement"
            a_ok = result.risk_bound_total >= adr
            print(
                f"  [Answer] |AdR|={adr:.4f}  delta={a_delta:+.4f} ({a_dir})  "
                f"ALoss_T={aloss_target:.4f}  ALoss_P={aloss_proxy:.4f}  "
                f"{'PASS' if a_ok else 'VIOLATED'}"
            )

        print(f"  ({elapsed:.1f}s)")

        results = [result]

        # =============================================================
        # Report & save
        # =============================================================
        self.report(results)
        target_slug = target_model_id.split("/")[-1].lower()
        abs_tag = "_absorbed" if absorbed else ""
        stem = f"prism_ft_{target_slug}_{task_name}_n{num_samples}{abs_tag}"
        self.save(results, filename=f"{stem}.json")
        self.save_csv(results, filename=f"{stem}_qa.csv", loss_mode="full")
        if has_answer:
            self.save_csv(results, filename=f"{stem}_ans.csv", loss_mode="answer")
        return results

    # ------------------------------------------------------------------
    # Custom report
    # ------------------------------------------------------------------
    @staticmethod
    def report(results: list) -> None:
        if not results:
            print("(no results)")
            return

        r = results[0]
        absorbed = r.extra.get("mode") == "scale_absorbed"
        mode_tag = " (scale-absorbed)" if absorbed else ""
        ds_tag = ""
        if "dataset" in r.extra:
            ds_tag = f"  [data: {r.extra['dataset']}, n={r.extra.get('num_samples', '?')}]"

        hdr = f"  PRISM Finetuning Results{mode_tag}{ds_tag}"
        sep = "=" * max(len(hdr) + 4, 72)
        print(f"\n{sep}")
        print(hdr)
        print(sep)
        print(f"  Target (instruct): {r.extra.get('target_model', '?')}")
        print(f"  Proxy  (base)    : {r.extra.get('proxy_model', '?')}")

        rho_T_val = r.extra.get("rho_T")
        if rho_T_val is not None:
            print(f"  rho_T = {rho_T_val:.6f}")

        if "K_feat_tight" in r.extra:
            K_f = r.extra["K_feat_tight"]
            K_fn = r.extra.get("K_feat_naive")
            K_fe = r.extra.get("K_feat_empirical")
            K_p = r.extra.get("K_pred_theory")
            K_pe = r.extra.get("K_pred_empirical")
            print(
                f"  K_feat:  {K_f:.4f} (tight)  "
                + (f"{K_fn:.4f} (naive)  " if K_fn is not None else "")
                + (f"{K_fe:.4f} (emp)" if K_fe is not None else "")
            )
            print(
                f"  K_pred:  {K_p:.4f} (theory) "
                + (f"{K_pe:.4f} (emp)" if K_pe is not None else "")
            )

        print("-" * len(sep))

        geo_cols = "Omega     Scale      Shape       Head       FeatErr"
        if absorbed:
            geo_cols = "Omega     Shape       Head       FeatErr"

        print(f"  {geo_cols}")
        if absorbed:
            print(
                f"  {r.omega:.4f}    "
                f"{r.shape_mismatch:.6f}   {r.head_discrepancy:.6f}   "
                f"{r.feature_error:.6f}"
            )
        else:
            print(
                f"  {r.omega:.4f}    {r.scale_mismatch:.6f}   "
                f"{r.shape_mismatch:.6f}   {r.head_discrepancy:.6f}   "
                f"{r.feature_error:.6f}"
            )

        print("-" * len(sep))
        dr = abs(r.loss_target - r.loss_proxy)
        delta = r.loss_target - r.loss_proxy
        direction = "forgetting" if delta > 0 else "improvement"
        bound = r.risk_bound_total
        status = "PASS" if bound is not None and bound >= dr else "VIOLATED"

        print(f"  Loss_T(instruct) = {r.loss_target:.4f}   "
              f"PPL_T = {r.extra.get('perplexity_target', 0):.2f}")
        print(f"  Loss_P(base)     = {r.loss_proxy:.4f}   "
              f"PPL_P = {r.extra.get('perplexity_proxy', 0):.2f}")
        print(f"  delta_loss       = {delta:+.4f}  ({direction})")
        print(f"  |dR|             = {dr:.4f}")
        print(f"  Bound            = {bound:.4f}   [{status}]")

        if "answer_loss_target" in r.extra:
            alt = r.extra["answer_loss_target"]
            alp = r.extra["answer_loss_proxy"]
            adr = abs(alt - alp)
            a_delta = alt - alp
            a_dir = "forgetting" if a_delta > 0 else "improvement"
            a_status = "PASS" if bound is not None and bound >= adr else "VIOLATED"
            print(f"\n  [Answer-only]")
            print(f"  ALoss_T = {alt:.4f}   APPL_T = {r.extra.get('answer_ppl_target', 0):.2f}")
            print(f"  ALoss_P = {alp:.4f}   APPL_P = {r.extra.get('answer_ppl_proxy', 0):.2f}")
            print(f"  delta   = {a_delta:+.4f}  ({a_dir})")
            print(f"  |AdR|   = {adr:.4f}   Bound = {bound:.4f}   [{a_status}]")

        print(f"{sep}\n")
