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
from ..data.loaders import get_task_metadata, load_task_data
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

        # --- Dataset-specific Z extraction and loss modes ---
        task_meta = get_task_metadata(task_name)
        z_mode = cfg_data.get("z_mode") or task_meta["z_mode"]
        loss_mode = task_meta["loss_mode"]

        mode_tag = " (scale-absorbed)" if absorbed else ""
        print(f"{'=' * 72}")
        print(f"  PRISM Experiment: Instruction-Tuning Forgetting{mode_tag}")
        print(f"{'=' * 72}")
        print(f"  Target (instruct): {target_model_id}")
        print(f"  Proxy  (base)    : {proxy_model_id}")
        print(f"  Dataset          : {task_name}  z_mode={z_mode}  loss_mode={loss_mode}  (n={num_samples}, max_len={max_length})")

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

        print(f"Extracting target features and loss (single forward pass, z_mode={z_mode}) ...")
        Z_T, loss_stats_T = extractor.extract_features_and_loss_per_sample(
            target_model, dataloader, self.tensor_device,
            chunk_size=self.logit_chunk_size,
            z_mode=z_mode,
        )
        H_T = extractor.extract_head(target_model)

        # Full-text loss (always computed)
        losses_T = loss_stats_T["losses"]
        full_loss_target = losses_T.mean().item()
        full_ppl_target = math.exp(full_loss_target)

        # Answer-only loss
        has_answer = loss_stats_T["has_answer_loss"]
        aloss_target = (
            loss_stats_T["answer_losses"].mean().item() if has_answer else None
        )
        appl_target = math.exp(aloss_target) if aloss_target is not None else None

        # Select primary loss based on loss_mode
        if loss_mode == "answer":
            loss_target = aloss_target
            ppl_target = appl_target
        else:
            loss_target = full_loss_target
            ppl_target = full_ppl_target

        info = f"  Target: FullLoss={full_loss_target:.4f}  FullPPL={full_ppl_target:.2f}"
        if aloss_target is not None:
            info += f"  ALoss={aloss_target:.4f}  APPL={appl_target:.2f}"
        info += f"  Z={tuple(Z_T.shape)}  H={tuple(H_T.shape)}  z_mode={z_mode}"
        print(info)

        del target_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  Target model freed from VRAM.")

        # --- Lipschitz constants (computed from target head) ---
        rho_T_orig = Z_T.norm("fro").item() / math.sqrt(Z_T.shape[0])
        K_theory = UnifiedBound.theoretical_K(H_T)
        # Empirical K_feat must use losses paired with Z (Appendix A):
        # for last_context_token Z, use answer_losses (not full-text).
        # For concat Z, use per-token losses (1:1 pairing).
        if z_mode == "concat" and loss_stats_T["token_losses"] is not None:
            paired_losses = loss_stats_T["token_losses"]
        elif loss_mode != "full" and has_answer:
            paired_losses = loss_stats_T["answer_losses"]
        else:
            paired_losses = losses_T
        K_empirical_feat = UnifiedBound.estimate_lipschitz_lm(Z_T, paired_losses)
        K_feat = K_theory["K_feat"]
        K_pred = K_theory["K_pred"]
        K_pred_empirical = loss_stats_T["grad_norm_p95"]

        print(f"\n  ρ_T = {rho_T_orig:.6f}")
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
            "rho_T": rho_T_orig,
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

        print(f"Extracting proxy features and loss (single forward pass, z_mode={z_mode}) ...")
        Z_P, loss_stats_P = extractor.extract_features_and_loss_per_sample(
            proxy_model, dataloader, self.tensor_device,
            chunk_size=self.logit_chunk_size,
            z_mode=z_mode,
        )

        if not torch.isfinite(Z_P).all():
            print("  WARNING: proxy features contain NaN/Inf — aborting.")
            del proxy_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return []

        H_P = extractor.extract_head(proxy_model)

        # Full-text loss (always)
        full_loss_proxy = loss_stats_P["losses"].mean().item()

        # Answer-only loss
        aloss_proxy = (
            loss_stats_P["answer_losses"].mean().item() if has_answer else None
        )
        appl_proxy = math.exp(aloss_proxy) if aloss_proxy is not None else None

        info = f"  Proxy: FullLoss={full_loss_proxy:.4f}  FullPPL={math.exp(full_loss_proxy):.2f}"
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
        # Phase 3 — Metrics (Procrustes alignment, W = W_opt)
        # =============================================================
        print(f"\n--- Phase 3: PRISM Metrics (Procrustes) ---")

        target_short = target_model_id.split("/")[-1]
        proxy_short = proxy_model_id.split("/")[-1]
        label = f"{proxy_short} -> {target_short}"

        result = self.compute_metrics(
            Z_T, H_T, Z_P, H_P,
            label=label, absorbed=absorbed,
            K_feat=K_feat, K_pred=K_pred,
        )

        # Store all loss variants in extra
        result.extra["full_loss_target"] = full_loss_target
        result.extra["full_loss_proxy"] = full_loss_proxy
        result.extra["full_ppl_target"] = full_ppl_target
        result.extra["full_ppl_proxy"] = math.exp(full_loss_proxy)

        if has_answer and aloss_proxy is not None:
            result.extra["answer_loss_target"] = aloss_target
            result.extra["answer_loss_proxy"] = aloss_proxy
            result.extra["answer_ppl_target"] = appl_target
            result.extra["answer_ppl_proxy"] = appl_proxy

        # Select primary loss based on loss_mode
        if loss_mode == "answer":
            result.loss_target = aloss_target
            result.loss_proxy = aloss_proxy
            result.extra["perplexity_target"] = appl_target
            result.extra["perplexity_proxy"] = appl_proxy
        else:
            result.loss_target = full_loss_target
            result.loss_proxy = full_loss_proxy
            result.extra["perplexity_target"] = full_ppl_target
            result.extra["perplexity_proxy"] = math.exp(full_loss_proxy)

        result.extra.update(lipschitz_info)
        result.extra["dataset"] = task_name
        result.extra["z_mode"] = z_mode
        result.extra["loss_mode"] = loss_mode
        result.extra["num_samples"] = num_samples
        result.extra["target_model"] = target_model_id
        result.extra["proxy_model"] = proxy_model_id

        elapsed = time.time() - t0
        rho_P = result.rho_proxy
        scale_s = "" if absorbed else f"Scale={result.scale_mismatch:.6f}  "
        bound_s = f"{result.risk_bound_total:.4f}" if result.risk_bound_total is not None else "—"

        # Geometry line
        print(f"  [Geom]   ρ_T={rho_T_orig:.4f}  ρ_P={rho_P:.4f}  Ω={result.omega:.4f}  {scale_s}"
              f"Shape={result.shape_mismatch:.6f}  Head={result.head_discrepancy:.6f}  "
              f"Bound={bound_s}  "
              f"K_f={K_feat:.4f}({K_empirical_feat['p95']:.4f})  "
              f"K_p={K_pred:.4f}({K_pred_empirical:.4f})  ({elapsed:.1f}s)")

        # Full-text loss line
        full_dr = abs(full_loss_target - full_loss_proxy)
        full_delta = full_loss_target - full_loss_proxy
        full_dir = "forgetting" if full_delta > 0 else "improvement"
        if loss_mode == "full":
            status = "PASS" if result.risk_bound_total is not None and result.risk_bound_total >= full_dr else "VIOLATED"
            print(f"  [Full]   |dR|={full_dr:.4f}  Bound={bound_s}  {status}  "
                  f"Loss_T={full_loss_target:.4f}  Loss_P={full_loss_proxy:.4f}  "
                  f"PPL_T={full_ppl_target:.2f}  PPL_P={math.exp(full_loss_proxy):.2f}  "
                  f"delta={full_delta:+.4f} ({full_dir})")
        else:
            print(f"  [Full*]  |dR|={full_dr:.4f}  (ref, Z unpaired)  "
                  f"Loss_T={full_loss_target:.4f}  Loss_P={full_loss_proxy:.4f}  "
                  f"PPL_T={full_ppl_target:.2f}  PPL_P={math.exp(full_loss_proxy):.2f}  "
                  f"delta={full_delta:+.4f} ({full_dir})")

        # Answer-only loss line
        if has_answer and aloss_proxy is not None:
            adr = abs(aloss_target - aloss_proxy)
            a_delta = aloss_target - aloss_proxy
            a_dir = "forgetting" if a_delta > 0 else "improvement"
            status = "PASS" if result.risk_bound_total is not None and result.risk_bound_total >= adr else "VIOLATED"
            print(f"  [Answer] |AdR|={adr:.4f}  Bound={bound_s}  {status}  "
                  f"ALoss_T={aloss_target:.4f}  ALoss_P={aloss_proxy:.4f}  "
                  f"APPL_T={appl_target:.2f}  APPL_P={appl_proxy:.2f}  "
                  f"delta={a_delta:+.4f} ({a_dir})")


        results = [result]

        # =============================================================
        # Report & save
        # =============================================================
        self.report(results)
        target_slug = target_model_id.split("/")[-1].lower()
        abs_tag = "_absorbed" if absorbed else ""
        z_tag = f"_{z_mode}" if z_mode != task_meta["z_mode"] else ""
        stem = f"prism_ft_{target_slug}_{task_name}_n{num_samples}{z_tag}{abs_tag}"
        self.save(results, filename=f"{stem}.json")
        # Always save full-text CSV (for analysis / debug)
        self.save_csv(results, filename=f"{stem}_full.csv", loss_mode="full")
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
            z_m = r.extra.get("z_mode", "?")
            l_m = r.extra.get("loss_mode", "?")
            ds_tag = (f"  [data: {r.extra['dataset']}, n={r.extra.get('num_samples', '?')}, "
                      f"z_mode={z_m}, loss_mode={l_m}]")

        hdr = f"  PRISM Finetuning Results{mode_tag}{ds_tag}"
        sep = "=" * max(len(hdr) + 4, 72)
        print(f"\n{sep}")
        print(hdr)
        print(sep)
        print(f"  Target (instruct): {r.extra.get('target_model', '?')}")
        print(f"  Proxy  (base)    : {r.extra.get('proxy_model', '?')}")

        rho_T_val = r.extra.get("rho_T")
        if rho_T_val is not None:
            print(f"  ρ_T = {rho_T_val:.6f}    ρ_P = {r.rho_proxy:.6f}")

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

        # --- Primary loss ---
        bound = r.risk_bound_total
        loss_mode = r.extra.get("loss_mode", "full")
        dr = abs(r.loss_target - r.loss_proxy) if r.loss_target is not None and r.loss_proxy is not None else None
        if dr is not None:
            delta = r.loss_target - r.loss_proxy
            direction = "forgetting" if delta > 0 else "improvement"
            status = "PASS" if bound is not None and bound >= dr else "VIOLATED"

            print(f"  Primary loss (loss_mode={loss_mode}):")
            print(f"  Loss_T(instruct) = {r.loss_target:.4f}   "
                  f"PPL_T = {r.extra.get('perplexity_target', 0):.2f}")
            print(f"  Loss_P(base)     = {r.loss_proxy:.4f}   "
                  f"PPL_P = {r.extra.get('perplexity_proxy', 0):.2f}")
            print(f"  delta_loss       = {delta:+.4f}  ({direction})")
            print(f"  |dR|             = {dr:.4f}")
            print(f"  Bound            = {bound:.4f}   [{status}]")

        # --- Full-text loss table ---
        has_full = "full_loss_target" in r.extra
        if has_full:
            is_paired = (loss_mode == "full")
            tag = "" if is_paired else " (ref, Z unpaired)"
            flt = r.extra["full_loss_target"]
            flp = r.extra["full_loss_proxy"]
            fdr = abs(flt - flp)
            f_delta = flt - flp
            f_dir = "forgetting" if f_delta > 0 else "improvement"
            print(f"\n  [Full-text Loss{tag}]")
            print(f"  Loss_T = {flt:.4f}   PPL_T = {r.extra.get('full_ppl_target', 0):.2f}")
            print(f"  Loss_P = {flp:.4f}   PPL_P = {r.extra.get('full_ppl_proxy', 0):.2f}")
            print(f"  delta  = {f_delta:+.4f}  ({f_dir})")
            if is_paired:
                f_status = "PASS" if bound is not None and bound >= fdr else "VIOLATED"
                print(f"  |dR|   = {fdr:.4f}   Bound = {bound:.4f}   [{f_status}]")
            else:
                print(f"  |dR|   = {fdr:.4f}")

        # --- Answer-only loss ---
        if "answer_loss_target" in r.extra:
            alt = r.extra["answer_loss_target"]
            alp = r.extra["answer_loss_proxy"]
            adr = abs(alt - alp)
            a_delta = alt - alp
            a_dir = "forgetting" if a_delta > 0 else "improvement"
            a_status = "PASS" if bound is not None and bound >= adr else "VIOLATED"
            print(f"\n  [Answer-only Loss]")
            print(f"  ALoss_T = {alt:.4f}   APPL_T = {r.extra.get('answer_ppl_target', 0):.2f}")
            print(f"  ALoss_P = {alp:.4f}   APPL_P = {r.extra.get('answer_ppl_proxy', 0):.2f}")
            print(f"  delta   = {a_delta:+.4f}  ({a_dir})")
            print(f"  |AdR|   = {adr:.4f}   Bound = {bound:.4f}   [{a_status}]")

        # --- Final-answer-only loss (GSM8K) ---
        if "final_loss_target" in r.extra:
            flt = r.extra["final_loss_target"]
            flp = r.extra["final_loss_proxy"]
            fdr = abs(flt - flp)
            f_delta = flt - flp
            f_dir = "forgetting" if f_delta > 0 else "improvement"
            f_status = "PASS" if bound is not None and bound >= fdr else "VIOLATED"
            print(f"\n  [Final-answer-only Loss (GSM8K: number only)]")
            print(f"  FLoss_T = {flt:.4f}   FPPL_T = {r.extra.get('final_ppl_target', 0):.2f}")
            print(f"  FLoss_P = {flp:.4f}   FPPL_P = {r.extra.get('final_ppl_proxy', 0):.2f}")
            print(f"  delta   = {f_delta:+.4f}  ({f_dir})")
            print(f"  |FdR|   = {fdr:.4f}   Bound = {bound:.4f}   [{f_status}]")

        print(f"{sep}\n")
