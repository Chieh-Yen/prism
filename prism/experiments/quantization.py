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

        target_model_id = cfg_target.get("model", "NousResearch/Llama-2-7b-hf")
        quant_repo = cfg_proxy.get("model", "TheBloke/Llama-2-7b-GGUF")
        quant_bits: List[str] = cfg_proxy.get("quantization_bits", list(_LLAMA2_GGUF_MAP.keys()))

        task_name = cfg_data.get("task", "wikitext")
        num_samples = cfg_data.get("num_samples", 256)
        batch_size = cfg_data.get("batch_size", 8)
        max_length = cfg_data.get("max_length", 512)

        print(f"{'=' * 72}")
        print(f"  PRISM Experiment: Quantization Quality Estimation")
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
        loss_target = self.compute_lm_loss(target_model, dataloader, self.device)
        ppl_target = math.exp(loss_target)
        print(f"  Target: Loss={loss_target:.4f}  PPL={ppl_target:.2f}  "
              f"Z={tuple(Z_T.shape)}  H={tuple(H_T.shape)}")

        del target_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  Target model freed from VRAM.")

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

            result = self.compute_metrics(Z_T, H_T, Z_P, H_P, force_identity=True, label=label)

            result.loss_target = loss_target
            result.loss_proxy = self.compute_lm_loss(proxy_model, dataloader, self.device)
            result.extra["perplexity_target"] = ppl_target
            result.extra["perplexity_proxy"] = math.exp(result.loss_proxy)

            results.append(result)

            dr = abs(result.loss_target - result.loss_proxy)
            ppl_p = result.extra["perplexity_proxy"]
            elapsed = time.time() - t0
            print(f"    Omega={result.omega:.4f}  Scale={result.scale_mismatch:.6f}  "
                  f"Shape={result.shape_mismatch:.6f}  Head={result.head_discrepancy:.6f}  "
                  f"Bound={result.risk_bound_total:.4f}  |dR|={dr:.4f}  "
                  f"Loss_T={loss_target:.4f}  Loss_P={result.loss_proxy:.4f}  "
                  f"PPL_T={ppl_target:.2f}  PPL_P={ppl_p:.2f}  ({elapsed:.1f}s)")

            del proxy_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.report(results)
        self.save(results)
        return results

    @staticmethod
    def report(results: list) -> None:
        """Print table with perplexity and |ΔR| columns."""
        if not results:
            print("(no results)")
            return

        header = (
            f"{'Label':<20s} {'Omega':>8s} {'Scale':>10s} {'Shape':>10s} "
            f"{'Head':>10s} {'Bound':>8s} {'|dR|':>8s} "
            f"{'Loss_T':>8s} {'Loss_P':>8s} {'PPL_T':>8s} {'PPL_P':>8s}"
        )
        sep = "-" * len(header)
        print(f"\n{'=' * len(header)}")
        print("  PRISM Quantization Results")
        print(f"{'=' * len(header)}")
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
            print(
                f"{r.label:<20s} {r.omega:>8.4f} {r.scale_mismatch:>10.6f} "
                f"{r.shape_mismatch:>10.6f} {r.head_discrepancy:>10.6f} "
                f"{bt:>8s} {dr_s:>8s} "
                f"{lt:>8s} {lp:>8s} {ppl_t:>8s} {ppl_p:>8s}"
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
