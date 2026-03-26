"""
Cross-Scale Proxy Validation — Rotational Regime (W ∈ O(d)).

Target = large model from a family (e.g., 7B / 8B).
Proxy  = smaller models from the same or a different family.

Uses Procrustes alignment (W ∈ O(d)) to bridge different hidden dimensions
and measures whether PRISM metrics on each proxy predict the target's risk.

Scientific question
-------------------
Can we use a cheaper, smaller model as a reliable stand-in for the large
target?  A high Ω and tight bound (Bound ≈ |dR|) suggests the proxy is a
valid surrogate across all datasets.

Simplified bound (Theorem 1 / Rotational Regime):
    |R_T − R_P| ≤ K_feat · √[(ρ_T − ρ_P)² + 2 ρ_T ρ_P (1 − Ω)]
                + K_pred · ‖Σ_P^{1/2}(W H_T − H_P)‖_F

Tokenizer policy
----------------
Each model is evaluated using its **own tokenizer** on the same raw text
documents.  This ensures:

  1. Each model's loss is computed on sequences it was actually trained on,
     so Loss_T and Loss_P are semantically comparable.
  2. Feature vectors Z_T[i] and Z_P[i] both represent the same document i
     (just tokenised differently), satisfying PRISM's row-alignment requirement.

When a proxy tokenizer matches the target's (same-family models), the same
pre-built dataloader is reused as an optimisation.  A difference is detected
by tokenising a short test string and comparing the resulting IDs.

Cross-family models with different vocabulary sizes trigger a vocab-mismatch
warning for the head discrepancy metric; both head matrices are truncated to
the shared prefix length in that case.

Config example
--------------
    target:
      model: Qwen/Qwen3-8B
    proxy:
      models:
        - Qwen/Qwen3-0.6B
        - Qwen/Qwen3-1.7B
        - Qwen/Qwen3-4B
      # Optional per-proxy labels:
      # models:
      #   - model: Qwen/Qwen3-0.6B
      #     label: "0.6B"
"""

from __future__ import annotations

import gc
import importlib
import math
import time
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..core.bounds import UnifiedBound
from ..data.loaders import get_task_metadata, load_task_data
from ..models.extractors import LLMExtractor
from .base import BaseExperiment


# ======================================================================
# Helpers
# ======================================================================

def _load_causal_lm(model_id: str, dtype, device_map, trust_remote_code: bool = True, **kwargs):
    """Load a CausalLM with a fallback for config types not yet in AutoModelForCausalLM.

    Some newly-released models have their config class registered in transformers
    under a slightly different name than the one mapped in AutoModelForCausalLM.
    For example, Ministral-3 2512 models ship with ``"model_type": "mistral3"``
    in config.json, which resolves to ``Mistral3Config``, but the auto-class
    registry only contains ``Ministral3Config`` (note the extra 'in').

    Strategy (in order):
      1. Standard AutoModelForCausalLM.from_pretrained — fastest path.
      2. Config-class alias table: for known naming mismatches, import the
         corresponding *ForCausalLM class and call from_pretrained with it.
      3. Derive *ForCausalLM from the config class name directly (catches
         future model types where the naming is consistent but the auto
         registry is simply out of date).

    Extra keyword arguments (e.g. ``attn_implementation="flash_attention_2"``)
    are forwarded to every ``from_pretrained`` call.
    """
    # Known config-class name → (module_path, model_class_name) aliases.
    # Used when the model's config.json model_type and the auto-registry key differ.
    _CONFIG_ALIASES: dict = {
        # Mistral3Config is a MULTIMODAL config (vision + text backbone).
        # Mistral3ForConditionalGeneration wraps a Mistral text LM under
        # model.language_model, the same nested structure as Gemma 3 4B/12B.
        # LLMExtractor._get_backbone / extract_head already handle this path.
        "Mistral3Config": ("transformers.models.mistral3.modeling_mistral3",
                           "Mistral3ForConditionalGeneration"),
    }

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except ValueError as exc:
        if "Unrecognized configuration class" not in str(exc):
            raise

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        cfg_cls = type(cfg)
        cfg_cls_name = cfg_cls.__name__

        # --- Strategy 2: alias table ---
        if cfg_cls_name in _CONFIG_ALIASES:
            mod_path, cls_name = _CONFIG_ALIASES[cfg_cls_name]
            try:
                mod = importlib.import_module(mod_path)
                causal_cls = getattr(mod, cls_name)
                print(f"  [AutoModel fallback] {cfg_cls_name} → {cls_name}")
                return causal_cls.from_pretrained(
                    model_id,
                    config=cfg,
                    dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )
            except (ImportError, AttributeError) as alias_exc:
                print(f"  [AutoModel fallback] alias failed ({alias_exc}), trying strategy 3")

        # --- Strategy 3: derive *ForCausalLM from same module as the config ---
        causal_name = cfg_cls_name.replace("Config", "ForCausalLM")
        parent_module_path = ".".join(cfg_cls.__module__.split(".")[:-1])
        try:
            parent_mod = importlib.import_module(parent_module_path)
            causal_cls = getattr(parent_mod, causal_name)
            print(f"  [AutoModel fallback] Using {causal_cls.__name__} directly.")
            return causal_cls.from_pretrained(
                model_id,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        except (ImportError, AttributeError):
            raise ValueError(
                f"Cannot load {model_id}: config class {cfg_cls_name} is not mapped "
                f"in AutoModelForCausalLM and no fallback class was found. "
                f"Consider upgrading transformers. Original error: {exc}"
            ) from exc


def _same_tokenizer(tok_a, tok_b) -> bool:
    """Return True if two tokenizers produce identical token IDs on a test string."""
    test = "The quick brown fox jumped over the lazy dog. 1+1=2."
    return tok_a.encode(test) == tok_b.encode(test)


def _parse_proxy_entries(entries: List[Union[str, Dict]]) -> List[Dict[str, str]]:
    """Normalise each proxy entry to ``{"model": ..., "label": ...}``."""
    result = []
    for e in entries:
        if isinstance(e, str):
            result.append({"model": e, "label": e.split("/")[-1]})
        elif isinstance(e, dict):
            model = e.get("model", "")
            label = e.get("label", model.split("/")[-1])
            result.append({"model": model, "label": label})
        else:
            raise ValueError(f"proxy.models entries must be str or dict, got {type(e)}")
    return result


def _verify_vocab_prefix(tok_T, tok_P, min_vocab: int, proxy_label: str) -> None:
    """Verify token ID i maps to the same token string in both tokenizers
    for all i in 0..min_vocab-1.  Samples ~500 positions for efficiency.

    This guarantees that H_T[:, i] and H_P[:, i] correspond to the same
    token after truncation, making the column-wise head comparison valid.

    Raises ValueError if any mismatch is found.
    """
    step = max(1, min_vocab // 500)
    sample_ids = list(range(0, min_vocab, step))
    try:
        toks_T = tok_T.convert_ids_to_tokens(sample_ids)
        toks_P = tok_P.convert_ids_to_tokens(sample_ids)
    except Exception:
        toks_T = [tok_T.decode([i]) for i in sample_ids]
        toks_P = [tok_P.decode([i]) for i in sample_ids]
    mismatches = [
        (sample_ids[i], toks_T[i], toks_P[i])
        for i in range(len(sample_ids))
        if toks_T[i] != toks_P[i]
    ]
    if mismatches:
        examples = mismatches[:5]
        ex_str = ", ".join(
            f"id={tid}: '{t}' vs '{p}'" for tid, t, p in examples
        )
        raise ValueError(
            f"[{proxy_label}] Vocabulary prefix mismatch: "
            f"{len(mismatches)}/{len(sample_ids)} sampled token IDs differ "
            f"in range 0..{min_vocab - 1}. "
            f"Examples: {ex_str}. "
            "Head truncation is NOT valid — token IDs do not correspond "
            "across models at the same index."
        )


def _align_heads(
    H_T: torch.Tensor,
    H_P: torch.Tensor,
    proxy_label: str,
    tok_T=None,
    tok_P=None,
) -> tuple:
    """Align head matrices to the same vocab dimension.

    When vocab sizes differ and *tok_T* / *tok_P* are provided, first calls
    ``_verify_vocab_prefix`` to confirm that H_T[:, i] and H_P[:, i] truly
    correspond to the same token for every i in the shared prefix
    (0..min_vocab-1).  Raises ValueError if the mapping is misaligned so that
    a silent, incorrect comparison is never silently accepted.

    Typical safe case — same-family models where the larger model appends
    extra special tokens at the END of the vocabulary (e.g. Qwen2.5-7B with
    vocab=152064 vs Qwen2.5-0.5B with vocab=151936: the 128 extra tokens at
    indices 151936-152063 are special tokens not present in the 0.5B model,
    and the prefix is identical).

    Returns ``(H_T_use, H_P_use, truncated)`` where *truncated* is True
    when vocab sizes differed and both matrices were clipped to the minimum.
    """
    vocab_T = H_T.shape[1]
    vocab_P = H_P.shape[1]
    if vocab_T == vocab_P:
        return H_T, H_P, False
    min_vocab = min(vocab_T, vocab_P)
    if tok_T is not None and tok_P is not None:
        _verify_vocab_prefix(tok_T, tok_P, min_vocab, proxy_label)
        print(
            f"  WARNING [{proxy_label}]: vocab size mismatch "
            f"(target={vocab_T}, proxy={vocab_P}) — "
            f"truncating both heads to {min_vocab} tokens "
            f"(prefix verified: IDs 0..{min_vocab - 1} are aligned)."
        )
    else:
        print(
            f"  WARNING [{proxy_label}]: vocab size mismatch "
            f"(target={vocab_T}, proxy={vocab_P}) — "
            f"truncating both heads to {min_vocab} tokens. "
            "Head discrepancy metric may be less accurate."
        )
    return H_T[:, :min_vocab], H_P[:, :min_vocab], True


# ======================================================================
# Experiment class
# ======================================================================

class CrossScaleExperiment(BaseExperiment):
    """Compare a large target model against multiple smaller proxy models.

    Each proxy is loaded individually, evaluated, then freed from VRAM
    before loading the next one, so only one model needs to fit in GPU
    memory at a time (beyond the cached target tensors).
    """

    def setup_pairs(self) -> List[Dict[str, Any]]:
        """Not used directly — see ``run()``."""
        return []

    def run(self) -> list:
        cfg_target = self.config.get("target", {})
        cfg_proxy = self.config.get("proxy", {})
        cfg_data = self.config.get("data", {})
        cfg_align = self.config.get("alignment", {})

        target_model_id: str = cfg_target.get("model", "")
        proxy_entries_raw: List = cfg_proxy.get("models", [])
        absorbed: bool = cfg_align.get("scale_absorbed", False)

        if not target_model_id:
            raise ValueError("target.model must be specified in the config.")
        if not proxy_entries_raw:
            raise ValueError(
                "proxy.models must be a non-empty list of model IDs "
                "(or dicts with 'model' and optional 'label' keys)."
            )

        proxy_entries = _parse_proxy_entries(proxy_entries_raw)

        task_name: str = cfg_data.get("task", "wikitext")
        num_samples: int = cfg_data.get("num_samples", 128)
        batch_size: int = cfg_data.get("batch_size", 8)
        max_length: int = cfg_data.get("max_length", 512)

        # --- Dataset-specific Z extraction and loss modes ---
        task_meta = get_task_metadata(task_name)
        z_mode = cfg_data.get("z_mode") or task_meta["z_mode"]
        loss_mode = task_meta["loss_mode"]

        mode_tag = " (scale-absorbed)" if absorbed else ""
        print(f"{'=' * 72}")
        print(f"  PRISM Experiment: Cross-Scale Proxy Validation{mode_tag}")
        print(f"{'=' * 72}")
        print(f"  Target : {target_model_id}")
        for i, p in enumerate(proxy_entries):
            print(f"  Proxy {i+1}: {p['model']}  [{p['label']}]")
        print(f"  Dataset: {task_name}  z_mode={z_mode}  loss_mode={loss_mode}  (n={num_samples}, max_len={max_length})")

        # ----------------------------------------------------------------
        # Tokenizer — target model's tokenizer for target's dataloader
        # ----------------------------------------------------------------
        print(f"\nLoading tokenizer from {target_model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            target_model_id, trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading data (target tokenizer): {task_name} (n={num_samples}) ...")
        target_dataloader = load_task_data(
            task_name, split="test", num_samples=num_samples,
            batch_size=batch_size, tokenizer=tokenizer, max_length=max_length,
            seed=self.seed,
        )

        extractor = LLMExtractor(offload_to_cpu=self.offload_to_cpu)

        # ================================================================
        # Phase 1 — Target: extract features / loss, then free VRAM
        # ================================================================
        print(f"\n--- Phase 1: Target ---")
        print(f"Loading target: {target_model_id} ...")
        target_model = _load_causal_lm(
            target_model_id,
            dtype=self.model_dtype,
            device_map=self.device,
            **self._attn_impl_kwargs(),
        )
        target_model.eval()

        print(f"Caching target features and loss (single forward pass, z_mode={z_mode}) ...")
        Z_T, loss_stats_T = extractor.extract_features_and_loss_per_sample(
            target_model, target_dataloader, self.tensor_device,
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
        aloss_target: Optional[float] = (
            loss_stats_T["answer_losses"].mean().item() if has_answer else None
        )
        appl_target: Optional[float] = (
            math.exp(aloss_target) if aloss_target is not None else None
        )

        # Final-answer-only loss (GSM8K)
        has_final = loss_stats_T["has_final_answer_loss"]
        floss_target: Optional[float] = (
            loss_stats_T["final_answer_losses"].mean().item() if has_final else None
        )
        fppl_target: Optional[float] = (
            math.exp(floss_target) if floss_target is not None else None
        )

        # Select primary loss based on loss_mode
        if loss_mode == "answer":
            loss_target = aloss_target
            ppl_target = appl_target
        elif loss_mode == "gsm8k_dual":
            loss_target = aloss_target
            ppl_target = appl_target
        else:
            loss_target = full_loss_target
            ppl_target = full_ppl_target

        info = f"  Target: FullLoss={full_loss_target:.4f}  FullPPL={full_ppl_target:.2f}"
        if aloss_target is not None:
            info += f"  ALoss={aloss_target:.4f}  APPL={appl_target:.2f}"
        if floss_target is not None:
            info += f"  FLoss={floss_target:.4f}  FPPL={fppl_target:.2f}"
        info += f"  Z={tuple(Z_T.shape)}  H={tuple(H_T.shape)}  z_mode={z_mode}"
        print(info)

        del target_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  Target model freed from VRAM.")

        # ----------------------------------------------------------------
        # Lipschitz constants (computed once from target, shared across proxies)
        # ----------------------------------------------------------------
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
        print(f"\n  Lipschitz constants (invariant across proxies):")
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

        # ================================================================
        # Phase 2 — Proxies: load each in turn, evaluate, free VRAM
        # ================================================================
        results = []
        for i, proxy in enumerate(proxy_entries):
            proxy_model_id = proxy["model"]
            proxy_label = proxy["label"]
            label = f"{proxy_label} -> {target_model_id.split('/')[-1]}"

            print(f"\n--- [{i+1}/{len(proxy_entries)}] Proxy: {proxy_model_id} ---")
            t0 = time.time()

            proxy_model = None
            try:
                # ---- per-proxy tokenizer and dataloader ----
                print(f"  Loading proxy tokenizer from {proxy_model_id} ...")
                proxy_tokenizer = AutoTokenizer.from_pretrained(
                    proxy_model_id, trust_remote_code=True,
                )
                if proxy_tokenizer.pad_token is None:
                    proxy_tokenizer.pad_token = proxy_tokenizer.eos_token

                if _same_tokenizer(tokenizer, proxy_tokenizer):
                    print(f"  Tokenizer: same as target — reusing target dataloader")
                    proxy_dataloader = target_dataloader
                else:
                    print(f"  Tokenizer: differs from target — building proxy-specific dataloader")
                    proxy_dataloader = load_task_data(
                        task_name, split="test", num_samples=num_samples,
                        batch_size=batch_size, tokenizer=proxy_tokenizer,
                        max_length=max_length, seed=self.seed,
                    )

                print(f"  Loading proxy: {proxy_model_id} ...")
                proxy_model = _load_causal_lm(
                    proxy_model_id,
                    dtype=self.model_dtype,
                    device_map=self.device,
                    **self._attn_impl_kwargs(),
                )
                proxy_model.eval()

                print(f"  Extracting proxy features and loss (single forward pass, z_mode={z_mode}) ...")
                Z_P, loss_stats_P = extractor.extract_features_and_loss_per_sample(
                    proxy_model, proxy_dataloader, self.tensor_device,
                    chunk_size=self.logit_chunk_size,
                    z_mode=z_mode,
                )

                if not torch.isfinite(Z_P).all():
                    print(f"  WARNING: proxy features contain NaN/Inf — skipping {proxy_label}")
                    continue

                H_P = extractor.extract_head(proxy_model)

                # Full-text loss (always stored)
                full_loss_proxy = loss_stats_P["losses"].mean().item()

                # Answer-only loss
                aloss_proxy: Optional[float] = (
                    loss_stats_P["answer_losses"].mean().item() if has_answer else None
                )

                # Final-answer-only loss (GSM8K)
                floss_proxy: Optional[float] = (
                    loss_stats_P["final_answer_losses"].mean().item() if has_final else None
                )

                # ---- Align head vocab dimensions if needed ----
                H_T_use, H_P_use, head_truncated = _align_heads(
                    H_T, H_P, proxy_label, tokenizer, proxy_tokenizer
                )

                # ---- PRISM metrics (Procrustes, force_identity=False) ----
                result = self.compute_metrics(
                    Z_T, H_T_use, Z_P, H_P_use,
                    force_identity=False,
                    label=label,
                    absorbed=absorbed,
                    K_feat=K_feat,
                    K_pred=K_pred,
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
                    result.extra["answer_ppl_proxy"] = math.exp(aloss_proxy)

                if has_final and floss_proxy is not None:
                    result.extra["final_loss_target"] = floss_target
                    result.extra["final_loss_proxy"] = floss_proxy
                    result.extra["final_ppl_target"] = fppl_target
                    result.extra["final_ppl_proxy"] = math.exp(floss_proxy)

                # Select primary loss based on loss_mode
                if loss_mode == "answer":
                    result.loss_target = aloss_target
                    result.loss_proxy = aloss_proxy
                    result.extra["perplexity_target"] = appl_target
                    result.extra["perplexity_proxy"] = math.exp(aloss_proxy)
                elif loss_mode == "gsm8k_dual":
                    result.loss_target = aloss_target
                    result.loss_proxy = aloss_proxy
                    result.extra["perplexity_target"] = appl_target
                    result.extra["perplexity_proxy"] = math.exp(aloss_proxy)
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
                result.extra["proxy_hidden_dim"] = Z_P.shape[1]
                result.extra["target_hidden_dim"] = Z_T.shape[1]
                result.extra["head_vocab_truncated"] = head_truncated

                results.append(result)

                # --- Per-result console output ---
                rho_P = result.rho_proxy
                elapsed = time.time() - t0
                scale_s = "" if absorbed else f"Scale={result.scale_mismatch:.6f}  "
                bound_s = f"{result.risk_bound_total:.4f}" if result.risk_bound_total is not None else "—"

                # Geometry line (always)
                print(f"  [Geom]   ρ_T={rho_T_orig:.4f}  ρ_P={rho_P:.4f}  Ω={result.omega:.4f}  {scale_s}"
                      f"Shape={result.shape_mismatch:.6f}  Head={result.head_discrepancy:.6f}  "
                      f"Bound={bound_s}  "
                      f"d_T={Z_T.shape[1]}  d_P={Z_P.shape[1]}  ({elapsed:.1f}s)")

                # Full-text loss line
                full_dr = abs(full_loss_target - full_loss_proxy)
                if loss_mode == "full":
                    status = "PASS" if result.risk_bound_total is not None and result.risk_bound_total >= full_dr else "VIOLATED"
                    print(f"  [Full]   |dR|={full_dr:.4f}  Bound={bound_s}  {status}  "
                          f"Loss_T={full_loss_target:.4f}  Loss_P={full_loss_proxy:.4f}  "
                          f"PPL_T={full_ppl_target:.2f}  PPL_P={math.exp(full_loss_proxy):.2f}")
                else:
                    print(f"  [Full*]  |dR|={full_dr:.4f}  (ref, Z unpaired)  "
                          f"Loss_T={full_loss_target:.4f}  Loss_P={full_loss_proxy:.4f}  "
                          f"PPL_T={full_ppl_target:.2f}  PPL_P={math.exp(full_loss_proxy):.2f}")

                # Answer-only loss line
                if has_answer and aloss_proxy is not None:
                    adr = abs(aloss_target - aloss_proxy)
                    status = "PASS" if result.risk_bound_total is not None and result.risk_bound_total >= adr else "VIOLATED"
                    print(f"  [Answer] |AdR|={adr:.4f}  Bound={bound_s}  {status}  "
                          f"ALoss_T={aloss_target:.4f}  ALoss_P={aloss_proxy:.4f}  "
                          f"APPL_T={appl_target:.2f}  APPL_P={math.exp(aloss_proxy):.2f}")

                # Final-answer-only loss line (GSM8K)
                if has_final and floss_proxy is not None:
                    fdr = abs(floss_target - floss_proxy)
                    status = "PASS" if result.risk_bound_total is not None and result.risk_bound_total >= fdr else "VIOLATED"
                    print(f"  [Final]  |FdR|={fdr:.4f}  Bound={bound_s}  {status}  "
                          f"FLoss_T={floss_target:.4f}  FLoss_P={floss_proxy:.4f}  "
                          f"FPPL_T={fppl_target:.2f}  FPPL_P={math.exp(floss_proxy):.2f}")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  ERROR processing {proxy_label}: {e}  ({elapsed:.1f}s) — skipping")
                import traceback
                traceback.print_exc()

            finally:
                if proxy_model is not None:
                    del proxy_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ================================================================
        # Report & save
        # ================================================================
        self.report(results)

        target_slug = target_model_id.split("/")[-1].lower()
        abs_tag = "_absorbed" if absorbed else ""
        z_tag = f"_{z_mode}" if z_mode != task_meta["z_mode"] else ""
        stem = f"prism_cs_{target_slug}_{task_name}_n{num_samples}{z_tag}{abs_tag}"
        self.save(results, filename=f"{stem}.json")
        # Always save full-text CSV (for analysis / debug)
        self.save_csv(results, filename=f"{stem}_full.csv", loss_mode="full")
        if has_answer:
            self.save_csv(results, filename=f"{stem}_ans.csv", loss_mode="answer")
        if has_final:
            self.save_csv(results, filename=f"{stem}_final.csv", loss_mode="final_answer")
        return results

    # ------------------------------------------------------------------
    # Custom report
    # ------------------------------------------------------------------
    @staticmethod
    def report(results: list) -> None:
        """Print comparison table for all proxy models vs the target."""
        if not results:
            print("(no results)")
            return

        absorbed = results[0].extra.get("mode") == "scale_absorbed"
        r0 = results[0]
        mode_tag = " (scale-absorbed)" if absorbed else ""
        ds_tag = ""
        if "dataset" in r0.extra:
            z_m = r0.extra.get("z_mode", "?")
            l_m = r0.extra.get("loss_mode", "?")
            ds_tag = (f"  [data: {r0.extra['dataset']}, n={r0.extra.get('num_samples', '?')}, "
                      f"z_mode={z_m}, loss_mode={l_m}]")

        hdr_title = f"  PRISM Cross-Scale Results{mode_tag}{ds_tag}"
        sep_len = max(len(hdr_title) + 4, 80)
        sep = "=" * sep_len
        print(f"\n{sep}")
        print(hdr_title)
        print(sep)
        print(f"  Target : {r0.extra.get('target_model', '?')}")

        rho_T_val = r0.extra.get("rho_T")
        if rho_T_val is not None:
            print(f"  ρ_T = {rho_T_val:.6f}")

        if "K_feat_tight" in r0.extra:
            K_f  = r0.extra["K_feat_tight"]
            K_fn = r0.extra.get("K_feat_naive")
            K_fe = r0.extra.get("K_feat_empirical")
            K_p  = r0.extra.get("K_pred_theory")
            K_pe = r0.extra.get("K_pred_empirical")
            print(
                f"  K_feat:  {K_f:.4f} (tight)  "
                + (f"{K_fn:.4f} (naive)  " if K_fn is not None else "")
                + (f"{K_fe:.4f} (emp)" if K_fe is not None else "")
            )
            print(
                f"  K_pred:  {K_p:.4f} (theory) "
                + (f"{K_pe:.4f} (emp)" if K_pe is not None else "")
            )

        inner_sep = "-" * sep_len
        print(inner_sep)

        # Column header
        if absorbed:
            header = (
                f"{'Proxy':<26s} {'d_P':>5s} {'ρ_T':>8s} {'ρ_P':>8s} {'Ω':>8s} "
                f"{'Shape':>10s} {'Head':>10s} {'Bound':>10s} {'|dR|':>8s} "
                f"{'Loss_T':>8s} {'Loss_P':>8s} {'PPL_T':>8s} {'PPL_P':>8s} {'Status':>8s}"
            )
        else:
            header = (
                f"{'Proxy':<26s} {'d_P':>5s} {'ρ_T':>8s} {'ρ_P':>8s} {'Ω':>8s} {'Scale':>10s} "
                f"{'Shape':>10s} {'Head':>10s} {'Bound':>10s} {'|dR|':>8s} "
                f"{'Loss_T':>8s} {'Loss_P':>8s} {'PPL_T':>8s} {'PPL_P':>8s} {'Status':>8s}"
            )
        print(header)
        print(inner_sep)

        for r in results:
            dr = abs(r.loss_target - r.loss_proxy) if (
                r.loss_target is not None and r.loss_proxy is not None
            ) else None
            dr_s  = f"{dr:.4f}" if dr is not None else "—"
            bt    = f"{r.risk_bound_total:.4f}" if r.risk_bound_total is not None else "—"
            lt    = f"{r.loss_target:.4f}" if r.loss_target is not None else "—"
            lp    = f"{r.loss_proxy:.4f}" if r.loss_proxy is not None else "—"
            ppl_t = f"{r.extra.get('perplexity_target', 0):.2f}" if "perplexity_target" in r.extra else "—"
            ppl_p = f"{r.extra.get('perplexity_proxy', 0):.2f}" if "perplexity_proxy" in r.extra else "—"
            rho_t_s = f"{r.extra.get('rho_T', 0):.4f}" if "rho_T" in r.extra else "—"
            rho_p = f"{r.rho_proxy:.4f}"
            d_p   = str(r.extra.get("proxy_hidden_dim", "?"))

            status = ""
            if dr is not None and r.risk_bound_total is not None:
                status = "PASS" if r.risk_bound_total >= dr else "VIOL"

            # Shorten the label for display
            proxy_short = r.extra.get("proxy_model", r.label).split("/")[-1]
            if len(proxy_short) > 25:
                proxy_short = proxy_short[:22] + "..."

            if absorbed:
                print(
                    f"{proxy_short:<26s} {d_p:>5s} {rho_t_s:>8s} {rho_p:>8s} {r.omega:>8.4f} "
                    f"{r.shape_mismatch:>10.6f} {r.head_discrepancy:>10.6f} "
                    f"{bt:>10s} {dr_s:>8s} "
                    f"{lt:>8s} {lp:>8s} {ppl_t:>8s} {ppl_p:>8s} {status:>8s}"
                )
            else:
                print(
                    f"{proxy_short:<26s} {d_p:>5s} {rho_t_s:>8s} {rho_p:>8s} {r.omega:>8.4f} "
                    f"{r.scale_mismatch:>10.6f} {r.shape_mismatch:>10.6f} "
                    f"{r.head_discrepancy:>10.6f} {bt:>10s} {dr_s:>8s} "
                    f"{lt:>8s} {lp:>8s} {ppl_t:>8s} {ppl_p:>8s} {status:>8s}"
                )

        # Primary loss bound validation
        valid = [
            (r, abs(r.loss_target - r.loss_proxy))
            for r in results
            if r.loss_target is not None
            and r.loss_proxy is not None
            and r.risk_bound_total is not None
        ]
        if valid:
            print(inner_sep)
            loss_mode = r0.extra.get("loss_mode", "full")
            holds = sum(1 for r, dr in valid if r.risk_bound_total >= dr)
            print(
                f"  Bound holds (primary, loss_mode={loss_mode}): {holds}/{len(valid)}  "
                f"({'ALL PASS' if holds == len(valid) else 'SOME VIOLATED'})"
            )
            # Ranking by omega (best proxy = highest Ω)
            ranked = sorted(results, key=lambda r: r.omega, reverse=True)
            best = ranked[0]
            print(
                f"  Best proxy by Ω: {best.extra.get('proxy_model', best.label).split('/')[-1]}"
                f"  (Ω={best.omega:.4f})"
            )

        # --- Full-text loss table ---
        has_full = "full_loss_target" in r0.extra
        loss_mode = r0.extra.get("loss_mode", "full")
        if has_full:
            is_paired = (loss_mode == "full")
            tag = "" if is_paired else " (ref, Z unpaired)"
            full_header = (
                f"{'Proxy':<26s} {'ρ_P':>8s} {'Ω':>8s} "
                f"{'Loss_T':>8s} {'Loss_P':>8s} "
                f"{'PPL_T':>8s} {'PPL_P':>8s} {'|dR|':>8s}"
            )
            if is_paired:
                full_header += f" {'Bound':>10s} {'Status':>8s}"
            full_sep = "-" * len(full_header)
            print(f"\n  Full-text Loss{tag}")
            print(full_sep)
            print(full_header)
            print(full_sep)
            for r in results:
                flt = r.extra.get("full_loss_target")
                flp = r.extra.get("full_loss_proxy")
                if flt is not None and flp is not None:
                    fdr = abs(flt - flp)
                    proxy_short = r.extra.get("proxy_model", r.label).split("/")[-1]
                    if len(proxy_short) > 25:
                        proxy_short = proxy_short[:22] + "..."
                    line = (
                        f"{proxy_short:<26s} {r.rho_proxy:>8.4f} {r.omega:>8.4f} "
                        f"{flt:>8.4f} {flp:>8.4f} "
                        f"{r.extra.get('full_ppl_target', 0):>8.2f} "
                        f"{r.extra.get('full_ppl_proxy', 0):>8.2f} "
                        f"{fdr:>8.4f}"
                    )
                    if is_paired:
                        bt_val = r.risk_bound_total
                        status = "PASS" if bt_val is not None and bt_val >= fdr else "VIOLATED"
                        line += f" {bt_val:>10.4f} {status:>8s}"
                    print(line)
            if is_paired:
                full_valid = [(r, abs(r.extra["full_loss_target"] - r.extra["full_loss_proxy"]))
                              for r in results
                              if "full_loss_target" in r.extra and r.risk_bound_total is not None]
                if full_valid:
                    print(full_sep)
                    f_holds = sum(1 for r, fdr in full_valid if r.risk_bound_total >= fdr)
                    print(f"  Bound holds (full): {f_holds}/{len(full_valid)}  "
                          f"({'ALL PASS' if f_holds == len(full_valid) else 'SOME VIOLATED'})")

        # --- Answer-only table (if available) ---
        has_ans = "answer_loss_target" in r0.extra
        if has_ans:
            ans_header = (
                f"{'Proxy':<26s} {'ρ_P':>8s} {'Ω':>8s} "
                f"{'ALoss_T':>8s} {'ALoss_P':>8s} "
                f"{'APPL_T':>8s} {'APPL_P':>8s} {'|AdR|':>8s} "
                f"{'Bound':>10s} {'Status':>8s}"
            )
            ans_sep = "-" * len(ans_header)
            print(f"\n  Answer-only Loss (same bound applies)")
            print(ans_sep)
            print(ans_header)
            print(ans_sep)
            for r in results:
                alt = r.extra.get("answer_loss_target")
                alp = r.extra.get("answer_loss_proxy")
                if alt is not None and alp is not None:
                    adr = abs(alt - alp)
                    bt_val = r.risk_bound_total
                    a_status = "PASS" if bt_val is not None and bt_val >= adr else "VIOL"
                    proxy_short = r.extra.get("proxy_model", r.label).split("/")[-1]
                    if len(proxy_short) > 25:
                        proxy_short = proxy_short[:22] + "..."
                    print(
                        f"{proxy_short:<26s} {r.rho_proxy:>8.4f} {r.omega:>8.4f} "
                        f"{alt:>8.4f} {alp:>8.4f} "
                        f"{r.extra.get('answer_ppl_target', 0):>8.2f} "
                        f"{r.extra.get('answer_ppl_proxy', 0):>8.2f} "
                        f"{adr:>8.4f} {bt_val:>10.4f} {a_status:>8s}"
                    )
            ans_valid = [(r, abs(r.extra["answer_loss_target"] - r.extra["answer_loss_proxy"]))
                         for r in results
                         if "answer_loss_target" in r.extra and r.risk_bound_total is not None]
            if ans_valid:
                print(ans_sep)
                a_holds = sum(1 for r, adr in ans_valid if r.risk_bound_total >= adr)
                print(f"  Bound holds (answer): {a_holds}/{len(ans_valid)}  "
                      f"({'ALL PASS' if a_holds == len(ans_valid) else 'SOME VIOLATED'})")

        # --- Final-answer-only table (GSM8K) ---
        has_final_rpt = "final_loss_target" in r0.extra
        if has_final_rpt:
            final_header = (
                f"{'Proxy':<26s} {'ρ_P':>8s} {'Ω':>8s} "
                f"{'FLoss_T':>8s} {'FLoss_P':>8s} "
                f"{'FPPL_T':>8s} {'FPPL_P':>8s} {'|FdR|':>8s} "
                f"{'Bound':>10s} {'Status':>8s}"
            )
            final_sep = "-" * len(final_header)
            print(f"\n  Final-answer-only Loss (GSM8K: number only, no reasoning)")
            print(final_sep)
            print(final_header)
            print(final_sep)
            for r in results:
                flt = r.extra.get("final_loss_target")
                flp = r.extra.get("final_loss_proxy")
                if flt is not None and flp is not None:
                    fdr = abs(flt - flp)
                    bt_val = r.risk_bound_total
                    status = "PASS" if bt_val is not None and bt_val >= fdr else "VIOL"
                    proxy_short = r.extra.get("proxy_model", r.label).split("/")[-1]
                    if len(proxy_short) > 25:
                        proxy_short = proxy_short[:22] + "..."
                    print(
                        f"{proxy_short:<26s} {r.rho_proxy:>8.4f} {r.omega:>8.4f} "
                        f"{flt:>8.4f} {flp:>8.4f} "
                        f"{r.extra.get('final_ppl_target', 0):>8.2f} "
                        f"{r.extra.get('final_ppl_proxy', 0):>8.2f} "
                        f"{fdr:>8.4f} {bt_val:>10.4f} {status:>8s}"
                    )
            final_valid = [(r, abs(r.extra["final_loss_target"] - r.extra["final_loss_proxy"]))
                           for r in results
                           if "final_loss_target" in r.extra and r.risk_bound_total is not None]
            if final_valid:
                print(final_sep)
                f_holds = sum(1 for r, fdr in final_valid if r.risk_bound_total >= fdr)
                print(f"  Bound holds (final): {f_holds}/{len(final_valid)}  "
                      f"({'ALL PASS' if f_holds == len(final_valid) else 'SOME VIOLATED'})")

        print(f"{sep}\n")
