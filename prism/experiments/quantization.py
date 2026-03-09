"""
Quantization Quality Estimation — Identity Regime (W = I).

Target = full-precision model (BF16 by default; configurable via computing.model_dtype).
Proxy  = quantized model at each bit width.
The alignment map W degenerates to the identity.

The simplified bound (Eq. 6) becomes:
    |R_F − R_Q| ≈ K_feat · √[ (ρ_F − ρ_Q)² + 2 ρ_F ρ_Q (1 − Ω) ]

Quantisation backends
---------------------
Three backends are supported, selected by a **prefix** in each quant tag:

* **No prefix** (e.g. ``Q8_0``, ``Q4_K_M``) → load from a GGUF repo.
* **``bnb:``** prefix (e.g. ``bnb:nf4``, ``bnb:int8``) → load from the
  *target* model with bitsandbytes on-the-fly quantisation.
* **``gptq:``** prefix (e.g. ``gptq:TheBloke/Llama-2-7B-GPTQ``) → load a
  pre-quantised GPTQ model.  Optionally specify a branch with ``@``:
  ``gptq:TheBloke/Llama-2-7B-GPTQ@gptq-4bit-32g-actorder_True``.

Supported ``bnb:`` tags: ``int8``, ``nf4``, ``fp4``, ``nf4-dq``
(NF4 with double quantisation, aka QLoRA-style).
"""

from __future__ import annotations

import gc
import math
import re
import time
from typing import Any, Dict, List

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from ..core.bounds import UnifiedBound
from ..data.loaders import load_task_data
from ..models.extractors import LLMExtractor
from .base import BaseExperiment


# ======================================================================
# BitsAndBytes config factory
# ======================================================================
_BNB_CONFIGS: Dict[str, Any] = {
    "int8": lambda: BitsAndBytesConfig(load_in_8bit=True),
    "nf4": lambda: BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ),
    "fp4": lambda: BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.float16,
    ),
    "nf4-dq": lambda: BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    ),
}


def _is_bnb(quant_tag: str) -> bool:
    return quant_tag.startswith("bnb:")


def _bnb_type(quant_tag: str) -> str:
    return quant_tag[4:]


# ======================================================================
# dtype helpers — same model weights, different native precision
# ======================================================================
_DTYPE_LABELS: Dict[str, str] = {
    "float16":  "FP16",
    "bfloat16": "BF16",
    "float32":  "FP32",
}


def _is_dtype(quant_tag: str) -> bool:
    return quant_tag.startswith("dtype:")


def _parse_dtype(quant_tag: str) -> str:
    return quant_tag[6:]   # e.g. "float16"


# ======================================================================
# GPTQ helpers
# ======================================================================
def _is_gptq(quant_tag: str) -> bool:
    return quant_tag.startswith("gptq:")


def _parse_gptq(quant_tag: str) -> tuple:
    """Parse ``gptq:REPO`` or ``gptq:REPO@REVISION`` → (repo, revision|None)."""
    body = quant_tag[5:]
    if "@" in body:
        repo, rev = body.rsplit("@", 1)
        return repo, rev
    return body, None


# ======================================================================
# GGUF helpers
# ======================================================================
def _gguf_filename(template: str, quant: str) -> str:
    """Build a GGUF filename from a template and a quantisation tag."""
    return template.format(quant=quant)


def _derive_gguf_template(quant_repo: str) -> str:
    """Auto-derive a GGUF filename template from a HuggingFace repo name.

    Two conventions are handled:

    1. **TheBloke** — ``TheBloke/{ModelName}-GGUF``
       files:  ``{model-name}.{QUANT}.gguf``  (lowercased, dot separator)
       Example: ``TheBloke/Llama-2-7b-GGUF`` → ``llama-2-7b.{quant}.gguf``

    2. **Official** — ``Org/{ModelName}-GGUF``
       files:  ``{ModelName}-{QUANT}.gguf``  (original case, dash separator)
       Example: ``Qwen/Qwen3-8B-GGUF`` → ``Qwen3-8B-{quant}.gguf``

    Override via ``proxy.gguf_template`` in the YAML config to skip this.
    """
    name = quant_repo.split("/")[-1]
    stem = re.sub(r"-GGUF$", "", name, flags=re.IGNORECASE)
    org = quant_repo.split("/")[0] if "/" in quant_repo else ""
    if org.lower() == "thebloke":
        return f"{stem.lower()}.{{quant}}.gguf"
    return f"{stem}-{{quant}}.gguf"


def _display_label(quant_tag: str) -> str:
    """Human-readable name for a quant tag."""
    if _is_bnb(quant_tag):
        return _bnb_type(quant_tag).upper()
    if _is_gptq(quant_tag):
        repo, rev = _parse_gptq(quant_tag)
        short = repo.split("/")[-1]
        return f"GPTQ({rev})" if rev else f"GPTQ({short})"
    if _is_dtype(quant_tag):
        dt = _parse_dtype(quant_tag)
        return _DTYPE_LABELS.get(dt, dt.upper())
    return quant_tag


def _bnb_requantize(
    model: torch.nn.Module,
    bnb_config: BitsAndBytesConfig,
    device: str,
) -> torch.nn.Module:
    """Replace ``nn.Linear`` layers with BnB-quantised equivalents in-place.

    Intended for checkpoints that carry a pre-applied quantisation (e.g.
    ``FineGrainedFP8Config``) which ``from_pretrained`` cannot combine with a
    ``BitsAndBytesConfig`` in a single call.

    The caller must have already loaded the model to CPU with
    ``dtype=bfloat16``.  On hardware that does not support the checkpoint's
    native format (e.g. FP8 on a GPU with compute capability < 8.9),
    transformers automatically dequantises the weights to BF16, so all
    ``nn.Linear`` layers already hold regular BF16 tensors at this point.

    After in-place layer replacement the model is dispatched to *device*,
    which triggers bitsandbytes' on-CUDA weight quantisation via the
    ``Int8Params.to()`` / ``Params4bit.to()`` overrides.
    """
    import bitsandbytes as bnb
    from transformers.integrations.bitsandbytes import should_convert_module

    skip = getattr(bnb_config, "llm_int8_skip_modules", None) or ["lm_head"]
    is_int8 = bool(bnb_config.load_in_8bit)
    replaced = 0

    for module_name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if not should_convert_module(module_name, skip):
            continue

        w = module.weight.data          # BF16, CPU
        b = module.bias.data if module.bias is not None else None

        if is_int8:
            new = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=b is not None,
                has_fp16_weights=bnb_config.llm_int8_has_fp16_weight,
                threshold=bnb_config.llm_int8_threshold,
            )
            new.weight = bnb.nn.Int8Params(
                w,
                requires_grad=False,
                has_fp16_weights=bnb_config.llm_int8_has_fp16_weight,
            )
        else:
            new = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=b is not None,
                compute_dtype=bnb_config.bnb_4bit_compute_dtype,
                compress_statistics=bnb_config.bnb_4bit_use_double_quant,
                quant_type=bnb_config.bnb_4bit_quant_type,
            )
            new.weight = bnb.nn.Params4bit(
                w,
                requires_grad=False,
                quant_type=bnb_config.bnb_4bit_quant_type,
                compress_statistics=bnb_config.bnb_4bit_use_double_quant,
            )

        if b is not None:
            new.bias = torch.nn.Parameter(b, requires_grad=False)
        new.source_cls = type(module)
        new.requires_grad_(False)
        model.set_submodule(module_name, new)
        replaced += 1

    print(f"  In-place BnB replacement: {replaced} Linear → {'Int8' if is_int8 else '4bit'} layers.")
    model.to(device)    # triggers Int8Params.to() / Params4bit.to() → BnB quantisation
    return model


def _load_model(model_id: str, **kwargs) -> torch.nn.Module:
    """Load a model for causal-LM tasks.

    Primary path: ``AutoModelForCausalLM`` (pure text models).
    Fallback: ``AutoModelForImageTextToText`` for vision-language models
    whose config (e.g. ``Mistral3Config``) is not registered under
    ``AutoModelForCausalLM``.  The ``LLMExtractor`` handles VL model
    structures transparently by navigating to the text backbone.
    """
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError as exc:
        if "Unrecognized configuration class" not in str(exc):
            raise
        cfg = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=kwargs.get("trust_remote_code", True),
        )
        if not hasattr(cfg, "text_config"):
            raise
        print(
            f"  (VL model detected [{cfg.__class__.__name__}];"
            " loading text backbone via AutoModelForImageTextToText)"
        )
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)


class QuantizationExperiment(BaseExperiment):
    """Compare full-precision target with quantised variants (proxy), W = I."""

    def setup_pairs(self) -> List[Dict[str, Any]]:
        """Not used directly — see ``run()``."""
        return []

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
            import flash_attn  # noqa: F401 — availability probe only
        except ImportError:
            print("  [flash_attn] package not found — running without Flash Attention 2.")
            return {}
        if self.model_dtype == torch.float32:
            print("  [flash_attn] skipped — flash_attention_2 requires fp16/bf16, not fp32.")
            return {}
        return {"attn_implementation": "flash_attention_2"}

    # ------------------------------------------------------------------
    # Proxy loading — dispatches between GGUF and bitsandbytes
    # ------------------------------------------------------------------
    def _load_proxy_gguf(
        self, quant_repo: str, filename: str,
    ) -> torch.nn.Module:
        """Load a GGUF-quantised proxy."""
        print(f"  Loading proxy: {filename} from {quant_repo} ...")
        proxy = _load_model(
            quant_repo,
            gguf_file=filename,
            dtype=self.model_dtype,
            device_map=self.device,
            trust_remote_code=True,
            **self._attn_impl_kwargs(),
        )
        return proxy

    def _load_proxy_bnb(
        self, bnb_tag: str, model_id: str,
    ) -> torch.nn.Module:
        """Load a bitsandbytes-quantised proxy from the original model."""
        if bnb_tag not in _BNB_CONFIGS:
            raise ValueError(
                f"Unknown bnb quant type '{bnb_tag}'. "
                f"Available: {sorted(_BNB_CONFIGS)}"
            )
        print(f"  Loading proxy: {model_id} [bnb:{bnb_tag}] ...")
        # Pre-load config to detect whether the checkpoint already has its own
        # quantisation (e.g. FineGrainedFP8Config for Ministral-3-8B-Instruct).
        # Passing a BitsAndBytesConfig alongside a different quantisation class
        # raises a conflict error in transformers.  When a pre-applied config is
        # detected we use a two-step path:
        #   1. Load to CPU with dtype=bfloat16 — transformers automatically
        #      dequantises the weights to BF16 on hardware that does not support
        #      the native format (e.g. FP8 on compute capability < 8.9).
        #   2. Replace nn.Linear layers with BnB-quantised equivalents in-place
        #      via _bnb_requantize, then dispatch to GPU (triggers on-CUDA
        #      BnB weight quantisation through Int8Params / Params4bit).
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        existing_quant = getattr(config, "quantization_config", None)

        if existing_quant is None:
            proxy = _load_model(
                model_id,
                quantization_config=_BNB_CONFIGS[bnb_tag](),
                device_map=self.device,
                trust_remote_code=True,
                **self._attn_impl_kwargs(),
            )
        else:
            print(
                f"  (pre-quantised checkpoint [{type(existing_quant).__name__}];"
                f" loading to CPU for dequantisation, then re-quantising with BnB)"
            )
            proxy = _load_model(
                model_id,
                dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
            )
            proxy = _bnb_requantize(proxy, _BNB_CONFIGS[bnb_tag](), self.device)
        return proxy

    def _load_proxy_gptq(
        self, repo: str, revision: str | None,
    ) -> torch.nn.Module:
        """Load a pre-quantised GPTQ model from HuggingFace."""
        rev_tag = f" @{revision}" if revision else ""
        print(f"  Loading proxy: {repo}{rev_tag} [GPTQ] ...")
        kwargs: dict = dict(
            device_map=self.device,
            trust_remote_code=True,
            **self._attn_impl_kwargs(),
        )
        if revision:
            kwargs["revision"] = revision
        proxy = _load_model(repo, **kwargs)
        return proxy

    def _load_proxy_dtype(
        self, dtype_str: str, model_id: str,
    ) -> torch.nn.Module:
        """Load target model weights at a different native dtype (precision-only proxy).

        E.g. ``dtype:float16`` loads the same checkpoint the target was loaded from
        but casts to FP16, letting the experiment measure the BF16→FP16 gap as a
        reference point against heavier quantisation tiers.
        """
        dtype = getattr(torch, dtype_str)
        label = _DTYPE_LABELS.get(dtype_str, dtype_str.upper())
        print(f"  Loading proxy: {model_id} [{label}] ...")
        proxy = _load_model(
            model_id,
            dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
            **self._attn_impl_kwargs(),
        )
        return proxy

    def _load_proxy(
        self,
        quant_tag: str,
        quant_repo: str,
        gguf_tpl: str,
        target_model_id: str,
    ) -> torch.nn.Module:
        """Dispatch proxy loading based on quant tag prefix."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if _is_dtype(quant_tag):
            proxy = self._load_proxy_dtype(_parse_dtype(quant_tag), target_model_id)
        elif _is_gptq(quant_tag):
            repo, rev = _parse_gptq(quant_tag)
            proxy = self._load_proxy_gptq(repo, rev)
        elif _is_bnb(quant_tag):
            proxy = self._load_proxy_bnb(_bnb_type(quant_tag), target_model_id)
        else:
            filename = _gguf_filename(gguf_tpl, quant_tag)
            proxy = self._load_proxy_gguf(quant_repo, filename)
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
        quant_bits: List[str] = cfg_proxy.get("quantization_bits", [
            "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K",
        ])
        gguf_tpl: str = cfg_proxy.get("gguf_template") or _derive_gguf_template(quant_repo)
        absorbed: bool = cfg_align.get("scale_absorbed", False)

        task_name = cfg_data.get("task", "wikitext")
        num_samples = cfg_data.get("num_samples", 256)
        batch_size = cfg_data.get("batch_size", 8)
        max_length = cfg_data.get("max_length", 512)

        has_gguf = any(not _is_bnb(q) and not _is_gptq(q) and not _is_dtype(q) for q in quant_bits)
        has_bnb = any(_is_bnb(q) for q in quant_bits)
        has_gptq = any(_is_gptq(q) for q in quant_bits)
        has_dtype = any(_is_dtype(q) for q in quant_bits)

        mode_tag = " (scale-absorbed)" if absorbed else ""
        print(f"{'=' * 72}")
        print(f"  PRISM Experiment: Quantization Quality Estimation{mode_tag}")
        print(f"{'=' * 72}")
        print(f"  Target : {target_model_id}  [{_DTYPE_LABELS.get(str(self.model_dtype).split('.')[-1], str(self.model_dtype))}]")
        if has_dtype:
            dtype_tags = [_parse_dtype(q) for q in quant_bits if _is_dtype(q)]
            print(f"  Dtype  : {', '.join(_DTYPE_LABELS.get(d, d.upper()) for d in dtype_tags)}")
        if has_gguf:
            print(f"  GGUF   : {quant_repo}  (template: {gguf_tpl})")
        if has_bnb:
            bnb_tags = [_bnb_type(q) for q in quant_bits if _is_bnb(q)]
            print(f"  BnB    : {', '.join(bnb_tags)}")
        if has_gptq:
            gptq_repos = set()
            for q in quant_bits:
                if _is_gptq(q):
                    repo, _ = _parse_gptq(q)
                    gptq_repos.add(repo)
            print(f"  GPTQ   : {', '.join(sorted(gptq_repos))}")
        print(f"  Quants : {', '.join(_display_label(q) for q in quant_bits)}")

        print(f"Loading tokenizer from {target_model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(target_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading data: {task_name} (n={num_samples}) ...")
        dataloader = load_task_data(
            task_name, split="test", num_samples=num_samples,
            batch_size=batch_size, tokenizer=tokenizer, max_length=max_length,
            seed=self.seed,
        )

        # --- Phase 1: load target, extract everything, then free VRAM ---
        target_dtype_label = _DTYPE_LABELS.get(str(self.model_dtype).split(".")[-1], str(self.model_dtype).upper())
        print(f"Loading target ({target_dtype_label}): {target_model_id} ...")
        target_model = _load_model(
            target_model_id, dtype=self.model_dtype, device_map=self.device,
            trust_remote_code=True,
            **self._attn_impl_kwargs(),
        )
        target_model.eval()
        extractor = LLMExtractor(offload_to_cpu=self.offload_to_cpu)

        print("Caching target features and loss (single forward pass) ...")
        Z_T, loss_stats = extractor.extract_features_and_loss_per_sample(
            target_model, dataloader, self.tensor_device,
            chunk_size=self.logit_chunk_size,
        )
        H_T = extractor.extract_head(target_model)
        losses_T = loss_stats["losses"]
        loss_target = losses_T.mean().item()
        ppl_target = math.exp(loss_target)
        has_answer = loss_stats["has_answer_loss"]
        aloss_target = loss_stats["answer_losses"].mean().item() if has_answer else None
        appl_target = math.exp(aloss_target) if aloss_target is not None else None
        target_info = (f"  Target: Loss={loss_target:.4f}  PPL={ppl_target:.2f}")
        if aloss_target is not None:
            target_info += f"  ALoss={aloss_target:.4f}  APPL={appl_target:.2f}"
        target_info += f"  Z={tuple(Z_T.shape)}  H={tuple(H_T.shape)}"
        print(target_info)

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
            disp = _display_label(bit_label)
            label = f"{target_dtype_label} vs {disp}"
            print(f"\n--- [{i+1}/{len(quant_bits)}] {label} ---")
            t0 = time.time()

            proxy_model = None
            try:
                proxy_model = self._load_proxy(bit_label, quant_repo, gguf_tpl, target_model_id)

                Z_P, proxy_stats = extractor.extract_features_and_loss_per_sample(
                    proxy_model, dataloader, self.tensor_device,
                    chunk_size=self.logit_chunk_size,
                )

                if not torch.isfinite(Z_P).all():
                    print(f"  WARNING: proxy features contain NaN/Inf — skipping {disp}")
                    continue

                H_P = extractor.extract_head(proxy_model)

                result = self.compute_metrics(
                    Z_T, H_T, Z_P, H_P,
                    force_identity=True, label=label, absorbed=absorbed,
                    K_feat=K_feat, K_pred=K_pred,
                )
                result.loss_target = loss_target
                result.loss_proxy = proxy_stats["losses"].mean().item()
                result.extra["perplexity_target"] = ppl_target
                result.extra["perplexity_proxy"] = math.exp(result.loss_proxy)
                if has_answer:
                    aloss_proxy = proxy_stats["answer_losses"].mean().item()
                    result.extra["answer_loss_target"] = aloss_target
                    result.extra["answer_loss_proxy"] = aloss_proxy
                    result.extra["answer_ppl_target"] = appl_target
                    result.extra["answer_ppl_proxy"] = math.exp(aloss_proxy)
                result.extra.update(lipschitz_info)
                result.extra["dataset"] = task_name
                result.extra["num_samples"] = num_samples
                result.extra["target_model"] = target_model_id
                if _is_dtype(bit_label):
                    lbl = _DTYPE_LABELS.get(_parse_dtype(bit_label), _parse_dtype(bit_label).upper())
                    result.extra["proxy_model"] = f"{target_model_id} [{lbl}]"
                elif _is_gptq(bit_label):
                    repo, rev = _parse_gptq(bit_label)
                    rev_tag = f"@{rev}" if rev else ""
                    result.extra["proxy_model"] = f"{repo}{rev_tag} [GPTQ]"
                elif _is_bnb(bit_label):
                    result.extra["proxy_model"] = f"{target_model_id} [bnb:{_bnb_type(bit_label)}]"
                else:
                    result.extra["proxy_model"] = f"{quant_repo}/{_gguf_filename(gguf_tpl, bit_label)}"

                results.append(result)

                dr = abs(result.loss_target - result.loss_proxy)
                ppl_p = result.extra["perplexity_proxy"]
                elapsed = time.time() - t0
                scale_s = "" if absorbed else f"Scale={result.scale_mismatch:.6f}  "
                print(f"  [Full]   ρ_T={rho_T_orig:.4f}  Ω={result.omega:.4f}  {scale_s}"
                      f"Shape={result.shape_mismatch:.6f}  Head={result.head_discrepancy:.6f}  "
                      f"Bound={result.risk_bound_total:.4f}  |dR|={dr:.4f}  "
                      f"Loss_T={loss_target:.4f}  Loss_P={result.loss_proxy:.4f}  "
                      f"PPL_T={ppl_target:.2f}  PPL_P={ppl_p:.2f}  "
                      f"K_f={K_feat:.4f}({K_empirical_feat['p95']:.4f})  "
                      f"K_p={K_pred:.4f}({K_pred_empirical:.4f})  ({elapsed:.1f}s)")
                if has_answer:
                    adr = abs(aloss_target - aloss_proxy)
                    print(f"  [Answer] |AdR|={adr:.4f}  Bound={result.risk_bound_total:.4f}  "
                          f"{'PASS' if result.risk_bound_total >= adr else 'VIOLATED'}  "
                          f"ALoss_T={aloss_target:.4f}  ALoss_P={aloss_proxy:.4f}  "
                          f"APPL_T={appl_target:.2f}  APPL_P={math.exp(aloss_proxy):.2f}")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  ERROR processing {disp}: {e}  ({elapsed:.1f}s) — skipping")

            finally:
                del proxy_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.report(results)
        model_slug = target_model_id.split("/")[-1].lower()
        abs_tag = "_absorbed" if absorbed else ""
        stem = f"prism_{model_slug}_{task_name}_n{num_samples}{abs_tag}"
        self.save(results, filename=f"{stem}.json")
        self.save_csv(results, filename=f"{stem}_qa.csv", loss_mode="full")
        if has_answer:
            self.save_csv(results, filename=f"{stem}_ans.csv", loss_mode="answer")
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
            print(f"  Bound holds (full): {holds}/{len(valid)}  "
                  f"({'ALL PASS' if holds == len(valid) else 'SOME VIOLATED'})")

        # --- Answer-only table (if available) ---
        has_ans = "answer_loss_target" in r0.extra
        if has_ans:
            ans_header = (
                f"{'Label':<20s} {'ALoss_T':>8s} {'ALoss_P':>8s} "
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
                    status = "PASS" if bt_val is not None and bt_val >= adr else "VIOLATED"
                    print(
                        f"{r.label:<20s} {alt:>8.4f} {alp:>8.4f} "
                        f"{r.extra.get('answer_ppl_target', 0):>8.2f} "
                        f"{r.extra.get('answer_ppl_proxy', 0):>8.2f} "
                        f"{adr:>8.4f} "
                        f"{bt_val:>10.4f} {status:>8s}"
                    )
            ans_valid = [(r, abs(r.extra["answer_loss_target"] - r.extra["answer_loss_proxy"]))
                         for r in results
                         if "answer_loss_target" in r.extra and r.risk_bound_total is not None]
            if ans_valid:
                print(ans_sep)
                a_holds = sum(1 for r, adr in ans_valid if r.risk_bound_total >= adr)
                print(f"  Bound holds (answer): {a_holds}/{len(ans_valid)}  "
                      f"({'ALL PASS' if a_holds == len(ans_valid) else 'SOME VIOLATED'})")

        print(f"{'=' * len(header)}\n")
