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
* **``awq:``** prefix (e.g. ``awq:TheBloke/Llama-2-7B-AWQ``) → load a
  pre-quantised AWQ model.  Optionally specify a branch with ``@``.

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
# AWQ helpers
# ======================================================================
def _is_awq(quant_tag: str) -> bool:
    return quant_tag.startswith("awq:")


def _parse_awq(quant_tag: str) -> tuple:
    """Parse ``awq:REPO`` or ``awq:REPO@REVISION`` → (repo, revision|None)."""
    body = quant_tag[4:]
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


def _proxy_model_name(
    quant_tag: str, target_model_id: str,
    quant_repo: str, gguf_tpl: str,
) -> str:
    """Full proxy model identifier for CSV/JSON output."""
    if _is_dtype(quant_tag):
        lbl = _DTYPE_LABELS.get(_parse_dtype(quant_tag), _parse_dtype(quant_tag).upper())
        return f"{target_model_id} [{lbl}]"
    if _is_gptq(quant_tag):
        repo, rev = _parse_gptq(quant_tag)
        return f"{repo}@{rev} [GPTQ]" if rev else f"{repo} [GPTQ]"
    if _is_awq(quant_tag):
        repo, rev = _parse_awq(quant_tag)
        return f"{repo}@{rev} [AWQ]" if rev else f"{repo} [AWQ]"
    if _is_bnb(quant_tag):
        return f"{target_model_id} [bnb:{_bnb_type(quant_tag)}]"
    return f"{quant_repo}/{_gguf_filename(gguf_tpl, quant_tag)}"


def _display_label(quant_tag: str) -> str:
    """Human-readable name for a quant tag."""
    if _is_bnb(quant_tag):
        return _bnb_type(quant_tag).upper()
    if _is_gptq(quant_tag):
        repo, rev = _parse_gptq(quant_tag)
        short = repo.split("/")[-1]
        return f"GPTQ({rev})" if rev else f"GPTQ({short})"
    if _is_awq(quant_tag):
        repo, rev = _parse_awq(quant_tag)
        short = repo.split("/")[-1]
        return f"AWQ({rev})" if rev else f"AWQ({short})"
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
        # GGUF dequantization materialises float32 tensors first; `dtype` may not
        # be applied until after the "Loading weights" progress bar completes.
        # Force the cast here so the proxy is stored in the target dtype (BF16)
        # rather than float32, cutting VRAM usage by ~16 GB for an 8B model.
        if next(proxy.parameters()).dtype != self.model_dtype:
            proxy = proxy.to(self.model_dtype)
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

    def _load_proxy_hf(
        self, repo: str, revision: str | None, backend: str,
    ) -> torch.nn.Module:
        """Load a pre-quantised GPTQ or AWQ model from HuggingFace."""
        rev_tag = f" @{revision}" if revision else ""
        print(f"  Loading proxy: {repo}{rev_tag} [{backend}] ...")
        kwargs: dict = dict(
            device_map=self.device,
            trust_remote_code=True,
            **self._attn_impl_kwargs(),
        )
        if revision:
            kwargs["revision"] = revision
        return _load_model(repo, **kwargs)

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

        # Disable GC during model loading: the GGUF dequantization loop
        # creates hundreds of large tensors, and Python's cyclic GC sweeps
        # over all tracked objects (including the already-instantiated model
        # graph).  This causes O(n_tensors * n_model_objects) overhead that
        # makes loading ~100× slower (22 s/tensor vs 0.3 s/tensor).
        _gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            if _is_dtype(quant_tag):
                proxy = self._load_proxy_dtype(_parse_dtype(quant_tag), target_model_id)
            elif _is_gptq(quant_tag):
                repo, rev = _parse_gptq(quant_tag)
                proxy = self._load_proxy_hf(repo, rev, "GPTQ")
            elif _is_awq(quant_tag):
                repo, rev = _parse_awq(quant_tag)
                proxy = self._load_proxy_hf(repo, rev, "AWQ")
            elif _is_bnb(quant_tag):
                proxy = self._load_proxy_bnb(_bnb_type(quant_tag), target_model_id)
            else:
                filename = _gguf_filename(gguf_tpl, quant_tag)
                proxy = self._load_proxy_gguf(quant_repo, filename)
        finally:
            if _gc_was_enabled:
                gc.enable()
        proxy.eval()
        return proxy

    # ------------------------------------------------------------------
    # Helpers for multi-z_mode paired-loss selection
    # ------------------------------------------------------------------
    @staticmethod
    def _select_paired_losses(zm, loss_mode, loss_stats):
        """Return the losses tensor correctly paired with Z for K_feat estimation."""
        if zm == "concat" and loss_stats["token_losses"] is not None:
            return loss_stats["token_losses"]
        if loss_mode == "answer" and loss_stats["has_answer_loss"]:
            return loss_stats["answer_losses"]
        return loss_stats["losses"]

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

        # --- Control Variate ablation config ---
        cfg_cv = self.config.get("cv_ablation", {})
        cv_cal_sizes: List[int] = cfg_cv.get("calibration_sizes", [])
        cv_n_trials: int = cfg_cv.get("n_trials", 20)

        # --- Dataset-specific Z extraction and loss modes ---
        from ..data.loaders import get_task_metadata
        task_meta = get_task_metadata(task_name)
        loss_mode = task_meta["loss_mode"]

        # z_modes resolution: explicit list > single override > all supported
        if cfg_data.get("z_modes"):
            z_modes = list(cfg_data["z_modes"])
        elif cfg_data.get("z_mode"):
            z_modes = [cfg_data["z_mode"]]
        else:
            z_modes = ["concat"]

        has_gguf = any(not _is_bnb(q) and not _is_gptq(q) and not _is_awq(q) and not _is_dtype(q) for q in quant_bits)
        has_awq = any(_is_awq(q) for q in quant_bits)
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
        if has_awq:
            awq_repos = set()
            for q in quant_bits:
                if _is_awq(q):
                    repo, _ = _parse_awq(q)
                    awq_repos.add(repo)
            print(f"  AWQ    : {', '.join(sorted(awq_repos))}")
        print(f"  Quants : {', '.join(_display_label(q) for q in quant_bits)}")
        print(f"  Data   : {task_name}  z_modes={z_modes}  loss_mode={loss_mode}")

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

        # --- Phase 1: load target, extract ALL z_modes in one forward pass ---
        target_dtype_label = _DTYPE_LABELS.get(str(self.model_dtype).split(".")[-1], str(self.model_dtype).upper())
        print(f"Loading target ({target_dtype_label}): {target_model_id} ...")
        target_model = _load_model(
            target_model_id, dtype=self.model_dtype, device_map=self.device,
            trust_remote_code=True,
            **self._attn_impl_kwargs(),
        )
        target_model.eval()
        extractor = LLMExtractor()

        print(f"Caching target features and loss (single forward pass, z_modes={z_modes}) ...")
        Z_dict_T, loss_stats = extractor.extract_features_and_loss_per_sample(
            target_model, dataloader, self.tensor_device,
            chunk_size=self.logit_chunk_size,
            z_modes=z_modes,
        )
        H_T = extractor.extract_head(target_model)

        # Compute primary loss based on loss_mode
        has_answer = loss_stats["has_answer_loss"]
        if loss_mode == "answer" and has_answer:
            loss_target = loss_stats["answer_losses"].mean().item()
        else:
            loss_target = loss_stats["losses"].mean().item()
        ppl_target = math.exp(loss_target)

        target_info = f"  Target: Loss={loss_target:.4f}  PPL={ppl_target:.2f}  (mode={loss_mode})"
        for zm in z_modes:
            target_info += f"  Z[{zm}]={tuple(Z_dict_T[zm].shape)}"
        target_info += f"  H={tuple(H_T.shape)}"
        print(target_info)

        del target_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  Target model freed from VRAM.")

        # --- Lipschitz constants per z_mode ---
        K_theory = UnifiedBound.theoretical_K(H_T)
        K_feat = K_theory["K_feat"]
        K_pred = K_theory["K_pred"]
        K_pred_empirical = loss_stats["grad_norm_p95"]

        rho_T_per_zm: Dict[str, float] = {}
        K_emp_per_zm: Dict[str, Dict] = {}
        lipschitz_per_zm: Dict[str, Dict] = {}

        for zm in z_modes:
            Z_T_zm = Z_dict_T[zm]
            rho_T = Z_T_zm.norm("fro").item() / math.sqrt(Z_T_zm.shape[0])
            rho_T_per_zm[zm] = rho_T
            paired = self._select_paired_losses(zm, loss_mode, loss_stats)
            K_emp = UnifiedBound.estimate_lipschitz_lm(Z_T_zm, paired)
            K_emp_per_zm[zm] = K_emp
            lipschitz_per_zm[zm] = {
                "K_feat_tight": K_feat,
                "K_feat_naive": K_theory["K_feat_naive"],
                "K_feat_empirical": K_emp["K_feat_empirical"],
                "K_feat_empirical_median": K_emp["median"],
                "K_feat_empirical_max": K_emp["max"],
                "K_pred_theory": K_pred,
                "K_pred_empirical": K_pred_empirical,
                "K_pred_empirical_mean": loss_stats["grad_norm_mean"],
                "K_pred_empirical_max": loss_stats["grad_norm_max"],
                "H_T_spectral": K_theory["H_T_spectral"],
                "max_pairwise_dist": K_theory["max_pairwise_dist"],
                "rho_T": rho_T,
            }

        print(f"\n  Lipschitz constants (shared: K_feat_tight={K_feat:.4f}, K_pred={K_pred:.4f}):")
        for zm in z_modes:
            rho_T = rho_T_per_zm[zm]
            K_emp = K_emp_per_zm[zm]
            print(f"    [{zm}]  ρ_T={rho_T:.6f}  K_feat_emp={K_emp['p95']:.4f}  "
                  f"(median={K_emp['median']:.4f}, max={K_emp['max']:.4f})")

        # Move cached target tensors to CPU so the full GPU is available for each
        # proxy model + its forward pass.  Procrustes / bound computations are pure
        # linear algebra and run fine on CPU.
        Z_dict_T = {k: v.cpu() for k, v in Z_dict_T.items()}
        H_T = H_T.cpu()
        for k, v in loss_stats.items():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                loss_stats[k] = v.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Phase 2: load each proxy one at a time ---
        results = []
        for i, bit_label in enumerate(quant_bits):
            disp = _display_label(bit_label)
            base_label = f"{target_dtype_label} vs {disp}"
            print(f"\n--- [{i+1}/{len(quant_bits)}] {base_label} ---")
            t0 = time.time()

            proxy_model = None
            try:
                proxy_model = self._load_proxy(bit_label, quant_repo, gguf_tpl, target_model_id)

                Z_dict_P, proxy_stats = extractor.extract_features_and_loss_per_sample(
                    proxy_model, dataloader, self.tensor_device,
                    chunk_size=self.logit_chunk_size,
                    z_modes=z_modes,
                )

                # Check all z_modes for NaN/Inf
                bad = [zm for zm in z_modes if not torch.isfinite(Z_dict_P[zm]).all()]
                if bad:
                    print(f"  WARNING: proxy features contain NaN/Inf in {bad} — skipping {disp}")
                    continue

                H_P = extractor.extract_head(proxy_model)

                # Move proxy features to CPU and free proxy model immediately so the
                # next GGUF load has the full GPU budget available.
                Z_dict_P = {k: v.cpu() for k, v in Z_dict_P.items()}
                H_P = H_P.cpu()
                for k, v in proxy_stats.items():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        proxy_stats[k] = v.cpu()
                del proxy_model
                proxy_model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Proxy loss (same for all z_modes — from same forward pass)
                if loss_mode == "answer" and has_answer:
                    loss_proxy = proxy_stats["answer_losses"].mean().item()
                else:
                    loss_proxy = proxy_stats["losses"].mean().item()
                ppl_proxy = math.exp(loss_proxy)

                # Proxy model label (shared across z_modes)
                proxy_model_name = _proxy_model_name(
                    bit_label, target_model_id, quant_repo, gguf_tpl)

                elapsed = time.time() - t0

                # --- One result per z_mode ---
                for zm in z_modes:
                    rho_T = rho_T_per_zm[zm]
                    K_emp = K_emp_per_zm[zm]

                    result = self.compute_metrics(
                        Z_dict_T[zm], H_T, Z_dict_P[zm], H_P,
                        force_identity=True, label=base_label, absorbed=absorbed,
                        K_feat=K_feat, K_pred=K_pred,
                    )
                    result.loss_target = loss_target
                    result.loss_proxy = loss_proxy
                    result.extra["perplexity_target"] = ppl_target
                    result.extra["perplexity_proxy"] = ppl_proxy
                    result.extra.update(lipschitz_per_zm[zm])
                    result.extra["dataset"] = task_name
                    result.extra["z_mode"] = zm
                    result.extra["loss_mode"] = loss_mode
                    result.extra["num_samples"] = num_samples
                    result.extra["target_model"] = target_model_id
                    result.extra["proxy_model"] = proxy_model_name
                    results.append(result)

                    # Console output
                    rho_P = result.rho_proxy
                    scale_s = "" if absorbed else f"Sc={result.scale_mismatch:.6f} "
                    bound_s = f"{result.risk_bound_total:.4f}" if result.risk_bound_total is not None else "—"
                    dr = abs(loss_target - loss_proxy)
                    status = ""
                    if result.risk_bound_total is not None:
                        status = "PASS" if result.risk_bound_total >= dr else "VIOLATED"

                    print(f"  [{zm:<20s}] ρ_T={rho_T:.4f} ρ_P={rho_P:.4f} Ω={result.omega:.4f} "
                          f"{scale_s}δ={result.shape_mismatch:.6f} γ={result.head_discrepancy:.6f} "
                          f"Bound={bound_s} |dR|={dr:.4f} {status} "
                          f"K_f_emp={K_emp['p95']:.4f}"
                          + (f"  ({elapsed:.1f}s)" if zm == z_modes[0] else ""))

                # --- Control Variate Ablation ---
                if cv_cal_sizes:
                    if loss_mode == "answer" and has_answer:
                        persample_T = loss_stats["answer_losses"]
                        persample_P = proxy_stats["answer_losses"]
                    else:
                        persample_T = loss_stats["losses"]
                        persample_P = proxy_stats["losses"]

                    N_full = persample_T.shape[0]
                    R_T_full = persample_T.mean().item()
                    R_P_full = persample_P.mean().item()

                    cv_results_for_proxy = []
                    for n_cal in cv_cal_sizes:
                        if n_cal >= N_full:
                            continue
                        errors = []
                        for trial in range(cv_n_trials):
                            rng = torch.Generator().manual_seed(self.seed + trial)
                            idx = torch.randperm(N_full, generator=rng)[:n_cal]
                            delta_hat = persample_T[idx].mean().item() - persample_P[idx].mean().item()
                            R_T_hat = R_P_full + delta_hat
                            errors.append(abs(R_T_hat - R_T_full))
                        mean_err = sum(errors) / len(errors)
                        std_err = (sum((e - mean_err) ** 2 for e in errors) / len(errors)) ** 0.5
                        cv_results_for_proxy.append({
                            "n_cal": n_cal,
                            "mean_abs_error": mean_err,
                            "std_abs_error": std_err,
                            "R_T_full": R_T_full,
                            "R_P_full": R_P_full,
                        })
                        print(f"    CV n_cal={n_cal:5d}: |R̂_T - R_T| = {mean_err:.6f} ± {std_err:.6f}")

                    # Store CV ablation in the first z_mode's result
                    if cv_results_for_proxy:
                        results[-len(z_modes)].extra["cv_ablation"] = cv_results_for_proxy

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  ERROR processing {disp}: {e}  ({elapsed:.1f}s) — skipping")

            finally:
                del proxy_model
                # Free proxy-iteration tensors to reclaim VRAM before next
                # proxy load.  With concat mode, stale Z_dict_P + H_P can
                # hold >6 GB on GPU, starving the next model load.
                Z_dict_P = None  # type: ignore[assignment]
                H_P = None       # type: ignore[assignment]
                proxy_stats = None  # type: ignore[assignment]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # --- Report and save per z_mode ---
        model_slug = target_model_id.split("/")[-1].lower()
        abs_tag = "_absorbed" if absorbed else ""
        for zm in z_modes:
            zm_results = [r for r in results if r.extra.get("z_mode") == zm]
            if not zm_results:
                continue
            print(f"\n{'=' * 72}")
            print(f"  z_mode: {zm}")
            print(f"{'=' * 72}")
            self.report(zm_results)
            stem = f"prism_{model_slug}_{task_name}_n{num_samples}_{zm}{abs_tag}"
            self.save(zm_results, filename=f"{stem}.json")
            self.save_csv(zm_results, filename=f"{stem}.csv", loss_mode=loss_mode)
        return results

    @staticmethod
    def report(results: list) -> None:
        """Print table with perplexity and |ΔR| columns.

        Automatically detects scale-absorbed mode and omits the Scale column.
        Prints Lipschitz constants in the header.
        Includes rho_P per row, z_mode/loss_mode info, and all loss tables.
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
                f"{'Label':<20s} {'ρ_T':>8s} {'ρ_P':>8s} {'Omega':>8s} {'Shape':>10s} "
                f"{'Head':>10s} {'Bound':>10s} {'|dR|':>8s} "
                f"{'Loss_T':>8s} {'Loss_P':>8s} {'PPL_T':>8s} {'PPL_P':>8s} "
                f"{'K_f':>8s} {kf_hdr:>8s} {'K_p':>8s} {kp_hdr:>8s}"
            )
        else:
            header = (
                f"{'Label':<20s} {'ρ_T':>8s} {'ρ_P':>8s} {'Omega':>8s} {'Scale':>10s} {'Shape':>10s} "
                f"{'Head':>10s} {'Bound':>10s} {'|dR|':>8s} "
                f"{'Loss_T':>8s} {'Loss_P':>8s} {'PPL_T':>8s} {'PPL_P':>8s} "
                f"{'K_f':>8s} {kf_hdr:>8s} {'K_p':>8s} {kp_hdr:>8s}"
            )

        mode_tag = " (scale-absorbed)" if absorbed else ""
        ds_tag = ""
        if "dataset" in r0.extra:
            z_m = r0.extra.get("z_mode", "?")
            l_m = r0.extra.get("loss_mode", "?")
            ds_tag = (f"  [data: {r0.extra['dataset']}, n={r0.extra.get('num_samples', '?')}, "
                      f"z_mode={z_m}, loss_mode={l_m}]")
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
            rho_t_s = f"{r.extra.get('rho_T', 0):.4f}" if "rho_T" in r.extra else "—"
            rho_p_s = f"{r.rho_proxy:.4f}"
            if absorbed:
                print(
                    f"{r.label:<20s} {rho_t_s:>8s} {rho_p_s:>8s} {r.omega:>8.4f} "
                    f"{r.shape_mismatch:>10.6f} {r.head_discrepancy:>10.6f} "
                    f"{bt:>10s} {dr_s:>8s} "
                    f"{lt:>8s} {lp:>8s} {ppl_t:>8s} {ppl_p:>8s} "
                    f"{kf:>8s} {kfe:>8s} {kp:>8s} {kpe:>8s}"
                )
            else:
                print(
                    f"{r.label:<20s} {rho_t_s:>8s} {rho_p_s:>8s} {r.omega:>8.4f} {r.scale_mismatch:>10.6f} "
                    f"{r.shape_mismatch:>10.6f} {r.head_discrepancy:>10.6f} "
                    f"{bt:>10s} {dr_s:>8s} "
                    f"{lt:>8s} {lp:>8s} {ppl_t:>8s} {ppl_p:>8s} "
                    f"{kf:>8s} {kfe:>8s} {kp:>8s} {kpe:>8s}"
                )

        # Bound validation
        valid = [(r, abs(r.loss_target - r.loss_proxy))
                 for r in results
                 if r.loss_target is not None and r.loss_proxy is not None and r.risk_bound_total is not None]
        if valid:
            print(sep)
            loss_mode = r0.extra.get("loss_mode", "full")
            holds = sum(1 for r, dr in valid if r.risk_bound_total >= dr)
            print(f"  Bound holds (loss_mode={loss_mode}): {holds}/{len(valid)}  "
                  f"({'ALL PASS' if holds == len(valid) else 'SOME VIOLATED'})")

        print(f"{'=' * len(header)}\n")
