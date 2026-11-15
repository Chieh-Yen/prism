"""
Quantization experiment: target against quantized proxies of the same
architecture.

Target = the model loaded at computing.model_dtype (bfloat16 in configs/quantization.yaml).
Proxy  = one quantized variant per entry of ``proxy.quantization_bits``.

Every comparison is evaluated under the identity alignment W = I and under
the Procrustes alignment (PRISMMetrics.compute_all).  With identical heads
and W = I the head term is zero and the bound is the feature term
    K_feat · √[ (ρ_T − ρ_P)² + 2 ρ_T ρ_P (1 − Ω) ].

Proxy kinds
-----------
The kind of a proxy is selected by the **prefix** of its quant tag:

* **No prefix** (e.g. ``Q8_0``, ``Q4_K_M``) → GGUF file of ``proxy.model``.
* **``bnb:``** prefix (e.g. ``bnb:nf4``, ``bnb:int8``) → the *target*
  checkpoint, quantized on the fly by bitsandbytes.
* **``gptq:``** prefix (e.g. ``gptq:ModelCloud/Meta-Llama-3.1-8B-gptq-4bit``)
  → a pre-quantised GPTQ repository, loaded at the revision pinned in
  ``configs/model_revisions.yaml``.  ``gptq:REPO@REVISION`` loads the given
  commit instead of the pinned one.
* **``dtype:``** prefix (e.g. ``dtype:float16``) → the target checkpoint
  loaded at another native dtype.

Supported ``bnb:`` tags: ``int8``, ``nf4``, ``fp4``.
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
from ..core.metrics import PRISMMetrics
from ..data.loaders import load_task_data
from ..models.extractors import LLMExtractor
from .base import BaseExperiment
import os
from pathlib import Path

# Remote code is opt-in (PRISM_TRUST_REMOTE_CODE=1). Every model family used in
# the paper is native to the pinned transformers release, so the default is off.
_TRUST_REMOTE_CODE = os.environ.get("PRISM_TRUST_REMOTE_CODE", "0") == "1"

# ----------------------------------------------------------------------
# Pinned revisions (configs/model_revisions.yaml): every HF repo listed there
# is loaded at a fixed commit so a re-run sees the same weights, tokenizer and
# config. A repo that is not listed is loaded at its head; pinned_revision
# prints which of the two applies, once per repository.
# ----------------------------------------------------------------------
_REVISIONS_PATH = Path(__file__).resolve().parents[2] / "configs" / "model_revisions.yaml"
_REVISIONS: Dict[str, str] | None = None
_REVISION_REPORTED: set = set()      # repositories already reported by pinned_revision


def pinned_revision(model_id: str) -> str | None:
    """Commit hash for ``model_id`` from configs/model_revisions.yaml, or None.

    The first time a repository is asked for, the outcome is printed:

      * listed in the file: one line on stdout with the revision used;
      * not pinned: a warning on stderr. This happens when PRISM_UNPINNED=1 is
        set, when configs/model_revisions.yaml is missing, or when
        ``model_id`` is not listed in it. The function then returns None; a
        caller that has no revision of its own then passes none, and the
        load uses the head of the repository (or a local path as it is).

    No exception is raised in either case, and the return value is the same
    as without the messages.
    """
    global _REVISIONS
    import sys

    def _report(revision, reason=None):
        if model_id in _REVISION_REPORTED:
            return
        _REVISION_REPORTED.add(model_id)
        if revision is not None:
            print(f"[revision] {model_id}: pinned revision {revision}")
        else:
            print(f"[revision] WARNING: no pinned revision for {model_id} "
                  f"({reason}); unless the caller passes one, the head of the "
                  f"repository is loaded.", file=sys.stderr)

    if os.environ.get("PRISM_UNPINNED", "0") == "1":
        _report(None, "PRISM_UNPINNED=1")
        return None
    if _REVISIONS is None:
        try:
            import yaml
            with open(_REVISIONS_PATH) as f:
                _REVISIONS = yaml.safe_load(f) or {}
        except FileNotFoundError:
            _REVISIONS = {}
    revision = _REVISIONS.get(model_id)
    if revision is not None:
        _report(revision)
    elif not _REVISIONS_PATH.exists():
        _report(None, f"{_REVISIONS_PATH} not found")
    else:
        _report(None, f"not listed in {_REVISIONS_PATH.name}")
    return revision


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
}


def _is_bnb(quant_tag: str) -> bool:
    return quant_tag.startswith("bnb:")


def _bnb_type(quant_tag: str) -> str:
    return quant_tag[4:]


# ======================================================================
# dtype helpers; same model weights, different native precision
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

    ``Org/{ModelName}-GGUF`` → files ``{ModelName}-{QUANT}.gguf`` (original
    case, dash separator).
    Example: ``Qwen/Qwen3-8B-GGUF`` → ``Qwen3-8B-{quant}.gguf``

    Override via ``proxy.gguf_template`` in the YAML config to skip this.
    """
    name = quant_repo.split("/")[-1]
    stem = re.sub(r"-GGUF$", "", name, flags=re.IGNORECASE)
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
    # An explicit revision (gptq:REPO@COMMIT) takes precedence over the pin.
    if "revision" not in kwargs:
        _rev = pinned_revision(model_id)
        if _rev is not None:
            kwargs["revision"] = _rev
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError as exc:
        if "Unrecognized configuration class" not in str(exc):
            raise
        _cfg_kwargs = {"revision": kwargs["revision"]} if "revision" in kwargs else {}
        cfg = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=kwargs.get("trust_remote_code", _TRUST_REMOTE_CODE),
            **_cfg_kwargs,
        )
        if not hasattr(cfg, "text_config"):
            raise
        print(
            f"  (VL model detected [{cfg.__class__.__name__}];"
            " loading text backbone via AutoModelForImageTextToText)"
        )
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)


class QuantizationExperiment(BaseExperiment):
    """Compare full-precision target with quantised variants (proxy).

    Both W = I (identity alignment) and W = W_N (Procrustes alignment)
    are computed; Bound is the smaller of Bound_I and Bound_W.
    """

    # ------------------------------------------------------------------
    # Proxy loading; dispatches between GGUF and bitsandbytes
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
            trust_remote_code=_TRUST_REMOTE_CODE,
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
        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=_TRUST_REMOTE_CODE,
            revision=pinned_revision(model_id))
        existing_quant = getattr(config, "quantization_config", None)

        if existing_quant is None:
            proxy = _load_model(
                model_id,
                quantization_config=_BNB_CONFIGS[bnb_tag](),
                device_map=self.device,
                trust_remote_code=_TRUST_REMOTE_CODE,
                **self._attn_impl_kwargs(),
            )
        else:
            proxy = _load_model(
                model_id,
                dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=_TRUST_REMOTE_CODE,
            )
            proxy = _bnb_requantize(proxy, _BNB_CONFIGS[bnb_tag](), self.device)
        return proxy

    def _load_proxy_hf(
        self, repo: str, revision: str | None, backend: str,
    ) -> torch.nn.Module:
        """Load a pre-quantised GPTQ model from HuggingFace."""
        revision = revision or pinned_revision(repo)
        rev_tag = f" @{revision}" if revision else ""
        print(f"  Loading proxy: {repo}{rev_tag} [{backend}] ...")
        kwargs: dict = dict(
            device_map=self.device,
            trust_remote_code=_TRUST_REMOTE_CODE,
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
            trust_remote_code=_TRUST_REMOTE_CODE,
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
            proxy = self._load_proxy_hf(repo, rev, "GPTQ")
        elif _is_bnb(quant_tag):
            proxy = self._load_proxy_bnb(_bnb_type(quant_tag), target_model_id)
        else:
            filename = _gguf_filename(gguf_tpl, quant_tag)
            proxy = self._load_proxy_gguf(quant_repo, filename)
        proxy.eval()
        return proxy

    @staticmethod
    def _check_token_rows(Z_dict, stats, who: str) -> None:
        """Check that Z and ``token_losses`` have the same number of rows."""
        tok = stats.get("token_losses")
        if "concat" not in Z_dict or tok is None:
            return
        n_z, n_tok = Z_dict["concat"].shape[0], tok.numel()
        if n_z != n_tok:
            raise RuntimeError(
                f"{who}: {n_z} feature rows but {n_tok} token losses.")

    def run(self) -> list:
        """Cache target features/loss, free VRAM, then load proxies one at a time."""
        cfg_target = self.config.get("target", {})
        cfg_proxy = self.config.get("proxy", {})
        cfg_data = self.config.get("data", {})

        target_model_id = cfg_target.get("model")
        if not target_model_id:
            raise ValueError("configuration key 'target.model' is missing")
        quant_bits: List[str] = cfg_proxy.get("quantization_bits", [
            "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K",
        ])
        # proxy.model is the GGUF repository; it is needed when a GGUF variant
        # (a tag without prefix) is requested.
        quant_repo = cfg_proxy.get("model")
        if not quant_repo:
            if any(not _is_bnb(q) and not _is_gptq(q)
                   and not _is_dtype(q) for q in quant_bits):
                raise ValueError(
                    "configuration key 'proxy.model' (the GGUF repository) is "
                    "missing, and proxy.quantization_bits lists GGUF variants")
            quant_repo = ""
        gguf_tpl: str = cfg_proxy.get("gguf_template") or _derive_gguf_template(quant_repo)

        task_name = cfg_data.get("task", "mmlu")
        num_samples = cfg_data.get("num_samples", 256)
        batch_size = cfg_data.get("batch_size", 8)
        max_length = cfg_data.get("max_length", 512)

        # --- Dataset-specific Z extraction and loss modes ---
        from ..data.loaders import get_task_metadata
        task_meta = get_task_metadata(task_name)
        loss_mode = task_meta["loss_mode"]

        # Feature mode of the task (TASK_REGISTRY: "concat" for every task).
        z_modes = [task_meta["z_mode"]]

        has_gguf = any(not _is_bnb(q) and not _is_gptq(q) and not _is_dtype(q) for q in quant_bits)
        has_bnb = any(_is_bnb(q) for q in quant_bits)
        has_gptq = any(_is_gptq(q) for q in quant_bits)
        has_dtype = any(_is_dtype(q) for q in quant_bits)

        print(f"{'=' * 72}")
        print("  PRISM Experiment: Quantization Quality Estimation")
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
        print(f"  Data   : {task_name}  z_modes={z_modes}  loss_mode={loss_mode}")

        print(f"Loading tokenizer from {target_model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            target_model_id, trust_remote_code=_TRUST_REMOTE_CODE,
            revision=pinned_revision(target_model_id))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # The feature and loss extraction assumes right padding (position t of
        # a sequence is column t of the batch).
        tokenizer.padding_side = "right"

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
            trust_remote_code=_TRUST_REMOTE_CODE,
            **self._attn_impl_kwargs(),
        )
        target_model.eval()
        extractor = LLMExtractor()

        print(f"Caching target features and loss (single forward pass, z_modes={z_modes}) ...")
        Z_dict_T, loss_stats = extractor.extract_features_and_loss_per_sample(
            target_model, dataloader, self.tensor_device,
            z_modes=z_modes,
        )
        self._check_token_rows(Z_dict_T, loss_stats, "target")
        H_T = extractor.extract_head(target_model)

        # Compute both full-text and answer-only losses (stored for CSV).
        has_answer = loss_stats["has_answer_loss"]
        full_loss_target = loss_stats["losses"].mean().item()
        full_ppl_target = math.exp(full_loss_target)
        if has_answer:
            answer_loss_target = loss_stats["answer_losses"].mean().item()
            answer_ppl_target = math.exp(answer_loss_target)
        else:
            answer_loss_target = answer_ppl_target = None

        if loss_mode == "answer" and has_answer:
            sample_loss_target = answer_loss_target
        else:
            sample_loss_target = full_loss_target
        _tok_T = loss_stats["token_losses"]
        token_loss_target = (_tok_T.mean().item()
                             if _tok_T is not None and _tok_T.numel() > 0
                             else sample_loss_target)
        ppl_target = math.exp(sample_loss_target)

        target_info = f"  Target: Loss={sample_loss_target:.8f}  PPL={ppl_target:.2f}  (mode={loss_mode})"
        for zm in z_modes:
            target_info += f"  Z[{zm}]={tuple(Z_dict_T[zm].shape)}"
        target_info += f"  H={tuple(H_T.shape)}"
        print(target_info)

        del target_model
        gc.collect()

        # theoretical_K benefits from GPU matmuls for large-vocab heads,
        # so compute it before moving H_T to CPU.
        K_theory = UnifiedBound.theoretical_K(H_T)
        K_feat = K_theory["K_feat"]
        K_pred = K_theory["K_pred"]

        # Move everything to CPU; full GPU is now free for proxy models.
        # Z_dict_T is already CPU (features moved in extractor).
        H_T = H_T.cpu()
        for k, v in loss_stats.items():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                loss_stats[k] = v.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  Target model freed; tensors moved to CPU.")

        # --- Lipschitz constants per z_mode (all on CPU) ---
        rho_T_per_zm: Dict[str, float] = {}
        lipschitz_per_zm: Dict[str, Dict] = {}

        for zm in z_modes:
            Z_T_zm = Z_dict_T[zm]
            rho_T = PRISMMetrics.rms_scale(Z_T_zm)
            rho_T_per_zm[zm] = rho_T
            lipschitz_per_zm[zm] = {
                "K_feat_tight": K_feat,
                "K_feat_naive": K_theory["K_feat_naive"],
                "K_pred_theory": K_pred,
                "H_T_spectral": K_theory["H_T_spectral"],
                "max_pairwise_dist": K_theory["max_pairwise_dist"],
                "rho_T": rho_T,
            }

        print(f"\n  Lipschitz constants:")
        print(f"    K_feat={K_feat:.8f}  K_pred={K_pred:.8f}")
        for zm in z_modes:
            print(f"    [{zm}]  ρ_T={rho_T_per_zm[zm]:.8f}")

        # --- Phase 2: load each proxy one at a time ---
        results = []
        skipped: List[str] = []     # variants that produced no result, with the reason
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
                    z_modes=z_modes,
                )
                self._check_token_rows(Z_dict_P, proxy_stats, disp)

                # Z_dict_P features are already on CPU from extractor.
                # Check all z_modes for NaN/Inf.
                bad = [zm for zm in z_modes if not torch.isfinite(Z_dict_P[zm]).all()]
                if bad:
                    print(f"  WARNING: proxy features contain NaN/Inf in {bad} — skipping {disp}")
                    skipped.append(f"{disp}: features contain NaN/Inf")
                    continue

                H_P = extractor.extract_head(proxy_model)

                # Free proxy model immediately; H_P is a detached copy.
                H_P = H_P.cpu()
                for k, v in proxy_stats.items():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        proxy_stats[k] = v.cpu()
                del proxy_model
                proxy_model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Proxy loss; both full-text and answer-only
                full_loss_proxy = proxy_stats["losses"].mean().item()
                full_ppl_proxy = math.exp(full_loss_proxy)
                if has_answer:
                    answer_loss_proxy = proxy_stats["answer_losses"].mean().item()
                    answer_ppl_proxy = math.exp(answer_loss_proxy)
                else:
                    answer_loss_proxy = answer_ppl_proxy = None

                if loss_mode == "answer" and has_answer:
                    sample_loss_proxy = answer_loss_proxy
                else:
                    sample_loss_proxy = full_loss_proxy
                _tok_P = proxy_stats["token_losses"]
                token_loss_proxy = (_tok_P.mean().item()
                                    if _tok_P is not None and _tok_P.numel() > 0
                                    else sample_loss_proxy)

                # Proxy model label (shared across z_modes)
                proxy_model_name = _proxy_model_name(
                    bit_label, target_model_id, quant_repo, gguf_tpl)

                elapsed = time.time() - t0

                # --- One result per z_mode (dual-W: W=I and W=opt) ---
                for zm in z_modes:
                    rho_T = rho_T_per_zm[zm]
                    Z_T_zm, Z_P_zm = Z_dict_T[zm], Z_dict_P[zm]
                    d = Z_T_zm.shape[1]
                    W_I = torch.eye(d, dtype=torch.float32)

                    # Compute metrics under both alignment choices
                    result_I = self.compute_metrics(
                        Z_T_zm, H_T, Z_P_zm, H_P,
                        W=W_I, label=base_label,
                        K_feat=K_feat, K_pred=K_pred,
                    )
                    result_W = self.compute_metrics(
                        Z_T_zm, H_T, Z_P_zm, H_P,
                        W=None, label=base_label,
                        K_feat=K_feat, K_pred=K_pred,
                    )

                    # Theoretical bounds (filled by fill_result inside compute_metrics)
                    bound_I_th = result_I.risk_bound_total
                    bound_W_th = result_W.risk_bound_total
                    bound_th = min(bound_I_th, bound_W_th)

                    # Primary result: the W with the smaller bound
                    result = result_I if bound_I_th <= bound_W_th else result_W
                    result.risk_bound_total = bound_th

                    if zm == "concat" and loss_stats["token_losses"] is not None:
                        zm_loss_t, zm_loss_p = token_loss_target, token_loss_proxy
                    else:
                        zm_loss_t, zm_loss_p = sample_loss_target, sample_loss_proxy
                    result.loss_target = zm_loss_t
                    result.loss_proxy = zm_loss_p
                    result.extra["perplexity_target"] = math.exp(zm_loss_t)
                    result.extra["perplexity_proxy"] = math.exp(zm_loss_p)

                    # Both loss variants for CSV
                    result.extra["full_loss_target"] = full_loss_target
                    result.extra["full_loss_proxy"] = full_loss_proxy
                    result.extra["full_ppl_target"] = full_ppl_target
                    result.extra["full_ppl_proxy"] = full_ppl_proxy
                    if answer_loss_target is not None:
                        result.extra["answer_loss_target"] = answer_loss_target
                        result.extra["answer_loss_proxy"] = answer_loss_proxy
                        result.extra["answer_ppl_target"] = answer_ppl_target
                        result.extra["answer_ppl_proxy"] = answer_ppl_proxy

                    # Dual-W metrics
                    result.extra["omega_I"] = result_I.omega
                    result.extra["omega_W"] = result_W.omega
                    result.extra["delta_I"] = result_I.feature_error
                    result.extra["delta_W"] = result_W.feature_error
                    result.extra["gamma_I"] = result_I.head_discrepancy
                    result.extra["gamma_W"] = result_W.head_discrepancy
                    result.extra["bound_I_th"] = bound_I_th
                    result.extra["bound_W_th"] = bound_W_th
                    result.extra["bound_th"] = bound_th

                    result.extra.update(lipschitz_per_zm[zm])
                    result.extra["dataset"] = task_name
                    result.extra["z_mode"] = zm
                    result.extra["loss_mode"] = loss_mode
                    result.extra["num_samples"] = num_samples
                    result.extra["target_model"] = target_model_id
                    result.extra["proxy_model"] = proxy_model_name
                    results.append(result)

                    # Console output; dual-W format
                    rho_P = result_I.rho_proxy
                    dr = abs(zm_loss_t - zm_loss_p)
                    print(f"  [{zm:<20s}] ρ_T={rho_T:.8f} ρ_P={rho_P:.8f}")
                    print(f"    W=I:   Ω={result_I.omega:.8f} δ={result_I.feature_error:.8f} "
                          f"γ={result_I.head_discrepancy:.8f}  "
                          f"Bnd={bound_I_th:.8f}")
                    print(f"    W=opt: Ω={result_W.omega:.8f} δ={result_W.feature_error:.8f} "
                          f"γ={result_W.head_discrepancy:.8f}  "
                          f"Bnd={bound_W_th:.8f}")
                    print(f"    → min  Bnd={bound_th:.8f} "
                          f"|dR|={dr:.8f}"
                          + (f"  ({elapsed:.1f}s)" if zm == z_modes[0] else ""))

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  ERROR processing {disp}: {e}  ({elapsed:.1f}s) — skipping")
                skipped.append(f"{disp}: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")

            finally:
                del proxy_model
                Z_dict_P = None  # type: ignore[assignment]
                H_P = None       # type: ignore[assignment]
                proxy_stats = None  # type: ignore[assignment]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # --- Report and save per z_mode ---
        model_slug = target_model_id.split("/")[-1].lower()
        for zm in z_modes:
            zm_results = [r for r in results if r.extra.get("z_mode") == zm]
            if not zm_results:
                continue
            print(f"\n{'=' * 72}")
            print(f"  z_mode: {zm}")
            print(f"{'=' * 72}")
            self.report(zm_results)
            stem = f"prism_{model_slug}_{task_name}_n{num_samples}_{zm}"
            self.save(zm_results, filename=f"{stem}.json")
            self.save_csv(zm_results, filename=f"{stem}.csv", loss_mode=loss_mode)

        # One line per variant that produced no result (the loader or the
        # extraction raised, or the features were not finite). The prefix
        # "[skipped]" is what run_quantization.sh counts.
        print(f"Variants with a result: {len(quant_bits) - len(skipped)} of {len(quant_bits)}")
        for item in skipped:
            print(f"[skipped] {target_model_id} | {task_name} | {item}")
        return results

    @staticmethod
    def report(results: list) -> None:
        """Print the dual-W summary table."""
        if not results:
            print("(no results)")
            return

        r0 = results[0]

        header = (
            f"{'Label':<20s} {'ρ_T':>13s} {'ρ_P':>13s} "
            f"{'Ω_I':>12s} {'Ω_W':>12s} "
            f"{'δ_I':>12s} {'δ_W':>12s} "
            f"{'γ_I':>12s} {'γ_W':>12s} "
            f"{'Bnd':>12s} "
            f"{'|dR|':>12s} "
            f"{'Loss_T':>12s} {'Loss_P':>12s}"
        )

        ds_tag = ""
        if "dataset" in r0.extra:
            z_m = r0.extra.get("z_mode", "?")
            l_m = r0.extra.get("loss_mode", "?")
            ds_tag = (f"  [data: {r0.extra['dataset']}, n={r0.extra.get('num_samples', '?')}, "
                      f"z_mode={z_m}, loss_mode={l_m}]")
        sep = "-" * len(header)
        print(f"\n{'=' * len(header)}")
        print(f"  PRISM Quantization Results{ds_tag}")
        print(f"{'=' * len(header)}")

        rho_T_val = r0.extra.get("rho_T")
        if rho_T_val is not None:
            print(f"  ρ_T = {rho_T_val:.8f}")

        if "K_feat_tight" in r0.extra:
            K_f = r0.extra["K_feat_tight"]
            K_fn = r0.extra.get("K_feat_naive")
            K_p = r0.extra.get("K_pred_theory")
            print(f"  K_feat:  {K_f:.8f} (tight)  "
                  + (f"{K_fn:.8f} (naive)" if K_fn is not None else ""))
            print(f"  K_pred:  {K_p:.8f}")
            print(sep)

        print(header)
        print(sep)

        for r in results:
            dr = abs(r.loss_target - r.loss_proxy) if r.loss_target is not None and r.loss_proxy is not None else None
            dr_s = f"{dr:.8f}" if dr is not None else "—"
            lt = f"{r.loss_target:.8f}" if r.loss_target is not None else "—"
            lp = f"{r.loss_proxy:.8f}" if r.loss_proxy is not None else "—"
            rho_t_s = f"{r.extra.get('rho_T', 0):.8f}" if "rho_T" in r.extra else "—"
            rho_p_s = f"{r.rho_proxy:.8f}"

            o_I = f"{r.extra.get('omega_I', r.omega):.8f}"
            o_W = f"{r.extra.get('omega_W', r.omega):.8f}"
            d_I = f"{r.extra.get('delta_I', r.feature_error):.8f}"
            d_W = f"{r.extra.get('delta_W', r.feature_error):.8f}"
            g_I = f"{r.extra.get('gamma_I', r.head_discrepancy):.8f}"
            g_W = f"{r.extra.get('gamma_W', r.head_discrepancy):.8f}"

            bnd_th = f"{r.extra.get('bound_th', r.risk_bound_total):.8f}" if r.risk_bound_total is not None else "—"

            print(
                f"{r.label:<20s} {rho_t_s:>13s} {rho_p_s:>13s} "
                f"{o_I:>12s} {o_W:>12s} "
                f"{d_I:>12s} {d_W:>12s} "
                f"{g_I:>12s} {g_W:>12s} "
                f"{bnd_th:>12s} "
                f"{dr_s:>12s} "
                f"{lt:>12s} {lp:>12s}"
            )

        print(f"{'=' * len(header)}\n")
