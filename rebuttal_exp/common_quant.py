"""
Shared helpers for the rebuttal-window quantization-side experiments (E1, E3).

Reuses the repo's own loading conventions:
  * variant list is parsed from exp_result/quantization/quantization_merged_slim.csv
    (`proxy_model` strings written by prism/experiments/quantization.py), so the
    exact same proxies as the paper are reloaded — no re-specification drift;
  * proxy loading mirrors QuantizationExperiment._load_proxy_* and imports its
    _BNB_CONFIGS / _load_model directly.

GPU-side module: requires torch / transformers (run on the 5090 box).
"""

from __future__ import annotations

import csv
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from prism.experiments.quantization import _BNB_CONFIGS, _load_model   # noqa: E402
from prism.models.extractors import LLMExtractor                       # noqa: E402

CSV_PATH = REPO / "exp_result" / "quantization" / "quantization_merged_slim.csv"

# CSV target_model strings for the two main-text families.
FAMILIES = {
    "llama": "meta-llama/Meta-Llama-3.1-8B",
    "qwen": "Qwen/Qwen3-8B-Base",
}

BENCHMARKS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]


# ----------------------------------------------------------------------
# Variant list from the paper's own result CSV
# ----------------------------------------------------------------------
def parse_proxy_spec(proxy_model: str, label: str, target_model: str) -> Dict:
    """Turn a CSV `proxy_model` string into a loader spec.

    Formats produced by the quantization experiment:
      '<repo>/<file>.gguf'      -> gguf
      '<repo> [bnb:nf4]'        -> bitsandbytes
      '<repo> [GPTQ]'           -> pre-quantised HF repo
      '<repo> [FP16]'           -> dtype-only proxy of the target checkpoint
    """
    s = proxy_model.strip()
    if s.endswith(".gguf"):
        repo, filename = s.rsplit("/", 1)
        return {"kind": "gguf", "repo": repo, "file": filename, "label": label}
    if s.endswith("]") and "[" in s:
        repo, tag = s.rsplit("[", 1)
        repo, tag = repo.strip(), tag.rstrip("]").strip()
        if tag.lower().startswith("bnb:"):
            return {"kind": "bnb", "repo": repo, "tag": tag.split(":", 1)[1],
                    "label": label}
        if tag.upper() == "GPTQ":
            return {"kind": "gptq", "repo": repo, "label": label}
        if tag.upper() in ("FP16", "FLOAT16"):
            return {"kind": "dtype", "repo": repo, "dtype": "float16",
                    "label": label}
    raise ValueError(f"Unrecognised proxy_model format: {proxy_model!r}")


def variants_from_csv(family: str) -> List[Dict]:
    """Distinct proxies of a family, in the CSV's own label set."""
    target = FAMILIES[family]
    seen, specs = set(), []
    for r in csv.DictReader(open(CSV_PATH)):
        if r["target_model"] != target:
            continue
        key = (r["proxy_model"], r["Label"])
        if key in seen:
            continue
        seen.add(key)
        specs.append(parse_proxy_spec(r["proxy_model"], r["Label"], target))
    return specs


def risk_gaps_from_csv(family: str) -> Dict[tuple, float]:
    """(Label, dataset) -> |MdR| for the family (empirical risk gaps)."""
    target = FAMILIES[family]
    out = {}
    for r in csv.DictReader(open(CSV_PATH)):
        if r["target_model"] == target:
            try:
                out[(r["Label"], r["dataset"])] = float(r["|MdR|"])
            except ValueError:
                pass
    return out


# ----------------------------------------------------------------------
# Proxy loading (mirrors QuantizationExperiment._load_proxy_*)
# ----------------------------------------------------------------------
def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_target(model_id: str, device: str = "cuda:0"):
    return _load_model(
        model_id, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    ).eval()


def _load_gptq_with_fallback(repo: str, device: str):
    """GPTQ repos with desc_act=True crash on gptqmodel>=7:
    `type object 'BACKEND' has no attribute 'EXLLAMA_V1'` — optimum still
    maps legacy exllama configs to an enum member that gptqmodel 7.x
    removed. Downgrading gptqmodel is unreliable (no wheels for new
    torch/python). Instead, override the kernel choice at load time via
    GPTQConfig; transformers merges it over the checkpoint's own config.
    (bits=4 matches every GPTQ repo in the paper's variant manifest.)"""
    from transformers import GPTQConfig                    # lazy import
    attempts = [
        ("as-is", None),
        ("exllama_v2", dict(bits=4, exllama_config={"version": 2})),
        ("no_exllama", dict(bits=4, use_exllama=False)),
    ]
    last_exc = None
    for tag, qc_kwargs in attempts:
        try:
            kwargs = dict(device_map=device, trust_remote_code=True)
            if qc_kwargs is not None:
                print(f"  [gptq-fallback] retrying {repo} with {tag}")
                kwargs["quantization_config"] = GPTQConfig(**qc_kwargs)
            return _load_model(repo, **kwargs)
        except Exception as exc:                           # noqa: BLE001
            msg = str(exc)
            if qc_kwargs is None and "EXLLAMA" not in msg \
                    and "BACKEND" not in msg:
                raise          # unrelated failure — surface it immediately
            last_exc = exc
    raise last_exc


def load_proxy(spec: Dict, device: str = "cuda:0"):
    free_cuda()
    kind = spec["kind"]
    if kind == "gguf":
        m = _load_model(spec["repo"], gguf_file=spec["file"],
                        dtype=torch.bfloat16, device_map=device,
                        trust_remote_code=True)
    elif kind == "bnb":
        m = _load_model(spec["repo"],
                        quantization_config=_BNB_CONFIGS[spec["tag"]](),
                        device_map=device, trust_remote_code=True)
    elif kind == "gptq":
        m = _load_gptq_with_fallback(spec["repo"], device)
    elif kind == "dtype":
        m = _load_model(spec["repo"], dtype=getattr(torch, spec["dtype"]),
                        device_map=device, trust_remote_code=True)
    else:
        raise ValueError(f"Unknown proxy kind {kind!r}")
    return m.eval()


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------
def extract_Z(model, dataloader, device: str = "cuda:0") -> torch.Tensor:
    """Token-level (concat) features, float32 on CPU — paper protocol."""
    return LLMExtractor().extract_features(
        model, dataloader, device, z_mode="concat",
    )


def subsample_tokens(Z_T: torch.Tensor, Z_P: torch.Tensor,
                     cap: int, seed: int = 0):
    """Same random token subset for both sides (paired positions)."""
    n = min(Z_T.shape[0], Z_P.shape[0])
    if Z_T.shape[0] != Z_P.shape[0]:
        print(f"  [warn] token count mismatch T={Z_T.shape[0]} P={Z_P.shape[0]}; "
              f"truncating to {n}")
        Z_T, Z_P = Z_T[:n], Z_P[:n]
    if n <= cap:
        return Z_T, Z_P
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:cap].sort().values
    return Z_T[idx], Z_P[idx]
