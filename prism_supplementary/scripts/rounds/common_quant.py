"""
Shared helpers for the quantization-side scripts: the three-seed round in
scripts/rounds/ and the eight-sequence screen in scripts/response/screen_n8/.

Reuses the package's own loading conventions:
  * the variant list is read from data/variant_manifest.csv, which holds, for
    each quantized variant of the two main-text families, the `proxy_model`
    string and the `Label` that prism/experiments/quantization.py writes for
    it. The manifest has no result columns: it only fixes which proxies are
    loaded, and in which order;
  * proxy loading mirrors QuantizationExperiment._load_proxy_* and imports its
    _BNB_CONFIGS / _load_model directly, so models are loaded at the revisions
    pinned in configs/model_revisions.yaml and remote code stays opt-in
    (PRISM_TRUST_REMOTE_CODE=1).

GPU-side module: requires torch / transformers.
"""

from __future__ import annotations

import csv
import gc
import sys
from pathlib import Path
from typing import Dict, List

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent          # supplementary root (holds prism/)
sys.path.insert(0, str(REPO))

from prism.experiments.quantization import (_BNB_CONFIGS,              # noqa: E402
                                            _TRUST_REMOTE_CODE,
                                            _load_model)
from prism.models.extractors import LLMExtractor                       # noqa: E402

MANIFEST_PATH = HERE / "data" / "variant_manifest.csv"

# `target_model` strings of the two main-text families.
FAMILIES = {
    "llama": "meta-llama/Meta-Llama-3.1-8B",
    "qwen": "Qwen/Qwen3-8B-Base",
}

BENCHMARKS = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]


# ----------------------------------------------------------------------
# Variant list from the manifest
# ----------------------------------------------------------------------
def parse_proxy_spec(proxy_model: str, label: str, target_model: str) -> Dict:
    """Turn a `proxy_model` string into a loader spec.

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
        if tag.upper() == "FP16":
            return {"kind": "dtype", "repo": repo, "dtype": "float16",
                    "label": label}
    raise ValueError(f"Unrecognised proxy_model format: {proxy_model!r}")


def variants_from_csv(family: str) -> List[Dict]:
    """Distinct proxies of a family, in manifest order."""
    target = FAMILIES[family]
    seen, specs = set(), []
    for r in csv.DictReader(open(MANIFEST_PATH)):
        if r["target_model"] != target:
            continue
        key = (r["proxy_model"], r["Label"])
        if key in seen:
            continue
        seen.add(key)
        specs.append(parse_proxy_spec(r["proxy_model"], r["Label"], target))
    return specs


def risk_gaps_from_csv(family: str, results_csv) -> Dict[tuple, float]:
    """(Label, dataset) -> empirical risk gap, for the family.

    `results_csv` is one result table, or a list of them, written by
    BaseExperiment.save_csv, i.e. by run_quantization.py, which writes one
    file per model and dataset (prism_<model>_<dataset>_n<N>_<z_mode>.csv).
    Rows are selected by `target_model`; rows whose `z_mode` column is not
    "concat" are skipped (the scores joined with these gaps are computed on
    token-level features).

    The gap of a row is read from the first of these columns that is present
    and not empty:

      `|AdR|`  answer-only gap: |mean_T - mean_P| of the mean over sequences
               of each sequence's mean answer-token loss. Written by
               save_csv for the question-answering benchmarks (all five
               benchmarks used here).
      `|dR|`   full-sequence gap (all tokens); the only gap column for tasks
               without an answer span.

    A (Label, dataset) pair that occurs twice with different gaps (for
    example files of two sample sizes) is an error.
    """
    target = FAMILIES[family]
    if isinstance(results_csv, (str, Path)):
        results_csv = [results_csv]
    out, source = {}, {}
    for path in results_csv:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("target_model") != target:
                    continue
                if r.get("z_mode") not in (None, "", "concat"):
                    continue
                raw = next((r[c] for c in ("|AdR|", "|dR|")
                            if r.get(c) not in (None, "")), None)
                if raw is None:
                    continue
                try:
                    gap = float(raw)
                except ValueError:
                    continue
                key = (r["Label"], r["dataset"])
                if key in out and out[key] != gap:
                    raise ValueError(
                        f"{key} has two different risk gaps ({out[key]} in "
                        f"{source[key]}, {gap} in {path}); pass the result "
                        f"files of one run only")
                out[key], source[key] = gap, path
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
        trust_remote_code=_TRUST_REMOTE_CODE,
    ).eval()


def _load_gptq_with_fallback(repo: str, device: str):
    """Load a pre-quantized GPTQ repository; if its kernel backend is unavailable, retry with the exllama-v2 kernel, then without exllama."""
    from transformers import GPTQConfig                    # lazy import
    attempts = [
        ("as-is", None),
        ("exllama_v2", dict(bits=4, exllama_config={"version": 2})),
        ("no_exllama", dict(bits=4, use_exllama=False)),
    ]
    last_exc = None
    for tag, qc_kwargs in attempts:
        try:
            kwargs = dict(device_map=device, trust_remote_code=_TRUST_REMOTE_CODE)
            if qc_kwargs is not None:
                print(f"  [gptq-fallback] retrying {repo} with {tag}")
                kwargs["quantization_config"] = GPTQConfig(**qc_kwargs)
            return _load_model(repo, **kwargs)
        except Exception as exc:                           # noqa: BLE001
            msg = str(exc)
            if qc_kwargs is None and "EXLLAMA" not in msg \
                    and "BACKEND" not in msg:
                raise          # unrelated failure
            last_exc = exc
    raise last_exc


def load_proxy(spec: Dict, device: str = "cuda:0"):
    free_cuda()
    kind = spec["kind"]
    if kind == "gguf":
        m = _load_model(spec["repo"], gguf_file=spec["file"],
                        dtype=torch.bfloat16, device_map=device,
                        trust_remote_code=_TRUST_REMOTE_CODE)
    elif kind == "bnb":
        m = _load_model(spec["repo"],
                        quantization_config=_BNB_CONFIGS[spec["tag"]](),
                        device_map=device, trust_remote_code=_TRUST_REMOTE_CODE)
    elif kind == "gptq":
        m = _load_gptq_with_fallback(spec["repo"], device)
    elif kind == "dtype":
        m = _load_model(spec["repo"], dtype=getattr(torch, spec["dtype"]),
                        device_map=device, trust_remote_code=_TRUST_REMOTE_CODE)
    else:
        raise ValueError(f"Unknown proxy kind {kind!r}")
    return m.eval()


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------
def extract_Z(model, dataloader, device: str = "cuda:0") -> torch.Tensor:
    """Token-level (concat) features, float32 on the CPU."""
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
