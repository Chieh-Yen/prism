#!/usr/bin/env python3
"""Lipschitz constants of the bound for one target model.

The bound is K_feat * delta + K_pred * gamma. Both constants depend only on
the target model, not on the variant or the benchmark:

    K_feat = max_{j,k} ||h_j - h_k||_2   over the columns h_j of the target head
    K_pred = sqrt(2)

They are computed by ``prism.core.bounds.UnifiedBound.theoretical_K`` from the
output head of the BF16 target, the same call the quantization pipeline uses.
``scale_toggle.py`` does not store them, so ``analyze.py`` reads them from the
small JSON this script writes: {"target", "K_feat", "K_pred"}.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from prism.core.bounds import UnifiedBound
from prism.models.extractors import LLMExtractor
from prism.experiments.quantization import pinned_revision  # configs/model_revisions.yaml
from transformers import AutoModelForCausalLM

# Remote code is opt-in (PRISM_TRUST_REMOTE_CODE=1); the default is off.
_TRUST_REMOTE_CODE = os.environ.get("PRISM_TRUST_REMOTE_CODE", "0") == "1"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True,
                    help="Hugging Face id of the target model (loaded in BF16)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, help="output JSON path")
    a = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        a.target, dtype=torch.bfloat16, device_map=a.device,
        trust_remote_code=_TRUST_REMOTE_CODE,
        revision=pinned_revision(a.target)).eval()
    H_T = LLMExtractor().extract_head(model)
    k = UnifiedBound.theoretical_K(H_T)
    out = dict(target=a.target, K_feat=k["K_feat"], K_pred=k["K_pred"])
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{a.target}: K_feat {out['K_feat']:.6f}  K_pred {out['K_pred']:.6f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
