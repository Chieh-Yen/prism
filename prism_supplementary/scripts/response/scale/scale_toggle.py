#!/usr/bin/env python3
"""Scale-axis intervention: switch outlier-channel preservation on and off.

The paper names outlier-channel preservation as the remedy the scale axis
points to (Sec. 4.4). The 8-bit path of bitsandbytes implements it: channels
whose activation magnitude exceeds ``llm_int8_threshold`` are kept in FP16 and
the rest are quantized. The default threshold is 6.0; a very large threshold
(1e9) quantizes every channel and changes nothing else. The reported settings
are 6.0 and 1e9. The output field ``outliers_preserved`` is true for a
threshold in (0, 1e6) and false otherwise; a threshold of 0 switches the
outlier path of bitsandbytes off, and thresholds other than 6.0 and 1e9 were
not run.

For each benchmark the BF16 target is evaluated once and the 8-bit variant
once per threshold, on the same sequences. Each (benchmark, threshold) cell
stores the two losses, their absolute difference and the PRISM decomposition
under identity alignment (W = I): rho_T, rho_P, the scale term
(rho_T - rho_P)^2, the shape term 2 rho_T rho_P (1 - Omega), the feature term
delta and the head term gamma.

Output: a JSON list with one record per cell, rewritten after every cell.
``analyze.py`` in this folder turns it into the reported table.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from prism.data.loaders import load_task_data, get_task_metadata
from prism.models.extractors import LLMExtractor
from prism.core.metrics import PRISMMetrics
from prism.experiments.quantization import pinned_revision  # configs/model_revisions.yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Remote code is opt-in (PRISM_TRUST_REMOTE_CODE=1). Both models used here are
# native to the pinned transformers release, so the default is off.
_TRUST_REMOTE_CODE = os.environ.get("PRISM_TRUST_REMOTE_CODE", "0") == "1"


def paired_losses(zm, loss_mode, st):
    if zm == "concat" and st.get("token_losses") is not None:
        return st["token_losses"]
    if loss_mode == "answer" and st.get("has_answer_loss"):
        return st["answer_losses"]
    return st["losses"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True,
                    help="Hugging Face id of the target model (loaded in BF16)")
    ap.add_argument("--tasks", default="arc,mmlu,squad,triviaqa,gsm8k")
    ap.add_argument("--thresholds", default="6.0,1e9",
                    help="llm_int8_threshold values; 6.0 = bitsandbytes default "
                         "(outlier channels kept in FP16), 1e9 = every channel quantized")
    ap.add_argument("--n", type=int, default=512,
                    help="number of test sequences per benchmark")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42,
                    help="seed of the dataset shuffle that selects the sequences")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, help="output JSON path")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(
        a.target, trust_remote_code=_TRUST_REMOTE_CODE,
        revision=pinned_revision(a.target))
    if tok.pad_token is None:
        # Llama ships without a pad token; the forgetting trainer does the same thing.
        tok.pad_token = tok.eos_token
    # The extraction assumes right padding.
    tok.padding_side = "right"
    ex = LLMExtractor()
    rows = []

    for task in a.tasks.split(","):
        meta = get_task_metadata(task)
        zm, lmode = meta["z_mode"], meta["loss_mode"]
        print(f"\n===== {task} =====", flush=True)
        dl = load_task_data(task, split="test", num_samples=a.n,
                            batch_size=a.batch_size, tokenizer=tok,
                            max_length=a.max_length, seed=a.seed)

        tgt = AutoModelForCausalLM.from_pretrained(
            a.target, dtype=torch.bfloat16, device_map=a.device,
            trust_remote_code=_TRUST_REMOTE_CODE,
            revision=pinned_revision(a.target)).eval()
        Zd_T, st_T = ex.extract_features_and_loss_per_sample(tgt, dl, a.device, z_modes=[zm])
        H_T = ex.extract_head(tgt).cpu()
        Z_T = Zd_T[zm]
        loss_T = paired_losses(zm, lmode, st_T).float().mean().item()
        del tgt; gc.collect(); torch.cuda.empty_cache()
        print(f"  target loss {loss_T:.4f}  rows {Z_T.shape[0]}", flush=True)

        for th in a.thresholds.split(","):
            thv = float(th)
            t0 = time.time()
            cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=thv)
            px = AutoModelForCausalLM.from_pretrained(
                a.target, quantization_config=cfg, device_map=a.device,
                trust_remote_code=_TRUST_REMOTE_CODE,
                revision=pinned_revision(a.target)).eval()
            Zd_P, st_P = ex.extract_features_and_loss_per_sample(px, dl, a.device, z_modes=[zm])
            Z_P = Zd_P[zm]
            H_P = ex.extract_head(px).cpu()
            loss_P = paired_losses(zm, lmode, st_P).float().mean().item()
            del px; gc.collect(); torch.cuda.empty_cache()

            md = a.device
            try:
                W_I = torch.eye(Z_T.shape[1], dtype=torch.float32, device=md)
                r = PRISMMetrics.compute_all(
                    Z_T.to(md).float(), H_T.to(md).float(),
                    Z_P.to(md).float(), H_P.to(md).float(),
                    W=W_I, label=f"int8@{th}")
                del W_I
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
                print("    metrics OOM on GPU, falling back to CPU", flush=True)
                W_I = torch.eye(Z_T.shape[1], dtype=torch.float32)
                r = PRISMMetrics.compute_all(Z_T, H_T, Z_P, H_P, W=W_I, label=f"int8@{th}")
                del W_I
            gc.collect(); torch.cuda.empty_cache()
            row = dict(task=task, target=a.target, threshold=thv,
                       outliers_preserved=(0.0 < thv < 1e6),
                       loss_T=loss_T, loss_P=loss_P, dR=abs(loss_T - loss_P),
                       rho_T=r.rho_target, rho_P=r.rho_proxy,
                       d_rho=r.rho_target - r.rho_proxy,
                       scale_term=r.scale_mismatch, shape_term=r.shape_mismatch,
                       omega=r.omega, delta=r.feature_error,
                       gamma=r.head_discrepancy, secs=round(time.time() - t0, 1))
            rows.append(row)
            print(f"    th={th:>6}  |dR| {row['dR']:.4f}  rho_P {row['rho_P']:.3f} "
                  f"(rho_T {row['rho_T']:.3f}, dRho {row['d_rho']:+.3f})  "
                  f"omega {row['omega']:.5f}  delta {row['delta']:.3f} "
                  f"gamma {row['gamma']:.3f}", flush=True)
            with open(a.out, "w") as f:
                json.dump(rows, f, indent=1)

        del Z_T, H_T; gc.collect(); torch.cuda.empty_cache()
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
