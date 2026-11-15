#!/usr/bin/env python3
"""Head experiment 1: hybrids of a GGUF variant and the original model.

For each GGUF variant, four models are evaluated on identical inputs
(same loader, same n, same seed):

  target    original backbone  + original head (BF16)
  variant   quantized backbone + quantized head (the GGUF file as loaded)
  hybrid A  quantized backbone + original head
  hybrid B  original backbone  + the variant's quantized head

"Backbone" is everything before the output head (lm_head). Hybrid A keeps the
variant's features, so its feature term equals the variant's and its head term
is exactly 0. Hybrid B keeps the target's features, so the head is its only
source of error. The script checks both statements on the extracted features
(`backbone_identical`, `head_only_identical`).

The quantized head is the lm_head weight of the GGUF file as transformers
dequantizes it; it is run in BF16, like every other weight. head_precision.py
compares the output tensors of the GGUF files it loads entry for entry.

Metrics use W = I against the target (Z_T, H_T). The gaps are |R_T - R| in
nats, where R is the answer-token cross-entropy on the n sequences. Each row
also stores the constants of the bound (K_feat, K_pred) for the target head.

Field names in the output rows:
  loss_P, dR                     the variant
  loss_P_fp16head, dR_fp16head   hybrid A (delta_fp16head, gamma_fp16head)
  loss_T_quanthead, dR_quanthead hybrid B (delta_quanthead, gamma_quanthead)
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # supplementary root (holds prism/)

from prism.data.loaders import load_task_data, get_task_metadata   # noqa: E402
from prism.models.extractors import LLMExtractor                   # noqa: E402
from prism.core.metrics import PRISMMetrics                        # noqa: E402
from prism.core.bounds import UnifiedBound                         # noqa: E402
from prism.experiments.quantization import pinned_revision         # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM       # noqa: E402

# Remote code is opt-in (PRISM_TRUST_REMOTE_CODE=1); the default is off.
_TRUST_REMOTE_CODE = os.environ.get("PRISM_TRUST_REMOTE_CODE", "0") == "1"


def paired_losses(zm, loss_mode, st):
    if zm == "concat" and st.get("token_losses") is not None:
        return st["token_losses"]
    if loss_mode == "answer" and st.get("has_answer_loss"):
        return st["answer_losses"]
    return st["losses"]


def load_model(model_id, **kw):
    return AutoModelForCausalLM.from_pretrained(
        model_id, revision=pinned_revision(model_id),
        trust_remote_code=_TRUST_REMOTE_CODE, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", required=True,
                    help="original model, e.g. Qwen/Qwen3-8B-Base")
    ap.add_argument("--gguf_repo", required=True,
                    help="GGUF repository, e.g. mradermacher/Qwen3-8B-Base-GGUF")
    ap.add_argument("--gguf_tpl", required=True,
                    help="GGUF file name with a {quant} field, "
                         "e.g. 'Qwen3-8B-Base.{quant}.gguf'")
    ap.add_argument("--quants", default="Q6_K")
    ap.add_argument("--tasks", default="arc,mmlu,squad,triviaqa,gsm8k")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--metrics_device", default="cuda:0",
                    help="where to run the PRISM metric contractions")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = a.device
    tok = AutoTokenizer.from_pretrained(a.target, revision=pinned_revision(a.target),
                                        trust_remote_code=_TRUST_REMOTE_CODE)
    if tok.pad_token is None:
        # Llama ships without a pad token.
        tok.pad_token = tok.eos_token
    # The extraction assumes right padding.
    tok.padding_side = "right"
    ex = LLMExtractor()
    rows = []
    K = None

    for task in a.tasks.split(","):
        meta = get_task_metadata(task)
        zm, lmode = meta["z_mode"], meta["loss_mode"]
        print(f"\n===== {task} (z_mode={zm}, loss_mode={lmode}) =====", flush=True)
        dl = load_task_data(task, split="test", num_samples=a.n,
                            batch_size=a.batch_size, tokenizer=tok,
                            max_length=a.max_length, seed=a.seed)

        t0 = time.time()
        tgt = load_model(a.target, dtype=torch.bfloat16, device_map=dev)
        tgt.eval()
        Zd_T, st_T = ex.extract_features_and_loss_per_sample(
            tgt, dl, dev, z_modes=[zm])
        H_T = ex.extract_head(tgt).cpu()          # (d, V)
        Z_T = Zd_T[zm]
        lT = paired_losses(zm, lmode, st_T)
        loss_T = lT.float().mean().item()
        del tgt; gc.collect(); torch.cuda.empty_cache()
        print(f"  target done in {time.time()-t0:.0f}s  loss_T={loss_T:.4f}  "
              f"Z_T={tuple(Z_T.shape)}", flush=True)

        if K is None:
            # Constants of the bound; they depend on the target head only.
            K = UnifiedBound.theoretical_K(H_T.to(a.metrics_device))
            gc.collect(); torch.cuda.empty_cache()
            print(f"  K_feat={K['K_feat']:.6f}  K_pred={K['K_pred']:.6f}", flush=True)

        for quant in a.quants.split(","):
            fn = a.gguf_tpl.format(quant=quant)
            print(f"  --- {quant} ({fn}) ---", flush=True)
            t1 = time.time()
            px = load_model(a.gguf_repo, gguf_file=fn,
                            dtype=torch.bfloat16, device_map=dev)
            px.eval()
            Zd_P, st_P = ex.extract_features_and_loss_per_sample(
                px, dl, dev, z_modes=[zm])
            Z_P = Zd_P[zm]
            H_P = ex.extract_head(px).cpu()
            loss_P = paired_losses(zm, lmode, st_P).float().mean().item()

            # ---- hybrid A: restore the target's head, same backbone ----
            w = px.lm_head.weight.data
            orig = w.clone()
            px.lm_head.weight.data = H_T.T.contiguous().to(w.device, w.dtype)
            Zd_S, st_S = ex.extract_features_and_loss_per_sample(
                px, dl, dev, z_modes=[zm])
            loss_S = paired_losses(zm, lmode, st_S).float().mean().item()
            Z_S = Zd_S[zm]
            backbone_identical = bool(torch.equal(Z_P, Z_S))
            px.lm_head.weight.data = orig
            del px; gc.collect(); torch.cuda.empty_cache()

            # ---- hybrid B: target backbone + the variant's quantized head ----
            # The mirror of hybrid A. The features are the target's own, so the
            # only error source is the head.
            tb = load_model(a.target, dtype=torch.bfloat16, device_map=dev)
            tb.eval()
            wb = tb.lm_head.weight.data
            ob = wb.clone()
            tb.lm_head.weight.data = H_P.T.contiguous().to(wb.device, wb.dtype)
            Zd_B, st_B = ex.extract_features_and_loss_per_sample(
                tb, dl, dev, z_modes=[zm])
            loss_B = paired_losses(zm, lmode, st_B).float().mean().item()
            # guards against tied embeddings: the backbone must be untouched
            head_only_identical = bool(torch.equal(Zd_B[zm], Z_T))
            tb.lm_head.weight.data = ob
            del tb, Zd_B, st_B, ob
            gc.collect(); torch.cuda.empty_cache()

            # Metrics on GPU: the head term contracts Sigma^{1/2} (d x d) with
            # (d x V) and V is 128k-152k, which is minutes on CPU and seconds here.
            md = a.metrics_device
            W_I = torch.eye(Z_T.shape[1], dtype=torch.float32, device=md)
            Zt_, Ht_ = Z_T.to(md).float(), H_T.to(md).float()
            Zp_, Hp_ = Z_P.to(md).float(), H_P.to(md).float()
            r_P = PRISMMetrics.compute_all(Zt_, Ht_, Zp_, Hp_, W=W_I, label=f"{quant}")
            r_S = PRISMMetrics.compute_all(Zt_, Ht_, Zp_, Ht_, W=W_I,
                                           label=f"{quant}+fp16head")
            r_B = PRISMMetrics.compute_all(Zt_, Ht_, Zt_, Hp_, W=W_I,
                                           label=f"{quant}+quanthead")
            del Zt_, Ht_, Zp_, Hp_, W_I
            gc.collect(); torch.cuda.empty_cache()

            row = dict(
                target=a.target, gguf_repo=a.gguf_repo,
                task=task, quant=quant, n_samples=a.n, n_rows=int(Z_T.shape[0]),
                loss_T=loss_T, loss_P=loss_P, loss_P_fp16head=loss_S,
                dR=abs(loss_T - loss_P), dR_fp16head=abs(loss_T - loss_S),
                omega=r_P.omega, rho_T=r_P.rho_target, rho_P=r_P.rho_proxy,
                delta=r_P.feature_error, gamma=r_P.head_discrepancy,
                gamma_fp16head=r_S.head_discrepancy,
                delta_fp16head=r_S.feature_error,
                backbone_identical=backbone_identical,
                loss_T_quanthead=loss_B,
                dR_quanthead=abs(loss_T - loss_B),
                gamma_quanthead=r_B.head_discrepancy,
                delta_quanthead=r_B.feature_error,
                head_only_identical=head_only_identical,
                K_feat=K["K_feat"], K_pred=K["K_pred"],
                secs=round(time.time() - t1, 1),
            )
            rows.append(row)
            print(f"    hybrid B: |dR|(target backbone + quant head) = "
                  f"{row['dR_quanthead']:.4f}  delta={row['delta_quanthead']:.4g}  "
                  f"gamma={row['gamma_quanthead']:.3f}  "
                  f"head_only_identical={head_only_identical}", flush=True)
            print(f"    |dR| {row['dR']:.4f} -> {row['dR_fp16head']:.4f}   "
                  f"gamma {row['gamma']:.3f} -> {row['gamma_fp16head']:.3g}   "
                  f"delta {row['delta']:.3f}  backbone_identical={backbone_identical}",
                  flush=True)
            with open(a.out, "w") as f:
                json.dump(rows, f, indent=1)

        del Z_T, H_T; gc.collect(); torch.cuda.empty_cache()

    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
