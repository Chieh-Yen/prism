#!/usr/bin/env python3
"""Head experiment 2: head precision x backbone grid.

Grid, on identical inputs (same loader, n and seed as head_swap.py):
  backbone in {bf16 (the target itself), GGUF files given by --backbones}
  head     in {bf16, q8_0, q6_k, q5_0, q4_0, q3_0, q2_0}

  bf16      the target's own lm_head
  q8_0/q5_0/q4_0  the block formats Q8_0, Q5_0 and Q4_0 (blocks of 32, FP16
            scale), applied to the target's lm_head by the torch functions
            below. The functions are compared with gguf-py's reference
            quantizers at start-up on the first 2048 rows of the head; the
            share of differing entries and the largest absolute difference
            are stored in info["quantizer_check"].
  q6_k      the lm_head weight of the GGUF file of the first backbone in
            --backbones, as transformers dequantizes it. For every backbone
            file the script checks that the file's own lm_head equals this
            tensor entry for entry and stores the result in
            info["native_head_is_q6k"].
  q3_0/q2_0 the Q4_0 scheme with 3 and 2 bits, i.e. fewer levels (stress
            range). These are not GGUF formats and have no gguf-py reference.

All heads are run in BF16, as transformers does when it dequantizes GGUF.
Metrics use W = I against the target (Z_T, H_T), as in head_swap.py.

Each row stores the feature term delta and the head term gamma, and
  bound = K_feat * delta + K_pred * gamma,
with K_feat and K_pred from UnifiedBound.theoretical_K on the target head
(stored in info).
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np
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

HEADS = ["bf16", "q8_0", "q6_k", "q5_0", "q4_0", "q3_0", "q2_0"]
LEGACY_BITS = {"q5_0": 5, "q4_0": 4, "q3_0": 3, "q2_0": 2}


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


def _q_legacy(B32, bits):
    """Q{bits}_0 on blocks (n, 32) float32, after gguf-py's Q4_0/Q5_0."""
    half = 2 ** (bits - 1)
    imax = B32.abs().argmax(-1, keepdim=True)
    mx = torch.gather(B32, -1, imax)
    d = mx / -half
    idd = torch.where(d == 0, torch.zeros_like(d), 1.0 / d)
    q = torch.trunc(B32.double() * idd.double() + (half + 0.5)).float().clamp(0, 2 ** bits - 1)
    return (q - half) * d.half().float()


def _q8_0(B32):
    d = B32.abs().amax(-1, keepdim=True) / 127
    idd = torch.where(d == 0, torch.zeros_like(d), 1.0 / d)
    x = B32 * idd
    q = torch.sign(x) * torch.floor(x.abs() + 0.5)          # roundf: half away from zero
    return q * d.half().float()


def quantize_head(W, kind, dev, chunk=8192):
    """W: (V, d) lm_head weight. Blocks of 32 run along d, as in GGUF."""
    out = torch.empty(W.shape, dtype=torch.bfloat16)
    for i in range(0, W.shape[0], chunk):
        B = W[i:i + chunk].to(dev, torch.float32)
        shp = B.shape
        B = B.reshape(-1, 32)
        Q = _q8_0(B) if kind == "q8_0" else _q_legacy(B, LEGACY_BITS[kind])
        out[i:i + chunk] = Q.reshape(shp).to(torch.bfloat16).cpu()
    return out


def check_against_gguf(W, dev, rows=2048):
    """Compare the torch quantizers with gguf-py on the first `rows` rows of the head.

    Returns, per format, the fraction of entries that differ and the largest
    absolute difference.
    """
    import gguf
    from gguf import quants
    T = gguf.GGMLQuantizationType
    x = W[:rows].float().numpy()
    rep = {}
    for kind, t in [("q8_0", T.Q8_0), ("q5_0", T.Q5_0), ("q4_0", T.Q4_0)]:
        ref = quants.dequantize(quants.quantize(x, t), t)
        B = torch.from_numpy(x).to(dev).reshape(-1, 32)
        mine = (_q8_0(B) if kind == "q8_0" else _q_legacy(B, LEGACY_BITS[kind])).reshape(x.shape).cpu().numpy()
        rep[kind] = dict(frac_mismatch=float(np.mean(mine != ref)),
                         max_abs_diff=float(np.abs(mine - ref).max()))
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", required=True,
                    help="original model, e.g. Qwen/Qwen3-8B-Base")
    ap.add_argument("--gguf_repo", required=True,
                    help="GGUF repository, e.g. mradermacher/Qwen3-8B-Base-GGUF")
    ap.add_argument("--gguf_tpl", required=True,
                    help="GGUF file name with a {quant} field, "
                         "e.g. 'Qwen3-8B-Base.{quant}.gguf'")
    ap.add_argument("--backbones", default="Q4_K_M,Q2_K")
    ap.add_argument("--heads", default=",".join(HEADS))
    ap.add_argument("--tasks", default="arc,mmlu,squad,triviaqa,gsm8k")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = a.device
    heads_wanted = a.heads.split(",")
    tasks = a.tasks.split(",")
    backbones = [b for b in a.backbones.split(",") if b]

    tok = AutoTokenizer.from_pretrained(a.target, revision=pinned_revision(a.target),
                                        trust_remote_code=_TRUST_REMOTE_CODE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # The extraction assumes right padding.
    tok.padding_side = "right"
    ex = LLMExtractor()
    info = dict(target=a.target, gguf_repo=a.gguf_repo, backbones=backbones,
                heads=heads_wanted, n=a.n, seed=a.seed)
    rows = []

    def dump():
        with open(a.out, "w") as f:
            json.dump(dict(info=info, rows=rows), f, indent=1)

    loaders = {}
    for task in tasks:
        loaders[task] = load_task_data(task, split="test", num_samples=a.n,
                                       batch_size=a.batch_size, tokenizer=tok,
                                       max_length=a.max_length, seed=a.seed)

    # ---- the q6_k head: lm_head of the first GGUF backbone file ----
    t0 = time.time()
    g = load_model(a.gguf_repo, gguf_file=a.gguf_tpl.format(quant=backbones[0]),
                   dtype=torch.bfloat16, device_map="cpu")
    W_q6k = g.lm_head.weight.data.clone()
    del g; gc.collect()
    print(f"q6_k head read from {backbones[0]} in {time.time()-t0:.0f}s", flush=True)

    # ---- target: features and losses under every head ----
    tgt = load_model(a.target, dtype=torch.bfloat16, device_map=dev)
    tgt.eval()
    assert tgt.lm_head.weight.data_ptr() != tgt.get_input_embeddings().weight.data_ptr(), "tied head"
    W_T = tgt.lm_head.weight.data.detach().cpu().clone()          # (V, d) bf16
    info["quantizer_check"] = check_against_gguf(W_T, "cpu")   # on CPU, like the quantizers below
    print("quantizer check vs gguf-py:", info["quantizer_check"], flush=True)

    heads = {"bf16": W_T, "q6_k": W_q6k}
    for k in heads_wanted:
        if k not in heads:
            heads[k] = quantize_head(W_T, k, "cpu")
    nT = W_T.float().norm().item()
    info["head_rel_err"] = {k: (heads[k].float() - W_T.float()).norm().item() / nT for k in heads_wanted}
    print("head rel. Frobenius error:", info["head_rel_err"], flush=True)
    dump()

    ZT, HT_f, lossT = {}, W_T.T.float(), {}
    W_I = None

    def run_heads(model, backbone, task, Zref_store):
        nonlocal W_I
        meta = get_task_metadata(task)
        zm, lmode = meta["z_mode"], meta["loss_mode"]
        w = model.lm_head.weight
        orig = w.data.clone()
        Z0 = None
        for hk in heads_wanted:
            t1 = time.time()
            w.data = heads[hk].to(w.device, w.dtype)
            Zd, st = ex.extract_features_and_loss_per_sample(model, loaders[task], dev, z_modes=[zm])
            Z = Zd[zm]
            loss = paired_losses(zm, lmode, st).float().mean().item()
            same_backbone = True if Z0 is None else bool(torch.equal(Z, Z0))
            if Z0 is None:
                Z0 = Z
                if backbone == "bf16":
                    Zref_store[task] = Z
            if backbone == "bf16" and hk == "bf16":
                lossT[task] = loss
            Zt = ZT[task] if task in ZT else Zref_store[task]
            if W_I is None or W_I.shape[0] != Zt.shape[1]:
                W_I = torch.eye(Zt.shape[1], dtype=torch.float32, device=dev)
            r = PRISMMetrics.compute_all(Zt.to(dev).float(), HT_f.to(dev), Z.to(dev).float(),
                                         heads[hk].T.float().to(dev), W=W_I, label=f"{backbone}/{hk}")
            rows.append(dict(task=task, backbone=backbone, head=hk, n_rows=int(Z.shape[0]),
                             loss=loss, same_backbone_as_first_head=same_backbone,
                             omega=r.omega, rho_T=r.rho_target, rho_P=r.rho_proxy,
                             delta=r.feature_error, gamma=r.head_discrepancy,
                             secs=round(time.time() - t1, 1)))
            print(f"  [{task}] {backbone:7s} head={hk:5s} loss={loss:.4f} delta={r.feature_error:.3f} "
                  f"gamma={r.head_discrepancy:.3f} same_bb={same_backbone} ({time.time()-t1:.0f}s)", flush=True)
            gc.collect(); torch.cuda.empty_cache()
        w.data = orig

    for task in tasks:
        run_heads(tgt, "bf16", task, ZT)
        dump()
    del tgt; gc.collect(); torch.cuda.empty_cache()

    for bb in backbones:
        t0 = time.time()
        px = load_model(a.gguf_repo, gguf_file=a.gguf_tpl.format(quant=bb),
                        dtype=torch.bfloat16, device_map=dev)
        px.eval()
        info.setdefault("native_head_is_q6k", {})[bb] = bool(torch.equal(px.lm_head.weight.data.cpu(), W_q6k))
        print(f"== backbone {bb} loaded in {time.time()-t0:.0f}s; native head == q6_k: "
              f"{info['native_head_is_q6k'][bb]}", flush=True)
        for task in tasks:
            run_heads(px, bb, task, ZT)
            dump()
        del px; gc.collect(); torch.cuda.empty_cache()

    # Constants of the bound; they depend on the target head only.
    K = UnifiedBound.theoretical_K(HT_f.to(dev))
    info["K_feat"], info["K_pred"] = K["K_feat"], K["K_pred"]
    print(f"K_feat={K['K_feat']:.6f}  K_pred={K['K_pred']:.6f}", flush=True)

    for r in rows:
        r["loss_T"] = lossT[r["task"]]
        r["dR_signed"] = r["loss"] - r["loss_T"]
        r["dR"] = abs(r["dR_signed"])
        r["bound"] = info["K_feat"] * r["delta"] + info["K_pred"] * r["gamma"]
        r["bound_holds"] = r["dR"] <= r["bound"]
    dump()
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
