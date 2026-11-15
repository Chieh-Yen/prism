#!/usr/bin/env python3
"""Layer-wise and module-wise shape-drift profile of GGUF variants.

Theorem 1 needs the pre-head feature: that is the only point where a linear
read-out turns geometry into loss, so the *bound* does not transfer to
intermediate layers. The shape part of the decomposition does transfer as a
descriptive profile, and that is what this script records between a BF16
target and each GGUF variant, on the same sequences:

  * the residual stream: every entry of ``hidden_states`` (index 0 is the
    embedding output, index i is the stream after block i, and the last index
    is the final hidden state after the final norm);
  * the outputs of the q, k and v projections of every block: the output of
    the linear layers ``self_attn.q_proj``, ``k_proj`` and ``v_proj``
    (forward hooks; 0-based block index), i.e. the full vector over all
    heads, before any later per-head operation.

All non-pad tokens are used, prompt included. Each token vector is divided by
its own L2 norm before accumulation, so that a few tokens with very large
activations cannot dominate the inner product. This is the per-token
rescaling that the RMSNorm in front of every block also applies (up to the
factor sqrt(d)); the learned gain of RMSNorm is not applied here. After this
normalisation Omega is the mean per-token cosine
between the target's and the variant's feature, the stored RMS scales are 1
by construction, and the profile carries the shape term only.

Statistics are accumulated streaming (per read-out point: the sum of squared
norms for each model and the cross inner product), so memory is O(num_layers)
rather than O(tokens x layers x width).

Output: a JSON list with one record per variant, rewritten after every
variant. ``make_table.py`` in this folder turns it into the reported table.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from prism.data.loaders import load_task_data
from prism.experiments.quantization import pinned_revision  # configs/model_revisions.yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

# Remote code is opt-in (PRISM_TRUST_REMOTE_CODE=1). Both model families used
# here are native to the pinned transformers release, so the default is off.
_TRUST_REMOTE_CODE = os.environ.get("PRISM_TRUST_REMOTE_CODE", "0") == "1"

MODULES = ("q_proj", "k_proj", "v_proj")


def _attach(model, modules, store):
    """Forward hooks on each decoder layer's attention projections."""
    handles = []
    for li, layer in enumerate(model.model.layers):
        for m in modules:
            mod = getattr(layer.self_attn, m)
            handles.append(mod.register_forward_hook(
                lambda _mod, _inp, out, key=(li, m): store.__setitem__(key, out.detach())))
    return handles


def _acc(zt, zp, sel, ss_t, ss_p, cross, li):
    zt = zt.reshape(-1, zt.shape[-1]).float()
    zp = zp.reshape(-1, zp.shape[-1]).float()
    if sel is not None:
        zt, zp = zt[sel], zp[sel]
    zt = torch.nn.functional.normalize(zt, dim=1)
    zp = torch.nn.functional.normalize(zp, dim=1)
    ss_t[li] += (zt * zt).sum(dim=1).double().sum().item()
    ss_p[li] += (zp * zp).sum(dim=1).double().sum().item()
    cross[li] += (zt * zp).sum(dim=1).double().sum().item()
    return zt.shape[0]


@torch.no_grad()
def layer_stats(model_t, model_p, dl, device, n_layers, modules):
    """Per-layer (rho_T, rho_P, Omega) on the residual stream, plus the same
    three numbers on the outputs of the attention projections named in
    ``modules``, all on unit-normalised token vectors."""
    n_dec = n_layers - 1
    ss_t = torch.zeros(n_layers, dtype=torch.float64)
    ss_p = torch.zeros(n_layers, dtype=torch.float64)
    cross = torch.zeros(n_layers, dtype=torch.float64)
    mod_acc = {m: tuple(torch.zeros(n_dec, dtype=torch.float64) for _ in range(3))
               for m in modules}
    store_t, store_p = {}, {}
    h_t = _attach(model_t, modules, store_t)
    h_p = _attach(model_p, modules, store_p)
    rows = 0
    try:
        for batch in dl:
            b = {k: v.to(device) for k, v in batch.items()
                 if k in ("input_ids", "attention_mask")}
            m = b.get("attention_mask")
            sel = m.bool().reshape(-1) if m is not None else None
            ot = model_t(**b, output_hidden_states=True).hidden_states
            op = model_p(**b, output_hidden_states=True).hidden_states
            for li in range(n_layers):
                nrow = _acc(ot[li], op[li], sel, ss_t, ss_p, cross, li)
            rows += nrow
            for mod in modules:
                a_t, a_p, a_x = mod_acc[mod]
                for li in range(n_dec):
                    _acc(store_t[(li, mod)], store_p[(li, mod)], sel, a_t, a_p, a_x, li)
            store_t.clear(); store_p.clear()
            del ot, op
    finally:
        for h in h_t + h_p:
            h.remove()
    def _fin(a, b, c):
        return ((a / max(rows, 1)).sqrt(), (b / max(rows, 1)).sqrt(),
                c / (a.sqrt() * b.sqrt()).clamp(min=1e-12))
    rho_t, rho_p, omega = _fin(ss_t, ss_p, cross)
    mods = {mod: _fin(*mod_acc[mod]) for mod in modules}
    return rho_t, rho_p, omega, rows, mods


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True,
                    help="Hugging Face id of the target model (loaded in BF16)")
    ap.add_argument("--gguf_repo", required=True,
                    help="Hugging Face repo holding the GGUF variants")
    ap.add_argument("--gguf_tpl", required=True,
                    help="GGUF file name with a {quant} placeholder")
    ap.add_argument("--quants", required=True,
                    help="comma-separated GGUF tags; the reported runs use "
                         "Q6_K,Q4_K_M,Q2_K for Qwen3-8B-Base and Q6_K for Llama-3.1-8B")
    ap.add_argument("--task", default="mmlu")
    ap.add_argument("--n", type=int, default=128,
                    help="number of test sequences")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=1024)
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
    dl = load_task_data(a.task, split="test", num_samples=a.n,
                        batch_size=a.batch_size, tokenizer=tok,
                        max_length=a.max_length, seed=a.seed)

    print(f"loading target {a.target}", flush=True)
    mt = AutoModelForCausalLM.from_pretrained(
        a.target, dtype=torch.bfloat16, device_map=a.device,
        trust_remote_code=_TRUST_REMOTE_CODE,
        revision=pinned_revision(a.target)).eval()
    n_layers = mt.config.num_hidden_layers + 1

    out = []
    for quant in a.quants.split(","):
        fn = a.gguf_tpl.format(quant=quant)
        print(f"--- {quant} ---", flush=True)
        t0 = time.time()
        mp = AutoModelForCausalLM.from_pretrained(
            a.gguf_repo, gguf_file=fn, dtype=torch.bfloat16,
            device_map=a.device, trust_remote_code=_TRUST_REMOTE_CODE,
            revision=pinned_revision(a.gguf_repo)).eval()
        rt, rp, om, rows, mods = layer_stats(mt, mp, dl, a.device, n_layers, MODULES)
        del mp; gc.collect(); torch.cuda.empty_cache()
        out.append(dict(target=a.target, quant=quant, task=a.task, rows=rows,
                        rho_T=[round(x, 4) for x in rt.tolist()],
                        rho_P=[round(x, 4) for x in rp.tolist()],
                        omega=[round(x, 6) for x in om.tolist()],
                        modules={k: dict(rho_T=[round(x, 4) for x in v[0].tolist()],
                                         rho_P=[round(x, 4) for x in v[1].tolist()],
                                         omega=[round(x, 6) for x in v[2].tolist()])
                                 for k, v in mods.items()},
                        secs=round(time.time() - t0, 1)))
        worst = int(torch.argmin(om).item())
        print(f"  hidden states={n_layers} rows={rows} "
              f"min omega {om.min():.5f} at hidden-state index {worst}; "
              f"final hidden state omega {om[-1]:.5f}  ({time.time()-t0:.0f}s)", flush=True)
        for k, v in mods.items():
            print(f"  {k}: min omega {v[2].min():.5f} at 0-based block index "
                  f"{int(torch.argmin(v[2]))}, mean omega {v[2].mean():.5f}", flush=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
