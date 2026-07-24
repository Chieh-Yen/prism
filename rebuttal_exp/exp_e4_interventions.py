#!/usr/bin/env python3
"""
E4 — Single-axis controlled interventions (8VrD-W3: "no controlled
interventions that change only activation scale, backbone geometry, or the
output head").

Design: inject three families of *pure* perturbations into ONE base model
(default Llama-3.1-8B) and show each PRISM term responds selectively while
the empirical risk gap |dR| moves with it.

  scale-only   final-norm output * alpha, alpha in {0.5, 0.8, 1.25, 2.0}
               -> only (rho_T - rho_P)^2 moves;  1-Omega ~ 0;  gamma = 0
  shape-only   final hidden states rotated by R = expm(theta * A_hat),
               A_hat skew-symmetric, ||A_hat||_2 = 1, theta in
               {0.05, 0.1, 0.2, 0.4} rad -> token norms preserved exactly:
               scale = 0 to numerical precision; only 1-Omega moves; gamma = 0
  head-only    lm_head RTN-quantised per-row at {8, 6, 4, 3} bits, backbone
               untouched -> delta = 0 exactly; only gamma moves

All losses are *real* (the perturbation runs inside the forward pass), so
each row also reports the measured |dR| and verifies bound validity.
Per-config sup-residual  max_i ||z_T,i - z_P,i||  is dumped for E5's
concentration-corollary numerical illustration.

Cost: 1 model load + ~13 configs x (512-sample forward + d x d metrics)
~= 30-45 min on the RTX 5090.

Usage (repo root, GPU box):
    python rebuttal_exp/exp_e4_interventions.py \
        --model meta-llama/Meta-Llama-3.1-8B --dataset mmlu

Output: rebuttal_exp/out/E4/interventions_{model_short}_{dataset}.csv
        rebuttal_exp/out/E4/E4_results_{model_short}_{dataset}.md
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from prism.core.bounds import UnifiedBound                  # noqa: E402
from prism.core.metrics import PRISMMetrics                 # noqa: E402
from prism.data.loaders import load_task_data               # noqa: E402
from prism.models.extractors import LLMExtractor            # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

OUT_DIR = HERE / "out" / "E4"

SCALE_ALPHAS = [0.5, 0.8, 1.25, 2.0]
SHAPE_THETAS = [0.05, 0.1, 0.2, 0.4]
HEAD_BITS = [8, 6, 4, 3]


def find_final_norm(model):
    """Final pre-lm_head RMSNorm across supported architectures."""
    for path in ("model.norm", "model.model.norm", "transformer.ln_f"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("Cannot locate final norm module")


def make_rotation(d: int, theta: float, device, seed: int = 0) -> torch.Tensor:
    """R = expm(theta * A_hat), A_hat skew-symmetric with unit spectral norm."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    G = torch.randn(d, d, generator=g)
    A = (G - G.T) / 2
    A = A.to(device=device, dtype=torch.float32)
    A = A / torch.linalg.matrix_norm(A, ord=2)
    return torch.matrix_exp(theta * A)          # orthogonal (d, d)


def rtn_quantize_rows(W: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-row round-to-nearest quantization of (V, d) weights."""
    Wf = W.float()
    qmax = 2 ** (bits - 1) - 1
    scale = Wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qmax
    return ((Wf / scale).round().clamp(-qmax - 1, qmax) * scale).to(W.dtype)


def extract(model, dl, device):
    """(Z_concat, mean token loss, per-token count) in one pass."""
    Z, stats = LLMExtractor().extract_features_and_loss_per_sample(
        model, dl, device, z_mode="concat",
    )
    tok = stats["token_losses"]
    loss = tok.mean().item() if tok is not None else stats["losses"].mean().item()
    return Z, loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--dataset", default="mmlu")
    ap.add_argument("--num_samples", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    short = args.model.split("/")[-1].lower()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dl = load_task_data(args.dataset, split="test",
                        num_samples=args.num_samples,
                        batch_size=args.batch_size, tokenizer=tokenizer,
                        max_length=args.max_length, seed=args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=True,
    ).eval()
    norm_mod = find_final_norm(model)
    extractor = LLMExtractor()
    H_T = extractor.extract_head(model).float().cpu()      # (d, V)
    d = H_T.shape[0]
    K = UnifiedBound.theoretical_K(H_T.to(args.device))
    K_feat, K_pred = K["K_feat"], K["K_pred"]
    print(f"K_feat={K_feat:.4f}  K_pred={K_pred:.4f}  d={d}")

    # ── Reference (unperturbed) pass ────────────────────────────────────
    print("Reference pass ...")
    Z_T, loss_T = extract(model, dl, args.device)
    print(f"  Z_T={list(Z_T.shape)}  loss_T={loss_T:.4f}")

    hook_state = {"mode": None, "alpha": 1.0, "R": None}

    def hook(_mod, _inp, out):
        if hook_state["mode"] == "scale":
            return out * hook_state["alpha"]
        if hook_state["mode"] == "shape":
            R = hook_state["R"]
            return (out.to(R.dtype) @ R.T).to(out.dtype)
        return out

    handle = norm_mod.register_forward_hook(hook)

    rows = []

    def evaluate(config_label, family, H_P=None):
        t0 = time.time()
        Z_P, loss_P = extract(model, dl, args.device)
        H_P_use = H_P if H_P is not None else H_T
        res = PRISMMetrics.compute_all(
            Z_T.to(args.device), H_T.to(args.device),
            Z_P.to(args.device), H_P_use.to(args.device),
            W=torch.eye(d, device=args.device), label=config_label,
        )
        sup_res = (Z_T.to(args.device) - Z_P.to(args.device)) \
            .norm(dim=1).max().item()
        bound = UnifiedBound.compute_bound(
            res.omega, res.rho_target, res.rho_proxy, res.head_discrepancy,
            K_feat=K_feat, K_pred=K_pred,
        )
        b_total = bound.get("risk_bound_total",
                            K_feat * res.feature_error
                            + K_pred * res.head_discrepancy)
        row = {
            "config": config_label, "family": family,
            "scale_term": res.scale_mismatch,
            "shape_term": res.shape_mismatch,
            "one_minus_omega": 1 - res.omega,
            "gamma": res.head_discrepancy,
            "delta": K_feat * res.feature_error,
            "bound": b_total,
            "|dR|": abs(loss_P - loss_T),
            "bound_holds": b_total >= abs(loss_P - loss_T),
            "sup_residual": sup_res,
            "rho_T": res.rho_target, "rho_P": res.rho_proxy,
        }
        rows.append(row)
        print(f"  {config_label:<16s} scale={row['scale_term']:.3e} "
              f"shape={row['shape_term']:.3e} gamma={row['gamma']:.3e} "
              f"B={row['bound']:.2f} |dR|={row['|dR|']:.4f} "
              f"holds={row['bound_holds']} ({time.time() - t0:.0f}s)")
        del Z_P

    # identity sanity row
    hook_state["mode"] = None
    evaluate("identity", "control")

    # (a) scale-only
    for a in SCALE_ALPHAS:
        hook_state.update(mode="scale", alpha=a)
        evaluate(f"scale_x{a}", "scale")

    # (b) shape-only (norm-preserving rotation)
    for th in SHAPE_THETAS:
        R = make_rotation(d, th, args.device, seed=args.seed)
        hook_state.update(mode="shape", R=R)
        evaluate(f"rot_{th}rad", "shape")
        hook_state["R"] = None
        torch.cuda.empty_cache()

    # (c) head-only (RTN-quantised lm_head; backbone untouched)
    hook_state["mode"] = None
    lm_head_w = model.lm_head.weight
    W_orig = lm_head_w.data.clone()
    for bits in HEAD_BITS:
        Wq = rtn_quantize_rows(W_orig, bits)
        lm_head_w.data.copy_(Wq)
        evaluate(f"head_rtn_{bits}bit", "head", H_P=Wq.T.contiguous().float().cpu())
    lm_head_w.data.copy_(W_orig)
    handle.remove()

    # ── Outputs ─────────────────────────────────────────────────────────
    stem = f"{short}_{args.dataset}"
    with open(OUT_DIR / f"interventions_{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = [f"# E4 — single-axis interventions ({args.model}, {args.dataset})", "",
          f"K_feat={K_feat:.4f}, K_pred={K_pred:.4f}; identity alignment W=I; "
          f"n={args.num_samples} samples.", "",
          "| config | family | scale term | shape term | gamma | bound B "
          "| measured dR | bound holds |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['config']} | {r['family']} | {r['scale_term']:.3e} "
                  f"| {r['shape_term']:.3e} | {r['gamma']:.3e} "
                  f"| {r['bound']:.2f} | {r['|dR|']:.4f} "
                  f"| {'yes' if r['bound_holds'] else 'NO'} |")

    # Selectivity: per family, how much each axis moved vs the control row.
    ctrl = rows[0]
    md += ["", "## Selectivity (max |term - control| within each family)", "",
           "| family | scale term | shape term | gamma |", "|---|---|---|---|"]
    for fam in ("scale", "shape", "head"):
        sub = [r for r in rows if r["family"] == fam]
        if not sub:
            continue
        dscale = max(abs(r["scale_term"] - ctrl["scale_term"]) for r in sub)
        dshape = max(abs(r["shape_term"] - ctrl["shape_term"]) for r in sub)
        dgamma = max(abs(r["gamma"] - ctrl["gamma"]) for r in sub)
        md.append(f"| {fam}-only | {dscale:.3e} | {dshape:.3e} | {dgamma:.3e} |")
    md += ["", "Expected pattern: each family's own column dominates its row by "
           "orders of magnitude — that is the axis-identifiability claim.", ""]
    (OUT_DIR / f"E4_results_{stem}.md").write_text("\n".join(md))
    print("\n".join(md[-12:]))


if __name__ == "__main__":
    main()
