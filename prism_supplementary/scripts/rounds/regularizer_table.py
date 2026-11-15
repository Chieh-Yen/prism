#!/usr/bin/env python3
"""Table of the fixed-weight regularizer comparison of
run_regularizer_baselines.sh, from the run JSONs. No GPU.

The table of Appendix I of the paper is not printed by this script; it comes
from the unified protocol (scripts/response/forgetting/).

Reads the runs written by train_forgetting_baselines.py,

  <root>/<method>/<tag>/seed<N>/<model>/<task>/prism_forgetting_metrics.json

(<tag> is lam<lambda>, or top<K> for layer_freeze), and prints, per
fine-tuning task, at the checkpoint --step (default 300):

  runs        one line per (method, configuration, seed), for every run
              found; no configuration is left out:
                forgetting   mean |dR| over the downstream benchmarks (every
                             evaluated task except the fine-tuning task)
                target loss  the trainer's eval_loss on the fine-tuning task.
                             Its rows depend on how the run was started: the
                             whole held-out slice with --eval_keep_anchor_rows
                             (run_regularizer_baselines.sh passes it), the
                             slice without the anchor rows otherwise
                mean Omega   mean of Omega over the same benchmarks
  seeds       mean and SAMPLE standard deviation (``statistics.stdev``,
              divisor n - 1) of forgetting and of target loss over the seeds,
              computed from the unrounded per-seed values, in two blocks:
                * the six configurations of the `seeds` stage of
                  run_regularizer_baselines.sh (PRINTED below): no
                  regularization, anchor-CE lambda 0.01, L2-SP lambda 0.01,
                  EWC lambda 0.1, shape regularizer lambda 1, layer freezing
                  top 16;
                * every other configuration with at least two seeds.
  matched     among the runs of the sweep seed (42), for every method, the
              configuration whose target loss is closest to the target loss
              of the shape regularizer (`trace`) at that seed. The
              selection is applied to every configuration found, the whole
              sweep grid included.

Runs without a checkpoint at --step are listed as such.  When the same run
exists under two model directory names (meta-llama-3.1-8b and llama-3.1-8b
are the same weights), the one under the name starting with "meta-" is used.

One model per table: when the root holds runs of more than one model, the
script stops and asks for --model (the model directory name, e.g.
llama-3.1-8b or qwen3-8b-base).

A last block lists the configurations of the seed-42 sweep
(run_regularizer_baselines.sh) for which no run was found.

Usage:  python3 scripts/rounds/regularizer_table.py [--root DIR] [--step 300]
                                                    [--model NAME]
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # supplementary root
DEFAULT_ROOT = os.path.join(ROOT, "out", "rounds", "regularizers")
REF_METHOD = "trace"            # the shape regularizer: reference of the matched block
SWEEP_SEED = "seed42"           # seed of the sweep; the matched block uses it only
# (method, configuration) of the six configurations of the `seeds` stage of
# run_regularizer_baselines.sh.
PRINTED = [
    ("none", "lam0"),
    ("replay", "lam0.01"),
    ("l2sp", "lam0.01"),
    ("ewc", "lam0.1"),
    ("trace", "lam1"),
    ("layer_freeze", "top16"),
]
SWEEP = {                       # seed-42 sweep of run_regularizer_baselines.sh
    "layer_freeze": ["top4", "top8", "top16"],
    "ewc": ["lam0.0001", "lam0.001", "lam0.01", "lam0.1"],
    "l2sp": ["lam0.0001", "lam0.001", "lam0.01", "lam0.1"],
}


def load_step(path, step):
    try:
        d = json.load(open(path))
    except (json.JSONDecodeError, ValueError, OSError):
        return "BAD"          # empty or truncated file
    for c in d.get("checkpoints", []):
        if c.get("step") == step:
            # eval-row provenance, recorded by the trainer when present
            c = dict(c)
            c["_n_eval_rows"] = d.get("experiment", {}).get("n_eval_rows")
            return c
    return None


def model_name(directory):
    """Model directory name without a leading "meta-" (same weights)."""
    return directory[len("meta-"):] if directory.startswith("meta-") else directory


def summarize(ck, ft_task):
    held = [b for b in ck["tasks"] if b != ft_task]
    dr = [ck["tasks"][b]["delta_risk"] for b in held]
    om = [ck["tasks"][b]["omega"] for b in held]
    return {
        "dR": statistics.mean(dr),
        "omega": statistics.mean(om),
        "eval_loss": ck.get("eval_loss"),
        "n_eval_rows": ck.get("_n_eval_rows"),
        "n_held": len(held),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="output root of train_forgetting_baselines.py")
    ap.add_argument("--step", type=int, default=300)
    ap.add_argument("--model", default=None,
                    help="model directory name (e.g. llama-3.1-8b, qwen3-8b-base); "
                         "required when the root holds runs of more than one model")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(f"no {args.root}")

    pat = os.path.join(args.root, "*", "*", "seed*", "*", "*",
                       "prism_forgetting_metrics.json")
    files = glob.glob(pat)
    models = sorted({model_name(f.split(os.sep)[-3]) for f in files})
    if args.model is not None:
        wanted = model_name(args.model.lower())
        if wanted not in models:
            raise SystemExit(f"no run of model '{args.model}' under {args.root} "
                             f"(found: {', '.join(models) or 'none'})")
        files = [f for f in files if model_name(f.split(os.sep)[-3]) == wanted]
    elif len(models) > 1:
        raise SystemExit(f"{args.root} holds runs of more than one model "
                         f"({', '.join(models)}); pass --model NAME to print "
                         f"the table of one of them")
    by_key = {}
    for f in files:
        parts = f.split(os.sep)
        method, lam, seed, model, task = parts[-6], parts[-5], parts[-4], parts[-3], parts[-2]
        key = (task, method, lam, seed)
        prefer = model.startswith("meta-")
        if key not in by_key or (prefer and not by_key[key][1]):
            by_key[key] = (f, prefer)

    rows = defaultdict(list)   # task -> list of (method, lam, seed, summary|None)
    for (task, method, lam, seed), (f, _) in by_key.items():
        ck = load_step(f, args.step)
        if ck == "BAD":
            rows[task].append((method, lam, seed, "BAD"))
        else:
            rows[task].append((method, lam, seed,
                               summarize(ck, task) if ck else None))

    if not rows:
        raise SystemExit(f"no prism_forgetting_metrics.json found under {args.root}")

    for task in sorted(rows):
        print(f"\n{'='*72}\nfine-tuning task = {task}   (forgetting = mean |dR| over the "
              f"downstream benchmarks, step {args.step})\n{'='*72}")
        entries = sorted(rows[task], key=lambda r: (r[0], r[1], r[2]))
        print(f"{'method':<14}{'config':<12}{'seed':<8}"
              f"{'forgetting':<12}{'target loss':<13}{'mean Omega':<10}")
        configs = defaultdict(list)   # method -> [(dR, eval_loss, omega, lam, seed)]
        for m, l, sd, s in entries:
            if s == "BAD":
                print(f"{m:<14}{l:<12}{sd:<8}(JSON empty or truncated)")
                continue
            if s is None:
                print(f"{m:<14}{l:<12}{sd:<8}(no step-{args.step} checkpoint)")
                continue
            print(f"{m:<14}{l:<12}{sd:<8}{s['dR']:<12.3f}"
                  f"{(s['eval_loss'] or float('nan')):<13.3f}{s['omega']:<10.3f}")
            configs[m].append((s["dR"], s["eval_loss"], s["omega"], l, sd))

        per_cfg = defaultdict(list)   # (method, config) -> [(dR, eval_loss)]
        for m in configs:
            for dr, ev, om, l, sd in configs[m]:
                per_cfg[(m, l)].append((dr, ev))

        def seed_line(m, l):
            vals = per_cfg[(m, l)]
            dr = [v[0] for v in vals]
            ev = [v[1] for v in vals if v[1] is not None]
            dr_txt = (f"{statistics.mean(dr):.3f} ± {statistics.stdev(dr):.3f}"
                      if len(dr) >= 2 else f"{dr[0]:.3f} (one seed)")
            ev_txt = (f"{statistics.mean(ev):.3f} ± {statistics.stdev(ev):.3f}"
                      if len(ev) >= 2 else "-")
            print(f"    {m:<14}{l:<12}{len(vals):<7}{dr_txt:<18}{ev_txt}")

        hdr = f"    {'method':<14}{'config':<12}{'seeds':<7}{'forgetting':<18}{'target loss':<18}"
        print("\n  configurations of the seeds stage: "
              "mean ± sample sd (divisor n - 1) over seeds")
        print(hdr)
        for m, l in PRINTED:
            if (m, l) in per_cfg:
                seed_line(m, l)
            else:
                print(f"    {m:<14}{l:<12}(no run found)")
        others = sorted(k for k in per_cfg if k not in PRINTED and len(per_cfg[k]) >= 2)
        if others:
            print("\n  other configurations with at least two seeds: "
                  "mean ± sample sd (divisor n - 1) over seeds")
            print(hdr)
            for m, l in others:
                seed_line(m, l)

        # Matched block: the sweep seed only.
        sweep = {m: [c for c in configs[m] if c[4] == SWEEP_SEED and c[1] is not None]
                 for m in configs}
        refs = sweep.get(REF_METHOD, [])
        if len(refs) == 1:
            ref = refs[0][1]
            print(f"\n  matched target loss, {SWEEP_SEED} runs only (reference = {REF_METHOD} "
                  f"{refs[0][3]}, target loss {ref:.3f}; per method, the configuration "
                  f"closest to it):")
            for m in sorted(sweep):
                if not sweep[m]:
                    continue
                dr, ev, om, l, sd = min(sweep[m], key=lambda c: abs(c[1] - ref))
                print(f"    {m+' '+l:<24} {sd:<8} target loss {ev:.3f} ({ev - ref:+.3f})   "
                      f"forgetting {dr:.3f}")
        elif not refs:
            print(f"\n  (no {SWEEP_SEED} run of {REF_METHOD}: no matched block)")
        else:
            print(f"\n  ({len(refs)} {SWEEP_SEED} configurations of {REF_METHOD} found "
                  f"({', '.join(c[3] for c in refs)}): the reference of the matched "
                  f"block is not unique, no matched block)")

        n_rows = {s["n_eval_rows"] for m, l, sd, s in entries
                  if isinstance(s, dict) and s.get("n_eval_rows") is not None}
        if len(n_rows) > 1:
            print(f"\n  NOTE: the runs above record different numbers of evaluation rows "
                  f"({sorted(n_rows)}): some were started with --eval_keep_anchor_rows "
                  f"and some without, so their target losses are not on the same rows.")
        elif len(n_rows) == 1:
            print(f"\n  target loss: every run that records it was evaluated on "
                  f"{n_rows.pop()} rows.")

    print(f"\n{'='*72}\nseed-42 sweep configurations without a run\n{'='*72}")
    n_missing = 0
    for task in sorted(rows):
        have = {(m, l) for (t, m, l, sd) in by_key if t == task and sd == "seed42"}
        for method, tags in SWEEP.items():
            for tag in tags:
                if (method, tag) not in have:
                    n_missing += 1
                    print(f"  missing: {task:<11} {method:<13} {tag}")
    if not n_missing:
        print("  none")


if __name__ == "__main__":
    main()
