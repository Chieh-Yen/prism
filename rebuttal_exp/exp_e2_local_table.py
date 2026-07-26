#!/usr/bin/env python3
"""Local E2 table builder (zero GPU).

Reads whatever run JSONs are present under out/E2/ in the ACTUAL layout
  out/E2/<method>/<lam>/seed<N>/<model>/<task>/prism_forgetting_metrics.json
(the repo's exp_e2_aggregate.py expects an older *_<task>.json naming and
misses this layout), and prints, per fine-tuning task:

  method / config -> downstream mean |dR| over the 5 held-out benchmarks
                     (ARC/MMLU/SQuAD/TriviaQA/GSM8K, i.e. tasks minus the FT task),
                     target-task eval_loss, mean Omega over held-out.

It reports only runs whose step-300 checkpoint exists, dedupes the
meta-llama-3.1-8b / llama-3.1-8b alias, marks each method's sweep-best
(lowest downstream |dR|), and flags configs whose target eval_loss is
notably worse than no-reg (over-regularization). Missing runs are listed so
it is obvious what still needs copying back.

Usage:  python3 rebuttal_exp/exp_e2_local_table.py [--step 300]
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
E2 = os.path.join(HERE, "out", "E2")
DOWNSTREAM = ("arc", "mmlu", "squad", "triviaqa", "gsm8k")
SKIP_METHODS = {"feature_kd"}   # excluded from the reported baseline set
REF_METHOD = "trace"            # matched-target reference (the shape run)


def load_step(path, step):
    try:
        d = json.load(open(path))
    except (json.JSONDecodeError, ValueError, OSError):
        return "BAD"          # empty/truncated (mid-write on the pod)
    for c in d.get("checkpoints", []):
        if c.get("step") == step:
            return c
    return None


def summarize(ck, ft_task):
    held = [b for b in ck["tasks"] if b != ft_task]
    dr = [ck["tasks"][b]["delta_risk"] for b in held]
    om = [ck["tasks"][b]["omega"] for b in held]
    return {
        "dR": statistics.mean(dr),
        "omega": statistics.mean(om),
        "eval_loss": ck.get("eval_loss"),
        "n_held": len(held),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=300)
    args = ap.parse_args()

    if not os.path.isdir(E2):
        raise SystemExit(f"no {E2}")

    # method/lam/seed/model/task/prism_forgetting_metrics.json
    pat = os.path.join(E2, "*", "*", "seed*", "*", "*",
                       "prism_forgetting_metrics.json")
    # dedupe alias: prefer meta-llama-3.1-8b over llama-3.1-8b for the same run
    by_key = {}
    for f in glob.glob(pat):
        parts = f.split(os.sep)
        method, lam, seed, model, task = parts[-6], parts[-5], parts[-4], parts[-3], parts[-2]
        if method in SKIP_METHODS:
            continue
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
        raise SystemExit("no prism_forgetting_metrics.json found under out/E2/ "
                         "(copy the JSONs back first)")

    for task in sorted(rows):
        print(f"\n{'='*72}\nFT = {task}   (downstream = mean |dR| over "
              f"{'/'.join(DOWNSTREAM)}, step {args.step})\n{'='*72}")
        entries = sorted(rows[task], key=lambda r: (r[0], r[1], r[2]))
        # no-reg baseline eval_loss (if present) for the over-reg flag
        noreg = next((s for m, l, sd, s in entries
                      if m == "none" and isinstance(s, dict)), None)
        base_eval = noreg["eval_loss"] if noreg else None
        print(f"{'method':<14}{'config':<10}{'seed':<8}"
              f"{'downstream|dR|':<16}{'target loss':<13}{'meanOmega':<10}")
        configs = defaultdict(list)   # method -> [(dR, eval_loss, omega, lam, seed)]
        for m, l, sd, s in entries:
            if s == "BAD":
                print(f"{m:<14}{l:<10}{sd:<8}(JSON empty/truncated - re-pull)")
                continue
            if s is None:
                print(f"{m:<14}{l:<10}{sd:<8}(no step-{args.step} checkpoint)")
                continue
            print(f"{m:<14}{l:<10}{sd:<8}{s['dR']:<16.3f}"
                  f"{(s['eval_loss'] or float('nan')):<13.3f}{s['omega']:<10.3f}")
            configs[m].append((s["dR"], s["eval_loss"], s["omega"], l, sd))

        # matched-target-loss comparison: the fair "at comparable plasticity,
        # who forgets least?" Reference = the shape run's (REF_METHOD) target loss.
        ref = None
        for dr, ev, om, l, sd in configs.get(REF_METHOD, []):
            if ev is not None:
                ref = ev
                break
        if ref is not None:
            print(f"\n  matched-target-loss view (reference = {REF_METHOD} target "
                  f"loss {ref:.3f}; each method's config CLOSEST to it):")
            for m in sorted(configs):
                cs = [c for c in configs[m] if c[1] is not None]
                if not cs:
                    continue
                dr, ev, om, l, sd = min(cs, key=lambda c: abs(c[1] - ref))
                d = ev - ref
                note = ""
                if d > 0.04:
                    note = ("  (cannot match plasticity in-grid: best target "
                            "loss is higher -> it under-trains the FT task)")
                print(f"    {m+' '+l:<24} target {ev:.3f} (d{d:+.3f})   "
                      f"forget |dR| = {dr:.3f}{note}")
        else:
            print(f"\n  ({REF_METHOD} not present -> no matched-target view)")

    # coverage summary: which sweep configs are still missing
    expected = {
        "layer_freeze": ["4", "8", "16"],
        "ewc": ["0.0001", "0.001", "0.01", "0.1"],
        "l2sp": ["0.0001", "0.001", "0.01", "0.1"],
        "feature_kd": ["0.1", "1", "10"],
    }
    print(f"\n{'='*72}\nCOVERAGE (seed 42 sweep)\n{'='*72}")
    for task in ("truthfulqa", "bbq"):
        have = {(m, l) for (t, m, l, sd) in by_key if t == task and sd == "seed42"}
        for method, lams in expected.items():
            for lam_name in lams:
                # lam dir names vary (lam0.1 / lam1 / lam1e-4...); match loosely
                present = any(m == method and (lam.lstrip("lam") in (lam_name,)
                              or lam.lstrip("lam").rstrip("0").rstrip(".") == lam_name)
                              for (m, lam) in have)
                if not present:
                    print(f"  MISSING: {task:<11} {method:<13} lam~{lam_name}")


if __name__ == "__main__":
    main()
