#!/usr/bin/env python3
"""
E2 aggregator — collect every run produced by train_forgetting_baselines.py
(plus, optionally, the paper's own no-reg/trace runs) into the rebuttal
baseline table:

    per (model, FT task, method):
        sweep-best lambda (min mean downstream |dR| @ step 300, seed-mean)
        mean +/- sd downstream |dR| across seeds at best lambda
        % change vs no-reg
        target-task loss (plasticity check, eQL6-W4/S5)

Also writes out/E2/best_lambdas.env so script_E2.sh stage 2 (multi-seed)
can pick up the sweep winners automatically.

Zero GPU; stdlib only.

Usage:
    python rebuttal_exp/exp_e2_aggregate.py            # E2 runs only
    python rebuttal_exp/exp_e2_aggregate.py --with-paper-runs
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
E2_ROOT = HERE / "out" / "E2"
PAPER_ROOTS = [
    REPO / "exp_result" / "regularization",
    REPO / "forgetting_exp_log_safety_clear",
    REPO / "forgetting_exp_log_safety",
]
DOWNSTREAM = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]
ANALYSIS_STEP = 300


def norm_model(name: str) -> str:
    n = name.lower()
    if "llama" in n:
        return "llama"
    if "qwen" in n:
        return "qwen"
    return n


def checkpoint_at(data, step):
    best = None
    for ck in data["checkpoints"]:
        if ck["step"] <= step and (best is None or ck["step"] > best["step"]):
            best = ck
    return best


def summarize(path: Path):
    d = json.load(open(path))
    ck = checkpoint_at(d, ANALYSIS_STEP)
    if ck is None:
        return None
    ds = [ck["tasks"][t]["delta_risk"] for t in DOWNSTREAM if t in ck["tasks"]]
    ft = path.stem.replace("prism_forgetting_metrics_", "")
    return {
        "step": ck["step"],
        "mean_dR": statistics.mean(ds) if ds else float("nan"),
        "per_bench": {t: ck["tasks"][t]["delta_risk"]
                      for t in DOWNSTREAM if t in ck["tasks"]},
        "target_loss": ck["tasks"].get(ft, {}).get("loss_P"),
        "omega_mean": statistics.mean(
            [ck["tasks"][t]["omega"] for t in DOWNSTREAM if t in ck["tasks"]]),
    }


def collect_e2():
    """runs[(model, task, method, lam)] -> {seed: summary}"""
    runs = defaultdict(dict)
    if not E2_ROOT.is_dir():
        return runs
    for f in E2_ROOT.glob("*/*/seed*/*/*/prism_forgetting_metrics_*.json"):
        method = f.parents[4].name
        # sweep tag is "lam{value}" for penalty methods, "top{K}" for
        # layer_freeze (K = number of unfrozen top layers)
        lam = f.parents[3].name.removeprefix("lam").removeprefix("top")
        seed = f.parents[2].name.removeprefix("seed")
        model = norm_model(f.parents[1].name)
        task = f.stem.replace("prism_forgetting_metrics_", "")
        s = summarize(f)
        if s:
            runs[(model, task, method, lam)][seed] = s
    return runs


def collect_paper():
    runs = defaultdict(dict)
    seen = set()
    for root in PAPER_ROOTS:
        if not root.is_dir():
            continue
        for lam_dir in root.iterdir():
            if not lam_dir.is_dir():
                continue
            try:
                lam = f"{float(lam_dir.name):g}"
            except ValueError:
                continue
            for f in lam_dir.glob("*/prism_forgetting_metrics_*.json"):
                model = norm_model(f.parent.name)
                task = f.stem.replace("prism_forgetting_metrics_", "")
                method = "none(paper)" if lam == "0" else "trace(paper)"
                key = (model, task, method, lam)
                if key in seen:
                    continue
                seen.add(key)
                s = summarize(f)
                if s:
                    runs[key]["42"] = s
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-paper-runs", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="keep runs whose last checkpoint < analysis step "
                         "(default: drop them — they are aborted runs and "
                         "not comparable to step-300 anchors)")
    args = ap.parse_args()

    runs = collect_e2()
    if args.with_paper_runs:
        for k, v in collect_paper().items():
            runs.setdefault(k, v)
    if not args.allow_partial:
        dropped = []
        for key in list(runs):
            runs[key] = {s: v for s, v in runs[key].items()
                         if v["step"] == ANALYSIS_STEP}
            if not runs[key]:
                dropped.append(key)
                del runs[key]
        if dropped:
            print(f"[info] dropped {len(dropped)} partial runs "
                  f"(< step {ANALYSIS_STEP}): "
                  + ", ".join("/".join(k) for k in dropped[:6])
                  + (" ..." if len(dropped) > 6 else ""))
    if not runs:
        raise SystemExit("No runs found under rebuttal_exp/out/E2/ "
                         "(use --with-paper-runs to include paper trees).")

    # no-reg anchors per (model, task). Fresh pod-round 'none' takes
    # precedence over 'none(paper)': mixing a paper-round anchor with
    # pod-round method runs is a cross-round comparison (only valid once
    # the backfill canary certifies the rounds match — and even then the
    # same-round anchor is the cleaner default).
    noreg, noreg_key = {}, {}
    for pref in ("none", "none(paper)"):
        for (model, task, method, lam), seeds in runs.items():
            if method == pref and (model, task) not in noreg:
                vals = [s["mean_dR"] for s in seeds.values()]
                noreg[(model, task)] = statistics.mean(vals)
                noreg_key[(model, task)] = (model, task, method, lam)

    # sweep-best lambda per (model, task, method)
    best = {}
    for (model, task, method, lam), seeds in runs.items():
        if method in ("none", "none(paper)"):
            continue
        m = statistics.mean([s["mean_dR"] for s in seeds.values()])
        key = (model, task, method)
        if key not in best or m < best[key][1]:
            best[key] = (lam, m)

    md = ["# E2 — regularizer baseline table (auto-generated)",
          "",
          f"Downstream mean |dR| at step {ANALYSIS_STEP} over "
          f"{', '.join(DOWNSTREAM)}; lower is better. "
          "Target-task loss = plasticity check (lower is better).", ""]
    env_lines = []
    for (model, task) in sorted({(m, t) for (m, t, *_ ) in best} |
                                set(noreg)):
        md.append(f"\n## {model} / FT: {task}\n")
        md.append("| method | best lam | mean dR (seeds) | sd | n seeds "
                  "| vs no-reg | target loss |")
        md.append("|---|---|---|---|---|---|---|")
        base = noreg.get((model, task))
        if base is not None:
            # same source as the '% vs no-reg' baseline (fresh-first rule)
            key0 = noreg_key[(model, task)]
            seeds0 = runs[key0]
            tl = [s["target_loss"] for s in seeds0.values()
                  if s["target_loss"] is not None]
            src = key0[2]        # 'none' (fresh) or 'none(paper)' — shown
            md.append(f"| no-reg [{src}] | - | {base:.4f} | "
                      f"{statistics.pstdev([s['mean_dR'] for s in seeds0.values()]):.4f} "
                      f"| {len(seeds0)} | - | "
                      f"{statistics.mean(tl):.4f} |" if tl else
                      f"| no-reg [{src}] | - | {base:.4f} | - | {len(seeds0)} | - | - |")
        for (m2, t2, method), (lam, _) in sorted(best.items()):
            if (m2, t2) != (model, task):
                continue
            seeds = runs[(m2, t2, method, lam)]
            vals = [s["mean_dR"] for s in seeds.values()]
            mean, sd = statistics.mean(vals), statistics.pstdev(vals)
            rel = (f"{100 * (base - mean) / base:+.1f}%"
                   if base else "-")
            tl = [s["target_loss"] for s in seeds.values()
                  if s["target_loss"] is not None]
            tls = f"{statistics.mean(tl):.4f}" if tl else "-"
            md.append(f"| {method} | {lam} | {mean:.4f} | {sd:.4f} "
                      f"| {len(vals)} | {rel} | {tls} |")
            env_key = f"BEST_{method.upper()}_{model.upper()}_{task.upper()}"
            env_lines.append(f"{env_key.replace('(PAPER)', '_PAPER')}={lam}")

    E2_ROOT.mkdir(parents=True, exist_ok=True)
    (E2_ROOT / "E2_table.md").write_text("\n".join(md))
    (E2_ROOT / "best_lambdas.env").write_text("\n".join(env_lines) + "\n")
    print("\n".join(md))
    print(f"\n[written] {E2_ROOT / 'E2_table.md'}")
    print(f"[written] {E2_ROOT / 'best_lambdas.env'}")


if __name__ == "__main__":
    main()
