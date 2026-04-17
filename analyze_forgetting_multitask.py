#!/usr/bin/env python3
"""
Analyze PRISM forgetting experiment results.

Reads the JSON files produced by train_forgetting_multitask.py and generates:
  1. Per-model forgetting curves (Ω vs step, |ΔR| vs step)
  2. Cross-task forgetting matrix (heatmap at final checkpoint)
  3. Scatter plot data: PRISM bound vs empirical |ΔR| across all checkpoints
  4. Spearman rank correlation between bound and |ΔR|

Usage:
    python analyze_forgetting_multitask.py --results_dir checkpoints/forgetting_multitask

    # Single model:
    python analyze_forgetting_multitask.py \
        --json checkpoints/forgetting_multitask/llama-3.1-8b/gsm8k/prism_forgetting_metrics.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_results(json_path: str) -> Dict[str, Any]:
    with open(json_path) as f:
        return json.load(f)


def collect_all_results(results_dir: str) -> List[Dict[str, Any]]:
    """Find all prism_forgetting_metrics.json files under results_dir."""
    pattern = os.path.join(results_dir, "**", "prism_forgetting_metrics.json")
    paths = sorted(glob.glob(pattern, recursive=True))
    results = []
    for p in paths:
        data = load_results(p)
        data["_path"] = p
        results.append(data)
    return results


# ── Analysis functions ────────────────────────────────────────────────────

def print_forgetting_curves(data: Dict[str, Any]):
    """Print Ω and |ΔR| across training steps for each eval task."""
    exp = data["experiment"]
    model = exp["model"].split("/")[-1]
    trained = exp["trained_task"]
    checkpoints = data["checkpoints"]

    print(f"\n{'=' * 90}")
    print(f"  Forgetting Curves: {model} fine-tuned on {trained}")
    print(f"{'=' * 90}")

    tasks = list(checkpoints[0]["tasks"].keys())

    # Header
    print(f"\n  {'step':>6s}", end="")
    for t in tasks:
        marker = "*" if t == trained else " "
        print(f"  | {t}{marker}: {'Omega':>10s} {'|dR|':>8s} {'Bound':>8s}", end="")
    print()
    print(f"  {'─' * 6}", end="")
    for _ in tasks:
        print(f"  | {'─' * 34}", end="")
    print()

    for ckpt in checkpoints:
        step = ckpt["step"]
        print(f"  {step:6d}", end="")
        for t in tasks:
            r = ckpt["tasks"][t]
            omega = r["omega"]
            dr = r["delta_risk"]
            bound = r["bound_total"]
            bound_s = f"{bound:8.4f}" if bound is not None else "       —"
            print(f"  | {' ' * (len(t)+1)}  {omega:10.6f} {dr:8.4f} {bound_s}", end="")
        print()


def format_float(value: Any, width: int = 9, precision: int = 4) -> str:
    if value is None:
        return f"{'—':>{width}s}"
    return f"{float(value):>{width}.{precision}f}"


def format_bool(value: Any, width: int = 5) -> str:
    if value is True:
        return f"{'yes':>{width}s}"
    if value is False:
        return f"{'no':>{width}s}"
    return f"{'—':>{width}s}"


def print_task_checkpoint_tables(data: Dict[str, Any]):
    """Print checkpoint tables grouped by eval task for one fine-tune run."""
    exp = data["experiment"]
    model = exp["model"].split("/")[-1]
    trained = exp["trained_task"]
    checkpoints = data["checkpoints"]

    if not checkpoints:
        return

    tasks = list(checkpoints[0]["tasks"].keys())

    print(f"\n{'=' * 150}")
    print(f"  Checkpoint Summary by Task: {model} fine-tuned on {trained}")
    print(f"{'=' * 150}")

    for task in tasks:
        marker = " *" if task == trained else ""
        print(f"\n  Task: {task}{marker}")
        print(
            f"  {'Checkpoint':>10s}  {'ρ_T':>9s}  {'ρ_P':>9s}  {'Ω':>9s}  {'Scale':>9s}"
            f"  {'Shape':>10s}  {'δ':>9s}  {'γ':>9s}  {'Bound':>9s}"
            f"  {'Loss_T':>9s}  {'Loss_P':>9s}  {'|ΔR|':>9s}  {'Bound':>9s}  {'Holds':>5s}"
        )
        print(f"  {'─' * 140}")

        for ckpt in checkpoints:
            r = ckpt["tasks"][task]
            print(
                f"  {ckpt['step']:10d}"
                f"  {format_float(r.get('rho_T'), 9, 2)}"
                f"  {format_float(r.get('rho_P'), 9, 2)}"
                f"  {format_float(r.get('omega'), 9, 6)}"
                f"  {format_float(r.get('scale'), 9, 6)}"
                f"  {format_float(r.get('shape'), 10, 6)}"
                f"  {format_float(r.get('delta'), 9, 4)}"
                f"  {format_float(r.get('gamma'), 9, 4)}"
                f"  {format_float(r.get('bound_total'), 9, 4)}"
                f"  {format_float(r.get('loss_T'), 9, 4)}"
                f"  {format_float(r.get('loss_P'), 9, 4)}"
                f"  {format_float(r.get('delta_risk'), 9, 4)}"
                f"  {format_float(r.get('bound_total'), 9, 4)}"
                f"  {format_bool(r.get('bound_holds'), 5)}"
            )


def compute_correlations(all_data: List[Dict[str, Any]]):
    """Compute Spearman correlation between PRISM bound and |ΔR| across all checkpoints."""
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("  [scipy not installed — skipping correlation analysis]")
        return

    bounds, delta_risks = [], []
    bounds_by_model: Dict[str, tuple] = {}  # model -> (bounds_list, dr_list)

    for data in all_data:
        exp = data["experiment"]
        model = exp["model"].split("/")[-1]
        trained = exp["trained_task"]
        key = f"{model}/{trained}"

        if key not in bounds_by_model:
            bounds_by_model[key] = ([], [])

        for ckpt in data["checkpoints"]:
            for task, r in ckpt["tasks"].items():
                b = r.get("bound_total")
                dr = r.get("delta_risk")
                if b is not None and dr is not None:
                    bounds.append(b)
                    delta_risks.append(dr)
                    bounds_by_model[key][0].append(b)
                    bounds_by_model[key][1].append(dr)

    if len(bounds) < 3:
        print("  Not enough data points for correlation analysis.")
        return

    print(f"\n{'=' * 70}")
    print(f"  Spearman Rank Correlation: PRISM Bound vs |ΔR|")
    print(f"{'=' * 70}")

    # Overall
    rho, pval = spearmanr(bounds, delta_risks)
    n_holds = sum(1 for b, d in zip(bounds, delta_risks) if b >= d)
    print(f"\n  Overall (n={len(bounds)}):")
    print(f"    Spearman ρ = {rho:.4f}  (p = {pval:.2e})")
    print(f"    Bound holds: {n_holds}/{len(bounds)} ({100*n_holds/len(bounds):.1f}%)")

    # Per model/task
    print(f"\n  Per training run:")
    print(f"  {'Run':<30s}  {'n':>4s}  {'Spearman':>10s}  {'p-value':>10s}  {'Holds%':>7s}")
    print(f"  {'─' * 70}")
    for key in sorted(bounds_by_model):
        b_list, dr_list = bounds_by_model[key]
        if len(b_list) < 3:
            continue
        r, p = spearmanr(b_list, dr_list)
        nh = sum(1 for b, d in zip(b_list, dr_list) if b >= d)
        print(f"  {key:<30s}  {len(b_list):4d}  {r:10.4f}  {p:10.2e}  {100*nh/len(b_list):6.1f}%")


def export_csv(all_data: List[Dict[str, Any]], output_path: str):
    """Export all results to a flat CSV for external analysis."""
    import csv

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model", "trained_task", "step",
        "eval_task", "omega", "rho_T", "rho_P",
        "scale", "shape", "delta", "gamma",
        "bound_total", "loss_T", "loss_P", "delta_risk",
        "loss_T_full", "loss_P_full", "delta_risk_full",
        "bound_holds", "train_loss", "eval_loss",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for data in all_data:
            exp = data["experiment"]
            model = exp["model"].split("/")[-1]
            trained = exp["trained_task"]

            for ckpt in data["checkpoints"]:
                for task, r in ckpt["tasks"].items():
                    row = {
                        "model": model,
                        "trained_task": trained,
                        "step": ckpt["step"],
                        "eval_task": task,
                        "train_loss": ckpt.get("train_loss"),
                        "eval_loss": ckpt.get("eval_loss"),
                    }
                    for field in fieldnames:
                        if field in row:
                            continue
                        row[field] = r.get(field)
                    writer.writerow(row)

    print(f"  CSV exported: {output_path}  ({sum(len(d['checkpoints']) * 5 for d in all_data)} rows)")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Analyze PRISM forgetting experiment results")
    p.add_argument("--results_dir", type=str, default="checkpoints/forgetting_multitask",
                   help="Root directory containing prism_forgetting_metrics.json files")
    p.add_argument("--json", type=str, default=None,
                   help="Path to a single JSON file (overrides --results_dir)")
    p.add_argument("--csv", type=str, default=None,
                   help="Export all results to CSV at this path")
    args = p.parse_args()

    # Load results
    if args.json:
        all_data = [load_results(args.json)]
        all_data[0]["_path"] = args.json
    else:
        all_data = collect_all_results(args.results_dir)

    if not all_data:
        print(f"No results found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(all_data)} experiment(s):")
    for d in all_data:
        exp = d["experiment"]
        n_ckpt = len(d["checkpoints"])
        print(f"  {exp['model'].split('/')[-1]} / {exp['trained_task']}"
              f"  ({n_ckpt} checkpoints)  [{d['_path']}]")

    # Per-experiment analysis
    for data in all_data:
        print_task_checkpoint_tables(data)

    # Cross-experiment correlation
    compute_correlations(all_data)

    # CSV export
    if args.csv:
        export_csv(all_data, args.csv)
    else:
        default_csv = os.path.join(args.results_dir, "forgetting_all_results.csv")
        export_csv(all_data, default_csv)


if __name__ == "__main__":
    main()
