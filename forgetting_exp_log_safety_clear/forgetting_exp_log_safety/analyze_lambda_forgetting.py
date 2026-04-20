"""Analyze trace-norm regularization effect on forgetting.

Loads prism_forgetting_metrics_{task}.json across (lambda, model, target_task),
computes ΔR = loss_P - loss_T per (step, eval_task), aligns to a common last
step, and reports target improvement vs. downstream forgetting trade-off.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
LAMBDAS = ["0.0", "0.1", "0.5", "1.0"]
MODELS = ["llama", "qwen"]
TARGETS = ["bbq", "lima", "no_robots", "social_iqa", "truthfulqa"]
DOWNSTREAM = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]
ALIGN_STEP = 300  # common last step across most runs


def load_run(path: Path):
    with open(path) as f:
        d = json.load(f)
    trained = d["experiment"]["trained_task"]
    series = defaultdict(dict)  # eval_task -> {step: delta_risk_signed}
    for cp in d["checkpoints"]:
        step = cp["step"]
        for task, m in cp["tasks"].items():
            series[task][step] = m["loss_P"] - m["loss_T"]
    return trained, dict(series)


def collect():
    """Return table[lam][model][target] = {eval_task: {step: ΔR}}."""
    table = defaultdict(lambda: defaultdict(dict))
    for lam in LAMBDAS:
        for model in MODELS:
            d = ROOT / lam / model
            if not d.is_dir():
                continue
            for f in sorted(d.glob("prism_forgetting_metrics_*.json")):
                trained, series = load_run(f)
                table[lam][model][trained] = series
    return table


def at_step(series: dict, step: int):
    """Return ΔR at exact step, else last step <= step, else last available."""
    steps = sorted(series.keys())
    if not steps:
        return None, None
    if step in series:
        return series[step], step
    leq = [s for s in steps if s <= step]
    if leq:
        s = leq[-1]
        return series[s], s
    s = steps[-1]
    return series[s], s


def summarize(table):
    """Build rows: (model, target, lambda, Δtarget, avg_forget, max_forget, net, step_used)."""
    rows = []
    for model in MODELS:
        for target in TARGETS:
            for lam in LAMBDAS:
                runs = table[lam][model]
                if target not in runs:
                    rows.append((model, target, lam, None, None, None, None, None, "MISSING"))
                    continue
                series = runs[target]
                # target task
                tgt_series = series.get(target, {})
                d_tgt, s_tgt = at_step(tgt_series, ALIGN_STEP)
                # downstream
                forgets = []
                for ds in DOWNSTREAM:
                    ds_series = series.get(ds, {})
                    d_ds, _ = at_step(ds_series, ALIGN_STEP)
                    if d_ds is not None:
                        forgets.append(d_ds)
                if not forgets or d_tgt is None:
                    rows.append((model, target, lam, None, None, None, None, None, "EMPTY"))
                    continue
                avg_f = float(np.mean(forgets))
                max_f = float(np.max(forgets))
                net = (-d_tgt) - avg_f
                note = "" if s_tgt == ALIGN_STEP else f"step={s_tgt}"
                rows.append((model, target, lam, d_tgt, avg_f, max_f, net, s_tgt, note))
    return rows


def print_table(rows):
    hdr = f"{'model':<6} {'target':<12} {'λ':<5} {'Δtarget':>9} {'avgForget':>10} {'maxForget':>10} {'net':>8} {'step':>6}  note"
    print(hdr)
    print("-" * len(hdr))
    for model, target, lam, dt, af, mf, net, step, note in rows:
        if dt is None:
            print(f"{model:<6} {target:<12} {lam:<5} {'-':>9} {'-':>10} {'-':>10} {'-':>8} {'-':>6}  {note}")
            continue
        print(f"{model:<6} {target:<12} {lam:<5} {dt:>+9.4f} {af:>+10.4f} {mf:>+10.4f} {net:>+8.4f} {step:>6}  {note}")


def validate_hypotheses(rows):
    """H1: λ=0.0 target<0 & avg_forget>0.
       H2: avg_forget decreases with λ.
       H3: trade-off knee exists."""
    print("\n=== Hypothesis checks ===")
    by_mt = defaultdict(dict)  # (model, target) -> lam -> (dt, af)
    for model, target, lam, dt, af, mf, net, step, note in rows:
        if dt is None:
            continue
        by_mt[(model, target)][lam] = (dt, af, net, step)

    # H1
    print("\n[H1] At λ=0.0: target improves (Δ<0) and downstream degrades (avg_forget>0)?")
    for (model, target), lams in by_mt.items():
        if "0.0" not in lams:
            continue
        dt, af, net, step = lams["0.0"]
        target_ok = dt < 0
        forget_ok = af > 0
        flag = "OK" if (target_ok and forget_ok) else "WARN"
        print(f"  {model}/{target}: Δtarget={dt:+.4f} avg_forget={af:+.4f} -> {flag}")

    # H2
    print("\n[H2] avg_forget should decrease monotonically as λ increases:")
    for (model, target), lams in by_mt.items():
        seq = [(lam, lams[lam][1]) for lam in LAMBDAS if lam in lams]
        if len(seq) < 2:
            continue
        afs = [v for _, v in seq]
        diffs = np.diff(afs)
        mono = all(d <= 0 for d in diffs)
        trend = " -> ".join(f"{lam}:{af:+.3f}" for lam, af in seq)
        print(f"  {model}/{target}: {trend}  [{'monotonic' if mono else 'non-monotonic'}]")

    # H3 trade-off, knee: pick lambda with best net
    print("\n[H3] Best λ per (model, target) by net = (−Δtarget) − avg_forget:")
    best_counts = defaultdict(int)
    for (model, target), lams in by_mt.items():
        best_lam, best_net = None, -np.inf
        for lam, (dt, af, net, step) in lams.items():
            if net is not None and net > best_net:
                best_net, best_lam = net, lam
        if best_lam:
            best_counts[best_lam] += 1
            print(f"  {model}/{target}: best λ = {best_lam} (net={best_net:+.4f})")
    print(f"\n[H3 aggregate] λ winning counts: {dict(best_counts)}")


def plot_curves(table, out_path: Path):
    """Grid: rows=models, cols=targets. Each panel: ΔR vs step for all eval tasks & λ."""
    fig, axes = plt.subplots(len(MODELS), len(TARGETS), figsize=(24, 8), sharex=False)
    lam_colors = {"0.0": "#d62728", "0.1": "#ff7f0e", "0.5": "#2ca02c", "1.0": "#1f77b4"}

    for i, model in enumerate(MODELS):
        for j, target in enumerate(TARGETS):
            ax = axes[i, j]
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            for lam in LAMBDAS:
                runs = table[lam][model]
                if target not in runs:
                    continue
                series = runs[target]
                # target task: solid thick
                if target in series:
                    steps = sorted(series[target].keys())
                    vals = [series[target][s] for s in steps]
                    ax.plot(steps, vals, color=lam_colors[lam], lw=2.2, ls="-",
                            label=f"λ={lam} target" if i == 0 and j == 0 else None)
                # downstream: thin dashed, averaged
                agg = defaultdict(list)
                for ds in DOWNSTREAM:
                    if ds in series:
                        for s, v in series[ds].items():
                            agg[s].append(v)
                if agg:
                    steps = sorted(agg)
                    means = [np.mean(agg[s]) for s in steps]
                    ax.plot(steps, means, color=lam_colors[lam], lw=1.2, ls=":",
                            label=f"λ={lam} downstream avg" if i == 0 and j == 0 else None)
            ax.set_title(f"{model} / {target}")
            ax.set_xlabel("step")
            if j == 0:
                ax.set_ylabel("ΔR = loss_P − loss_T")
            ax.grid(alpha=0.3)
    # single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"\nSaved curve plot: {out_path}")


def check_lower_than_zero(table):
    """For every (model, target, eval_task), test whether ΔR at λ∈{0.1, 0.5} is
    lower than at λ=0.0, using two criteria:
      (A) end-point: at last common step ≤ ALIGN_STEP (300).
      (B) trajectory mean: mean ΔR over common steps in (0, ALIGN_STEP].
    Reports per (model, target) how many of 6 eval tasks satisfy each criterion,
    and prints offending cases."""
    compare_lams = ["0.1", "0.5"]
    print("\n=== Does λ∈{0.1, 0.5} lower ΔR vs λ=0.0? ===")
    print("(ΔR = loss_P − loss_T; lower = less forgetting on downstream, more improvement on target)")

    totals = {lam: {"A_ok": 0, "A_total": 0, "B_ok": 0, "B_total": 0} for lam in compare_lams}
    offenders = {lam: [] for lam in compare_lams}

    for model in MODELS:
        print(f"\n--- {model} ---")
        header = f"{'target':<12} {'eval':<11} {'λ=0.0 end':>10} {'λ=0.1 end':>10} {'λ=0.5 end':>10}  A(0.1) A(0.5)  B(0.1) B(0.5)"
        print(header)
        for target in TARGETS:
            runs_by_lam = {lam: table[lam][model].get(target, {}) for lam in ["0.0"] + compare_lams}
            if not runs_by_lam["0.0"]:
                continue
            eval_tasks = [target] + DOWNSTREAM
            for ev in eval_tasks:
                series = {lam: runs_by_lam[lam].get(ev, {}) for lam in ["0.0"] + compare_lams}
                if not series["0.0"]:
                    continue
                # end-point: last common step <= ALIGN_STEP
                common_end = None
                steps_0 = sorted(s for s in series["0.0"] if s <= ALIGN_STEP)
                for s in reversed(steps_0):
                    if all(s in series[lam] for lam in compare_lams if series[lam]):
                        common_end = s
                        break
                # trajectory common steps
                common_all = set(s for s in series["0.0"] if s <= ALIGN_STEP)
                for lam in compare_lams:
                    if series[lam]:
                        common_all &= set(series[lam].keys())
                common_all = sorted(common_all)

                vals_end = {lam: series[lam].get(common_end) if common_end is not None else None
                            for lam in ["0.0"] + compare_lams}
                mean_vals = {lam: (float(np.mean([series[lam][s] for s in common_all]))
                                    if common_all and series[lam] else None)
                              for lam in ["0.0"] + compare_lams}

                flags_A, flags_B = {}, {}
                for lam in compare_lams:
                    if not series[lam]:
                        flags_A[lam], flags_B[lam] = "--", "--"
                        continue
                    # (A) end-point
                    if vals_end[lam] is not None and vals_end["0.0"] is not None:
                        ok = vals_end[lam] < vals_end["0.0"]
                        totals[lam]["A_total"] += 1
                        totals[lam]["A_ok"] += int(ok)
                        flags_A[lam] = "✓" if ok else "✗"
                        if not ok:
                            offenders[lam].append(
                                (model, target, ev, "A",
                                 vals_end["0.0"], vals_end[lam],
                                 vals_end[lam] - vals_end["0.0"]))
                    else:
                        flags_A[lam] = "--"
                    # (B) trajectory mean
                    if mean_vals[lam] is not None and mean_vals["0.0"] is not None:
                        ok = mean_vals[lam] < mean_vals["0.0"]
                        totals[lam]["B_total"] += 1
                        totals[lam]["B_ok"] += int(ok)
                        flags_B[lam] = "✓" if ok else "✗"
                        if not ok:
                            offenders[lam].append(
                                (model, target, ev, "B",
                                 mean_vals["0.0"], mean_vals[lam],
                                 mean_vals[lam] - mean_vals["0.0"]))
                    else:
                        flags_B[lam] = "--"

                v0 = f"{vals_end['0.0']:+.3f}" if vals_end["0.0"] is not None else "   -  "
                v1 = f"{vals_end['0.1']:+.3f}" if vals_end["0.1"] is not None else "   -  "
                v5 = f"{vals_end['0.5']:+.3f}" if vals_end["0.5"] is not None else "   -  "
                tag = "*" if ev == target else " "
                print(f"{target:<12} {tag}{ev:<10} {v0:>10} {v1:>10} {v5:>10}   "
                      f"{flags_A.get('0.1','--'):>4}   {flags_A.get('0.5','--'):>4}   "
                      f"{flags_B.get('0.1','--'):>4}   {flags_B.get('0.5','--'):>4}")

    print("\n=== Summary (satisfaction rates) ===")
    for lam in compare_lams:
        t = totals[lam]
        if t["A_total"]:
            print(f"  λ={lam}: end-point (A)  {t['A_ok']}/{t['A_total']} "
                  f"= {100*t['A_ok']/t['A_total']:.1f}% lower than λ=0.0")
        if t["B_total"]:
            print(f"  λ={lam}: trajectory-mean (B)  {t['B_ok']}/{t['B_total']} "
                  f"= {100*t['B_ok']/t['B_total']:.1f}% lower than λ=0.0")

    print("\n=== Offenders (λ not lower than 0.0) ===")
    for lam in compare_lams:
        if not offenders[lam]:
            print(f"  λ={lam}: none")
            continue
        print(f"  λ={lam}:")
        for model, target, ev, crit, v0, vL, diff in offenders[lam]:
            tgt_flag = " (TARGET)" if ev == target else ""
            print(f"    {crit} {model}/{target}/{ev}{tgt_flag}: "
                  f"0.0={v0:+.4f}, {lam}={vL:+.4f}, diff={diff:+.4f}")

    # aggregate per (model, target): fraction of 6 eval tasks satisfied
    print("\n=== Per (model, target) satisfaction count on end-point (A) ===")
    print(f"{'model':<6} {'target':<12} {'λ=0.1 OK/total':>16} {'λ=0.5 OK/total':>16}")
    for model in MODELS:
        for target in TARGETS:
            runs_by_lam = {lam: table[lam][model].get(target, {}) for lam in ["0.0", "0.1", "0.5"]}
            if not runs_by_lam["0.0"]:
                continue
            eval_tasks = [target] + DOWNSTREAM
            cnt = {lam: [0, 0] for lam in ["0.1", "0.5"]}
            for ev in eval_tasks:
                s0 = runs_by_lam["0.0"].get(ev, {})
                if not s0:
                    continue
                # end common step
                steps_0 = sorted(s for s in s0 if s <= ALIGN_STEP)
                if not steps_0:
                    continue
                for lam in ["0.1", "0.5"]:
                    sL = runs_by_lam[lam].get(ev, {})
                    if not sL:
                        continue
                    common_end = None
                    for s in reversed(steps_0):
                        if s in sL:
                            common_end = s
                            break
                    if common_end is None:
                        continue
                    cnt[lam][1] += 1
                    if sL[common_end] < s0[common_end]:
                        cnt[lam][0] += 1
            print(f"{model:<6} {target:<12} "
                  f"{cnt['0.1'][0]}/{cnt['0.1'][1]:<14} "
                  f"{cnt['0.5'][0]}/{cnt['0.5'][1]:<14}")


def plot_grouped(table, out_dir: Path):
    """4 grouped figures. Each figure: rows = targets, cols = 6 eval tasks
    (row's target + 5 downstream). Subplot: x=step 0–300, y=ΔR, lines=λ.

      - combined_llama_truthfulqa_bbq.pdf   (2 rows)
      - combined_llama_others.pdf           (3 rows: lima, no_robots, social_iqa)
      - combined_qwen_truthfulqa_bbq.pdf    (2 rows)
      - combined_qwen_others.pdf            (3 rows)
    """
    out_dir.mkdir(exist_ok=True)
    lam_colors = {"0.0": "#d62728", "0.1": "#ff7f0e", "0.5": "#2ca02c", "1.0": "#1f77b4"}
    x_max = 300
    groups = [
        ("truthfulqa_bbq", ["truthfulqa", "bbq"]),
        ("others", ["lima", "no_robots", "social_iqa"]),
    ]

    for model in MODELS:
        for suffix, targets_in_group in groups:
            n_rows = len(targets_in_group)
            fig, axes = plt.subplots(n_rows, 6, figsize=(22, 3.1 * n_rows),
                                     sharex=True, squeeze=False)
            for r, target in enumerate(targets_in_group):
                eval_tasks = [target] + DOWNSTREAM
                for c, ev in enumerate(eval_tasks):
                    ax = axes[r, c]
                    ax.axhline(0, color="gray", lw=0.6, ls="--")
                    plotted = False
                    for lam in LAMBDAS:
                        runs = table[lam][model]
                        if target not in runs:
                            continue
                        ev_series = runs[target].get(ev, {})
                        if not ev_series:
                            continue
                        steps = sorted(s for s in ev_series if s <= x_max)
                        if not steps:
                            continue
                        vals = [ev_series[s] for s in steps]
                        ax.plot(steps, vals, color=lam_colors[lam], lw=1.7,
                                marker="o", markersize=3, label=f"λ={lam}")
                        plotted = True
                    is_target = (ev == target)
                    title = f"{ev}" + (" (target)" if is_target else "")
                    ax.set_title(title, fontsize=10,
                                 fontweight="bold" if is_target else "normal")
                    ax.grid(alpha=0.3)
                    ax.set_xlim(0, x_max)
                    if r == n_rows - 1:
                        ax.set_xlabel("step")
                    if c == 0:
                        ax.set_ylabel(f"{target}\nΔR = loss_P − loss_T", fontsize=10)
                    if not plotted:
                        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                                transform=ax.transAxes, color="gray")
            # single legend from first subplot that has data
            handles, labels = [], []
            for ax in axes.flat:
                h, l = ax.get_legend_handles_labels()
                if h:
                    handles, labels = h, l
                    break
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=4,
                           bbox_to_anchor=(0.5, 1.0 + 0.015 * (4 - n_rows)))
            fig.suptitle(f"{model}  —  targets: {', '.join(targets_in_group)}",
                         fontsize=13, y=1.0 + 0.02 * (4 - n_rows))
            plt.tight_layout()
            out = out_dir / f"combined_{model}_{suffix}.pdf"
            plt.savefig(out, bbox_inches="tight")
            plt.close()
            print(f"  saved {out.name}")


def plot_per_target(table, out_dir: Path):
    """One figure per (model, target). 6 subplots = 6 eval tasks.
    x = step in [0, 300]. y = ΔR = loss_P - loss_T. Lines = λ."""
    out_dir.mkdir(exist_ok=True)
    lam_colors = {"0.0": "#d62728", "0.1": "#ff7f0e", "0.5": "#2ca02c", "1.0": "#1f77b4"}
    x_max = 300

    for model in MODELS:
        for target in TARGETS:
            eval_tasks = [target] + DOWNSTREAM  # 6 panels, target first
            # check if any lambda has this run
            has_any = any(target in table[lam][model] for lam in LAMBDAS)
            if not has_any:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
            axes = axes.flatten()
            for k, ev in enumerate(eval_tasks):
                ax = axes[k]
                ax.axhline(0, color="gray", lw=0.6, ls="--")
                for lam in LAMBDAS:
                    runs = table[lam][model]
                    if target not in runs:
                        continue
                    ev_series = runs[target].get(ev, {})
                    if not ev_series:
                        continue
                    steps = sorted(s for s in ev_series if s <= x_max)
                    if not steps:
                        continue
                    vals = [ev_series[s] for s in steps]
                    ax.plot(steps, vals, color=lam_colors[lam], lw=1.8,
                            marker="o", markersize=3, label=f"λ={lam}")
                is_target = (ev == target)
                title = f"{ev}" + (" (target)" if is_target else "")
                ax.set_title(title, fontsize=11, fontweight="bold" if is_target else "normal")
                ax.grid(alpha=0.3)
                if k >= 3:
                    ax.set_xlabel("step")
                if k % 3 == 0:
                    ax.set_ylabel("ΔR = loss_P − loss_T")
                ax.set_xlim(0, x_max)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
            fig.suptitle(f"{model} / target={target}  (ΔR per eval task, step 0–{x_max})",
                         fontsize=13, y=1.06)
            plt.tight_layout()
            out = out_dir / f"per_target_{model}_{target}.pdf"
            plt.savefig(out, bbox_inches="tight")
            plt.close()
            print(f"  saved {out.name}")


def plot_tradeoff(rows, out_path: Path):
    """Per (model, target), x=avg_forget, y=−Δtarget, points colored by λ."""
    by_mt = defaultdict(dict)
    for model, target, lam, dt, af, mf, net, step, note in rows:
        if dt is None:
            continue
        by_mt[(model, target)][lam] = (dt, af)
    fig, axes = plt.subplots(len(MODELS), len(TARGETS), figsize=(24, 8), sharex=False, sharey=False)
    lam_colors = {"0.0": "#d62728", "0.1": "#ff7f0e", "0.5": "#2ca02c", "1.0": "#1f77b4"}
    for i, model in enumerate(MODELS):
        for j, target in enumerate(TARGETS):
            ax = axes[i, j]
            key = (model, target)
            if key not in by_mt:
                ax.set_title(f"{model}/{target} (no data)")
                continue
            xs, ys, lams_seq = [], [], []
            for lam in LAMBDAS:
                if lam in by_mt[key]:
                    dt, af = by_mt[key][lam]
                    xs.append(af)
                    ys.append(-dt)
                    lams_seq.append(lam)
            ax.plot(xs, ys, "-", color="gray", lw=0.8, alpha=0.6)
            for x, y, lam in zip(xs, ys, lams_seq):
                ax.scatter(x, y, s=80, color=lam_colors[lam], label=f"λ={lam}" if i == 0 and j == 0 else None, zorder=3)
                ax.annotate(lam, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
            ax.axhline(0, color="gray", lw=0.4, ls="--")
            ax.axvline(0, color="gray", lw=0.4, ls="--")
            ax.set_title(f"{model}/{target}")
            ax.set_xlabel("avg downstream forgetting →")
            if j == 0:
                ax.set_ylabel("target improvement (−Δ) ↑")
            ax.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved trade-off plot: {out_path}")


def recommend(rows):
    print("\n=== Recommendations ===")
    by_mt = defaultdict(dict)
    for model, target, lam, dt, af, mf, net, step, note in rows:
        if dt is None:
            continue
        by_mt[(model, target)][lam] = dict(dt=dt, af=af, mf=mf, net=net, step=step, note=note)

    # (1) ideal target tasks: λ=0.0 shows both strong target gain AND clear forgetting;
    #     and all four lambdas available for that target for BOTH models.
    print("\n(1) Target-task ranking (criterion: |Δtarget@λ=0| × avg_forget@λ=0, averaged across models, only if all 4 λ present):")
    scores = {}
    for target in TARGETS:
        ok = True
        contrast = []
        for model in MODELS:
            key = (model, target)
            if any(lam not in by_mt.get(key, {}) for lam in LAMBDAS):
                ok = False
                break
            d0 = by_mt[key]["0.0"]
            contrast.append((-d0["dt"]) * max(d0["af"], 0))
        if ok:
            scores[target] = float(np.mean(contrast))
    for target, s in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {target}: contrast score = {s:.4f}")
    missing = [t for t in TARGETS if t not in scores]
    if missing:
        print(f"  (excluded due to missing λ runs: {missing})")

    # (2) per-target best λ
    print("\n(2) Best λ per target (averaged net across models):")
    per_target = defaultdict(list)
    for (model, target), lams in by_mt.items():
        for lam, d in lams.items():
            per_target[(target, lam)].append(d["net"])
    target_best = {}
    for target in TARGETS:
        best_lam, best_net = None, -np.inf
        for lam in LAMBDAS:
            nets = per_target.get((target, lam), [])
            if len(nets) == len(MODELS):  # both models present
                m = float(np.mean(nets))
                if m > best_net:
                    best_net, best_lam = m, lam
        if best_lam:
            target_best[target] = (best_lam, best_net)
            print(f"  {target}: λ*={best_lam}, avg net={best_net:+.4f}")

    # (3) global best λ
    print("\n(3) Global λ recommendation (avg net across all (model, target) with all four λ):")
    glob = defaultdict(list)
    for (model, target), lams in by_mt.items():
        if any(lam not in lams for lam in LAMBDAS):
            continue
        for lam in LAMBDAS:
            glob[lam].append(lams[lam]["net"])
    for lam in LAMBDAS:
        if glob[lam]:
            print(f"  λ={lam}: avg net = {np.mean(glob[lam]):+.4f}  (n={len(glob[lam])})")


def main():
    table = collect()
    rows = summarize(table)
    print_table(rows)
    validate_hypotheses(rows)
    check_lower_than_zero(table)
    plot_curves(table, ROOT / "forgetting_curves.pdf")
    plot_tradeoff(rows, ROOT / "forgetting_tradeoff.pdf")
    print("\nPer-target figures:")
    plot_per_target(table, ROOT / "per_target_figs")
    print("\nGrouped figures:")
    plot_grouped(table, ROOT / "grouped_figs")
    recommend(rows)


if __name__ == "__main__":
    main()
