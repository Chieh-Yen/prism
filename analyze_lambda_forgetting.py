"""Analyze regularization effect on LoRA forgetting (post gradient-bug fix).

Loads prism_forgetting_metrics_{task}.json across (config, model, target_task),
where each config is (method, λ) drawn from {trace, replay}, plus a single
"baseline" config representing **no regularization** (sourced from
regularization_replay/0.0/, which is identical to trace/0.0/ — both reduce to
plain LoRA without any regularization penalty).

Naming convention:
  - "baseline" or "no reg"  ←  the run with no regularization (replay λ=0.0)
                              We avoid calling this "λ=0" because λ=0 is
                              method-specific (trace λ=0 or replay λ=0); both
                              are the SAME experiment, and the unified label
                              "no reg" makes the cross-method comparison clear.
  - "trace λ=X"             ←  trace-norm shape regularizer with weight X
  - "replay λ=X"            ←  replay-CE baseline with weight X

Data paths:
  regularization_exp/exp_result/regularization/<lam>/...        (trace)
  regularization_exp/exp_result/regularization_replay/<lam>/... (replay; 0.0 = baseline)

Outputs:
  paper/figures/forgetting/combined_<model>_truthfulqa_bbq.pdf  (paper Fig 4 source)
  paper/figures/forgetting/forgetting_curves_reg.pdf
  paper/figures/forgetting/forgetting_tradeoff_reg.pdf
  paper/figures/forgetting/per_target_figs/per_target_<model>_<target>.pdf
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════
ROOT        = Path(__file__).resolve().parent
DATA_TRACE  = ROOT / "regularization_exp" / "exp_result" / "regularization"
DATA_REPLAY = ROOT / "regularization_exp" / "exp_result" / "regularization_replay"
FIG_DIR     = ROOT / "paper" / "figures" / "forgetting"

ALIGN_STEP  = 300
MODELS      = ["llama", "qwen"]
TARGETS     = ["truthfulqa", "bbq"]
DOWNSTREAM  = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]

# ═══════════════════════════════════════════════════════════════════
# Config registry
# Each config: (key, method, lam, label, color, marker)
# - key       : short id used as dict key (e.g., "baseline", "trace_1.0")
# - method    : 'baseline' | 'trace' | 'replay'
# - lam       : λ value as string (informational; for baseline this is the
#               source λ "0.0" of replay/0.0)
# - label     : display name in tables / legends
# - color     : plot color
# - marker    : plot marker
# ═══════════════════════════════════════════════════════════════════
ALL_CONFIGS = [
    # baseline (no regularization)
    ("baseline",      "baseline", "0.0",   "no reg",          "#7f7f7f", "o"),
    # trace-norm sweep
    ("trace_0.01",    "trace",    "0.01",  "trace λ=0.01",    "#1a9850", "v"),
    ("trace_0.05",    "trace",    "0.05",  "trace λ=0.05",    "#66bd63", "^"),
    ("trace_0.1",     "trace",    "0.1",   "trace λ=0.1",     "#a6d96a", "<"),
    ("trace_0.5",     "trace",    "0.5",   "trace λ=0.5",     "#fdae61", ">"),
    ("trace_1.0",     "trace",    "1.0",   "trace λ=1.0",     "#d73027", "D"),
    # replay sweep (excludes 0.0 since that IS the baseline)
    ("replay_0.001",  "replay",   "0.001", "replay λ=0.001",  "#deebf7", "v"),
    ("replay_0.005",  "replay",   "0.005", "replay λ=0.005",  "#9ecae1", "^"),
    ("replay_0.01",   "replay",   "0.01",  "replay λ=0.01",   "#6baed6", "<"),
    ("replay_0.05",   "replay",   "0.05",  "replay λ=0.05",   "#3182bd", ">"),
    ("replay_0.1",    "replay",   "0.1",   "replay λ=0.1",    "#08519c", "D"),
]

# Configs used in the paper-grade combined figure (Fig 4 in paper).
# Recommended pair: baseline + replay 0.01 + trace 1.0
#   - Each near per-method |ΔR|-best across the sweep:
#     * Replay λ=0.01 has the lowest 4-cell mean |ΔR| (0.349) among replay
#       λ ∈ {0.001, 0.005, 0.01, 0.05, 0.1}; replay degrades for λ > 0.01.
#     * Trace λ=1.0 has the lowest 4-cell mean |ΔR| (0.314) and is also the
#       sweep maximum; trace decreases monotonically with λ on Llama.
#   - λ ratio is 100× (2 orders of magnitude); we frame as "each method at
#     its sweep-best" rather than as a fixed ratio.
#
# Scope of the paper claim (Fig 4 caption, neurips_2026.tex line 384):
#   "trace cuts downstream MEAN |ΔR| further than the replay baseline"
# This is a mean-level claim across the 2x5 (target x downstream) grid for
# each fine-tuning task; per-cell wins are mixed (trace wins 7/10 across the
# Llama TruthfulQA-FT + BBQ-FT cells) but trace's wins are larger in
# magnitude than replay's, so the mean comparison is unambiguous in trace's
# favor. Per-cell numbers are in the appendix tables (table_compare_*.tex).
PLOT_CONFIGS = [
    ("baseline",    "baseline", "0.0",  "no reg",                  "#7f7f7f", "o"),
    ("replay_0.01", "replay",   "0.01", "replay λ=0.01 (baseline)", "#1f77b4", "s"),
    ("trace_1.0",   "trace",    "1.0",  "trace λ=1.0 (ours)",       "#d62728", "^"),
]

# Configs used in fast hypothesis checks / trade-off scatter
ANALYSIS_CONFIGS = ALL_CONFIGS  # full sweep


# ═══════════════════════════════════════════════════════════════════
# Path resolution + loading
# ═══════════════════════════════════════════════════════════════════
def get_json_path(method: str, lam: str, model: str, ft_task: str) -> Path:
    """Resolve path; baseline maps to replay/0.0."""
    if method == "baseline":
        return DATA_REPLAY / "0.0" / model / f"prism_forgetting_metrics_{ft_task}.json"
    if method == "trace":
        return DATA_TRACE / lam / model / f"prism_forgetting_metrics_{ft_task}.json"
    if method == "replay":
        return DATA_REPLAY / lam / model / f"prism_forgetting_metrics_{ft_task}.json"
    raise ValueError(f"Unknown method: {method!r}")


def load_run(path: Path):
    """Return (trained_task, series_dict). series[eval]={step: ΔR (signed)}."""
    if not path.exists():
        return None, None
    with open(path) as f:
        d = json.load(f)
    trained = d["experiment"]["trained_task"]
    series = defaultdict(dict)
    for cp in d["checkpoints"]:
        step = cp["step"]
        for task, m in cp["tasks"].items():
            lt = m.get("loss_T")
            lp = m.get("loss_P")
            if lt is None or lp is None:
                continue
            series[task][step] = lp - lt
    return trained, dict(series)


def collect():
    """table[config_key][model][target] = {eval_task: {step: signed ΔR}}."""
    table: dict = defaultdict(lambda: defaultdict(dict))
    for key, method, lam, *_ in ALL_CONFIGS:
        for model in MODELS:
            for target in TARGETS:
                p = get_json_path(method, lam, model, target)
                trained, series = load_run(p)
                if series is None:
                    continue
                table[key][model][target] = series
    return table


def at_step(series: dict, step: int):
    """Return (ΔR at exact step, step_used). Falls back to last step ≤ step."""
    if not series:
        return None, None
    if step in series:
        return series[step], step
    leq = [s for s in series.keys() if s <= step]
    if leq:
        s = max(leq)
        return series[s], s
    s = max(series.keys())
    return series[s], s


# ═══════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════
def summarize(table):
    """rows: (model, target, config_key, label, Δtarget, avgFor, maxFor, net, step, note)."""
    config_lookup = {c[0]: c for c in ALL_CONFIGS}
    rows = []
    for model in MODELS:
        for target in TARGETS:
            for key, method, lam, label, *_ in ALL_CONFIGS:
                runs = table[key][model]
                if target not in runs:
                    rows.append((model, target, key, label, None, None, None, None, None, "MISSING"))
                    continue
                series = runs[target]
                # target-task ΔR
                tgt_series = series.get(target, {})
                d_tgt, s_tgt = at_step(tgt_series, ALIGN_STEP)
                # downstream
                forgets = []
                for ds in DOWNSTREAM:
                    d_ds, _ = at_step(series.get(ds, {}), ALIGN_STEP)
                    if d_ds is not None:
                        forgets.append(d_ds)
                if d_tgt is None or not forgets:
                    rows.append((model, target, key, label, None, None, None, None, None, "EMPTY"))
                    continue
                avg_f = float(np.mean(forgets))
                max_f = float(np.max(forgets))
                net = (-d_tgt) - avg_f
                note = "" if s_tgt == ALIGN_STEP else f"step={s_tgt}"
                rows.append((model, target, key, label, d_tgt, avg_f, max_f, net, s_tgt, note))
    return rows


def print_table(rows):
    hdr = (f"{'model':<6} {'target':<12} {'config':<18} "
           f"{'Δtarget':>9} {'avgForget':>10} {'maxForget':>10} "
           f"{'net':>8} {'step':>6}  note")
    print(hdr)
    print("-" * len(hdr))
    for model, target, key, label, dt, af, mf, net, step, note in rows:
        if dt is None:
            print(f"{model:<6} {target:<12} {label:<18} "
                  f"{'-':>9} {'-':>10} {'-':>10} {'-':>8} {'-':>6}  {note}")
            continue
        print(f"{model:<6} {target:<12} {label:<18} "
              f"{dt:>+9.4f} {af:>+10.4f} {mf:>+10.4f} {net:>+8.4f} "
              f"{step:>6}  {note}")


# ═══════════════════════════════════════════════════════════════════
# Hypothesis checks (adapted for two-method comparison)
# ═══════════════════════════════════════════════════════════════════
def validate_hypotheses(rows):
    """H1 (baseline behavior): no-reg shows target<0 & avg_forget>0.
       H2 (monotonicity per method): avg_forget decreases with λ.
       H3 (cross-method): trace strongest λ vs replay strongest λ."""
    print("\n=== Hypothesis checks ===")
    by_mt = defaultdict(dict)
    for model, target, key, label, dt, af, mf, net, step, note in rows:
        if dt is None:
            continue
        by_mt[(model, target)][key] = (label, dt, af, net, step)

    # H1: baseline
    print("\n[H1] no reg: target improves (Δ<0) and downstream degrades (avg_forget>0)?")
    for (model, target), cfgs in by_mt.items():
        if "baseline" not in cfgs:
            continue
        _label, dt, af, net, step = cfgs["baseline"]
        target_ok = dt < 0
        forget_ok = af > 0
        flag = "OK" if (target_ok and forget_ok) else "WARN"
        print(f"  {model}/{target}: Δtarget={dt:+.4f} avg_forget={af:+.4f} -> {flag}")

    # H2: monotonic decrease per method
    print("\n[H2] avg_forget should decrease as λ increases (within each method):")
    for method in ("trace", "replay"):
        method_cfgs = [c for c in ALL_CONFIGS if c[1] == method]
        method_cfgs.sort(key=lambda c: float(c[2]))
        for (model, target), cfgs in by_mt.items():
            seq = [(c[3], cfgs[c[0]][2]) for c in method_cfgs if c[0] in cfgs]
            if len(seq) < 2:
                continue
            afs = [v for _, v in seq]
            mono = all(d <= 0 for d in np.diff(afs))
            trend = " -> ".join(f"{lab.split()[1]}:{af:+.3f}" for lab, af in seq)
            tag = "monotonic" if mono else "non-monotonic"
            print(f"  [{method:6s}] {model}/{target}: {trend}  [{tag}]")

    # H3: trace best vs replay best
    print("\n[H3] Best config per (model, target) by net = (−Δtarget) − avg_forget:")
    win_by_method = defaultdict(int)
    for (model, target), cfgs in by_mt.items():
        best_key, best_label, best_net = None, None, -np.inf
        for k, (lab, dt, af, net, step) in cfgs.items():
            if net > best_net:
                best_net, best_key, best_label = net, k, lab
        if best_key:
            method = best_key.split("_")[0] if "_" in best_key else best_key
            win_by_method[method] += 1
            print(f"  {model}/{target}: best = {best_label}  (net={best_net:+.4f})")
    print(f"\n[H3 aggregate] Wins by method: {dict(win_by_method)}")


# ═══════════════════════════════════════════════════════════════════
# Lower-than-baseline check (trace vs replay vs baseline)
# ═══════════════════════════════════════════════════════════════════
def check_lower_than_baseline(table, compare_keys=None):
    """For every (model, target, eval_task), test whether ΔR at each compare_key
    is lower than at baseline (no reg).  Two criteria:
      (A) end-point at last common step ≤ ALIGN_STEP
      (B) trajectory mean over common steps in (0, ALIGN_STEP]
    """
    if compare_keys is None:
        compare_keys = [c[0] for c in ALL_CONFIGS if c[0] != "baseline"]
    print("\n=== Does each config lower ΔR vs baseline (no reg)? ===")
    print("(ΔR = loss_P − loss_T; lower = less forgetting / more target gain)")

    label_lookup = {c[0]: c[3] for c in ALL_CONFIGS}
    totals = {k: {"A_ok": 0, "A_total": 0, "B_ok": 0, "B_total": 0}
              for k in compare_keys}
    offenders = {k: [] for k in compare_keys}

    for model in MODELS:
        for target in TARGETS:
            base_runs = table["baseline"][model].get(target, {})
            if not base_runs:
                continue
            eval_tasks = [target] + DOWNSTREAM
            for ev in eval_tasks:
                base_series = base_runs.get(ev, {})
                if not base_series:
                    continue
                steps_b = sorted(s for s in base_series if s <= ALIGN_STEP)
                for ck in compare_keys:
                    cmp_series = (table[ck][model].get(target, {}) or {}).get(ev, {})
                    if not cmp_series:
                        continue
                    # (A) end-point common step
                    common_end = None
                    for s in reversed(steps_b):
                        if s in cmp_series:
                            common_end = s
                            break
                    if common_end is not None:
                        b = base_series[common_end]
                        c = cmp_series[common_end]
                        ok = c < b
                        totals[ck]["A_total"] += 1
                        totals[ck]["A_ok"] += int(ok)
                        if not ok:
                            offenders[ck].append(
                                (model, target, ev, "A", b, c, c - b))
                    # (B) trajectory mean over common steps
                    common_all = sorted(set(s for s in steps_b
                                            if s in cmp_series))
                    if common_all:
                        b_mean = float(np.mean([base_series[s] for s in common_all]))
                        c_mean = float(np.mean([cmp_series[s] for s in common_all]))
                        ok = c_mean < b_mean
                        totals[ck]["B_total"] += 1
                        totals[ck]["B_ok"] += int(ok)
                        if not ok:
                            offenders[ck].append(
                                (model, target, ev, "B", b_mean, c_mean,
                                 c_mean - b_mean))

    print("\n=== Summary (satisfaction rates: % of cells where config beats baseline) ===")
    for ck in compare_keys:
        t = totals[ck]
        lab = label_lookup[ck]
        if t["A_total"]:
            print(f"  {lab:<18} end-point (A)       "
                  f"{t['A_ok']}/{t['A_total']} = "
                  f"{100*t['A_ok']/t['A_total']:.1f}%")
        if t["B_total"]:
            print(f"  {lab:<18} trajectory-mean (B) "
                  f"{t['B_ok']}/{t['B_total']} = "
                  f"{100*t['B_ok']/t['B_total']:.1f}%")

    print("\n=== Offenders (config NOT lower than baseline) ===")
    for ck in compare_keys:
        if not offenders[ck]:
            print(f"  {label_lookup[ck]}: none")
            continue
        print(f"  {label_lookup[ck]}:")
        for model, target, ev, crit, b, c, diff in offenders[ck]:
            tgt_flag = " (TARGET)" if ev == target else ""
            print(f"    {crit} {model}/{target}/{ev}{tgt_flag}: "
                  f"baseline={b:+.4f}, {label_lookup[ck]}={c:+.4f}, "
                  f"diff={diff:+.4f}")


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════
TASK_DISPLAY = {
    "bbq": "BBQ", "truthfulqa": "TruthfulQA",
    "arc": "ARC", "mmlu": "MMLU", "squad": "SQuAD",
    "triviaqa": "TriviaQA", "gsm8k": "GSM8K",
}
MODEL_DISPLAY = {"llama": "Llama-3.1-8B", "qwen": "Qwen3-8B-Base"}
ROW_DISPLAY = {k: f"FT: {v}" for k, v in TASK_DISPLAY.items()}


def plot_combined(table, plot_configs, out_dir: Path):
    """Per (model): one figure with rows = targets (truthfulqa, bbq),
    cols = 6 eval tasks (target + 5 downstream). Lines = configs.
    This is the paper Fig 4 source."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x_max = ALIGN_STEP

    for model in MODELS:
        nrow, ncol = len(TARGETS), 6
        legend_reserve_inch = 2.0
        fig_height = nrow * 4.6 + legend_reserve_inch
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(ncol * 6.0, fig_height),
                                 squeeze=False)
        top_frac = (fig_height - legend_reserve_inch) / fig_height
        plt.subplots_adjust(hspace=0.28, wspace=0.22, top=top_frac,
                            bottom=0.14, left=0.07, right=0.985)

        seen_keys = []
        for ri, target in enumerate(TARGETS):
            eval_tasks = [target] + DOWNSTREAM
            for ci, ev in enumerate(eval_tasks):
                ax = axes[ri][ci]
                ax.axhline(0, color="black", ls="--", lw=2.6, alpha=0.8,
                           zorder=2)
                plotted = False
                for li, (key, method, lam, label, color, marker) in enumerate(plot_configs):
                    runs = table[key][model]
                    if target not in runs:
                        continue
                    ev_series = runs[target].get(ev, {})
                    if not ev_series:
                        continue
                    steps = sorted(s for s in ev_series if s <= x_max)
                    if not steps:
                        continue
                    vals = [ev_series[s] for s in steps]
                    ax.plot(steps, vals, color=color, marker=marker,
                            linestyle="-", markersize=16,
                            markeredgecolor="k", markeredgewidth=1.1,
                            lw=3.6, alpha=0.88, zorder=3 + li,
                            clip_on=False)
                    plotted = True
                    if key not in seen_keys:
                        seen_keys.append(key)

                ax.set_xlim(0, x_max)
                ax.tick_params(labelsize=24)
                ax.tick_params(axis="both", which="minor", length=0)
                ax.grid(True, which="major", ls=":", alpha=0.35)

                if ri == 0:
                    col_title = "Fine-tune Dataset" if ci == 0 else TASK_DISPLAY[ev]
                    ax.set_title(col_title, fontsize=38,
                                 fontweight="bold", pad=15)
                if ci == 0:
                    ax.set_ylabel(
                        ROW_DISPLAY[target] + "\n" + r"$\Delta\mathcal{R}$",
                        fontsize=30, fontweight="bold", labelpad=8)
                    ax.yaxis.set_label_coords(-0.26, 0.5)
                if ri == nrow - 1:
                    ax.set_xlabel("Training Step", fontsize=30, labelpad=8)

                if not plotted:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, color="0.6", fontsize=25)

        # Legend
        legend_entries = []
        for (key, method, lam, label, color, marker) in plot_configs:
            if key not in seen_keys:
                continue
            legend_entries.append((
                Line2D([0], [0], marker=marker, color=color, linestyle="-",
                       markerfacecolor=color, markeredgecolor="k",
                       markeredgewidth=1.1, markersize=20, lw=3.8),
                label))
        legend_entries.append((
            Line2D([0], [0], color="black", ls="--", lw=2.6, alpha=0.8),
            r"$\Delta\mathcal{R}=0$"))
        handles, labels = zip(*legend_entries)
        leg = fig.legend(handles, labels, loc="upper center",
                         bbox_to_anchor=(0.52, 0.998),
                         ncol=min(len(labels), 7), fontsize=30,
                         frameon=True, fancybox=True, edgecolor="black",
                         handletextpad=0.6, columnspacing=1.8, borderpad=0.6)
        leg.get_frame().set_linewidth(0.8)

        out = out_dir / f"combined_{model}_truthfulqa_bbq.pdf"
        fig.savefig(str(out), format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out.name}")


def plot_curves(table, configs, out_path: Path):
    """Diagnostic: rows=models, cols=targets. Each panel: ΔR vs step
    for target (solid) and downstream avg (dashed), lines=configs."""
    fig, axes = plt.subplots(len(MODELS), len(TARGETS),
                             figsize=(18, 10), sharex=False)
    if len(TARGETS) == 1:
        axes = axes.reshape(-1, 1)
    for i, model in enumerate(MODELS):
        for j, target in enumerate(TARGETS):
            ax = axes[i, j]
            ax.axhline(0, color="gray", lw=1.6, ls="--")
            for key, method, lam, label, color, marker in configs:
                runs = table[key][model]
                if target not in runs:
                    continue
                series = runs[target]
                # target solid
                if target in series:
                    steps = sorted(series[target].keys())
                    vals = [series[target][s] for s in steps]
                    ax.plot(steps, vals, color=color, lw=3.0, ls="-",
                            label=f"{label} target"
                            if i == 0 and j == 0 else None)
                # downstream dashed mean
                agg = defaultdict(list)
                for ds in DOWNSTREAM:
                    if ds in series:
                        for s, v in series[ds].items():
                            agg[s].append(v)
                if agg:
                    steps = sorted(agg)
                    means = [np.mean(agg[s]) for s in steps]
                    ax.plot(steps, means, color=color, lw=2.0, ls=":",
                            label=f"{label} downstream avg"
                            if i == 0 and j == 0 else None)
            ax.set_title(f"{model} / {target}", fontsize=15, fontweight="bold")
            ax.set_xlabel("step", fontsize=12)
            if j == 0:
                ax.set_ylabel(r"$\Delta\mathcal{R} = \mathrm{loss}_P - \mathrm{loss}_T$",
                              fontsize=12)
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.02), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"\nSaved curve plot: {out_path}")


def plot_tradeoff(rows, out_path: Path):
    """Per (model, target), x=avg_forget, y=−Δtarget, points colored by config."""
    config_meta = {c[0]: c for c in ALL_CONFIGS}
    by_mt = defaultdict(dict)
    for model, target, key, label, dt, af, mf, net, step, note in rows:
        if dt is None:
            continue
        by_mt[(model, target)][key] = (dt, af)

    fig, axes = plt.subplots(len(MODELS), len(TARGETS),
                             figsize=(18, 10), sharex=False, sharey=False)
    if len(TARGETS) == 1:
        axes = axes.reshape(-1, 1)
    for i, model in enumerate(MODELS):
        for j, target in enumerate(TARGETS):
            ax = axes[i, j]
            key_view = (model, target)
            if key_view not in by_mt:
                ax.set_title(f"{model}/{target} (no data)",
                             fontsize=15, fontweight="bold")
                continue
            xs, ys, key_seq = [], [], []
            for c in ALL_CONFIGS:
                k = c[0]
                if k in by_mt[key_view]:
                    dt, af = by_mt[key_view][k]
                    xs.append(af)
                    ys.append(-dt)
                    key_seq.append(k)
            ax.plot(xs, ys, "-", color="gray", lw=1.5, alpha=0.4)
            for x, y, k in zip(xs, ys, key_seq):
                _, _, _, label, color, marker = config_meta[k]
                ax.scatter(x, y, s=180, color=color, marker=marker,
                           edgecolors="k", linewidths=1.0,
                           label=label if i == 0 and j == 0 else None,
                           zorder=3)
            ax.axhline(0, color="gray", lw=1.4, ls="--")
            ax.axvline(0, color="gray", lw=1.4, ls="--")
            ax.set_title(f"{model}/{target}", fontsize=15, fontweight="bold")
            ax.set_xlabel("avg downstream forgetting →", fontsize=12)
            if j == 0:
                ax.set_ylabel("target improvement (−Δ) ↑", fontsize=12)
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.02), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved trade-off plot: {out_path}")


def plot_per_target(table, configs, out_dir: Path):
    """One figure per (model, target). 6 subplots = 6 eval tasks. Lines=configs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x_max = ALIGN_STEP
    for model in MODELS:
        for target in TARGETS:
            has_any = any(target in table[c[0]][model] for c in configs)
            if not has_any:
                continue
            fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=True)
            axes = axes.flatten()
            eval_tasks = [target] + DOWNSTREAM
            for k, ev in enumerate(eval_tasks):
                ax = axes[k]
                ax.axhline(0, color="gray", lw=1.6, ls="--")
                for ckey, method, lam, label, color, marker in configs:
                    runs = table[ckey][model]
                    if target not in runs:
                        continue
                    ev_series = runs[target].get(ev, {})
                    if not ev_series:
                        continue
                    steps = sorted(s for s in ev_series if s <= x_max)
                    if not steps:
                        continue
                    vals = [ev_series[s] for s in steps]
                    ax.plot(steps, vals, color=color, lw=2.4,
                            marker=marker, markersize=8,
                            markeredgecolor="k", markeredgewidth=0.8,
                            label=label)
                is_target = (ev == target)
                title = f"{ev}" + (" (target)" if is_target else "")
                ax.set_title(title, fontsize=16,
                             fontweight="bold" if is_target else "normal")
                ax.tick_params(axis="both", labelsize=12)
                ax.grid(alpha=0.3)
                if k >= 3:
                    ax.set_xlabel("step", fontsize=14)
                if k % 3 == 0:
                    ax.set_ylabel(r"$\Delta\mathcal{R}$", fontsize=14)
                ax.set_xlim(0, x_max)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=6,
                       bbox_to_anchor=(0.5, 1.04), fontsize=11)
            fig.suptitle(
                f"{MODEL_DISPLAY[model]} / target={TASK_DISPLAY[target]}",
                fontsize=18, fontweight="bold", y=1.08)
            plt.tight_layout()
            out = out_dir / f"per_target_{model}_{target}.pdf"
            plt.savefig(out, bbox_inches="tight")
            plt.close()
            print(f"  saved {out.name}")


# ═══════════════════════════════════════════════════════════════════
# Recommendations
# ═══════════════════════════════════════════════════════════════════
def recommend(rows):
    print("\n=== Recommendations ===")
    by_mt = defaultdict(dict)
    label_lookup = {c[0]: c[3] for c in ALL_CONFIGS}
    for model, target, key, label, dt, af, mf, net, step, note in rows:
        if dt is None:
            continue
        by_mt[(model, target)][key] = dict(dt=dt, af=af, mf=mf,
                                            net=net, step=step, note=note)

    # Per (model, target) best by net
    print("\n(1) Best config per (model, target) by net:")
    win_by_method = defaultdict(int)
    for (model, target), cfgs in by_mt.items():
        best_key, best_net = None, -np.inf
        for k, d in cfgs.items():
            if d["net"] > best_net:
                best_net, best_key = d["net"], k
        if best_key:
            method = best_key.split("_")[0] if "_" in best_key else best_key
            win_by_method[method] += 1
            print(f"  {model}/{target}: {label_lookup[best_key]:<18} "
                  f"net={best_net:+.4f}")
    print(f"\n  → Wins by method: {dict(win_by_method)}")

    # Recommended pair comparison: trace λ=1.0 (sweep max + |ΔR|-best) vs
    # replay λ=0.01 (replay's |ΔR|-best across sweep)
    print("\n(2) Recommended pair comparison (trace λ=1.0 vs replay λ=0.01):")
    for (model, target), cfgs in by_mt.items():
        if ("trace_1.0" not in cfgs or "replay_0.01" not in cfgs
                or "baseline" not in cfgs):
            continue
        b = cfgs["baseline"]
        t = cfgs["trace_1.0"]
        r = cfgs["replay_0.01"]
        print(f"  {model}/{target}:")
        print(f"    baseline:        Δtgt={b['dt']:+.4f}, avgFor={b['af']:+.4f}, net={b['net']:+.4f}")
        print(f"    replay λ=0.01:   Δtgt={r['dt']:+.4f}, avgFor={r['af']:+.4f}, net={r['net']:+.4f}")
        print(f"    trace λ=1.0:     Δtgt={t['dt']:+.4f}, avgFor={t['af']:+.4f}, net={t['net']:+.4f}")
        winner = "trace λ=1.0" if t["net"] > r["net"] else "replay λ=0.01"
        print(f"    → winner (by net): {winner}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    table = collect()
    rows = summarize(table)

    print_table(rows)
    validate_hypotheses(rows)
    check_lower_than_baseline(table,
                              compare_keys=["trace_1.0", "trace_0.5",
                                            "replay_0.1", "replay_0.01"])

    print("\n=== Plots ===")
    print("Combined paper figures:")
    plot_combined(table, PLOT_CONFIGS, FIG_DIR)
    print("Diagnostic plots:")
    plot_curves(table, PLOT_CONFIGS, FIG_DIR / "forgetting_curves_reg.pdf")
    plot_tradeoff(rows, FIG_DIR / "forgetting_tradeoff_reg.pdf")
    print("Per-target detail figures:")
    plot_per_target(table, PLOT_CONFIGS, FIG_DIR / "per_target_figs")

    recommend(rows)


if __name__ == "__main__":
    main()
