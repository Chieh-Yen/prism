#!/usr/bin/env python3
"""Tables for the scale-axis intervention, from the JSON written by scale_toggle.py.

Two settings of the 8-bit bitsandbytes variant are compared per benchmark:

    outliers kept    llm_int8_threshold = 6.0 (default; outlier channels stay in FP16)
    all quantized    llm_int8_threshold = 1e9 (every channel quantized)

Every quantity is recomputed from the stored primitives (loss_T, loss_P, rho_T,
rho_P, omega, gamma), and the derived fields stored next to them are checked
against the recomputation:

    gap         |loss_T - loss_P|                      (nats)
    scale term  (rho_T - rho_P)^2
    shape term  2 rho_T rho_P (1 - Omega)
    delta       sqrt(scale term + shape term)
    bound       K_feat * delta + K_pred * gamma

K_feat and K_pred are properties of the target head. They are read from the
JSON written by head_constants.py (one file per results file, same order).

The ratio columns are "all quantized / outliers kept". A change of the gap
counts only if it exceeds the noise floor (NOISE_FLOOR, fixed at 0.02 nats).

The same tables are printed for every results file: the per-benchmark gaps and
term ratios, the absolute terms and the bound, and the summary. With more than
one file an across-models table follows.

Usage:
    python analyze.py --results QWEN.json LLAMA.json --constants QWEN_K.json LLAMA_K.json
"""
from __future__ import annotations
import argparse, json, math

DISPLAY = {"squad": "SQuAD", "triviaqa": "TriviaQA", "mmlu": "MMLU",
           "arc": "ARC", "gsm8k": "GSM8K"}

KEPT, ALLQ = "outliers kept", "all quantized"

# Noise floor, in nats: gap changes below it are read as noise. It is a
# constant of the analysis, not an option.
NOISE_FLOOR = 0.02


def derive(rec, k_feat, k_pred):
    """Recompute every reported quantity of one cell from its stored primitives."""
    rho_t, rho_p, omega, gamma = rec["rho_T"], rec["rho_P"], rec["omega"], rec["gamma"]
    scale = (rho_t - rho_p) ** 2
    shape = 2.0 * rho_t * rho_p * (1.0 - omega)
    delta = math.sqrt(max(scale + shape, 0.0))
    gap = abs(rec["loss_T"] - rec["loss_P"])
    stored = {"dR": gap, "scale_term": scale, "shape_term": shape, "delta": delta}
    dev = max(abs(rec[k] - v) for k, v in stored.items())
    return dict(gap=gap, scale=scale, shape=shape, delta=delta, gamma=gamma,
                bound=k_feat * delta + k_pred * gamma, dev=dev)


def load(results_path, constants_path):
    with open(results_path) as f:
        records = json.load(f)
    with open(constants_path) as f:
        const = json.load(f)
    targets = sorted({r["target"] for r in records})
    if len(targets) != 1:
        raise SystemExit(f"{results_path}: expected one target, found {targets}")
    target = targets[0]
    if "target" not in const:
        raise SystemExit(f"{constants_path}: no 'target' field; a constants file must name "
                         f"the model it was computed for (see head_constants.py)")
    if const["target"] != target:
        raise SystemExit(f"{constants_path} is for {const['target']}, "
                         f"but {results_path} is for {target}")
    k_feat, k_pred = float(const["K_feat"]), float(const["K_pred"])
    cells = {}
    for r in records:
        setting = KEPT if r["outliers_preserved"] else ALLQ
        if (r["task"], setting) in cells:
            raise SystemExit(f"{results_path}: duplicate cell {r['task']} / {setting}")
        cells[(r["task"], setting)] = dict(derive(r, k_feat, k_pred), threshold=r["threshold"])
    tasks = list(dict.fromkeys(r["task"] for r in records))
    for t in tasks:
        for setting in (KEPT, ALLQ):
            if (t, setting) not in cells:
                raise SystemExit(f"{results_path}: missing cell {t} / {setting}")
    return dict(target=target, k_feat=k_feat, k_pred=k_pred, tasks=tasks, cells=cells,
                n_records=len(records))


def fold(x):
    """Ratio as text: one decimal from 1 upward, two decimals below."""
    return f"{x:.1f}x" if x >= 1.0 else f"{x:.2f}x"


def verdict(change, floor):
    """change = gap(all quantized) - gap(outliers kept)."""
    return "beyond noise" if abs(change) > floor else "within noise"


def report(m, floor):
    c, tasks = m["cells"], m["tasks"]
    name = lambda t: DISPLAY.get(t, t)
    print("=" * 100)
    print(f"{m['target']}   K_feat = {m['k_feat']:.4f}   K_pred = {m['k_pred']:.4f}   "
          f"cells = {m['n_records']}")
    th = {setting: sorted({c[(t, setting)]["threshold"] for t in tasks}) for setting in (KEPT, ALLQ)}
    print(f"llm_int8_threshold: {KEPT} = {th[KEPT]}, {ALLQ} = {th[ALLQ]}")
    print(f"largest |stored derived field - recomputed| = "
          f"{max(v['dev'] for v in c.values()):.1e}")
    print("=" * 100)

    print("\nGap and term ratios (ratio = all quantized / outliers kept)")
    print(f"{'benchmark':10s} {'gap kept':>9s} {'gap all-q':>10s} {'scale ratio':>12s} "
          f"{'shape ratio':>12s} {'gap change':>11s}  reading at the {floor:g}-nat floor")
    for t in tasks:
        a, b = c[(t, KEPT)], c[(t, ALLQ)]
        change = b["gap"] - a["gap"]
        print(f"{name(t):10s} {a['gap']:9.4f} {b['gap']:10.4f} "
              f"{fold(b['scale'] / a['scale']):>12s} {fold(b['shape'] / a['shape']):>12s} "
              f"{change:+11.4f}  {verdict(change, floor)}")

    print("\nAbsolute terms and bound")
    print(f"{'benchmark':10s} {'setting':14s} {'scale term':>11s} {'shape term':>11s} "
          f"{'delta':>8s} {'gamma':>7s} {'bound':>9s} {'gap':>8s}  bound >= gap")
    for t in tasks:
        for setting in (KEPT, ALLQ):
            v = c[(t, setting)]
            print(f"{name(t):10s} {setting:14s} {v['scale']:11.4f} {v['shape']:11.2f} "
                  f"{v['delta']:8.3f} {v['gamma']:7.3f} {v['bound']:9.2f} {v['gap']:8.4f}  "
                  f"{'yes' if v['bound'] >= v['gap'] else 'NO'}")

    n_cells = len(c)
    scale_ratio = [c[(t, ALLQ)]["scale"] / c[(t, KEPT)]["scale"] for t in tasks]
    shape_ratio = [c[(t, ALLQ)]["shape"] / c[(t, KEPT)]["shape"] for t in tasks]
    change = [c[(t, ALLQ)]["gap"] - c[(t, KEPT)]["gap"] for t in tasks]
    kept_scale = [c[(t, KEPT)]["scale"] for t in tasks]
    summary = dict(
        target=m["target"],
        head_zero=sum(v["gamma"] == 0.0 for v in c.values()),
        bound_holds=sum(v["bound"] >= v["gap"] for v in c.values()),
        bound_lower_kept=sum(c[(t, KEPT)]["bound"] < c[(t, ALLQ)]["bound"] for t in tasks),
        n_cells=n_cells, n_tasks=len(tasks),
        scale_ratio=(min(scale_ratio), max(scale_ratio)),
        shape_ratio=(min(shape_ratio), max(shape_ratio)),
        kept_scale=(min(kept_scale), max(kept_scale)),
        max_abs_change=max(abs(x) for x in change),
        beyond_floor=sum(abs(x) > floor for x in change),
    )
    s = summary
    print("\nSummary")
    print(f"  head term exactly zero                      : {s['head_zero']} / {n_cells} cells")
    print(f"  bound >= gap                                : {s['bound_holds']} / {n_cells} cells")
    print(f"  bound lower with outliers kept              : "
          f"{s['bound_lower_kept']} / {s['n_tasks']} benchmarks")
    print(f"  scale term ratio, min to max                : "
          f"{fold(s['scale_ratio'][0])} to {fold(s['scale_ratio'][1])}")
    print(f"  shape term ratio, min to max                : "
          f"{fold(s['shape_ratio'][0])} to {fold(s['shape_ratio'][1])}")
    print(f"  scale term with outliers kept, min to max   : "
          f"{s['kept_scale'][0]:.3f} to {s['kept_scale'][1]:.3f}")
    print(f"  largest |gap change| from the switch        : {s['max_abs_change']:.4f} nats")
    print(f"  benchmarks whose gap change exceeds {floor:g} nats: "
          f"{s['beyond_floor']} / {s['n_tasks']}")
    print()
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True,
                    help="JSON files written by scale_toggle.py, one per target model; "
                         "the same tables are printed for each")
    ap.add_argument("--constants", nargs="+", required=True,
                    help="JSON files written by head_constants.py, in the same order")
    a = ap.parse_args()
    if len(a.results) != len(a.constants):
        raise SystemExit("--results and --constants need the same number of files")

    summaries = [report(load(r, k), NOISE_FLOOR) for r, k in zip(a.results, a.constants)]

    if len(summaries) > 1:
        print("=" * 100)
        print("Across models")
        print("=" * 100)
        print(f"{'target':28s} {'scale term, outliers kept':>27s} {'largest |gap change|':>21s} "
              f"{'beyond floor':>13s} {'bound >= gap':>13s}")
        for s in summaries:
            rng = f"{s['kept_scale'][0]:.3f} to {s['kept_scale'][1]:.3f}"
            print(f"{s['target']:28s} {rng:>27s} {s['max_abs_change']:21.4f} "
                  f"{s['beyond_floor']:>9d} / {s['n_tasks']} "
                  f"{s['bound_holds']:>8d} / {s['n_cells']}")
        total = sum(s["n_cells"] for s in summaries)
        print(f"\n  bound >= gap in {sum(s['bound_holds'] for s in summaries)} / {total} cells; "
              f"head term exactly zero in {sum(s['head_zero'] for s in summaries)} / {total} cells")


if __name__ == "__main__":
    main()
