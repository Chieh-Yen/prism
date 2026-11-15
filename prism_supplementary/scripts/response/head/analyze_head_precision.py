#!/usr/bin/env python3
"""Analysis of head experiment 2 (output of head_precision.py). No GPU.

A cell is (model, benchmark, backbone, head). Each cell stores the loss, the
feature term delta and the head term gamma against the target (BF16 backbone,
BF16 head), and the gap |R_target - R| in nats (dR; dR_signed keeps the sign).

Quantities printed
  cells       number of cells; cells whose features are identical, entry for
              entry, to those under the first head on the same backbone.
  bound       K_feat * delta + K_pred * gamma >= gap, recomputed here from the
              stored delta and gamma, with K_feat and K_pred read from info.
  pairs       pairs of head settings inside one group, both heads quantized
              (the BF16 head is left out: its head term and its outcome are 0
              by construction). A pair counts only if its two outcomes differ
              by >= NOISE_FLOOR nats, and is "ranked correctly" if the head with the
              larger head term has the larger outcome. Ties in the head term
              count as wrong.
                head-only gap   group = (model, benchmark), original backbone;
                                outcome = gap of the cell.
                lowering        group = (model, benchmark, quantized backbone);
                                outcome = gap with that head minus gap with the
                                BF16 head on the same backbone, i.e. how much
                                restoring the original head lowers the gap.
              "Q8_0 to Q4_0 only" keeps the pairs whose two heads are both
              among Q8_0, Q6_K, Q5_0, Q4_0.
  interaction for a quantized backbone b and a quantized head h, on signed gaps:
                I = dR(b, h) - dR(b, BF16) - dR(BF16, h);
              reported as |I| / |dR(b, h)| over the cells with
              |dR(b, h)| >= NOISE_FLOOR.
  per format  median over the benchmarks of the head-only gap and of the head
              term, per model and head format, on the original backbone.
  restoring   the lowering defined above, per model and head format, over the
              (benchmark, quantized backbone) cells: its largest value, the
              largest absolute change of the gap, and the number of cells in
              which the lowering is >= NOISE_FLOOR.

NOISE_FLOOR is fixed at 0.02 nats and applies to gaps and to differences of
gaps (see the constant below).

usage:
  python analyze_head_precision.py FILE [FILE ...]
"""
from __future__ import annotations
import argparse, itertools, json, statistics

# Noise floor of the head experiments, in nats: gaps and differences of gaps
# below it are treated as noise. It is a constant of the analysis, not an
# option.
NOISE_FLOOR = 0.02
ORIGINAL = "bf16"
PRACTICAL = ["q8_0", "q6_k", "q5_0", "q4_0"]
HEAD_ORDER = ["bf16", "q8_0", "q6_k", "q5_0", "q4_0", "q3_0", "q2_0"]
HEAD_NAME = {"bf16": "BF16", "q8_0": "Q8_0", "q6_k": "Q6_K", "q5_0": "Q5_0",
             "q4_0": "Q4_0", "q3_0": "3-bit", "q2_0": "2-bit"}


def load(paths):
    cells, meta = {}, {}
    for p in paths:
        d = json.load(open(p))
        info = d["info"]
        model = info["target"].split("/")[-1]
        if "K_feat" not in info:
            raise SystemExit(f"{p}: incomplete run (no K_feat in info)")
        kf, kp = info["K_feat"], info["K_pred"]
        m = meta.setdefault(model, dict(K_feat=kf, K_pred=kp, n=info["n"],
                                        seed=info["seed"], files=[]))
        if (m["K_feat"], m["K_pred"]) != (kf, kp):
            raise SystemExit(f"{p}: constants differ between files of {model}")
        m["files"].append(dict(path=p, quantizer_check=info.get("quantizer_check"),
                               native=info.get("native_head_is_q6k")))
        for r in d["rows"]:
            if "dR" not in r:
                raise SystemExit(f"{p}: incomplete run (rows without a gap)")
            key = (model, r["task"], r["backbone"], r["head"])
            if key in cells:
                raise SystemExit(f"duplicate cell {key}")
            cells[key] = r
    return cells, meta


def ranked_pairs(items, floor, keep):
    """items: (head, head term, outcome). Returns (correct, counted, wrong pairs)."""
    c = n = 0
    wrong = []
    for (h1, g1, y1), (h2, g2, y2) in itertools.combinations(items, 2):
        if not (keep(h1) and keep(h2)) or abs(y1 - y2) < floor:
            continue
        n += 1
        if (g1 - g2) * (y1 - y2) > 0:
            c += 1
        else:
            wrong.append(((h1, g1, y1), (h2, g2, y2)))
    return c, n, wrong


def main():
    ap = argparse.ArgumentParser(description="Analysis of head experiment 2 (head precision grid).")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    F = NOISE_FLOOR
    G, meta = load(a.files)
    models = sorted(meta)
    tasks = sorted({k[1] for k in G})
    backbones = sorted({k[2] for k in G if k[2] != ORIGINAL}, reverse=True)
    heads = [h for h in HEAD_ORDER if any(k[3] == h for k in G)]
    qheads = [h for h in heads if h != ORIGINAL]
    expected = len(models) * len(tasks) * (1 + len(backbones)) * len(heads)

    print(f"cells: {len(G)} (complete grid: {expected} = {len(models)} models x {len(tasks)} benchmarks "
          f"x {1 + len(backbones)} backbones x {len(heads)} heads)")
    if len(G) != expected:
        raise SystemExit("the grid is incomplete")
    print(f"  backbones: original, {', '.join(backbones)};  heads: {', '.join(HEAD_NAME[h] for h in heads)}")
    for m in models:
        print(f"  {m}: n = {meta[m]['n']}, seed = {meta[m]['seed']}, K_feat = {meta[m]['K_feat']:.4f} "
              f"(stored), K_pred = {meta[m]['K_pred']:.4f}")
        notes = []
        for f in meta[m]["files"]:
            qc = f["quantizer_check"] or {}
            note = ("    quantizers vs gguf-py, share of entries that agree: " + ", ".join(
                f"{HEAD_NAME[k]} {100 * (1 - v['frac_mismatch']):.2f}%" for k, v in qc.items())
                + f";  lm_head of each GGUF backbone file equals the q6_k head: {f['native']}")
            if note not in notes:
                notes.append(note)
        print("\n".join(notes))
    same = sum(bool(r["same_backbone_as_first_head"]) for r in G.values())
    print(f"  features identical across heads on the same backbone: {same} / {len(G)}")

    holds = sum(r["dR"] <= meta[k[0]]["K_feat"] * r["delta"] + meta[k[0]]["K_pred"] * r["gamma"]
                for k, r in G.items())
    print(f"\nbound  K_feat * delta + K_pred * gamma >= gap: holds in {holds} / {len(G)} cells")

    print(f"\npairs of quantized head settings whose outcomes differ by >= {F} nats, "
          f"ranked correctly by the head term")
    subsets = [("all heads", lambda h: True), ("Q8_0 to Q4_0 only", lambda h: h in PRACTICAL)]
    for title, groups, outcome in [
        ("head-only gap (original backbone)",
         [(m, t, ORIGINAL) for m in models for t in tasks],
         lambda m, t, b, h: G[(m, t, b, h)]["dR"]),
        ("lowering of the gap when the original head is restored (quantized backbones)",
         [(m, t, b) for m in models for t in tasks for b in backbones],
         lambda m, t, b, h: G[(m, t, b, h)]["dR"] - G[(m, t, b, ORIGINAL)]["dR"]),
    ]:
        print(f"  {title}")
        for name, keep in subsets:
            tot = {m: [0, 0] for m in models}
            for m, t, b in groups:
                items = [(h, G[(m, t, b, h)]["gamma"], outcome(m, t, b, h)) for h in qheads]
                c, n, _ = ranked_pairs(items, F, keep)
                tot[m][0] += c; tot[m][1] += n
            print(f"    {name:18s} {sum(v[0] for v in tot.values())} / {sum(v[1] for v in tot.values())}"
                  f"   (" + ", ".join(f"{m} {tot[m][0]} / {tot[m][1]}" for m in models) + ")")

    rel = []
    for m, t, b, h in itertools.product(models, tasks, backbones, qheads):
        full = G[(m, t, b, h)]
        i = full["dR_signed"] - G[(m, t, b, ORIGINAL)]["dR_signed"] - G[(m, t, ORIGINAL, h)]["dR_signed"]
        if full["dR"] >= F:
            rel.append(abs(i) / full["dR"])
    n_all = len(models) * len(tasks) * len(backbones) * len(qheads)
    print(f"\ninteraction |I| / |dR|: median {statistics.median(rel):.3f} "
          f"over the {len(rel)} of {n_all} cells with |dR| >= {F}")

    print("\nper head format, original backbone, median over the benchmarks")
    print(f"  {'':22s}" + "".join(f"{HEAD_NAME[h]:>9s}" for h in qheads))
    med_gap, med_term = {}, {}
    for m in models:
        for h in qheads:
            med_gap[(m, h)] = statistics.median(G[(m, t, ORIGINAL, h)]["dR"] for t in tasks)
            med_term[(m, h)] = statistics.median(G[(m, t, ORIGINAL, h)]["gamma"] for t in tasks)
        print(f"  {m:14s} gap    " + "".join(f"{med_gap[(m, h)]:9.4f}" for h in qheads))
    for m in models:
        print(f"  {m:14s} term   " + "".join(f"{med_term[(m, h)]:9.2f}" for h in qheads))
    for m in models:
        mx = {h: max(G[(m, t, ORIGINAL, h)]["dR"] for t in tasks) for h in qheads}
        prac = [h for h in qheads if h in PRACTICAL]
        print(f"  {m:14s} Q8_0 to Q4_0: largest median {max(med_gap[(m, h)] for h in prac):.4f}, "
              f"largest single gap {max(mx[h] for h in prac):.4f} "
              f"({'all below' if max(mx[h] for h in prac) < F else 'not all below'} {F})")
    if len(models) == 2:
        lo, hi = sorted(models, key=lambda m: med_term[(m, qheads[0])])
        fm = sum(med_term[(hi, h)] > med_term[(lo, h)] for h in qheads)
        print(f"  head term larger on {hi} than on {lo} (medians): {fm} / {len(qheads)} head formats")

    print(f"\nrestoring the original head on a quantized backbone: lowering of the gap in nats "
          f"({len(tasks)} benchmarks x {len(backbones)} backbones = {len(tasks) * len(backbones)} cells)")
    print(f"  {'model':14s} {'head':>6s} {'largest':>9s} {'largest |.|':>12s} {'>= ' + str(F):>8s}")
    for m in models:
        for h in qheads:
            low = [G[(m, t, b, h)]["dR"] - G[(m, t, b, ORIGINAL)]["dR"] for t in tasks for b in backbones]
            print(f"  {m:14s} {HEAD_NAME[h]:>6s} {max(low):+9.4f} "
                  f"{max(abs(x) for x in low):12.4f} {sum(x >= F for x in low):5d} / {len(low)}")


if __name__ == "__main__":
    main()
