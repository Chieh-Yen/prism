#!/usr/bin/env python3
"""Peak table of the layer-wise profile, from the JSON written by layer_profile.py.

Entries are the shape drift 1 - Omega between the target and a GGUF variant,
where Omega is the mean per-token cosine (0 = no drift).

Numbering
---------
Blocks are numbered from 1.

  residual stream   ``omega[i]`` of a record is read at ``hidden_states[i]``:
                    i = 0 is the embedding output, i = 1 .. L-1 is the stream
                    after block i, and i = L (L = number of blocks) is the
                    final hidden state, i.e. the stream after block L and the
                    final norm. The peak is taken over i = 1 .. L, one value
                    per block. When it falls on i = L it is the value at the
                    output and is printed as "(output)". The embedding output
                    (i = 0) is not a block and is not part of the table.
  q/k/v projections ``modules[m].omega[j]`` is read at the output of the
                    projection in the block with 0-based index j, so the
                    reported block is j + 1. The peak is taken over all blocks.

Besides the table the script prints the token counts, the ratio between the
v_proj and q_proj peaks, and, for every target with several variants, how
often the ordering "fewer bits, more drift" is violated over the read-out
points of the table (every block of the residual stream and of q, k, v).

Usage:
    python make_table.py --results QWEN.json LLAMA.json [--profiles]
"""
from __future__ import annotations
import argparse, json, re

MODULES = ("q_proj", "k_proj", "v_proj")


def drift(omegas):
    return [1.0 - x for x in omegas]


def argmax(values, start=0):
    """Index of the first maximum of values[start:], in the indexing of values."""
    return max(range(start, len(values)), key=lambda i: (values[i], -i))


def bits(quant):
    """Nominal bit-width of a GGUF tag such as Q4_K_M."""
    m = re.match(r"Q(\d+)", quant)
    if m is None:
        raise SystemExit(f"cannot read a bit-width from the variant tag {quant!r}")
    return int(m.group(1))


def summarise(rec):
    for key, vals in [("residual", rec["rho_T"] + rec["rho_P"])] + [
            (m, rec["modules"][m]["rho_T"] + rec["modules"][m]["rho_P"]) for m in MODULES]:
        if any(abs(v - 1.0) > 1e-3 for v in vals):
            raise SystemExit(
                f"{rec['target']} {rec['quant']} ({key}): stored RMS scales are not 1")
    res = drift(rec["omega"])
    n_blocks = len(res) - 1
    for m in MODULES:
        if len(rec["modules"][m]["omega"]) != n_blocks:
            raise SystemExit(f"{rec['target']} {rec['quant']}: {m} has "
                             f"{len(rec['modules'][m]['omega'])} entries, expected {n_blocks}")
    peak_i = argmax(res, start=1)
    out = dict(name=f"{rec['target'].split('/')[-1]} {rec['quant']}",
               target=rec["target"], quant=rec["quant"], task=rec["task"], rows=rec["rows"],
               n_blocks=n_blocks, residual=res,
               res_peak=res[peak_i], res_peak_block=peak_i,
               res_peak_is_output=(peak_i == n_blocks),
               output=res[n_blocks], modules={})
    for m in MODULES:
        d = drift(rec["modules"][m]["omega"])
        j = argmax(d)
        out["modules"][m] = dict(drift=d, peak=d[j], block=j + 1)
    return out


def cell(value, block, is_output=False):
    return f"{value:.3f} ({'output' if is_output else block})"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True,
                    help="JSON files written by layer_profile.py")
    ap.add_argument("--profiles", action="store_true",
                    help="also print the full depth profile of every variant")
    a = ap.parse_args()

    recs = []
    for path in a.results:
        with open(path) as f:
            recs += [summarise(r) for r in json.load(f)]

    print("Peak table (shape drift 1 - Omega, 0 = no drift; blocks numbered from 1)\n")
    print("| variant | residual stream, peak (block) | at output (final hidden state) | "
          "q_proj peak (block) | k_proj peak (block) | v_proj peak (block) |")
    print("|---|--:|--:|--:|--:|--:|")
    for r in recs:
        mods = " | ".join(cell(r["modules"][m]["peak"], r["modules"][m]["block"]) for m in MODULES)
        print(f"| {r['name']} | {cell(r['res_peak'], r['res_peak_block'], r['res_peak_is_output'])} "
              f"| {r['output']:.3f} | {mods} |")

    print("\nSame entries with six decimals, and the read-out sizes")
    print(f"{'variant':24s} {'task':6s} {'tokens':>7s} {'blocks':>7s} {'residual peak':>20s} "
          f"{'output':>9s} {'q_proj peak':>15s} {'k_proj peak':>15s} {'v_proj peak':>15s}")
    for r in recs:
        where = "output" if r["res_peak_is_output"] else f"{r['res_peak_block']}"
        mods = " ".join(f"{r['modules'][m]['peak']:.6f} ({r['modules'][m]['block']})".rjust(15)
                        for m in MODULES)
        peak = f"{r['res_peak']:.6f} ({where})"
        print(f"{r['name']:24s} {r['task']:6s} {r['rows']:7d} {r['n_blocks']:7d} "
              f"{peak:>20s} {r['output']:9.6f} {mods}")

    print("\nRatio of the module peaks: v_proj peak / q_proj peak (each peak taken over all blocks)")
    for r in recs:
        p = {m: r["modules"][m]["peak"] for m in MODULES}
        print(f"  {r['name']:24s} {p['v_proj'] / p['q_proj']:6.2f}")

    print("\nOrdering by bit-width (fewer bits, strictly more drift), per target")
    by_target = {}
    for r in recs:
        by_target.setdefault(r["target"], []).append(r)
    for target, group in by_target.items():
        if len(group) < 2:
            print(f"  {target}: one variant, nothing to order")
            continue
        group = sorted(group, key=lambda r: bits(r["quant"]))
        order = " > ".join(r["quant"] for r in group)
        total_points = total_viol = 0
        for point in ("residual",) + MODULES:
            series = [r["residual"][1:] if point == "residual" else r["modules"][point]["drift"]
                      for r in group]
            n_points = len(series[0])
            viol = sum(not all(series[k][i] > series[k + 1][i] for k in range(len(series) - 1))
                       for i in range(n_points))
            total_points += n_points
            total_viol += viol
            label = "residual stream (after block 1 .. L)" if point == "residual" else point
            print(f"  {target}: {order} violated at {viol} of {n_points} points, {label}")
        print(f"  {target}: {total_viol} violations over {total_points} read-out points")

    if not a.profiles:
        return
    print("\nDepth profiles of 1 - Omega")
    for r in recs:
        print(f"  {r['name']}")
        print("    residual (after block 1 .. L): "
              + " ".join(f"{x:.4f}" for x in r["residual"][1:]))
        for m in MODULES:
            print(f"    {m} (block 1 .. L): "
                  + " ".join(f"{x:.4f}" for x in r["modules"][m]["drift"]))


if __name__ == "__main__":
    main()
