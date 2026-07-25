#!/usr/bin/env python3
"""
E3 part C — recover the six (ref_task, size) configs from the screen log.

Background: the 2026-07-24 part-C run completed all six configs
(ref in {truthfulqa, wikitext} x n in {8, 32, 128}, trace lambda=1.0,
Llama-TruthfulQA, seed 42, step 300) but every run wrote to the SAME
output directory, so only the last JSON survived (script_E3.sh has since
been fixed to use a per-config output_root). The PRISM checkpoint tables
are all printed in the screen log, however — this script parses them back.

For each config: the step-300 [Empirical Risk] table (per-task Loss_T /
Loss_P / |dR|) and the geometry table's Omega column. Summary per config:
downstream mean |dR| (5 benchmarks), target-task loss_P, mean downstream
Omega — plus lambda=0 / lambda=1.0 anchors recomputed from the paper-run
trees for direct comparison, and a cross-check of the last config against
the surviving JSON.

Zero GPU, stdlib only.

Usage:  python3 rebuttal_exp/exp_e3_partc_from_log.py \
            [--log rebuttal_exp/out/E3/screen.E3.20260724_093318.log]

Output: rebuttal_exp/out/E3/partC_summary.md  (+ partC_summary.csv)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_DIR = HERE / "out" / "E3"

DOWNSTREAM = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]
FT_TASK = "truthfulqa"
ANALYSIS_STEP = 300

MARKER_RE = re.compile(r">>> trace lam=1\.0 ref=(\w+) n=(\d+)")
STEP_RE = re.compile(r"PRISM evaluation @ step (\d+)")
# Risk rows:  "  arc            0.4886    8.0521    7.5635  424.9973    yes"
RISK_RE = re.compile(
    r"^\s{2}(\w+)\s*\*?\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(yes|NO)\s*$")
# Geometry rows have 8 numeric columns; Omega is the 3rd.
GEOM_RE = re.compile(
    r"^\s{2}(\w+)\s*\*?\s+" + r"([-\d.]+)\s+" * 7 + r"([-\d.]+)\s*$")


def parse_log(path: Path):
    text = path.read_text(errors="replace").replace("\r", "\n")
    lines = text.split("\n")

    # section boundaries
    sections = []          # (ref, n, start_idx)
    for i, ln in enumerate(lines):
        m = MARKER_RE.search(ln)
        if m:
            sections.append((m.group(1), int(m.group(2)), i))
    configs = []
    for s, (ref, n, start) in enumerate(sections):
        end = sections[s + 1][2] if s + 1 < len(sections) else len(lines)
        # last "@ step <= ANALYSIS_STEP" block in the section
        best_step, best_idx = -1, None
        for i in range(start, end):
            m = STEP_RE.search(lines[i])
            if m and best_step <= int(m.group(1)) <= ANALYSIS_STEP:
                best_step, best_idx = int(m.group(1)), i
        if best_idx is None:
            configs.append({"ref": ref, "n": n, "step": None})
            continue
        risk, omega = {}, {}
        in_risk = False
        # No fixed window: tqdm \r-junk inflates line counts unpredictably.
        # Scan until the risk table closes (6 rows seen or [saved] marker).
        for i in range(best_idx, end):
            ln = lines[i]
            if "[Empirical Risk]" in ln:
                in_risk = True
                continue
            if in_risk and (ln.strip().startswith("=" * 20)
                            or "[saved]" in ln or len(risk) >= 6):
                break
            mg = GEOM_RE.match(ln)
            if mg and not in_risk and mg.group(1) in DOWNSTREAM + [FT_TASK]:
                omega[mg.group(1)] = float(mg.group(4))   # 3rd numeric = Omega
            mr = RISK_RE.match(ln)
            if mr and in_risk and mr.group(1) in DOWNSTREAM + [FT_TASK]:
                risk[mr.group(1)] = {
                    "loss_T": float(mr.group(2)),
                    "loss_P": float(mr.group(3)),
                    "dR": float(mr.group(4)),
                }
        ds = [risk[t]["dR"] for t in DOWNSTREAM if t in risk]
        oms = [omega[t] for t in DOWNSTREAM if t in omega]
        configs.append({
            "ref": ref, "n": n, "step": best_step,
            "mean_dR_downstream": statistics.mean(ds) if ds else None,
            "target_loss_P": risk.get(FT_TASK, {}).get("loss_P"),
            "mean_omega_downstream": statistics.mean(oms) if oms else None,
            "per_bench_dR": {t: risk[t]["dR"] for t in DOWNSTREAM if t in risk},
        })
    return configs


def tree_anchor(lam_dirs):
    """mean downstream |dR| @300 from the first paper-run tree that has a
    complete llama/truthfulqa run for the given lambda dir names."""
    roots = [REPO / "exp_result" / "regularization",
             REPO / "forgetting_exp_log_safety_clear",
             REPO / "forgetting_exp_log_safety"]
    for root in roots:
        for lam in lam_dirs:
            p = root / lam / "llama" / f"prism_forgetting_metrics_{FT_TASK}.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            best = None
            for ck in d["checkpoints"]:
                if ck["step"] <= ANALYSIS_STEP and \
                        (best is None or ck["step"] > best["step"]):
                    best = ck
            if best is None or best["step"] < ANALYSIS_STEP:
                continue
            ds = [best["tasks"][t]["delta_risk"]
                  for t in DOWNSTREAM if t in best["tasks"]]
            tgt = best["tasks"].get(FT_TASK, {}).get("loss_P")
            return statistics.mean(ds), tgt, str(p.relative_to(REPO))
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(
        OUT_DIR / "screen.E3.20260724_093318.log"))
    args = ap.parse_args()
    configs = parse_log(Path(args.log))

    anchor0, tgt0, src0 = tree_anchor(["0.0", "0"])
    anchor1, tgt1, src1 = tree_anchor(["1.0", "1"])

    # cross-check last config against the surviving JSON, if present
    surv = (OUT_DIR / "reg_sensitivity" / "trace" / "lam1" / "seed42"
            / "llama-3.1-8b" / FT_TASK / "prism_forgetting_metrics.json")
    check_line = "surviving JSON not found — cross-check skipped"
    last = configs[-1] if configs else None
    if surv.exists() and last and last.get("mean_dR_downstream") is not None:
        d = json.load(open(surv))
        best = max((ck for ck in d["checkpoints"]
                    if ck["step"] <= ANALYSIS_STEP), key=lambda c: c["step"])
        js = statistics.mean(best["tasks"][t]["delta_risk"]
                             for t in DOWNSTREAM if t in best["tasks"])
        ok = abs(js - last["mean_dR_downstream"]) < 5e-3
        check_line = (f"cross-check vs surviving JSON (last config "
                      f"{last['ref']}/n={last['n']}): log {last['mean_dR_downstream']:.4f} "
                      f"vs json {js:.4f} -> {'MATCH' if ok else 'MISMATCH'}")
    elif last and last.get("mean_dR_downstream") is None:
        check_line = "last config failed to parse — cross-check skipped"

    a0 = (f"no-reg (lambda=0) mean downstream |dR| = {anchor0:.4f} [{src0}]"
          if anchor0 is not None else "no-reg (lambda=0) anchor: NOT FOUND")
    a1 = (f"paper config (ref=task, n=32, lambda=1.0) = {anchor1:.4f} [{src1}]"
          if anchor1 is not None else
          "lambda=1.0 tree anchor: no complete step-300 run in any tree "
          "(known E10.md caveat) — use this run's (task, 32) row itself")
    md = ["# E3 part C — recovered from screen log "
          "(trace lambda=1.0, Llama-TruthfulQA, seed 42, step 300)", "",
          "**PROTOCOL CAVEAT — INTERNAL REFERENCE ONLY.** This round ran at "
          "lr = 2e-4 (the trainer's old inherited default), but every "
          "paper-round tree run records lr = 1e-5 (launch-time override; "
          "see `experiment.lr` in exp_result/regularization/*/*.json). "
          "These six numbers are an lr-stress variant, NOT the paper-"
          "protocol sensitivity result — do not cite them in the rebuttal; "
          "the 'vs no-reg' column compares across lr rounds and is "
          "therefore not meaningful. Part C must be re-run with the fixed "
          "defaults (lr 1e-5, per-config output_root, lambda 0.5).", "",
          f"Anchors from the paper-run trees: {a0}; {a1}.", "",
          f"{check_line}", "",
          "| ref domain | n | step | mean downstream |dR| | vs no-reg | "
          "target loss_P | mean downstream Omega |",
          "|---|---|---|---|---|---|---|"]
    rows_csv = []
    for c in configs:
        if c["step"] is None or c["mean_dR_downstream"] is None:
            md.append(f"| {c['ref']} | {c['n']} | PARSE FAIL | - | - | - | - |")
            continue
        rel = (100 * (anchor0 - c["mean_dR_downstream"]) / anchor0
               if anchor0 else float("nan"))
        md.append(f"| {c['ref']} | {c['n']} | {c['step']} "
                  f"| {c['mean_dR_downstream']:.4f} | {rel:+.1f}% "
                  f"| {c['target_loss_P']:.4f} "
                  f"| {c['mean_omega_downstream']:.4f} |")
        rows_csv.append({"ref": c["ref"], "n": c["n"], "step": c["step"],
                         "mean_dR_downstream": c["mean_dR_downstream"],
                         "benefit_vs_noreg_pct": rel,
                         "target_loss_P": c["target_loss_P"],
                         "mean_omega_downstream": c["mean_omega_downstream"],
                         **{f"dR_{t}": c["per_bench_dR"].get(t)
                            for t in DOWNSTREAM}})

    md += ["", "Reading: positive 'vs no-reg' = trace at that (domain, size) "
           "REDUCES forgetting relative to lambda=0; negative = it makes "
           "forgetting worse. The (task, 32) row is a same-seed rerun of the "
           "paper configuration — use it as the internal reproduction check.",
           ""]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "partC_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    (OUT_DIR / "partC_summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
