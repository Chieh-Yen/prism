#!/usr/bin/env python3
"""
E15 — PAIRED reference-draw ablation for the shape regularizer (8VrD-Q3).

WHY A NEW EXPERIMENT INSTEAD OF REUSING E3 PART C
    E3 part C answered "does the reference set matter?" with a domain swap:
    task-domain TruthfulQA (n=32, |dR| 0.7358) vs generic WikiText (n=32,
    0.7731).  That contrast is UNPAIRED in three ways at once, so its
    0.04-nat gap cannot be attributed to domain:

      (i)   kind of reference item.  TruthfulQA rows go through a formatter
            AND a prompt_formatter, so the batch carries `prompt_length`
            and the task's loss_mode is "answer"; WikiText rows are raw
            text with no prompt boundary and loss_mode "full".
      (ii)  delivered size.  TextDataset drops blank/whitespace rows AFTER
            the n-row select, and wikitext-2-raw's test split is largely
            blank lines and " = Header = " lines, so "n=32" delivers fewer
            than 32 sequences and a very different token budget.  Size in
            SEQUENCES is matched; size in TOKENS is not.
      (iii) no within-condition scale.  One run per cell leaves nothing to
            say whether 0.04 nats is large or inside the run-to-run floor.

    E15 moves the manipulated variable inside the paper's own setting: same
    task-domain reference, same n=32, same formatter and prompt boundary —
    only WHICH 32 sequences, taken as DISJOINT windows of the one fixed
    shuffle the paper uses (seed 42 + 1000 = 1042).  Everything else
    (training seed, data order, LoRA init, lambda, lr, reg_every_k, steps)
    is identical, so offset 0 IS the paper's own run and comes for free,
    and a replicate of offset 0 measures the floor the draw spread has to be
    read against.

    Default lambda is 1.0 — the paper's headline operating point, whose
    complete step-300 run lives in regularization_exp/exp_result (0.6813,
    the trace column of Table 2).  Use --lambda_reg 0.5 to sit beside
    E3-C's size ablation instead (0.7358).

TWO ZERO-GPU MODES
    --preflight   Materialise every cell's reference set through the exact
                  trainer code path and report: delivered sequence count,
                  total and answer-region token counts, a content
                  fingerprint, and the pairwise row overlap between cells.
                  This turns "disjoint" and "matched size" into
                  measurements rather than intentions, and it is where the
                  WikiText unpairedness of E3-C becomes a number.
                  Needs a tokenizer; no model, no GPU.

    --aggregate   Read the per-cell prism_forgetting_metrics.json files
                  written by script_E15.sh (plus the paper-tree draw-0
                  anchor and the no-reg anchor) and emit the draw table:
                  per-draw mean downstream |dR|, per-benchmark |dR|, mean
                  Omega, target loss_P, then mean +- sd and CV across
                  draws, with the replicate pair printed as the floor.

Usage (repo root):
    python3 rebuttal_exp/exp_e15_refdraw_ablation.py --preflight
    python3 rebuttal_exp/exp_e15_refdraw_ablation.py --aggregate

Outputs: rebuttal_exp/out/E15/preflight_refsets.md
         rebuttal_exp/out/E15/refdraw_summary.{md,csv}
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

OUT_DIR = HERE / "out" / "E15"

DOWNSTREAM = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]
FT_TASK = "truthfulqa"
ANALYSIS_STEP = 300

# Cells are "ref_task:n:offset[:suffix]".  The suffix only renames the
# output directory, so a same-config replicate can sit beside the original.
#   offset 0   = the paper's own reference set (shuffle 1042, rows 0..31)
#   offset 32/64/96 = disjoint same-size windows of that shuffle
DEFAULT_PREFLIGHT_CELLS = ("truthfulqa:32:0 truthfulqa:32:32 "
                           "truthfulqa:32:64 truthfulqa:32:96 wikitext:32:0")
DEFAULT_AGGREGATE_CELLS = ("truthfulqa:32:32 truthfulqa:32:64 "
                           "truthfulqa:32:96 truthfulqa:32:0:rep")

CELL_JSON = ("trace/lam{lam:g}/seed{seed}/{model_short}/{task}/"
             "prism_forgetting_metrics.json")

# ── Which tree is the paper round ────────────────────────────────────────
# `regularization_exp/exp_result/` is the authoritative post-gradient-fix
# sweep: every run is lr 1e-5, seed 42, n_ref 32, k 8, max_steps 300, model
# meta-llama/Llama-3.1-8B, and its per-benchmark rows reproduce the paper's
# Table-2 dump exactly (paper/exp_replay_trace.md §4):
#     lambda 0 (no reg) 0.8434 | 0.01 0.8004 | 0.05 0.7757 | 0.1 0.7603
#     lambda 0.5 0.7358 | lambda 1.0 0.6813   <- Table 2's trace number
# The top-level `exp_result/regularization/` copy is NOT interchangeable:
# its 0.0 cell is the SUPERSEDED 1/8-gradient-bug baseline (1.4830, and a
# 700-step cosine schedule read at step 300), and its 1.0 cell is truncated
# at step 150. Anchoring on it is what produced E3.md §8.2's "-51%".
PAPER_ROUND = "regularization_exp/exp_result"
DRAW0_BY_LAMBDA = {
    1.0: f"{PAPER_ROUND}/regularization/1.0/llama/"
         "prism_forgetting_metrics_truthfulqa.json",   # 0.6813 = Table 2
    0.5: f"{PAPER_ROUND}/regularization/0.5/llama/"
         "prism_forgetting_metrics_truthfulqa.json",   # 0.7358 = E3-C's anchor
    0.1: f"{PAPER_ROUND}/regularization/0.1/llama/"
         "prism_forgetting_metrics_truthfulqa.json",
}
NOREG_PAPER = (f"{PAPER_ROUND}/regularization_replay/0.0/llama/"
               "prism_forgetting_metrics_truthfulqa.json")          # 0.8434
NOREG_SUPERSEDED = ("exp_result/regularization/0.0/llama/"
                    "prism_forgetting_metrics_truthfulqa.json")     # 1.4830


# ══════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════
def parse_cells(spec: str) -> List[Tuple[str, int, int, str]]:
    cells = []
    for tok in spec.split():
        parts = tok.split(":")
        if len(parts) == 3:
            parts.append("")
        if len(parts) != 4:
            sys.exit(f"bad cell '{tok}' — expected ref_task:n:offset[:suffix]")
        cells.append((parts[0], int(parts[1]), int(parts[2]), parts[3]))
    return cells


def cell_tag(ref_task: str, n: int, offset: int, suffix: str = "") -> str:
    tag = f"{ref_task}_n{n}_off{offset}"
    return f"{tag}_{suffix}" if suffix else tag


def step_ck(path: Path, step: int = ANALYSIS_STEP) -> Optional[Dict]:
    """The checkpoint record at exactly `step`, plus its experiment block."""
    try:
        d = json.load(open(path))
    except Exception:
        return None
    for ck in d.get("checkpoints", []):
        if ck.get("step") == step:
            return {"ck": ck, "experiment": d.get("experiment", {})}
    return None


def summarise_ck(ck: Dict) -> Dict:
    tasks = ck["tasks"]
    ds = [tasks[t]["delta_risk"] for t in DOWNSTREAM if t in tasks]
    om = [tasks[t]["omega"] for t in DOWNSTREAM if t in tasks]
    return {
        "mean_dR_downstream": statistics.mean(ds) if ds else None,
        "mean_omega_downstream": statistics.mean(om) if om else None,
        "target_loss_P": tasks.get(FT_TASK, {}).get("loss_P"),
        "n_downstream": len(ds),
        "per_bench_dR": {t: tasks[t]["delta_risk"]
                         for t in DOWNSTREAM if t in tasks},
    }


def fmt(v: Optional[float], nd: int = 4) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


# ══════════════════════════════════════════════════════════════════════
# preflight — what the trainer will actually feed the regularizer
# ══════════════════════════════════════════════════════════════════════
def preflight(cells: List[Tuple[str, int, int, str]], tokenizer_id: str,
              ref_seed: int, max_length: int, batch_size: int) -> None:
    from transformers import AutoTokenizer
    from prism.data.loaders import get_task_metadata, load_task_data

    tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    rows = []
    for ref_task, n, offset, suffix in cells:
        meta = get_task_metadata(ref_task)
        dl = load_task_data(
            ref_task, split="test", num_samples=n, batch_size=batch_size,
            tokenizer=tok, max_length=max_length,
            seed=ref_seed, offset=offset,
        )
        ds = dl.dataset
        ids, attn = ds.encodings["input_ids"], ds.encodings["attention_mask"]
        kept = int(ids.shape[0])
        per_seq = attn.sum(dim=1)
        total_tokens = int(per_seq.sum())
        if ds.prompt_lengths is not None:
            answer_tokens: Optional[int] = int(
                (per_seq - ds.prompt_lengths).clamp(min=0).sum())
        else:
            answer_tokens = None
        # Per-sequence fingerprint over unpadded token ids — identifies the
        # same source row across cells without needing the raw text back.
        fp = {hashlib.sha256(
                  ",".join(map(str, ids[i][attn[i].bool()].tolist())).encode()
              ).hexdigest()[:16] for i in range(kept)}
        rows.append({
            "tag": cell_tag(ref_task, n, offset, suffix),
            "ref_task": ref_task, "n_requested": n, "offset": offset,
            "n_delivered": kept, "dropped_blank": n - kept,
            "total_tokens": total_tokens, "answer_tokens": answer_tokens,
            "has_prompt_boundary": ds.prompt_lengths is not None,
            "loss_mode": meta["loss_mode"], "z_mode": meta["z_mode"],
            "setwise_fp": hashlib.sha256(
                "".join(sorted(fp)).encode()).hexdigest()[:16],
            "_fp": fp,
        })
        print(f"  {rows[-1]['tag']:<26s} kept {kept:>3d}/{n}  "
              f"tokens {total_tokens:>6d}  answer "
              f"{answer_tokens if answer_tokens is not None else 'n/a'}")

    md = [f"# E15 preflight — reference sets as delivered", "",
          f"tokenizer `{tokenizer_id}`, ref_seed {ref_seed}, "
          f"max_length {max_length}, reg_batch_size {batch_size}", "",
          "Every row is built through the same `load_task_data(split=\"test\", "
          "num_samples=n, seed=ref_seed, offset=offset)` call the trainer "
          "makes, so these are the sequences the regularizer sees, not a "
          "reconstruction of them.", "",
          "Delivered counts, blank-row drops, overlaps and fingerprints are "
          "dataset-level and tokenizer-independent; the two token columns are "
          "not, so run this with the tokenizer of the model the cells train "
          "on before quoting them.", "",
          "**`role`**: only task-domain rows become training cells. A "
          "`context only` row is never trained on by E15 — it is the E3-C "
          "comparison point whose unpairedness this table measures, and the "
          "control that shows the overlap matrix is really comparing source "
          "rows. Drop it with `--cells \"...\"` if the log should contain "
          "task rows alone.", "",
          "| cell | role | delivered / requested | blank dropped | total "
          "tokens | answer tokens | prompt boundary | loss_mode | set "
          "fingerprint |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        # Only task-domain cells become training runs. A non-task row is here
        # as CONTEXT: it is the E3-C comparison point whose unpairedness this
        # table quantifies, and it doubles as the control for the overlap
        # matrix (an all-zero column against a set that is genuinely
        # different proves the fingerprints identify source rows).
        role = "draw (trained)" if r["ref_task"] == FT_TASK else "context only"
        md.append(
            f"| {r['tag']} | {role} "
            f"| {r['n_delivered']} / {r['n_requested']} "
            f"| {r['dropped_blank']} | {r['total_tokens']} "
            f"| {r['answer_tokens'] if r['answer_tokens'] is not None else '—'} "
            f"| {'yes' if r['has_prompt_boundary'] else 'NO'} "
            f"| {r['loss_mode']} | `{r['setwise_fp']}` |")

    # pairwise overlap — the disjointness certificate
    md += ["", "## Pairwise row overlap (shared sequences)", "",
           "| | " + " | ".join(r["tag"] for r in rows) + " |",
           "|---" * (len(rows) + 1) + "|"]
    for a in rows:
        md.append(f"| {a['tag']} | "
                  + " | ".join(str(len(a["_fp"] & b["_fp"])) for b in rows)
                  + " |")

    same = [r for r in rows if r["ref_task"] == FT_TASK]
    off_diag = [len(a["_fp"] & b["_fp"]) for a in same for b in same
                if a is not b]
    if not off_diag:
        verdict = "n/a (fewer than two task-domain cells)"
    elif max(off_diag) == 0:
        verdict = "DISJOINT (zero shared sequences in every pair)"
    else:
        verdict = f"NOT DISJOINT — max pairwise overlap {max(off_diag)}"
    md += ["", f"**Task-domain draws: {verdict}.**"]
    if len(same) > 1:
        tc = [r["total_tokens"] for r in same]
        md.append(
            f"Their token budgets span {min(tc)}–{max(tc)} "
            f"({100 * (max(tc) - min(tc)) / statistics.mean(tc):.1f}% of the "
            f"mean) at a fixed 32 sequences. That residual spread is the "
            f"honest size mismatch that survives inside the paired design, "
            f"and it is what any |dR| spread has to be read against.")

    other = [r for r in rows if r["ref_task"] != FT_TASK]
    if other and same:
        base = same[0]
        md += ["", "## The unpairedness of the E3-C domain contrast, in numbers",
               ""]
        for r in other:
            md.append(
                f"- **{r['ref_task']} n={r['n_requested']}**: "
                f"{r['n_delivered']} sequences delivered vs "
                f"{base['n_delivered']} for the task draw "
                f"({r['dropped_blank']} blank rows dropped vs "
                f"{base['dropped_blank']}); {r['total_tokens']} tokens vs "
                f"{base['total_tokens']}; prompt boundary "
                f"{'present' if r['has_prompt_boundary'] else 'ABSENT'} vs "
                f"present; loss_mode {r['loss_mode']} vs {base['loss_mode']}. "
                f"A gap measured against this cell moves domain, item kind "
                f"and token budget together.")
        md.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "preflight_refsets.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[saved] {OUT_DIR / 'preflight_refsets.md'}")


# ══════════════════════════════════════════════════════════════════════
# aggregate — the draw table
# ══════════════════════════════════════════════════════════════════════
def noreg_anchor() -> Tuple[Optional[float], Optional[str], bool]:
    """mean downstream |dR| @300 of the PAPER ROUND's lambda=0 run (0.8434,
    the value Table 2 reports as "no reg"). Falls back to the superseded
    pre-gradient-fix baseline only if the paper tree is missing, and says so
    — quoting a reduction % against 1.4830 is a protocol error, not a round
    difference."""
    got = step_ck(REPO / NOREG_PAPER)
    if got:
        return (summarise_ck(got["ck"])["mean_dR_downstream"],
                NOREG_PAPER, False)
    got = step_ck(REPO / NOREG_SUPERSEDED)
    if got:
        return (summarise_ck(got["ck"])["mean_dR_downstream"],
                NOREG_SUPERSEDED, True)
    return None, None, False


def aggregate(cells: List[Tuple[str, int, int, str]], lam: float, seed: int,
              model_short: str, task: str, draw0: Optional[Path],
              replicate_tag: str) -> None:
    root = OUT_DIR / "reg_refdraw"
    found: List[Dict] = []

    if draw0 is not None and draw0.exists():
        got = step_ck(draw0)
        if got:
            exp = got["experiment"]
            s = summarise_ck(got["ck"])
            if s["mean_dR_downstream"] is not None:
                found.append({
                    "tag": f"{task}_n32_off0", "source": "paper tree (draw 0)",
                    "path": str(draw0.relative_to(REPO)),
                    "ref_seed": exp.get("ref_seed", seed + 1000),
                    "ref_offset": exp.get("ref_offset", 0),
                    "reg_samples": exp.get("reg_samples"),
                    "lr": exp.get("lr"), "lambda_reg": exp.get("lambda_reg", lam),
                    **s,
                })
        else:
            print(f"  [warn] draw-0 anchor has no step-{ANALYSIS_STEP} "
                  f"record: {draw0}")

    for ref_task, n, offset, suffix in cells:
        tag = cell_tag(ref_task, n, offset, suffix)
        p = root / tag / CELL_JSON.format(lam=lam, seed=seed,
                                          model_short=model_short, task=task)
        got = step_ck(p)
        if not got:
            print(f"  [pending] {tag} — no step-{ANALYSIS_STEP} record "
                  f"at {p}")
            continue
        exp = got["experiment"]
        s = summarise_ck(got["ck"])
        if s["mean_dR_downstream"] is None:
            print(f"  [skip] {tag} — step-{ANALYSIS_STEP} record has no "
                  f"downstream tasks")
            continue
        env = exp.get("env") or {}
        found.append({
            "tag": tag, "source": "E15", "path": str(p.relative_to(REPO)),
            "ref_seed": exp.get("ref_seed"),
            "ref_offset": exp.get("ref_offset"),
            "reg_samples": exp.get("reg_samples"),
            "lr": exp.get("lr"), "lambda_reg": exp.get("lambda_reg"),
            "env": "/".join(env.get(k, "?") for k in ("torch", "transformers",
                                                      "peft")),
            **s,
        })

    if not found:
        sys.exit("no completed cells — run script_E15.sh first")

    a0, a0src, a0_stale = noreg_anchor()

    # protocol guard: pooling cells that differ in lr / lambda / n is exactly
    # the E3 part-C postmortem failure mode. Refuse to hide it.
    lrs = {f["lr"] for f in found}
    lams = {f["lambda_reg"] for f in found if f["lambda_reg"] is not None}
    ns = {f["reg_samples"] for f in found if f["reg_samples"] is not None}
    bad = [name for name, s in (("lr", lrs), ("lambda", lams),
                                ("reg_samples", ns)) if len(s) > 1]
    guard = ("OK — lr, lambda and reference size identical across cells"
             if not bad else
             f"**MIXED PROTOCOL, DO NOT POOL**: {', '.join(bad)} differ "
             f"(lr {sorted(lrs)}, lambda {sorted(lams)}, n {sorted(ns)})")
    # Library versions are recorded per cell (older paper-round JSONs have no
    # env block, so only compare cells that carry one).
    envs = {f["env"] for f in found if f.get("env")}
    if len(envs) > 1:
        guard += (f"  ⚠️ **MIXED ENVIRONMENT** across cells "
                  f"(torch/transformers/peft: {sorted(envs)}) — the draw "
                  f"spread now also contains a library difference")

    # The spread is over cells that went through THIS script path. The
    # paper-tree draw 0 was launched from train_forgetting_multitask.py, so
    # pooling it with the E15 cells would fold a script-path difference into
    # the draw sd. It serves as the reproduction anchor instead, and the
    # replicate cell (offset 0 through this script) is draw 0 of the paired
    # set.
    draws = [f for f in found if f["source"] == "E15"]
    vals = [f["mean_dR_downstream"] for f in draws]

    md = [f"# E15 — paired reference-draw ablation", "",
          f"trace lambda={lam:g}, {model_short} / {task}, training seed "
          f"{seed}, step {ANALYSIS_STEP}, reference pool = the task's own "
          f"test split.", "",
          f"- protocol guard: {guard}",
          f"- no-reg (lambda=0) mean downstream |dR| = "
          + (f"{a0:.4f} [`{a0src}`]" if a0 is not None else "NOT FOUND")
          + ("  ⚠️ **SUPERSEDED BASELINE** (pre-gradient-fix, 700-step "
             "schedule): do not quote a reduction % against it — the paper "
             "round's no-reg is 0.8434 in "
             "`regularization_exp/exp_result/regularization_replay/0.0/`"
             if a0_stale else ""),
          "",
          "Every cell shares the training seed, data order, LoRA init, "
          "lambda, lr, reg_every_k and step budget. The single manipulated "
          "variable is which 32 sequences form D_ref, taken as disjoint "
          "windows of one fixed shuffle — see `preflight_refsets.md` for the "
          "disjointness certificate and the per-draw token counts.", "",
          "| cell | ref_seed | offset | mean downstream \\|dR\\| | mean Omega "
          "| target loss_P | " + " | ".join(f"\\|dR\\| {t}" for t in DOWNSTREAM)
          + " | source |",
          "|---" * (7 + len(DOWNSTREAM)) + "|"]
    rows_csv = []
    for f in found:
        md.append(
            f"| {f['tag']} | {f['ref_seed']} | {f['ref_offset']} "
            f"| {fmt(f['mean_dR_downstream'])} "
            f"| {fmt(f['mean_omega_downstream'])} "
            f"| {fmt(f['target_loss_P'])} | "
            + " | ".join(fmt(f["per_bench_dR"].get(t)) for t in DOWNSTREAM)
            + f" | {f['source']} |")
        rows_csv.append({
            "cell": f["tag"], "source": f["source"],
            "ref_seed": f["ref_seed"], "ref_offset": f["ref_offset"],
            "reg_samples": f["reg_samples"], "lr": f["lr"],
            "lambda_reg": f["lambda_reg"],
            "mean_dR_downstream": f["mean_dR_downstream"],
            "mean_omega_downstream": f["mean_omega_downstream"],
            "target_loss_P": f["target_loss_P"],
            **{f"dR_{t}": f["per_bench_dR"].get(t) for t in DOWNSTREAM},
            "path": f["path"],
        })

    md += ["", "## Spread across draws (this script path only)", ""]
    if len(vals) >= 2:
        m, sd = statistics.mean(vals), statistics.stdev(vals)
        md.append(f"- draws pooled: **{m:.4f} ± {sd:.4f}** nats "
                  f"(n={len(vals)}, CV {100 * sd / m:.1f}%, range "
                  f"{min(vals):.4f}–{max(vals):.4f}, spread "
                  f"{max(vals) - min(vals):.4f})")
        if a0 is not None and not a0_stale:
            red = [100 * (a0 - v) / a0 for v in vals]
            md.append(f"- benefit vs the paper round's no-reg ({a0:.4f}) "
                      f"spans {min(red):.1f}%–{max(red):.1f}% across draws")
    else:
        md.append(f"- only {len(vals)} draw(s) complete — spread undefined")

    rep = next((f for f in found if f["tag"] == replicate_tag), None)
    paper = next((f for f in found if f["source"].startswith("paper")), None)
    if rep and paper:
        d = abs(rep["mean_dR_downstream"] - paper["mean_dR_downstream"])
        md.append(
            f"- **reproduction canary / floor**: the replicate of draw 0 "
            f"(same reference set, same config, this script) lands at "
            f"{rep['mean_dR_downstream']:.4f} vs the paper-tree "
            f"{paper['mean_dR_downstream']:.4f}, i.e. |Δ| = {d:.4f} nats with "
            f"the reference set held FIXED. That covers nondeterminism plus "
            f"the multitask→baselines script path, so it is a CONSERVATIVE "
            f"floor for the draw effect.")
        if len(vals) >= 2 and d > 0:
            md.append(f"- draw spread / floor = "
                      f"{(max(vals) - min(vals)) / d:.2f}× — at or below ~1× "
                      f"means the draw is indistinguishable from run noise")
        elif len(vals) >= 2:
            md.append("- floor measured as exactly 0 — treat the draw spread "
                      "as the whole effect")
    else:
        md.append(f"- replicate cell `{replicate_tag}` not complete — the "
                  f"draw spread has no measured floor yet, so report it as an "
                  f"UPPER BOUND on the draw effect")

    md += ["", "Reading: this is the ablation 8VrD-Q3 asks for, run inside "
           "the paper's own setting. It answers 'is the specific "
           "32-sequence reference set load-bearing?' without moving domain, "
           "item kind and token budget at the same time — the three "
           "confounds bundled into E3-C's WikiText contrast, quantified in "
           "`preflight_refsets.md`. Every cell shares the paper round's "
           "protocol (lr 1e-5, seed 42, k 8, 300 steps, n_ref 32), so the "
           "absolute gaps are directly comparable to the paper's own "
           "lambda sweep: no reg 0.8434 / trace 0.5 0.7358 / trace 1.0 "
           "0.6813.", ""]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "refdraw_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    (OUT_DIR / "refdraw_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[saved] {OUT_DIR / 'refdraw_summary.md'}")


# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preflight", action="store_true",
                    help="materialise and certify the reference sets (zero GPU)")
    ap.add_argument("--aggregate", action="store_true",
                    help="build the draw table from completed cells (zero GPU)")
    ap.add_argument("--cells", default=None,
                    help="'ref_task:n:offset[:suffix]' list; preflight "
                         "defaults to the four task draws + the wikitext "
                         "point, aggregate to the three new draws + replicate")
    ap.add_argument("--ref_seed", type=int, default=1042,
                    help="shuffle seed of the reference pool (default 1042 = "
                         "paper seed 42 + 1000)")
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--reg_batch_size", type=int, default=8)
    ap.add_argument("--lambda_reg", type=float, default=1.0,
                    help="1.0 = the paper's headline operating point (Table 2 "
                         "trace, 0.6813); 0.5 = the point E3-C's size "
                         "ablation sits on (0.7358)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_short", default="llama-3.1-8b")
    ap.add_argument("--task", default=FT_TASK)
    ap.add_argument("--draw0", default=None,
                    help="JSON of the paper-round run that IS draw 0 "
                         "(reference offset 0 of shuffle 1042); default "
                         "resolves from --lambda_reg against the paper round")
    ap.add_argument("--replicate_tag", default="truthfulqa_n32_off0_rep",
                    help="cell tag holding the same-config rerun of draw 0")
    args = ap.parse_args()

    if not (args.preflight or args.aggregate):
        ap.error("pick --preflight and/or --aggregate")

    if args.preflight:
        preflight(parse_cells(args.cells or DEFAULT_PREFLIGHT_CELLS),
                  args.tokenizer, args.ref_seed, args.max_length,
                  args.reg_batch_size)
    if args.aggregate:
        # "none" (or an empty string) suppresses the anchor entirely, which is
        # what per-size blocks need: the paper tree's draw 0 is n=32, so
        # pooling it into an n=8 or n=16 block would mix reference sizes and
        # (correctly) trip the protocol guard.
        if args.draw0 in ("none", ""):
            rel = None
        else:
            rel = args.draw0 or DRAW0_BY_LAMBDA.get(args.lambda_reg)
        if rel is None:
            print(f"  [warn] no paper-round draw-0 anchor known for lambda="
                  f"{args.lambda_reg:g} — the replicate cell becomes the only "
                  f"reference for the floor")
        p = REPO / rel if rel else None
        aggregate(parse_cells(args.cells or DEFAULT_AGGREGATE_CELLS),
                  args.lambda_reg, args.seed, args.model_short, args.task,
                  p, args.replicate_tag)


if __name__ == "__main__":
    main()
