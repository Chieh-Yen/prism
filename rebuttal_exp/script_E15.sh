#!/usr/bin/env bash
# ============================================================
# E15 — PAIRED reference-draw ablation, regularizer side  (8VrD-Q3)
#
#   The question E3 part C left unpaired: at the paper's own n=32, does it
#   matter WHICH 32 sequences are the reference set? E3-C answered a
#   different question (task-domain vs WikiText) with a contrast that moves
#   domain, item kind (prompt boundary / loss_mode) and token budget at
#   once. E15 holds all three fixed and moves only the draw.
#
#   Design: reference = the task's OWN test split (paper setting), n=32,
#   shuffle seed 1042 (= paper seed 42 + 1000), sliced into DISJOINT
#   windows. offset 0 IS the paper-round run — free anchor, not rerun:
#     LAMBDA=1.0 -> 0.6813 = the trace column of the paper's Table 2
#     LAMBDA=0.5 -> 0.7358 = the point E3-C's size ablation sits on
#   (both from regularization_exp/exp_result/, the post-gradient-fix sweep;
#   the paper round's no-reg is 0.8434, NOT the 1.4830 in the top-level
#   exp_result/regularization/0.0 copy — that one is the superseded
#   1/8-gradient-bug, 700-step baseline.) New cells:
#
#     truthfulqa:32:32   draw 1   rows 32..63    of shuffle 1042
#     truthfulqa:32:64   draw 2   rows 64..95
#     truthfulqa:32:0:rep  replicate of draw 0 -> nondeterminism floor
#                          AND same-protocol reproduction canary
#     truthfulqa:32:96   draw 3   rows 96..127
#
#   Cell order is the order of evidentiary value: after 2 cells there is a
#   3-point spread (with the anchor), after 3 the spread has a measured
#   floor, the 4th tightens the sd. Measured on one RTX 5090, 300 steps,
#   Llama-3.1-8B: ~74 min at n=8, ~79 min at n=32, ~89 min at n=128 per
#   cell (E3 rerun tqdm bars + the paper round's own logs) -> ~5.3 h for
#   the four n=32 cells.
#
#   PART 0 (zero GPU, run it first, anywhere) certifies that the draws are
#   disjoint and that their token budgets match — and quantifies, in the
#   same table, how far the E3-C WikiText cell was from matched.
#
#   TruthfulQA's test pool is validation[80%:] = 163 rows, so FIVE disjoint
#   32-row windows fit (offsets 0/32/64/96/128, rows 0..159); offset 128 is
#   the spare draw if the spread wants a fifth point. offset+n > 163 raises
#   in the loader rather than silently wrapping.
#
#   SIZE LADDER (32 -> 16 -> 8), each size its own block of 4 cells:
#     N=32  E15_CELLS="truthfulqa:32:32 truthfulqa:32:64 \
#                      truthfulqa:32:0:rep truthfulqa:32:96"   (anchor free)
#     N=16  E15_CELLS="truthfulqa:16:0 truthfulqa:16:16 \
#                      truthfulqa:16:32 truthfulqa:16:48"      (10 windows)
#     N=8   E15_CELLS="truthfulqa:8:0 truthfulqa:8:8 \
#                      truthfulqa:8:16 truthfulqa:8:24"         (20 windows)
#   Only n=32 has a free offset-0 anchor (the paper's reference size), so the
#   16 and 8 blocks must run their own offset 0. The floor is a property of
#   the pipeline, not of n, so measure it once in the n=32 block and reuse it.
#   n=128 gives only ONE window in a 163-row pool, so a DISJOINT draw
#   ablation is impossible at that size: the second-best is offset 35
#   (35..162), which shares 93/128 rows. Report the overlap if you run it.
#
#   RESUME: cells that already hold a step-300 json are skipped outright;
#   a cell that died mid-run restarts from the newest checkpoint its metrics
#   json covers and keeps the earlier PRISM records (train_forgetting_
#   baselines.py --resume auto, the default). A config collision in an
#   existing output_root aborts instead of merging.
#
# Knobs: PARTS="0 A Z", CUDA_GPU, MODEL, LAMBDA=0.5, E15_CELLS, REF_SEED
#
# Usage:
#   PARTS=0   bash rebuttal_exp/script_E15.sh     # preflight only, no GPU
#   bash rebuttal_exp/script_E15.sh               # preflight + runs + table
#   PARTS=Z   bash rebuttal_exp/script_E15.sh     # re-aggregate only
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."           # repo root

PARTS="${PARTS:-0 A Z}"
# The paper-tree run (= draw 0) used exactly this repo string; keeping it
# identical is what makes the replicate a reproduction canary rather than a
# near-miss. (E3-C used the Meta-Llama alias; do not mix the two here.)
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
MODEL_SHORT="$(echo "${MODEL##*/}" | tr '[:upper:]' '[:lower:]')"
TASK="${TASK:-truthfulqa}"
# 1.0 = the paper's headline operating point (Table 2). LAMBDA=0.5 to pair
# with E3-C's size ablation instead.
LAMBDA="${LAMBDA:-1.0}"
SEED="${SEED:-42}"
REF_SEED="${REF_SEED:-$((SEED + 1000))}"
N="${N:-32}"
export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"

# "ref_task:n:offset[:suffix]" — suffix only renames the output dir, so the
# replicate can sit beside the original.
E15_CELLS="${E15_CELLS:-truthfulqa:${N}:32 truthfulqa:${N}:64 truthfulqa:${N}:0:rep truthfulqa:${N}:96}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="rebuttal_exp/out/E15"
LOG="$OUT/screen.E15.${TS}.log"
mkdir -p "$OUT"
FAIL=0

# ── Part 0: preflight (zero GPU) ────────────────────────────────────────
if [[ " $PARTS " == *" 0 "* ]]; then
    echo "=== E15 part 0 (reference-set preflight, zero GPU) ===" | tee -a "$LOG"
    python3 rebuttal_exp/exp_e15_refdraw_ablation.py --preflight \
        --ref_seed "$REF_SEED" --tokenizer "$MODEL" \
        2>&1 | tee -a "$LOG" || FAIL=1
fi

# ── Part A: the runs ────────────────────────────────────────────────────
cell_done () {   # output_root -> 0 if this cell's step-300 json exists
    # NOTE the json is prism_forgetting_metrics.json — script_E3.sh's
    # equivalent gate looked for prism_forgetting_metrics_<task>.json (the
    # paper-tree naming) and therefore never actually skipped anything.
    python3 - "$1" "$LAMBDA" "$MODEL_SHORT" "$TASK" "$SEED" <<'PY'
import json, os, sys
root, lam, model_short, task, seed = sys.argv[1:6]
p = os.path.join(root, "trace", f"lam{float(lam):g}", f"seed{seed}",
                 model_short, task, "prism_forgetting_metrics.json")
try:
    cks = json.load(open(p)).get("checkpoints", [])
    sys.exit(0 if any(c.get("step") == 300 for c in cks) else 1)
except Exception:
    sys.exit(1)
PY
}

if [[ " $PARTS " == *" A "* ]]; then
    echo "=== E15 part A (paired draws, GPU) ===" | tee -a "$LOG"
    for cell in $E15_CELLS; do
        IFS=: read -r ref_task n offset suffix <<< "$cell"
        suffix="${suffix:-}"
        tag="${ref_task}_n${n}_off${offset}${suffix:+_$suffix}"
        root="$OUT/reg_refdraw/${tag}"
        if cell_done "$root"; then
            echo "=== SKIP (step-300 json exists) $tag ===" | tee -a "$LOG"
            continue
        fi
        echo ">>> $tag  (ref_seed=$REF_SEED offset=$offset) $(date)" | tee -a "$LOG"
        # seed stays 42 for every cell: the training run must be IDENTICAL
        # across draws or the ablation stops being paired. --ref_seed is what
        # decouples the reference draw from it.
        python rebuttal_exp/train_forgetting_baselines.py \
            --model "$MODEL" --task "$TASK" \
            --method trace --lambda_reg "$LAMBDA" \
            --seed "$SEED" --max_steps 300 \
            --ref_task "$ref_task" --reg_samples "$n" \
            --ref_seed "$REF_SEED" --ref_offset "$offset" \
            --output_root "$root" \
            2>&1 | tee -a "$LOG" || FAIL=1
    done
fi

# ── Part Z: aggregate (zero GPU) ────────────────────────────────────────
if [[ " $PARTS " == *" Z "* ]]; then
    echo "=== E15 part Z (draw table, zero GPU) ===" | tee -a "$LOG"
    python3 rebuttal_exp/exp_e15_refdraw_ablation.py --aggregate \
        --cells "$E15_CELLS" --lambda_reg "$LAMBDA" --seed "$SEED" \
        --model_short "$MODEL_SHORT" --task "$TASK" \
        2>&1 | tee -a "$LOG" || FAIL=1
fi

echo "=== E15 done ($(date)); outputs in $OUT ===" | tee -a "$LOG"
exit $FAIL
