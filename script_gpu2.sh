#!/usr/bin/env bash
# ============================================================
# GPU 2 — inference/quantization side + canary-gated trimmed E3C
#
#   1. E1 REDO=gsm8k (skip)  (~2 h; refreshes the 16k-subsample
#                            artifact cells; CKA/SVCCA only move down)
#   2. E12 cost-table regen (zero GPU; E1 gsm8k timings changed ->
#                            refresh draft 144.0 s series if median moved)
#   3. E8                   (~20 min; 2026-07-25 CUDA device fix — this
#                            run REQUIRES the fresh sync)
#   4. E3 part B            (~1.5-2 h; size ablation x 3 stability draws)
#   5. E9                   (~1 h; gsm8k answer-span)
#   6. E3C trimmed          (waits for GPU 1's canary verdict:
#                              MATCH -> 3 cells (tree anchor = (task,32))
#                              NEAR  -> 4 cells (adds fresh task:32 anchor)
#                              MISMATCH/absent -> skipped, message printed)
#
# Every stage is independently resumable (E1/E3B resume per variant;
# E3C skips completed cells) — safe to re-launch this wrapper.
#
# Knobs: CUDA_GPU (default 1), WAIT_CANARY_H (default 8), N_DRAWS
# Run AFTER ./sync_to_runpod.sh.
# ============================================================
set -uo pipefail
cd "$(dirname "$0")"

export CUDA_GPU="${CUDA_GPU:-1}"
CHECK="rebuttal_exp/out/E2/backfill_check.md"
LOG="rebuttal_exp/out/screen.gpu2.$(date +%Y%m%d_%H%M%S).log"
mkdir -p rebuttal_exp/out
FAIL=0

step () { echo "=== [gpu2 $(date '+%m-%d %H:%M')] $* ===" | tee -a "$LOG"; }

# step "E1 REDO=gsm8k (~1 h/family)"
# REDO="gsm8k" bash rebuttal_exp/script_E1.sh 2>&1 | tee -a "$LOG" || FAIL=1

step "E12 cost-table regen from refreshed E1 timings (zero GPU)"
SKIP_MEASURE=1 bash rebuttal_exp/script_E12.sh 2>&1 | tee -a "$LOG" || FAIL=1

step "E8 isometry lite (~20 min)"
bash rebuttal_exp/script_E8.sh 2>&1 | tee -a "$LOG" || FAIL=1

step "E3 part B: size ablation + stability draws (~1.5-2 h)"
PARTS="B" bash rebuttal_exp/script_E3.sh 2>&1 | tee -a "$LOG" || FAIL=1

step "E9 answer-span (~1 h)"
bash rebuttal_exp/script_E9.sh 2>&1 | tee -a "$LOG" || FAIL=1

# ---------------- E3C: wait for GPU 1's canary ----------------
WAIT_S=$(( ${WAIT_CANARY_H:-8} * 3600 ))
waited=0
while [[ ! -f "$CHECK" && $waited -lt $WAIT_S ]]; do
    step "waiting for canary ($CHECK) ... $((waited/60)) min elapsed"
    sleep 600; waited=$((waited + 600))
done

if [[ ! -f "$CHECK" ]]; then
    step "E3C SKIPPED: no canary verdict after ${WAIT_CANARY_H:-8} h — run later:"
    step "  E3C_CELLS=\"truthfulqa:8 truthfulqa:128 wikitext:32\" PARTS=C bash rebuttal_exp/script_E3.sh"
elif grep -q "Verdict: MISMATCH" "$CHECK"; then
    step "E3C SKIPPED: canary MISMATCH — no training on an unreproduced environment"
else
    CELLS="truthfulqa:8 truthfulqa:128 wikitext:32"
    if grep -q "Verdict: NEAR" "$CHECK"; then
        CELLS="truthfulqa:32 $CELLS"   # fresh (task,32) anchor: tree not citable
        step "canary NEAR -> E3C with fresh anchor, 4 cells (~6 h)"
    else
        step "canary MATCH -> E3C trimmed, 3 cells (~4.5 h; tree lam0.5@300 = (task,32) anchor)"
    fi
    E3C_CELLS="$CELLS" PARTS="C" bash rebuttal_exp/script_E3.sh \
        2>&1 | tee -a "$LOG" || FAIL=1
fi

step "gpu2 queue done (FAIL=$FAIL) — pull out/{E1,E12,E8,E3,E9} back"
exit $FAIL
