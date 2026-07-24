#!/usr/bin/env bash
# ============================================================
# E7 — Free-running generation subset      (AC-D; G3T9-W2, 8VrD-W4/Q2)
#
# AC gives an either/or: run free-running experiments OR narrow the
# claims to teacher-forced. Strategy = both: this small experiment
# (one family, one benchmark subset, greedy decoding) + the corollary
# restated as teacher-forced-only in the revision.
#
# GPU budget: ~2-2.5 h for the Llama family at N=100 prompts.
# Knobs: FAMILY, DATASET, N_PROMPTS, MAX_NEW_TOKENS, CUDA_GPU
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."           # repo root

FAMILY="${FAMILY:-llama}"
DATASET="${DATASET:-mmlu}"
N_PROMPTS="${N_PROMPTS:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="rebuttal_exp/out/E7"
LOG="$OUT/screen.E7.${TS}.log"
mkdir -p "$OUT"

python rebuttal_exp/exp_e7_freerun.py \
    --family "$FAMILY" --dataset "$DATASET" \
    --num_prompts "$N_PROMPTS" --max_new_tokens "$MAX_NEW_TOKENS" \
    2>&1 | tee -a "$LOG"

echo "=== E7 done ($(date)); outputs in $OUT ===" | tee -a "$LOG"
