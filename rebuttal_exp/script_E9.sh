#!/usr/bin/env bash
# ============================================================
# E9 — GSM8K final-answer-span variant        (pCi8-W4)
#
# Restrict features AND per-token CE to the final-answer span
# ("#### <number>" onward) instead of the full CoT — does the
# graded span restore the rank signal that CoT averaging
# dilutes (App. F.3's SNR diagnosis)?
#
# GPU budget: ~1 h for the Llama family on 1x RTX 5090
#             (12 variant loads dominate; GSM8K only).
# Knobs: FAMILY=llama|qwen, NUM_SAMPLES=512, FALLBACK_K=8, CUDA_GPU
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."           # repo root

FAMILY="${FAMILY:-llama}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
FALLBACK_K="${FALLBACK_K:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="rebuttal_exp/out/E9"
LOG="$OUT/screen.E9.${TS}.log"
mkdir -p "$OUT"

python rebuttal_exp/exp_e9_answer_span.py \
    --family "$FAMILY" --num_samples "$NUM_SAMPLES" \
    --fallback_last_k "$FALLBACK_K" \
    2>&1 | tee -a "$LOG"

echo "=== E9 done ($(date)); outputs in $OUT ===" | tee -a "$LOG"
