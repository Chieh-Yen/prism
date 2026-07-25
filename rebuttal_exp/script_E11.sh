#!/usr/bin/env bash
# ============================================================
# E11 — Base vs instruct: first data point beyond PTQ /
#       frozen-head LoRA                       (G3T9-Q1)
#
# Full PRISM decomposition (W = I, nonzero head term) between
# base and instruct checkpoints on the paper's 5 benchmarks —
# App. C.3's scoped-out regime, run end-to-end with a bound-
# validity check and a PTQ context anchor from the paper CSV.
#
# GPU budget: ~30-45 min per pair; default (llama qwen) ~1.5 h.
# Knobs: PAIRS="llama qwen" (also: mistral ministral qwen25),
#        NUM_SAMPLES=512, CUDA_GPU
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."           # repo root

PAIRS="${PAIRS:-llama qwen}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="rebuttal_exp/out/E11"
LOG="$OUT/screen.E11.${TS}.log"
mkdir -p "$OUT"

# shellcheck disable=SC2086
python rebuttal_exp/exp_e11_base_instruct.py \
    --pairs $PAIRS --num_samples "$NUM_SAMPLES" \
    2>&1 | tee -a "$LOG"

echo "=== E11 done ($(date)); outputs in $OUT ===" | tee -a "$LOG"
