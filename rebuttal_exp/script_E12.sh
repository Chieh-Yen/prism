#!/usr/bin/env bash
# ============================================================
# E12 — GPU-cost table                          (G3T9-W1)
#
# Two stages in one entry point:
#   1. E12b measurement (GPU, ~5-8 min at n=256): GSM8K greedy
#      decode x1 to natural EOS (ceiling 1024, hits counted) vs
#      the exact PRISM teacher-forced extraction call x1 — the
#      point: TF is MUCH faster than decoding, measured not
#      assumed. Replaces the biggest benchmark-side assumption.
#   2. Cost table (CPU, ~1 min): harvests E1 [load]/extract and
#      E7 [gen] timings, three-tier benchmark estimate
#      (floor / standard / maj@8), gsm8k tier = the measurement.
#
# Prereqs: E1 + E7 screen logs in out/E1, out/E7.
# Knobs: SAMPLES=256 (512 = zero-footnote), SKIP_MEASURE=1,
#        MODEL, CUDA_GPU
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."           # repo root

SAMPLES="${SAMPLES:-256}"
SKIP_MEASURE="${SKIP_MEASURE:-0}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B}"
export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="rebuttal_exp/out/E12"
LOG="$OUT/screen.E12.${TS}.log"
mkdir -p "$OUT"
FAIL=0

if [[ "$SKIP_MEASURE" != "1" ]]; then
    echo "=== E12b gsm8k measurement (GPU, ~5-8 min) ===" | tee -a "$LOG"
    python rebuttal_exp/exp_e12_gsm8k_measure.py \
        --model "$MODEL" --num_samples "$SAMPLES" \
        2>&1 | tee -a "$LOG" || FAIL=1
fi

echo "=== E12 cost table (CPU) ===" | tee -a "$LOG"
python rebuttal_exp/exp_e12_cost_table.py --model "$MODEL" \
    2>&1 | tee -a "$LOG" || FAIL=1

echo "=== E12 done ($(date)); outputs in $OUT ===" | tee -a "$LOG"
exit $FAIL
