#!/usr/bin/env bash
# ============================================================
# E4 — Single-axis controlled interventions      (8VrD-W3)
#
# scale-only (final-norm rescale) / shape-only (norm-preserving
# rotation) / head-only (RTN-quantised lm_head) injected into ONE
# base model; shows selective response of each PRISM term with the
# empirical |dR| measured in the same pass.
#
# GPU budget: ~30-45 min per (model, dataset) on 1x RTX 5090.
# Default: Llama-3.1-8B on MMLU; add wikitext as a corpus control.
#
# Knobs: MODEL, DATASETS="mmlu wikitext", CUDA_GPU
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."           # repo root

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
DATASETS="${DATASETS:-mmlu wikitext}"
export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="rebuttal_exp/out/E4"
LOG="$OUT/screen.E4.${TS}.log"
mkdir -p "$OUT"
FAIL=0

for ds in $DATASETS; do
    echo "=== E4 $MODEL / $ds ($(date)) ===" | tee -a "$LOG"
    python rebuttal_exp/exp_e4_interventions.py \
        --model "$MODEL" --dataset "$ds" \
        2>&1 | tee -a "$LOG" || FAIL=1
done

echo "=== E4 done ($(date)); outputs in $OUT ===" | tee -a "$LOG"
exit $FAIL
