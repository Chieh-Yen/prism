#!/usr/bin/env bash
# ============================================================
# E3 — Reference-set ablation           (G3T9-W2, 8VrD-Q3)
#
#   Part A  benchmark-independent reference (ZERO GPU — reuses the
#           paper CSV's wikitext / fineweb_edu rows). Runs anywhere.
#   Part B  reference-SIZE ablation {8,16,32,64,128} on wikitext,
#           Llama family.                     ~1-1.5 h GPU
#   Part C  regularizer-side sensitivity (8VrD-Q3): trace lam=1.0,
#           Llama-TruthfulQA, sizes {8,32,128} x domains
#           {truthfulqa (paper), wikitext}.   6 runs x ~25 min ~= 2.5 h
#
# Knobs: PARTS="A B C", CUDA_GPU, FAMILY=llama
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."           # repo root

PARTS="${PARTS:-A B C}"
FAMILY="${FAMILY:-llama}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="rebuttal_exp/out/E3"
LOG="$OUT/screen.E3.${TS}.log"
mkdir -p "$OUT"
FAIL=0

if [[ " $PARTS " == *" A "* ]]; then
    echo "=== E3 part A (csv-only, zero GPU) ===" | tee -a "$LOG"
    python rebuttal_exp/exp_e3_refset_ablation.py --csv-only \
        2>&1 | tee -a "$LOG" || FAIL=1
fi

if [[ " $PARTS " == *" B "* ]]; then
    echo "=== E3 part B (size ablation, GPU) ===" | tee -a "$LOG"
    python rebuttal_exp/exp_e3_refset_ablation.py --family "$FAMILY" \
        2>&1 | tee -a "$LOG" || FAIL=1
fi

if [[ " $PARTS " == *" C "* ]]; then
    echo "=== E3 part C (regularizer sensitivity, GPU) ===" | tee -a "$LOG"
    # lambda follows the paper's llama-TQA operating point (0.5 — the value
    # with full E10 coverage); lr comes from the trainer's fixed 1e-5
    # default (the paper-round launch value). Override: LAMBDA=1.0.
    LAMBDA="${LAMBDA:-0.5}"
    for ref_task in truthfulqa wikitext; do
        for n in 8 32 128; do
            echo ">>> trace lam=$LAMBDA ref=$ref_task n=$n ($(date))" | tee -a "$LOG"
            # Per-config output_root: ref_task/reg_samples are NOT part of
            # the trainer's own directory scheme — without this, all six
            # configs overwrite one another (2026-07-24 postmortem).
            python rebuttal_exp/train_forgetting_baselines.py \
                --model "$MODEL" --task truthfulqa \
                --method trace --lambda_reg "$LAMBDA" --seed 42 --max_steps 300 \
                --ref_task "$ref_task" --reg_samples "$n" \
                --output_root "rebuttal_exp/out/E3/reg_sensitivity/${ref_task}_n${n}" \
                2>&1 | tee -a "$LOG" || FAIL=1
        done
    done
fi

echo "=== E3 done ($(date)); outputs in $OUT ===" | tee -a "$LOG"
exit $FAIL
