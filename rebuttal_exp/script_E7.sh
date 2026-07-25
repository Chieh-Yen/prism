#!/usr/bin/env bash
# ============================================================
# E7 — Free-running generation subset      (AC-D; G3T9-W2, 8VrD-W4/Q2)
#
# AC gives an either/or: run free-running experiments OR narrow the
# claims to teacher-forced. Strategy = both: this small experiment
# (one family, one benchmark subset, greedy decoding) + the corollary
# restated as teacher-forced-only in the revision.
#
# GPU budget: ~2-2.5 h for the Llama family at N=100 prompts, PER SEED.
# Multi-seed robustness (AC-D / 8VrD-W4-Q2): set SEEDS="42 43 44" to run three
# independent prompt subsets in one invocation. Greedy decoding is
# deterministic, so each seed only re-draws the prompt subset; the script
# auto-aggregates to mean +/- sd across seeds when >=2 seed CSVs exist.
# Knobs: FAMILY, DATASET, N_PROMPTS, MAX_NEW_TOKENS, MIN_NEW_TOKENS, SEEDS,
#        CUDA_GPU
# NOTE (2026-07-25 postmortem): DATASET=mmlu degenerates — multiple-choice
# prompts stop after ~1 greedy token, so the free-run column collapses onto
# teacher-forcing. Rerun with DATASET=gsm8k (natural long CoT continuations)
# or MIN_NEW_TOKENS=128 (EOS suppressed; disclose). The old mmlu CSV must be
# moved aside first, or resume will keep its degenerate rows.
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."           # repo root

FAMILY="${FAMILY:-llama}"
DATASET="${DATASET:-mmlu}"
# SEEDS: space-separated prompt-subset seeds. Back-compat: a single SEED still
# works (SEEDS defaults to it). seed 42 -> {fam}_{ds}_freerun.csv; seed s!=42
# -> {fam}_{ds}_s{s}_freerun.csv. For the robustness run use SEEDS="42 43 44".
SEEDS="${SEEDS:-${SEED:-42}}"
N_PROMPTS="${N_PROMPTS:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-0}"
LOW_RAM="${LOW_RAM:-0}"            # 1 = target not CPU-resident (small-RAM pods)
EXTRA=""
[[ "$LOW_RAM" == "1" ]] && EXTRA="--low_ram"
export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="rebuttal_exp/out/E7"
LOG="$OUT/screen.E7.${TS}.log"
mkdir -p "$OUT"

for SEED in $SEEDS; do
    echo ">>> E7 seed $SEED ($(date))" | tee -a "$LOG"
    python rebuttal_exp/exp_e7_freerun.py \
        --family "$FAMILY" --dataset "$DATASET" \
        --num_prompts "$N_PROMPTS" --max_new_tokens "$MAX_NEW_TOKENS" \
        --min_new_tokens "$MIN_NEW_TOKENS" --seed "$SEED" ${EXTRA:+$EXTRA} \
        2>&1 | tee -a "$LOG"
done

# Cross-seed mean +/- sd (zero GPU). Always fold in seed 42 (the primary
# subset) and dedupe, so `SEEDS="43 44"` still aggregates over all three; the
# aggregator silently skips any seed whose CSV is absent.
AGG_SEEDS="$(echo "42 $SEEDS" | tr ' ' '\n' | sort -un | tr '\n' ' ')"
if [[ "$(echo "$AGG_SEEDS" | wc -w)" -ge 2 ]]; then
    echo "=== E7 seed aggregation ($AGG_SEEDS) ===" | tee -a "$LOG"
    python rebuttal_exp/exp_e7_seed_aggregate.py \
        --family "$FAMILY" --dataset "$DATASET" --seeds $AGG_SEEDS \
        2>&1 | tee -a "$LOG" || true
fi

echo "=== E7 done ($(date)); outputs in $OUT ===" | tee -a "$LOG"
