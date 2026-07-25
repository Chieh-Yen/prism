#!/usr/bin/env bash
# ============================================================
# E8 — isometry-restriction cost vs bit-width     (pCi8-W6)
#
# iso_gain = 1 - res_lin/res_orth in the top-r principal
# subspace: how much residual does restricting alignment to
# rotation x scale COST versus arbitrary linear? ~0 = the
# near-isometry premise holds. Replaces the flawed in-E1
# iso_dev column (underdetermined when n_tokens < d).
#
# Reuses E1's target-feature cache; loads 5 proxies only.
# GPU budget: ~15-25 min (llama, mmlu+squad).
# Knobs: FAMILY, TIERS="FP16 Q8_0 Q4_K_M Q2_K NF4", CUDA_GPU
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."           # repo root

FAMILY="${FAMILY:-llama}"
export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="rebuttal_exp/out/E8"
LOG="$OUT/screen.E8.${TS}.log"
mkdir -p "$OUT"

# shellcheck disable=SC2086
python rebuttal_exp/exp_e8_isometry.py --family "$FAMILY" \
    ${TIERS:+--tiers $TIERS} \
    2>&1 | tee -a "$LOG"

echo "=== E8 done ($(date)); outputs in $OUT ===" | tee -a "$LOG"
