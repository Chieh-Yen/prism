#!/usr/bin/env bash
# ============================================================
# E1 — CKA / SVCCA / Procrustes-distance same-protocol ranking
#      comparison against PRISM.        (G3T9-W3, pCi8-W3)
#
# GPU budget: ~1.5-2.5 h per family on 1x RTX 5090
#             (dominated by proxy re-loading, esp. GGUF dequant).
# Families: llama (Meta-Llama-3.1-8B), qwen (Qwen3-8B-Base)
#           — the paper's main-text 2x5 grid. Ministral/DeepSeek
#           replication is deferred to the revision (E1.md sec. 5).
#
# Knobs:
#   FAMILIES="llama qwen"   subset of families
#   NUM_SAMPLES=512         per-benchmark samples (paper: 512)
#   TOKEN_CAP=16384         paired-token subsample for CKA/SVCCA
#   CUDA_GPU=0
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."           # repo root

FAMILIES="${FAMILIES:-llama qwen}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
TOKEN_CAP="${TOKEN_CAP:-16384}"
export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="rebuttal_exp/out/E1/screen.E1.${TS}.log"
mkdir -p rebuttal_exp/out/E1
FAIL=0

for fam in $FAMILIES; do
    echo "=== E1 family=$fam ($(date)) ===" | tee -a "$LOG"
    python rebuttal_exp/exp_e1_similarity_baselines.py \
        --family "$fam" \
        --num_samples "$NUM_SAMPLES" \
        --token_cap "$TOKEN_CAP" \
        2>&1 | tee -a "$LOG" || FAIL=1
done

echo "=== E1 done ($(date)); outputs in rebuttal_exp/out/E1/ ===" | tee -a "$LOG"
exit $FAIL
