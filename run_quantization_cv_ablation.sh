#!/usr/bin/env bash
# ============================================================
# PRISM — Control Variate Ablation: Sample Efficiency
#
# Tests how many calibration samples are needed for the
# geometric control variate estimator (Eq. 8 in the paper):
#
#   R̂_T(D_test) = R_P(D_test) + [R_T(D_cal) - R_P(D_cal)]
#
# Protocol:
#   1. Run the full quantization experiment with a large N_FULL
#      to obtain ground-truth per-sample losses for both target
#      and proxy.
#   2. For each calibration size n in CAL_SIZES:
#      - Subsample n samples as D_cal
#      - Compute the calibrated bias Δ̂ = R_T(D_cal) - R_P(D_cal)
#      - Estimate R̂_T = R_P(D_full) + Δ̂
#      - Report |R̂_T - R_T(D_full)| as estimation error
#
# The experiment reuses the quantization pipeline but with
# cv_ablation.calibration_sizes=[...] to trigger the ablation.
#
# Usage:
#   bash run_cv_ablation.sh                     # GPU 0
#   CUDA_GPU=1 bash run_cv_ablation.sh          # GPU 1
#   MULTI_GPU=1 bash run_cv_ablation.sh         # GPU 0+1
# ============================================================
set -euo pipefail

# ── GPU selection ────────────────────────────────────────────
MULTI_GPU="${MULTI_GPU:-0}"
if [[ "$MULTI_GPU" == "1" ]]; then
    GPUIDS="0,1"
    DEVICE_OVERRIDE="device=auto"
else
    GPUIDS="${CUDA_GPU:-0}"
    DEVICE_OVERRIDE=""
fi

# ── Experiment parameters ────────────────────────────────────
N_FULL=2000                                  # Full test set size (ground truth)
CAL_SIZES="50,100,200,500,1000"              # Calibration sizes to test
N_TRIALS=20                                  # Random subsampling trials per size
CFG="configs/quantization.yaml"
LOG="screen.cv_ablation.log"

run() {
    echo ">>> $*" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$GPUIDS" python run.py --config "$CFG" \
        ${DEVICE_OVERRIDE:+"$DEVICE_OVERRIDE"} \
        cv_ablation.calibration_sizes="[$CAL_SIZES]" \
        cv_ablation.n_trials=$N_TRIALS \
        "$@" 2>&1 | tee -a "$LOG"
}

# ── Datasets ─────────────────────────────────────────────────
# Focus on MMLU (14K full benchmark) and WikiText (LM loss)
# to demonstrate the estimator across both MC and LM paradigms.
DATASETS_CV="mmlu wikitext"

run_cv_datasets() {
    local args=("$@")
    for DS in $DATASETS_CV; do
        run "${args[@]}" data.task=$DS data.num_samples=$N_FULL
    done
}

# ============================================================
# Representative models (one per family)
# ============================================================

# --- Llama-3.1-8B-Instruct ---
LLAMA_TARGET="target.model=meta-llama/Meta-Llama-3.1-8B-Instruct"
LLAMA_GGUF="proxy.model=bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
LLAMA_BITS="proxy.quantization_bits=[\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:nf4,\
awq:hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4]"

run_cv_datasets $LLAMA_TARGET $LLAMA_GGUF "$LLAMA_BITS"

# --- Qwen3-8B (Instruct / Thinking) ---
QWEN_TARGET="target.model=Qwen/Qwen3-8B"
QWEN_GGUF="proxy.model=Qwen/Qwen3-8B-GGUF"
QWEN_BITS="proxy.quantization_bits=[\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:nf4,\
awq:Qwen/Qwen3-8B-AWQ]"

run_cv_datasets $QWEN_TARGET $QWEN_GGUF "$QWEN_BITS"

# --- Mistral-7B-Instruct-v0.3 ---
MIS_TARGET="target.model=mistralai/Mistral-7B-Instruct-v0.3"
MIS_GGUF="proxy.model=bartowski/Mistral-7B-Instruct-v0.3-GGUF"
MIS_BITS="proxy.quantization_bits=[\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:nf4,\
awq:solidrust/Mistral-7B-Instruct-v0.3-AWQ]"

run_cv_datasets $MIS_TARGET $MIS_GGUF "$MIS_BITS"

echo "========================================"
echo "  CV ablation complete."
echo "  Results in: ./results/quantization/"
echo "  Log:        $LOG"
echo "========================================"
