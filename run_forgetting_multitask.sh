#!/usr/bin/env bash
# ============================================================
# PRISM — Multi-Task Catastrophic Forgetting Experiment
#
# Fine-tune 2 models × 5 tasks with LoRA, then run PRISM
# forgetting inference on all checkpoint × eval_task combinations.
#
# Stage 1: LoRA fine-tune on each task → save adapter checkpoints
# Stage 2: PRISM inference on every checkpoint × eval task
#
# Environment variables (all optional):
#   CUDA_GPU=0          GPU index (default: 0)
#   MODELS="llama qwen" Which models to run (default: both)
#   TASKS="arc mmlu squad triviaqa gsm8k"  Which tasks (default: all 5)
#   STAGES="1 2"        Which stages to run (default: both)
#   N_EVAL=256          Samples per eval task in Stage 2 (default: 256)
#
# Examples:
#   bash run_forgetting_multitask.sh
#   CUDA_GPU=0 MODELS="llama" TASKS="gsm8k" bash run_forgetting_multitask.sh
#   STAGES="2" bash run_forgetting_multitask.sh     # stage 2 only
#   MODELS="qwen" TASKS="arc mmlu" STAGES="1" bash run_forgetting_multitask.sh
# ============================================================
set -euo pipefail

# ── GPU selection ──────────────────────────────────────────────────────────
GPUID="${CUDA_GPU:-0}"

# ── Model definitions ─────────────────────────────────────────────────────
# Short name → HuggingFace model ID
declare -A MODEL_IDS
MODEL_IDS[llama]="meta-llama/Llama-3.1-8B"
MODEL_IDS[qwen]="Qwen/Qwen3-8B-Base"

# ── Parameters ────────────────────────────────────────────────────────────
MODELS="${MODELS:-llama qwen}"
TASKS="${TASKS:-arc mmlu squad triviaqa gsm8k}"
STAGES="${STAGES:-1 2}"
N_EVAL="${N_EVAL:-256}"

CKPT_ROOT="./checkpoints/forgetting_multitask"
RESULT_ROOT="./results/forgetting_multitask"
CFG="configs/forgetting_multitask.yaml"
LOG="screen_forgetting_multitask.log"

# ── Helpers ───────────────────────────────────────────────────────────────
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ── Stage 1: LoRA Fine-Tuning ─────────────────────────────────────────────
if echo "$STAGES" | grep -qw "1"; then
    log ""
    log "============================================================"
    log "  STAGE 1: LoRA Fine-Tuning"
    log "  Models : $MODELS"
    log "  Tasks  : $TASKS"
    log "============================================================"

    for MODEL_KEY in $MODELS; do
        MODEL_ID="${MODEL_IDS[$MODEL_KEY]}"
        MODEL_SHORT=$(echo "$MODEL_ID" | awk -F'/' '{print tolower($NF)}')

        for TASK in $TASKS; do
            log ""
            log "--- [Stage 1] model=$MODEL_KEY  task=$TASK ---"

            OUT_DIR="${CKPT_ROOT}/${MODEL_SHORT}/${TASK}"

            # Skip if checkpoints already exist
            if [ -d "$OUT_DIR" ] && ls "$OUT_DIR"/checkpoint-* >/dev/null 2>&1; then
                log "    Checkpoints already exist in $OUT_DIR — skipping."
                continue
            fi

            CUDA_VISIBLE_DEVICES="$GPUID" python train_forgetting_multitask.py \
                --model "$MODEL_ID" \
                --task "$TASK" \
                --output_dir "$OUT_DIR" \
                2>&1 | tee -a "$LOG"

            log "--- [Stage 1] model=$MODEL_KEY  task=$TASK done ---"
        done
    done

    log ""
    log "===== Stage 1 complete. Checkpoints under: $CKPT_ROOT ====="
fi

# ── Stage 2: PRISM Forgetting Inference ───────────────────────────────────
if echo "$STAGES" | grep -qw "2"; then
    log ""
    log "============================================================"
    log "  STAGE 2: PRISM Forgetting Inference"
    log "  Models : $MODELS"
    log "  Tasks  : $TASKS"
    log "  Eval   : arc mmlu squad triviaqa gsm8k"
    log "  n_eval : $N_EVAL"
    log "============================================================"

    for MODEL_KEY in $MODELS; do
        MODEL_ID="${MODEL_IDS[$MODEL_KEY]}"
        MODEL_SHORT=$(echo "$MODEL_ID" | awk -F'/' '{print tolower($NF)}')

        for TASK in $TASKS; do
            CKPT_DIR="${CKPT_ROOT}/${MODEL_SHORT}/${TASK}"
            OUT_DIR="${RESULT_ROOT}/${MODEL_SHORT}/trained_${TASK}"

            if [ ! -d "$CKPT_DIR" ]; then
                log "    WARNING: No checkpoints at $CKPT_DIR — skipping."
                continue
            fi

            log ""
            log "--- [Stage 2] model=$MODEL_KEY  trained=$TASK ---"

            CUDA_VISIBLE_DEVICES="$GPUID" python infer_forgetting_multitask.py \
                --config "$CFG" \
                --base_model "$MODEL_ID" \
                --checkpoint_dir "$CKPT_DIR" \
                --eval_tasks arc mmlu squad triviaqa gsm8k \
                --num_samples "$N_EVAL" \
                --output_dir "$OUT_DIR" \
                2>&1 | tee -a "$LOG"

            log "--- [Stage 2] model=$MODEL_KEY  trained=$TASK done ---"
        done
    done

    log ""
    log "===== Stage 2 complete. Results under: $RESULT_ROOT ====="
fi

log ""
log "========================================"
log "  run_forgetting_multitask.sh complete."
log "  Checkpoints : $CKPT_ROOT/"
log "  Results     : $RESULT_ROOT/"
log "  Log         : $LOG"
log "========================================"
