#!/usr/bin/env bash
# ============================================================
# PRISM — Multi-Task Catastrophic Forgetting Experiment
#
# LoRA fine-tune 2 models × 8 tasks.  PRISM forgetting metrics
# (Ω, δ, γ, Bound, |ΔR|) are computed online at every checkpoint
# via the built-in callback — no separate inference stage needed.
#
# For each fine-tuning task, PRISM eval runs on the trained task
# plus the 5 base tasks (arc, mmlu, squad, triviaqa, gsm8k) — 6 total.
#
# Terminology (paper Sec 3.2):
#   Target = base model θ₀  (frozen reference)
#   Proxy  = fine-tuned θ_t  (drifting model)
#
# Environment variables (all optional):
#   CUDA_GPU=0          GPU index (default: 0)
#   MODELS="llama qwen" Which models (default: both)
#   TASKS="truthfulqa bbq social_iqa arc mmlu squad triviaqa gsm8k"
#
# Examples:
#   bash run_forgetting_multitask.sh
#   CUDA_GPU=0 MODELS="llama" TASKS="truthfulqa" bash run_forgetting_multitask.sh
#   MODELS="qwen" TASKS="bbq social_iqa" bash run_forgetting_multitask.sh
# ============================================================
set -euo pipefail

# ── GPU ───────────────────────────────────────────────────────────────────
GPUID="${CUDA_GPU:-0}"

# ── Model definitions ─────────────────────────────────────────────────────
declare -A MODEL_IDS
MODEL_IDS[llama]="meta-llama/Llama-3.1-8B"
MODEL_IDS[qwen]="Qwen/Qwen3-8B-Base"

# ── Parameters ────────────────────────────────────────────────────────────
MODELS="${MODELS:-llama qwen}"
TASKS="${TASKS:-truthfulqa bbq social_iqa arc mmlu squad triviaqa gsm8k}"

CKPT_ROOT="./checkpoints/forgetting_multitask"
LOG="screen_forgetting_multitask.log"
ALT_LOG="screen.forgetting.log"

# ── Helpers ───────────────────────────────────────────────────────────────
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG" "$ALT_LOG"; }

# ── Main loop ─────────────────────────────────────────────────────────────
log ""
log "============================================================"
log "  PRISM Forgetting — LoRA Fine-Tuning + Online Monitoring"
log "  Models : $MODELS"
log "  Tasks  : $TASKS"
log "  GPU    : $GPUID"
log "============================================================"

for MODEL_KEY in $MODELS; do
    MODEL_ID="${MODEL_IDS[$MODEL_KEY]}"
    MODEL_SHORT=$(echo "$MODEL_ID" | awk -F'/' '{print tolower($NF)}')

    for TASK in $TASKS; do
        OUT_DIR="${CKPT_ROOT}/${MODEL_SHORT}/${TASK}"

        # Skip if PRISM metrics JSON already exists
        if [ -f "$OUT_DIR/prism_forgetting_metrics.json" ]; then
            log "  [skip] $MODEL_KEY / $TASK — prism_forgetting_metrics.json exists"
            continue
        fi

        log ""
        log "─── model=$MODEL_KEY  task=$TASK ───"

        CUDA_VISIBLE_DEVICES="$GPUID" python train_forgetting_multitask.py \
            --model "$MODEL_ID" \
            --task "$TASK" \
            --output_dir "$OUT_DIR" \
            --lr 2e-5 \
            2>&1 | tee -a "$LOG" "$ALT_LOG"

        log "─── model=$MODEL_KEY  task=$TASK  done ───"
    done
done

log ""
log "========================================"
log "  All runs complete."
log "  Checkpoints + PRISM metrics: $CKPT_ROOT/"
log "  Log: $LOG"
log "========================================"
