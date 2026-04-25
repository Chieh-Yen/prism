#!/usr/bin/env bash
# ============================================================
# PRISM — Multi-Task Catastrophic Forgetting Experiment
#
# LoRA fine-tune 2 models × 10 tasks.  PRISM forgetting metrics
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
# Bound constants (paper Eq. 8, Appendix A): K_feat = max_{j,k} ||h_j - h_k||_2
# and K_pred = sqrt(2) are now computed automatically from the base model's
# lm_head inside train_forgetting_multitask.py (UnifiedBound.theoretical_K).
# No K=1 placeholder — the bound Theorem 2 holds as stated.
#
# Environment variables (all optional):
#   CUDA_GPU=0          GPU index (default: 0)
#   MODELS="llama qwen" Which models (default: both)
#   TASKS="truthfulqa bbq social_iqa no_robots lima arc mmlu squad triviaqa gsm8k"
#   SHAPE_REG=1         Enable PRISM shape regularizer 1−Ω_I (default: off)
#   LAMBDA_SHAPE=0.1    Shape reg weight (default: 0.1)
#   REPLAY_REG=1        Enable replay-CE baseline (default: off, mutually
#                       exclusive with SHAPE_REG)
#   LAMBDA_REPLAY=0.1   Replay reg weight (default: 0.1)
#
# Examples:
#   bash run_forgetting_multitask.sh
#   SHAPE_REG=1 bash run_forgetting_multitask.sh
#   SHAPE_REG=1 LAMBDA_SHAPE=1.0 TASKS="arc" bash run_forgetting_multitask.sh
#   REPLAY_REG=1 LAMBDA_REPLAY=0.1 bash run_forgetting_multitask.sh   # baseline run
# ============================================================
set -euo pipefail

# ── GPU ───────────────────────────────────────────────────────────────────
GPUID="${CUDA_GPU:-0}"

# ── CUDA allocator: expandable segments reduce fragmentation on varying
#    sequence lengths (dynamic padding at max_length=1024 leaves ~10 GB
#    reserved-but-unallocated on Qwen3 without this).  Numerics unchanged.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Model definitions ─────────────────────────────────────────────────────
declare -A MODEL_IDS
MODEL_IDS[llama]="meta-llama/Llama-3.1-8B"
MODEL_IDS[qwen]="Qwen/Qwen3-8B-Base"

# ── Parameters ────────────────────────────────────────────────────────────
MODELS="${MODELS:-llama qwen}"
TASKS="${TASKS:-truthfulqa bbq social_iqa no_robots lima arc mmlu squad triviaqa gsm8k}"
TASKS="${TASKS:-no_robots lima truthfulqa bbq social_iqa squad triviaqa gsm8k arc mmlu}"

# ── Regularizer selection (mutually exclusive) ───────────────────────────
# Set at most one of:
#   SHAPE_REG=1   + LAMBDA_SHAPE=...   → 1−Ω_I shape regularizer (PRISM)
#   REPLAY_REG=1  + LAMBDA_REPLAY=...  → CE on the same 32 ref samples (baseline)
SHAPE_REG="${SHAPE_REG:-0}"
LAMBDA_SHAPE="${LAMBDA_SHAPE:-0.1}"
REPLAY_REG="${REPLAY_REG:-0}"
LAMBDA_REPLAY="${LAMBDA_REPLAY:-0.1}"

if [ "$SHAPE_REG" = "1" ] && [ "$REPLAY_REG" = "1" ]; then
    echo "ERROR: SHAPE_REG and REPLAY_REG are mutually exclusive — set only one." >&2
    exit 2
fi

REG_ARGS=""
if [ "$SHAPE_REG" = "1" ]; then
    REG_ARGS="--lambda_shape $LAMBDA_SHAPE --reg_every_k 8 --reg_samples 32"
elif [ "$REPLAY_REG" = "1" ]; then
    REG_ARGS="--lambda_replay $LAMBDA_REPLAY --reg_every_k 8 --reg_samples 32"
fi

# Optional: cap training to MAX_STEPS (else use task-specific defaults
# from train_forgetting_multitask.py, e.g. 700 for truthfulqa/bbq).
if [ -n "${MAX_STEPS:-}" ]; then
    REG_ARGS="$REG_ARGS --max_steps $MAX_STEPS"
fi

CKPT_ROOT="./checkpoints/forgetting_multitask"
if [ "$SHAPE_REG" = "1" ]; then
    CKPT_ROOT="./checkpoints/forgetting_multitask_shape_lam${LAMBDA_SHAPE}"
elif [ "$REPLAY_REG" = "1" ]; then
    CKPT_ROOT="./checkpoints/forgetting_multitask_replay_lam${LAMBDA_REPLAY}"
fi
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
log "  Alloc  : PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
if [ "$SHAPE_REG" = "1" ]; then
log "  Reg    : SHAPE   (λ=$LAMBDA_SHAPE)"
elif [ "$REPLAY_REG" = "1" ]; then
log "  Reg    : REPLAY  (λ=$LAMBDA_REPLAY)"
else
log "  Reg    : OFF"
fi
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
            --lr 1e-5 \
            $REG_ARGS \
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
