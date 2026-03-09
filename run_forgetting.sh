#!/usr/bin/env bash
# ============================================================
# PRISM — Catastrophic Forgetting Experiment (Llama-3-8B / GSM8K)
#
# Stage 1: Fine-tune Llama-3-8B on GSM8K, save checkpoint every 50 steps.
#   Three modes: full | head_only | backbone_only
#
# Stage 2: PRISM forgetting inference on every checkpoint.
#   Eval tasks: mmlu | arc | wikitext
#   Results written to ./results/forgetting/<mode>/<task>/
#
# Environment variables (all optional):
#   CUDA_GPU=0          GPU index (default: 0); ignored when MULTI_GPU=1
#   MULTI_GPU=1         Expose GPU 0+1; Trainer / HF will auto-distribute
#   MODES="full"        Space-separated subset of modes to run (default: all three)
#   STAGES="1 2"        Which stages to run (default: both)
#   MAX_STEPS=500       Training steps per mode (default: 500)
#   SAVE_STEPS=50       Checkpoint frequency in steps (default: 50)
#   N_EVAL=256          Samples per eval task in Stage 2 (default: 256)
#
# Examples:
#   bash run_forgetting.sh                           # all modes, both stages, GPU 0
#   CUDA_GPU=1 bash run_forgetting.sh                # GPU 1
#   MULTI_GPU=1 bash run_forgetting.sh               # multi-GPU
#   MODES="full" bash run_forgetting.sh              # single mode
#   STAGES="2" MODES="head_only" bash run_forgetting.sh  # stage 2 only
# ============================================================
set -euo pipefail

# ── GPU selection ──────────────────────────────────────────────────────────────
MULTI_GPU="${MULTI_GPU:-0}"
if [[ "$MULTI_GPU" == "1" ]]; then
    GPUIDS="0,1"
    DEVICE_OVERRIDE="device=auto"
else
    GPUIDS="${CUDA_GPU:-0}"
    DEVICE_OVERRIDE=""
fi

# ── Experiment parameters ──────────────────────────────────────────────────────
BASE_MODEL="meta-llama/Meta-Llama-3-8B"
CKPT_ROOT="./checkpoints/forgetting"
CFG="configs/forgetting_gsm8k.yaml"
LOG="screen_forgetting.log"

MODES="${MODES:-full head_only backbone_only}"
STAGES="${STAGES:-1 2}"
EVAL_TASKS="mmlu arc wikitext"

MAX_STEPS="${MAX_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-50}"
N_EVAL="${N_EVAL:-256}"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "$*" | tee -a "$LOG"; }

run_prism() {
    log ">>> python run.py $*"
    CUDA_VISIBLE_DEVICES="$GPUIDS" python run.py --config "$CFG" \
        ${DEVICE_OVERRIDE:+"$DEVICE_OVERRIDE"} "$@" 2>&1 | tee -a "$LOG"
}

# Build a PRISM-style list "[path1,path2,...]" from checkpoint directories
# sorted numerically by step number.
build_ckpt_list() {
    local mode="$1"
    python3 - <<PYEOF
import glob, os
ckpts = sorted(
    glob.glob("$CKPT_ROOT/$mode/checkpoint-*"),
    key=lambda p: int(p.rsplit("-", 1)[-1])
)
if not ckpts:
    raise SystemExit(f"No checkpoints found under $CKPT_ROOT/$mode/")
print("[" + ",".join(ckpts) + "]")
PYEOF
}

# ── Stage 1: Fine-tune ─────────────────────────────────────────────────────────
if echo "$STAGES" | grep -qw "1"; then
    log ""
    log "============================================================"
    log "  STAGE 1: Fine-tune $BASE_MODEL on GSM8K"
    log "  Modes   : $MODES"
    log "  Steps   : $MAX_STEPS  (save every $SAVE_STEPS)"
    log "============================================================"

    for MODE in $MODES; do
        log ""
        log "--- [Stage 1] mode=$MODE ---"
        CUDA_VISIBLE_DEVICES="$GPUIDS" python train_forgetting.py \
            --model   "$BASE_MODEL" \
            --mode    "$MODE" \
            --output_dir "$CKPT_ROOT/$MODE" \
            --max_steps  "$MAX_STEPS" \
            --save_steps "$SAVE_STEPS" \
            2>&1 | tee -a "$LOG"
        log "--- [Stage 1] mode=$MODE done ---"
    done

    log ""
    log "===== Stage 1 complete. Checkpoints under: $CKPT_ROOT ====="
fi

# ── Stage 2: PRISM forgetting inference ───────────────────────────────────────
if echo "$STAGES" | grep -qw "2"; then
    log ""
    log "============================================================"
    log "  STAGE 2: PRISM forgetting inference"
    log "  Modes : $MODES"
    log "  Tasks : $EVAL_TASKS"
    log "  n     : $N_EVAL samples per task"
    log "============================================================"

    for MODE in $MODES; do
        log ""
        log "--- [Stage 2] mode=$MODE ---"

        CKPT_LIST="$(build_ckpt_list "$MODE")"
        log "    Checkpoints: $CKPT_LIST"

        for TASK in $EVAL_TASKS; do
            log ""
            log "    [Stage 2] mode=$MODE  task=$TASK"
            run_prism \
                "target.model=$BASE_MODEL" \
                "proxy.checkpoints=$CKPT_LIST" \
                "data.task=$TASK" \
                "data.num_samples=$N_EVAL" \
                "output.dir=./results/forgetting/$MODE/$TASK"
        done

        log "--- [Stage 2] mode=$MODE done ---"
    done

    log ""
    log "===== Stage 2 complete. Results under: ./results/forgetting/ ====="
fi

log ""
log "========================================"
log "  run_forgetting.sh complete."
log "  Checkpoints : $CKPT_ROOT/"
log "  Results     : ./results/forgetting/"
log "  Log         : $LOG"
log "========================================"
