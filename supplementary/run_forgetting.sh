#!/usr/bin/env bash
# ============================================================
# PRISM Forgetting — Regularization Sweep (Replay + Trace-Norm)
#
# Wraps run_forgetting_one.sh to sweep λ for two regularizers
# across {truthfulqa, bbq} on Llama, then on Qwen. Replay (CE on the
# 32 fixed reference samples) is run first as the apples-to-apples
# baseline; trace-norm (1−Ω_I shape regularizer) follows.
#
# Run order (sequential):
#   for MODEL in llama, qwen:
#     Phase 1 — REPLAY:  for TASK in {truthfulqa, bbq}, for λ in REPLAY_LAMBDAS
#     Phase 2 — SHAPE :  for TASK in {truthfulqa, bbq}, for λ in SHAPE_LAMBDAS
#
# Per-run output dirs are auto-disambiguated by (method, λ) via
# run_forgetting_one.sh's CKPT_ROOT logic, so re-running this
# script resumes (the inner script skips already-completed
# (model, task) combos via the prism_forgetting_metrics.json check).
#
# Default λ grids (override via env vars). These match the ranges in
# Sec.~5.1 of the paper, with λ=0 added to the replay grid as the no-reg
# anchor (the regularizer is computed but contributes 0 to the loss, so
# training is functionally equivalent to unregularized fine-tuning).
#
#   REPLAY: {0, 0.001, 0.005, 0.01, 0.05, 0.1}
#   SHAPE : {0.01, 0.05, 0.1, 0.5, 1.0}
#
# A single invocation of this wrapper covers every configuration reported
# in Sec.~5.4 (no-reg + replay sweep + trace-norm sweep, both models, both
# fine-tuning tasks).
#
# Environment knobs:
#   CUDA_GPU=0                                       GPU index
#   REPLAY_LAMBDAS="0 0.001 0.005 0.01 0.05 0.1"     space-separated
#   SHAPE_LAMBDAS="0.01 0.05 0.1 0.5 1.0"            space-separated
#   MODELS_ORDER="llama qwen"                        space-separated
#   TASKS_ORDER="truthfulqa bbq"                     space-separated
#
# Examples:
#   bash run_forgetting.sh
#   REPLAY_LAMBDAS="0.05 0.1" SHAPE_LAMBDAS="0.5" bash run_forgetting.sh
# ============================================================
set -uo pipefail
# Note: -e intentionally omitted — a single OOM / transient failure
#       shouldn't kill a multi-day sweep. Failures are logged + counted.

GPUID="${CUDA_GPU:-0}"
MAX_STEPS="${MAX_STEPS:-300}"
REPLAY_LAMBDAS="${REPLAY_LAMBDAS:-0 0.001 0.005 0.01 0.05 0.1}"
SHAPE_LAMBDAS="${SHAPE_LAMBDAS:-0.01 0.05 0.1 0.5 1.0}"
MODELS_ORDER="${MODELS_ORDER:-llama qwen}"
TASKS_ORDER="${TASKS_ORDER:-truthfulqa bbq}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="screen_forgetting_${TIMESTAMP}.log"
ln -sf "$LOG" screen_forgetting_latest.log

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ── Pre-compute total run count for progress display ──────────────────────
n_models=0; for _ in $MODELS_ORDER; do n_models=$((n_models + 1)); done
n_tasks=0;  for _ in $TASKS_ORDER;  do n_tasks=$((n_tasks + 1));   done
n_replay=0; for _ in $REPLAY_LAMBDAS; do n_replay=$((n_replay + 1)); done
n_shape=0;  for _ in $SHAPE_LAMBDAS;  do n_shape=$((n_shape + 1));   done
TOTAL_RUNS=$(( n_models * n_tasks * (n_replay + n_shape) ))
DONE=0
FAILED=()

log ""
log "============================================================"
log "  Forgetting regularization sweep"
log "  Models  : $MODELS_ORDER"
log "  Tasks   : $TASKS_ORDER"
log "  Replay λ: $REPLAY_LAMBDAS  (${n_replay} values)"
log "  Shape  λ: $SHAPE_LAMBDAS   (${n_shape} values)"
log "  Total runs : $TOTAL_RUNS"
log "  Max steps  : $MAX_STEPS"
log "  GPU: $GPUID"
log "  Log: $LOG"
log "============================================================"

run_one() {
    local model="$1" task="$2" method="$3" lam="$4"
    DONE=$((DONE + 1))

    log ""
    log "─── [${DONE}/${TOTAL_RUNS}]  ${model} / ${task} / ${method} λ=${lam} ───"

    local rc=0
    if [ "$method" = "replay" ]; then
        CUDA_GPU="$GPUID" MAX_STEPS="$MAX_STEPS" \
            REPLAY_REG=1 LAMBDA_REPLAY="$lam" \
            MODELS="$model" TASKS="$task" \
            bash run_forgetting_one.sh 2>&1 | tee -a "$LOG"
        rc=${PIPESTATUS[0]}
    elif [ "$method" = "shape" ]; then
        CUDA_GPU="$GPUID" MAX_STEPS="$MAX_STEPS" \
            SHAPE_REG=1 LAMBDA_SHAPE="$lam" \
            MODELS="$model" TASKS="$task" \
            bash run_forgetting_one.sh 2>&1 | tee -a "$LOG"
        rc=${PIPESTATUS[0]}
    fi

    if [ "$rc" -ne 0 ]; then
        FAILED+=("${model}/${task}/${method}/λ=${lam} (rc=${rc})")
        log "  ✗ FAILED: ${model}/${task}/${method}/λ=${lam} (rc=${rc})"
    fi
}

# ── Sweep ─────────────────────────────────────────────────────────────────
for MODEL in $MODELS_ORDER; do
    log ""
    log "════════════════════════════════════════════════════════════"
    log "  MODEL: $MODEL"
    log "════════════════════════════════════════════════════════════"

    log ""
    log "── Phase 1 — REPLAY baseline ──"
    for TASK in $TASKS_ORDER; do
        for L in $REPLAY_LAMBDAS; do
            run_one "$MODEL" "$TASK" "replay" "$L"
        done
    done

    log ""
    log "── Phase 2 — SHAPE (trace-norm) ──"
    for TASK in $TASKS_ORDER; do
        for L in $SHAPE_LAMBDAS; do
            run_one "$MODEL" "$TASK" "shape" "$L"
        done
    done
done

# ── Summary ───────────────────────────────────────────────────────────────
log ""
log "============================================================"
if [ "${#FAILED[@]}" -eq 0 ]; then
    log "  Sweep complete — all ${TOTAL_RUNS} runs OK."
else
    log "  Sweep complete with ${#FAILED[@]} failure(s) of ${TOTAL_RUNS}:"
    for f in "${FAILED[@]}"; do
        log "    ✗ $f"
    done
fi
log "  Log: $LOG"
log "============================================================"

[ "${#FAILED[@]}" -eq 0 ]
