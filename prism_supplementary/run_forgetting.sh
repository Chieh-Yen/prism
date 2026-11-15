#!/usr/bin/env bash
# ============================================================
# PRISM forgetting: regularization sweep (replay and shape regularizer)
#
# The λ sweeps of the submission (Sec. 4.5). Wraps run_forgetting_one.sh
# to sweep λ for two regularizers across {truthfulqa, bbq} on Llama, then
# on Qwen. Replay (cross-entropy on the 32 anchor sequences, which are
# held-out sequences of the fine-tuning task; "anchor-CE" elsewhere in
# this package) is run first; the shape regularizer (1−Ω_I) follows.
#
# Run order (sequential):
#   for MODEL in llama, qwen:
#     Phase 1 — REPLAY:  for TASK in {truthfulqa, bbq}, for λ in REPLAY_LAMBDAS
#     Phase 2 — SHAPE :  for TASK in {truthfulqa, bbq}, for λ in SHAPE_LAMBDAS
#
# Per-run output dirs are auto-disambiguated by (method, λ) via
# run_forgetting_one.sh's CKPT_ROOT logic, so re-running this
# script continues the sweep: the inner script skips a (model, task) run
# whose prism_forgetting_metrics.json holds the record of the final step, and
# starts an interrupted run again from step 0.
#
# Default λ grids (override via env vars). These are the ranges of
# Sec. 4.1 of the paper, with λ=0 added to the replay grid as the
# unregularized run.
#
#   REPLAY: {0, 0.001, 0.005, 0.01, 0.05, 0.1}
#   SHAPE : {0.01, 0.05, 0.1, 0.5, 1.0}
#
# The λ=0 entry passes `--lambda_replay 0` (with `--lambda_shape` at its
# default 0) to train_forgetting.py, which then uses the library's default
# Trainer: no anchor set is built, no penalty is computed and no anchor
# forward pass is run. This is the unregularized run of the submission. It
# is not the unregularized reference of the unified protocol
# (scripts/response/forgetting/), which runs the regularized training loop
# at `--lambda_shape 1e-9`; the two loops scale the loss differently under
# gradient accumulation (see the trainer selection in train_forgetting.py).
#
# A single invocation of this wrapper covers the unregularized run, the
# replay sweep and the shape-regularizer sweep for both models and both
# fine-tuning tasks.
#
# Environment knobs:
#   CUDA_GPU=0                                       GPU index
#   REPLAY_LAMBDAS="0 0.001 0.005 0.01 0.05 0.1"     space-separated; an
#       empty value (REPLAY_LAMBDAS="") runs no replay run
#   SHAPE_LAMBDAS="0.01 0.05 0.1 0.5 1.0"            space-separated; an
#       empty value (SHAPE_LAMBDAS="") runs no shape-regularizer run
#   MODELS_ORDER="llama qwen"                        space-separated
#   TASKS_ORDER="truthfulqa bbq"                     space-separated
#   EVAL_KEEP_ANCHOR_ROWS=1                          1 (default): the trainer's
#       eval_loss is computed on the whole held-out slice of the fine-tuning
#       task (--eval_keep_anchor_rows), which is what the submission's sweeps
#       report as target loss. 0: the 32 anchor rows are removed from the eval
#       set, as in the unified protocol. Evaluation only.
#
# Table of the sweep (target loss and downstream |ΔR| per λ), no GPU:
#   python scripts/sweep_table.py --root checkpoints
#
# Examples:
#   bash run_forgetting.sh
#   REPLAY_LAMBDAS="0.05 0.1" SHAPE_LAMBDAS="0.5" bash run_forgetting.sh
#   MODELS_ORDER=llama TASKS_ORDER=truthfulqa bash run_forgetting.sh   # one cell
#   MODELS_ORDER=llama TASKS_ORDER=truthfulqa REPLAY_LAMBDAS="" bash run_forgetting.sh
#                                             # one cell, shape regularizer only
# ============================================================
set -uo pipefail

GPUID="${CUDA_GPU:-0}"
MAX_STEPS="${MAX_STEPS:-300}"
# No colon in the expansion: an explicitly empty list is kept and runs nothing.
REPLAY_LAMBDAS="${REPLAY_LAMBDAS-0 0.001 0.005 0.01 0.05 0.1}"
SHAPE_LAMBDAS="${SHAPE_LAMBDAS-0.01 0.05 0.1 0.5 1.0}"
MODELS_ORDER="${MODELS_ORDER:-llama qwen}"
TASKS_ORDER="${TASKS_ORDER:-truthfulqa bbq}"
EVAL_KEEP_ANCHOR_ROWS="${EVAL_KEEP_ANCHOR_ROWS:-1}"

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
log "  Eval rows  : $([ "$EVAL_KEEP_ANCHOR_ROWS" = "1" ] && echo "whole held-out slice (--eval_keep_anchor_rows)" || echo "held-out slice without the anchor rows")"
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
            EVAL_KEEP_ANCHOR_ROWS="$EVAL_KEEP_ANCHOR_ROWS" \
            REPLAY_REG=1 LAMBDA_REPLAY="$lam" \
            MODELS="$model" TASKS="$task" \
            bash run_forgetting_one.sh 2>&1 | tee -a "$LOG"
        rc=${PIPESTATUS[0]}
    elif [ "$method" = "shape" ]; then
        CUDA_GPU="$GPUID" MAX_STEPS="$MAX_STEPS" \
            EVAL_KEEP_ANCHOR_ROWS="$EVAL_KEEP_ANCHOR_ROWS" \
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
    log "── Phase 2 — SHAPE (shape regularizer) ──"
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
