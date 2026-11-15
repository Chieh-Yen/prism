#!/usr/bin/env bash
# ============================================================
# PRISM forgetting: single-seed λ sweep (anchor-CE and shape regularizer)
#
# Wraps run_forgetting_one.sh to sweep λ for two penalties across
# {truthfulqa, bbq} on Llama, then on Qwen, with the library's default
# trainer at λ=0. Cross-entropy on the anchor set (anchor-CE; the anchor set
# is 32 held-out sequences of the fine-tuning task) is run first; the shape
# regularizer (1−Ω_I) follows. The λ=0 runs are the unregularized LoRA
# checkpoint series of Sec. 4.2; the sweep is tabulated by
# scripts/sweep_table.py. The forgetting comparison of Sec. 4.5 and
# Appendices H.2, H.3, I, J is not run here but by scripts/response/forgetting/.
#
# Run order (sequential):
#   for MODEL in llama, qwen:
#     Phase 1: ANCHOR-CE: for TASK in {truthfulqa, bbq}, for λ in REPLAY_LAMBDAS
#     Phase 2: SHAPE    : for TASK in {truthfulqa, bbq}, for λ in SHAPE_LAMBDAS
#
# Per-run output dirs are auto-disambiguated by (method, λ) via
# run_forgetting_one.sh's CKPT_ROOT logic, so re-running this
# script continues the sweep: the inner script skips a (model, task) run
# whose prism_forgetting_metrics.json holds the record of the final step, and
# starts an interrupted run again from step 0.
#
# Default λ grids (override via env vars); λ=0 in the anchor-CE grid is the
# unregularized run.
#
#   ANCHOR-CE (REPLAY_LAMBDAS): {0, 0.001, 0.005, 0.01, 0.05, 0.1}
#   SHAPE     (SHAPE_LAMBDAS) : {0.01, 0.05, 0.1, 0.5, 1.0}
#
# The λ=0 entry passes `--lambda_replay 0` (with `--lambda_shape` at its
# default 0) to train_forgetting.py, which then uses the library's default
# Trainer: no anchor set is built, no penalty is computed and no anchor
# forward pass is run. It is not the unregularized reference of the unified
# protocol (scripts/response/forgetting/), which runs the regularized training loop
# at `--lambda_shape 1e-9`; the two loops scale the loss differently under
# gradient accumulation (see the trainer selection in train_forgetting.py).
#
# Environment knobs:
#   CUDA_GPU=0                                       GPU index
#   REPLAY_LAMBDAS="0 0.001 0.005 0.01 0.05 0.1"     space-separated; an
#       empty value (REPLAY_LAMBDAS="") runs no anchor-CE run
#   SHAPE_LAMBDAS="0.01 0.05 0.1 0.5 1.0"            space-separated; an
#       empty value (SHAPE_LAMBDAS="") runs no shape-regularizer run
#   MODELS_ORDER="llama qwen"                        space-separated
#   TASKS_ORDER="truthfulqa bbq"                     space-separated
#
# Every run is started with --eval_keep_anchor_rows: the trainer's eval_loss
# is computed on the whole held-out slice of the fine-tuning task, which is
# what scripts/sweep_table.py reports as target loss.
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
#   REPLAY_LAMBDAS="0" SHAPE_LAMBDAS="" bash run_forgetting.sh
#                                             # unregularized checkpoint series only
# ============================================================
set -uo pipefail

GPUID="${CUDA_GPU:-0}"
MAX_STEPS="${MAX_STEPS:-300}"
# No colon in the expansion: an explicitly empty list is kept and runs nothing.
REPLAY_LAMBDAS="${REPLAY_LAMBDAS-0 0.001 0.005 0.01 0.05 0.1}"
SHAPE_LAMBDAS="${SHAPE_LAMBDAS-0.01 0.05 0.1 0.5 1.0}"
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
log "  Anchor-CE λ: $REPLAY_LAMBDAS  (${n_replay} values)"
log "  Shape  λ: $SHAPE_LAMBDAS   (${n_shape} values)"
log "  Total runs : $TOTAL_RUNS"
log "  Max steps  : $MAX_STEPS"
log "  Eval rows  : whole held-out slice (--eval_keep_anchor_rows)"
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
    log "── Phase 1: ANCHOR-CE (cross-entropy on the anchor set) ──"
    for TASK in $TASKS_ORDER; do
        for L in $REPLAY_LAMBDAS; do
            run_one "$MODEL" "$TASK" "replay" "$L"
        done
    done

    log ""
    log "── Phase 2: SHAPE (shape regularizer) ──"
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
