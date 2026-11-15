#!/usr/bin/env bash
# ============================================================
# PRISM: single-configuration LoRA forgetting experiment (paper Sec. 4.5)
#
# LoRA fine-tune 2 base models × 2 fine-tuning tasks (TruthfulQA, BBQ).
# PRISM forgetting metrics (Ω, δ, γ, Bound, |ΔR|) are computed online at
# every checkpoint via a training callback, so no separate inference
# stage is needed. For each fine-tuning task, PRISM eval runs on the
# 5 downstream benchmarks (ARC, MMLU, SQuAD, TriviaQA, GSM8K),
# disjoint from the fine-tuning task.
#
# Called by run_forgetting.sh to sweep regularizer × λ; can also be
# invoked directly for a single configuration.
#
# Terminology (paper Sec. 3.1):
#   Target = base model θ₀  (frozen reference)
#   Proxy  = fine-tuned θ_t  (drifting model)
#
# Environment variables (all optional):
#   CUDA_GPU=0           GPU index (default: 0)
#   MODELS="llama qwen"  Models to fine-tune (default: both)
#   TASKS="truthfulqa bbq"  Fine-tuning tasks (default: both)
#   SHAPE_REG=1          Enable shape regularizer 1−Ω (default: off)
#   LAMBDA_SHAPE=0.1     Shape reg weight (default: 0.1). Setting it without
#                        SHAPE_REG=1 is an error.
#   REPLAY_REG=1         Enable replay-CE baseline (default: off,
#                        mutually exclusive with SHAPE_REG)
#   LAMBDA_REPLAY=0.1    Replay reg weight (default: 0.1). Setting it without
#                        REPLAY_REG=1 is an error.
#   EVAL_KEEP_ANCHOR_ROWS=1  (default 1) pass --eval_keep_anchor_rows to
#                        train_forgetting.py: the trainer's eval_loss is
#                        computed on the whole held-out slice of the
#                        fine-tuning task, as in the λ sweeps of the
#                        submission. With 0 the 32 anchor rows are removed
#                        from the eval set (unified protocol). Evaluation
#                        only; training is the same either way.
#   MAX_STEPS=300        Number of training steps. Unset when this script is
#                        called directly: train_forgetting.py then uses its
#                        task default (700 for truthfulqa and bbq).
#                        run_forgetting.sh passes 300, the step at which the
#                        paper reports results.
#
# With neither regularizer enabled, or with a weight of 0, train_forgetting.py
# uses the library's default Trainer (no anchor set, no penalty).
#
# A (model, task) run is skipped only when its prism_forgetting_metrics.json
# holds the record of the final step: the max_steps recorded in that file,
# which must equal MAX_STEPS when MAX_STEPS is set. Any other run directory
# (an interrupted run, or a run made with another number of steps) is started
# again from step 0 (train_forgetting.py has no resume) and its metrics file
# is overwritten.
#
# Examples:
#   bash run_forgetting_one.sh                                  # no-reg baseline
#   SHAPE_REG=1 LAMBDA_SHAPE=1.0 bash run_forgetting_one.sh     # shape regularizer
#   REPLAY_REG=1 LAMBDA_REPLAY=0.01 bash run_forgetting_one.sh  # replay
# ============================================================
set -euo pipefail

# ── GPU ───────────────────────────────────────────────────────────────────
GPUID="${CUDA_GPU:-0}"

# ── CUDA allocator: expandable segments reduce fragmentation with varying
#    sequence lengths (dynamic padding).  Numerics unchanged.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Model definitions ─────────────────────────────────────────────────────
declare -A MODEL_IDS
MODEL_IDS[llama]="meta-llama/Meta-Llama-3.1-8B"
MODEL_IDS[qwen]="Qwen/Qwen3-8B-Base"

# ── Parameters ────────────────────────────────────────────────────────────
MODELS="${MODELS:-llama qwen}"
TASKS="${TASKS:-truthfulqa bbq}"

# ── Regularizer selection (mutually exclusive) ───────────────────────────
# Set at most one of:
#   SHAPE_REG=1   + LAMBDA_SHAPE=...   → 1−Ω_I shape regularizer (PRISM)
#   REPLAY_REG=1  + LAMBDA_REPLAY=...  → CE on the same 32 anchor sequences (baseline)
SHAPE_REG="${SHAPE_REG:-0}"
REPLAY_REG="${REPLAY_REG:-0}"

# A weight without its switch would be ignored and the run would be an
# unregularized one written to the unregularized directory: refuse.
if [ -n "${LAMBDA_SHAPE+x}" ] && [ "$SHAPE_REG" != "1" ]; then
    echo "ERROR: LAMBDA_SHAPE=$LAMBDA_SHAPE is set but SHAPE_REG is not 1: set SHAPE_REG=1 or unset LAMBDA_SHAPE." >&2
    exit 2
fi
if [ -n "${LAMBDA_REPLAY+x}" ] && [ "$REPLAY_REG" != "1" ]; then
    echo "ERROR: LAMBDA_REPLAY=$LAMBDA_REPLAY is set but REPLAY_REG is not 1: set REPLAY_REG=1 or unset LAMBDA_REPLAY." >&2
    exit 2
fi

LAMBDA_SHAPE="${LAMBDA_SHAPE:-0.1}"
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
# from train_forgetting.py, e.g. 700 for truthfulqa/bbq).
if [ -n "${MAX_STEPS:-}" ]; then
    REG_ARGS="$REG_ARGS --max_steps $MAX_STEPS"
fi

# Evaluation rows of the trainer's eval_loss (see header). Default: the whole
# held-out slice, as in the submission's sweeps.
EVAL_KEEP_ANCHOR_ROWS="${EVAL_KEEP_ANCHOR_ROWS:-1}"
if [ "$EVAL_KEEP_ANCHOR_ROWS" = "1" ]; then
    REG_ARGS="$REG_ARGS --eval_keep_anchor_rows"
fi

CKPT_ROOT="./checkpoints/forgetting"
if [ "$SHAPE_REG" = "1" ]; then
    CKPT_ROOT="./checkpoints/forgetting_shape_lam${LAMBDA_SHAPE}"
elif [ "$REPLAY_REG" = "1" ]; then
    CKPT_ROOT="./checkpoints/forgetting_replay_lam${LAMBDA_REPLAY}"
fi
LOG="screen_forgetting_one.log"
ALT_LOG="screen_forgetting.log"

# ── Helpers ───────────────────────────────────────────────────────────────
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG" "$ALT_LOG"; }

# is_complete <metrics.json>: 0 if the file holds the record of the final
# step. The final step is the max_steps the file itself records; when
# MAX_STEPS is set, the recorded max_steps must equal it. The file is written
# from the first checkpoint on, so its existence alone does not mean the run
# finished.
is_complete() {
    [ -f "$1" ] || return 1
    python - "$1" "${MAX_STEPS:-}" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    final = int(d["experiment"]["max_steps"])
    if sys.argv[2] and int(sys.argv[2]) != final:
        raise ValueError("run was made with another number of steps")
    done = any(c.get("step") == final for c in d.get("checkpoints", []))
except Exception:
    done = False
sys.exit(0 if done else 1)
PY
}

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
if [ "$EVAL_KEEP_ANCHOR_ROWS" = "1" ]; then
log "  Eval   : whole held-out slice (--eval_keep_anchor_rows)"
else
log "  Eval   : held-out slice without the anchor rows"
fi
log "============================================================"

for MODEL_KEY in $MODELS; do
    MODEL_ID="${MODEL_IDS[$MODEL_KEY]}"
    MODEL_SHORT=$(echo "$MODEL_ID" | awk -F'/' '{print tolower($NF)}')

    for TASK in $TASKS; do
        OUT_DIR="${CKPT_ROOT}/${MODEL_SHORT}/${TASK}"

        # Skip only if the metrics JSON holds the record of the final step.
        if is_complete "$OUT_DIR/prism_forgetting_metrics.json"; then
            log "  [skip] $MODEL_KEY / $TASK: final step recorded in prism_forgetting_metrics.json"
            continue
        fi
        if [ -f "$OUT_DIR/prism_forgetting_metrics.json" ]; then
            log "  [restart] $MODEL_KEY / $TASK: prism_forgetting_metrics.json holds no finished run with this number of steps; the run starts again from step 0"
        fi

        log ""
        log "─── model=$MODEL_KEY  task=$TASK ───"

        CUDA_VISIBLE_DEVICES="$GPUID" python train_forgetting.py \
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
