#!/usr/bin/env bash
# ============================================================
# Fixed-weight regularizer comparison on one cell: layer freezing, EWC and
# L2-SP next to the unregularized run, anchor-CE (`replay`) and the shape
# regularizer (`trace`), on Llama-3.1-8B / TruthfulQA, 300 steps. The weights
# are set in advance (stages below); `none` and `layer_freeze` use the
# library's default Trainer. The table of Appendix I of the paper is not produced here but by
# the unified protocol (scripts/response/forgetting/).
#
# Stages, run in order. A re-launched stage skips the runs whose metrics JSON
# holds the checkpoint at step MAX_STEPS (SKIP_DONE). An interrupted run of
# layer_freeze, none, trace or replay is continued from its last recorded
# checkpoint. An interrupted L2-SP or EWC run is not resumable (its reference
# point, the adapter initialisation, is not stored): the trainer stops with a
# message, the run is listed under FAILED RUNS, and it has to be restarted
# after deleting its run directory. A failed run is recorded and the loop
# continues.
#
#   STAGE=sweep   seed 42: layer_freeze top-K in {4, 8, 16}; EWC and L2-SP at
#                 lambda in {1e-4, 1e-3, 1e-2, 1e-1}                (11 runs)
#   STAGE=seeds   seeds 43 and 44 at one fixed configuration of each
#                 baseline (layer_freeze top 16, EWC
#                 lambda 0.1, L2-SP lambda 0.01)                    ( 6 runs)
#                 and seeds 42, 43, 44 of `none`, `trace` at lambda TRACE_LAM
#                 and `replay` at lambda 0.01                       ( 9 runs)
#   STAGE=table   no GPU: regularizer_table.py on the runs of MODEL
#
# Every run is started with --eval_keep_anchor_rows: the trainer's eval_loss
# (the "target loss" column of regularizer_table.py) is computed on the whole
# held-out slice of the fine-tuning task, anchor rows included.
# EVAL_KEEP_ANCHOR_ROWS=0 removes the anchor rows from the eval set instead
# (unified protocol). Evaluation only; training is the same either way.
#
# Knobs: MODEL, TASKS, CUDA_GPU, MAX_STEPS (300), TRACE_LAM (1.0),
#        SKIP_DONE (1), EVAL_KEEP_ANCHOR_ROWS (1).
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/../.."        # supplementary root

STAGE="${STAGE:-sweep}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B}"
TASKS="${TASKS:-truthfulqa}"
MAX_STEPS="${MAX_STEPS:-300}"
# lambda of the shape regularizer in the `seeds` stage (default 1.0).
TRACE_LAM="${TRACE_LAM:-1.0}"
# SKIP_DONE=1: a re-launched stage skips the runs whose metrics JSON already
# holds the checkpoint at step MAX_STEPS. SKIP_DONE=0 disables the check and
# launches every run of the stage; the trainer then refuses an existing L2-SP
# or EWC run directory (see the header) and the run is listed under FAILED
# RUNS.
SKIP_DONE="${SKIP_DONE:-1}"
EVAL_ARGS=()
if [[ "${EVAL_KEEP_ANCHOR_ROWS:-1}" == "1" ]]; then
    EVAL_ARGS=(--eval_keep_anchor_rows)
fi
export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="out/rounds/regularizers"
LOG="$OUT/run.${STAGE}.${TS}.log"
mkdir -p "$OUT"
FAILURES=()

is_done () {  # method lambda_or_topK seed task model  ->  0 if the final checkpoint is recorded
    [[ "$SKIP_DONE" == "1" ]] || return 1
    python3 - "$1" "$2" "$3" "$4" "$5" "$MAX_STEPS" "$OUT" <<'PY'
import json, os, sys
method, x, seed, task, model, steps, out = sys.argv[1:8]
tag = f"top{x}" if method == "layer_freeze" else f"lam{float(x):g}"
p = os.path.join(out, method, tag, f"seed{seed}", model.split("/")[-1].lower(), task,
                 "prism_forgetting_metrics.json")
try:
    cks = json.load(open(p)).get("checkpoints", [])
    sys.exit(0 if any(c.get("step") == int(steps) for c in cks) else 1)
except Exception:
    sys.exit(1)
PY
}

run_one () {  # method lambda seed task model
    local method="$1" lam="$2" seed="$3" task="$4" model="$5"
    local tag="[$method lam=$lam seed=$seed $task]"
    if is_done "$method" "$lam" "$seed" "$task" "$model"; then
        echo "=== SKIP (step-$MAX_STEPS checkpoint recorded) $tag ===" | tee -a "$LOG"
        return 0
    fi
    echo ">>> $tag ($(date))" | tee -a "$LOG"
    python scripts/rounds/train_forgetting_baselines.py \
        --model "$model" --task "$task" \
        --method "$method" --lambda_reg "$lam" \
        --seed "$seed" --max_steps "$MAX_STEPS" --output_root "$OUT" \
        ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} \
        2>&1 | tee -a "$LOG"
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        FAILURES+=("$tag")
        echo "!!! FAILED $tag" | tee -a "$LOG"
    fi
}

run_freeze () {  # topK seed task model
    local top="$1" seed="$2" task="$3" model="$4"
    local tag="[layer_freeze top=$top seed=$seed $task]"
    if is_done layer_freeze "$top" "$seed" "$task" "$model"; then
        echo "=== SKIP (step-$MAX_STEPS checkpoint recorded) $tag ===" | tee -a "$LOG"
        return 0
    fi
    echo ">>> $tag ($(date))" | tee -a "$LOG"
    python scripts/rounds/train_forgetting_baselines.py \
        --model "$model" --task "$task" \
        --method layer_freeze --lora_top_layers "$top" \
        --seed "$seed" --max_steps "$MAX_STEPS" --output_root "$OUT" \
        ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} \
        2>&1 | tee -a "$LOG"
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        FAILURES+=("$tag")
        echo "!!! FAILED $tag" | tee -a "$LOG"
    fi
}

# Sweep grids. The EWC Fisher is normalised to unit mean, so EWC and L2-SP
# share one lambda scale.
L2SP_LAMS="1e-4 1e-3 1e-2 1e-1"
EWC_LAMS="1e-4 1e-3 1e-2 1e-1"
FREEZE_TOPS="4 8 16"     # layer_freeze: LoRA on the top-K layers only

case "$STAGE" in
  sweep)
    for task in $TASKS; do
        for top in $FREEZE_TOPS; do run_freeze "$top" 42 "$task" "$MODEL"; done
        for lam in $EWC_LAMS;  do run_one ewc  "$lam" 42 "$task" "$MODEL"; done
        for lam in $L2SP_LAMS; do run_one l2sp "$lam" 42 "$task" "$MODEL"; done
    done
    ;;

  seeds)
    for task in $TASKS; do
        case "$task" in
            # The fixed configurations of the `seeds` stage (TruthfulQA).
            truthfulqa) FREEZE_TOP=16; EWC_LAM=1e-1; L2SP_LAM=1e-2 ;;
            *) echo "!! seeds: no configuration recorded for task=$task" | tee -a "$LOG"; exit 1 ;;
        esac
        for seed in 43 44; do
            run_freeze "$FREEZE_TOP" "$seed" "$task" "$MODEL"
            run_one ewc  "$EWC_LAM"  "$seed" "$task" "$MODEL"
            run_one l2sp "$L2SP_LAM" "$seed" "$task" "$MODEL"
        done
        for seed in 42 43 44; do
            run_one none   0            "$seed" "$task" "$MODEL"
            run_one trace  "$TRACE_LAM" "$seed" "$task" "$MODEL"
            run_one replay 0.01         "$seed" "$task" "$MODEL"
        done
    done
    ;;

  table)
    python3 scripts/rounds/regularizer_table.py --root "$OUT" --step "$MAX_STEPS" \
        --model "${MODEL##*/}" 2>&1 | tee -a "$LOG"
    ;;

  *) echo "Unknown STAGE=$STAGE (sweep|seeds|table)"; exit 2 ;;
esac

echo "=== $STAGE done ($(date)) ===" | tee -a "$LOG"
if [[ ${#FAILURES[@]} -gt 0 ]]; then
    { echo "FAILED RUNS:"; printf '%s\n' "${FAILURES[@]}"; } | tee -a "$LOG"
    exit 1
fi
