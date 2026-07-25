#!/usr/bin/env bash
# ============================================================
# E2 — Regularizer baselines: EWC / L2-SP / feature-KD
#      (pCi8-W5, G3T9-W3, 8VrD-W3/Q4, eQL6-W4)
#
# Three stages, run in order (each is resumable; individual run
# failures are recorded and the loop continues, per repo contract):
#
#   STAGE=sweep      4 methods (layer_freeze/ewc/l2sp/feature_kd) x 3-4
#                    grid points x 2 tasks x seed 42
#                    28 runs x ~25 min  ~= 12 h
#   STAGE=seeds      best-setting x seeds {43,44} for new methods (16 runs)
#                    + trace/replay/none x seeds {42,43,44} x 2 tasks
#                      (18 runs; regenerates complete step-300 anchors)
#                    34 runs           ~= 14 h
#   STAGE=aggregate  zero GPU — writes out/E2/E2_table.md
#
# Optional Qwen replication (best setting only, 1 seed):
#   STAGE=qwen       7 methods x 2 tasks = 14 runs ~= 7 h
#
# Total committed budget (sweep+seeds+aggregate): ~24 h on 1x RTX 5090.
# layer_freeze (LoRA on top-K layers only) is the AC-named low-cost
# continual-learning baseline — it runs FIRST in the sweep.
#
# Knobs: MODEL, TASKS, SEEDS, CUDA_GPU, MAX_STEPS (default 300).
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."           # repo root

STAGE="${STAGE:-sweep}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-8B}"
TASKS="${TASKS:-truthfulqa bbq}"
MAX_STEPS="${MAX_STEPS:-300}"
# trace anchor lambda for seeds/qwen stages. Default 0.5 = the only lambda
# with a complete paper-round step-300 run. Decision rule: after
# STAGE=backfill completes lambda=1.0 (and its canary MATCHes), compare
# 1.0@300 vs 0.5@300 (0.7357 for llama-TQA) — if 1.0 is sweep-best, rerun
# seeds with TRACE_LAM=1.0 instead of editing this file. (backfill itself
# always runs lambda=1.0 — that is its purpose; this knob does not affect it.)
TRACE_LAM="${TRACE_LAM:-0.5}"
export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="rebuttal_exp/out/E2"
LOG="$OUT/screen.E2.${STAGE}.${TS}.log"
mkdir -p "$OUT"
FAILURES=()

# Gated-repo lookups (PEFT adapter save) 401 without a token — harmless here
# (vocab never resized) but noisy; a token also speeds up dataset pulls.
if [[ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "[warn] HF_TOKEN not set: expect benign 401 warnings at PEFT save;" \
         "export HF_TOKEN to silence (see E2.md risk table)." | tee -a "$LOG"
fi

run_one () {  # method lambda seed task model [extra trainer args...]
    local method="$1" lam="$2" seed="$3" task="$4" model="$5"
    shift 5
    local tag="[$method lam=$lam seed=$seed $task $*]"
    echo ">>> $tag ($(date))" | tee -a "$LOG"
    python rebuttal_exp/train_forgetting_baselines.py \
        --model "$model" --task "$task" \
        --method "$method" --lambda_reg "$lam" \
        --seed "$seed" --max_steps "$MAX_STEPS" "$@" \
        2>&1 | tee -a "$LOG"
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        FAILURES+=("$tag")
        echo "!!! FAILED $tag" | tee -a "$LOG"
    fi
}

# Lambda grids (unit-mean Fisher makes ewc share l2sp's scale; see E2.md).
# l2sp/ewc get a 4th point (1e-1): at lr=1e-5 the LoRA drift ||theta-theta0||^2
# is small, so 1e-2 lands only "moderate" — without a strong-end point a
# boundary-hitting sweep-best would force a rerun.
L2SP_LAMS="1e-4 1e-3 1e-2 1e-1"
EWC_LAMS="1e-4 1e-3 1e-2 1e-1"
KD_LAMS="0.1 1.0 10"
FREEZE_TOPS="4 8 16"     # layer_freeze: LoRA on top-K layers (AC-named baseline)

run_freeze () {  # topK seed task model
    local top="$1" seed="$2" task="$3" model="$4"
    local tag="[layer_freeze top=$top seed=$seed $task]"
    echo ">>> $tag ($(date))" | tee -a "$LOG"
    python rebuttal_exp/train_forgetting_baselines.py \
        --model "$model" --task "$task" \
        --method layer_freeze --lora_top_layers "$top" \
        --seed "$seed" --max_steps "$MAX_STEPS" \
        2>&1 | tee -a "$LOG"
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        FAILURES+=("$tag")
        echo "!!! FAILED $tag" | tee -a "$LOG"
    fi
}

case "$STAGE" in
  sweep)
    # AC-named baselines first (layer_freeze, ewc), then the rest.
    for task in $TASKS; do
        for top in $FREEZE_TOPS; do run_freeze "$top" 42 "$task" "$MODEL"; done
        for lam in $EWC_LAMS;  do run_one ewc        "$lam" 42 "$task" "$MODEL"; done
        for lam in $L2SP_LAMS; do run_one l2sp       "$lam" 42 "$task" "$MODEL"; done
        for lam in $KD_LAMS;   do run_one feature_kd "$lam" 42 "$task" "$MODEL"; done
    done
    python rebuttal_exp/exp_e2_aggregate.py --with-paper-runs \
        2>&1 | tee -a "$LOG" || true
    ;;

  seeds)
    # Best lambdas from the sweep (fall back to grid mid-points).
    [[ -f "$OUT/best_lambdas.env" ]] && source "$OUT/best_lambdas.env"
    for task in $TASKS; do
        TU="$(echo "$task" | tr '[:lower:]' '[:upper:]')"
        var="BEST_LAYER_FREEZE_LLAMA_${TU}"
        top="${!var:-8}"
        for seed in 43 44; do
            run_freeze "$top" "$seed" "$task" "$MODEL"
        done
        for m in L2SP EWC FEATURE_KD; do
            var="BEST_${m}_LLAMA_${TU}"
            lam="${!var:-1e-3}"
            for seed in 43 44; do
                run_one "$(echo "$m" | tr '[:upper:]' '[:lower:]')" \
                        "$lam" "$seed" "$task" "$MODEL"
            done
        done
        # Paper-config anchors, multi-seed. Default TRACE_LAM=0.5: the only
        # operating point with a complete step-300 paper-round run (the old
        # "lam=1.0 -> 0.618" quote was a step-150 misread of an aborted run;
        # E2.md §4 erratum). If the lam=1.0 backfill turns out sweep-best,
        # rerun with TRACE_LAM=1.0. lr comes from the trainer's 1e-5 default.
        for seed in 42 43 44; do
            run_one none   0            "$seed" "$task" "$MODEL"
            run_one trace  "$TRACE_LAM" "$seed" "$task" "$MODEL"
            run_one replay 0.01         "$seed" "$task" "$MODEL"
        done
    done
    ;;

  qwen)
    [[ -f "$OUT/best_lambdas.env" ]] && source "$OUT/best_lambdas.env"
    for task in $TASKS; do
        TU="$(echo "$task" | tr '[:lower:]' '[:upper:]')"
        var="BEST_LAYER_FREEZE_LLAMA_${TU}"
        run_freeze "${!var:-8}" 42 "$task" "$QWEN_MODEL"
        for m in L2SP EWC FEATURE_KD; do
            var="BEST_${m}_LLAMA_${TU}"
            run_one "$(echo "$m" | tr '[:upper:]' '[:lower:]')" \
                    "${!var:-1e-3}" 42 "$task" "$QWEN_MODEL"
        done
        run_one none   0            42 "$task" "$QWEN_MODEL"
        run_one trace  "$TRACE_LAM" 42 "$task" "$QWEN_MODEL"
        run_one replay 0.01         42 "$task" "$QWEN_MODEL"
    done
    ;;

  backfill)
    # Complete the lambda=1.0 runs the paper-round trees left unfinished:
    #   llama/truthfulqa (interrupted @150), llama/bbq (@50).
    # qwen truthfulqa/bbq lam=1.0 are already complete — nothing to do there.
    # Fresh 300-step reruns, same seed/protocol (lr 1e-5 via trainer default;
    # reg_every_k=8 default — MATCHES the interrupted runs AND the tree's
    # lam 0.1/0.5 rows for TQA/bbq, so the completed lam grid stays
    # protocol-homogeneous. Do NOT pass k=4 here: it would break both the
    # canary comparability and the lam-grid pairing).
    # DOUBLES AS THE ENVIRONMENT-REPRODUCTION CANARY for the whole E2
    # campaign: the checker below compares steps 25..interrupt against the
    # old trajectories — a match certifies pod-round == paper-round
    # (same hardware class: the paper round also ran on a single RTX 5090).
    # Deliberately NOT backfilled: lima/no_robots lam=1.0 ({llama,qwen}).
    # Those FT tasks are the post-submission 5-task extension (E10), not
    # part of the submitted paper's 2x2 grid; the rebuttal only ever cites
    # their lambda*=0.5 numbers (already computed, zero GPU). Completing
    # their lam=1.0 column would cost ~6 h for a column nothing cites.
    for task in truthfulqa bbq; do
        run_one trace 1.0 42 "$task" "$MODEL"
    done
    python3 rebuttal_exp/exp_e2_backfill_check.py 2>&1 | tee -a "$LOG" || true
    ;;

  aggregate)
    python rebuttal_exp/exp_e2_aggregate.py --with-paper-runs \
        2>&1 | tee -a "$LOG"
    ;;

  *) echo "Unknown STAGE=$STAGE (sweep|seeds|qwen|aggregate|backfill)"; exit 2 ;;
esac

echo "=== E2 $STAGE done ($(date)) ===" | tee -a "$LOG"
if [[ ${#FAILURES[@]} -gt 0 ]]; then
    printf 'FAILED RUNS:\n%s\n' "${FAILURES[@]}" | tee -a "$LOG"
    exit 1
fi
