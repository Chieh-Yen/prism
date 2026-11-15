#!/usr/bin/env bash
# ============================================================
# PRISM: quantization experiment suite (paper Sec. 4.2)
# 5 datasets × 7 models × the PTQ variants listed in the matrix file.
#
# Source of truth for model/dataset matrix:
#   configs/quantization_matrix.yaml
#
# Environment knobs:
#   CUDA_GPU=0|1              single-GPU index (ignored when MULTI_GPU=1)
#   MULTI_GPU=0|1             expose GPU 0+1 and pass device=auto
#   NUM_SAMPLES=512           data.num_samples override (paper default: 512)
#   MODELS="id1,id2"          optional subset by matrix model id
#   DATASETS="mmlu,arc"       optional subset by dataset id
#   MATRIX=configs/..yaml     matrix file path override
#   DRY_RUN=1                 print commands without executing; writes no
#                             log file
#
# Needs bash >= 4 (mapfile). A variant that raises an exception during
# loading, extraction or the metric computation is skipped by
# run_quantization.py and the grid continues; the summary at the end lists
# the skipped variants.
# ============================================================
set -uo pipefail

MULTI_GPU="${MULTI_GPU:-0}"
if [[ "$MULTI_GPU" == "1" ]]; then
    GPUIDS="0,1"
    DEVICE_OVERRIDE="device=auto"
else
    GPUIDS="${CUDA_GPU:-0}"
    DEVICE_OVERRIDE=""
fi

N="${NUM_SAMPLES:-512}"
CFG="${CFG:-configs/quantization.yaml}"
MATRIX="${MATRIX:-configs/quantization_matrix.yaml}"
MODELS="${MODELS:-}"
DATASETS="${DATASETS:-}"
DRY_RUN="${DRY_RUN:-0}"

if (( BASH_VERSINFO[0] < 4 )); then
    echo "Error: run_quantization.sh needs bash >= 4 (found $BASH_VERSION)" >&2
    exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
    LOG=""
else
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    LOG="screen.quantization.${TIMESTAMP}.log"
    ln -sf "$LOG" screen.quantization.latest.log
fi

FAILURES=()
PY_MATRIX=(python scripts/quantization_matrix.py --matrix "$MATRIX")

log() {
    if [[ -n "$LOG" ]]; then
        echo "$*" | tee -a "$LOG"
    else
        echo "$*"
    fi
}

run() {
    local args=("$@")
    local cmd=(python run_quantization.py --config "$CFG")
    [[ -n "$DEVICE_OVERRIDE" ]] && cmd+=("$DEVICE_OVERRIDE")
    cmd+=("${args[@]}")

    log ">>> ${cmd[*]}"
    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi

    CUDA_VISIBLE_DEVICES="$GPUIDS" "${cmd[@]}" 2>&1 | tee -a "$LOG"
}

run_all_datasets() {
    local model_tag="$1"
    shift
    local args=("$@")
    local ds
    for ds in "${DATASETS_ALL[@]}"; do
        if ! run "${args[@]}" "data.task=$ds" "data.num_samples=$N"; then
            log "!!! FAILED: ${model_tag} / $ds"
            FAILURES+=("${model_tag} | $ds")
        fi
    done
}

DATASETS_RAW="$("${PY_MATRIX[@]}" datasets --datasets "$DATASETS")" || exit 1
mapfile -t DATASETS_ALL <<<"$DATASETS_RAW"
if [[ ${#DATASETS_ALL[@]} -eq 0 ]]; then
    echo "Error: no datasets resolved from matrix" >&2
    exit 1
fi

MODELS_RAW="$("${PY_MATRIX[@]}" models --models "$MODELS")" || exit 1
mapfile -t MODEL_ROWS <<<"$MODELS_RAW"
if [[ ${#MODEL_ROWS[@]} -eq 0 ]]; then
    echo "Error: no models resolved from matrix" >&2
    exit 1
fi

log "Resolved ${#MODEL_ROWS[@]} model(s) × ${#DATASETS_ALL[@]} dataset(s)"
if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN=1 (commands are printed only)"
fi

for row in "${MODEL_ROWS[@]}"; do
    IFS=$'\t' read -r model_id target_override proxy_override template_override bits_override <<<"$row"
    # The template override is always passed: an empty value makes the runner
    # derive the GGUF file name from the proxy repository name.
    model_args=("$target_override" "$proxy_override" "$template_override" "$bits_override")
    run_all_datasets "$target_override" "${model_args[@]}"
done

log "========================================"
if [[ "$DRY_RUN" == "1" ]]; then
    log "  DRY_RUN=1: nothing was executed."
    log "========================================"
    exit 0
fi

# Variants skipped inside a run: run_quantization.py prints one
# "[skipped] <target> | <dataset> | <variant>: <reason>" line for each.
SKIPPED=()
while IFS= read -r line; do
    [[ -n "$line" ]] && SKIPPED+=("$line")
done < <(grep '^\[skipped\] ' "$LOG" || true)

if [[ ${#FAILURES[@]} -eq 0 ]]; then
    log "  All (model, dataset) runs finished."
else
    log "  ${#FAILURES[@]} (model, dataset) run(s) failed:"
    for f in "${FAILURES[@]}"; do
        log "    FAILED: $f"
    done
fi
log "  Skipped variants (no result written): ${#SKIPPED[@]}"
for s in ${SKIPPED[@]+"${SKIPPED[@]}"}; do
    log "    ${s#\[skipped\] }"
done
log "  Results in: ./results/quantization/"
log "  Log:        $LOG"
log "========================================"

[[ ${#FAILURES[@]} -eq 0 ]]
