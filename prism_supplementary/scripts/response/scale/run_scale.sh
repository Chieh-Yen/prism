#!/usr/bin/env bash
# Scale-axis intervention: the reported configurations, then the tables.
#
#   two targets (Qwen3-8B-Base, Llama-3.1-8B) x five benchmarks x two settings
#   of llm_int8_threshold (6.0 = outlier channels kept in FP16, 1e9 = every
#   channel quantized), n = 512 test sequences per benchmark.
#
# Knobs: OUT_DIR (default out/response/scale), PYTHON (default python),
#        CUDA_VISIBLE_DEVICES (the scripts use cuda:0 of the visible devices).
set -euo pipefail
cd "$(dirname "$0")/../../.."        # supplementary root

OUT_DIR="${OUT_DIR:-out/response/scale}"
PYTHON="${PYTHON:-python}"
HERE=scripts/response/scale
mkdir -p "$OUT_DIR"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TARGETS=("Qwen/Qwen3-8B-Base" "meta-llama/Llama-3.1-8B")
TAGS=("qwen3_8b_base" "llama_3_1_8b")

RESULTS=()
CONSTANTS=()
for i in 0 1; do
    result="$OUT_DIR/scale_toggle_${TAGS[$i]}.json"
    constants="$OUT_DIR/head_constants_${TAGS[$i]}.json"
    "$PYTHON" "$HERE/scale_toggle.py" --target "${TARGETS[$i]}" \
        --thresholds 6.0,1e9 --tasks squad,triviaqa,mmlu,arc,gsm8k --n 512 \
        --out "$result" 2>&1 | tee "$OUT_DIR/scale_toggle_${TAGS[$i]}.log"
    "$PYTHON" "$HERE/head_constants.py" --target "${TARGETS[$i]}" --out "$constants"
    RESULTS+=("$result")
    CONSTANTS+=("$constants")
done

"$PYTHON" "$HERE/analyze.py" --results "${RESULTS[@]}" --constants "${CONSTANTS[@]}" \
    | tee "$OUT_DIR/scale_tables.txt"
