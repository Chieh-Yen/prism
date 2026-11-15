#!/usr/bin/env bash
# Eight-sequence screen: launcher.
#
#   bash scripts/response/screen_n8/run_screen.sh run       # GPU: the 20 variants that need no GPTQ loader
#   bash scripts/response/screen_n8/run_screen.sh gptq      # GPU: the 3 GPTQ variants (environment with a GPTQ loader)
#   bash scripts/response/screen_n8/run_screen.sh analyze   # no GPU: the tables
#
# The two GPU stages write to the same CSVs.  `run` ends with status 1 if a
# variant could not be loaded; `analyze` refuses an incomplete grid.
#
# knobs: CUDA_GPU (0), FAMILIES ("llama qwen"), OUT (out/response/screen_n8),
#        SIZE_ROUND (unset; output directory of the three-seed round, adds
#        section 5 of the analysis), PYTHON (python3)
#
# `run` and `gptq` are resumable: finished (variant, seed, benchmark) blocks
# are skipped and the target features are cached under $OUT/{family}_ZT/.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # supplementary root

STAGE="${1:-analyze}"
FAMILIES="${FAMILIES:-llama qwen}"
OUT="${OUT:-out/response/screen_n8}"
HERE=scripts/response/screen_n8
PYTHON="${PYTHON:-python3}"

# Variant labels (column Label of scripts/rounds/data/variant_manifest.csv).
LLAMA_PLAIN=("BF16 vs FP16" "BF16 vs Q8_0" "BF16 vs Q6_K" "BF16 vs Q5_K_M" "BF16 vs Q4_K_M"
             "BF16 vs Q3_K_M" "BF16 vs Q2_K" "BF16 vs INT8" "BF16 vs NF4" "BF16 vs FP4")
QWEN_PLAIN=("${LLAMA_PLAIN[@]}")
LLAMA_GPTQ=("BF16 vs GPTQ(Meta-Llama-3.1-8B-gptq-4bit)" "BF16 vs GPTQ(Meta-Llama-3.1-8B-GPTQ)")
QWEN_GPTQ=("BF16 vs GPTQ(Qwen3-8B-GPTQ)")

case "$STAGE" in
  run)
    export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"
    for fam in $FAMILIES; do
      case "$fam" in
        llama) "$PYTHON" "$HERE/run_screen.py" --family llama --out "$OUT" --only "${LLAMA_PLAIN[@]}" ;;
        qwen)  "$PYTHON" "$HERE/run_screen.py" --family qwen  --out "$OUT" --only "${QWEN_PLAIN[@]}" ;;
        *) echo "unknown family $fam" >&2; exit 2 ;;
      esac
    done
    ;;
  gptq)
    # Run in an environment that has a GPTQ loader (see the README).
    export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"
    for fam in $FAMILIES; do
      case "$fam" in
        llama) "$PYTHON" "$HERE/run_screen.py" --family llama --out "$OUT" --only "${LLAMA_GPTQ[@]}" ;;
        qwen)  "$PYTHON" "$HERE/run_screen.py" --family qwen  --out "$OUT" --only "${QWEN_GPTQ[@]}" ;;
        *) echo "unknown family $fam" >&2; exit 2 ;;
      esac
    done
    ;;
  analyze)
    CSVS=()
    for fam in $FAMILIES; do CSVS+=("$OUT/${fam}_screen.csv"); done
    if [[ -n "${SIZE_ROUND:-}" ]]; then
      "$PYTHON" "$HERE/analyze_screen.py" --csv "${CSVS[@]}" --size-round "$SIZE_ROUND"
    else
      "$PYTHON" "$HERE/analyze_screen.py" --csv "${CSVS[@]}"
    fi
    ;;
  *)
    echo "usage: $0 {run|gptq|analyze}" >&2
    exit 2
    ;;
esac
