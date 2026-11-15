#!/usr/bin/env bash
# ============================================================
# Head experiments 1 (hybrids) and 2 (head precision grid).
#
#   STAGE=swap       head_swap.py
#                      Qwen3-8B-Base   GGUF Q6_K, Q4_K_M, Q2_K   (15 cells)
#                      Llama-3.1-8B    GGUF Q6_K                 ( 5 cells)
#   STAGE=precision  head_precision.py, backbones {original, Q4_K_M, Q2_K}
#                    x 7 heads, both models (210 cells). Each model is run
#                    as two processes (arc,mmlu,squad and triviaqa,gsm8k).
#   STAGE=analyze    analyze_head_swap.py, analyze_head_precision.py (no GPU);
#                    runs both analyses and needs the outputs of both
#                    experiments
#   STAGE=all        the three stages in order (default)
#
# Five benchmarks, n = 512, seed 42 (script defaults).
# Knobs: CUDA_GPU (default 0), OUT (default out/response/head),
#        PYTHON (default python3).
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/../../.."        # supplementary root

STAGE="${STAGE:-all}"
OUT="${OUT:-out/response/head}"
PY="${PYTHON:-python3}"
HERE=scripts/response/head
export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

QWEN=(--target Qwen/Qwen3-8B-Base --gguf_repo mradermacher/Qwen3-8B-Base-GGUF
      --gguf_tpl 'Qwen3-8B-Base.{quant}.gguf')
LLAMA=(--target meta-llama/Llama-3.1-8B --gguf_repo QuantFactory/Meta-Llama-3.1-8B-GGUF
       --gguf_tpl 'Meta-Llama-3.1-8B.{quant}.gguf')
TASKS=arc,mmlu,squad,triviaqa,gsm8k

if [[ "$STAGE" == "swap" || "$STAGE" == "all" ]]; then
  "$PY" $HERE/head_swap.py "${QWEN[@]}" --quants Q6_K,Q4_K_M,Q2_K \
      --tasks $TASKS --n 512 --metrics_device cuda:0 \
      --out "$OUT/head_swap_qwen3_8b_base.json"
  echo "head_swap Qwen3-8B-Base exit=$?"
  "$PY" $HERE/head_swap.py "${LLAMA[@]}" --quants Q6_K \
      --tasks $TASKS --n 512 --metrics_device cuda:0 \
      --out "$OUT/head_swap_llama_3_1_8b.json"
  echo "head_swap Llama-3.1-8B exit=$?"
fi

if [[ "$STAGE" == "precision" || "$STAGE" == "all" ]]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
  for part in "a arc,mmlu,squad" "b triviaqa,gsm8k"; do
    set -- $part
    "$PY" $HERE/head_precision.py "${QWEN[@]}" --backbones Q4_K_M,Q2_K \
        --tasks "$2" --n 512 --out "$OUT/head_precision_qwen3_8b_base_$1.json"
    echo "head_precision Qwen3-8B-Base ($2) exit=$?"
    "$PY" $HERE/head_precision.py "${LLAMA[@]}" --backbones Q4_K_M,Q2_K \
        --tasks "$2" --n 512 --out "$OUT/head_precision_llama_3_1_8b_$1.json"
    echo "head_precision Llama-3.1-8B ($2) exit=$?"
  done
fi

if [[ "$STAGE" == "analyze" || "$STAGE" == "all" ]]; then
  "$PY" $HERE/analyze_head_swap.py \
      "$OUT/head_swap_qwen3_8b_base.json" "$OUT/head_swap_llama_3_1_8b.json" \
      | tee "$OUT/analyze_head_swap.txt"
  "$PY" $HERE/analyze_head_precision.py \
      "$OUT"/head_precision_qwen3_8b_base_[ab].json "$OUT"/head_precision_llama_3_1_8b_[ab].json \
      | tee "$OUT/analyze_head_precision.txt"
fi
