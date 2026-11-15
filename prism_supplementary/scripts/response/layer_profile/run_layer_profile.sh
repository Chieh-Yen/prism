#!/usr/bin/env bash
# Layer-wise / module-wise profile: the reported configurations, then the table.
#
#   Qwen3-8B-Base against its GGUF variants Q6_K, Q4_K_M and Q2_K, and
#   Llama-3.1-8B against its GGUF variant Q6_K; 128 MMLU test sequences.
#
# Knobs: OUT_DIR (default out/response/layer_profile), PYTHON (default python),
#        CUDA_VISIBLE_DEVICES (the script uses cuda:0 of the visible devices).
set -euo pipefail
cd "$(dirname "$0")/../../.."        # supplementary root

OUT_DIR="${OUT_DIR:-out/response/layer_profile}"
PYTHON="${PYTHON:-python}"
HERE=scripts/response/layer_profile
mkdir -p "$OUT_DIR"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PYTHON" "$HERE/layer_profile.py" --target Qwen/Qwen3-8B-Base \
    --gguf_repo mradermacher/Qwen3-8B-Base-GGUF \
    --gguf_tpl 'Qwen3-8B-Base.{quant}.gguf' \
    --quants Q6_K,Q4_K_M,Q2_K --task mmlu --n 128 \
    --out "$OUT_DIR/layer_profile_qwen3_8b_base.json" \
    2>&1 | tee "$OUT_DIR/layer_profile_qwen3_8b_base.log"

"$PYTHON" "$HERE/layer_profile.py" --target meta-llama/Llama-3.1-8B \
    --gguf_repo QuantFactory/Meta-Llama-3.1-8B-GGUF \
    --gguf_tpl 'Meta-Llama-3.1-8B.{quant}.gguf' \
    --quants Q6_K --task mmlu --n 128 \
    --out "$OUT_DIR/layer_profile_llama_3_1_8b.json" \
    2>&1 | tee "$OUT_DIR/layer_profile_llama_3_1_8b.log"

"$PYTHON" "$HERE/make_table.py" \
    --results "$OUT_DIR/layer_profile_qwen3_8b_base.json" \
              "$OUT_DIR/layer_profile_llama_3_1_8b.json" \
    | tee "$OUT_DIR/layer_profile_table.txt"
