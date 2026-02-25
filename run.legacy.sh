#!/usr/bin/env bash
# ============================================================
# PRISM — Full Quantization Experiment Suite
# 6 datasets × 3 models, GGUF + bitsandbytes (NF4/FP4/INT8)
# ============================================================
set -euo pipefail

GPU="${CUDA_GPU:-1}"
N=128
CFG="configs/quantization.yaml"
LOG="screen.log"

run() {
    echo ">>> $*" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$GPU" python run.py --config "$CFG" "$@" 2>&1 | tee -a "$LOG"
}

DATASETS_PLAIN="c4 lambada wikitext"
DATASETS_QA="gsm8k mmlu arc"
DATASETS_ALL="c4 lambada wikitext gsm8k mmlu arc"

# ============================================================
# Model 1: Llama-2-7b  (GGUF: TheBloke, 6 quant levels + BnB)
# ============================================================
LLAMA_TARGET="target.model=NousResearch/Llama-2-7b-hf"
LLAMA_GGUF="proxy.model=TheBloke/Llama-2-7b-GGUF"
LLAMA_BITS="proxy.quantization_bits=[Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,bnb:int8,bnb:nf4,bnb:fp4]"

for DS in $DATASETS_ALL; do
    run $LLAMA_TARGET $LLAMA_GGUF "$LLAMA_BITS" data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 2: Mistral-7B-v0.1  (GGUF: TheBloke + BnB)
# ============================================================
MISTRAL_TARGET="target.model=mistralai/Mistral-7B-v0.1"
MISTRAL_GGUF="proxy.model=TheBloke/Mistral-7B-v0.1-GGUF"
MISTRAL_BITS="proxy.quantization_bits=[Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,bnb:int8,bnb:nf4,bnb:fp4]"

for DS in $DATASETS_ALL; do
    run $MISTRAL_TARGET $MISTRAL_GGUF "$MISTRAL_BITS" data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 3: Qwen3-8B  (GGUF: official + BnB, max_length=512)
# Qwen3 has 151K vocab → needs shorter sequences on 20GB GPU
# No Q3/Q2 GGUF available for Qwen3-8B
# ============================================================
QWEN_TARGET="target.model=Qwen/Qwen3-8B"
QWEN_GGUF="proxy.model=Qwen/Qwen3-8B-GGUF"
QWEN_BITS="proxy.quantization_bits=[Q8_0,Q6_K,Q5_K_M,Q4_K_M,bnb:int8,bnb:nf4,bnb:fp4]"
QWEN_MAXLEN="data.max_length=512"

for DS in $DATASETS_ALL; do
    run $QWEN_TARGET $QWEN_GGUF "$QWEN_BITS" $QWEN_MAXLEN data.task=$DS data.num_samples=$N
done

echo "========================================"
echo "  All experiments complete."
echo "  Results in: ./results/quantization/"
echo "  Log:        $LOG"
echo "========================================"
