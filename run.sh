#!/usr/bin/env bash
# ============================================================
# PRISM — Full Quantization Experiment Suite
# 6 datasets × 3 models, GGUF + BnB (NF4/FP4/INT8) + GPTQ
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
# Model 1: Llama-2-7b  (GGUF + BnB + GPTQ)
#   GPTQ main = 4bit-128g;  branch = 4bit-32g
# ============================================================
LLAMA_TARGET="target.model=NousResearch/Llama-2-7b-hf"
LLAMA_GGUF="proxy.model=TheBloke/Llama-2-7b-GGUF"
LLAMA_BITS="proxy.quantization_bits=[Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,bnb:int8,bnb:nf4,bnb:fp4,gptq:TheBloke/Llama-2-7B-GPTQ,gptq:TheBloke/Llama-2-7B-GPTQ@gptq-4bit-32g-actorder_True]"

for DS in $DATASETS_ALL; do
    run $LLAMA_TARGET $LLAMA_GGUF "$LLAMA_BITS" data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 2: Mistral-7B-v0.1  (GGUF + BnB + GPTQ)
#   GPTQ main = 4bit-128g;  branch = 4bit-32g
#   (8bit branches use Triton kernels that may crash — omitted)
# ============================================================
MISTRAL_TARGET="target.model=mistralai/Mistral-7B-v0.1"
MISTRAL_GGUF="proxy.model=TheBloke/Mistral-7B-v0.1-GGUF"
MISTRAL_BITS="proxy.quantization_bits=[Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,bnb:int8,bnb:nf4,bnb:fp4,gptq:TheBloke/Mistral-7B-v0.1-GPTQ,gptq:TheBloke/Mistral-7B-v0.1-GPTQ@gptq-4bit-32g-actorder_True]"

for DS in $DATASETS_ALL; do
    run $MISTRAL_TARGET $MISTRAL_GGUF "$MISTRAL_BITS" data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 3: Qwen3-8B  (GGUF + BnB + GPTQ, max_length=512)
# Qwen3 has 151K vocab → needs shorter sequences on 20GB GPU
# No Q3/Q2 GGUF available for Qwen3-8B
#   GPTQ: AlphaGaO 4bit-128g + JunHowie Int8
# ============================================================
QWEN_TARGET="target.model=Qwen/Qwen3-8B"
QWEN_GGUF="proxy.model=Qwen/Qwen3-8B-GGUF"
QWEN_BITS="proxy.quantization_bits=[Q8_0,Q6_K,Q5_K_M,Q4_K_M,bnb:int8,bnb:nf4,bnb:fp4,gptq:AlphaGaO/Qwen3-8B-GPTQ,gptq:JunHowie/Qwen3-8B-GPTQ-Int8]"
QWEN_MAXLEN="data.max_length=512"

for DS in $DATASETS_ALL; do
    run $QWEN_TARGET $QWEN_GGUF "$QWEN_BITS" $QWEN_MAXLEN data.task=$DS data.num_samples=$N
done

echo "========================================"
echo "  All experiments complete."
echo "  Results in: ./results/quantization/"
echo "  Log:        $LOG"
echo "========================================"
