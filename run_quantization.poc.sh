#!/usr/bin/env bash
# ============================================================
# PRISM — Full Quantization Experiment Suite
# 6 datasets × 5 models, GGUF + BnB (NF4/FP4/INT8) + GPTQ
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
#   GPTQ: taharmasmaliyev07 8bit-128g (gptqmodel 5.6.12)
# ============================================================
LLAMA_TARGET="target.model=NousResearch/Llama-2-7b-hf"
LLAMA_GGUF="proxy.model=TheBloke/Llama-2-7b-GGUF"
LLAMA_BITS="proxy.quantization_bits=[Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,bnb:int8,bnb:nf4,bnb:fp4,gptq:TheBloke/Llama-2-7B-GPTQ,gptq:TheBloke/Llama-2-7B-GPTQ@gptq-4bit-32g-actorder_True,gptq:TheBloke/Llama-2-7B-GPTQ@gptq-4bit-64g-actorder_True,gptq:TheBloke/Llama-2-7B-GPTQ@gptq-4bit-128g-actorder_True,gptq:taharmasmaliyev07/Llama-2-7b-hf-gptq-int8]"

for DS in $DATASETS_ALL; do
    run $LLAMA_TARGET $LLAMA_GGUF "$LLAMA_BITS" data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 2: Mistral-7B-v0.1  (GGUF + BnB + GPTQ)
#   GPTQ main = 4bit-128g;  branch = 4bit-32g
#   GPTQ: taharmasmaliyev07 8bit-128g (gptqmodel 5.6.12)
# ============================================================
MISTRAL_TARGET="target.model=mistralai/Mistral-7B-v0.1"
MISTRAL_GGUF="proxy.model=TheBloke/Mistral-7B-v0.1-GGUF"
MISTRAL_BITS="proxy.quantization_bits=[Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,bnb:int8,bnb:nf4,bnb:fp4,gptq:TheBloke/Mistral-7B-v0.1-GPTQ,gptq:TheBloke/Mistral-7B-v0.1-GPTQ@gptq-4bit-32g-actorder_True,gptq:TheBloke/Mistral-7B-v0.1-GPTQ@gptq-8bit-32g-actorder_True,gptq:TheBloke/Mistral-7B-v0.1-GPTQ@gptq-8bit-128g-actorder_True]"

for DS in $DATASETS_ALL; do
    run $MISTRAL_TARGET $MISTRAL_GGUF "$MISTRAL_BITS" data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 3: Qwen3-8B-Base  (GGUF + BnB, max_length=512)
# Target switched to Base model for pre-training-era analysis.
# Qwen3 has 151K vocab → shorter sequences to fit 20GB GPU.
#
# GGUF : Qwen/Qwen3-8B-Base-GGUF
#         Verify this repo exists on HuggingFace before running.
#         Files expected: Qwen3-8B-Base-{quant}.gguf
#         No Q3/Q2 GGUF typically available for Qwen3-8B.
#
# GPTQ : No Base-specific GPTQ repo confirmed at time of writing.
#         AlphaGaO/Qwen3-8B-GPTQ and JunHowie/Qwen3-8B-GPTQ-*
#         target the Instruct model — do NOT use for Base.
#         Uncomment and update QWEN_BITS when a Base GPTQ repo is found.
# ============================================================
QWEN_TARGET="target.model=Qwen/Qwen3-8B-Base"
QWEN_GGUF="proxy.model=Qwen/Qwen3-8B-Base-GGUF"
QWEN_BITS="proxy.quantization_bits=[Q8_0,Q6_K,Q5_K_M,Q4_K_M,bnb:int8,bnb:nf4,bnb:fp4,gptq:Efficient-ML/Qwen3-8B-base-gptq-w4-128,gptq:Efficient-ML/Qwen3-8B-base-gptq-w8-128]"
# QWEN_BITS="proxy.quantization_bits=[Q8_0,Q6_K,Q5_K_M,Q4_K_M,bnb:int8,bnb:nf4,bnb:fp4,gptq:AlphaGaO/Qwen3-8B-GPTQ,gptq:JunHowie/Qwen3-8B-GPTQ-Int8]"
# QWEN_BITS="proxy.quantization_bits=[Q8_0,Q6_K,Q5_K_M,Q4_K_M,bnb:int8,bnb:nf4,bnb:fp4,gptq:REPO/Qwen3-8B-Base-GPTQ]"
QWEN_MAXLEN="data.max_length=512"

for DS in $DATASETS_ALL; do
    run $QWEN_TARGET $QWEN_GGUF "$QWEN_BITS" $QWEN_MAXLEN data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 4: OLMo-3-1025-7B  (BnB only)
# Architecture: Olmo3ForCausalLM, vocab=100K, hidden=4096
# GGUF: not supported (transformers GGUF loader lacks olmo2 arch)
# GPTQ: not available for base model
# ============================================================
OLMO_TARGET="target.model=allenai/Olmo-3-1025-7B"
OLMO_GGUF="proxy.model=allenai/Olmo-3-1025-7B"
OLMO_BITS="proxy.quantization_bits=[bnb:int8,bnb:nf4,bnb:fp4]"

for DS in $DATASETS_ALL; do
    run $OLMO_TARGET $OLMO_GGUF "$OLMO_BITS" data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 5: Llama-3.1-8B  (GGUF + BnB + GPTQ)
# Architecture: LlamaForCausalLM, vocab=128K, hidden=4096
# This is the BASE (pre-trained) model, not Instruct.
#
# GGUF : bartowski/Meta-Llama-3.1-8B-GGUF
#         Files: Meta-Llama-3.1-8B-{quant}.gguf
#         (auto-derived template matches bartowski naming convention)
#
# GPTQ : TechxGenus/Meta-Llama-3.1-8B-GPTQ
#         Verify this repo exists; substitute with any confirmed
#         Base-model GPTQ repo if needed.
#
# max_length: keeping default 2048 — hidden_size=4096 same as Llama-2.
#             Reduce to 1024 if OOM (vocab=128K makes lm_head ~1GB fp16).
# ============================================================
LLAMA31_TARGET="target.model=meta-llama/Llama-3.1-8B"
LLAMA31_GGUF="proxy.model=bartowski/Meta-Llama-3.1-8B-GGUF"
LLAMA31_BITS="proxy.quantization_bits=[Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,bnb:int8,bnb:nf4,bnb:fp4,gptq:TechxGenus/Meta-Llama-3.1-8B-GPTQ]"

for DS in $DATASETS_ALL; do
    run $LLAMA31_TARGET $LLAMA31_GGUF "$LLAMA31_BITS" data.task=$DS data.num_samples=$N
done

echo "========================================"
echo "  All experiments complete."
echo "  Results in: ./results/quantization/"
echo "  Log:        $LOG"
echo "========================================"
