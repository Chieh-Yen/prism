#!/usr/bin/env bash
# ============================================================
# PRISM — Full Quantization Experiment Suite
# 6 datasets × 6 models
#
# Quantization tiers (all models):
#   FP16  → dtype:float16           precision-drift reference (BF16→FP16)
#   INT8  → bnb:int8  + Q8_0       BitsAndBytes 8-bit / GGUF 8-bit
#   FP6   → Q6_K                   GGUF 6-bit k-quant (FP6 equivalent)
#   5-bit → Q5_K_M                 GGUF 5-bit k-quant
#   NF4   → bnb:nf4  + Q4_K_M     BitsAndBytes NF4 / GGUF 4-bit
#   FP4   → bnb:fp4                BitsAndBytes FP4
#   3-bit → Q3_K_M                 GGUF 3-bit k-quant
#   2-bit → Q2_K                   GGUF 2-bit k-quant
#   GPTQ  → gptq:REPO              pre-quantised GPTQ (INT4 / INT8)
#
# Target models (all loaded in BF16):
#   1. meta-llama/Meta-Llama-3.1-8B          Base
#   2. Qwen/Qwen3-8B-Base                    Base
#   3. mistralai/Ministral-3-8B-Base-2512    Base
#   4. meta-llama/Meta-Llama-3.1-8B-Instruct Instruct
#   5. Qwen/Qwen3-8B                         Instruct / Thinking
#   6. mistralai/Ministral-3-8B-Instruct-2512 Instruct
#
# NOTE: GGUF repos for Ministral-3 are inferred from the bartowski
# naming convention.  Verify on HuggingFace before running.
# ============================================================
set -euo pipefail

GPU="${CUDA_GPU:-0}"
N=512
CFG="configs/quantization.yaml"
LOG="screen.log"

run() {
    echo ">>> $*" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$GPU" python run.py --config "$CFG" "$@" 2>&1 | tee -a "$LOG"
}

DATASETS_ALL="c4 lambada wikitext gsm8k mmlu arc"

# ============================================================
# Model 1: Meta-Llama-3.1-8B  (Base)
# Arch  : LlamaForCausalLM, vocab=128K, hidden=4096
# GGUF  : bartowski/Meta-Llama-3.1-8B-GGUF
#          files: Meta-Llama-3.1-8B-{quant}.gguf
# GPTQ  : TechxGenus/Meta-Llama-3.1-8B-GPTQ  (INT4, w4-128g)
# ============================================================
LLAMA31B_TARGET="target.model=meta-llama/Meta-Llama-3.1-8B"
LLAMA31B_GGUF="proxy.model=bartowski/Meta-Llama-3.1-8B-GGUF"
LLAMA31B_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:TechxGenus/Meta-Llama-3.1-8B-GPTQ]"

for DS in $DATASETS_ALL; do
    run $LLAMA31B_TARGET $LLAMA31B_GGUF "$LLAMA31B_BITS" \
        data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 2: Qwen3-8B-Base
# Arch  : Qwen3ForCausalLM, vocab=151K, hidden=4096
# GGUF  : Qwen/Qwen3-8B-Base-GGUF
#          files: Qwen3-8B-Base-{quant}.gguf
#          (Q3/Q2 availability not guaranteed — experiment will
#          skip failed GGUF loads automatically)
# GPTQ  : Efficient-ML/Qwen3-8B-base-gptq-w4-128  (INT4)
#          Efficient-ML/Qwen3-8B-base-gptq-w8-128  (INT8)
# ============================================================
QWEN3B_TARGET="target.model=Qwen/Qwen3-8B-Base"
QWEN3B_GGUF="proxy.model=Qwen/Qwen3-8B-Base-GGUF"
QWEN3B_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:Efficient-ML/Qwen3-8B-base-gptq-w4-128,\
gptq:Efficient-ML/Qwen3-8B-base-gptq-w8-128]"

for DS in $DATASETS_ALL; do
    run $QWEN3B_TARGET $QWEN3B_GGUF "$QWEN3B_BITS" \
        data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 3: Ministral-3-8B-Base-2512
# Arch  : Mistral3ForConditionalGeneration (text-only backbone),
#          vocab=131K (Tekken v7), hidden=5120
# GGUF  : bartowski/Ministral-3-8B-Base-2512-GGUF
#          files: Ministral-3-8B-Base-2512-{quant}.gguf
#          *** Verify this repo exists before running ***
# GPTQ  : no confirmed repo — omitted
# ============================================================
MIN3B_TARGET="target.model=mistralai/Ministral-3-8B-Base-2512"
MIN3B_GGUF="proxy.model=bartowski/Ministral-3-8B-Base-2512-GGUF"
MIN3B_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4]"

for DS in $DATASETS_ALL; do
    run $MIN3B_TARGET $MIN3B_GGUF "$MIN3B_BITS" \
        data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 4: Meta-Llama-3.1-8B-Instruct
# Arch  : LlamaForCausalLM, vocab=128K, hidden=4096
# GGUF  : bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
#          files: Meta-Llama-3.1-8B-Instruct-{quant}.gguf
# GPTQ  : TechxGenus/Meta-Llama-3.1-8B-Instruct-GPTQ  (INT4, w4-128g)
# ============================================================
LLAMA31I_TARGET="target.model=meta-llama/Meta-Llama-3.1-8B-Instruct"
LLAMA31I_GGUF="proxy.model=bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
LLAMA31I_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:TechxGenus/Meta-Llama-3.1-8B-Instruct-GPTQ]"

for DS in $DATASETS_ALL; do
    run $LLAMA31I_TARGET $LLAMA31I_GGUF "$LLAMA31I_BITS" \
        data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 5: Qwen3-8B  (Instruct / Thinking)
# Arch  : Qwen3ForCausalLM, vocab=151K, hidden=4096
# GGUF  : Qwen/Qwen3-8B-GGUF
#          files: Qwen3-8B-{quant}.gguf
#          (Q3/Q2 availability not guaranteed)
# GPTQ  : Efficient-ML/Qwen3-8B-gptq-w4-128  (INT4)
#          Efficient-ML/Qwen3-8B-gptq-w8-128  (INT8)
# ============================================================
QWEN3I_TARGET="target.model=Qwen/Qwen3-8B"
QWEN3I_GGUF="proxy.model=Qwen/Qwen3-8B-GGUF"
QWEN3I_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:Efficient-ML/Qwen3-8B-gptq-w4-128,\
gptq:Efficient-ML/Qwen3-8B-gptq-w8-128]"

for DS in $DATASETS_ALL; do
    run $QWEN3I_TARGET $QWEN3I_GGUF "$QWEN3I_BITS" \
        data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 6: Ministral-3-8B-Instruct-2512
# Arch  : Mistral3ForConditionalGeneration (text-only backbone),
#          vocab=131K (Tekken v7), hidden=5120
# GGUF  : bartowski/Ministral-3-8B-Instruct-2512-GGUF
#          files: Ministral-3-8B-Instruct-2512-{quant}.gguf
#          *** Verify this repo exists before running ***
# GPTQ  : no confirmed repo — omitted
# ============================================================
MIN3I_TARGET="target.model=mistralai/Ministral-3-8B-Instruct-2512"
MIN3I_GGUF="proxy.model=bartowski/Ministral-3-8B-Instruct-2512-GGUF"
MIN3I_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4]"

for DS in $DATASETS_ALL; do
    run $MIN3I_TARGET $MIN3I_GGUF "$MIN3I_BITS" \
        data.task=$DS data.num_samples=$N
done

echo "========================================"
echo "  All experiments complete."
echo "  Results in: ./results/quantization/"
echo "  Log:        $LOG"
echo "========================================"
