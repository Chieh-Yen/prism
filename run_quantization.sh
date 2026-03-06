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
#   GPTQ  → gptq:REPO              pre-quantised GPTQ (see per-model notes)
#
# Target models (all loaded in BF16):
#   1. meta-llama/Meta-Llama-3.1-8B          Base
#   2. Qwen/Qwen3-8B-Base                    Base
#   3. mistralai/Ministral-3-8B-Base-2512    Base
#   4. meta-llama/Meta-Llama-3.1-8B-Instruct Instruct
#   5. Qwen/Qwen3-8B                         Instruct / Thinking
#   6. mistralai/Ministral-3-8B-Instruct-2512 Instruct
#
# GPTQ loading notes:
#   • All gptq: entries use AutoModelForCausalLM.from_pretrained (standard HF).
#   • Efficient-ML repos use .pth format — require the GPTQ-for-Qwen3
#     custom inference script and will FAIL with standard transformers.
#     They are kept in the list; PRISM will skip them with an error message.
#   • Ministral-3: no confirmed public GPTQ repos — omitted.
#
# GGUF repo notes:
#   • Ministral-3 bartowski repos are inferred from naming convention.
#     Verify on HuggingFace before running.
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
# GPTQ  : ModelCloud/Meta-Llama-3.1-8B-gptq-4bit  INT4-g128  GPTQModel 0.9.9
#          shuyuej/Meta-Llama-3.1-8B-GPTQ          INT4       ExLlama v1 format
#          (TechxGenus has Llama-3 only, not 3.1 — omitted)
#          (No confirmed INT8-group GPTQ for the base model at time of writing)
# ============================================================
LLAMA31B_TARGET="target.model=meta-llama/Meta-Llama-3.1-8B"
LLAMA31B_GGUF="proxy.model=bartowski/Meta-Llama-3.1-8B-GGUF"
LLAMA31B_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:ModelCloud/Meta-Llama-3.1-8B-gptq-4bit,\
gptq:shuyuej/Meta-Llama-3.1-8B-GPTQ]"

for DS in $DATASETS_ALL; do
    run $LLAMA31B_TARGET $LLAMA31B_GGUF "$LLAMA31B_BITS" \
        data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 2: Qwen3-8B-Base
# Arch  : Qwen3ForCausalLM, vocab=151K, hidden=4096
# GGUF  : Qwen/Qwen3-8B-Base-GGUF
#          files: Qwen3-8B-Base-{quant}.gguf
#          (Q3/Q2 availability not guaranteed)
# GPTQ  : Efficient-ML repos below use .pth format — will be skipped
#          by PRISM unless the GPTQ-for-Qwen3 custom script is active.
#          Efficient-ML/Qwen3-8B-base-gptq-w4-128      INT4-g128
#          Efficient-ML/Qwen3-8B-base-gptq-w8-128      INT8-g128
#          Efficient-ML/Qwen3-8B-base-gptq-w4-perchannel  INT4-perchan
#          Efficient-ML/Qwen3-8B-base-gptq-w8-perchannel  INT8-perchan
#          AlphaGaO/Qwen3-8B-GPTQ                      INT4-g128 Marlin
#            ↑ post-quant fine-tuned on distillation data — scores may
#              not reflect clean quantization error; verify before use.
# ============================================================
QWEN3B_TARGET="target.model=Qwen/Qwen3-8B-Base"
QWEN3B_GGUF="proxy.model=Qwen/Qwen3-8B-Base-GGUF"
QWEN3B_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:Efficient-ML/Qwen3-8B-base-gptq-w4-128,\
gptq:Efficient-ML/Qwen3-8B-base-gptq-w8-128,\
gptq:Efficient-ML/Qwen3-8B-base-gptq-w4-perchannel,\
gptq:Efficient-ML/Qwen3-8B-base-gptq-w8-perchannel,\
gptq:AlphaGaO/Qwen3-8B-GPTQ]"

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
# GPTQ  : no confirmed public repo — omitted
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
# GPTQ  : hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4  INT4-g128 AutoGPTQ
#          ModelCloud/Meta-Llama-3.1-8B-Instruct-gptq-4bit      INT4-g128 GPTQModel
#          shuyuej/Meta-Llama-3.1-8B-Instruct-GPTQ              INT4-g128 ExLlama v1
#          (No confirmed INT8-group standard GPTQ for Llama-3.1-8B-Instruct)
# ============================================================
LLAMA31I_TARGET="target.model=meta-llama/Meta-Llama-3.1-8B-Instruct"
LLAMA31I_GGUF="proxy.model=bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
LLAMA31I_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4,\
gptq:ModelCloud/Meta-Llama-3.1-8B-Instruct-gptq-4bit,\
gptq:shuyuej/Meta-Llama-3.1-8B-Instruct-GPTQ]"

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
# GPTQ  : Efficient-ML repos below use .pth format — will be skipped
#          by PRISM unless the GPTQ-for-Qwen3 custom script is active.
#          Efficient-ML/Qwen3-8B-gptq-w4-128      INT4-g128
#          Efficient-ML/Qwen3-8B-gptq-w8-128      INT8-g128
#          Efficient-ML/Qwen3-8B-gptq-w4-perchannel  INT4-perchan
#          Efficient-ML/Qwen3-8B-gptq-w8-perchannel  INT8-perchan
#          Standard-format repos (transformers-compatible):
#          JunHowie/Qwen3-8B-GPTQ-Int8             INT8-g128  AutoGPTQ
#          RedHatAI/Qwen3-8B-quantized.w4a16        INT4-g64   llm-compressor
# ============================================================
QWEN3I_TARGET="target.model=Qwen/Qwen3-8B"
QWEN3I_GGUF="proxy.model=Qwen/Qwen3-8B-GGUF"
QWEN3I_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:Efficient-ML/Qwen3-8B-gptq-w4-128,\
gptq:Efficient-ML/Qwen3-8B-gptq-w8-128,\
gptq:Efficient-ML/Qwen3-8B-gptq-w4-perchannel,\
gptq:Efficient-ML/Qwen3-8B-gptq-w8-perchannel,\
gptq:JunHowie/Qwen3-8B-GPTQ-Int8,\
gptq:RedHatAI/Qwen3-8B-quantized.w4a16]"

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
# GPTQ  : no confirmed public repo — omitted
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
