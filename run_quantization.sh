#!/usr/bin/env bash
# ============================================================
# PRISM — Full Quantization Experiment Suite
# 6 datasets × 15 models
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
#   1.  meta-llama/Meta-Llama-3.1-8B              Base
#   2.  Qwen/Qwen3-8B-Base                        Base
#   3.  mistralai/Ministral-3-8B-Base-2512        Base
#   4.  meta-llama/Meta-Llama-3.1-8B-Instruct     Instruct
#   5.  Qwen/Qwen3-8B                             Instruct / Thinking
#   6.  mistralai/Ministral-3-8B-Instruct-2512    Instruct
#   7.  mistralai/Mistral-7B-v0.3                 Base
#   8.  mistralai/Mistral-7B-Instruct-v0.3        Instruct
#   9.  deepseek-ai/DeepSeek-R1-Distill-Llama-8B  Distilled / Reasoning
#   10. Qwen/Qwen2.5-7B                           Base
#   11. Qwen/Qwen2.5-7B-Instruct                  Instruct
#   12. google/gemma-3-4b                         Base
#   13. google/gemma-3-4b-it                      Instruct
#   14. google/gemma-2-9b                         Base
#   15. google/gemma-2-9b-it                      Instruct
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

# ── GPU selection ─────────────────────────────────────────────
# MULTI_GPU=0 (default) → single GPU selected by CUDA_GPU (default 0)
# MULTI_GPU=1           → expose both GPU 0 and GPU 1; device=auto
#                         lets HuggingFace distribute each model across them
#
# Examples:
#   bash run_quantization.sh                     # GPU 0
#   CUDA_GPU=1 bash run_quantization.sh          # GPU 1
#   MULTI_GPU=1 bash run_quantization.sh         # GPU 0+1
MULTI_GPU="${MULTI_GPU:-0}"
if [[ "$MULTI_GPU" == "1" ]]; then
    GPUIDS="0,1"
    DEVICE_OVERRIDE="device=auto"
else
    GPUIDS="${CUDA_GPU:-0}"
    DEVICE_OVERRIDE=""
fi

N=512
CFG="configs/quantization.yaml"
LOG="screen.log"

run() {
    echo ">>> $*" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$GPUIDS" python run.py --config "$CFG" ${DEVICE_OVERRIDE:+"$DEVICE_OVERRIDE"} "$@" 2>&1 | tee -a "$LOG"
}

DATASETS_ALL="lambada c4 wikitext gsm8k mmlu arc"

# ============================================================
# Model 1: Meta-Llama-3.1-8B  (Base)
# Arch  : LlamaForCausalLM, vocab=128K, hidden=4096
# GGUF  : QuantFactory/Meta-Llama-3.1-8B-GGUF  (community, public)
#          files: Meta-Llama-3.1-8B.{quant}.gguf  (dot convention — gguf_template required)
#          NOTE: bartowski/Meta-Llama-3.1-8B-GGUF is gated (inherits Meta license)
# GPTQ  : ModelCloud/Meta-Llama-3.1-8B-gptq-4bit  INT4-g128  GPTQModel 0.9.9
#          shuyuej/Meta-Llama-3.1-8B-GPTQ          INT4       ExLlama v1 format
#          (TechxGenus has Llama-3 only, not 3.1 — omitted)
#          (No confirmed INT8-group GPTQ for the base model at time of writing)
# ============================================================
LLAMA31B_TARGET="target.model=meta-llama/Meta-Llama-3.1-8B"
LLAMA31B_GGUF="proxy.model=QuantFactory/Meta-Llama-3.1-8B-GGUF"
LLAMA31B_TPL="proxy.gguf_template=Meta-Llama-3.1-8B.{quant}.gguf"
LLAMA31B_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:ModelCloud/Meta-Llama-3.1-8B-gptq-4bit,\
gptq:shuyuej/Meta-Llama-3.1-8B-GPTQ]"


#for DS in $DATASETS_ALL; do
#    run $LLAMA31B_TARGET $LLAMA31B_GGUF $LLAMA31B_TPL "$LLAMA31B_BITS" \
#        data.task=$DS data.num_samples=$N
#done

DATASETS_ALL="lambada c4 wikitext gsm8k mmlu arc"

# ============================================================
# Model 2: Qwen3-8B-Base
# Arch  : Qwen3ForCausalLM, vocab=151K, hidden=4096
# GGUF  : mradermacher/Qwen3-8B-Base-GGUF  (community; Qwen org has no Base GGUF)
#          files: Qwen3-8B-Base.{quant}.gguf  (dot convention)
#          explicit gguf_template required — auto-derive uses dash separator
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
QWEN3B_GGUF="proxy.model=mradermacher/Qwen3-8B-Base-GGUF"
QWEN3B_TPL="proxy.gguf_template=Qwen3-8B-Base.{quant}.gguf"
QWEN3B_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:Efficient-ML/Qwen3-8B-base-gptq-w4-128,\
gptq:Efficient-ML/Qwen3-8B-base-gptq-w8-128,\
gptq:Efficient-ML/Qwen3-8B-base-gptq-w4-perchannel,\
gptq:Efficient-ML/Qwen3-8B-base-gptq-w8-perchannel,\
gptq:AlphaGaO/Qwen3-8B-GPTQ]"


#for DS in $DATASETS_ALL; do
#    run $QWEN3B_TARGET $QWEN3B_GGUF $QWEN3B_TPL "$QWEN3B_BITS" \
#        data.task=$DS data.num_samples=$N
#done

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


#for DS in $DATASETS_ALL; do
#    run $LLAMA31I_TARGET $LLAMA31I_GGUF "$LLAMA31I_BITS" \
#        data.task=$DS data.num_samples=$N
#done

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

DATASETS_ALL="wikitext gsm8k mmlu arc"

for DS in $DATASETS_ALL; do
    run $QWEN3I_TARGET $QWEN3I_GGUF "$QWEN3I_BITS" \
        data.task=$DS data.num_samples=$N
done

DATASETS_ALL="lambada c4 wikitext gsm8k mmlu arc"

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

# ============================================================
# Model 7: Mistral-7B-v0.3  (Base)
# Arch  : MistralForCausalLM, vocab=32768, hidden=4096
# GGUF  : bartowski/Mistral-7B-v0.3-GGUF
#          files: Mistral-7B-v0.3-{quant}.gguf
# GPTQ  : iproskurina/Mistral-7B-v0.3-GPTQ-4bit-g128  INT4-g128  AutoGPTQ
# ============================================================
MIS7B_TARGET="target.model=mistralai/Mistral-7B-v0.3"
MIS7B_GGUF="proxy.model=bartowski/Mistral-7B-v0.3-GGUF"
MIS7B_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:iproskurina/Mistral-7B-v0.3-GPTQ-4bit-g128]"

for DS in $DATASETS_ALL; do
    run $MIS7B_TARGET $MIS7B_GGUF "$MIS7B_BITS" \
        data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 8: Mistral-7B-Instruct-v0.3  (Instruct)
# Arch  : MistralForCausalLM, vocab=32768, hidden=4096
# GGUF  : bartowski/Mistral-7B-Instruct-v0.3-GGUF
#          files: Mistral-7B-Instruct-v0.3-{quant}.gguf
# GPTQ  : thesven/Mistral-7B-Instruct-v0.3-GPTQ  INT4  community AutoGPTQ
# ============================================================
MIS7I_TARGET="target.model=mistralai/Mistral-7B-Instruct-v0.3"
MIS7I_GGUF="proxy.model=bartowski/Mistral-7B-Instruct-v0.3-GGUF"
MIS7I_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:thesven/Mistral-7B-Instruct-v0.3-GPTQ]"

for DS in $DATASETS_ALL; do
    run $MIS7I_TARGET $MIS7I_GGUF "$MIS7I_BITS" \
        data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 9: DeepSeek-R1-Distill-Llama-8B  (Distilled / Reasoning)
# Arch  : LlamaForCausalLM, vocab=128K, hidden=4096
# GGUF  : bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF
#          files: DeepSeek-R1-Distill-Llama-8B-{quant}.gguf
# GPTQ  : jakiAJK/DeepSeek-R1-Distill-Llama-8B_GPTQ-int4  INT4-g128  community
#          (no tier-1 provider GPTQ confirmed at time of writing)
# ============================================================
DSR1_TARGET="target.model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DSR1_GGUF="proxy.model=bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF"
DSR1_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:jakiAJK/DeepSeek-R1-Distill-Llama-8B_GPTQ-int4]"


for DS in $DATASETS_ALL; do
    run $DSR1_TARGET $DSR1_GGUF "$DSR1_BITS" \
        data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 10: Qwen2.5-7B  (Base)
# Arch  : Qwen2ForCausalLM, vocab=151K, hidden=3584
# GGUF  : QuantFactory/Qwen2.5-7B-GGUF  (community)
#          files: Qwen2.5-7B.{quant}.gguf  (QuantFactory dot convention)
#          explicit gguf_template required — auto-derive uses dash separator
# GPTQ  : no confirmed public repo for the base model — omitted
# ============================================================
QWEN25B_TARGET="target.model=Qwen/Qwen2.5-7B"
QWEN25B_GGUF="proxy.model=QuantFactory/Qwen2.5-7B-GGUF"
QWEN25B_TPL="proxy.gguf_template=Qwen2.5-7B.{quant}.gguf"
QWEN25B_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4]"

for DS in $DATASETS_ALL; do
    run $QWEN25B_TARGET $QWEN25B_GGUF $QWEN25B_TPL "$QWEN25B_BITS" \
        data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 11: Qwen2.5-7B-Instruct  (Instruct)
# Arch  : Qwen2ForCausalLM, vocab=151K, hidden=3584
# GGUF  : Qwen/Qwen2.5-7B-Instruct-GGUF  (official)
#          files: Qwen2.5-7B-Instruct-{quant}.gguf
# GPTQ  : Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4  INT4-g128  official AutoGPTQ
#          Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8  INT8-g128  official AutoGPTQ
# ============================================================
QWEN25I_TARGET="target.model=Qwen/Qwen2.5-7B-Instruct"
QWEN25I_GGUF="proxy.model=Qwen/Qwen2.5-7B-Instruct-GGUF"
QWEN25I_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4,\
gptq:Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8]"


for DS in $DATASETS_ALL; do
    run $QWEN25I_TARGET $QWEN25I_GGUF "$QWEN25I_BITS" \
        data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 12: Gemma-3-4B  (Base)
# Arch  : Gemma3ForCausalLM, vocab=262K, hidden=2560
# GGUF  : no confirmed public GGUF for the base model — BnB only
# GPTQ  : no confirmed public repo — omitted
# NOTE  : gated model — requires HuggingFace license acceptance at
#          https://huggingface.co/google/gemma-3-4b before running
# ============================================================
GEMMA3B_TARGET="target.model=google/gemma-3-4b"
GEMMA3B_GGUF="proxy.model=google/gemma-3-4b"
GEMMA3B_BITS="proxy.quantization_bits=[\
dtype:float16,\
bnb:int8,bnb:nf4,bnb:fp4]"

for DS in $DATASETS_ALL; do
    run $GEMMA3B_TARGET $GEMMA3B_GGUF "$GEMMA3B_BITS" \
        data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 13: Gemma-3-4B-IT  (Instruct)
# Arch  : Gemma3ForCausalLM, vocab=262K, hidden=2560
# GGUF  : bartowski/google_gemma-3-4b-it-GGUF
#          files: google_gemma-3-4b-it-{quant}.gguf
#          (bartowski prefixes Google models with "google_")
# GPTQ  : circulus/gemma-3-4b-it-gptq  INT4-g128  gptqmodel 2.1.0
# ============================================================
GEMMA3I_TARGET="target.model=google/gemma-3-4b-it"
GEMMA3I_GGUF="proxy.model=bartowski/google_gemma-3-4b-it-GGUF"
GEMMA3I_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:circulus/gemma-3-4b-it-gptq]"


for DS in $DATASETS_ALL; do
    run $GEMMA3I_TARGET $GEMMA3I_GGUF "$GEMMA3I_BITS" \
        data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 14: Gemma-2-9B  (Base)
# Arch  : Gemma2ForCausalLM, vocab=256K, hidden=3584
# GGUF  : QuantFactory/gemma-2-9b-GGUF  (community)
#          files: gemma-2-9b.{quant}.gguf  (QuantFactory dot convention)
#          explicit gguf_template required — auto-derive uses dash separator
# GPTQ  : ModelCloud/gemma-2-9b-gptq-4bit  INT4-g128  gptqmodel 0.9.2
# ============================================================
GEMMA2B_TARGET="target.model=google/gemma-2-9b"
GEMMA2B_GGUF="proxy.model=QuantFactory/gemma-2-9b-GGUF"
GEMMA2B_TPL="proxy.gguf_template=gemma-2-9b.{quant}.gguf"
GEMMA2B_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:ModelCloud/gemma-2-9b-gptq-4bit]"


for DS in $DATASETS_ALL; do
    run $GEMMA2B_TARGET $GEMMA2B_GGUF $GEMMA2B_TPL "$GEMMA2B_BITS" \
        data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 15: Gemma-2-9B-IT  (Instruct)
# Arch  : Gemma2ForCausalLM, vocab=256K, hidden=3584
# GGUF  : bartowski/gemma-2-9b-it-GGUF
#          files: gemma-2-9b-it-{quant}.gguf
# GPTQ  : ModelCloud/gemma-2-9b-it-gptq-4bit  INT4-g128  gptqmodel 0.9.2
#          marcsun13/gemma-2-9b-it-GPTQ         INT4-g128  standard transformers
# ============================================================
GEMMA2I_TARGET="target.model=google/gemma-2-9b-it"
GEMMA2I_GGUF="proxy.model=bartowski/gemma-2-9b-it-GGUF"
GEMMA2I_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:ModelCloud/gemma-2-9b-it-gptq-4bit,\
gptq:marcsun13/gemma-2-9b-it-GPTQ]"


for DS in $DATASETS_ALL; do
    run $GEMMA2I_TARGET $GEMMA2I_GGUF "$GEMMA2I_BITS" \
        data.task=$DS data.num_samples=$N
done

echo "========================================"
echo "  All experiments complete."
echo "  Results in: ./results/quantization/"
echo "  Log:        $LOG"
echo "========================================"
