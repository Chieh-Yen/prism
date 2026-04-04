#!/usr/bin/env bash
# ============================================================
# PRISM — Full Quantization Experiment Suite
# 7 datasets × 11 models, multi-z_mode per dataset
#
# Each (model, dataset) pair runs ALL z_modes in a single
# forward pass via data.z_modes=[...]:
#   Corpus (WikiText, FineWeb-Edu):              mean_pool + concat
#   Q&A (MMLU, ARC, GSM8K, TriviaQA, SQuAD):    last_context_token + concat + last_token
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
#   AWQ   → awq:REPO               pre-quantised AWQ  (see per-model notes)
#
# Target models (all loaded in BF16):
#   1.  Qwen/Qwen3-8B-Base                        Base
#   2.  meta-llama/Meta-Llama-3.1-8B              Base
#   3.  mistralai/Ministral-3-8B-Base-2512        Base
#   4.  Qwen/Qwen3-8B                             Instruct / Thinking
#   5.  meta-llama/Meta-Llama-3.1-8B-Instruct     Instruct
#   6.  mistralai/Ministral-3-8B-Instruct-2512    Instruct
#   7.  deepseek-ai/DeepSeek-R1-Distill-Llama-8B  Distilled / Reasoning
#   8.  Qwen/Qwen2.5-7B                           Base
#   9.  Qwen/Qwen2.5-7B-Instruct                  Instruct
#   10. mistralai/Mistral-7B-v0.3                 Base
#   11. mistralai/Mistral-7B-Instruct-v0.3        Instruct
#
# GPTQ/AWQ loading notes:
#   • All gptq:/awq: entries use AutoModelForCausalLM.from_pretrained (standard HF).
#   • Efficient-ML repos use .pth format — require the GPTQ-for-Qwen3
#     custom inference script and will FAIL with standard transformers.
#     They are kept in the list; PRISM will skip them with an error message.
#   • Ministral-3: no confirmed public GPTQ/AWQ repos — omitted.
#
# GGUF repo notes:
#   • Ministral-3 Base   : mradermacher/Ministral-3-8B-Base-2512-GGUF
#                          files: Ministral-3-8B-Base-2512.{quant}.gguf (dot convention)
#   • Ministral-3 Instruct: bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF
#                          files: mistralai_Ministral-3-8B-Instruct-2512-{quant}.gguf
#   NOTE: transformers 5.x GGUF support for mistral3 was patched locally in
#         transformers/integrations/ggml.py and modeling_gguf_pytorch_utils.py.
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
LOG="screen.quantization.log"

run() {
    echo ">>> $*" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$GPUIDS" python run.py --config "$CFG" ${DEVICE_OVERRIDE:+"$DEVICE_OVERRIDE"} "$@" 2>&1 | tee -a "$LOG"
}

# z_modes auto-resolved from TASK_REGISTRY.z_modes_all per dataset:
#   corpus (wikitext, fineweb_edu):                   [mean_pool, concat]
#   Q&A (gsm8k, mmlu, arc, triviaqa, squad):           [last_context_token, concat, last_token]

DATASETS_ALL="wikitext fineweb_edu gsm8k mmlu arc triviaqa squad"

# Helper: run all 7 datasets for a model
# Usage: run_all_datasets MODEL_ARGS...
run_all_datasets() {
    local args=("$@")
    for DS in $DATASETS_ALL; do
        run "${args[@]}" data.task=$DS data.num_samples=$N
    done
}

# ============================================================
# Model 1: Qwen3-8B-Base
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

run_all_datasets $QWEN3B_TARGET $QWEN3B_GGUF $QWEN3B_TPL "$QWEN3B_BITS"

# ============================================================
# Model 2: Meta-Llama-3.1-8B  (Base)
# Arch  : LlamaForCausalLM, vocab=128K, hidden=4096
# GGUF  : QuantFactory/Meta-Llama-3.1-8B-GGUF  (community, public)
#          files: Meta-Llama-3.1-8B.{quant}.gguf  (dot convention — gguf_template required)
#          NOTE: bartowski/Meta-Llama-3.1-8B-GGUF is gated (inherits Meta license)
# GPTQ  : ModelCloud/Meta-Llama-3.1-8B-gptq-4bit  INT4-g128  GPTQModel 0.9.9
#          shuyuej/Meta-Llama-3.1-8B-GPTQ          INT4       ExLlama v1 format
# AWQ   : UCLA-EMC/Meta-Llama-3.1-8B-AWQ-INT4     INT4       community
# ============================================================
LLAMA31B_TARGET="target.model=meta-llama/Meta-Llama-3.1-8B"
LLAMA31B_GGUF="proxy.model=QuantFactory/Meta-Llama-3.1-8B-GGUF"
LLAMA31B_TPL="proxy.gguf_template=Meta-Llama-3.1-8B.{quant}.gguf"
LLAMA31B_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:ModelCloud/Meta-Llama-3.1-8B-gptq-4bit,\
gptq:shuyuej/Meta-Llama-3.1-8B-GPTQ,\
awq:UCLA-EMC/Meta-Llama-3.1-8B-AWQ-INT4]"

run_all_datasets $LLAMA31B_TARGET $LLAMA31B_GGUF $LLAMA31B_TPL "$LLAMA31B_BITS"

# ============================================================
# Model 3: Ministral-3-8B-Base-2512
# Arch  : Mistral3ForConditionalGeneration (VL backbone; text path used),
#          vocab=131K (Tekken v7), hidden=4096
# GGUF  : mradermacher/Ministral-3-8B-Base-2512-GGUF
#          files: Ministral-3-8B-Base-2512.{quant}.gguf  (dot convention)
#          NOTE: transformers mistral3 GGUF support patched locally.
# GPTQ  : no confirmed public repo — omitted
# ============================================================
MIN3B_TARGET="target.model=mistralai/Ministral-3-8B-Base-2512"
MIN3B_GGUF="proxy.model=mradermacher/Ministral-3-8B-Base-2512-GGUF"
MIN3B_TPL="proxy.gguf_template=Ministral-3-8B-Base-2512.{quant}.gguf"
MIN3B_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4]"

run_all_datasets $MIN3B_TARGET $MIN3B_GGUF $MIN3B_TPL "$MIN3B_BITS"

# ============================================================
# Model 4: Qwen3-8B  (Instruct / Thinking)
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
gptq:RedHatAI/Qwen3-8B-quantized.w4a16,\
awq:Qwen/Qwen3-8B-AWQ]"

run_all_datasets $QWEN3I_TARGET $QWEN3I_GGUF "$QWEN3I_BITS"

# ============================================================
# Model 5: Meta-Llama-3.1-8B-Instruct
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
gptq:shuyuej/Meta-Llama-3.1-8B-Instruct-GPTQ,\
awq:hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4]"

run_all_datasets $LLAMA31I_TARGET $LLAMA31I_GGUF "$LLAMA31I_BITS"

# ============================================================
# Model 6: Ministral-3-8B-Instruct-2512
# Arch  : Mistral3ForConditionalGeneration (VL backbone; text path used),
#          vocab=131K (Tekken v7), hidden=4096
# GGUF  : bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF
#          files: mistralai_Ministral-3-8B-Instruct-2512-{quant}.gguf
#          NOTE: transformers mistral3 GGUF support patched locally.
# BnB   : Hub checkpoint is FP8 (FineGrainedFP8Config).  PRISM handles this
#          via a two-step path: load to CPU (transformers auto-dequantises FP8
#          → BF16 on hardware with compute capability < 8.9), then replace
#          Linear layers with BnB-quantised equivalents in-place and dispatch
#          to GPU.  See quantization.py:_bnb_requantize().
# AWQ   : cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit  INT4  community
# GPTQ  : no confirmed public repo — omitted
# ============================================================
MIN3I_TARGET="target.model=mistralai/Ministral-3-8B-Instruct-2512"
MIN3I_GGUF="proxy.model=bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF"
MIN3I_TPL="proxy.gguf_template=mistralai_Ministral-3-8B-Instruct-2512-{quant}.gguf"
MIN3I_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
awq:cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit]"

run_all_datasets $MIN3I_TARGET $MIN3I_GGUF $MIN3I_TPL "$MIN3I_BITS"

# ============================================================
# Model 7: DeepSeek-R1-Distill-Llama-8B  (Distilled / Reasoning)
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
gptq:jakiAJK/DeepSeek-R1-Distill-Llama-8B_GPTQ-int4,\
awq:casperhansen/deepseek-r1-distill-llama-8b-awq]"

run_all_datasets $DSR1_TARGET $DSR1_GGUF "$DSR1_BITS"

# ============================================================
# Model 8: Qwen2.5-7B  (Base)
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

run_all_datasets $QWEN25B_TARGET $QWEN25B_GGUF $QWEN25B_TPL "$QWEN25B_BITS"

# ============================================================
# Model 9: Qwen2.5-7B-Instruct  (Instruct)
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
gptq:Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8,\
awq:Qwen/Qwen2.5-7B-Instruct-AWQ]"

run_all_datasets $QWEN25I_TARGET $QWEN25I_GGUF "$QWEN25I_BITS"

# ============================================================
# Model 10: Mistral-7B-v0.3  (Base)
# Arch  : MistralForCausalLM, vocab=32768, hidden=4096
# GGUF  : mradermacher/Mistral-7B-v0.3-GGUF
#          files: Mistral-7B-v0.3.{quant}.gguf  (dot convention)
#          explicit gguf_template required — auto-derive uses dash separator
# GPTQ  : iproskurina/Mistral-7B-v0.3-GPTQ-4bit-g128  INT4-g128  AutoGPTQ
# ============================================================
MIS7B_TARGET="target.model=mistralai/Mistral-7B-v0.3"
MIS7B_GGUF="proxy.model=mradermacher/Mistral-7B-v0.3-GGUF"
MIS7B_TPL="proxy.gguf_template=Mistral-7B-v0.3.{quant}.gguf"
MIS7B_BITS="proxy.quantization_bits=[\
dtype:float16,\
Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K,\
bnb:int8,bnb:nf4,bnb:fp4,\
gptq:iproskurina/Mistral-7B-v0.3-GPTQ-4bit-g128,\
awq:solidrust/Mistral-7B-v0.3-AWQ]"

run_all_datasets $MIS7B_TARGET $MIS7B_GGUF $MIS7B_TPL "$MIS7B_BITS"

# ============================================================
# Model 11: Mistral-7B-Instruct-v0.3  (Instruct)
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
gptq:thesven/Mistral-7B-Instruct-v0.3-GPTQ,\
awq:solidrust/Mistral-7B-Instruct-v0.3-AWQ]"

run_all_datasets $MIS7I_TARGET $MIS7I_GGUF "$MIS7I_BITS"

echo "========================================"
echo "  All experiments complete."
echo "  Results in: ./results/quantization/"
echo "  Log:        $LOG"
echo "========================================"
