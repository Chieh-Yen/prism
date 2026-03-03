#!/usr/bin/env bash
# ============================================================
# PRISM — Instruction-Tuning Forgetting Experiment Suite
# 6 datasets × 4 models  (base → instruct, Procrustes regime)
# ============================================================
set -euo pipefail

GPU="${CUDA_GPU:-1}"
N=128
CFG="configs/finetuning.yaml"
LOG="screen_finetuning.log"

run() {
    echo ">>> $*" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$GPU" python run.py --config "$CFG" "$@" 2>&1 | tee -a "$LOG"
}

DATASETS_ALL="c4 lambada wikitext gsm8k mmlu arc"

# ============================================================
# Model 1: Llama-2-7b  (base → chat)
# ============================================================
LLAMA_TARGET="target.model=NousResearch/Llama-2-7b-chat-hf"
LLAMA_PROXY="proxy.model=NousResearch/Llama-2-7b-hf"

for DS in $DATASETS_ALL; do
    run $LLAMA_TARGET $LLAMA_PROXY data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 2: Mistral-7B-v0.1  (base → instruct)
# ============================================================
MISTRAL_TARGET="target.model=mistralai/Mistral-7B-Instruct-v0.1"
MISTRAL_PROXY="proxy.model=mistralai/Mistral-7B-v0.1"

for DS in $DATASETS_ALL; do
    run $MISTRAL_TARGET $MISTRAL_PROXY data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 3: Qwen3-8B  (base → instruct, max_length=512)
# Qwen3 has 151K vocab → shorter sequences to fit 20GB GPU
# ============================================================
QWEN_TARGET="target.model=Qwen/Qwen3-8B"
QWEN_PROXY="proxy.model=Qwen/Qwen3-8B-Base"
QWEN_MAXLEN="data.max_length=512"

for DS in $DATASETS_ALL; do
    run $QWEN_TARGET $QWEN_PROXY $QWEN_MAXLEN data.task=$DS data.num_samples=$N
done

# ============================================================
# Model 4: OLMo-3-1025-7B  (base → instruct)
# ============================================================
OLMO_TARGET="target.model=allenai/Olmo-3-7B-Instruct"
OLMO_PROXY="proxy.model=allenai/Olmo-3-1025-7B"

for DS in $DATASETS_ALL; do
    run $OLMO_TARGET $OLMO_PROXY data.task=$DS data.num_samples=$N
done

echo "========================================"
echo "  All finetuning experiments complete."
echo "  Results in: ./results/finetuning/"
echo "  Log:        $LOG"
echo "========================================"
