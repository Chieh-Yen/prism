#!/usr/bin/env bash
# ============================================================
# PRISM — Cross-Scale Proxy Validation Experiment Suite
# 6 datasets × 3 model families (smaller proxy → larger target)
#
# Scientific question: do PRISM metrics on a small proxy model
# reliably predict the large target's performance?
#
# Tokenizer note:
#   All proxies in the SAME family share the target's tokenizer.
#   Cross-family comparisons require a compatible vocabulary.
#
# Model families:
#   Family 1 — Qwen3-Base  (Qwen3 tokenizer, vocab=151K)
#     Target : Qwen/Qwen3-8B-Base
#     Proxies: Qwen/Qwen3-0.6B-Base, Qwen/Qwen3-1.7B-Base, Qwen/Qwen3-4B-Base
#              All Base variants — no instruction tuning, pure LM pre-training.
#
#   Family 2 — Llama-2  (LlamaTokenizer, vocab=32K, hidden=4096)
#     Target : NousResearch/Llama-2-7b-hf
#     Proxies: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T  [scale proxy]
#              mistralai/Mistral-7B-v0.1  [cross-arch proxy, same hidden/vocab]
#              → measures feature geometry difference between Llama-2 and Mistral
#
#   Family 3 — Mistral  (LlamaTokenizer, vocab=32K, hidden=4096)
#     Target : mistralai/Mistral-7B-v0.1
#     Proxies: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T  [scale proxy]
#              NousResearch/Llama-2-7b-hf  [cross-arch proxy, same hidden/vocab]
#              → symmetric counterpart of the Llama-2 experiment above
#
#   Cross-arch note (Llama-2 ↔ Mistral):
#     Both share hidden_size=4096 and LlamaTokenizer (vocab=32K), so inputs
#     are identical and head dimensions match exactly.  Ω quantifies how
#     similar the two models' internal feature geometry is across datasets.
#
# Usage:
#   bash run_cross_scale.sh
#   CUDA_GPU=0 bash run_cross_scale.sh
#   FAMILIES="qwen3" bash run_cross_scale.sh   # run only one family
# ============================================================
set -euo pipefail

GPU="${CUDA_GPU:-1}"
N="${NUM_SAMPLES:-128}"
CFG="configs/cross_scale.yaml"
LOG="screen_cross_scale.log"
FAMILIES="${FAMILIES:-qwen3 llama2 mistral}"

run() {
    echo ">>> $*" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$GPU" python run.py --config "$CFG" "$@" 2>&1 | tee -a "$LOG"
}

DATASETS_ALL="c4 lambada wikitext gsm8k mmlu arc"

# ============================================================
# Family 1: Qwen3-Base  (pure pre-trained, no instruction tuning)
# All Qwen3-Base models share tokenizer and vocab (151K tokens).
# Proxies cover 0.6B → 1.7B → 4B vs target 8B-Base.
# ============================================================
run_qwen3() {
    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "  Family 1: Qwen3-Base  (target=Qwen3-8B-Base)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    QWEN_TARGET="target.model=Qwen/Qwen3-8B-Base"
    QWEN_PROXIES="proxy.models=[Qwen/Qwen3-0.6B-Base,Qwen/Qwen3-1.7B-Base,Qwen/Qwen3-4B-Base]"
    QWEN_TARGET="target.model=Qwen/Qwen3-8B"
    QWEN_PROXIES="proxy.models=[Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B]"
    QWEN_MAXLEN="data.max_length=1024"   # Qwen3 151K vocab → shorter seqs on 20GB GPU

    for DS in $DATASETS_ALL; do
        run "$QWEN_TARGET" "$QWEN_PROXIES" "$QWEN_MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/qwen3
    done
}

# ============================================================
# Family 2: Llama-2
# Target : NousResearch/Llama-2-7b-hf  (hidden=4096, vocab=32K)
# Proxies:
#   - TinyLlama-1.1B        [scale proxy:  same Llama-2 arch, hidden=2048]
#   - mistralai/Mistral-7B-v0.1  [cross-arch proxy: hidden=4096, same vocab=32K]
#
# The Mistral proxy entry answers: how different is Mistral-7B's internal
# feature geometry from Llama-2-7B's?  Since hidden dims and vocab match
# exactly, head discrepancy is fully comparable without any truncation.
# ============================================================
run_llama2() {
    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "  Family 2: Llama-2  (target=Llama-2-7b-hf)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    LLAMA_TARGET="target.model=NousResearch/Llama-2-7b-hf"
    LLAMA_PROXIES="proxy.models=[TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T,mistralai/Mistral-7B-v0.1]"
    LLAMA_MAXLEN="data.max_length=1024"

    for DS in $DATASETS_ALL; do
        run "$LLAMA_TARGET" "$LLAMA_PROXIES" "$LLAMA_MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/llama2
    done
}

# ============================================================
# Family 3: Mistral
# Target : mistralai/Mistral-7B-v0.1  (hidden=4096, vocab=32K, LlamaTokenizer)
# Proxies:
#   - TinyLlama-1.1B             [scale proxy:  same tokenizer, hidden=2048]
#   - NousResearch/Llama-2-7b-hf [cross-arch proxy: hidden=4096, same vocab=32K]
#
# Symmetric counterpart of the Llama-2 experiment.  Using Llama-2-7B as
# a proxy for Mistral-7B measures the same geometry gap from the other
# direction; the Ω values should match (Procrustes is symmetric).
# ============================================================
run_mistral() {
    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "  Family 3: Mistral  (target=Mistral-7B-v0.1)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    MISTRAL_TARGET="target.model=mistralai/Mistral-7B-v0.1"
    MISTRAL_PROXIES="proxy.models=[TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T,NousResearch/Llama-2-7b-hf]"
    MISTRAL_MAXLEN="data.max_length=1024"

    for DS in $DATASETS_ALL; do
        run "$MISTRAL_TARGET" "$MISTRAL_PROXIES" "$MISTRAL_MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/mistral
    done
}

# ============================================================
# Dispatch
# ============================================================
for FAMILY in $FAMILIES; do
    case "$FAMILY" in
        qwen3)    run_qwen3  ;;
        llama2)   run_llama2 ;;
        mistral)  run_mistral ;;
        *)
            echo "Unknown family '$FAMILY'. Choose from: qwen3 llama2 mistral" >&2
            exit 1
            ;;
    esac
done

echo ""
echo "========================================"
echo "  All cross-scale experiments complete."
echo "  Results in: ./results/cross_scale/"
echo "  Log:        $LOG"
echo "========================================"
