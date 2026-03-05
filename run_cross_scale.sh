#!/usr/bin/env bash
# ============================================================
# PRISM — Cross-Scale Proxy Validation Experiment Suite
# 6 datasets × 6 model families (smaller proxy → larger target)
#
# Scientific question: do PRISM metrics on a small proxy model
# reliably predict the large target's performance?
#
# Each model uses its OWN tokenizer for feature extraction and loss
# computation (per-model tokenizer policy in cross_scale.py).
#
# Model families:
#   Family 1 — Qwen3-Base  (Qwen3 tokenizer, vocab=151K)
#     Target : Qwen/Qwen3-8B
#     Proxies: Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B, Qwen/Qwen3-4B
#
#   Family 2 — Llama-2  (LlamaTokenizer, vocab=32K)
#     Target : NousResearch/Llama-2-7b-hf
#     Proxies: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
#              mistralai/Mistral-7B-v0.1  [cross-arch, same vocab=32K]
#
#   Family 3 — Mistral  (LlamaTokenizer, vocab=32K)
#     Target : mistralai/Mistral-7B-v0.1
#     Proxies: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
#              NousResearch/Llama-2-7b-hf  [cross-arch, same vocab=32K]
#
#   Family 4 — Ministral-3  (Tekken tokenizer, vocab=131K)  [NO auth required]
#     Target : mistralai/Ministral-3-8B-Base-2512
#     Proxies: mistralai/Ministral-3-3B-Base-2512
#
#   Family 5 — Gemma-2  (Gemma tokenizer, vocab=256K)  [requires Google HF auth]
#     Target : google/gemma-2-9b
#     Proxies: google/gemma-2-2b
#
#   Family 6 — Gemma-3  (Gemma3 tokenizer, vocab=256K) [requires Google HF auth]
#     Target : google/gemma-3-12b-pt
#     Proxies: google/gemma-3-270m, google/gemma-3-1b-pt, google/gemma-3-4b-pt
#     Note: 270M and 1B are text-only; 4B and 12B are multimodal
#           (Gemma3ForConditionalGeneration) — LLMExtractor handles both.
#
# Usage:
#   bash run_cross_scale.sh
#   CUDA_GPU=0 bash run_cross_scale.sh
#   FAMILIES="qwen3" bash run_cross_scale.sh           # run only one family
#   FAMILIES="ministral3 gemma2 gemma3" bash run_cross_scale.sh
# ============================================================
set -euo pipefail

GPU="${CUDA_GPU:-1}"
N="${NUM_SAMPLES:-128}"
CFG="configs/cross_scale.yaml"
LOG="screen_cross_scale.log"
FAMILIES="${FAMILIES:-qwen3 llama2 mistral ministral3 gemma2 gemma3}"

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


    # QWEN_TARGET="target.model=Qwen/Qwen3-8B-Base"
    # QWEN_PROXIES="proxy.models=[Qwen/Qwen3-0.6B-Base,Qwen/Qwen3-1.7B-Base,Qwen/Qwen3-4B-Base]"
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
# Family 4: Ministral-3  (publicly accessible, no HF auth needed)
# Target : mistralai/Ministral-3-8B-Base-2512  (hidden=4096, vocab=131K, Tekken-v7)
# Proxy  : mistralai/Ministral-3-3B-Base-2512  (hidden=3072, vocab=131K, Tekken-v7)
#
# Both models share the same Tekken v7 tokenizer → same tokenizer detected,
# target_dataloader is reused for efficiency.
# ============================================================
run_ministral3() {
    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "  Family 4: Ministral-3  (target=Ministral-3-8B-Base-2512)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    M3_TARGET="target.model=mistralai/Ministral-3-8B-Base-2512"
    M3_PROXIES="proxy.models=[mistralai/Ministral-3-3B-Base-2512]"
    M3_MAXLEN="data.max_length=1024"   # Tekken 131K vocab → same as Qwen3 precaution

    for DS in $DATASETS_ALL; do
        run "$M3_TARGET" "$M3_PROXIES" "$M3_MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/ministral3
    done
}

# ============================================================
# Family 5: Gemma-2  (requires Google HuggingFace authorization)
# Target : google/gemma-2-9b   (hidden=3584, vocab=256K, Gemma tokenizer)
# Proxy  : google/gemma-2-2b   (hidden=2304, vocab=256K, Gemma tokenizer)
#
# Both models share the same SentencePiece Gemma tokenizer.
# To use: huggingface-cli login  (accept Gemma license at hf.co/google/gemma-2-9b)
# ============================================================
run_gemma2() {
    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "  Family 5: Gemma-2  (target=gemma-2-9b)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    G2_TARGET="target.model=google/gemma-2-9b"
    G2_PROXIES="proxy.models=[google/gemma-2-2b]"
    G2_MAXLEN="data.max_length=1024"   # Gemma 256K vocab → shorter seqs on 20GB GPU

    for DS in $DATASETS_ALL; do
        run "$G2_TARGET" "$G2_PROXIES" "$G2_MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/gemma2
    done
}

# ============================================================
# Family 6: Gemma-3  (requires Google HuggingFace authorization)
# Target : google/gemma-3-12b-pt  (multimodal, hidden=2560, vocab=256K)
# Proxies:
#   - google/gemma-3-270m    [text-only, Gemma3ForCausalLM, hidden=1152]
#   - google/gemma-3-1b-pt   [text-only, Gemma3ForCausalLM, hidden=1152]
#   - google/gemma-3-4b-pt   [multimodal, Gemma3ForConditionalGeneration, hidden=2560]
#
# Note on architecture:
#   270M and 1B use Gemma3ForCausalLM (standard backbone at model.model).
#   4B, 12B, 27B use Gemma3ForConditionalGeneration (backbone at
#   model.language_model.model — handled by the updated LLMExtractor).
#
# Note: gemma-3-270m has no "-pt" suffix (base model released without suffix).
# To use: huggingface-cli login  (accept Gemma license at hf.co/google/gemma-3-12b-pt)
# ============================================================
run_gemma3() {
    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "  Family 6: Gemma-3  (target=gemma-3-12b-pt)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    G3_TARGET="target.model=google/gemma-3-12b-pt"
    G3_PROXIES="proxy.models=[google/gemma-3-270m,google/gemma-3-1b-pt,google/gemma-3-4b-pt]"
    G3_MAXLEN="data.max_length=512"    # 12B multimodal model — conservative for 20GB GPU

    for DS in $DATASETS_ALL; do
        run "$G3_TARGET" "$G3_PROXIES" "$G3_MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/gemma3
    done
}

# ============================================================
# Dispatch
# ============================================================
for FAMILY in $FAMILIES; do
    case "$FAMILY" in
        qwen3)      run_qwen3      ;;
        llama2)     run_llama2     ;;
        mistral)    run_mistral    ;;
        ministral3) run_ministral3 ;;
        gemma2)     run_gemma2     ;;
        gemma3)     run_gemma3     ;;
        *)
            echo "Unknown family '$FAMILY'. Choose from: qwen3 llama2 mistral ministral3 gemma2 gemma3" >&2
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
