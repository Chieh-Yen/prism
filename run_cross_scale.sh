#!/usr/bin/env bash
# ============================================================
# PRISM — Cross-Scale Proxy Validation Experiment Suite
# 6 datasets × 10 model families (dense lower-triangular: every
# smaller-proxy → every larger-target within each family)
#
# Scientific question: do PRISM metrics on a small proxy model
# reliably predict the large target's performance?
#
# Each model uses its OWN tokenizer for feature extraction and loss
# computation (per-model tokenizer policy in cross_scale.py).
#
# ── Qwen families (all use Qwen tokenizer, vocab=151K) ──────────────
#   qwen3base   — Qwen3  Base:      0.6B/1.7B/4B/8B/14B-Base  (dense ▽)
#   qwen3       — Qwen3  (thinking): 0.6B/1.7B/4B/8B/14B      (dense ▽)
#   qwen25base  — Qwen2.5 Base:     0.5B/1.5B/3B/7B/14B       (dense ▽)
#   qwen25      — Qwen2.5 Instruct: 0.5B/1.5B/3B/7B/14B-Inst  (dense ▽)
#
# ── Other families ───────────────────────────────────────────────────
#   llama2      — Llama-2 (vocab=32K) + cross-arch Mistral proxy
#   mistral     — Mistral-7B (vocab=32K) + cross-arch Llama-2 proxy
#   ministral3  — Ministral-3 2512 (multimodal wrapper, vocab=131K)
#   gemma2      — Gemma-2 2B→9B (vocab=256K)  [Google HF auth required]
#   gemma3      — Gemma-3 270M/1B/4B/12B-pt   (dense ▽) [Google HF auth]
#
# Dense lower-triangle (▽): for each target T, proxies = all models
# smaller than T within the same family.  Results land in per-target
# subdirs, e.g. results/cross_scale/qwen3base/target_14B/.
#
# Usage:
#   bash run_cross_scale.sh
#   CUDA_GPU=1 bash run_cross_scale.sh
#   FAMILIES="qwen25 qwen25base qwen3 qwen3base" bash run_cross_scale.sh
#   FAMILIES="gemma2 gemma3" bash run_cross_scale.sh
# ============================================================
set -euo pipefail

# ── GPU selection ─────────────────────────────────────────────
# MULTI_GPU=0 (default) → single GPU selected by CUDA_GPU (default 0)
# MULTI_GPU=1           → expose both GPU 0 and GPU 1; device=auto
#                         lets HuggingFace distribute each model across them
#
# Examples:
#   bash run_cross_scale.sh                               # GPU 0
#   CUDA_GPU=1 bash run_cross_scale.sh                    # GPU 1
#   MULTI_GPU=1 bash run_cross_scale.sh                   # GPU 0+1
#   MULTI_GPU=1 FAMILIES="gemma3" bash run_cross_scale.sh # GPU 0+1, gemma3 only
MULTI_GPU="${MULTI_GPU:-0}"
if [[ "$MULTI_GPU" == "1" ]]; then
    GPUIDS="0,1"
    DEVICE_OVERRIDE="device=auto"
else
    GPUIDS="${CUDA_GPU:-0}"
    DEVICE_OVERRIDE=""
fi

N="${NUM_SAMPLES:-128}"
CFG="configs/cross_scale.yaml"
LOG="screen_cross_scale.log"
FAMILIES="${FAMILIES:-qwen25 qwen25base qwen3 qwen3base llama2 mistral ministral3 gemma2 gemma3}"

run() {
    echo ">>> $*" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$GPUIDS" python run.py --config "$CFG" ${DEVICE_OVERRIDE:+"$DEVICE_OVERRIDE"} "$@" 2>&1 | tee -a "$LOG"
}

DATASETS_ALL="c4 lambada wikitext gsm8k mmlu arc"

# ============================================================
# Qwen families — all share Qwen tokenizer (vocab=151K).
# max_length=1024 to keep logit peak VRAM manageable on 20GB GPUs.
#
# Dense lower-triangle: run every (proxy < target) pair within the
# family.  Each target level writes to its own subdirectory.
# ============================================================

# ── Qwen3 Base ───────────────────────────────────────────────────────
# Pure pre-trained base; matches Qwen2.5-Base for apples-to-apples.
# Sizes : 0.6B-Base  1.7B-Base  4B-Base  8B-Base  14B-Base
# ▽ pairs:
#   target=14B-Base  proxies=[0.6B, 1.7B, 4B, 8B]-Base
#   target= 8B-Base  proxies=[0.6B, 1.7B, 4B]-Base
#   target= 4B-Base  proxies=[0.6B, 1.7B]-Base
#   target= 1.7B-Base proxies=[0.6B]-Base
run_qwen3base() {
    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "  Qwen3 Base  (dense lower-triangle, up to 14B-Base)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    MAXLEN="data.max_length=1024"

    # target = 14B-Base
    for DS in $DATASETS_ALL; do
        run "target.model=Qwen/Qwen3-14B-Base" \
            "proxy.models=[Qwen/Qwen3-0.6B-Base,Qwen/Qwen3-1.7B-Base,Qwen/Qwen3-4B-Base,Qwen/Qwen3-8B-Base]" \
            "$MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/qwen3base/target_14B
    done

    # target = 8B-Base
    for DS in $DATASETS_ALL; do
        run "target.model=Qwen/Qwen3-8B-Base" \
            "proxy.models=[Qwen/Qwen3-0.6B-Base,Qwen/Qwen3-1.7B-Base,Qwen/Qwen3-4B-Base]" \
            "$MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/qwen3base/target_8B
    done

    # target = 4B-Base
    for DS in $DATASETS_ALL; do
        run "target.model=Qwen/Qwen3-4B-Base" \
            "proxy.models=[Qwen/Qwen3-0.6B-Base,Qwen/Qwen3-1.7B-Base]" \
            "$MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/qwen3base/target_4B
    done

    # target = 1.7B-Base
    for DS in $DATASETS_ALL; do
        run "target.model=Qwen/Qwen3-1.7B-Base" \
            "proxy.models=[Qwen/Qwen3-0.6B-Base]" \
            "$MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/qwen3base/target_1.7B
    done
}

# ── Qwen3 (thinking / non-base) ──────────────────────────────────────
# Default Qwen3 models (thinking mode enabled by default in chat templates).
# Sizes : 0.6B  1.7B  4B  8B  14B
# ▽ pairs:
#   target=14B  proxies=[0.6B, 1.7B, 4B, 8B]
#   target= 8B  proxies=[0.6B, 1.7B, 4B]
#   target= 4B  proxies=[0.6B, 1.7B]
#   target= 1.7B proxies=[0.6B]
run_qwen3() {
    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "  Qwen3 (thinking)  (dense lower-triangle, up to 14B)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    MAXLEN="data.max_length=1024"

    # target = 14B
    for DS in $DATASETS_ALL; do
        run "target.model=Qwen/Qwen3-14B" \
            "proxy.models=[Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,Qwen/Qwen3-8B]" \
            "$MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/qwen3/target_14B
    done

    # target = 8B
    for DS in $DATASETS_ALL; do
        run "target.model=Qwen/Qwen3-8B" \
            "proxy.models=[Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B]" \
            "$MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/qwen3/target_8B
    done

    # target = 4B
    for DS in $DATASETS_ALL; do
        run "target.model=Qwen/Qwen3-4B" \
            "proxy.models=[Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B]" \
            "$MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/qwen3/target_4B
    done

    # target = 1.7B
    for DS in $DATASETS_ALL; do
        run "target.model=Qwen/Qwen3-1.7B" \
            "proxy.models=[Qwen/Qwen3-0.6B]" \
            "$MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/qwen3/target_1.7B
    done
}

# ── Qwen2.5 Base ─────────────────────────────────────────────────────
# Pure pre-trained base models; cleaner comparison for LM perplexity.
# Sizes : 0.5B  1.5B  3B  7B  14B
# ▽ pairs:
#   target=14B  proxies=[0.5B, 1.5B, 3B, 7B]
#   target= 7B  proxies=[0.5B, 1.5B, 3B]
#   target= 3B  proxies=[0.5B, 1.5B]
#   target= 1.5B proxies=[0.5B]
run_qwen25base() {
    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "  Qwen2.5 Base  (dense lower-triangle, up to 14B)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    MAXLEN="data.max_length=1024"

    # target = 14B
    for DS in $DATASETS_ALL; do
        run "target.model=Qwen/Qwen2.5-14B" \
            "proxy.models=[Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B,Qwen/Qwen2.5-3B,Qwen/Qwen2.5-7B]" \
            "$MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/qwen25base/target_14B
    done

    # target = 7B
    for DS in $DATASETS_ALL; do
        run "target.model=Qwen/Qwen2.5-7B" \
            "proxy.models=[Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B,Qwen/Qwen2.5-3B]" \
            "$MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/qwen25base/target_7B
    done

    # target = 3B
    for DS in $DATASETS_ALL; do
        run "target.model=Qwen/Qwen2.5-3B" \
            "proxy.models=[Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B]" \
            "$MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/qwen25base/target_3B
    done

    # target = 1.5B
    for DS in $DATASETS_ALL; do
        run "target.model=Qwen/Qwen2.5-1.5B" \
            "proxy.models=[Qwen/Qwen2.5-0.5B]" \
            "$MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/qwen25base/target_1.5B
    done
}

# ── Qwen2.5 Instruct ─────────────────────────────────────────────────
# Instruction-tuned models; useful for Q&A datasets (GSM8K, MMLU, ARC).
# Sizes : 0.5B  1.5B  3B  7B  14B  (all -Instruct)
# ▽ pairs:
#   target=14B-Instruct  proxies=[0.5B, 1.5B, 3B, 7B]-Instruct
#   target= 7B-Instruct  proxies=[0.5B, 1.5B, 3B]-Instruct
#   target= 3B-Instruct  proxies=[0.5B, 1.5B]-Instruct
#   target= 1.5B-Instruct proxies=[0.5B]-Instruct
run_qwen25() {
    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "  Qwen2.5 Instruct  (dense lower-triangle, up to 14B-Instruct)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    MAXLEN="data.max_length=1024"

    # target = 14B-Instruct
    for DS in $DATASETS_ALL; do
        run "target.model=Qwen/Qwen2.5-14B-Instruct" \
            "proxy.models=[Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct]" \
            "$MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/qwen25/target_14B
    done

    # target = 7B-Instruct
    for DS in $DATASETS_ALL; do
        run "target.model=Qwen/Qwen2.5-7B-Instruct" \
            "proxy.models=[Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct]" \
            "$MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/qwen25/target_7B
    done

    # target = 3B-Instruct
    for DS in $DATASETS_ALL; do
        run "target.model=Qwen/Qwen2.5-3B-Instruct" \
            "proxy.models=[Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct]" \
            "$MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/qwen25/target_3B
    done

    # target = 1.5B-Instruct
    for DS in $DATASETS_ALL; do
        run "target.model=Qwen/Qwen2.5-1.5B-Instruct" \
            "proxy.models=[Qwen/Qwen2.5-0.5B-Instruct]" \
            "$MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/qwen25/target_1.5B
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
    # Ministral-3 2512 uses Mistral3ForConditionalGeneration (multimodal wrapper).
    # Text backbone lives at model.language_model — extra memory vs pure causal LM.
    M3_MAXLEN="data.max_length=1024"

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
# Sizes : 270M  1B-pt  4B-pt  12B-pt
# ▽ pairs:
#   target=12B-pt  proxies=[270M, 1B-pt, 4B-pt]
#   target= 4B-pt  proxies=[270M, 1B-pt]
#   target= 1B-pt  proxies=[270M]
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
    echo "  Family 6: Gemma-3  (dense lower-triangle, up to 12B-pt)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    G3_MAXLEN="data.max_length=1024"    # 12B multimodal model — conservative for 20GB GPU

    # target = 12B-pt
    for DS in $DATASETS_ALL; do
        run "target.model=google/gemma-3-12b-pt" \
            "proxy.models=[google/gemma-3-270m,google/gemma-3-1b-pt,google/gemma-3-4b-pt]" \
            "$G3_MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/gemma3/target_12B
    done

    # target = 4B-pt
    for DS in $DATASETS_ALL; do
        run "target.model=google/gemma-3-4b-pt" \
            "proxy.models=[google/gemma-3-270m,google/gemma-3-1b-pt]" \
            "$G3_MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/gemma3/target_4B
    done

    # target = 1B-pt
    for DS in $DATASETS_ALL; do
        run "target.model=google/gemma-3-1b-pt" \
            "proxy.models=[google/gemma-3-270m]" \
            "$G3_MAXLEN" \
            data.task="$DS" data.num_samples="$N" \
            output.dir=./results/cross_scale/gemma3/target_1B
    done
}

# ============================================================
# Dispatch
# ============================================================
for FAMILY in $FAMILIES; do
    case "$FAMILY" in
        qwen3base)  run_qwen3base  ;;
        qwen3)      run_qwen3      ;;
        qwen25base) run_qwen25base ;;
        qwen25)     run_qwen25     ;;
        llama2)     run_llama2     ;;
        mistral)    run_mistral    ;;
        ministral3) run_ministral3 ;;
        gemma2)     run_gemma2     ;;
        gemma3)     run_gemma3     ;;
        *)
            echo "Unknown family '$FAMILY'. Choose from: qwen25 qwen25base qwen3 qwen3base llama2 mistral ministral3 gemma2 gemma3" >&2
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
