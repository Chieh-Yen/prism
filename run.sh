# 測試長距離理解衰退用 lambada
#CUDA_VISIBLE_DEVICES=1 python run.py --config configs/quantization.yaml data.task=lambada data.num_samples=128 2>&1 | tee -a screen.log
#CUDA_VISIBLE_DEVICES=1 python run.py --config configs/quantization.yaml data.task=lambada data.num_samples=128 alignment.scale_absorbed=true 2>&1 | tee -a screen.log

# 仍可用 wikitext
#CUDA_VISIBLE_DEVICES=1 python run.py --config configs/quantization.yaml data.task=wikitext data.num_samples=128 2>&1 | tee -a screen.log
#CUDA_VISIBLE_DEVICES=1 python run.py --config configs/quantization.yaml data.task=wikitext data.num_samples=128 alignment.scale_absorbed=true 2>&1 | tee -a screen.log

# 預設用 C4（推薦）
#CUDA_VISIBLE_DEVICES=1 python run.py --config configs/quantization.yaml data.task=c4 data.num_samples=128 2>&1 | tee -a screen.log
#CUDA_VISIBLE_DEVICES=1 python run.py --config configs/quantization.yaml data.task=c4 data.num_samples=128 alignment.scale_absorbed=true 2>&1 | tee -a screen.log

# 數學推理
#CUDA_VISIBLE_DEVICES=0 python run.py --config configs/quantization.yaml data.task=gsm8k data.num_samples=128 2>&1 | tee -a screen.log.0
#CUDA_VISIBLE_DEVICES=0 python run.py --config configs/quantization.yaml data.task=gsm8k data.num_samples=128 alignment.scale_absorbed=true 2>&1 | tee -a screen.log.0

# 多學科知識（MMLU）
#CUDA_VISIBLE_DEVICES=0 python run.py --config configs/quantization.yaml data.task=mmlu data.num_samples=128 2>&1 | tee -a screen.log.0
#CUDA_VISIBLE_DEVICES=0 python run.py --config configs/quantization.yaml data.task=mmlu data.num_samples=128 alignment.scale_absorbed=true 2>&1 | tee -a screen.log.0

# 科學推理（ARC-Challenge）
#CUDA_VISIBLE_DEVICES=0 python run.py --config configs/quantization.yaml data.task=arc data.num_samples=128 2>&1 | tee -a screen.log
#CUDA_VISIBLE_DEVICES=0 python run.py --config configs/quantization.yaml data.task=arc data.num_samples=128 alignment.scale_absorbed=true 2>&1 | tee -a screen.log.0

#python run.py --config configs/quantization.yaml \
#    target.model=mistralai/Mistral-7B-v0.1 \
#    proxy.model=TheBloke/Mistral-7B-v0.1-GGUF \
#    data.task=c4 data.num_samples=128

# ============================================================
# Mistral-7B-v0.1 (template auto-derived: mistral-7b-v0.1.{quant}.gguf)
# ============================================================

# C4
#CUDA_VISIBLE_DEVICES=1 python run.py --config configs/quantization.yaml \
#    target.model=mistralai/Mistral-7B-v0.1 \
#    proxy.model=TheBloke/Mistral-7B-v0.1-GGUF \
#    data.task=c4 data.num_samples=128 2>&1 | tee -a screen.log
#CUDA_VISIBLE_DEVICES=1 python run.py --config configs/quantization.yaml \
#    target.model=mistralai/Mistral-7B-v0.1 \
#    proxy.model=TheBloke/Mistral-7B-v0.1-GGUF \
#    data.task=c4 data.num_samples=128 alignment.scale_absorbed=true 2>&1 | tee -a screen.log

# GSM8K
#CUDA_VISIBLE_DEVICES=1 python run.py --config configs/quantization.yaml \
#    target.model=mistralai/Mistral-7B-v0.1 \
#    proxy.model=TheBloke/Mistral-7B-v0.1-GGUF \
#    data.task=gsm8k data.num_samples=128 2>&1 | tee -a screen.log
#CUDA_VISIBLE_DEVICES=1 python run.py --config configs/quantization.yaml \
#    target.model=mistralai/Mistral-7B-v0.1 \
#    proxy.model=TheBloke/Mistral-7B-v0.1-GGUF \
#    data.task=gsm8k data.num_samples=128 alignment.scale_absorbed=true 2>&1 | tee -a screen.log

# ============================================================
# Qwen3-8B (template auto-derived: Qwen3-8B-{quant}.gguf)
# Available quants: Q8_0, Q6_K, Q5_K_M, Q5_0, Q4_K_M
# ============================================================
QWEN3_ARGS="target.model=Qwen/Qwen3-4B proxy.model=Qwen/Qwen3-4B-GGUF"
QWEN3_BITS="proxy.quantization_bits=[Q8_0,Q6_K,Q5_K_M,Q4_K_M]"
QWEN3_MAXLEN="data.max_length=2048"

# C4
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/quantization.yaml \
    $QWEN3_ARGS $QWEN3_BITS $QWEN3_MAXLEN data.task=c4 data.num_samples=128 2>&1 | tee -a screen.log
#CUDA_VISIBLE_DEVICES=1 python run.py --config configs/quantization.yaml \
#    $QWEN3_ARGS $QWEN3_BITS $QWEN3_MAXLEN data.task=c4 data.num_samples=128 alignment.scale_absorbed=true 2>&1 | tee -a screen.log
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/quantization.yaml \
    $QWEN3_ARGS $QWEN3_BITS $QWEN3_MAXLEN data.task=lambada data.num_samples=128 2>&1 | tee -a screen.log
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/quantization.yaml \
    $QWEN3_ARGS $QWEN3_BITS $QWEN3_MAXLEN data.task=wikitext data.num_samples=128 2>&1 | tee -a screen.log

# GSM8K
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/quantization.yaml \
    $QWEN3_ARGS $QWEN3_BITS $QWEN3_MAXLEN data.task=gsm8k data.num_samples=128 2>&1 | tee -a screen.log
#CUDA_VISIBLE_DEVICES=1 python run.py --config configs/quantization.yaml \
#    $QWEN3_ARGS $QWEN3_BITS $QWEN3_MAXLEN data.task=gsm8k data.num_samples=128 alignment.scale_absorbed=true 2>&1 | tee -a screen.log
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/quantization.yaml \
    $QWEN3_ARGS $QWEN3_BITS $QWEN3_MAXLEN data.task=mmlu data.num_samples=128 2>&1 | tee -a screen.log
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/quantization.yaml \
    $QWEN3_ARGS $QWEN3_BITS $QWEN3_MAXLEN data.task=arc data.num_samples=128 2>&1 | tee -a screen.log

