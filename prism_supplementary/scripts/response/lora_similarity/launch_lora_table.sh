#!/usr/bin/env bash
# LoRA checkpoint-ranking table: PRISM bound against 1-CKA and 1-SVCCA on the
# ten Llama-3.1-8B cells, three seeds.
#
#   bash scripts/response/lora_similarity/launch_lora_table.sh OUT_ROOT
#
# Knob: PYTHON (default python); the GPU is the one CUDA_VISIBLE_DEVICES exposes.
set -euo pipefail
OUT_ROOT=${1:?usage: launch_lora_table.sh OUT_ROOT}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}

"$PYTHON" "$HERE/run_series.py" --out_root "$OUT_ROOT"
"$PYTHON" "$HERE/analyze.py" --out_root "$OUT_ROOT"
