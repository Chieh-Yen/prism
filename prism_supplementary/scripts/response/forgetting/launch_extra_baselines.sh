#!/usr/bin/env bash
# LwF and representation KD (hint MSE) on Llama-3.1-8B / TruthfulQA under the
# same rule, with the weighted penalty lambda * P of each selected point.
#
#   bash scripts/response/forgetting/launch_extra_baselines.sh OUT_ROOT
#
# Knob: PYTHON (default python); the GPU is the one CUDA_VISIBLE_DEVICES exposes.
#
# OUT_ROOT must be the directory launch_main_table.sh wrote to: the reference
# and the shape-regularizer runs of that cell are read from it.
set -euo pipefail
OUT_ROOT=${1:?usage: launch_extra_baselines.sh OUT_ROOT}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}

"$PYTHON" "$HERE/run_unified.py" --out_root "$OUT_ROOT" --jobs "$HERE/jobs_extra_baselines.txt"
"$PYTHON" "$HERE/eval_target_disjoint.py" --out_root "$OUT_ROOT" --model llama
"$PYTHON" "$HERE/select_lambda.py" --out_root "$OUT_ROOT"
"$PYTHON" "$HERE/penalty_report.py" --out_root "$OUT_ROOT"
