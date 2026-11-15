#!/usr/bin/env bash
# The Appendix I baselines L2-SP and EWC on the four (model, task) cells under
# the rule of the unified protocol, next to the same-loop reference and the
# shape regularizer.
#
#   bash scripts/response/forgetting/launch_appendix_baselines.sh OUT_ROOT
#
# Knob: PYTHON (default python); the GPU is the one CUDA_VISIBLE_DEVICES exposes.
#
# OUT_ROOT must be the directory launch_main_table.sh wrote to: the reference
# and the shape-regularizer runs of the four cells are read from it.
set -euo pipefail
OUT_ROOT=${1:?usage: launch_appendix_baselines.sh OUT_ROOT}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}

"$PYTHON" "$HERE/run_unified.py" --out_root "$OUT_ROOT" --jobs "$HERE/jobs_appendix_baselines.txt"
for model in llama qwen; do
  "$PYTHON" "$HERE/eval_target_disjoint.py" --out_root "$OUT_ROOT" --model "$model"
done
"$PYTHON" "$HERE/select_lambda.py" --out_root "$OUT_ROOT"
"$PYTHON" "$HERE/penalty_report.py" --out_root "$OUT_ROOT"
