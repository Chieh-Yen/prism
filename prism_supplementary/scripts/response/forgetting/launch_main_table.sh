#!/usr/bin/env bash
# Main forgetting table: no regularization (same loop), anchor-CE, generic
# replay and the shape regularizer on four (model, task) cells, three seeds.
#
#   bash scripts/response/forgetting/launch_main_table.sh OUT_ROOT
#
# Knob: PYTHON (default python); the GPU is the one CUDA_VISIBLE_DEVICES exposes.
#
# Runs every job of jobs_main_table.txt (finished runs are skipped), evaluates
# the target-task loss outside the anchor set, and prints the tables.
set -euo pipefail
OUT_ROOT=${1:?usage: launch_main_table.sh OUT_ROOT}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}

"$PYTHON" "$HERE/run_unified.py" --out_root "$OUT_ROOT" --jobs "$HERE/jobs_main_table.txt"
for model in llama qwen; do
  "$PYTHON" "$HERE/eval_target_disjoint.py" --out_root "$OUT_ROOT" --model "$model"
done
"$PYTHON" "$HERE/select_lambda.py" --out_root "$OUT_ROOT" \
  || echo "select_lambda.py: search not finished; run it with --emit for the next runs"
