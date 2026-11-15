#!/usr/bin/env bash
# Layer freezing (Appendix I): LoRA adapters on the top 16 layers only, no
# penalty, trained in the loop of the unified protocol, on the four
# (model, task) cells, three seeds.
#
#   bash scripts/response/forgetting/launch_layer_freezing.sh OUT_ROOT
#
# Knob: PYTHON (default python); the GPU is the one CUDA_VISIBLE_DEVICES exposes.
#
# Runs every job of jobs_layer_freezing.txt (finished runs are skipped) and
# evaluates the target-task loss outside the anchor set. Layer freezing has no
# weight to select, so no selection follows: forgetting is read from each
# run's prism_forgetting_metrics.json and the target-task loss from
# OUT_ROOT/target_loss_disjoint_<model>.json (see the README).
set -euo pipefail
OUT_ROOT=${1:?usage: launch_layer_freezing.sh OUT_ROOT}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}

"$PYTHON" "$HERE/run_unified.py" --out_root "$OUT_ROOT" --jobs "$HERE/jobs_layer_freezing.txt"
for model in llama qwen; do
  "$PYTHON" "$HERE/eval_target_disjoint.py" --out_root "$OUT_ROOT" --model "$model"
done
