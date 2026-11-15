#!/usr/bin/env bash
# Lambda search of the unified protocol, from a list of starting runs.
#
#   bash scripts/response/forgetting/search_lambda.sh OUT_ROOT START_JOBS [BUDGET]
#
# START_JOBS lists the runs each method starts from, one
# "method model task lambda seed" per line (seed 42); a method without a
# starting run in a cell is not searched there.  BUDGET is the tolerance in
# nats (default 0.01).
#
# 1. train the reference (no regularization, same loop; three seeds per cell)
#    and the starting runs;
# 2. evaluate the target-task loss outside the anchor set for all new runs;
# 3. ask select_lambda.py --emit for the next runs (the next grid value above
#    the current boundary at seed 42, or seeds 43/44 at the boundary; see its
#    docstring) and train them;
# 4. repeat 2-3 until nothing is requested, then print the tables.
#
# See "Lambda search" in the README.
#
# Knob: PYTHON (default python); the GPU is the one CUDA_VISIBLE_DEVICES exposes.
set -euo pipefail
OUT_ROOT=${1:?usage: search_lambda.sh OUT_ROOT START_JOBS [BUDGET]}
START_JOBS=${2:?usage: search_lambda.sh OUT_ROOT START_JOBS [BUDGET]}
BUDGET=${3:-0.01}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}

evaluate () {
  for model in llama qwen; do
    "$PYTHON" "$HERE/eval_target_disjoint.py" --out_root "$OUT_ROOT" --model "$model"
  done
}

"$PYTHON" "$HERE/run_unified.py" --out_root "$OUT_ROOT" --methods noreg_sameloop
"$PYTHON" "$HERE/run_unified.py" --out_root "$OUT_ROOT" --jobs "$START_JOBS"
PREVIOUS=""
while true; do
  evaluate
  JOBS=$("$PYTHON" "$HERE/select_lambda.py" --out_root "$OUT_ROOT" --budget "$BUDGET" --emit)
  [ -z "$JOBS" ] && break
  if [ "$JOBS" = "$PREVIOUS" ]; then
    echo "the requested runs finished without a target-task loss; stopping" >&2
    exit 1
  fi
  PREVIOUS=$JOBS
  echo "next runs:"; echo "$JOBS"
  echo "$JOBS" | "$PYTHON" "$HERE/run_unified.py" --out_root "$OUT_ROOT" --jobs -
done
"$PYTHON" "$HERE/select_lambda.py" --out_root "$OUT_ROOT" --budget "$BUDGET"
