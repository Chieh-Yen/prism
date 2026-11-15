#!/usr/bin/env bash
# Sensitivity to the anchor draw (Appendix I): the shape regularizer on
# Llama-3.1-8B / TruthfulQA at lambda 1e3, seeds 42, 43, 44, with the anchor set
# redrawn as four pairwise-disjoint windows of each seed's shuffle of the
# held-out split (--ref_offset k * n, k = 0..3), at n = 32, 16 and 8 sequences
# (--reg_samples n): 36 runs. All other arguments are those of the unified
# protocol (protocol.COMMON_ARGS). Window k = 0 at n = 32 is the anchor set of
# the main runs.
#
#   bash scripts/response/forgetting/launch_anchor_draw.sh OUT_ROOT
#
# Knob: PYTHON (default python); the GPU is the one CUDA_VISIBLE_DEVICES exposes.
#
# Run (n, k, seed) is written below OUT_ROOT/n<n>_off<k*n>/; finished runs are
# skipped. anchor_draw_table.py prints the forgetting of every run and the mean
# and population sd per n.
set -euo pipefail
OUT_ROOT=${1:?usage: launch_anchor_draw.sh OUT_ROOT}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}
TRAINER="$HERE/../../rounds/train_forgetting_baselines.py"
COMMON=$("$PYTHON" -c "import sys; sys.path.insert(0, '$HERE'); import protocol as P; print(' '.join(P.COMMON_ARGS))")

for n in 32 16 8; do
  for k in 0 1 2 3; do
    off=$((k * n))
    for seed in 42 43 44; do
      out="$OUT_ROOT/n${n}_off${off}"
      if [ -e "$out/trace/lam1000/seed${seed}/llama-3.1-8b/truthfulqa/prism_forgetting_metrics.json" ] \
         && grep -q '"step": 300' "$out/trace/lam1000/seed${seed}/llama-3.1-8b/truthfulqa/prism_forgetting_metrics.json"; then
        continue
      fi
      # shellcheck disable=SC2086
      "$PYTHON" "$TRAINER" --model meta-llama/Llama-3.1-8B --task truthfulqa \
        --seed "$seed" --method trace --lambda_reg 1000 --output_root "$out" \
        --save_steps 300 $COMMON --reg_samples "$n" --ref_offset "$off"
    done
  done
done
"$PYTHON" "$HERE/anchor_draw_table.py" --out_root "$OUT_ROOT"
