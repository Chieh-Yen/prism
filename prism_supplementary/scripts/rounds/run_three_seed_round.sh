#!/usr/bin/env bash
# ============================================================
# Three-seed round of the reference-set-size comparison (Appendix F.2).
#
#   For each family, seed, reference size and benchmark, every score
#   (1-CKA, 1-SVCCA, the feature terms, the bound B_N) and the risk gap are
#   computed on the same features from one forward pass
#   (three_seed_round.py); size_study.py then builds the table.
#
#   stages
#     dry     no GPU. Prints the plan.
#     run     GPU. Loads the target once and each variant once per family and
#             loops over seeds, sizes and benchmarks. Resumable: rows are
#             appended per (seed, variant), finished cells are skipped, and
#             target features are cached per (seed, size, benchmark).
#             Ends with the `table` stage.
#     table   no GPU. Checks that the grid is complete and writes
#             out/rounds/size_study/{family}_size_study.md.
#
#   usage (from anywhere; the script changes to the supplementary root)
#     bash scripts/rounds/run_three_seed_round.sh dry
#     bash scripts/rounds/run_three_seed_round.sh run      # both families, seeds 43 44 45
#     FAMILIES=llama bash scripts/rounds/run_three_seed_round.sh run
#     bash scripts/rounds/run_three_seed_round.sh table
#
#   knobs: CUDA_GPU (0), FAMILIES ("llama qwen"), SEEDS ("43 44 45"),
#          SIZES ("8 32 128 512") reference sizes in sequences,
#          CHUNK (8192) token rows per float64 accumulation chunk (a memory
#          setting; it does not change the result),
#          FORCE=1 moves existing seed files aside and recomputes.
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/../.."   # supplementary root

STAGE="${1:-dry}"
export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"
FAMILIES="${FAMILIES:-llama qwen}"
SEEDS="${SEEDS:-43 44 45}"
SIZES="${SIZES:-8 32 128 512}"
PY=scripts/rounds/three_seed_round.py
OUT=out/rounds/three_seed
mkdir -p "$OUT"
LOG="$OUT/run.$(date +%Y%m%d_%H%M%S).log"
FORCE_FLAG=""
[[ "${FORCE:-0}" == "1" ]] && FORCE_FLAG="--force"

step () { echo "=== [$(date '+%m-%d %H:%M')] $* ===" | tee -a "$LOG"; }

table () {
    step "completeness check and tables (no GPU)"
    python3 "$PY" --check 2>&1 | tee -a "$LOG"
    for fam in $FAMILIES; do
        python3 scripts/rounds/size_study.py --family "$fam" 2>&1 | tee -a "$LOG"
    done
}

case "$STAGE" in
  dry)
    step "plan (no GPU)"
    for fam in $FAMILIES; do
        # shellcheck disable=SC2086
        python3 "$PY" --dry-run --family "$fam" --sizes $SIZES --seeds $SEEDS 2>&1 | tee -a "$LOG"
    done
    ;;

  run)
    step "three-seed round: families=[$FAMILIES] seeds=[$SEEDS] sizes=[$SIZES]"
    for fam in $FAMILIES; do
        step "family=$fam"
        # shellcheck disable=SC2086
        python3 "$PY" --family "$fam" --seeds $SEEDS --sizes $SIZES \
                --chunk "${CHUNK:-8192}" $FORCE_FLAG 2>&1 | tee -a "$LOG"
    done
    table
    ;;

  table)
    table
    ;;

  *)
    echo "usage: $0 {dry|run|table}" >&2
    exit 2
    ;;
esac

step "stage=$STAGE done (log: $LOG)"
