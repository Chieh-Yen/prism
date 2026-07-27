#!/usr/bin/env bash
# ============================================================
# E1-D: fresh, self-contained round of the similarity comparison
#
#   Produces one table, all five rows on the SAME features:
#     1-CKA | 1-SVCCA | 1-Omega_N | feature arm_N | PRISM B_N
#   as mean +- sd over three seeds the paper never used (43/44/45).
#
#   Why: E1 mixes pipelines (similarity recomputed, bound + |dR| joined from
#   the paper CSV), which is what forced the draft's "within-block comparisons
#   only" caveat and left delta_N = 0.901 (re-extracted) next to 0.873 (paper
#   Table 3). Here |dR| and B_N are recomputed too, so the table is internally
#   paired. It also fixes the paper round's omega artefact: float32 + the clamp
#   at prism/core/metrics.py:244 stores a literal 1.0 for 11/12 gsm8k cells.
#
#   stages
#     dry     zero GPU. Prints the plan and proves the clamp is in the paper's
#             metric path. Run this first.
#     run     GPU. Loads the target ONCE and each proxy ONCE, then loops seeds x
#             benchmarks inside. Resumable: rows are appended per (seed, proxy),
#             and a completed pair is never reloaded.
#             ~13 model loads/family instead of 39 -> ~20-30 min/family at 3
#             seeds, versus ~75 min for the naive seed-outer order. Loading a
#             GGUF/GPTQ 8B proxy is 1-2 min and dominates; a 5-benchmark forward
#             pass is ~5-8 s and the float64 metric ~2 s.
#     probe   zero GPU. Times the metric on random tensors and extrapolates the
#             wall clock for the whole grid. RUN THIS BEFORE `run`.
#     report  zero GPU. Builds out/E1D/table.md (+ diagnostics.md).
#     study   zero GPU. E14 size study only -> out/E14/{family}_size_study.md
#             (`run` already calls it per family).
#
#   usage
#     bash rebuttal_exp/script_E1D.sh dry
#     bash rebuttal_exp/script_E1D.sh run           # both families, seeds 43 44 45
#     FAMILIES=llama SEEDS="43 44 45" bash rebuttal_exp/script_E1D.sh run
#     bash rebuttal_exp/script_E1D.sh report
#
#   knobs: CUDA_GPU (0), FAMILIES ("llama qwen"), SEEDS ("43 44 45"),
#          SIZES ("8 32 128 512") reference-slice sizes; the largest is the paper
#                 protocol and doubles as the main table, FORCE=1
#          CHUNK (8192) tokens per float64 accumulation chunk. Lower it if the
#          metric OOMs next to the model; it does not change the result.
#
#   safe to re-launch after any interruption: the (seed, proxy) pairs already in
#   out/E1D/*_seed*.csv are skipped without loading the proxy, and the target
#   features are cached per (seed, benchmark) with atomic writes.
#   `report` REFUSES to emit a table if any cell is missing (pass
#   --allow-incomplete to override, which marks the table NOT final).
#
#   BEFORE YOU RUN, note the one real risk (also in the script docstring):
#   fixing the omega artefact RAISES the shape-core ranking, so the Table 3
#   ladder Omega_N 0.806 -> delta_N 0.873 -> B_N 0.912 may flatten. The
#   "machinery buys +0.106" line currently used for pCi8-W3 depends on that
#   ladder. If it flattens, rewrite that thread around the four-outputs framing
#   (ranking is one output of four) and report the new numbers as they come.
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-dry}"
export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"
FAMILIES="${FAMILIES:-llama qwen}"
SEEDS="${SEEDS:-43 44 45}"
SIZES="${SIZES:-8 32 128 512}"
PY=rebuttal_exp/exp_e1d_fresh_round.py
OUT=rebuttal_exp/out/E1D
mkdir -p "$OUT"
LOG="$OUT/screen.E1D.$(date +%Y%m%d_%H%M%S).log"
FORCE_FLAG=""
[[ "${FORCE:-0}" == "1" ]] && FORCE_FLAG="--force"

step () { echo "=== [E1D $(date '+%m-%d %H:%M')] $* ===" | tee -a "$LOG"; }

case "$STAGE" in
  dry)
    step "dry-run (zero GPU)"
    python3 "$PY" --dry-run --sizes $SIZES --seeds $SEEDS 2>&1 | tee -a "$LOG"
    ;;

  probe)
    step "probe (no model load): time the metric, extrapolate the whole grid"
    python3 "$PY" --probe --sizes $SIZES --seeds $SEEDS 2>&1 | tee -a "$LOG"
    ;;

  run)
    step "fresh round: families=[$FAMILIES] seeds=[$SEEDS]"
    for fam in $FAMILIES; do
      step "family=$fam"
      # shellcheck disable=SC2086
      python3 "$PY" --family "$fam" --seeds $SEEDS --sizes $SIZES \
              --chunk "${CHUNK:-8192}" $FORCE_FLAG 2>&1 | tee -a "$LOG"
      step "E14 size study (zero GPU) for $fam"
      python3 rebuttal_exp/exp_e14_size_study.py --family "$fam" 2>&1 \
              | tail -20 | tee -a "$LOG"
    done
    step "building table"
    python3 "$PY" --report 2>&1 | tee -a "$LOG"
    ;;

  report)
    step "report (zero GPU)"
    python3 "$PY" --report 2>&1 | tee -a "$LOG"
    ;;

  study)
    step "E14 size study only (zero GPU)"
    for fam in $FAMILIES; do
      python3 rebuttal_exp/exp_e14_size_study.py --family "$fam" 2>&1 | tee -a "$LOG"
    done
    ;;

  *)
    echo "usage: $0 {dry|probe|run|report|study}" >&2
    exit 2
    ;;
esac

step "stage=$STAGE done (log: $LOG)"
