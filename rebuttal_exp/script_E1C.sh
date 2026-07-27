#!/usr/bin/env bash
# ============================================================
# E1-C: reproduce the PAPER's feature round, then emit paper-round CKA/SVCCA
#
#   why: E1 joins bound_I/bound_W from the paper CSV but recomputes
#        cka/svcca/procr_dist on a fresh extraction. The two rounds agree to
#        3 decimals on arc/mmlu/squad/triviaqa and disagree on gsm8k, because
#        the paper CSV stores Omega = exactly 1.0 for 11/12 gsm8k cells
#        (saturation tracks token count: gsm8k 52184 tokens vs mmlu 511).
#        Until we can reproduce the paper's round we cannot state a
#        paper-pipeline CKA/SVCCA number.
#
#   stages
#     selftest  zero GPU. Verifies the delta identity that the acceptance test
#               rests on (168/168 llama, 154/154 qwen) + prints the saturation
#               map. ALWAYS run this first; if it fails, stop.
#     compare   zero GPU. old(16k) vs new(52k) gsm8k token budget, from the two
#               extractions already on disk. Result: largest |old-new| = 0.0000
#               over all cells -> the cap axis is INERT, so the sweep must chase
#               the metric path (cast/dtype/clamp), not the token budget.
#     sweep     GPU. gsm8k only, both families. Inner (free) axes: token cap x
#               feature cast (none/bf16/fp16) x accumulation dtype x omega clamp;
#               outer axes (need re-extraction): num_samples x max_length.
#               Scores every config against the paper's rho/Omega.
#               Writes out/E1C/{family}_reconcile.md.
#     lock      GPU. Re-runs all 5 benchmarks at ONE config (set LOCK=...)
#               and writes out/E1C/{family}_paperround.csv.
#     report    zero GPU. Spearman per benchmark + paired bootstrap
#               (feature arm vs CKA/SVCCA, same features) -> out/E1C/report.md
#
#   usage
#     bash rebuttal_exp/script_E1C.sh selftest
#     bash rebuttal_exp/script_E1C.sh compare                # zero GPU, 2 s
#     bash rebuttal_exp/script_E1C.sh sweep                  # ~50 min/family
#     LOCK="cap=52184,dtype=float64,clamp=on" \
#       bash rebuttal_exp/script_E1C.sh lock                 # ~25 min/family
#     bash rebuttal_exp/script_E1C.sh report
#
#   knobs: CUDA_GPU (default 0), FAMILIES (default "llama qwen"), LOCK
#
#   reading the verdict in {family}_reconcile.md
#     MATCH    -> the paper-round CKA/SVCCA in {family}_paperround.csv are
#                 certified; they are the downward-revised Block 1 numbers.
#     NEAR     -> report the residual; treat as indicative, not certified.
#     NO MATCH -> the divergence is not cap/dtype/clamp. Do NOT pick the
#                 closest config and call it the paper's. Fall back to the
#                 difference-invariance argument already in the draft
#                 (all three scores share features; per-cell gap <= 0.036).
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."                     # repo root

STAGE="${1:-selftest}"
export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"
FAMILIES="${FAMILIES:-llama qwen}"
PY=rebuttal_exp/exp_e1c_paper_round_reconcile.py
OUT=rebuttal_exp/out/E1C
mkdir -p "$OUT"
LOG="$OUT/screen.E1C.$(date +%Y%m%d_%H%M%S).log"

step () { echo "=== [E1C $(date '+%m-%d %H:%M')] $* ===" | tee -a "$LOG"; }

case "$STAGE" in
  selftest)
    step "selftest (zero GPU): delta identity + saturation map"
    python3 "$PY" --selftest 2>&1 | tee -a "$LOG"
    ;;

  compare)
    step "compare (zero GPU): old(16k) vs new(52k) gsm8k token budget"
    python3 "$PY" --compare-runs 2>&1 | tee -a "$LOG"
    ;;

  sweep)
    step "sweep on gsm8k only (the one benchmark that disagrees)"
    for fam in $FAMILIES; do
      step "sweep family=$fam"
      python3 "$PY" --family "$fam" --benchmarks gsm8k 2>&1 | tee -a "$LOG"
      echo "--- verdicts ($fam) ---" | tee -a "$LOG"
      grep -E '^\| `cap=' "$OUT/${fam}_reconcile.md" 2>/dev/null | tee -a "$LOG"
    done
    step "read $OUT/*_reconcile.md, pick the MATCH config, then: LOCK=... $0 lock"
    ;;

  lock)
    if [[ -z "${LOCK:-}" ]]; then
      echo "!! set LOCK, e.g. LOCK=\"cap=52184,dtype=float64,clamp=on\"" >&2
      exit 2
    fi
    step "lock config=$LOCK over all 5 benchmarks"
    for fam in $FAMILIES; do
      step "lock family=$fam"
      python3 "$PY" --family "$fam" \
              --benchmarks arc mmlu squad triviaqa gsm8k \
              --lock-config "$LOCK" 2>&1 | tee -a "$LOG"
    done
    step "then: $0 report"
    ;;

  report)
    step "report (zero GPU): Spearman + paired bootstrap"
    python3 "$PY" --report 2>&1 | tee -a "$LOG"
    ;;

  *)
    echo "usage: $0 {selftest|compare|sweep|lock|report}" >&2
    exit 2
    ;;
esac

step "stage=$STAGE done (log: $LOG)"
