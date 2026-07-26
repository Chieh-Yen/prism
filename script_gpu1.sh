#!/usr/bin/env bash
# ============================================================
# GPU 1 — E2 training critical path (AC condition C)
#
#   backfill (~3 h, incl. canary check) -> CANARY GATE ->
#   sweep (~12 h) -> seeds (~14 h) -> aggregate (zero GPU)
#
# SAFE TO RE-LAUNCH after any interruption: completed runs are
# skipped via their step-300 metrics JSON (SKIP_DONE in
# script_E2.sh). The trainer has no mid-run resume, so only the
# single run that was in flight restarts from its own step 0.
#
# Canary gate (out/E2/backfill_check.md):
#   MATCH    -> proceed; tree anchors citable (flag CANARY_MATCH)
#   NEAR     -> proceed; sweep/seeds are a self-consistent fresh
#               round, but old-tree numbers must NOT be mixed and
#               E3C needs a fresh (task,32) anchor (flag CANARY_NEAR)
#   MISMATCH -> ABORT before burning 26 h; triage per TODOs.parallel.md
#
# TRACE_LAM stays 0.5 by default. If the printed lam=1.0@300 value
# beats 0.7357 (lower mean downstream |dR| = less forgetting), rerun:
#   TRACE_LAM=1.0 STAGE=seeds bash rebuttal_exp/script_E2.sh
# (skip-done makes that rerun touch only the 6 trace rows.)
#
# Knobs: CUDA_GPU (default 0), TRACE_LAM (default 0.5)
#   SKIP_BBQ=1   run TruthfulQA ONLY across sweep+seeds (drop BBQ) to finish
#                the TQA side first when time is tight. BBQ is simply never
#                looped; already-done TQA runs still skip, so a relaunch jumps
#                straight to the remaining TQA work. Re-enable later with a
#                plain relaunch (SKIP_BBQ unset) — BBQ runs then fill in.
# Run AFTER ./sync_to_runpod.sh.
# ============================================================
set -uo pipefail
cd "$(dirname "$0")"

export CUDA_GPU="${CUDA_GPU:-0}"

# SKIP_BBQ=1 -> TruthfulQA only. Exported TASKS is honored by the sweep/seeds
# task loops in script_E2.sh; the backfill trace anchors are already complete
# so they skip regardless of this.
if [[ "${SKIP_BBQ:-0}" == "1" ]]; then
    export TASKS="truthfulqa"
else
    export TASKS="${TASKS:-truthfulqa bbq}"
fi

OUT="rebuttal_exp/out/E2"
CHECK="$OUT/backfill_check.md"
LOG="$OUT/screen.gpu1.$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUT"
FAIL=0

step () { echo "=== [gpu1 $(date '+%m-%d %H:%M')] $* ===" | tee -a "$LOG"; }

step "E2 tasks = [$TASKS]$( [[ "${SKIP_BBQ:-0}" == "1" ]] && echo '  (SKIP_BBQ=1: BBQ disabled)' )"

step "STAGE=backfill (~3 h; lambda=1.0 TQA/bbq + canary check)"
STAGE=backfill bash rebuttal_exp/script_E2.sh 2>&1 | tee -a "$LOG" || FAIL=1

# ---------------- canary gate ----------------
if [[ ! -f "$CHECK" ]]; then
    step "ABORT: $CHECK missing — backfill or checker failed; do NOT start sweep/seeds"
    exit 1
fi
if grep -q "Verdict: MISMATCH" "$CHECK"; then
    step "ABORT: canary MISMATCH — environment does not reproduce the paper round."
    step "Triage: pip list | grep -E 'torch|transformers|peft'; diff the trainer config echo vs paper JSON experiment block (TODOs.parallel.md)"
    exit 1
fi
if grep -q "Verdict: NEAR" "$CHECK"; then
    step "canary NEAR — mild cross-machine drift: proceeding (fresh round is self-consistent; NEVER mix old-tree numbers). E3C will add a fresh (task,32) anchor."
    touch "$OUT/CANARY_NEAR"
else
    step "canary MATCH — environment certified; paper-tree anchors citable"
    touch "$OUT/CANARY_MATCH"
fi

# TRACE_LAM decision aid (informational; default 0.5 proceeds regardless)
python3 - <<'PY' 2>&1 | tee -a "$LOG"
import glob, json, statistics
DOWNSTREAM = ["arc", "mmlu", "squad", "triviaqa", "gsm8k"]
hits = glob.glob("rebuttal_exp/out/E2/trace/lam1/seed42/*/truthfulqa/"
                 "prism_forgetting_metrics_truthfulqa.json")
if hits:
    cks = json.load(open(hits[0])).get("checkpoints", [])
    v = [statistics.mean(c["tasks"][t]["delta_risk"]
                         for t in DOWNSTREAM if t in c["tasks"])
         for c in cks if c.get("step") == 300]
    if v:
        better = "lam=1.0 WINS -> consider TRACE_LAM=1.0 for seeds" \
            if v[0] < 0.7357 else "lam=0.5 stands (default)"
        print(f"[decision aid] trace lam=1.0 @300 mean downstream |dR| = "
              f"{v[0]:.4f} vs lam=0.5 anchor 0.7357 -> {better}")
    else:
        print("[decision aid] lam=1.0 json has no step-300 checkpoint")
else:
    print("[decision aid] lam=1.0 TQA json not found")
PY

step "STAGE=sweep (~12 h; layer_freeze/ewc/l2sp/feature_kd grids)"
STAGE=sweep bash rebuttal_exp/script_E2.sh 2>&1 | tee -a "$LOG" || FAIL=1

step "STAGE=seeds (~14 h; sweep-best x {43,44} + none/trace/replay x {42,43,44})"
STAGE=seeds bash rebuttal_exp/script_E2.sh 2>&1 | tee -a "$LOG" || FAIL=1

step "STAGE=aggregate (zero GPU)"
STAGE=aggregate bash rebuttal_exp/script_E2.sh 2>&1 | tee -a "$LOG" || FAIL=1

step "gpu1 queue done (FAIL=$FAIL) — pull out/E2 back and fill [TBD-E2]"
exit $FAIL
