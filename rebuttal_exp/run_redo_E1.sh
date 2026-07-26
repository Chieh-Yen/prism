#!/usr/bin/env bash
# ============================================================
# run_redo_E1.sh — reconcile the gsm8k Omega_N inconsistency
#                  (G3T9-W3 / pCi8-W3 rigor fix, 2026-07-26)
#
# WHY THIS EXISTS
# ---------------
# The E1 metrics CSV joins B_I/B_W and reads Omega_W from the PAPER CSV
# (paper pipeline), while CKA/SVCCA/Procr/omega_I are RE-EXTRACTED (E1
# pipeline). On gsm8k these disagree: paper 1-Omega_W = +0.48 ("shape core
# saturates", the source of the draft's "machinery buys +0.088"), but the
# E1 re-extraction gives 1-omega_I/1-CKA = +0.94. The old story blamed a
# 16k-token subsample artifact — but the REDO now re-extracts gsm8k at the
# FULL 65536 cap and still returns +0.94, so it is NOT a subsample artifact;
# it is a paper-vs-E1 pipeline difference.
#
# The missing piece was the shape core (Omega_N) computed on the SAME
# re-extracted features as CKA. exp_e1_similarity_baselines.py now emits a
# native nuclear `omega_W` column (added 2026-07-26). This script forces a
# clean, fully consistent re-extraction so every benchmark has that column,
# then regenerates the derived tables. Read §3 of E1.exp.md afterwards: if
# pod 1-Omega_W ~ pod 1-Omega_I ~ 1-CKA (all high) on gsm8k, the "+0.48
# saturates / +0.088 gain" claim is paper-pipeline-only and must be rewritten;
# if pod 1-Omega_W stays low, the shape-core story survives in the pod gauge.
#
# COST: full re-extract, ~1.5-2.5 h per family on 1x RTX 5090 (proxy reload
#       dominates; target features are reused from out/E1/{fam}_ZT/). The
#       nuclear-omega SVD adds a few seconds per (variant, benchmark).
#
# KNOBS
#   FAMILIES="llama qwen"   subset / order of families
#   CUDA_GPU=0              GPU index (run llama on 0, qwen on 1 to parallelize)
#   TOKEN_CAP=65536         paired-token cap (full gsm8k)
#   NUM_SAMPLES=512         paper protocol
#   FORCE=0                 1 = re-extract even if omega_W already present
#
# PARALLEL (two cards, ~halves wall-clock):
#   FAMILIES=llama CUDA_GPU=0 bash rebuttal_exp/run_redo_E1.sh &
#   FAMILIES=qwen  CUDA_GPU=1 bash rebuttal_exp/run_redo_E1.sh &
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."           # repo root

FAMILIES="${FAMILIES:-llama qwen}"
TOKEN_CAP="${TOKEN_CAP:-65536}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
FORCE="${FORCE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="rebuttal_exp/out/E1"
LOG="$OUT/screen.redoE1.${TS}.log"
mkdir -p "$OUT"
FAIL=0

log() { echo "$@" | tee -a "$LOG"; }

log "=== run_redo_E1 start ($(date)); families='$FAMILIES' cap=$TOKEN_CAP GPU=${CUDA_GPU:-0} ==="

# ── 1. Force a clean full recompute so `omega_W` is populated everywhere ──
# Older CSVs lack the omega_W column; resume would keep those rows with a
# NaN shape core. Move the CSV aside (target-feature cache is untouched, so
# only proxy reload + extraction re-runs). Skip if omega_W already present
# unless FORCE=1.
for fam in $FAMILIES; do
    csv="$OUT/${fam}_metrics.csv"
    if [[ -f "$csv" ]]; then
        if head -1 "$csv" | grep -q "omega_W" && [[ "$FORCE" != "1" ]]; then
            log ">>> $fam: omega_W already in CSV — skipping re-extract "
            log "    (set FORCE=1 to recompute anyway)."
            continue
        fi
        bak="$csv.pre_omegaW.${TS}.bak"
        cp "$csv" "$bak"
        rm -f "$csv"
        log ">>> $fam: backed up old CSV -> $(basename "$bak"); forcing full recompute"
    fi
done

# ── 2. Re-extract (GPU). Full run: no --redo, no resume rows to keep. ────
for fam in $FAMILIES; do
    csv="$OUT/${fam}_metrics.csv"
    if [[ -f "$csv" ]] && head -1 "$csv" | grep -q "omega_W" && [[ "$FORCE" != "1" ]]; then
        continue   # already reconciled above
    fi
    log "=== E1 re-extract family=$fam ($(date)) ==="
    python rebuttal_exp/exp_e1_similarity_baselines.py \
        --family "$fam" \
        --num_samples "$NUM_SAMPLES" \
        --token_cap "$TOKEN_CAP" \
        2>&1 | tee -a "$LOG" || FAIL=1
done

# ── 3. Regenerate derived tables (zero GPU) ──────────────────────────────
log "=== regenerate subgroup_analysis.md + E1.exp.md ($(date)) ==="
python rebuttal_exp/exp_e1_subgroups.py 2>&1 | tee -a "$LOG" || FAIL=1
python rebuttal_exp/exp_e1_report.py    2>&1 | tee -a "$LOG" || FAIL=1

# ── 4. Reconciliation pointer ────────────────────────────────────────────
log ""
log "=== run_redo_E1 done ($(date)); FAIL=$FAIL ==="
log "NEXT: read out/E1/*_spearman.md (new 1-Omega_W column) and E1.exp.md §3."
log "  - If gsm8k pod 1-Omega_W ~ 1-CKA (high): drop the draft's 'Omega_N"
log "    +0.48 saturates' + revisit pCi8-W3 '+0.088 gain' (paper-pipeline-only)."
log "  - If gsm8k pod 1-Omega_W stays low: restate the shape-core story in the"
log "    pod gauge, consistently sourced with CKA."
log "  Headline B_I/B_W/CKA/SVCCA means are unaffected (already confirmed)."
exit $FAIL
