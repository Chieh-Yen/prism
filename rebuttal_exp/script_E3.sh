#!/usr/bin/env bash
# ============================================================
# E3 — Reference-set ablation           (G3T9-W2, 8VrD-Q3)
#
#   Part A  benchmark-independent reference (ZERO GPU — reuses the
#           paper CSV's wikitext / fineweb_edu rows). Runs anywhere.
#   Part B  reference-SIZE ablation {8,16,32,64,128} on wikitext,
#           Llama family, N_DRAWS independent draws per size
#           (stability).                      ~1.5-2 h GPU
#   Part C  regularizer-side sensitivity (8VrD-Q3): trace lam=0.5
#           (paper operating point), Llama-TruthfulQA; cells set by
#           E3C_CELLS ("ref:n" pairs).        ~1.5 h GPU per run
#           Completed cells (step-300 json) are skipped on re-entry.
#
# Knobs: PARTS="A B C", CUDA_GPU, FAMILY=llama, N_DRAWS=3, LAMBDA=0.5,
#        E3C_CELLS (default full 3x2 grid; trimmed cross design =
#        "truthfulqa:8 truthfulqa:128 wikitext:32", with the paper tree's
#        lam0.5@300 anchor standing in for (truthfulqa,32) after the E2
#        canary MATCHes)
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."           # repo root

PARTS="${PARTS:-A B C}"
FAMILY="${FAMILY:-llama}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B}"   # 統一用 CSV 同名 repo(同權重 alias);省 15G 重複 cache
export CUDA_VISIBLE_DEVICES="${CUDA_GPU:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="rebuttal_exp/out/E3"
LOG="$OUT/screen.E3.${TS}.log"
mkdir -p "$OUT"
FAIL=0

if [[ " $PARTS " == *" A "* ]]; then
    echo "=== E3 part A (csv-only, zero GPU) ===" | tee -a "$LOG"
    python rebuttal_exp/exp_e3_refset_ablation.py --csv-only \
        2>&1 | tee -a "$LOG" || FAIL=1
fi

if [[ " $PARTS " == *" B "* ]]; then
    echo "=== E3 part B (size ablation + stability draws, GPU) ===" | tee -a "$LOG"
    python rebuttal_exp/exp_e3_refset_ablation.py --family "$FAMILY" \
        --n_draws "${N_DRAWS:-3}" \
        2>&1 | tee -a "$LOG" || FAIL=1
fi

if [[ " $PARTS " == *" C "* ]]; then
    echo "=== E3 part C (regularizer sensitivity, GPU) ===" | tee -a "$LOG"
    # lambda follows the paper's llama-TQA operating point (0.5 — the value
    # with full E10 coverage); lr comes from the trainer's fixed 1e-5
    # default (the paper-round launch value). Override: LAMBDA=1.0.
    LAMBDA="${LAMBDA:-0.5}"
    # Cells as "ref:n" pairs. Default = full 3x2 grid. Trimmed cross design
    # (see header) skips (truthfulqa,32) — the paper tree's lam0.5@300 run
    # IS that cell once the E2 backfill canary certifies the environment.
    E3C_CELLS="${E3C_CELLS:-truthfulqa:8 truthfulqa:32 truthfulqa:128 wikitext:8 wikitext:32 wikitext:128}"
    c_done () {  # output_root -> 0 if the cell's step-300 json exists
        python3 - "$1" "$LAMBDA" "$MODEL" <<'PY'
import json, os, sys
root, lam, model = sys.argv[1:4]
p = os.path.join(root, "trace", f"lam{float(lam):g}", "seed42",
                 model.split("/")[-1].lower(), "truthfulqa",
                 "prism_forgetting_metrics_truthfulqa.json")
try:
    cks = json.load(open(p)).get("checkpoints", [])
    sys.exit(0 if any(c.get("step") == 300 for c in cks) else 1)
except Exception:
    sys.exit(1)
PY
    }
    for cell in $E3C_CELLS; do
        ref_task="${cell%%:*}"; n="${cell##*:}"
        # Per-config output_root: ref_task/reg_samples are NOT part of
        # the trainer's own directory scheme — without this, all six
        # configs overwrite one another (2026-07-24 postmortem).
        root="rebuttal_exp/out/E3/reg_sensitivity/${ref_task}_n${n}"
        if c_done "$root"; then
            echo "=== SKIP (step-300 json exists) trace lam=$LAMBDA ref=$ref_task n=$n ===" | tee -a "$LOG"
            continue
        fi
        echo ">>> trace lam=$LAMBDA ref=$ref_task n=$n ($(date))" | tee -a "$LOG"
        python rebuttal_exp/train_forgetting_baselines.py \
            --model "$MODEL" --task truthfulqa \
            --method trace --lambda_reg "$LAMBDA" --seed 42 --max_steps 300 \
            --ref_task "$ref_task" --reg_samples "$n" \
            --output_root "$root" \
            2>&1 | tee -a "$LOG" || FAIL=1
    done
fi

echo "=== E3 done ($(date)); outputs in $OUT ===" | tee -a "$LOG"
exit $FAIL
