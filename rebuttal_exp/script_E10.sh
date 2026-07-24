#!/usr/bin/env bash
# ============================================================
# E10 — Table-22 failure cells vs PRISM axis diagnosis
#       (8VrD-W3 negative cells)
#
# ZERO GPU: reads the existing forgetting/regularization JSONs from
#   exp_result/regularization/, forgetting_exp_log_safety_clear/,
#   forgetting_exp_log_safety/   (first tree wins per (lambda, cell);
#   runs that stopped before step 300 are excluded automatically).
# Runs in seconds on any machine.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."           # repo root

python3 rebuttal_exp/exp_e10_axis_alignment.py
echo "Outputs: rebuttal_exp/out/E10/{E10_results.md,cells.csv}"
