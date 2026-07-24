#!/usr/bin/env bash
# ============================================================
# E6 — Slack quantification + LOO isotonic calibration
#      (pCi8-W2, G3T9-W4, 8VrD-W2/Q1)
#
# ZERO GPU: pure analysis of the existing paper CSV
# (exp_result/quantization/quantization_merged_slim.csv).
# Runs in seconds on any machine.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."           # repo root

python3 rebuttal_exp/exp_e6_slack_calibration.py
echo "Outputs: rebuttal_exp/out/E6/{E6_results.md,slack_summary.csv,calibration.csv}"
