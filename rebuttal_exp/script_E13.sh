#!/usr/bin/env bash
# ============================================================
# E13 — Full 20-cell correlation matrix, paper 2x2x5 grid
#       (8VrD-W4(1), AC-E: all-cell aggregates, not Llama-only)
#
# ZERO GPU: reads the lambda=0 forgetting JSONs from
#   exp_result/regularization/0.0/ (fallback: forgetting_exp_log trees).
# Includes a protocol-scan checksum against the paper's published
# numbers (Llama mean 0.831; Qwen-BBQ -0.34/-0.66) — see E13.md.
# Runs in seconds on any machine.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."           # repo root

python3 rebuttal_exp/exp_e13_fullmatrix.py
echo "Outputs: rebuttal_exp/out/E13/{E13_results.md,matrix.csv}"
