#!/usr/bin/env bash
# ============================================================
# PRISM — environment installer
#
# Installs all dependencies in the correct order:
#   1. torch with CUDA 12.8 wheel  (must come first)
#   2. setuptools >= 71            (GPTQModel pyproject.toml fix)
#   3. requirements.txt            (everything else)
#   4. GPTQModel with              (requires torch at build time)
#      --no-build-isolation
#
# Background — two GPTQModel install issues:
#   a) Without --no-build-isolation:
#      pip creates an isolated build env that has no torch, so
#      GPTQModel's setup.py fails with "Unable to detect torch version".
#   b) With --no-build-isolation + setuptools < 71:
#      setuptools rejects pyproject.toml's `license = "Apache-2.0"`
#      (SPDX string, PEP 639) — only accepted by setuptools >= 71.0.
#   Fix: upgrade setuptools first, then use --no-build-isolation.
#
# Usage:
#   bash install.sh
#   bash install.sh --skip-torch   # if torch already installed
# ============================================================
set -euo pipefail

SKIP_TORCH=false
for arg in "$@"; do
    [[ "$arg" == "--skip-torch" ]] && SKIP_TORCH=true
done

echo "============================================"
echo "  PRISM environment installer"
echo "============================================"

# ── Step 1: torch ────────────────────────────────────────────
if [[ "$SKIP_TORCH" == false ]]; then
    echo ""
    echo "[1/4] Installing torch==2.10.0 (CUDA 12.8) ..."
    pip install \
        torch==2.10.0 \
        torchaudio==2.10.0 \
        torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128
    echo "      torch installed."
else
    echo "[1/4] Skipping torch (--skip-torch)."
fi

# Verify torch is importable before proceeding to GPTQModel
python -c "import torch; print('      torch version:', torch.__version__)" || {
    echo "ERROR: torch not found after install. Aborting."
    exit 1
}

# ── Step 2: setuptools >= 71 ──────────────────────────────────
# GPTQModel 5.7.0 pyproject.toml uses `license = "Apache-2.0"` (PEP 639
# SPDX string). setuptools < 71.0 validates against the PEP 621 schema
# which only allows {file: ...} or {text: ...}, rejecting the SPDX string.
# setuptools >= 71.0 (2024-07-04) added full PEP 639 support.
echo ""
echo "[2/4] Upgrading setuptools >= 71 (required by GPTQModel pyproject.toml) ..."
pip install "setuptools>=71"
python -c "import setuptools; print('      setuptools version:', setuptools.__version__)"

# ── Step 3: requirements.txt ─────────────────────────────────
echo ""
echo "[3/4] Installing requirements.txt ..."
pip install -r requirements.txt
echo "      requirements.txt installed."

# ── Step 4: GPTQModel ─────────────────────────────────────────
echo ""
echo "[4/4] Installing GPTQModel==5.7.0 (--no-build-isolation) ..."
# --no-build-isolation: skips pip's isolated build sandbox so that
#   (a) GPTQModel's setup.py can detect the pre-installed torch version
#   (b) the upgraded setuptools (>= 71) is used for pyproject.toml validation
pip install -v GPTQModel==5.7.0 --no-build-isolation
echo "      GPTQModel installed."

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Installation complete."
echo "  Verify with: python -c \"import GPTQModel; print(GPTQModel.__version__)\""
echo "============================================"
