#!/usr/bin/env bash
# ============================================================
# PRISM — environment installer
#
# Installs all dependencies in the correct order:
#   1. torch with CUDA 12.8 wheel
#   2. setuptools >= 71
#   3. requirements.txt
#   4. GPTQModel (with --no-build-isolation)
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
echo ""
echo "[2/4] Upgrading setuptools >= 71 ..."
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
pip install -v GPTQModel==5.7.0 --no-build-isolation
echo "      GPTQModel installed."

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Installation complete."
echo "  Verify with: python -c \"import GPTQModel; print(GPTQModel.__version__)\""
echo "============================================"
