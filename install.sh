#!/usr/bin/env bash
# ============================================================
# PRISM — environment installer
#
# Installs all dependencies in the correct order:
#   1. torch with CUDA 12.8 wheel  (must come first)
#   2. requirements.txt            (everything else)
#   3. GPTQModel with              (requires torch at build time)
#      --no-build-isolation
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
    echo "[1/3] Installing torch==2.10.0 (CUDA 12.8) ..."
    pip install \
        torch==2.10.0 \
        torchaudio==2.10.0 \
        torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128
    echo "      torch installed."
else
    echo "[1/3] Skipping torch (--skip-torch)."
fi

# Verify torch is importable before proceeding to GPTQModel
python -c "import torch; print('      torch version:', torch.__version__)" || {
    echo "ERROR: torch not found after install. Aborting."
    exit 1
}

# ── Step 2: requirements.txt ─────────────────────────────────
echo ""
echo "[2/3] Installing requirements.txt ..."
pip install -r requirements.txt
echo "      requirements.txt installed."

# ── Step 3: GPTQModel ─────────────────────────────────────────
echo ""
echo "[3/3] Installing GPTQModel==5.7.0 (--no-build-isolation) ..."
# --no-build-isolation: lets GPTQModel's setup.py see the already-installed
# torch, so it can detect the torch version without a fresh isolated env.
pip install GPTQModel==5.7.0 --no-build-isolation
echo "      GPTQModel installed."

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Installation complete."
echo "  Verify with: python -c \"import GPTQModel; print(GPTQModel.__version__)\""
echo "============================================"
