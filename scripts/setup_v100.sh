#!/bin/bash
# AKA-Local V100 Evaluator Setup Script
# Run on the V100 server: bash setup_v100.sh
set -e

echo "=== AKA-Local V100 Setup ==="
echo ""

# ---- Check CUDA ----
echo "[1/6] Checking CUDA..."
if [ -d "/usr/local/cuda-11.8" ]; then
    echo "  PASS: CUDA 11.8 found"
    CUDA_HOME="/usr/local/cuda-11.8"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_VER=$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',')
    echo "  PASS: CUDA $CUDA_VER found at /usr/local/cuda"
    CUDA_HOME="/usr/local/cuda"
else
    echo "  FAIL: CUDA not found at /usr/local/cuda-11.8 or /usr/local/cuda"
    echo "  Install CUDA 11.8 from: https://developer.nvidia.com/cuda-11-8-0-download-archive"
    exit 1
fi

# ---- Check nvcc ----
echo "[2/6] Checking nvcc..."
NVCC="$CUDA_HOME/bin/nvcc"
if [ -x "$NVCC" ]; then
    $NVCC --version | head -1
    echo "  PASS: nvcc found"
else
    echo "  FAIL: nvcc not found at $NVCC"
    exit 1
fi

# ---- Check Python ----
echo "[3/6] Checking Python..."
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "  FAIL: Python not found"
    exit 1
fi
$PYTHON --version
echo "  PASS: Python found"

# ---- Check PyTorch (optional) ----
echo "[4/6] Checking PyTorch..."
if $PYTHON -c "import torch; print(torch.__version__)" 2>/dev/null; then
    echo "  PASS: PyTorch available"
else
    echo "  WARN: PyTorch not installed (optional, needed for correctness checks)"
    echo "  Install: pip install torch"
fi

# ---- Create evaluator directory ----
echo "[5/6] Creating evaluator directory..."
EVAL_DIR="$HOME/cuda_kernel_experiments/evaluator"
mkdir -p "$EVAL_DIR"
echo "  Created: $EVAL_DIR"

# ---- Check for evaluator files ----
echo "[6/6] Checking evaluator files..."
if [ -f "$EVAL_DIR/evaluate.py" ] && [ -f "$EVAL_DIR/eval.sh" ]; then
    echo "  PASS: evaluate.py and eval.sh already present"
else
    echo "  WARN: Evaluator files missing from $EVAL_DIR"
    echo ""
    echo "  You need to copy evaluate.py and eval.sh to this directory."
    echo "  These files are in the atrex-kernel-agent-win/tools/ directory"
    echo "  of the atrex-bench repository."
    echo ""
    echo "  Example:"
    echo "    scp atrex-kernel-agent-win/tools/evaluate.py user@v100:$EVAL_DIR/"
    echo "    scp atrex-kernel-agent-win/tools/eval.sh user@v100:$EVAL_DIR/"
    echo "    ssh user@v100 chmod +x $EVAL_DIR/eval.sh"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Ensure evaluate.py and eval.sh are in $EVAL_DIR/"
echo "  2. From your local machine, run: python -m lab.cli doctor --env v100"
echo ""
