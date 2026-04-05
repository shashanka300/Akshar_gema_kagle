#!/bin/bash
# Akshar Benchmark — Setup Script (uv)
# Run this on your RTX 5090 machine

set -e

echo "=== Akshar Benchmark Setup ==="

# 1. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "[0/3] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi

# 2. Create venv and install dependencies
echo "[1/3] Creating virtual environment and installing dependencies..."
uv sync

# 3. Install PyTorch with CUDA 12.4
echo "[2/3] Installing PyTorch + CUDA 12.4..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Create directories
echo "[3/3] Creating data directories..."
mkdir -p data/indic_hw results

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo ""
echo "  # Activate environment"
echo "  source .venv/bin/activate"
echo "  # or prefix commands with: uv run"
echo ""
echo "  # Download Hindi dataset from HuggingFace"
echo "  uv run python benchmark_gemma4_indic_hw.py --download"
echo ""
echo "  # For other scripts, download from CVIT:"
echo "  # https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-hw-words"
echo "  # Extract into data/indic_hw/<script_name>/"
echo ""
echo "  # Run benchmark with E4B (fits in 32GB bf16)"
echo "  uv run python benchmark_gemma4_indic_hw.py --run --model google/gemma-4-E4B-it --samples 100"
echo ""
echo "  # Run with 26B MoE (4-bit quantized, ~14GB VRAM)"
echo "  uv run python benchmark_gemma4_indic_hw.py --run --model google/gemma-4-26B-A4B-it --quantize 4bit --samples 50"
echo ""
