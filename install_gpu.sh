#!/bin/bash
# GPU Installation Script for Reels Generator - UV Version
# Run this on a machine with NVIDIA GPU

echo "=========================================="
echo "🎮 Reels Generator - GPU Setup Script (UV)"
echo "=========================================="
echo ""

# 1. Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "🚀 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Check for NVIDIA GPU
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠️  nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Check CUDA version for the index URL
CUDA_VERSION="cu124"  # Default to 12.4; adjust to cu118, cu121 etc. if needed

echo "=========================================="
echo "Setting up environment and packages..."
echo "=========================================="
echo ""

# 2. Create and activate virtual environment using uv
# UV creates a .venv folder by default
echo "Creating/Syncing virtual environment..."
uv venv --python 3.14  # Using 3.11 to avoid the imghdr/3.13 issues discussed earlier
sleep 5
source .venv\\bin\\activate.bat

echo ""
echo "1. Installing PyTorch with CUDA $CUDA_VERSION..."
# We use --extra-index-url so uv can still find non-torch packages on PyPI
uv pip install torch torchvision torchaudio --extra-index-url "https://download.pytorch.org/whl/$CUDA_VERSION"

echo ""
echo "2. Installing AI Models..."
uv pip install ultralytics openai-whisper

echo ""
echo "3. Installing Application Dependencies..."
# Bundling these for faster resolution
uv pip install streamlit opencv-python numpy pillow tqdm loguru ffmpeg-python yt-dlp openai anthropic standard-imghdr

echo ""
echo "=========================================="
echo "Verifying GPU Setup..."
echo "=========================================="
echo ""

# Use 'uv run' to ensure we use the correct environment context
uv run python << 'VERIFY'
import torch
import cv2
import sys

print("=" * 50)
print(f"Python Version: {sys.version.split()[0]}")
print("GPU Verification Results")
print("=" * 50)

# PyTorch
print(f"\n📁 PyTorch:")
cuda_ok = torch.cuda.is_available()
print(f"   Version: {torch.__version__}")
print(f"   CUDA available: {cuda_ok}")
if cuda_ok:
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️ CUDA not available - check index URL / drivers")

# Ultralytics
try:
    from ultralytics import YOLO
    print(f"\n📁 Ultralytics YOLO: Installed")
    model = YOLO('yolov8n.pt')
    print(f"   Device: {'cuda' if cuda_ok else 'cpu'}")
except Exception as e:
    print(f"\n📁 Ultralytics YOLO: Error - {e}")

# Whisper
try:
    import whisper
    print(f"\n📁 Whisper: Installed")
    print(f"   Device: {'cuda' if cuda_ok else 'cpu'}")
except Exception as e:
    print(f"\n📁 Whisper: Error - {e}")

# OpenCV
print(f"\n📁 OpenCV:")
print(f"   Version: {cv2.__version__}")

print("\n" + "=" * 50)
VERIFY

echo ""
echo "✅ GPU Setup Complete with UV!"
echo "To run your app, use: uv run streamlit run app.py"