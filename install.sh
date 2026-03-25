#!/bin/bash
# Reels Generator Installation Script

echo "=========================================="
echo "🎬 Reels Generator - Installation Script"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment (optional)
read -p "Create virtual environment? (y/n): " CREATE_VENV

if [ "$CREATE_VENV" = "y" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment activated."
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
echo ""

# Core dependencies
pip install streamlit streamlit-extras

# Video processing
pip install opencv-python opencv-contrib-python ffmpeg-python moviepy imageio-ffmpeg

# YouTube download
pip install yt-dlp

# Audio transcription
pip install openai-whisper

# Object detection
pip install ultralytics torch torchvision

# AI integration
pip install openai anthropic

# Utilities
pip install numpy pandas pillow tqdm python-dotenv loguru scipy

# Audio processing
pip install pydub librosa

echo ""
echo "=========================================="
echo "✅ Installation complete!"
echo "=========================================="
echo ""
echo "To run the application:"
echo "  python run.py"
echo ""
echo "Or directly with Streamlit:"
echo "  streamlit run app.py"
echo ""