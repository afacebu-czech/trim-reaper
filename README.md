# 🎬 Reels Generator

AI-Powered Viral Content Creator with automatic subject tracking and smart cropping.

## Features

- **📹 Multiple Input Sources**
  - Drag and drop video upload
  - YouTube URL download
  - Support for multiple video formats (MP4, MOV, AVI, MKV, WebM, FLV, WMV)

- **🎯 Smart Subject Detection**
  - YOLO-based object and person detection (GPU accelerated)
  - Face detection with Haar cascades
  - Motion detection for moving subjects
  - Automatic subject tracking across frames
  - Priority tracking for people and faces

- **🎙️ Audio Transcription**
  - OpenAI Whisper integration (GPU accelerated)
  - Multiple model sizes (tiny to large)
  - Word-level timestamps
  - Language auto-detection

- **🔥 Viral Content Detection**
  - AI-powered moment identification
  - Engagement keyword analysis
  - Emotional content scoring
  - Story arc detection

- **🎥 Smart Panning & Cropping**
  - Subject-following camera movement
  - Smooth cinematic panning
  - Multiple smoothness levels
  - Auto-center fallback

- **🤖 Multi-Provider AI Integration**
  - OpenAI GPT-4
  - Anthropic Claude
  - Local rule-based fallback

- **📱 Reel-Optimized Output**
  - 9:16 aspect ratio (1080x1920)
  - Multiple quality presets
  - H.264 encoding
  - Fast start for streaming

- **🎮 GPU Acceleration**
  - CUDA support for PyTorch
  - GPU-accelerated YOLO detection
  - GPU-accelerated Whisper transcription
  - Automatic hardware detection

## Installation

### Option 1: CPU Installation (No GPU Required)

```bash
# Clone or download the project
cd reels-generator

# Run the installation script
chmod +x install.sh
./install.sh
```

### Option 2: GPU Installation (NVIDIA GPU Required)

**Prerequisites:**

- NVIDIA GPU with CUDA support
- NVIDIA drivers installed (check with `nvidia-smi`)
- CUDA Toolkit 11.8+ or 12.x

```bash
# Check your CUDA version
nvidia-smi  # Shows driver and CUDA version

# Run the GPU installation script
chmod +x install_gpu.sh
./install_gpu.sh
```

**Manual GPU Installation:**

```bash
# 1. Install PyTorch with CUDA (adjust CUDA version as needed)
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Install GPU-accelerated packages
pip install ultralytics openai-whisper

# 3. Install other dependencies
pip install streamlit opencv-python numpy pillow tqdm loguru ffmpeg-python yt-dlp openai anthropic
```

### Verify GPU Setup

```bash
# Check GPU configuration
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

### Manual Install (CPU)

```bash
# Install core dependencies
pip install streamlit opencv-python numpy pillow

# Install video processing
pip install ffmpeg-python moviepy

# Install YouTube download
pip install yt-dlp

# Install Whisper for transcription
pip install openai-whisper

# Install YOLO for detection
pip install ultralytics torch

# Install AI providers
pip install openai anthropic
```

### FFmpeg Requirement

FFmpeg is required for video processing. Install it based on your OS:

- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org)

## Usage

### Starting the Application

```bash
# Run with the startup script
python run.py

# Or directly with Streamlit
streamlit run app.py
```

### Using the Interface

1. **Upload Video**
   - Drag and drop a video file, OR
   - Paste a YouTube URL and click "Download"

2. **Configure Settings** (Sidebar)
   - Select AI provider (OpenAI/Anthropic/Local)
   - Set target reel duration
   - Choose output quality
   - Set panning smoothness

3. **Start Analysis**
   - Click "Start Analysis" to process the video
   - Wait for transcription and subject detection

4. **Review Viral Moments**
   - See AI-detected viral segments
   - View scores and reasons
   - Select a segment or create custom

5. **Edit & Export**
   - Fine-tune timing
   - Set panning mode
   - Generate and download your reel

## GPU Performance Tips

### Recommended Settings by GPU Memory

| GPU Memory | Whisper Model | YOLO Model | Batch Size |
| ---------- | ------------- | ---------- | ---------- |
| < 4 GB     | tiny          | yolov8n    | 1          |
| 4-8 GB     | small         | yolov8m    | 2          |
| 8-16 GB    | medium        | yolov8l    | 4          |
| > 16 GB    | large         | yolov8x    | 8          |

### GPU-Specific Configurations

```python
# In the app sidebar, you can configure:
# - Detection Model: yolov8n/s/m/l/x (larger = more accurate but slower)
# - Whisper Model: tiny/base/small/medium/large

# For best GPU performance:
# 1. Use larger models when GPU memory allows
# 2. Reduce sample_rate for more accurate detection
# 3. Enable GPU in Advanced Options
```

## Configuration

### API Keys

Set your API keys in the sidebar or as environment variables:

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

### Quality Presets

| Preset | CRF | Bitrate | Use Case         |
| ------ | --- | ------- | ---------------- |
| Low    | 28  | 2 Mbps  | Quick previews   |
| Medium | 23  | 4 Mbps  | Standard quality |
| High   | 18  | 8 Mbps  | Recommended      |
| Ultra  | 15  | 15 Mbps | Maximum quality  |

### Panning Modes

| Mode      | Smoothing | Use Case            |
| --------- | --------- | ------------------- |
| None      | 0%        | Static center crop  |
| Low       | 20%       | Minimal movement    |
| Medium    | 50%       | Balanced tracking   |
| High      | 80%       | Smooth following    |
| Cinematic | 95%       | Movie-like movement |

## Project Structure

```
reels-generator/
├── app.py                    # Main Streamlit application
├── run.py                    # Startup script
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── install.sh               # Installation script
├── modules/
│   ├── __init__.py
│   ├── video_downloader.py   # YouTube/video download
│   ├── subject_detector.py   # YOLO-based detection
│   ├── transcription.py      # Whisper transcription
│   ├── viral_detector.py     # AI viral analysis
│   ├── video_processor.py    # FFmpeg processing
│   └── ai_integration.py     # Multi-provider AI
├── outputs/                  # Generated reels
├── temp/                     # Temporary files
└── models/                   # Model weights
```

## Technical Details

### Subject Detection

- Uses YOLOv8 for real-time object detection
- 80 COCO classes supported
- Priority tracking for people and faces
- Position smoothing for stable tracking

### Transcription

- OpenAI Whisper models (tiny/base/small/medium/large)
- Automatic language detection
- Word-level timestamps for precise clipping
- Sentiment analysis integration

### Video Processing

- FFmpeg for encoding and format conversion
- OpenCV for frame-level manipulation
- Hardware acceleration when available
- Automatic audio extraction and sync

### Viral Detection

- Keyword-based hook detection
- Engagement pattern analysis
- Emotional content scoring
- AI-powered moment selection

## Troubleshooting

### Common Issues

**FFmpeg not found**

```bash
# Install FFmpeg
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg      # macOS
```

**CUDA out of memory**

- Use smaller Whisper model (tiny/base)
- Use smaller YOLO model (yolov8n)
- Reduce video resolution before processing

**Slow processing**

- Enable GPU acceleration (install CUDA)
- Use lower sample rates
- Choose smaller models

**Import errors**

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## License

This project is for educational purposes. Please respect content creators' rights when downloading and processing videos.

## Acknowledgments

- [Streamlit](https://streamlit.io/) - UI framework
- [OpenAI Whisper](https://github.com/openai/whisper) - Transcription
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection
- [FFmpeg](https://ffmpeg.org/) - Video processing
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Video downloading
