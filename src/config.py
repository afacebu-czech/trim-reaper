"""
Configuration settings for Reels Generator
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp"
MODELS_DIR = BASE_DIR / "models"

# Create directories
for dir_path in [OUTPUT_DIR, TEMP_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Video Settings
REEL_WIDTH = 1080
REEL_HEIGHT = 1920
REEL_FPS = 30
MAX_REEL_DURATION = 180  # seconds
MIN_REEL_DURATION = 15  # seconds

# Default Settings
DEFAULT_QUALITY = "high"
DEFAULT_PANNING = "medium"
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_DETECTION_MODEL = "yolov8s"

# Processing Settings
SAMPLE_RATE_DETECTION = 5  # Process every Nth frame for detection
SAMPLE_RATE_MOTION = 10  # Process every Nth frame for motion analysis

# Quality Presets
QUALITY_PRESETS = {
    'low': {
        'crf': 28,
        'preset': 'ultrafast',
        'bitrate': '2M',
        'resolution': (720, 1280)
    },
    'medium': {
        'crf': 23,
        'preset': 'fast',
        'bitrate': '4M',
        'resolution': (1080, 1920)
    },
    'high': {
        'crf': 18,
        'preset': 'medium',
        'bitrate': '8M',
        'resolution': (1080, 1920)
    },
    'ultra': {
        'crf': 15,
        'preset': 'slow',
        'bitrate': '15M',
        'resolution': (1080, 1920)
    }
}

# AI Provider Settings
AI_PROVIDERS = {
    'openai': {
        'name': 'OpenAI (GPT-4)',
        'models': ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
        'default_model': 'gpt-4'
    },
    'anthropic': {
        'name': 'Anthropic (Claude)',
        'models': ['claude-3-opus-20240229', 'claude-3-sonnet-20240229'],
        'default_model': 'claude-3-sonnet-20240229'
    },
    'zai': {
        'name': 'ZAI (GLM-4.7-Flash)',
        'models': ['GLM-4.7-Flash'],
        'default_model': 'GLM-4.7-Flash'
    },
    'local': {
        'name': 'Local (Rule-based)',
        'models': ['rule-based'],
        'default_model': 'rule-based'
    }
}

# Detection Classes (COCO)
PRIORITY_SUBJECTS = [
    'person', 'face', 'cell phone', 'laptop', 'tv', 
    'car', 'dog', 'cat', 'bird', 'sports ball'
]

# Viral Keywords
VIRAL_KEYWORDS = {
    'hooks': [
        'wait for it', 'watch until', 'you won\'t believe',
        'plot twist', 'here\'s the thing', 'let me tell you'
    ],
    'ctas': [
        'follow for more', 'like and subscribe', 'share this',
        'comment below', 'duet this', 'stitch this'
    ],
    'emphasis': [
        'literally', 'actually', 'seriously', 'honestly',
        'obviously', 'absolutely', 'definitely'
    ]
}

# Supported video formats
SUPPORTED_FORMATS = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv']

# API Keys (from environment)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
ZAI_API_KEY = os.environ.get('ZAI_API_KEY', '')