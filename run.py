#!/usr/bin/env python3
"""
Reels Generator - Run Script
Start the Streamlit application
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    
    required = [
        'streamlit',
        'opencv-python',
        'numpy',
        'pillow'
    ]
    
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True

def check_optional_dependencies():
    """Check optional dependencies and warn"""
    
    optional = {
        'whisper': 'Audio transcription (pip install openai-whisper)',
        'ultralytics': 'Subject detection (pip install ultralytics)',
        'yt_dlp': 'YouTube download (pip install yt-dlp)',
        'openai': 'OpenAI integration (pip install openai)',
        'anthropic': 'Anthropic integration (pip install anthropic)',
        'torch': 'GPU acceleration (pip install torch)'
    }
    
    missing = []
    
    for package, description in optional.items():
        try:
            __import__(package)
        except ImportError:
            missing.append((package, description))
    
    if missing:
        print("\nOptional dependencies not installed:")
        for package, description in missing:
            print(f"  - {package}: {description}")
        print("\nSome features may be limited.")

def main():
    """Main entry point"""
    
    print("=" * 50)
    print("🎬 Reels Generator")
    print("   AI-Powered Viral Content Creator")
    print("=" * 50)
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    check_optional_dependencies()
    
    print("\nStarting Streamlit server...")
    print("Press Ctrl+C to stop")
    print()
    
    # Get the app path
    app_path = Path(__file__).parent / "app.py"
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            str(app_path),
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error starting Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()