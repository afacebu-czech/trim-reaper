"""
Reels Generator - AI-Powered Viral Content Creator
Creates engaging reels from videos with automatic subject tracking and smart cropping
"""

import streamlit as st
import os
import sys
import time
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Reels Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #4a5568;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e7f3ff;
        border: 1px solid #b8daff;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Import modules with error handling
@st.cache_resource
def load_modules():
    """Load all required modules"""
    modules = {}
    try:
        from src.modules.video_downloader import VideoDownloader
        modules['downloader'] = VideoDownloader
    except ImportError as e:
        st.warning(f"Video downloader module not fully loaded: {e}")
        modules['downloader'] = None
    
    try:
        from src.modules.subject_detector import SubjectDetector
        modules['detector'] = SubjectDetector
    except ImportError as e:
        st.warning(f"Subject detector module not fully loaded: {e}")
        modules['detector'] = None
    
    try:
        from src.modules.transcription import TranscriptionEngine
        modules['transcription'] = TranscriptionEngine
    except ImportError as e:
        st.warning(f"Transcription module not fully loaded: {e}")
        modules['transcription'] = None
    
    try:
        from src.modules.viral_detector import ViralContentDetector
        modules['viral'] = ViralContentDetector
    except ImportError as e:
        st.warning(f"Viral detector module not fully loaded: {e}")
        modules['viral'] = None
    
    try:
        from src.modules.video_processor import VideoProcessor
        modules['processor'] = VideoProcessor
    except ImportError as e:
        st.warning(f"Video processor module not fully loaded: {e}")
        modules['processor'] = None
    
    try:
        from src.modules.ai_integration import AIIntegration
        modules['ai'] = AIIntegration
    except ImportError as e:
        st.warning(f"AI integration module not fully loaded: {e}")
        modules['ai'] = None
    
    try:
        from src.modules.speaker_tracker import SpeakerTracker
        modules['speaker_tracker'] = SpeakerTracker
    except ImportError as e:
        st.warning(f"Speaker tracker module not fully loaded: {e}")
        modules['speaker_tracker'] = None
    
    return modules

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    default_states = {
        'video_path': None,
        'video_info': {},
        'transcription': None,
        'viral_segments': [],
        'detected_subjects': [],
        'processing_stage': 'idle',
        'output_path': None,
        'analysis_data': {},
        'ai_provider': 'openai',
        'reel_duration': 60,
        'output_quality': 'high',
        'panning_smoothness': 'medium',
        'selected_segment': None,
        'preview_frames': [],
        'processing_log': [],
        # Completion flags (persist across tabs)
        'analysis_completed': False,
        'viral_completed': False,
        'export_completed': False,
        # Speaker tracking data
        'speaker_tracking': {},
        'speaker_timeline': [],
        'focus_points': [],
        'is_multi_speaker': False,
        # Track last video to detect changes
        'last_video_path': None,
        'last_video_hash': None,
        # Track downloaded quality
        'downloaded_quality': None,
    }
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

def clear_all_session_data():
    """Clear all session data when video changes or is removed"""
    keys_to_clear = [
        'video_info', 'transcription', 'viral_segments',
        'detected_subjects', 'processing_stage', 'output_path', 'analysis_data',
        'selected_segment', 'preview_frames', 'processing_log',
        'analysis_completed', 'viral_completed', 'export_completed',
        'speaker_tracking', 'speaker_timeline', 'focus_points', 'is_multi_speaker',
        'downloaded_quality'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            if key == 'processing_stage':
                st.session_state[key] = 'idle'
            elif key in ['video_info', 'analysis_data', 'speaker_tracking']:
                st.session_state[key] = {}
            elif key in ['viral_segments', 'preview_frames', 'processing_log', 'speaker_timeline', 'focus_points']:
                st.session_state[key] = []
            elif key in ['analysis_completed', 'viral_completed', 'export_completed', 'is_multi_speaker']:
                st.session_state[key] = False
            else:
                st.session_state[key] = None

def check_video_change():
    """Check if video has changed and clear data if so"""
    current_video = st.session_state.get('video_path')
    last_video = st.session_state.get('last_video_path')
    
    # Check if video was removed (cleared)
    if last_video is not None and current_video is None:
        clear_all_session_data()
        st.session_state.last_video_path = None
        return
    
    # Check if video changed
    if current_video != last_video:
        if last_video is not None:
            # Video has changed, clear all data
            clear_all_session_data()
        st.session_state.last_video_path = current_video

init_session_state()
check_video_change()

# Sidebar Configuration
def render_sidebar():
    """Render the sidebar with configuration options"""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # AI Provider Selection
        st.markdown("#### 🤖 AI Provider")
        ai_provider = st.selectbox(
            "Select AI Provider",
            options=["openai", "anthropic", "zai", "local"],
            format_func=lambda x: {
                "openai": "OpenAI (GPT-4)",
                "anthropic": "Anthropic (Claude)",
                "zai": "ZAI (GLM-4.7-Flash)",
                "local": "Local (Offline)"
            }[x],
            index=2,
            key="ai_provider_select"
        )
        st.session_state.ai_provider = ai_provider
        
        # API Key input based on provider
        if ai_provider == "openai":
            api_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
            os.environ["OPENAI_API_KEY"] = api_key
        elif ai_provider == "anthropic":
            api_key = st.text_input("Anthropic API Key", type="password", key="anthropic_key")
        
        st.markdown("---")
        
        # Reel Settings
        st.markdown("#### 📱 Reel Settings")
        
        reel_duration = st.slider(
            "Target Duration (seconds)",
            min_value=15,
            max_value=180,
            value=st.session_state.reel_duration,
            step=5,
            key="duration_slider"
        )
        st.session_state.reel_duration = reel_duration
        
        output_quality = st.select_slider(
            "Output Quality",
            options=["low", "medium", "high", "ultra"],
            value=st.session_state.output_quality,
            key="quality_slider"
        )
        st.session_state.output_quality = output_quality
        
        panning_smoothness = st.select_slider(
            "Panning Smoothness",
            options=["none", "low", "medium", "high", "cinematic"],
            value=st.session_state.panning_smoothness,
            key="panning_slider"
        )
        st.session_state.panning_smoothness = panning_smoothness
        
        st.markdown("---")
        
        # Advanced Options
        with st.expander("🔧 Advanced Options"):
            st.checkbox("Enable face tracking", value=True, key="face_tracking")
            st.checkbox("Enable motion detection", value=True, key="motion_detection")
            st.checkbox("Preserve original audio", value=True, key="preserve_audio")
            st.checkbox("Add auto-captions", value=False, key="auto_captions")
            st.checkbox("Apply color correction", value=True, key="color_correction")
            st.checkbox("Enable speaker tracking", value=True, key="speaker_tracking_enabled")
            
            whisper_model = st.selectbox(
                "Whisper Model",
                options=["tiny", "base", "small", "medium", "large"],
                index=2,
                key="whisper_model"
            )
            
            detection_model = st.selectbox(
                "Detection Model",
                options=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                index=0,
                key="detection_model",
                help="For performance: yolov8n | For quality: yolov8s"
            )
        
        st.markdown("---")
        
        # System Status
        st.markdown("#### 📊 System Status")
        
        # Check dependencies
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            try:
                import cv2
                st.success("OpenCV ✓")
            except:
                st.error("OpenCV ✗")
            
            try:
                import whisper
                st.success("Whisper ✓")
            except:
                st.error("Whisper ✗")
        
        with status_col2:
            try:
                import torch
                st.success("PyTorch ✓")
            except:
                st.error("PyTorch ✗")
            
            try:
                import ffmpeg
                st.success("FFmpeg ✓")
            except:
                st.error("FFmpeg ✗")
        
        # GPU Status
        try:
            import torch
            if torch.cuda.is_available():
                st.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            else:
                st.info("💻 Running on CPU")
        except:
            pass
        
        # Progress indicators
        if st.session_state.video_path:
            st.markdown("---")
            st.markdown("#### 📈 Progress")
            
            if st.session_state.analysis_completed:
                st.success("✅ Analysis Complete")
            elif st.session_state.processing_stage == 'analyzing':
                st.info("🔄 Analyzing...")
            else:
                st.warning("⏳ Analysis Pending")
            
            if st.session_state.viral_completed:
                st.success("✅ Viral Detection Complete")
            elif st.session_state.processing_stage == 'viral_detection':
                st.info("🔄 Detecting...")
            else:
                st.warning("⏳ Viral Detection Pending")

# Main Content Area
def render_main():
    """Render the main content area"""
    st.markdown('<h1 class="main-header">🎬 Reels Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #718096; font-size: 1.1rem;">AI-Powered Viral Content Creator with Smart Subject Tracking</p>', unsafe_allow_html=True)
    
    # Load modules
    modules = load_modules()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📥 Input", 
        "🔍 Analysis", 
        "🎯 Viral Detection", 
        "✂️ Editor", 
        "📤 Export"
    ])
    
    # Tab 1: Input
    with tab1:
        render_input_tab(modules)
    
    # Tab 2: Analysis
    with tab2:
        render_analysis_tab(modules)
    
    # Tab 3: Viral Detection
    with tab3:
        render_viral_tab(modules)
    
    # Tab 4: Editor
    with tab4:
        render_editor_tab(modules)
    
    # Tab 5: Export
    with tab5:
        render_export_tab(modules)

def render_input_tab(modules):
    """Render the input tab with file upload and YouTube options"""
    st.markdown("### 📥 Video Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📁 Upload Video")
        st.markdown("Drag and drop or click to upload a video file")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mov", "avi", "mkv", "webm", "flv", "wmv"],
            help="Supported formats: MP4, MOV, AVI, MKV, WebM, FLV, WMV"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            temp_dir = Path(tempfile.gettempdir()) / "reels_generator"
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / uploaded_file.name
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state.video_path = str(temp_path)
            st.success(f"✅ Video uploaded: {uploaded_file.name}")
            
            # Show video info
            file_size = uploaded_file.size / (1024 * 1024)
            st.info(f"📊 File size: {file_size:.2f} MB")
    
    with col2:
        st.markdown("#### 🎬 YouTube Download")
        st.markdown("Paste a YouTube URL to download the video")
        
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key="youtube_url_input"
        )
        
        # Quality selection options - initialize from session state if available
        youtube_quality = st.session_state.get('youtube_quality_select', "1080p (Full HD)")
        available_qualities = []
        video_info_yt = {}
        
        # Track if quality options are loaded
        quality_ready = False 
        
        # Check URL and get available qualities
        if youtube_url and modules['downloader']:
            try:
                downloader = modules['downloader']()
                
                # Check if it's a valid Youtube URL
                platform = downloader.detect_platform(youtube_url)

                if platform == 'youtube':
                    # Get video info
                    with st.spinner("Fetching video info..."):
                        video_info_yt = downloader.get_video_info(youtube_url)
                        
                    if video_info_yt and 'error' not in video_info_yt:
                        # Show video info preview
                        st.markdown(f"**📹 {video_info_yt.get('title', 'Unknown')[:60]}...**")
                        
                        info_cols = st.columns(3)
                        with info_cols[0]:
                            st.metric("Duration", video_info_yt.get('duration_formatted', 'N/A'))
                        with info_cols[1]:
                            st.metric("Max Res", f"{video_info_yt.get('max_resolution', 'N/A')}p")
                        with info_cols[2]:
                            st.metric("Views", f"{video_info_yt.get('view_count', 0):,}")
                            
                        # Get available qualities
                        available_qualities = downloader.get_available_qualities(youtube_url)
                        
                        # Quality selector
                        quality_options = [q['name'] for q in available_qualities if q['available']]
                        
                        if quality_options:
                            # Get current selection from session state or default
                            current_selection = st.session_state.get('youtube_quality_select', "1080p (Full HD)")
                            
                            # Find index for current selection or default to 0
                            default_index = quality_options.index(current_selection) if current_selection in quality_options else (
                                quality_options.index("1080p (Full HD)") if "1080p (Full HD)" in quality_options else 0
                            )
                            
                            youtube_quality = st.selectbox(
                                "🎥 Select Quality",
                                options=quality_options,
                                index=default_index,
                                help="Higher quality = larger file size",
                                key="youtube_quality_select"
                            )
                            
                            # Quality options are now ready
                            quality_ready = True
                            
                            # Show quality description
                            for q in available_qualities:
                                if q['name'] == youtube_quality:
                                    size_info = f" (~{q['estimated_size_mb']}MB)" if q.get('estimated_size_mb') else ""
                                    st.caption(f"ℹ️ {q['description']}{size_info}")
                                    break
                        else:
                            st.warning("Could not fetch available qualities. Will use best available.")
                            
                            # Allow download with fallback
                            quality_ready = True
                    
                    else:
                        error_msg = video_info_yt.get('error', 'Unknown error')
                        st.error(f"Could not fetch video info: {error_msg}")
                
                else:
                    error_msg = video_info_yt.get('error', 'Unknown error')
                    st.error(f"Could not fetch video info: {error_msg}")
                    
            except Exception as e:
                st.warning(f"Could not validate URL: {str(e)}")
                
        else:
            # No Url entered yet
            if not youtube_url:
                st.info("Paste a YouTube URL to see available quality options")
                
        # Download button - disabled until quality options are loaded
        download_disabled = not quality_ready
        download_help = "Select quality options above before downloading" if download_disabled else "Click to download the video"
                
        if st.button("⬇️ Download from YouTube", use_container_width=True, type="primary", disabled=download_disabled, help=download_help):
            if youtube_url and modules['downloader']:
                # Get selected quality from session state (persists across reruns)
                selected_quality = st.session_state.get('youtube_quality_select', "1080p (Full HD)")
                
                # Delete old video file if exists (for re-downloading with different quality)
                old_video_path = st.session_state.get('video_path')
                if old_video_path and os.path.exists(old_video_path):
                    try:
                        os.remove(old_video_path)
                        st.info(f"🗑️ Removed previous video file")
                    except Exception as e:
                        st.warning(f"Could not remove old file: {e}")
                
                # Clear all analysis data before downloading new video
                clear_all_session_data()
                
                with st.spinner(f"Downloading at {selected_quality}..."):
                    try:
                        downloader = modules['downloader']()
                        selected_quality = st.session_state.get('youtube_quality_select', "1080p (Full HD)")
                        video_path = downloader.download(youtube_url, quality=selected_quality)
                        if video_path:
                            st.session_state.video_path = video_path
                            st.session_state.downloaded_quality = selected_quality
                            st.success(f"✅ Video downloaded successfully at {selected_quality}!")
                            
                            # Show file info
                            if os.path.exists(video_path):
                                file_size = os.path.getsize(video_path) / (1024 * 1024)
                                st.info(f"📊 Downloaded: {file_size:.2f} MB | Quality: {selected_quality}")
                                
                        else:
                            st.error("Failed to download video. Please check the URL.")
                    except Exception as e:
                        st.error(f"Error downloading video: {str(e)}")
            elif not youtube_url:
                st.warning("Please enter a YouTube URL")
            else:
                st.error("Downloader module not available. Please install required dependencies.")
        
        # Quick format options
        st.markdown("---")
        st.markdown("#### 📋 Quick Options")
        
        if st.button("🔄 Clear Input", use_container_width=True):
            # Delete video file from disk if exists
            video_path = st.session_state.get('video_path')
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception:
                    pass  # Silently fail if file can't be deleted
            
            # Clear all session data and reset to initial state
            clear_all_session_data()
            st.session_state.video_path = None
            st.session_state.last_video_path = None
            st.rerun()
    
    # Video Preview
    if st.session_state.video_path:
        st.markdown("---")
        st.markdown("### 📺 Video Preview")
        
        # Show downloaded quality if available
        if st.session_state.get("downloaded_quality"):
            st.caption(f"Downloaded Quality: **{st.session_state.downloaded_quality}**")
        
        video_col, info_col = st.columns([2, 1])
        
        with video_col:
            st.video(st.session_state.video_path)
        
        with info_col:
            st.markdown("#### 📊 Video Information")
            
            # Get video info if processor is available
            if modules['processor']:
                try:
                    processor = modules['processor']()
                    video_info = processor.get_video_info(st.session_state.video_path)
                    st.session_state.video_info = video_info
                    
                    info_items = [
                        ("Resolution", f"{video_info.get('width', 'N/A')}x{video_info.get('height', 'N/A')}"),
                        ("Duration", f"{video_info.get('duration', 0):.2f}s"),
                        ("FPS", f"{video_info.get('fps', 0):.2f}"),
                        ("Codec", video_info.get('codec', 'N/A')),
                        ("Bitrate", f"{video_info.get('bitrate', 0) / 1000:.0f} kbps"),
                        ("Audio", video_info.get('audio_codec', 'N/A'))
                    ]
                    
                    for label, value in info_items:
                        st.markdown(f"**{label}:** {value}")
                        
                except Exception as e:
                    st.warning(f"Could not retrieve video info: {str(e)}")
            
            # Show processing options
            st.markdown("---")
            st.markdown("#### 🚀 Next Steps")
            
            if st.button("▶️ Start Analysis", use_container_width=True, type="primary"):
                st.session_state.processing_stage = 'analyzing'
                st.rerun()

def render_analysis_tab(modules):
    """Render the analysis tab with transcription and subject detection"""
    st.markdown("### 🔍 Video Analysis")
    
    if not st.session_state.video_path:
        st.warning("⚠️ Please upload a video first in the Input tab")
        return
    
    # Run analysis if not completed and not in progress
    if not st.session_state.analysis_completed and st.session_state.processing_stage != 'analyzing':
        st.warning("⚠️ Please click the 'Start Analysis' button in Input tab to start the analysis.")
        return
    
    # Analysis Progress
    if st.session_state.processing_stage == 'analyzing' and not st.session_state.analysis_completed:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Transcription
        status_text.markdown("🎙️ **Step 1/4:** Transcribing audio with Whisper...")
        progress_bar.progress(5)
        
        if modules['transcription']:
            try:
                transcriber = modules['transcription'](model_size=st.session_state.whisper_model)
                transcription = transcriber.transcribe(st.session_state.video_path)
                st.session_state.transcription = transcription
                progress_bar.progress(25)
                status_text.success("✅ Transcription completed!")
            except Exception as e:
                st.error(f"Transcription error: {str(e)}")
                progress_bar.progress(25)
        else:
            progress_bar.progress(25)
            st.warning("Transcription module not available")
        
        # Step 2: Subject Detection
        status_text.markdown("🎯 **Step 2/4:** Detecting subjects (faces, bodies, motion)...")
        progress_bar.progress(30)
        
        if modules['detector']:
            try:
                detector = modules['detector'](model_name=st.session_state.detection_model)
                subjects = detector.detect_subjects(st.session_state.video_path, sample_rate=5)
                st.session_state.detected_subjects = subjects
                progress_bar.progress(55)
                
                # Show detection summary
                detection_count = len(subjects.get('detections', []))
                class_counts = subjects.get('class_counts', {})
                if class_counts:
                    status_text.success(f"✅ Subject detection completed! Found {detection_count} detections ({len(class_counts)} types)")
                else:
                    status_text.success("✅ Subject detection completed! Motion analysis done.")
            except Exception as e:
                import traceback
                st.error(f"Detection error: {str(e)}")
                st.code(traceback.format_exc())
                progress_bar.progress(55)
        else:
            progress_bar.progress(55)
            st.warning("Detector module not available")
        
        # Step 3: Multi-Speaker Tracking (for interviews/talks)
        status_text.markdown("👥 **Step 3/4:** Analyzing speakers for interview/talk scenarios...")
        progress_bar.progress(60)
        
        try:
            if modules['speaker_tracker'] and st.session_state.get('speaker_tracking_enabled', True):
                from src.modules.speaker_tracker import SpeakerTracker
                speaker_tracker = SpeakerTracker(use_gpu=True)
                tracking_results = speaker_tracker.track_subjects(
                    st.session_state.video_path,
                    st.session_state.detected_subjects,
                    st.session_state.transcription,
                    sample_rate=5
                )
                st.session_state.speaker_tracking = tracking_results
                st.session_state.speaker_timeline = tracking_results.get('speaker_timeline', [])
                st.session_state.focus_points = tracking_results.get('focus_points', [])
                st.session_state.is_multi_speaker = tracking_results.get('is_multi_speaker', False)
                progress_bar.progress(75)
                status_text.success(f"✅ Speaker tracking completed! {tracking_results.get('subject_count', 0)} subjects tracked")
            else:
                progress_bar.progress(75)
        except Exception as e:
            st.warning(f"Speaker tracking error: {str(e)}")
            logger.warning(f"Speaker tracking error: {str(e)}")
            progress_bar.progress(75)
        
        # Step 4: Motion Analysis
        status_text.markdown("🎥 **Step 4/4:** Analyzing motion patterns...")
        progress_bar.progress(80)
        
        try:
            if modules['processor']:
                processor = modules['processor']()
                motion_data = processor.analyze_motion(st.session_state.video_path)
                st.session_state.analysis_data['motion'] = motion_data
        except Exception as e:
            st.warning(f"Motion analysis error: {str(e)}")
        
        progress_bar.progress(100)
        status_text.success("✅ Analysis completed!")
        st.session_state.processing_stage = 'analyzed'
        st.session_state.analysis_completed = True
        time.sleep(1)
        st.rerun()
    
    # Display Analysis Results - show whenever analysis is completed
    if st.session_state.analysis_completed:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎙️ Transcription Results")
            
            if st.session_state.transcription:
                transcription = st.session_state.transcription
                
                # Summary metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Duration", f"{transcription.get('duration', 0):.1f}s")
                with metric_col2:
                    st.metric("Segments", len(transcription.get('segments', [])))
                with metric_col3:
                    st.metric("Language", transcription.get('language', 'N/A').upper())
                
                # Full transcript with timestamps
                st.markdown("##### 📝 Full Transcript")
                
                with st.expander("View Full Transcript", expanded=True):
                    for segment in transcription.get('segments', [])[:50]:  # Show first 50
                        start = segment.get('start', 0)
                        end = segment.get('end', 0)
                        text = segment.get('text', '')
                        st.markdown(f"[{start:.1f}s - {end:.1f}s] {text}")
                    
                    if len(transcription.get('segments', [])) > 50:
                        st.info(f"... and {len(transcription.get('segments', [])) - 50} more segments")
            else:
                st.info("No transcription available")
        
        with col2:
            st.markdown("#### 🎯 Subject Detection")
            
            if st.session_state.detected_subjects:
                subjects = st.session_state.detected_subjects
                
                # Detection methods used
                detection_methods = subjects.get('detection_methods', {})
                if detection_methods:
                    st.markdown("##### 🔧 Detection Methods Used")
                    method_status = []
                    if detection_methods.get('yolo'):
                        method_status.append("✅ YOLO")
                    else:
                        method_status.append("⚠️ YOLO (unavailable)")
                    
                    if detection_methods.get('opencv_faces'):
                        method_status.append("✅ Face Detection")
                    else:
                        method_status.append("❌ Face Detection")
                    
                    if detection_methods.get('opencv_body'):
                        method_status.append("✅ Body Detection")
                    else:
                        method_status.append("❌ Body Detection")
                    
                    if detection_methods.get('motion'):
                        method_status.append("✅ Motion Detection")
                    else:
                        method_status.append("❌ Motion Detection")
                    
                    st.markdown(" | ".join(method_status))
                
                # Summary
                total_detections = len(subjects.get('detections', []))
                st.metric("Total Detections", total_detections)
                
                # Subject types breakdown
                subject_counts = subjects.get('class_counts', {})
                if subject_counts:
                    st.markdown("##### 📊 Detection Breakdown")
                    
                    import pandas as pd
                    df = pd.DataFrame([
                        {"Subject": k, "Count": v} 
                        for k, v in subject_counts.items()
                    ])
                    st.bar_chart(df.set_index('Subject'))
                    
                    # Show detection presence percentage
                    st.markdown("##### 📈 Subject Presence")
                    presence = subjects.get('subject_presence', {})
                    if presence:
                        for subject, pct in sorted(presence.items(), key=lambda x: x[1], reverse=True)[:5]:
                            st.markdown(f"**{subject}:** {pct:.1f}% of frames")
                else:
                    st.warning("No specific subjects detected, but motion/activity was analyzed")
                
                # Prominent subjects
                prominent = subjects.get('prominent_subjects', [])
                if prominent:
                    st.markdown("##### ⭐ Prominent Subjects")
                    for subj in prominent[:3]:
                        st.markdown(f"- **{subj['class']}**: {subj['count']} detections, {subj['area_percentage']:.1f}% of frame")
                
                # Multi-speaker info
                if st.session_state.is_multi_speaker:
                    st.markdown("---")
                    st.markdown("#### 👥 Multi-Speaker Detection")
                    st.info("🎯 Interview/Talk scenario detected!")
                    
                    tracking = st.session_state.speaker_tracking
                    if tracking:
                        tracked_subjects = tracking.get('tracked_subjects', {})
                        st.markdown(f"**Tracked Subjects:** {len(tracked_subjects)}")
                        
                        # Show speaker segments
                        speaker_segments = tracking.get('speaker_segments', [])
                        if speaker_segments:
                            st.markdown("##### 🗣️ Speaker Timeline")
                            for seg in speaker_segments[:10]:
                                speaker_id = seg.get('speaker_id', 0)
                                start = seg.get('start', 0)
                                end = seg.get('end', 0)
                                duration = seg.get('duration', 0)
                                transcript_preview = seg.get('transcript', '')[:50] + "..."
                                st.markdown(f"**Speaker {speaker_id}:** {start:.1f}s - {end:.1f}s ({duration:.1f}s)")
                                if transcript_preview:
                                    st.markdown(f"   _\"{transcript_preview[:30]}...\"_")
                            
                            if len(speaker_segments) > 10:
                                st.info(f"... and {len(speaker_segments) - 10} more segments")
            else:
                st.info("No subjects detected - video will use center crop")
        
        # Motion Analysis
        st.markdown("---")
        st.markdown("#### 📈 Motion Analysis")
        
        motion_data = st.session_state.analysis_data.get('motion', {})
        if motion_data:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Motion", f"{motion_data.get('avg_motion', 0):.2f}")
            with col2:
                st.metric("Peak Motion", f"{motion_data.get('peak_motion', 0):.2f}")
            with col3:
                st.metric("Motion Score", f"{motion_data.get('motion_score', 0):.1f}/100")
            with col4:
                st.metric("Dynamic Scenes", motion_data.get('dynamic_scenes', 0))
            
            # Motion intensity over time
            if motion_data.get('motion_timeline'):
                import pandas as pd
                df = pd.DataFrame({
                    'Time (s)': range(len(motion_data['motion_timeline'])),
                    'Motion Intensity': motion_data['motion_timeline']
                })
                st.line_chart(df.set_index('Time (s)'))
        else:
            st.info("Motion analysis not available")
        
        # Proceed button
        st.markdown("---")
        
        if st.session_state.viral_completed:
            st.success("✅ Viral detection completed! Check the 'Viral Detection' tab for results.")
        else:
            st.info("🎯 Analysis complete! Switch to the 'Viral Detection' tab to find viral moments.")
            if st.button("🎯 Start Viral Detection", use_container_width=True, type="primary"):
                st.session_state.processing_stage = 'viral_detection'
                st.rerun()

def render_viral_tab(modules):
    """Render the viral content detection tab"""
    st.markdown("### 🎯 Viral Content Detection")
    
    if not st.session_state.video_path:
        st.warning("⚠️ Please upload a video first in the Input tab")
        return
    
    # Check if analysis is completed
    if not st.session_state.analysis_completed:
        st.warning("⚠️ Please complete the analysis first in the Analysis tab")
        return
    
    # Run viral detection if not completed
    if not st.session_state.viral_completed and st.session_state.processing_stage != 'viral_detection':
        st.session_state.processing_stage = 'viral_detection'
        st.rerun()
        return
    
    # AI-Powered Viral Detection
    if st.session_state.processing_stage == 'viral_detection' and not st.session_state.viral_completed:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.markdown("🤖 **Analyzing content for viral potential...**")
        progress_bar.progress(20)
        
        try:
            # Initialize viral detector if module is available
            viral_detector = None
            if modules.get('viral'):
                try:
                    ai_provider = st.session_state.get('ai_provider', 'local')
                    viral_detector = modules['viral'](ai_provider=ai_provider)
                    status_text.markdown("✅ Viral detector initialized")
                except Exception as e:
                    status_text.warning(f"Viral detector init failed: {e}, using fallback")
            
            progress_bar.progress(40)
            status_text.markdown("🧠 **Identifying viral moments...**")
            
            # Prepare analysis data safely
            analysis_data = {
                'transcription': st.session_state.get('transcription', {}),
                'subjects': st.session_state.get('detected_subjects', {}),
                'motion': st.session_state.analysis_data.get('motion', {}) if st.session_state.get('analysis_data') else {},
                'video_info': st.session_state.get('video_info', {}),
                'speaker_tracking': st.session_state.get('speaker_tracking', {}),
            }
            
            if viral_detector:
                # Use actual viral detector
                viral_segments = viral_detector.detect_viral_segments(
                    analysis_data,
                    target_duration=st.session_state.get('reel_duration', 60)
                )
                
                progress_bar.progress(80)
                status_text.markdown("📊 **Scoring and ranking segments...**")
                
                # Score segments
                viral_segments = viral_detector.score_segments(viral_segments)
            else:
                # Use fallback/mock detection
                progress_bar.progress(60)
                status_text.markdown("📊 **Analyzing content patterns...**")
                viral_segments = create_mock_viral_segments()
            
            st.session_state.viral_segments = viral_segments
            progress_bar.progress(100)
            status_text.success("✅ Viral detection completed!")
            
        except Exception as e:
            import traceback
            st.error(f"Viral detection error: {str(e)}")
            st.code(traceback.format_exc())
            # Create mock data for demonstration
            st.session_state.viral_segments = create_mock_viral_segments()
        
        st.session_state.processing_stage = 'viral_detected'
        st.session_state.viral_completed = True
        time.sleep(1)
        st.rerun()
    
    # Display Viral Segments - show whenever viral detection is completed
    if st.session_state.viral_completed:
        st.markdown("#### 🏆 Top Viral Moments")
        
        segments = st.session_state.viral_segments
        
        if segments:
            # Top recommendation
            best_segment = segments[0] if segments else None
            
            if best_segment:
                st.markdown("### 🥇 Recommended Clip")
                
                rec_col1, rec_col2, rec_col3 = st.columns(3)
                
                with rec_col1:
                    st.metric("Viral Score", f"{best_segment.get('score', 0):.0f}/100")
                with rec_col2:
                    start = best_segment.get('start', 0)
                    end = best_segment.get('end', 60)
                    st.metric("Duration", f"{end - start:.1f}s")
                with rec_col3:
                    st.metric("Category", best_segment.get('category', 'General'))
                
                st.markdown(f"**Reason:** {best_segment.get('reason', 'High engagement potential')}")
                
                if best_segment.get('transcript'):
                    with st.expander("View Transcript"):
                        st.markdown(best_segment['transcript'])
            
            # All segments
            st.markdown("---")
            st.markdown("#### 📋 All Detected Moments")
            
            for i, segment in enumerate(segments[:10]):  # Show top 10
                with st.container():
                    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
                    
                    with col1:
                        # Rank badge
                        badge = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
                        st.markdown(f"### {badge}")
                    
                    with col2:
                        start = segment.get('start', 0)
                        end = segment.get('end', 60)
                        st.markdown(f"**Timestamp:** {format_time(start)} - {format_time(end)}")
                        st.markdown(f"**Category:** {segment.get('category', 'N/A')}")
                    
                    with col3:
                        # Score bar
                        score = segment.get('score', 0)
                        st.progress(score / 100)
                        st.markdown(f"Score: {score:.0f}")
                    
                    with col4:
                        if st.button("Select", key=f"select_{i}", use_container_width=True):
                            st.session_state.selected_segment = segment
                            st.session_state.processing_stage = 'editing'
                            st.rerun()
                    
                    # Reason
                    st.markdown(f"<small>💡 {segment.get('reason', 'No reason provided')}</small>", unsafe_allow_html=True)
                    st.markdown("---")
            
            # Custom segment selection
            st.markdown("#### ✂️ Custom Segment Selection")
            
            video_info = st.session_state.video_info
            max_duration = video_info.get('duration', 300)
            
            col1, col2 = st.columns(2)
            
            with col1:
                custom_start = st.number_input(
                    "Start Time (seconds)",
                    min_value=0.0,
                    max_value=max_duration,
                    value=0.0,
                    step=0.5
                )
            
            with col2:
                custom_end = st.number_input(
                    "End Time (seconds)",
                    min_value=0.0,
                    max_value=max_duration,
                    value=min(60.0, max_duration),
                    step=0.5
                )
            
            if st.button("🎯 Use Custom Segment", use_container_width=True):
                custom_segment = {
                    'start': custom_start,
                    'end': custom_end,
                    'score': 50,
                    'category': 'Custom',
                    'reason': 'User-selected segment'
                }
                st.session_state.selected_segment = custom_segment
                st.session_state.processing_stage = 'editing'
                st.rerun()
        else:
            st.info("No viral segments detected. Try adjusting your settings.")

def render_editor_tab(modules):
    """Render the editor tab for fine-tuning the output"""
    st.markdown("### ✂️ Reel Editor")
    
    if not st.session_state.video_path:
        st.warning("⚠️ Please upload a video first in the Input tab")
        return
    
    if not st.session_state.viral_completed:
        st.warning("⚠️ Please complete viral detection first in the Viral Detection tab")
        return
    
    if not st.session_state.selected_segment:
        st.warning("⚠️ Please select a segment from the Viral Detection tab")
        st.info("👆 Go to the 'Viral Detection' tab and click 'Select' on a segment to edit it.")
        return
    
    segment = st.session_state.selected_segment
    
    # Segment info
    st.markdown("#### 📊 Selected Segment")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Start", format_time(segment.get('start', 0)))
    with col2:
        st.metric("End", format_time(segment.get('end', 60)))
    with col3:
        duration = segment.get('end', 60) - segment.get('start', 0)
        st.metric("Duration", f"{duration:.1f}s")
    with col4:
        st.metric("Score", f"{segment.get('score', 0):.0f}")
    
    # Edit controls
    st.markdown("---")
    st.markdown("#### 🎛️ Edit Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### ⏱️ Timing Adjustments")
        
        video_info = st.session_state.video_info
        max_duration = video_info.get('duration', 300)
        
        edit_start = st.slider(
            "Start Time",
            min_value=0.0,
            max_value=max_duration,
            value=float(segment.get('start', 0)),
            step=0.1,
            format="%.1fs"
        )
        
        edit_end = st.slider(
            "End Time",
            min_value=0.0,
            max_value=max_duration,
            value=float(segment.get('end', 60)),
            step=0.1,
            format="%.1fs"
        )
        
        st.session_state.selected_segment['start'] = edit_start
        st.session_state.selected_segment['end'] = edit_end
    
    with col2:
        st.markdown("##### 🎬 Panning Settings")
        
        # Show multi-speaker option if detected
        if st.session_state.is_multi_speaker:
            st.info("👥 Multi-speaker video detected - predictive focus switching enabled")
        
        pan_mode = st.selectbox(
            "Panning Mode",
            options=["auto", "center", "follow_subject", "follow_speaker", "cinematic", "static"],
            format_func=lambda x: {
                "auto": "Auto (AI Recommended)",
                "center": "Center Crop",
                "follow_subject": "Follow Subject",
                "follow_speaker": "Follow Speaker (Interview Mode)",
                "cinematic": "Cinematic Pan",
                "static": "Static Crop"
            }[x]
        )
        
        if pan_mode == "follow_subject":
            tracking_subject = st.selectbox(
                "Track Subject",
                options=["auto", "person", "face", "object"],
                format_func=lambda x: {
                    "auto": "Auto Detect",
                    "person": "Person",
                    "face": "Face",
                    "object": "Main Object"
                }[x]
            )
        
        zoom_level = st.slider(
            "Zoom Level",
            min_value=1.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Higher values zoom in more"
        )
    
    # Preview section
    st.markdown("---")
    st.markdown("#### 👁️ Preview")
    
    if st.button("🎬 Generate Preview", use_container_width=True):
        with st.spinner("Generating preview frames..."):
            if modules['processor']:
                try:
                    processor = modules['processor']()
                    
                    # Generate preview frames
                    preview_frames = processor.generate_preview(
                        st.session_state.video_path,
                        edit_start,
                        edit_end,
                        pan_mode=pan_mode,
                        zoom=zoom_level
                    )
                    
                    st.session_state.preview_frames = preview_frames
                    
                except Exception as e:
                    st.error(f"Preview generation error: {str(e)}")
            else:
                st.warning("Processor module not available for preview")
    
    # Display preview frames
    if st.session_state.preview_frames:
        st.markdown("##### Preview Frames")
        
        frames = st.session_state.preview_frames
        cols = min(len(frames), 5)
        
        preview_cols = st.columns(cols)
        for i, frame in enumerate(frames[:cols]):
            with preview_cols[i]:
                if isinstance(frame, str) and os.path.exists(frame):
                    st.image(frame, caption=f"Frame {i+1}")
                elif isinstance(frame, np.ndarray):
                    st.image(frame, caption=f"Frame {i+1}")
    
    # Output settings
    st.markdown("---")
    st.markdown("#### 📱 Output Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        output_format = st.selectbox(
            "Output Format",
            options=["mp4", "mov", "webm"],
            index=0
        )
    
    with col2:
        resolution = st.selectbox(
            "Resolution",
            options=["1080x1920 (FHD)", "720x1280 (HD)", "540x960 (SD)"],
            index=0
        )
    
    with col3:
        fps = st.selectbox(
            "Frame Rate",
            options=["30 fps", "60 fps", "Original"],
            index=0
        )
    
    # Generate button
    st.markdown("---")
    
    if st.button("🎬 Generate Reel", use_container_width=True, type="primary"):
        st.session_state.processing_stage = 'exporting'
        st.rerun()

def render_export_tab(modules):
    """Render the export tab for generating the final reel"""
    st.markdown("### 📤 Export Reel")
    
    if not st.session_state.video_path:
        st.warning("⚠️ Please upload a video first in the Input tab")
        return
    
    if not st.session_state.selected_segment:
        st.warning("⚠️ Please select and edit a segment first in the Editor tab")
        return
    
    segment = st.session_state.selected_segment
    
    if st.session_state.processing_stage == 'exporting' and not st.session_state.export_completed:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Prepare clip
        status_text.markdown("✂️ **Step 1/4:** Extracting video segment...")
        progress_bar.progress(10)
        
        # Step 2: Process video
        status_text.markdown("🎬 **Step 2/4:** Applying panning and cropping...")
        progress_bar.progress(30)
        
        # Step 3: Final encoding
        status_text.markdown("🎥 **Step 3/4:** Encoding final output...")
        progress_bar.progress(60)
        
        if modules['processor']:
            try:
                processor = modules['processor']()
                
                output_dir = Path("export")
                output_dir.mkdir(exist_ok=True)
                
                # Prepare tracking points from focus_points if available
                tracking_points = None
                if st.session_state.focus_points:
                    tracking_points = st.session_state.focus_points
                
                output_path = processor.generate_reel(
                    input_path=st.session_state.video_path,
                    output_dir=str(output_dir),
                    start_time=segment.get('start', 0),
                    end_time=segment.get('end', 60),
                    quality=st.session_state.output_quality,
                    panning=st.session_state.panning_smoothness,
                    tracking_points=tracking_points
                )
                
                # Verify the output file exists
                if output_path and os.path.exists(output_path):
                    st.session_state.output_path = output_path
                    progress_bar.progress(90)
                else:
                    st.error("Failed to create output file. Check if FFmpeg is installed correctly")
                    st.session_state.output_path = None
                    st.session_state.processing_stage = 'export_failed'
                    return
                    
            except Exception as e:
                import traceback
                st.error(f"Error generating reel: {str(e)}")
                with st.expander("View Error Details"):
                    st.code(traceback.format_exc())
                st.session_state.output_path = None
                st.session_state.processing_stage = 'export_failed'
                return
        else:
            st.error("Video processor module not available")
            st.session_state.output_path = None
            st.session_state.processing_stage = 'export_failed'
            return
        
        # Step 4: Finalize - only if we got here successfully
        status_text.markdown("✅ **Step 4/4:** Finalizing...")
        progress_bar.progress(100)
        status_text.success("✅ Reel generated successfully!")
        
        st.session_state.processing_stage = 'exported'
        st.session_state.export_completed = True
        time.sleep(1)
        st.rerun()
    
    # Display results
    if st.session_state.export_completed:
        st.markdown("#### 🎉 Your Reel is Ready!")
        
        # Success message
        st.markdown("""
        <div class="success-box">
            <h4>✅ Export Complete!</h4>
            <p>Your reel has been generated and is ready for download.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Output preview
        output_path = st.session_state.get('output_path')
        if output_path and os.path.exists(output_path):
            st.video(output_path)
            
            # Download button
            with open(output_path, "rb") as f: 
                video_bytes = f.read()
                
                st.download_button(
                    label="📥 Download Reel",
                    data=video_bytes,
                    file_name="reel_output.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    type="primary"
                )
                
                # Show file size
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                st.info(f"Expected path: {output_path}")
        else:
            st.info("Output file will be available after processing completes")
        
        # Output details
        st.markdown("---")
        st.markdown("#### 📊 Output Details")
        
        segment = st.session_state.selected_segment or {}
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            duration = segment.get('end', 60) - segment.get('start', 0)
            st.metric("Duration", f"{duration:.1f}s")
        with col2:
            st.metric("Quality", st.session_state.output_quality.title())
        with col3:
            st.metric("Panning", st.session_state.panning_smoothness.title())
        with col4:
            st.metric("Format", "MP4 (H.264)")
        
        # Summary
        st.markdown("---")
        st.markdown("#### 📋 Processing Summary")
        
        transcription = st.session_state.get('transcription') or {}
        detected_subjects = st.session_state.get('detected_subjects') or {}
        viral_segments = st.session_state.get('viral_segments') or []
        
        summary_items = [
            ("Video Input", st.session_state.video_info.get('original_name', 'Uploaded video') if st.session_state.get('video_info') else 'Uploaded Video'),
            ("Transcription", f"{len(st.session_state.transcription.get('segments', []))} segments" if st.session_state.transcription else "N/A"),
            ("Subjects Detected", len(st.session_state.detected_subjects.get('detections', [])) if st.session_state.detected_subjects else 0),
            ("Multi-Speaker", "Yes" if st.session_state.is_multi_speaker else "No"),
            ("Viral Segments", len(st.session_state.viral_segments)),
            ("AI Provider", st.session_state.ai_provider.title()),
            ("Output Path", st.session_state.output_path or "N/A")
        ]
        
        # ------------------------ Updated version
        #         transcription = st.session_state.get('transcription') or {}
        # detected_subjects = st.session_state.get('detected_subjects') or {}
        # viral_segments = st.session_state.get('viral_segments') or []
        
        # summary_items = [
        #     ("Video Input", st.session_state.video_info.get('original_name', 'Uploaded video') if st.session_state.get('video_info') else 'Uploaded video'),
        #     ("Transcription", f"{len(transcription.get('segments', []))} segments" if transcription else "N/A"),
        #     ("Subjects Detected", len(detected_subjects.get('detections', [])) if detected_subjects else 0),
        #     ("Multi-Speaker", "Yes" if st.session_state.get('is_multi_speaker') else "No"),
        #     ("Viral Segments", len(viral_segments)),
        #     ("AI Provider", st.session_state.ai_provider.title()),
        #     ("Output Path", output_path or "N/A")
        
        for label, value in summary_items:
            st.markdown(f"**{label}:** {value}")
        
        # New reel button
        st.markdown("---")
        if st.button("🔄 Create Another Reel", use_container_width=True):
            # Reset state but keep video
            st.session_state.processing_stage = 'analyzed'
            st.session_state.selected_segment = None
            st.session_state.viral_segments = []
            st.session_state.output_path = None
            st.session_state.viral_completed = False
            st.session_state.export_completed = False
            st.rerun()
    else:
        # Show export info
        st.info("👆 Select a segment in the Editor tab, then click 'Generate Reel' to export.")

# Utility Functions
def format_time(seconds: float) -> str:
    """Format seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def create_mock_viral_segments() -> List[Dict]:
    """Create mock viral segments for demonstration"""
    video_info = st.session_state.video_info
    duration = video_info.get('duration', 300)
    
    segments = []
    
    # Create sample segments
    sample_times = [
        (0, min(60, duration), "Opening segment", 85),
        (min(30, duration * 0.1), min(90, duration * 0.3), "Key moment", 92),
        (min(120, duration * 0.4), min(180, duration * 0.6), "Highlight", 88),
        (min(180, duration * 0.6), min(240, duration * 0.8), "Conclusion", 75),
    ]
    
    categories = ["Entertainment", "Education", "Tutorial", "Highlight", "Story"]
    reasons = [
        "High energy moment with engaging content",
        "Clear audio with important information",
        "Visual interest with subject movement",
        "Strong emotional content detected",
        "Peak engagement potential based on transcript"
    ]
    
    for start, end, title, score in sample_times:
        if start < duration:
            segments.append({
                'start': start,
                'end': min(end, duration),
                'score': score,
                'category': categories[len(segments) % len(categories)],
                'reason': reasons[len(segments) % len(reasons)],
                'title': title,
                'transcript': f"Sample transcript for segment {len(segments) + 1}..."
            })
    
    # Sort by score
    segments.sort(key=lambda x: x['score'], reverse=True)
    
    return segments

# Main execution
def main():
    """Main entry point"""
    render_sidebar()
    render_main()

if __name__ == "__main__":
    main()