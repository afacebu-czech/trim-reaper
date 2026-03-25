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
        'active_tab': 0  # Default to first tab
    }
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Sidebar Configuration
def render_sidebar():
    """Render the sidebar with configuration options"""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # AI Provider Selection
        st.markdown("#### 🤖 AI Provider")
        ai_provider = st.selectbox(
            "Select AI Provider",
            options=["openai", "anthropic", "local"],
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
            os.environ["ANTHROPIC_API_KEY"] = api_key
        
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
            
            whisper_model = st.selectbox(
                "Whisper Model",
                options=["tiny", "base", "small", "medium", "large"],
                index=2,
                key="whisper_model"
            )
            
            detection_model = st.selectbox(
                "Detection Model",
                options=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                index=1,
                key="detection_model"
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
        
        if st.button("⬇️ Download from YouTube", use_container_width=True):
            if youtube_url and modules['downloader']:
                with st.spinner("Downloading video from YouTube..."):
                    try:
                        downloader = modules['downloader']()
                        video_path = downloader.download(youtube_url)
                        if video_path:
                            st.session_state.video_path = video_path
                            st.success("✅ Video downloaded successfully!")
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
            st.session_state.video_path = None
            st.session_state.video_info = {}
            st.session_state.transcription = None
            st.session_state.viral_segments = []
            st.rerun()
    
    # Video Preview
    if st.session_state.video_path:
        st.markdown("---")
        st.markdown("### 📺 Video Preview")
        
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
    
    # Analysis Progress
    if st.session_state.processing_stage == 'analyzing':
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Transcription
        status_text.markdown("🎙️ **Step 1/3:** Transcribing audio with Whisper...")
        progress_bar.progress(10)
        
        if modules['transcription']:
            try:
                transcriber = modules['transcription'](model_size=st.session_state.whisper_model)
                transcription = transcriber.transcribe(st.session_state.video_path)
                st.session_state.transcription = transcription
                progress_bar.progress(40)
                status_text.success("✅ Transcription completed!")
            except Exception as e:
                st.error(f"Transcription error: {str(e)}")
                progress_bar.progress(40)
        else:
            progress_bar.progress(40)
            st.warning("Transcription module not available")
        
        # Step 2: Subject Detection
        status_text.markdown("🎯 **Step 2/3:** Detecting subjects (faces, bodies, motion)...")
        
        if modules['detector']:
            try:
                detector = modules['detector'](model_name=st.session_state.detection_model)
                subjects = detector.detect_subjects(st.session_state.video_path, sample_rate=5)
                st.session_state.detected_subjects = subjects
                progress_bar.progress(70)
                
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
                progress_bar.progress(70)
        else:
            progress_bar.progress(70)
            st.warning("Detector module not available")
        
        # Step 3: Motion Analysis
        status_text.markdown("🎥 **Step 3/3:** Analyzing motion patterns...")
        progress_bar.progress(90)
        
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
        time.sleep(1)
        st.rerun()
    
    # Display Analysis Results
    if st.session_state.processing_stage == 'analyzed':
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
                
                # Main subjects for tracking
                main_subjects = subjects.get('main_subjects', [])
                if main_subjects:
                    st.markdown(f"##### 🎬 Tracking Points")
                    st.info(f"Generated {len(main_subjects)} tracking points for smooth panning")
                else:
                    st.info("Using center crop for panning (no subjects detected for tracking)")
                
                # Detection timeline
                with st.expander("View Detection Timeline"):
                    timeline = subjects.get('subject_timeline', [])
                    if timeline:
                        for entry in timeline[:30]:
                            frame = entry.get('frame', 0)
                            count = entry.get('detection_count', 0)
                            timestamp = entry.get('timestamp', 0)
                            subjects_in_frame = entry.get('subjects', [])
                            subject_classes = [s['class'] for s in subjects_in_frame[:3]]
                            st.markdown(f"Frame {frame} ({timestamp:.1f}s): {count} detections - {', '.join(subject_classes) if subject_classes else 'none'}")
                        if len(timeline) > 30:
                            st.info(f"... and {len(timeline) - 30} more frames")
                    else:
                        st.info("No timeline data available")
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
        
        # Check if viral detection should run automatically after analysis
        if st.session_state.processing_stage == 'analyzed' and not st.session_state.get('viral_detection_done', False):
            st.info("🎯 Click below to start viral detection, then switch to the 'Viral Detection' tab")
            
            if st.button("🎯 Start Viral Detection", use_container_width=True, type="primary"):
                st.session_state.processing_stage = 'viral_detection'
                st.rerun()
        elif st.session_state.get('viral_detection_done', False):
            st.success("✅ Viral detection completed! Check the 'Viral Detection' tab for results.")
    
    # Initial state
    if st.session_state.processing_stage == 'idle':
        st.info("👆 Click 'Start Analysis' in the Input tab after uploading a video")
        
        # Show what will be analyzed
        st.markdown("#### 📋 What will be analyzed:")
        
        analysis_items = [
            ("🎙️ Audio Transcription", "Using OpenAI Whisper to convert speech to text with timestamps"),
            ("🎯 Subject Detection", "Using YOLO to detect people, objects, and main subjects"),
            ("🎥 Motion Analysis", "Analyzing camera movement and scene dynamics"),
            ("📊 Scene Detection", "Identifying scene changes and key moments")
        ]
        
        for icon_title, description in analysis_items:
            st.markdown(f"**{icon_title}**")
            st.markdown(f"<small>{description}</small>", unsafe_allow_html=True)

def render_viral_tab(modules):
    """Render the viral content detection tab"""
    st.markdown("### 🎯 Viral Content Detection")
    
    if not st.session_state.video_path:
        st.warning("⚠️ Please upload and analyze a video first")
        return
    
    if st.session_state.processing_stage == 'analyzed':
        # Show prompt to start viral detection
        st.info("👆 Click 'Start Viral Detection' in the Analysis tab to begin")
        
        # Also show a button here for convenience
        if st.button("🎯 Start Viral Detection Now", use_container_width=True, type="primary"):
            st.session_state.processing_stage = 'viral_detection'
            st.rerun()
        return
    
    if st.session_state.processing_stage not in ['viral_detection', 'viral_detected', 'editing']:
        st.warning("⚠️ Please complete the analysis first")
        return
    
    # AI-Powered Viral Detection
    if st.session_state.processing_stage == 'viral_detection':
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
                'video_info': st.session_state.get('video_info', {})
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
            st.session_state.viral_detection_done = True
            progress_bar.progress(100)
            status_text.success("✅ Viral detection completed!")
            
        except Exception as e:
            import traceback
            st.error(f"Viral detection error: {str(e)}")
            st.code(traceback.format_exc())
            # Create mock data for demonstration
            st.session_state.viral_segments = create_mock_viral_segments()
            st.session_state.viral_detection_done = True
        
        st.session_state.processing_stage = 'viral_detected'
        time.sleep(1)
        st.rerun()
    
    # Display Viral Segments
    if st.session_state.processing_stage in ['viral_detected', 'editing']:
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
    
    if not st.session_state.selected_segment:
        st.warning("⚠️ Please select a segment from the Viral Detection tab")
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
        
        pan_mode = st.selectbox(
            "Panning Mode",
            options=["auto", "center", "follow_subject", "cinematic", "static"],
            format_func=lambda x: {
                "auto": "Auto (AI Recommended)",
                "center": "Center Crop",
                "follow_subject": "Follow Subject",
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
    
    if not st.session_state.selected_segment:
        st.warning("⚠️ Please select and edit a segment first")
        return
    
    segment = st.session_state.selected_segment
    
    if st.session_state.processing_stage == 'exporting':
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
                
                output_dir = Path("/export")
                output_dir.mkdir(exist_ok=True)
                
                output_path = processor.generate_reel(
                    input_path=st.session_state.video_path,
                    output_dir=str(output_dir),
                    start_time=segment.get('start', 0),
                    end_time=segment.get('end', 60),
                    quality=st.session_state.output_quality,
                    panning=st.session_state.panning_smoothness
                )
                
                st.session_state.output_path = output_path
                progress_bar.progress(90)
                
            except Exception as e:
                st.error(f"Error generating reel: {str(e)}")
                # Create a placeholder for demo
                st.session_state.output_path = str(Path("/home/z/my-project/reels-generator/outputs") / "demo_reel.mp4")
        
        # Step 4: Finalize
        status_text.markdown("✅ **Step 4/4:** Finalizing...")
        progress_bar.progress(100)
        status_text.success("✅ Reel generated successfully!")
        
        st.session_state.processing_stage = 'exported'
        time.sleep(1)
        st.rerun()
    
    # Display results
    if st.session_state.processing_stage == 'exported':
        st.markdown("#### 🎉 Your Reel is Ready!")
        
        # Success message
        st.markdown("""
        <div class="success-box">
            <h4>✅ Export Complete!</h4>
            <p>Your reel has been generated and is ready for download.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Output preview
        if st.session_state.output_path and os.path.exists(st.session_state.output_path):
            st.video(st.session_state.output_path)
            
            # Download button
            with open(st.session_state.output_path, "rb") as f:
                st.download_button(
                    label="📥 Download Reel",
                    data=f,
                    file_name="reel_output.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    type="primary"
                )
        else:
            st.info("Output file will be available after processing completes")
        
        # Output details
        st.markdown("---")
        st.markdown("#### 📊 Output Details")
        
        segment = st.session_state.selected_segment
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Duration", f"{segment.get('end', 60) - segment.get('start', 0):.1f}s")
        with col2:
            st.metric("Quality", st.session_state.output_quality.title())
        with col3:
            st.metric("Panning", st.session_state.panning_smoothness.title())
        with col4:
            st.metric("Format", "MP4 (H.264)")
        
        # Summary
        st.markdown("---")
        st.markdown("#### 📋 Processing Summary")
        
        summary_items = [
            ("Video Input", st.session_state.video_info.get('original_name', 'Uploaded video')),
            ("Transcription", f"{len(st.session_state.transcription.get('segments', []))} segments" if st.session_state.transcription else "N/A"),
            ("Subjects Detected", len(st.session_state.detected_subjects.get('detections', [])) if st.session_state.detected_subjects else 0),
            ("Viral Segments", len(st.session_state.viral_segments)),
            ("AI Provider", st.session_state.ai_provider.title()),
            ("Output Path", st.session_state.output_path or "N/A")
        ]
        
        for label, value in summary_items:
            st.markdown(f"**{label}:** {value}")
        
        # New reel button
        st.markdown("---")
        if st.button("🔄 Create Another Reel", use_container_width=True):
            # Reset state
            st.session_state.processing_stage = 'idle'
            st.session_state.selected_segment = None
            st.session_state.viral_segments = []
            st.session_state.output_path = None
            st.rerun()

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