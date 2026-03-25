"""
Video Processor Module
Handles video processing, cropping, panning, and encoding using FFmpeg and OpenCV
"""

import os
import cv2
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from dataclasses import dataclass
import shutil

@dataclass
class VideoInfo:
    """Video metadata"""
    width: int
    height: int
    fps: float
    duration: float
    codec: str
    bitrate: int
    audio_codec: str
    has_audio: bool

class VideoProcessor:
    """
    Process videos for reel generation
    Handles cropping, panning, encoding, and effects
    """
    
    # Reel aspect ratio (9:16)
    REEL_WIDTH = 1080
    REEL_HEIGHT = 1920
    REEL_FPS = 30
    
    # Quality presets
    QUALITY_PRESETS = {
        'low': {'crf': 28, 'preset': 'ultrafast', 'bitrate': '2M'},
        'medium': {'crf': 23, 'preset': 'fast', 'bitrate': '4M'},
        'high': {'crf': 18, 'preset': 'medium', 'bitrate': '8M'},
        'ultra': {'crf': 15, 'preset': 'slow', 'bitrate': '15M'}
    }
    
    # Panning smoothness levels
    SMOOTHNESS_LEVELS = {
        'none': {'smoothing': 0, 'interpolation': 'none'},
        'low': {'smoothing': 0.2, 'interpolation': 'linear'},
        'medium': {'smoothing': 0.5, 'interpolation': 'cubic'},
        'high': {'smoothing': 0.8, 'interpolation': 'spline'},
        'cinematic': {'smoothing': 0.95, 'interpolation': 'spline'}
    }
    
    def __init__(self):
        """Initialize video processor"""
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """Check if FFmpeg is available"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("FFmpeg is available")
            else:
                logger.warning("FFmpeg check failed")
        except FileNotFoundError:
            logger.warning("FFmpeg not found. Some features may not work.")
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get video metadata
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Try OpenCV first
        cap = cv2.VideoCapture(video_path)
        
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Get additional info with FFmpeg
            codec = 'unknown'
            bitrate = 0
            audio_codec = 'none'
            has_audio = False
            
            try:
                result = subprocess.run(
                    ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                     '-show_format', '-show_streams', video_path],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    import json
                    data = json.loads(result.stdout)
                    
                    # Get format info
                    fmt = data.get('format', {})
                    bitrate = int(fmt.get('bit_rate', 0))
                    
                    # Get stream info
                    for stream in data.get('streams', []):
                        if stream.get('codec_type') == 'video':
                            codec = stream.get('codec_name', 'unknown')
                        elif stream.get('codec_type') == 'audio':
                            has_audio = True
                            audio_codec = stream.get('codec_name', 'unknown')
                            
            except Exception as e:
                logger.warning(f"FFprobe error: {str(e)}")
            
            return {
                'width': width,
                'height': height,
                'fps': fps,
                'duration': duration,
                'frame_count': frame_count,
                'codec': codec,
                'bitrate': bitrate,
                'audio_codec': audio_codec,
                'has_audio': has_audio,
                'original_name': os.path.basename(video_path)
            }
        
        raise ValueError(f"Could not open video: {video_path}")
    
    def analyze_motion(self, video_path: str, sample_rate: int = 10) -> Dict[str, Any]:
        """
        Analyze motion intensity in video
        
        Args:
            video_path: Path to video file
            sample_rate: Analyze every Nth frame
            
        Returns:
            Motion analysis results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        motion_values = []
        prev_frame = None
        
        frame_idx = 0
        dynamic_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                if prev_frame is not None:
                    # Calculate difference
                    diff = cv2.absdiff(prev_frame, gray)
                    motion_score = np.mean(diff)
                    motion_values.append(float(motion_score))
                    
                    # Check if dynamic
                    if motion_score > 5:
                        dynamic_count += 1
                
                prev_frame = gray
            
            frame_idx += 1
        
        cap.release()
        
        if motion_values:
            avg_motion = float(np.mean(motion_values))
            peak_motion = float(np.max(motion_values))
            motion_score = min(100, avg_motion * 10 + peak_motion * 0.5)
        else:
            avg_motion = 0
            peak_motion = 0
            motion_score = 0
        
        return {
            'avg_motion': avg_motion,
            'peak_motion': peak_motion,
            'motion_score': motion_score,
            'dynamic_scenes': dynamic_count,
            'motion_timeline': motion_values,
            'total_frames': total_frames
        }
    
    def generate_preview(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        pan_mode: str = "auto",
        zoom: float = 1.0,
        num_frames: int = 5
    ) -> List[np.ndarray]:
        """
        Generate preview frames
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            pan_mode: Panning mode
            zoom: Zoom level
            num_frames: Number of preview frames
            
        Returns:
            List of preview frame arrays
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate crop dimensions for 9:16
        crop_width = int(height * 9 / 16)
        crop_height = height
        
        if crop_width > width:
            crop_width = width
            crop_height = int(width * 16 / 9)
        
        # Apply zoom
        crop_width = int(crop_width / zoom)
        crop_height = int(crop_height / zoom)
        
        preview_frames = []
        
        duration = end_time - start_time
        frame_interval = duration / (num_frames + 1)
        
        for i in range(num_frames):
            timestamp = start_time + frame_interval * (i + 1)
            frame_number = int(timestamp * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # Apply crop
                center_x = width // 2
                center_y = height // 2
                
                x1 = max(0, center_x - crop_width // 2)
                y1 = max(0, center_y - crop_height // 2)
                x2 = min(width, x1 + crop_width)
                y2 = min(height, y1 + crop_height)
                
                cropped = frame[y1:y2, x1:x2]
                
                # Resize to reel dimensions
                resized = cv2.resize(cropped, (self.REEL_WIDTH, self.REEL_HEIGHT))
                
                # Convert BGR to RGB
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                
                preview_frames.append(rgb)
        
        cap.release()
        
        return preview_frames
    
    def generate_reel(
        self,
        input_path: str,
        output_dir: str,
        start_time: float,
        end_time: float,
        quality: str = "high",
        panning: str = "medium",
        tracking_points: Optional[List[Dict]] = None,
        add_captions: bool = False,
        caption_data: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate a reel from video
        
        Args:
            input_path: Path to input video
            output_dir: Output directory
            start_time: Start time in seconds
            end_time: End time in seconds
            quality: Output quality preset
            panning: Panning smoothness level
            tracking_points: Optional tracking points for smooth panning
            add_captions: Whether to add captions
            caption_data: Caption data if adding captions
            
        Returns:
            Path to generated reel
        """
        logger.info(f"Generating reel from {input_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Get video info
        video_info = self.get_video_info(input_path)
        
        # Get quality settings
        quality_settings = self.QUALITY_PRESETS.get(quality, self.QUALITY_PRESETS['high'])
        smoothness_settings = self.SMOOTHNESS_LEVELS.get(panning, self.SMOOTHNESS_LEVELS['medium'])
        
        # Output path
        output_path = os.path.join(output_dir, f"reel_{int(start_time)}_{int(end_time)}.mp4")
        
        # Process video
        if tracking_points:
            output_path = self._process_with_tracking(
                input_path, output_path, tracking_points,
                start_time, end_time, quality_settings
            )
        else:
            output_path = self._process_with_ffmpeg(
                input_path, output_path, start_time, end_time,
                quality_settings, smoothness_settings
            )
        
        # Add captions if requested
        if add_captions and caption_data:
            output_path = self._add_captions(output_path, caption_data)
        
        logger.success(f"Reel generated: {output_path}")
        
        return output_path
    
    def _process_with_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        quality: Dict,
        smoothness: Dict
    ) -> str:
        """Process video using FFmpeg with crop filter"""
        
        # Get video dimensions
        video_info = self.get_video_info(input_path)
        width = video_info['width']
        height = video_info['height']
        
        # Calculate crop dimensions for 9:16
        crop_height = height
        crop_width = int(height * 9 / 16)
        
        if crop_width > width:
            crop_width = width
            crop_height = int(width * 16 / 9)
        
        # Calculate center crop position
        x_offset = (width - crop_width) // 2
        y_offset = (height - crop_height) // 2
        
        duration = end_time - start_time
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', input_path,
            '-t', str(duration),
            '-vf', f'crop={crop_width}:{crop_height}:{x_offset}:{y_offset},scale={self.REEL_WIDTH}:{self.REEL_HEIGHT}',
            '-c:v', 'libx264',
            '-crf', str(quality['crf']),
            '-preset', quality['preset'],
            '-c:a', 'aac',
            '-b:a', '192k',
            '-movflags', '+faststart',
            output_path
        ]
        
        logger.info(f"Running FFmpeg: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                # Try simpler approach
                return self._process_with_opencv(input_path, output_path, start_time, end_time)
            
            return output_path
            
        except Exception as e:
            logger.error(f"FFmpeg processing error: {str(e)}")
            return self._process_with_opencv(input_path, output_path, start_time, end_time)
    
    def _process_with_tracking(
        self,
        input_path: str,
        output_path: str,
        tracking_points: List[Dict],
        start_time: float,
        end_time: float,
        quality: Dict
    ) -> str:
        """Process video with subject tracking"""
        
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create tracking lookup
        tracking_lookup = {tp['frame']: tp for tp in tracking_points}
        
        # Calculate dimensions
        crop_height = height
        crop_width = int(height * 9 / 16)
        
        if crop_width > width:
            crop_width = width
            crop_height = int(width * 16 / 9)
        
        # Temp output
        temp_output = output_path.replace('.mp4', '_temp.mp4')
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, self.REEL_FPS, 
                              (self.REEL_WIDTH, self.REEL_HEIGHT))
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Smoothing parameters
        prev_center_x = width // 2
        prev_center_y = height // 2
        smoothing = 0.3
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Get tracking point
            relative_frame = frame_idx - start_frame
            
            if frame_idx in tracking_lookup:
                target = tracking_lookup[frame_idx]['center']
                target_x, target_y = target
            else:
                target_x, target_y = prev_center_x, prev_center_y
            
            # Smooth the movement
            center_x = int(prev_center_x * (1 - smoothing) + target_x * smoothing)
            center_y = int(prev_center_y * (1 - smoothing) + target_y * smoothing)
            
            prev_center_x = center_x
            prev_center_y = center_y
            
            # Calculate crop coordinates
            x1 = max(0, min(center_x - crop_width // 2, width - crop_width))
            y1 = max(0, min(center_y - crop_height // 2, height - crop_height))
            x2 = x1 + crop_width
            y2 = y1 + crop_height
            
            # Crop and resize
            cropped = frame[y1:y2, x1:x2]
            resized = cv2.resize(cropped, (self.REEL_WIDTH, self.REEL_HEIGHT))
            
            out.write(resized)
        
        cap.release()
        out.release()
        
        # Re-encode with audio
        return self._add_audio(input_path, temp_output, output_path, start_time, end_time)
    
    def _process_with_opencv(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float
    ) -> str:
        """Fallback processing using OpenCV"""
        
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate crop
        crop_height = height
        crop_width = int(height * 9 / 16)
        
        if crop_width > width:
            crop_width = width
            crop_height = int(width * 16 / 9)
        
        x_offset = (width - crop_width) // 2
        y_offset = (height - crop_height) // 2
        
        # Temp output
        temp_output = output_path.replace('.mp4', '_temp.mp4')
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, self.REEL_FPS,
                              (self.REEL_WIDTH, self.REEL_HEIGHT))
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            
            if not ret:
                break
            
            cropped = frame[y_offset:y_offset+crop_height, 
                           x_offset:x_offset+crop_width]
            resized = cv2.resize(cropped, (self.REEL_WIDTH, self.REEL_HEIGHT))
            out.write(resized)
        
        cap.release()
        out.release()
        
        # Add audio
        return self._add_audio(input_path, temp_output, output_path, start_time, end_time)
    
    def _add_audio(
        self,
        input_path: str,
        video_path: str,
        output_path: str,
        start_time: float,
        end_time: float
    ) -> str:
        """Add audio from original video"""
        
        duration = end_time - start_time
        
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-ss', str(start_time),
            '-i', input_path,
            '-t', str(duration),
            '-map', '0:v',
            '-map', '1:a?',
            '-c:v', 'libx264',
            '-crf', '18',
            '-preset', 'medium',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-movflags', '+faststart',
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Remove temp file
                if os.path.exists(video_path):
                    os.remove(video_path)
                return output_path
            else:
                logger.warning(f"Audio merge failed: {result.stderr}")
                # Just rename temp file
                if os.path.exists(video_path):
                    shutil.move(video_path, output_path)
                return output_path
                
        except Exception as e:
            logger.warning(f"Audio merge error: {str(e)}")
            if os.path.exists(video_path):
                shutil.move(video_path, output_path)
            return output_path
    
    def _add_captions(self, video_path: str, captions: List[Dict]) -> str:
        """Add captions to video using FFmpeg"""
        
        # Create SRT file
        srt_path = video_path.replace('.mp4', '.srt')
        
        with open(srt_path, 'w') as f:
            for i, cap in enumerate(captions, 1):
                start = self._seconds_to_srt_time(cap['start'])
                end = self._seconds_to_srt_time(cap['end'])
                text = cap['text']
                
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
        
        output_path = video_path.replace('.mp4', '_captioned.mp4')
        
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', f"subtitles={srt_path}:force_style='Fontsize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2'",
            '-c:a', 'copy',
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True)
            return output_path
        except Exception as e:
            logger.warning(f"Caption error: {str(e)}")
            return video_path
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
    
    def apply_color_correction(
        self,
        video_path: str,
        output_path: str,
        brightness: float = 0,
        contrast: float = 1.0,
        saturation: float = 1.0
    ) -> str:
        """Apply color correction to video"""
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        temp_output = output_path.replace('.mp4', '_color.mp4')
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        for _ in range(total_frames):
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Apply corrections
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            
            # Apply saturation
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = hsv[:,:,1] * saturation
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        return self._add_audio(video_path, temp_output, output_path, 0, total_frames/fps)
    
    def create_thumbnail(
        self,
        video_path: str,
        timestamp: float,
        output_path: str
    ) -> str:
        """Create a thumbnail from video"""
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        cap.release()
        
        if ret:
            # Resize for thumbnail
            thumbnail = cv2.resize(frame, (self.REEL_WIDTH, self.REEL_HEIGHT))
            cv2.imwrite(output_path, thumbnail)
            return output_path
        
        return None


# Test
if __name__ == "__main__":
    processor = VideoProcessor()
    print("Video Processor initialized")
    print(f"Reel dimensions: {processor.REEL_WIDTH}x{processor.REEL_HEIGHT}")